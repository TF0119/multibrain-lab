"""BrainPolicy（神経中核の方策ラッパ）の形・勾配・状態リセット（PLAN §4.2〜§4.4、§7）。

小さな乱数配線（test_ports.py と同じ作り方）で単体確認し、data/ と CUDA がある
環境では全規模の中核でも確かめる。
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch.distributions import Normal

from multibrain.brain.build import build_core
from multibrain.brain.core import BETA, BrainCore, Ports
from multibrain.brain.policy import LOG_STD_MIN, BrainPolicy

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(
    not (DATA / "ports.json").exists(),
    reason="data/ports.json が無い（scripts/build_ports.py を先に実行）")

K, B = 2, 3


@pytest.fixture(scope="module")
def small():
    """ports の全ニューロンを含む最小の乱数配線 (w0, ports, device)。"""
    d = json.loads((DATA / "ports.json").read_text())
    ids = sorted({i for p in d["inputs"] + d["outputs"] for i in p["neurons"]})
    body_ids = np.array(ids, dtype=np.int64)
    n = len(body_ids)
    rng = np.random.default_rng(0)
    w0 = sp.random(n, n, density=20 / n, random_state=rng,
                   dtype=np.float32, format="csr") * 0.05
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ports = Ports.load(DATA / "ports.json", body_ids)
    return w0, ports, dev


@pytest.fixture
def policy(small):
    w0, ports, dev = small
    core = BrainCore(w0, ports, n_brains=K, device=dev)
    return BrainPolicy(core, n_envs=B)


def test_step_shapes_and_tanh(policy):
    dev = policy.log_std.device
    state = policy.init_state()
    obs = torch.rand(B, 76, device=dev) * 2 - 1
    state, out = policy.step(state, obs)
    J = policy.act_dim
    assert out["u"].shape == (K, B, J)
    assert out["a"].shape == (K, B, J)
    assert out["logp"].shape == (K, B)
    assert out["mu"].shape == (K, B, J)
    assert out["cmd"].shape == (K, B, J)
    assert out["s"].shape == (policy.core.N, K * B)
    torch.testing.assert_close(out["a"], torch.tanh(out["u"]))
    torch.testing.assert_close(out["cmd"], torch.tanh(out["mu"]))


def test_pre_tanh_forward_matches_readout(policy):
    """core.forward(pre_tanh=True) の第 2 返り値が tanh 前で、readout と一致する。"""
    core, dev = policy.core, policy.log_std.device
    state = core.init_state(B)
    obs = torch.rand(K, B, 76, device=dev) * 2 - 1
    _, a = core(state, obs)
    _, mu = core(state, obs, pre_tanh=True)
    torch.testing.assert_close(a, torch.tanh(mu))


def test_logp_matches_normal(policy):
    dev = policy.log_std.device
    state = policy.init_state()
    obs = torch.rand(B, 76, device=dev) * 2 - 1
    _, out = policy.step(state, obs)
    std = policy.log_std.clamp(min=LOG_STD_MIN).exp().unsqueeze(1)
    ref = Normal(out["mu"], std.expand_as(out["mu"])).log_prob(out["u"]).sum(-1)
    torch.testing.assert_close(out["logp"], ref)
    torch.testing.assert_close(policy.log_prob(out["mu"], out["u"]), ref)


def test_std_floor(policy):
    """log_std を -10 にしても std は下限 0.1 に切り詰められる。"""
    with torch.no_grad():
        policy.log_std.fill_(-10.0)
    mu = torch.zeros(K, B, policy.act_dim, device=policy.log_std.device)
    d = policy.dist(mu)
    assert d.scale.min().item() == pytest.approx(0.1)


def test_backward_reaches_all_params(policy):
    dev = policy.log_std.device
    policy.zero_grad(set_to_none=True)
    state = policy.init_state()
    loss = 0.0
    for _ in range(4):
        obs = torch.rand(B, 76, device=dev) * 2 - 1
        state, out = policy.step(state, obs)
        loss = loss + (out["a"] ** 2).mean()
    loss.backward()
    params = {"raw_g": policy.core.raw_g, "b": policy.core.b,
              "raw_tau_m": policy.core.raw_tau_m, "raw_tau_s": policy.core.raw_tau_s,
              "in_gain": policy.core.in_gain, "out_w": policy.core.out_w,
              "log_std": policy.log_std}
    for name, p in params.items():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_reset_state_only_masked_envs(policy):
    dev = policy.log_std.device
    state = policy.init_state()
    for _ in range(4):
        obs = torch.rand(B, 76, device=dev) * 2 - 1
        state, _ = policy.step(state, obs)
    before = tuple(x.detach().clone() for x in state)
    mask = torch.tensor([True, False, True], device=dev)
    new_state = policy.reset_state(state, mask)
    expected = (0.0, 0.0, 1.0 / (1.0 + math.exp(1.0 / BETA)))     # v, I, s
    for i, (new, old) in enumerate(zip(new_state, before)):
        nv, ov = new.view(-1, K, B), old.view(-1, K, B)
        for b in (0, 2):                                        # mask の環境は初期値
            assert torch.all(nv[:, :, b] == expected[i])
        assert torch.equal(nv[:, :, 1], ov[:, :, 1])            # 他の環境は不変
    # 元の state は書き換えられていない（in-place でない）
    for new, old in zip(state, before):
        assert torch.equal(new, old)
    # 勾配の履歴は残る（reset 後も backward が届く）
    assert all(x.grad_fn is not None for x in new_state)


def test_applied_action(small):
    w0, ports, dev = small
    J = len(ports.joints)
    pol1 = BrainPolicy(BrainCore(w0, ports, n_brains=1, device=dev), n_envs=B)
    a1 = torch.randn(1, B, J, device=dev)
    torch.testing.assert_close(pol1.applied_action(a1), a1[0])
    pol2 = BrainPolicy(BrainCore(w0, ports, n_brains=2, device=dev), n_envs=B)
    a2 = torch.randn(2, B, J, device=dev)
    out = pol2.applied_action(a2)
    assert out.shape == (B, J)
    torch.testing.assert_close(out, a2.mean(0))


def test_param_groups(policy):
    groups = policy.param_groups()
    grouped = [p for g in groups for p in g["params"]]
    all_params = list(policy.parameters())
    # 全パラメータをちょうど 1 回ずつ含む
    assert len(grouped) == len(all_params)
    assert {id(p) for p in grouped} == {id(p) for p in all_params}
    # §7 の学習率: 中核 1e-4、ポートと読出し（log_std 含む）3e-4
    core_ids = {id(p) for p in (policy.core.raw_g, policy.core.b,
                                policy.core.raw_tau_m, policy.core.raw_tau_s)}
    port_ids = {id(p) for p in (policy.core.in_gain, policy.core.out_w,
                                policy.log_std)}
    for g in groups:
        ids = {id(p) for p in g["params"]}
        if ids == core_ids:
            assert g["lr"] == pytest.approx(1e-4)
        elif ids == port_ids:
            assert g["lr"] == pytest.approx(3e-4)
        else:
            pytest.fail(f"未知のグループ: {ids}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA が無い")
def test_full_scale_build_and_backward():
    if not (DATA / "graph.npz").exists():
        pytest.skip("data/graph.npz が無い")
    core, info = build_core(1)
    assert info["N"] == 71618
    assert info["nnz"] > 7.8e6
    assert info["c"] == pytest.approx(0.0207, rel=0.2)
    pol = BrainPolicy(core, n_envs=8)
    dev = core.w0.device
    state = pol.init_state()
    loss = 0.0
    for _ in range(4):
        obs = torch.rand(8, 76, device=dev) * 2 - 1
        state, out = pol.step(state, obs)
        loss = loss + (out["a"] ** 2).mean()
    loss.backward()
    for name, p in {"raw_g": core.raw_g, "b": core.b, "raw_tau_m": core.raw_tau_m,
                    "raw_tau_s": core.raw_tau_s, "in_gain": core.in_gain,
                    "out_w": core.out_w, "log_std": pol.log_std}.items():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    s = out["s"].detach()
    s_mean = float(s.mean())
    assert 0.01 <= s_mean <= 0.2, f"s_mean={s_mean}"
    assert float((s > 0.99).float().mean()) < 0.01
    print(f"\nfull-scale: N={info['N']} nnz={info['nnz']} "
          f"c={info['c']:.5f} s_mean={s_mean:.4f}")


def test_other_env_count_and_forward_mu(policy):
    """評価のように別の環境数で同じ方策を使えること、forward_mu が step の
    mu と一致し標本化しないこと。"""
    dev = policy.log_std.device
    B2 = policy.B + 2
    state = policy.init_state(B2)
    obs = torch.rand(B2, policy.obs_dim, device=dev) * 2 - 1
    st1, out = policy.step(state, obs)
    assert out["mu"].shape == (policy.K, B2, policy.act_dim)
    st2, mu = policy.forward_mu(state, obs)
    torch.testing.assert_close(mu, out["mu"])
    for a, b in zip(st1, st2):
        torch.testing.assert_close(a, b)
    mask = torch.zeros(B2, dtype=torch.bool, device=dev)
    mask[0] = True
    v, I, s = policy.reset_state(st2, mask)
    assert v.shape == (policy.core.N, policy.K * B2)
    # リセットした環境の列（k·B2+0）だけが初期値
    for k in range(policy.K):
        assert float(v[:, k * B2].abs().max()) == 0.0
    assert float(v[:, 1].abs().max()) > 0.0
