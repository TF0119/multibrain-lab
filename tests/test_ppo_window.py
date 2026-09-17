"""WindowPPO（切り詰め窓 BPTT の PPO、PLAN §7）の検証。

小さな乱数配線（test_brain_policy.py と同じ作り方）と本物の WarpBodyEnv で、
nworld=8、window=4、collect_len=8 で軽く回す。最重要の試験は、収集直後に
窓の開始状態から再計算した logp が buf["logp"] を再現すること（窓の中で
リセットが起きた場合も含む）。CUDA と data/ が無ければ skip。
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

wp = pytest.importorskip("warp", reason="warp not installed")
mujoco_warp = pytest.importorskip("mujoco_warp",
                                  reason="mujoco_warp not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device for mujoco_warp", allow_module_level=True)

from multibrain.body.env import WarpBodyEnv
from multibrain.brain.core import BrainCore, Ports
from multibrain.brain.policy import BrainPolicy
from multibrain.learning import WindowPPO, WindowPPOConfig

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(
    not (DATA / "ports.json").exists(),
    reason="data/ports.json が無い")

NWORLD = 8
WINDOW = 4
COLLECT_LEN = 8


@pytest.fixture(scope="module")
def env():
    e = WarpBodyEnv(nworld=NWORLD, seed=0)
    yield e
    e.close()


@pytest.fixture(scope="module")
def wiring():
    """ports の全ニューロンを含む最小の乱数配線 (w0, ports)。"""
    d = json.loads((DATA / "ports.json").read_text())
    ids = sorted({i for p in d["inputs"] + d["outputs"] for i in p["neurons"]})
    body_ids = np.array(ids, dtype=np.int64)
    n = len(body_ids)
    rng = np.random.default_rng(0)
    w0 = sp.random(n, n, density=20 / n, random_state=rng,
                   dtype=np.float32, format="csr") * 0.05
    ports = Ports.load(DATA / "ports.json", body_ids)
    return w0, ports


@pytest.fixture
def policy(wiring):
    w0, ports = wiring
    core = BrainCore(w0, ports, n_brains=1, device="cuda")
    return BrainPolicy(core, n_envs=NWORLD)


def _make(env, policy, **kw):
    cfg = WindowPPOConfig(window=WINDOW, collect_len=COLLECT_LEN, **kw)
    return WindowPPO(env, policy, cfg=cfg)


def test_one_update(env, policy):
    ppo = _make(env, policy)
    stats = ppo.update()

    assert ppo.body_steps == COLLECT_LEN * NWORLD
    assert ppo.updates == 1
    for key in ("loss", "pg_loss", "v_loss", "entropy", "clip_frac",
                "mean_reward", "steps_per_s", "success_rate",
                "max_consecutive", "collect_s", "update_s",
                "gpu_mem_alloc_mb", "gpu_mem_reserved_mb",
                "core_grad_norm", "ports_grad_norm"):
        assert key in stats, key
        assert math.isfinite(stats[key]), key
    assert len(stats["terms"]) == 10
    assert all(math.isfinite(v) for v in stats["terms"].values())
    # 中核にもポートにも勾配が届いている
    assert stats["core_grad_norm"] > 0.0
    assert stats["ports_grad_norm"] > 0.0


def test_replay_reproduces_logp(env, policy):
    """収集直後、窓の開始状態から再計算した logp が buf["logp"] と一致する。

    episode_steps=3 で両方の窓の中 (t=2, t=5) にリセットを起こし、
    再計算が buf["end"] のマスクで同じ位置に状態を戻すことも確かめる。
    """
    ppo = _make(env, policy)
    orig_steps = env.episode_steps
    env.episode_steps = WINDOW - 1
    try:
        ppo._obs = env.reset()              # episode_step = 0 に揃える
        ppo._state = policy.init_state(NWORLD)
        buf, _ = ppo.collect()
    finally:
        env.episode_steps = orig_steps

    # t=2（窓 0 の中）と t=5（窓 1 の中）で全環境が done になる
    assert buf["end"][WINDOW - 2].all()
    assert buf["end"][2 * WINDOW - 3].all()

    for ckpt in (False, True):
        ppo.cfg.checkpoint = ckpt
        with torch.no_grad():
            for w in range(COLLECT_LEN // WINDOW):
                mus = ppo._window_mus(buf, w)
                for i, mu in enumerate(mus):
                    t = w * WINDOW + i
                    logp = ppo.policy.log_prob(mu, buf["u"][t])[0]
                    torch.testing.assert_close(
                        logp, buf["logp"][t], rtol=0.0, atol=1e-4)
    # 窓の開始状態は detach 済み（前の窓へ勾配を流さない）
    assert all(not x.requires_grad
               for st in buf["win_state"] for x in st)


def test_update_moves_all_params(env, policy):
    """1 更新で中核 4 パラメータと in_gain、out_w、log_std がすべて動く。"""
    ppo = _make(env, policy)
    named = {"raw_g": policy.core.raw_g, "b": policy.core.b,
             "raw_tau_m": policy.core.raw_tau_m,
             "raw_tau_s": policy.core.raw_tau_s,
             "in_gain": policy.core.in_gain, "out_w": policy.core.out_w,
             "log_std": policy.log_std}
    before = {k: p.detach().clone() for k, p in named.items()}
    ppo.update()
    for k, p in named.items():
        assert not torch.equal(before[k], p.detach()), \
            f"{k} が更新されていない"


def test_save_load_roundtrip(env, wiring, tmp_path):
    w0, ports = wiring
    pol1 = BrainPolicy(BrainCore(w0, ports, n_brains=1, device="cuda"),
                       n_envs=NWORLD)
    ppo = _make(env, pol1)
    ppo.update()
    ckpt = tmp_path / "ckpt.pt"
    ppo.save(ckpt)

    pol2 = BrainPolicy(BrainCore(w0, ports, n_brains=1, device="cuda"),
                       n_envs=NWORLD)
    ppo2 = WindowPPO.load(ckpt, env, pol2)
    assert ppo2.body_steps == ppo.body_steps
    assert ppo2.updates == ppo.updates

    gen = torch.Generator(device="cuda").manual_seed(0)
    obs = torch.rand(4, env.obs_dim, generator=gen, device="cuda") * 4 - 2
    with torch.no_grad():
        _, a1 = ppo.policy.act_deterministic(ppo.policy.init_state(4), obs)
        _, a2 = ppo2.policy.act_deterministic(ppo2.policy.init_state(4), obs)
    torch.testing.assert_close(a1, a2)
