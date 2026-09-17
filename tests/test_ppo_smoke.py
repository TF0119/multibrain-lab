"""Smoke test for the PPO learner on WarpBodyEnv (PLAN.md §7).

nworld=8, collect_len=16: one update must run end to end with a finite
loss and actually move the policy parameters, and a save/load roundtrip
must reproduce identical outputs. Needs mujoco_warp on CUDA; skips
otherwise, same as test_env.py.
"""

import math
from pathlib import Path

import pytest
import torch

wp = pytest.importorskip("warp", reason="warp not installed")
mujoco_warp = pytest.importorskip("mujoco_warp",
                                  reason="mujoco_warp not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device for mujoco_warp", allow_module_level=True)

from multibrain.body.env import WarpBodyEnv
from multibrain.learning import PPO, PPOConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

NWORLD = 8
COLLECT_LEN = 16


@pytest.fixture(scope="module")
def env():
    e = WarpBodyEnv(nworld=NWORLD, seed=0)
    yield e
    e.close()


def test_one_update(env):
    ppo = PPO(env, cfg=PPOConfig(collect_len=COLLECT_LEN))
    before = [p.detach().clone() for p in ppo.policy.parameters()]
    stats = ppo.update()

    assert ppo.body_steps == COLLECT_LEN * NWORLD
    assert ppo.updates == 1
    for key in ("loss", "pg_loss", "v_loss", "entropy", "clip_frac",
                "mean_reward", "steps_per_s", "success_rate",
                "max_consecutive", "gpu_mem_alloc_mb",
                "gpu_mem_reserved_mb"):
        assert key in stats
        assert math.isfinite(stats[key]), key
    assert len(stats["terms"]) == 9
    assert all(math.isfinite(v) for v in stats["terms"].values())
    changed = [not torch.equal(b, p.detach())
               for b, p in zip(before, ppo.policy.parameters())]
    assert any(changed), "policy parameters did not update"


def test_save_load_roundtrip(env, tmp_path):
    ppo = PPO(env, cfg=PPOConfig(collect_len=COLLECT_LEN))
    ppo.update()
    ckpt = tmp_path / "ckpt.pt"
    ppo.save(ckpt)

    ppo2 = PPO.load(ckpt, env)
    assert ppo2.body_steps == ppo.body_steps
    assert ppo2.updates == ppo.updates

    gen = torch.Generator(device="cuda").manual_seed(0)
    obs = torch.rand(4, env.obs_dim, generator=gen, device="cuda") * 4 - 2
    act = torch.rand(4, env.act_dim, generator=gen, device="cuda") * 2 - 1
    with torch.no_grad():
        nobs1, nobs2 = ppo.norm.normalize(obs), ppo2.norm.normalize(obs)
        torch.testing.assert_close(nobs1, nobs2)
        torch.testing.assert_close(ppo.policy.act_deterministic(nobs1),
                                   ppo2.policy.act_deterministic(nobs2))
        torch.testing.assert_close(ppo.value(nobs1, act),
                                   ppo2.value(nobs1, act))


def test_buffer_logp_matches_stored_obs(env):
    """The buffer keeps the normalized obs the policy sampled from, so
    re-evaluating log_prob on it reproduces logp_old exactly (ratio = 1
    before any optimizer step)."""
    ppo = PPO(env, cfg=PPOConfig(collect_len=COLLECT_LEN))
    buf, _ = ppo.collect()
    T, B = COLLECT_LEN, NWORLD
    with torch.no_grad():
        logp = ppo.policy.log_prob(buf["obs"].reshape(T * B, -1),
                                   buf["u"].reshape(T * B, -1))
    torch.testing.assert_close(logp, buf["logp"].reshape(T * B),
                               rtol=0, atol=1e-5)
    # stored obs are the normalized ones: within the clip range
    assert buf["obs"].abs().max() <= ppo.norm.clip
