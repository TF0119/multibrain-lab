"""Tests for WarpBodyEnv and the start poses (PLAN.md §5.1, §10.2).

Everything here needs mujoco_warp on CUDA; the module skips otherwise.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

wp = pytest.importorskip("warp", reason="warp not installed")
mujoco_warp = pytest.importorskip("mujoco_warp",
                                  reason="mujoco_warp not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device for mujoco_warp", allow_module_level=True)

from multibrain.body.env import WarpBodyEnv
from multibrain.body.layout import BodyLayout
from multibrain.body.reward import TERMS
from multibrain.body.start_poses import eval_starts, start_qpos

REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "humanoid.xml"

NWORLD = 8


@pytest.fixture(scope="module")
def mjm():
    return mujoco.MjModel.from_xml_path(str(XML_PATH))


@pytest.fixture(scope="module")
def layout(mjm):
    return BodyLayout.from_model(mjm)


@pytest.fixture
def env():
    e = WarpBodyEnv(nworld=NWORLD, seed=0)
    yield e
    e.close()


def test_reset_obs_finite(env):
    obs = env.reset()
    assert obs.shape == (NWORLD, 76)
    assert obs.device.type == "cuda"
    assert torch.isfinite(obs).all()


def test_start_qpos_heights(layout):
    rng = np.random.default_rng(7)
    for kind in ("supine", "prone", "side"):
        for _ in range(4):
            q = start_qpos(layout, kind, rng)
            assert q.shape == (layout.nq,)
            assert q[2] < 0.5          # lying starts stay near the floor
    for _ in range(4):
        q = start_qpos(layout, "standing", rng)
        assert q[2] == pytest.approx(0.84, abs=0.03)
        # joints stay inside their ranges even with the +/-2 deg jitter
        assert (q[layout.qpos_adr] >= layout.range_lo - 1e-6).all()
        assert (q[layout.qpos_adr] <= layout.range_hi + 1e-6).all()


def test_env_start_heights(env, layout):
    """rise_and_stand resets put every world in a lying pose; a
    stand_balance env resets to ~0.84 m."""
    env.reset()
    z = env.qpos[:, 2].cpu().numpy()
    assert (z < 0.5).all()

    e2 = WarpBodyEnv(nworld=4, task="stand_balance", seed=1)
    try:
        e2.reset()
        z2 = e2.qpos[:, 2].cpu().numpy()
        np.testing.assert_allclose(z2, 0.84, atol=0.03)
    finally:
        e2.close()


def test_random_rollout_100_steps(env):
    obs = env.reset()
    gen = torch.Generator(device="cuda").manual_seed(0)
    info = None
    reward = None
    for _ in range(100):
        a = torch.rand((NWORLD, env.act_dim), generator=gen,
                       device="cuda") * 2.0 - 1.0
        obs, reward, done, info = env.step(a)
        assert torch.isfinite(obs).all()
        assert torch.isfinite(reward).all()
    assert reward.shape == (NWORLD,)
    assert set(info["terms"].keys()) == set(TERMS)
    assert len(info["terms"]) == 10
    assert not info["nonfinite"].any()
    assert not info["overflow"].any()


def test_masked_reset_keeps_other_worlds(env):
    env.reset()
    gen = torch.Generator(device="cuda").manual_seed(1)
    for _ in range(5):
        env.step(torch.rand((NWORLD, env.act_dim), generator=gen,
                            device="cuda") * 2.0 - 1.0)
    before = env.qpos.clone()

    mask = torch.zeros(NWORLD, dtype=torch.bool, device="cuda")
    mask[[0, 3, 5]] = True
    env.reset(mask)
    after = env.qpos

    # untouched worlds keep their exact state; reset worlds are back at a
    # lying start height (rise_and_stand starts are all on the floor)
    torch.testing.assert_close(after[~mask], before[~mask],
                               rtol=0.0, atol=0.0)
    assert (after[mask, 2] < 0.5).all()
    # reset worlds really did move (qvel was nonzero after stepping)
    assert not torch.equal(after[mask], before[mask])


def test_episode_timeout(env):
    env.episode_steps = 10
    env.reset()
    done = None
    info = None
    for i in range(10):
        _, _, done, info = env.step(
            torch.zeros(NWORLD, env.act_dim, device="cuda"))
        if i < 9:
            assert not done.any(), f"done early at step {i + 1}"
    assert done.all()
    assert info["time_out"].all()
    assert not info["nonfinite"].any()


def test_eval_starts(layout):
    starts = eval_starts(layout, n=24, seed=12345)
    assert len(starts) == 24
    counts = {}
    for kind, q in starts:
        counts[kind] = counts.get(kind, 0) + 1
        assert q.shape == (layout.nq,)
    assert counts == {"supine": 8, "prone": 8, "side": 8}

    again = eval_starts(layout, n=24, seed=12345)
    for (k1, q1), (k2, q2) in zip(starts, again):
        assert k1 == k2
        np.testing.assert_array_equal(q1, q2)
