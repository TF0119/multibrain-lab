"""Tests for the 76-dim observation vector (PLAN.md §3.5) and for
mujoco_warp vs CPU MuJoCo sensordata parity, reward sensors included.

CPU-side tests build sensordata from a real MjData and run it through
observe(); the last test steps both engines on the same control
sequence and compares raw sensordata.
"""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from multibrain.body.layout import BodyLayout
from multibrain.body.observation import OBS_DIM, OBS_SLICES, observe

REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "humanoid.xml"
JOINT_GROUPS = REPO_ROOT / "configs" / "joint_groups.yaml"

# Pelvis freejoint orientation for lying flat on the back (face-up).
# scripts/check_body.py calls the *negative* pitch pose "prone", but
# measured against the model it is the supine one: with this quat the
# touch_chest site ends up at z=0.3 and touch_back at z=0.1 (chest up,
# back down), and the pelvis-frame gravity reads exactly (-1,0,0),
# i.e. gravity pulls toward the back. The +sin(π/4) "supine" pose in
# check_body.py is the mirror image (face-down, gravity +x), so the
# pose labels in that script are swapped; this test uses the pose that
# actually lies on the back, as the spec's (-1,0,0) requires.
SUPINE_QUAT = [math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0]


@pytest.fixture(scope="module")
def mjm():
    return mujoco.MjModel.from_xml_path(str(XML_PATH))


@pytest.fixture(scope="module")
def layout(mjm):
    return BodyLayout.from_model(mjm, JOINT_GROUPS)


def _observe1(layout, d):
    """observe() on a single MjData, returned as a (76,) tensor."""
    s = torch.as_tensor(np.asarray(d.sensordata), dtype=torch.float32)
    return observe(layout, s.unsqueeze(0))[0]


def test_obs_slices_tile_obs_dim():
    assert OBS_DIM == 76
    covered = np.zeros(OBS_DIM, dtype=bool)
    for sl in OBS_SLICES.values():
        covered[sl] = True
    assert covered.all()


def test_standing_rest(mjm, layout):
    """Standing rest right after mj_forward: normalized joints inside
    [-1,1], gravity straight down in the pelvis frame, height = h_ref,
    both feet loaded (~245 N each, i.e. half the weight; clamps to 1.0
    after the /50 N normalization) and no other touch site firing.

    The limb capsules end one radius short of their child joint (see
    build_xml.py), so the foot boxes are the lowest geoms and carry the
    weight; with the earlier geometry the shin tips penetrated the floor
    and the feet read 0.
    """
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    obs = _observe1(layout, d)
    assert obs.shape == (OBS_DIM,)

    joints = obs[OBS_SLICES["jointpos"]]
    assert joints.min() >= -1.0 - 1e-5 and joints.max() <= 1.0 + 1e-5

    np.testing.assert_allclose(
        obs[OBS_SLICES["gravity"]].numpy(), [0.0, 0.0, -1.0], atol=1e-5)
    assert obs[OBS_SLICES["height"]].item() == pytest.approx(1.0, abs=1e-5)
    touch = obs[OBS_SLICES["touch"]]
    feet = [layout.touch_names.index(n) for n in ("touch_foot_L", "touch_foot_R")]
    others = [i for i in range(len(layout.touch_names)) if i not in feet]
    assert (touch[feet] == 1.0).all()      # ~245 N each, clamped
    assert (touch[others] == 0).all()


def test_supine_rest(mjm, layout):
    """Lying flat on the back: drop the pelvis 1 s and let it settle.

    The pose settles on its own (damped joints, measured: pelvis z =
    0.15 m and the pitch quat unchanged after 1 s), so no initial joint
    angles are needed. Gravity must read (-1,0,0) in the pelvis frame:
    body +x (the front) points world +z (face up), so world -z maps to
    body -x — the pull is toward the back, i.e. the body rests supine.
    Verified numerically: sensordata gives exactly (-1, 0, 0).
    """
    d = mujoco.MjData(mjm)
    d.qpos[2] = 0.20
    d.qpos[3:7] = SUPINE_QUAT
    mujoco.mj_forward(mjm, d)
    for _ in range(200):  # 1 s at the 5 ms timestep
        mujoco.mj_step(mjm, d)

    obs = _observe1(layout, d)
    np.testing.assert_allclose(
        obs[OBS_SLICES["gravity"]].numpy(), [-1.0, 0.0, 0.0], atol=1e-2)

    # at least one dorsal touch site must fire (measured: touch_pelvis
    # carries ~240 N -> clamps to 1.0; touch_back and touch_head stay 0
    # because the torso capsule and head rest above the floor plane)
    touch = obs[OBS_SLICES["touch"]]
    dorsal = [layout.touch_names.index(n)
              for n in ("touch_back", "touch_pelvis", "touch_head")]
    assert touch[dorsal].max() > 0


def test_random_poses_bounded(mjm, layout):
    """20 random valid poses: the bounded obs slices stay inside their
    ranges (angles [-1,1], velocities [-3,3], touch [0,1])."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = mujoco.MjData(mjm)
        d.qpos[:] = mjm.qpos0
        d.qpos[0] = rng.uniform(-0.5, 0.5)
        d.qpos[1] = rng.uniform(-0.5, 0.5)
        d.qpos[2] = rng.uniform(0.1, 1.2)
        u = rng.normal(size=4)
        d.qpos[3:7] = u / np.linalg.norm(u)
        d.qpos[layout.qpos_adr] = rng.uniform(
            layout.range_lo, layout.range_hi)
        # +/-40 rad/s exceeds the 10*3 = 30 rad/s clamp point on purpose
        # so the jointvel clamp is exercised, not just the linear range
        d.qvel[layout.dof_adr] = rng.uniform(-40.0, 40.0, 27)
        d.qvel[:6] = rng.uniform(-3.0, 3.0, 6)
        mujoco.mj_forward(mjm, d)

        obs = _observe1(layout, d)
        assert obs.shape == (OBS_DIM,)
        jp = obs[OBS_SLICES["jointpos"]]
        assert jp.min() >= -1.0 - 1e-5 and jp.max() <= 1.0 + 1e-5
        jv = obs[OBS_SLICES["jointvel"]]
        assert jv.min() >= -3.0 and jv.max() <= 3.0
        tc = obs[OBS_SLICES["touch"]]
        assert tc.min() >= 0.0 and tc.max() <= 1.0


def test_batched_matches_single(mjm, layout):
    """A (B, nsensordata) batch gives the same rows as one-at-a-time."""
    rng = np.random.default_rng(1)
    rows = []
    singles = []
    for _ in range(3):
        d = mujoco.MjData(mjm)
        d.qpos[:] = mjm.qpos0
        d.qpos[2] = rng.uniform(0.2, 1.0)
        u = rng.normal(size=4)
        d.qpos[3:7] = u / np.linalg.norm(u)
        d.qpos[layout.qpos_adr] = rng.uniform(
            layout.range_lo, layout.range_hi)
        mujoco.mj_forward(mjm, d)
        s = torch.as_tensor(np.asarray(d.sensordata), dtype=torch.float32)
        rows.append(s)
        singles.append(_observe1(layout, d))
    batched = observe(layout, torch.stack(rows))
    assert batched.shape == (3, OBS_DIM)
    for i in range(3):
        torch.testing.assert_close(batched[i], singles[i])


def test_mujoco_warp_matches_cpu(mjm, layout):
    """mujoco_warp accepts the XML and its sensordata matches CPU MuJoCo.

    nworld=4, 10 body steps x 4 physics steps; each body step draws a
    random ctrl applied identically to all 4 worlds and to the CPU
    rollout, then raw sensordata must agree at rtol=1e-3 / atol=1e-3
    (the loaded feet read ~150-300 N of touch force, where the two
    solvers differ at ~1e-4 relative).

    The ctrl amplitude is +/-0.1 rather than the full +/-1 ctrlrange on
    purpose: with +-1 the feet re-tap the floor around body step ~4 and
    the two engines' contact solutions diverge chaotically (measured:
    47 N on touch_foot_R, 2.6 rad/s on jointvel after 10 steps). +-0.1
    keeps the feet planted without slipping, so the comparison tests
    the sensor computation itself rather than contact chaos.
    """
    wp = pytest.importorskip("warp", reason="warp not installed")
    mujoco_warp = pytest.importorskip(
        "mujoco_warp", reason="mujoco_warp not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device for mujoco_warp")
    wp.init()

    mjd0 = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd0)

    m = mujoco_warp.put_model(mjm)
    d = mujoco_warp.put_data(mjm, mjd0, nworld=4)

    cpu = mujoco.MjData(mjm)
    cpu.qpos[:] = mjd0.qpos
    cpu.qvel[:] = mjd0.qvel
    cpu.ctrl[:] = mjd0.ctrl
    mujoco.mj_forward(mjm, cpu)

    rng = np.random.default_rng(0)
    for _ in range(10):
        ctrl = rng.uniform(-0.1, 0.1, mjm.nu).astype(np.float32)
        d.ctrl = wp.array(np.tile(ctrl, (4, 1)))
        cpu.ctrl[:] = ctrl
        for _ in range(4):
            mujoco_warp.step(m, d)
            mujoco.mj_step(mjm, cpu)
    wp.synchronize()

    warp_sd = wp.to_torch(d.sensordata).cpu().numpy().astype(np.float64)
    cpu_sd = np.asarray(cpu.sensordata)
    assert warp_sd.shape == (4, cpu_sd.shape[0])
    for w in range(4):
        np.testing.assert_allclose(
            warp_sd[w], cpu_sd, rtol=1e-3, atol=1e-3,
            err_msg=f"world {w} sensordata diverged from CPU")

    # the reward/success sensors (§3.5 second table) must actually be
    # computed on the Warp side, not left at zero
    reward_names = (
        ["framezaxis_torso", "framelinvel_imu"]
        + [f"framelinvel_{s}" for s in layout.impact_site_names]
        + ["framexaxis_touch_head"])
    for name in reward_names:
        sl = layout.sensor[name]
        assert np.abs(warp_sd[:, sl]).max() > 1e-2, \
            f"{name} reads all-zero in mujoco_warp"
