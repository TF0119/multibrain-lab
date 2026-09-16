"""Tests for the §5.2 success judgment (contacts / standing_now /
SuccessTracker) on CPU MuJoCo sensordata.

Physical poses are settled by the dynamics for ~1 s before judging, since
the touch forces and velocities only become meaningful once the body rests
on the floor. Pose quaternions come from scripts/check_body.py POSES.
"""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from multibrain.body.layout import BodyLayout
from multibrain.body.success import (SuccessConfig, SuccessTracker, contacts,
                                     standing_now, tilt_cos)

REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "humanoid.xml"
JOINT_GROUPS = REPO_ROOT / "configs" / "joint_groups.yaml"
TASK_YAML = REPO_ROOT / "configs" / "task.yaml"
REWARD_YAML = REPO_ROOT / "configs" / "reward.yaml"

SUBSTEPS = 4  # physics steps per 20 ms body step (configs/task.yaml)

# pelvis freejoint quats (w, x, y, z) and start heights; same values as
# scripts/check_body.py POSES (supine = -90 deg about y, prone = +90 deg,
# side = +90 deg about x)
POSES = {
    "standing": ([1.0, 0.0, 0.0, 0.0], 0.84),
    "supine":   ([math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0], 0.20),
    "prone":    ([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0], 0.20),
    "side":     ([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0], 0.20),
}


@pytest.fixture(scope="module")
def mjm():
    return mujoco.MjModel.from_xml_path(str(XML_PATH))


@pytest.fixture(scope="module")
def layout(mjm):
    return BodyLayout.from_model(mjm, JOINT_GROUPS)


@pytest.fixture(scope="module")
def scfg():
    return SuccessConfig.from_yaml(TASK_YAML, REWARD_YAML)


def _sd(d):
    """(1, nsensordata) float32 tensor from a stepped MjData."""
    return torch.as_tensor(np.asarray(d.sensordata),
                           dtype=torch.float32).unsqueeze(0)


def _make(mjm, pose, z=None, joints_deg=None):
    """MjData placed in one of POSES, optionally with joint angles (deg)."""
    d = mujoco.MjData(mjm)
    quat, z0 = POSES[pose]
    d.qpos[2] = z0 if z is None else z
    d.qpos[3:7] = quat
    for name, deg in (joints_deg or {}).items():
        jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[mjm.jnt_qposadr[jid]] = math.radians(deg)
    mujoco.mj_forward(mjm, d)
    return d


def _settle(d, mjm, seconds):
    for _ in range(round(seconds / 0.005)):
        mujoco.mj_step(mjm, d)


def _col(layout, name):
    return layout.touch_names.index(name)


def test_config_from_yaml(scfg):
    """hold_steps/milestones are seconds / body_step_s; the tilt threshold is
    cos(15 deg); force and ratio thresholds come straight from the yamls."""
    assert scfg.hold_steps == 500                      # 10 s / 0.02 s
    assert scfg.tilt_cos_min == pytest.approx(math.cos(math.radians(15.0)))
    assert scfg.height_ratio_min == 0.9
    assert scfg.horizontal_speed_max == 0.2
    assert scfg.contact_force_min == 1.0
    assert scfg.milestone_steps == [50, 150, 500]      # 1 s / 3 s / 10 s


def test_standing_rest_is_standing(mjm, layout, scfg):
    """Standing rest + ctrl=0 for 0.1 s (20 physics steps): both feet on the
    floor, nothing else touching, torso upright, pelvis at ~h_ref and
    ~stationary -> standing_now True."""
    d = _make(mjm, "standing")
    _settle(d, mjm, 0.1)
    s = _sd(d)

    c = contacts(layout, s, scfg.contact_force_min)[0]
    feet = {n: c[_col(layout, n)].item()
            for n in ("touch_foot_L", "touch_foot_R")}
    assert all(feet.values())
    nonfoot = [i for i, n in enumerate(layout.touch_names)
               if not n.startswith("touch_foot")]
    assert not c[nonfoot].any()

    assert tilt_cos(layout, s).item() == pytest.approx(1.0, abs=1e-2)
    h = s[0, layout.sensor["framepos_imu"].start + 2].item()
    assert h / layout.h_ref == pytest.approx(1.0, abs=0.05)
    hspeed = s[0, layout.sensor["framelinvel_imu"]].numpy()[:2]
    assert np.linalg.norm(hspeed) == pytest.approx(0.0, abs=0.05)

    assert standing_now(layout, scfg, s).item() is True


def test_contacts_threshold_and_order(mjm, layout, scfg):
    """contacts() is (B, 12) bool in TOUCH_SITES order, True iff force >=
    force_min (boundary inclusive)."""
    d = _make(mjm, "standing")
    _settle(d, mjm, 0.1)
    s = _sd(d).clone()
    fmin = scfg.contact_force_min

    c = contacts(layout, s, fmin)
    assert c.shape == (1, 12) and c.dtype == torch.bool
    assert c[0, _col(layout, "touch_foot_L")]
    assert not c[0, _col(layout, "touch_hand_L")]

    hand = layout.sensor["touch_hand_L"].start
    s[0, hand] = fmin - 1e-3
    assert not contacts(layout, s, fmin)[0, _col(layout, "touch_hand_L")]
    s[0, hand] = fmin
    assert contacts(layout, s, fmin)[0, _col(layout, "touch_hand_L")]


@pytest.mark.parametrize("pose,expected_any", [
    # measured on this body: supine settles on feet + pelvis, prone on
    # feet + knees + hands + pelvis, side on one foot + both hands
    ("supine", ["touch_back", "touch_pelvis", "touch_head"]),
    ("prone", ["touch_chest", "touch_pelvis", "touch_knee_L",
               "touch_knee_R"]),
    ("side", ["touch_hand_L", "touch_hand_R", "touch_elbow_L",
              "touch_elbow_R", "touch_knee_L", "touch_knee_R",
              "touch_pelvis", "touch_chest", "touch_back", "touch_head"]),
])
def test_lying_poses_are_not_standing(mjm, layout, scfg, pose, expected_any):
    """Supine / prone / side after 1 s of settling: standing_now False and
    at least one non-foot site touches (supine: back/pelvis/head, prone:
    chest/pelvis/knee, side: any non-foot site)."""
    d = _make(mjm, pose)
    _settle(d, mjm, 1.0)
    s = _sd(d)

    assert standing_now(layout, scfg, s).item() is False
    c = contacts(layout, s, scfg.contact_force_min)[0]
    nonfoot = [i for i, n in enumerate(layout.touch_names)
               if not n.startswith("touch_foot")]
    assert c[nonfoot].any()
    assert any(c[_col(layout, n)] for n in expected_any)


def test_kneel_is_not_standing(mjm, layout, scfg):
    """Kneeling: hips flexed 18 deg, knees 95 deg, pelvis dropped to 0.46 m.
    On this body the pose settles onto both knees + both feet (measured:
    h ~ 0.41, tilt ~ 45 deg, contacts hold through 1.5 s of settling), so
    standing_now is False while the knee sites carry load."""
    d = _make(mjm, "standing", z=0.46, joints_deg={
        "hip_L_x": 18.0, "hip_R_x": 18.0, "knee_L": 95.0, "knee_R": 95.0})
    _settle(d, mjm, 1.5)
    s = _sd(d)

    assert standing_now(layout, scfg, s).item() is False
    c = contacts(layout, s, scfg.contact_force_min)[0]
    assert c[_col(layout, "touch_knee_L")]
    assert c[_col(layout, "touch_knee_R")]


def test_success_tracker_counts_and_resets(scfg):
    """499 consecutive standing steps: no success. The 500th sets
    success_now + first_success; the 501st keeps success_now but clears
    first_success. Milestones fire exactly once at 50 / 150 / 500.
    reset(mask) zeroes only the masked env's counters."""
    tr = SuccessTracker(scfg, n_envs=2, device="cpu")
    milestones = torch.zeros(2, 3, dtype=torch.bool)

    for i in range(1, 499):
        out = tr.update(torch.ones(2, dtype=torch.bool))
        assert not out["success_now"].any()
        assert not out["first_success"].any()
        milestones |= out["milestone"]

    out = tr.update(torch.ones(2, dtype=torch.bool))       # step 499
    assert not out["success_now"].any()
    milestones |= out["milestone"]

    out = tr.update(torch.ones(2, dtype=torch.bool))       # step 500
    assert out["success_now"].all()
    assert out["first_success"].all()
    milestones |= out["milestone"]

    out = tr.update(torch.ones(2, dtype=torch.bool))       # step 501
    assert out["success_now"].all()
    assert not out["first_success"].any()

    # each milestone fired exactly once, at 50 / 150 / 500
    assert milestones.all()
    tr2 = SuccessTracker(scfg, n_envs=1, device="cpu")
    hits = [[] for _ in range(3)]
    for i in range(1, 502):
        m = tr2.update(torch.ones(1, dtype=torch.bool))["milestone"][0]
        for k in range(3):
            if m[k]:
                hits[k].append(i)
    assert hits == [[50], [150], [500]]

    # a False step breaks the streak
    out = tr2.update(torch.zeros(1, dtype=torch.bool))
    assert tr2.consecutive.item() == 0
    assert not out["success_now"].any()

    # reset(mask) affects only the masked env
    tr3 = SuccessTracker(scfg, n_envs=2, device="cpu")
    for _ in range(600):
        tr3.update(torch.ones(2, dtype=torch.bool))
    tr3.reset(torch.tensor([True, False]))
    assert tr3.consecutive.tolist() == [0, 600]
    assert tr3.achieved.tolist() == [False, True]
    out = tr3.update(torch.ones(2, dtype=torch.bool))
    assert out["success_now"].tolist() == [False, True]
    # env 1 already achieved: its first_success stays cleared
    assert out["first_success"].tolist() == [False, False]
