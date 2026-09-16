import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from multibrain.body.build_xml import build_xml

REPO_ROOT = Path(__file__).resolve().parents[1]
JOINT_GROUPS = REPO_ROOT / "configs" / "joint_groups.yaml"


@pytest.fixture(scope="module")
def joints():
    with open(JOINT_GROUPS) as f:
        return yaml.safe_load(f)["joints"]


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(build_xml(JOINT_GROUPS))


def test_joint_count_and_groups(joints):
    assert len(joints) == 27
    names = [j["name"] for j in joints]
    assert len(set(names)) == 27, "duplicate joint names"
    groups = {}
    for j in joints:
        groups[j["group"]] = groups.get(j["group"], 0) + 1
    assert groups == {"left": 11, "center": 5, "right": 11}
    # no missing: every left joint has a right twin and vice versa
    for n in names:
        if n.endswith("_L") or "_L_" in n:
            assert n.replace("_L", "_R") in names or n.replace("_L_", "_R_") in names


def test_model_structure(model, joints):
    assert model.nu == 27
    hinge = sum(1 for t in model.jnt_type if t == mujoco.mjtJoint.mjJNT_HINGE)
    assert hinge == 27
    act_names = {model.actuator(i).name for i in range(model.nu)}
    assert act_names == {j["name"] for j in joints}


def test_total_mass(model):
    d = mujoco.MjData(model)
    mujoco.mj_setConst(model, d)
    assert model.body_mass.sum() == pytest.approx(50.0, abs=0.5)


def test_sensor_counts(model):
    # 27 jointpos + 27 jointvel + framequat/gyro/velocimeter/framepos + 12 touch
    assert model.nsensor == 27 + 27 + 4 + 12
    counts = {}
    for i in range(model.nsensor):
        counts[model.sensor_type[i]] = counts.get(model.sensor_type[i], 0) + 1
    assert counts[mujoco.mjtSensor.mjSENS_JOINTPOS] == 27
    assert counts[mujoco.mjtSensor.mjSENS_JOINTVEL] == 27
    assert counts[mujoco.mjtSensor.mjSENS_TOUCH] == 12
    for t in (mujoco.mjtSensor.mjSENS_FRAMEQUAT, mujoco.mjtSensor.mjSENS_GYRO,
              mujoco.mjtSensor.mjSENS_VELOCIMETER, mujoco.mjtSensor.mjSENS_FRAMEPOS):
        assert counts[t] == 1


def _lowest_geom_z(model, d):
    lo = math.inf
    for g in range(1, model.ngeom):  # skip floor plane
        R = d.geom_xmat[g].reshape(3, 3)
        half = model.geom_aabb[g][3:]
        ext = np.abs(R) @ half
        lo = min(lo, d.geom_xpos[g, 2] - ext[2])
    return lo


def _posed(model, joint_name, angle):
    """Kinematics only: set one hinge to `angle` rad, no dynamics."""
    d = mujoco.MjData(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    d.qpos[model.jnt_qposadr[jid]] = angle
    mujoco.mj_kinematics(model, d)
    return d


def _body_pos(model, d, name):
    return d.xpos[model.body(name).id].copy()


def _toe(model, d, side):
    """World position of the +x (toe) tip of the foot geom."""
    body_id = model.body(f"foot_{side}").id
    gid = next(g for g in range(model.ngeom)
               if model.geom_bodyid[g] == body_id)
    R = d.geom_xmat[gid].reshape(3, 3)
    return d.geom_xpos[gid] + R @ np.array([model.geom_size[gid][0], 0.0, 0.0])


def test_joint_axis_signs(model):
    """Positive joint rotation must move the limb in the anatomically
    positive direction (kinematics only, gravity off)."""
    d0 = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d0)

    # knee +1.0 rad: shank swings back -> foot x decreases
    d = _posed(model, "knee_L", 1.0)
    assert _body_pos(model, d, "foot_L")[0] < _body_pos(model, d0, "foot_L")[0]

    # hip_L_x +1.0: thigh raises forward -> shin x increases
    d = _posed(model, "hip_L_x", 1.0)
    assert _body_pos(model, d, "shin_L")[0] > _body_pos(model, d0, "shin_L")[0]

    # elbow_L +1.0: flexion brings the hand forward -> hand x increases
    d = _posed(model, "elbow_L", 1.0)
    assert _body_pos(model, d, "hand_L")[0] > _body_pos(model, d0, "hand_L")[0]

    # ankle_L_x +0.5: dorsiflexion -> toe tip z rises
    d = _posed(model, "ankle_L_x", 0.5)
    assert _toe(model, d, "L")[2] > _toe(model, d0, "L")[2]

    # shoulder_*_y +1.0: abduction — left hand +y, right hand -y
    d = _posed(model, "shoulder_L_y", 1.0)
    assert _body_pos(model, d, "hand_L")[1] > _body_pos(model, d0, "hand_L")[1]
    d = _posed(model, "shoulder_R_y", 1.0)
    assert _body_pos(model, d, "hand_R")[1] < _body_pos(model, d0, "hand_R")[1]

    # hip_*_z +0.5: mirrored rotation -> toe y displacements have opposite signs
    dL = _posed(model, "hip_L_z", 0.5)
    dR = _posed(model, "hip_R_z", 0.5)
    dy_L = _toe(model, dL, "L")[1] - _toe(model, d0, "L")[1]
    dy_R = _toe(model, dR, "R")[1] - _toe(model, d0, "R")[1]
    assert dy_L * dy_R < 0


def test_uncontrolled_fall(model):
    """§3.8: ctrl=0 from standing -> falls within 2 s; final state has no NaN
    and no geom resting below -0.02 m. The transient deepest point while
    falling is not a pass criterion."""
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    z0 = d.qpos[2]
    for _ in range(400):  # 2 s at 5 ms
        mujoco.mj_step(model, d)
    assert not np.isnan(d.qpos).any()
    assert _lowest_geom_z(model, d) > -0.02
    assert d.qpos[2] < 0.7 * z0
    # transient minimum is recorded by scripts/check_body.py, not asserted
