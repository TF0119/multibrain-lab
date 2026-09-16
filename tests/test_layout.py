"""Tests for BodyLayout (PLAN.md §3.5): name-resolved sensordata indices."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from multibrain.body.layout import BodyLayout
from multibrain.body.build_xml import IMPACT_SITES, TOUCH_SITES

REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "humanoid.xml"
JOINT_GROUPS = REPO_ROOT / "configs" / "joint_groups.yaml"


@pytest.fixture(scope="module")
def mjm():
    return mujoco.MjModel.from_xml_path(str(XML_PATH))


@pytest.fixture(scope="module")
def joints():
    with open(JOINT_GROUPS) as f:
        return yaml.safe_load(f)["joints"]


@pytest.fixture(scope="module")
def layout(mjm):
    return BodyLayout.from_model(mjm, JOINT_GROUPS)


def _adr(mjm, name):
    sid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SENSOR, name)
    assert sid >= 0, name
    return int(mjm.sensor_adr[sid]), int(mjm.sensor_dim[sid])


def test_sensor_and_data_counts(mjm, layout):
    assert mjm.nsensor == 77
    # 27+27+4+3+3+3+12 observation sensors + 3+3+12+3 reward sensors
    assert mjm.nsensordata == 79 + 21 == 100
    assert layout.nsensordata == 100
    assert len(layout.sensor) == mjm.nsensor
    # slices tile [0, nsensordata) exactly, in sensor declaration order
    covered = np.zeros(mjm.nsensordata, dtype=bool)
    for sl in layout.sensor.values():
        covered[sl] = True
    assert covered.all()


def test_joint_layout_matches_yaml(layout, joints):
    names = [j["name"] for j in joints]
    assert layout.joint_names == names
    assert len(names) == 27
    np.testing.assert_allclose(
        layout.range_lo, np.deg2rad([j["range_deg"][0] for j in joints]))
    np.testing.assert_allclose(
        layout.range_hi, np.deg2rad([j["range_deg"][1] for j in joints]))
    np.testing.assert_allclose(
        layout.gear, [j["torque"] for j in joints])
    assert layout.nu == 27
    assert layout.h_ref == pytest.approx(0.84)


def test_joint_addresses(mjm, layout):
    for i, name in enumerate(layout.joint_names):
        jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert layout.qpos_adr[i] == mjm.jnt_qposadr[jid]
        assert layout.dof_adr[i] == mjm.jnt_dofadr[jid]


def test_indices_match_name_lookup(mjm, layout):
    for i, n in enumerate(layout.joint_names):
        adr, dim = _adr(mjm, f"jointpos_{n}")
        assert dim == 1 and layout.jointpos_idx[i] == adr
        adr, dim = _adr(mjm, f"jointvel_{n}")
        assert dim == 1 and layout.jointvel_idx[i] == adr
    np.testing.assert_array_equal(
        layout.framequat_idx, np.arange(*_slice_args(mjm, "framequat_imu")))
    np.testing.assert_array_equal(
        layout.gyro_idx, np.arange(*_slice_args(mjm, "gyro_imu")))
    np.testing.assert_array_equal(
        layout.velocimeter_idx, np.arange(*_slice_args(mjm, "velocimeter_imu")))
    np.testing.assert_array_equal(
        layout.framepos_idx, np.arange(*_slice_args(mjm, "framepos_imu")))
    np.testing.assert_array_equal(
        layout.torso_zaxis_idx, np.arange(*_slice_args(mjm, "framezaxis_torso")))
    np.testing.assert_array_equal(
        layout.pelvis_linvel_idx, np.arange(*_slice_args(mjm, "framelinvel_imu")))
    np.testing.assert_array_equal(
        layout.head_xaxis_idx,
        np.arange(*_slice_args(mjm, "framexaxis_touch_head")))


def _slice_args(mjm, name):
    adr, dim = _adr(mjm, name)
    return adr, adr + dim


def test_touch_indices(mjm, layout):
    assert layout.touch_names == [n for n, _, _, _ in TOUCH_SITES]
    assert len(layout.touch_idx) == 12
    for i, n in enumerate(layout.touch_names):
        adr, dim = _adr(mjm, n)
        assert dim == 1 and layout.touch_idx[i] == adr
    np.testing.assert_array_equal(
        layout.foot_touch_idx,
        [layout.touch_idx[layout.touch_names.index("touch_foot_L")],
         layout.touch_idx[layout.touch_names.index("touch_foot_R")]])
    assert len(layout.nonfoot_touch_idx) == 10
    assert set(layout.nonfoot_touch_idx).isdisjoint(set(layout.foot_touch_idx))
    assert (set(layout.nonfoot_touch_idx) | set(layout.foot_touch_idx)
            == set(layout.touch_idx))


def test_impact_indices(mjm, layout):
    assert layout.impact_site_names == IMPACT_SITES
    assert layout.impact_linvel_idx.shape == (4, 3)
    assert layout.impact_touch_idx.shape == (4,)
    for k, site in enumerate(IMPACT_SITES):
        adr, dim = _adr(mjm, f"framelinvel_{site}")
        assert dim == 3
        np.testing.assert_array_equal(
            layout.impact_linvel_idx[k], np.arange(adr, adr + 3))
        adr, dim = _adr(mjm, site)
        assert dim == 1 and layout.impact_touch_idx[k] == adr


def test_to_torch(layout):
    import torch
    tl = layout.to("cpu")
    assert isinstance(tl.jointpos_idx, torch.Tensor)
    assert tl.jointpos_idx.dtype == torch.long
    assert tl.range_lo.dtype == torch.float32
    np.testing.assert_allclose(tl.gear.numpy(), layout.gear)
    np.testing.assert_array_equal(tl.touch_idx.numpy(), layout.touch_idx)
    # numpy version is untouched
    assert isinstance(layout.jointpos_idx, np.ndarray)
    # non-array fields carried over
    assert tl.joint_names == layout.joint_names
    assert tl.sensor == layout.sensor
    assert tl.h_ref == layout.h_ref
