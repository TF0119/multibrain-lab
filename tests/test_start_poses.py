"""Start poses (PLAN.md §5.1): settled height, no floor penetration, no launch."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from multibrain.body.layout import BodyLayout
from multibrain.body.start_poses import (CLEARANCE, EVAL_KINDS, eval_starts,
                                          lowest_geom_z, start_qpos)

XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "humanoid.xml"


@pytest.fixture(scope="module")
def mjm():
    return mujoco.MjModel.from_xml_path(str(XML_PATH))


@pytest.fixture(scope="module")
def layout(mjm):
    return BodyLayout.from_model(mjm)


def _lowest(mjm, q):
    d = mujoco.MjData(mjm)
    d.qpos[:] = q
    mujoco.mj_kinematics(mjm, d)
    return lowest_geom_z(mjm, d)


@pytest.mark.parametrize("kind", ["standing", "supine", "prone", "side"])
def test_settled_lowest_point(mjm, layout, kind):
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = start_qpos(layout, kind, rng, mjm)
        assert _lowest(mjm, q) == pytest.approx(CLEARANCE, abs=1e-4)


def test_unsettled_lying_starts_penetrate(mjm, layout):
    """Documents why settling is needed: the nominal 0.20 m pelvis height
    with scattered joints puts limbs well below the floor."""
    rng = np.random.default_rng(0)
    lows = [_lowest(mjm, start_qpos(layout, "prone", rng)) for _ in range(30)]
    assert min(lows) < -0.1


def test_eval_starts_settled(mjm, layout):
    starts = eval_starts(layout, n=24, seed=12345, mjm=mjm)
    assert [k for k, _ in starts].count("supine") == 8
    for kind, q in starts:
        assert kind in EVAL_KINDS
        assert _lowest(mjm, q) == pytest.approx(CLEARANCE, abs=1e-4)


def test_no_launch_under_zero_ctrl(mjm, layout):
    """Lying starts released with zero control must settle, not fly: the
    pelvis never rises above 0.45 m within 2 s (CPU MuJoCo)."""
    rng = np.random.default_rng(1)
    for kind in EVAL_KINDS:
        for _ in range(4):
            d = mujoco.MjData(mjm)
            d.qpos[:] = start_qpos(layout, kind, rng, mjm)
            mujoco.mj_forward(mjm, d)
            h_max = 0.0
            for _ in range(400):
                mujoco.mj_step(mjm, d)
                h_max = max(h_max, float(d.qpos[2]))
            assert h_max < 0.45, (kind, h_max)
            assert not np.isnan(d.qpos).any()
