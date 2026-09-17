"""Reward loophole tests of PLAN.md §5.3: jumping, stopping on the knees,
and propping the pelvis up on the hands must not pay better than standing.

Physics rollouts use CPU MuJoCo with 4 physics steps per body step, d.act
as the applied torque and the commanded ctrl as the action, like
tests/test_reward.py. Synthetic sensordata rows are assembled by sensor
name through BodyLayout.sensor (no hard-coded indices).
"""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from multibrain.body.layout import BodyLayout
from multibrain.body.reward import Reward, RewardConfig
from multibrain.body.success import SuccessConfig, SuccessTracker, standing_now

REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "humanoid.xml"
TASK_YAML = REPO_ROOT / "configs" / "task.yaml"
REWARD_YAML = REPO_ROOT / "configs" / "reward.yaml"
SUBSTEPS = 4


@pytest.fixture(scope="module")
def mjm():
    return mujoco.MjModel.from_xml_path(str(XML_PATH))


@pytest.fixture(scope="module")
def layout(mjm):
    return BodyLayout.from_model(mjm)


@pytest.fixture(scope="module")
def rcfg():
    return RewardConfig.from_yaml(REWARD_YAML)


@pytest.fixture(scope="module")
def scfg():
    return SuccessConfig.from_yaml(TASK_YAML, REWARD_YAML)


def _sd(d):
    return torch.as_tensor(np.asarray(d.sensordata), dtype=torch.float32)[None]


def _rollout(mjm, d, layout, rcfg, scfg, n_steps, ctrl_fn):
    """Body-step rollout -> list of (terms, standing) per step."""
    reward = Reward(layout, rcfg, 1, "cpu")
    tracker = SuccessTracker(scfg, 1, "cpu")
    reward.reset(torch.ones(1, dtype=torch.bool), _sd(d))
    out = []
    for i in range(n_steps):
        d.ctrl[:] = ctrl_fn(i)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(mjm, d)
        s = _sd(d)
        standing = standing_now(layout, scfg, s)
        tr = tracker.update(standing)
        act = torch.as_tensor(np.asarray(d.act), dtype=torch.float32)[None]
        ctrl = torch.as_tensor(np.asarray(d.ctrl), dtype=torch.float32)[None]
        total, terms = reward.step(s, act, ctrl, standing, tr["first_success"])
        out.append((float(total), {k: float(v) for k, v in terms.items()},
                    bool(standing)))
    return out


# The passive symmetric body stands for ~2.7 s after settling before it
# starts to drift (pelvis speed > 0.2 m/s), so the comparisons below use
# 2 s windows.
STILL_STEPS = 100


def _standing_data(mjm, settle_steps=40):
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    for _ in range(settle_steps):  # 0.2 s, feet take the load
        mujoco.mj_step(mjm, d)
    return d


def _synthetic(layout, **named):
    """One sensordata row from {sensor_name: value or vector}."""
    s = torch.zeros(1, layout.nsensordata)
    for name, val in named.items():
        sl = layout.sensor[name]
        s[0, sl] = torch.as_tensor(val, dtype=torch.float32)
    return s


# ---------------------------------------------------------------- jumping

def test_jump_synthetic_height_telescopes(layout, rcfg):
    """Height progress over any closed height loop sums to zero: bobbing the
    pelvis 0.2 m <-> 0.6 m for 20 cycles earns nothing, and the rest of the
    terms can only subtract."""
    reward = Reward(layout, rcfg, 1, "cpu")
    lo = _synthetic(layout, framepos_imu=[0.0, 0.0, 0.2])
    hi = _synthetic(layout, framepos_imu=[0.0, 0.0, 0.6])
    reward.reset(torch.ones(1, dtype=torch.bool), lo)
    zeros = torch.zeros(1, layout.nu)
    f = torch.zeros(1, dtype=torch.bool)
    hp = tot = dense = 0.0
    for _ in range(20):
        for s in (hi, lo):
            total, terms = reward.step(s, zeros, zeros, f, f)
            hp += float(terms["height_progress"])
            dense += float(terms["height"])
            tot += float(total)
    assert abs(hp) < 1e-5
    # the only thing bobbing earns is the dense height term, i.e. exactly
    # what resting at the mean height (0.4 m) would earn
    assert tot == pytest.approx(dense, abs=1e-5)
    assert dense == pytest.approx(
        40 * rcfg.coef["height"] * 0.4 / rcfg.h_ref, abs=1e-4)


def test_hopping_pays_less_than_standing(mjm, layout, rcfg, scfg):
    """Physics: from standing, a +/-1 square wave (0.4 s period) on the hips,
    knees and ankles for 2 s. Height progress telescopes to
    2 (h_T - h_0) / h_ref, and the total is far below 2 s of quiet standing
    (which collects the 1.0 standing + 0.2 uprightness terms every step)."""
    names = layout.joint_names
    drive = [names.index(n) for n in ("hip_L_x", "hip_R_x", "knee_L",
                                       "knee_R", "ankle_L_x", "ankle_R_x")]

    d = _standing_data(mjm)
    h0 = float(d.sensordata[layout.framepos_idx[2]])

    def square(i):
        c = np.zeros(mjm.nu)
        c[drive] = 1.0 if (i // 10) % 2 == 0 else -1.0  # 10 steps = 0.2 s
        return c

    hop = _rollout(mjm, d, layout, rcfg, scfg, STILL_STEPS, square)
    hT = float(d.sensordata[layout.framepos_idx[2]])
    hp = sum(t["height_progress"] for _, t, _ in hop)
    assert hp == pytest.approx(
        rcfg.coef["height_progress"] * (hT - h0) / rcfg.h_ref, abs=1e-4)

    still = _rollout(mjm, _standing_data(mjm), layout, rcfg, scfg,
                     STILL_STEPS, lambda i: np.zeros(mjm.nu))
    assert all(st for _, _, st in still), "quiet standing must count as standing"
    hop_total = sum(tot for tot, _, _ in hop)
    still_total = sum(tot for tot, _, _ in still)
    assert still_total > STILL_STEPS * 1.0  # standing term alone
    assert hop_total < 0.25 * still_total


# ---------------------------------------------------------- kneeling stop

def test_kneeling_pays_less_than_standing(mjm, layout, rcfg, scfg):
    """Kneeling (hip 18 deg, knee 95 deg, pelvis 0.46 m, settled 1.5 s; the
    pose of tests/test_success.py) held for 2 s: never standing, and the
    per-step reward stays at or below the 0.2 uprightness ceiling, far below
    quiet standing's ~1.2 per step."""
    d = mujoco.MjData(mjm)
    d.qpos[2] = 0.46
    for n, deg in (("hip_L_x", 18.0), ("hip_R_x", 18.0),
                   ("knee_L", 95.0), ("knee_R", 95.0)):
        d.qpos[layout.qpos_adr[layout.joint_names.index(n)]] = math.radians(deg)
    mujoco.mj_forward(mjm, d)
    for _ in range(300):  # 1.5 s settle
        mujoco.mj_step(mjm, d)

    kneel = _rollout(mjm, d, layout, rcfg, scfg, STILL_STEPS,
                     lambda i: np.zeros(mjm.nu))
    assert not any(st for _, _, st in kneel)
    per_step = sum(tot for tot, _, _ in kneel) / len(kneel)
    assert per_step <= 0.25

    still = _rollout(mjm, _standing_data(mjm), layout, rcfg, scfg,
                     STILL_STEPS, lambda i: np.zeros(mjm.nu))
    assert per_step < sum(tot for tot, _, _ in still) / len(still)


# ------------------------------------------------- propping up on the hands

def test_propped_pelvis_pays_less_than_standing(layout, rcfg):
    """Synthetic: pelvis at 0.85 h_ref propped on both hands (torso z axis
    tilted to cos 0.7), feet loaded too. Per step: uprightness 0.2 * 0.7
    plus dense height 0.05 * 0.85 minus extra contact 0.05 * 2 hands
    = 0.0825, versus ~1.25 for standing."""
    s = _synthetic(layout, framepos_imu=[0.0, 0.0, 0.85 * rcfg.h_ref],
                   framezaxis_torso=[0.0, math.sqrt(1 - 0.7 ** 2), 0.7],
                   touch_hand_L=30.0, touch_hand_R=30.0,
                   touch_foot_L=200.0, touch_foot_R=200.0)
    reward = Reward(layout, rcfg, 1, "cpu")
    reward.reset(torch.ones(1, dtype=torch.bool), s)
    zeros = torch.zeros(1, layout.nu)
    f = torch.zeros(1, dtype=torch.bool)
    total, terms = reward.step(s, zeros, zeros, f, f)
    expected = (rcfg.coef["uprightness"] * 0.7
                + rcfg.coef["height"] * 0.85
                - rcfg.coef["extra_contact"] * 2)
    assert float(total) == pytest.approx(expected, abs=1e-6)
    assert float(terms["extra_contact"]) == pytest.approx(
        -rcfg.coef["extra_contact"] * 2, abs=1e-6)
    assert float(total) < rcfg.coef["standing"] + rcfg.coef["uprightness"]
