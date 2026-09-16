"""Tests for the §5.3 reward on CPU MuJoCo sensordata.

act is d.act (the first-order-filtered torque, §3.3), action is the ctrl
command sent to the model. Sensordata is read once per body step (4 x 5 ms
physics steps), matching how the env will drive it.
"""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import yaml

from multibrain.body.layout import BodyLayout
from multibrain.body.reward import TERMS, Reward, RewardConfig
from multibrain.body.success import SuccessConfig, standing_now

REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "humanoid.xml"
JOINT_GROUPS = REPO_ROOT / "configs" / "joint_groups.yaml"
TASK_YAML = REPO_ROOT / "configs" / "task.yaml"
REWARD_YAML = REPO_ROOT / "configs" / "reward.yaml"

SUBSTEPS = 4  # physics steps per 20 ms body step (configs/task.yaml)

# pelvis quat lying flat on the back (scripts/check_body.py POSES["supine"])
SUPINE_QUAT = [math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0]


@pytest.fixture(scope="module")
def mjm():
    return mujoco.MjModel.from_xml_path(str(XML_PATH))


@pytest.fixture(scope="module")
def layout(mjm):
    return BodyLayout.from_model(mjm, JOINT_GROUPS)


@pytest.fixture(scope="module")
def scfg():
    return SuccessConfig.from_yaml(TASK_YAML, REWARD_YAML)


@pytest.fixture(scope="module")
def rcfg():
    return RewardConfig.from_yaml(REWARD_YAML)


def _sd(d):
    """(1, nsensordata) float32 tensor from a stepped MjData."""
    return torch.as_tensor(np.asarray(d.sensordata),
                           dtype=torch.float32).unsqueeze(0)


def _act(d):
    return torch.as_tensor(np.asarray(d.act), dtype=torch.float32).unsqueeze(0)


def _ctrl(d):
    return torch.as_tensor(np.asarray(d.ctrl), dtype=torch.float32).unsqueeze(0)


def _impact_state(layout, rcfg, s):
    """(touch (4,) bool, vz_down (4,), head_xaxis_z) from one sensordata row."""
    touch_idx = [layout.sensor[n].start for n in rcfg.impact_sites]
    vel_idx = [layout.sensor[f"framelinvel_{n}"] for n in rcfg.impact_sites]
    touch = s[0, touch_idx] >= rcfg.contact_force_min
    vz = torch.stack([torch.clamp(-s[0, sl][2], min=0.0) for sl in vel_idx])
    hz = s[0, layout.sensor["framexaxis_touch_head"].start + 2]
    return touch, vz, hz


def _expected_pain(rcfg, seq):
    """Reference pain_impact series from a recorded per-step impact state.

    seq: list of (touch (4,), vz (4,), head_xaxis_z); index 0 is the state
    at reset, so the pain on body step t uses seq[t] and seq[t + 1].
    """
    out = []
    for t in range(len(seq) - 1):
        touch_p, vz_p, _ = seq[t]
        touch_n, _, hz = seq[t + 1]
        onset = touch_n & ~touch_p
        v_excess = torch.clamp(vz_p - rcfg.v0, min=0.0, max=rcfg.v_cap)
        pain = 0.0
        for k, name in enumerate(rcfg.impact_sites):
            w = rcfg.impact_weight[name]
            if name == "touch_head" and hz < rcfg.face_axis_z_max:
                w = rcfg.impact_weight["face"]
            pain -= rcfg.coef["pain_impact"] * w * float(onset[k]) \
                * float(v_excess[k]) ** 2
        out.append(pain)
    return out


def _rollout(mjm, d, reward, layout, scfg, rcfg, n_steps, ctrl_fn,
             record=None):
    """Step `d` for n_steps body steps with ctrl_fn(i) -> (27,) ctrl,
    feeding reward.step each body step. Returns list of terms dicts."""
    all_terms = []
    for i in range(n_steps):
        d.ctrl[:] = ctrl_fn(i)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(mjm, d)
        s = _sd(d)
        if record is not None:
            record.append(_impact_state(layout, rcfg, s))
        total, terms = reward.step(
            s, _act(d), _ctrl(d), standing_now(layout, scfg, s),
            torch.zeros(1, dtype=torch.bool))
        all_terms.append(terms)
    return all_terms


def test_fall_pain_impact(mjm, layout, scfg, rcfg):
    """§5.3 test: standing + 3 deg forward lean, ctrl=0 for 2 s. Total
    pain_impact < 0, and each impact site's pain is charged only on the
    steps where its contact starts (per-site onset counting, verified
    against a per-site reference computed from the recorded touch and
    downward-speed sequences)."""
    d = mujoco.MjData(mjm)
    th = math.radians(3.0)
    d.qpos[3:7] = [math.cos(th / 2), 0.0, math.sin(th / 2), 0.0]
    mujoco.mj_forward(mjm, d)

    reward = Reward(layout, rcfg, 1, "cpu")
    s0 = _sd(d)
    reward.reset(torch.ones(1, dtype=torch.bool), s0)
    seq = [_impact_state(layout, rcfg, s0)]

    terms_list = _rollout(mjm, d, reward, layout, scfg, rcfg, 100,
                          lambda i: np.zeros(mjm.nu), record=seq)
    pain = torch.stack([t["pain_impact"] for t in terms_list])[:, 0]

    assert pain.sum().item() < 0.0
    # at least one impact site actually hit the floor
    assert torch.stack([x[0] for x in seq]).any()

    expected = torch.tensor(_expected_pain(rcfg, seq), dtype=torch.float32)
    torch.testing.assert_close(pain, expected, rtol=1e-5, atol=1e-6)
    # every nonzero pain step is explained by an onset that step
    onset_steps = [t for t in range(len(seq) - 1)
                   if (seq[t + 1][0] & ~seq[t][0]).any()]
    assert set((pain != 0).nonzero()[0].tolist()) <= set(onset_steps)


def _supine_rest_height(mjm):
    d = mujoco.MjData(mjm)
    d.qpos[2] = 0.20
    d.qpos[3:7] = SUPINE_QUAT
    mujoco.mj_forward(mjm, d)
    for _ in range(200):  # 1 s
        mujoco.mj_step(mjm, d)
    return float(d.qpos[2])


@pytest.mark.parametrize("lift,expect_zero", [(0.01, True), (0.3, False)])
def test_supine_drop_pain(mjm, layout, scfg, rcfg, lift, expect_zero):
    """§5.3 test: supine released just above its resting height must not
    hurt — from +1 cm the impact sites arrive at ~0.44 m/s < v0 = 0.5.
    From +0.3 m (~2.4 m/s) the total pain_impact is negative."""
    rest_z = _supine_rest_height(mjm)

    d = mujoco.MjData(mjm)
    d.qpos[2] = rest_z + lift
    d.qpos[3:7] = SUPINE_QUAT
    mujoco.mj_forward(mjm, d)

    reward = Reward(layout, rcfg, 1, "cpu")
    reward.reset(torch.ones(1, dtype=torch.bool), _sd(d))
    terms_list = _rollout(mjm, d, reward, layout, scfg, rcfg, 100,
                          lambda i: np.zeros(mjm.nu))
    pain_sum = sum(t["pain_impact"].item() for t in terms_list)
    if expect_zero:
        assert pain_sum == 0.0
    else:
        assert pain_sum < 0.0


def test_pain_joint_limit(mjm, layout, scfg, rcfg):
    """§5.3 test: ctrl=+1 on every joint for 1 s drives joints into their
    +limit band with pushing torque -> pain_joint_limit < 0 on the last
    step. ctrl=0 for another 0.5 s lets the filter (tau = 40 ms) relax act
    to ~0 -> |pain_joint_limit| < 1e-3."""
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    reward = Reward(layout, rcfg, 1, "cpu")
    reward.reset(torch.ones(1, dtype=torch.bool), _sd(d))

    terms_list = _rollout(mjm, d, reward, layout, scfg, rcfg, 50,
                          lambda i: np.ones(mjm.nu))
    assert terms_list[-1]["pain_joint_limit"].item() < 0.0

    terms_list = _rollout(mjm, d, reward, layout, scfg, rcfg, 25,
                          lambda i: np.zeros(mjm.nu))
    assert abs(terms_list[-1]["pain_joint_limit"].item()) < 1e-3


def test_fatigue_energy_uses_act(layout, rcfg, mjm):
    """act = +/-1 on all 27 joints -> -coef * mean(1) = -0.02 exactly."""
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    s = _sd(d)
    for sign in (1.0, -1.0):
        reward = Reward(layout, rcfg, 1, "cpu")
        reward.reset(torch.ones(1, dtype=torch.bool), s)
        act = torch.full((1, 27), float(sign))
        _, terms = reward.step(s, act, torch.zeros(1, 27),
                               torch.zeros(1, dtype=torch.bool),
                               torch.zeros(1, dtype=torch.bool))
        assert terms["fatigue_energy"].item() == pytest.approx(
            -rcfg.coef["fatigue_energy"], abs=1e-6)


def test_fatigue_action_rate(layout, rcfg, mjm):
    """Alternating action +1/-1: step 1 sees (1-0)^2 -> -0.05; from step 2
    on the jump is 2 -> -coef * 4 = -0.2 per step."""
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    s = _sd(d)
    reward = Reward(layout, rcfg, 1, "cpu")
    reward.reset(torch.ones(1, dtype=torch.bool), s)
    act = torch.zeros(1, 27)
    for i in range(4):
        action = torch.full((1, 27), 1.0 if i % 2 == 0 else -1.0)
        _, terms = reward.step(s, act, action,
                               torch.zeros(1, dtype=torch.bool),
                               torch.zeros(1, dtype=torch.bool))
        expected = (-rcfg.coef["fatigue_action_rate"] * (1.0 if i == 0 else 4.0))
        assert terms["fatigue_action_rate"].item() == pytest.approx(
            expected, abs=1e-6)


def test_height_progress_telescopes(mjm, layout, scfg, rcfg):
    """Sum of height_progress over a rollout equals
    coef * (h_T - h_0) / h_ref (telescoping series)."""
    d = mujoco.MjData(mjm)
    d.qpos[2] = 0.20
    d.qpos[3:7] = SUPINE_QUAT
    mujoco.mj_forward(mjm, d)

    reward = Reward(layout, rcfg, 1, "cpu")
    s0 = _sd(d)
    reward.reset(torch.ones(1, dtype=torch.bool), s0)
    h0 = s0[0, layout.sensor["framepos_imu"].start + 2].item()

    terms_list = _rollout(mjm, d, reward, layout, scfg, rcfg, 50,
                          lambda i: np.zeros(mjm.nu))
    hT = _sd(d)[0, layout.sensor["framepos_imu"].start + 2].item()
    hp_sum = sum(t["height_progress"].item() for t in terms_list)
    assert hp_sum == pytest.approx(
        rcfg.coef["height_progress"] * (hT - h0) / rcfg.h_ref, abs=1e-5)


def test_standing_and_first_success_terms(layout, rcfg, mjm):
    """standing / first_success are simply coefficient x bool."""
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    s = _sd(d)
    reward = Reward(layout, rcfg, 1, "cpu")
    reward.reset(torch.ones(1, dtype=torch.bool), s)
    zero = torch.zeros(1, 27)
    for st, fs in [(True, False), (False, True), (True, True),
                   (False, False)]:
        _, terms = reward.step(
            s, zero, zero,
            torch.tensor([st]), torch.tensor([fs]))
        assert terms["standing"].item() == pytest.approx(
            rcfg.coef["standing"] * st, abs=1e-7)
        assert terms["first_success"].item() == pytest.approx(
            rcfg.coef["first_success"] * fs, abs=1e-6)


def test_extra_contact_height_gate(mjm, layout, rcfg):
    """extra_contact is nonzero only when a non-foot site touches AND
    h_t >= 0.8 h_ref. Checked by forcing a hand channel above the force
    threshold on a standing snapshot (h ~ h_ref passes the gate) and by
    a real supine snapshot (pelvis touches but h is below the gate)."""
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    for _ in range(20):
        mujoco.mj_step(mjm, d)
    s_stand = _sd(d)

    d2 = mujoco.MjData(mjm)
    d2.qpos[2] = 0.20
    d2.qpos[3:7] = SUPINE_QUAT
    mujoco.mj_forward(mjm, d2)
    for _ in range(200):
        mujoco.mj_step(mjm, d2)
    s_supine = _sd(d2)

    zero = torch.zeros(1, 27)
    bfalse = torch.zeros(1, dtype=torch.bool)

    def extra_contact(s):
        reward = Reward(layout, rcfg, 1, "cpu")
        reward.reset(torch.ones(1, dtype=torch.bool), s)
        _, terms = reward.step(s, zero, zero, bfalse, bfalse)
        return terms["extra_contact"].item()

    # standing, feet only -> 0 despite h passing the gate
    assert extra_contact(s_stand) == 0.0
    # standing with a forced hand contact -> -coef * 1
    s_forced = s_stand.clone()
    s_forced[0, layout.sensor["touch_hand_L"].start] = 10.0
    assert extra_contact(s_forced) == pytest.approx(
        -rcfg.coef["extra_contact"], abs=1e-7)
    # supine: pelvis touches but h << 0.8 h_ref -> 0
    assert extra_contact(s_supine) == 0.0


def test_batch_matches_single(mjm, layout, scfg, rcfg):
    """A B=2 batch produces the same terms as two independent B=1 runs
    fed the same per-env inputs; every term is (B,) float32."""

    def record_rollout(d, n, ctrl_fn):
        """Replayable (sensordata, act, action, standing) per body step."""
        mujoco.mj_forward(mjm, d)
        rows = []
        for i in range(n):
            d.ctrl[:] = ctrl_fn(i)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(mjm, d)
            s = _sd(d)
            rows.append((s, _act(d), _ctrl(d),
                         standing_now(layout, scfg, s)))
        return rows

    # env 0: standing rest nudged by a smooth ctrl; env 1: supine drop
    d0 = mujoco.MjData(mjm)
    roll0 = record_rollout(
        d0, 20, lambda i: 0.1 * np.sin(0.3 * i + np.arange(mjm.nu)))
    d1 = mujoco.MjData(mjm)
    d1.qpos[2] = 0.30
    d1.qpos[3:7] = SUPINE_QUAT
    roll1 = record_rollout(d1, 20, lambda i: np.zeros(mjm.nu))

    init = [roll0[0][0], roll1[0][0]]
    mask = torch.ones(1, dtype=torch.bool)

    singles = []
    for env in range(2):
        r = Reward(layout, rcfg, 1, "cpu")
        r.reset(mask, init[env])
        out = []
        for s, act, actn, st in (roll0, roll1)[env]:
            _, terms = r.step(s, act, actn, st,
                              torch.zeros(1, dtype=torch.bool))
            out.append(terms)
        singles.append(out)

    rb = Reward(layout, rcfg, 2, "cpu")
    rb.reset(torch.ones(2, dtype=torch.bool), torch.cat(init))
    for i in range(20):
        s = torch.cat([roll0[i][0], roll1[i][0]])
        act = torch.cat([roll0[i][1], roll1[i][1]])
        actn = torch.cat([roll0[i][2], roll1[i][2]])
        st = torch.cat([roll0[i][3], roll1[i][3]])
        total, terms = rb.step(s, act, actn, st,
                               torch.zeros(2, dtype=torch.bool))
        assert total.shape == (2,)
        for name in TERMS:
            assert terms[name].shape == (2,)
            assert terms[name].dtype == torch.float32
            for env in range(2):
                torch.testing.assert_close(
                    terms[name][env], singles[env][i][name][0],
                    rtol=1e-5, atol=1e-6)


def test_coefficients_come_from_yaml(mjm, layout, rcfg, tmp_path):
    """A reward.yaml copy with one coefficient changed must change that
    term's value (nothing is hard-coded)."""
    with open(REWARD_YAML) as f:
        y = yaml.safe_load(f)
    y["coef"]["fatigue_energy"] = 0.2   # 10x the shipped 0.02
    patched = tmp_path / "reward.yaml"
    patched.write_text(yaml.safe_dump(y))
    cfg2 = RewardConfig.from_yaml(patched)
    assert cfg2.coef["fatigue_energy"] == 0.2

    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    s = _sd(d)
    act = torch.ones(1, 27)
    for cfg, expected in ((rcfg, -0.02), (cfg2, -0.2)):
        reward = Reward(layout, cfg, 1, "cpu")
        reward.reset(torch.ones(1, dtype=torch.bool), s)
        _, terms = reward.step(s, act, torch.zeros(1, 27),
                               torch.zeros(1, dtype=torch.bool),
                               torch.zeros(1, dtype=torch.bool))
        assert terms["fatigue_energy"].item() == pytest.approx(
            expected, abs=1e-6)
