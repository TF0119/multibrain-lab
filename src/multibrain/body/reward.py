"""Reward of PLAN.md §5.3 on a batch of sensordata tensors (B envs, torch).

All coefficients and thresholds come from configs/reward.yaml (via
RewardConfig); nothing is hard-coded. The ten terms are:

  height_progress      c * (h_t - h_{t-1}) / h_ref
  height               c * h_t / h_ref            (dense; gradient at every height)
  uprightness          c * clamp(cos tilt, -1, 1) * 1[h_t >= 0.5 h_ref]
  standing             c * 1[standing]
  first_success        c * 1[first_success]
  pain_impact          -c * sum_k w_k onset_k min([vz_k - v0]+^2, v_cap^2)
  pain_joint_limit     -c * mean_j xi_j^2 [sign(q~_j) act_j]+
  fatigue_energy       -c * mean_j act_j^2
  fatigue_action_rate  -c * mean_j (a_j - a_{j,t-1})^2
  extra_contact        -c * (# non-foot contacts) * 1[h_t >= 0.8 h_ref]

`act` is MuJoCo's d.act — the normalized torque after the first-order
filter, i.e. what is actually applied to the joints (§3.3). `action` is
the policy's raw command before the filter.
"""

from dataclasses import dataclass

import numpy as np
import torch
import yaml

from .layout import BodyLayout
from .observation import _idx, _val
from .success import tilt_cos

TERMS = ["height_progress", "height", "uprightness", "standing", "first_success",
         "pain_impact", "pain_joint_limit", "fatigue_energy",
         "fatigue_action_rate", "extra_contact"]


@dataclass
class RewardConfig:
    h_ref: float
    coef: dict                    # term name -> coefficient (9 entries)
    uprightness_min_height_ratio: float
    extra_contact_min_height_ratio: float
    contact_force_min: float      # N
    impact_sites: list            # site names, order of the impact state
    impact_weight: dict           # site name (and "face") -> weight
    face_axis_z_max: float
    v0: float                     # m/s
    v_cap: float                  # m/s
    joint_limit_band: float       # fraction of the half-range

    @classmethod
    def from_yaml(cls, path) -> "RewardConfig":
        with open(path) as f:
            y = yaml.safe_load(f)
        return cls(
            h_ref=float(y["h_ref"]),
            coef={k: float(v) for k, v in y["coef"].items()},
            uprightness_min_height_ratio=
                float(y["uprightness"]["min_height_ratio"]),
            extra_contact_min_height_ratio=
                float(y["extra_contact"]["min_height_ratio"]),
            contact_force_min=float(y["contact"]["force_min_n"]),
            impact_sites=list(y["impact"]["sites"]),
            impact_weight={k: float(v)
                           for k, v in y["impact"]["weight"].items()},
            face_axis_z_max=float(y["impact"]["face_axis_z_max"]),
            v0=float(y["impact"]["v0"]),
            v_cap=float(y["impact"]["v_cap"]),
            joint_limit_band=float(y["joint_limit"]["band"]),
        )


class Reward:
    """Stateful per-step reward for a batch of B envs (§5.3)."""

    def __init__(self, layout: BodyLayout, cfg: RewardConfig,
                 n_envs: int, device):
        self.layout = layout
        self.cfg = cfg
        self.h_prev = torch.zeros(n_envs, device=device)
        self.a_prev = torch.zeros(n_envs, int(layout.nu), device=device)
        k = len(cfg.impact_sites)
        self.touch_prev = torch.zeros(n_envs, k, dtype=torch.bool,
                                      device=device)
        self.vz_prev = torch.zeros(n_envs, k, device=device)

        # sensordata indices are resolved by name so that the site order of
        # reward.yaml is honored (it need not match build_xml.IMPACT_SITES)
        sensor = layout.sensor
        self._impact_touch_idx = torch.as_tensor(
            [sensor[s].start for s in cfg.impact_sites],
            dtype=torch.long, device=device)
        linvel_idx = []
        for s in cfg.impact_sites:
            sl = sensor[f"framelinvel_{s}"]
            linvel_idx.append(range(sl.start, sl.stop))
        self._impact_linvel_idx = torch.as_tensor(
            linvel_idx, dtype=torch.long, device=device)
        self._w_site = torch.as_tensor(
            [cfg.impact_weight[s] for s in cfg.impact_sites],
            dtype=torch.float32, device=device)
        self._head_col = (cfg.impact_sites.index("touch_head")
                          if "touch_head" in cfg.impact_sites else None)

    def _impact_state(self, s: torch.Tensor):
        """(h_t, touch_now, vz_now) for the impact sites at this step."""
        h_t = s[:, _idx(s, self.layout.framepos_idx)][:, 2]
        touch_now = s[:, self._impact_touch_idx] >= self.cfg.contact_force_min
        vel = s[:, self._impact_linvel_idx]           # (B, K, 3)
        vz_now = torch.clamp(-vel[..., 2], min=0.0)   # downward speed
        return h_t, touch_now, vz_now

    def reset(self, mask: torch.Tensor, sensordata: torch.Tensor) -> None:
        """Re-seed h_prev/touch_prev/vz_prev from the current sensordata and
        zero a_prev for the masked envs (start of a new trial)."""
        h_t, touch_now, vz_now = self._impact_state(sensordata)
        m = mask.to(torch.bool).unsqueeze(-1)
        self.h_prev = torch.where(mask, h_t, self.h_prev)
        self.touch_prev = torch.where(m, touch_now, self.touch_prev)
        self.vz_prev = torch.where(m, vz_now, self.vz_prev)
        self.a_prev = torch.where(m, torch.zeros_like(self.a_prev),
                                self.a_prev)

    def step(self, sensordata: torch.Tensor, act: torch.Tensor,
             action: torch.Tensor, standing: torch.Tensor,
             first_success: torch.Tensor):
        """One body step -> (total (B,), terms {name: (B,) float32}).

        `terms` holds the coefficient-applied value of each of TERMS.
        """
        s = sensordata
        cfg = self.cfg
        dt = s.dtype
        n_envs = s.shape[0]

        h_t, touch_now, vz_now = self._impact_state(s)
        tilt = torch.clamp(tilt_cos(self.layout, s), -1.0, 1.0)

        terms = {}
        terms["height_progress"] = (
            cfg.coef["height_progress"] * (h_t - self.h_prev) / cfg.h_ref)
        terms["height"] = cfg.coef["height"] * h_t / cfg.h_ref
        terms["uprightness"] = (
            cfg.coef["uprightness"] * tilt
            * (h_t >= cfg.uprightness_min_height_ratio
               * cfg.h_ref).to(dt))
        terms["standing"] = cfg.coef["standing"] * standing.to(dt)
        terms["first_success"] = (cfg.coef["first_success"]
                                  * first_success.to(dt))

        # pain: one charge per contact onset, scaled by the site's downward
        # speed on the previous body step (§5.3)
        onset = touch_now & ~self.touch_prev
        v_excess = torch.clamp(self.vz_prev - cfg.v0, min=0.0,
                               max=cfg.v_cap)
        w = self._w_site.to(dt).unsqueeze(0).repeat(n_envs, 1)
        if self._head_col is not None:
            face = (s[:, _idx(s, self.layout.head_xaxis_idx)][:, 2]
                    < cfg.face_axis_z_max)
            w[:, self._head_col] = torch.where(
                face, s.new_full((n_envs,), cfg.impact_weight["face"]),
                w[:, self._head_col])
        terms["pain_impact"] = (
            -cfg.coef["pain_impact"]
            * (w * onset.to(dt) * v_excess * v_excess).sum(-1))

        # pain: torque pressing a joint into the last `band` of its range;
        # q~ uses the same normalization as the observation (§3.5)
        q = s[:, _idx(s, self.layout.jointpos_idx)]
        lo = _val(s, self.layout.range_lo)
        hi = _val(s, self.layout.range_hi)
        qn = 2.0 * (q - lo) / (hi - lo) - 1.0
        xi = torch.clamp(
            (qn.abs() - (1.0 - cfg.joint_limit_band)) / cfg.joint_limit_band,
            min=0.0)
        push = torch.clamp(torch.sign(qn) * act, min=0.0)
        terms["pain_joint_limit"] = (
            -cfg.coef["pain_joint_limit"] * (xi * xi * push).mean(-1))

        terms["fatigue_energy"] = (
            -cfg.coef["fatigue_energy"] * (act * act).mean(-1))
        da = action - self.a_prev
        terms["fatigue_action_rate"] = (
            -cfg.coef["fatigue_action_rate"] * (da * da).mean(-1))

        nonfoot = s[:, _idx(s, self.layout.nonfoot_touch_idx)] \
            >= cfg.contact_force_min
        terms["extra_contact"] = (
            -cfg.coef["extra_contact"] * nonfoot.to(dt).sum(-1)
            * (h_t >= cfg.extra_contact_min_height_ratio
               * cfg.h_ref).to(dt))

        self.h_prev = h_t
        self.a_prev = action.detach().clone()
        self.touch_prev = touch_now
        self.vz_prev = vz_now

        terms = {k: terms[k].to(torch.float32) for k in TERMS}
        total = torch.stack([terms[k] for k in TERMS]).sum(dim=0)
        return total, terms
