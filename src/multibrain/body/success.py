"""Success judgment of PLAN.md §5.2 on a batch of sensordata tensors.

A trial succeeds when all of the following hold for `success.hold_s`
consecutive body steps (configs/task.yaml):

  - both feet touch the floor (touch force >= contact.force_min_n, which
    lives in configs/reward.yaml) and no other touch site does,
  - the torso z axis is within tilt_deg_max of vertical,
  - pelvis height >= height_ratio_min * h_ref,
  - pelvis horizontal speed in world coordinates <= horizontal_speed_max.

Milestones (first 1 s / 3 s / 10 s of standing) are reported for logging
only; they do not affect the judgment.
"""

import math
from dataclasses import dataclass, field

import torch
import yaml

from .layout import BodyLayout
from .observation import _idx


@dataclass
class SuccessConfig:
    hold_steps: int                 # success.hold_s / body_step_s
    tilt_cos_min: float             # cos(success.tilt_deg_max)
    height_ratio_min: float         # pelvis height / h_ref
    horizontal_speed_max: float     # m/s
    contact_force_min: float        # N, reward.yaml contact.force_min_n
    milestone_steps: list           # success.milestones_s / body_step_s

    @classmethod
    def from_yaml(cls, task_path, reward_path) -> "SuccessConfig":
        with open(task_path) as f:
            task = yaml.safe_load(f)
        with open(reward_path) as f:
            reward = yaml.safe_load(f)
        step = float(task["body_step_s"])
        sc = task["success"]
        return cls(
            hold_steps=round(float(sc["hold_s"]) / step),
            tilt_cos_min=math.cos(math.radians(float(sc["tilt_deg_max"]))),
            height_ratio_min=float(sc["height_ratio_min"]),
            horizontal_speed_max=float(sc["horizontal_speed_max"]),
            contact_force_min=float(reward["contact"]["force_min_n"]),
            milestone_steps=[round(float(m) / step)
                             for m in sc["milestones_s"]],
        )


def contacts(layout: BodyLayout, sensordata: torch.Tensor,
             force_min: float) -> torch.Tensor:
    """(B, 12) bool — touch force >= force_min, in TOUCH_SITES order."""
    return sensordata[:, _idx(sensordata, layout.touch_idx)] >= force_min


def tilt_cos(layout: BodyLayout, sensordata: torch.Tensor) -> torch.Tensor:
    """(B,) — world z of the torso z axis, i.e. cos of the tilt angle.

    Also used by the uprightness reward term (§5.3).
    """
    return sensordata[:, _idx(sensordata, layout.torso_zaxis_idx)][:, 2]


def standing_now(layout: BodyLayout, cfg: SuccessConfig,
                 sensordata: torch.Tensor) -> torch.Tensor:
    """(B,) bool — every §5.2 condition holds at this body step."""
    s = sensordata
    feet = s[:, _idx(s, layout.foot_touch_idx)] >= cfg.contact_force_min
    nonfoot = s[:, _idx(s, layout.nonfoot_touch_idx)] >= cfg.contact_force_min
    height = s[:, _idx(s, layout.framepos_idx)][:, 2]
    hspeed = s[:, _idx(s, layout.pelvis_linvel_idx)][:, :2].norm(dim=-1)
    return (feet.all(dim=-1)
            & ~nonfoot.any(dim=-1)
            & (tilt_cos(layout, s) >= cfg.tilt_cos_min)
            & (height >= cfg.height_ratio_min * float(layout.h_ref))
            & (hspeed <= cfg.horizontal_speed_max))


class SuccessTracker:
    """Per-env counter of consecutive standing steps (§5.2)."""

    def __init__(self, cfg: SuccessConfig, n_envs: int, device):
        self.cfg = cfg
        self.consecutive = torch.zeros(n_envs, dtype=torch.long,
                                       device=device)
        self.achieved = torch.zeros(n_envs, dtype=torch.bool, device=device)
        self._milestones = torch.as_tensor(list(cfg.milestone_steps),
                                           dtype=torch.long, device=device)

    def reset(self, mask: torch.Tensor) -> None:
        """Zero the counters of the masked envs (start of a new trial)."""
        mask = mask.to(torch.bool)
        self.consecutive = torch.where(
            mask, torch.zeros_like(self.consecutive), self.consecutive)
        self.achieved = torch.where(
            mask, torch.zeros_like(self.achieved), self.achieved)

    def update(self, standing: torch.Tensor) -> dict:
        """Advance the counters by one body step.

        Returns a dict with (B,) bool tensors `standing`, `success_now`
        (consecutive >= hold_steps), `first_success` (reached hold_steps for
        the first time in this trial), and `milestone` (B, 3) — consecutive
        has just reached each of cfg.milestone_steps.
        """
        self.consecutive = torch.where(
            standing.to(torch.bool), self.consecutive + 1,
            torch.zeros_like(self.consecutive))
        success_now = self.consecutive >= self.cfg.hold_steps
        first_success = success_now & ~self.achieved
        self.achieved = self.achieved | success_now
        milestone = (self.consecutive.unsqueeze(-1)
                     == self._milestones.unsqueeze(0))
        return {
            "standing": standing,
            "success_now": success_now,
            "first_success": first_success,
            "milestone": milestone,
        }
