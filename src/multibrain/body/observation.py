"""The 76-dim observation vector of PLAN.md §3.5, built from sensordata.

Layout of the observation (OBS_SLICES):
  [0:27]   joint angles, normalized to [-1, 1] by the joint range
  [27:54]  joint velocities, qd/10 clamped to [-3, 3]
  [54:57]  gravity direction in the pelvis (imu) frame, R(q_imu)^T (0,0,-1)
  [57:60]  pelvis angular velocity (gyro)
  [60:63]  pelvis linear velocity (velocimeter)
  [63]     pelvis height / h_ref
  [64:76]  touch forces, F/50 clamped to [0, 1] (TOUCH_SITES order)
"""

import numpy as np
import torch

from .layout import BodyLayout

OBS_DIM = 76

OBS_SLICES = {
    "jointpos": slice(0, 27),
    "jointvel": slice(27, 54),
    "gravity": slice(54, 57),
    "gyro": slice(57, 60),
    "velocimeter": slice(60, 63),
    "height": slice(63, 64),
    "touch": slice(64, 76),
}


def _idx(sensordata: torch.Tensor, idx) -> torch.Tensor:
    """Index array (numpy or torch) as a long tensor on sensordata's device."""
    if isinstance(idx, torch.Tensor):
        return idx.to(device=sensordata.device, dtype=torch.long)
    return torch.as_tensor(np.asarray(idx), dtype=torch.long,
                           device=sensordata.device)


def _val(sensordata: torch.Tensor, arr) -> torch.Tensor:
    """Layout array (numpy or torch) as a tensor matching sensordata."""
    if isinstance(arr, torch.Tensor):
        return arr.to(device=sensordata.device, dtype=sensordata.dtype)
    return torch.as_tensor(np.asarray(arr), dtype=sensordata.dtype,
                           device=sensordata.device)


def observe(layout: BodyLayout, sensordata: torch.Tensor) -> torch.Tensor:
    """(B, nsensordata) sensordata -> (B, 76) observation."""
    s = sensordata
    q = s[:, _idx(s, layout.jointpos_idx)]
    qd = s[:, _idx(s, layout.jointvel_idx)]
    quat = s[:, _idx(s, layout.framequat_idx)]   # (B, 4) w, x, y, z
    gyro = s[:, _idx(s, layout.gyro_idx)]
    vel = s[:, _idx(s, layout.velocimeter_idx)]
    pos = s[:, _idx(s, layout.framepos_idx)]
    touch = s[:, _idx(s, layout.touch_idx)]

    lo = _val(s, layout.range_lo)
    hi = _val(s, layout.range_hi)
    qn = 2.0 * (q - lo) / (hi - lo) - 1.0
    qdn = torch.clamp(qd / 10.0, -3.0, 3.0)

    # gravity (0,0,-1) in the imu frame = -(3rd row of R); with quat (w,x,y,z):
    # row 2 of R is (2(xz-wy), 2(yz+wx), 1-2(x^2+y^2))
    w, x, y, z = quat.unbind(-1)
    gravity = torch.stack([
        2.0 * (w * y - x * z),
        -2.0 * (y * z + w * x),
        2.0 * (x * x + y * y) - 1.0,
    ], dim=-1)

    height = (pos[:, 2] / float(layout.h_ref)).unsqueeze(-1)
    contact = torch.clamp(touch / 50.0, 0.0, 1.0)

    return torch.cat([qn, qdn, gravity, gyro, vel, height, contact], dim=-1)
