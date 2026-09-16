"""Start poses for the tasks of PLAN.md §5.1.

Base orientations and pelvis heights come from scripts/check_body.py POSES
(supine = -90 deg pitch about +y, prone = +90 deg, side = +90 deg roll
about +x). Every start adds a uniform yaw U(-pi, pi); lying starts scatter
each joint around the middle of its range by +/-20% of the half-range.

`standing` keeps the rest joints plus a small jitter (joints +/-2 deg,
pelvis tilt up to 2 deg about a random horizontal axis): the exactly
symmetric passive body balances upright forever, so the `stand_balance`
start needs a perturbation (PLAN.md §3.8, §5.1). Its pelvis height is the
rest height plus U(0, 0.02 m).

`layout` may be the numpy BodyLayout (BodyLayout.from_model) or a device
copy (.to(device)); the arrays used here are read back as numpy.
"""

import math

import numpy as np

# (w, x, y, z) pelvis quats and pelvis heights, same values as
# scripts/check_body.py POSES.
POSES = {
    "standing": ([1.0, 0.0, 0.0, 0.0], 0.84),
    "supine": ([math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0],
               0.20),
    "prone": ([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
              0.20),
    "side": ([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
             0.20),
}

LYING_JOINT_SCATTER = 0.20            # fraction of the half-range
STANDING_JOINT_JITTER = math.radians(2.0)
STANDING_TILT_MAX = math.radians(2.0)
STANDING_HEIGHT_JITTER = 0.02         # m, added to the rest pelvis height

EVAL_KINDS = ("supine", "prone", "side")


def _np(x) -> np.ndarray:
    """Layout array (numpy or torch, any device) as a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _qmul(a, b):
    """Quaternion product a*b, (w, x, y, z) order (rotates by b then a)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _axis_angle(axis, ang):
    s = math.sin(ang / 2.0)
    return (math.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s)


def start_qpos(layout, kind: str, rng: np.random.Generator) -> np.ndarray:
    """(nq,) qpos for one trial starting in `kind` pose.

    kind is one of "standing", "supine", "prone", "side".
    """
    base_quat, z = POSES[kind]

    qpos = np.zeros(int(layout.nq), dtype=np.float64)
    qpos[2] = z
    yaw = rng.uniform(-math.pi, math.pi)
    quat = _qmul(_axis_angle((0.0, 0.0, 1.0), yaw), base_quat)

    lo = _np(layout.range_lo)
    hi = _np(layout.range_hi)
    adr = _np(layout.qpos_adr)
    if kind == "standing":
        qpos[2] = z + rng.uniform(0.0, STANDING_HEIGHT_JITTER)
        # small lean about a random horizontal axis (see module docstring)
        azimuth = rng.uniform(0.0, 2.0 * math.pi)
        tilt = rng.uniform(0.0, STANDING_TILT_MAX)
        quat = _qmul(
            _axis_angle((math.cos(azimuth), math.sin(azimuth), 0.0), tilt),
            quat)
        joints = np.clip(
            rng.uniform(-STANDING_JOINT_JITTER, STANDING_JOINT_JITTER,
                        size=lo.shape[0]),
            lo, hi)
    else:
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        joints = mid + rng.uniform(
            -LYING_JOINT_SCATTER, LYING_JOINT_SCATTER,
            size=lo.shape[0]) * half
    qpos[3:7] = quat
    qpos[adr] = joints
    return qpos.astype(np.float32)


def eval_starts(layout, n: int = 24, seed: int = 12345):
    """Fixed evaluation starts: supine / prone / side, n/3 of each.

    Fixed seed — these are the §5.2 evaluation conditions and must not
    overlap the training stream. Not used during learning.
    """
    rng = np.random.default_rng(seed)
    per = n // len(EVAL_KINDS)
    out = []
    for kind in EVAL_KINDS:
        for _ in range(per):
            out.append((kind, start_qpos(layout, kind, rng)))
    return out
