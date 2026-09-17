"""Start poses for the tasks of PLAN.md §5.1.

Base orientations and pelvis heights come from scripts/check_body.py POSES
(supine = -90 deg pitch about +y, prone = +90 deg, side = +90 deg roll
about +x). Every start adds a uniform yaw U(-pi, pi); lying starts jitter
each joint around the rest (extended) pose by +/-LYING_JOINT_JITTER so
the limbs stay roughly in the body plane. Scattering around the middle
of the ranges (bent knees and elbows) made limbs point out of that plane
and, once the height is settled on the lowest point, lifted the pelvis
to 0.5 m and dropped the body from there.

`standing` keeps the rest joints plus a small jitter (joints +/-2 deg,
pelvis tilt up to 2 deg about a random horizontal axis): the exactly
symmetric passive body balances upright forever, so the `stand_balance`
start needs a perturbation (PLAN.md §3.8, §5.1). Its pelvis height is the
rest height plus U(0, 0.02 m).

`layout` may be the numpy BodyLayout (BodyLayout.from_model) or a device
copy (.to(device)); the arrays used here are read back as numpy.

When the CPU model `mjm` is given, the pelvis height is settled so that the
lowest point of any body geom sits CLEARANCE above the floor: the scattered
joints otherwise push hands and feet up to 0.4 m below the floor, and the
penetration recovery launches the body metres into the air.
"""

import math

import mujoco
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

LYING_JOINT_JITTER = math.radians(15.0)   # around the rest pose
STANDING_JOINT_JITTER = math.radians(2.0)
STANDING_TILT_MAX = math.radians(2.0)
STANDING_HEIGHT_JITTER = 0.02         # m, added to the rest pelvis height

EVAL_KINDS = ("supine", "prone", "side")
CLEARANCE = 0.005                     # m, lowest geom point above the floor


def lowest_geom_z(mjm, d) -> float:
    """Exact lowest world-z of all body geoms (floor plane excluded)."""
    lo = math.inf
    for g in range(mjm.ngeom):
        if mjm.geom_bodyid[g] == 0:
            continue
        t = int(mjm.geom_type[g])
        c = d.geom_xpos[g]
        R = d.geom_xmat[g].reshape(3, 3)
        size = mjm.geom_size[g]
        if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            z = c[2] - size[0]
        elif t in (int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                   int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            axis_z = R[2, 2] * size[1]
            z = min(c[2] - axis_z, c[2] + axis_z) - size[0]
        else:  # box (and anything else): rotated half extents
            z = c[2] - float(np.abs(R[2]) @ size)
        lo = min(lo, float(z))
    return lo


def settle_height(mjm, qpos: np.ndarray, clearance: float = CLEARANCE):
    """Return qpos with qpos[2] shifted so lowest_geom_z == clearance."""
    d = mujoco.MjData(mjm)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(mjm, d)
    out = np.array(qpos, dtype=np.float64)
    out[2] += clearance - lowest_geom_z(mjm, d)
    return out


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


def start_qpos(layout, kind: str, rng: np.random.Generator,
               mjm=None) -> np.ndarray:
    """(nq,) qpos for one trial starting in `kind` pose.

    kind is one of "standing", "supine", "prone", "side". With `mjm` the
    pelvis height is settled on the lowest geom point (see module doc);
    without it the nominal POSES height is used as-is.
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
        joints = np.clip(
            rng.uniform(-LYING_JOINT_JITTER, LYING_JOINT_JITTER,
                        size=lo.shape[0]),
            lo, hi)
    qpos[3:7] = quat
    qpos[adr] = joints
    if mjm is not None:
        qpos = settle_height(mjm, qpos)
    return qpos.astype(np.float32)


def eval_starts(layout, n: int = 24, seed: int = 12345, mjm=None,
                kinds=EVAL_KINDS):
    """Fixed evaluation starts: n/len(kinds) of each kind.

    Fixed seed — these are the §5.2 evaluation conditions and must not
    overlap the training stream. Not used during learning. `kinds`
    defaults to the rise_and_stand starts; a balance task must pass its
    own (evaluating a balance policy from the floor measures the wrong
    thing).
    """
    rng = np.random.default_rng(seed)
    kinds = list(kinds)
    per = max(1, n // len(kinds))
    out = []
    for kind in kinds:
        for _ in range(per):
            out.append((kind, start_qpos(layout, kind, rng, mjm)))
    return out[:n]
