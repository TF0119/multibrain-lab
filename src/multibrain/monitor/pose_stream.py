"""Real-time pose streaming to the browser viewer (PLAN.md §11.1).

Wire protocol, version 1 ("MBP1"):

- On connect the server sends one JSON text message, the "meta": joint axes,
  the body tree, per-body geom primitives, the rest qpos, the body -> VRM
  bone map, and segment direction hints. Everything needed to reconstruct
  world poses lives in the meta so frames stay tiny.
- After that it sends binary frames:

      u32  magic  "MBP1"
      f64  sim_t  (seconds)
      u32  seq
      f32  qpos[nq]   (freejoint 7 + hinge angles, model order)

  ~150 bytes per frame for the 27-DoF humanoid.

The viewer does forward kinematics itself (body offsets + joint axes from
meta), so the stream carries qpos only.

Non-blocking contract: ``PoseStreamer.submit()`` runs on the caller's thread,
does one small array copy + struct pack, and touches no socket. All network
I/O lives on a dedicated thread running an asyncio loop. Per client we keep
at most the newest frame; a slow client loses old frames, never the other
way around. If ``websockets`` is missing or the port cannot be bound the
streamer disables itself and ``submit()`` becomes a no-op.

``WarpPoseSource`` feeds env-0 qpos from a mujoco_warp simulation without
synchronizing the caller: each ``enqueue()`` issues a device->device copy
plus a device->pinned-host copy and records a CUDA event; ``read()`` only
checks the event and returns the completed copy once (or None -> not yet).
Loop order per body step: ``read()`` the previous step's copy and submit
it, then ``enqueue()`` the current one. Reading right after enqueueing in
the same step almost always finds the event still pending (the GPU is
busy with that very step), so nearly every frame would be dropped.
"""

import asyncio
import json
import math
import struct
import threading
import time
import warnings

import numpy as np

from .activity import encode_activity

MAGIC = b"MBP1"

# MuJoCo body -> VRM humanoid bone (VRMHumanBoneName in three-vrm).
VRM_BONE_MAP = {
    "pelvis": "hips",
    "torso": "chest",
    "head": "head",
    "upper_arm_L": "leftUpperArm",
    "forearm_L": "leftLowerArm",
    "hand_L": "leftHand",
    "thigh_L": "leftUpperLeg",
    "shin_L": "leftLowerLeg",
    "foot_L": "leftFoot",
    "upper_arm_R": "rightUpperArm",
    "forearm_R": "rightLowerArm",
    "hand_R": "rightHand",
    "thigh_R": "rightUpperLeg",
    "shin_R": "rightLowerLeg",
    "foot_R": "rightFoot",
}

# Direction of the body segment's "bone" in the body's local frame, used by
# the viewer to align each VRM bone to the segment at the rest pose.
# None = no own direction; the viewer inherits the nearest mapped ancestor's
# alignment (head inherits chest, hand inherits forearm).
SEGMENT_DIR = {
    "pelvis": (0.0, 0.0, 1.0),
    "torso": (0.0, 0.0, 1.0),
    "head": None,
    "upper_arm_L": (0.0, 0.0, -1.0),
    "forearm_L": (0.0, 0.0, -1.0),
    "hand_L": None,
    "thigh_L": (0.0, 0.0, -1.0),
    "shin_L": (0.0, 0.0, -1.0),
    # foot box: center 0.04 forward, -0.02 down (see build_xml.TOUCH_SITES)
    "foot_L": (0.894, 0.0, -0.447),
    "upper_arm_R": (0.0, 0.0, -1.0),
    "forearm_R": (0.0, 0.0, -1.0),
    "hand_R": None,
    "thigh_R": (0.0, 0.0, -1.0),
    "shin_R": (0.0, 0.0, -1.0),
    "foot_R": (0.894, 0.0, -0.447),
}


def _qmul(a, b):
    """Quaternion product a*b, (w, x, y, z) order."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _qrot(q, v):
    """Rotate vector v by quaternion q (w, x, y, z)."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(qvec, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _axis_angle(axis, ang):
    s = math.sin(ang / 2.0)
    return (math.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s)


def _norm_axis(axis):
    n = math.sqrt(sum(a * a for a in axis))
    return tuple(a / n for a in axis)


def build_meta(mjm, hz=30.0, body_step_s=0.02, condition=""):
    """Build the JSON meta message for a compiled MuJoCo model.

    Contains everything the viewer needs for FK and rendering: body tree,
    joint axes, geom primitives, rest qpos, VRM bone map, segment dirs.
    """
    import mujoco

    # hinge joints in model order
    joints = []
    hinge_of = {}  # model joint index -> index in `joints`
    for j in range(mjm.njnt):
        if mjm.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j)
        hinge_of[j] = len(joints)
        joints.append(
            {
                "name": name,
                "axis": [float(a) for a in mjm.jnt_axis[j]],
                "qposadr": int(mjm.jnt_qposadr[j]),
            }
        )

    # body tree (world body 0 is implicit; pelvis carries the freejoint)
    bodies = []
    for i in range(1, mjm.nbody):
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i)
        pid = int(mjm.body_parentid[i])
        pname = (
            mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, pid) if pid > 0 else None
        )
        has_free = any(
            mjm.jnt_bodyid[j] == i and mjm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            for j in range(mjm.njnt)
        )
        bodies.append(
            {
                "name": name,
                "parent": pname,
                "pos": [float(v) for v in mjm.body_pos[i]],
                "free": bool(has_free),
                "joints": [
                    hinge_of[j]
                    for j in range(mjm.njnt)
                    if mjm.jnt_bodyid[j] == i and j in hinge_of
                ],
            }
        )

    # geom primitives per body (viewer fallback figure + collision overlay)
    geoms = {}
    for g in range(mjm.ngeom):
        bid = int(mjm.geom_bodyid[g])
        if bid == 0:
            continue  # floor
        bname = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, bid)
        t = int(mjm.geom_type[g])
        pos = [float(v) for v in mjm.geom_pos[g]]
        quat = tuple(float(v) for v in mjm.geom_quat[g])
        size = [float(v) for v in mjm.geom_size[g]]
        ent = None
        if t in (int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                 int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            ax = _qrot(quat, (0.0, 0.0, 1.0))
            hl = size[1]
            a = [pos[k] - ax[k] * hl for k in range(3)]
            b = [pos[k] + ax[k] * hl for k in range(3)]
            ent = {"type": "capsule", "a": a, "b": b, "r": size[0]}
        elif t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            ent = {"type": "sphere", "pos": pos, "r": size[0]}
        elif t == int(mujoco.mjtGeom.mjGEOM_BOX):
            ent = {"type": "box", "pos": pos, "quat": list(quat), "half": size}
        if ent is not None:
            geoms.setdefault(bname, []).append(ent)

    return {
        "type": "meta",
        "version": 1,
        "model": "multibrain_humanoid",
        "condition": condition,
        "hz": float(hz),
        "body_step_s": float(body_step_s),
        "nq": int(mjm.nq),
        "joints": joints,
        "bodies": bodies,
        "geoms": geoms,
        "rest_qpos": [float(v) for v in mjm.qpos0],
        "vrm_map": {b: VRM_BONE_MAP[b] for b in VRM_BONE_MAP if b in geoms
                    or any(bd["name"] == b for bd in bodies)},
        "segment_dir": {b: d for b, d in SEGMENT_DIR.items()
                        if any(bd["name"] == b for bd in bodies)},
    }


def fk(meta, qpos):
    """Forward kinematics matching the viewer's JS implementation.

    Returns {body_name: (pos3, quat_wxyz)} in MuJoCo world coordinates.
    A body's local rotation is the ordered product of its hinge
    axis-angle rotations; joints sit at the body origin.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    info = {b["name"]: b for b in meta["bodies"]}
    joints = meta["joints"]
    out = {}

    def world(name):
        if name in out:
            return out[name]
        b = info[name]
        if b["free"]:
            wpos = (float(qpos[0]), float(qpos[1]), float(qpos[2]))
            wquat = (
                float(qpos[3]),
                float(qpos[4]),
                float(qpos[5]),
                float(qpos[6]),
            )
        else:
            ppos, pquat = world(b["parent"])
            lp = _qrot(pquat, b["pos"])
            wpos = tuple(ppos[k] + lp[k] for k in range(3))
            lquat = (1.0, 0.0, 0.0, 0.0)
            for ji in b["joints"]:
                j = joints[ji]
                lquat = _qmul(lquat, _axis_angle(j["axis"], qpos[j["qposadr"]]))
            wquat = _qmul(pquat, lquat)
        out[name] = (wpos, wquat)
        return out[name]

    for b in meta["bodies"]:
        world(b["name"])
    return out


def encode_frame(seq, t, qpos):
    """Pack one binary pose frame (protocol MBP1)."""
    q = np.asarray(qpos, dtype="<f4").reshape(-1)
    return MAGIC + struct.pack("<dI", float(t), int(seq)) + q.tobytes()


def decode_frame(buf):
    """Decode a binary pose frame -> (t, seq, qpos float32 array)."""
    buf = bytes(buf)
    if buf[:4] != MAGIC:
        raise ValueError("bad magic")
    t, seq = struct.unpack("<dI", buf[4:16])
    return t, seq, np.frombuffer(buf[16:], dtype="<f4")


class PoseStreamer:
    """WebSocket pose server on a dedicated thread (PLAN §11.1).

    submit() is safe to call from the training loop every body step: it
    rate-limits to `hz`, packs the frame, and hands it to the asyncio loop
    with call_soon_threadsafe. With no clients it returns immediately.
    """

    def __init__(self, meta, host="127.0.0.1", port=8765, hz=30.0,
                 activity_hz=10.0):
        self.meta = dict(meta)
        self.meta["hz"] = float(hz)
        self.host = host
        self.port = int(port)
        self.hz = float(hz)
        self._min_interval = 1.0 / self.hz if self.hz > 0 else 0.0
        self.activity_hz = float(activity_hz)
        self._activity_interval = (
            1.0 / self.activity_hz if self.activity_hz > 0 else 0.0)
        self._has_activity = "activity" in self.meta
        self._last_activity = 0.0
        self.enabled = False
        self._loop = None
        self._server = None
        self._thread = None
        self._ready = threading.Event()
        # ws -> ({"pose": frame|None, "activity": frame|None}, asyncio.Event)
        self._clients = {}
        self._n_clients = 0
        self._seq = 0
        self._last_sent = 0.0
        self.stats = {"submitted": 0, "sent": 0, "dropped_rate": 0,
                      "activity_submitted": 0, "activity_sent": 0}

    def start(self):
        try:
            import websockets  # noqa: F401
        except ImportError:
            warnings.warn("websockets not installed; PoseStreamer disabled")
            return self
        self._thread = threading.Thread(
            target=self._run, name="pose-stream", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(10.0):
            warnings.warn("PoseStreamer start timed out")
        return self

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:  # bind failure etc -> disable, never crash
            warnings.warn(f"PoseStreamer disabled: {e}")
            self._ready.set()
            return
        self.enabled = True
        self._ready.set()
        self._loop.run_forever()

    async def _serve(self):
        import websockets

        self._server = await websockets.serve(
            self._handler, self.host, self.port, compression=None
        )
        sock = self._server.sockets[0]
        self.port = int(sock.getsockname()[1])

    async def _handler(self, ws):
        # per client, per frame kind, keep only the newest frame
        state = {"pose": None, "activity": None}
        ready = asyncio.Event()
        self._clients[ws] = (state, ready)
        self._n_clients = len(self._clients)
        try:
            await ws.send(json.dumps(self.meta))
            while True:
                await ready.wait()
                ready.clear()
                if not self.enabled:
                    break
                for kind in ("pose", "activity"):
                    frame = state[kind]
                    if frame is None:
                        continue
                    state[kind] = None
                    await ws.send(frame)
                    self.stats["sent" if kind == "pose"
                               else "activity_sent"] += 1
        except Exception:
            pass
        finally:
            self._clients.pop(ws, None)
            self._n_clients = len(self._clients)

    def submit(self, t, qpos):
        """Offer a frame. Returns True if it was queued for broadcast."""
        if not self.enabled or self._n_clients == 0:
            return False
        now = time.monotonic()
        if now - self._last_sent < self._min_interval:
            self.stats["dropped_rate"] += 1
            return False
        self._last_sent = now
        self._seq += 1
        frame = encode_frame(self._seq, t, qpos)
        self.stats["submitted"] += 1
        try:
            self._loop.call_soon_threadsafe(self._publish, "pose", frame)
        except RuntimeError:
            self.enabled = False
            return False
        return True

    def wants_activity(self):
        """True when an MBA1 activity frame is due (activity_hz limit).

        Read-only peek: it does not consume the slot — submit_activity()
        checks the same condition again and marks the time.
        """
        if not self.enabled or self._n_clients == 0 or not self._has_activity:
            return False
        return time.monotonic() - self._last_activity >= self._activity_interval

    def submit_activity(self, t, stats, sample):
        """Offer an MBA1 activity frame. Same contract as submit()."""
        if not self.enabled or self._n_clients == 0 or not self._has_activity:
            return False
        now = time.monotonic()
        if now - self._last_activity < self._activity_interval:
            return False
        self._last_activity = now
        self._seq += 1
        frame = encode_activity(self._seq, t, stats, sample)
        self.stats["activity_submitted"] += 1
        try:
            self._loop.call_soon_threadsafe(self._publish, "activity", frame)
        except RuntimeError:
            self.enabled = False
            return False
        return True

    def _publish(self, kind, frame):
        for state, ready in self._clients.values():
            state[kind] = frame  # overwrite: only the newest survives
            ready.set()

    def close(self):
        self.enabled = False
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._shutdown)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _shutdown(self):
        self._loop.create_task(self._shutdown_async())

    async def _shutdown_async(self):
        # wake every handler blocked in ready.wait() so it can observe the
        # closed socket and finish; otherwise wait_closed() times out and
        # the interpreter reports destroyed pending tasks at exit
        for _state, ready in list(self._clients.values()):
            ready.set()
        for ws in list(self._clients):
            try:
                await asyncio.wait_for(ws.close(), 1.0)
            except Exception:
                pass
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), 1.0)
            except Exception:
                pass
        self._loop.stop()


class WarpPoseSource:
    """Non-blocking env-`env` qpos readout from a mujoco_warp sim.

    Call ``read()`` first and ``enqueue()`` last in each body step. enqueue
    issues a device->device copy plus a device->pinned-host copy and records
    a CUDA event — all async, the caller never waits on the GPU. read checks
    the recorded event with ``query()`` (no sync) and returns the completed
    host copy once, or None when the copy hasn't landed yet so the caller
    can drop the frame.
    """

    def __init__(self, d, nq, env=0):
        import torch
        import warp as wp

        self.env = int(env)
        self._qpos_view = wp.to_torch(d.qpos)[self.env]  # (nq,) GPU view
        self._stage = torch.empty(nq, device="cuda")
        self._host = torch.empty(nq, pin_memory=True)
        self._event = torch.cuda.Event()
        self._armed = False

    def enqueue(self):
        self._stage.copy_(self._qpos_view, non_blocking=True)
        self._host.copy_(self._stage, non_blocking=True)
        self._event.record()
        self._armed = True

    def read(self):
        if not self._armed or not self._event.query():
            return None
        self._armed = False
        return self._host.numpy().copy()
