"""Stream env-0 pose to the browser viewer (PLAN.md §11.1).

CPU modes (default):
  --mode fall    standing start, zero ctrl (uncontrolled fall)
  --mode wave    standing start, sinusoidal ctrl per joint (shows joint motion)
  --mode prone   prone start, zero ctrl
  --replay FILE  stream a saved qpos sequence (T x nq .npy) — same frames as
                 live mode, used for post-hoc checkpoint replay

--warp runs the physics in mujoco_warp with --nworld environments and streams
env 0 through WarpPoseSource (non-blocking staged GPU->host copy).

--speed 1.0 paces the sim at wall-clock x speed (0 = as fast as possible).
Ctrl+C to stop.

Usage:
  .venv/bin/python scripts/stream_body.py --mode wave
  .venv/bin/python scripts/stream_body.py --warp --nworld 64 --speed 0
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "humanoid.xml"

SUBSTEPS = 4          # 5 ms physics x4 = 20 ms body step


def poses_start(mjm, pose):
    import math

    d = mujoco.MjData(mjm)
    quat, z = {
        "standing": ((1.0, 0.0, 0.0, 0.0), 0.84),
        "supine": ((math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0), 0.20),
        "prone": ((math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0), 0.20),
    }[pose]
    d.qpos[2] = z
    d.qpos[3:7] = quat
    mujoco.mj_forward(mjm, d)
    return d


def run_cpu(mjm, args, streamer):
    rng = np.random.default_rng(args.seed)
    phase = rng.uniform(0, 2 * np.pi, mjm.nu)
    d = poses_start(mjm, "prone" if args.mode == "prone" else "standing")
    t_wall0 = time.monotonic()
    t_sim0 = d.time
    steps = 0
    print(f"[cpu] mode={args.mode} streaming on ws://{args.host}:{streamer.port}")
    while True:
        if args.mode == "wave":
            d.ctrl[:] = 0.6 * np.sin(2 * np.pi * 0.4 * d.time + phase)
        elif args.mode == "fall":
            d.ctrl[:] = 0.0
        for _ in range(SUBSTEPS):
            mujoco.mj_step(mjm, d)
        streamer.submit(d.time, d.qpos)
        steps += 1
        if args.duration and d.time - t_sim0 > args.duration:
            break
        if args.speed > 0:
            target = (d.time - t_sim0) / args.speed
            lag = target - (time.monotonic() - t_wall0)
            if lag > 0:
                time.sleep(min(lag, 0.05))
        if args.duration == 0 and args.speed <= 0 and steps % 5000 == 0:
            print(f"[cpu] t={d.time:.1f}s sent={streamer.stats}")
    print(f"[cpu] done t={d.time:.2f}s stats={streamer.stats}")


def run_replay(mjm, args, streamer):
    q = np.load(args.replay)
    assert q.ndim == 2 and q.shape[1] == mjm.nq, q.shape
    dt = args.dt if args.dt else 0.02
    print(f"[replay] {args.replay}: {q.shape[0]} frames @ {dt}s")
    t0 = time.monotonic()
    while True:
        for i in range(q.shape[0]):
            streamer.submit(i * dt, q[i])
            if args.speed > 0:
                lag = i * dt / args.speed - (time.monotonic() - t0)
                if lag > 0:
                    time.sleep(min(lag, 0.05))
        if not args.loop:
            break
    print(f"[replay] done stats={streamer.stats}")


def run_warp(mjm, args, streamer):
    import mujoco_warp
    import warp as wp
    from multibrain.monitor import WarpPoseSource

    wp.init()
    mjd0 = poses_start(mjm, "standing")
    m = mujoco_warp.put_model(mjm)
    d = mujoco_warp.put_data(mjm, mjd0, nworld=args.nworld)
    src = WarpPoseSource(d, mjm.nq, env=args.env)
    rng = np.random.default_rng(args.seed)
    phase = rng.uniform(0, 2 * np.pi, (args.nworld, mjm.nu))
    t_wall0 = time.monotonic()
    body = 0
    print(f"[warp] nworld={args.nworld} env={args.env} "
          f"ws://{args.host}:{streamer.port}")
    while True:
        t = body * SUBSTEPS * mjm.opt.timestep
        if args.mode == "wave":
            d.ctrl = wp.array(
                (0.6 * np.sin(2 * np.pi * 0.4 * t + phase)).astype(np.float32))
        for _ in range(SUBSTEPS):
            mujoco_warp.step(m, d)
        src.enqueue()
        q = src.read()
        if q is not None:
            streamer.submit(t, q)
        body += 1
        if args.duration and t > args.duration:
            break
        if args.speed > 0:
            lag = t / args.speed - (time.monotonic() - t_wall0)
            if lag > 0:
                time.sleep(min(lag, 0.05))
    wp.synchronize()
    print(f"[warp] done t={t:.2f}s stats={streamer.stats}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fall",
                    choices=["fall", "wave", "prone"])
    ap.add_argument("--replay", default=None, help="qpos .npy file to replay")
    ap.add_argument("--dt", type=float, default=0.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--warp", action="store_true")
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from multibrain.monitor import PoseStreamer, build_meta

    mjm = mujoco.MjModel.from_xml_path(str(XML_PATH))
    meta = build_meta(mjm, hz=args.hz, condition="demo")
    streamer = PoseStreamer(meta, host=args.host, port=args.port,
                            hz=args.hz).start()
    if not streamer.enabled:
        raise SystemExit("streamer failed to start")
    try:
        if args.replay:
            run_replay(mjm, args, streamer)
        elif args.warp:
            run_warp(mjm, args, streamer)
        else:
            run_cpu(mjm, args, streamer)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        streamer.close()


if __name__ == "__main__":
    main()
