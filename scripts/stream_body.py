"""Stream env-0 pose to the browser viewer (PLAN.md §11.1).

CPU modes (default):
  --mode fall    standing start, zero ctrl (uncontrolled fall)
  --mode wave    standing start, sinusoidal ctrl per joint (shows joint motion)
  --mode prone   prone start, zero ctrl
  --replay FILE  stream a saved qpos sequence (T x nq .npy) — same frames as
                 live mode, used for post-hoc checkpoint replay

--warp runs the physics in mujoco_warp with --nworld environments and streams
env 0 through WarpPoseSource (non-blocking staged GPU->host copy).

--activity additionally runs the full-scale BrainCore on random observations
(~U(-1,1)) and streams env-0 neural activity as MBA1 frames (PLAN §11.2):
group stats + fixed sample at up to --activity-hz. --brains N sets K.

--speed 1.0 paces the sim at wall-clock x speed (0 = as fast as possible).
Ctrl+C to stop.

Usage:
  .venv/bin/python scripts/stream_body.py --mode wave
  .venv/bin/python scripts/stream_body.py --warp --nworld 64 --speed 0
  .venv/bin/python scripts/stream_body.py --mode wave --activity --brains 3
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "humanoid.xml"

SUBSTEPS = 4          # 5 ms physics x4 = 20 ms body step
N_OBS = 76            # observation dims (PLAN §3.5)
TARGET_RHO = 0.9      # same as bench_core.py


def poses_start(mjm, pose):
    import math

    d = mujoco.MjData(mjm)
    quat, z = {
        "standing": ((1.0, 0.0, 0.0, 0.0), 0.84),
        "supine": ((math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0), 0.20),
        "prone": ((math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0), 0.20),
    }[pose]
    d.qpos[2] = z
    d.qpos[3:7] = quat
    mujoco.mj_forward(mjm, d)
    return d


def setup_activity(args):
    """Build the full-scale BrainCore like bench_core.py (PLAN §11.2 demo)."""
    import json

    import pandas as pd
    import torch

    from multibrain.brain.core import (
        BETA,
        BrainCore,
        Ports,
        csr_to_torch,
        spectral_radius,
    )
    from multibrain.data.malecns import Graph
    from multibrain.monitor import (
        BrainActivitySource,
        activity_meta,
        build_groups,
        build_sample,
    )

    data = REPO_ROOT / "data"
    dev = torch.device("cuda")
    graph = Graph.load(data / "graph.npz")
    ports_d = json.loads((data / "ports.json").read_text())
    neurons = pd.read_parquet(data / "neurons.parquet")
    groups = build_groups(neurons, ports_d)
    sample = build_sample(neurons, groups)
    k = args.brains
    names = (["mono"] if k == 1
             else ["L", "C", "R"] if k == 3
             else [f"b{i}" for i in range(k)])
    ports = Ports.load(data / "ports.json", graph.body_ids)
    w_unit = graph.base_weights(1.0)
    rho = spectral_radius(csr_to_torch(w_unit, dev))
    c = TARGET_RHO / (rho / (4 * BETA))
    core = BrainCore(graph.base_weights(c), ports, n_brains=k, device=dev)
    src = BrainActivitySource(groups, sample, graph.body_ids.shape[0], k,
                              device=dev, env=args.env)
    print(f"[activity] N={graph.body_ids.shape[0]} K={k} G={len(groups)} "
          f"M={len(sample['rows'])} c={c:.4f}")
    return core, src, activity_meta(groups, sample, names)


def run_cpu(mjm, args, streamer, act=None):
    rng = np.random.default_rng(args.seed)
    phase = rng.uniform(0, 2 * np.pi, mjm.nu)
    d = poses_start(mjm, "prone" if args.mode == "prone" else "standing")
    t_wall0 = time.monotonic()
    t_sim0 = d.time
    steps = 0
    state = None
    if act is not None:
        import torch
        core, act_src, _ = act
        state = core.init_state(1)
        print(f"[cpu] activity: {core.K} brain(s) on {core.w0.device}")
    print(f"[cpu] mode={args.mode} streaming on ws://{args.host}:{streamer.port}")
    while True:
        if args.mode == "wave":
            d.ctrl[:] = 0.6 * np.sin(2 * np.pi * 0.4 * d.time + phase)
        elif args.mode == "fall":
            d.ctrl[:] = 0.0
        for _ in range(SUBSTEPS):
            mujoco.mj_step(mjm, d)
        streamer.submit(d.time, d.qpos)
        if act is not None:
            obs = torch.rand(core.K, 1, N_OBS, device=core.w0.device) * 2 - 1
            with torch.no_grad():
                state, _ = core(state, obs)
            # previous step's copy first (never race the copy issued now)
            r = act_src.read()
            if r is not None:
                streamer.submit_activity(d.time, *r)
            if streamer.wants_activity():
                act_src.enqueue(state[2])
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


def run_warp(mjm, args, streamer, act=None):
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
    state = None
    if act is not None:
        import torch
        core, act_src, _ = act
        state = core.init_state(args.nworld)
        print(f"[warp] activity: {core.K} brain(s) x {args.nworld} envs")
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
        if act is not None:
            obs = torch.rand(core.K, args.nworld, N_OBS,
                             device=core.w0.device) * 2 - 1
            with torch.no_grad():
                state, _ = core(state, obs)
            r = act_src.read()
            if r is not None:
                streamer.submit_activity(t, *r)
            if streamer.wants_activity():
                act_src.enqueue(state[2])
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
    ap.add_argument("--activity", action="store_true",
                    help="stream neuron activity (MBA1) from a BrainCore "
                         "driven by random observations")
    ap.add_argument("--brains", type=int, default=1,
                    help="number of brains K for --activity")
    ap.add_argument("--activity-hz", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from multibrain.monitor import PoseStreamer, build_meta

    mjm = mujoco.MjModel.from_xml_path(str(XML_PATH))
    meta = build_meta(mjm, hz=args.hz, condition="demo")
    act = None
    if args.activity:
        data = REPO_ROOT / "data"
        missing = [f for f in ("graph.npz", "ports.json", "neurons.parquet")
                   if not (data / f).exists()]
        if missing:
            print(f"[activity] missing data files: {missing}; "
                  "exiting (run scripts/build_graph.py + build_ports.py first)")
            return
        if args.replay:
            print("[activity] --replay has no neural state; ignored")
        else:
            import torch
            if not torch.cuda.is_available():
                print("[activity] CUDA not available; BrainCore needs a GPU")
                return
            act = setup_activity(args)
            meta["activity"] = act[2]
    streamer = PoseStreamer(meta, host=args.host, port=args.port,
                            hz=args.hz, activity_hz=args.activity_hz).start()
    if not streamer.enabled:
        raise SystemExit("streamer failed to start")
    try:
        if args.replay:
            run_replay(mjm, args, streamer)
        elif args.warp:
            run_warp(mjm, args, streamer, act)
        else:
            run_cpu(mjm, args, streamer, act)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        streamer.close()


if __name__ == "__main__":
    main()
