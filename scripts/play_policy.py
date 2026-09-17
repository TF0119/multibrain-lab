"""Replay a trained mlp_control checkpoint and stream env 0 to the viewer.

Loads runs/<name>/ckpt.pt (PPO.save), builds a WarpBodyEnv with the 24
fixed evaluation starts (or random task starts with --random), runs the
deterministic policy (a = tanh(mu), frozen obs normalization) for one
episode at wall-clock speed --speed, and prints a per-world summary:
start kind, max / final pelvis height, longest standing streak, success.
--loop repeats episodes until Ctrl+C; --world picks which world is
streamed as env 0 by rotating the start list.

Usage:
  .venv/bin/python scripts/play_policy.py runs/g1_smoke/ckpt.pt --port 8766
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "humanoid.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--nworld", type=int, default=24)
    ap.add_argument("--task", default="rise_and_stand")
    ap.add_argument("--random", action="store_true",
                    help="random task starts instead of the 24 eval starts")
    ap.add_argument("--world", type=int, default=0,
                    help="which eval start is placed in env 0 (streamed)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="wall-clock pacing (0 = as fast as possible)")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from multibrain.body.env import WarpBodyEnv
    from multibrain.body.layout import BodyLayout
    from multibrain.body.start_poses import eval_starts
    from multibrain.learning import PPO

    streamer = None
    if not args.no_stream:
        from multibrain.monitor import PoseStreamer, build_meta
        mjm = mujoco.MjModel.from_xml_path(str(XML_PATH))
        streamer = PoseStreamer(build_meta(mjm, condition="mlp_control/play"),
                                host=args.host, port=args.port).start()
        print(f"[stream] ws://{args.host}:{streamer.port}", flush=True)

    env = WarpBodyEnv(nworld=args.nworld, task=args.task, seed=args.seed,
                      streamer=streamer)
    ppo = PPO.load(args.ckpt, env)
    ppo.norm.freeze()
    layout = BodyLayout.from_model(env.mjm)
    starts = eval_starts(layout, n=24)
    starts = starts[args.world:] + starts[:args.world]
    dt = env.body_step_s
    h_idx = layout.framepos_idx[2]
    try:
        while True:
            if args.random:
                obs = env.reset()
                kinds = ["random"] * env.nworld
            else:
                rows = [q for _, q in starts[:env.nworld]]
                while len(rows) < env.nworld:
                    rows += rows[:env.nworld - len(rows)]
                obs = env.reset_to(np.stack(rows))
                kinds = [k for k, _ in starts[:env.nworld]]
                kinds += kinds[:env.nworld - len(kinds)]
            n = env.nworld
            h_max = torch.zeros(n, device=env.device)
            streak = torch.zeros(n, dtype=torch.long, device=env.device)
            achieved = torch.zeros(n, dtype=torch.bool, device=env.device)
            t0 = time.monotonic()
            with torch.no_grad():
                for i in range(env.episode_steps):
                    a = ppo.policy.act_deterministic(ppo.norm.normalize(obs))
                    obs, r, done, info = env.step(a)
                    h = env.sensordata()[:, h_idx]
                    h_max = torch.maximum(h_max, h)
                    streak = torch.maximum(streak, env.tracker.consecutive)
                    achieved |= info["first_success"]
                    if args.speed > 0:
                        lag = (i + 1) * dt / args.speed - (time.monotonic() - t0)
                        if lag > 0:
                            time.sleep(min(lag, 0.05))
            h_fin = env.sensordata()[:, h_idx].cpu().numpy()
            rows = []
            for w in range(n):
                rows.append({"world": w, "start": kinds[w],
                             "h_max": round(float(h_max[w]), 3),
                             "h_final": round(float(h_fin[w]), 3),
                             "max_standing_s": round(float(streak[w]) * dt, 2),
                             "success": bool(achieved[w])})
            print(json.dumps({"episode_s": env.episode_steps * dt,
                              "successes": int(achieved.sum()),
                              "h_max_mean": round(float(h_max.mean()), 3),
                              "h_final_mean": round(float(h_fin.mean()), 3)}),
                  flush=True)
            for row in rows:
                print(json.dumps(row), flush=True)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("stopped")
    finally:
        env.close()
        if streamer is not None:
            streamer.close()


if __name__ == "__main__":
    main()
