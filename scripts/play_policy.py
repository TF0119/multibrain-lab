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
    from multibrain.learning import PPO, replay_episodes

    streamer = None
    if not args.no_stream:
        from multibrain.monitor import PoseStreamer, build_meta
        mjm = mujoco.MjModel.from_xml_path(str(XML_PATH))
        meta = build_meta(mjm, condition="mlp_control")
        meta["mode"] = "replay"
        streamer = PoseStreamer(meta, host=args.host, port=args.port).start()
        print(f"[stream] ws://{args.host}:{streamer.port}", flush=True)

    env = WarpBodyEnv(nworld=args.nworld, task=args.task, seed=args.seed,
                      streamer=streamer)
    ppo = PPO.load(args.ckpt, env)
    layout = BodyLayout.from_model(env.mjm)
    if args.random:
        rng = np.random.default_rng(args.seed)
        from multibrain.body.start_poses import start_qpos
        kinds_all = ["supine", "prone", "side"]
        starts = [(kinds_all[i % 3], start_qpos(layout, kinds_all[i % 3], rng, env.mjm))
                  for i in range(env.nworld)]
    else:
        starts = eval_starts(layout, n=24, mjm=env.mjm, kinds=env._starts)
        starts = starts[args.world:] + starts[:args.world]
    dt = env.body_step_s

    def on_episode(episode, rows):
        print(json.dumps({"episode": episode, "episode_s": env.episode_steps * dt,
                          "successes": sum(r["success"] for r in rows),
                          "h_max_mean": round(sum(r["h_max"] for r in rows) / len(rows), 3),
                          "h_final_mean": round(sum(r["h_final"] for r in rows) / len(rows), 3)}),
              flush=True)
        for row in rows:
            print(json.dumps(row), flush=True)

    try:
        replay_episodes(ppo, env, [q for _, q in starts], [k for k, _ in starts],
                        streamer=streamer, speed=args.speed, loop=args.loop,
                        status_extra={"ckpt": str(args.ckpt),
                                      "trained_body_steps": ppo.body_steps},
                        on_episode=on_episode)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        env.close()
        if streamer is not None:
            streamer.close()


if __name__ == "__main__":
    main()
