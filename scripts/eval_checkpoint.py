"""Evaluate a saved policy on its task's own fixed start poses (PLAN §5.2).

Complements the in-training eval: works on any checkpoint after the fact,
uses the start kinds of the task being evaluated (a balance policy judged
from the floor measures the wrong thing), and reports the §5.2 quantities
directly — how many of the 24 starts held the success condition for
`hold_s`, the longest and mean continuous standing time, how many survived
the whole trial without falling, and the pelvis height at the end.

Usage:
  .venv/bin/python scripts/eval_checkpoint.py runs/sb_term/ckpt_best.pt \
      --task stand_balance
  .venv/bin/python scripts/eval_checkpoint.py runs/g1_e/ckpt_best.pt --json
"""

import argparse
import json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--task", default="rise_and_stand")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--mono", action="store_true",
                    help="checkpoint from train_mono.py (WindowPPO/BrainPolicy)")
    ap.add_argument("--json", action="store_true", help="print JSON only")
    args = ap.parse_args()

    from multibrain.body.env import WarpBodyEnv
    from multibrain.body.layout import BodyLayout
    from multibrain.body.start_poses import eval_starts

    env = WarpBodyEnv(nworld=args.n, task=args.task, seed=args.seed + 777)
    if args.mono:
        from multibrain.brain import BrainPolicy, build_core
        from multibrain.learning import WindowPPO
        core, _ = build_core(1, device=str(env.device))
        policy = BrainPolicy(core, n_envs=args.n)
        ppo = WindowPPO.load(args.ckpt, env, policy)
        state = policy.init_state(args.n)
    else:
        from multibrain.learning import PPO
        ppo = PPO.load(args.ckpt, env)
        state = None
    ppo.norm.freeze()

    layout = BodyLayout.from_model(env.mjm)
    starts = eval_starts(layout, n=args.n, seed=args.seed, mjm=env.mjm,
                         kinds=env._starts)
    obs = env.reset_to(np.stack([q for _, q in starts]))
    dev = env.device
    streak = torch.zeros(args.n, dtype=torch.long, device=dev)
    achieved = torch.zeros(args.n, dtype=torch.bool, device=dev)
    alive = torch.ones(args.n, dtype=torch.bool, device=dev)
    h_idx = layout.framepos_idx[2]
    h_max = torch.zeros(args.n, device=dev)

    with torch.no_grad():
        for _ in range(env.episode_steps):
            nobs = ppo.norm.normalize(obs)
            if args.mono:
                state, a = ppo.policy.act_deterministic(state, nobs)
                a = ppo.policy.applied_action(a)
            else:
                a = ppo.policy.act_deterministic(nobs)
            obs, _, _, info = env.step(a)
            streak = torch.maximum(streak, env.tracker.consecutive)
            achieved |= info["first_success"]
            alive = alive & ~info.get(
                "fallen", torch.zeros_like(alive))
            h_max = torch.maximum(h_max, env.sensordata()[:, h_idx])

    dt = env.body_step_s
    hold_s = env.success_cfg.hold_steps * dt
    out = {
        "ckpt": args.ckpt, "task": args.task, "n": args.n,
        "starts": sorted({k for k, _ in starts}),
        f"success_{hold_s:.0f}s": int(achieved.sum()),
        "max_standing_s": round(float(streak.max()) * dt, 2),
        "mean_standing_s": round(float(streak.float().mean()) * dt, 2),
        "never_fell": int(alive.sum()),
        "h_max_mean": round(float(h_max.mean()), 3),
        "h_final_mean": round(float(env.sensordata()[:, h_idx].mean()), 3),
    }
    print(json.dumps(out, ensure_ascii=False))
    if not args.json:
        print(f"{out[f'success_{hold_s:.0f}s']}/{args.n} starts held the "
              f"success condition for {hold_s:.0f} s; longest continuous "
              f"stand {out['max_standing_s']} s")
    env.close()


if __name__ == "__main__":
    main()
