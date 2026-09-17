"""PPO training loop for the `mlp_control` condition (PLAN.md §7, §10).

Runs GaussianMlpPolicy + ValueMlp on WarpBodyEnv, one JSONL line per
update in <out>/log.jsonl and a checkpoint <out>/ckpt.pt when
--total-steps (the body-step budget, §10.3's max_body_steps) is reached.
Without --total-steps the run refuses to start (§10.3).

--eval-every N: every N body steps the deterministic policy (a=tanh(mu),
frozen obs norm) is run on the 24 fixed eval_starts (supine/prone/side
x8) for one 60 s episode each, on a dedicated 24-world env; successes go
to the log. 0 disables.

--stream starts a PoseStreamer so env 0 is visible in the viewer (§11.1).

Usage:
  .venv/bin/python scripts/train_mlp.py --nworld 64 --total-steps 2000000 \
      --task rise_and_stand --seed 0 --out runs/mlp_s0 --stream
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "humanoid.xml"

EVAL_SEED = 12345           # eval_starts' fixed seed (§5.2)
EVAL_N = 24                 # 8 each of supine / prone / side


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--total-steps", type=int, default=None,
                    help="body-step budget (= §10.3 max_body_steps); "
                         "required, the run will not start without it")
    ap.add_argument("--task", default="rise_and_stand",
                    choices=["rise_and_stand", "stand_balance"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stream", action="store_true",
                    help="serve env-0 poses on a PoseStreamer")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", default="runs/mlp_control",
                    help="output dir for log.jsonl and ckpt.pt")
    ap.add_argument("--eval-every", type=float, default=5e6,
                    help="body steps between evaluations; 0 disables")
    ap.add_argument("--max-compute-hours", type=float, default=None,
                    help="optional wall-clock cap (§10.3)")
    ap.add_argument("--collect-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None,
                    help="Adam learning rate (default: PPOConfig, 3e-4)")
    ap.add_argument("--entropy-coef", type=float, default=None,
                    help="entropy bonus coefficient (default: 0)")
    ap.add_argument("--reward-cfg", default=None,
                    help="alternative reward yaml (default configs/reward.yaml)")
    ap.add_argument("--no-play-after", action="store_true",
                    help="with --stream: exit when done instead of replaying "
                         "the final policy for the viewer")
    return ap.parse_args()


def run_eval(ppo, eval_env, starts):
    """One 60 s deterministic episode from each eval start.

    Returns {"n": 24, "successes": k, "per_kind": {...}}. Done worlds are
    deliberately not reset — the episode assignment is fixed per world.
    """
    env = eval_env
    n = env.nworld
    qpos = np.stack([q for _, q in starts[:n]])
    kinds = [k for k, _ in starts[:n]]

    ppo.norm.freeze()
    obs = env.reset_to(qpos)
    achieved = torch.zeros(n, dtype=torch.bool, device=env.device)
    first_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
    from multibrain.body.layout import BodyLayout as _BL
    h_idx = _BL.from_model(env.mjm).framepos_idx[2]
    h_max = torch.zeros(n, device=env.device)
    streak = torch.zeros(n, dtype=torch.long, device=env.device)
    with torch.no_grad():
        for i in range(env.episode_steps):
            a = ppo.policy.act_deterministic(ppo.norm.normalize(obs))
            obs, _, _, info = env.step(a)
            h_max = torch.maximum(h_max, env.sensordata()[:, h_idx])
            streak = torch.maximum(streak, env.tracker.consecutive)
            fs = info["first_success"]
            first_step = torch.where(fs & (first_step < 0),
                                     torch.full_like(first_step, i),
                                     first_step)
            achieved |= fs
    ppo.norm.unfreeze()

    h_fin = env.sensordata()[:, h_idx]
    ok = achieved.cpu().numpy()
    t_first = first_step.cpu().numpy()
    per_kind = {}
    for i, k in enumerate(kinds):
        d = per_kind.setdefault(k, {"n": 0, "ok": 0})
        d["n"] += 1
        d["ok"] += int(ok[i])
    return {
        "n": n,
        "successes": int(ok.sum()),
        "per_kind": per_kind,
        "first_success_body_steps": [int(t) for t in t_first if t >= 0],
        "h_max_mean": round(float(h_max.mean()), 4),
        "h_max_best": round(float(h_max.max()), 4),
        # h_max is dominated by the passive transient right after the reset
        # (measured: identical to 4 decimals across four evals 5M steps
        # apart), so the final height is what actually tracks progress
        "h_final_mean": round(float(h_fin.mean()), 4),
        "h_final_best": round(float(h_fin.max()), 4),
        "max_standing_s": round(float(streak.max()) * env.body_step_s, 2),
    }


def main():
    args = parse_args()
    if args.total_steps is None:
        raise SystemExit(
            "--total-steps is required (PLAN.md §10.3: a run without a "
            "max_body_steps budget must not start)")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for mujoco_warp")

    import mujoco

    from multibrain.body.env import WarpBodyEnv
    from multibrain.body.layout import BodyLayout
    from multibrain.body.start_poses import eval_starts
    from multibrain.learning import PPO, PPOConfig

    torch.manual_seed(args.seed)

    streamer = None
    if args.stream:
        from multibrain.monitor import PoseStreamer, build_meta
        mjm = mujoco.MjModel.from_xml_path(str(XML_PATH))
        meta = build_meta(mjm, condition="mlp_control")
        meta["mode"] = "train"
        streamer = PoseStreamer(meta, host=args.host,
                                port=args.port).start()
        if not streamer.enabled:
            raise SystemExit("streamer failed to start")
        print(f"[stream] ws://{args.host}:{streamer.port}")

    env_kw = {}
    if args.reward_cfg:
        env_kw["reward_cfg"] = args.reward_cfg
    env = WarpBodyEnv(nworld=args.nworld, task=args.task,
                      seed=args.seed, streamer=streamer, **env_kw)
    cfg_kw = {"collect_len": args.collect_len}
    if args.lr is not None:
        cfg_kw["lr"] = args.lr
    if args.entropy_coef is not None:
        cfg_kw["entropy_coef"] = args.entropy_coef
    ppo = PPO(env, cfg=PPOConfig(**cfg_kw))
    print(f"[config] {ppo.cfg} reward_cfg={args.reward_cfg or 'configs/reward.yaml'}",
          flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    ckpt_path = out / "ckpt.pt"
    best_path = out / "ckpt_best.pt"
    # best = most eval successes, ties broken by the mean FINAL pelvis height
    # (the peak is the reset transient, see run_eval)
    best_score = (-1, -1.0)

    eval_env = None
    starts = None
    if args.eval_every and args.eval_every > 0:
        eval_env = WarpBodyEnv(nworld=EVAL_N, task=args.task,
                               seed=args.seed + 1000, **env_kw)
        starts = eval_starts(
            BodyLayout.from_model(eval_env.mjm), n=EVAL_N, seed=EVAL_SEED,
            mjm=eval_env.mjm)
    next_eval = args.eval_every
    t_start = time.monotonic()

    def log(rec):
        line = json.dumps(rec)
        with open(log_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    try:
        while ppo.body_steps < args.total_steps:
            stats = ppo.update()
            log({"type": "update", **stats})
            if streamer is not None:
                streamer.submit_status({
                    "mode": "train", "body_steps": ppo.body_steps,
                    "total_steps": int(args.total_steps),
                    "update": ppo.updates,
                    "elapsed_s": round(time.monotonic() - t_start, 1),
                    "steps_per_s": round(stats["steps_per_s"]),
                    "mean_reward": round(stats["mean_reward"], 4),
                    "standing_frac": round(stats["standing_frac"], 4),
                    "successes": stats["successes"],
                    "max_consecutive": stats["max_consecutive"]})
            if (eval_env is not None
                    and ppo.body_steps >= next_eval):
                ev = run_eval(ppo, eval_env, starts)
                score = (ev["successes"], ev["h_final_mean"])
                is_best = score > best_score
                if is_best:
                    best_score = score
                    ppo.save(best_path)
                log({"type": "eval", "body_steps": ppo.body_steps,
                     "best": is_best, **ev})
                next_eval += args.eval_every
            if (args.max_compute_hours is not None
                    and time.monotonic() - t_start
                    > args.max_compute_hours * 3600.0):
                log({"type": "stop", "reason": "max_compute_hours",
                     "body_steps": ppo.body_steps})
                break
    except (Exception, KeyboardInterrupt) as e:
        # §10.3: save the checkpoint before stopping on failure too
        ppo.save(ckpt_path)
        log({"type": "stop", "reason": f"{type(e).__name__}: {e}",
             "body_steps": ppo.body_steps})
        if isinstance(e, KeyboardInterrupt):
            print(f"interrupted; saved {ckpt_path}")
        else:
            raise
    else:
        ppo.save(ckpt_path)
        log({"type": "done", "body_steps": ppo.body_steps,
             "checkpoint": str(ckpt_path)})
        if streamer is not None and not args.no_play_after:
            # keep the viewer alive: replay the final policy on the 24
            # evaluation starts until Ctrl+C (status mode "replay")
            from multibrain.learning import replay_episodes
            env.close()
            env = WarpBodyEnv(nworld=EVAL_N, task=args.task,
                              seed=args.seed + 2000, streamer=streamer)
            play_starts = eval_starts(
                BodyLayout.from_model(env.mjm), n=EVAL_N, seed=EVAL_SEED,
                mjm=env.mjm)
            ppo_play = PPO.load(ckpt_path, env)
            print("[stream] training done; replaying the final policy "
                  "(Ctrl+C to stop)", flush=True)
            try:
                replay_episodes(
                    ppo_play, env, [q for _, q in play_starts],
                    [k for k, _ in play_starts], streamer=streamer,
                    speed=1.0, loop=True,
                    status_extra={"ckpt": str(ckpt_path),
                                  "trained_body_steps": ppo.body_steps})
            except KeyboardInterrupt:
                pass
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()
        if streamer is not None:
            streamer.close()


if __name__ == "__main__":
    main()
