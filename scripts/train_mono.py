"""PPO training loop for the `mono` condition (PLAN.md §2, §7, §10).

One full-scale MaleCNS-derived neural core drives all 27 DoF. Learning is
WindowPPO (truncated-window BPTT, §7): collection of --collect-len body
steps is split into --window windows; each update replays one window at a
time from its stored detached start state and takes one optimizer step
per window. Separate Adam learning rates for the core (--lr-core, 1e-4)
and ports/readout (--lr-ports, 3e-4).

Logging mirrors train_mlp.py: one JSONL line per update in
<out>/log.jsonl plus the §12.3 neural diagnostics (s_mean,
s_saturated_frac, v_abs_max, core/ports grad norms), a checkpoint
<out>/ckpt.pt at --total-steps (required, §10.3) and <out>/ckpt_best.pt
for the best eval (successes, then mean peak pelvis height).

--stream serves env-0 poses on a PoseStreamer (§11.1) and, unless
--no-activity, env-0 neuron activity as MBA1 frames (§11.2). WindowPPO
has no per-body-step hook, so activity is delivered **once per update**
from the post-collection state ppo._state[2]; the effective rate is
~1/(collect+update) Hz (~0.2 Hz at 64 envs, ~0.1 Hz at 256) and
--activity-hz acts only as a cap. Delivery order per update is
read() -> submit_activity() -> enqueue() (reading after enqueueing in
the same step would always find the event pending).

Usage:
  .venv/bin/python scripts/train_mono.py --nworld 64 --total-steps 2000000 \
      --task rise_and_stand --seed 0 --out runs/mono_s0 --stream
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "humanoid.xml"
DATA = REPO_ROOT / "data"

EVAL_SEED = 12345           # eval_starts' fixed seed (§5.2)
EVAL_N = 24                 # 8 each of supine / prone / side


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=64,
                    help="parallel envs (§4.5 baseline 64 for the neural "
                         "core, not mlp_control's 256)")
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
    ap.add_argument("--out", default="runs/mono",
                    help="output dir for log.jsonl and ckpt.pt")
    ap.add_argument("--eval-every", type=float, default=5e6,
                    help="body steps between evaluations; 0 disables")
    ap.add_argument("--max-compute-hours", type=float, default=None,
                    help="optional wall-clock cap (§10.3)")
    ap.add_argument("--window", type=int, default=16,
                    help="truncated BPTT window in body steps (§7)")
    ap.add_argument("--collect-len", type=int, default=64)
    ap.add_argument("--lr-core", type=float, default=1e-4,
                    help="Adam lr for the neural core (§7)")
    ap.add_argument("--lr-ports", type=float, default=3e-4,
                    help="Adam lr for ports, readout and value (§7)")
    ap.add_argument("--lr", type=float, default=None,
                    help="shortcut: set both --lr-core and --lr-ports")
    ap.add_argument("--entropy-coef", type=float, default=None,
                    help="entropy bonus coefficient (default: 0)")
    ap.add_argument("--checkpoint", action="store_true",
                    help="torch.utils.checkpoint each window step "
                         "(cuts peak GPU memory at some speed cost)")
    ap.add_argument("--activity-hz", type=float, default=10.0,
                    help="cap on the activity frame rate (§11.2); the "
                         "effective rate is once per update")
    ap.add_argument("--no-activity", action="store_true",
                    help="do not stream neuron activity")
    ap.add_argument("--reward-cfg", default=None,
                    help="alternative reward yaml (default configs/reward.yaml)")
    ap.add_argument("--no-play-after", action="store_true",
                    help="with --stream: exit when done instead of replaying "
                         "the final policy for the viewer")
    return ap.parse_args()


def run_eval(ppo, eval_env, starts):
    """One 60 s deterministic episode from each eval start.

    Same protocol as train_mlp.run_eval (done worlds are not reset — the
    episode assignment is fixed per world), but the neural policy carries
    a state: it is initialized once at the start of the evaluation.
    """
    env = eval_env
    n = env.nworld
    qpos = np.stack([q for _, q in starts[:n]])
    kinds = [k for k, _ in starts[:n]]

    ppo.norm.freeze()
    obs = env.reset_to(qpos)
    state = ppo.policy.init_state(n)
    achieved = torch.zeros(n, dtype=torch.bool, device=env.device)
    first_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
    from multibrain.body.layout import BodyLayout as _BL
    h_idx = _BL.from_model(env.mjm).framepos_idx[2]
    h_max = torch.zeros(n, device=env.device)
    streak = torch.zeros(n, dtype=torch.long, device=env.device)
    with torch.no_grad():
        for i in range(env.episode_steps):
            state, a = ppo.policy.act_deterministic(
                state, ppo.norm.normalize(obs))
            obs, _, _, info = env.step(ppo.policy.applied_action(a))
            h_max = torch.maximum(h_max, env.sensordata()[:, h_idx])
            streak = torch.maximum(streak, env.tracker.consecutive)
            fs = info["first_success"]
            first_step = torch.where(fs & (first_step < 0),
                                     torch.full_like(first_step, i),
                                     first_step)
            achieved |= fs
    ppo.norm.unfreeze()

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
        "max_standing_s": round(float(streak.max()) * env.body_step_s, 2),
    }


def replay_policy(ppo, env, starts, kinds, streamer=None, act_src=None,
                  speed=1.0, loop=True, status_extra=None):
    """Deterministic replay of a BrainPolicy with viewer streaming.

    replay_episodes() assumes a stateless policy (act_deterministic(obs)
    -> a); BrainPolicy carries (v, I, s), so this mirrors that loop with
    the state re-initialized at the start of every trial. Activity keeps
    streaming on the same read -> submit -> enqueue order as training.
    """
    from multibrain.body.layout import BodyLayout

    ppo.norm.freeze()
    layout = BodyLayout.from_model(env.mjm)
    h_idx = layout.framepos_idx[2]
    dt = env.body_step_s
    n = env.nworld
    rows_q = list(starts)
    while len(rows_q) < n:
        rows_q += rows_q[:n - len(rows_q)]
    episode = 0
    t_wall0 = time.monotonic()
    while True:
        obs = env.reset_to(np.stack(rows_q[:n]))
        state = ppo.policy.init_state(n)
        t0 = time.monotonic()
        last_status = 0.0
        with torch.no_grad():
            for i in range(env.episode_steps):
                state, a = ppo.policy.act_deterministic(
                    state, ppo.norm.normalize(obs))
                obs, r, done, info = env.step(ppo.policy.applied_action(a))
                if act_src is not None and streamer is not None:
                    rr = act_src.read()
                    if rr is not None:
                        streamer.submit_activity(i * dt, *rr)
                    if streamer.wants_activity():
                        act_src.enqueue(state[2])
                now = time.monotonic()
                if streamer is not None and now - last_status >= 1.0:
                    last_status = now
                    streamer.submit_status({
                        "mode": "replay", "episode": episode,
                        "episode_t": round((i + 1) * dt, 1),
                        "episode_s": env.episode_steps * dt,
                        "elapsed_s": round(now - t_wall0, 1),
                        **(status_extra or {})})
                if speed > 0:
                    lag = (i + 1) * dt / speed - (now - t0)
                    if lag > 0:
                        time.sleep(min(lag, 0.05))
        episode += 1
        if not loop:
            return


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
    from multibrain.brain import BrainPolicy, build_core
    from multibrain.learning import ValueMlp, WindowPPO, WindowPPOConfig

    torch.manual_seed(args.seed)

    # Full-scale mono core. The wiring and the port assignment are fixed
    # across runs and seeds (build_core's own seed=0 for Ports.load), so
    # --seed only drives initialization, sampling and env resets.
    core, info = build_core(1, device="cuda")

    streamer = None
    groups = sample = None
    if args.stream:
        from multibrain.monitor import (
            PoseStreamer,
            activity_meta,
            build_groups,
            build_meta,
            build_sample,
        )
        mjm = mujoco.MjModel.from_xml_path(str(XML_PATH))
        meta = build_meta(mjm, condition="mono")
        meta["mode"] = "train"
        if not args.no_activity:
            if (DATA / "neurons.parquet").exists():
                import pandas as pd
                neurons = pd.read_parquet(DATA / "neurons.parquet")
                ports_d = json.loads((DATA / "ports.json").read_text())
                groups = build_groups(neurons, ports_d)
                sample = build_sample(neurons, groups)
                meta["activity"] = activity_meta(groups, sample, ["mono"],
                                                 hz=args.activity_hz)
            else:
                print("[stream] data/neurons.parquet missing; "
                      "activity HUD disabled", flush=True)
        streamer = PoseStreamer(meta, host=args.host, port=args.port,
                                activity_hz=args.activity_hz).start()
        if not streamer.enabled:
            raise SystemExit("streamer failed to start")
        print(f"[stream] ws://{args.host}:{streamer.port}", flush=True)

    env_kw = {}
    if args.reward_cfg:
        env_kw["reward_cfg"] = args.reward_cfg
    env = WarpBodyEnv(nworld=args.nworld, task=args.task,
                      seed=args.seed, streamer=streamer, **env_kw)

    act_src = None
    if streamer is not None and groups is not None:
        from multibrain.monitor import BrainActivitySource
        act_src = BrainActivitySource(groups, sample, info["N"], 1,
                                      device=env.device, env=0)

    policy = BrainPolicy(core, n_envs=args.nworld)
    value = ValueMlp(env.obs_dim, env.act_dim)
    lr_core = args.lr if args.lr is not None else args.lr_core
    lr_ports = args.lr if args.lr is not None else args.lr_ports
    cfg_kw = {"window": args.window, "collect_len": args.collect_len,
              "lr_core": lr_core, "lr_ports": lr_ports,
              "checkpoint": args.checkpoint}
    if args.entropy_coef is not None:
        cfg_kw["entropy_coef"] = args.entropy_coef
    ppo = WindowPPO(env, policy, value=value, cfg=WindowPPOConfig(**cfg_kw))
    n_pol = sum(p.numel() for p in policy.parameters())
    n_val = sum(p.numel() for p in ppo.value.parameters())
    print(f"[config] mono core: N={info['N']} nnz={info['nnz']} "
          f"c={info['c']:.4f} rho_unit={info['rho_unit']:.4f} "
          f"learnable={n_pol + n_val} (policy {n_pol} + value {n_val})",
          flush=True)
    print(f"[config] {ppo.cfg} nworld={args.nworld} seed={args.seed} "
          f"task={args.task} "
          f"reward_cfg={args.reward_cfg or 'configs/reward.yaml'}",
          flush=True)
    if act_src is not None:
        print(f"[config] activity: once per update (WindowPPO has no "
              f"per-step hook); effective rate ~= 1/(collect+update) "
              f"(~0.2 Hz at nworld=64, ~0.1 Hz at 256); "
              f"--activity-hz {args.activity_hz} is only a cap",
              flush=True)
    elif args.stream:
        print("[config] activity: disabled (--no-activity)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    ckpt_path = out / "ckpt.pt"
    best_path = out / "ckpt_best.pt"
    # best = most eval successes, ties broken by the mean peak pelvis height
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
            # §12.3 diagnostics from the post-collection neural state
            v, _, s = ppo._state
            s_mean = float(s.mean())
            s_sat = float((s > 0.99).float().mean())
            v_abs = float(v.abs().max())
            period = stats["collect_s"] + stats["update_s"]
            eff_hz = (min(args.activity_hz, 1.0 / period)
                      if act_src is not None else 0.0)
            log({"type": "update", **stats,
                 "s_mean": round(s_mean, 6),
                 "s_saturated_frac": round(s_sat, 6),
                 "v_abs_max": round(v_abs, 4),
                 "activity_per": "update",
                 "activity_hz": round(eff_hz, 3)})
            # activity: read the previous update's copy, submit it, then
            # enqueue this update's state (never enqueue -> read in the
            # same step; the event would still be pending)
            if act_src is not None:
                r = act_src.read()
                if r is not None:
                    streamer.submit_activity(
                        ppo.body_steps / env.nworld * env.body_step_s, *r)
                if streamer.wants_activity():
                    act_src.enqueue(s)
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
                    "max_consecutive": stats["max_consecutive"],
                    "activity_hz": round(eff_hz, 3),
                    "core_grad_norm": round(stats["core_grad_norm"], 4)})
            if (eval_env is not None
                    and ppo.body_steps >= next_eval):
                ev = run_eval(ppo, eval_env, starts)
                score = (ev["successes"], ev["h_max_mean"])
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
            # keep the viewer alive: replay the final policy from the 24
            # evaluation starts until Ctrl+C (status mode "replay").
            # BrainPolicy is stateful, so replay_policy re-initializes
            # the neural state at the start of every trial.
            env.close()
            env = WarpBodyEnv(nworld=EVAL_N, task=args.task,
                              seed=args.seed + 2000, streamer=streamer)
            play_starts = eval_starts(
                BodyLayout.from_model(env.mjm), n=EVAL_N, seed=EVAL_SEED,
                mjm=env.mjm)
            ppo_play = WindowPPO.load(ckpt_path, env, policy, ppo.value)
            print("[stream] training done; replaying the final policy "
                  "(Ctrl+C to stop)", flush=True)
            try:
                replay_policy(
                    ppo_play, env, [q for _, q in play_starts],
                    [k for k, _ in play_starts], streamer=streamer,
                    act_src=act_src, speed=1.0, loop=True,
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
