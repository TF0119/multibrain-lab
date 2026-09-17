"""Deterministic replay of a trained policy with viewer streaming.

Used by scripts/play_policy.py and by scripts/train_mlp.py after training
(so the viewer keeps showing the final policy instead of a closed socket).
The status message tells the viewer it is watching a replay, which
checkpoint, and where in the episode it is.
"""

import time

import numpy as np
import torch


def replay_episodes(ppo, env, starts, kinds, streamer=None, speed=1.0,
                    loop=True, status_extra=None, on_episode=None):
    """Run 60 s deterministic episodes from `starts` (list of qpos rows).

    Returns after one episode when loop is False; otherwise runs until
    KeyboardInterrupt. on_episode(rows) receives per-world summaries.
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
    kinds = list(kinds) + list(kinds)[:n - len(kinds)]
    episode = 0
    t_wall0 = time.monotonic()
    while True:
        obs = env.reset_to(np.stack(rows_q[:n]))
        h_max = torch.zeros(n, device=env.device)
        streak = torch.zeros(n, dtype=torch.long, device=env.device)
        achieved = torch.zeros(n, dtype=torch.bool, device=env.device)
        t0 = time.monotonic()
        last_status = 0.0
        with torch.no_grad():
            for i in range(env.episode_steps):
                a = ppo.policy.act_deterministic(ppo.norm.normalize(obs))
                obs, r, done, info = env.step(a)
                h = env.sensordata()[:, h_idx]
                h_max = torch.maximum(h_max, h)
                streak = torch.maximum(streak, env.tracker.consecutive)
                achieved |= info["first_success"]
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
        h_fin = env.sensordata()[:, h_idx].cpu().numpy()
        rows = [{"world": w, "start": kinds[w],
                 "h_max": round(float(h_max[w]), 3),
                 "h_final": round(float(h_fin[w]), 3),
                 "max_standing_s": round(float(streak[w]) * dt, 2),
                 "success": bool(achieved[w])} for w in range(n)]
        if on_episode is not None:
            on_episode(episode, rows)
        episode += 1
        if not loop:
            return rows
