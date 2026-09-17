"""Truncated-window BPTT PPO for the BrainPolicy (PLAN.md §7).

The MLP learner (ppo.py) shuffles the flattened (T*B) batch freely; the
recurrent neural core cannot — time only mixes at *window* granularity.
A collection of T = cfg.collect_len body steps is split into windows of
cfg.window steps. Each update replays one window at a time from its
stored (detached) start state, backpropagates through that window only,
and takes one optimizer step per window. The window order is shuffled
each epoch; the time axis inside a window is always replayed in order
and the computation graph never crosses a window boundary (§7: at full
scale one 16-step window already costs ~2.8 GB at 64 envs, so joining
windows would quadruple the peak).

For the replay to reproduce the rollout exactly, collection stores the
three things §7 requires: each window's start state (detached clones),
the *normalized* observations the policy actually saw (same reason as
ppo.py — re-normalizing with moved statistics would shift logp_old), and
the per-step episode-end mask. When the env resets mid-window the replay
must reset the neural state at the same step (policy.reset_state on
buf["end"]), otherwise the recomputed trajectory diverges.

The buffer layout, GAE and time-out bootstrapping mirror ppo.py:
``term`` is a real termination, ``end`` any episode end, and time-out
transitions bootstrap through the value at the timed-out state. The
optimizer is a single Adam over policy.param_groups(lr_core, lr_ports)
plus the value net at lr_ports. The grad-norm cap is applied to the
policy (core + ports together) and the value separately; the core and
port grad norms are logged individually for diagnosis.
"""

import dataclasses
import time
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint as _ckpt

from multibrain.brain.policy import BrainPolicy
from .mlp_policy import ValueMlp
from .running_norm import RunningNorm


@dataclass
class WindowPPOConfig:
    window: int = 16            # T, body steps per window (§7)
    collect_len: int = 64
    epochs: int = 4
    clip: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    lr_core: float = 1e-4       # §7: neural core
    lr_ports: float = 3e-4      # §7: ports and readout
    vf_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    checkpoint: bool = False    # wrap each step in torch.utils.checkpoint


def _grad_norm(params) -> float:
    """Total grad norm of a parameter subset (for logging only)."""
    sq = None
    for p in params:
        if p.grad is not None:
            s = p.grad.detach().pow(2).sum()
            sq = s if sq is None else sq + s
    return 0.0 if sq is None else float(sq.sqrt())


class WindowPPO:
    """PPO driver for a recurrent BrainPolicy on a WarpBodyEnv.

    update() collects cfg.collect_len body steps and runs one windowed
    PPO update, returning a flat dict of scalars for logging (same keys
    as PPO plus update_s and the core/ports grad norms). save()/load()
    carry policy, value, norm, optimizer, step counters and the config;
    load() takes the policy and value instances because the core's fixed
    wiring lives outside the state dict.
    """

    def __init__(self, env, policy: BrainPolicy,
                 value: ValueMlp | None = None,
                 cfg: WindowPPOConfig | None = None):
        self.env = env
        self.cfg = cfg or WindowPPOConfig()
        assert policy.K == 1, "K > 1 is M3 work (brain coupling)"
        assert policy.act_dim == env.act_dim, \
            (policy.act_dim, env.act_dim)
        assert self.cfg.collect_len % self.cfg.window == 0, \
            (self.cfg.collect_len, self.cfg.window)
        self.policy = policy.to(env.device)
        self.value = (value or ValueMlp(env.obs_dim, env.act_dim)
                      ).to(env.device)
        self.norm = RunningNorm(env.obs_dim).to(env.device)
        groups = self.policy.param_groups(self.cfg.lr_core,
                                          self.cfg.lr_ports)
        groups.append({"params": list(self.value.parameters()),
                       "lr": self.cfg.lr_ports})
        self.optim = torch.optim.Adam(groups)
        self.body_steps = 0
        self.updates = 0
        self._obs = None            # current obs carried across collections
        self._state = None          # neural state carried across collections
        self._act_view = None       # live GPU view of env's d.act

    # ---------------------------------------------------------- rollout

    def _applied_torque(self) -> torch.Tensor:
        """Live (B, act_dim) GPU view of the torque currently in d.act."""
        if self._act_view is None:
            self._act_view = self.env._wp.to_torch(self.env.d.act)
        return self._act_view

    def collect(self):
        """Run T body steps; returns the rollout buffer dict and stats."""
        env, cfg, pol = self.env, self.cfg, self.policy
        T, B, Tw = cfg.collect_len, env.nworld, cfg.window
        dev = env.device
        od, ad = env.obs_dim, env.act_dim

        buf = {
            "obs": torch.zeros(T, B, od, device=dev),
            "u": torch.zeros(T, B, ad, device=dev),
            "logp": torch.zeros(T, B, device=dev),
            "rew": torch.zeros(T, B, device=dev),
            "val": torch.zeros(T, B, device=dev),
            "vnext": torch.zeros(T, B, device=dev),
            "term": torch.zeros(T, B, dtype=torch.bool, device=dev),
            "end": torch.zeros(T, B, dtype=torch.bool, device=dev),
            "time_out": torch.zeros(T, B, dtype=torch.bool, device=dev),
            "act_prev": torch.zeros(T, B, ad, device=dev),
            "win_state": [],
        }
        if self._obs is None:
            self._obs = env.reset()
        if self._state is None:
            self._state = pol.init_state(B)

        # GPU-side stat accumulators; read once after the loop
        rew_sum = torch.zeros((), device=dev)
        term_sums = defaultdict(lambda: torch.zeros((), device=dev))
        n_ep = torch.zeros((), device=dev)
        n_succ = torch.zeros((), device=dev)
        standing_sum = torch.zeros((), device=dev)
        max_consec = torch.zeros((), dtype=torch.long, device=dev)

        t0 = time.perf_counter()
        for t in range(T):
            if t % Tw == 0:
                # detached clone of the window start state: no gradient
                # may flow to before the window (§7)
                buf["win_state"].append(
                    tuple(x.detach().clone() for x in self._state))
            obs = self._obs
            self.norm.update(obs)
            nobs = self.norm.normalize(obs)
            act_prev = self._applied_torque()
            with torch.no_grad():
                self._state, out = pol.step(self._state, nobs)
                a = pol.applied_action(out["a"])
                v = self.value(nobs, act_prev)

            # the smoothness term sees the deterministic command tanh(mu)
            obs2, rew, done, info = env.step(a, command=out["cmd"][0])
            nobs2 = self.norm.normalize(obs2)
            with torch.no_grad():
                v2 = self.value(nobs2, self._applied_torque())
            time_out = info["time_out"]

            buf["obs"][t] = nobs        # normalized, as seen by the policy
            buf["u"][t] = out["u"][0]   # K=1: (B, J), pre-tanh sample
            buf["logp"][t] = out["logp"][0]
            # a non-finite world can emit a NaN reward; it is done anyway
            buf["rew"][t] = torch.nan_to_num(rew, nan=0.0,
                                             posinf=0.0, neginf=0.0)
            buf["val"][t] = v
            buf["vnext"][t] = v2
            buf["term"][t] = done & ~time_out
            buf["end"][t] = done
            buf["time_out"][t] = time_out
            buf["act_prev"][t] = act_prev

            rew_sum += rew.nan_to_num(nan=0.0, posinf=0.0,
                                    neginf=0.0).sum()
            for k, tv in info["terms"].items():
                term_sums[k] += tv.nan_to_num(nan=0.0).sum()
            n_ep += done.sum()
            n_succ += info["first_success"].sum()
            standing_sum += info["success_now"].sum()
            max_consec = torch.maximum(max_consec,
                                       env.tracker.consecutive.max())

            # the neural state resets exactly where the env did; the same
            # mask is replayed in the update via buf["end"]
            with torch.no_grad():
                self._state = pol.reset_state(self._state, done)

            # the only host sync per body step: does anyone need a reset?
            if bool(done.any()):
                self._obs = env.reset(done)
            else:
                self._obs = obs2

        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n = float(T * B)
        scal = torch.stack([rew_sum, n_ep, n_succ, standing_sum,
                            max_consec.to(torch.float32)]).cpu()
        stats = {
            "mean_reward": float(scal[0]) / n,
            "episodes": int(scal[1]),
            "successes": int(scal[2]),
            "success_rate": float(scal[2]) / max(float(scal[1]), 1.0),
            "standing_frac": float(scal[3]) / n,
            "max_consecutive": int(scal[4]),
            "collect_s": dt,
            "steps_per_s": n / max(dt, 1e-9),
            "terms": {k: float(v.cpu()) / n for k, v in term_sums.items()},
        }
        return buf, stats

    # ----------------------------------------------------------- update

    def _gae(self, buf):
        """Advantage and return targets from the (T, B) buffers."""
        T, B = buf["rew"].shape
        adv = torch.zeros(T, B, device=self.env.device)
        last = torch.zeros(B, device=self.env.device)
        for t in reversed(range(T)):
            not_term = (~buf["term"][t]).to(torch.float32)
            not_end = (~buf["end"][t]).to(torch.float32)
            delta = (buf["rew"][t]
                     + self.cfg.gamma * buf["vnext"][t] * not_term
                     - buf["val"][t])
            last = delta + self.cfg.gamma * self.cfg.lam * not_end * last
            adv[t] = last
        return adv, adv + buf["val"]

    def _fwd_mu(self, v, I, s, obs):
        """forward_mu unrolled to flat args for torch.utils.checkpoint."""
        return self.policy.forward_mu((v, I, s), obs)

    def _window_mus(self, buf, w: int):
        """Recompute window w's mu sequence from its stored start state.

        Time runs in order; the env resets recorded in buf["end"] are
        replayed at the same steps so the trajectory matches collection.
        """
        Tw = self.cfg.window
        state = tuple(x.clone() for x in buf["win_state"][w])
        mus = []
        for t in range(w * Tw, (w + 1) * Tw):
            if self.cfg.checkpoint:
                state, mu = _ckpt(self._fwd_mu, *state, buf["obs"][t],
                                  use_reentrant=False)
            else:
                state, mu = self.policy.forward_mu(state, buf["obs"][t])
            mus.append(mu)
            state = self.policy.reset_state(state, buf["end"][t])
        return mus

    def _update(self, buf):
        cfg = self.cfg
        Tw = cfg.window
        n_win = cfg.collect_len // Tw
        adv, ret = self._gae(buf)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pol_params = list(self.policy.parameters())
        val_params = list(self.value.parameters())
        core_params = [self.policy.core.raw_g, self.policy.core.b,
                       self.policy.core.raw_tau_m,
                       self.policy.core.raw_tau_s]
        port_params = [self.policy.core.in_gain, self.policy.core.out_w,
                       self.policy.log_std]
        n_units = cfg.epochs * n_win
        pg_l = v_l = ent = clip_f = core_gn = ports_gn = 0.0
        for _ in range(cfg.epochs):
            # only the window order may be shuffled, never the time axis
            for w in torch.randperm(n_win).tolist():
                self.optim.zero_grad(set_to_none=True)
                loss_w = buf["logp"].new_zeros(())
                # per-step diagnostics, averaged over the window (taking only
                # the last step's values would misreport the whole window)
                w_pg = w_vl = w_ent = w_clip = 0.0
                for i, mu in enumerate(self._window_mus(buf, w)):
                    t = w * Tw + i
                    logp = self.policy.log_prob(mu, buf["u"][t])[0]
                    ratio = (logp - buf["logp"][t]).exp()
                    pg = -torch.min(
                        ratio * adv[t],
                        ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip)
                        * adv[t]).mean()
                    v = self.value(buf["obs"][t], buf["act_prev"][t])
                    vl = 0.5 * (v - ret[t]).pow(2).mean()
                    e = self.policy.entropy(mu)[0].mean()
                    loss_w = loss_w + (pg + cfg.vf_coef * vl
                                       - cfg.entropy_coef * e) / Tw
                    w_pg += float(pg.detach()) / Tw
                    w_vl += float(vl.detach()) / Tw
                    w_ent += float(e.detach()) / Tw
                    w_clip += float(((ratio - 1.0).abs() > cfg.clip)
                                    .to(torch.float32).mean().detach()) / Tw
                loss_w.backward()
                core_gn += _grad_norm(core_params) / n_units
                ports_gn += _grad_norm(port_params) / n_units
                torch.nn.utils.clip_grad_norm_(pol_params,
                                               cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(val_params,
                                               cfg.max_grad_norm)
                self.optim.step()

                pg_l += w_pg / n_units
                v_l += w_vl / n_units
                ent += w_ent / n_units
                clip_f += w_clip / n_units
        return {"loss": pg_l + cfg.vf_coef * v_l - cfg.entropy_coef * ent,
                "pg_loss": pg_l, "v_loss": v_l, "entropy": ent,
                "clip_frac": clip_f, "core_grad_norm": core_gn,
                "ports_grad_norm": ports_gn}

    def update(self):
        """One collect + windowed PPO update; returns the log dict."""
        buf, stats = self.collect()
        t0 = time.perf_counter()
        stats.update(self._update(buf))
        if self.env.device.type == "cuda":
            torch.cuda.synchronize()
        stats["update_s"] = time.perf_counter() - t0
        self.body_steps += self.cfg.collect_len * self.env.nworld
        self.updates += 1
        stats["body_steps"] = self.body_steps
        stats["update"] = self.updates
        if self.env.device.type == "cuda":
            stats["gpu_mem_alloc_mb"] = (
                torch.cuda.max_memory_allocated(self.env.device) / 2 ** 20)
            stats["gpu_mem_reserved_mb"] = (
                torch.cuda.max_memory_reserved(self.env.device) / 2 ** 20)
        else:
            stats["gpu_mem_alloc_mb"] = stats["gpu_mem_reserved_mb"] = 0.0
        return stats

    # -------------------------------------------------------- checkpoint

    def save(self, path) -> None:
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "norm": self.norm.state_dict(),
            "norm_frozen": self.norm.frozen,
            "optimizer": self.optim.state_dict(),
            "body_steps": self.body_steps,
            "updates": self.updates,
            "obs_dim": self.env.obs_dim,
            "act_dim": self.env.act_dim,
            "cfg": dataclasses.asdict(self.cfg),
        }, path)

    @classmethod
    def load(cls, path, env, policy: BrainPolicy,
             value: ValueMlp | None = None,
             map_location=None) -> "WindowPPO":
        """Rebuild a WindowPPO bound to `env` from a save() checkpoint.

        `policy` must wrap the same fixed wiring (w0, ports) as the saved
        one — those tensors are plain attributes, not state-dict entries.
        """
        ck = torch.load(path, map_location=map_location or env.device)
        obj = cls(env, policy, value, cfg=WindowPPOConfig(**ck["cfg"]))
        obj.policy.load_state_dict(ck["policy"])
        obj.value.load_state_dict(ck["value"])
        obj.norm.load_state_dict(ck["norm"])
        obj.norm.frozen = bool(ck["norm_frozen"])
        obj.optim.load_state_dict(ck["optimizer"])
        obj.body_steps = int(ck["body_steps"])
        obj.updates = int(ck["updates"])
        return obj
