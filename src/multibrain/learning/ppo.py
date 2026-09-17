"""PPO learner for the `mlp_control` condition (PLAN.md §7).

One shared policy/value is trained from `nworld` parallel environments
(§10.1: the environments share the model, not the bodies). Per collection
of T = cfg.collect_len body steps the buffers hold (T, B) rows of obs,
pre-tanh action u, logp, reward, value, and the done flags split into
real terminations vs. episode time-outs. Time-out transitions bootstrap
through the value at the timed-out state (the env does not auto-reset, so
that value is read before the reset), while out-of-bounds and non-finite
terminations are treated as done.

GAE (gamma=0.99, lam=0.95) runs over the T axis; the recursion stops at
any episode end so advantages never leak into the next trial. Each update
shuffles the flattened (T*B) batch into cfg.n_minibatch minibatches for
cfg.epochs epochs; clip 0.2, Adam lr 3e-4, entropy coef 0, grad-norm 1.0
— the §7 initial values.

Observations pass through a RunningNorm (updated during collection only).
The buffer stores the *normalized* observation that the policy actually
saw, so the update evaluates the ratio on exactly the input that produced
logp_old (re-normalizing raw obs with the final statistics would shift it).
The value input is [normalized obs, last applied torque] where the torque
is the GPU view of env's d.act, i.e. the torque that produced the obs.

The gradient-norm limit (max_grad_norm) is applied to the policy and the
value network separately, so a large value loss cannot shrink the policy
update through a shared norm.

Per body step the rollout does at most one host sync, to decide whether
any world needs reset; all statistic accumulators stay on the GPU and are
read once per collection.
"""

import dataclasses
import math
import time
from collections import defaultdict
from dataclasses import dataclass

import torch

from .mlp_policy import GaussianMlpPolicy, ValueMlp
from .running_norm import RunningNorm


@dataclass
class PPOConfig:
    collect_len: int = 64        # T, body steps per collection (§7)
    epochs: int = 4
    clip: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    vf_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    n_minibatch: int = 4


class PPO:
    """PPO driver bound to a WarpBodyEnv.

    update() collects cfg.collect_len body steps and runs one PPO update,
    returning a flat dict of scalars for logging. save()/load() carry
    policy, value, norm, optimizer, step counters and the config.
    """

    def __init__(self, env, policy: GaussianMlpPolicy | None = None,
                 value: ValueMlp | None = None,
                 cfg: PPOConfig | None = None):
        self.env = env
        self.cfg = cfg or PPOConfig()
        dev = env.device
        self.policy = (policy or GaussianMlpPolicy(env.obs_dim, env.act_dim)
                       ).to(dev)
        self.value = (value or ValueMlp(env.obs_dim, env.act_dim)).to(dev)
        self.norm = RunningNorm(env.obs_dim).to(dev)
        self.optim = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=self.cfg.lr)
        self.body_steps = 0
        self.updates = 0
        self._obs = None            # current obs carried across collections
        self._act_view = None       # live GPU view of env's d.act

    # ---------------------------------------------------------- rollout

    def _applied_torque(self) -> torch.Tensor:
        """Live (B, act_dim) GPU view of the torque currently in d.act."""
        if self._act_view is None:
            self._act_view = self.env._wp.to_torch(self.env.d.act)
        return self._act_view

    def collect(self):
        """Run T body steps; returns the rollout buffer dict and stats."""
        env, cfg = self.env, self.cfg
        T, B = cfg.collect_len, env.nworld
        dev = env.device
        od, ad = env.obs_dim, env.act_dim

        buf = {
            "obs": torch.zeros(T, B, od, device=dev),
            "act": torch.zeros(T, B, ad, device=dev),
            "u": torch.zeros(T, B, ad, device=dev),
            "logp": torch.zeros(T, B, device=dev),
            "rew": torch.zeros(T, B, device=dev),
            "val": torch.zeros(T, B, device=dev),
            "vnext": torch.zeros(T, B, device=dev),
            "term": torch.zeros(T, B, dtype=torch.bool, device=dev),
            "end": torch.zeros(T, B, dtype=torch.bool, device=dev),
        }
        if self._obs is None:
            self._obs = env.reset()

        # GPU-side stat accumulators; read once after the loop
        rew_sum = torch.zeros((), device=dev)
        term_sums = defaultdict(lambda: torch.zeros((), device=dev))
        n_ep = torch.zeros((), device=dev)
        n_succ = torch.zeros((), device=dev)
        standing_sum = torch.zeros((), device=dev)
        max_consec = torch.zeros((), dtype=torch.long, device=dev)

        t0 = time.perf_counter()
        for t in range(T):
            obs = self._obs
            self.norm.update(obs)
            nobs = self.norm.normalize(obs)
            act_prev = self._applied_torque()
            with torch.no_grad():
                u, a, logp = self.policy.act(nobs)
                v = self.value(nobs, act_prev)

            obs2, rew, done, info = env.step(a)
            nobs2 = self.norm.normalize(obs2)
            with torch.no_grad():
                v2 = self.value(nobs2, self._applied_torque())
            time_out = info["time_out"]

            buf["obs"][t] = nobs          # normalized, as seen by the policy
            buf["act"][t] = act_prev
            buf["u"][t] = u
            buf["logp"][t] = logp
            # a non-finite world can emit a NaN reward; it is done anyway
            buf["rew"][t] = torch.nan_to_num(rew, nan=0.0,
                                             posinf=0.0, neginf=0.0)
            buf["val"][t] = v
            buf["vnext"][t] = v2
            buf["term"][t] = done & ~time_out
            buf["end"][t] = done

            rew_sum += rew.nan_to_num(nan=0.0, posinf=0.0,
                                    neginf=0.0).sum()
            for k, tv in info["terms"].items():
                term_sums[k] += tv.nan_to_num(nan=0.0).sum()
            n_ep += done.sum()
            n_succ += info["first_success"].sum()
            standing_sum += info["success_now"].sum()
            max_consec = torch.maximum(max_consec,
                                       env.tracker.consecutive.max())

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

    def _update(self, buf):
        cfg = self.cfg
        n = cfg.collect_len * self.env.nworld
        obs = buf["obs"].reshape(n, -1)
        act = buf["act"].reshape(n, -1)
        u = buf["u"].reshape(n, -1)
        logp_old = buf["logp"].reshape(n)
        adv, ret = self._gae(buf)
        adv = (adv.reshape(n) - adv.mean()) / (adv.std() + 1e-8)
        ret = ret.reshape(n)

        pg_l = v_l = ent = clip_f = 0.0
        pol_params = list(self.policy.parameters())
        val_params = list(self.value.parameters())
        for _ in range(cfg.epochs):
            for mb in torch.randperm(n, device=obs.device).chunk(
                    cfg.n_minibatch):
                nobs = obs[mb]                # already normalized
                logp = self.policy.log_prob(nobs, u[mb])
                ratio = (logp - logp_old[mb]).exp()
                pg = -torch.min(
                    ratio * adv[mb],
                    ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip)
                    * adv[mb]).mean()
                v = self.value(nobs, act[mb])
                vl = 0.5 * (v - ret[mb]).pow(2).mean()
                e = self.policy.entropy(nobs).mean()
                loss = pg + cfg.vf_coef * vl - cfg.entropy_coef * e

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pol_params, cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(val_params, cfg.max_grad_norm)
                self.optim.step()

                pg_l += float(pg.detach()) / (cfg.epochs * cfg.n_minibatch)
                v_l += float(vl.detach()) / (cfg.epochs * cfg.n_minibatch)
                ent += float(e.detach()) / (cfg.epochs * cfg.n_minibatch)
                clip_f += float(((ratio - 1.0).abs() > cfg.clip)
                                .to(torch.float32).mean().detach()) \
                    / (cfg.epochs * cfg.n_minibatch)
        return {"loss": pg_l + cfg.vf_coef * v_l - cfg.entropy_coef * ent,
                "pg_loss": pg_l, "v_loss": v_l, "entropy": ent,
                "clip_frac": clip_f}

    def update(self):
        """One collect + PPO update; returns the per-update log dict."""
        buf, stats = self.collect()
        stats.update(self._update(buf))
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
    def load(cls, path, env, map_location=None) -> "PPO":
        """Rebuild a PPO bound to `env` from a save() checkpoint."""
        ck = torch.load(path, map_location=map_location or env.device)
        ppo = cls(env, cfg=PPOConfig(**ck["cfg"]))
        ppo.policy.load_state_dict(ck["policy"])
        ppo.value.load_state_dict(ck["value"])
        ppo.norm.load_state_dict(ck["norm"])
        ppo.norm.frozen = bool(ck["norm_frozen"])
        ppo.optim.load_state_dict(ck["optimizer"])
        ppo.body_steps = int(ck["body_steps"])
        ppo.updates = int(ck["updates"])
        return ppo
