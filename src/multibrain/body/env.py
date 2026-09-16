"""Batched MuJoCo Warp environment for the tasks of PLAN.md §5.1.

One mujoco_warp Data of `nworld` worlds advances 4 physics steps (4 x 5 ms
= one 20 ms body step, §3.6) per call to step(); those 4 steps are captured
into a single CUDA graph and replayed with wp.capture_launch (§10.2). All
state I/O goes through torch views on the warp arrays (wp.to_torch /
wp.from_torch for ctrl) so nothing crosses the host in the step path.

Observation (76), reward (9 terms) and success tracking are wired to
observation.py, reward.py and success.py; termination is episode length
(episode_s / body_step_s), pelvis xy beyond termination.max_pelvis_xy_m,
and non-finite state when termination.on_nonfinite. The contact/constraint
overflow bitmask (d.overflow) is reported in info every step (§10.2).

The env never resets itself: the caller must call reset(done) on the
finished worlds so TimeLimit bootstrapping stays correct.

All torch work that touches warp state is enqueued on an ExternalStream
wrapping warp's stream (same convention as scripts/check_body.py), so
ctrl writes, graph launches and the obs/reward/success kernels stay
ordered without host synchronization. Each call ends with the caller's
stream waiting on that stream; step() itself never syncs the host.
"""

import contextlib
import warnings
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from .layout import BodyLayout
from .observation import OBS_DIM, observe
from .reward import Reward, RewardConfig
from .start_poses import start_qpos
from .success import SuccessConfig, SuccessTracker, standing_now

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XML = REPO_ROOT / "assets" / "humanoid.xml"
DEFAULT_TASK_CFG = REPO_ROOT / "configs" / "task.yaml"
DEFAULT_REWARD_CFG = REPO_ROOT / "configs" / "reward.yaml"

WARMUP_BODY_STEPS = 8     # warmup before the graph capture, like check_body


class WarpBodyEnv:
    obs_dim = OBS_DIM     # 76
    act_dim = 27

    def __init__(self, nworld: int, task: str = "rise_and_stand",
                 device="cuda", xml_path=DEFAULT_XML,
                 task_cfg=DEFAULT_TASK_CFG, reward_cfg=DEFAULT_REWARD_CFG,
                 seed: int = 0, streamer=None, njmax: int = 512):
        import mujoco_warp
        import warp as wp

        self._wp = wp
        self._mjw = mujoco_warp
        self.device = torch.device(device)
        self._cuda = self.device.type == "cuda"
        self.nworld = int(nworld)
        self.task = task
        self.rng = np.random.default_rng(seed)

        with open(task_cfg) as f:
            task_y = yaml.safe_load(f)
        self.body_step_s = float(task_y["body_step_s"])
        self.substeps = int(task_y["physics_substeps"])
        spec = task_y["tasks"][task]
        self.episode_steps = round(
            float(spec["episode_s"]) / self.body_step_s)
        self._starts = list(spec["starts"])
        term = task_y["termination"]
        self._max_pelvis_xy = float(term["max_pelvis_xy_m"])
        self._on_nonfinite = bool(term["on_nonfinite"])

        self.mjm = mujoco.MjModel.from_xml_path(str(xml_path))
        mjd0 = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, mjd0)

        wp.init()
        self.m = mujoco_warp.put_model(self.mjm)
        # njmax is the per-world constraint row budget: the default guess
        # (64) overflows as soon as bodies lie flat — 27 joint friction/
        # limit rows plus ~6 rows per contact need more. M0 measured at
        # most 24 contacts per world, so 512 is a comfortable margin.
        self.d = mujoco_warp.put_data(self.mjm, mjd0, nworld=self.nworld,
                                      njmax=njmax)

        # _layout_host keeps numpy arrays for start_qpos; layout is the
        # device copy used by observe / reward / success
        self._layout_host = BodyLayout.from_model(self.mjm)
        self.layout = self._layout_host.to(self.device)
        assert int(self.layout.nu) == self.act_dim
        assert int(self.layout.nq) == int(self.mjm.nq)

        # torch views into the warp state (zero-copy); ctrl is re-aliased
        # to a tensor we own so policy output lands straight in the sim
        self._qpos = wp.to_torch(self.d.qpos)
        self._qvel = wp.to_torch(self.d.qvel)
        self._act = wp.to_torch(self.d.act)
        self._time = wp.to_torch(self.d.time)
        self._sensordata = wp.to_torch(self.d.sensordata)
        self._overflow = wp.to_torch(self.d.overflow)
        self.ctrl = torch.zeros((self.nworld, self.act_dim),
                                device=self.device)
        self.d.ctrl = wp.from_torch(self.ctrl)

        self.reward_cfg = RewardConfig.from_yaml(reward_cfg)
        self.success_cfg = SuccessConfig.from_yaml(task_cfg, reward_cfg)
        self.reward = Reward(self.layout, self.reward_cfg, self.nworld,
                             self.device)
        self.tracker = SuccessTracker(self.success_cfg, self.nworld,
                                      self.device)
        self.episode_step = torch.zeros(self.nworld, dtype=torch.long,
                                        device=self.device)

        self._streamer = streamer
        self._pose_src = None
        if streamer is not None and self._cuda:
            from multibrain.monitor import WarpPoseSource
            self._pose_src = WarpPoseSource(self.d, int(self.mjm.nq), env=0)
        self._body_steps = 0      # host-side clock for stream timestamps

        # torch ops that touch warp state run on warp's stream
        self._tstream = (torch.cuda.ExternalStream(
            wp.get_stream().cuda_stream) if self._cuda else None)

        self._graph = None
        if self._cuda:
            with torch.cuda.stream(self._tstream):
                for _ in range(WARMUP_BODY_STEPS * self.substeps):
                    mujoco_warp.step(self.m, self.d)
                wp.synchronize()
                try:
                    with wp.ScopedCapture() as capture:
                        for _ in range(self.substeps):
                            mujoco_warp.step(self.m, self.d)
                    self._graph = capture.graph
                except Exception as e:
                    warnings.warn(
                        "CUDA graph capture failed; stepping without a "
                        f"graph: {type(e).__name__}: {e}")

    def _warp_stream(self):
        """Context running torch ops on warp's stream (no-op on CPU)."""
        if not self._cuda:
            return contextlib.nullcontext()
        # order caller-side producers (e.g. the policy's action) first
        self._tstream.wait_stream(torch.cuda.current_stream())
        return torch.cuda.stream(self._tstream)

    def _release_stream(self):
        """Make the caller's stream wait on the warp stream."""
        if self._cuda:
            torch.cuda.current_stream().wait_stream(self._tstream)

    @property
    def qpos(self):
        """(nworld, nq) torch view of the warp qpos."""
        return self._qpos

    def sensordata(self) -> torch.Tensor:
        """(nworld, nsensordata) GPU view of d.sensordata."""
        return self._sensordata

    def reset(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Re-initialize the masked worlds (all of them when mask is None)
        to a fresh start pose and return (nworld, obs_dim) observations.

        Writes go straight into the warp arrays via the torch views; a
        single mujoco_warp.forward then refreshes sensordata. Start poses
        are drawn uniformly from the task's `starts` list.
        """
        if mask is None:
            mask = torch.ones(self.nworld, dtype=torch.bool,
                              device=self.device)
        else:
            mask = mask.to(device=self.device, dtype=torch.bool)

        with self._warp_stream():
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                kinds = self.rng.integers(0, len(self._starts),
                                          size=idx.numel())
                qpos = np.stack([
                    start_qpos(self._layout_host, self._starts[int(k)],
                               self.rng)
                    for k in kinds])
                qpos_t = torch.as_tensor(qpos, dtype=torch.float32,
                                         device=self.device)
                self._qpos[idx] = qpos_t
                self._qvel[idx] = 0.0
                self._act[idx] = 0.0
                self.ctrl[idx] = 0.0
                self._time[idx] = 0.0
                self._overflow[idx] = 0   # sticky bitmask; clear per trial
                self._mjw.forward(self.m, self.d)
            self.reward.reset(mask, self._sensordata)
            self.tracker.reset(mask)
            self.episode_step = torch.where(
                mask, torch.zeros_like(self.episode_step),
                self.episode_step)
            obs = observe(self.layout, self._sensordata)
        self._release_stream()
        return obs

    def step(self, action: torch.Tensor):
        """One 20 ms body step -> (obs, reward, done, info).

        `action` (nworld, act_dim) is clamped to [-1, 1] into ctrl. done =
        episode_steps reached | pelvis xy out of bounds | non-finite state
        (when termination.on_nonfinite). Worlds marked done keep stepping
        until the caller resets them — PPO consumes info["time_out"] for
        bootstrapping. Nothing here syncs the host.
        """
        with self._warp_stream():
            self.ctrl.copy_(action).clamp_(-1.0, 1.0)
            if self._graph is not None:
                self._wp.capture_launch(self._graph)
            else:
                for _ in range(self.substeps):
                    self._mjw.step(self.m, self.d)

            s = self._sensordata
            obs = observe(self.layout, s)
            standing = standing_now(self.layout, self.success_cfg, s)
            tr = self.tracker.update(standing)
            total, terms = self.reward.step(
                s, self._act, self.ctrl, standing, tr["first_success"])

            self.episode_step += 1
            self._body_steps += 1
            time_out = self.episode_step >= self.episode_steps
            out_of_bounds = (self._qpos[:, :2].norm(dim=-1)
                             >= self._max_pelvis_xy)
            nonfinite = ~(torch.isfinite(self._qpos).all(dim=-1)
                          & torch.isfinite(self._qvel).all(dim=-1))
            done = time_out | out_of_bounds
            if self._on_nonfinite:
                done = done | nonfinite
            info = {
                "terms": terms,
                "success_now": tr["success_now"],
                "first_success": tr["first_success"],
                "milestone": tr["milestone"],
                "nonfinite": nonfinite,
                "time_out": time_out,
                "out_of_bounds": out_of_bounds,
                "overflow": self._overflow != 0,
            }

            # env-0 pose for the viewer: non-blocking staged copy, the
            # previously completed frame is what actually gets submitted
            if self._pose_src is not None:
                self._pose_src.enqueue()
                q = self._pose_src.read()
                if q is not None:
                    self._streamer.submit(
                        self._body_steps * self.body_step_s, q)
        self._release_stream()
        return obs, total, done, info

    def close(self):
        if self._cuda:
            self._wp.synchronize()
