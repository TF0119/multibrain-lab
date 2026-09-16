"""MLP policy and value for the `mlp_control` condition (PLAN.md §7).

GaussianMlpPolicy: obs (76) -> two hidden layers -> mu (27), the pre-tanh
mean; the applied command is a = tanh(u) with u ~ N(mu, std). log_std is
a per-joint parameter initialized at log(0.3). PPO optimizes the
distribution over u (pre-squash), so log_prob is evaluated on u.

ValueMlp: (obs, act) -> scalar. Input is 103 = 76 obs + 27 of the last
applied torque (env d.act), matching the §7 value-function spec. It is
used during learning only and takes no part in the deployed controller.
"""

import math

import torch
from torch import nn
from torch.distributions import Normal

OBS_DIM = 76
ACT_DIM = 27
HIDDEN = (256, 256)
LOG_STD0 = math.log(0.3)


def _mlp(in_dim: int, hidden, out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def _ortho(net: nn.Sequential, out_gain: float) -> None:
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight, gain=out_gain)


class GaussianMlpPolicy(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden=HIDDEN, log_std0: float = LOG_STD0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.net = _mlp(self.obs_dim, tuple(hidden), self.act_dim)
        self.log_std = nn.Parameter(
            torch.full((self.act_dim,), float(log_std0)))
        _ortho(self.net, out_gain=0.01)

    def dist(self, obs: torch.Tensor) -> Normal:
        mu = self.net(obs)
        return Normal(mu, self.log_std.exp().expand_as(mu))

    def act(self, obs: torch.Tensor):
        """Sample -> (u, a = tanh(u), logp(u)). logp is over u, pre-tanh."""
        d = self.dist(obs)
        u = d.sample()
        return u, torch.tanh(u), d.log_prob(u).sum(-1)

    def log_prob(self, obs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """(B,) log probability of pre-tanh action u under the policy."""
        return self.dist(obs).log_prob(u).sum(-1)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        return self.dist(obs).entropy().sum(-1)

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluation action: a = tanh(mu), no exploration noise."""
        return torch.tanh(self.net(obs))


class ValueMlp(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden=HIDDEN):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.net = _mlp(self.obs_dim + self.act_dim, tuple(hidden), 1)
        _ortho(self.net, out_gain=1.0)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """(B, obs_dim), (B, act_dim) -> (B,) value."""
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)
