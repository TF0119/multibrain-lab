"""神経中核（BrainCore）を PPO の方策として包む（PLAN §4.2〜§4.4）。

GaussianMlpPolicy と同じ形にそろえる: tanh を通す前の値 u にガウス分布を置き、
a = tanh(u) を身体へ渡す。読出しの重み和（tanh 前）が分布の平均 mu で、
標準偏差は脳・関節ごとに学習する（§4.4）。内部状態 (v, I, s) はモジュール内に
隠さず (N, K·B) の列のまま持ち回る。環境 b、脳 k の列は k·B + b（BrainCore の
並びと同じ）。学習ループ側が窓の開始で detach や保存を行う。
"""

import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from multibrain.brain.core import BETA, BrainCore

OBS_DIM = 76
LOG_STD0 = math.log(0.3)      # §4.4
LOG_STD_MIN = math.log(0.1)   # mlp_policy.py と同じ探索の下限（§7）


class BrainPolicy(nn.Module):
    """K 脳ぶんの神経中核を方策として使う（§4.2〜§4.4）。"""

    def __init__(self, core: BrainCore, n_envs: int, log_std0: float = LOG_STD0):
        super().__init__()
        self.core = core
        self.obs_dim = OBS_DIM
        self.act_dim = len(core.ports.joints)
        self.K = core.K
        self.B = n_envs
        self.log_std = nn.Parameter(
            torch.full((self.K, self.act_dim), float(log_std0), device=core.w0.device))

    def _expand_obs(self, obs: Tensor) -> Tensor:
        """(B, 76) → (K, B, 76)。全脳に同じ全身の観測を渡す（§2）。expand なので複製しない。

        B は obs から取る。self.B は既定の環境数にすぎず、評価（24 開始）のように
        別の環境数で同じ方策を使うことがあるため、ここで固定しない。
        """
        return obs.unsqueeze(0).expand(self.K, obs.shape[0], self.obs_dim)

    def init_state(self, n_envs: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """環境ぶんの初期状態 (v, I, s)。v=0、I=0、s=σ(-1/β)。既定は self.B。"""
        return self.core.init_state(self.B if n_envs is None else n_envs)

    def reset_state(self, state, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """mask (B,) が立った環境の列だけを初期値に戻す（v=0、I=0、s=σ(-1/β)）。

        in-place 代入は使わず torch.where で新しいテンソルを返す。
        他の環境の列は値も勾配の履歴も変わらない。
        """
        v, I, s = state
        # 環境数は状態の列数から取る（評価では self.B と異なることがある）
        assert v.shape[1] == self.K * mask.shape[0], (v.shape, mask.shape)
        col = mask.repeat(self.K).unsqueeze(0)                  # (1, K·B): 列 k·B+b ← 環境 b
        s0 = 1.0 / (1.0 + math.exp(1.0 / BETA))
        return (torch.where(col, 0.0, v),
                torch.where(col, 0.0, I),
                torch.where(col, s0, s))

    def dist(self, mu: Tensor) -> Normal:
        """mu: (K, B, J) → tanh 前の行動 u のガウス分布。std は log_std を下限で切り詰めて exp。"""
        std = self.log_std.clamp(min=LOG_STD_MIN).exp().unsqueeze(1)     # (K, 1, J)
        return Normal(mu, std.expand_as(mu))

    def step(self, state, obs: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor], dict]:
        """obs: (B, 76)（全脳に同じ全身の観測、§2）。1 身体ステップ進める。

        out = {"u": (K,B,J) tanh 前の標本, "a": tanh(u), "logp": (K,B) u の対数確率,
               "mu": (K,B,J) tanh 前の平均, "cmd": tanh(mu), "s": (N, K·B) 活動}
        """
        state, mu = self.core(state, self._expand_obs(obs), pre_tanh=True)
        d = self.dist(mu)
        u = d.rsample()     # reparametrize: (a**2).mean() の勾配が mu と log_std まで届くように
        out = {"u": u, "a": torch.tanh(u), "logp": d.log_prob(u).sum(-1),
               "mu": mu, "cmd": torch.tanh(mu), "s": state[2]}
        return state, out

    def forward_mu(self, state, obs: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor]:
        """1 身体ステップ進めて (状態, mu) を返す。標本化しない。

        切り詰め窓の再計算で使う。再計算では保存した u の対数確率を評価するので、
        標本を引き直す必要がなく、引くと乱数を無駄に消費する。
        """
        return self.core(state, self._expand_obs(obs), pre_tanh=True)

    def act_deterministic(self, state, obs: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor]:
        """評価用の決定的行動 a = tanh(mu)（探索ノイズなし）。"""
        state, mu = self.core(state, self._expand_obs(obs), pre_tanh=True)
        return state, torch.tanh(mu)

    def log_prob(self, mu: Tensor, u: Tensor) -> Tensor:
        """(K,B): tanh 前の行動 u の対数確率。"""
        return self.dist(mu).log_prob(u).sum(-1)

    def entropy(self, mu: Tensor) -> Tensor:
        """(K,B): tanh 前の行動分布のエントロピー。"""
        return self.dist(mu).entropy().sum(-1)

    def applied_action(self, a: Tensor) -> Tensor:
        """(K,B,J) の各脳の行動を身体に加える (B,J) にまとめる。

        mono（K=1）は a[0]。K>1 は平均（tri_symmetric 相当）。
        自由度の分担（tri_body）と合成シナプス結合は M3 で実装する。
        """
        return a[0] if self.K == 1 else a.mean(0)

    def param_groups(self, lr_core: float = 1e-4, lr_ports: float = 3e-4) -> list[dict]:
        """§7 の学習率。core = {raw_g, b, raw_tau_m, raw_tau_s}、
        ports = {in_gain, out_w, log_std}。全パラメータを漏れなく含むことを assert する。"""
        groups = [{"params": [self.core.raw_g, self.core.b,
                              self.core.raw_tau_m, self.core.raw_tau_s],
                   "lr": lr_core},
                  {"params": [self.core.in_gain, self.core.out_w, self.log_std],
                   "lr": lr_ports}]
        grouped = [p for g in groups for p in g["params"]]
        all_params = list(self.parameters())
        assert len(grouped) == len(all_params) \
            and {id(p) for p in grouped} == {id(p) for p in all_params}, \
            "param_groups が全パラメータをちょうど 1 回ずつ含んでいない"
        return groups
