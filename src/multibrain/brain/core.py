"""PLAN §4.2 の神経中核（連続値版）と、§4.3 の入力ポート、§4.4 の読出し。

K 個の脳が同じ固定配線 W0 を共有し、学習パラメータと状態を別に持つ。
状態は (N, K·B) の列並びで持ち、疎行列積 W0 @ S を全脳まとめて一回で行う。
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor, nn

BETA = 0.25          # 出力 σ((v − 1) / β) の温度
V_CLAMP = 5.0
N_BASIS = 8          # スカラー信号あたりのガウス型応答曲線の数
BASIS_WIDTH = 0.35
PARAM_RANGE = {"g": (0.1, 10.0), "b": (-2.0, 2.0), "tau_m": (0.010, 0.200), "tau_s": (0.005, 0.050)}
PARAM_INIT = {"g": 1.0, "b": 0.0, "tau_m": 0.020, "tau_s": 0.005}


def csr_to_torch(w: sp.csr_matrix, device) -> Tensor:
    return torch.sparse_csr_tensor(torch.from_numpy(w.indptr).to(torch.int32), torch.from_numpy(w.indices).to(torch.int32),
                                   torch.from_numpy(w.data).float(), size=w.shape, device=device)


@dataclass
class Ports:
    """ports.json を添字テンソルに変換したもの。bodyId → 行番号の変換は body_ids で行う。"""
    signals: list[str]
    joints: list[str]
    in_neuron: Tensor      # (P,) ポートニューロンの行番号
    in_signal: Tensor      # (P,) 観測ベクトル内の信号番号
    in_center: Tensor      # (P,) 応答曲線の中心
    out_row: Tensor        # (Q,) 関節番号
    out_neuron: Tensor     # (Q,) 運動ニューロンの行番号
    out_init: Tensor       # (Q,) 初期重み

    @classmethod
    def load(cls, path: Path, body_ids: np.ndarray, seed: int = 0) -> "Ports":
        d = json.loads(Path(path).read_text())
        lookup = {int(b): i for i, b in enumerate(body_ids.tolist())}
        rng = np.random.default_rng(seed)
        centers = np.linspace(-1.0, 1.0, N_BASIS)
        signals = [p["signal"] for p in d["inputs"]]
        in_neuron, in_signal, in_center = [], [], []
        for k, p in enumerate(d["inputs"]):
            for b in p["neurons"]:
                in_neuron.append(lookup[b]); in_signal.append(k); in_center.append(centers[rng.integers(N_BASIS)])
        joints = d["joints"]
        jidx = {j: i for i, j in enumerate(joints)}
        out_row, out_neuron, out_init = [], [], []
        for p in d["outputs"]:
            for b, w in zip(p["neurons"], p["init_weight"]):
                out_row.append(jidx[p["joint"]]); out_neuron.append(lookup[b]); out_init.append(w)
        t = lambda x, dt=torch.long: torch.tensor(x, dtype=dt)
        return cls(signals, joints, t(in_neuron), t(in_signal), t(in_center, torch.float32),
                   t(out_row), t(out_neuron), t(out_init, torch.float32))

    def to(self, device) -> "Ports":
        return Ports(self.signals, self.joints, *(x.to(device) for x in
                     (self.in_neuron, self.in_signal, self.in_center, self.out_row, self.out_neuron, self.out_init)))


def _logit_param(value: float, lo: float, hi: float) -> float:
    """[lo, hi] を対数空間で表した無制約パラメータの値。端の値は 0.1% だけ内側に寄せる。"""
    frac = min(max((math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo)), 1e-3), 1 - 1e-3)
    return math.log(frac / (1 - frac))


class BrainCore(nn.Module):
    """K 脳ぶんの神経中核。forward は 1 身体ステップ（連続値版では 1 神経更新）。"""

    def __init__(self, w0: sp.csr_matrix, ports: Ports, n_brains: int, dt: float = 0.020, device="cuda"):
        super().__init__()
        self.N, self.K, self.dt = w0.shape[0], n_brains, dt
        self.w0 = csr_to_torch(w0, device)
        self.ports = ports.to(device)
        K, N = n_brains, self.N
        self.raw_g = nn.Parameter(torch.full((K, N), _logit_param(PARAM_INIT["g"], *PARAM_RANGE["g"]), device=device))
        self.b = nn.Parameter(torch.zeros(K, N, device=device))
        self.raw_tau_m = nn.Parameter(torch.full((K, N), _logit_param(PARAM_INIT["tau_m"], *PARAM_RANGE["tau_m"]), device=device))
        self.raw_tau_s = nn.Parameter(torch.full((K, N), _logit_param(PARAM_INIT["tau_s"], *PARAM_RANGE["tau_s"]), device=device))
        self.in_gain = nn.Parameter(torch.ones(K, len(ports.in_neuron), device=device))       # a_i ≥ 0（絶対値で使う）
        self.out_w = nn.Parameter(ports.out_init.to(device).unsqueeze(0).repeat(K, 1))       # w_ji

    @staticmethod
    def _bounded(raw: Tensor, lo: float, hi: float) -> Tensor:
        return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(raw))

    def params(self) -> dict[str, Tensor]:
        return {"g": self._bounded(self.raw_g, *PARAM_RANGE["g"]),
                "b": self.b.clamp(*PARAM_RANGE["b"]),
                "tau_m": self._bounded(self.raw_tau_m, *PARAM_RANGE["tau_m"]),
                "tau_s": self._bounded(self.raw_tau_s, *PARAM_RANGE["tau_s"])}

    def n_learnable(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_state(self, n_envs: int) -> tuple[Tensor, Tensor, Tensor]:
        z = torch.zeros(self.N, self.K * n_envs, device=self.w0.device)
        return z, z.clone(), torch.sigmoid((z - 1.0) / BETA)

    def encode(self, obs: Tensor) -> Tensor:
        """obs: (K, B, S) → 感覚電流 u: (N, K·B)。"""
        K, B, _ = obs.shape
        x = obs[:, :, self.ports.in_signal]                                    # (K, B, P)
        phi = torch.exp(-0.5 * ((x - self.ports.in_center) / BASIS_WIDTH) ** 2)
        phi = phi * self.in_gain.abs().unsqueeze(1)                              # (K, B, P)
        u = torch.zeros(K, B, self.N, device=obs.device)
        u.index_add_(2, self.ports.in_neuron, phi)
        return u.permute(2, 0, 1).reshape(self.N, K * B)

    def readout_pre(self, s: Tensor) -> Tensor:
        """s: (N, K·B) → tanh を通す前の読出し値 (K, B, J)。"""
        K = self.K
        r = s.view(self.N, K, -1)[self.ports.out_neuron]                         # (Q, K, B)
        contrib = r * self.out_w.t().unsqueeze(2)                                # (Q, K, B)
        out = torch.zeros(len(self.ports.joints), K, r.shape[2], device=s.device)
        out.index_add_(0, self.ports.out_row, contrib)
        return out.permute(1, 2, 0)

    def readout(self, s: Tensor) -> Tensor:
        """s: (N, K·B) → 正規化トルク a: (K, B, J)。"""
        return torch.tanh(self.readout_pre(s))

    def forward(self, state, obs: Tensor, m: Tensor | None = None, pre_tanh: bool = False):
        """1 更新。state = (v, I, s)、obs: (K, B, S)、m: 合成シナプス電流 (N, K·B) または None。
        pre_tanh=True のとき第 2 返り値は tanh 前の読出し値（§4.4 のガウス分布の平均）。"""
        v, I, s = state
        p = self.params()
        K, B = obs.shape[:2]
        col = lambda x: x.t().unsqueeze(2)                                       # (K, N) → (N, K, 1) で放送
        alpha, gamma, g, b = (col(torch.exp(-self.dt / p["tau_m"])), col(torch.exp(-self.dt / p["tau_s"])),
                              col(p["g"]), col(p["b"]))
        drive = g * torch.sparse.mm(self.w0, s).view(self.N, K, B) + self.encode(obs).view(self.N, K, B)
        if m is not None:
            drive = drive + m.view(self.N, K, B)
        I = gamma * I.view(self.N, K, B) + (1 - gamma) * drive
        v = (alpha * v.view(self.N, K, B) + (1 - alpha) * (I + b)).clamp(-V_CLAMP, V_CLAMP)
        s = torch.sigmoid((v - 1.0) / BETA)
        flat = lambda x: x.reshape(self.N, K * B)
        s = flat(s)
        out = self.readout_pre(s) if pre_tanh else self.readout(s)
        return (flat(v), flat(I), s), out


def spectral_radius(w0: Tensor, iters: int = 100) -> float:
    """べき乗法で |λ_max|(W0) を近似する。"""
    x = torch.randn(w0.shape[0], 1, device=w0.device)
    x /= x.norm()
    lam = 0.0
    for _ in range(iters):
        y = torch.sparse.mm(w0, x)
        lam = y.norm().item()
        x = y / max(lam, 1e-12)
    return lam
