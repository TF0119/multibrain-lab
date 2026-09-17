"""graph.npz と ports.json から全規模の BrainCore を組み立てる。

bench_core.py と stream_body.py の setup_activity にある手順を一箇所にまとめたもの。
重みのスケール c は、動作点で線形化した結合行列のスペクトル半径が target_rho
になるように決める（PLAN §4.1）。
"""

from pathlib import Path

import torch

from multibrain.brain.core import BETA, BrainCore, Ports, csr_to_torch, spectral_radius
from multibrain.data.malecns import Graph

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_RHO = 0.9      # bench_core.py と同じ


def build_core(n_brains: int, data_dir=REPO_ROOT / "data", device="cuda",
               target_rho: float = TARGET_RHO, seed: int = 0) -> tuple[BrainCore, dict]:
    """graph.npz と ports.json から全規模の BrainCore を組み立てる。

    c は bench_core.py と同じ規則で決める:
      rho = spectral_radius(csr_to_torch(graph.base_weights(1.0), device))
      c = target_rho / (rho / (4 * BETA))
    返り値の dict は {"N", "nnz", "c", "rho_unit", "n_learnable", "body_ids"}。
    data_dir に graph.npz か ports.json が無ければ FileNotFoundError を投げる。
    """
    data_dir = Path(data_dir)
    missing = [f for f in ("graph.npz", "ports.json") if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"{data_dir} に必要なファイルが無い: {missing}")
    dev = torch.device(device)
    graph = Graph.load(data_dir / "graph.npz")
    ports = Ports.load(data_dir / "ports.json", graph.body_ids, seed)
    rho = spectral_radius(csr_to_torch(graph.base_weights(1.0), dev))
    c = target_rho / (rho / (4 * BETA))
    w0 = graph.base_weights(c)
    core = BrainCore(w0, ports, n_brains=n_brains, device=dev)
    info = {"N": core.N, "nnz": int(w0.nnz), "c": c, "rho_unit": rho,
            "n_learnable": core.n_learnable(), "body_ids": graph.body_ids}
    return core, info
