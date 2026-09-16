"""神経中核の全規模の順伝播と逆伝播を実測し、data/benchmark_core.json に書く（M0）。

使い方: python scripts/bench_core.py [--envs 64] [--window 16] [--brains 1 10]
物理計算は含まない。観測は乱数、損失は読出しの二乗和で、勾配が中核のパラメータに届くことも確かめる。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from multibrain.brain.core import BETA, BrainCore, Ports, csr_to_torch, spectral_radius
from multibrain.data.malecns import Graph

TARGET_RHO = 0.9
N_OBS = 76


def timed(fn, repeat: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat


def bench(core: BrainCore, n_envs: int, window: int, use_checkpoint: bool) -> dict:
    K, dev = core.K, core.w0.device
    obs = torch.rand(K, n_envs, N_OBS, device=dev) * 2 - 1
    state = core.init_state(n_envs)

    def step_no_grad():
        nonlocal state
        with torch.no_grad():
            state, _ = core(state, obs)

    def one_step(v, I, s):
        (v, I, s), a = core((v, I, s), obs)
        return v, I, s, a

    def window_fwd_bwd():
        st = tuple(x.detach() for x in core.init_state(n_envs))
        loss = 0.0
        for _ in range(window):
            if use_checkpoint:
                *st, a = checkpoint(one_step, *st, use_reentrant=False)
            else:
                *st, a = one_step(*st)
            loss = loss + (a ** 2).mean()
        core.zero_grad(set_to_none=True)
        loss.backward()

    for _ in range(3):
        step_no_grad()
    t_fwd = timed(step_no_grad, 20)
    torch.cuda.reset_peak_memory_stats()
    window_fwd_bwd()
    t_win = timed(window_fwd_bwd, 3)
    peak = torch.cuda.max_memory_allocated() / 1e9
    grads = {n: float(p.grad.norm()) for n, p in core.named_parameters() if p.grad is not None}
    with torch.no_grad():
        st = core.init_state(n_envs)
        for _ in range(50):
            st, a = core(st, obs)
        v, _, s = st
    return {
        "brains": K, "envs_per_brain": n_envs, "columns": K * n_envs, "window": window, "checkpoint": use_checkpoint,
        "forward_ms_per_step": t_fwd * 1e3,
        "window_fwd_bwd_ms": t_win * 1e3,
        "fwd_bwd_ms_per_body_step": t_win * 1e3 / window,
        "body_steps_per_s_fwd_only": K * n_envs / t_fwd,
        "body_steps_per_s_fwd_bwd": K * n_envs * window / t_win,
        "peak_gpu_GB": peak,
        "grad_norms": grads,
        "activity_after_50_steps": {"s_mean": float(s.mean()), "s_saturated_frac": float((s > 0.95).float().mean()),
                                    "v_mean": float(v.mean()), "v_abs_max": float(v.abs().max())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--brains", type=int, nargs="+", default=[1, 10])
    args = ap.parse_args()
    data = Path(args.data)
    dev = torch.device("cuda")

    graph = Graph.load(data / "graph.npz")
    ports = Ports.load(data / "ports.json", graph.body_ids)
    w_unit = graph.base_weights(1.0)
    rho_unit = spectral_radius(csr_to_torch(w_unit, dev))
    sigma_prime_max = 1.0 / (4 * BETA)
    c = TARGET_RHO / (rho_unit * sigma_prime_max)
    w0 = graph.base_weights(c)
    report = {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "N": int(w0.shape[0]), "nnz": int(w0.nnz),
        "weights": {"rho_unit": rho_unit, "sigma_prime_max": sigma_prime_max, "c": c, "target_rho_linearized": TARGET_RHO,
                    "w0_abs_mean": float(np.abs(w0.data).mean()), "w0_abs_max": float(np.abs(w0.data).max())},
        "ports": {"input_neurons": int(len(ports.in_neuron)), "output_entries": int(len(ports.out_neuron))},
        "runs": [],
    }
    for K, ckpt in [(K, c) for K in args.brains for c in ((False, True) if K == 1 else (True,))]:
        core = BrainCore(w0, ports, n_brains=K, device=dev)
        report["n_learnable_per_brain"] = core.n_learnable() // K
        try:
            r = bench(core, args.envs, args.window, ckpt)
        except torch.OutOfMemoryError as e:
            r = {"brains": K, "envs_per_brain": args.envs, "checkpoint": ckpt, "error": "CUDA OOM", "detail": str(e)[:200]}
        report["runs"].append(r)
        print(json.dumps(r, indent=2))
        del core
        torch.cuda.empty_cache()

    out = data / "benchmark_core.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
