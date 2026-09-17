"""Smoke test for scripts/train_mono.py (PLAN.md §7, §11.2, §12.3).

Runs the script as a subprocess at toy scale
(--nworld 4 --window 4 --collect-len 8 --total-steps 64, no eval, no
replay) and checks that it exits cleanly and that runs/mono_smoke/
log.jsonl contains `update` lines carrying the §12.3 neural diagnostics
and a `done` line. Must finish well under 5 minutes. Skips without CUDA
or data/ (same condition as test_ppo_window.py).
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "runs" / "mono_smoke"

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available()
         and (DATA / "graph.npz").exists()
         and (DATA / "ports.json").exists()
         and (DATA / "neurons.parquet").exists()),
    reason="CUDA か data/ が無い",
)


def test_train_mono_smoke():
    cmd = [
        sys.executable, str(REPO / "scripts" / "train_mono.py"),
        "--nworld", "4", "--window", "4", "--collect-len", "8",
        "--total-steps", "64", "--eval-every", "0",
        "--out", str(OUT), "--no-play-after",
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\n--- stdout ---\n{proc.stdout[-4000:]}"
        f"\n--- stderr ---\n{proc.stderr[-4000:]}")

    lines = [json.loads(l) for l in
             (OUT / "log.jsonl").read_text().splitlines() if l.strip()]
    updates = [l for l in lines if l.get("type") == "update"]
    dones = [l for l in lines if l.get("type") == "done"]
    assert updates, "log.jsonl に update 行が無い"
    assert dones, "log.jsonl に done 行が無い"
    for key in ("s_mean", "s_saturated_frac", "v_abs_max",
                "core_grad_norm", "ports_grad_norm"):
        assert key in updates[-1], key
        assert math.isfinite(updates[-1][key]), key
