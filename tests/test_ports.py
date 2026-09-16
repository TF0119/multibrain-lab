"""ports.json と神経中核の整合性。data/ が無い環境では飛ばす。"""

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from multibrain.brain.core import BrainCore, Ports

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (DATA / "ports.json").exists(), reason="data/ports.json が無い（scripts/build_ports.py を先に実行）")


@pytest.fixture(scope="module")
def ports_json():
    return json.loads((DATA / "ports.json").read_text())


def test_ports_cover_observation_and_joints(ports_json):
    assert len(ports_json["inputs"]) == 76
    assert len(ports_json["outputs"]) == 27
    assert sorted(p["joint"] for p in ports_json["outputs"]) == sorted(ports_json["joints"])
    assert all(p["n"] >= 8 for p in ports_json["inputs"])


def test_input_and_output_neurons_are_disjoint(ports_json):
    ins = {i for p in ports_json["inputs"] for i in p["neurons"]}
    outs = {i for p in ports_json["outputs"] for i in p["neurons"]}
    assert not (ins & outs)
    assert sum(p["n"] for p in ports_json["inputs"]) == len(ins), "入力ニューロンは信号間で重複しない"


def test_core_shapes_on_small_random_graph(ports_json):
    """ports の bodyId をすべて含む小さな乱数配線で、形と勾配の到達を確かめる。"""
    ids = sorted({i for p in ports_json["inputs"] + ports_json["outputs"] for i in p["neurons"]})
    body_ids = np.array(ids, dtype=np.int64)
    n = len(body_ids)
    rng = np.random.default_rng(0)
    w0 = sp.random(n, n, density=20 / n, random_state=rng, dtype=np.float32, format="csr") * 0.05
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ports = Ports.load(DATA / "ports.json", body_ids)
    core = BrainCore(w0, ports, n_brains=2, device=dev)
    obs = torch.rand(2, 3, 76, device=dev) * 2 - 1
    state = core.init_state(3)
    loss = 0.0
    for _ in range(4):
        state, a = core(state, obs)
        loss = loss + (a ** 2).mean()
    assert a.shape == (2, 3, 27)
    assert state[0].shape == (n, 6)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in core.parameters())
