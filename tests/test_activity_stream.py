"""Tests for the neuron activity stream, protocol MBA1 (PLAN.md §11.2)."""

import json
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from multibrain.monitor import (
    GROUPS,
    SAMPLE_GROUPS,
    SAMPLE_PER_GROUP,
    BrainActivitySource,
    PoseStreamer,
    activity_meta,
    build_groups,
    build_meta,
    build_sample,
    decode_activity,
    decode_frame,
    encode_activity,
)
from multibrain.monitor.activity import MAGIC as ACTIVITY_MAGIC

REPO = Path(__file__).resolve().parents[1]
XML_PATH = REPO / "assets" / "humanoid.xml"
DATA = REPO / "data"
needs_data = pytest.mark.skipif(
    not ((DATA / "neurons.parquet").exists() and (DATA / "ports.json").exists()),
    reason="data/ が無い（scripts/build_graph.py, build_ports.py を先に実行）",
)


@pytest.fixture(scope="module")
def mjm():
    return mujoco.MjModel.from_xml_path(str(XML_PATH))


def test_activity_codec():
    K, G, M = 3, 12, 512
    rng = np.random.default_rng(0)
    stats = rng.random((K, G, 2), dtype=np.float32)
    sample = rng.integers(0, 256, (K, M), dtype=np.uint8)
    buf = encode_activity(9, 1.5, stats, sample)
    assert buf[:4] == b"MBA1"
    assert len(buf) == 4 + 8 + 4 + 8 + K * G * 2 * 4 + K * M
    t, seq, stats2, sample2 = decode_activity(buf)
    assert t == 1.5 and seq == 9
    np.testing.assert_array_equal(stats2, stats)
    np.testing.assert_array_equal(sample2, sample)


@needs_data
def test_build_groups_counts():
    import pandas as pd

    neurons = pd.read_parquet(DATA / "neurons.parquet")
    ports = json.loads((DATA / "ports.json").read_text())
    groups = build_groups(neurons, ports)
    by_id = {g["id"]: g for g in groups}
    assert [g["id"] for g in groups] == [g["id"] for g in GROUPS]
    assert by_id["sensory"]["n"] == 11782
    assert by_id["descending"]["n"] == 1314
    assert by_id["ascending"]["n"] == 1846
    assert by_id["motor"]["n"] == 815
    assert by_id["kenyon"]["n"] == 4064
    assert by_id["cx"]["n"] == 2950
    assert by_id["all"]["n"] == 71618
    assert by_id["input_port"]["n"] == sum(p["n"] for p in ports["inputs"])
    outs = {b for p in ports["outputs"] for b in p["neurons"]}
    assert by_id["output_port"]["n"] == len(outs)
    for g in groups:
        rows = g["rows"]
        assert rows.dtype == np.int64
        assert len(np.unique(rows)) == g["n"]
        assert (np.diff(rows) > 0).all()


@needs_data
def test_build_sample():
    import pandas as pd

    neurons = pd.read_parquet(DATA / "neurons.parquet")
    ports = json.loads((DATA / "ports.json").read_text())
    groups = build_groups(neurons, ports)
    sample = build_sample(neurons, groups)
    assert len(sample["rows"]) == len(SAMPLE_GROUPS) * SAMPLE_PER_GROUP
    assert len(np.unique(sample["rows"])) == len(sample["rows"])
    assert neurons["somaLocation"].iloc[sample["rows"]].notna().all()
    assert sample["pos"].dtype == np.float32
    assert sample["pos"].min() >= -1.0 and sample["pos"].max() <= 1.0
    for gid in SAMPLE_GROUPS:
        gi = next(i for i, g in enumerate(groups) if g["id"] == gid)
        assert (sample["group"] == gi).sum() == SAMPLE_PER_GROUP
        assert set(sample["rows"][sample["group"] == gi]) <= set(groups[gi]["rows"].tolist())


def test_activity_source_cpu():
    import torch

    N, K, B, env = 12, 2, 3, 1
    groups = [
        {"id": "g0", "label": "a", "n": 4, "rows": np.array([0, 1, 2, 3])},
        {"id": "g1", "label": "b", "n": 3, "rows": np.array([4, 5, 6])},
        {"id": "all", "label": "all", "n": N, "rows": np.arange(N)},
    ]
    sample = {"rows": np.array([0, 5, 9]), "group": np.array([0, 1, 2]),
              "pos": np.zeros((3, 3), np.float32)}
    src = BrainActivitySource(groups, sample, N, K, device="cpu", env=env)
    rng = np.random.default_rng(0)
    s_np = rng.random((N, K * B)).astype(np.float32)
    assert src.read() is None  # nothing enqueued yet
    src.enqueue(torch.from_numpy(s_np))
    out = src.read()
    assert out is not None
    stats, samp = out
    se = s_np.reshape(N, K, B)[:, :, env]
    for gi, g in enumerate(groups):
        np.testing.assert_allclose(stats[:, gi, 0], se[g["rows"]].mean(0), atol=1e-6)
        np.testing.assert_allclose(stats[:, gi, 1], (se[g["rows"]] > 0.5).mean(0), atol=1e-6)
    expect = np.rint(se[sample["rows"]] * 255).clip(0, 255).astype(np.uint8).T
    np.testing.assert_array_equal(samp, expect)


def _connect(port):
    from websockets.sync.client import connect

    return connect(f"ws://127.0.0.1:{port}", open_timeout=5, legacy=True)


def test_stream_activity_end_to_end(mjm):
    meta = build_meta(mjm)
    groups = [{"id": "g", "label": "g", "n": 2, "rows": np.arange(2)}]
    sample = {"rows": np.arange(3), "group": np.zeros(3, int),
              "pos": np.zeros((3, 3), np.float32)}
    meta["activity"] = activity_meta(groups, sample, ["mono"])
    json.dumps(meta)  # meta must stay JSON-serializable
    streamer = PoseStreamer(meta, port=0, hz=60, activity_hz=10).start()
    assert streamer.enabled
    try:
        ws = _connect(streamer.port)
        meta_msg = json.loads(ws.recv(timeout=5))
        assert meta_msg["activity"]["brains"] == ["mono"]
        assert meta_msg["activity"]["groups"][0]["id"] == "g"
        time.sleep(0.05)  # let the handler register before submits
        stats = np.zeros((1, 1, 2), np.float32)
        samp = np.zeros((1, 3), np.uint8)
        for k in range(10):
            streamer.submit(0.02 * k, np.zeros(34))
            if streamer.wants_activity():
                streamer.submit_activity(0.02 * k, stats, samp)
            time.sleep(0.05)
        got = set()
        mba1 = None
        for _ in range(30):
            try:
                msg = ws.recv(timeout=2)
            except TimeoutError:
                break
            if not isinstance(msg, bytes):
                continue
            got.add(msg[:4])
            if msg[:4] == ACTIVITY_MAGIC:
                mba1 = msg
            if {b"MBP1", b"MBA1"} <= got:
                break
        assert b"MBP1" in got and b"MBA1" in got
        t, seq, st, sa = decode_activity(mba1)
        assert st.shape == (1, 1, 2) and sa.shape == (1, 3)
        assert streamer.stats["activity_submitted"] >= 1
        assert streamer.stats["activity_sent"] >= 1
        # activity_hz=10: a fast burst of 100 calls submits at most ~2
        before = streamer.stats["activity_submitted"]
        for _ in range(100):
            streamer.submit_activity(0.0, stats, samp)
        assert streamer.stats["activity_submitted"] - before <= 2
        ws.close()
    finally:
        streamer.close()


def test_no_activity_meta_disables(mjm):
    meta = build_meta(mjm)  # no "activity" key: e.g. mlp_control
    streamer = PoseStreamer(meta, port=0, hz=30).start()
    try:
        ws = _connect(streamer.port)
        ws.recv(timeout=5)  # meta
        time.sleep(0.05)
        assert not streamer.wants_activity()
        assert streamer.submit_activity(0.0, np.zeros((1, 1, 2), np.float32),
                                        np.zeros((1, 1), np.uint8)) is False
        ws.close()
    finally:
        streamer.close()
