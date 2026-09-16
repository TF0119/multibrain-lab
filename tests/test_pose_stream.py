"""Tests for the viewer pose stream (PLAN.md §11.1)."""

import json
import socket
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from multibrain.monitor import (
    PoseStreamer,
    build_meta,
    decode_frame,
    encode_frame,
    fk,
)

XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "humanoid.xml"


@pytest.fixture(scope="module")
def mjm():
    return mujoco.MjModel.from_xml_path(str(XML_PATH))


@pytest.fixture(scope="module")
def meta(mjm):
    return build_meta(mjm)


def test_meta_structure(mjm, meta):
    assert meta["version"] == 1
    assert meta["nq"] == 34
    assert len(meta["joints"]) == 27
    assert len(meta["bodies"]) == 15
    free = [b for b in meta["bodies"] if b["free"]]
    assert len(free) == 1 and free[0]["name"] == "pelvis"
    # every body has geoms, a bone mapping, and a segment-dir entry
    for b in meta["bodies"]:
        assert meta["geoms"].get(b["name"]), b["name"]
        assert b["name"] in meta["vrm_map"], b["name"]
        assert b["name"] in meta["segment_dir"], b["name"]
    # parents form a tree rooted at pelvis
    roots = [b["name"] for b in meta["bodies"] if b["parent"] is None]
    assert roots == ["pelvis"]
    for b in meta["bodies"]:
        if b["name"] == "pelvis":
            continue
        assert b["parent"] in {x["name"] for x in meta["bodies"]}
    # rest_qpos is a valid qpos
    assert len(meta["rest_qpos"]) == mjm.nq
    assert meta["rest_qpos"][2] == pytest.approx(0.84)


def _quat_close(q1, q2, atol=1e-5):
    d = abs(sum(a * b for a, b in zip(q1, q2)))
    assert d == pytest.approx(1.0, abs=atol)


def test_fk_matches_mujoco(mjm, meta):
    rng = np.random.default_rng(1)
    for _ in range(20):
        d = mujoco.MjData(mjm)
        d.qpos[:] = mjm.qpos0
        d.qpos[0] = rng.uniform(-0.5, 0.5)
        d.qpos[1] = rng.uniform(-0.5, 0.5)
        d.qpos[2] = rng.uniform(0.1, 1.0)
        # random valid root quat
        u = rng.normal(size=4)
        d.qpos[3:7] = u / np.linalg.norm(u)
        for j in range(mjm.njnt):
            if mjm.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            lo, hi = mjm.jnt_range[j]
            d.qpos[mjm.jnt_qposadr[j]] = rng.uniform(lo, hi)
        mujoco.mj_forward(mjm, d)
        poses = fk(meta, d.qpos)
        for i in range(1, mjm.nbody):
            name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i)
            pos, quat = poses[name]
            np.testing.assert_allclose(
                pos, d.xpos[i], atol=1e-9, err_msg=name)
            _quat_close(quat, tuple(d.xquat[i]))


def test_frame_codec():
    q = np.linspace(-1, 1, 34).astype(np.float64)
    buf = encode_frame(7, 1.25, q)
    assert buf[:4] == b"MBP1"
    t, seq, back = decode_frame(buf)
    assert t == 1.25 and seq == 7
    np.testing.assert_allclose(back, q, atol=1e-6)


def _connect(port):
    from websockets.sync.client import connect

    return connect(f"ws://127.0.0.1:{port}", open_timeout=5, legacy=True)


def test_stream_end_to_end(meta):
    streamer = PoseStreamer(meta, port=0, hz=60).start()
    assert streamer.enabled
    try:
        ws = _connect(streamer.port)
        meta_msg = json.loads(ws.recv(timeout=5))
        assert meta_msg["type"] == "meta"
        assert meta_msg["nq"] == 34
        time.sleep(0.05)  # let the handler register before submits
        rng = np.random.default_rng(0)
        for k in range(5):
            streamer.submit(0.02 * k, rng.normal(size=34))
            time.sleep(0.02)  # above the 1/60 s interval
        t, seq, qpos = decode_frame(ws.recv(timeout=5))
        assert seq >= 1 and len(qpos) == 34
        ws.close()
        assert streamer.stats["sent"] >= 1
    finally:
        streamer.close()


def test_rate_limit_and_latest_only(meta):
    streamer = PoseStreamer(meta, port=0, hz=30).start()
    try:
        ws = _connect(streamer.port)
        ws.recv(timeout=5)  # meta
        time.sleep(0.05)
        for k in range(200):  # far faster than 30 Hz
            streamer.submit(k * 0.02, np.full(34, k))
        assert streamer.stats["dropped_rate"] > 100
        assert streamer.stats["submitted"] <= 2
        ws.close()
    finally:
        streamer.close()


def test_no_clients_is_cheap(meta):
    streamer = PoseStreamer(meta, port=0, hz=30).start()
    try:
        t0 = time.monotonic()
        for k in range(1000):
            assert streamer.submit(k, np.zeros(34)) is False
        assert time.monotonic() - t0 < 1.0
        assert streamer.stats["submitted"] == 0
    finally:
        streamer.close()


def test_port_busy_disables(meta):
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        streamer = PoseStreamer(meta, port=port, hz=30).start()
        assert not streamer.enabled
        assert streamer.submit(0.0, np.zeros(34)) is False
        streamer.close()
    finally:
        blocker.close()
