"""Neuron activity streaming for the viewer HUD (PLAN.md §11.2).

Instead of shipping all N neurons per frame, the watched env (env 0) sends
per-brain activity in two granularities:

- Group aggregates: for the 12 groups below, mean activity ``s`` and the
  active fraction (``s > 0.5``) per brain.
- Fixed sample: M = 512 neurons (``SAMPLE_PER_GROUP`` per group drawn from
  the 8 ``SAMPLE_GROUPS`` that have somaLocation), quantized to uint8.

Wire protocol "MBA1" (little endian):

    u32  magic "MBA1"
    f64  t        (seconds)
    u32  seq
    u16  K        n_brains
    u16  G        n_groups
    u16  M        n_sample
    u16  reserved = 0
    f32  stats[K][G][2]   ([..., 0] = mean, [..., 1] = active fraction)
    u8   sample[K][M]

Group and sample layout travels once in the JSON meta under "activity";
frames carry only the numbers.

``BrainActivitySource`` mirrors ``WarpPoseSource``: ``enqueue()`` issues the
group sparse matmul, the sample gather, and the device->pinned-host copies
on the caller's stream without ever synchronizing; ``read()`` only queries
a CUDA event and returns the completed copy once (or None -> not yet).
Loop order per body step: ``read()`` the previous step's copy and submit
it, then ``wants_activity()`` -> ``enqueue()`` for the next one.
"""

import struct

import numpy as np
import pandas as pd

MAGIC = b"MBA1"
SAMPLE_PER_GROUP = 64
SAMPLE_SEED = 0

_SENSORY = ("vnc_sensory", "cb_sensory", "sensory_ascending", "sensory_descending")
_MOTOR = ("vnc_motor", "cb_motor")

# The 12 groups in PLAN §11.2 table order. rule: ("superclass", names) /
# ("class", name) / ("port", inputs|outputs) / ("all", None).
GROUPS = [
    {"id": "sensory",           "label": "感覚",       "rule": ("superclass", _SENSORY)},
    {"id": "input_port",        "label": "入力ポート", "rule": ("port", "inputs")},
    {"id": "descending",        "label": "下行性",     "rule": ("superclass", ("descending_neuron",))},
    {"id": "ascending",         "label": "上行性",     "rule": ("superclass", ("ascending_neuron",))},
    {"id": "motor",             "label": "運動",       "rule": ("superclass", _MOTOR)},
    {"id": "output_port",       "label": "出力ポート", "rule": ("port", "outputs")},
    {"id": "cb_intrinsic",      "label": "脳内在",     "rule": ("superclass", ("cb_intrinsic",))},
    {"id": "vnc_intrinsic",     "label": "神経索内在", "rule": ("superclass", ("vnc_intrinsic",))},
    {"id": "kenyon",            "label": "キノコ体",   "rule": ("class", "Kenyon_Cell")},
    {"id": "cx",                "label": "中心複合体", "rule": ("class", "CX")},
    {"id": "visual_projection", "label": "視覚投射",   "rule": ("superclass", ("visual_projection",))},
    {"id": "all",               "label": "全体",       "rule": ("all", None)},
]

# Groups eligible for the fixed sample (the ones with somaLocation).
SAMPLE_GROUPS = ["descending", "ascending", "motor", "cb_intrinsic",
                 "vnc_intrinsic", "kenyon", "cx", "visual_projection"]


def build_groups(neurons: pd.DataFrame, ports: dict) -> list[dict]:
    """Resolve the 12 groups to neuron row indices (ascending int64).

    ``neurons`` rows are used as-is (row number == graph row == bodyId order).
    Port groups map ports.json bodyIds through the bodyId column.
    """
    lookup = pd.Series(np.arange(len(neurons), dtype=np.int64),
                       index=neurons["bodyId"].to_numpy())
    out = []
    for spec in GROUPS:
        kind, key = spec["rule"]
        if kind == "all":
            rows = np.arange(len(neurons), dtype=np.int64)
        elif kind == "port":
            ids = np.unique(np.concatenate(
                [np.asarray(p["neurons"], dtype=np.int64) for p in ports[key]]))
            rows = lookup.loc[ids].to_numpy(dtype=np.int64)
        elif kind == "class":
            rows = np.flatnonzero(
                neurons["class"].eq(key).fillna(False).to_numpy(dtype=bool))
        else:
            rows = np.flatnonzero(
                neurons["superclass"].isin(key).fillna(False).to_numpy(dtype=bool))
        rows = np.sort(rows.astype(np.int64))
        out.append({"id": spec["id"], "label": spec["label"],
                    "n": int(len(rows)), "rows": rows})
    return out


def build_sample(neurons: pd.DataFrame, groups: list[dict],
                 seed: int = SAMPLE_SEED, per_group: int = SAMPLE_PER_GROUP) -> dict:
    """Pick the fixed sample: ``per_group`` neurons per SAMPLE_GROUPS group.

    Only rows with a non-null somaLocation are eligible; rows are unique
    across the whole sample (class-based groups overlap superclass groups).
    ``pos`` is somaLocation min/max-normalized per axis to [-1, 1] over all
    neurons that have a somaLocation.
    """
    rng = np.random.default_rng(seed)
    has_soma = neurons["somaLocation"].notna().to_numpy()
    soma = np.stack(neurons.loc[has_soma, "somaLocation"].to_numpy()).astype(np.float64)
    lo = soma.min(axis=0)
    span = np.maximum(soma.max(axis=0) - lo, 1.0)
    index = {g["id"]: i for i, g in enumerate(groups)}
    taken = np.zeros(len(neurons), dtype=bool)
    rows, grp, pos = [], [], []
    for gid in SAMPLE_GROUPS:
        cand = groups[index[gid]]["rows"]
        cand = cand[has_soma[cand] & ~taken[cand]]
        k = min(per_group, len(cand))
        sel = np.sort(cand[rng.choice(len(cand), size=k, replace=False)])
        taken[sel] = True
        s = np.stack(neurons["somaLocation"].iloc[sel].to_numpy()).astype(np.float64)
        rows.append(sel)
        grp.append(np.full(k, index[gid], dtype=np.int64))
        pos.append(((s - lo) / span * 2.0 - 1.0).astype(np.float32))
    return {"rows": np.concatenate(rows), "group": np.concatenate(grp),
            "pos": np.concatenate(pos)}


def activity_meta(groups: list[dict], sample: dict, brains: list[str],
                  hz: float = 10.0) -> dict:
    """JSON-safe dict for ``meta["activity"]``: layout of every MBA1 frame."""
    return {
        "hz": float(hz),
        "brains": [str(b) for b in brains],
        "groups": [{"id": g["id"], "label": g["label"], "n": int(g["n"])}
                   for g in groups],
        "sample": {
            "n": int(len(sample["rows"])),
            "rows": [int(r) for r in sample["rows"]],
            "group": [int(g) for g in sample["group"]],
            "pos": [[float(v) for v in p] for p in sample["pos"]],
        },
    }


def encode_activity(seq: int, t: float, stats: np.ndarray, sample: np.ndarray) -> bytes:
    """Pack one MBA1 frame. stats: (K, G, 2) f32, sample: (K, M) u8."""
    stats = np.ascontiguousarray(stats, dtype="<f4")
    sample = np.ascontiguousarray(sample, dtype=np.uint8)
    K, G = stats.shape[:2]
    M = sample.shape[1]
    return (MAGIC + struct.pack("<dI4H", float(t), int(seq), K, G, M, 0)
            + stats.tobytes() + sample.tobytes())


def decode_activity(buf) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Decode an MBA1 frame -> (t, seq, stats (K,G,2) f32, sample (K,M) u8)."""
    buf = bytes(buf)
    if buf[:4] != MAGIC:
        raise ValueError("bad magic")
    t, seq, K, G, M, _ = struct.unpack("<dI4H", buf[4:24])
    off = 24
    stats = np.frombuffer(buf[off:off + K * G * 2 * 4], dtype="<f4").reshape(K, G, 2).copy()
    off += K * G * 2 * 4
    sample = np.frombuffer(buf[off:off + K * M], dtype=np.uint8).reshape(K, M).copy()
    return t, seq, stats, sample


class BrainActivitySource:
    """Non-blocking env-`env` activity readout from a BrainCore state.

    Same contract as ``WarpPoseSource``: ``enqueue(s)`` runs on the caller's
    stream — group stats via one (G, N) sparse matmul, sample via a gather —
    then copies into pinned host buffers and records a CUDA event.
    ``read()`` only queries the event; it never syncs the GPU. On a CPU
    device everything is synchronous and there is no event.
    """

    def __init__(self, groups, sample, n_neurons: int, n_brains: int,
                 device="cuda", env: int = 0):
        import torch

        dev = torch.device(device)
        self.env = int(env)
        self.N, self.K = int(n_neurons), int(n_brains)
        self.G, self.M = len(groups), len(sample["rows"])
        crow, cols, vals = [0], [], []
        for g in groups:
            r = np.asarray(g["rows"], dtype=np.int64)
            cols.append(r.astype(np.int32))
            vals.append(np.full(len(r), 1.0 / max(len(r), 1), dtype=np.float32))
            crow.append(crow[-1] + len(r))
        self._ind = torch.sparse_csr_tensor(
            torch.from_numpy(np.asarray(crow, dtype=np.int32)),
            torch.from_numpy(np.concatenate(cols)),
            torch.from_numpy(np.concatenate(vals)),
            size=(self.G, self.N), device=dev)
        self._sample_rows = torch.from_numpy(
            np.asarray(sample["rows"], dtype=np.int64)).to(dev)
        pinned = dev.type == "cuda"
        self._stats = torch.empty(self.K, self.G, 2, dtype=torch.float32,
                                  pin_memory=pinned)
        self._samp = torch.empty(self.K, self.M, dtype=torch.uint8,
                                 pin_memory=pinned)
        self._event = torch.cuda.Event() if pinned else None
        self._armed = False

    def enqueue(self, s) -> None:
        """Queue stats+sample for env `env` from s: (N, K*B) activity."""
        import torch

        s_env = s.view(self.N, self.K, -1)[:, :, self.env].float()   # (N, K)
        mean = torch.sparse.mm(self._ind, s_env)                     # (G, K)
        active = torch.sparse.mm(self._ind, (s_env > 0.5).float())
        stats = torch.stack([mean, active], -1).permute(1, 0, 2)     # (K, G, 2)
        samp = (s_env[self._sample_rows] * 255).round().clamp(0, 255)
        self._stats.copy_(stats, non_blocking=True)
        self._samp.copy_(samp.to(torch.uint8).t().contiguous(), non_blocking=True)
        if self._event is not None:
            self._event.record()
        self._armed = True

    def read(self):
        """The completed (stats, sample) host copy of the last ``enqueue()``,
        or None while it is still in flight. Consumed once: a second call
        returns None until the next ``enqueue()``. Call it at the start of
        the next body step, before deciding whether to enqueue again, so the
        query never races the copy issued in the same step."""
        if not self._armed:
            return None
        if self._event is not None and not self._event.query():
            return None
        self._armed = False
        return self._stats.numpy().copy(), self._samp.numpy().copy()
