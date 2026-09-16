"""MaleCNS v1.0 の flat-connectome を PLAN §4.1 の規則で読み、ニューロン集合と結合行列を作る。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import scipy.sparse as sp

FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "stats": "body-stats-male-cns-v1.0-minconf-0.5.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
EXCLUDED_SUPERCLASS = ("ol_intrinsic", "ol_sensory")
MIN_WEIGHT = 2
SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0,
        "dopamine": 0.0, "serotonin": 0.0, "octopamine": 0.0}
CONF_MIN = 0.5


def load_neurons(raw: Path) -> pd.DataFrame:
    """集合に入るニューロンの注釈。bodyId 昇順、index はそのまま行番号（= ニューロン番号）。"""
    ann = feather.read_table(raw / FILES["annotations"]).to_pandas()
    keep = ann[(ann["status"] == "Traced") & ~ann["superclass"].isin(EXCLUDED_SUPERCLASS)]
    keep = keep.sort_values("bodyId").reset_index(drop=True)
    for col in ("superclass", "class", "subclass", "type", "entryNerve", "somaSide", "rootSide", "somaNeuromere"):
        keep[col] = keep[col].astype("string")
    return keep


def load_signs(raw: Path, neurons: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """前シナプス側の符号 s_j。確信度 0.5 未満は細胞型の予測で置き換え、それでも不明なら +1。"""
    nt = feather.read_table(raw / FILES["neurotransmitters"],
                            columns=["body", "consensus_nt", "predicted_nt_confidence", "celltype_predicted_nt"])
    nt = nt.to_pandas().drop_duplicates("body").set_index("body").reindex(neurons["bodyId"])
    low = nt["predicted_nt_confidence"].isna() | (nt["predicted_nt_confidence"] < CONF_MIN)
    chosen = nt["consensus_nt"].astype("string")
    chosen[low] = nt.loc[low, "celltype_predicted_nt"].astype("string")
    sign = chosen.map(SIGN)
    unresolved = sign.isna()
    sign = sign.fillna(1.0).to_numpy(dtype=np.float32)
    summary = {"chosen": chosen.fillna("<NA>").value_counts().to_dict(),
               "unresolved_default_plus": int(unresolved.sum()),
               "modulatory_zero": int((sign == 0).sum())}
    return sign, summary


def load_edges(raw: Path, neurons: pd.DataFrame) -> sp.csr_matrix:
    """n_ij（j → i のシナプス数、min_weight 以上）を CSR で返す。行 i が後シナプス側。"""
    ids = pa.array(neurons["bodyId"].to_numpy(), type=pa.int64())
    w = feather.read_table(raw / FILES["weights"], memory_map=True)
    w = w.filter(pc.and_(pc.is_in(w["body_pre"], value_set=ids), pc.is_in(w["body_post"], value_set=ids)))
    w = w.filter(pc.greater_equal(w["weight"], MIN_WEIGHT))
    lookup = pd.Series(np.arange(len(neurons)), index=neurons["bodyId"].to_numpy())
    row = lookup[w["body_post"].to_numpy()].to_numpy()
    col = lookup[w["body_pre"].to_numpy()].to_numpy()
    n = len(neurons)
    return sp.csr_matrix((w["weight"].to_numpy().astype(np.float32), (row, col)), shape=(n, n))


def load_in_synapses(raw: Path, neurons: pd.DataFrame) -> np.ndarray:
    """D_i：stats の post（データセット全体での入力シナプス総数）。"""
    ids = pa.array(neurons["bodyId"].to_numpy(), type=pa.int64())
    st = feather.read_table(raw / FILES["stats"], columns=["body", "post"])
    st = st.filter(pc.is_in(st["body"], value_set=ids)).to_pandas().drop_duplicates("body")
    st = st.set_index("body").reindex(neurons["bodyId"])
    return st["post"].fillna(0).to_numpy(dtype=np.float32)


@dataclass
class Graph:
    body_ids: np.ndarray       # (N,) int64
    n_syn: sp.csr_matrix       # (N, N) float32、n_ij
    sign: np.ndarray           # (N,) float32、前シナプス側の符号 s_j
    in_synapses: np.ndarray    # (N,) float32、D_i

    def base_weights(self, c: float) -> sp.csr_matrix:
        """w0_ij = c · s_j · n_ij / sqrt(D_i)。"""
        d = np.sqrt(np.maximum(self.in_synapses, 1.0))
        w = self.n_syn.multiply(self.sign[None, :]).multiply(1.0 / d[:, None]).tocsr()
        w.data = (c * w.data).astype(np.float32)
        w.eliminate_zeros()
        return w

    def save(self, path: Path) -> None:
        np.savez(path, body_ids=self.body_ids, indptr=self.n_syn.indptr, indices=self.n_syn.indices,
                 n_syn=self.n_syn.data, sign=self.sign, in_synapses=self.in_synapses)

    @classmethod
    def load(cls, path: Path) -> "Graph":
        z = np.load(path)
        n = len(z["body_ids"])
        return cls(z["body_ids"], sp.csr_matrix((z["n_syn"], z["indices"], z["indptr"]), shape=(n, n)),
                   z["sign"], z["in_synapses"])


def build_graph(raw: Path) -> tuple[Graph, pd.DataFrame, dict]:
    neurons = load_neurons(raw)
    sign, sign_summary = load_signs(raw, neurons)
    n_syn = load_edges(raw, neurons)
    d = load_in_synapses(raw, neurons)
    return Graph(neurons["bodyId"].to_numpy(), n_syn, sign, d), neurons, sign_summary
