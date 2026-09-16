"""PLAN §4.1 の規則でニューロン集合と接続を選び、規模（N、nnz）と注釈の分布を実測する。

使い方: python scripts/measure_scale.py [--raw data/raw] [--out data/m0_scale.json]
weights（1.5 億行）は pyarrow のまま扱い、pandas に変換しない。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

NAMES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "stats": "body-stats-male-cns-v1.0-minconf-0.5.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
EXCLUDED_SUPERCLASS = {"ol_intrinsic", "ol_sensory"}
MIN_WEIGHT = 2
SIGN = {"acetylcholine": +1, "gaba": -1, "glutamate": -1, "histamine": -1,
        "dopamine": 0, "serotonin": 0, "octopamine": 0, "unclear": None}


def vc(series, n=None):
    s = series.astype("string").value_counts(dropna=False)
    return {str(k): int(v) for k, v in (s if n is None else s.head(n)).items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/m0_scale.json")
    args = ap.parse_args()
    raw = Path(args.raw)
    report = {"rules": {"status": "Traced", "excluded_superclass": sorted(EXCLUDED_SUPERCLASS),
                        "min_weight": MIN_WEIGHT, "sign": {k: v for k, v in SIGN.items() if v is not None}}}

    # --- ニューロン集合 ---
    ann = feather.read_table(raw / NAMES["annotations"]).to_pandas()
    traced = ann[ann["status"] == "Traced"]
    has_soma = traced["somaLocation"].notna()
    keep = traced[~traced["superclass"].isin(EXCLUDED_SUPERCLASS)]
    report["neurons"] = {
        "annotation_rows": len(ann),
        "traced": len(traced),
        "traced_with_soma": int(has_soma.sum()),
        "traced_superclass_na": int(traced["superclass"].isna().sum()),
        "kept_N": len(keep),
        "kept_with_soma": int(keep["somaLocation"].notna().sum()),
        "kept_by_superclass": vc(keep["superclass"]),
    }
    body_ids = pa.array(keep["bodyId"].to_numpy(), type=pa.int64())
    body_set = set(keep["bodyId"].tolist())
    print(f"N = {len(keep)} (traced {len(traced)}, excluded OL {len(traced) - len(keep)})")

    # --- 注釈の分布（§4.3、§4.4、§6 の割当に使う） ---
    sens = keep[keep["superclass"].isin(["vnc_sensory", "cb_sensory", "sensory_ascending", "sensory_descending"])]
    sens_key = (sens["class"].astype("string").fillna("-") + " | " + sens["subclass"].astype("string").fillna("-")
                + " | " + sens["entryNerve"].astype("string").fillna("-"))
    motor = keep[keep["superclass"].isin(["vnc_motor", "cb_motor"])]
    motor_key = motor["somaNeuromere"].astype("string").fillna("-") + " | " + motor["type"].astype("string").fillna("-")
    dn = keep[keep["superclass"] == "descending_neuron"]
    an = keep[keep["superclass"] == "ascending_neuron"]
    vp = keep[keep["superclass"] == "visual_projection"]
    report["annotations"] = {
        "sensory_total": len(sens),
        "sensory_by_class_subclass_nerve": vc(sens_key, 60),
        "motor_total": len(motor),
        "motor_by_neuromere_type": vc(motor_key),
        "descending_total": len(dn),
        "descending_type_prefix": vc(dn["type"].astype("string").str.extract(r"^(DN[a-z]+)")[0]),
        "descending_types": int(dn["type"].nunique()),
        "ascending_total": len(an),
        "ascending_types": int(an["type"].nunique()),
        "visual_projection_total": len(vp),
        "visual_projection_type_prefix": vc(vp["type"].astype("string").str.extract(r"^([A-Za-z]+)")[0], 20),
    }

    # --- 符号 ---
    nt = feather.read_table(raw / NAMES["neurotransmitters"],
                            columns=["body", "consensus_nt", "predicted_nt", "predicted_nt_confidence",
                                     "celltype_predicted_nt"]).to_pandas()
    nt = nt[nt["body"].isin(body_set)].set_index("body")
    nt = nt.reindex(keep["bodyId"])
    conf = nt["predicted_nt_confidence"]
    low = conf.isna() | (conf < 0.5)
    chosen = nt["consensus_nt"].astype("string").copy()
    chosen[low] = nt.loc[low, "celltype_predicted_nt"].astype("string")
    report["sign"] = {
        "with_nt_row": int(nt["consensus_nt"].notna().sum()),
        "consensus_nt": vc(nt["consensus_nt"]),
        "confidence_below_0.5_or_na": int(low.sum()),
        "chosen_after_celltype_fallback": vc(chosen),
        "still_unclear_or_na": int(((chosen == "unclear") | chosen.isna()).sum()),
        "modulatory_zero_sign": int(chosen.isin(["dopamine", "serotonin", "octopamine"]).sum()),
    }

    # --- 接続 ---
    w = feather.read_table(raw / NAMES["weights"], memory_map=True)
    report["edges"] = {"all_rows": w.num_rows, "all_synapses": int(pc.sum(w["weight"]).as_py())}
    mask = pc.and_(pc.is_in(w["body_pre"], value_set=body_ids), pc.is_in(w["body_post"], value_set=body_ids))
    w = w.filter(mask)
    report["edges"]["within_set_rows"] = w.num_rows
    report["edges"]["within_set_synapses"] = int(pc.sum(w["weight"]).as_py())
    w = w.filter(pc.greater_equal(w["weight"], MIN_WEIGHT))
    weight = w["weight"].to_numpy()
    report["edges"]["nnz_min_weight"] = w.num_rows
    report["edges"]["synapses_min_weight"] = int(weight.sum())
    report["edges"]["weight_percentiles"] = {str(q): float(np.percentile(weight, q)) for q in (50, 90, 99, 99.9)}
    report["edges"]["weight_max"] = int(weight.max())

    post = w["body_post"].to_numpy()
    pre = w["body_pre"].to_numpy()
    in_deg = np.unique(post, return_counts=True)[1]
    out_deg = np.unique(pre, return_counts=True)[1]
    report["degree"] = {
        "neurons_with_input": int(len(in_deg)), "neurons_with_output": int(len(out_deg)),
        "in_degree_mean": float(in_deg.mean()), "in_degree_p99": float(np.percentile(in_deg, 99)),
        "out_degree_mean": float(out_deg.mean()), "out_degree_max": int(out_deg.max()),
    }
    isolated = len(body_set) - len(set(post.tolist()) | set(pre.tolist()))
    report["degree"]["isolated_in_set"] = int(isolated)

    # --- D_i：stats の post（全入力シナプス）と、集合内の入力シナプス和 ---
    st = feather.read_table(raw / NAMES["stats"], columns=["body", "pre", "post"])
    st = st.filter(pc.is_in(st["body"], value_set=body_ids)).to_pandas().set_index("body").reindex(keep["bodyId"])
    report["D_i"] = {
        "stats_post_median": float(st["post"].median()), "stats_post_mean": float(st["post"].mean()),
        "stats_pre_median": float(st["pre"].median()),
        "within_set_in_synapses_mean": float(report["edges"]["synapses_min_weight"] / len(keep)),
    }

    # --- メモリ見積もり（§4.5 の式に実測値を入れる） ---
    N, nnz = len(keep), w.num_rows
    report["memory_estimate"] = {
        "weights_csr_fp16_int32_MB": nnz * 6 / 1e6,
        "state_per_env_per_brain_KB": 3 * N * 2 / 1e3,
        "history_T16_env128_brain10_GB": 10 * 128 * 16 * 3 * N * 2 / 1e9,
        "spmm_GFLOP_per_step_B1024": 2 * nnz * 1024 / 1e9,
    }

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("neurons", "sign", "edges", "degree", "D_i", "memory_estimate")},
                     indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
