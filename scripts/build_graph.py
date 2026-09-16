"""PLAN §4.1 の規則で結合行列を作り、data/graph.npz と data/neurons.parquet に保存する。"""

import argparse
import json
from pathlib import Path

from multibrain.data.malecns import EXCLUDED_SUPERCLASS, MIN_WEIGHT, build_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)

    graph, neurons, sign_summary = build_graph(Path(args.raw))
    graph.save(out / "graph.npz")
    neurons.to_parquet(out / "neurons.parquet")

    manifest_path = out / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["graph"] = {
        "rules": {"status": "Traced", "excluded_superclass": list(EXCLUDED_SUPERCLASS), "min_weight": MIN_WEIGHT,
                  "weight": "w0_ij = c * s_j * n_ij / sqrt(D_i), D_i = stats.post"},
        "N": int(len(graph.body_ids)),
        "nnz": int(graph.n_syn.nnz),
        "synapses": int(graph.n_syn.data.sum()),
        "sign": sign_summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["graph"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
