"""取得した feather のスキーマと、注釈列の値分布を表示する（M0 の最初の確認）。

使い方: python scripts/inspect_schema.py [--raw data/raw]
weights と stats は大きいので、スキーマと先頭行と行数だけを見る。
"""

import argparse
from pathlib import Path

import pyarrow.feather as feather
import pyarrow as pa

NAMES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "stats": "body-stats-male-cns-v1.0-minconf-0.5.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
MAX_UNIQUE = 40


def show_schema(key: str, table: pa.Table) -> None:
    print(f"\n===== {key}: {table.num_rows} rows × {table.num_columns} cols =====")
    for field in table.schema:
        print(f"  {field.name:32s} {field.type}")


def show_values(table: pa.Table) -> None:
    df = table.to_pandas()
    print(df.head(3).to_string())
    for field in table.schema:
        col, t = field.name, field.type
        if pa.types.is_list(t) or pa.types.is_large_list(t):
            continue
        if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_dictionary(t):
            s = df[col].astype("string")
            n, null = s.nunique(dropna=True), s.isna().sum()
            print(f"\n--- {col}: {n} unique, {null} null")
            print(s.value_counts(dropna=False).head(MAX_UNIQUE if n <= MAX_UNIQUE else 15).to_string())
        elif pa.types.is_integer(t) or pa.types.is_floating(t):
            print(f"\n--- {col}: min {df[col].min()}, max {df[col].max()}, null {df[col].isna().sum()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    args = ap.parse_args()
    raw = Path(args.raw)

    for key in ("annotations", "neurotransmitters"):
        t = feather.read_table(raw / NAMES[key])
        show_schema(key, t)
        show_values(t)

    for key in ("stats", "weights"):
        t = feather.read_table(raw / NAMES[key])
        show_schema(key, t)
        print(t.slice(0, 5).to_pandas().to_string())


if __name__ == "__main__":
    main()
