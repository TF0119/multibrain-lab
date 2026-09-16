"""MaleCNS v1.0 の flat-connectome から計画に必要な 4 ファイルを取得し、manifest を書く。

使い方: python scripts/fetch_malecns.py [--out data/raw]
再実行時は、既存ファイルのサイズがサーバの Content-Length と一致すれば再取得しない。
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "stats": "body-stats-male-cns-v1.0-minconf-0.5.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
CHUNK = 1 << 22


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


def download(url: str, dst: Path) -> None:
    size = remote_size(url)
    if dst.exists() and dst.stat().st_size == size:
        print(f"skip  {dst.name} ({size / 1e6:.0f} MB, already complete)")
        return
    done = dst.stat().st_size if dst.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    mode = "ab" if done else "wb"
    with urllib.request.urlopen(req) as r, dst.open(mode) as f:
        while chunk := r.read(CHUNK):
            f.write(chunk)
            done += len(chunk)
            print(f"\r{dst.name}: {done / 1e6:.0f}/{size / 1e6:.0f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    if dst.stat().st_size != size:
        raise RuntimeError(f"{dst.name}: size mismatch {dst.stat().st_size} != {size}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "male-cns:v1.0",
        "source": BASE,
        "license": "CC-BY",
        "fetched": date.today().isoformat(),
        "files": {},
    }
    for key, name in FILES.items():
        dst = out / name
        download(BASE + name, dst)
        manifest["files"][key] = {"name": name, "bytes": dst.stat().st_size, "sha256": sha256(dst)}
        print(f"{key:18s} {manifest['files'][key]['bytes']:>12d}  {manifest['files'][key]['sha256'][:16]}…")

    (out.parent / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out.parent / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
