"""PLAN §4.3（入力ポート）と §4.4（出力ポート）の規則を MaleCNS の注釈に適用し、data/ports.json を書く。

使い方: python scripts/build_ports.py [--neurons data/neurons.parquet] [--out data/ports.json]
選択規則はこのファイルにだけ書く。どの規則と代用が使われたかは ports.json に残す。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
SIDES = ("L", "R")
LIMB = {"arm": "T2", "leg": "T3"}                      # 人型の四肢 → ハエの体節
NERVE = {"T1": "ProLN", "T2": "MesoLN", "T3": "MetaLN"}  # 脚神経
PROXIMAL = {"arm": "shoulder", "leg": "hip"}
DISTAL = {"arm": ("elbow", "wrist"), "leg": ("knee", "ankle")}
CONTACT_SITES = ["foot_L", "foot_R", "hand_L", "hand_R", "knee_L", "knee_R", "elbow_L", "elbow_R",
                 "pelvis", "chest", "back", "head"]
MIN_GROUP = 8
SENSORY_SUPERCLASS = ("vnc_sensory", "cb_sensory", "sensory_ascending", "sensory_descending")

# 関節の型 → (主動筋の type 注釈, 拮抗筋の type 注釈)
MOTOR_TEMPLATE = {
    "prox_x": (["Tergopleural/Pleural promotor MN", "Tr flexor MN"], ["Pleural remotor/abductor MN", "Tr extensor MN"]),
    "prox_y": (["Sternal adductor MN"], ["Pleural remotor/abductor MN"]),
    "prox_z": (["Sternal anterior rotator MN"], ["Sternal posterior rotator MN"]),
    "mid": (["Ti flexor MN", "Acc. ti flexor MN"], ["Ti extensor MN"]),
    "dist_x": (["Ta depressor MN"], ["Ta levator MN"]),
    "dist_y": (["ltm MN", "ltm1-tibia MN", "ltm2-femur MN"], ["ltm MN", "ltm1-tibia MN", "ltm2-femur MN"]),
}


def joint_names() -> list[str]:
    names = ["waist_z", "waist_y", "waist_x", "neck_y", "neck_z"]
    for s in SIDES:
        names += [f"shoulder_{s}_x", f"shoulder_{s}_y", f"shoulder_{s}_z", f"elbow_{s}", f"wrist_{s}"]
    for s in SIDES:
        names += [f"hip_{s}_x", f"hip_{s}_y", f"hip_{s}_z", f"knee_{s}", f"ankle_{s}_x", f"ankle_{s}_y"]
    return names


class Selector:
    def __init__(self, neurons: pd.DataFrame):
        self.n = neurons
        self.sens = neurons[neurons["superclass"].isin(SENSORY_SUPERCLASS)].copy()
        self.sens["side"] = self.sens["rootSide"].fillna(self.sens["somaSide"])
        self.motor = neurons[neurons["superclass"].isin(["vnc_motor", "cb_motor"])].copy()
        self.rng = np.random.default_rng(SEED)
        self.used_inputs: set[int] = set()

    # ---- 感覚 ----
    def sensory(self, **cond) -> pd.DataFrame:
        df = self.sens
        for k, v in cond.items():
            df = df[df[k].isin(v)] if isinstance(v, (list, tuple, set)) else df[df[k] == v]
        return df

    def split(self, df: pd.DataFrame, k: int) -> list[list[int]]:
        ids = df["bodyId"].to_numpy().copy()
        self.rng.shuffle(ids)
        return [sorted(int(x) for x in part) for part in np.array_split(ids, k)]

    # ---- 運動 ----
    def motor_pool(self, neuromere: str, side: str) -> pd.DataFrame:
        m = self.motor
        return m[(m["somaNeuromere"] == neuromere) & (m["somaSide"] == side)]

    def motor_types(self, pool: pd.DataFrame, types: list[str]) -> list[int]:
        return sorted(int(x) for x in pool[pool["type"].isin(types)]["bodyId"])


def build_inputs(sel: Selector) -> list[dict]:
    ports: list[dict] = []

    def add(signal: str, ids: list[int], selector: str, fallback: str | None = None):
        ports.append({"signal": signal, "n": len(ids), "selector": selector, "fallback": fallback, "neurons": ids})

    # 四肢の固有感覚：弦音器は遠位二関節、毛板と leg 注釈は近位三軸
    for limb, seg in LIMB.items():
        for side in SIDES:
            base = dict(**{"class": "mechanosensory_proprioceptive"}, entryNerve=NERVE[seg], side=side)
            cho = sel.sensory(**base, subclass="chordotonal organ")
            prox = sel.sensory(**base)
            prox = prox[~prox.index.isin(cho.index)]
            distal_signals = [f"{kind}:{j}_{side}{ax}" for j in DISTAL[limb]
                              for ax in (["_x"] if j in ("ankle",) else [""]) for kind in ("jointpos", "jointvel")]
            if limb == "leg":
                distal_signals += [f"jointpos:ankle_{side}_y", f"jointvel:ankle_{side}_y"]
            for sig, ids in zip(distal_signals, sel.split(cho, len(distal_signals))):
                add(sig, ids, f"proprioceptive chordotonal organ, {NERVE[seg]}, side {side}")
            prox_signals = [f"{kind}:{PROXIMAL[limb]}_{side}_{ax}" for ax in "xyz" for kind in ("jointpos", "jointvel")]
            for sig, ids in zip(prox_signals, sel.split(prox, len(prox_signals))):
                add(sig, ids, f"proprioceptive hair plate / leg (non-chordotonal), {NERVE[seg]}, side {side}")

    # 腰：腹部の固有感覚。首：前胸神経の毛板
    ab = sel.sensory(**{"class": "mechanosensory_proprioceptive"}, subclass="abdomen")
    for sig, ids in zip([f"{k}:waist_{ax}" for ax in "zyx" for k in ("jointpos", "jointvel")], sel.split(ab, 6)):
        add(sig, ids, "proprioceptive, subclass abdomen")
    neck = sel.sensory(subclass="hair plate", entryNerve="PrN")
    for sig, ids in zip([f"{k}:neck_{ax}" for ax in "yz" for k in ("jointpos", "jointvel")], sel.split(neck, 4)):
        add(sig, ids, "hair plate, PrN (neck)")

    # 重力、角速度、線速度と高さ
    grav = sel.sensory(subclass="wind_gravity")
    for i, ids in enumerate(sel.split(grav, 3)):
        add(f"gravity:{i}", ids, "Johnston's organ wind_gravity")
    halt = sel.sensory(entryNerve="DMetaN", subclass=["haltere", "campaniform sensilla"])
    for i, ids in enumerate(sel.split(halt, 3)):
        add(f"gyro:{i}", ids, "haltere campaniform sensilla, DMetaN")
    aud = sel.sensory(subclass="auditory")
    parts = sel.split(aud, 4)
    for i in range(3):
        add(f"velocimeter:{i}", parts[i], "Johnston's organ auditory", fallback="no biological counterpart")
    add("height", parts[3], "Johnston's organ auditory", fallback="no biological counterpart")

    # 接触：四肢は脚の触覚神経を遠位と近位に分ける。体幹は背板の剛毛、頭は口器の機械感覚
    for limb, seg in LIMB.items():
        for side in SIDES:
            tact = sel.sensory(**{"class": "mechanosensory_tactile"}, entryNerve=NERVE[seg], side=side)
            dist, prox = sel.split(tact, 2)
            d_name, p_name = ("hand", "elbow") if limb == "arm" else ("foot", "knee")
            add(f"touch:{d_name}_{side}", dist, f"tactile, {NERVE[seg]}, side {side} (distal half)")
            add(f"touch:{p_name}_{side}", prox, f"tactile, {NERVE[seg]}, side {side} (proximal half)")
    notum = sel.sensory(**{"class": "mechanosensory_tactile"}, subclass="notum")
    back, chest = sel.split(notum, 2)
    add("touch:back", back, "tactile notum (dorsal thorax)")
    add("touch:chest", chest, "tactile notum (dorsal thorax)", fallback="no ventral thorax bristle annotation; notum used")
    pelvis = sel.sensory(**{"class": "mechanosensory_tactile"}, entryNerve=["AbN1", "AbN2", "AbN3", "AbN4", "AbNT"])
    fb = None
    if len(pelvis) < MIN_GROUP:
        pelvis = sel.sensory(**{"class": "unknown_sensory"}, subclass="abdomen")
        fb = "no abdominal tactile annotation; unknown_sensory abdomen used"
    add("touch:pelvis", sorted(int(x) for x in pelvis["bodyId"]), "abdominal nerves", fallback=fb)
    head = sel.sensory(**{"class": "mechanosensory"}, entryNerve="MxLbN")
    add("touch:head", sorted(int(x) for x in head["bodyId"]), "mechanosensory MxLbN (mouthparts)",
        fallback="head bristles not annotated; mouthpart mechanosensory used")
    return ports


def build_outputs(sel: Selector) -> list[dict]:
    ports: list[dict] = []

    def add(joint: str, ag: list[int], ant: list[int], pool: list[int], rule: str, note: str):
        both = set(ag) & set(ant)
        w = {}
        for i in ag:
            w[i] = 0.0 if i in both else 1.0 / len(ag)
        for i in ant:
            w[i] = 0.0 if i in both else -1.0 / len(ant)
        for i in pool:
            w.setdefault(i, 0.0)
        ports.append({"joint": joint, "rule": rule, "note": note, "n_agonist": len(ag), "n_antagonist": len(ant),
                      "n_total": len(w), "neurons": sorted(w), "init_weight": [w[i] for i in sorted(w)]})

    for limb, seg in LIMB.items():
        prox = PROXIMAL[limb]
        mid, dist = DISTAL[limb]
        joints = {f"{prox}_{{s}}_x": "prox_x", f"{prox}_{{s}}_y": "prox_y", f"{prox}_{{s}}_z": "prox_z",
                  f"{mid}_{{s}}": "mid", f"{dist}_{{s}}" + ("_x" if limb == "leg" else ""): "dist_x"}
        if limb == "leg":
            joints[f"{dist}_{{s}}_y"] = "dist_y"
        for side in SIDES:
            pool_df = sel.motor_pool(seg, side)
            pool = sorted(int(x) for x in pool_df["bodyId"])
            for tmpl, key in joints.items():
                joint = tmpl.format(s=side)
                ag_t, ant_t = MOTOR_TEMPLATE[key]
                ag, ant = sel.motor_types(pool_df, ag_t), sel.motor_types(pool_df, ant_t)
                note = f"{seg} {side}: agonist {ag_t} -> {len(ag)}, antagonist {ant_t} -> {len(ant)}"
                if not ag and not ant:
                    add(joint, [], [], pool, "pool_only", note + "; no annotated group, whole leg pool with zero init")
                elif len(ag) < MIN_GROUP or len(ant) < MIN_GROUP:
                    add(joint, ag, ant, pool, "aggregated", note + f"; group < {MIN_GROUP}, whole leg pool added with zero init")
                else:
                    add(joint, ag, ant, [], "annotated", note)

    m = sel.motor
    abdominal = m[m["somaNeuromere"].fillna("").str.match(r"^A\d+$") & m["type"].fillna("").str.startswith("MNad")]
    ab_l = sorted(int(x) for x in abdominal[abdominal["somaSide"] == "L"]["bodyId"])
    ab_r = sorted(int(x) for x in abdominal[abdominal["somaSide"] == "R"]["bodyId"])
    add("waist_x", [], [], ab_l + ab_r, "pool_only", "abdominal MNad*: dorsal/ventral not annotated, zero init")
    add("waist_y", ab_l, ab_r, [], "annotated", "abdominal MNad*: left (+) vs right (−)")
    add("waist_z", ab_l, ab_r, [], "annotated", "abdominal MNad*: left (+) vs right (−)")
    neck = m[m["type"].fillna("").str.match(r"^CvN\d+$")]
    nk_l = sorted(int(x) for x in neck[neck["somaSide"] == "L"]["bodyId"])
    nk_r = sorted(int(x) for x in neck[neck["somaSide"] == "R"]["bodyId"])
    add("neck_y", [], [], nk_l + nk_r, "pool_only", "cervical nerve MNs (CvN*), zero init")
    add("neck_z", nk_l, nk_r, [], "annotated", "cervical nerve MNs: left (+) vs right (−)")
    return ports


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", default="data/neurons.parquet")
    ap.add_argument("--out", default="data/ports.json")
    args = ap.parse_args()

    neurons = pd.read_parquet(args.neurons)
    sel = Selector(neurons)
    inputs = build_inputs(sel)
    outputs = build_outputs(sel)

    joints = joint_names()
    out_joints = [p["joint"] for p in outputs]
    assert sorted(out_joints) == sorted(joints), set(joints) ^ set(out_joints)
    expected_inputs = {f"{k}:{j}" for j in joints for k in ("jointpos", "jointvel")}
    expected_inputs |= {f"gravity:{i}" for i in range(3)} | {f"gyro:{i}" for i in range(3)}
    expected_inputs |= {f"velocimeter:{i}" for i in range(3)} | {"height"} | {f"touch:{s}" for s in CONTACT_SITES}
    got = {p["signal"] for p in inputs}
    assert got == expected_inputs, got ^ expected_inputs
    assert len(inputs) == 76

    in_ids = [i for p in inputs for i in p["neurons"]]
    out_ids = [i for p in outputs for i in p["neurons"]]
    result = {
        "seed": SEED,
        "rules": {"arm": LIMB["arm"], "leg": LIMB["leg"], "min_group": MIN_GROUP,
                  "side_column": "rootSide, fallback somaSide (sensory); somaSide (motor)"},
        "summary": {
            "input_ports": len(inputs), "input_neurons": len(in_ids), "input_neurons_unique": len(set(in_ids)),
            "input_min_n": min(p["n"] for p in inputs), "input_fallbacks": sum(p["fallback"] is not None for p in inputs),
            "output_ports": len(outputs), "output_neurons_unique": len(set(out_ids)),
            "output_rules": pd.Series([p["rule"] for p in outputs]).value_counts().to_dict(),
            "input_output_overlap": len(set(in_ids) & set(out_ids)),
        },
        "joints": joints,
        "inputs": inputs,
        "outputs": outputs,
    }
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print("\ninputs (signal, n, fallback):")
    for p in inputs:
        print(f"  {p['signal']:26s} {p['n']:5d}  {p['fallback'] or ''}")
    print("\noutputs (joint, rule, agonist, antagonist, total):")
    for p in outputs:
        print(f"  {p['joint']:14s} {p['rule']:10s} {p['n_agonist']:3d} {p['n_antagonist']:3d} {p['n_total']:4d}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
