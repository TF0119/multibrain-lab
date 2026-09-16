"""M0 body-side measurements for PLAN.md §3.8.

CPU (MuJoCo): uncontrolled fall from standing; all actuators at ctrl=±1.

GPU (mujoco_warp), two separate measurements:
  - contacts: synced loop, nworld=64, standing + supine + prone + side starts,
    max per-world contact count (d.contact.worldid bincount over d.nacon).
  - throughput: 200 body steps (x4 physics steps) captured as a CUDA graph via
    wp.ScopedCapture and replayed with wp.capture_launch; ctrl updated each
    body step by a torch uniform_ on the GPU tensor that d.ctrl wraps
    (wp.from_torch, no host round-trip). Sync only around the timed region.
    nworld = 64, 128, 256, 512, 1024.

GPU memory per run: torch.cuda.mem_get_info() free-mem delta before put_data
vs after warmup, plus wp.get_mempool_used_mem_current delta.

Writes data/body_check.json.
"""

import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "humanoid.xml"
OUT_PATH = REPO_ROOT / "data" / "body_check.json"

SUBSTEPS = 4          # physics steps per 20 ms body step (5 ms timestep)
BODY_STEPS = 200
WARMUP_STEPS = 20


def lowest_geom_z(model, d):
    lo = math.inf
    for g in range(1, model.ngeom):
        ext = np.abs(d.geom_xmat[g].reshape(3, 3)) @ model.geom_aabb[g][3:]
        lo = min(lo, d.geom_xpos[g, 2] - ext[2])
    return lo


def cpu_checks(model):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    z0 = float(d.qpos[2])
    lo = math.inf
    for _ in range(400):
        mujoco.mj_step(model, d)
        lo = min(lo, lowest_geom_z(model, d))
    lo_final = lowest_geom_z(model, d)
    fall = {
        "pelvis_z_init": z0,
        "pelvis_z_final": float(d.qpos[2]),
        "fell_below_70pct": bool(d.qpos[2] < 0.7 * z0),
        # transient deepest point is recorded only; the pass criterion is the
        # final state (no geom resting below -0.02 m)
        "min_geom_z_transient": float(lo),
        "min_geom_z_final": float(lo_final),
        "penetration_ok": bool(lo_final > -0.02),
        "nan": bool(np.isnan(d.qpos).any()),
    }

    # drive every actuator to each end of ctrl range; check for NaN and
    # that hinge angles stay inside their ranges
    extremes = {"nan": False, "range_violations": []}
    for sign in (1.0, -1.0):
        d2 = mujoco.MjData(model)
        mujoco.mj_forward(model, d2)
        d2.ctrl[:] = sign
        for _ in range(200):
            mujoco.mj_step(model, d2)
        if np.isnan(d2.qpos).any() or np.isnan(d2.qvel).any():
            extremes["nan"] = True
            break
        for j in range(model.njnt):
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            lo_r, hi_r = model.jnt_range[j]
            q = d2.qpos[model.jnt_qposadr[j]]
            if q < lo_r - 1e-3 or q > hi_r + 1e-3:
                extremes["range_violations"].append(
                    {"joint": model.joint(j).name, "q": float(q),
                     "range": [float(lo_r), float(hi_r)], "ctrl": sign})
    return {"fall": fall, "ctrl_extremes": extremes}


# pose qpos[3:7] quaternions (w, x, y, z) for the pelvis freejoint, and z heights
POSES = {
    "standing": ([1.0, 0.0, 0.0, 0.0], 0.84),
    "supine":   ([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0], 0.20),
    "prone":    ([math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0], 0.20),
    "side":     ([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0], 0.20),
}


def _cuda_free():
    import torch
    return torch.cuda.mem_get_info()[0]


def _mempool_used():
    import warp as wp
    return wp.get_mempool_used_mem_current(wp.get_device("cuda:0"))


def contact_run(mjm, mjd0, nworld):
    """Synced loop measuring per-world contact counts. Not timed."""
    import warp as wp
    import mujoco_warp

    rng = np.random.default_rng(0)
    m = mujoco_warp.put_model(mjm)
    d = mujoco_warp.put_data(mjm, mjd0, nworld=nworld)

    def max_world_ncon():
        nacon = int(d.nacon.numpy()[0])
        if nacon == 0:
            return 0
        counts = np.bincount(
            d.contact.worldid.numpy()[:nacon], minlength=nworld)
        return int(counts.max())

    max_ncon = 0
    for _ in range(BODY_STEPS):
        d.ctrl = wp.array(rng.uniform(-1, 1, (nworld, mjm.nu)).astype(np.float32))
        for _ in range(SUBSTEPS):
            mujoco_warp.step(m, d)
        max_ncon = max(max_ncon, max_world_ncon())
    wp.synchronize()
    return {
        "nworld": nworld,
        "max_ncon_per_world": max_ncon,
        "naconmax_allocated": int(d.naconmax),
        "nan_in_qpos": bool(np.isnan(d.qpos.numpy()).any()),
    }


def throughput_run(mjm, mjd0, nworld):
    """Unsynced, CUDA-graph-captured timing of 200 body steps."""
    import warp as wp
    import mujoco_warp
    import torch

    free0 = _cuda_free()
    pool0 = _mempool_used()
    m = mujoco_warp.put_model(mjm)
    d = mujoco_warp.put_data(mjm, mjd0, nworld=nworld)

    # d.ctrl aliases this GPU tensor; uniform_() updates it in place each step.
    # The torch op is enqueued on an ExternalStream wrapping warp's current
    # stream, so ctrl updates and physics/graph launches stay ordered on one
    # CUDA stream without host synchronization.
    ctrl_t = torch.zeros((nworld, mjm.nu), device="cuda")
    d.ctrl = wp.from_torch(ctrl_t)
    tstream = torch.cuda.ExternalStream(wp.get_stream().cuda_stream)

    with torch.cuda.stream(tstream):
        for _ in range(WARMUP_STEPS):
            ctrl_t.uniform_(-1.0, 1.0)
            for _ in range(SUBSTEPS):
                mujoco_warp.step(m, d)
        wp.synchronize()

        free1 = _cuda_free()
        pool1 = _mempool_used()
        mem = {
            "cuda_alloc_bytes": int(free0 - free1),
            "mempool_used_delta_bytes": int(pool1 - pool0),
        }

        result = {"nworld": nworld, **mem}
        try:
            with wp.ScopedCapture() as capture:
                for _ in range(SUBSTEPS):
                    mujoco_warp.step(m, d)
            graph = capture.graph

            t0 = time.perf_counter()
            for _ in range(BODY_STEPS):
                ctrl_t.uniform_(-1.0, 1.0)
                wp.capture_launch(graph)
            wp.synchronize()
            dt = time.perf_counter() - t0
        except Exception as e:
            # fall back to uncaptured stepping, flagged in the output
            result["graph_error"] = f"{type(e).__name__}: {e}"
            result["no_graph"] = True
            t0 = time.perf_counter()
            for _ in range(BODY_STEPS):
                ctrl_t.uniform_(-1.0, 1.0)
                for _ in range(SUBSTEPS):
                    mujoco_warp.step(m, d)
            wp.synchronize()
            dt = time.perf_counter() - t0

    result["elapsed_s"] = dt
    result["body_steps_per_s"] = nworld * BODY_STEPS / dt
    result["ms_per_body_step"] = dt / BODY_STEPS * 1000.0
    result["nan_in_qpos"] = bool(np.isnan(d.qpos.numpy()).any())
    return result


def make_mjd(mjm, pose):
    quat, z = POSES[pose]
    mjd = mujoco.MjData(mjm)
    mjd.qpos[2] = z
    mjd.qpos[3:7] = quat
    mujoco.mj_forward(mjm, mjd)
    return mjd


def main():
    xml = XML_PATH.read_text()
    mjm = mujoco.MjModel.from_xml_string(xml)
    results = {"xml": str(XML_PATH), "cpu": cpu_checks(mjm),
               "warp": {"contacts": {}, "throughput": {}}}

    try:
        import warp as wp
        wp.init()
        free_start = _cuda_free()

        # contact counts: nworld=64, all four start poses (synced, untimed)
        for pose in POSES:
            try:
                results["warp"]["contacts"][pose] = contact_run(
                    mjm, make_mjd(mjm, pose), 64)
            except Exception as e:
                results["warp"]["contacts"][pose] = {"error": str(e)}

        # throughput: standing start, CUDA-graph captured
        for n in (64, 128, 256, 512, 1024):
            try:
                results["warp"]["throughput"][f"n{n}"] = throughput_run(
                    mjm, make_mjd(mjm, "standing"), n)
            except Exception as e:
                results["warp"]["throughput"][f"n{n}"] = {
                    "error": f"{type(e).__name__}: {e}"}

        # warp uses its own CUDA mempool; the free-mem drop across the whole
        # section approximates cumulative peak device allocation
        results["warp"]["cuda_free_drop_bytes"] = int(free_start - _cuda_free())
    except Exception as e:
        results["warp"]["init_error"] = f"{type(e).__name__}: {e}"

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
