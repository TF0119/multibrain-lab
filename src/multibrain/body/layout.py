"""Name-resolved index layout for the humanoid model (PLAN.md §3.5).

All sensordata indices are resolved from sensor names via
mj_name2id + sensor_adr/sensor_dim — positions are never hard-coded.
Joint order is the order of configs/joint_groups.yaml, which is also the
actuator (ctrl) order.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .build_xml import DEFAULT_JOINT_GROUPS, IMPACT_SITES, TOUCH_SITES


def _sensor_slice(mjm: mujoco.MjModel, name: str) -> slice:
    sid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise KeyError(f"no sensor named {name!r}")
    adr = int(mjm.sensor_adr[sid])
    return slice(adr, adr + int(mjm.sensor_dim[sid]))


@dataclass
class BodyLayout:
    joint_names: list[str]          # 27, joint_groups.yaml order = ctrl order
    qpos_adr: np.ndarray            # (27,) qpos address of each joint
    dof_adr: np.ndarray             # (27,) qvel address of each joint
    range_lo: np.ndarray            # (27,) rad
    range_hi: np.ndarray            # (27,) rad
    gear: np.ndarray                # (27,) actuator gear = torque limit (N·m)
    h_ref: float                    # reference standing pelvis height (0.84)
    nq: int
    nv: int
    nu: int
    nsensordata: int
    sensor: dict                    # every sensor name -> slice into sensordata
    jointpos_idx: np.ndarray        # (27,) in joint_names order
    jointvel_idx: np.ndarray        # (27,)
    framequat_idx: np.ndarray       # (4,) imu quat (w, x, y, z)
    gyro_idx: np.ndarray            # (3,)
    velocimeter_idx: np.ndarray     # (3,)
    framepos_idx: np.ndarray        # (3,)
    touch_names: list[str]          # 12, build_xml.TOUCH_SITES order
    touch_idx: np.ndarray           # (12,) sensordata index of each touch site
    foot_touch_idx: np.ndarray      # (2,) sensordata indices of the feet
    nonfoot_touch_idx: np.ndarray   # (10,) the remaining touch sites
    torso_zaxis_idx: np.ndarray     # (3,) torso z axis in world (§5.2 tilt)
    pelvis_linvel_idx: np.ndarray   # (3,) pelvis world linvel (§5.2)
    impact_site_names: list[str]    # IMPACT_SITES order
    impact_linvel_idx: np.ndarray   # (4, 3) linvel of each impact site
    impact_touch_idx: np.ndarray    # (4,) touch force index of each impact site
    head_xaxis_idx: np.ndarray      # (3,) head x axis in world (§5.3 face)

    @classmethod
    def from_model(cls, mjm: mujoco.MjModel,
                   joint_groups_path: Path = DEFAULT_JOINT_GROUPS
                   ) -> "BodyLayout":
        with open(joint_groups_path) as f:
            joints = yaml.safe_load(f)["joints"]
        joint_names = [j["name"] for j in joints]

        jids = np.array([mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, n)
                         for n in joint_names])
        assert (jids >= 0).all(), "joint_groups.yaml names missing from model"
        aids = np.array([mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                         for n in joint_names])
        assert (aids >= 0).all(), "every joint needs a same-named actuator"

        # values come from the model; assert they agree with the yaml source
        qpos_adr = mjm.jnt_qposadr[jids].astype(np.int64)
        dof_adr = mjm.jnt_dofadr[jids].astype(np.int64)
        range_lo = mjm.jnt_range[jids, 0].copy()
        range_hi = mjm.jnt_range[jids, 1].copy()
        gear = mjm.actuator_gear[aids, 0].copy()
        np.testing.assert_allclose(
            range_lo, np.deg2rad([j["range_deg"][0] for j in joints]),
            rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            range_hi, np.deg2rad([j["range_deg"][1] for j in joints]),
            rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            gear, [j["torque"] for j in joints], rtol=0, atol=1e-12)

        sensor = {}
        for i in range(mjm.nsensor):
            name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_SENSOR, i)
            sensor[name] = slice(int(mjm.sensor_adr[i]),
                                 int(mjm.sensor_adr[i] + mjm.sensor_dim[i]))

        def idx(name):
            sl = sensor[name]
            return np.arange(sl.start, sl.stop, dtype=np.int64)

        touch_names = [name for name, _, _, _ in TOUCH_SITES]
        jointpos_idx = np.array([sensor[f"jointpos_{n}"].start
                                 for n in joint_names], dtype=np.int64)
        jointvel_idx = np.array([sensor[f"jointvel_{n}"].start
                                 for n in joint_names], dtype=np.int64)
        touch_idx = np.array([sensor[n].start for n in touch_names],
                             dtype=np.int64)
        is_foot = np.array([n.startswith("touch_foot") for n in touch_names])

        # declaration order in the model must match the order the index
        # arrays assume (yaml order for joints, TOUCH_SITES for touch)
        def declared_names(sensortype):
            return [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_SENSOR, i)
                    for i in range(mjm.nsensor)
                    if mjm.sensor_type[i] == sensortype]
        st = mujoco.mjtSensor
        assert declared_names(st.mjSENS_JOINTPOS) == \
            [f"jointpos_{n}" for n in joint_names]
        assert declared_names(st.mjSENS_JOINTVEL) == \
            [f"jointvel_{n}" for n in joint_names]
        assert declared_names(st.mjSENS_TOUCH) == touch_names
        assert declared_names(st.mjSENS_FRAMELINVEL) == \
            ["framelinvel_imu"] + [f"framelinvel_{s}" for s in IMPACT_SITES]

        # every index array must equal what name lookup yields
        for i, n in enumerate(joint_names):
            assert sensor[f"jointpos_{n}"] == slice(jointpos_idx[i],
                                                  jointpos_idx[i] + 1)
            assert sensor[f"jointvel_{n}"] == slice(jointvel_idx[i],
                                                  jointvel_idx[i] + 1)
        for i, n in enumerate(touch_names):
            assert sensor[n] == slice(touch_idx[i], touch_idx[i] + 1)

        impact_linvel_idx = np.stack(
            [idx(f"framelinvel_{s}") for s in IMPACT_SITES])
        return cls(
            joint_names=joint_names,
            qpos_adr=qpos_adr,
            dof_adr=dof_adr,
            range_lo=range_lo,
            range_hi=range_hi,
            gear=gear,
            h_ref=float(mjm.qpos0[2]),
            nq=int(mjm.nq),
            nv=int(mjm.nv),
            nu=int(mjm.nu),
            nsensordata=int(mjm.nsensordata),
            sensor=sensor,
            jointpos_idx=jointpos_idx,
            jointvel_idx=jointvel_idx,
            framequat_idx=idx("framequat_imu"),
            gyro_idx=idx("gyro_imu"),
            velocimeter_idx=idx("velocimeter_imu"),
            framepos_idx=idx("framepos_imu"),
            touch_names=touch_names,
            touch_idx=touch_idx,
            foot_touch_idx=touch_idx[is_foot],
            nonfoot_touch_idx=touch_idx[~is_foot],
            torso_zaxis_idx=idx("framezaxis_torso"),
            pelvis_linvel_idx=idx("framelinvel_imu"),
            impact_site_names=list(IMPACT_SITES),
            impact_linvel_idx=impact_linvel_idx,
            impact_touch_idx=np.array([sensor[s].start for s in IMPACT_SITES],
                                      dtype=np.int64),
            head_xaxis_idx=idx("framexaxis_touch_head"),
        )

    def to(self, device) -> "BodyLayout":
        """Copy with index/range/gear arrays as torch tensors on `device`."""
        import torch
        kw = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                dtype = (torch.long if np.issubdtype(v.dtype, np.integer)
                         else torch.float32)
                kw[f.name] = torch.as_tensor(v).to(device=device, dtype=dtype)
        return dataclasses.replace(self, **kw)
