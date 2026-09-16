"""Generate assets/humanoid.xml from configs/joint_groups.yaml and PLAN.md §3.

Segment masses come from the §3.1 table (de Leva ratios, total 50 kg).
Lengths given in §3.1 are used verbatim (thigh 0.38, shin 0.37, upper arm 0.26,
forearm 0.24, head radius 0.09, foot box 0.22x0.09x0.04).
Lengths not given in the table are chosen so the figure fits a 1.58 m height:
  - pelvis box 0.28 x 0.18 x 0.18 (from §3.7 skeleton)
  - torso capsule length 0.42 (upper+mid trunk mass merged into one body, 15.0 kg)
  - waist joint sits 0.12 above the pelvis center
  - neck joint 0.48 above the torso origin, head center 0.06 above that
  Standing leg: hip drop 0.05 + thigh 0.38 + shin 0.37 + foot drop 0.04 = 0.84 m
  pelvis height. Head top = 0.84+0.12+0.48+0.06+0.09 = 1.59 m ~ 1.58 m.
Frame convention (per §3.7 skeleton): x = fore-aft (front is +x), y = lateral
(left is +y), z = up.
"""

import math
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JOINT_GROUPS = REPO_ROOT / "configs" / "joint_groups.yaml"
DEFAULT_OUT = REPO_ROOT / "assets" / "humanoid.xml"

def _axis_str(j):
    return " ".join(str(v) for v in j["axis"])

# Segment constants: (not all are in §3.1; see module docstring for provenance)
PELVIS_Z = 0.84          # hip drop + thigh + shin + foot drop
HIP_DROP = 0.05
HIP_LATERAL = 0.09
WAIST_Z = 0.12           # torso (waist joint) origin above pelvis center
TORSO_LEN = 0.42
TORSO_RADIUS = 0.13
NECK_Z = 0.48            # head (neck joint) origin above torso origin
HEAD_Z = 0.06            # head geom center above neck joint
HEAD_R = 0.09
SHOULDER_Y = 0.20        # lateral offset of shoulder joint from torso midline
SHOULDER_Z = 0.38        # shoulder height above torso origin
UPPER_ARM_LEN = 0.26
FOREARM_LEN = 0.24
THIGH_LEN = 0.38
SHIN_LEN = 0.37

# masses from §3.1 (kg); torso = upper trunk 7.7 + mid trunk 7.3
MASS = {
    "pelvis": 6.2, "torso": 15.0, "head": 3.3,
    "upper_arm": 1.3, "forearm": 0.7, "hand": 0.3,
    "thigh": 7.4, "shin": 2.4, "foot": 0.65,
}

# touch sites: name -> (body, pos, size) — 12 sites per §3.5
TOUCH_SITES = [
    ("touch_foot_L",  "foot_L",    "0.04 0 -0.02", "0.11 0.045 0.02"),
    ("touch_foot_R",  "foot_R",    "0.04 0 -0.02", "0.11 0.045 0.02"),
    ("touch_hand_L",  "hand_L",    "0 0 -0.045",   "0.04 0.045 0.045"),
    ("touch_hand_R",  "hand_R",    "0 0 -0.045",   "0.04 0.045 0.045"),
    ("touch_knee_L",  "shin_L",    "0.05 0 -0.02", "0.05 0.05 0.05"),
    ("touch_knee_R",  "shin_R",    "0.05 0 -0.02", "0.05 0.05 0.05"),
    ("touch_elbow_L", "forearm_L", "0.04 0 -0.01", "0.045 0.045 0.045"),
    ("touch_elbow_R", "forearm_R", "0.04 0 -0.01", "0.045 0.045 0.045"),
    ("touch_pelvis",  "pelvis",    "0 0 0",        "0.14 0.09 0.09"),
    ("touch_chest",   "torso",     "0.13 0 0.30",  "0.03 0.10 0.08"),
    ("touch_back",    "torso",     "-0.13 0 0.30", "0.03 0.10 0.08"),
    ("touch_head",    "head",      f"0 0 {HEAD_Z}", f"{HEAD_R} {HEAD_R} {HEAD_R}"),
]

# parent-child body pairs excluded from contact (§3.4)
EXCLUDE_PAIRS = [
    ("pelvis", "torso"), ("torso", "head"),
    ("torso", "upper_arm_L"), ("torso", "upper_arm_R"),
    ("upper_arm_L", "forearm_L"), ("upper_arm_R", "forearm_R"),
    ("forearm_L", "hand_L"), ("forearm_R", "hand_R"),
    ("pelvis", "thigh_L"), ("pelvis", "thigh_R"),
    ("thigh_L", "shin_L"), ("thigh_R", "shin_R"),
    ("shin_L", "foot_L"), ("shin_R", "foot_R"),
]

BODY_TOUCH_SITES = {}
for _name, _body, _pos, _size in TOUCH_SITES:
    BODY_TOUCH_SITES.setdefault(_body, []).append((_name, _pos, _size))


def _joints_for(joints, body):
    return [j for j in joints if j["parent_body"] == body]


def _joint_xml(j):
    lo, hi = j["range_deg"]
    return (f'      <joint name="{j["name"]}" axis="{_axis_str(j)}" '
            f'range="{lo} {hi}"/>\n')


def _sites_xml(body):
    out = ""
    for name, pos, size in BODY_TOUCH_SITES.get(body, []):
        out += f'      <site name="{name}" type="box" pos="{pos}" size="{size}"/>\n'
    return out


def _leg(joints, side):
    s = f'      <body name="thigh_{side}" pos="0 {HIP_LATERAL} -{HIP_DROP}">\n'
    s += "".join(_joint_xml(j) for j in _joints_for(joints, f"thigh_{side}"))
    s += (f'      <geom type="capsule" fromto="0 0 0 0 0 -{THIGH_LEN}" size="0.06" '
          f'mass="{MASS["thigh"]}"/>\n')
    s += f'      <body name="shin_{side}" pos="0 0 -{THIGH_LEN}">\n'
    s += "".join(_joint_xml(j) for j in _joints_for(joints, f"shin_{side}"))
    s += (f'      <geom type="capsule" fromto="0 0 0 0 0 -{SHIN_LEN}" size="0.045" '
          f'mass="{MASS["shin"]}"/>\n')
    s += _sites_xml(f"shin_{side}")
    s += f'      <body name="foot_{side}" pos="0 0 -{SHIN_LEN}">\n'
    s += "".join(_joint_xml(j) for j in _joints_for(joints, f"foot_{side}"))
    s += (f'      <geom type="box" pos="0.04 0 -0.02" size="0.11 0.045 0.02" '
          f'mass="{MASS["foot"]}"/>\n')
    s += _sites_xml(f"foot_{side}")
    s += "      </body>\n      </body>\n      </body>\n"
    return s


def _arm(joints, side):
    s = f'      <body name="upper_arm_{side}" pos="0 {SHOULDER_Y} {SHOULDER_Z}">\n'
    s += "".join(_joint_xml(j) for j in _joints_for(joints, f"upper_arm_{side}"))
    s += (f'      <geom type="capsule" fromto="0 0 0 0 0 -{UPPER_ARM_LEN}" size="0.045" '
          f'mass="{MASS["upper_arm"]}"/>\n')
    s += f'      <body name="forearm_{side}" pos="0 0 -{UPPER_ARM_LEN}">\n'
    s += "".join(_joint_xml(j) for j in _joints_for(joints, f"forearm_{side}"))
    s += (f'      <geom type="capsule" fromto="0 0 0 0 0 -{FOREARM_LEN}" size="0.04" '
          f'mass="{MASS["forearm"]}"/>\n')
    s += _sites_xml(f"forearm_{side}")
    s += f'      <body name="hand_{side}" pos="0 0 -{FOREARM_LEN}">\n'
    s += "".join(_joint_xml(j) for j in _joints_for(joints, f"hand_{side}"))
    s += (f'      <geom type="box" pos="0 0 -0.045" size="0.035 0.045 0.045" '
          f'mass="{MASS["hand"]}"/>\n')
    s += _sites_xml(f"hand_{side}")
    s += "      </body>\n      </body>\n      </body>\n"
    return s


def build_xml(joint_groups_path=DEFAULT_JOINT_GROUPS):
    with open(joint_groups_path) as f:
        joints = yaml.safe_load(f)["joints"]

    x = '<mujoco model="multibrain_humanoid">\n'
    # multiccd disabled: mujoco_warp rejects non-zero geom margin for box-box
    # pairs handled by MULTICCD; margin is kept for penetration control, so the
    # flag is turned off instead (per the warp error message's own suggestion).
    x += '  <option timestep="0.005"><flag multiccd="disable"/></option>\n'
    x += '  <default>\n'
    # solreflimit/solimplimit are not in §3.2 (which specifies only armature,
    # damping, frictionloss): with soft default limits, ctrl=±1 pushes joints
    # up to ~5 deg past their range; these keep overshoot under ~0.05 deg.
    x += ('    <joint type="hinge" armature="0.02" damping="2" frictionloss="0.2" '
          'limited="true" solreflimit="0.01 1" solimplimit="0.99 0.999 0.001"/>\n')
    # solimp/solref are not in §3.4 (which specifies only condim/friction):
    # with the plain defaults a 50 kg body at 5 ms steps penetrates the floor by
    # ~8 cm transiently while falling; the stiffer solimp keeps the resting
    # penetration within ~2 cm (§3.8 judges the final state). margin=0.01 on
    # the floor plane (below) only extends contact detection 1 cm — larger
    # margins were rejected as a design choice: the body would rest floating
    # and touch sensors would fire above the floor. The margin lives on the
    # floor rather than the default class because mujoco_warp rejects non-zero
    # margin on geoms that form BOX-BOX pairs (pelvis/hands/feet are boxes).
    x += ('    <geom condim="3" friction="1.0 0.005 0.0001" '
          'solimp="0.9 0.99 0.001" solref="0.01 1"/>\n')
    x += ('    <general dyntype="filter" dynprm="0.04" ctrlrange="-1 1" '
          'ctrllimited="true"/>\n')
    x += '  </default>\n'
    x += '  <worldbody>\n'
    x += '    <geom name="floor" type="plane" size="0 0 1" margin="0.01"/>\n'
    x += f'    <body name="pelvis" pos="0 0 {PELVIS_Z}">\n'
    x += '      <freejoint/>\n'
    x += f'      <geom type="box" size="0.14 0.09 0.09" mass="{MASS["pelvis"]}"/>\n'
    x += '      <site name="imu"/>\n'
    x += _sites_xml("pelvis")
    x += f'      <body name="torso" pos="0 0 {WAIST_Z}">\n'
    x += "".join(_joint_xml(j) for j in _joints_for(joints, "torso"))
    x += (f'      <geom type="capsule" fromto="0 0 0 0 0 {TORSO_LEN}" '
          f'size="{TORSO_RADIUS}" mass="{MASS["torso"]}"/>\n')
    x += _sites_xml("torso")
    x += f'      <body name="head" pos="0 0 {NECK_Z}">\n'
    x += "".join(_joint_xml(j) for j in _joints_for(joints, "head"))
    x += f'      <geom type="sphere" pos="0 0 {HEAD_Z}" size="{HEAD_R}" mass="{MASS["head"]}"/>\n'
    x += _sites_xml("head")
    x += "      </body>\n"
    x += _arm(joints, "L")
    x += _arm(joints, "R")
    x += "      </body>\n"
    x += _leg(joints, "L")
    x += _leg(joints, "R")
    x += "    </body>\n  </worldbody>\n"

    x += "  <contact>\n"
    for b1, b2 in EXCLUDE_PAIRS:
        x += f'    <exclude body1="{b1}" body2="{b2}"/>\n'
    x += "  </contact>\n"

    x += "  <actuator>\n"
    for j in joints:
        x += f'    <general name="{j["name"]}" joint="{j["name"]}" gear="{j["torque"]}"/>\n'
    x += "  </actuator>\n"

    x += "  <sensor>\n"
    for j in joints:
        x += f'    <jointpos joint="{j["name"]}"/>\n'
    for j in joints:
        x += f'    <jointvel joint="{j["name"]}"/>\n'
    x += '    <framequat objtype="site" objname="imu"/>\n'
    x += '    <gyro site="imu"/>\n'
    x += '    <velocimeter site="imu"/>\n'
    x += '    <framepos objtype="site" objname="imu"/>\n'
    for name, _, _, _ in TOUCH_SITES:
        x += f'    <touch site="{name}"/>\n'
    x += "  </sensor>\n</mujoco>\n"
    return x


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    xml = build_xml()
    out.write_text(xml)
    print(f"wrote {out} ({len(xml.splitlines())} lines)")


if __name__ == "__main__":
    main()
