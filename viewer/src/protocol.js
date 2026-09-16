// MBP1 プロトコルと FK（DOM 非依存。main.js と検証スクリプトの両方から使う）。
import * as THREE from 'three';

export const MAGIC = 'MBP1';

// MuJoCo 身体座標 (x 前, y 左, z 上) -> three.js (x 左, y 上, z 前)。
// 基底変換は (x,y,z)->(y,z,x) の循環置換 = (1,1,1) 軸まわり -120°。
// 表示用の剛体姿勢は Q_disp = RC * Q_body として持ち回る。
// これは「MuJoCo ローカル成分 -> three 世界」への写像なので、
// ローカル量（オフセット・関節軸・ジオム座標）は生の成分のまま使い、
// RC を掛けるのは根の変換だけで済む。
export const RC = new THREE.Quaternion(-0.5, -0.5, -0.5, 0.5);
export const RC_INV = RC.clone().invert();
export const mjPos = (v) => new THREE.Vector3(v[1], v[2], v[0]);
const rootQuat = (w, x, y, z) =>
  RC.clone().multiply(new THREE.Quaternion(x, y, z, w));

export function decodeFrame(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(
    dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== MAGIC) throw new Error(`bad magic ${magic}`);
  const t = dv.getFloat64(4, true);
  const seq = dv.getUint32(12, true);
  return { t, seq, qpos: new Float32Array(buf.slice(16)) };
}

// meta の木と軸から剛体姿勢を復元する（Q_disp = RC * Q_body 規約）。
export function fk(meta, qpos) {
  const poses = new Map();
  const info = new Map(meta.bodies.map((b) => [b.name, b]));
  const tmpV = new THREE.Vector3();
  function world(name) {
    const hit = poses.get(name);
    if (hit) return hit;
    const b = info.get(name);
    let pos, quat;
    if (b.free) {
      pos = mjPos([qpos[0], qpos[1], qpos[2]]);
      quat = rootQuat(qpos[3], qpos[4], qpos[5], qpos[6]);
    } else {
      const p = world(b.parent);
      pos = tmpV.set(b.pos[0], b.pos[1], b.pos[2])
        .applyQuaternion(p.quat).add(p.pos).clone();
      const lq = new THREE.Quaternion();
      for (const ji of b.joints) {
        const j = meta.joints[ji];
        lq.multiply(new THREE.Quaternion().setFromAxisAngle(
          tmpV.set(j.axis[0], j.axis[1], j.axis[2]).normalize(),
          qpos[j.qposadr]));
      }
      quat = p.quat.clone().multiply(lq);
    }
    const out = { pos, quat };
    poses.set(name, out);
    return out;
  }
  meta.bodies.forEach((b) => world(b.name));
  return poses;
}
