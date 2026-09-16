// MultiBrain Lab viewer (PLAN.md §11.1)
// WebSocket で届く qpos 列（根 7 + 27 ヒンジ）を FK して
// VRM モデルか簡易人形にリアルタイムで反映する。
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';
import { decodeFrame, fk } from './protocol.js';
import { MAGIC_ACTIVITY, decodeActivityFrame, ActivityHud } from './activity.js';

// 接続先は ?ws=ws://host:port か ?port=8975 で上書きできる（既定 8765）。
const qs = new URLSearchParams(location.search);
const WS_URL = qs.get('ws') ??
  `ws://${location.hostname}:${qs.get('port') ?? 8765}`;

// ---------- 簡易人形（VRM が無い場合の常用表示） ----------
const SIDE_COLOR = { center: 0x9aa4ad, L: 0xe0a458, R: 0x58a7e0 };
function bodyColor(name) {
  if (name.endsWith('_L')) return SIDE_COLOR.L;
  if (name.endsWith('_R')) return SIDE_COLOR.R;
  return SIDE_COLOR.center;
}

function geomMesh(g, color) {
  const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.8 });
  let mesh;
  if (g.type === 'capsule' || g.type === 'cylinder') {
    const a = new THREE.Vector3(...g.a), b = new THREE.Vector3(...g.b);
    const len = a.distanceTo(b);
    mesh = new THREE.Mesh(new THREE.CapsuleGeometry(g.r, len, 4, 10), mat);
    mesh.position.copy(a).add(b).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0), b.sub(a).normalize());
  } else if (g.type === 'box') {
    mesh = new THREE.Mesh(new THREE.BoxGeometry(
      g.half[0] * 2, g.half[1] * 2, g.half[2] * 2), mat);
    mesh.position.set(...g.pos);
    if (g.quat) mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  } else if (g.type === 'sphere') {
    mesh = new THREE.Mesh(new THREE.SphereGeometry(g.r, 16, 12), mat);
    mesh.position.set(...g.pos);
  }
  return mesh;
}

class StickFigure {
  constructor(meta) {
    this.group = new THREE.Group();
    this.bodies = new Map();
    for (const b of meta.bodies) {
      const g = new THREE.Group();
      for (const geo of meta.geoms[b.name] ?? []) {
        const m = geomMesh(geo, bodyColor(b.name));
        if (m) g.add(m);
      }
      this.group.add(g);
      this.bodies.set(b.name, g);
    }
  }
  apply(poses) {
    for (const [name, g] of this.bodies) {
      const p = poses.get(name);
      if (!p) continue;
      g.position.copy(p.pos);
      g.quaternion.copy(p.quat);
    }
  }
}

// ---------- VRM 写像（PLAN §11.1 のオフセット方式） ----------
// 各骨について、rest 姿勢で MuJoCo 体節方向へ骨方向を最短弧で合わせる
// オフセット K を読込時に計算し、実行時は B = W_body * K で世界姿勢を決める。
const DIR_CHILD = {
  hips: 'spine', chest: 'neck',
  leftUpperArm: 'leftLowerArm', leftLowerArm: 'leftHand',
  leftUpperLeg: 'leftLowerLeg', leftLowerLeg: 'leftFoot', leftFoot: 'leftToes',
  rightUpperArm: 'rightLowerArm', rightLowerArm: 'rightHand',
  rightUpperLeg: 'rightLowerLeg', rightLowerLeg: 'rightFoot', rightFoot: 'rightToes',
};
const BONE_ORDER = [
  'hips', 'spine', 'chest', 'upperChest', 'neck', 'head',
  'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
  'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand',
  'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes',
  'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes',
];

class VrmDriver {
  constructor(vrm, meta) {
    this.vrm = vrm;
    this.meta = meta;
    const humanoid = vrm.humanoid;
    const bone = (name) =>
      (humanoid.getRawBoneNode?.(name) ??
        humanoid.getBoneNode?.(name) ??
        humanoid.getNormalizedBoneNode?.(name) ?? null);

    // モデルの向きを +Z にそろえる（VRM0 系は -Z 向きのことがある）。
    // 足->つま先の水平方向から向きを測り、補正してから rest を採る。
    const footL = bone('leftFoot'), toesL = bone('leftToes');
    if (footL && toesL) {
      const fwd = toesL.getWorldPosition(new THREE.Vector3())
        .sub(footL.getWorldPosition(new THREE.Vector3()));
      fwd.y = 0;
      if (fwd.lengthSq() > 1e-6) {
        const yaw = Math.atan2(fwd.x, fwd.z);
        vrm.scene.rotation.y -= yaw;
      }
    }
    vrm.scene.updateMatrixWorld(true);

    const rest = fk(meta, meta.rest_qpos);
    const parentOf = new Map(meta.bodies.map((b) => [b.name, b.parent]));

    // pass 1: 体節方向のある骨の最短弧アライン R[body]
    const R = new Map();
    for (const [body, boneName] of Object.entries(meta.vrm_map)) {
      const s = meta.segment_dir[body];
      const childName = DIR_CHILD[boneName];
      const node = bone(boneName);
      const child = childName && bone(childName);
      if (!s || !node || !child) continue;
      const v = child.getWorldPosition(new THREE.Vector3())
        .sub(node.getWorldPosition(new THREE.Vector3()));
      if (v.lengthSq() < 1e-8) continue;
      const d0 = new THREE.Vector3(...s).applyQuaternion(rest.get(body).quat);
      R.set(body, new THREE.Quaternion().setFromUnitVectors(v.normalize(), d0));
    }
    // pass 2: 方向を持たない体節（head, hand）は最近の祖先の R を継ぐ
    const alignOf = (body) => {
      let b = body;
      while (b && !R.has(b)) b = parentOf.get(b);
      return R.get(b) ?? new THREE.Quaternion();
    };

    // 処理順に骨を集め、K と rest ローカルを記録
    this.bones = [];
    const bodyOfBone = new Map(
      Object.entries(meta.vrm_map).map(([b, n]) => [n, b]));
    for (const name of BONE_ORDER) {
      const node = bone(name);
      if (!node) continue;
      const body = bodyOfBone.get(name) ?? null;
      const B0 = node.getWorldQuaternion(new THREE.Quaternion());
      let K = null;
      if (body && rest.get(body)) {
        const Brest = alignOf(body).clone().multiply(B0);
        K = rest.get(body).quat.clone().invert().multiply(Brest);
      }
      this.bones.push({
        node, name, body, K,
        restLocal: node.quaternion.clone(),
      });
    }

    // hips 位置の追随用
    this.hips = bone('hips');
    this.hipsRestLocalPos = this.hips.position.clone();
    this.hipsParentInv = this.hips.parent
      .getWorldQuaternion(new THREE.Quaternion()).invert();
    this.pelvisRest = rest.get('pelvis').pos.clone();
    const hipsW = this.hips.getWorldPosition(new THREE.Vector3());
    this.scale = this.pelvisRest.y > 1e-6 ? hipsW.y / this.pelvisRest.y : 1.0;

    this.memo = new Map();
    this._pw = new THREE.Quaternion();
    this._desired = new THREE.Quaternion();
    this._v = new THREE.Vector3();
  }

  // ノードの親の世界姿勢（処理済み骨は memo の値、中間ノードはローカル積算）
  parentQuat(node) {
    const p = node.parent;
    if (!p) return new THREE.Quaternion();
    if (this.memo.has(p)) return this.memo.get(p);
    return this.parentQuat(p).clone().multiply(p.quaternion);
  }

  apply(poses) {
    this.memo.clear();
    for (const e of this.bones) {
      const pw = this.parentQuat(e.node);
      const desired = this._desired;
      if (e.body && e.K) {
        desired.copy(poses.get(e.body).quat).multiply(e.K);
      } else {
        desired.copy(pw).multiply(e.restLocal);
      }
      e.node.quaternion.copy(pw).invert().multiply(desired);
      this.memo.set(e.node, desired.clone());
    }
    // hips の位置 = rest + (骨盤位置 - rest 骨盤位置) * 身長比
    const pelvis = poses.get('pelvis');
    if (pelvis && this.hips) {
      const off = this._v.copy(pelvis.pos).sub(this.pelvisRest)
        .multiplyScalar(this.scale).applyQuaternion(this.hipsParentInv);
      this.hips.position.copy(this.hipsRestLocalPos).add(off);
    }
  }
}

// ---------- 画面 ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101418);
scene.fog = new THREE.Fog(0x101418, 8, 24);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.05, 60);
camera.position.set(1.8, 1.4, 2.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.8, 0);

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x20242a, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(2.5, 4, 1.5);
scene.add(sun);
scene.add(new THREE.GridHelper(8, 16, 0x33404c, 0x222b33));

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

const statusEl = document.getElementById('status');
const infoEl = document.getElementById('info');
const logEl = document.getElementById('log');
const log = (msg) => {
  const line = document.createElement('div');
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.prepend(line);
  while (logEl.children.length > 40) logEl.lastChild.remove();
};

// ---------- 状態と配線 ----------
let meta = null;
let stick = null;
let vrmDriver = null;
let latest = null;      // {t, seq, qpos}
let viewMode = 'both';

function setView(mode) {
  viewMode = mode;
  for (const btn of document.querySelectorAll('#controls button[data-view]')) {
    btn.classList.toggle('on', btn.dataset.view === mode);
  }
  const showVrm = mode !== 'stick' && vrmDriver;
  const showStick = mode !== 'vrm' || !vrmDriver;
  if (vrmDriver) vrmDriver.vrm.scene.visible = showVrm;
  if (stick) stick.group.visible = showStick;
}
document.querySelectorAll('#controls button[data-view]').forEach((btn) => {
  btn.addEventListener('click', () => setView(btn.dataset.view));
});

// 活動 HUD（PLAN §11.2）。meta.activity が無い条件では隠したまま。
let activityHud = null;
let activityOn = true;
const actBtn = document.querySelector('#controls button[data-activity]');
actBtn?.addEventListener('click', () => {
  activityOn = !activityOn;
  actBtn.classList.toggle('on', activityOn);
  activityHud?.setVisible(activityOn);
});

function loadVrm() {
  const loader = new GLTFLoader();
  loader.register((p) => new VRMLoaderPlugin(p));
  loader.load('/avatar.vrm', (gltf) => {
    const vrm = gltf.userData.vrm;
    scene.add(vrm.scene);
    vrmDriver = new VrmDriver(vrm, meta);
    setView(viewMode);
    log(`VRM 読込: ${vrm.meta?.name ?? 'avatar.vrm'}`);
  }, undefined, (err) => {
    log(`avatar.vrm を読めません（簡易人形で表示）: ${err?.message ?? err}`);
  });
}

function onMeta(msg) {
  meta = msg;
  stick = new StickFigure(meta);
  scene.add(stick.group);
  loadVrm();
  setView('both');
  const actEl = document.getElementById('activity');
  if (meta.activity) {
    actEl.replaceChildren();
    activityHud = new ActivityHud(actEl, meta.activity);
    activityHud.setVisible(activityOn);
  } else {
    activityHud = null;
    actEl.style.display = 'none';
  }
  log(`meta: ${meta.bodies.length} bodies, ${meta.joints.length} joints` +
      `${meta.condition ? ` (${meta.condition})` : ''}`);
}

let backoff = 500;
function connect() {
  const ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    backoff = 500;
    statusEl.textContent = `接続中 ${WS_URL}`;
    statusEl.className = 'ok';
  };
  ws.onmessage = (ev) => {
    try {
      if (typeof ev.data === 'string') {
        onMeta(JSON.parse(ev.data));
      } else {
        // 先頭 4 バイトで MBP1（姿勢）と MBA1（活動）を振り分ける。
        const dv = new DataView(ev.data);
        const magic = String.fromCharCode(
          dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
        if (magic === MAGIC_ACTIVITY) {
          activityHud?.push(decodeActivityFrame(ev.data));
        } else {
          latest = decodeFrame(ev.data);
        }
      }
    } catch (e) {
      log(`frame error: ${e.message}`);
    }
  };
  ws.onclose = () => {
    statusEl.textContent = `切断 — ${(backoff / 1000).toFixed(1)}s 後に再接続`;
    statusEl.className = 'err';
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 8000);
  };
  ws.onerror = () => ws.close();
}
connect();

let frames = 0, fpsT = performance.now(), fps = 0;
function tick() {
  requestAnimationFrame(tick);
  if (latest && meta) {
    const poses = fk(meta, latest.qpos);
    stick?.apply(poses);
    vrmDriver?.apply(poses);
    frames++;
    const now = performance.now();
    if (now - fpsT > 500) {
      fps = Math.round(frames * 1000 / (now - fpsT));
      frames = 0; fpsT = now;
    }
    infoEl.textContent =
      `t=${latest.t.toFixed(2)}s seq=${latest.seq} ${fps}fps`;
  }
  activityHud?.render();
  controls.update();
  renderer.render(scene, camera);
}
tick();
