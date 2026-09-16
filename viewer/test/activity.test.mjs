// decodeActivityFrame の復号試験（node --test）。three や DOM に依存しない。
import test from 'node:test';
import assert from 'node:assert/strict';
import { MAGIC_ACTIVITY, decodeActivityFrame } from '../src/activity.js';

// 仕様どおりのバイト列を組み立てる:
// "MBA1" | f64 t (LE) | u32 seq | u16 K | u16 G | u16 M | u16 reserved
//        | f32[K][G][2] | u8[K][M]
function buildFrame({ t = 1.5, seq = 42, K = 3, G = 12, M = 512 } = {}) {
  const statsLen = K * G * 2;
  const buf = new ArrayBuffer(24 + statsLen * 4 + K * M);
  const dv = new DataView(buf);
  for (let i = 0; i < 4; i++) dv.setUint8(i, MAGIC_ACTIVITY.charCodeAt(i));
  dv.setFloat64(4, t, true);
  dv.setUint32(12, seq, true);
  dv.setUint16(16, K, true);
  dv.setUint16(18, G, true);
  dv.setUint16(20, M, true);
  dv.setUint16(22, 0, true);
  const stats = new Float32Array(statsLen);
  for (let i = 0; i < statsLen; i++) stats[i] = (i % 97) / 97;
  new Uint8Array(buf, 24, statsLen * 4).set(new Uint8Array(stats.buffer));
  const sample = new Uint8Array(buf, 24 + statsLen * 4, K * M);
  for (let i = 0; i < sample.length; i++) sample[i] = (i * 7 + 3) & 0xff;
  return { buf, stats, sample };
}

test('ヘッダと配列が仕様どおり復号される（K=3, G=12, M=512）', () => {
  const { buf, stats, sample } = buildFrame({ t: 12.25, seq: 987 });
  const f = decodeActivityFrame(buf);
  assert.equal(f.t, 12.25);
  assert.equal(f.seq, 987);
  assert.equal(f.K, 3);
  assert.equal(f.G, 12);
  assert.equal(f.M, 512);
  assert.ok(f.stats instanceof Float32Array);
  assert.ok(f.sample instanceof Uint8Array);
  assert.equal(f.stats.length, stats.length);
  assert.equal(f.sample.length, sample.length);
  for (let i = 0; i < stats.length; i++) assert.equal(f.stats[i], stats[i]);
  for (let i = 0; i < sample.length; i++) assert.equal(f.sample[i], sample[i]);
});

test('小さい K/G/M でも復号できる（K=1, G=2, M=3）', () => {
  const { buf, stats, sample } = buildFrame({ t: 0, seq: 0, K: 1, G: 2, M: 3 });
  const f = decodeActivityFrame(buf);
  assert.equal(f.K, 1);
  assert.equal(f.G, 2);
  assert.equal(f.M, 3);
  // stats[k][g][0]=平均, [k][g][1]=活動率 の順で読めること
  assert.equal(f.stats[0], stats[0]);
  assert.equal(f.stats[3], stats[3]);
  assert.equal(f.sample[2], sample[2]);
});

test('magic が MBA1 でないと throw する', () => {
  const { buf } = buildFrame();
  new Uint8Array(buf, 0, 4).set([0x4d, 0x42, 0x50, 0x31]); // "MBP1"
  assert.throws(() => decodeActivityFrame(buf), /bad magic/);
});

test('ヘッダ未満や中身の足りないフレームは throw する', () => {
  assert.throws(() => decodeActivityFrame(new ArrayBuffer(8)));
  const { buf } = buildFrame({ K: 3, G: 12, M: 512 });
  assert.throws(() => decodeActivityFrame(buf.slice(0, buf.byteLength - 1)));
});
