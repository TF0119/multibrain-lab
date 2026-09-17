// MBA1 活動フレームの復号と活動 HUD（PLAN.md §11.2）
// 復号部は DOM 非依存（node --test で試験する）。描画は canvas 2D のみ。
export const MAGIC_ACTIVITY = 'MBA1';

// "MBA1" | f64 t | u32 seq | u16 K | u16 G | u16 M | u16 reserved
//        | f32[K][G][2]（平均活動, 活動率）| u8[K][M]（標本の活動 x255）
const HEADER = 24;

export function decodeActivityFrame(buf) {
  if (buf.byteLength < HEADER) throw new Error('short frame');
  const dv = new DataView(buf);
  const magic = String.fromCharCode(
    dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== MAGIC_ACTIVITY) throw new Error(`bad magic ${magic}`);
  const t = dv.getFloat64(4, true);
  const seq = dv.getUint32(12, true);
  const K = dv.getUint16(16, true);
  const G = dv.getUint16(18, true);
  const M = dv.getUint16(20, true);
  const statsBytes = K * G * 2 * 4;
  if (buf.byteLength < HEADER + statsBytes + K * M) {
    throw new Error('short frame');
  }
  return {
    t, seq, K, G, M,
    stats: new Float32Array(buf.slice(HEADER, HEADER + statsBytes)),
    sample: new Uint8Array(
      buf.slice(HEADER + statsBytes, HEADER + statsBytes + K * M)),
  };
}

// 脳の色: K=3 は簡易人形の左右色（左 橙、中央 灰、右 青）、K=1 は水色。
function brainColors(K) {
  if (K === 3) return ['#e0a458', '#9aa4ad', '#58a7e0'];
  if (K === 1) return ['#8fd3ff'];
  return Array.from({ length: K },
    (_, i) => `hsl(${Math.round(i * 360 / K)}, 60%, 65%)`);
}

// 活動 0→1 を 暗(#1d242c) → 橙(#ffb347) → 白(#fff2c8) で色付けする LUT。
const HEAT = (() => {
  const c0 = [0x1d, 0x24, 0x2c], c1 = [0xff, 0xb3, 0x47], c2 = [0xff, 0xf2, 0xc8];
  return Array.from({ length: 256 }, (_, i) => {
    const v = i / 255;
    const [a, b, u] = v < 0.5 ? [c0, c1, v * 2] : [c1, c2, (v - 0.5) * 2];
    return `rgb(${a.map((x, j) => Math.round(x + (b[j] - x) * u)).join(',')})`;
  });
})();

const clamp01 = (v) => Math.max(0, Math.min(1, v));

export class ActivityHud {
  constructor(container, activityMeta) {
    this.el = container;
    this.meta = activityMeta;
    this.K = activityMeta.brains?.length ?? 1;
    this.G = activityMeta.groups?.length ?? 0;
    this.M = activityMeta.sample?.n ?? 0;
    this.hz = activityMeta.hz ?? 10;
    this.win = Math.max(2, Math.round(this.hz * 10)); // 直近 10 秒
    this.colors = brainColors(this.K);
    // スパークライン用リングバッファ: hist[(k*G+g)*win + slot]
    this.hist = new Float32Array(this.K * this.G * this.win);
    this.head = 0;
    this.count = 0;
    this.frame = null;
    this.dirty = false;
    this.visible = false;
    // 目盛り: 'auto' はフレームの最大値を 1 とし最大値を数字で示す（乱数観測では
    // 平均活動が 0.02 程度で、絶対目盛りだとバーも点群も空に見えるため）。
    this.scaleMode = 'auto';
    this.W = 328;
    this.labelW = 116;
    this.rowH = Math.max(10, 4 + this.K * 4);
    this._build();
  }

  _sec(title) {
    const d = document.createElement('div');
    d.className = 'act-sec';
    const h = document.createElement('div');
    h.className = 'act-title';
    h.textContent = title;
    d.appendChild(h);
    this.el.appendChild(d);
    return d;
  }

  _canvas(w, h) {
    const dpr = Math.min(globalThis.devicePixelRatio ?? 1, 2);
    const c = document.createElement('canvas');
    c.width = w * dpr;
    c.height = h * dpr;
    c.style.aspectRatio = `${w} / ${h}`;
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [c, ctx];
  }

  setScaleMode(mode) {
    this.scaleMode = mode === 'abs' ? 'abs' : 'auto';
    for (const b of this.scaleBtns) {
      b.classList.toggle('on', b.dataset.scale === this.scaleMode);
    }
    this.dirty = true;
  }

  _build() {
    // 枠1: 群ごとの活動バー（目盛りの切替ボタンつき）
    const s1 = this._sec('群の活動');
    const bar = document.createElement('div');
    bar.className = 'act-scale';
    this.scaleBtns = [];
    for (const [mode, label] of [['auto', '自動'], ['abs', '絶対']]) {
      const b = document.createElement('button');
      b.dataset.scale = mode;
      b.textContent = label;
      b.classList.toggle('on', mode === this.scaleMode);
      b.addEventListener('click', () => this.setScaleMode(mode));
      bar.appendChild(b);
      this.scaleBtns.push(b);
    }
    this.scaleNote = document.createElement('span');
    this.scaleNote.className = 'act-note';
    bar.appendChild(this.scaleNote);
    s1.appendChild(bar);
    const legend = document.createElement('div');
    legend.className = 'act-legend';
    (this.meta.brains ?? []).forEach((b, k) => {
      const s = document.createElement('span');
      s.style.color = this.colors[k];
      s.textContent = `■ ${b}`;
      legend.appendChild(s);
    });
    s1.appendChild(legend);
    this.barsH = this.G * this.rowH;
    [this.barsCanvas, this.barsCtx] = this._canvas(this.W, this.barsH);
    s1.appendChild(this.barsCanvas);

    // 枠2: 直近 10 秒のスパークライン
    const s2 = this._sec(`直近 10 秒（${this.hz} Hz）`);
    this.sparkRowH = 14;
    this.sparkH = this.G * this.sparkRowH;
    [this.sparkCanvas, this.sparkCtx] = this._canvas(this.W, this.sparkH);
    s2.appendChild(this.sparkCanvas);

    // 枠3: 標本の点群（脳ごとに一枚）
    const s3 = this._sec('標本の活動');
    this.cloudNote = document.createElement('div');
    this.cloudNote.className = 'act-note';
    s3.appendChild(this.cloudNote);
    const row = document.createElement('div');
    row.className = 'act-clouds';
    this.clouds = [];
    const gap = 6;
    const size = Math.min(180, Math.floor((this.W - (this.K - 1) * gap) / this.K));
    for (let k = 0; k < this.K; k++) {
      const cell = document.createElement('div');
      cell.className = 'act-cloud';
      const [c, ctx] = this._canvas(size, size);
      c.style.width = '100%';
      const cap = document.createElement('div');
      cap.className = 'act-cap';
      cap.style.color = this.colors[k];
      cap.textContent = this.meta.brains?.[k] ?? `b${k}`;
      cell.appendChild(c);
      cell.appendChild(cap);
      row.appendChild(cell);
      this.clouds.push({ c, ctx, size });
    }
    s3.appendChild(row);
  }

  push(frame) {
    this.frame = frame;
    const kMax = Math.min(frame.K, this.K);
    const gMax = Math.min(frame.G, this.G);
    for (let k = 0; k < kMax; k++) {
      for (let g = 0; g < gMax; g++) {
        this.hist[(k * this.G + g) * this.win + this.head] =
          frame.stats[(k * frame.G + g) * 2];
      }
    }
    this.head = (this.head + 1) % this.win;
    if (this.count < this.win) this.count++;
    this.dirty = true;
  }

  setVisible(v) {
    this.visible = v;
    this.el.style.display = v ? 'block' : 'none';
    if (v) this.dirty = true;
  }

  render() {
    if (!this.dirty || !this.visible || !this.frame) return;
    this._drawBars();
    this._drawSpark();
    this._drawClouds();
    this.dirty = false;
  }

  // 枠1: 群 × 脳の横バー（平均活動）+ 活動率の目盛り
  _drawBars() {
    const ctx = this.barsCtx, f = this.frame;
    const barX = this.labelW, barW = this.W - this.labelW - 2;
    ctx.clearRect(0, 0, this.W, this.barsH);
    ctx.font = '9px ui-monospace, monospace';
    ctx.textBaseline = 'middle';
    const gMax = Math.min(f.G, this.G), kMax = Math.min(f.K, this.K);
    const bh = (this.rowH - 6) / kMax;
    let top = 0;
    for (let k = 0; k < kMax; k++) {
      for (let g = 0; g < gMax; g++) top = Math.max(top, f.stats[(k * f.G + g) * 2]);
    }
    const auto = this.scaleMode === 'auto' && top > 0;
    const unit = auto ? top : 1;
    this.scaleNote.textContent = auto
      ? `平均の最大 ${top.toFixed(4)} を 1 とする（目盛りは活動率）`
      : '0〜1 の絶対目盛り';
    for (let g = 0; g < gMax; g++) {
      const y = g * this.rowH;
      const grp = this.meta.groups[g];
      ctx.fillStyle = '#9aa4ad';
      ctx.fillText(`${grp.label ?? grp.id} ${grp.n ?? ''}`,
        0, y + this.rowH / 2, this.labelW - 6);
      ctx.fillStyle = '#1d242c';
      ctx.fillRect(barX, y + 2, barW, this.rowH - 4);
      for (let k = 0; k < kMax; k++) {
        const mean = f.stats[(k * f.G + g) * 2];
        const rate = f.stats[(k * f.G + g) * 2 + 1];
        ctx.fillStyle = this.colors[k];
        ctx.fillRect(barX, y + 3 + k * bh,
          clamp01(mean / unit) * barW, Math.max(1, bh - 1));
        ctx.fillStyle = '#e8eef3';
        ctx.fillRect(barX + clamp01(rate) * barW - 0.5,
          y + 3 + k * bh, 1, Math.max(1, bh - 1));
      }
    }
  }

  // 枠2: 群ごと 1 段、脳ごとの線で直近 10 秒の平均活動
  _drawSpark() {
    const ctx = this.sparkCtx;
    const lineX = this.labelW, lineW = this.W - this.labelW - 2;
    ctx.clearRect(0, 0, this.W, this.sparkH);
    ctx.font = '9px ui-monospace, monospace';
    ctx.textBaseline = 'middle';
    const kMax = Math.min(this.frame.K, this.K);
    const autoMode = this.scaleMode === 'auto';
    for (let g = 0; g < this.G; g++) {
      const y = g * this.sparkRowH, h = this.sparkRowH;
      const grp = this.meta.groups[g];
      ctx.fillStyle = '#9aa4ad';
      ctx.fillText(`${grp.label ?? grp.id}`, 0, y + h / 2, this.labelW - 6);
      ctx.strokeStyle = '#2a333d';
      ctx.strokeRect(lineX + 0.5, y + 1.5, lineW - 1, h - 3);
      // 自動目盛り: この行（全脳）の直近ウィンドウの最大値を 1 とする
      let rowMax = 0;
      if (autoMode) {
        for (let k = 0; k < kMax; k++) {
          const base = (k * this.G + g) * this.win;
          for (let i = 0; i < this.count; i++) {
            const idx = (this.head - this.count + i + this.win) % this.win;
            rowMax = Math.max(rowMax, this.hist[base + idx]);
          }
        }
      }
      const unit = autoMode && rowMax > 0 ? rowMax : 1;
      if (autoMode && rowMax > 0) {
        ctx.fillStyle = '#6f7b86';
        ctx.textAlign = 'right';
        ctx.fillText(rowMax.toFixed(3), lineX + lineW - 3, y + h / 2);
        ctx.textAlign = 'left';
      }
      for (let k = 0; k < kMax; k++) {
        ctx.strokeStyle = this.colors[k];
        ctx.beginPath();
        const base = (k * this.G + g) * this.win;
        for (let i = 0; i < this.count; i++) {
          const idx = (this.head - this.count + i + this.win) % this.win;
          const v = clamp01(this.hist[base + idx] / unit);
          const x = lineX + (i / (this.win - 1)) * (lineW - 2) + 1;
          const py = y + h - 2 - v * (h - 4);
          if (i === 0) ctx.moveTo(x, py); else ctx.lineTo(x, py);
        }
        ctx.stroke();
      }
    }
  }

  // 枠3: 標本の点群。横 = x、縦 = z（z 小さいほど上 = 脳が上、神経索が下）
  _drawClouds() {
    const pos = this.meta.sample?.pos;
    if (!pos) return;
    const f = this.frame;
    const mMax = Math.min(f.M, this.M, pos.length);
    let top = 0;
    for (let i = 0; i < f.sample.length; i++) top = Math.max(top, f.sample[i]);
    const auto = this.scaleMode === 'auto' && top > 0;
    const gain = auto ? 255 / top : 1;
    this.cloudNote.textContent = auto
      ? `標本の最大 ${top}/255 を最も明るい色とする`
      : '0〜255 の絶対目盛り';
    for (let k = 0; k < Math.min(f.K, this.K); k++) {
      const { ctx, size } = this.clouds[k];
      ctx.clearRect(0, 0, size, size);
      ctx.fillStyle = '#101418';
      ctx.fillRect(0, 0, size, size);
      for (let m = 0; m < mMax; m++) {
        const px = (clamp01(pos[m][0] * 0.5 + 0.5)) * (size - 4) + 2;
        const py = (1 - clamp01(pos[m][2] * 0.5 + 0.5)) * (size - 4) + 2;
        ctx.fillStyle = HEAT[Math.min(255, Math.round(f.sample[k * f.M + m] * gain))];
        ctx.fillRect(px - 1, py - 1, 2.5, 2.5);
      }
    }
  }
}
