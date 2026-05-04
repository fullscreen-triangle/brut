// Mouse / pointer sensor.
//
// Per rambling-trembling-sensor.tex: postural CoP decomposes into a slow
// supraspinal "rambling" component (< 0.5 Hz) and a fast spinal-loop
// "trembling" component (0.5-3 Hz). The same architecture applies to any
// motor target-tracking task — including cursor pointing. We sample
// pointer events, build a velocity time series, and spectrally decompose
// into rambling / trembling band powers.
//
// Per-second aggregates:
//   distance         px traversed in last second
//   peakVelocity     max |v| in last second (px/s)
//   meanVelocity     mean |v| in last second
//   ramblingPower    spectral energy in 0.05-0.5 Hz
//   tremblingPower   spectral energy in 0.5-3.0 Hz
//   clicks           pointer-down events in last second
//   scrollDelta      |Δscroll| in last second

const HISTORY_SECONDS = 60;

const RAMBLING_LO = 0.05;
const RAMBLING_HI = 0.5;
const TREMBLING_LO = 0.5;
const TREMBLING_HI = 3.0;

interface PointerSample {
  t: number;
  x: number;
  y: number;
}

export interface MouseWindow {
  distance: number;        // px in last `windowMs`
  peakVelocity: number;    // px/s
  meanVelocity: number;    // px/s
  ramblingPower: number;   // arbitrary units
  tremblingPower: number;  // arbitrary units
  rtRatio: number;         // ramblingPower / (ramblingPower + tremblingPower)
  clicks: number;
  scrollDelta: number;     // |Δscroll| (px)
  active: boolean;         // any pointer activity in last second
}

export class MouseSensor {
  private samples: PointerSample[] = [];
  private clickTimes: number[] = [];
  private scrollEvents: { t: number; dy: number; dx: number }[] = [];
  private active = false;
  private lastEventMs = 0;

  start(): void {
    if (this.active) return;
    this.active = true;
    window.addEventListener('pointermove', this.onMove, true);
    window.addEventListener('pointerdown', this.onDown, true);
    window.addEventListener('wheel', this.onWheel, { capture: true, passive: true });
  }

  stop(): void {
    if (!this.active) return;
    this.active = false;
    window.removeEventListener('pointermove', this.onMove, true);
    window.removeEventListener('pointerdown', this.onDown, true);
    window.removeEventListener('wheel', this.onWheel, true);
    this.samples.length = 0;
    this.clickTimes.length = 0;
    this.scrollEvents.length = 0;
  }

  isActive(): boolean {
    return this.active;
  }

  msSinceLastEvent(): number {
    if (this.lastEventMs === 0) return Infinity;
    return performance.now() - this.lastEventMs;
  }

  windowStats(windowMs = 1000): MouseWindow {
    const now = performance.now();
    const cutoff = now - windowMs;

    // Distance + peak/mean velocity.
    let distance = 0;
    let peak = 0;
    let velSum = 0;
    let velCount = 0;
    for (let i = 1; i < this.samples.length; i++) {
      const a = this.samples[i - 1];
      const b = this.samples[i];
      if (b.t < cutoff) continue;
      const dt = (b.t - a.t) / 1000;
      if (dt <= 0) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.hypot(dx, dy);
      const v = d / dt;
      distance += d;
      if (v > peak) peak = v;
      velSum += v;
      velCount += 1;
    }
    const meanVelocity = velCount > 0 ? velSum / velCount : 0;

    // Spectral split on a longer (30 s) tail of the velocity series.
    const { ramblingPower, tremblingPower } = this.spectralSplit(30_000);
    const totalPow = ramblingPower + tremblingPower;
    const rtRatio = totalPow > 1e-9 ? ramblingPower / totalPow : 0;

    const clicks = this.clickTimes.filter((t) => t >= cutoff).length;
    let scrollDelta = 0;
    for (const e of this.scrollEvents) {
      if (e.t >= cutoff) scrollDelta += Math.abs(e.dx) + Math.abs(e.dy);
    }

    const active = this.lastEventMs > cutoff;

    return {
      distance,
      peakVelocity: peak,
      meanVelocity,
      ramblingPower,
      tremblingPower,
      rtRatio,
      clicks,
      scrollDelta,
      active,
    };
  }

  /**
   * Build a velocity series over the last `tailMs` ms (resampled to 30 Hz),
   * and integrate the power in the rambling and trembling bands.
   */
  private spectralSplit(tailMs: number): { ramblingPower: number; tremblingPower: number } {
    const fs = 30; // Hz — resampling rate
    const n = Math.floor((tailMs / 1000) * fs);
    if (this.samples.length < 2 || n < 32) {
      return { ramblingPower: 0, tremblingPower: 0 };
    }
    const now = performance.now();
    const t0 = now - tailMs;

    // Resample velocity at uniform 30 Hz over [t0, now].
    const series = new Float32Array(n);
    let j = 1;
    for (let i = 0; i < n; i++) {
      const t = t0 + (i / fs) * 1000;
      while (j < this.samples.length && this.samples[j].t < t) j++;
      if (j >= this.samples.length) break;
      const a = this.samples[j - 1];
      const b = this.samples[j];
      const dt = (b.t - a.t) / 1000;
      if (dt <= 0) {
        series[i] = 0;
        continue;
      }
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const v = Math.hypot(dx, dy) / dt;
      series[i] = v;
    }

    // Detrend.
    let m = 0;
    for (let i = 0; i < n; i++) m += series[i];
    m /= n;
    for (let i = 0; i < n; i++) series[i] -= m;

    // Goertzel power scan in each band.
    const stepHz = 0.05;
    let ramb = 0;
    let trem = 0;
    for (let f = RAMBLING_LO; f <= RAMBLING_HI; f += stepHz) {
      ramb += goertzelPower(series, fs, f);
    }
    for (let f = TREMBLING_LO; f <= TREMBLING_HI; f += stepHz) {
      trem += goertzelPower(series, fs, f);
    }
    return { ramblingPower: ramb, tremblingPower: trem };
  }

  // ── private handlers ────────────────────────────────────────────────
  private onMove = (e: PointerEvent): void => {
    const t = performance.now();
    this.samples.push({ t, x: e.clientX, y: e.clientY });
    this.lastEventMs = t;
    this.evictOld(t);
  };

  private onDown = (_e: PointerEvent): void => {
    const t = performance.now();
    this.clickTimes.push(t);
    this.lastEventMs = t;
    this.evictOld(t);
  };

  private onWheel = (e: WheelEvent): void => {
    const t = performance.now();
    this.scrollEvents.push({ t, dx: e.deltaX, dy: e.deltaY });
    this.lastEventMs = t;
    this.evictOld(t);
  };

  private evictOld(t: number): void {
    const cutoff = t - HISTORY_SECONDS * 1000;
    while (this.samples.length > 0 && this.samples[0].t < cutoff) this.samples.shift();
    while (this.clickTimes.length > 0 && this.clickTimes[0] < cutoff) this.clickTimes.shift();
    while (this.scrollEvents.length > 0 && this.scrollEvents[0].t < cutoff) this.scrollEvents.shift();
  }
}

function goertzelPower(a: Float32Array, sampleRateHz: number, freqHz: number): number {
  const n = a.length;
  if (n === 0) return 0;
  const k = freqHz / sampleRateHz;
  const w = 2 * Math.PI * k;
  const cosw = Math.cos(w);
  const coeff = 2 * cosw;
  let q1 = 0;
  let q2 = 0;
  for (let i = 0; i < n; i++) {
    const q0 = coeff * q1 - q2 + a[i];
    q2 = q1;
    q1 = q0;
  }
  return q1 * q1 + q2 * q2 - q1 * q2 * coeff;
}
