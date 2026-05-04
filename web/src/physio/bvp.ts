// CPU-side BVP buffer + cardiac analysis.
// Receives one global BVP sample per frame (the spatial mean of valid R_c
// patches' detrended green amplitude, produced by the GPU). This buffer feeds
// HR / RMSSD / R_c / S-entropy estimation.

const BUFFER_SECONDS = 12;

export interface BvpStats {
  hrBpm: number;            // 0 if not yet estimable
  rmssdMs: number;          // 0 if < 2 valid intervals
  rc: number;               // Kuramoto-style coherence in [0, 1]
  cv: number;               // RR coefficient of variation
  sk: number;               // S_k: Shannon entropy of RR distribution / log(N)
  st: number;               // S_t: temporal integration since session start
  se: number;               // S_e: entropy utilisation against max
  beats: number;            // detected beat count in window
  filled: number;           // how full the buffer is, [0, 1]
}

export class BvpAnalyzer {
  private samples: number[] = [];
  private sampleTimes: number[] = [];
  private sessionStartMs: number;
  private sampleRateHz = 30;

  constructor() {
    this.sessionStartMs = performance.now();
  }

  setSampleRate(hz: number): void {
    this.sampleRateHz = Math.max(1, hz);
  }

  push(sample: number, tMs: number): void {
    this.samples.push(sample);
    this.sampleTimes.push(tMs);
    const cutoff = tMs - BUFFER_SECONDS * 1000;
    while (this.sampleTimes.length > 0 && this.sampleTimes[0] < cutoff) {
      this.samples.shift();
      this.sampleTimes.shift();
    }
  }

  compute(): BvpStats {
    const n = this.samples.length;
    const filled = Math.min(1, n / (BUFFER_SECONDS * this.sampleRateHz));
    if (n < this.sampleRateHz * 4) {
      return zero(filled);
    }

    // Detrend (linear) over the buffer.
    const detrended = linearDetrend(this.samples);

    // Bandpass: simple difference of moving averages (hi-pass cardiac envelope).
    // Approx 0.7-3 Hz with windows ~4 and ~32 samples at 30 Hz.
    const band = bandpassMA(detrended, 4, 32);

    // Peak detect: local maxima above adaptive threshold.
    const peaks = detectPeaks(band, 0.5);

    // Convert peaks to RR intervals (ms).
    const rr: number[] = [];
    for (let i = 1; i < peaks.length; i++) {
      const dtMs = this.sampleTimes[peaks[i]] - this.sampleTimes[peaks[i - 1]];
      // Reject physiologically implausible intervals.
      if (dtMs > 333 && dtMs < 1500) rr.push(dtMs);
    }

    if (rr.length < 2) {
      return zero(filled);
    }

    const meanRR = mean(rr);
    const hrBpm = 60000 / meanRR;

    // RMSSD.
    let sqDiff = 0;
    for (let i = 1; i < rr.length; i++) {
      const d = rr[i] - rr[i - 1];
      sqDiff += d * d;
    }
    const rmssdMs = Math.sqrt(sqDiff / Math.max(1, rr.length - 1));

    const cv = rmssdMs / meanRR;
    const rc = Math.exp(-2 * Math.PI * Math.PI * cv * cv);

    // S-entropy stack -------------------------------------------------
    // S_k: normalised Shannon entropy over an RR histogram (HRV complexity).
    const sk = shannonRR(rr);

    // S_t: temporal integration since session start (cardio-neural-integration eq. 5).
    // tau_circadian ~ 90 min for ultradian; we use 5 min so the demo crosses the
    // saturation knee on a useful timescale.
    const tauChar = 5 * 60 * 1000;
    const st = 1 - Math.exp(-(tMs(this.sessionStartMs)) / tauChar);

    // S_e: entropy utilisation, log p(n, M) / log p_max. We treat the observed
    // distribution variance against a reference max-variance scenario.
    const rrVar = variance(rr, meanRR);
    const refVar = Math.pow(0.1 * meanRR, 2); // 10% of mean RR as reference scale
    const se = Math.min(1, Math.max(0, Math.log(1 + rrVar / refVar) / Math.log(2)));

    return {
      hrBpm,
      rmssdMs,
      rc,
      cv,
      sk,
      st,
      se,
      beats: rr.length + 1,
      filled,
    };
  }
}

function zero(filled: number): BvpStats {
  return { hrBpm: 0, rmssdMs: 0, rc: 0, cv: 0, sk: 0, st: 0, se: 0, beats: 0, filled };
}

function linearDetrend(a: number[]): Float32Array {
  const n = a.length;
  let sx = 0, sy = 0, sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) {
    sx += i;
    sy += a[i];
    sxy += i * a[i];
    sxx += i * i;
  }
  const denom = n * sxx - sx * sx;
  const slope = denom > 1e-9 ? (n * sxy - sx * sy) / denom : 0;
  const intercept = (sy - slope * sx) / n;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = a[i] - (intercept + slope * i);
  return out;
}

function bandpassMA(a: Float32Array, narrow: number, wide: number): Float32Array {
  const n = a.length;
  const sN = movingAverage(a, narrow);
  const sW = movingAverage(a, wide);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = sN[i] - sW[i];
  return out;
}

function movingAverage(a: Float32Array, w: number): Float32Array {
  const n = a.length;
  const out = new Float32Array(n);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    acc += a[i];
    if (i >= w) acc -= a[i - w];
    out[i] = acc / Math.min(i + 1, w);
  }
  return out;
}

function detectPeaks(a: Float32Array, sigmaMul: number): number[] {
  const n = a.length;
  if (n < 3) return [];
  let m = 0;
  for (let i = 0; i < n; i++) m += a[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) {
    const d = a[i] - m;
    v += d * d;
  }
  v = Math.sqrt(v / n);
  const thr = m + sigmaMul * v;
  const peaks: number[] = [];
  // Refractory ~250 ms at any sample rate; here we approximate as 6 samples.
  const refractory = 6;
  let lastPeak = -refractory;
  for (let i = 1; i < n - 1; i++) {
    if (a[i] > thr && a[i] > a[i - 1] && a[i] > a[i + 1] && i - lastPeak >= refractory) {
      peaks.push(i);
      lastPeak = i;
    }
  }
  return peaks;
}

function mean(a: number[]): number {
  let s = 0;
  for (const v of a) s += v;
  return s / Math.max(1, a.length);
}

function variance(a: number[], m: number): number {
  let s = 0;
  for (const v of a) {
    const d = v - m;
    s += d * d;
  }
  return s / Math.max(1, a.length);
}

function shannonRR(rr: number[]): number {
  const n = rr.length;
  if (n < 4) return 0;
  // Histogram with sqrt-N bins over the observed range.
  const bins = Math.max(4, Math.round(Math.sqrt(n)));
  let lo = Infinity, hi = -Infinity;
  for (const v of rr) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (hi - lo < 1) return 0;
  const counts = new Uint32Array(bins);
  const w = (hi - lo) / bins;
  for (const v of rr) {
    const k = Math.min(bins - 1, Math.floor((v - lo) / w));
    counts[k]++;
  }
  let h = 0;
  for (const c of counts) {
    if (c === 0) continue;
    const p = c / n;
    h -= p * Math.log2(p);
  }
  return h / Math.log2(bins);
}

function tMs(start: number): number {
  return performance.now() - start;
}
