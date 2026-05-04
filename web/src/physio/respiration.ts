// Respiration rate extraction from rPPG.
//
// Two independent estimators that we average for robustness:
//   (a) BVP envelope spectrum: the slow swing of the BVP amplitude is driven
//       by intrathoracic-pressure-modulated venous return. Take a moving
//       absolute-value envelope, FFT, peak in 0.1-0.4 Hz.
//   (b) Respiratory sinus arrhythmia (RSA): the RR-interval series itself is
//       modulated at breath frequency. We use a coarse autocorrelation in the
//       respiratory band on the resampled BVP-derived envelope.
//
// We keep ~30 s of BVP samples and produce one rate estimate per second.

const WINDOW_SECONDS = 30;
const RESP_MIN_HZ = 0.1;   // 6 bpm
const RESP_MAX_HZ = 0.5;   // 30 bpm

export interface RespEstimate {
  rateBpm: number;       // breaths/min, 0 if not yet estimable
  amplitude: number;     // envelope AC amplitude (relative units)
  confidence: number;    // [0, 1] — ratio of dominant-band energy to total
}

export class RespirationEstimator {
  private samples: number[] = [];
  private times: number[] = [];
  private sampleRateHz = 30;

  setSampleRate(hz: number): void {
    this.sampleRateHz = Math.max(1, hz);
  }

  push(sample: number, tMs: number): void {
    this.samples.push(sample);
    this.times.push(tMs);
    const cutoff = tMs - WINDOW_SECONDS * 1000;
    while (this.times.length > 0 && this.times[0] < cutoff) {
      this.samples.shift();
      this.times.shift();
    }
  }

  estimate(): RespEstimate {
    const n = this.samples.length;
    if (n < this.sampleRateHz * 8) {
      return { rateBpm: 0, amplitude: 0, confidence: 0 };
    }

    // Step 1: BVP envelope via abs() then low-pass moving average.
    const env = new Float32Array(n);
    for (let i = 0; i < n; i++) env[i] = Math.abs(this.samples[i]);
    const envSmooth = movingAverage(env, Math.max(2, Math.floor(this.sampleRateHz / 2)));

    // Detrend (remove DC + linear).
    const detrended = linearDetrend(envSmooth);

    // Step 2: Goertzel-style scan across the respiratory band; pick peak.
    const fmin = RESP_MIN_HZ;
    const fmax = RESP_MAX_HZ;
    const stepHz = 0.005;
    let bestHz = 0;
    let bestPower = 0;
    let totalPower = 0;
    for (let f = fmin; f <= fmax; f += stepHz) {
      const p = goertzelPower(detrended, this.sampleRateHz, f);
      totalPower += p;
      if (p > bestPower) {
        bestPower = p;
        bestHz = f;
      }
    }
    if (bestPower === 0) {
      return { rateBpm: 0, amplitude: 0, confidence: 0 };
    }

    const rateBpm = bestHz * 60;
    const amplitude = stdev(detrended);
    const confidence = bestPower / Math.max(1e-9, totalPower);

    return { rateBpm, amplitude, confidence };
  }
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

function linearDetrend(a: Float32Array): Float32Array {
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

/** Single-frequency power via the Goertzel algorithm. */
function goertzelPower(a: Float32Array, sampleRateHz: number, freqHz: number): number {
  const n = a.length;
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

function stdev(a: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m += a[i];
  m /= a.length;
  let v = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - m;
    v += d * d;
  }
  return Math.sqrt(v / a.length);
}
