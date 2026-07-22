// Microphone beat detector — the audio clock, from the room.
//
// Nobody training uses a computer; they use a phone, and the music is playing
// in the room. So the clock does not need the audio file or a streaming API —
// it *listens*. `getUserMedia({ audio: true })` opens the phone mic (the same
// permission flow the camera uses), and Web Audio extracts the beat grid from
// whatever is playing: gym speakers, earbuds, a boombox.
//
// Pipeline:
//   mic → AnalyserNode (FFT) → spectral flux → onset peaks → tempo (BPM) via
//   inter-onset autocorrelation → phase-locked beat grid → onBeat callbacks.
//
// The output matches the BeatClock's `feedBeat()` seam exactly, so a live mic
// replaces the synthetic grid with no change to any downstream consumer: the
// rPPG capture protocols, the effort index, and the exercise agents all key off
// the same beat events whether the clock is synthetic or heard.
//
// Electronic music is the easy, clean case (loud quantised grid, minimal
// drift); the detector is tuned for that material but degrades gracefully.

import { log } from '../util/log';

export interface MicBeatOptions {
  /** Fired once per detected beat, with the current BPM estimate. */
  onBeat: (bpm: number) => void;
  /** Fired when the BPM estimate is re-locked. Optional. */
  onTempo?: (bpm: number) => void;
  /** Plausible tempo band for the material. Default 90–160 (electronic). */
  bpmRange?: [number, number];
}

export interface MicBeatHandle {
  stream: MediaStream;
  stop(): void;
  /** Latest BPM estimate. */
  bpm(): number;
}

const FFT_SIZE = 1024;
const FLUX_HISTORY = 512;          // ~ a few seconds of onset envelope
const MIN_BEAT_GAP_MS = 250;       // reject double-triggers (≤ 240 BPM)

/**
 * Open the microphone and start detecting beats. Mirrors camera/stream.ts:
 * one async open, a handle with stop().
 */
export async function startMicBeat(opts: MicBeatOptions): Promise<MicBeatHandle> {
  const [bpmMin, bpmMax] = opts.bpmRange ?? [90, 160];

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,   // we want the raw room, not voice-cleaned audio
      noiseSuppression: false,
      autoGainControl: false,
    },
    video: false,
  });

  const AudioCtx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AudioCtx();
  const src = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = FFT_SIZE;
  analyser.smoothingTimeConstant = 0.0;   // we do our own smoothing on the flux
  src.connect(analyser);

  const bins = analyser.frequencyBinCount;
  const spectrum = new Float32Array(bins);
  let prevSpectrum = new Float32Array(bins);

  // Onset envelope: recent spectral-flux values with timestamps.
  const flux: number[] = [];
  const fluxT: number[] = [];
  const onsetTimes: number[] = [];   // ms timestamps of detected onsets

  let bpm = 120;
  let lastBeatMs = 0;
  let nextBeatMs = 0;                 // phase-locked predicted beat time
  let running = true;

  log(`mic ready: listening for beats (${bpmMin}–${bpmMax} BPM) device="${stream.getAudioTracks()[0]?.label ?? '?'}"`);

  function spectralFlux(now: number): number {
    analyser.getFloatFrequencyData(spectrum);   // dB
    let sum = 0;
    for (let i = 0; i < bins; i++) {
      // Positive spectral difference (energy increases) drives onsets.
      const d = spectrum[i] - prevSpectrum[i];
      if (d > 0) sum += d;
    }
    prevSpectrum.set(spectrum);
    flux.push(sum);
    fluxT.push(now);
    if (flux.length > FLUX_HISTORY) { flux.shift(); fluxT.shift(); }
    return sum;
  }

  /** Adaptive-threshold peak-pick on the newest flux sample. */
  function isOnset(): boolean {
    const n = flux.length;
    if (n < 8) return false;
    const w = flux.slice(Math.max(0, n - 24));
    const mean = w.reduce((a, b) => a + b, 0) / w.length;
    const std = Math.sqrt(w.reduce((s, v) => s + (v - mean) ** 2, 0) / w.length);
    const thresh = mean + 1.5 * std;
    const cur = flux[n - 1];
    const prev = flux[n - 2];
    return cur > thresh && cur >= prev;   // local rise above adaptive threshold
  }

  /** Estimate BPM from inter-onset intervals by autocorrelation over the band. */
  function estimateTempo(): void {
    if (onsetTimes.length < 6) return;
    const recent = onsetTimes.slice(-32);
    // Candidate beat periods over the plausible band, 1 BPM resolution.
    let bestBpm = bpm;
    let bestScore = -Infinity;
    for (let candidate = bpmMin; candidate <= bpmMax; candidate++) {
      const periodMs = 60_000 / candidate;
      // Score: how well onset times fall on a grid of this period (phase-free).
      let score = 0;
      for (const t of recent) {
        const phase = (t % periodMs) / periodMs;       // 0..1
        const d = Math.min(phase, 1 - phase);          // distance to nearest grid line
        score += Math.cos(2 * Math.PI * d);            // 1 at grid, −1 at anti-grid
      }
      if (score > bestScore) { bestScore = score; bestBpm = candidate; }
    }
    if (Math.abs(bestBpm - bpm) >= 1) {
      bpm = bestBpm;
      opts.onTempo?.(bpm);
    }
  }

  function tick(): void {
    if (!running) return;
    const now = performance.now();
    spectralFlux(now);

    if (isOnset() && now - lastBeatMs > MIN_BEAT_GAP_MS) {
      onsetTimes.push(now);
      if (onsetTimes.length > 64) onsetTimes.shift();
      estimateTempo();
    }

    // Phase-locked beat emission: emit a beat when the predicted grid time
    // arrives, and nudge the grid toward the nearest recent onset so it stays
    // locked to the music even as the estimate refines.
    const beatMs = 60_000 / bpm;
    if (nextBeatMs === 0) nextBeatMs = now + beatMs;
    if (now >= nextBeatMs) {
      // Nudge phase toward the closest onset within half a beat.
      const near = onsetTimes.find((t) => Math.abs(t - now) < beatMs / 2);
      const phaseCorrection = near !== undefined ? (near - now) * 0.25 : 0;
      lastBeatMs = now;
      nextBeatMs = now + beatMs + phaseCorrection;
      opts.onBeat(bpm);
    }

    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);

  return {
    stream,
    bpm: () => bpm,
    stop() {
      running = false;
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
    },
  };
}
