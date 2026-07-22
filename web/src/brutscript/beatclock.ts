// Beat clock — the musical clock that gates the rPPG pipeline.
//
// A camera at a fixed frame rate measures time arbitrarily. Audio running in
// parallel has a BPM: a periodic structure of beats, bars, and phrases.
// Synchronising the pipeline to that structure turns the camera into a
// time-gated instrument — measurements happen at beat-defined windows, not
// arbitrary slices. The beat is the clock, and it is the invariant along
// which the athlete's physiology propagates.
//
// This module is the clock source. It emits `beat`, `bar`, and `phrase`
// events on a grid derived from a BPM estimate, and exposes a snapshot of
// musical position (`bpm`, `beat`, `bar_pos`, `phrase`) suitable for pushing
// onto the BrutScript signal bus. Every rPPG capture protocol keys off beat
// position:
//
//   beat 1 of bar         → DC-baseline window   (mean RGB, skin-temp proxy)
//   beat 3 of bar         → AC-amplitude window  (vasodilation factor η)
//   every 4 bars (phrase) → full optical inversion (θ, PCHR decomposition)
//   every 8 bars          → effort index → decision model (exercise change?)
//
// The pipeline does not run continuously. It runs in windows the musical
// structure defines: measurement at the least sufficient interval.
//
// Source model. For electronic music the beat grid is quantised and
// grid-locked, with minimal BPM drift, so a synthetic constant-BPM grid is a
// faithful stand-in for a real onset detector on that material. `feedBeat()`
// lets a real detector (Web Audio onset analysis, an external tempo tracker)
// drive the same event stream later without changing any consumer.

// ─── Musical position ─────────────────────────────────────────────────────────

/** Position of a beat within the musical structure. Beats and bars are 1-indexed. */
export interface BeatPosition {
  /** Absolute beat count since the clock started (1-indexed, monotone). */
  index: number;
  /** Beat within the current bar: 1..beatsPerBar. */
  beatInBar: number;
  /** Absolute bar count since start (1-indexed). */
  bar: number;
  /** Absolute phrase count since start (1-indexed). A phrase is barsPerPhrase bars. */
  phrase: number;
  /** True on the first beat of a bar. */
  isBarStart: boolean;
  /** True on the first beat of a phrase. */
  isPhraseStart: boolean;
  /** Current BPM estimate at this beat. */
  bpm: number;
  /** Session-relative timestamp of this beat, ms. */
  t: number;
}

/** A signal snapshot suitable for pushing onto the BrutScript bus. */
export interface BeatSignals {
  bpm: number;
  /** Beat within bar, 1..beatsPerBar. */
  beat: number;
  /** Bar within phrase, 1..barsPerPhrase. */
  bar_pos: number;
  /** Absolute phrase index. */
  phrase: number;
  /** Absolute beat index (monotone). */
  beat_index: number;
}

export interface BeatClockOptions {
  /** Initial tempo. Electronic sets typically 120–140. Default 128. */
  bpm?: number;
  /** Beats per bar (time signature numerator). Default 4. */
  beatsPerBar?: number;
  /** Bars per phrase — the full-inversion cadence. Default 4. */
  barsPerPhrase?: number;
  /** Fired on every beat. */
  onBeat?: (pos: BeatPosition) => void;
  /** Fired on the first beat of each bar. */
  onBar?: (pos: BeatPosition) => void;
  /** Fired on the first beat of each phrase (the full-inversion boundary). */
  onPhrase?: (pos: BeatPosition) => void;
  /**
   * Clock driver. `interval` (default) schedules beats internally from the BPM.
   * `manual` fires nothing on its own — call `feedBeat()` from a real detector.
   */
  driver?: 'interval' | 'manual';
}

// ─── Beat clock ────────────────────────────────────────────────────────────────

export class BeatClock {
  private bpm: number;
  private readonly beatsPerBar: number;
  private readonly barsPerPhrase: number;
  private readonly onBeat?: (pos: BeatPosition) => void;
  private readonly onBar?: (pos: BeatPosition) => void;
  private readonly onPhrase?: (pos: BeatPosition) => void;
  private readonly driver: 'interval' | 'manual';

  private beatIndex = 0;                // beats emitted so far
  private timer: ReturnType<typeof setTimeout> | null = null;
  private startMs = 0;
  private running = false;
  private last: BeatPosition | null = null;

  constructor(opts: BeatClockOptions = {}) {
    this.bpm = Math.max(1, opts.bpm ?? 128);
    this.beatsPerBar = Math.max(1, opts.beatsPerBar ?? 4);
    this.barsPerPhrase = Math.max(1, opts.barsPerPhrase ?? 4);
    this.onBeat = opts.onBeat;
    this.onBar = opts.onBar;
    this.onPhrase = opts.onPhrase;
    this.driver = opts.driver ?? 'interval';
  }

  /** Milliseconds between beats at the current BPM. */
  get beatMs(): number { return 60_000 / this.bpm; }

  /** Current tempo estimate. */
  get currentBpm(): number { return this.bpm; }

  /** Update tempo mid-session (e.g. a track change or detector re-estimate). */
  setBpm(bpm: number): void {
    this.bpm = Math.max(1, bpm);
  }

  start(nowMs = performance.now()): void {
    if (this.running) return;
    this.running = true;
    this.startMs = nowMs;
    this.beatIndex = 0;
    if (this.driver === 'interval') this.scheduleNext(nowMs);
  }

  stop(): void {
    this.running = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
  }

  /**
   * Drive one beat from an external detector (manual driver, or to inject a
   * detected onset over the interval grid). Returns the emitted position.
   */
  feedBeat(nowMs = performance.now()): BeatPosition {
    return this.emitBeat(nowMs);
  }

  /** The most recently emitted position, or null before the first beat. */
  get position(): BeatPosition | null { return this.last; }

  /** Current musical position as a bus-pushable signal snapshot. */
  signals(): BeatSignals {
    const p = this.last;
    if (!p) {
      return { bpm: this.bpm, beat: 1, bar_pos: 1, phrase: 1, beat_index: 0 };
    }
    const barInPhrase = ((p.bar - 1) % this.barsPerPhrase) + 1;
    return {
      bpm: p.bpm,
      beat: p.beatInBar,
      bar_pos: barInPhrase,
      phrase: p.phrase,
      beat_index: p.index,
    };
  }

  // ── Internals ──────────────────────────────────────────────────────────────

  private scheduleNext(refMs: number): void {
    if (!this.running) return;
    // Anchor to the ideal grid so drift does not accumulate: the k-th beat is
    // due at startMs + k * beatMs. Schedule relative to that, not to "now".
    const dueAt = this.startMs + (this.beatIndex + 1) * this.beatMs;
    const delay = Math.max(0, dueAt - refMs);
    this.timer = setTimeout(() => {
      const t = performance.now();
      this.emitBeat(t);
      this.scheduleNext(t);
    }, delay);
  }

  private emitBeat(nowMs: number): BeatPosition {
    this.beatIndex += 1;
    const idx = this.beatIndex;
    const beatInBar = ((idx - 1) % this.beatsPerBar) + 1;
    const bar = Math.floor((idx - 1) / this.beatsPerBar) + 1;
    const phrase = Math.floor((bar - 1) / this.barsPerPhrase) + 1;
    const isBarStart = beatInBar === 1;
    const isPhraseStart = isBarStart && ((bar - 1) % this.barsPerPhrase === 0);

    const pos: BeatPosition = {
      index: idx,
      beatInBar,
      bar,
      phrase,
      isBarStart,
      isPhraseStart,
      bpm: this.bpm,
      t: nowMs - this.startMs,
    };
    this.last = pos;

    this.onBeat?.(pos);
    if (isBarStart) this.onBar?.(pos);
    if (isPhraseStart) this.onPhrase?.(pos);
    return pos;
  }
}

// ─── Effort index ──────────────────────────────────────────────────────────────

/**
 * The effort index HR/BPM. It costs nothing extra — it falls out of the
 * synchronisation itself once BPM is on the bus alongside HR.
 *
 *   ratio ≈ 1  → entrainment: the athlete moves with the music.
 *   ratio > 1  → working harder than the music is driving.
 *   ratio < 1  → recovery underway.
 */
export function effortIndex(hr: number, bpm: number): number {
  if (bpm <= 0) return 0;
  return hr / bpm;
}

export type EffortRegime = 'recovering' | 'entrained' | 'working';

/** Classify the effort index into a regime with a tolerance band around 1. */
export function effortRegime(ratio: number, band = 0.08): EffortRegime {
  if (ratio < 1 - band) return 'recovering';
  if (ratio > 1 + band) return 'working';
  return 'entrained';
}

// ─── Capture protocol allocation ───────────────────────────────────────────────

/** Which measurement protocol a given beat position calls for. */
export type CaptureProtocol =
  | 'dc_baseline'      // beat 1: mean RGB → DC baseline, skin-temp proxy
  | 'ac_amplitude'     // beat 3: AC amplitude → vasodilation η
  | 'full_inversion'   // phrase boundary: complete optical state θ + PCHR
  | 'idle';            // other beats: no capture

/**
 * Beat position determines measurement protocol — the least-sufficient-interval
 * allocation. Phrase boundaries take precedence (full inversion subsumes the
 * per-beat windows).
 */
export function protocolFor(pos: BeatPosition): CaptureProtocol {
  if (pos.isPhraseStart) return 'full_inversion';
  if (pos.beatInBar === 1) return 'dc_baseline';
  if (pos.beatInBar === 3) return 'ac_amplitude';
  return 'idle';
}
