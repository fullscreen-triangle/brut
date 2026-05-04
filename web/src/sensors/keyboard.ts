// Keyboard sensor.
//
// Per neuro-muscular-derivation.tex: motor commands are the efferent half
// of a closed charge circulation. Keystrokes are *direct readouts* of that
// efferent stream, available at sub-ms timestamp resolution from the
// browser. We capture every key event (excluding repeats) and produce two
// kinds of features:
//
//   1. Per-event:   dwell  = key_up - key_down                (force-application duration)
//                   flight = next_key_down - this_key_up      (motor program transition)
//                   iki    = next_key_down - this_key_down    (inter-keystroke interval)
//   2. Per-second:  count, mean_iki, std_iki, mean_dwell,
//                   backspace_rate, pause_count
//
// Pause count = number of IKIs > 1500 ms in the last second. Pauses
// distinguish cognitive engagement from motor execution.

const PAUSE_MS = 1500;
const HISTORY_SECONDS = 60;
const SUPPRESS = new Set([
  'Shift', 'Control', 'Alt', 'Meta', 'CapsLock', 'NumLock', 'ScrollLock',
  'Tab', 'Insert', 'Escape', 'OS',
]);

interface KeyEvent {
  code: string;
  key: string;
  type: 'down' | 'up';
  t: number;
}

interface KeystrokeFeature {
  t: number;          // key-down timestamp
  code: string;
  dwellMs: number;    // up - down
  flightMs: number;   // next-down - this-up; -1 if last
  ikiMs: number;      // next-down - this-down; -1 if last
  isBackspace: boolean;
}

export interface KeyboardWindow {
  count: number;
  meanIki: number;        // ms; 0 if no events
  stdIki: number;
  meanDwell: number;
  backspaceRate: number;  // 0..1
  pauseCount: number;     // IKIs > PAUSE_MS in window
  bursty: number;         // cv of IKIs (std/mean), 0 if N<2
}

export class KeyboardSensor {
  private events: KeyEvent[] = [];
  private features: KeystrokeFeature[] = [];
  private downByCode = new Map<string, KeyEvent>();
  private active = false;
  private lastEventMs = 0;

  start(): void {
    if (this.active) return;
    this.active = true;
    window.addEventListener('keydown', this.onDown, true);
    window.addEventListener('keyup', this.onUp, true);
  }

  stop(): void {
    if (!this.active) return;
    this.active = false;
    window.removeEventListener('keydown', this.onDown, true);
    window.removeEventListener('keyup', this.onUp, true);
    this.events.length = 0;
    this.features.length = 0;
    this.downByCode.clear();
  }

  isActive(): boolean {
    return this.active;
  }

  msSinceLastEvent(): number {
    if (this.lastEventMs === 0) return Infinity;
    return performance.now() - this.lastEventMs;
  }

  /**
   * Aggregate the last `windowMs` of keystrokes into per-window features.
   * Default 1000 ms — matches the dashboard 1-Hz cadence.
   */
  windowStats(windowMs = 1000): KeyboardWindow {
    const now = performance.now();
    const cutoff = now - windowMs;
    const recent = this.features.filter((f) => f.t >= cutoff);
    const count = recent.length;

    if (count === 0) {
      return {
        count: 0,
        meanIki: 0,
        stdIki: 0,
        meanDwell: 0,
        backspaceRate: 0,
        pauseCount: 0,
        bursty: 0,
      };
    }

    const ikis = recent.filter((f) => f.ikiMs > 0).map((f) => f.ikiMs);
    const dwells = recent.map((f) => f.dwellMs).filter((v) => v > 0);

    const meanIki = mean(ikis);
    const stdIki = stdev(ikis, meanIki);
    const meanDwell = mean(dwells);
    const backspaceCount = recent.filter((f) => f.isBackspace).length;
    const backspaceRate = backspaceCount / count;
    const pauseCount = ikis.filter((v) => v > PAUSE_MS).length;
    const bursty = meanIki > 0 ? stdIki / meanIki : 0;

    return { count, meanIki, stdIki, meanDwell, backspaceRate, pauseCount, bursty };
  }

  /** Last `nMs` of IKIs as a Float32Array, suitable for spectral analysis. */
  recentIkiSeries(nMs = 30_000): Float32Array {
    const cutoff = performance.now() - nMs;
    const xs = this.features
      .filter((f) => f.t >= cutoff && f.ikiMs > 0)
      .map((f) => f.ikiMs);
    return Float32Array.from(xs);
  }

  // ── private handlers ────────────────────────────────────────────────
  private onDown = (e: KeyboardEvent): void => {
    if (e.repeat) return;
    if (SUPPRESS.has(e.key)) return;
    const t = performance.now();
    const ev: KeyEvent = { code: e.code, key: e.key, type: 'down', t };
    this.events.push(ev);
    this.downByCode.set(e.code, ev);
    this.lastEventMs = t;
    this.evictOld(t);
  };

  private onUp = (e: KeyboardEvent): void => {
    if (SUPPRESS.has(e.key)) return;
    const t = performance.now();
    const down = this.downByCode.get(e.code);
    if (!down) return;
    this.downByCode.delete(e.code);

    const dwellMs = t - down.t;

    // Find the previous keystroke feature to fill in flight/iki for it.
    if (this.features.length > 0) {
      const prev = this.features[this.features.length - 1];
      if (prev.flightMs < 0) {
        // We don't yet know prev's flight — fill it now by treating the most
        // recent key-up as the boundary. This is approximate when several keys
        // overlap; that's fine for aggregate statistics.
      }
    }

    // Append a new feature for this completed keystroke.
    const feature: KeystrokeFeature = {
      t: down.t,
      code: e.code,
      dwellMs,
      flightMs: -1,
      ikiMs: -1,
      isBackspace: e.code === 'Backspace' || e.key === 'Backspace',
    };

    // Backfill the previous feature's flight/iki based on this new keystroke.
    if (this.features.length > 0) {
      const prev = this.features[this.features.length - 1];
      if (prev.flightMs < 0) prev.flightMs = down.t - (prev.t + prev.dwellMs);
      if (prev.ikiMs < 0) prev.ikiMs = down.t - prev.t;
    }

    this.features.push(feature);
    this.lastEventMs = t;
    this.evictOld(t);
  };

  private evictOld(t: number): void {
    const cutoff = t - HISTORY_SECONDS * 1000;
    while (this.features.length > 0 && this.features[0].t < cutoff) {
      this.features.shift();
    }
    while (this.events.length > 0 && this.events[0].t < cutoff) {
      this.events.shift();
    }
  }
}

function mean(a: number[]): number {
  if (a.length === 0) return 0;
  let s = 0;
  for (const v of a) s += v;
  return s / a.length;
}

function stdev(a: number[], m: number): number {
  if (a.length < 2) return 0;
  let s = 0;
  for (const v of a) {
    const d = v - m;
    s += d * d;
  }
  return Math.sqrt(s / (a.length - 1));
}
