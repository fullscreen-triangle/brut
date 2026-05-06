// Blink detector from MediaPipe FaceLandmarker output.
//
// Eye Aspect Ratio (EAR) — Soukupova & Cech 2016 — collapses during a blink:
//
//   EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
//
// where p1..p6 are 6 specific landmarks ringing each eye. We compute EAR for
// both eyes per frame, average them, threshold + refractory to detect blinks.
// Each blink is a timestamp event; we maintain a 60 s rolling window and
// derive blinks/min and inter-blink intervals.

import type { FaceLandmarkerResult } from '@mediapipe/tasks-vision';

// MediaPipe 478-point face mesh — eye landmark indices that work well for EAR.
const LEFT_EYE = [33, 160, 158, 133, 153, 144];   // p1, p2, p3, p4, p5, p6
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

const EAR_BLINK_THRESHOLD = 0.20;
const EAR_OPEN_THRESHOLD = 0.24;     // hysteresis to avoid jitter at threshold
const REFRACTORY_MS = 250;
const HISTORY_SECONDS = 60;

interface BlinkEvent {
  t: number;
  minEar: number;
}

export interface BlinkWindow {
  countLastSecond: number;
  bpmRate: number;          // blinks per minute, last 60 s
  ear: number;              // current instantaneous EAR (0 if no face)
  totalSession: number;
}

export class BlinkDetector {
  private events: BlinkEvent[] = [];
  private isClosed = false;
  private minEarInBlink = Infinity;
  private lastBlinkMs = -REFRACTORY_MS;
  private lastEar = 0;
  private totalSession = 0;
  private hasFace = false;

  ingest(result: FaceLandmarkerResult | null): void {
    if (!result || result.faceLandmarks.length === 0) {
      this.hasFace = false;
      this.isClosed = false;
      this.minEarInBlink = Infinity;
      return;
    }
    this.hasFace = true;
    const lm = result.faceLandmarks[0];
    const earL = computeEAR(lm, LEFT_EYE);
    const earR = computeEAR(lm, RIGHT_EYE);
    const ear = 0.5 * (earL + earR);
    this.lastEar = ear;
    const t = performance.now();

    // Hysteretic blink state machine.
    if (!this.isClosed && ear < EAR_BLINK_THRESHOLD) {
      // Blink onset.
      this.isClosed = true;
      this.minEarInBlink = ear;
    } else if (this.isClosed) {
      if (ear < this.minEarInBlink) this.minEarInBlink = ear;
      if (ear > EAR_OPEN_THRESHOLD) {
        // Blink offset — record it if past refractory.
        if (t - this.lastBlinkMs > REFRACTORY_MS) {
          this.events.push({ t, minEar: this.minEarInBlink });
          this.lastBlinkMs = t;
          this.totalSession += 1;
        }
        this.isClosed = false;
        this.minEarInBlink = Infinity;
      }
    }

    // Evict old events.
    const cutoff = t - HISTORY_SECONDS * 1000;
    while (this.events.length > 0 && this.events[0].t < cutoff) this.events.shift();
  }

  windowStats(): BlinkWindow {
    const now = performance.now();
    const oneSecAgo = now - 1000;
    let countLastSecond = 0;
    for (let i = this.events.length - 1; i >= 0; i--) {
      if (this.events[i].t < oneSecAgo) break;
      countLastSecond += 1;
    }
    const bpmRate = (this.events.length * 60) / HISTORY_SECONDS;
    return {
      countLastSecond,
      bpmRate,
      ear: this.hasFace ? this.lastEar : 0,
      totalSession: this.totalSession,
    };
  }

  /** Recent blink timestamps (ms epoch via performance.now()), oldest first. */
  recentTimestamps(): number[] {
    return this.events.map((e) => e.t);
  }
}

function computeEAR(
  landmarks: Array<{ x: number; y: number }>,
  idx: number[],
): number {
  const p1 = landmarks[idx[0]];
  const p2 = landmarks[idx[1]];
  const p3 = landmarks[idx[2]];
  const p4 = landmarks[idx[3]];
  const p5 = landmarks[idx[4]];
  const p6 = landmarks[idx[5]];
  const v1 = dist(p2, p6);
  const v2 = dist(p3, p5);
  const h = dist(p1, p4);
  if (h < 1e-6) return 0;
  return (v1 + v2) / (2 * h);
}

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
