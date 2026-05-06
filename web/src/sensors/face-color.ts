// Face-color sensor.
//
// Extracts mean RGB across the face ROI (forehead + bilateral cheeks) by
// drawing the current video frame into an offscreen 2D canvas and averaging
// the pixels inside each sub-region. This gives the optical model what it
// needs — DC RGB per skin region — without a GPU readback path.
//
// Why offscreen 2D instead of using the rPPG GPU pipeline:
//
//   - rPPG already pulls the green channel into a circular buffer for HR.
//     Adding R+B to that buffer triples its size for marginal HR benefit.
//   - The optical model only needs DC mean RGB once per second, not per frame.
//   - 2D canvas getImageData on a 64×64 region is ~1 ms — cheap, and runs
//     completely on the CPU off the rendering critical path.

import type { FaceROI } from '../camera/landmarks';

export interface FaceColorSample {
  forehead: { r: number; g: number; b: number };
  leftCheek: { r: number; g: number; b: number };
  rightCheek: { r: number; g: number; b: number };
  combined: { r: number; g: number; b: number };
}

/** Width of the offscreen capture canvas. We scale the video down for cheap sampling. */
const CAPTURE_W = 320;
const CAPTURE_H = 180;

export class FaceColorSensor {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private lastSampleMs = 0;

  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.width = CAPTURE_W;
    this.canvas.height = CAPTURE_H;
    const ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('2D context unavailable for face-color sensor');
    this.ctx = ctx;
  }

  /** Sample the current video frame; returns null if there's nothing to sample. */
  sample(video: HTMLVideoElement, roi: FaceROI | null): FaceColorSample | null {
    if (!roi || video.videoWidth === 0) return null;
    const now = performance.now();
    if (now - this.lastSampleMs < 200) return null;  // throttle to ~5 Hz
    this.lastSampleMs = now;

    this.ctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);

    return {
      forehead: this.meanRgbInRoi(roi.forehead),
      leftCheek: this.meanRgbInRoi(roi.leftCheek),
      rightCheek: this.meanRgbInRoi(roi.rightCheek),
      combined: this.meanRgbInRoi(roi.bbox),
    };
  }

  private meanRgbInRoi(rect: { x: number; y: number; w: number; h: number }): {
    r: number;
    g: number;
    b: number;
  } {
    const x = Math.max(0, Math.min(CAPTURE_W - 1, Math.floor(rect.x * CAPTURE_W)));
    const y = Math.max(0, Math.min(CAPTURE_H - 1, Math.floor(rect.y * CAPTURE_H)));
    const w = Math.max(1, Math.min(CAPTURE_W - x, Math.floor(rect.w * CAPTURE_W)));
    const h = Math.max(1, Math.min(CAPTURE_H - y, Math.floor(rect.h * CAPTURE_H)));

    const data = this.ctx.getImageData(x, y, w, h).data;
    let sumR = 0, sumG = 0, sumB = 0;
    const n = data.length / 4;
    for (let i = 0; i < data.length; i += 4) {
      sumR += data[i];
      sumG += data[i + 1];
      sumB += data[i + 2];
    }
    return { r: sumR / n / 255, g: sumG / n / 255, b: sumB / n / 255 };
  }
}
