// 2D canvas overlay: face ROI bbox + MediaPipe sub-region rectangles.

import type { FaceROI } from '../camera/landmarks';

export class Overlay {
  private ctx: CanvasRenderingContext2D;

  constructor(private canvas: HTMLCanvasElement) {
    const c = canvas.getContext('2d');
    if (!c) throw new Error('overlay canvas context failed');
    this.ctx = c;
  }

  resizeToVideo(video: HTMLVideoElement): void {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  drawROI(roi: FaceROI | null): void {
    if (!roi) return;
    const { width: W, height: H } = this.canvas;
    const ctx = this.ctx;

    ctx.strokeStyle = 'rgba(95, 175, 255, 0.6)';
    ctx.lineWidth = 1.5;
    rect(ctx, roi.bbox, W, H);

    ctx.strokeStyle = 'rgba(127, 212, 127, 0.5)';
    ctx.lineWidth = 1.0;
    rect(ctx, roi.forehead, W, H);
    rect(ctx, roi.leftCheek, W, H);
    rect(ctx, roi.rightCheek, W, H);
  }
}

function rect(
  ctx: CanvasRenderingContext2D,
  r: { x: number; y: number; w: number; h: number },
  W: number,
  H: number,
): void {
  ctx.strokeRect(r.x * W, r.y * H, r.w * W, r.h * H);
}
