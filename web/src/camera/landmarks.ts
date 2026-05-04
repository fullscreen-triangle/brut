import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';
import { log } from '../util/log';

export interface FaceROI {
  // Forehead and cheek ROI in normalized [0, 1] image coords (origin top-left, x = right, y = down).
  forehead: { x: number; y: number; w: number; h: number };
  leftCheek: { x: number; y: number; w: number; h: number };
  rightCheek: { x: number; y: number; w: number; h: number };
  // Bounding box covering all three.
  bbox: { x: number; y: number; w: number; h: number };
}

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';

let landmarker: FaceLandmarker | null = null;

export async function initFaceLandmarker(): Promise<void> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
  log('face landmarker ready');
}

export function detectFace(video: HTMLVideoElement, timestampMs: number): FaceLandmarkerResult | null {
  if (!landmarker) return null;
  return landmarker.detectForVideo(video, timestampMs);
}

// MediaPipe FaceLandmarker emits 478 landmark indices. Reference subsets for ROIs:
// https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
const FOREHEAD_IDX = [10, 67, 109, 338, 297]; // forehead patch
const LCHEEK_IDX = [50, 101, 36, 205, 187, 123]; // subject's left, image right
const RCHEEK_IDX = [280, 330, 266, 425, 411, 352]; // subject's right, image left

export function extractFaceROI(result: FaceLandmarkerResult | null): FaceROI | null {
  if (!result || result.faceLandmarks.length === 0) return null;
  const lm = result.faceLandmarks[0];

  const bboxOf = (idxs: number[]) => {
    let xmin = 1, ymin = 1, xmax = 0, ymax = 0;
    for (const i of idxs) {
      const p = lm[i];
      if (p.x < xmin) xmin = p.x;
      if (p.y < ymin) ymin = p.y;
      if (p.x > xmax) xmax = p.x;
      if (p.y > ymax) ymax = p.y;
    }
    return { x: xmin, y: ymin, w: xmax - xmin, h: ymax - ymin };
  };

  const forehead = inset(bboxOf(FOREHEAD_IDX), 0.05);
  const leftCheek = inset(bboxOf(LCHEEK_IDX), 0.1);
  const rightCheek = inset(bboxOf(RCHEEK_IDX), 0.1);

  const xmin = Math.min(forehead.x, leftCheek.x, rightCheek.x);
  const ymin = Math.min(forehead.y, leftCheek.y, rightCheek.y);
  const xmax = Math.max(forehead.x + forehead.w, leftCheek.x + leftCheek.w, rightCheek.x + rightCheek.w);
  const ymax = Math.max(forehead.y + forehead.h, leftCheek.y + leftCheek.h, rightCheek.y + rightCheek.h);

  return {
    forehead,
    leftCheek,
    rightCheek,
    bbox: { x: xmin, y: ymin, w: xmax - xmin, h: ymax - ymin },
  };
}

function inset(b: { x: number; y: number; w: number; h: number }, frac: number) {
  const ix = b.w * frac;
  const iy = b.h * frac;
  return { x: b.x + ix, y: b.y + iy, w: Math.max(0, b.w - 2 * ix), h: Math.max(0, b.h - 2 * iy) };
}

export function destroyFaceLandmarker(): void {
  landmarker?.close();
  landmarker = null;
}
