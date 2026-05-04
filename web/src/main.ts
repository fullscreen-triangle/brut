// BRUT Observatory — entry point.
//
// Closed-circuit physiological observatory:
//   camera --> GPU shader pipeline --> coherence field --> screen --> eye
// All metrics computed in a single render loop. No backend, no persistence.

import { startCamera, stopCamera, type CameraStream } from './camera/stream';
import { initFaceLandmarker, detectFace, extractFaceROI, destroyFaceLandmarker } from './camera/landmarks';
import { initGpu, resizeCanvasToDisplay, type GpuContext } from './gpu/device';
import { createRppgPipeline, type RppgPipeline } from './gpu/rppg';
import { BvpAnalyzer } from './physio/bvp';
import { Overlay } from './ui/overlay';
import { renderPanel } from './ui/panel';
import { log, setStatus } from './util/log';

interface AppState {
  camera: CameraStream | null;
  gpu: GpuContext | null;
  rppg: RppgPipeline | null;
  overlay: Overlay | null;
  bvp: BvpAnalyzer;
  running: boolean;
  rafId: number | null;
  fpsAcc: number;
  fpsCount: number;
  lastFpsLog: number;
  lastFrameTs: number;
}

const state: AppState = {
  camera: null,
  gpu: null,
  rppg: null,
  overlay: null,
  bvp: new BvpAnalyzer(),
  running: false,
  rafId: null,
  fpsAcc: 0,
  fpsCount: 0,
  lastFpsLog: 0,
  lastFrameTs: 0,
};

const startBtn = document.getElementById('start') as HTMLButtonElement;
const stopBtn = document.getElementById('stop') as HTMLButtonElement;
const video = document.getElementById('video') as HTMLVideoElement;
const overlayCanvas = document.getElementById('overlay') as HTMLCanvasElement;
const gpuCanvas = document.getElementById('gpu') as HTMLCanvasElement;

startBtn.addEventListener('click', () => { void start(); });
stopBtn.addEventListener('click', () => { stop(); });

async function start(): Promise<void> {
  if (state.running) return;
  startBtn.disabled = true;

  try {
    setStatus('opening camera');
    state.camera = await startCamera(video);

    setStatus('loading face landmarker');
    await initFaceLandmarker();

    setStatus('initialising webgpu');
    state.gpu = await initGpu(gpuCanvas);
    resizeCanvasToDisplay(gpuCanvas);

    setStatus('building rppg pipeline');
    state.rppg = await createRppgPipeline(state.gpu);

    state.overlay = new Overlay(overlayCanvas);
    state.overlay.resizeToVideo(video);

    // Camera fps is updated continuously from frame deltas; 30 is just the seed.
    state.bvp.setSampleRate(30);

    state.running = true;
    stopBtn.disabled = false;
    setStatus('running');
    state.lastFrameTs = performance.now();
    state.rafId = requestAnimationFrame(frame);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    log(`start failed: ${msg}`);
    setStatus(`error: ${msg}`);
    startBtn.disabled = false;
    stop();
  }
}

function stop(): void {
  state.running = false;
  if (state.rafId !== null) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }
  if (state.camera) {
    stopCamera(state.camera);
    state.camera = null;
  }
  destroyFaceLandmarker();
  state.rppg?.destroy();
  state.rppg = null;
  state.gpu = null;
  state.overlay = null;
  state.bvp = new BvpAnalyzer();
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus('stopped');
}

async function frame(tNowDom: number): Promise<void> {
  if (!state.running || !state.camera || !state.gpu || !state.rppg || !state.overlay) return;

  const now = performance.now();
  const dt = Math.max(1, now - state.lastFrameTs);
  state.lastFrameTs = now;
  state.fpsAcc += 1000 / dt;
  state.fpsCount += 1;

  if (now - state.lastFpsLog > 2000) {
    const fps = state.fpsAcc / Math.max(1, state.fpsCount);
    log(`fps=${fps.toFixed(1)}`);
    state.bvp.setSampleRate(fps);
    state.lastFpsLog = now;
    state.fpsAcc = 0;
    state.fpsCount = 0;
  }

  // WebGPU swap chain picks up canvas-size changes automatically; we only
  // need to keep the canvas backing store in sync with CSS layout.
  resizeCanvasToDisplay(gpuCanvas);
  state.overlay.resizeToVideo(video);

  const lmResult = detectFace(video, tNowDom);
  const roi = extractFaceROI(lmResult);

  state.overlay.clear();
  state.overlay.drawROI(roi);

  if (roi) {
    const result = await state.rppg.tick(video, roi, 1000 / dt, now);
    if (result.globalBvp !== 0 || result.rcMean !== 0) {
      state.bvp.push(result.globalBvp, now);
    }
    const stats = state.bvp.compute();
    renderPanel(stats, { rcMean: result.rcMean, rcStd: result.rcStd, snr: result.snr });
  } else {
    setStatus('searching for face');
  }

  state.rafId = requestAnimationFrame(frame);
}
