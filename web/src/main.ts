// BRUT Observatory — entry point.
//
// Closed-circuit physiological observatory with anatomical glb corners,
// click-to-invoke side panels, and a bottom crossfilter dashboard.

import { startCamera, stopCamera, type CameraStream } from './camera/stream';
import { initFaceLandmarker, detectFace, extractFaceROI, destroyFaceLandmarker } from './camera/landmarks';
import { initGpu, resizeCanvasToDisplay, type GpuContext } from './gpu/device';
import { createRppgPipeline, type RppgPipeline } from './gpu/rppg';
import { BvpAnalyzer } from './physio/bvp';
import { RespirationEstimator } from './physio/respiration';
import { classifyRegime, type Regime } from './physio/regimes';
import { Overlay } from './ui/overlay';
import { mountAnatomyCorners, type AnatomyHandles } from './anatomy/corners';
import './ui/panels';
import { mountHeartPanel, type HeartPanelHandle } from './ui/heart-panel';
import { mountLungsPanel, type LungsPanelHandle } from './ui/lungs-panel';
import { mountDashboard, type Dashboard } from './charts/dashboard';
import { log, setStatus } from './util/log';

const REGIME_INDEX: Record<Regime, number> = {
  turbulent: 0,
  aperture: 1,
  cascade: 2,
  coherent: 3,
  'phase-locked': 4,
};

interface AppState {
  camera: CameraStream | null;
  gpu: GpuContext | null;
  rppg: RppgPipeline | null;
  overlay: Overlay | null;
  anatomy: AnatomyHandles | null;
  heartPanel: HeartPanelHandle | null;
  lungsPanel: LungsPanelHandle | null;
  dashboard: Dashboard | null;
  bvp: BvpAnalyzer;
  resp: RespirationEstimator;
  running: boolean;
  rafId: number | null;
  fpsAcc: number;
  fpsCount: number;
  lastFpsLog: number;
  lastFrameTs: number;
  lastDashPush: number;
}

const state: AppState = {
  camera: null,
  gpu: null,
  rppg: null,
  overlay: null,
  anatomy: null,
  heartPanel: null,
  lungsPanel: null,
  dashboard: null,
  bvp: new BvpAnalyzer(),
  resp: new RespirationEstimator(),
  running: false,
  rafId: null,
  fpsAcc: 0,
  fpsCount: 0,
  lastFpsLog: 0,
  lastFrameTs: 0,
  lastDashPush: 0,
};

const startBtn = document.getElementById('start') as HTMLButtonElement;
const stopBtn = document.getElementById('stop') as HTMLButtonElement;
const video = document.getElementById('video') as HTMLVideoElement;
const overlayCanvas = document.getElementById('overlay') as HTMLCanvasElement;
const gpuCanvas = document.getElementById('gpu') as HTMLCanvasElement;
const heartCanvas = document.getElementById('heart-canvas') as HTMLCanvasElement;
const lungsCanvas = document.getElementById('lungs-canvas') as HTMLCanvasElement;
const hrMini = document.getElementById('hr-mini')!;
const rrMini = document.getElementById('rr-mini')!;

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

    setStatus('mounting anatomy');
    state.anatomy = await mountAnatomyCorners(heartCanvas, lungsCanvas);

    setStatus('mounting panels');
    state.heartPanel = mountHeartPanel();
    state.lungsPanel = mountLungsPanel();
    state.dashboard = mountDashboard(document.getElementById('drawer-body')!);

    state.overlay = new Overlay(overlayCanvas);
    state.overlay.resizeToVideo(video);

    state.bvp.setSampleRate(30);
    state.resp.setSampleRate(30);

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
  state.anatomy?.destroy();
  state.heartPanel?.destroy();
  state.lungsPanel?.destroy();
  state.dashboard?.destroy();
  state.rppg = null;
  state.anatomy = null;
  state.heartPanel = null;
  state.lungsPanel = null;
  state.dashboard = null;
  state.gpu = null;
  state.overlay = null;
  state.bvp = new BvpAnalyzer();
  state.resp = new RespirationEstimator();
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
    state.resp.setSampleRate(fps);
    state.lastFpsLog = now;
    state.fpsAcc = 0;
    state.fpsCount = 0;
  }

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
      state.resp.push(result.globalBvp, now);
    }
    const stats = state.bvp.compute();
    const respEst = state.resp.estimate();

    // Drive anatomy tempo from live signals.
    if (stats.hrBpm > 30 && state.anatomy) state.anatomy.setHeartHr(stats.hrBpm);
    if (respEst.rateBpm > 4 && state.anatomy) state.anatomy.setLungsRespRate(respEst.rateBpm);

    // Corner mini labels.
    hrMini.textContent = stats.hrBpm > 0 ? `${stats.hrBpm.toFixed(0)} bpm` : '— bpm';
    rrMini.textContent = respEst.rateBpm > 0 ? `${respEst.rateBpm.toFixed(0)} bpm` : '— bpm';

    // Side panels.
    const regime = classifyRegime(stats.rc);
    state.heartPanel?.update({
      hrBpm: stats.hrBpm,
      rmssd: stats.rmssdMs,
      rc: stats.rc,
      sk: stats.sk,
      st: stats.st,
      se: stats.se,
      regime,
    });

    // SpO2 from a Hill-curve forward eval at typical arterial PO2 = 100 mmHg.
    // This is the model's prediction at sea-level normoxia until the V/Q
    // closure brings in actual alveolar gas equation output.
    const arterialPO2 = 100;
    const sat = Math.pow(arterialPO2, 2.7) / (Math.pow(27, 2.7) + Math.pow(arterialPO2, 2.7));
    state.lungsPanel?.update({
      rrBpm: respEst.rateBpm,
      respConfidence: respEst.confidence,
      spo2Estimate: sat * 100,
      arterialPO2,
    });

    // Dashboard push (one record per second to keep crossfilter responsive).
    if (now - state.lastDashPush > 1000 && stats.beats >= 2) {
      state.lastDashPush = now;
      state.dashboard?.push({
        t: now,
        hr: stats.hrBpm,
        rmssd: stats.rmssdMs,
        rc: stats.rc,
        sk: stats.sk,
        st: stats.st,
        se: stats.se,
        regime: REGIME_INDEX[regime],
        rrBpm: respEst.rateBpm,
      });
    }
  } else {
    setStatus('searching for face');
  }

  state.rafId = requestAnimationFrame(frame);
}
