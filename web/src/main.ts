// BRUT Observatory — entry point.
//
// Closed-circuit physiological observatory with anatomical glb corners,
// click-to-invoke side panels, and a bottom crossfilter dashboard.
//
// Per frame:
//   1. Capture face ROI via MediaPipe.
//   2. Run the WebGPU rPPG pipeline; get a global BVP sample + per-pixel R_c.
//   3. Push the BVP sample into the analyzer + respiration estimator.
//   4. Algebraically invert the cardiac equations of state (physio/eos.ts)
//      against the BVP-derived features to obtain a fitted cardiac state.
//   5. Drive anatomy tempos, side panels, and the crossfilter dashboard
//      from the fitted state.

import { startCamera, stopCamera, type CameraStream } from './camera/stream';
import { initFaceLandmarker, detectFace, extractFaceROI, destroyFaceLandmarker } from './camera/landmarks';
import { initGpu, resizeCanvasToDisplay, type GpuContext } from './gpu/device';
import { createRppgPipeline, type RppgPipeline } from './gpu/rppg';
import { BvpAnalyzer } from './physio/bvp';
import { RespirationEstimator } from './physio/respiration';
import { classifyRegime, type Regime } from './physio/regimes';
import { inferState, derive, REST_STATE, StateSmoother, type CardiacState } from './physio/eos';
import { KeyboardSensor } from './sensors/keyboard';
import { MouseSensor } from './sensors/mouse';
import { Overlay } from './ui/overlay';
import { mountAnatomyCorners, type AnatomyHandles } from './anatomy/corners';
import './ui/panels';
import { mountHeartPanel, type HeartPanelHandle } from './ui/heart-panel';
import { mountLungsPanel, type LungsPanelHandle } from './ui/lungs-panel';
import { mountDashboard, type Dashboard } from './charts/dashboard';
import { initStealthToggle } from './ui/stealth';
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
  smoother: StateSmoother;
  keyboard: KeyboardSensor;
  mouse: MouseSensor;
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
  smoother: new StateSmoother(0.18),
  keyboard: new KeyboardSensor(),
  mouse: new MouseSensor(),
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
const echoA2cCanvas = document.getElementById('echo-a2c-canvas') as HTMLCanvasElement;
const echoA5cCanvas = document.getElementById('echo-a5c-canvas') as HTMLCanvasElement;
const echoPsaxCanvas = document.getElementById('echo-psax-canvas') as HTMLCanvasElement;
const hrMini = document.getElementById('hr-mini')!;
const rrMini = document.getElementById('rr-mini')!;
const dotCamera = document.getElementById('dot-camera')!;
const dotKeyboard = document.getElementById('dot-keyboard')!;
const dotMouse = document.getElementById('dot-mouse')!;

initStealthToggle();

// Motor sensors run from page load — independent of the camera. Even if
// the user never clicks "start", we record their motor activity in the
// dashboard. This matches the framework's claim that all data is useful.
state.keyboard.start();
state.mouse.start();
setSensorDot(dotKeyboard, 'idle');
setSensorDot(dotMouse, 'idle');

startBtn.addEventListener('click', () => { void start(); });
stopBtn.addEventListener('click', () => { stop(); });

// Mount the dashboard immediately too, so motor data is collected and
// visualised before the camera is ever started.
const dashboardHost = document.getElementById('drawer-body')!;
state.dashboard = mountDashboard(dashboardHost);

// 1-Hz aggregator for motor-only mode (when the camera isn't running).
setInterval(() => {
  if (state.running) return; // camera path pushes its own records
  pushMotorRecord();
}, 1000);

function setSensorDot(el: HTMLElement, level: 'live' | 'idle' | 'warn' | 'dead'): void {
  el.classList.remove('live', 'idle', 'warn', 'dead');
  el.classList.add(level);
}

function pushMotorRecord(): void {
  const kw = state.keyboard.windowStats(1000);
  const mw = state.mouse.windowStats(1000);
  // Only push if anything happened — keeps the dashboard responsive.
  if (kw.count === 0 && mw.distance < 1 && mw.clicks === 0 && mw.scrollDelta === 0) {
    setSensorDot(dotKeyboard, state.keyboard.msSinceLastEvent() < 5000 ? 'idle' : 'idle');
    setSensorDot(dotMouse, state.mouse.msSinceLastEvent() < 5000 ? 'idle' : 'idle');
    return;
  }
  setSensorDot(dotKeyboard, kw.count > 0 ? 'live' : 'idle');
  setSensorDot(dotMouse, mw.distance > 1 ? 'live' : 'idle');
  state.dashboard?.push({
    t: performance.now(),
    hr: 0, rmssd: 0, rc: 0, sk: 0, st: 0, se: 0, regime: 0, rrBpm: 0,
    ees: 0, ea: 0, ef: 0, sv: 0, co: 0,
    keyCount: kw.count,
    meanIki: kw.meanIki,
    meanDwell: kw.meanDwell,
    backspaceRate: kw.backspaceRate,
    bursty: kw.bursty,
    mouseDistance: mw.distance,
    mousePeakVel: mw.peakVelocity,
    ramblingPower: mw.ramblingPower,
    tremblingPower: mw.tremblingPower,
    rtRatio: mw.rtRatio,
  });
}

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

    setStatus('mounting anatomy (5 glbs)');
    state.anatomy = await mountAnatomyCorners({
      heartCanvas,
      lungsCanvas,
      echoA2cCanvas,
      echoA5cCanvas,
      echoPsaxCanvas,
    });

    setStatus('mounting panels');
    state.heartPanel = mountHeartPanel();
    state.lungsPanel = mountLungsPanel();
    // Dashboard already mounted at page load so motor records flow before
    // the camera starts. We do NOT remount it here.

    state.overlay = new Overlay(overlayCanvas);
    state.overlay.resizeToVideo(video);

    state.bvp.setSampleRate(30);
    state.resp.setSampleRate(30);
    state.smoother.reset();

    setSensorDot(dotCamera, 'live');

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
  // Dashboard stays mounted — motor records keep flowing.
  state.rppg = null;
  state.anatomy = null;
  state.heartPanel = null;
  state.lungsPanel = null;
  state.gpu = null;
  state.overlay = null;
  state.bvp = new BvpAnalyzer();
  state.resp = new RespirationEstimator();
  state.smoother = new StateSmoother(0.18);
  setSensorDot(dotCamera, 'idle');
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

    // Cardiac equations-of-state inversion. Only run once we have a stable HR.
    let cardiacState: CardiacState = REST_STATE;
    if (stats.hrBpm >= 30) {
      const inferred = inferState({
        HR_bpm: stats.hrBpm,
        rel_amplitude: stats.relAmplitude,
        // dicrotic-notch detection from rPPG is not yet implemented; default to baseline.
      });
      cardiacState = state.smoother.update(inferred);
    }

    // Drive anatomy tempo from the fitted state (HR canonical from inverter).
    if (cardiacState.HR > 30 && state.anatomy) state.anatomy.setHeartHr(cardiacState.HR);
    if (respEst.rateBpm > 4 && state.anatomy) state.anatomy.setLungsRespRate(respEst.rateBpm);

    // Corner mini labels.
    hrMini.textContent = stats.hrBpm > 0 ? `${cardiacState.HR.toFixed(0)} bpm` : '— bpm';
    rrMini.textContent = respEst.rateBpm > 0 ? `${respEst.rateBpm.toFixed(0)} bpm` : '— bpm';

    // Side panels driven by the fitted state.
    const regimeRc = classifyRegime(stats.rc);
    state.heartPanel?.update(cardiacState, {
      rmssd: stats.rmssdMs,
      rc: stats.rc,
      sk: stats.sk,
      st: stats.st,
      se: stats.se,
      regimeRc,
    });

    state.lungsPanel?.update({
      rrBpm: respEst.rateBpm,
      respConfidence: respEst.confidence,
      altitudeM: 0,         // placeholder until a sensor or user input arrives
      paco2: 40,            // typical resting; later derived from respiration
    });

    // Dashboard push: one record per second, fusing cardiac + motor.
    if (now - state.lastDashPush > 1000 && stats.beats >= 2) {
      state.lastDashPush = now;
      const der = derive(cardiacState);
      const kw = state.keyboard.windowStats(1000);
      const mw = state.mouse.windowStats(1000);
      setSensorDot(dotKeyboard, kw.count > 0 ? 'live' : 'idle');
      setSensorDot(dotMouse, mw.distance > 1 ? 'live' : 'idle');
      state.dashboard?.push({
        t: now,
        hr: cardiacState.HR,
        rmssd: stats.rmssdMs,
        rc: stats.rc,
        sk: stats.sk,
        st: stats.st,
        se: stats.se,
        regime: REGIME_INDEX[regimeRc],
        rrBpm: respEst.rateBpm,
        ees: cardiacState.Ees,
        ea: cardiacState.Ea,
        ef: der.EF,
        sv: der.SV,
        co: der.CO,
        keyCount: kw.count,
        meanIki: kw.meanIki,
        meanDwell: kw.meanDwell,
        backspaceRate: kw.backspaceRate,
        bursty: kw.bursty,
        mouseDistance: mw.distance,
        mousePeakVel: mw.peakVelocity,
        ramblingPower: mw.ramblingPower,
        tremblingPower: mw.tremblingPower,
        rtRatio: mw.rtRatio,
      });
    }
  } else {
    setStatus('searching for face');
  }

  state.rafId = requestAnimationFrame(frame);
}
