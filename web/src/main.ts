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
import { inferState, derive, REST_STATE, StateSmoother, pchrDecompose, type CardiacState } from './physio/eos';
import {
  inferSkinState,
  skinTemperatureC,
  VasodilationTracker,
  SKIN_BASELINE,
  type SkinState,
} from './physio/skin-optics';
import { KeyboardSensor } from './sensors/keyboard';
import { MouseSensor } from './sensors/mouse';
import { BlinkDetector } from './sensors/blinks';
import { MelanopicTracker } from './sensors/melanopic';
import { FaceColorSensor } from './sensors/face-color';
import { Overlay } from './ui/overlay';
import { mountAnatomyCorners, type AnatomyHandles } from './anatomy/corners';
import './ui/panels';
import { mountHeartPanel, type HeartPanelHandle } from './ui/heart-panel';
import { mountLungsPanel, type LungsPanelHandle } from './ui/lungs-panel';
import { mountDashboard, type Dashboard } from './charts/dashboard';
import { initStealthToggle } from './ui/stealth';
import { mountLanding } from './ui/landing';
import { createPulseSvg, type PulseSvgHandle, type PulseState } from './ui/pulse-svg';
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
  blinks: BlinkDetector;
  melanopic: MelanopicTracker;
  faceColor: FaceColorSensor;
  vaso: VasodilationTracker;
  lastSkinState: SkinState;
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
  blinks: new BlinkDetector(),
  melanopic: new MelanopicTracker(),
  faceColor: new FaceColorSensor(),
  vaso: new VasodilationTracker(),
  lastSkinState: { ...SKIN_BASELINE },
  running: false,
  rafId: null,
  fpsAcc: 0,
  fpsCount: 0,
  lastFpsLog: 0,
  lastFrameTs: 0,
  lastDashPush: 0,
};

// Sensor indicator pulse SVG handles — created in initObservatory().
let cameraPulse:   PulseSvgHandle | null = null;
let keyboardPulse: PulseSvgHandle | null = null;
let mousePulse:    PulseSvgHandle | null = null;

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

// ── Boot sequence: landing first, observatory after begin ────────────
// Show the minimal landing screen with the liquid-fill heart. Sensors
// and the dashboard mount only AFTER the user clicks begin, so the
// landing stays clean and nothing about the observatory is computed
// before the user has consented to engage with it.

const landing = mountLanding();
void landing.beginPromise.then(() => {
  initObservatory();
});

function initObservatory(): void {
  initStealthToggle();
  initSensorPulses();

  // Motor sensors run from begin — independent of the camera. Even if
  // the user never clicks "start", we record their motor activity in the
  // dashboard. This matches the framework's claim that all data is useful.
  state.keyboard.start();
  state.mouse.start();
  setSensorDot(dotKeyboard, 'idle', keyboardPulse);
  setSensorDot(dotMouse, 'idle', mousePulse);

  startBtn.addEventListener('click', () => { void start(); });
  stopBtn.addEventListener('click', () => { stop(); });

  // Mount the dashboard immediately so motor data is collected and
  // visualised before the camera is ever started.
  const dashboardHost = document.getElementById('drawer-body')!;
  state.dashboard = mountDashboard(dashboardHost);

  // 1-Hz aggregator for motor-only mode (when the camera isn't running).
  setInterval(() => {
    if (state.running) return; // camera path pushes its own records
    pushMotorRecord();
  }, 1000);
}

function setSensorDot(el: HTMLElement, level: PulseState, pulse?: PulseSvgHandle | null): void {
  el.classList.remove('live', 'idle', 'warn', 'dead');
  el.classList.add(level);
  pulse?.setState(level);
}

function initSensorPulses(): void {
  // Camera → jugular (1.4 s ≈ 60 BPM visual; duration updated from live HR).
  const camEl = document.getElementById('dot-camera');
  if (camEl) {
    cameraPulse = createPulseSvg('jugular', { viewBox: '228 63 180 68', strokeWidth: 2 });
    camEl.appendChild(cameraPulse.el);
  }
  // Keyboard → pulsar (2.5 s, deliberate; matches inter-keystroke cadence).
  const kbEl = document.getElementById('dot-keyboard');
  if (kbEl) {
    keyboardPulse = createPulseSvg('pulsar', { viewBox: '228 63 180 68', strokeWidth: 2 });
    kbEl.appendChild(keyboardPulse.el);
  }
  // Mouse → bleed (1.2 s, erratic; matches trembling/rambling character).
  const msEl = document.getElementById('dot-mouse');
  if (msEl) {
    mousePulse = createPulseSvg('bleed', { viewBox: '228 63 180 68', strokeWidth: 2 });
    msEl.appendChild(mousePulse.el);
  }
}

function pushMotorRecord(): void {
  const kw = state.keyboard.windowStats(1000);
  const mw = state.mouse.windowStats(1000);
  const ml = state.melanopic.tick();
  const bw = state.blinks.windowStats();
  // Only push if any sensor actually has fresh data this tick.
  if (
    kw.count === 0 &&
    mw.distance < 1 &&
    mw.clicks === 0 &&
    mw.scrollDelta === 0 &&
    bw.countLastSecond === 0
  ) {
    setSensorDot(dotKeyboard, 'idle', keyboardPulse);
    setSensorDot(dotMouse, 'idle', mousePulse);
    return;
  }
  setSensorDot(dotKeyboard, kw.count > 0 ? 'live' : 'idle', keyboardPulse);
  setSensorDot(dotMouse, mw.distance > 1 ? 'live' : 'idle', mousePulse);
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
    blinks: bw.countLastSecond,
    blinksPerMin: bw.bpmRate,
    melFlux: ml.flux,
    melSensitivity: ml.sensitivity,
    melLoadMlxH: ml.cumulativeMlxHours,
    T_skin_C: 0,
    vasodilation: 0,
    spo2Optical: 0,
    dHRautonomic: 0,
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

    setSensorDot(dotCamera, 'live', cameraPulse);

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
  state.blinks = new BlinkDetector();
  state.vaso.reset();
  state.lastSkinState = { ...SKIN_BASELINE };
  // Reset corner labels — anatomy is destroyed but the DOM text persists.
  hrMini.textContent = '— bpm';
  rrMini.textContent = '— bpm';
  setSensorDot(dotCamera, 'idle', cameraPulse);
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

  // Blink detection runs whenever a face is visible — independent of rPPG.
  state.blinks.ingest(lmResult);

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

    // Skin-optics inversion: sample face ROI mean RGB, invert the layered
    // optical model to recover (melanin, [Hb], oxygenation), and combine with
    // the BVP amplitude to derive a vasodilation factor → skin temperature.
    const colorSample = state.faceColor.sample(video, roi);
    const vaso = state.vaso.update(stats.relAmplitude, 1.0);
    let skinState: SkinState = state.lastSkinState;
    if (colorSample) {
      skinState = inferSkinState(colorSample.combined, vaso, state.lastSkinState);
      state.lastSkinState = skinState;
    } else {
      skinState = { ...skinState, vasodilation: vaso };
    }
    const T_skin_C = skinTemperatureC(vaso);
    const spo2 = skinState.oxygenation;

    // Cardiac equations-of-state inversion. Only run once we have a stable HR.
    // Now feeds T_skin and SpO₂ into the PCHR decomposition so HR is split
    // into intrinsic + metabolic + hypoxic + autonomic, instead of attributing
    // everything to the autonomic axis.
    let cardiacState: CardiacState = REST_STATE;
    let pchr = pchrDecompose(0, undefined, undefined);
    if (stats.hrBpm >= 30) {
      const inferred = inferState({
        HR_bpm: stats.hrBpm,
        rel_amplitude: stats.relAmplitude,
        T_skin_C,
        SpO2: spo2,
      });
      cardiacState = state.smoother.update(inferred);
      pchr = pchrDecompose(stats.hrBpm, T_skin_C, spo2, cardiacState.HR);
    }

    // Compute derived hemodynamics once and reuse for shader, panels, dashboard.
    const der = derive(cardiacState);

    // Drive anatomy tempo + strain shader ONLY when we have a real fit.
    // Gating on stats.hrBpm (the raw measurement) — not on cardiacState.HR,
    // which is REST_STATE.HR=70 by default and would mislead before any
    // actual measurement.
    const haveCardiacFit = stats.hrBpm >= 30;
    // Sync camera pulse indicator speed to measured HR so the ECG animation
    // travels at the user's actual cardiac cadence.
    if (haveCardiacFit && cameraPulse) {
      cameraPulse.setDuration(60 / cardiacState.HR);
    }
    if (state.anatomy) {
      if (haveCardiacFit) {
        state.anatomy.setHeartHr(cardiacState.HR);
        state.anatomy.setCardiacFit({
          HR: cardiacState.HR,
          Ees: cardiacState.Ees,
          Ea: cardiacState.Ea,
          EDV: cardiacState.EDV,
          ESV: der.ESV,
          EF: der.EF,
          Rc: stats.rc > 0 ? stats.rc : 0.85,
        });
      } else {
        // No fit yet — pause cardiac glb tempo and put strain shader into
        // its neutral / no-signal mode (uHR=0 triggers the grey fragment path).
        state.anatomy.setHeartHr(0);
        state.anatomy.setCardiacFit({
          HR: 0, Ees: 0, Ea: 0, EDV: 0, ESV: 0, EF: 0, Rc: 0,
        });
      }
      if (respEst.rateBpm > 4) {
        state.anatomy.setLungsRespRate(respEst.rateBpm);
      } else {
        state.anatomy.setLungsRespRate(0);
      }
    }

    // Corner mini labels — keep them honest about whether there's a live signal.
    hrMini.textContent = haveCardiacFit ? `${cardiacState.HR.toFixed(0)} bpm` : '— bpm';
    rrMini.textContent = respEst.rateBpm > 4 ? `${respEst.rateBpm.toFixed(0)} bpm` : '— bpm';

    // Side panels driven by the fitted state. Pass an HR=0 marker when no
    // fit exists yet, so the heart panel shows '—' instead of REST_STATE
    // defaults that would look like real numbers.
    const regimeRc = classifyRegime(stats.rc);
    const panelState: CardiacState = haveCardiacFit
      ? cardiacState
      : { ...cardiacState, HR: 0 };
    state.heartPanel?.update(panelState, {
      rmssd: stats.rmssdMs,
      rc: stats.rc,
      sk: stats.sk,
      st: stats.st,
      se: stats.se,
      regimeRc,
      T_skin_C: haveCardiacFit ? T_skin_C : undefined,
      vasodilation: haveCardiacFit ? vaso : undefined,
      spo2_proxy: haveCardiacFit ? spo2 : undefined,
      pchr: haveCardiacFit ? pchr : undefined,
    });

    state.lungsPanel?.update({
      rrBpm: respEst.rateBpm,
      respConfidence: respEst.confidence,
      altitudeM: 0,         // placeholder until a sensor or user input arrives
      paco2: 40,            // typical resting; later derived from respiration
    });

    // Dashboard push: one record per second, fusing cardiac + motor + blinks + melanopic.
    if (now - state.lastDashPush > 1000 && stats.beats >= 2) {
      state.lastDashPush = now;
      const kw = state.keyboard.windowStats(1000);
      const mw = state.mouse.windowStats(1000);
      const bw = state.blinks.windowStats();
      const ml = state.melanopic.tick();
      setSensorDot(dotKeyboard, kw.count > 0 ? 'live' : 'idle', keyboardPulse);
      setSensorDot(dotMouse, mw.distance > 1 ? 'live' : 'idle', mousePulse);
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
        blinks: bw.countLastSecond,
        blinksPerMin: bw.bpmRate,
        melFlux: ml.flux,
        melSensitivity: ml.sensitivity,
        melLoadMlxH: ml.cumulativeMlxHours,
        T_skin_C,
        vasodilation: vaso,
        spo2Optical: spo2,
        dHRautonomic: pchr.dHR_autonomic,
      });
    }
  } else {
    setStatus('searching for face');
  }

  state.rafId = requestAnimationFrame(frame);
}
