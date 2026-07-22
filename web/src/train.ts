// BRUT Train — the mobile, beat-gated training instance.
//
// Opened from a phone browser (a link off the desktop site, or "Add to Home
// Screen" as a PWA), this is a fullscreen portrait training shell. It asks for
// two permissions, exactly as the observatory asks for the camera:
//
//   • camera — front-facing rPPG: heart rate and its PCHR decomposition from
//     the face, per the layered-optical-ppg method.
//   • microphone — the beat clock, heard from the room (gym speakers, earbuds,
//     a boombox). The music's BPM gates when and how the camera measures.
//
// The athlete never picks the exercise. A set of exercise agents each hold one
// exercise; at every musical phrase boundary the body's beat-synchronised
// physiological evidence is evaluated for two-factor relevance (does it advance
// this exercise's purpose, and is it a coherent exertion), and if the body has
// made its case the next exercise takes over. The GLB model shows the current
// exercise, moving at the tempo of the music.
//
// This entry reuses the observatory's physiology building blocks (camera, rPPG,
// skin optics, PCHR) and the beat clock + exercise agents; it does not load the
// full observatory UI.

import { startCamera, stopCamera, type CameraStream } from './camera/stream';
import { initFaceLandmarker, detectFace, extractFaceROI, destroyFaceLandmarker } from './camera/landmarks';
import { initGpu, resizeCanvasToDisplay, type GpuContext } from './gpu/device';
import { createRppgPipeline, type RppgPipeline } from './gpu/rppg';
import { BvpAnalyzer } from './physio/bvp';
import { VasodilationTracker } from './physio/skin-optics';
import { inferSkinState, skinTemperatureC, SKIN_BASELINE, type SkinState } from './physio/skin-optics';
import { pchrDecompose } from './physio/eos';
import { FaceColorSensor } from './sensors/face-color';
import { startMicBeat, type MicBeatHandle } from './audio/mic-beat';
import { BeatClock, effortIndex, effortRegime, protocolFor, type BeatPosition } from './brutscript/beatclock';
import {
  EXERCISE_ROSTER,
  FIRST_EXERCISE,
  evaluateTransition,
  type ExerciseAgent,
  type PhraseEvidence,
} from './brutscript/exercise-agents';
import { mountGlbViewer, CARDIAC_GLB_CATALOGUE, type GlbViewerHandle } from './brutscript/sandbox-glb';

// ─── DOM shell ─────────────────────────────────────────────────────────────────

const $ = (id: string) => document.getElementById(id)!;

interface TrainState {
  camera: CameraStream | null;
  gpu: GpuContext | null;
  rppg: RppgPipeline | null;
  mic: MicBeatHandle | null;
  clock: BeatClock | null;
  glb: GlbViewerHandle | null;
  bvp: BvpAnalyzer;
  vaso: VasodilationTracker;
  faceColor: FaceColorSensor;
  lastSkin: SkinState;
  active: ExerciseAgent | null;
  effortSamples: number[];
  latest: {
    hr: number; effort: number; dhrAutonomic: number; dhrMetabolic: number;
    dhrHypoxic: number; vasodilation: number; tSkin: number; rcMean: number;
  };
  rafId: number | null;
  running: boolean;
}

const state: TrainState = {
  camera: null, gpu: null, rppg: null, mic: null, clock: null, glb: null,
  bvp: new BvpAnalyzer(),
  vaso: new VasodilationTracker(),
  faceColor: new FaceColorSensor(),
  lastSkin: { ...SKIN_BASELINE },
  active: null,
  effortSamples: [],
  latest: { hr: 0, effort: 0, dhrAutonomic: 0, dhrMetabolic: 0, dhrHypoxic: 0, vasodilation: 1, tSkin: 33, rcMean: 0 },
  rafId: null,
  running: false,
};

// ─── Start / stop ───────────────────────────────────────────────────────────────

async function start(): Promise<void> {
  if (state.running) return;
  setStatus('requesting camera + microphone…');

  const video = $('train-video') as HTMLVideoElement;
  const gpuCanvas = $('train-gpu') as HTMLCanvasElement;
  const glbCanvas = $('train-glb') as HTMLCanvasElement;

  // 1. Camera (front-facing rPPG).
  try {
    state.camera = await startCamera(video);
  } catch (err) {
    setStatus(`camera denied: ${String(err)}`);
    return;
  }

  // 2. WebGPU rPPG pipeline. Guard: some mobile browsers lack WebGPU — the
  //    beat/effort/agent loop still runs, just without live HR from the camera.
  try {
    state.gpu = await initGpu(gpuCanvas);
    state.rppg = await createRppgPipeline(state.gpu);
    await initFaceLandmarker();
  } catch (err) {
    setStatus(`rPPG unavailable (${String(err)}) — running beat + effort only`);
  }

  // 3. GLB exercise character.
  try {
    state.glb = await mountGlbViewer(glbCanvas, CARDIAC_GLB_CATALOGUE[0].url);
  } catch { /* non-fatal: training runs without the avatar */ }

  // 4. Microphone beat clock. The mic drives the BeatClock via feedBeat();
  //    the clock owns bar/phrase structure and the capture-protocol allocation.
  state.clock = new BeatClock({
    bpm: 128,
    beatsPerBar: 4,
    barsPerPhrase: 4,
    driver: 'manual',                         // the mic is the driver, not a timer
    onBeat: (pos) => onBeatTick(pos),
    onPhrase: (pos) => onPhraseBoundary(pos),
  });
  state.clock.start();

  try {
    state.mic = await startMicBeat({
      bpmRange: [90, 160],
      onBeat: (bpm) => { state.clock?.setBpm(bpm); state.clock?.feedBeat(); },
      onTempo: (bpm) => { setBpmReadout(bpm); },
    });
  } catch (err) {
    setStatus(`microphone denied: ${String(err)} — using internal 128 BPM grid`);
    // Fall back to the clock's own interval grid so training still works.
    state.clock.stop();
    state.clock = new BeatClock({
      bpm: 128, beatsPerBar: 4, barsPerPhrase: 4,
      onBeat: (pos) => onBeatTick(pos),
      onPhrase: (pos) => onPhraseBoundary(pos),
    });
    state.clock.start();
  }

  // 5. First exercise agent.
  loadExercise(EXERCISE_ROSTER[FIRST_EXERCISE]);

  state.running = true;
  $('train-start').setAttribute('hidden', '');
  $('train-stop').removeAttribute('hidden');
  $('train-hud').removeAttribute('hidden');
  setStatus('training');

  // 6. Camera frame loop (produces HR/PCHR; the beat clock reads it at beats).
  state.rafId = requestAnimationFrame(frame);
}

function stop(): void {
  state.running = false;
  if (state.rafId !== null) { cancelAnimationFrame(state.rafId); state.rafId = null; }
  state.clock?.stop(); state.clock = null;
  state.mic?.stop(); state.mic = null;
  if (state.camera) { stopCamera(state.camera); state.camera = null; }
  state.glb?.destroy(); state.glb = null;
  destroyFaceLandmarker();
  state.active = null;
  state.effortSamples = [];
  $('train-stop').setAttribute('hidden', '');
  $('train-start').removeAttribute('hidden');
  $('train-hud').setAttribute('hidden', '');
  setStatus('stopped');
}

// ─── Camera frame → HR + PCHR ────────────────────────────────────────────────────

let lastFrameMs = performance.now();

async function frame(): Promise<void> {
  if (!state.running) return;
  const now = performance.now();
  const dt = Math.max(1, now - lastFrameMs);
  lastFrameMs = now;

  const video = $('train-video') as HTMLVideoElement;
  const gpuCanvas = $('train-gpu') as HTMLCanvasElement;

  if (state.rppg && state.gpu && video.readyState >= 2) {
    resizeCanvasToDisplay(gpuCanvas);
    const lm = detectFace(video, now);
    const roi = extractFaceROI(lm);
    if (roi) {
      const result = await state.rppg.tick(video, roi, 1000 / dt, now);
      if (result.globalBvp !== 0 || result.rcMean !== 0) state.bvp.push(result.globalBvp, now);
      const stats = state.bvp.compute();

      const colorSample = state.faceColor.sample(video, roi);
      const vaso = state.vaso.update(stats.relAmplitude, 1.0);
      if (colorSample) {
        state.lastSkin = inferSkinState(colorSample.combined, vaso, state.lastSkin);
      } else {
        state.lastSkin = { ...state.lastSkin, vasodilation: vaso };
      }
      const tSkin = skinTemperatureC(vaso);
      const spo2 = state.lastSkin.oxygenation;

      if (stats.hrBpm >= 30) {
        const pchr = pchrDecompose(stats.hrBpm, tSkin, spo2);
        state.latest.hr = stats.hrBpm;
        state.latest.dhrAutonomic = pchr.dHR_autonomic;
        state.latest.dhrMetabolic = pchr.dHR_metabolic;
        state.latest.dhrHypoxic = pchr.dHR_hypoxic;
      }
      state.latest.vasodilation = vaso;
      state.latest.tSkin = tSkin;
      state.latest.rcMean = result.rcMean;
    }
  }

  state.rafId = requestAnimationFrame(frame);
}

// ─── Beat tick: effort index + protocol + GLB tempo ──────────────────────────────

function onBeatTick(pos: BeatPosition): void {
  if (!state.clock) return;
  const bpm = state.clock.currentBpm;
  const effort = effortIndex(state.latest.hr, bpm);
  state.latest.effort = effort;
  state.effortSamples.push(effort);

  const protocol = protocolFor(pos);
  const regime = effortRegime(effort);

  // The GLB moves at the music's tempo, not the heart's.
  state.glb?.setTempoHz(bpm / 60);

  updateHud(bpm, effort, regime, protocol, pos);
}

// ─── Phrase boundary: the body makes its case ────────────────────────────────────

function onPhraseBoundary(_pos: BeatPosition): void {
  if (!state.active) return;
  const s = state.effortSamples;
  const meanEffort = s.length ? s.reduce((a, b) => a + b, 0) / s.length : 0;
  const trend = s.length >= 2 ? s[s.length - 1] - s[0] : 0;
  state.effortSamples = [];

  const evidence: PhraseEvidence = {
    effort: meanEffort,
    dhr_autonomic: state.latest.dhrAutonomic,
    dhr_metabolic: state.latest.dhrMetabolic,
    dhr_hypoxic: state.latest.dhrHypoxic,
    vasodilation: state.latest.vasodilation,
    t_skin: state.latest.tSkin,
    rc_mean: state.latest.rcMean,
    effort_trend: trend,
  };

  const verdict = evaluateTransition(state.active, evidence);
  logCase(`${state.active.label}: ${verdict.reason}`, verdict.relevant);

  if (verdict.relevant && verdict.next) {
    const next = EXERCISE_ROSTER[verdict.next];
    if (next) loadExercise(next);
    else finishSession();
  }
}

function loadExercise(agent: ExerciseAgent): void {
  state.active = agent;
  state.effortSamples = [];
  $('train-exercise').textContent = agent.label;
  $('train-target').textContent = `target effort ${agent.purpose.effortTarget.toFixed(2)}×`;
  logCase(`▶ ${agent.label} — bring your effort to ${agent.purpose.effortTarget.toFixed(2)}× the beat`, true);
}

function finishSession(): void {
  logCase('session complete — the body worked through every exercise', true);
  stop();
}

// ─── HUD ─────────────────────────────────────────────────────────────────────────

function updateHud(bpm: number, effort: number, regime: string, protocol: string, pos: BeatPosition): void {
  $('train-bpm').textContent = `${Math.round(bpm)} BPM`;
  $('train-hr').textContent = state.latest.hr >= 30 ? `${Math.round(state.latest.hr)} bpm` : '— bpm';
  $('train-effort').textContent = `${effort.toFixed(2)}×`;
  $('train-regime').textContent = regime;
  $('train-regime').className = `train-regime train-regime-${regime}`;
  $('train-barpos').textContent = `bar ${((pos.bar - 1) % 4) + 1}/4 · beat ${pos.beatInBar}/4`;
  $('train-protocol').textContent = protocol.replace('_', ' ');
}

function setBpmReadout(bpm: number): void {
  $('train-bpm').textContent = `${Math.round(bpm)} BPM`;
}

function logCase(msg: string, highlight: boolean): void {
  const el = $('train-log');
  const line = document.createElement('div');
  line.className = highlight ? 'train-log-line train-log-hi' : 'train-log-line';
  line.textContent = msg;
  el.prepend(line);
  while (el.childElementCount > 8) el.lastElementChild?.remove();
}

function setStatus(msg: string): void {
  $('train-status').textContent = msg;
}

// ─── Boot ─────────────────────────────────────────────────────────────────────────

$('train-begin-btn').addEventListener('click', () => { void start(); });
$('train-stop').addEventListener('click', () => stop());

// Register the service worker for installable / offline-capable behaviour.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/train-sw.js').catch(() => { /* offline-only nicety */ });
  });
}
