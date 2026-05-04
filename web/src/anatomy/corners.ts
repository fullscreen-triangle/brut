// Concrete heart, lungs, and echo-plane glb widgets, plus the click-to-invoke
// wiring. The corners (TL/TR) carry the system-overview anatomies; the
// echo-plane strip in the centre carries the model-based echocardiogram —
// three standard imaging planes (A2C, A5C, PSAX) all tempo-locked to the
// fitted cardiac state.

import { mountGlb, type GlbWidget } from './glb';

const HEART_GLB = '/glb/heart__animated_anatomical_3d_model.glb';
const LUNGS_GLB = '/glb/lungs_exhale_front_view.glb';

const ECHO_A2C_GLB = '/glb/cardiac_anatomy_apical_2_chamber_echo_plane.glb';
const ECHO_A5C_GLB = '/glb/cardiac_anatomy_apical_5_chamber_echo_plane.glb';
const ECHO_PSAX_GLB = '/glb/cardiac_anatomy_psax_aortic_valve_echo_plane.glb';

const HEART_BASELINE_BPM = 60;
const LUNGS_BASELINE_BPM = 12;

export interface AnatomyHandles {
  heart: GlbWidget;
  lungs: GlbWidget;
  echo: {
    a2c: GlbWidget;
    a5c: GlbWidget;
    psax: GlbWidget;
  };
  setHeartHr(bpm: number): void;       // drives heart corner + all echo planes
  setLungsRespRate(bpm: number): void;
  destroy(): void;
}

export interface AnatomyCanvases {
  heartCanvas: HTMLCanvasElement;
  lungsCanvas: HTMLCanvasElement;
  echoA2cCanvas: HTMLCanvasElement;
  echoA5cCanvas: HTMLCanvasElement;
  echoPsaxCanvas: HTMLCanvasElement;
}

export async function mountAnatomyCorners(canvases: AnatomyCanvases): Promise<AnatomyHandles> {
  const [heart, lungs, a2c, a5c, psax] = await Promise.all([
    mountGlb({
      url: HEART_GLB,
      canvas: canvases.heartCanvas,
      framePadding: 1.5,
      ambientYawRate: 0.35,
    }),
    mountGlb({
      url: LUNGS_GLB,
      canvas: canvases.lungsCanvas,
      framePadding: 1.4,
      ambientYawRate: 0.18,
    }),
    mountGlb({
      url: ECHO_A2C_GLB,
      canvas: canvases.echoA2cCanvas,
      framePadding: 1.35,
      ambientYawRate: 0.0,
    }),
    mountGlb({
      url: ECHO_A5C_GLB,
      canvas: canvases.echoA5cCanvas,
      framePadding: 1.35,
      ambientYawRate: 0.0,
    }),
    mountGlb({
      url: ECHO_PSAX_GLB,
      canvas: canvases.echoPsaxCanvas,
      framePadding: 1.35,
      ambientYawRate: 0.0,
    }),
  ]);

  // Until we have a fitted HR / respiration, no tempo: animations stay paused.
  // (Echo planes don't ambient-rotate; they're meant to look like a clinical
  // imaging plane — still until they have a heartbeat to display.)
  for (const w of [heart, lungs, a2c, a5c, psax]) w.setTempoHz(0);

  return {
    heart,
    lungs,
    echo: { a2c, a5c, psax },
    setHeartHr(bpm: number): void {
      heart.setTempoFromBpm(bpm, HEART_BASELINE_BPM);
      a2c.setTempoFromBpm(bpm, HEART_BASELINE_BPM);
      a5c.setTempoFromBpm(bpm, HEART_BASELINE_BPM);
      psax.setTempoFromBpm(bpm, HEART_BASELINE_BPM);
    },
    setLungsRespRate(bpm: number): void {
      lungs.setTempoFromBpm(bpm, LUNGS_BASELINE_BPM);
    },
    destroy(): void {
      heart.destroy();
      lungs.destroy();
      a2c.destroy();
      a5c.destroy();
      psax.destroy();
    },
  };
}
