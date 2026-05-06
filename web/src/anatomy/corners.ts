// Concrete heart, lungs, and echo-plane glb widgets, plus the click-to-invoke
// wiring. The four cardiac glbs share one StrainMaterial whose uniforms are
// driven from the fitted CardiacState — making the rendered surface IS the
// observation of regional strain (per observation-computation.tex), not a
// separately-computed visualisation.

import { Color } from 'three';
import { mountGlb, type GlbWidget } from './glb';
import {
  createStrainMaterial,
  updateStrainUniforms,
  type StrainMaterial,
  type CardiacFitInput,
} from './strain-shader';

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
  setHeartHr(bpm: number): void;
  setLungsRespRate(bpm: number): void;
  /**
   * Push the fitted cardiac state into the shared StrainMaterial. The phase
   * uniform advances autonomously inside each glb's render loop; the
   * non-phase uniforms (Ees, Ea, EDV, ESV, EF, R_c, HR) come from here.
   */
  setCardiacFit(fit: CardiacFitInput): void;
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
  // One shared strain material drives all four cardiac glbs. Each glb gets
  // its own onTick callback that copies its local phase counter into the
  // material's uPhase uniform every render frame. Since the renders are
  // serialised (each glb has its own RAF), the most recent write wins for
  // any given paint, which is what we want — they're all driven at the
  // same tempo so phases agree.
  const strain: StrainMaterial = createStrainMaterial(new Color(0.95, 0.92, 1.0));

  const onCardiacTick = (phase: number, t: number): void => {
    strain.uniforms.uPhase.value = phase;
    strain.uniforms.uTime.value = t;
  };

  const [heart, lungs, a2c, a5c, psax] = await Promise.all([
    mountGlb({
      url: HEART_GLB,
      canvas: canvases.heartCanvas,
      framePadding: 1.5,
      ambientYawRate: 0.35,
      overrideMaterial: strain.material,
      onTick: onCardiacTick,
      pulseAmplitude: 0.06,
      pulseStyle: 'cardiac',
    }),
    mountGlb({
      url: LUNGS_GLB,
      canvas: canvases.lungsCanvas,
      framePadding: 1.45,
      ambientYawRate: 0.18,
      pulseAmplitude: 0.10,
      pulseStyle: 'respiratory',
    }),
    mountGlb({
      url: ECHO_A2C_GLB,
      canvas: canvases.echoA2cCanvas,
      framePadding: 1.4,
      ambientYawRate: 0.0,
      overrideMaterial: strain.material,
      onTick: onCardiacTick,
      pulseAmplitude: 0.07,
      pulseStyle: 'cardiac',
    }),
    mountGlb({
      url: ECHO_A5C_GLB,
      canvas: canvases.echoA5cCanvas,
      framePadding: 1.4,
      ambientYawRate: 0.0,
      overrideMaterial: strain.material,
      onTick: onCardiacTick,
      pulseAmplitude: 0.07,
      pulseStyle: 'cardiac',
    }),
    mountGlb({
      url: ECHO_PSAX_GLB,
      canvas: canvases.echoPsaxCanvas,
      framePadding: 1.4,
      ambientYawRate: 0.0,
      overrideMaterial: strain.material,
      onTick: onCardiacTick,
      pulseAmplitude: 0.07,
      pulseStyle: 'cardiac',
    }),
  ]);

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
    setCardiacFit(fit: CardiacFitInput): void {
      updateStrainUniforms(strain, fit);
    },
    destroy(): void {
      heart.destroy();
      lungs.destroy();
      a2c.destroy();
      a5c.destroy();
      psax.destroy();
      strain.material.dispose();
    },
  };
}
