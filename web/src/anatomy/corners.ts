// Concrete heart + lungs corner widgets and the click-to-invoke wiring.

import { mountGlb, type GlbWidget } from './glb';

const HEART_GLB = '/glb/heart__animated_anatomical_3d_model.glb';
const LUNGS_GLB = '/glb/lungs_exhale_front_view.glb';

const HEART_BASELINE_BPM = 60;     // animation clip's "natural" cadence assumption
const LUNGS_BASELINE_BPM = 12;     // ~12 breaths/min as the natural lungs cadence

export interface AnatomyHandles {
  heart: GlbWidget;
  lungs: GlbWidget;
  setHeartHr(bpm: number): void;
  setLungsRespRate(bpm: number): void;
  destroy(): void;
}

export async function mountAnatomyCorners(
  heartCanvas: HTMLCanvasElement,
  lungsCanvas: HTMLCanvasElement,
): Promise<AnatomyHandles> {
  // Mount in parallel so first paint is fast.
  const [heart, lungs] = await Promise.all([
    mountGlb({
      url: HEART_GLB,
      canvas: heartCanvas,
      framePadding: 1.5,
      ambientYawRate: 0.35,
    }),
    mountGlb({
      url: LUNGS_GLB,
      canvas: lungsCanvas,
      framePadding: 1.4,
      ambientYawRate: 0.18,
    }),
  ]);

  // Until we have a fitted HR / respiration, show ambient yaw at clip speed 0.
  heart.setTempoHz(0);
  lungs.setTempoHz(0);

  return {
    heart,
    lungs,
    setHeartHr(bpm: number): void {
      heart.setTempoFromBpm(bpm, HEART_BASELINE_BPM);
    },
    setLungsRespRate(bpm: number): void {
      lungs.setTempoFromBpm(bpm, LUNGS_BASELINE_BPM);
    },
    destroy(): void {
      heart.destroy();
      lungs.destroy();
    },
  };
}
