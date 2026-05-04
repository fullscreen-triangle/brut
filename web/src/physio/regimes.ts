// Five-regime classification for the cardiac Kuramoto order parameter R_c.
// Boundaries follow the wearable-recalibrated thresholds reported in the
// cardio-neural-integration validation (Oura Ring, 86 nights).

export type Regime = 'turbulent' | 'aperture' | 'cascade' | 'coherent' | 'phase-locked';

const RC_BOUNDS: Array<{ regime: Regime; min: number }> = [
  { regime: 'phase-locked', min: 0.947 },
  { regime: 'coherent',     min: 0.930 },
  { regime: 'cascade',      min: 0.900 },
  { regime: 'aperture',     min: 0.850 },
  { regime: 'turbulent',    min: 0.000 },
];

export function classifyRegime(rc: number): Regime {
  for (const b of RC_BOUNDS) if (rc >= b.min) return b.regime;
  return 'turbulent';
}

// Two-failure-mode discriminator (cardio-neural-integration thm. 2):
//   D_rigid = (1 - S_e) * 1{R_c > 0.95}
// Returns 'rigidity' if pathological phase-locking is plausible, 'decoherence'
// if R_c is collapsing, otherwise null.
export function failureMode(rc: number, se: number): 'rigidity' | 'decoherence' | null {
  if (rc > 0.95 && se < 0.5) return 'rigidity';
  if (rc < 0.30) return 'decoherence';
  return null;
}
