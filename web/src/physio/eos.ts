// Cardiac equations of state — forward and inverse.
//
// Implements the closed-form relations from cardiac-equations-of-state.tex:
//
//   ESPVR:         P_es  = E_es * (V_es - V_d)
//   EDPVR:         P_ed  = alpha * (exp(beta * (V - V_0)) - 1)
//   V-A coupling:  SV    = E_es * (EDV - V_d) / (E_es + E_a)
//   Conservation:  CO    = HR * SV
//                  ESV   = EDV - SV
//                  EF    = SV / EDV
//   Stroke work:   SW    = SV * (P_es - P_ed_avg)
//
// The forward direction (`derive`) is rigorous: given a state, we evaluate
// the equations and return all dependent quantities.
//
// The inverse direction (`inferState`) takes BVP-derived observations and
// algebraically recovers the state under the framework's strong prior. We
// fix the EDPVR parameters and dead volume V_d to their resting paper
// defaults and treat (HR, E_es, E_a, EDV) as the inferable state, with
// physiologically meaningful update rules:
//
//   HR     <- direct observation
//   SV     <- prior_SV * (rel_amplitude)
//   E_es   <- prior_E_es * (1 + 0.04 * (HR - HR_rest))   inotropy with HR
//   E_a    <- prior_E_a * (1 + dicrotic_shift)           afterload from notch
//   EDV    <- algebraic inversion of V-A coupling for the inferred SV
//
// This is not curve-fitting and there are no free parameters. Every
// constant comes from cardiac-equations-of-state.tex Section 9 + companion
// derivations. The strong prior is *the framework itself*.

export interface CardiacState {
  // Inferable state ----------------------------------------------------
  HR: number;        // bpm
  Ees: number;       // mmHg/mL — end-systolic elastance
  Ea: number;        // mmHg/mL — arterial elastance
  EDV: number;       // mL — end-diastolic volume
  // Fixed defaults (paper baselines, can be adjusted later) ------------
  Vd: number;        // mL — dead volume
  alpha: number;     // mmHg
  beta: number;      // 1/mL
  V0: number;        // mL
  Pmax: number;      // mmHg — peak ejection pressure (used for SW)
}

/** Resting state, matching cardiac-equations-of-state.tex Section 9.1. */
export const REST_STATE: CardiacState = {
  HR: 70,
  Ees: 2.0,
  Ea: 1.3,
  EDV: 120,
  Vd: 10,
  alpha: 0.7,
  beta: 0.04,
  V0: 60,
  Pmax: 120,
};

export interface DerivedHemodynamics {
  SV: number;          // mL
  ESV: number;         // mL
  EF: number;          // 0..1
  CO: number;          // L/min
  Pes: number;         // mmHg — end-systolic pressure
  Ped: number;         // mmHg — end-diastolic pressure
  MAP: number;         // mmHg — approximate mean arterial pressure
  pulse: number;       // mmHg — pulse pressure
  SW: number;          // mmHg·mL — stroke work
  EesEaRatio: number;  // dimensionless — V-A coupling ratio
  regime: CardiacRegime;
}

export type CardiacRegime =
  | 'rest'
  | 'submaximal-exercise'
  | 'maximal-exercise'
  | 'compensated-systolic-HF'
  | 'hypertension'
  | 'hypovolemia'
  | 'distributive-shock';

export function derive(s: CardiacState): DerivedHemodynamics {
  const SV = (s.Ees * (s.EDV - s.Vd)) / (s.Ees + s.Ea);
  const ESV = s.EDV - SV;
  const EF = SV / Math.max(1e-6, s.EDV);
  const CO = (s.HR * SV) / 1000;
  const Pes = s.Ees * (ESV - s.Vd);
  const Ped = s.alpha * (Math.exp(s.beta * (s.EDV - s.V0)) - 1);
  // MAP ≈ diastolic + (1/3) * pulse pressure ≈ Pes - (1/3) * (Pes - Ped)
  const pulse = Math.max(0, Pes - Ped);
  const MAP = Ped + pulse / 3;
  const SW = SV * (Pes - Ped) * 0.5;        // approximate PV-loop area
  const ratio = s.Ees / s.Ea;
  const regime = classifyRegime(s.HR, ratio, EF, ESV, s.EDV);

  return { SV, ESV, EF, CO, Pes, Ped, MAP, pulse, SW, EesEaRatio: ratio, regime };
}

/**
 * Regime classification per cardiac-equations-of-state.tex Section 9.
 * Decision rules use HR + V-A coupling ratio + ejection fraction to map onto
 * the seven regimes from the paper. This is a deterministic mapping; for the
 * cardio-neural-integration paper's R_c-based classification, see
 * physio/regimes.ts.
 */
export function classifyRegime(
  HR: number,
  ratio: number,
  EF: number,
  ESV: number,
  EDV: number,
): CardiacRegime {
  if (EF < 0.4 && EDV > 180) return 'compensated-systolic-HF';
  if (HR > 160 && ratio > 1.0) return 'maximal-exercise';
  if (HR > 100 && HR <= 160) return 'submaximal-exercise';
  if (ratio < 0.7 && EF > 0.55) return 'distributive-shock';
  if (ratio > 1.8 && ESV / EDV < 0.5 && HR < 80) return 'hypertension';
  if (EDV < 90 && HR > 90) return 'hypovolemia';
  return 'rest';
}

// ── Inverse: BVP features → state ──────────────────────────────────────

export interface BvpFeatures {
  HR_bpm: number;            // direct observation
  rel_amplitude: number;     // current AC amplitude / 30 s baseline (1.0 = baseline)
  rel_dicrotic?: number;     // dicrotic notch position 0..1; 0.5 = baseline
                             // (lower = earlier notch = stiffer arteries / higher E_a)
  /**
   * Skin temperature in °C, optional. When supplied, drives Q_10 metabolic
   * decomposition of HR per sensor-disambiguation.tex Eq. 22 (PCHR):
   *
   *   ΔHR_met  = α_T · ΔT_skin · HR₀
   *   ΔHR_auto = HR_obs − HR_intrinsic − ΔHR_met − ΔHR_O₂
   *
   * α_T ≈ 0.08 °C⁻¹ from Q_10 ≈ 2.3 (sensor-disambiguation Theorem 2.1).
   */
  T_skin_C?: number;
  /** Optional SpO₂ proxy (0..1). Drives the hypoxic ΔHR_O2 term. */
  SpO2?: number;
}

/** PCHR decomposition: how the observed HR splits into physiological drivers. */
export interface PchrDecomposition {
  HR_obs: number;        // bpm — the raw observation
  HR_intrinsic: number;  // bpm — sino-atrial pacemaker baseline
  dHR_metabolic: number; // bpm — metabolic / thermal drive (Q_10 term)
  dHR_hypoxic: number;   // bpm — hypoxic compensation
  dHR_autonomic: number; // bpm — autonomic residual (sympathovagal balance)
}

const ALPHA_T_PER_C = 0.08;     // °C⁻¹ — sensor-disambiguation Theorem 2.1
const BETA_O2 = 0.15;           // per 10% SpO₂ drop — sensor-disambiguation Theorem 2.2
const T_REF_C = 33.0;           // resting forehead reference (°C)
const HR_INTRINSIC_DEFAULT = 60;

export function pchrDecompose(
  HR_obs: number,
  T_skin_C: number | undefined,
  SpO2: number | undefined,
  HR_intrinsic = HR_INTRINSIC_DEFAULT,
): PchrDecomposition {
  const dT = T_skin_C !== undefined ? T_skin_C - T_REF_C : 0;
  const dHR_metabolic = ALPHA_T_PER_C * dT * HR_intrinsic;
  const dHR_hypoxic = SpO2 !== undefined ? BETA_O2 * (1 - SpO2) * HR_intrinsic : 0;
  const dHR_autonomic = HR_obs - HR_intrinsic - dHR_metabolic - dHR_hypoxic;
  return {
    HR_obs,
    HR_intrinsic,
    dHR_metabolic,
    dHR_hypoxic,
    dHR_autonomic,
  };
}

/**
 * Algebraic inversion of the cardiac equations of state given BVP-derived
 * observations. Closed-form, no iterative fitting.
 *
 * The trick is that we treat the framework's published rest-state values as
 * the strong prior and only let observations move the parameters along
 * specific axes:
 *
 *   - HR moves directly (we observe it).
 *   - SV moves with rel_amplitude (PPG amplitude scales with stroke volume
 *     up to a constant gain that we treat as captured by the baseline
 *     normalisation).
 *   - E_es scales mildly with HR via inotropy (Eq. M2/E4 of the paper).
 *   - E_a scales with dicrotic-notch shift (timing of aortic valve closure
 *     in the Windkessel response is set by C_a and TPR; rel_dicrotic
 *     parameterises this).
 *   - EDV is then the unique value that closes the V-A coupling identity
 *     SV = E_es*(EDV - V_d)/(E_es + E_a) for the inferred SV.
 */
export function inferState(obs: BvpFeatures, prior: CardiacState = REST_STATE): CardiacState {
  const HR = clamp(obs.HR_bpm, 30, 220);
  const HR_rest = prior.HR;

  // Subtract metabolic and hypoxic drives before scaling inotropy. The
  // autonomic residual (HR_obs − HR_intrinsic − ΔHR_met − ΔHR_O2) is what
  // actually reflects sympathetic activation; using raw HR overestimates
  // contractility when the user is just warm.
  const pchr = pchrDecompose(HR, obs.T_skin_C, obs.SpO2, prior.HR);
  const autonomicHR = pchr.HR_intrinsic + pchr.dHR_autonomic;

  // Inotropy: E_es scales with the autonomic-driven part of HR (paper Eq. M2).
  const inotropyFactor = 1 + 0.04 * (autonomicHR - HR_rest);
  const Ees = clamp(prior.Ees * inotropyFactor, 0.4, 8.0);

  // Afterload: dicrotic notch position carries E_a information.
  // Default to baseline if the BVP feature wasn't extractable.
  const dicShift = obs.rel_dicrotic !== undefined
    ? clamp(0.5 - obs.rel_dicrotic, -0.4, 0.4)
    : 0;
  const Ea = clamp(prior.Ea * (1 + 1.5 * dicShift), 0.3, 4.0);

  // Stroke volume from amplitude.
  const SV_baseline = (prior.Ees * (prior.EDV - prior.Vd)) / (prior.Ees + prior.Ea);
  const SV = clamp(SV_baseline * obs.rel_amplitude, 10, 200);

  // EDV from inverted V-A coupling identity:
  //   SV = E_es * (EDV - V_d) / (E_es + E_a)
  //   => EDV = SV * (E_es + E_a) / E_es + V_d
  const EDV = clamp((SV * (Ees + Ea)) / Math.max(1e-6, Ees) + prior.Vd, 60, 250);

  // Pmax: peak ejection pressure tracks afterload + contractility.
  const Pmax = clamp(120 * (Ea / prior.Ea) * (Ees / prior.Ees), 60, 240);

  return {
    ...prior,
    HR,
    Ees,
    Ea,
    EDV,
    Pmax,
  };
}

/** Exponential moving-average smoother for the cardiac state vector. */
export class StateSmoother {
  private state: CardiacState | null = null;

  constructor(private alpha = 0.18) {}

  update(next: CardiacState): CardiacState {
    if (this.state === null) {
      this.state = { ...next };
      return this.state;
    }
    const a = this.alpha;
    const s = this.state;
    s.HR = ema(s.HR, next.HR, a);
    s.Ees = ema(s.Ees, next.Ees, a);
    s.Ea = ema(s.Ea, next.Ea, a);
    s.EDV = ema(s.EDV, next.EDV, a);
    s.Pmax = ema(s.Pmax, next.Pmax, a);
    return s;
  }

  reset(): void {
    this.state = null;
  }
}

function ema(prev: number, next: number, a: number): number {
  return prev * (1 - a) + next * a;
}

function clamp(x: number, lo: number, hi: number): number {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}
