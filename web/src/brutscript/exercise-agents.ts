// Exercise agents — the NPC layer that reads the body at beat-synchronised
// intervals and decides when the exercise changes.
//
// Each exercise type is a separate agent. An agent has:
//
//   • a PURPOSE — a target physiological state it exists to bring the athlete
//     to (the attractor of its drive, in the sense of the companion account
//     "The Purpose of a Character"). Concretely: a target effort band and the
//     PCHR signature that band should produce.
//
//   • an ANALYSIS SCRIPT — its own BrutScript, wiring exactly the inversions
//     and decompositions this exercise needs, in the sequence it needs them.
//     The agent "decides the analysis to carry out and the sequence" by *being*
//     that script. Different exercises measure different things.
//
// The athlete never asks to change exercises. At each phrase boundary the
// accumulated physiological evidence — HR decomposition, vasodilation, skin-
// temperature trend, effort index — is the argument the body makes to the
// current exercise agent. The agent evaluates that argument against its own
// reasons for the current exercise. If the argument is sufficient, the exercise
// changes. This is the two-factor relevance test of "The Physiology of
// Response": a transition is committed only if the evidence both
//
//   (a) advances the athlete toward the agent's purpose  (purpose gain > 0), and
//   (b) is physiologically coherent                       (a real exertion
//       signature, not warmth, anxiety, or a spoofed heart rate).
//
// Neither factor alone suffices. A high HR from a warm face (metabolic drive
// without autonomic drive) fails coherence; hitting the number by anxiety
// (autonomic spike with no metabolic/vasodilation trajectory) fails coherence
// too. The layered rPPG decomposition is exactly what makes the coherence
// factor checkable — it is the state amplitude-only rPPG cannot recover.

// ─── Physiological evidence (the body's argument) ──────────────────────────────

/**
 * The evidence accumulated over a phrase, read at beat-synchronised intervals.
 * These are the env values the analysis script produces; the agent reads them
 * off the live BrutScript at the phrase boundary.
 */
export interface PhraseEvidence {
  /** HR/BPM effort index over the phrase (mean). */
  effort: number;
  /** Autonomic drive component ΔHRa (genuine neural drive). */
  dhr_autonomic: number;
  /** Metabolic (thermal) drive component ΔHRm. */
  dhr_metabolic: number;
  /** Hypoxic drive component ΔHRo. */
  dhr_hypoxic: number;
  /** Vasodilation factor η. */
  vasodilation: number;
  /** Skin-temperature proxy, °C. */
  t_skin: number;
  /** Cross-channel coherence / regime confidence (rc_mean-like), 0..1. */
  rc_mean: number;
  /** Trend in effort across the phrase (slope sign): rising / flat / falling. */
  effort_trend: number;
}

// ─── Exercise agent ────────────────────────────────────────────────────────────

/** A candidate a body's argument can move the session toward. */
export interface ExerciseAgent {
  id: string;
  label: string;
  /** The analysis script this agent runs while it is active. */
  script: string;
  /**
   * The agent's purpose: the target effort band and required drive signature.
   * Purpose gain is how far the evidence has moved into (or past) this band.
   */
  purpose: {
    /** Effort index the athlete should be sustaining under this exercise. */
    effortTarget: number;
    /** Half-width of the acceptable band around the target. */
    effortBand: number;
    /**
     * The exercise is "achieved" (ready to advance) when effort has reached
     * this multiple of the target — i.e. the body has made its case that this
     * exercise is done and the next should begin.
     */
    advanceAt: number;
  };
  /** Exercises this agent can hand off to, in preference order. */
  successors: string[];
}

// ─── Two-factor relevance ──────────────────────────────────────────────────────

export interface TransitionVerdict {
  /** Purpose gain: signed distance the evidence has moved toward advancing. */
  purposeGain: number;
  /** Whether the evidence advances toward the agent's purpose. */
  purposeAdvancing: boolean;
  /** Coherence margin: ≥ 0 means the exertion signature is physiologically real. */
  coherenceMargin: number;
  /** Whether the evidence is physiologically coherent. */
  coherent: boolean;
  /** Relevant iff both factors clear their bars — only then is a transition committed. */
  relevant: boolean;
  /** Chosen successor exercise id, when relevant. */
  next: string | null;
  /** Human-readable account of the decision, for the trace/console. */
  reason: string;
}

/**
 * Coherence test on the body's argument. A real exertion has a *structured*
 * PCHR signature: autonomic drive dominates, and the metabolic/vasodilation
 * trajectory is consistent with genuine work rather than a static thermal or
 * emotional offset. Returns a margin (≥ 0 coherent, < 0 incoherent) so the
 * verdict can report how comfortably the argument passed.
 */
export function coherenceMargin(e: PhraseEvidence): number {
  // Genuine exertion: autonomic drive is the largest single component and is
  // clearly positive. A warm face alone shows up as metabolic drive with weak
  // autonomic drive; an anxiety spike shows autonomic drive with no supporting
  // vasodilation/metabolic trajectory. Require all three to agree.
  const autonomicDominant = e.dhr_autonomic - Math.max(e.dhr_metabolic, e.dhr_hypoxic);
  const vasodilationSupports = (e.vasodilation - 1.0);   // work dilates: η > 1
  const channelTrust = e.rc_mean - 0.85;                 // signal actually locked
  // The margin is the weakest of the corroborating signals: coherence fails if
  // any one of them is absent, matching "all routes must agree" (vanishing
  // holonomy) from the physiology account.
  return Math.min(autonomicDominant / 10, vasodilationSupports, channelTrust);
}

/**
 * Evaluate the body's phrase-boundary argument against the active agent's
 * purpose. Commit a transition only if it is two-factor relevant.
 */
export function evaluateTransition(agent: ExerciseAgent, e: PhraseEvidence): TransitionVerdict {
  // Purpose gain: how far effort has pushed past the advance threshold. Positive
  // means the body is arguing this exercise is achieved and the next should run.
  const advanceEffort = agent.purpose.effortTarget * agent.purpose.advanceAt;
  const purposeGain = e.effort - advanceEffort;
  const purposeAdvancing = purposeGain > 0 && e.effort_trend >= 0;

  const margin = coherenceMargin(e);
  const coherent = margin >= 0;

  const relevant = purposeAdvancing && coherent;
  const next = relevant ? (agent.successors[0] ?? null) : null;

  let reason: string;
  if (!purposeAdvancing && !coherent) {
    reason = `hold: effort ${e.effort.toFixed(2)} below advance ${advanceEffort.toFixed(2)} and signature incoherent (margin ${margin.toFixed(2)})`;
  } else if (!purposeAdvancing) {
    reason = `hold: coherent exertion but effort ${e.effort.toFixed(2)} has not cleared advance ${advanceEffort.toFixed(2)}`;
  } else if (!coherent) {
    reason = `hold: effort target met but signature incoherent (margin ${margin.toFixed(2)}) — warmth/anxiety/spoof, not work`;
  } else {
    reason = `advance → ${next}: body cleared effort ${e.effort.toFixed(2)} ≥ ${advanceEffort.toFixed(2)} with coherent exertion (margin ${margin.toFixed(2)})`;
  }

  return { purposeGain, purposeAdvancing, coherenceMargin: margin, coherent, relevant, next, reason };
}

// ─── The exercise roster ───────────────────────────────────────────────────────
//
// Each agent's script wires exactly what that exercise needs. All three share
// the beat-gated signal bus (bpm, beat, bar_pos, phrase, effort) the sandbox
// provides; they differ in which decompositions they run and what they watch.

const WARMUP_SCRIPT = `-- Exercise agent: WARM-UP
-- Purpose: bring the athlete into entrainment (HR tracking BPM).
-- Runs the light protocol — baseline + effort index only.

source rppg    { signal bvp, rc_mean, hr, rmssd; rate 30hz }
source face_rgb { signal r, g, b; rate 1hz }
source beat    { signal bpm, beat, bar_pos, phrase, effort }

layer skin_optics from face_rgb {
  invert melanin      from b   using beer_lambert.blue
  invert vasodilation from bvp using beer_lambert.blue baseline 30s sqrt_compress
  derive t_skin = 33.0 + 4.0 * (vasodilation - 1.0) clamp [27, 37]
}

decompose warmup from rppg, beat {
  term effort_idx = effort
  regime = classify(effort) {
    working    when effort >= 1.05
    entrained  when effort >= 0.92
    recovering otherwise
  }
}

watch entrained {
  when effort >= 0.92 and rc_mean >= 0.90
  emit "entrained" confidence 0.8
}
`;

const INTERVAL_SCRIPT = `-- Exercise agent: INTERVALS
-- Purpose: sustain work above the beat (HR > BPM) with a real autonomic drive.
-- Runs the full PCHR decomposition — this exercise needs to prove the exertion.

source rppg     { signal bvp, rc_mean, hr, rmssd, spo2; rate 30hz }
source face_rgb { signal r, g, b; rate 1hz }
source beat     { signal bpm, beat, bar_pos, phrase, effort }

layer skin_optics from face_rgb {
  invert melanin      from b   using beer_lambert.blue
  invert hb_conc      from r   given melanin using beer_lambert.red
  invert spo2         from g   given melanin, hb_conc using beer_lambert.green
  invert vasodilation from bvp using beer_lambert.blue baseline 30s sqrt_compress
  derive t_skin = 33.0 + 4.0 * (vasodilation - 1.0) clamp [27, 37]
}

decompose pchr from rppg, skin_optics {
  term hr_intrinsic  = baseline(hr, 300, 5)
  term dhr_metabolic = 0.08 * (t_skin - 33.0) * hr_intrinsic
  term dhr_hypoxic   = 0.15 * (1.0 - spo2) * hr_intrinsic
  term dhr_autonomic = hr - hr_intrinsic - dhr_metabolic - dhr_hypoxic
}

watch working_hard {
  when effort >= 1.10 and dhr_autonomic > 15.0
  emit "working_hard" confidence 0.85
}
`;

const COOLDOWN_SCRIPT = `-- Exercise agent: COOL-DOWN
-- Purpose: bring the athlete below the beat (HR < BPM), recovery underway.

source rppg     { signal bvp, rc_mean, hr, rmssd; rate 30hz }
source face_rgb { signal r, g, b; rate 1hz }
source beat     { signal bpm, beat, bar_pos, phrase, effort }

layer skin_optics from face_rgb {
  invert vasodilation from bvp using beer_lambert.blue baseline 30s sqrt_compress
  derive t_skin = 33.0 + 4.0 * (vasodilation - 1.0) clamp [27, 37]
}

decompose cooldown from rppg, beat {
  term effort_idx = effort
  regime = classify(effort) {
    working    when effort >= 1.05
    entrained  when effort >= 0.92
    recovering otherwise
  }
}

watch recovered {
  when effort < 0.90
  emit "recovered" confidence 0.8
}
`;

export const EXERCISE_ROSTER: Record<string, ExerciseAgent> = {
  warmup: {
    id: 'warmup',
    label: 'Warm-up',
    script: WARMUP_SCRIPT,
    purpose: { effortTarget: 0.92, effortBand: 0.08, advanceAt: 1.0 },
    successors: ['intervals'],
  },
  intervals: {
    id: 'intervals',
    label: 'Intervals',
    script: INTERVAL_SCRIPT,
    purpose: { effortTarget: 1.10, effortBand: 0.10, advanceAt: 1.0 },
    successors: ['cooldown'],
  },
  cooldown: {
    id: 'cooldown',
    label: 'Cool-down',
    script: COOLDOWN_SCRIPT,
    purpose: { effortTarget: 0.85, effortBand: 0.08, advanceAt: 1.0 },
    successors: [],
  },
};

export const FIRST_EXERCISE = 'warmup';
