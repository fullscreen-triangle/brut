// Skin-optics forward + inverse model.
//
// Treats the forehead as a layered partition network whose photon traversal
// follows the same geometry that mass-transfer-mechanisms.tex derives for
// chromatographic columns and fluid viscosity. The microbiome paper's
// "categorical aperture" sits at layer 0 (stratum corneum + surface lipids).
//
// Layer stack (light incident from camera direction = display + ambient):
//
//   Layer 0   air → stratum corneum interface (Fresnel, n ≈ 1.55)
//             + microbiome lipid film (categorical aperture; quasi-loss-free).
//   Layer 1   stratum corneum / epidermis: melanin absorption dominates,
//             negligible at λ_red, moderate at λ_green, strong at λ_blue.
//   Layer 2   dermis with capillary network: hemoglobin absorption depends
//             on total Hb concentration, oxygenation, and effective optical
//             path length (which is what vasodilation modulates).
//   Layer 3   subcutaneous fat / muscle: effectively diffuse return.
//
// Forward (state → predicted RGB):
//
//   R_λ = ρ_specular(λ) + (1 − ρ_specular) · T_epi(λ)² · A_dermis(λ) · D_diff
//
// where A_dermis(λ) is the chamber's spectral albedo accounting for hemoglobin,
// T_epi(λ) is the epidermal transmittance (melanin + scattering), and
// D_diff captures back-diffusion. The squared transmittance accounts for the
// down-and-back photon path. This is a coarsened band-averaged model; the
// constants come from textbook absorption coefficients (Prahl, OMLC) for
// HbO₂ / Hb / melanin in the camera's RGB centre wavelengths.
//
// Inverse (observed RGB → state):
//
//   The full nonlinear inverse is not algebraically solvable, but the model
//   is well-conditioned for direct inversion via channel separation:
//
//     blue   →  melanin index               (Hb absorption ≪ melanin in blue)
//     red    →  Hb concentration            (melanin and oxygenation weak)
//     green  →  oxygenation, vasodilation   (Hb absorption dominant)
//
//   We solve sequentially: m from B, [Hb] from (R, m), SpO2 from (G, m, [Hb]).
//   Vasodilation is read separately from the AC component — supplied externally
//   by the rPPG amplitude estimator, not from the DC RGB.
//
// T_skin from vasodilation:
//
//   T_skin = T_resting + κ · (vasodilation_factor − 1)
//
// where T_resting ≈ 33 °C for resting forehead (conservative; per dermatology
// literature the resting forehead skin temperature in 22–24 °C ambient is
// 32–34 °C). κ ≈ 4 K per unit dilation factor — the characteristic dilation
// scale that rotates skin from cold (vasoconstriction) to hot (full dilation).

const T_RESTING_C = 33.0;       // resting forehead skin temperature (°C)
const KAPPA_T_K = 4.0;          // K per unit vasodilation factor
const VASO_TAU_S = 60.0;        // smoothing τ for vasodilation (s)

// Band-centre absorption coefficients (cm⁻¹ at typical chromophore concentrations).
// These are "effective" values pre-multiplied by characteristic concentrations
// for normal Caucasian skin; treat as the model's internal calibration.
//
//   λ_R ≈ 620 nm:  HbO₂ low,  Hb low,  melanin very low
//   λ_G ≈ 540 nm:  HbO₂ high (peak), Hb high (peak),  melanin moderate
//   λ_B ≈ 460 nm:  HbO₂ very high, Hb very high, melanin high
const ABS = {
  // hemoglobin (band-averaged, scaled per unit [Hb] in g/dL):
  hbR: 0.060,   hbG: 0.560,   hbB: 0.420,
  // oxygenation effect on green channel (HbO₂ peak shifts vs deoxyHb):
  hbO2_minus_hb_G: 0.180,
  // melanin (per unit melanin index 0..1):
  melR: 0.05,   melG: 0.18,   melB: 0.42,
  // baseline scattering / specular constant:
  specular: 0.04,
} as const;

// Effective forehead path length (cm) — round-trip through epidermis + dermis.
const PATH_CM = 0.20;

export interface SkinState {
  melanin: number;        // 0..1 — pigmentation index
  hbConc: number;         // g/dL — total hemoglobin concentration (resting ~14 g/dL)
  oxygenation: number;    // 0..1 — fraction HbO₂ / total
  vasodilation: number;   // 1.0 = baseline; < 1 vasoconstricted, > 1 dilated
  microbiomeR: number;    // categorical richness (microbiome paper); affects layer-0 transmittance
}

export const SKIN_BASELINE: SkinState = {
  melanin: 0.15,
  hbConc: 14.0,
  oxygenation: 0.97,
  vasodilation: 1.0,
  microbiomeR: 4.5,       // log10 — healthy median per microbiome paper Eq. 25
};

export function forwardRGB(s: SkinState): { r: number; g: number; b: number } {
  // Surface specular (Fresnel) — small, λ-independent for our coarsened bands.
  // Modulated mildly by microbiome layer thickness, modeled as deviation from
  // healthy log R median (the paper's Section 5).
  const microbiomeAttenuation = 1.0 - 0.04 * (s.microbiomeR - 4.5);
  const spec = ABS.specular * microbiomeAttenuation;

  // Melanin transmittance per channel.
  const tmR = Math.exp(-ABS.melR * s.melanin * PATH_CM * 10);
  const tmG = Math.exp(-ABS.melG * s.melanin * PATH_CM * 10);
  const tmB = Math.exp(-ABS.melB * s.melanin * PATH_CM * 10);

  // Hemoglobin absorption with oxygenation modulation (green only — HbO₂ peak).
  // Vasodilation amplifies the effective optical-path length through dermal blood.
  const hbEff = s.hbConc * s.vasodilation;
  const taR = Math.exp(-ABS.hbR * hbEff * PATH_CM);
  const taGdeoxy = Math.exp(-ABS.hbG * hbEff * PATH_CM);
  const taGoxy = Math.exp(-(ABS.hbG + ABS.hbO2_minus_hb_G) * hbEff * PATH_CM);
  const taG = (1 - s.oxygenation) * taGdeoxy + s.oxygenation * taGoxy;
  const taB = Math.exp(-ABS.hbB * hbEff * PATH_CM);

  // Diffuse return from subcutaneous fat — channel-independent, ~0.5.
  const D = 0.55;

  // Round-trip: epidermis transmittance squared × dermis absorption × diffuse return.
  const r = spec + (1 - spec) * tmR * tmR * taR * D;
  const g = spec + (1 - spec) * tmG * tmG * taG * D;
  const b = spec + (1 - spec) * tmB * tmB * taB * D;

  return {
    r: clamp01(r),
    g: clamp01(g),
    b: clamp01(b),
  };
}

/** Algebraic inversion. Solves layer-by-layer using channel separation. */
export function inferSkinState(
  rgb: { r: number; g: number; b: number },
  vasodilationFactor: number,
  prior: SkinState = SKIN_BASELINE,
): SkinState {
  // Avoid log(<=0).
  const r = Math.max(0.01, Math.min(0.99, rgb.r));
  const g = Math.max(0.01, Math.min(0.99, rgb.g));
  const b = Math.max(0.01, Math.min(0.99, rgb.b));

  const D = 0.55;
  const spec = ABS.specular * (1.0 - 0.04 * (prior.microbiomeR - 4.5));
  const lift = (c: number) => Math.max(0.01, (c - spec) / (1 - spec) / D);

  const lR = lift(r);
  const lG = lift(g);
  const lB = lift(b);

  // Solve B for melanin assuming Hb absorption is small in blue compared to melanin.
  // lB ≈ exp(−2 · ABS.melB · m · PATH · 10) · exp(−ABS.hbB · [Hb] · PATH)
  // We drop the Hb term in B (small) for a first cut, then refine.
  const melanin = clamp(
    -Math.log(Math.max(1e-3, lB)) / (2 * ABS.melB * PATH_CM * 10),
    0.02, 0.85,
  );

  // Solve R for [Hb] given melanin (R is dominated by Hb; melanin contribution is small).
  // lR = exp(−2·ABS.melR·m·P·10) · exp(−ABS.hbR · [Hb] · P)
  const lRcorrected = lR * Math.exp(2 * ABS.melR * melanin * PATH_CM * 10);
  const hbConc = clamp(
    -Math.log(Math.max(1e-3, lRcorrected)) / (ABS.hbR * PATH_CM),
    8, 20,
  );

  // Solve G for oxygenation given (melanin, [Hb]).
  // lG corrected for melanin attenuation.
  const lGcorrected = lG * Math.exp(2 * ABS.melG * melanin * PATH_CM * 10);
  // lGcorrected = (1-α)·exp(-hbG·[Hb]·P) + α·exp(-(hbG+hbO2-hb)·[Hb]·P)
  //   where α = oxygenation
  const tDeoxy = Math.exp(-ABS.hbG * hbConc * vasodilationFactor * PATH_CM);
  const tOxy   = Math.exp(-(ABS.hbG + ABS.hbO2_minus_hb_G) * hbConc * vasodilationFactor * PATH_CM);
  let oxygenation = (lGcorrected - tDeoxy) / Math.max(1e-6, tOxy - tDeoxy);
  oxygenation = clamp(oxygenation, 0.5, 1.0);

  return {
    melanin,
    hbConc,
    oxygenation,
    vasodilation: vasodilationFactor,
    microbiomeR: prior.microbiomeR,
  };
}

/**
 * Vasodilation tracker. The pulsatile component of the green channel relative
 * to a long-rolling baseline is the vasodilation proxy: more dilation = more
 * blood volume oscillating with each heartbeat = bigger AC swing.
 *
 * We smooth with τ = 60 s so it tracks thermoregulatory changes (10s of
 * seconds) and not individual heartbeats.
 */
export class VasodilationTracker {
  private value = 1.0;
  private lastTickMs = performance.now();

  update(currentRelAmplitude: number, baselineRelAmplitude = 1.0): number {
    const t = performance.now();
    const dt = Math.max(0, (t - this.lastTickMs) / 1000);
    this.lastTickMs = t;

    // Map rel_amplitude to a vasodilation factor.
    //   rel_amplitude = 1.0 → baseline → factor 1.0
    //   rel_amplitude > 1   → vasodilation
    //   rel_amplitude < 1   → vasoconstriction
    // We compress the mapping (sqrt) to avoid huge swings from amplitude noise.
    const target = Math.sqrt(Math.max(0.1, currentRelAmplitude / Math.max(0.1, baselineRelAmplitude)));

    // First-order low-pass with τ = VASO_TAU_S.
    const alpha = 1 - Math.exp(-dt / VASO_TAU_S);
    this.value = this.value * (1 - alpha) + target * alpha;

    return this.value;
  }

  current(): number {
    return this.value;
  }

  reset(): void {
    this.value = 1.0;
  }
}

/** Skin temperature from vasodilation factor. Bounded to [27, 37] °C. */
export function skinTemperatureC(vasodilation: number): number {
  const t = T_RESTING_C + KAPPA_T_K * (vasodilation - 1.0);
  return Math.max(27, Math.min(37, t));
}

/** Q_10 metabolic factor for a given skin temperature, relative to T_REF (37 °C). */
export function metabolicQ10(skinTC: number, q10 = 2.3, refTC = 37.0): number {
  return Math.pow(q10, (skinTC - refTC) / 10);
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}
