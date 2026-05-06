// Melanopic-lux tracker.
//
// Models the user's circadian-disrupting screen exposure under the framework's
// premise that we know exactly what we render. The browser doesn't give us
// direct framebuffer luminance (without a costly readback every frame), so we
// estimate from the page's average rendered colour — which is dominated by:
//
//   - body background (black in stealth, dark grey otherwise)
//   - drawer background (rgba dark with alpha)
//   - chart colours (small area)
//   - glb canvases (variable but small fraction of screen)
//
// Per frame:
//   1. Sample the visible page as an averaged RGB colour (approximated from
//      computed styles and known UI footprints; refinement via canvas readback
//      is a v-next).
//   2. Compute melanopic flux:  m_flux = peak_lux * (0.06·R + 0.34·G + 0.60·B)
//      where the weighting approximates the melanopsin action spectrum
//      (peak ~480 nm) on sRGB primaries.
//   3. Multiply by time-of-day sensitivity. Human melanopic sensitivity to
//      circadian phase shifts roughly follows a cosine: peak ~3 am, trough ~3 pm.
//   4. Integrate dt to get cumulative mlux·hours since session start.
//
// Output is pushed once per second into the dashboard via main.ts.

const DISPLAY_PEAK_LUX = 300;        // typical bright-room display, cd/m^2 ≈ lux at the eye

export interface MelanopicSample {
  flux: number;            // current melanopic illuminance (mlux)
  sensitivity: number;     // current circadian sensitivity, 0..1
  weightedFlux: number;    // flux * sensitivity
  cumulativeMlxHours: number; // session integral of weightedFlux over hours
  isStealth: boolean;
  pageLuminanceFrac: number; // 0..1 — estimated fraction of peak page is currently emitting
}

export class MelanopicTracker {
  private cumulativeMlxH = 0;
  private lastTickMs = performance.now();

  /** Approximate average page RGB by summing CSS-declared backgrounds weighted by visible area. */
  private samplePageRgb(): { r: number; g: number; b: number } {
    // Body background dominates. Stealth class drives it pure black.
    const bg = parseRgb(getComputedStyle(document.body).backgroundColor);

    // Add a small contribution from open panels (chart text, accent lines).
    let acc = { r: bg.r, g: bg.g, b: bg.b };
    const openPanels = document.querySelectorAll('.side-panel.open, .drawer.open');
    if (openPanels.length > 0) {
      // A rough boost toward the panel-bg colour (rgba(8,12,18,0.55)) and
      // the accent blue used for chart strokes.
      acc = blend(acc, { r: 8, g: 12, b: 18 }, 0.15);
      acc = blend(acc, { r: 95, g: 175, b: 255 }, 0.04);
    }

    // Glb canvases are typically dimly coloured — small contribution.
    acc = blend(acc, { r: 28, g: 24, b: 30 }, 0.05);

    return acc;
  }

  /** Time-of-day sensitivity. Peak ~3 am, trough ~3 pm. */
  private circadianSensitivity(): number {
    const now = new Date();
    const h = now.getHours() + now.getMinutes() / 60 + now.getSeconds() / 3600;
    return 0.5 * (1 - Math.cos(((h - 3) * Math.PI) / 12));
  }

  tick(): MelanopicSample {
    const t = performance.now();
    const dtHours = Math.max(0, (t - this.lastTickMs) / 1000 / 3600);
    this.lastTickMs = t;

    const rgb = this.samplePageRgb();
    const r = rgb.r / 255;
    const g = rgb.g / 255;
    const b = rgb.b / 255;
    const pageLuminanceFrac = 0.06 * r + 0.34 * g + 0.60 * b;
    const flux = DISPLAY_PEAK_LUX * pageLuminanceFrac;

    const sens = this.circadianSensitivity();
    const weighted = flux * sens;
    this.cumulativeMlxH += weighted * dtHours;

    return {
      flux,
      sensitivity: sens,
      weightedFlux: weighted,
      cumulativeMlxHours: this.cumulativeMlxH,
      isStealth: document.body.classList.contains('stealth'),
      pageLuminanceFrac,
    };
  }
}

function parseRgb(str: string): { r: number; g: number; b: number } {
  // Handles "rgb(0, 0, 0)" and "rgba(8, 12, 18, 0.55)".
  const m = str.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
  if (!m) return { r: 0, g: 0, b: 0 };
  return { r: +m[1], g: +m[2], b: +m[3] };
}

function blend(
  a: { r: number; g: number; b: number },
  b: { r: number; g: number; b: number },
  t: number,
): { r: number; g: number; b: number } {
  return {
    r: a.r * (1 - t) + b.r * t,
    g: a.g * (1 - t) + b.g * t,
    b: a.b * (1 - t) + b.b * t,
  };
}
