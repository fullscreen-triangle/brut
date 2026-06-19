// Animated ECG-pulse SVG component.
//
// Ports the stroke-dashoffset travelling-signal technique from
// publications/sources/digital-pulse.html + pulse.css into a reusable
// TypeScript factory. Each pulse type (pulsar/jugular/bleed/flat) maps to a
// different dasharray and default animation speed, matching the physiological
// cadences they represent. Callers can override the duration in real-time so
// the animation tracks a live measured signal (e.g. camera-derived HR).

export type PulseType  = 'pulsar' | 'jugular' | 'bleed' | 'flat';
export type PulseState = 'live' | 'idle' | 'warn' | 'dead';

interface PulseCfg {
  dasharray: number;
  duration: number; // seconds
}

// Values taken directly from pulse.css. The 814 dashoffset swing is the
// total path length; individual dasharrays control segment visibility.
const CFG: Record<PulseType, PulseCfg> = {
  pulsar:  { dasharray: 281, duration: 2.5 },
  jugular: { dasharray: 497, duration: 1.4 },
  bleed:   { dasharray: 437, duration: 1.2 },
  flat:    { dasharray: 814, duration: 10  },
};

// Identical to the path in digital-pulse.html — ECG-like waveform on a
// 0–519 x 60–135 canvas, baseline at y=90.
const ECG_PATH =
  'M0,90L250,90Q257,60 262,87T267,95 270,88 273,92t6,35 7,-60T290,127 297,107' +
  's2,-11 10,-10 1,1 8,-10T319,95c6,4 8,-6 10,-17s2,10 9,11h210';

export interface PulseSvgHandle {
  el: SVGSVGElement;
  setState(s: PulseState): void;
  setDuration(seconds: number): void;
}

export function createPulseSvg(
  type: PulseType,
  opts: { viewBox?: string; strokeWidth?: number } = {}
): PulseSvgHandle {
  const cfg = CFG[type];
  const ns = 'http://www.w3.org/2000/svg';

  const svg = document.createElementNS(ns, 'svg') as SVGSVGElement;
  svg.setAttribute('viewBox', opts.viewBox ?? '100 60 420 76');
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.setAttribute('xmlns', ns);
  svg.classList.add('pulse-svg', `pulse-${type}`, 'idle');
  svg.style.setProperty('--pulse-dasharray', String(cfg.dasharray));
  svg.style.setProperty('--pulse-duration', `${cfg.duration}s`);

  const path = document.createElementNS(ns, 'path') as SVGPathElement;
  path.setAttribute('d', ECG_PATH);
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke-width', String(opts.strokeWidth ?? 2));
  path.setAttribute('stroke-linejoin', 'round');
  path.classList.add('pulse-path');
  svg.appendChild(path);

  return {
    el: svg,
    setState(s: PulseState): void {
      svg.classList.remove('live', 'idle', 'warn', 'dead');
      svg.classList.add(s);
    },
    setDuration(seconds: number): void {
      svg.style.setProperty('--pulse-duration', `${Math.max(0.3, seconds)}s`);
    },
  };
}
