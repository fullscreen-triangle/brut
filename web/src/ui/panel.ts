// Live readout panel. Pure DOM mutation, no framework.

import type { BvpStats } from '../physio/bvp';
import { classifyRegime, failureMode, type Regime } from '../physio/regimes';

const REGIME_COLOR: Record<Regime, string> = {
  'phase-locked': '#7fd47f',
  'coherent':     '#5fafff',
  'cascade':      '#ffd75f',
  'aperture':     '#ffaf5f',
  'turbulent':    '#ff5f5f',
};

const refs = {
  hr: document.getElementById('hr')!,
  rmssd: document.getElementById('rmssd')!,
  rc: document.getElementById('rc')!,
  regime: document.getElementById('regime')!,
  sk: document.getElementById('sk')!,
  st: document.getElementById('st')!,
  se: document.getElementById('se')!,
  snr: document.getElementById('snr')!,
  rcspread: document.getElementById('rcspread')!,
};

export function renderPanel(stats: BvpStats, fieldStats: { rcMean: number; rcStd: number; snr: number }): void {
  if (stats.beats >= 2) {
    refs.hr.textContent = stats.hrBpm.toFixed(1);
    refs.rmssd.textContent = stats.rmssdMs.toFixed(0);
    refs.rc.textContent = stats.rc.toFixed(3);
    const regime = classifyRegime(stats.rc);
    refs.regime.textContent = regime;
    (refs.regime as HTMLElement).style.color = REGIME_COLOR[regime];
    const mode = failureMode(stats.rc, stats.se);
    if (mode) {
      refs.regime.textContent += `  (${mode})`;
    }
  } else {
    refs.hr.textContent = '—';
    refs.rmssd.textContent = '—';
    refs.rc.textContent = '—';
    refs.regime.textContent = `acquiring (${(stats.filled * 100).toFixed(0)}%)`;
    (refs.regime as HTMLElement).style.color = '';
  }

  refs.sk.textContent = stats.sk > 0 ? stats.sk.toFixed(3) : '—';
  refs.st.textContent = stats.st > 0 ? stats.st.toFixed(3) : '—';
  refs.se.textContent = stats.se > 0 ? stats.se.toFixed(3) : '—';

  // Spatial spread of R_c -> the novel observation channel.
  refs.rcspread.textContent = fieldStats.rcStd > 0 ? fieldStats.rcStd.toFixed(3) : '—';
  // SNR proxy from autocorrelation peak vs band mean.
  refs.snr.textContent = fieldStats.snr !== 0 ? (10 * Math.log10(Math.max(1e-6, fieldStats.snr + 1))).toFixed(1) : '—';
}
