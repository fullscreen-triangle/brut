// Cardiac analysis panel content. Mounts the canonical cardiac charts from
// the BRUT publications. Right now: PV loop with ESPVR/EDPVR overlay,
// plus a live readout block. More charts (Frank-Starling, regime bar,
// PCHR decomposition) get added incrementally.

import { mountPvLoop, type PvLoopHandle } from '../charts/pv-loop';

export interface HeartPanelHandle {
  update(stats: { hrBpm: number; rmssd: number; rc: number; sk: number; st: number; se: number; regime: string }): void;
  destroy(): void;
}

export function mountHeartPanel(): HeartPanelHandle {
  const body = document.getElementById('heart-panel-body')!;
  body.innerHTML = '';

  // Stat block.
  const statBlock = document.createElement('div');
  statBlock.innerHTML = `
    <div class="row"><label>HR</label><span data-k="hr">—</span><span class="unit">bpm</span></div>
    <div class="row"><label>RMSSD</label><span data-k="rmssd">—</span><span class="unit">ms</span></div>
    <div class="row"><label>R_c</label><span data-k="rc">—</span><span class="unit"></span></div>
    <div class="row"><label>regime</label><span data-k="regime">—</span><span class="unit"></span></div>
    <div class="row"><label>S_k / S_t / S_e</label><span data-k="sentropy">—</span><span class="unit"></span></div>
  `;
  body.appendChild(statBlock);

  // PV loop chart.
  const pvBlock = document.createElement('div');
  pvBlock.className = 'chart-block';
  pvBlock.style.height = '280px';
  pvBlock.innerHTML = `<div class="chart-title">P-V loop · ESPVR / EDPVR</div><div class="chart-host" style="flex:1;min-height:0"></div>`;
  body.appendChild(pvBlock);
  const pvHost = pvBlock.querySelector('.chart-host') as HTMLElement;
  const pv: PvLoopHandle = mountPvLoop(pvHost);

  // Placeholder for the next chart cohort.
  const placeholder = document.createElement('div');
  placeholder.style.color = 'var(--dim)';
  placeholder.style.fontSize = '10px';
  placeholder.innerHTML = 'next: Frank-Starling · PCHR decomposition · regime ribbon · S-entropy 3D scatter';
  body.appendChild(placeholder);

  function update(stats: { hrBpm: number; rmssd: number; rc: number; sk: number; st: number; se: number; regime: string }): void {
    const set = (k: string, v: string): void => {
      const el = body.querySelector(`[data-k="${k}"]`);
      if (el) el.textContent = v;
    };
    set('hr', stats.hrBpm > 0 ? stats.hrBpm.toFixed(1) : '—');
    set('rmssd', stats.rmssd > 0 ? stats.rmssd.toFixed(0) : '—');
    set('rc', stats.rc > 0 ? stats.rc.toFixed(3) : '—');
    set('regime', stats.regime);
    set('sentropy', stats.sk > 0 ? `${stats.sk.toFixed(2)} / ${stats.st.toFixed(2)} / ${stats.se.toFixed(2)}` : '—');

    // For now we don't yet drive the PV loop from a fitted state — only
    // adjust EDV scaling roughly with HR as a placeholder until the UKF
    // produces real E_es / EDV / ESV. This will be replaced when the
    // forward-model fit lands.
    if (stats.hrBpm > 30) {
      const edv = clamp(160 - 0.6 * (stats.hrBpm - 60), 80, 200);
      const esv = clamp(70 - 0.4 * (stats.hrBpm - 60), 20, 110);
      pv.setState({ Edv: edv, Esv: esv });
    }
  }

  function destroy(): void {
    pv.destroy();
    body.innerHTML = '';
  }

  return { update, destroy };
}

function clamp(x: number, a: number, b: number): number {
  return Math.max(a, Math.min(b, x));
}
