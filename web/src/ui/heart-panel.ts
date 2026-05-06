// Cardiac analysis panel.
//
// Now driven by the fitted cardiac state from physio/eos.ts. The PV loop and
// Frank-Starling chart both read from the fitted (E_es, E_a, EDV) and the
// derived hemodynamics block displays everything that follows.

import { mountPvLoop, type PvLoopHandle } from '../charts/pv-loop';
import { mountFrankStarling, type FrankStarlingHandle } from '../charts/frank-starling';
import type { CardiacState, DerivedHemodynamics, PchrDecomposition } from '../physio/eos';
import { derive } from '../physio/eos';

export interface HeartPanelStats {
  rmssd: number;
  rc: number;
  sk: number;
  st: number;
  se: number;
  regimeRc: string;
  // Skin-optics derived metabolic inputs
  T_skin_C?: number;
  vasodilation?: number;
  spo2_proxy?: number;       // 0..1
  pchr?: PchrDecomposition;
}

export interface HeartPanelHandle {
  update(state: CardiacState, stats: HeartPanelStats): void;
  destroy(): void;
}

export function mountHeartPanel(): HeartPanelHandle {
  const body = document.getElementById('heart-panel-body')!;
  body.innerHTML = '';

  // Live state readout.
  const stat = document.createElement('div');
  stat.innerHTML = `
    <div class="row"><label>HR</label><span data-k="hr">—</span><span class="unit">bpm</span></div>
    <div class="row"><label>SV / EF</label><span data-k="svef">—</span><span class="unit">mL · %</span></div>
    <div class="row"><label>CO</label><span data-k="co">—</span><span class="unit">L/min</span></div>
    <div class="row"><label>EDV / ESV</label><span data-k="vols">—</span><span class="unit">mL</span></div>
    <div class="row"><label>E_es / E_a</label><span data-k="elast">—</span><span class="unit">mmHg/mL</span></div>
    <div class="row"><label>E_es / E_a ratio</label><span data-k="ratio">—</span><span class="unit"></span></div>
    <div class="row"><label>P_es / P_ed</label><span data-k="press">—</span><span class="unit">mmHg</span></div>
    <div class="row"><label>MAP · pulse</label><span data-k="map">—</span><span class="unit">mmHg</span></div>
    <div class="row"><label>stroke work</label><span data-k="sw">—</span><span class="unit">mmHg·mL</span></div>
    <div class="row"><label>T_skin (model)</label><span data-k="tskin">—</span><span class="unit">°C</span></div>
    <div class="row"><label>vasodilation</label><span data-k="vaso">—</span><span class="unit"></span></div>
    <div class="row"><label>SpO₂ (model)</label><span data-k="spo2pchr">—</span><span class="unit">%</span></div>
    <div class="row"><label>HR intrinsic</label><span data-k="hrint">—</span><span class="unit">bpm</span></div>
    <div class="row"><label>ΔHR met / hypox / auto</label><span data-k="hrcomps">—</span><span class="unit">bpm</span></div>
    <div class="row"><label>EOS regime</label><span data-k="eosregime">—</span><span class="unit"></span></div>
    <div class="row"><label>R_c regime</label><span data-k="rcregime">—</span><span class="unit"></span></div>
    <div class="row"><label>RMSSD · R_c</label><span data-k="hrv">—</span><span class="unit"></span></div>
    <div class="row"><label>S_k / S_t / S_e</label><span data-k="sentropy">—</span><span class="unit"></span></div>
  `;
  body.appendChild(stat);

  // PV loop.
  const pvBlock = document.createElement('div');
  pvBlock.className = 'chart-block';
  pvBlock.style.height = '260px';
  pvBlock.innerHTML = `<div class="chart-title">P-V loop · ESPVR / EDPVR (fitted)</div><div class="chart-host" style="flex:1;min-height:0"></div>`;
  body.appendChild(pvBlock);
  const pv: PvLoopHandle = mountPvLoop(pvBlock.querySelector('.chart-host') as HTMLElement);

  // Frank-Starling.
  const fsBlock = document.createElement('div');
  fsBlock.className = 'chart-block';
  fsBlock.style.height = '220px';
  fsBlock.innerHTML = `<div class="chart-title">Frank-Starling · operating point on inferred curve</div><div class="chart-host" style="flex:1;min-height:0"></div>`;
  body.appendChild(fsBlock);
  const fs: FrankStarlingHandle = mountFrankStarling(fsBlock.querySelector('.chart-host') as HTMLElement);

  function update(state: CardiacState, hpStats: HeartPanelStats): void {
    const set = (k: string, v: string): void => {
      const el = body.querySelector(`[data-k="${k}"]`);
      if (el) el.textContent = v;
    };
    // No live fit yet — show '—' across the readout instead of REST_STATE
    // defaults, which would mislead the user into thinking those are their
    // numbers. The PV loop and Frank-Starling charts also pause updates.
    if (state.HR <= 0) {
      const dash = ['hr', 'svef', 'co', 'vols', 'elast', 'ratio', 'press', 'map', 'sw',
                    'tskin', 'vaso', 'spo2pchr', 'hrint', 'hrcomps',
                    'eosregime', 'rcregime', 'hrv', 'sentropy'];
      for (const k of dash) set(k, '—');
      return;
    }

    const d: DerivedHemodynamics = derive(state);
    set('hr', state.HR.toFixed(1));
    set('svef', `${d.SV.toFixed(0)} · ${(d.EF * 100).toFixed(0)}`);
    set('co', d.CO.toFixed(2));
    set('vols', `${state.EDV.toFixed(0)} · ${d.ESV.toFixed(0)}`);
    set('elast', `${state.Ees.toFixed(2)} · ${state.Ea.toFixed(2)}`);
    set('ratio', d.EesEaRatio.toFixed(2));
    set('press', `${d.Pes.toFixed(0)} · ${d.Ped.toFixed(0)}`);
    set('map', `${d.MAP.toFixed(0)} · ${d.pulse.toFixed(0)}`);
    set('sw', d.SW.toFixed(0));
    set('tskin', hpStats.T_skin_C !== undefined ? hpStats.T_skin_C.toFixed(1) : '—');
    set('vaso', hpStats.vasodilation !== undefined ? hpStats.vasodilation.toFixed(2) : '—');
    set('spo2pchr', hpStats.spo2_proxy !== undefined ? (hpStats.spo2_proxy * 100).toFixed(1) : '—');
    if (hpStats.pchr) {
      set('hrint', hpStats.pchr.HR_intrinsic.toFixed(0));
      const fmt = (x: number): string => `${x >= 0 ? '+' : ''}${x.toFixed(1)}`;
      set('hrcomps', `${fmt(hpStats.pchr.dHR_metabolic)} · ${fmt(hpStats.pchr.dHR_hypoxic)} · ${fmt(hpStats.pchr.dHR_autonomic)}`);
    } else {
      set('hrint', '—');
      set('hrcomps', '—');
    }
    set('eosregime', d.regime);
    set('rcregime', hpStats.regimeRc);
    set('hrv', hpStats.rmssd > 0 ? `${hpStats.rmssd.toFixed(0)} ms · ${hpStats.rc.toFixed(3)}` : '—');
    set('sentropy', hpStats.sk > 0
      ? `${hpStats.sk.toFixed(2)} / ${hpStats.st.toFixed(2)} / ${hpStats.se.toFixed(2)}`
      : '—');

    pv.setState({
      Ees: state.Ees,
      Vd: state.Vd,
      alpha: state.alpha,
      beta: state.beta,
      V0: state.V0,
      Edv: state.EDV,
      Esv: d.ESV,
      Pmax: state.Pmax,
    });
    fs.setState(state);
  }

  function destroy(): void {
    pv.destroy();
    fs.destroy();
    body.innerHTML = '';
  }

  return { update, destroy };
}
