// Respiration analysis panel.
//
// Three live charts now:
//   1. Hb-O2 dissociation curve with current arterial operating point.
//   2. Alveolar gas equation: P_AO_2 vs altitude with current operating point.
//   3. Stat block with respiration rate, confidence, predicted P_AO2 and SpO2.
//
// Predicted SpO2 comes from the Hill model evaluated at the predicted P_AO2,
// which itself is the framework's forward prediction at the current altitude
// and PaCO2 (paco2 default 40 mmHg until we have it from a CO2 sensor).

import { mountHbDissociation, type HbCurveHandle } from '../charts/hb-dissociation';
import { mountAlveolarGas, type AlveolarGasHandle, alveolarPO2 } from '../charts/alveolar-gas';

const HILL_N = 2.7;
const P50 = 27;

function hbSat(pO2: number): number {
  const p = Math.pow(Math.max(0, pO2), HILL_N);
  return p / (Math.pow(P50, HILL_N) + p);
}

export interface LungsPanelHandle {
  update(stats: {
    rrBpm: number;
    respConfidence: number;
    altitudeM: number;     // 0 if unknown; later sourced from device sensors
    paco2: number;         // default 40 mmHg
  }): void;
  destroy(): void;
}

export function mountLungsPanel(): LungsPanelHandle {
  const body = document.getElementById('lungs-panel-body')!;
  body.innerHTML = '';

  const stat = document.createElement('div');
  stat.innerHTML = `
    <div class="row"><label>respiration</label><span data-k="rr">—</span><span class="unit">bpm</span></div>
    <div class="row"><label>resp confidence</label><span data-k="conf">—</span><span class="unit"></span></div>
    <div class="row"><label>altitude (assumed)</label><span data-k="alt">0</span><span class="unit">m</span></div>
    <div class="row"><label>P_AO₂ (model)</label><span data-k="pao2">—</span><span class="unit">mmHg</span></div>
    <div class="row"><label>SpO₂ (model)</label><span data-k="spo2">—</span><span class="unit">%</span></div>
  `;
  body.appendChild(stat);

  const dissBlock = document.createElement('div');
  dissBlock.className = 'chart-block';
  dissBlock.style.height = '220px';
  dissBlock.innerHTML = `<div class="chart-title">Hb-O₂ dissociation · operating point</div><div class="chart-host" style="flex:1;min-height:0"></div>`;
  body.appendChild(dissBlock);
  const diss: HbCurveHandle = mountHbDissociation(dissBlock.querySelector('.chart-host') as HTMLElement);

  const algBlock = document.createElement('div');
  algBlock.className = 'chart-block';
  algBlock.style.height = '220px';
  algBlock.innerHTML = `<div class="chart-title">alveolar gas equation · P_AO₂ vs altitude</div><div class="chart-host" style="flex:1;min-height:0"></div>`;
  body.appendChild(algBlock);
  const alg: AlveolarGasHandle = mountAlveolarGas(algBlock.querySelector('.chart-host') as HTMLElement);

  function update(stats: {
    rrBpm: number;
    respConfidence: number;
    altitudeM: number;
    paco2: number;
  }): void {
    const set = (k: string, v: string): void => {
      const el = body.querySelector(`[data-k="${k}"]`);
      if (el) el.textContent = v;
    };

    const pao2 = alveolarPO2(stats.altitudeM, stats.paco2);
    const sat = hbSat(pao2);

    set('rr', stats.rrBpm > 0 ? stats.rrBpm.toFixed(1) : '—');
    set('conf', stats.respConfidence > 0 ? stats.respConfidence.toFixed(2) : '—');
    set('alt', stats.altitudeM.toFixed(0));
    set('pao2', pao2.toFixed(0));
    set('spo2', (sat * 100).toFixed(1));

    diss.setPoint(pao2, sat);
    alg.setOperating(stats.altitudeM, stats.paco2);
  }

  function destroy(): void {
    diss.destroy();
    alg.destroy();
    body.innerHTML = '';
  }

  return { update, destroy };
}
