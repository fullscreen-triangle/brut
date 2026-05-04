// Respiration analysis panel content. Mounts the Hb-O2 dissociation curve
// with live operating point, plus respiration-rate readouts. More charts
// (alveolar gas, RSA spectrum, V/Q matching) come incrementally.

import { mountHbDissociation, type HbCurveHandle } from '../charts/hb-dissociation';

export interface LungsPanelHandle {
  update(stats: {
    rrBpm: number;
    respConfidence: number;
    spo2Estimate: number;     // 0..100, 0 if unknown
    arterialPO2: number;      // mmHg, default arterial
  }): void;
  destroy(): void;
}

export function mountLungsPanel(): LungsPanelHandle {
  const body = document.getElementById('lungs-panel-body')!;
  body.innerHTML = '';

  // Stat block.
  const statBlock = document.createElement('div');
  statBlock.innerHTML = `
    <div class="row"><label>respiration rate</label><span data-k="rr">—</span><span class="unit">bpm</span></div>
    <div class="row"><label>resp confidence</label><span data-k="conf">—</span><span class="unit"></span></div>
    <div class="row"><label>arterial P_O₂ (est)</label><span data-k="po2">—</span><span class="unit">mmHg</span></div>
    <div class="row"><label>SpO₂ (model)</label><span data-k="spo2">—</span><span class="unit">%</span></div>
  `;
  body.appendChild(statBlock);

  // Hb-O2 dissociation curve.
  const dissBlock = document.createElement('div');
  dissBlock.className = 'chart-block';
  dissBlock.style.height = '260px';
  dissBlock.innerHTML = `<div class="chart-title">Hb-O₂ dissociation · current operating point</div><div class="chart-host" style="flex:1;min-height:0"></div>`;
  body.appendChild(dissBlock);
  const dissHost = dissBlock.querySelector('.chart-host') as HTMLElement;
  const diss: HbCurveHandle = mountHbDissociation(dissHost);

  const placeholder = document.createElement('div');
  placeholder.style.color = 'var(--dim)';
  placeholder.style.fontSize = '10px';
  placeholder.innerHTML = 'next: alveolar gas equation · RSA spectrum · partition cascade waterfall · V/Q matching';
  body.appendChild(placeholder);

  function update(stats: {
    rrBpm: number;
    respConfidence: number;
    spo2Estimate: number;
    arterialPO2: number;
  }): void {
    const set = (k: string, v: string): void => {
      const el = body.querySelector(`[data-k="${k}"]`);
      if (el) el.textContent = v;
    };
    set('rr', stats.rrBpm > 0 ? stats.rrBpm.toFixed(1) : '—');
    set('conf', stats.respConfidence > 0 ? stats.respConfidence.toFixed(2) : '—');
    set('po2', stats.arterialPO2.toFixed(0));
    set('spo2', stats.spo2Estimate > 0 ? stats.spo2Estimate.toFixed(1) : '—');

    diss.setPoint(stats.arterialPO2, stats.spo2Estimate > 0 ? stats.spo2Estimate / 100 : undefined);
  }

  function destroy(): void {
    diss.destroy();
    body.innerHTML = '';
  }

  return { update, destroy };
}
