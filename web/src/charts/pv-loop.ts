// Pressure-Volume loop with ESPVR / EDPVR overlays
// per cardiac-equations-of-state.tex Section 4 + Theorem 4.1, 4.2.
//
//   P_es = E_es * (V - V_d)            (ESPVR — linear)
//   P_ed = alpha * (exp(beta*(V-V0))-1) (EDPVR — exponential)
//
// We render the static fans first; the live operating loop updates as the
// UKF produces fitted state. Until then we draw a representative loop at
// rest values from the cardiac equations of state paper.

import * as d3 from 'd3';
import {
  makeChart,
  syncChartSize,
  plotWidth,
  plotHeight,
  observeResize,
} from './chart-utils';

interface PvLoopState {
  // ESPVR
  Ees: number;       // mmHg/mL
  Vd: number;        // mL
  // EDPVR
  alpha: number;     // mmHg
  beta: number;      // 1/mL
  V0: number;        // mL
  // Operating loop
  Edv: number;       // mL
  Esv: number;       // mL
  Pmax: number;      // mmHg, peak ejection pressure
}

const REST_STATE: PvLoopState = {
  Ees: 2.0,
  Vd: 10,
  alpha: 0.7,
  beta: 0.04,
  V0: 60,
  Edv: 120,
  Esv: 50,
  Pmax: 120,
};

export interface PvLoopHandle {
  setState(s: Partial<PvLoopState>): void;
  destroy(): void;
}

export function mountPvLoop(container: HTMLElement): PvLoopHandle {
  const frame = makeChart(container, { margin: { left: 38, bottom: 28, top: 8, right: 12 } });
  let state: PvLoopState = { ...REST_STATE };

  function pvLoop(s: PvLoopState): Array<[number, number]> {
    // Idealized 4-segment loop: filling -> isovolumic contraction ->
    // ejection -> isovolumic relaxation. We approximate with a 60-point
    // closed curve through the corner pressures.
    const pts: Array<[number, number]> = [];
    const PfillStart = edpvr(s, s.Esv);
    const PfillEnd = edpvr(s, s.Edv);
    const Pes = espvr(s, s.Esv);

    // Filling: V Esv -> Edv at EDPVR pressure
    for (let i = 0; i <= 20; i++) {
      const t = i / 20;
      const V = s.Esv + t * (s.Edv - s.Esv);
      pts.push([V, edpvr(s, V)]);
    }
    // Isovolumic contraction at V=Edv from PfillEnd up to Pmax
    for (let i = 0; i <= 8; i++) {
      const t = i / 8;
      pts.push([s.Edv, PfillEnd + t * (s.Pmax - PfillEnd)]);
    }
    // Ejection: V Edv -> Esv at near-constant Pmax sloping to Pes
    for (let i = 0; i <= 20; i++) {
      const t = i / 20;
      const V = s.Edv - t * (s.Edv - s.Esv);
      const P = s.Pmax + t * (Pes - s.Pmax);
      pts.push([V, P]);
    }
    // Isovolumic relaxation at V=Esv from Pes down to PfillStart
    for (let i = 0; i <= 8; i++) {
      const t = i / 8;
      pts.push([s.Esv, Pes - t * (Pes - PfillStart)]);
    }
    return pts;
  }

  function espvr(s: PvLoopState, V: number): number {
    return Math.max(0, s.Ees * (V - s.Vd));
  }

  function edpvr(s: PvLoopState, V: number): number {
    if (V <= s.V0) return 0;
    return s.alpha * (Math.exp(s.beta * (V - s.V0)) - 1);
  }

  function draw(): void {
    if (!syncChartSize(container, frame) && frame.width === 0) return;
    const W = plotWidth(frame);
    const H = plotHeight(frame);
    if (W <= 0 || H <= 0) return;

    frame.g.selectAll('*').remove();

    const x = d3.scaleLinear().domain([0, 200]).range([0, W]);
    const y = d3.scaleLinear().domain([0, 160]).range([H, 0]);

    // Axes.
    frame.g
      .append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(0,${H})`)
      .call(d3.axisBottom(x).ticks(5));
    frame.g
      .append('g')
      .attr('class', 'axis')
      .call(d3.axisLeft(y).ticks(5));

    // Axis labels.
    frame.g
      .append('text')
      .attr('x', W / 2)
      .attr('y', H + 22)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--dim)')
      .style('font-size', '9px')
      .text('volume (mL)');
    frame.g
      .append('text')
      .attr('x', -H / 2)
      .attr('y', -28)
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('fill', 'var(--dim)')
      .style('font-size', '9px')
      .text('pressure (mmHg)');

    // ESPVR line.
    const espvrPts: Array<[number, number]> = [];
    for (let V = state.Vd; V <= 220; V += 5) espvrPts.push([V, espvr(state, V)]);
    const lineGen = d3
      .line<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(Math.min(d[1], 160)));
    frame.g
      .append('path')
      .datum(espvrPts)
      .attr('class', 'line-secondary')
      .attr('d', lineGen);

    // EDPVR curve.
    const edpvrPts: Array<[number, number]> = [];
    for (let V = state.V0; V <= 220; V += 2) edpvrPts.push([V, edpvr(state, V)]);
    frame.g
      .append('path')
      .datum(edpvrPts)
      .attr('class', 'line-secondary')
      .attr('d', lineGen);

    // Operating loop.
    const loopPts = pvLoop(state);
    frame.g
      .append('path')
      .datum(loopPts)
      .attr('class', 'line')
      .attr('d', lineGen.curve(d3.curveCatmullRomClosed));
  }

  draw();
  const stop = observeResize(container, draw);

  return {
    setState(s: Partial<PvLoopState>): void {
      state = { ...state, ...s };
      draw();
    },
    destroy(): void {
      stop();
      frame.svg.remove();
    },
  };
}
