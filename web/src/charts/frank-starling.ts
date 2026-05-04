// Frank-Starling chart with live operating point.
//
// Plots stroke volume vs end-diastolic volume per cardiac-equations-of-state.tex
// Eq. \ref{eq:frank_starling} for fixed E_es and E_a:
//
//   SV(EDV) = E_es * h(L_s(EDV)) * (EDV - V_d) / (E_es * h(...) + E_a)
//
// We use the simplified plateau form (h ≈ 1 for EDV in 80–200 mL) to keep
// it visually clean; the operating point comes from the fitted state.

import * as d3 from 'd3';
import {
  makeChart,
  syncChartSize,
  plotWidth,
  plotHeight,
  observeResize,
} from './chart-utils';
import type { CardiacState } from '../physio/eos';

export interface FrankStarlingHandle {
  setState(state: CardiacState): void;
  destroy(): void;
}

export function mountFrankStarling(container: HTMLElement): FrankStarlingHandle {
  const frame = makeChart(container, { margin: { top: 8, right: 12, bottom: 26, left: 38 } });
  let state: CardiacState | null = null;

  function curve(s: CardiacState): Array<[number, number]> {
    const out: Array<[number, number]> = [];
    for (let edv = 60; edv <= 220; edv += 4) {
      // Plateau approximation: h(L_s) ~ 1 in physiological range.
      const sv = (s.Ees * (edv - s.Vd)) / (s.Ees + s.Ea);
      out.push([edv, Math.max(0, sv)]);
    }
    return out;
  }

  function draw(): void {
    if (!syncChartSize(container, frame) && frame.width === 0) return;
    const W = plotWidth(frame);
    const H = plotHeight(frame);
    if (W <= 0 || H <= 0) return;

    frame.g.selectAll('*').remove();

    const x = d3.scaleLinear().domain([50, 230]).range([0, W]);
    const y = d3.scaleLinear().domain([0, 160]).range([H, 0]);

    frame.g
      .append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(0,${H})`)
      .call(d3.axisBottom(x).ticks(5));
    frame.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5));

    frame.g
      .append('text')
      .attr('x', W / 2).attr('y', H + 22)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--dim)').style('font-size', '9px')
      .text('EDV (mL)');
    frame.g
      .append('text')
      .attr('x', -H / 2).attr('y', -28)
      .attr('text-anchor', 'middle').attr('transform', 'rotate(-90)')
      .attr('fill', 'var(--dim)').style('font-size', '9px')
      .text('SV (mL)');

    if (!state) return;

    // Curve at current contractility / afterload.
    const data = curve(state);
    const lineGen = d3
      .line<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(d[1]))
      .curve(d3.curveMonotoneX);
    frame.g.append('path').datum(data).attr('class', 'line').attr('d', lineGen);

    // Reference curve at rest values for visual comparison.
    const restCurve = curve({ ...state, Ees: 2.0, Ea: 1.3 });
    frame.g
      .append('path')
      .datum(restCurve)
      .attr('class', 'line-secondary')
      .attr('d', lineGen);

    // Operating point.
    const SV = (state.Ees * (state.EDV - state.Vd)) / (state.Ees + state.Ea);
    frame.g
      .append('circle')
      .attr('class', 'point')
      .attr('r', 5)
      .attr('cx', x(state.EDV))
      .attr('cy', y(SV));
    frame.g
      .append('text')
      .attr('x', x(state.EDV) + 8)
      .attr('y', y(SV) - 6)
      .attr('fill', 'var(--accent)')
      .style('font-size', '10px')
      .text(`SV ${SV.toFixed(0)} · EDV ${state.EDV.toFixed(0)}`);
  }

  draw();
  const stop = observeResize(container, draw);

  return {
    setState(s: CardiacState): void {
      state = s;
      draw();
    },
    destroy(): void {
      stop();
      frame.svg.remove();
    },
  };
}
