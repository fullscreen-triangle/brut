// Alveolar gas equation chart per cardiovascular-derivation.tex Eq. \ref{eq:alveolar_gas}.
//
//   P_AO_2 = F_I O_2 (P_atm - P_H2O) - P_A CO_2 / R
//
// We plot P_AO_2 versus altitude (which lowers P_atm exponentially with
// scale height H ≈ 8 km), with the user's current operating point marked.
// This makes the framework's prediction visible: as altitude rises, alveolar
// O_2 falls, and the Hb dissociation curve in the same panel will sit at a
// lower saturation point.

import * as d3 from 'd3';
import {
  makeChart,
  syncChartSize,
  plotWidth,
  plotHeight,
  observeResize,
} from './chart-utils';

const FIO2 = 0.21;
const PH2O = 47;       // mmHg at 37 °C
const RER = 0.8;       // respiratory exchange ratio (mixed diet)
const SCALE_HEIGHT_M = 8000;

function patmAtAltitude(altM: number): number {
  // Barometric formula simplified.
  return 760 * Math.exp(-altM / SCALE_HEIGHT_M);
}

function alveolarPO2(altM: number, paco2: number = 40): number {
  const patm = patmAtAltitude(altM);
  return FIO2 * (patm - PH2O) - paco2 / RER;
}

export interface AlveolarGasHandle {
  setOperating(altM: number, paco2?: number): void;
  destroy(): void;
}

export function mountAlveolarGas(container: HTMLElement): AlveolarGasHandle {
  const frame = makeChart(container, { margin: { top: 8, right: 14, bottom: 26, left: 38 } });
  let altM = 0;
  let paco2 = 40;

  function draw(): void {
    if (!syncChartSize(container, frame) && frame.width === 0) return;
    const W = plotWidth(frame);
    const H = plotHeight(frame);
    if (W <= 0 || H <= 0) return;

    frame.g.selectAll('*').remove();

    const x = d3.scaleLinear().domain([0, 9000]).range([0, W]);
    const y = d3.scaleLinear().domain([0, 120]).range([H, 0]);

    frame.g
      .append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(0,${H})`)
      .call(d3.axisBottom(x).ticks(5).tickFormat((d) => `${(+d / 1000).toFixed(0)} km`));
    frame.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5));

    frame.g
      .append('text')
      .attr('x', W / 2).attr('y', H + 22)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--dim)').style('font-size', '9px')
      .text('altitude');
    frame.g
      .append('text')
      .attr('x', -H / 2).attr('y', -28)
      .attr('text-anchor', 'middle').attr('transform', 'rotate(-90)')
      .attr('fill', 'var(--dim)').style('font-size', '9px')
      .text('P_AO₂ (mmHg)');

    // Curve.
    const data: Array<[number, number]> = [];
    for (let h = 0; h <= 9000; h += 100) data.push([h, alveolarPO2(h, paco2)]);
    const lineGen = d3
      .line<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(Math.max(0, d[1])));
    frame.g.append('path').datum(data).attr('class', 'line').attr('d', lineGen);

    // Reference altitude markers.
    const refs: Array<[number, string]> = [
      [0, 'sea'],
      [3000, '3 km'],
      [5500, 'EBC'],
      [8849, 'Everest'],
    ];
    for (const [h, lbl] of refs) {
      const v = alveolarPO2(h, paco2);
      frame.g
        .append('line')
        .attr('class', 'line-secondary')
        .attr('x1', x(h)).attr('x2', x(h))
        .attr('y1', y(0)).attr('y2', y(v));
      frame.g
        .append('text')
        .attr('x', x(h)).attr('y', y(v) - 4)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--dim)').style('font-size', '8px')
        .text(lbl);
    }

    // Operating point.
    const op = alveolarPO2(altM, paco2);
    frame.g
      .append('circle')
      .attr('class', 'point')
      .attr('r', 5)
      .attr('cx', x(altM))
      .attr('cy', y(op));
    frame.g
      .append('text')
      .attr('x', x(altM) + 8)
      .attr('y', y(op) + 4)
      .attr('fill', 'var(--accent)').style('font-size', '10px')
      .text(`${op.toFixed(0)} mmHg @ ${(altM / 1000).toFixed(1)} km`);
  }

  draw();
  const stop = observeResize(container, draw);

  return {
    setOperating(alt: number, p?: number): void {
      altM = alt;
      if (p !== undefined) paco2 = p;
      draw();
    },
    destroy(): void {
      stop();
      frame.svg.remove();
    },
  };
}

export { alveolarPO2, patmAtAltitude };
