// Hemoglobin O2 dissociation curve (Hill equation, n=2.7, P50=27 mmHg)
// per cardiovascular-derivation.tex Eq. \ref{eq:hill}.
//
// Y(P_O2) = P_O2^n / (P50^n + P_O2^n)
//
// Live operating point is plotted from the current rPPG / model state
// (initially fixed to typical arterial; replaced by fitted state when
// the UKF lands).

import * as d3 from 'd3';
import {
  makeChart,
  syncChartSize,
  plotWidth,
  plotHeight,
  observeResize,
} from './chart-utils';

const P50 = 27;
const HILL_N = 2.7;

function saturation(pO2: number): number {
  const p = Math.pow(pO2, HILL_N);
  return p / (Math.pow(P50, HILL_N) + p);
}

export interface HbCurveHandle {
  setPoint(pO2: number, sat?: number): void;
  destroy(): void;
}

export function mountHbDissociation(container: HTMLElement): HbCurveHandle {
  const frame = makeChart(container);

  let currentPO2 = 100;
  let currentSat: number | null = null;

  function draw(): void {
    if (!syncChartSize(container, frame) && frame.width === 0) return;
    const W = plotWidth(frame);
    const H = plotHeight(frame);
    if (W <= 0 || H <= 0) return;

    frame.g.selectAll('*').remove();

    const x = d3.scaleLinear().domain([0, 110]).range([0, W]);
    const y = d3.scaleLinear().domain([0, 1]).range([H, 0]);

    // Curve points.
    const data = d3.range(0, 111, 1).map((p) => [p, saturation(p)] as [number, number]);

    // Axes.
    frame.g
      .append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(0,${H})`)
      .call(d3.axisBottom(x).ticks(5));
    frame.g
      .append('g')
      .attr('class', 'axis')
      .call(d3.axisLeft(y).ticks(5).tickFormat((d) => `${Math.round((+d) * 100)}%`));

    // Grid.
    frame.g
      .append('g')
      .attr('class', 'grid')
      .selectAll('line')
      .data(y.ticks(5))
      .enter()
      .append('line')
      .attr('x1', 0)
      .attr('x2', W)
      .attr('y1', (d) => y(d))
      .attr('y2', (d) => y(d));

    // Curve.
    const line = d3
      .line<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(d[1]))
      .curve(d3.curveMonotoneX);
    frame.g.append('path').datum(data).attr('class', 'line').attr('d', line);

    // Reference: P50 marker.
    frame.g
      .append('line')
      .attr('class', 'line-secondary')
      .attr('x1', x(P50))
      .attr('x2', x(P50))
      .attr('y1', y(0))
      .attr('y2', y(0.5));
    frame.g
      .append('text')
      .attr('x', x(P50) + 4)
      .attr('y', y(0.5) - 2)
      .attr('fill', 'var(--dim)')
      .style('font-size', '9px')
      .text(`P₅₀=${P50}`);

    // Operating point.
    const sat = currentSat ?? saturation(currentPO2);
    frame.g
      .append('circle')
      .attr('class', 'point')
      .attr('r', 4)
      .attr('cx', x(currentPO2))
      .attr('cy', y(sat));
    frame.g
      .append('text')
      .attr('x', x(currentPO2) + 6)
      .attr('y', y(sat) - 6)
      .attr('fill', 'var(--accent)')
      .style('font-size', '10px')
      .text(`${(sat * 100).toFixed(1)}% @ ${currentPO2.toFixed(0)}mmHg`);
  }

  draw();
  const stop = observeResize(container, draw);

  return {
    setPoint(pO2: number, sat?: number): void {
      currentPO2 = pO2;
      currentSat = sat ?? null;
      draw();
    },
    destroy(): void {
      stop();
      frame.svg.remove();
    },
  };
}
