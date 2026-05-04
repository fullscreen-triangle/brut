// Crossfilter dashboard.
//
// Mirrors the react-crossfilter pattern from your archived module: a single
// store holds the crossfilter dimensions and groups; each chart pushes a
// `redraw` into store.charts on mount; brushing any chart calls
// `redrawAll()`. Vanilla TS port — no React.
//
// Records pushed by main.ts have shape:
//   { t, hr, rmssd, rc, sk, st, se, regime, rrBpm }

import * as d3 from 'd3';
import crossfilter, { type Crossfilter, type Dimension, type Group } from 'crossfilter2';
import {
  makeChart,
  syncChartSize,
  plotWidth,
  plotHeight,
  observeResize,
} from './chart-utils';

export interface ObservatoryRecord {
  t: number;        // ms epoch
  hr: number;       // bpm (0 if no estimate yet)
  rmssd: number;    // ms
  rc: number;       // [0, 1]
  sk: number;
  st: number;
  se: number;
  regime: number;   // 0 turbulent .. 4 phase-locked
  rrBpm: number;    // breaths/min
}

export interface Dashboard {
  push(rec: ObservatoryRecord): void;
  destroy(): void;
}

interface ChartHandle {
  redraw: () => void;
}

interface CrossfilterStore {
  cf: Crossfilter<ObservatoryRecord>;
  dims: {
    t: Dimension<ObservatoryRecord, number>;
    hr: Dimension<ObservatoryRecord, number>;
    rc: Dimension<ObservatoryRecord, number>;
    rmssd: Dimension<ObservatoryRecord, number>;
    regime: Dimension<ObservatoryRecord, number>;
    rrBpm: Dimension<ObservatoryRecord, number>;
  };
  charts: ChartHandle[];
  redrawAll(): void;
}

const REGIME_NAMES = ['turb', 'apt', 'csc', 'coh', 'pl'];

export function mountDashboard(container: HTMLElement): Dashboard {
  // Create store --------------------------------------------------------
  const cf = crossfilter<ObservatoryRecord>([]);
  const dims = {
    t: cf.dimension((d) => d.t),
    hr: cf.dimension((d) => d.hr),
    rc: cf.dimension((d) => d.rc),
    rmssd: cf.dimension((d) => d.rmssd),
    regime: cf.dimension((d) => d.regime),
    rrBpm: cf.dimension((d) => d.rrBpm),
  };
  const charts: ChartHandle[] = [];
  const store: CrossfilterStore = {
    cf,
    dims,
    charts,
    redrawAll(): void {
      for (const c of charts) c.redraw();
    },
  };

  // Build six cells -----------------------------------------------------
  container.innerHTML = '';
  const cells = [
    { id: 'cell-hr', title: 'HR (bpm) — brushable', mount: (el: HTMLElement) => mountHrLine(el, store) },
    { id: 'cell-rc', title: 'R_c — brushable', mount: (el: HTMLElement) => mountRcLine(el, store) },
    { id: 'cell-resp', title: 'respiration rate', mount: (el: HTMLElement) => mountRespLine(el, store) },
    { id: 'cell-poincare', title: 'R_c × S_e (failure-mode plane)', mount: (el: HTMLElement) => mountFailureModeScatter(el, store) },
    { id: 'cell-tachogram', title: 'RR tachogram (RMSSD ribbon)', mount: (el: HTMLElement) => mountRmssdLine(el, store) },
    { id: 'cell-regime', title: 'regime occupancy', mount: (el: HTMLElement) => mountRegimeBar(el, store) },
  ];

  for (const c of cells) {
    const cell = document.createElement('div');
    cell.className = 'dash-cell';
    cell.id = c.id;
    cell.innerHTML = `<div class="cell-title">${c.title}</div><div class="cell-body"></div>`;
    container.appendChild(cell);
    const body = cell.querySelector('.cell-body') as HTMLElement;
    c.mount(body);
  }

  return {
    push(rec: ObservatoryRecord): void {
      cf.add([rec]);
      store.redrawAll();
    },
    destroy(): void {
      // crossfilter has no explicit dispose; null out references.
      charts.length = 0;
      cf.remove(() => true);
    },
  };
}

// ── Brushable line chart over time ─────────────────────────────────────
function mountTimeLine(
  container: HTMLElement,
  store: CrossfilterStore,
  group: Group<ObservatoryRecord, number, unknown>,
  yAccessor: (g: { value: unknown }) => number,
  color: string,
  yDomain?: [number, number],
): void {
  const frame = makeChart(container, { margin: { top: 6, right: 8, bottom: 18, left: 28 } });
  const brush = d3.brushX();
  let brushG: d3.Selection<SVGGElement, unknown, null, undefined> | null = null;

  function draw(): void {
    if (!syncChartSize(container, frame) && frame.width === 0) return;
    const W = plotWidth(frame);
    const H = plotHeight(frame);
    if (W <= 0 || H <= 0) return;

    frame.g.selectAll('*').remove();

    const all = group.all().filter((d) => yAccessor(d) > 0);
    if (all.length === 0) {
      frame.g
        .append('text')
        .attr('x', W / 2)
        .attr('y', H / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--dim)')
        .style('font-size', '10px')
        .text('acquiring…');
      return;
    }

    const tExtent = d3.extent(all, (d) => +d.key) as [number, number];
    const x = d3.scaleLinear().domain(tExtent).range([0, W]);
    const y = d3
      .scaleLinear()
      .domain(yDomain ?? (d3.extent(all, yAccessor) as [number, number]))
      .nice()
      .range([H, 0]);

    frame.g
      .append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(0,${H})`)
      .call(
        d3
          .axisBottom(x)
          .ticks(3)
          .tickFormat((d) => {
            const dt = new Date(+d);
            return `${dt.getMinutes().toString().padStart(2, '0')}:${dt
              .getSeconds()
              .toString()
              .padStart(2, '0')}`;
          }),
      );
    frame.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(3));

    const line = d3
      .line<{ key: number; value: unknown }>()
      .x((d) => x(d.key))
      .y((d) => y(yAccessor(d)))
      .curve(d3.curveMonotoneX);

    frame.g.append('path').datum(all).attr('class', 'line').attr('stroke', color).attr('d', line);

    // Brush.
    brush.extent([
      [0, 0],
      [W, H],
    ]);
    brushG = frame.g.append('g').attr('class', 'brush');
    brushG.call(brush);
    brush.on('end', (ev) => {
      if (!ev.selection) {
        store.dims.t.filterAll();
      } else {
        const [x0, x1] = ev.selection as [number, number];
        store.dims.t.filterRange([x.invert(x0), x.invert(x1)]);
      }
      store.redrawAll();
    });
  }

  draw();
  observeResize(container, draw);
  store.charts.push({ redraw: draw });
}

function mountHrLine(container: HTMLElement, store: CrossfilterStore): void {
  // Group HR by 1-second time bucket — we get one record per ~tick anyway.
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.hr);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, 'var(--accent)');
}

function mountRcLine(container: HTMLElement, store: CrossfilterStore): void {
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.rc);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, '#7fd47f', [0, 1]);
}

function mountRespLine(container: HTMLElement, store: CrossfilterStore): void {
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.rrBpm);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, '#ffaf5f', [0, 30]);
}

function mountRmssdLine(container: HTMLElement, store: CrossfilterStore): void {
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.rmssd);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, '#5fafff', [0, 200]);
}

// ── R_c × S_e scatter (the failure-mode plane) ─────────────────────────
function mountFailureModeScatter(container: HTMLElement, store: CrossfilterStore): void {
  const frame = makeChart(container, { margin: { top: 6, right: 8, bottom: 22, left: 28 } });
  const brush = d3.brush();

  function draw(): void {
    if (!syncChartSize(container, frame) && frame.width === 0) return;
    const W = plotWidth(frame);
    const H = plotHeight(frame);
    if (W <= 0 || H <= 0) return;

    frame.g.selectAll('*').remove();

    const all = store.cf.allFiltered().filter((d) => d.rc > 0 && d.se > 0);
    const x = d3.scaleLinear().domain([0, 1]).range([0, W]);
    const y = d3.scaleLinear().domain([0, 1]).range([H, 0]);

    frame.g
      .append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(0,${H})`)
      .call(d3.axisBottom(x).ticks(4));
    frame.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4));

    // Region shading: rigidity (top right, low S_e) and decoherence (left).
    frame.g
      .append('rect')
      .attr('x', x(0.95))
      .attr('y', y(0.5))
      .attr('width', W - x(0.95))
      .attr('height', H - y(0.5))
      .attr('fill', 'var(--hot)')
      .attr('fill-opacity', 0.08);
    frame.g
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', x(0.3))
      .attr('height', H)
      .attr('fill', 'var(--warn)')
      .attr('fill-opacity', 0.08);

    // Points.
    frame.g
      .selectAll('circle')
      .data(all)
      .enter()
      .append('circle')
      .attr('class', 'point')
      .attr('r', 1.6)
      .attr('cx', (d) => x(d.rc))
      .attr('cy', (d) => y(d.se));

    // Brush.
    brush.extent([
      [0, 0],
      [W, H],
    ]);
    const brushG = frame.g.append('g').attr('class', 'brush');
    brushG.call(brush);
    brush.on('end', (ev) => {
      if (!ev.selection) {
        store.dims.rc.filterAll();
      } else {
        const sel = ev.selection as [[number, number], [number, number]];
        const xRange: [number, number] = [x.invert(sel[0][0]), x.invert(sel[1][0])];
        store.dims.rc.filterRange(xRange);
      }
      store.redrawAll();
    });
  }

  draw();
  observeResize(container, draw);
  store.charts.push({ redraw: draw });
}

// ── Regime occupancy bars ──────────────────────────────────────────────
function mountRegimeBar(container: HTMLElement, store: CrossfilterStore): void {
  const frame = makeChart(container, { margin: { top: 6, right: 8, bottom: 22, left: 28 } });
  const group = store.dims.regime.group();

  function draw(): void {
    if (!syncChartSize(container, frame) && frame.width === 0) return;
    const W = plotWidth(frame);
    const H = plotHeight(frame);
    if (W <= 0 || H <= 0) return;

    frame.g.selectAll('*').remove();

    const buckets = group.all();
    const x = d3
      .scaleBand<number>()
      .domain([0, 1, 2, 3, 4])
      .range([0, W])
      .padding(0.2);
    const y = d3
      .scaleLinear()
      .domain([0, Math.max(1, d3.max(buckets, (d) => d.value as number) ?? 1)])
      .range([H, 0]);

    frame.g
      .append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(0,${H})`)
      .call(d3.axisBottom(x).tickFormat((d) => REGIME_NAMES[+d as number] ?? ''));
    frame.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(3));

    frame.g
      .selectAll('rect.bar')
      .data(buckets)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', (d) => x(+d.key) ?? 0)
      .attr('y', (d) => y(d.value as number))
      .attr('width', x.bandwidth())
      .attr('height', (d) => H - y(d.value as number))
      .attr('fill', (d) => regimeColor(+d.key));
  }

  draw();
  observeResize(container, draw);
  store.charts.push({ redraw: draw });
}

function regimeColor(r: number): string {
  switch (r) {
    case 0: return '#ff5f5f'; // turbulent
    case 1: return '#ffaf5f'; // aperture
    case 2: return '#ffd75f'; // cascade
    case 3: return '#5fafff'; // coherent
    case 4: return '#7fd47f'; // phase-locked
    default: return '#888';
  }
}
