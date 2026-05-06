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
import { putRecord, listSince, downloadJsonExport, clearAll, anchorNow } from '../persistence/db';

export interface ObservatoryRecord {
  t: number;        // ms epoch
  // Cardiac (0 if no camera / not yet estimated) -------------------------
  hr: number;       // bpm
  rmssd: number;    // ms
  rc: number;       // [0, 1]
  sk: number;
  st: number;
  se: number;
  regime: number;   // 0 turbulent .. 4 phase-locked
  rrBpm: number;    // breaths/min
  ees: number;      // mmHg/mL
  ea: number;       // mmHg/mL
  ef: number;       // 0..1
  sv: number;       // mL
  co: number;       // L/min
  // Motor (always recorded) -------------------------------------------
  keyCount: number;          // keystrokes in last 1 s
  meanIki: number;           // ms; 0 if no events
  meanDwell: number;         // ms
  backspaceRate: number;     // 0..1
  bursty: number;            // cv of IKIs
  mouseDistance: number;     // px in last 1 s
  mousePeakVel: number;      // px/s
  ramblingPower: number;
  tremblingPower: number;
  rtRatio: number;           // rambling / (rambling + trembling)
  // Perceptual / circadian (camera required for blinks; melanopic always) -
  blinks: number;            // blink count in last 1 s
  blinksPerMin: number;      // rolling 60 s
  melFlux: number;           // current melanopic illuminance (mlux)
  melSensitivity: number;    // current circadian-phase sensitivity, 0..1
  melLoadMlxH: number;       // cumulative mlux·hours since session start
  // Skin-optics derived metabolic inputs (camera required) ---------------
  T_skin_C: number;          // °C, 0 if not estimated
  vasodilation: number;      // 1.0 = baseline, > 1 dilated, < 1 constricted
  spo2Optical: number;       // 0..1, optical-model oxygenation proxy
  dHRautonomic: number;      // bpm — PCHR autonomic residual (neural drive)
}

export interface Dashboard {
  push(rec: ObservatoryRecord): void;
  /** Replay records from IndexedDB into the crossfilter. Called once at startup. */
  hydrate(daysBack?: number): Promise<number>;
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
    ees: Dimension<ObservatoryRecord, number>;
    ea: Dimension<ObservatoryRecord, number>;
    ef: Dimension<ObservatoryRecord, number>;
    meanIki: Dimension<ObservatoryRecord, number>;
    mouseDistance: Dimension<ObservatoryRecord, number>;
    rtRatio: Dimension<ObservatoryRecord, number>;
    blinksPerMin: Dimension<ObservatoryRecord, number>;
    melFlux: Dimension<ObservatoryRecord, number>;
    T_skin_C: Dimension<ObservatoryRecord, number>;
    vasodilation: Dimension<ObservatoryRecord, number>;
  };
  charts: ChartHandle[];
  redrawAll(): void;
}

export function mountDashboard(container: HTMLElement): Dashboard {
  // Anchor wall-clock <-> performance.now() so persisted records survive reloads.
  anchorNow();

  // Create store --------------------------------------------------------
  const cf = crossfilter<ObservatoryRecord>([]);
  const dims = {
    t: cf.dimension((d) => d.t),
    hr: cf.dimension((d) => d.hr),
    rc: cf.dimension((d) => d.rc),
    rmssd: cf.dimension((d) => d.rmssd),
    regime: cf.dimension((d) => d.regime),
    rrBpm: cf.dimension((d) => d.rrBpm),
    ees: cf.dimension((d) => d.ees),
    ea: cf.dimension((d) => d.ea),
    ef: cf.dimension((d) => d.ef),
    meanIki: cf.dimension((d) => d.meanIki),
    mouseDistance: cf.dimension((d) => d.mouseDistance),
    rtRatio: cf.dimension((d) => d.rtRatio),
    blinksPerMin: cf.dimension((d) => d.blinksPerMin),
    melFlux: cf.dimension((d) => d.melFlux),
    T_skin_C: cf.dimension((d) => d.T_skin_C),
    vasodilation: cf.dimension((d) => d.vasodilation),
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

  // Build cells (4 × 3 grid) -------------------------------------------
  container.innerHTML = '';
  const cells = [
    // Row 1 — cardiac measured
    { id: 'cell-hr', title: 'HR (bpm) — brushable', mount: (el: HTMLElement) => mountHrLine(el, store) },
    { id: 'cell-rc', title: 'R_c — brushable', mount: (el: HTMLElement) => mountRcLine(el, store) },
    { id: 'cell-rmssd', title: 'RMSSD (ms)', mount: (el: HTMLElement) => mountRmssdLine(el, store) },
    { id: 'cell-resp', title: 'resp rate (bpm)', mount: (el: HTMLElement) => mountRespLine(el, store) },
    // Row 2 — cardiac fitted
    { id: 'cell-co', title: 'CO (L/min) · fitted', mount: (el: HTMLElement) => mountCoLine(el, store) },
    { id: 'cell-ef', title: 'EF · fitted', mount: (el: HTMLElement) => mountEfLine(el, store) },
    { id: 'cell-elast', title: 'E_es × E_a · fitted state plane', mount: (el: HTMLElement) => mountElastanceScatter(el, store) },
    { id: 'cell-failure', title: 'R_c × S_e · failure-mode plane', mount: (el: HTMLElement) => mountFailureModeScatter(el, store) },
    // Row 3 — motor (efferent half of the closed circulation)
    { id: 'cell-iki', title: 'mean IKI (ms) · keystroke timing', mount: (el: HTMLElement) => mountIkiLine(el, store) },
    { id: 'cell-mouse', title: 'mouse distance (px/s)', mount: (el: HTMLElement) => mountMouseLine(el, store) },
    { id: 'cell-rt', title: 'rambling vs trembling (cursor)', mount: (el: HTMLElement) => mountRtSplit(el, store) },
    { id: 'cell-coupling', title: 'cardio-motor: HR × keystrokes', mount: (el: HTMLElement) => mountCouplingScatter(el, store) },
    // Row 4 — perceptual / circadian
    { id: 'cell-blinks', title: 'blinks per minute', mount: (el: HTMLElement) => mountBlinksLine(el, store) },
    { id: 'cell-mlflux', title: 'melanopic flux (mlux)', mount: (el: HTMLElement) => mountMelFluxLine(el, store) },
    { id: 'cell-mlload', title: 'cumulative melanopic load (mlux·h)', mount: (el: HTMLElement) => mountMelLoadLine(el, store) },
    { id: 'cell-blinkiki', title: 'engagement: blinks × IKI', mount: (el: HTMLElement) => mountBlinkIkiScatter(el, store) },
    // Row 5 — skin-optics derived metabolic state (model-grounded inversion)
    { id: 'cell-tskin', title: 'T_skin (°C) · model-derived', mount: (el: HTMLElement) => mountTskinLine(el, store) },
    { id: 'cell-vaso', title: 'vasodilation factor', mount: (el: HTMLElement) => mountVasoLine(el, store) },
    { id: 'cell-spo2opt', title: 'SpO₂ proxy (optical model)', mount: (el: HTMLElement) => mountSpo2OpticalLine(el, store) },
    { id: 'cell-hrauto', title: 'ΔHR autonomic (PCHR neural drive)', mount: (el: HTMLElement) => mountHRautonomicLine(el, store) },
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

  // Drawer header controls — export, clear, days-back replay slider.
  const drawerHeader = container.parentElement?.querySelector('header');
  if (drawerHeader && !drawerHeader.querySelector('.drawer-tools')) {
    const tools = document.createElement('div');
    tools.className = 'drawer-tools';
    tools.innerHTML = `
      <label class="ds-control">
        <span>history</span>
        <select id="days-back">
          <option value="0" selected>live only</option>
          <option value="1">1 d</option>
          <option value="3">3 d</option>
          <option value="7">7 d</option>
          <option value="30">30 d</option>
        </select>
      </label>
      <button class="ds-btn" id="export-json" title="download all data as JSON">export</button>
      <button class="ds-btn ds-btn-danger" id="clear-db" title="erase all stored records">clear</button>
    `;
    // Insert before the existing close button.
    const closeBtn = drawerHeader.querySelector('.close');
    if (closeBtn) drawerHeader.insertBefore(tools, closeBtn);
    else drawerHeader.appendChild(tools);

    tools.querySelector('#export-json')?.addEventListener('click', () => {
      void downloadJsonExport();
    });
    tools.querySelector('#clear-db')?.addEventListener('click', () => {
      // eslint-disable-next-line no-alert
      if (!confirm('Erase all stored observatory records? This cannot be undone.')) return;
      void clearAll();
      cf.remove(() => true);
      store.redrawAll();
    });
    tools.querySelector('#days-back')?.addEventListener('change', (ev) => {
      const days = parseInt((ev.target as HTMLSelectElement).value, 10) || 0;
      void hydrateDays(days);
    });
  }

  async function hydrateDays(days: number): Promise<number> {
    // Discard old hydrated records before replacing.
    cf.remove(() => true);
    if (days <= 0) {
      store.redrawAll();
      return 0;
    }
    const records = await listSince(days * 24 * 60 * 60 * 1000);
    if (records.length > 0) cf.add(records);
    store.redrawAll();
    return records.length;
  }

  return {
    push(rec: ObservatoryRecord): void {
      cf.add([rec]);
      store.redrawAll();
      void putRecord(rec);
    },
    hydrate(daysBack = 1): Promise<number> {
      return hydrateDays(daysBack);
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

    // Filled area underneath gives the chart visual weight; the line on top
    // keeps the precise locus crisp.
    const baseY = y.range()[0]; // bottom of plot area
    const area = d3
      .area<{ key: number; value: unknown }>()
      .x((d) => x(+d.key))
      .y0(baseY)
      .y1((d) => y(yAccessor(d)))
      .curve(d3.curveMonotoneX);

    const line = d3
      .line<{ key: number; value: unknown }>()
      .x((d) => x(+d.key))
      .y((d) => y(yAccessor(d)))
      .curve(d3.curveMonotoneX);

    frame.g
      .append('path')
      .datum(all)
      .attr('fill', color)
      .attr('fill-opacity', 0.28)
      .attr('stroke', 'none')
      .attr('d', area);

    frame.g
      .append('path')
      .datum(all)
      .attr('class', 'line')
      .attr('stroke', color)
      .attr('stroke-width', 1.4)
      .attr('fill', 'none')
      .attr('d', line);

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

function mountCoLine(container: HTMLElement, store: CrossfilterStore): void {
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.co);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, '#7fd47f', [0, 25]);
}

function mountEfLine(container: HTMLElement, store: CrossfilterStore): void {
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.ef);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, '#ffaf5f', [0, 1]);
}

function mountIkiLine(container: HTMLElement, store: CrossfilterStore): void {
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.meanIki);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, '#c89fff', [0, 800]);
}

function mountMouseLine(container: HTMLElement, store: CrossfilterStore): void {
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.mouseDistance);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, '#ff9fcf', [0, 4000]);
}

function mountBlinksLine(container: HTMLElement, store: CrossfilterStore): void {
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.blinksPerMin);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, '#9fdfff', [0, 40]);
}

function mountMelFluxLine(container: HTMLElement, store: CrossfilterStore): void {
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.melFlux);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, '#ffd96b', [0, 100]);
}

function mountTskinLine(container: HTMLElement, store: CrossfilterStore): void {
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.T_skin_C);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, '#ff8c66', [27, 37]);
}

function mountVasoLine(container: HTMLElement, store: CrossfilterStore): void {
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.vasodilation);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, '#ffafd6', [0.6, 1.6]);
}

function mountSpo2OpticalLine(container: HTMLElement, store: CrossfilterStore): void {
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.spo2Optical);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, '#7fdfff', [0.85, 1.0]);
}

function mountHRautonomicLine(container: HTMLElement, store: CrossfilterStore): void {
  const group = store.dims.t.group(Math.floor).reduceSum((d) => d.dHRautonomic);
  mountTimeLine(container, store, group as unknown as Group<ObservatoryRecord, number, unknown>,
    (g) => g.value as number, '#c8b0ff', [-30, 60]);
}

function mountMelLoadLine(container: HTMLElement, store: CrossfilterStore): void {
  // Cumulative — pull the latest record's melLoadMlxH per second.
  // The @types/crossfilter reduce signature is loose; cast to keep TS happy.
  const grouped = store.dims.t.group(Math.floor);
  const group = (grouped as unknown as {
    reduce: (
      add: (p: number, v: ObservatoryRecord) => number,
      remove: (p: number, v: ObservatoryRecord) => number,
      initial: () => number,
    ) => Group<ObservatoryRecord, number, unknown>;
  }).reduce(
    (p, v) => Math.max(p, v.melLoadMlxH),
    (p) => p,
    () => 0,
  );
  mountTimeLine(container, store, group, (g) => g.value as number, '#ffba9b');
}

// ── Rambling vs trembling stacked area (cursor motion) ─────────────────
function mountRtSplit(container: HTMLElement, store: CrossfilterStore): void {
  const frame = makeChart(container, { margin: { top: 6, right: 8, bottom: 22, left: 28 } });

  function draw(): void {
    if (!syncChartSize(container, frame) && frame.width === 0) return;
    const W = plotWidth(frame);
    const H = plotHeight(frame);
    if (W <= 0 || H <= 0) return;

    frame.g.selectAll('*').remove();

    const all = store.cf.allFiltered().filter((d) => (d.ramblingPower + d.tremblingPower) > 0);
    if (all.length === 0) {
      frame.g
        .append('text')
        .attr('x', W / 2)
        .attr('y', H / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--dim)')
        .style('font-size', '10px')
        .text('move the cursor…');
      return;
    }

    const tExtent = d3.extent(all, (d) => d.t) as [number, number];
    const x = d3.scaleLinear().domain(tExtent).range([0, W]);
    const y = d3.scaleLinear().domain([0, 1]).range([H, 0]);

    frame.g
      .append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(0,${H})`)
      .call(
        d3.axisBottom(x).ticks(3).tickFormat((d) => {
          const dt = new Date(+d);
          return `${dt.getMinutes().toString().padStart(2, '0')}:${dt
            .getSeconds()
            .toString()
            .padStart(2, '0')}`;
        }),
      );
    frame.g
      .append('g')
      .attr('class', 'axis')
      .call(d3.axisLeft(y).ticks(3).tickFormat((d) => `${(+d * 100).toFixed(0)}%`));

    // Two areas stacked: rambling fraction at the bottom (warm), trembling on top (cool).
    const ramblingFrac = (d: ObservatoryRecord): number => {
      const total = d.ramblingPower + d.tremblingPower;
      return total > 0 ? d.ramblingPower / total : 0;
    };

    const ramblingArea = d3
      .area<ObservatoryRecord>()
      .x((d) => x(d.t))
      .y0(H)
      .y1((d) => y(ramblingFrac(d)));

    const tremblingArea = d3
      .area<ObservatoryRecord>()
      .x((d) => x(d.t))
      .y0((d) => y(ramblingFrac(d)))
      .y1(0);

    frame.g
      .append('path')
      .datum(all)
      .attr('fill', '#ffaf5f')
      .attr('fill-opacity', 0.55)
      .attr('d', ramblingArea);

    frame.g
      .append('path')
      .datum(all)
      .attr('fill', '#5fafff')
      .attr('fill-opacity', 0.55)
      .attr('d', tremblingArea);

    frame.g
      .append('text')
      .attr('x', 4).attr('y', H - 4)
      .attr('fill', '#ffaf5f').style('font-size', '8px')
      .text('rambling (<0.5 Hz, supraspinal)');
    frame.g
      .append('text')
      .attr('x', 4).attr('y', 10)
      .attr('fill', '#5fafff').style('font-size', '8px')
      .text('trembling (0.5-3 Hz, spinal loop)');
  }

  draw();
  observeResize(container, draw);
  store.charts.push({ redraw: draw });
}

// ── Blinks × IKI engagement scatter ────────────────────────────────────
//
// Engagement-attention mapping: high attention → suppressed blinks +
// tight typing rhythm (small IKI). Low attention → bursty IKIs and frequent
// blinks. We plot blinks/min against meanIki and let the user brush.
function mountBlinkIkiScatter(container: HTMLElement, store: CrossfilterStore): void {
  const frame = makeChart(container, { margin: { top: 6, right: 8, bottom: 22, left: 28 } });

  function draw(): void {
    if (!syncChartSize(container, frame) && frame.width === 0) return;
    const W = plotWidth(frame);
    const H = plotHeight(frame);
    if (W <= 0 || H <= 0) return;

    frame.g.selectAll('*').remove();

    const all = store.cf.allFiltered().filter((d) => d.meanIki > 0 && d.blinksPerMin >= 0);
    if (all.length === 0) {
      frame.g
        .append('text')
        .attr('x', W / 2)
        .attr('y', H / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--dim)')
        .style('font-size', '10px')
        .text('type with the camera up…');
      return;
    }

    const x = d3.scaleLinear().domain([0, 800]).range([0, W]);
    const y = d3.scaleLinear().domain([0, 40]).range([H, 0]);

    frame.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(4));
    frame.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4));

    frame.g
      .append('text')
      .attr('x', W / 2).attr('y', H + 18)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--dim)').style('font-size', '8px')
      .text('mean IKI (ms)');
    frame.g
      .append('text')
      .attr('x', -H / 2).attr('y', -22)
      .attr('text-anchor', 'middle').attr('transform', 'rotate(-90)')
      .attr('fill', 'var(--dim)').style('font-size', '8px')
      .text('blinks/min');

    frame.g
      .selectAll('circle.bk')
      .data(all)
      .enter()
      .append('circle')
      .attr('class', 'point bk')
      .attr('r', 2)
      .attr('cx', (d) => x(d.meanIki))
      .attr('cy', (d) => y(d.blinksPerMin));
  }

  draw();
  observeResize(container, draw);
  store.charts.push({ redraw: draw });
}

// ── HR × keystroke-rate cardio-motor coupling ──────────────────────────
function mountCouplingScatter(container: HTMLElement, store: CrossfilterStore): void {
  const frame = makeChart(container, { margin: { top: 6, right: 8, bottom: 22, left: 28 } });

  function draw(): void {
    if (!syncChartSize(container, frame) && frame.width === 0) return;
    const W = plotWidth(frame);
    const H = plotHeight(frame);
    if (W <= 0 || H <= 0) return;

    frame.g.selectAll('*').remove();

    const all = store.cf.allFiltered().filter((d) => d.hr > 0 && (d.keyCount > 0 || d.mouseDistance > 0));
    const x = d3.scaleLinear().domain([40, 160]).range([0, W]);
    const y = d3.scaleLinear().domain([0, 12]).range([H, 0]);

    frame.g
      .append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(0,${H})`)
      .call(d3.axisBottom(x).ticks(4));
    frame.g
      .append('g')
      .attr('class', 'axis')
      .call(d3.axisLeft(y).ticks(4));

    frame.g
      .append('text')
      .attr('x', W / 2).attr('y', H + 18)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--dim)').style('font-size', '8px')
      .text('HR (bpm)');
    frame.g
      .append('text')
      .attr('x', -H / 2).attr('y', -22)
      .attr('text-anchor', 'middle').attr('transform', 'rotate(-90)')
      .attr('fill', 'var(--dim)').style('font-size', '8px')
      .text('keystrokes/s');

    frame.g
      .selectAll('circle.cm')
      .data(all)
      .enter()
      .append('circle')
      .attr('class', 'point cm')
      .attr('r', 2)
      .attr('cx', (d) => x(d.hr))
      .attr('cy', (d) => y(d.keyCount));
  }

  draw();
  observeResize(container, draw);
  store.charts.push({ redraw: draw });
}

// ── E_es × E_a fitted-state plane ──────────────────────────────────────
function mountElastanceScatter(container: HTMLElement, store: CrossfilterStore): void {
  const frame = makeChart(container, { margin: { top: 6, right: 8, bottom: 22, left: 28 } });
  const brush = d3.brush();

  function draw(): void {
    if (!syncChartSize(container, frame) && frame.width === 0) return;
    const W = plotWidth(frame);
    const H = plotHeight(frame);
    if (W <= 0 || H <= 0) return;

    frame.g.selectAll('*').remove();

    const all = store.cf.allFiltered().filter((d) => d.ees > 0 && d.ea > 0);
    const x = d3.scaleLinear().domain([0, 8]).range([0, W]);
    const y = d3.scaleLinear().domain([0, 4]).range([H, 0]);

    frame.g
      .append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(0,${H})`)
      .call(d3.axisBottom(x).ticks(4));
    frame.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4));

    // Reference: E_es / E_a = 1 line (optimal coupling; cardiac-eos.tex Theorem 5.4).
    const refLine: Array<[number, number]> = [
      [0, 0],
      [4, 4],
    ];
    const lineGen = d3
      .line<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(d[1]));
    frame.g
      .append('path')
      .datum(refLine)
      .attr('class', 'line-secondary')
      .attr('d', lineGen);

    frame.g
      .append('text')
      .attr('x', x(2.0)).attr('y', y(2.0) - 4)
      .attr('fill', 'var(--dim)')
      .style('font-size', '8px')
      .text('E_es = E_a (optimum)');

    // Points.
    frame.g
      .selectAll('circle.pt')
      .data(all)
      .enter()
      .append('circle')
      .attr('class', 'point pt')
      .attr('r', 2)
      .attr('cx', (d) => x(d.ees))
      .attr('cy', (d) => y(d.ea));

    // Highlight latest point.
    if (all.length > 0) {
      const last = all[all.length - 1];
      frame.g
        .append('circle')
        .attr('r', 5)
        .attr('cx', x(last.ees))
        .attr('cy', y(last.ea))
        .attr('fill', 'var(--accent)')
        .attr('stroke', 'var(--fg)')
        .attr('stroke-width', 1);
    }

    // Brush filter on E_es range.
    brush.extent([
      [0, 0],
      [W, H],
    ]);
    const brushG = frame.g.append('g').attr('class', 'brush');
    brushG.call(brush);
    brush.on('end', (ev) => {
      if (!ev.selection) {
        store.dims.ees.filterAll();
      } else {
        const sel = ev.selection as [[number, number], [number, number]];
        store.dims.ees.filterRange([x.invert(sel[0][0]), x.invert(sel[1][0])]);
      }
      store.redrawAll();
    });
  }

  draw();
  observeResize(container, draw);
  store.charts.push({ redraw: draw });
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

