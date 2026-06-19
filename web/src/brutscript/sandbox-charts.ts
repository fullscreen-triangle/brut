// BrutScript sandbox — D3 chart panel.
// Renders live trace data as a set of tab-selectable time-series and
// distribution charts. No React — pure D3 v7, matching the observatory pattern.

import * as d3 from 'd3';
import type { TraceEntry } from './runtime';

// ─── Colour tokens (match observatory CSS vars) ───────────────────────────────
const C = {
  bg:      '#08090c',
  line:    '#15191f',
  bright:  '#232830',
  fg:      '#e6e8eb',
  dim:     '#5a5e68',
  accent:  '#5fafff',
  green:   '#7fd47f',
  warn:    '#ffaf5f',
  hot:     '#ff5f5f',
  series:  ['#5fafff', '#7fd47f', '#ffaf5f', '#ff5f5f', '#bf9fef', '#5fd4d4'],
};

// ─── Signal colours by known name ────────────────────────────────────────────
const SIG_COLOR: Record<string, string> = {
  hr: C.hot, rc_mean: C.accent, rmssd: C.green, se: C.warn,
  sk: C.series[4], dhr_autonomic: C.hot, dhr_metabolic: C.warn,
  vasodilation: C.green, spo2: C.series[2], t_skin: C.warn,
  csci: C.accent, rt_ratio: C.series[5],
};

function sigColor(name: string): string {
  return SIG_COLOR[name] ?? C.series[Math.abs(hashStr(name)) % C.series.length];
}
function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return h;
}

// ─── Data model ──────────────────────────────────────────────────────────────

const MAX_POINTS = 600;

interface DataPoint { t: number; value: number; }
type SeriesStore = Map<string, DataPoint[]>;

function pushPoint(store: SeriesStore, name: string, t: number, value: number): void {
  if (typeof value !== 'number' || !isFinite(value)) return;
  let arr = store.get(name);
  if (!arr) { arr = []; store.set(name, arr); }
  arr.push({ t, value });
  if (arr.length > MAX_POINTS) arr.shift();
}

// ─── Chart container helpers ──────────────────────────────────────────────────

interface Margin { top: number; right: number; bottom: number; left: number; }
const MARGIN: Margin = { top: 12, right: 16, bottom: 32, left: 52 };

function svgSize(el: Element): { w: number; h: number; pw: number; ph: number } {
  const rect = el.getBoundingClientRect();
  const w = rect.width || 480;
  const h = rect.height || 220;
  return {
    w, h,
    pw: w - MARGIN.left - MARGIN.right,
    ph: h - MARGIN.top - MARGIN.bottom,
  };
}

// ─── Time-series chart ────────────────────────────────────────────────────────

interface TimeSeriesChart {
  update(store: SeriesStore, seriesNames: string[]): void;
  destroy(): void;
}

function createTimeSeriesChart(container: HTMLElement): TimeSeriesChart {
  const svg = d3.select(container)
    .append('svg')
    .attr('width', '100%')
    .attr('height', '100%')
    .style('display', 'block')
    .style('overflow', 'visible');

  const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

  const xAxis = g.append('g').attr('class', 'x-axis');
  const yAxis = g.append('g').attr('class', 'y-axis');
  const linesG = g.append('g').attr('class', 'lines');
  const legendG = g.append('g').attr('class', 'legend');

  const clip = svg.append('defs').append('clipPath').attr('id', `ts-clip-${Math.random().toString(36).slice(2)}`);
  const clipRect = clip.append('rect');
  linesG.attr('clip-path', `url(#${clip.attr('id')})`);

  styleAxisGroup(xAxis);
  styleAxisGroup(yAxis);

  function update(store: SeriesStore, seriesNames: string[]): void {
    const { pw, ph } = svgSize(container);
    clipRect.attr('width', pw).attr('height', ph + 4).attr('y', -2);

    const allPoints = seriesNames.flatMap(n => store.get(n) ?? []);
    if (allPoints.length === 0) return;

    const tMin = d3.min(allPoints, d => d.t)!;
    const tMax = d3.max(allPoints, d => d.t)!;
    const vMin = d3.min(allPoints, d => d.value)!;
    const vMax = d3.max(allPoints, d => d.value)!;
    const pad = (vMax - vMin) * 0.08 || 1;

    const xScale = d3.scaleLinear().domain([tMin, tMax]).range([0, pw]);
    const yScale = d3.scaleLinear().domain([vMin - pad, vMax + pad]).range([ph, 0]);

    xAxis.attr('transform', `translate(0,${ph})`).call(
      d3.axisBottom(xScale)
        .ticks(6)
        .tickFormat(v => `${((v as number) / 1000).toFixed(0)}s`)
    );
    yAxis.call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.3~g')));
    styleAxisGroup(xAxis);
    styleAxisGroup(yAxis);

    // Grid lines
    g.selectAll('.grid-h').remove();
    g.selectAll('.grid-h')
      .data(yScale.ticks(5))
      .join('line')
      .attr('class', 'grid-h')
      .attr('x1', 0).attr('x2', pw)
      .attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
      .attr('stroke', C.line).attr('stroke-width', 1);

    // Lines
    const line = d3.line<DataPoint>()
      .x(d => xScale(d.t))
      .y(d => yScale(d.value))
      .curve(d3.curveMonotoneX);

    linesG.selectAll<SVGPathElement, string>('.series-line')
      .data(seriesNames, d => d)
      .join(
        enter => enter.append('path')
          .attr('class', 'series-line')
          .attr('fill', 'none')
          .attr('stroke-width', 1.5)
          .attr('stroke-linejoin', 'round')
          .attr('stroke-linecap', 'round'),
        update => update,
        exit => exit.remove(),
      )
      .attr('stroke', n => sigColor(n))
      .attr('d', n => line(store.get(n) ?? []));

    // Legend
    legendG.attr('transform', `translate(${pw - 4}, 0)`);
    legendG.selectAll<SVGGElement, string>('.leg-item')
      .data(seriesNames, d => d)
      .join(
        enter => {
          const gi = enter.append('g').attr('class', 'leg-item');
          gi.append('line').attr('x1', -28).attr('x2', -10).attr('y1', 0).attr('y2', 0).attr('stroke-width', 1.5);
          gi.append('text').attr('x', -6).attr('y', 4).attr('text-anchor', 'end').style('font-size', '9px').style('fill', C.dim);
          return gi;
        },
        update => update,
      )
      .attr('transform', (_, i) => `translate(0, ${i * 14})`)
      .each(function(n) {
        d3.select(this).select('line').attr('stroke', sigColor(n));
        d3.select(this).select('text').text(n);
      });
  }

  return {
    update,
    destroy() { svg.remove(); },
  };
}

// ─── Coherence gauge (single-value arc) ──────────────────────────────────────

interface GaugeChart {
  update(value: number, label: string, regime: string): void;
  destroy(): void;
}

function createCoherenceGauge(container: HTMLElement): GaugeChart {
  const svg = d3.select(container)
    .append('svg')
    .attr('width', '100%')
    .attr('height', '100%')
    .style('display', 'block');

  const g = svg.append('g');
  const arc = g.append('path').attr('fill', 'none').attr('stroke-width', 8);
  const arcBg = g.append('path').attr('fill', 'none').attr('stroke-width', 8).attr('stroke', C.bright);
  const valText = g.append('text').attr('text-anchor', 'middle').style('font-family', 'var(--mono, monospace)').style('fill', C.fg);
  const labelText = g.append('text').attr('text-anchor', 'middle').style('font-family', 'var(--mono, monospace)').style('fill', C.dim).style('font-size', '10px');
  const regimeText = g.append('text').attr('text-anchor', 'middle').style('font-family', 'var(--mono, monospace)').style('font-size', '9px');

  function update(value: number, label: string, regime: string): void {
    const rect = container.getBoundingClientRect();
    const r = Math.min(rect.width, rect.height) * 0.38;
    const cx = rect.width / 2;
    const cy = rect.height * 0.55;
    g.attr('transform', `translate(${cx},${cy})`);

    const startAngle = -Math.PI * 0.75;
    const endAngle = Math.PI * 0.75;
    const valAngle = startAngle + (endAngle - startAngle) * Math.max(0, Math.min(1, value));

    const arcGen = d3.arc<unknown, { startAngle: number; endAngle: number; innerRadius: number; outerRadius: number }>();

    arcBg.attr('d', arcGen({ startAngle, endAngle, innerRadius: r - 4, outerRadius: r + 4 }) ?? '');

    const color = value >= 0.947 ? C.green : value >= 0.85 ? C.accent : value >= 0.3 ? C.warn : C.hot;
    arc
      .attr('stroke', color)
      .attr('d', arcGen({ startAngle, endAngle: valAngle, innerRadius: r - 4, outerRadius: r + 4 }) ?? '');

    const fontSize = Math.max(16, r * 0.38);
    valText.attr('y', -r * 0.08).style('font-size', `${fontSize}px`).style('fill', color).text(value.toFixed(3));
    labelText.attr('y', r * 0.22).text(label);
    regimeText.attr('y', r * 0.42).style('fill', color).text(regime);
  }

  return { update, destroy() { svg.remove(); } };
}

// ─── Divergence distribution bar chart ───────────────────────────────────────

interface DivBar {
  update(data: Array<{ label: string; count: number }>): void;
  destroy(): void;
}

function createDivergenceBar(container: HTMLElement): DivBar {
  const svg = d3.select(container)
    .append('svg')
    .attr('width', '100%').attr('height', '100%').style('display', 'block');
  const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

  function update(data: Array<{ label: string; count: number }>): void {
    const { pw, ph } = svgSize(container);
    const xScale = d3.scaleLinear().domain([0, d3.max(data, d => d.count) ?? 1]).range([0, pw]);
    const yScale = d3.scaleBand().domain(data.map(d => d.label)).range([0, ph]).padding(0.3);

    g.selectAll<SVGRectElement, {label:string;count:number}>('.div-bar')
      .data(data, d => d.label)
      .join('rect')
      .attr('class', 'div-bar')
      .attr('x', 0)
      .attr('y', d => yScale(d.label)!)
      .attr('height', yScale.bandwidth())
      .attr('width', d => xScale(d.count))
      .attr('fill', C.hot)
      .attr('rx', 2);

    g.selectAll<SVGTextElement, {label:string;count:number}>('.div-label')
      .data(data, d => d.label)
      .join('text')
      .attr('class', 'div-label')
      .attr('x', d => xScale(d.count) + 4)
      .attr('y', d => (yScale(d.label)! + yScale.bandwidth() / 2))
      .attr('dy', '0.35em')
      .style('font-size', '9px')
      .style('fill', C.fg)
      .text(d => d.count);

    g.selectAll<SVGTextElement, {label:string;count:number}>('.div-ylabel')
      .data(data, d => d.label)
      .join('text')
      .attr('class', 'div-ylabel')
      .attr('x', -4)
      .attr('y', d => (yScale(d.label)! + yScale.bandwidth() / 2))
      .attr('dy', '0.35em')
      .attr('text-anchor', 'end')
      .style('font-size', '9px')
      .style('fill', C.dim)
      .text(d => d.label);
  }

  return { update, destroy() { svg.remove(); } };
}

// ─── Style helpers ────────────────────────────────────────────────────────────

function styleAxisGroup(sel: d3.Selection<SVGGElement, unknown, null, undefined>): void {
  sel.selectAll('text').style('fill', C.dim).style('font-size', '9px').style('font-family', 'var(--mono,monospace)');
  sel.selectAll('path,line').attr('stroke', C.bright);
}

// ─── Public chart panel ───────────────────────────────────────────────────────

export interface ChartPanel {
  /** Feed a batch of trace entries; charts update from them. */
  ingestTrace(entries: TraceEntry[]): void;
  destroy(): void;
}

export function mountChartPanel(root: HTMLElement): ChartPanel {
  root.innerHTML = '';
  root.style.cssText = 'display:grid;grid-template-rows:auto 1fr;height:100%;overflow:hidden;';

  // ── Tab bar ───────────────────────────────────────────────────────────────
  const tabBar = document.createElement('div');
  tabBar.className = 'bs-tab-bar';
  const TABS = ['time series', 'coherence', 'divergences', 'signal table'];
  const tabBtns = TABS.map(t => {
    const b = document.createElement('button');
    b.textContent = t;
    b.className = 'bs-tab';
    b.dataset.tab = t;
    tabBar.appendChild(b);
    return b;
  });
  root.appendChild(tabBar);

  // ── Tab content area ──────────────────────────────────────────────────────
  const body = document.createElement('div');
  body.style.cssText = 'flex:1;overflow:hidden;position:relative;min-height:0;';
  root.appendChild(body);

  const panes = new Map<string, HTMLDivElement>();
  for (const t of TABS) {
    const p = document.createElement('div');
    p.className = 'bs-tab-pane';
    p.dataset.tab = t;
    p.style.cssText = 'display:none;width:100%;height:100%;overflow:auto;';
    body.appendChild(p);
    panes.set(t, p);
  }

  let activeTab = TABS[0];
  function switchTab(t: string): void {
    activeTab = t;
    for (const [name, pane] of panes) {
      pane.style.display = name === t ? 'block' : 'none';
    }
    tabBtns.forEach(b => b.classList.toggle('active', b.dataset.tab === t));
  }
  tabBtns.forEach(b => b.addEventListener('click', () => switchTab(b.dataset.tab!)));
  switchTab(TABS[0]);

  // ── Data stores ───────────────────────────────────────────────────────────
  const store: SeriesStore = new Map();
  const divergences = new Map<string, number>();
  const latest = new Map<string, number | string>();
  let rcValue = 0, regimeLabel = 'unknown';

  // ── Time-series panel ─────────────────────────────────────────────────────
  const tsPane = panes.get('time series')!;
  tsPane.style.cssText += 'display:block;padding:8px;height:100%;';
  const tsChart = createTimeSeriesChart(tsPane);

  // ── Coherence gauge ────────────────────────────────────────────────────────
  const cohPane = panes.get('coherence')!;
  cohPane.style.cssText += 'display:none;padding:8px;height:100%;';
  const gauge = createCoherenceGauge(cohPane);

  // ── Divergences ────────────────────────────────────────────────────────────
  const divPane = panes.get('divergences')!;
  divPane.style.cssText += 'display:none;padding:8px;height:100%;';
  const divBar = createDivergenceBar(divPane);

  // ── Signal table ──────────────────────────────────────────────────────────
  const tablePane = panes.get('signal table')!;
  tablePane.style.cssText += 'display:none;padding:8px;overflow:auto;height:100%;';
  const tableEl = document.createElement('table');
  tableEl.className = 'bs-signal-table';
  tablePane.appendChild(tableEl);

  // ── Ingest ────────────────────────────────────────────────────────────────
  function ingestTrace(entries: TraceEntry[]): void {
    for (const e of entries) {
      const t = e.t;

      // Collect numeric output values into store
      if (e.out) {
        for (const [k, v] of Object.entries(e.out)) {
          if (typeof v === 'number') {
            pushPoint(store, k, t, v);
            latest.set(k, v);
          } else if (typeof v === 'string') {
            latest.set(k, v);
          }
        }
      }
      // Track regime + rc
      if (e.out && 'regime' in e.out) regimeLabel = String(e.out.regime);
      if (e.out && 'rc_mean' in e.out && typeof e.out.rc_mean === 'number') rcValue = e.out.rc_mean;
      if (e.out && typeof (e.out as Record<string,unknown>)['rc_mean'] === 'number') rcValue = (e.out as Record<string,unknown>)['rc_mean'] as number;

      // Count diverge events
      if (e.step === 'diverge') {
        const key = e.block;
        divergences.set(key, (divergences.get(key) ?? 0) + 1);
      }
    }

    // Redraw visible tab
    if (activeTab === 'time series') {
      const names = [...store.keys()].filter(k => {
        const arr = store.get(k)!;
        return arr.length > 1;
      }).slice(0, 6);
      tsChart.update(store, names);
    } else if (activeTab === 'coherence') {
      gauge.update(rcValue, 'Rc — cardiac coherence', regimeLabel);
    } else if (activeTab === 'divergences') {
      divBar.update([...divergences.entries()].map(([label, count]) => ({ label, count })));
    } else if (activeTab === 'signal table') {
      renderTable();
    }
  }

  function renderTable(): void {
    const rows = [...latest.entries()];
    tableEl.innerHTML = `
      <thead><tr><th>signal</th><th>value</th></tr></thead>
      <tbody>${rows.map(([k, v]) =>
        `<tr><td class="sig-name">${k}</td><td class="sig-val" style="color:${typeof v === 'number' ? sigColor(k) : C.dim}">${
          typeof v === 'number' ? v.toFixed(4) : v
        }</td></tr>`
      ).join('')}</tbody>`;
  }

  return {
    ingestTrace,
    destroy() { tsChart.destroy(); gauge.destroy(); divBar.destroy(); },
  };
}
