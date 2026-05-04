// Shared D3 chart helpers. Tiny base on top of d3-selection / d3-scale / d3-axis.

import * as d3 from 'd3';

export interface ChartFrame {
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  g: d3.Selection<SVGGElement, unknown, null, undefined>;
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
}

export interface ChartFrameOptions {
  margin?: { top?: number; right?: number; bottom?: number; left?: number };
  className?: string;
}

const DEFAULT_MARGIN = { top: 8, right: 12, bottom: 22, left: 36 };

export function makeChart(
  container: HTMLElement,
  opts: ChartFrameOptions = {},
): ChartFrame {
  const margin = { ...DEFAULT_MARGIN, ...(opts.margin ?? {}) };

  // The svg fills the container; we recompute on resize.
  const svg = d3
    .select(container)
    .append('svg')
    .attr('class', `chart-svg ${opts.className ?? ''}`.trim())
    .attr('preserveAspectRatio', 'none');

  const g = svg.append('g').attr('class', 'chart-root');

  const frame: ChartFrame = {
    svg,
    g,
    width: 0,
    height: 0,
    margin,
  };

  return frame;
}

/** Apply current container size to the frame; returns true if it changed. */
export function syncChartSize(container: HTMLElement, frame: ChartFrame): boolean {
  const w = container.clientWidth;
  const h = container.clientHeight;
  if (w <= 0 || h <= 0) return false;
  if (frame.width === w && frame.height === h) return false;
  frame.width = w;
  frame.height = h;
  frame.svg
    .attr('width', w)
    .attr('height', h)
    .attr('viewBox', `0 0 ${w} ${h}`);
  frame.g.attr('transform', `translate(${frame.margin.left},${frame.margin.top})`);
  return true;
}

export function plotWidth(f: ChartFrame): number {
  return Math.max(0, f.width - f.margin.left - f.margin.right);
}

export function plotHeight(f: ChartFrame): number {
  return Math.max(0, f.height - f.margin.top - f.margin.bottom);
}

/** Watch container size and call `redraw` on changes via ResizeObserver. */
export function observeResize(container: HTMLElement, redraw: () => void): () => void {
  const ro = new ResizeObserver(() => redraw());
  ro.observe(container);
  return () => ro.disconnect();
}
