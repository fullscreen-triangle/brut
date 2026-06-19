// BrutScript runtime.
//
// Executes a WiringPlan against the live signal bus once per tick.
// Emits structured TraceEntry objects (NDJSON-serialisable) for every
// intermediate value and event.
//
// Inversion functions are looked up by their `using` string.  The ones
// implemented here correspond directly to skin-optics.ts and eos.ts.

import type { WiringPlan, Op, SerialExpr, SerialCondition } from './compiler';
import type { TimeWindow } from './ast';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface TraceEntry {
  t: number;                             // session-relative ms
  block: string;
  step: string;
  line: number;
  in?: Record<string, unknown>;
  out?: Record<string, unknown>;
  eval?: boolean;                        // for watch conditions
  value?: unknown;
  emit?: string;                         // watch fired event name
  confidence?: number;
  flag?: string;
  traces?: Record<string, unknown>;
}

type EnvValue = number | string;

type TraceHandler = (entry: TraceEntry) => void;
type ModelFn = (hub: string, task: string, inputs: Record<string, unknown>) => Promise<Record<string, unknown>>;

export interface RuntimeOptions {
  onTrace: TraceHandler;
  /** Resolve a model inference. Defaults to HuggingFace Inference API. */
  model?: ModelFn;
  /** API token for HuggingFace (only used if model is not provided) */
  hfToken?: string;
}

// ─── Rolling history for baseline() ──────────────────────────────────────────

class RollingHistory {
  private values: number[] = [];
  private times: number[] = [];

  push(v: number, t: number): void {
    this.values.push(v);
    this.times.push(t);
  }

  trim(windowMs: number, now: number): void {
    const cutoff = now - windowMs;
    while (this.times.length > 0 && this.times[0] < cutoff) {
      this.values.shift();
      this.times.shift();
    }
  }

  percentile(p: number): number {
    if (this.values.length === 0) return 0;
    const sorted = [...this.values].sort((a, b) => a - b);
    const idx = Math.floor((p / 100) * (sorted.length - 1));
    return sorted[Math.max(0, Math.min(sorted.length - 1, idx))];
  }

  mean(): number {
    if (this.values.length === 0) return 0;
    return this.values.reduce((a, b) => a + b, 0) / this.values.length;
  }

  std(): number {
    if (this.values.length < 2) return 0;
    const m = this.mean();
    return Math.sqrt(this.values.reduce((s, v) => s + (v - m) ** 2, 0) / this.values.length);
  }
}

// ─── Inversion functions ──────────────────────────────────────────────────────
// Named by `using` string, match the layered-optical-ppg derivation.

const ABSORPTION = {
  melanin:  { B: 0.84, G: 0.36, R: 0.10 },
  hbO2:     { B: 0.084, G: 0.112, R: 0.012 },
  hbDeox:   { B: 0.084, G: 0.064, R: 0.040 },
};

const L2 = 0.2; // epidermal path length (mm)
const L3 = 0.5; // dermal path length (mm)
const MU_MEL_B = ABSORPTION.melanin.B;
const MU_HB_R  = ABSORPTION.hbO2.R;
const T_DEOX_G = 0.30 as number; // normalised deoxy transmittance in green
const T_OXY_G  = 0.65 as number; // normalised oxy transmittance in green

type InversionFn = (primary: number, env: Map<string, EnvValue>) => number;

const INVERSIONS: Record<string, InversionFn> = {
  'beer_lambert.blue': (b, _env) => {
    // m = -ln(max(b, 0.01)) / (2 * mu_mel_B * L2)
    return -Math.log(Math.max(b, 0.01)) / (2 * MU_MEL_B * L2);
  },
  'beer_lambert.red': (r, env) => {
    const m = (env.get('melanin') as number) ?? 0.2;
    const vaso = (env.get('vasodilation') as number) ?? 1.0;
    const melaninAttenuation = Math.exp(-2 * ABSORPTION.melanin.R * L2 * m);
    const rTilde = Math.max(r / Math.max(melaninAttenuation, 0.01), 0.01);
    return -Math.log(rTilde) / (2 * MU_HB_R * L3 * Math.max(vaso, 0.1));
  },
  'beer_lambert.green': (g, env) => {
    const m = (env.get('melanin') as number) ?? 0.2;
    const melaninAttenuation = Math.exp(-2 * ABSORPTION.melanin.G * L2 * m);
    const gTilde = Math.max(g / Math.max(melaninAttenuation, 0.01), 0.01);
    if (T_OXY_G === T_DEOX_G) return 0.98;
    return Math.max(0, Math.min(1, (gTilde - T_DEOX_G) / (T_OXY_G - T_DEOX_G)));
  },
};

function applyInversion(using: string, primary: number, env: Map<string, EnvValue>): number {
  const fn = INVERSIONS[using];
  return fn ? fn(primary, env) : primary;
}

// ─── Expression evaluator ────────────────────────────────────────────────────

type HistoryStore = Map<string, RollingHistory>;

function evalExpr(e: SerialExpr, env: Map<string, EnvValue>, hist: HistoryStore, now: number): number | string {
  switch (e.kind) {
    case 'num': return e.value as number;
    case 'str': return e.value as string;
    case 'var': {
      const v = env.get(e.name as string);
      return v !== undefined ? v : 0;
    }
    case 'field': {
      const key = (e.obj as string) + '.' + (e.field as string);
      const v = env.get(key) ?? env.get(e.field as string);
      return v !== undefined ? v : 0;
    }
    case 'bin': {
      const l = evalExpr(e.left as SerialExpr, env, hist, now) as number;
      const r = evalExpr(e.right as SerialExpr, env, hist, now) as number;
      switch (e.op) {
        case '+': return l + r;
        case '-': return l - r;
        case '*': return l * r;
        case '/': return r === 0 ? 0 : l / r;
      }
      return 0;
    }
    case 'unary': return -(evalExpr(e.arg as SerialExpr, env, hist, now) as number);
    case 'call':  return evalBuiltin(e.name as string, (e.args as SerialExpr[]), env, hist, now);
    case 'first_match': {
      for (const n of e.names as string[]) {
        const v = env.get(n);
        if (v !== undefined && v !== '') return v;
      }
      return '';
    }
    default: return 0;
  }
}

function evalBuiltin(name: string, args: SerialExpr[], env: Map<string, EnvValue>, hist: HistoryStore, now: number): number {
  const num = (i: number) => evalExpr(args[i], env, hist, now) as number;

  switch (name) {
    case 'baseline': {
      const sigName = args[0]?.kind === 'var' ? (args[0] as unknown as {name:string}).name : 'unknown';
      const window = num(1) * 1000;
      const pct = num(2);
      let h = hist.get(sigName);
      if (!h) { h = new RollingHistory(); hist.set(sigName, h); }
      const sigVal = env.get(sigName) as number ?? 0;
      h.push(sigVal, now);
      h.trim(window, now);
      return h.percentile(pct);
    }
    case 'shannon_entropy': {
      // Approximate from a small rolling window
      const sigName = args[0]?.kind === 'var' ? (args[0] as unknown as {name:string}).name : 'bvp';
      const h = hist.get(sigName);
      if (!h) return 0.5;
      const vals = (h as any).values as number[];
      if (vals.length < 4) return 0.5;
      const min = Math.min(...vals), max = Math.max(...vals);
      if (max === min) return 0;
      const N = 16;
      const counts = new Array<number>(N).fill(0);
      for (const v of vals) {
        const bin = Math.min(N - 1, Math.floor(((v - min) / (max - min)) * N));
        counts[bin]++;
      }
      let H = 0;
      for (const c of counts) {
        if (c > 0) { const p = c / vals.length; H -= p * Math.log2(p); }
      }
      return H;
    }
    case 'sqrt_compress': return Math.sqrt(Math.max(0.1, num(0)));
    case 'mean': return args.length === 1 ? num(0) : args.reduce((s, _a, i) => s + num(i), 0) / args.length;
    case 'std': {
      if (args.length < 2) return 0;
      const vals = args.map((_a, i) => num(i));
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      return Math.sqrt(vals.reduce((s, v) => s + (v - m) ** 2, 0) / vals.length);
    }
    case 'abs': return Math.abs(num(0));
    case 'log2': return Math.log2(Math.max(num(0), 1e-10));
    case 'exp': return Math.exp(num(0));
    case 'cos': return Math.cos(num(0));
    default: return 0;
  }
}

// ─── Condition evaluator ──────────────────────────────────────────────────────

function evalCond(c: SerialCondition, env: Map<string, EnvValue>): boolean {
  switch (c.kind) {
    case 'cmp': {
      const left = env.get(c.left as string) as number ?? 0;
      const right = c.right as number;
      switch (c.op as string) {
        case '>':  return left > right;
        case '<':  return left < right;
        case '>=': return left >= right;
        case '<=': return left <= right;
        case '==': return left === right;
        case '!=': return left !== right;
      }
      return false;
    }
    case 'is': return (env.get(c.name as string) ?? '') === (c.label as string);
    case 'aligns': return (env.get(c.left as string) ?? '') === (env.get(c.right as string) ?? '');
    case 'and': return evalCond(c.left as SerialCondition, env) && evalCond(c.right as SerialCondition, env);
    case 'or':  return evalCond(c.left as SerialCondition, env) || evalCond(c.right as SerialCondition, env);
    case 'not': return !evalCond(c.cond as SerialCondition, env);
    default: return false;
  }
}

// Collect per-leaf evaluations for trace output
function traceCondLeaves(c: SerialCondition, env: Map<string, EnvValue>): Array<{ cond: string; eval: boolean; value: unknown }> {
  switch (c.kind) {
    case 'cmp': {
      const v = env.get(c.left as string) ?? 0;
      return [{ cond: `${c.left} ${c.op} ${c.right}`, eval: evalCond(c, env), value: v }];
    }
    case 'is': {
      const v = env.get(c.name as string) ?? '';
      return [{ cond: `${c.name} is ${c.label}`, eval: v === c.label, value: v }];
    }
    case 'aligns': {
      const lv = env.get(c.left as string) ?? '';
      const rv = env.get(c.right as string) ?? '';
      return [{ cond: `${c.left} aligns ${c.right}`, eval: lv === rv, value: `${lv} vs ${rv}` }];
    }
    case 'and': return [...traceCondLeaves(c.left as SerialCondition, env), ...traceCondLeaves(c.right as SerialCondition, env)];
    case 'or':  return [...traceCondLeaves(c.left as SerialCondition, env), ...traceCondLeaves(c.right as SerialCondition, env)];
    case 'not': return traceCondLeaves(c.cond as SerialCondition, env);
    default: return [];
  }
}

// ─── HuggingFace default inference adapter ────────────────────────────────────

async function hfInfer(hub: string, _task: string, inputs: Record<string, unknown>, token?: string): Promise<Record<string, unknown>> {
  const url = `https://api-inference.huggingface.co/models/${hub}`;
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  try {
    const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify({ inputs }) });
    if (!res.ok) return { error: `HTTP ${res.status}` };
    return await res.json() as Record<string, unknown>;
  } catch (err) {
    return { error: String(err) };
  }
}

// ─── Runtime ──────────────────────────────────────────────────────────────────

export class BrutScriptRuntime {
  private env = new Map<string, EnvValue>();
  private hist: HistoryStore = new Map();
  private traceLog: TraceEntry[] = [];

  constructor(
    private readonly plan: WiringPlan,
    private readonly opts: RuntimeOptions,
    private readonly sessionStartMs: number = performance.now(),
  ) {}

  // ── Public: push live signals ─────────────────────────────────────────────

  /** Call this once per sensor frame to update the named signal in the env. */
  pushSignal(name: string, value: EnvValue): void {
    this.env.set(name, value);
  }

  /** Call this once per second to drive the full evaluation. */
  async tick(nowMs: number): Promise<void> {
    const t = nowMs - this.sessionStartMs;
    for (const op of this.plan.ops) {
      await this.execOp(op, t, nowMs);
    }
  }

  /** Returns and clears the accumulated trace log since last drain. */
  drain(): TraceEntry[] {
    const out = this.traceLog;
    this.traceLog = [];
    return out;
  }

  /** Read a single env value (for external inspection). */
  read(name: string): EnvValue | undefined {
    return this.env.get(name);
  }

  // ── Op execution ─────────────────────────────────────────────────────────

  private async execOp(op: Op, t: number, nowMs: number): Promise<void> {
    switch (op.kind) {
      case 'source_bind':    this.execSourceBind(op, t); break;
      case 'invert':         this.execInvert(op, t, nowMs); break;
      case 'derive':
      case 'term':
      case 'coord':          this.execExprBind(op, t, nowMs); break;
      case 'regime_classify':this.execRegime(op, t); break;
      case 'failure_detect': this.execFailure(op, t); break;
      case 'pair_eval':      this.execPair(op, t, nowMs); break;
      case 'watch_eval':     this.execWatch(op, t); break;
      case 'model_infer':    await this.execModel(op, t); break;
      case 'explain_align':  this.execExplain(op, t, nowMs); break;
    }
  }

  private execSourceBind(op: Op, t: number): void {
    const v = this.env.get(op.channel!);
    if (v !== undefined) {
      this.emit({ t, block: op.block, step: `bind ${op.output}`, line: op.line, out: { [op.output!]: v } });
    }
  }

  private execInvert(op: Op, t: number, nowMs: number): void {
    const primary = this.env.get(op.inputSignal!) as number ?? 0;
    let result = applyInversion(op.using!, primary, this.env);

    if (op.sqrtCompress) result = Math.sqrt(Math.max(0.1, result));
    if (op.baseline) {
      const windowMs = windowToMs(op.baseline);
      let h = this.hist.get(op.output!);
      if (!h) { h = new RollingHistory(); this.hist.set(op.output!, h); }
      h.push(result, nowMs);
      h.trim(windowMs, nowMs);
    }

    const inputs: Record<string, unknown> = { [op.inputSignal!]: primary };
    for (const g of (op.given ?? [])) inputs[g] = this.env.get(g);

    this.env.set(op.output!, result);
    this.emit({ t, block: op.block, step: `invert ${op.output}`, line: op.line, in: inputs, out: { [op.output!]: result } });
  }

  private execExprBind(op: Op, t: number, nowMs: number): void {
    const result = evalExpr(op.expr!, this.env, this.hist, nowMs) as number;
    const clamped = op.clamp ? Math.max(op.clamp[0], Math.min(op.clamp[1], result)) : result;
    this.env.set(op.output!, clamped);
    this.emit({ t, block: op.block, step: `${op.kind} ${op.output}`, line: op.line, out: { [op.output!]: clamped } });
  }

  private execRegime(op: Op, t: number): void {
    const v = this.env.get(op.regimeInput!) as number ?? 0;
    let label = op.regimeOtherwise ?? 'turbulent';
    for (const c of (op.regimeCases ?? [])) {
      if (evalCond(c.cond, this.env)) { label = c.label; break; }
    }
    this.env.set('regime', label);
    this.env.set(op.block + '.regime', label);
    this.emit({ t, block: op.block, step: 'regime classify', line: op.line, in: { [op.regimeInput!]: v }, out: { regime: label } });
  }

  private execFailure(op: Op, t: number): void {
    let label = 'none';
    for (const c of (op.failureCases ?? [])) {
      if (evalCond(c.cond, this.env)) { label = c.label; break; }
    }
    this.env.set('failure', label);
    this.env.set(op.block + '.failure', label);
    this.emit({ t, block: op.block, step: 'failure detect', line: op.line, out: { failure: label } });
  }

  private execPair(op: Op, t: number, nowMs: number): void {
    const pred = evalExpr(op.predictedExpr!, this.env, this.hist, nowMs) as number;
    const obs  = evalExpr(op.observedExpr!,  this.env, this.hist, nowMs) as number;
    const absErr = Math.abs(pred - obs);
    const relErr = pred !== 0 ? absErr / Math.abs(pred) : absErr;
    const errorName = op.pairName! + '.error';
    this.env.set(errorName, relErr);
    this.emit({
      t, block: op.block, step: `pair ${op.pairName}`, line: op.line,
      out: { predicted: pred, observed: obs, abs_error: absErr, rel_error: relErr },
    });
  }

  private execWatch(op: Op, t: number): void {
    const leaves = traceCondLeaves(op.watchCond!, this.env);
    const fired = evalCond(op.watchCond!, this.env);

    if (fired) {
      this.env.set(op.watchName!, op.watchEmit!);
      this.emit({
        t, block: op.block, step: 'FIRED', line: op.line,
        emit: op.watchEmit, confidence: op.watchConfidence,
        out: leaves.reduce((o, l) => ({ ...o, [l.cond]: { eval: l.eval, value: l.value } }), {}),
      });
    } else {
      this.env.set(op.watchName!, '');
      const failed = leaves.filter(l => !l.eval);
      this.emit({
        t, block: op.block, step: 'not_fired', line: op.line,
        out: failed.reduce((o, l) => ({ ...o, [l.cond]: { eval: false, value: l.value } }), {}),
      });
    }
  }

  private async execModel(op: Op, t: number): Promise<void> {
    const inputs: Record<string, unknown> = {};
    for (const binding of (op.modelInputs ?? [])) {
      for (const sig of binding.signals) {
        const key = binding.field ? binding.from + '.' + binding.field : sig;
        inputs[sig] = this.env.get(key) ?? this.env.get(sig);
      }
    }

    const inferFn: ModelFn = this.opts.model ?? ((hub, task, inp) => hfInfer(hub, task, inp, this.opts.hfToken));
    const result = await inferFn(op.modelHub!, op.modelTask!, inputs);

    const outputs = op.modelOutputs ?? [];
    const resultVals = Array.isArray(result) ? result : Object.values(result);
    outputs.forEach((name, i) => {
      const v = (result as Record<string, unknown>)[name] ?? resultVals[i] ?? 0;
      this.env.set(op.block + '.' + name, v as EnvValue);
      this.env.set(name, v as EnvValue);
    });

    this.emit({ t, block: op.block, step: 'inference', line: op.line, in: inputs, out: result as Record<string, unknown> });
  }

  private execExplain(op: Op, t: number, nowMs: number): void {
    const brut  = evalExpr(op.brutPredExpr!, this.env, this.hist, nowMs);
    const model = evalExpr(op.modelPredExpr!, this.env, this.hist, nowMs);

    if (evalCond(op.agreeCond!, this.env)) {
      this.emit({ t, block: op.block, step: 'agree', line: op.line, out: { brut, model } });
      return;
    }

    for (const branch of (op.divergeBranches ?? [])) {
      if (!evalCond(branch.cond, this.env)) continue;

      const traceSnapshot: Record<string, unknown> = {};
      for (const tr of branch.traces) {
        const key = tr.field ? tr.block + '.' + tr.field : tr.block;
        traceSnapshot[key] = this.env.get(key) ?? this.env.get(tr.block);
      }

      for (const f of branch.flags) {
        this.emit({
          t, block: op.block, step: 'diverge', line: op.line,
          out: { brut, model },
          flag: f,
          traces: traceSnapshot,
        });
      }

      if (branch.flags.length === 0) {
        this.emit({ t, block: op.block, step: 'diverge', line: op.line, out: { brut, model }, traces: traceSnapshot });
      }

      break; // first matching diverge branch wins
    }
  }

  private emit(entry: TraceEntry): void {
    this.traceLog.push(entry);
    this.opts.onTrace(entry);
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function windowToMs(w: TimeWindow): number {
  switch (w.unit) {
    case 's': return w.amount * 1000;
    case 'm': return w.amount * 60_000;
    case 'h': return w.amount * 3_600_000;
    case 'd': return w.amount * 86_400_000;
  }
}
