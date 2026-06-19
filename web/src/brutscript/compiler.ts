// BrutScript compiler.
//
// Single-pass over the AST:
//   1. Signal-closure check — every referenced identifier must be in scope.
//   2. Type check — numeric vs. string vs. label domains.
//   3. Emit a WiringPlan — a serialisable description of the evaluation order
//      that the runtime executes against the live signal bus.

import type {
  Program, Block, Expr, Condition, TimeWindow,
  SourceBlock, LayerBlock, InvertStmt, DeriveStmt,
  DecomposeBlock, DecomposeStmt, TermStmt, CoordStmt,
  RegimeStmt, FailureStmt, PairStmt,
  WatchBlock, ModelBlock, ExplainBlock,
} from './ast';

// ─── Wiring plan types ────────────────────────────────────────────────────────
// The plan is a flat ordered list of ops. The runtime evaluates them in sequence
// each tick, maintaining a named value store (the environment).

export type OpKind =
  | 'source_bind'       // bind live bus channel to env name
  | 'invert'            // call a named inversion function
  | 'derive'            // evaluate an expression and bind
  | 'term'              // decompose term (same as derive but tagged)
  | 'coord'             // health coordinate (same as derive but tagged)
  | 'regime_classify'   // map a numeric env var to a regime label
  | 'failure_detect'    // map env vars to a failure label
  | 'pair_eval'         // evaluate predicted & observed, compute |error|
  | 'watch_eval'        // evaluate a compound condition, maybe emit event
  | 'model_infer'       // call HuggingFace / local model
  | 'explain_align';    // compare brut vs model, emit agree/diverge

export interface SerialCondition {
  kind: string;
  // For 'cmp': left, op, right
  // For 'is': name, label
  // For 'aligns': left, right
  // For 'and'/'or': left, right (nested SerialCondition)
  // For 'not': cond
  [key: string]: unknown;
}

export interface SerialExpr {
  kind: string;
  [key: string]: unknown;
}

export interface Op {
  kind: OpKind;
  // Common
  block: string;   // source block name for trace attribution
  line: number;
  col: number;
  // Output binding
  output?: string;
  // source_bind
  channel?: string;
  // invert
  inputSignal?: string;
  given?: string[];
  using?: string;
  baseline?: TimeWindow;
  sqrtCompress?: boolean;
  // derive / term / coord
  expr?: SerialExpr;
  clamp?: [number, number];
  // regime_classify
  regimeInput?: string;
  regimeCases?: Array<{ label: string; cond: SerialCondition }>;
  regimeOtherwise?: string;
  // failure_detect
  failureCases?: Array<{ label: string; cond: SerialCondition }>;
  // pair_eval
  pairName?: string;
  predictedExpr?: SerialExpr;
  observedExpr?: SerialExpr;
  // watch_eval
  watchName?: string;
  watchCond?: SerialCondition;
  watchEmit?: string;
  watchConfidence?: number;
  // model_infer
  modelHub?: string;
  modelTask?: string;
  modelInputs?: Array<{ signals: string[]; from: string; field?: string }>;
  modelOutputs?: string[];
  // explain_align
  explainName?: string;
  brutPredExpr?: SerialExpr;
  modelPredExpr?: SerialExpr;
  agreeCond?: SerialCondition;
  divergeBranches?: Array<{
    cond: SerialCondition;
    traces: Array<{ block: string; field?: string }>;
    flags: string[];
  }>;
  annotate?: Array<{ key: string; value: string }>;
}

export interface WiringPlan {
  ops: Op[];
  // Map of block.name → set of signal names it exposes (for sandbox display)
  exports: Record<string, string[]>;
}

// ─── Compiler errors ──────────────────────────────────────────────────────────

export interface CompileError { message: string; line: number; col: number; }

// ─── Compiler ─────────────────────────────────────────────────────────────────

type Domain = 'number' | 'string' | 'label' | 'unknown';

interface ScopeEntry { domain: Domain; block: string; }

export class Compiler {
  private scope = new Map<string, ScopeEntry>();
  readonly errors: CompileError[] = [];
  private ops: Op[] = [];
  private exports: Record<string, string[]> = {};

  compile(program: Program): WiringPlan {
    for (const block of program.blocks) {
      this.compileBlock(block);
    }
    return { ops: this.ops, exports: this.exports };
  }

  // ── Block dispatch ────────────────────────────────────────────────────────

  private compileBlock(block: Block): void {
    switch (block.kind) {
      case 'source':    return this.compileSource(block);
      case 'layer':     return this.compileLayer(block);
      case 'decompose': return this.compileDecompose(block);
      case 'watch':     return this.compileWatch(block);
      case 'model':     return this.compileModel(block);
      case 'explain':   return this.compileExplain(block);
    }
  }

  // ── Source ────────────────────────────────────────────────────────────────

  private compileSource(b: SourceBlock): void {
    this.exports[b.name] = b.signals;
    for (const sig of b.signals) {
      this.define(sig, 'number', b.name, b.loc.line, b.loc.col);
      this.ops.push({
        kind: 'source_bind',
        block: b.name,
        line: b.loc.line,
        col: b.loc.col,
        output: sig,
        channel: sig,
      });
    }
  }

  // ── Layer ─────────────────────────────────────────────────────────────────

  private compileLayer(b: LayerBlock): void {
    this.exports[b.name] = [];
    for (const dep of b.from) this.requireDefined(dep, b.loc.line, b.loc.col);

    for (const stmt of b.stmts) {
      if (stmt.kind === 'invert') this.compileInvert(stmt, b.name);
      else this.compileDeriveStmt(stmt, b.name);
    }
  }

  private compileInvert(s: InvertStmt, blockName: string): void {
    this.requireDefined(s.inputSignal, s.loc.line, s.loc.col);
    for (const g of s.given) this.requireDefined(g, s.loc.line, s.loc.col);
    this.define(s.output, 'number', blockName, s.loc.line, s.loc.col);
    this.exports[blockName].push(s.output);
    this.ops.push({
      kind: 'invert',
      block: blockName,
      line: s.loc.line,
      col: s.loc.col,
      output: s.output,
      inputSignal: s.inputSignal,
      given: s.given,
      using: s.using,
      baseline: s.baseline,
      sqrtCompress: s.sqrtCompress,
    });
  }

  private compileDeriveStmt(s: DeriveStmt, blockName: string): void {
    this.checkExpr(s.expr, s.loc.line, s.loc.col);
    this.define(s.output, 'number', blockName, s.loc.line, s.loc.col);
    this.exports[blockName].push(s.output);
    this.ops.push({
      kind: 'derive',
      block: blockName,
      line: s.loc.line,
      col: s.loc.col,
      output: s.output,
      expr: serialiseExpr(s.expr),
      clamp: s.clamp,
    });
  }

  // ── Decompose ─────────────────────────────────────────────────────────────

  private compileDecompose(b: DecomposeBlock): void {
    this.exports[b.name] = [];
    for (const dep of b.from) this.requireDefined(dep, b.loc.line, b.loc.col);

    for (const stmt of b.stmts) {
      this.compileDecomposeStmt(stmt, b.name);
    }
  }

  private compileDecomposeStmt(stmt: DecomposeStmt, blockName: string): void {
    switch (stmt.kind) {
      case 'term': {
        const s = stmt as unknown as TermStmt;
        this.checkExpr(s.expr, s.loc.line, s.loc.col);
        this.define(s.output, 'number', blockName, s.loc.line, s.loc.col);
        this.exports[blockName].push(s.output);
        this.ops.push({ kind: 'term', block: blockName, line: s.loc.line, col: s.loc.col, output: s.output, expr: serialiseExpr(s.expr) });
        break;
      }
      case 'coord': {
        const s = stmt as CoordStmt;
        this.checkExpr(s.expr, s.loc.line, s.loc.col);
        this.define(s.output, 'number', blockName, s.loc.line, s.loc.col);
        this.exports[blockName].push(s.output);
        this.ops.push({ kind: 'coord', block: blockName, line: s.loc.line, col: s.loc.col, output: s.output, expr: serialiseExpr(s.expr) });
        break;
      }
      case 'regime': {
        const s = stmt as RegimeStmt;
        this.requireDefined(s.input, s.loc.line, s.loc.col);
        this.define('regime', 'label', blockName, s.loc.line, s.loc.col);
        this.exports[blockName].push('regime');
        this.ops.push({
          kind: 'regime_classify', block: blockName, line: s.loc.line, col: s.loc.col,
          output: 'regime',
          regimeInput: s.input,
          regimeCases: s.cases.map(c => ({ label: c.label, cond: serialiseCond(c.cond) })),
          regimeOtherwise: s.otherwise,
        });
        break;
      }
      case 'failure': {
        const s = stmt as FailureStmt;
        this.define('failure', 'label', blockName, s.loc.line, s.loc.col);
        this.exports[blockName].push('failure');
        this.ops.push({
          kind: 'failure_detect', block: blockName, line: s.loc.line, col: s.loc.col,
          output: 'failure',
          failureCases: s.cases.map(c => ({ label: c.label, cond: serialiseCond(c.cond) })),
        });
        break;
      }
      case 'pair': {
        const s = stmt as PairStmt;
        this.checkExpr(s.predicted, s.loc.line, s.loc.col);
        this.checkExpr(s.observed, s.loc.line, s.loc.col);
        const errorName = s.name + '.error';
        this.define(errorName, 'number', blockName, s.loc.line, s.loc.col);
        this.exports[blockName].push(errorName);
        this.ops.push({
          kind: 'pair_eval', block: blockName, line: s.loc.line, col: s.loc.col,
          pairName: s.name,
          predictedExpr: serialiseExpr(s.predicted),
          observedExpr: serialiseExpr(s.observed),
          output: errorName,
        });
        break;
      }
    }
  }

  // ── Watch ─────────────────────────────────────────────────────────────────

  private compileWatch(b: WatchBlock): void {
    this.checkCond(b.cond, b.loc.line, b.loc.col);
    this.ops.push({
      kind: 'watch_eval',
      block: b.name,
      line: b.loc.line,
      col: b.loc.col,
      watchName: b.name,
      watchCond: serialiseCond(b.cond),
      watchEmit: b.emit,
      watchConfidence: b.confidence,
    });
  }

  // ── Model ─────────────────────────────────────────────────────────────────

  private compileModel(b: ModelBlock): void {
    this.exports[b.name] = b.outputs;
    for (const inp of b.inputs) {
      for (const sig of inp.signals) this.requireDefined(sig, b.loc.line, b.loc.col);
    }
    for (const out of b.outputs) {
      this.define(b.name + '.' + out, 'unknown', b.name, b.loc.line, b.loc.col);
    }
    this.ops.push({
      kind: 'model_infer',
      block: b.name,
      line: b.loc.line,
      col: b.loc.col,
      modelHub: b.hub,
      modelTask: b.task,
      modelInputs: b.inputs,
      modelOutputs: b.outputs,
    });
  }

  // ── Explain ───────────────────────────────────────────────────────────────

  private compileExplain(b: ExplainBlock): void {
    this.ops.push({
      kind: 'explain_align',
      block: b.name,
      line: b.loc.line,
      col: b.loc.col,
      explainName: b.name,
      brutPredExpr: serialiseExpr(b.brutPred),
      modelPredExpr: serialiseExpr(b.modelPred),
      agreeCond: serialiseCond(b.agreeCond),
      divergeBranches: b.divergeBranches.map(db => ({
        cond: serialiseCond(db.cond),
        traces: db.traces,
        flags: db.flags,
      })),
      annotate: b.annotate,
    });
  }

  // ── Scope helpers ─────────────────────────────────────────────────────────

  private define(name: string, domain: Domain, block: string, _line: number, _col: number): void {
    this.scope.set(name, { domain, block });
    // Also register dotted form: block.name
    this.scope.set(block + '.' + name, { domain, block });
  }

  private requireDefined(name: string, line: number, col: number): void {
    if (!this.scope.has(name) && !this.scope.has(name.split('.')[0])) {
      this.errors.push({ message: `'${name}' is not defined at this point in the program`, line, col });
    }
  }

  private checkExpr(e: Expr, line: number, col: number): void {
    switch (e.kind) {
      case 'var':
        this.requireDefined(e.name, e.loc.line, e.loc.col);
        break;
      case 'field':
        this.requireDefined(e.obj, e.loc.line, e.loc.col);
        break;
      case 'bin':
        this.checkExpr(e.left, line, col);
        this.checkExpr(e.right, line, col);
        break;
      case 'unary':
        this.checkExpr(e.arg, line, col);
        break;
      case 'call':
        for (const a of e.args) this.checkExpr(a, line, col);
        break;
      case 'first_match':
        for (const n of e.names) this.requireDefined(n, e.loc.line, e.loc.col);
        break;
    }
  }

  private checkCond(c: Condition, line: number, col: number): void {
    switch (c.kind) {
      case 'cmp':    this.requireDefined(c.left, c.loc.line, c.loc.col); break;
      case 'is':     this.requireDefined(c.name, c.loc.line, c.loc.col); break;
      case 'aligns': this.requireDefined(c.left, c.loc.line, c.loc.col); break;
      case 'and':    this.checkCond(c.left, line, col); this.checkCond(c.right, line, col); break;
      case 'or':     this.checkCond(c.left, line, col); this.checkCond(c.right, line, col); break;
      case 'not':    this.checkCond(c.cond, line, col); break;
    }
  }
}

// ─── Serialisation helpers ────────────────────────────────────────────────────
// Convert AST nodes to plain objects (no circular refs) for the wiring plan.

function serialiseExpr(e: Expr): SerialExpr {
  switch (e.kind) {
    case 'num':   return { kind: 'num', value: e.value };
    case 'str':   return { kind: 'str', value: e.value };
    case 'var':   return { kind: 'var', name: e.name };
    case 'field': return { kind: 'field', obj: e.obj, field: e.field };
    case 'bin':   return { kind: 'bin', op: e.op, left: serialiseExpr(e.left), right: serialiseExpr(e.right) };
    case 'unary': return { kind: 'unary', op: e.op, arg: serialiseExpr(e.arg) };
    case 'call':  return { kind: 'call', name: e.name, args: e.args.map(serialiseExpr) };
    case 'first_match': return { kind: 'first_match', names: e.names };
    default: return { kind: 'unknown' };
  }
}

function serialiseCond(c: Condition): SerialCondition {
  switch (c.kind) {
    case 'cmp':    return { kind: 'cmp', left: c.left, op: c.op, right: c.right };
    case 'is':     return { kind: 'is', name: c.name, label: c.label };
    case 'aligns': return { kind: 'aligns', left: c.left, right: c.right };
    case 'and':    return { kind: 'and', left: serialiseCond(c.left), right: serialiseCond(c.right) };
    case 'or':     return { kind: 'or', left: serialiseCond(c.left), right: serialiseCond(c.right) };
    case 'not':    return { kind: 'not', cond: serialiseCond(c.cond) };
    default: return { kind: 'unknown' };
  }
}

// ─── Public entry point ───────────────────────────────────────────────────────

export function compile(program: Program): { plan: WiringPlan; errors: CompileError[] } {
  const c = new Compiler();
  const plan = c.compile(program);
  return { plan, errors: c.errors };
}
