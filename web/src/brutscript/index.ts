// BrutScript public API.
//
// Usage:
//   import { BrutScript } from './brutscript';
//
//   const bs = new BrutScript(source, { onTrace: entry => console.log(entry) });
//   if (bs.errors.length) { ... }
//
//   // Each sensor frame:
//   bs.push('hr', stats.hrBpm);
//   bs.push('rc_mean', stats.rc);
//   // ...
//
//   // Each 1 Hz tick:
//   await bs.tick();

export { tokenise } from './lexer';
export { parse } from './parser';
export { compile } from './compiler';
export { BrutScriptRuntime } from './runtime';
export type { TraceEntry, RuntimeOptions } from './runtime';
export type { WiringPlan, Op } from './compiler';

export type { Program, Block } from './ast';

import { tokenise } from './lexer';
import { parse } from './parser';
import { compile } from './compiler';
import { BrutScriptRuntime } from './runtime';
import type { RuntimeOptions, TraceEntry } from './runtime';
import type { CompileError } from './compiler';
import type { ParseError } from './parser';

// Re-export for consumers who want strongly-typed errors
export type { CompileError, ParseError };

export type BrutScriptError =
  | { phase: 'lex';    message: string; line: number; col: number }
  | { phase: 'parse';  message: string; line: number; col: number }
  | { phase: 'compile'; message: string; line: number; col: number };

export class BrutScript {
  readonly errors: BrutScriptError[] = [];
  private runtime: BrutScriptRuntime | null = null;

  constructor(
    source: string,
    opts: RuntimeOptions,
    sessionStartMs = performance.now(),
  ) {
    // 1. Lex
    const tokens = tokenise(source);
    const lexErrors = tokens.filter(t => t.type === 9 /* TT.Error */);
    for (const e of lexErrors) {
      this.errors.push({ phase: 'lex', message: `unexpected character '${e.value}'`, line: e.line, col: e.col });
    }

    // 2. Parse
    const { program, errors: parseErrors } = parse(tokens);
    for (const e of parseErrors) {
      this.errors.push({ phase: 'parse', ...e });
    }

    // 3. Compile
    const { plan, errors: compileErrors } = compile(program);
    for (const e of compileErrors) {
      this.errors.push({ phase: 'compile', ...e });
    }

    if (this.errors.length === 0) {
      this.runtime = new BrutScriptRuntime(plan, opts, sessionStartMs);
    }
  }

  /** Push a live signal value into the environment. */
  push(name: string, value: number | string): void {
    this.runtime?.pushSignal(name, value);
  }

  /** Evaluate the plan for this tick. Call once per second. */
  async tick(nowMs = performance.now()): Promise<void> {
    await this.runtime?.tick(nowMs);
  }

  /** Drain accumulated trace entries. */
  drain(): TraceEntry[] {
    return this.runtime?.drain() ?? [];
  }

  /** Read a single env value. */
  read(name: string): number | string | undefined {
    return this.runtime?.read(name);
  }

  get ready(): boolean { return this.runtime !== null; }
}
