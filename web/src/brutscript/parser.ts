// BrutScript recursive-descent parser.
// Converts the token stream from the lexer into a typed AST.

import { type Token, TT } from './lexer';
import type {
  Program, Block, Loc,
  Expr, NumLit, StrLit, VarRef, FieldRef, BinOp, UnaryOp, CallExpr, FirstMatchExpr,
  Condition, CmpCond, IsCond, AlignsCond,
  TimeWindow, TimeUnit,
  SourceBlock,
  LayerBlock, LayerStmt, InvertStmt, DeriveStmt,
  DecomposeBlock, DecomposeStmt, TermStmt, CoordStmt,
  RegimeStmt, RegimeCase, FailureStmt, FailureCase, PairStmt,
  WatchBlock,
  ModelBlock, InputBinding,
  ExplainBlock, DivergeBranch, AnnotateItem,
} from './ast';

export interface ParseError { message: string; line: number; col: number; }

export class Parser {
  private pos = 0;
  readonly errors: ParseError[] = [];

  constructor(private readonly tokens: Token[]) {}

  // ── Public entry ──────────────────────────────────────────────────────────

  parse(): Program {
    const blocks: Block[] = [];
    while (!this.at(TT.EOF)) {
      const b = this.parseBlock();
      if (b) blocks.push(b);
    }
    return { blocks };
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  private peek(): Token { return this.tokens[this.pos] ?? this.tokens[this.tokens.length - 1]; }
  private at(type: TT): boolean { return this.peek().type === type; }
  private atAny(...types: TT[]): boolean { return types.includes(this.peek().type); }

  private advance(): Token { const t = this.peek(); if (!this.at(TT.EOF)) this.pos++; return t; }

  private expect(type: TT, msg?: string): Token {
    if (!this.at(type)) {
      const t = this.peek();
      this.errors.push({ message: msg ?? `expected token type ${type}, got '${t.value}'`, line: t.line, col: t.col });
      return t;
    }
    return this.advance();
  }

  private tryConsume(type: TT): boolean {
    if (this.at(type)) { this.advance(); return true; }
    return false;
  }

  private loc(): Loc { return { line: this.peek().line, col: this.peek().col }; }

  private identOrKeyword(): string {
    // Many keyword tokens are also valid identifier names in positions where a
    // name is expected (e.g. block names, signal names, using paths).
    const t = this.peek();
    if (t.type !== TT.EOF && t.type !== TT.LBrace && t.type !== TT.RBrace) {
      this.advance();
      return t.value;
    }
    return this.expect(TT.Ident).value;
  }

  private dottedName(): string {
    let s = this.identOrKeyword();
    while (this.tryConsume(TT.Dot)) s += '.' + this.identOrKeyword();
    return s;
  }

  // ── Top-level block dispatch ──────────────────────────────────────────────

  private parseBlock(): Block | null {
    const t = this.peek();
    switch (t.type) {
      case TT.Source:   return this.parseSource();
      case TT.Layer:    return this.parseLayer();
      case TT.Decompose:return this.parseDecompose();
      case TT.Watch:    return this.parseWatch();
      case TT.Model:    return this.parseModel();
      case TT.Explain:  return this.parseExplain();
      default:
        this.errors.push({ message: `unexpected token '${t.value}' at top level`, line: t.line, col: t.col });
        this.advance();
        return null;
    }
  }

  // ─── source ───────────────────────────────────────────────────────────────

  private parseSource(): SourceBlock {
    const loc = this.loc();
    this.expect(TT.Source);
    const name = this.identOrKeyword();
    this.expect(TT.LBrace);
    const signals: string[] = [];
    let rateHz = 30;

    while (!this.at(TT.RBrace) && !this.at(TT.EOF)) {
      if (this.tryConsume(TT.Signal)) {
        signals.push(this.identOrKeyword());
        while (this.tryConsume(TT.Comma)) signals.push(this.identOrKeyword());
      } else if (this.tryConsume(TT.Rate)) {
        rateHz = parseFloat(this.expect(TT.Number).value);
        this.tryConsume(TT.Hz);
      } else {
        this.errors.push({ message: `unexpected '${this.peek().value}' in source block`, ...this.loc() });
        this.advance();
      }
    }

    this.expect(TT.RBrace);
    return { kind: 'source', name, signals, rateHz, loc };
  }

  // ─── layer ───────────────────────────────────────────────────────────────

  private parseLayer(): LayerBlock {
    const loc = this.loc();
    this.expect(TT.Layer);
    const name = this.identOrKeyword();
    this.expect(TT.From);
    const from = this.parseIdentList();
    this.expect(TT.LBrace);
    const stmts: LayerStmt[] = [];

    while (!this.at(TT.RBrace) && !this.at(TT.EOF)) {
      if (this.at(TT.Invert)) stmts.push(this.parseInvert());
      else if (this.at(TT.Derive)) stmts.push(this.parseDerive());
      else { this.errors.push({ message: `unexpected '${this.peek().value}' in layer block`, ...this.loc() }); this.advance(); }
    }

    this.expect(TT.RBrace);
    return { kind: 'layer', name, from, stmts, loc };
  }

  private parseInvert(): InvertStmt {
    const loc = this.loc();
    this.expect(TT.Invert);
    const output = this.identOrKeyword();
    this.expect(TT.From);
    const inputSignal = this.identOrKeyword();

    let given: string[] = [];
    if (this.tryConsume(TT.Given)) {
      given = this.parseIdentList();
    }

    this.expect(TT.Using);
    const using = this.dottedName();

    let baseline: TimeWindow | undefined;
    if (this.tryConsume(TT.Baseline)) {
      baseline = this.parseTimeWindow();
    }

    const sqrtCompress = this.tryConsume(TT.SqrtCompress);
    return { kind: 'invert', output, inputSignal, given, using, baseline, sqrtCompress, loc };
  }

  private parseDerive(): DeriveStmt {
    const loc = this.loc();
    this.expect(TT.Derive);
    const output = this.identOrKeyword();
    this.expect(TT.Eq);
    const expr = this.parseExpr();

    let clamp: [number, number] | undefined;
    if (this.tryConsume(TT.Clamp)) {
      this.expect(TT.LBrace);
      const lo = parseFloat(this.expect(TT.Number).value);
      this.expect(TT.Comma);
      const hi = parseFloat(this.expect(TT.Number).value);
      this.expect(TT.RBrace);
      clamp = [lo, hi];
    }

    return { kind: 'derive', output, expr, clamp, loc };
  }

  // ─── decompose ───────────────────────────────────────────────────────────

  private parseDecompose(): DecomposeBlock {
    const loc = this.loc();
    this.expect(TT.Decompose);
    const name = this.identOrKeyword();
    this.expect(TT.From);
    const from = this.parseIdentList();
    this.expect(TT.LBrace);
    const stmts: DecomposeStmt[] = [];

    while (!this.at(TT.RBrace) && !this.at(TT.EOF)) {
      if (this.at(TT.Term))    stmts.push(this.parseTerm());
      else if (this.at(TT.Coord))   stmts.push(this.parseCoord());
      else if (this.at(TT.Regime))  stmts.push(this.parseRegime());
      else if (this.at(TT.Failure)) stmts.push(this.parseFailure());
      else if (this.at(TT.Pair))    stmts.push(this.parsePair());
      else { this.errors.push({ message: `unexpected '${this.peek().value}' in decompose block`, ...this.loc() }); this.advance(); }
    }

    this.expect(TT.RBrace);
    return { kind: 'decompose', name, from, stmts, loc };
  }

  private parseTerm(): TermStmt {
    const loc = this.loc();
    this.expect(TT.Term);
    const output = this.identOrKeyword();
    this.expect(TT.Eq);
    const expr = this.parseExpr();
    return { kind: 'term', output, expr, loc };
  }

  private parseCoord(): CoordStmt {
    const loc = this.loc();
    this.expect(TT.Coord);
    const output = this.identOrKeyword();
    this.expect(TT.Eq);
    const expr = this.parseExpr();
    return { kind: 'coord', output, expr, loc };
  }

  private parseRegime(): RegimeStmt {
    const loc = this.loc();
    this.expect(TT.Regime);
    this.expect(TT.Eq);
    this.expect(TT.Classify);
    this.expect(TT.LParen);
    const input = this.identOrKeyword();
    this.expect(TT.RParen);
    this.expect(TT.LBrace);

    const cases: RegimeCase[] = [];
    let otherwise = 'turbulent';

    while (!this.at(TT.RBrace) && !this.at(TT.EOF)) {
      if (this.tryConsume(TT.Otherwise)) {
        otherwise = this.identOrKeyword();
      } else {
        const label = this.identOrKeyword();
        this.expect(TT.When);
        const cond = this.parseAtomCond();
        cases.push({ label, cond });
      }
    }

    this.expect(TT.RBrace);
    return { kind: 'regime', input, cases, otherwise, loc };
  }

  private parseFailure(): FailureStmt {
    const loc = this.loc();
    this.expect(TT.Failure);
    this.expect(TT.Eq);
    this.expect(TT.Detect);
    this.expect(TT.LBrace);

    const cases: FailureCase[] = [];

    while (!this.at(TT.RBrace) && !this.at(TT.EOF)) {
      if (this.atAny(TT.None)) { this.advance(); this.expect(TT.Otherwise); break; }
      const label = this.identOrKeyword();
      this.expect(TT.When);
      const cond = this.parseCond();
      cases.push({ label, cond });
    }

    this.expect(TT.RBrace);
    return { kind: 'failure', cases, loc };
  }

  private parsePair(): PairStmt {
    const loc = this.loc();
    this.expect(TT.Pair);
    const name = this.identOrKeyword();
    this.expect(TT.Predicted);
    this.expect(TT.Eq);
    const predicted = this.parseExpr();
    this.expect(TT.Observed);
    this.expect(TT.Eq);
    const observed = this.parseExpr();
    return { kind: 'pair', name, predicted, observed, loc };
  }

  // ─── watch ───────────────────────────────────────────────────────────────

  private parseWatch(): WatchBlock {
    const loc = this.loc();
    this.expect(TT.Watch);
    const name = this.identOrKeyword();
    this.expect(TT.LBrace);
    this.expect(TT.When);
    const cond = this.parseCond();

    this.expect(TT.Emit);
    const emit = this.expect(TT.String).value;
    this.expect(TT.Confidence);
    const confidence = parseFloat(this.expect(TT.Number).value);

    this.expect(TT.RBrace);
    return { kind: 'watch', name, cond, emit, confidence, loc };
  }

  // ─── model ───────────────────────────────────────────────────────────────

  private parseModel(): ModelBlock {
    const loc = this.loc();
    this.expect(TT.Model);
    const name = this.identOrKeyword();
    this.expect(TT.LBrace);

    let hub = '';
    let task = '';
    const inputs: InputBinding[] = [];
    const outputs: string[] = [];

    while (!this.at(TT.RBrace) && !this.at(TT.EOF)) {
      if (this.tryConsume(TT.Hub)) {
        hub = this.expect(TT.String).value;
      } else if (this.tryConsume(TT.Task)) {
        task = this.identOrKeyword();
      } else if (this.tryConsume(TT.Input)) {
        this.expect(TT.LBrace);
        while (!this.at(TT.RBrace) && !this.at(TT.EOF)) {
          const signals: string[] = [this.identOrKeyword()];
          while (this.tryConsume(TT.Comma)) signals.push(this.identOrKeyword());
          this.expect(TT.From);
          const from = this.identOrKeyword();
          let field: string | undefined;
          if (this.tryConsume(TT.Dot)) field = this.identOrKeyword();
          inputs.push({ signals, from, field });
        }
        this.expect(TT.RBrace);
      } else if (this.tryConsume(TT.Output)) {
        outputs.push(this.identOrKeyword());
        while (this.tryConsume(TT.Comma)) outputs.push(this.identOrKeyword());
      } else {
        this.errors.push({ message: `unexpected '${this.peek().value}' in model block`, ...this.loc() });
        this.advance();
      }
    }

    this.expect(TT.RBrace);
    return { kind: 'model', name, hub, task, inputs, outputs, loc };
  }

  // ─── explain ─────────────────────────────────────────────────────────────

  private parseExplain(): ExplainBlock {
    const loc = this.loc();
    this.expect(TT.Explain);
    const name = this.identOrKeyword();
    this.expect(TT.LBrace);

    let brutPred: Expr = { kind: 'num', value: 0, loc };
    let modelPred: Expr = { kind: 'num', value: 0, loc };
    let agreeCond: Condition = { kind: 'cmp', left: '_agree', op: '==', right: 1, loc };
    const divergeBranches: DivergeBranch[] = [];
    const annotate: AnnotateItem[] = [];

    while (!this.at(TT.RBrace) && !this.at(TT.EOF)) {
      const kw = this.peek().value;
      if (kw === 'brut_prediction') {
        this.advance(); this.expect(TT.Eq);
        brutPred = this.parseExpr();
      } else if (kw === 'model_prediction' || kw === 'model_confidence') {
        this.advance(); this.expect(TT.Eq);
        modelPred = this.parseExpr();
      } else if (this.tryConsume(TT.Match)) {
        this.expect(TT.LBrace);
        while (!this.at(TT.RBrace) && !this.at(TT.EOF)) {
          if (this.tryConsume(TT.Agree)) {
            this.expect(TT.When);
            agreeCond = this.parseCond();
          } else if (this.tryConsume(TT.Diverge)) {
            const dloc = this.loc();
            this.expect(TT.When);
            const cond = this.parseCond();
            this.expect(TT.LBrace);
            const traces: DivergeBranch['traces'] = [];
            const flags: string[] = [];
            while (!this.at(TT.RBrace) && !this.at(TT.EOF)) {
              if (this.tryConsume(TT.Trace)) {
                const block = this.identOrKeyword();
                let field: string | undefined;
                if (this.tryConsume(TT.Dot)) field = this.identOrKeyword();
                traces.push({ block, field });
              } else if (this.tryConsume(TT.Flag)) {
                flags.push(this.expect(TT.String).value);
              } else {
                this.errors.push({ message: `unexpected '${this.peek().value}' in diverge block`, ...this.loc() });
                this.advance();
              }
            }
            this.expect(TT.RBrace);
            divergeBranches.push({ cond, traces, flags, loc: dloc });
          } else {
            this.errors.push({ message: `unexpected '${this.peek().value}' in match block`, ...this.loc() });
            this.advance();
          }
        }
        this.expect(TT.RBrace);
      } else if (this.tryConsume(TT.Annotate)) {
        this.identOrKeyword(); // consume target name (model_prediction etc.)
        this.expect(TT.With);
        this.expect(TT.LBrace);
        while (!this.at(TT.RBrace) && !this.at(TT.EOF)) {
          const key = this.identOrKeyword();
          this.expect(TT.Colon);
          const value = this.identOrKeyword();
          annotate.push({ key, value });
        }
        this.expect(TT.RBrace);
      } else {
        this.errors.push({ message: `unexpected '${kw}' in explain block`, ...this.loc() });
        this.advance();
      }
    }

    this.expect(TT.RBrace);
    return { kind: 'explain', name, brutPred, modelPred, agreeCond, divergeBranches, annotate, loc };
  }

  // ─── Conditions ───────────────────────────────────────────────────────────

  private parseCond(): Condition {
    let left = this.parseOrCond();
    while (this.tryConsume(TT.And)) {
      const right = this.parseOrCond();
      left = { kind: 'and', left, right, loc: left.loc };
    }
    return left;
  }

  private parseOrCond(): Condition {
    let left = this.parseUnaryOrAtomCond();
    while (this.tryConsume(TT.Or)) {
      const right = this.parseUnaryOrAtomCond();
      left = { kind: 'or', left, right, loc: left.loc };
    }
    return left;
  }

  private parseUnaryOrAtomCond(): Condition {
    if (this.tryConsume(TT.Not)) {
      const loc = this.loc();
      const cond = this.parseAtomCond();
      return { kind: 'not', cond, loc };
    }
    return this.parseAtomCond();
  }

  private parseAtomCond(): Condition {
    const loc = this.loc();
    const left = this.identOrKeyword();
    const t = this.peek();

    if (t.type === TT.Is) {
      this.advance();
      const label = this.identOrKeyword();
      return { kind: 'is', name: left, label, loc } as IsCond;
    }

    if (t.type === TT.Aligns) {
      this.advance();
      const right = this.identOrKeyword();
      return { kind: 'aligns', left, right, loc } as AlignsCond;
    }

    const opMap: Record<number, CmpCond['op']> = {
      [TT.Gt]: '>', [TT.Lt]: '<', [TT.GtEq]: '>=', [TT.LtEq]: '<=',
      [TT.EqEq]: '==', [TT.BangEq]: '!=',
    };
    if (opMap[t.type] !== undefined) {
      this.advance();
      const right = parseFloat(this.expect(TT.Number).value);
      return { kind: 'cmp', left, op: opMap[t.type], right, loc } as CmpCond;
    }

    this.errors.push({ message: `expected condition operator after '${left}'`, line: t.line, col: t.col });
    return { kind: 'cmp', left, op: '==', right: 0, loc } as CmpCond;
  }

  // ─── Expressions ──────────────────────────────────────────────────────────

  private parseExpr(): Expr { return this.parseAddSub(); }

  private parseAddSub(): Expr {
    let left = this.parseMulDiv();
    while (this.atAny(TT.Plus, TT.Minus)) {
      const op = this.advance().value as '+' | '-';
      const right = this.parseMulDiv();
      left = { kind: 'bin', op, left, right, loc: left.loc } as BinOp;
    }
    return left;
  }

  private parseMulDiv(): Expr {
    let left = this.parseUnary();
    while (this.atAny(TT.Star, TT.Slash)) {
      const op = this.advance().value as '*' | '/';
      const right = this.parseUnary();
      left = { kind: 'bin', op, left, right, loc: left.loc } as BinOp;
    }
    return left;
  }

  private parseUnary(): Expr {
    if (this.at(TT.Minus)) {
      const loc = this.loc();
      this.advance();
      return { kind: 'unary', op: '-', arg: this.parsePrimary(), loc } as UnaryOp;
    }
    return this.parsePrimary();
  }

  private parsePrimary(): Expr {
    const loc = this.loc();
    const t = this.peek();

    if (t.type === TT.Number) {
      this.advance();
      return { kind: 'num', value: parseFloat(t.value), loc } as NumLit;
    }

    if (t.type === TT.String) {
      this.advance();
      return { kind: 'str', value: t.value, loc } as StrLit;
    }

    if (t.type === TT.LParen) {
      this.advance();
      const e = this.parseExpr();
      this.expect(TT.RParen);
      return e;
    }

    if (t.type === TT.FirstMatch) {
      this.advance();
      this.expect(TT.LParen);
      const names: string[] = [this.identOrKeyword()];
      while (this.tryConsume(TT.Comma)) names.push(this.identOrKeyword());
      this.expect(TT.RParen);
      return { kind: 'first_match', names, loc } as FirstMatchExpr;
    }

    // Identifier — could be a plain var, a function call, or a field ref
    if (t.type !== TT.EOF) {
      const name = this.identOrKeyword();

      if (this.tryConsume(TT.LParen)) {
        // function call
        const args: Expr[] = [];
        if (!this.at(TT.RParen)) {
          args.push(this.parseExpr());
          while (this.tryConsume(TT.Comma)) args.push(this.parseExpr());
        }
        this.expect(TT.RParen);
        return { kind: 'call', name, args, loc } as CallExpr;
      }

      if (this.tryConsume(TT.Dot)) {
        const field = this.identOrKeyword();
        return { kind: 'field', obj: name, field, loc } as FieldRef;
      }

      return { kind: 'var', name, loc } as VarRef;
    }

    this.errors.push({ message: `unexpected token '${t.value}' in expression`, line: t.line, col: t.col });
    return { kind: 'num', value: 0, loc };
  }

  // ─── Helpers ──────────────────────────────────────────────────────────────

  private parseIdentList(): string[] {
    const list: string[] = [this.identOrKeyword()];
    while (this.tryConsume(TT.Comma)) list.push(this.identOrKeyword());
    return list;
  }

  private parseTimeWindow(): TimeWindow {
    const amount = parseFloat(this.expect(TT.Number).value);
    const unitMap: Record<number, TimeUnit> = {
      [TT.S]: 's', [TT.M]: 'm', [TT.H]: 'h', [TT.D]: 'd',
    };
    const unit: TimeUnit = unitMap[this.peek().type] ?? 's';
    if (unitMap[this.peek().type] !== undefined) this.advance();
    return { amount, unit };
  }
}

export function parse(tokens: Token[]): { program: Program; errors: ParseError[] } {
  const p = new Parser(tokens);
  return { program: p.parse(), errors: p.errors };
}
