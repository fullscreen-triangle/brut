// BrutScript lexer — converts source text to a flat token stream.

export const enum TT {
  // Literals
  Ident, Number, String,
  // Punctuation
  LBrace, RBrace, LParen, RParen, Comma, Dot, Semicolon, Colon,
  // Operators
  Plus, Minus, Star, Slash, Eq, EqEq, BangEq, Gt, Lt, GtEq, LtEq,
  // Keywords
  Source, Signal, Rate, Hz,
  Layer, From, Invert, Given, Using, Baseline, SqrtCompress, Clamp, Derive,
  Decompose, Term, Coord, Regime, Classify, When, Otherwise,
  Failure, Detect, None, Pair, Predicted, Observed,
  Watch, And, Or, Not, Emit, Confidence, Is, Aligns,
  Model, Hub, Task, Input, Output,
  Explain, Match, Agree, Diverge, Trace, Flag, Annotate, With, FirstMatch,
  Window, Percentile, P5, P10, P25, P50, P75, P90, P95, P99,
  // Time units
  S, M, H, D,
  // EOF / error
  EOF, Error,
}

export interface Token {
  type: TT;
  value: string;
  line: number;
  col: number;
}

const KEYWORDS: Record<string, TT> = {
  source: TT.Source, signal: TT.Signal, rate: TT.Rate, hz: TT.Hz,
  layer: TT.Layer, from: TT.From, invert: TT.Invert, given: TT.Given,
  using: TT.Using, baseline: TT.Baseline, sqrt_compress: TT.SqrtCompress,
  clamp: TT.Clamp, derive: TT.Derive,
  decompose: TT.Decompose, term: TT.Term, coord: TT.Coord,
  regime: TT.Regime, classify: TT.Classify, when: TT.When,
  otherwise: TT.Otherwise, failure: TT.Failure, detect: TT.Detect,
  none: TT.None, pair: TT.Pair, predicted: TT.Predicted, observed: TT.Observed,
  watch: TT.Watch, and: TT.And, or: TT.Or, not: TT.Not, emit: TT.Emit,
  confidence: TT.Confidence, is: TT.Is, aligns: TT.Aligns,
  model: TT.Model, hub: TT.Hub, task: TT.Task, input: TT.Input, output: TT.Output,
  explain: TT.Explain, match: TT.Match, agree: TT.Agree, diverge: TT.Diverge,
  trace: TT.Trace, flag: TT.Flag, annotate: TT.Annotate, with: TT.With,
  first_match: TT.FirstMatch, window: TT.Window, percentile: TT.Percentile,
  p5: TT.P5, p10: TT.P10, p25: TT.P25, p50: TT.P50,
  p75: TT.P75, p90: TT.P90, p95: TT.P95, p99: TT.P99,
  s: TT.S, m: TT.M, h: TT.H, d: TT.D,
};

export function tokenise(src: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  let line = 1;
  let lineStart = 0;

  function col(): number { return i - lineStart + 1; }
  function tok(type: TT, value: string, l = line, c = col()): Token {
    return { type, value, line: l, col: c };
  }

  while (i < src.length) {
    const c = src[i];

    // Whitespace
    if (c === '\n') { line++; lineStart = i + 1; i++; continue; }
    if (c === ' ' || c === '\t' || c === '\r') { i++; continue; }

    // Line comment
    if (src[i] === '-' && src[i + 1] === '-') {
      while (i < src.length && src[i] !== '\n') i++;
      continue;
    }

    const l = line, cl = col();

    // String
    if (c === '"') {
      i++;
      let s = '';
      while (i < src.length && src[i] !== '"') {
        if (src[i] === '\\' && i + 1 < src.length) { i++; s += src[i]; }
        else s += src[i];
        i++;
      }
      i++; // closing "
      tokens.push(tok(TT.String, s, l, cl));
      continue;
    }

    // Number
    if (c >= '0' && c <= '9' || (c === '.' && src[i + 1] >= '0' && src[i + 1] <= '9')) {
      let s = '';
      while (i < src.length && (src[i] >= '0' && src[i] <= '9' || src[i] === '.')) s += src[i++];
      tokens.push(tok(TT.Number, s, l, cl));
      continue;
    }

    // Identifier or keyword
    if (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c === '_') {
      let s = '';
      while (i < src.length && (src[i] >= 'a' && src[i] <= 'z' || src[i] >= 'A' && src[i] <= 'Z' || src[i] >= '0' && src[i] <= '9' || src[i] === '_')) s += src[i++];
      const kw = KEYWORDS[s.toLowerCase()];
      tokens.push(tok(kw !== undefined ? kw : TT.Ident, s, l, cl));
      continue;
    }

    // Two-char operators
    if (i + 1 < src.length) {
      const two = src[i] + src[i + 1];
      const t2: Record<string, TT> = { '>=': TT.GtEq, '<=': TT.LtEq, '==': TT.EqEq, '!=': TT.BangEq };
      if (t2[two] !== undefined) { tokens.push(tok(t2[two], two, l, cl)); i += 2; continue; }
    }

    // Single-char tokens
    const single: Record<string, TT> = {
      '{': TT.LBrace, '}': TT.RBrace, '(': TT.LParen, ')': TT.RParen,
      ',': TT.Comma, '.': TT.Dot, ';': TT.Semicolon, ':': TT.Colon,
      '+': TT.Plus, '-': TT.Minus, '*': TT.Star, '/': TT.Slash,
      '=': TT.Eq, '>': TT.Gt, '<': TT.Lt,
    };
    if (single[c] !== undefined) { tokens.push(tok(single[c], c, l, cl)); i++; continue; }

    tokens.push(tok(TT.Error, c, l, cl));
    i++;
  }

  tokens.push(tok(TT.EOF, '', line, col()));
  return tokens;
}
