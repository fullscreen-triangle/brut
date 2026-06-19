// BrutScript AST node types.
// Every node carries a source location for trace attribution.

export interface Loc { line: number; col: number; }

// ─── Expressions ─────────────────────────────────────────────────────────────

export type Expr =
  | NumLit
  | StrLit
  | VarRef
  | FieldRef
  | BinOp
  | UnaryOp
  | CallExpr
  | FirstMatchExpr;

export interface NumLit    { kind: 'num';   value: number; loc: Loc; }
export interface StrLit    { kind: 'str';   value: string; loc: Loc; }
export interface VarRef    { kind: 'var';   name: string;  loc: Loc; }
export interface FieldRef  { kind: 'field'; obj: string; field: string; loc: Loc; }
export interface BinOp     { kind: 'bin';   op: '+' | '-' | '*' | '/'; left: Expr; right: Expr; loc: Loc; }
export interface UnaryOp   { kind: 'unary'; op: '-'; arg: Expr; loc: Loc; }
export interface CallExpr  { kind: 'call';  name: string; args: Expr[]; loc: Loc; }
export interface FirstMatchExpr { kind: 'first_match'; names: string[]; loc: Loc; }

// ─── Conditions ───────────────────────────────────────────────────────────────

export type Condition =
  | CmpCond
  | IsCond
  | AlignsCond
  | AndCond
  | OrCond
  | NotCond;

export interface CmpCond     { kind: 'cmp';    left: string; op: '>' | '<' | '>=' | '<=' | '==' | '!='; right: number; loc: Loc; }
export interface IsCond      { kind: 'is';     name: string; label: string; loc: Loc; }
export interface AlignsCond  { kind: 'aligns'; left: string; right: string; loc: Loc; }
export interface AndCond     { kind: 'and';    left: Condition; right: Condition; loc: Loc; }
export interface OrCond      { kind: 'or';     left: Condition; right: Condition; loc: Loc; }
export interface NotCond     { kind: 'not';    cond: Condition; loc: Loc; }

// ─── Time window ─────────────────────────────────────────────────────────────

export type TimeUnit = 's' | 'm' | 'h' | 'd';
export interface TimeWindow { amount: number; unit: TimeUnit; }

// ─── Source block ─────────────────────────────────────────────────────────────

export interface SourceBlock {
  kind: 'source';
  name: string;
  signals: string[];
  rateHz: number;
  loc: Loc;
}

// ─── Layer block ─────────────────────────────────────────────────────────────

export type LayerStmt = InvertStmt | DeriveStmt;

export interface InvertStmt {
  kind: 'invert';
  output: string;
  inputSignal: string;
  given: string[];
  using: string;       // e.g. "beer_lambert.blue"
  baseline?: TimeWindow;
  sqrtCompress: boolean;
  loc: Loc;
}

export interface DeriveStmt {
  kind: 'derive';
  output: string;
  expr: Expr;
  clamp?: [number, number];
  loc: Loc;
}

export interface LayerBlock {
  kind: 'layer';
  name: string;
  from: string[];
  stmts: LayerStmt[];
  loc: Loc;
}

// ─── Decompose block ─────────────────────────────────────────────────────────

export type DecomposeStmt =
  | TermStmt
  | CoordStmt
  | RegimeStmt
  | FailureStmt
  | PairStmt;

export interface TermStmt {
  kind: 'term';
  output: string;
  expr: Expr;
  loc: Loc;
}

export interface CoordStmt {
  kind: 'coord';
  output: string;
  expr: Expr;
  loc: Loc;
}

export interface RegimeCase {
  label: string;
  cond: Condition;
}

export interface RegimeStmt {
  kind: 'regime';
  input: string;
  cases: RegimeCase[];
  otherwise: string;
  loc: Loc;
}

export interface FailureCase {
  label: string;
  cond: Condition;
}

export interface FailureStmt {
  kind: 'failure';
  cases: FailureCase[];
  loc: Loc;
}

export interface PairStmt {
  kind: 'pair';
  name: string;
  predicted: Expr;
  observed: Expr;
  loc: Loc;
}

export interface DecomposeBlock {
  kind: 'decompose';
  name: string;
  from: string[];
  stmts: DecomposeStmt[];
  loc: Loc;
}

// ─── Watch block ──────────────────────────────────────────────────────────────

export interface WatchBlock {
  kind: 'watch';
  name: string;
  cond: Condition;
  emit: string;
  confidence: number;
  loc: Loc;
}

// ─── Model block ──────────────────────────────────────────────────────────────

export interface InputBinding {
  signals: string[];   // identifiers to pass
  from: string;        // source block name
  field?: string;      // optional .field qualifier
}

export interface ModelBlock {
  kind: 'model';
  name: string;
  hub: string;
  task: string;
  inputs: InputBinding[];
  outputs: string[];
  loc: Loc;
}

// ─── Explain block ────────────────────────────────────────────────────────────

export interface DivergeBranch {
  cond: Condition;
  traces: Array<{ block: string; field?: string }>;
  flags: string[];
  loc: Loc;
}

export interface AnnotateItem {
  key: string;
  value: string;
}

export interface ExplainBlock {
  kind: 'explain';
  name: string;
  brutPred: Expr;
  modelPred: Expr;           // e.g. FieldRef model.label
  agreeCond: Condition;
  divergeBranches: DivergeBranch[];
  annotate: AnnotateItem[];
  loc: Loc;
}

// ─── Top-level program ───────────────────────────────────────────────────────

export type Block =
  | SourceBlock
  | LayerBlock
  | DecomposeBlock
  | WatchBlock
  | ModelBlock
  | ExplainBlock;

export interface Program {
  blocks: Block[];
}
