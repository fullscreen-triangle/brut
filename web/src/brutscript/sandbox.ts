// BrutScript sandbox — VSCode-style IDE mounted as a full-page overlay.
//
// Layout (three-pane):
//
//   ┌─ activity bar ─┬─── editor pane ────────────┬─── output pane ───────────┐
//   │  (icon rail)   │  Monaco-style textarea      │  tabs: charts | glb | log │
//   │                │  + compile/run controls      │                           │
//   └────────────────┴────────────────────────────┴───────────────────────────┘
//
// Editor: plain <textarea> with syntax-colour overlay (no Monaco dependency —
// keeps the bundle light). A CSS `.token` layer overlays coloured spans for
// keywords. The textarea captures Tab as two-space indent.
//
// Output tabs:
//   Charts  — D3 time-series + coherence gauge + divergence bar
//   Models  — GLB catalogue selector + 3D viewer + parse report
//   Console — raw NDJSON trace stream (filterable)

import { BrutScript } from './index';
import type { TraceEntry } from './runtime';
import { mountChartPanel } from './sandbox-charts';
import {
  BeatClock,
  effortIndex,
  effortRegime,
  protocolFor,
  type BeatPosition,
} from './beatclock';
import {
  EXERCISE_ROSTER,
  FIRST_EXERCISE,
  evaluateTransition,
  type ExerciseAgent,
  type PhraseEvidence,
} from './exercise-agents';
import {
  CARDIAC_GLB_CATALOGUE,
  parseGlb,
  mountGlbViewer,
  renderParseReport,
  type GlbViewerHandle,
  type GlbParseReport,
} from './sandbox-glb';

// ─── Default script shown on first open ──────────────────────────────────────

const DEFAULT_SCRIPT = `-- BrutScript: cardiac stress audit
-- Edit this script and press ▶ Run to compile and execute against live signals.

source rppg {
  signal bvp, rc_mean, hr, rmssd, sk, st, se
  rate   30hz
}

source face_rgb {
  signal r, g, b
  rate   1hz
}

source motor {
  signal mean_iki, rt_ratio
  rate   1hz
}

-- Beer-Lambert optical inversion
layer skin_optics from face_rgb {
  invert melanin      from b   using beer_lambert.blue
  invert hb_conc      from r   given melanin  using beer_lambert.red
  invert spo2         from g   given melanin, hb_conc  using beer_lambert.green
  invert vasodilation from bvp using beer_lambert.blue  baseline 30s  sqrt_compress
  derive t_skin = 33.0 + 4.0 * (vasodilation - 1.0)  clamp [27, 37]
}

-- PCHR decomposition
decompose pchr from rppg, skin_optics {
  term hr_intrinsic  = baseline(hr, 300, 5)
  term dhr_metabolic = 0.08 * (t_skin - 33.0) * hr_intrinsic
  term dhr_hypoxic   = 0.15 * (1.0 - spo2) * hr_intrinsic
  term dhr_autonomic = hr - hr_intrinsic - dhr_metabolic - dhr_hypoxic
}

-- S-entropy coordinates and regime
decompose s_entropy from rppg {
  coord sk = rmssd * (60000.0 / hr) / 1000.0
  coord se = shannon_entropy(bvp)

  regime = classify(rc_mean) {
    phase_locked  when rc_mean >= 0.947
    coherent      when rc_mean >= 0.930
    cascade       when rc_mean >= 0.900
    aperture      when rc_mean >= 0.850
    turbulent     otherwise
  }

  failure = detect {
    rigidity    when rc_mean > 0.95 and se < 0.5
    decoherence when rc_mean < 0.30
    none        otherwise
  }
}

-- Watch condition: stress onset
watch stress_onset {
  when  dhr_autonomic > 15.0
  and   rt_ratio      < 0.3
  and   rc_mean       < 0.85
  emit  "stress_onset"  confidence 0.8
}
`;

// ─── Keyword list for lightweight syntax highlighting ─────────────────────────

const KEYWORDS = new Set([
  'source', 'signal', 'rate', 'hz',
  'layer', 'from', 'invert', 'given', 'using', 'baseline', 'sqrt_compress', 'clamp', 'derive',
  'decompose', 'term', 'coord', 'regime', 'classify', 'when', 'otherwise',
  'failure', 'detect', 'none', 'pair', 'predicted', 'observed',
  'watch', 'and', 'or', 'not', 'emit', 'confidence', 'is', 'aligns',
  'model', 'hub', 'task', 'input', 'output',
  'explain', 'match', 'agree', 'diverge', 'trace', 'flag', 'annotate', 'with', 'first_match',
]);

function highlightLine(line: string): string {
  // Returns HTML with <span class="..."> wrappers. Escape first.
  const e = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Comment
  const ci = line.indexOf('--');
  if (ci >= 0) {
    return highlight(line.slice(0, ci)) + `<span class="bs-cm">${e(line.slice(ci))}</span>`;
  }
  return highlight(line);

  function highlight(s: string): string {
    return s.replace(/("(?:[^"\\]|\\.)*"|[\d.]+|[a-zA-Z_][a-zA-Z0-9_.]*)/g, (tok) => {
      if (tok.startsWith('"')) return `<span class="bs-str">${e(tok)}</span>`;
      if (/^\d/.test(tok)) return `<span class="bs-num">${e(tok)}</span>`;
      if (KEYWORDS.has(tok.toLowerCase())) return `<span class="bs-kw">${e(tok)}</span>`;
      return e(tok);
    });
  }
}

// ─── Sandbox class ────────────────────────────────────────────────────────────

export class BrutScriptSandbox {
  private root: HTMLElement;
  private textarea!: HTMLTextAreaElement;
  private highlight!: HTMLDivElement;
  private consoleEl!: HTMLDivElement;
  private statusBar!: HTMLDivElement;
  private glbCanvas!: HTMLCanvasElement;
  private glbReport!: HTMLDivElement;
  private glbSelectEl!: HTMLSelectElement;

  private bs: BrutScript | null = null;
  private tickInterval: ReturnType<typeof setInterval> | null = null;
  private beatClock: BeatClock | null = null;
  /** When true, the running session is agent-driven: exercise agents swap
   *  scripts at phrase boundaries. When false, the editor script runs as-is. */
  private agentMode = false;
  private activeExercise: ExerciseAgent | null = null;
  private effortSamples: number[] = [];
  private chartPanel: ReturnType<typeof mountChartPanel> | null = null;
  private glbViewer: GlbViewerHandle | null = null;
  private sessionStart = performance.now();
  private traceLines: string[] = [];
  private activeOutputTab = 'charts';
  private signalFeed: Map<string, number | string> = new Map();

  constructor(container: HTMLElement) {
    this.root = container;
    this.build();
  }

  // ── DOM construction ──────────────────────────────────────────────────────

  private build(): void {
    this.root.innerHTML = '';
    this.root.className = 'bs-sandbox';

    // Title bar
    const hfToken = (import.meta.env.VITE_HF_TOKEN as string | undefined) || undefined;
    const hfPill = hfToken
      ? `<span class="bs-hf-pill bs-hf-ok" title="HuggingFace token active">HF ●</span>`
      : `<span class="bs-hf-pill bs-hf-none" title="No VITE_HF_TOKEN — model blocks will fail">HF ○</span>`;

    const titleBar = div('bs-title-bar', `
      <div class="bs-title-left">
        <span class="bs-icon">⬡</span>
        <span class="bs-title-text">BrutScript — Cardiac Pipeline Sandbox</span>
        ${hfPill}
      </div>
      <div class="bs-title-right">
        <button class="bs-close-btn" title="Close sandbox">×</button>
      </div>`);
    this.root.appendChild(titleBar);
    titleBar.querySelector('.bs-close-btn')!.addEventListener('click', () => this.destroy());

    // Main layout: activity bar + editor + output
    const main = div('bs-main');
    this.root.appendChild(main);

    // Activity bar (icons)
    const actBar = div('bs-activity-bar');
    actBar.innerHTML = `
      <button class="bs-act" title="Explorer" data-pane="editor">≡</button>
      <button class="bs-act" title="Search">⌕</button>
      <button class="bs-act active" title="BrutScript">⬡</button>`;
    main.appendChild(actBar);

    // Editor pane
    const editorPane = div('bs-editor-pane');
    main.appendChild(editorPane);

    // Editor header (file tab)
    const editorHeader = div('bs-editor-header');
    editorHeader.innerHTML = `
      <div class="bs-file-tab active">
        <span class="bs-file-icon">◈</span>
        <span class="bs-file-name">pipeline.bs</span>
      </div>`;
    editorPane.appendChild(editorHeader);

    // Editor body: line numbers + textarea + highlight overlay
    const editorBody = div('bs-editor-body');
    editorPane.appendChild(editorBody);

    const lineNums = div('bs-line-nums');
    editorBody.appendChild(lineNums);

    const editorInner = div('bs-editor-inner');
    editorBody.appendChild(editorInner);

    this.highlight = div('bs-highlight-layer');
    this.highlight.setAttribute('aria-hidden', 'true');
    editorInner.appendChild(this.highlight);

    this.textarea = document.createElement('textarea');
    this.textarea.className = 'bs-editor-textarea';
    this.textarea.spellcheck = false;
    this.textarea.value = DEFAULT_SCRIPT;
    editorInner.appendChild(this.textarea);

    // Editor toolbar (below editor)
    const editorToolbar = div('bs-editor-toolbar');
    editorToolbar.innerHTML = `
      <div class="bs-toolbar-left">
        <span class="bs-lang-label">brutscript</span>
        <span class="bs-error-count" id="bs-err-count"></span>
      </div>
      <div class="bs-toolbar-right">
        <button class="bs-btn" id="bs-clear-btn">clear</button>
        <button class="bs-btn bs-btn-run" id="bs-run-btn">▶ run</button>
        <button class="bs-btn bs-btn-run" id="bs-train-btn" title="Beat-gated exercise agents: the body chooses the exercise">▶ train</button>
        <button class="bs-btn bs-btn-stop" id="bs-stop-btn" disabled>■ stop</button>
      </div>`;
    editorPane.appendChild(editorToolbar);

    // Output pane
    const outputPane = div('bs-output-pane');
    main.appendChild(outputPane);

    // Output header (tabs)
    const outputHeader = div('bs-output-header');
    outputHeader.innerHTML = `
      <div class="bs-output-tabs">
        <button class="bs-out-tab active" data-tab="charts">charts</button>
        <button class="bs-out-tab" data-tab="models">3d models</button>
        <button class="bs-out-tab" data-tab="console">console</button>
      </div>
      <div class="bs-output-actions">
        <button class="bs-mini-btn" id="bs-export-btn" title="Export trace as NDJSON">↓ export</button>
      </div>`;
    outputPane.appendChild(outputHeader);

    // Output body (tab panes)
    const outputBody = div('bs-output-body');
    outputPane.appendChild(outputBody);

    const chartsPane = div('bs-out-pane active', '');
    chartsPane.dataset.tab = 'charts';
    outputBody.appendChild(chartsPane);

    const modelsPane = div('bs-out-pane', '');
    modelsPane.dataset.tab = 'models';
    modelsPane.style.display = 'none';
    outputBody.appendChild(modelsPane);

    const consolePane = div('bs-out-pane', '');
    consolePane.dataset.tab = 'console';
    consolePane.style.display = 'none';
    outputBody.appendChild(consolePane);

    // Status bar (bottom)
    this.statusBar = div('bs-status-bar', '<span class="bs-status-text" id="bs-status-text">ready</span>');
    this.root.appendChild(this.statusBar);

    // ── Wire up charts panel ──────────────────────────────────────────────
    this.chartPanel = mountChartPanel(chartsPane);

    // ── Wire up console ───────────────────────────────────────────────────
    this.consoleEl = document.createElement('div');
    this.consoleEl.className = 'bs-console';
    consolePane.appendChild(this.consoleEl);

    // ── Wire up models pane ───────────────────────────────────────────────
    this.buildModelsPane(modelsPane);

    // ── Wire up editor events ─────────────────────────────────────────────
    this.textarea.addEventListener('input', () => this.updateEditor());
    this.textarea.addEventListener('scroll', () => {
      this.highlight.scrollTop = this.textarea.scrollTop;
      this.highlight.scrollLeft = this.textarea.scrollLeft;
      lineNums.scrollTop = this.textarea.scrollTop;
    });
    this.textarea.addEventListener('keydown', (ev) => {
      if (ev.key === 'Tab') {
        ev.preventDefault();
        const s = this.textarea.selectionStart;
        const e = this.textarea.selectionEnd;
        this.textarea.value = this.textarea.value.slice(0, s) + '  ' + this.textarea.value.slice(e);
        this.textarea.selectionStart = this.textarea.selectionEnd = s + 2;
        this.updateEditor();
      }
    });

    // Ctrl+Enter to run
    this.textarea.addEventListener('keydown', (ev) => {
      if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter') {
        ev.preventDefault();
        this.run();
      }
    });

    // ── Wire up output tabs ───────────────────────────────────────────────
    outputHeader.querySelectorAll('.bs-out-tab').forEach(btn => {
      (btn as HTMLElement).addEventListener('click', () => {
        const tab = (btn as HTMLElement).dataset.tab!;
        this.switchOutputTab(tab, outputHeader, outputBody);
      });
    });

    // ── Wire up buttons ───────────────────────────────────────────────────
    document.getElementById('bs-run-btn')?.addEventListener('click', () => this.run());
    document.getElementById('bs-train-btn')?.addEventListener('click', () => this.runAgents());
    document.getElementById('bs-stop-btn')?.addEventListener('click', () => this.stop());
    document.getElementById('bs-clear-btn')?.addEventListener('click', () => this.clearConsole());
    document.getElementById('bs-export-btn')?.addEventListener('click', () => this.exportTrace());

    // Initial highlight + line numbers
    this.updateEditor();
    this.updateLineNums(lineNums);
    this.textarea.addEventListener('input', () => this.updateLineNums(lineNums));
  }

  private buildModelsPane(pane: HTMLElement): void {
    pane.innerHTML = '';
    pane.style.cssText += 'display:flex;flex-direction:column;overflow:hidden;height:100%;';

    // Selector bar
    const selBar = div('bs-glb-selector-bar');
    this.glbSelectEl = document.createElement('select');
    this.glbSelectEl.className = 'bs-select';
    for (const entry of CARDIAC_GLB_CATALOGUE) {
      const opt = document.createElement('option');
      opt.value = entry.url;
      opt.textContent = entry.label;
      this.glbSelectEl.appendChild(opt);
    }
    const loadBtn = document.createElement('button');
    loadBtn.className = 'bs-btn';
    loadBtn.textContent = '⬡ load model';
    selBar.appendChild(this.glbSelectEl);
    selBar.appendChild(loadBtn);
    pane.appendChild(selBar);

    // Model description
    const descEl = div('bs-glb-desc');
    descEl.textContent = CARDIAC_GLB_CATALOGUE[0].description;
    pane.appendChild(descEl);
    this.glbSelectEl.addEventListener('change', () => {
      const entry = CARDIAC_GLB_CATALOGUE.find(e => e.url === this.glbSelectEl.value);
      if (entry) descEl.textContent = entry.description;
    });

    // Viewer + report split
    const viewerArea = div('bs-glb-viewer-area');
    pane.appendChild(viewerArea);

    this.glbCanvas = document.createElement('canvas');
    this.glbCanvas.className = 'bs-glb-canvas';
    viewerArea.appendChild(this.glbCanvas);

    this.glbReport = div('bs-glb-report');
    this.glbReport.textContent = 'Load a model to see its parse report.';
    viewerArea.appendChild(this.glbReport);

    loadBtn.addEventListener('click', () => this.loadGlbModel(this.glbSelectEl.value));
  }

  // ── GLB loading ────────────────────────────────────────────────────────────

  private async loadGlbModel(url: string): Promise<void> {
    this.setStatus(`parsing ${url.split('/').pop()}…`);
    this.glbReport.textContent = 'Parsing…';

    try {
      // Parse report
      const report: GlbParseReport = await parseGlb(url, this.sessionStart);
      renderParseReport(report, this.glbReport);

      // Destroy old viewer
      this.glbViewer?.destroy();
      this.glbViewer = null;

      // Mount new viewer
      this.glbViewer = await mountGlbViewer(this.glbCanvas, url);
      this.setStatus(`${report.meshes.length} meshes · ${report.totalVertices.toLocaleString()} verts · ${report.animationClips.length} clips`);
    } catch (err) {
      this.glbReport.innerHTML = `<span style="color:var(--hot)">Failed to load: ${err}</span>`;
      this.setStatus('glb load failed');
    }
  }

  // ── Run / stop ─────────────────────────────────────────────────────────────

  /**
   * Start a beat-gated training session driven by exercise agents. Unlike
   * `run()`, which executes the editor script as-is, this loads the first
   * exercise agent's own analysis script and lets the agents hand off to one
   * another at phrase boundaries as the body's evidence warrants. The athlete
   * does not pick the exercise — the beat-read body makes the case.
   */
  private runAgents(): void {
    this.stop();
    this.clearConsole();
    this.traceLines = [];
    this.sessionStart = performance.now();
    this.agentMode = true;
    this.effortSamples = [];

    const first = EXERCISE_ROSTER[FIRST_EXERCISE];
    this.loadExercise(first);
    if (!this.bs) return;   // compile error already reported

    const errCount = document.getElementById('bs-err-count');
    if (errCount) errCount.textContent = '';
    (document.getElementById('bs-run-btn') as HTMLButtonElement).disabled = true;
    (document.getElementById('bs-train-btn') as HTMLButtonElement).disabled = true;
    (document.getElementById('bs-stop-btn') as HTMLButtonElement).disabled = false;

    const initialBpm = asNum(this.signalFeed.get('bpm'), 128);
    this.beatClock = new BeatClock({
      bpm: initialBpm,
      beatsPerBar: 4,
      barsPerPhrase: 4,
      onBeat: (pos) => { void this.onBeatTick(pos); },
      onPhrase: (pos) => { if (this.agentMode) this.onPhraseBoundary(pos); },
    });
    this.beatClock.start();
    this.setStatus(`training — ${first.label} (beat-gated)`);
  }

  private run(): void {
    this.stop();
    this.clearConsole();
    this.traceLines = [];
    this.sessionStart = performance.now();
    this.agentMode = false;

    const source = this.textarea.value;
    const hfToken = (import.meta.env.VITE_HF_TOKEN as string | undefined) || undefined;
    this.bs = new BrutScript(source, {
      onTrace: (entry) => this.onTrace(entry),
      hfToken,
    }, this.sessionStart);

    const errCount = document.getElementById('bs-err-count');
    if (this.bs.errors.length > 0) {
      if (errCount) {
        errCount.textContent = `${this.bs.errors.length} error${this.bs.errors.length > 1 ? 's' : ''}`;
        errCount.style.color = 'var(--hot)';
      }
      for (const e of this.bs.errors) {
        this.logConsole(`[${e.phase}] line ${e.line}:${e.col} — ${e.message}`, 'error');
      }
      this.setStatus(`${this.bs.errors.length} compile error(s)`);
      this.bs = null;
      return;
    }

    if (errCount) { errCount.textContent = ''; }
    this.setStatus('running…');
    (document.getElementById('bs-run-btn') as HTMLButtonElement).disabled = true;
    (document.getElementById('bs-stop-btn') as HTMLButtonElement).disabled = false;

    // The beat is the clock. Instead of a fixed timer, the script is ticked on
    // the musical grid: each beat fires an evaluation, and beat position selects
    // the capture protocol (beat 1 → DC baseline, beat 3 → AC amplitude, phrase
    // boundary → full inversion). BPM, beat, bar, and phrase go onto the signal
    // bus, so `derive effort = hr / bpm` and beat-gated watches are scriptable.
    const initialBpm = asNum(this.signalFeed.get('bpm'), 128);
    this.beatClock = new BeatClock({
      bpm: initialBpm,
      beatsPerBar: 4,
      barsPerPhrase: 4,
      onBeat: (pos) => { void this.onBeatTick(pos); },
      onPhrase: (pos) => { if (this.agentMode) this.onPhraseBoundary(pos); },
    });
    this.beatClock.start();
  }

  /**
   * Phrase boundary: the body makes its case. Read the accumulated
   * physiological evidence off the live script, evaluate two-factor relevance
   * against the active exercise agent's purpose, and — only if the evidence
   * both advances the purpose and is physiologically coherent — swap the live
   * script to the successor exercise. The athlete never asks; the body argues.
   */
  private onPhraseBoundary(_pos: BeatPosition): void {
    if (!this.bs || !this.activeExercise) return;

    const samples = this.effortSamples;
    const meanEffort = samples.length ? samples.reduce((a, b) => a + b, 0) / samples.length : 0;
    const trend = samples.length >= 2 ? samples[samples.length - 1] - samples[0] : 0;
    this.effortSamples = [];

    const evidence: PhraseEvidence = {
      effort:        meanEffort,
      dhr_autonomic: asNum(this.bs.read('dhr_autonomic'), 0),
      dhr_metabolic: asNum(this.bs.read('dhr_metabolic'), 0),
      dhr_hypoxic:   asNum(this.bs.read('dhr_hypoxic'), 0),
      vasodilation:  asNum(this.bs.read('vasodilation'), 1.0),
      t_skin:        asNum(this.bs.read('t_skin'), 33.0),
      rc_mean:       asNum(this.bs.read('rc_mean'), 0.85),
      effort_trend:  trend,
    };

    const verdict = evaluateTransition(this.activeExercise, evidence);
    this.logConsole(`[phrase] ${this.activeExercise.label}: ${verdict.reason}`, verdict.relevant ? 'event' : 'trace');

    if (verdict.relevant && verdict.next) {
      const next = EXERCISE_ROSTER[verdict.next];
      if (next) this.loadExercise(next);
    }
  }

  /** Swap the live script to an exercise agent's own analysis script. */
  private loadExercise(agent: ExerciseAgent): void {
    this.activeExercise = agent;
    this.effortSamples = [];
    this.textarea.value = agent.script;
    this.updateEditor();
    const hfToken = (import.meta.env.VITE_HF_TOKEN as string | undefined) || undefined;
    this.bs = new BrutScript(agent.script, { onTrace: (e) => this.onTrace(e), hfToken }, this.sessionStart);
    if (this.bs.errors.length) {
      for (const e of this.bs.errors) this.logConsole(`[${e.phase}] line ${e.line}:${e.col} — ${e.message}`, 'error');
      this.setStatus(`exercise ${agent.label}: compile error`);
      this.bs = null;
      return;
    }
    this.setStatus(`exercise: ${agent.label}`);
    this.logConsole(`▶ exercise agent active: ${agent.label} — effort target ${agent.purpose.effortTarget}`, 'event');
  }

  /**
   * One beat of the clock: fold beat position + effort index onto the bus,
   * evaluate the script, then drive the GLB agent from BPM.
   */
  private async onBeatTick(pos: BeatPosition): Promise<void> {
    if (!this.bs || !this.beatClock) return;

    // If an external BPM estimate has arrived, re-lock the clock to it.
    const busBpm = this.signalFeed.get('bpm');
    if (typeof busBpm === 'number' && busBpm > 0) this.beatClock.setBpm(busBpm);

    // Push observatory signals, then overlay the beat-clock signals.
    for (const [name, value] of this.signalFeed) this.bs.push(name, value);

    const beat = this.beatClock.signals();
    const hr = asNum(this.signalFeed.get('hr'), asNum(this.bs.read('hr'), 0));
    const effort = effortIndex(hr, beat.bpm);
    const protocol = protocolFor(pos);
    if (this.agentMode) this.effortSamples.push(effort);

    this.bs.push('bpm', beat.bpm);
    this.bs.push('beat', beat.beat);
    this.bs.push('bar_pos', beat.bar_pos);
    this.bs.push('phrase', beat.phrase);
    this.bs.push('beat_index', beat.beat_index);
    this.bs.push('effort', effort);
    this.bs.push('effort_regime', effortRegime(effort));
    this.bs.push('protocol', protocol);

    await this.bs.tick();
    const entries = this.bs.drain();
    if (entries.length > 0) this.chartPanel?.ingestTrace(entries);

    // Drive the GLB agent. Tempo comes from the music (BPM), not HR: the model
    // moves with the track, while HR/BPM reads out as the effort index.
    if (this.glbViewer) {
      this.glbViewer.updateCardiacState({
        HR:  asNum(this.bs.read('hr'),  70),
        Ees: asNum(this.bs.read('ees'), 2.0),
        Ea:  asNum(this.bs.read('ea'),  1.3),
        EDV: asNum(this.bs.read('edv'), 120),
        ESV: asNum(this.bs.read('esv'), 50),
        EF:  asNum(this.bs.read('ef'),  0.58),
        Rc:  asNum(this.bs.read('rc_mean'), 0.85),
      });
      this.glbViewer.setTempoHz(beat.bpm / 60);
    }
  }

  private stop(): void {
    if (this.tickInterval) { clearInterval(this.tickInterval); this.tickInterval = null; }
    if (this.beatClock) { this.beatClock.stop(); this.beatClock = null; }
    this.bs = null;
    this.agentMode = false;
    this.activeExercise = null;
    this.effortSamples = [];
    (document.getElementById('bs-run-btn') as HTMLButtonElement | null)?.removeAttribute('disabled');
    (document.getElementById('bs-train-btn') as HTMLButtonElement | null)?.removeAttribute('disabled');
    (document.getElementById('bs-stop-btn') as HTMLButtonElement | null)?.setAttribute('disabled', '');
    this.setStatus('stopped');
  }

  // ── External signal injection ─────────────────────────────────────────────

  /** Called by the observatory to feed live signals into the running script. */
  public feedSignal(name: string, value: number | string): void {
    this.signalFeed.set(name, value);
  }

  // ── Trace handler ──────────────────────────────────────────────────────────

  private onTrace(entry: TraceEntry): void {
    const line = JSON.stringify(entry);
    this.traceLines.push(line);
    if (this.activeOutputTab === 'console') {
      this.appendConsoleLine(line, entry.step === 'diverge' ? 'warn' : entry.emit ? 'event' : 'trace');
    }
  }

  // ── Console ────────────────────────────────────────────────────────────────

  private logConsole(msg: string, cls = 'trace'): void {
    this.appendConsoleLine(msg, cls);
  }

  private appendConsoleLine(text: string, cls: string): void {
    const line = document.createElement('div');
    line.className = `bs-con-line bs-con-${cls}`;
    line.textContent = text;
    this.consoleEl.appendChild(line);
    // Auto-scroll only if near bottom
    const { scrollTop, scrollHeight, clientHeight } = this.consoleEl;
    if (scrollHeight - scrollTop - clientHeight < 60) {
      this.consoleEl.scrollTop = scrollHeight;
    }
    // Cap at 2000 lines
    while (this.consoleEl.children.length > 2000) {
      this.consoleEl.firstChild?.remove();
    }
  }

  private clearConsole(): void {
    this.consoleEl.innerHTML = '';
    this.traceLines = [];
  }

  // ── Editor sync ────────────────────────────────────────────────────────────

  private updateEditor(): void {
    const lines = this.textarea.value.split('\n');
    this.highlight.innerHTML = lines.map(l => `<div class="bs-hl-line">${highlightLine(l) || ' '}</div>`).join('');
    this.highlight.scrollTop = this.textarea.scrollTop;
    this.highlight.scrollLeft = this.textarea.scrollLeft;
  }

  private updateLineNums(el: HTMLElement): void {
    const count = this.textarea.value.split('\n').length;
    el.innerHTML = Array.from({ length: count }, (_, i) => `<div class="bs-lnum">${i + 1}</div>`).join('');
  }

  // ── Output tab switching ──────────────────────────────────────────────────

  private switchOutputTab(tab: string, header: HTMLElement, body: HTMLElement): void {
    this.activeOutputTab = tab;
    header.querySelectorAll('.bs-out-tab').forEach(b => {
      (b as HTMLElement).classList.toggle('active', (b as HTMLElement).dataset.tab === tab);
    });
    body.querySelectorAll('.bs-out-pane').forEach(p => {
      (p as HTMLElement).style.display = (p as HTMLElement).dataset.tab === tab ? 'flex' : 'none';
    });
  }

  // ── Status bar ─────────────────────────────────────────────────────────────

  private setStatus(msg: string): void {
    const el = document.getElementById('bs-status-text');
    if (el) el.textContent = msg;
  }

  // ── Export ─────────────────────────────────────────────────────────────────

  private exportTrace(): void {
    const blob = new Blob([this.traceLines.join('\n')], { type: 'application/x-ndjson' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `brutscript-trace-${Date.now()}.ndjson`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  destroy(): void {
    this.stop();
    this.chartPanel?.destroy();
    this.glbViewer?.destroy();
    this.root.innerHTML = '';
    this.root.className = '';
    this.root.style.display = 'none';
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function div(className: string, innerHTML = ''): HTMLDivElement {
  const el = document.createElement('div');
  el.className = className;
  if (innerHTML) el.innerHTML = innerHTML;
  return el;
}

function asNum(v: number | string | undefined, fallback: number): number {
  return typeof v === 'number' && isFinite(v) ? v : fallback;
}

// ─── Mount helper ─────────────────────────────────────────────────────────────

let sandboxInstance: BrutScriptSandbox | null = null;

export function openSandbox(): BrutScriptSandbox {
  let el = document.getElementById('brutscript-sandbox');
  if (!el) {
    el = document.createElement('div');
    el.id = 'brutscript-sandbox';
    document.body.appendChild(el);
  }
  el.style.display = '';
  if (sandboxInstance) return sandboxInstance;
  sandboxInstance = new BrutScriptSandbox(el);
  return sandboxInstance;
}

export function closeSandbox(): void {
  sandboxInstance?.destroy();
  sandboxInstance = null;
}
