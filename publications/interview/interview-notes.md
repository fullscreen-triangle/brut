# Interview Notes — Senior ML Developer + Founding Role
**Date:** 2026-06-20  
**Compensation:** Up to €120k/year + founding equity / stock  
**Company stack:** Flutter

---

## 0. The Core Angle — Scientific Legitimacy as Founding Contribution

This is the thread that runs through every answer. State it clearly once, early, and let everything else be evidence of it.

> "What I bring to a founding role is scientific legitimacy — the ability to verify that what the AI is actually doing corresponds to real physiology. That is not a nice-to-have in this space. It is the difference between a product that works in trials and one that works in deployment."

### Why this angle is strategically correct

The health-AI market has a specific problem that is not technical: **AI systems in biosensing look correct until they don't, and there is no framework for knowing which state you are in.** A model trained on a population dataset will overfit to the artefact distribution of the training hardware, the skin tone distribution of the training cohort, the activity profile of whoever wore the device. It will pass internal validation. It will fail in deployment. And nobody knows why, because there is no ground truth to compare against — only another model.

Kundai's position is: *I can build the ground truth.* BRUT's first-principles derivations — Beer-Lambert optical inversion, PCHR decomposition, Kuramoto coherence as regime discriminant — are not models. They are closed-form physical relationships. They do not overfit. They do not require labelled data. They provide the verifiable reference against which an AI system's output can be audited.

This is not a research luxury. It is a **commercial necessity** for any company trying to sell into:
- Clinical settings (where FDA/CE Mark requires traceable evidence)
- Enterprise/B2B wellness (where HR buyers demand audit trails)
- Insurance (where actuarial risk requires interpretable predictions)

### The B2B angle — removing dependence on consumer users

Consumer health apps have brutal retention economics: 90-day dropout, seasonal usage, difficulty collecting labelled ground truth at scale. B2B deployment changes all of that:

- **Corporate wellness / occupational health**: continuous monitoring of shift workers, pilots, surgeons — populations with high compliance, professional context, and willingness to pay per seat
- **Clinical trial support**: pharmaceutical companies need objective physiological endpoints for trials. A sensor pipeline with a verifiable physical basis is exactly what a CRO needs to replace subjective self-report
- **Insurance underwriting**: actuarial tables for biometric health data require interpretability. A black-box model cannot be used. A physics-grounded one can

The founding contribution is unlocking these markets by providing the scientific grounding that makes the AI output *auditable*. That is a different value proposition than "better accuracy on the benchmark" — and it is one that no pure ML team can replicate quickly.

---

## 1. The Position — What They Are Actually Hiring

This is not a standard engineering interview. "Founding role" signals they want someone who:

- Brings something the existing team cannot hire for on a job board
- Can own a technical domain with enough depth that it becomes a moat
- Has a view on where the business should go, not just how to build what was already decided

The scientific legitimacy angle is precisely what a Flutter/ML engineering team cannot generate internally. They can build fast. They cannot verify what they built against physical reality. That gap is what you fill.

**Frame every answer around:** *what I built, why the standard approach was insufficient, and what new doors the correct approach opens.*

---

## 2. Opening — How to Position Yourself

Lead with the frame, not the facts:

> "My work at TU Munich is about scientific verification of AI in physiological sensing — not building another model, but building the reference framework that tells you whether any model is working for the right reasons. BRUT is the implementation of that."

Then make the business case in one sentence:

> "The reason that matters commercially is that the markets where health AI is most valuable — clinical, B2B, insurance — all require that you can explain and verify what the system is doing. A purely empirical pipeline cannot do that. A physics-grounded one can."

---

## 3. BRUT — What to Say and What Numbers to Use

Do not narrate the pipeline. Lead with the claim, then give one precise mechanism to support it.

### The claim
> "87–93% accurate physiological interpretation from a phone camera — no dedicated hardware, no contact sensors. The accuracy comes from the theory, not from more data."

### If they ask how
**S-entropy framework:** The system computes a (Sk, St, Se) coordinate in [0,1]³ derived from Kuramoto coherence (Rc) measured across a 64×64 rPPG patch grid on WebGPU. Regime is classified by Rc threshold — phase-locked (>0.947), coherent (>0.930), cascade (>0.900), aperture (>0.850), turbulent. This is not a trained classifier. It is a geometric discriminant derived from coupled oscillator theory.

**Beer-Lambert inversion:** Four-step algebraic recovery of melanin, haemoglobin concentration, SpO2, and vasodilation from RGB face channels — no ML model, pure optical physics.

**PCHR decomposition:** HR_obs = HR_int + ΔHR_met + ΔHR_O2 + ΔHR_auto — partitions the heart rate signal into its causal contributors using known physiological coupling constants (α_T = 0.08/°C, β_O2 = 0.15).

### On the Rust/TypeScript architecture
- **Rust:** numerical core — oscillatory decomposition (rustfft, O(n log n), 10³ samples/sec), compression (10⁴ samples/sec), S-entropy navigation (ChaCha8 PRNG, constrained random walk, converges in 127–200 iterations, <0.1s/record)
- **TypeScript/WebGPU:** real-time rPPG pipeline in the browser — 64×64 patch grid, per-pixel Rc, full cardiac EOS state machine, zero server round-trips
- The split is principled: Rust for batch analysis and offline derivation, TypeScript for zero-latency streaming in the privacy model (no data leaves the device)

---

## 4. The Flutter Question — Technical Reality

They use Flutter. You need to address the bridge directly and confidently.

### What is actually true

**Rust → Flutter via flutter_rust_bridge (v2.x):**  
`flutter_rust_bridge_codegen` auto-generates Dart FFI bindings directly from Rust source — no manual binding code. Supports iOS, Android, macOS, Windows, Linux. The BRUT numerical core (oscillatory decomposition, S-entropy navigation, Beer-Lambert inversion) is pure Rust with no platform dependencies. It compiles to a native `.so`/`.dylib`/`.a` and is callable from Dart with essentially zero overhead for the compute-heavy path.

**TypeScript → Flutter:**  
No production transpiler exists. The correct framing is not "compile TypeScript to Flutter" — it is "the BrutScript DSL compiler and runtime logic is a 2,000-line self-contained TypeScript module with a clean public API. Porting the runtime to Dart is a 1–2 week translation task, not a rewrite. The grammar, AST, and semantics are already formally specified in the published BrutScript paper; the Dart implementation is a derivation of that spec, not an original design problem."

**On Flutter Web:**  
Rust compiles to WASM via `wasm-pack`. Flutter Web calls WASM through `dart:js` / `package:web` interop. The rPPG WebGPU pipeline would need a different path on Flutter Web (Flutter uses Impeller, not browser WebGPU) — this is the one genuine technical gap to acknowledge honestly.

### How to say this in the interview

> "Rust bridges to Flutter cleanly via flutter_rust_bridge — it auto-generates Dart FFI bindings from the Rust source, so the numerical core ports with no manual binding work. The BrutScript DSL logic is a week of Dart translation from a formal spec we already have. The rPPG pipeline is the real porting question — Flutter's rendering model is Impeller, not browser WebGPU, so the GPU patch grid would need to move to a compute shader via Flutter's GPU path. That is an engineering task I can scope and lead."

This answer does three things: demonstrates real technical knowledge, shows honesty about the gap, and immediately reframes it as a solvable problem you own.

---

## 5. Founder-Level Questions — What They Will Actually Ask

### "What is the biggest mistake you see in this space?"

> "Treating physiological sensing as a pure data problem. The entire field is racing to collect more labelled data to train better models. But the failure mode for consumer biosensors is not statistical — it is physical. A model trained on one device generation, one lighting distribution, one skin tone distribution will pass internal validation and fail in deployment. You cannot label your way out of shot noise and melanin variation. The right move is to model the physics and use the data to constrain the model, not replace it. The consequence is a system that generalises because it is grounded in something true, not because it memorised something large."

### "What do you bring that the current team doesn't have?"

This is the core answer. Say it directly:

> "Scientific legitimacy — the ability to verify that what the AI is doing corresponds to real physiology, not just to the training distribution. That unlocks B2B markets your consumer pipeline cannot reach. Clinical buyers, pharmaceutical CROs, enterprise HR — they all need an audit trail. A physics-grounded pipeline can provide one. A purely empirical model cannot, regardless of its accuracy number."

### "What would you build if you joined?"

Ask what their current pipeline does first. Then:

> "The first thing I would build is a verification layer — a set of first-principles checks that run alongside the AI model and flag when its output is physically implausible. Not as a replacement for the model, but as the thing that tells you when to trust it and when to escalate. That layer is also what makes the system sellable into regulated markets. It transforms 'AI health app' into 'auditable physiological monitoring system' — a completely different procurement conversation."

### "What is your relationship to the theoretical work versus the engineering?"

> "I do not separate them. BRUT's architecture came directly from theoretical questions — what information is preserved under each transformation, which mathematical structure is the right discriminant for regime classification. The Kuramoto coherence measure was chosen because coupled oscillator theory predicted it would work, and then validated. When theory and engineering are that tightly coupled you get systems that generalise because they are right, not because they overfit."

### "How do you think about the B2B opportunity versus consumer?"

> "Consumer is where you get adoption signal fast but the retention economics are brutal and the ground truth problem is hard. B2B changes both — you get professional compliance, high willingness to pay per seat, and contexts where the physiological interpretation is directly actionable: occupational health, clinical trials, insurance underwriting. The scientific grounding is what makes those conversations possible. A black-box model has no story for a clinical buyer. A physics-verifiable pipeline does."

### "How do you think about building a team around this?"

> "The scientific legitimacy function has to be owned, not consulted. That means at least one person on the founding team who can hold the mathematics and the systems engineering simultaneously — so that when a numerical result looks wrong, you can trace it from the AI output back to the physical model and find where the divergence is. That cross-layer debugging capability is what protects you from shipping confidence numbers that are technically high but physiologically meaningless."

---

## 6. Compensation and Equity

You have significant leverage. The founding role + equity package signals they have not found someone they consider a peer. Do not undersell.

- **Salary:** €120k is their ceiling. Do not negotiate down by being eager. Accept at the top or counter at market for founding engineers in European deep-tech (€130–150k is defensible at this level).
- **Equity:** Get the vesting schedule, cliff, and dilution protection in writing before accepting. Founding stock without anti-dilution protection in a seed-stage company is worth modeling carefully.
- **Title:** "Founding Engineer" or "Head of ML" matters for what you can do next. Get it explicit.

---

## 7. Questions to Ask Them

These signal strategic thinking, not just job-seeking:

1. "What is your current accuracy on the sensing pipeline, and what is the specific failure mode you most want to fix?"
2. "How do you think about the tradeoff between on-device inference and cloud — is that a product decision, a privacy decision, or a technical one?"
3. "What does the path to clinical validation look like for you, and do you see that as a regulatory challenge, a data challenge, or a scientific one?"
4. "What does the founding equity vest on — time, milestones, or both?"
5. "Who is the person I would be most technically dependent on, and are they staying?"

---

## 8. Things Not to Say

- Do not call BRUT "a wearable app" or "a health app." It is a physiological inference framework.
- Do not say "I can learn Flutter quickly" — say you already know the bridge path and have used Dart's FFI model conceptually through flutter_rust_bridge.
- Do not enumerate the pipeline stages as if reading a spec sheet. Tell the story of one stage as evidence of the whole.
- Do not volunteer that you are nervous about leaving academic research. Frame it as: "The interesting problems are now in deployment — the physics is right, the question is building the system that proves it at scale."

---

## 9. One-Paragraph Summary (say this if asked to introduce yourself)

> "I am a researcher at TU Munich working on scientific verification of AI in physiological sensing. I built BRUT — a framework that derives cardiac state, skin optics, and autonomic balance from first principles using a phone camera, achieving 87–93% accuracy without dedicated hardware. The distinguishing feature is that every inference step is grounded in closed-form physics — Beer-Lambert optical inversion, Kuramoto coherence, PCHR decomposition — so the output is auditable, not just accurate. That grounding is what I bring to a founding role: the ability to verify that what the AI is doing corresponds to real physiology, which is the prerequisite for selling into clinical, enterprise, and regulated markets that a consumer app cannot reach."

---

## 10. The Single Sentence (if you get 10 seconds)

> "I build the physics layer that makes AI in health trustworthy enough to sell to people who cannot afford to be wrong."

---

## 11. The Publication Portfolio — Five Papers Already Written

This is a founding-level asset that almost no startup in this space can match. Most raise their Series A without a single peer-reviewed paper. The corpus below represents a complete, internally consistent scientific framework — not five isolated papers, but five facets of the same partition-theoretic derivation of physiological sensing.

**The frame to use:**

> "We are not building a product and then trying to find the science. The science came first. We have five near-complete publications that provide the theoretical basis for everything the AI system does — and journals have already contacted us asking to publish. That is an unusual position to be in."

---

### Paper 1 — The Foundation
**"First-Principles Derivation of Cardiovascular-Pulmonary System Architecture from Categorical Fluid Dynamics and Transport Partition Theory"**
- **Directory:** `publications/cardio-vascular-derivation/`
- **Status:** ~90% complete
- **What it proves:** The cardiovascular system's architecture — 300M alveoli, 70m² surface area, 7μm capillary diameter, haemoglobin cooperativity n=2.8, Murray's cubic law, blood viscosity — derived entirely from partition theory and ideal gas mechanics. No empirical fitting, no free parameters.
- **Why it matters commercially:** Establishes that BRUT's physiological constants are not calibrated from data — they are derived from physics. This is the answer to any clinical buyer who asks "how do you know your model isn't overfit?"
- **Target journals:** *Journal of Theoretical Biology*, *PLOS Computational Biology*, *Frontiers in Physiology*

---

### Paper 2 — The Sensing Method
**"Layered Optical Inversion for Photoplethysmography: Recovering Vasodilation, Skin Temperature, and a Metabolic-Autonomic Decomposition of Heart Rate from RGB Camera Pixels"**
- **Directory:** `publications/layered-optical-ppg/`
- **Status:** ~95% complete — figures present (6 panels), validation JSONs present (6 experiments), captions file present
- **What it proves:** Beer-Lambert 4-layer inversion recovering melanin, [Hb], SpO2, vasodilation from RGB face channels algebraically. Then maps vasodilation to skin temperature via Q₁₀ law. Then decomposes HR_obs = HR_int + ΔHR_met + ΔHR_O2 + ΔHR_auto using known physiological coupling constants.
- **Why it matters commercially:** This is the paper that makes the camera-based sensing pipeline scientifically defensible. It directly addresses the "why should we trust a phone camera for health data" question — by showing each inference step is a closed-form inversion, not a learned mapping.
- **Target journals:** *Biomedical Optics Express*, *IEEE Transactions on Biomedical Engineering*, *npj Digital Medicine*
- **Closest to submission** — prioritise this one first.

---

### Paper 3 — The Cardiac Model
**"Equations of State for the Cardiac System: Partition-Theoretic Derivation of Ventricular Mechanics Across Physiological Regimes"**
- **Directory:** `publications/cardiac-equations-of-state/`
- **Status:** ~95% complete — figures present (4 panels), full PDF compiling
- **What it proves:** Frank-Starling law, Hill force-velocity relationship, end-systolic/diastolic PV relationships, ventricular-arterial coupling, Windkessel dynamics — all derived from partition theory. Covers 7 physiological regimes: rest, submaximal/maximal exercise, compensated/decompensated HF, hypertension, hypovolemia, distributive shock.
- **Why it matters commercially:** Gives the cardiac state machine in the BRUT observatory scientific backing. When the product reports Ees, Ea, EF — this paper is the citation that explains where those numbers come from and why they're trustworthy across different patient populations.
- **Target journals:** *Journal of Physiology*, *American Journal of Physiology: Heart and Circulatory Physiology*, *Frontiers in Cardiovascular Medicine*

---

### Paper 4 — The Unified Framework
**"On the Consequences of Cardiac Mechanics on Bounded Phase Space Dynamics in Coupled Metabolic Energetics and Neural Oscillatory Dynamics"**
- **Directory:** `publications/cardio-neural-integration/`
- **Status:** ~95% complete — figures present (5 panels), full PDF compiling
- **What it proves:** Unifies cardiac mechanics, neural oscillation, and metabolic energetics through a Bounded Phase Space Law. Derives the cardiac system as master oscillator in a 13-scale biological hierarchy. Defines S-entropy coordinates (Sk, St, Se), Kuramoto coherence as regime discriminant, and disease as coherence deficit with a derived therapeutic framework.
- **Why it matters commercially:** This is the theoretical paper that justifies the S-entropy health score — the core product metric. It connects the camera-based measurements to a rigorous definition of physiological health. Also the most ambitious claim: disease as coherence deficit is a testable, falsifiable scientific position, not a marketing statement.
- **Target journals:** *PLOS Biology*, *eLife*, *Nature Communications* (high-ambition target — this is the paper that could generate press coverage)

---

### Paper 5 — The Sensing System
**"Sensor Disambiguation Framework"** *(working title)*
- **Directory:** `publications/sensor-disambiguation/`
- **Status:** ~85% complete — figures present (5 panels: PCHR decomposition, S-entropy coordinates, temperature correction, cross-scale coherence, activity coupling)
- **What it proves:** How to disentangle multiple physiological sources when all signals are entangled — camera, keyboard, mouse. The PCHR decomposition, cross-scale coherence index (CSCI), and regime classification applied to real multi-sensor data. The sensor-agnostic framework that lets BRUT use whatever data is available.
- **Why it matters commercially:** This is the paper that supports the B2B sensor-agnostic pitch. The framework works on wearables, cameras, keyboards — any continuous biosignal source. Enterprise customers with different device ecosystems can all use the same verification layer.
- **Target journals:** *IEEE Sensors Journal*, *Sensors (MDPI)*, *Scientific Reports*

---

---

### Paper 6 — The B2B Tool
**"BrutScript: A Domain-Specific Language for Auditable Physiological AI Pipelines"**
- **Directory:** `publications/cardiac-scripting/`
- **Status:** ~20% (skeleton) — but the implementation is complete and the paper in `cardiac-scripting-vocubulary.tex` only needs to be finished against the working codebase
- **What it proves:** A formal DSL with a BNF grammar, denotational semantics, and a TypeScript compiler that produces structured NDJSON audit traces. Every AI inference step is traceable to a named block, a physical model, and a line number. The compiler is already running in the BRUT Observatory.
- **Why it is immediately a B2B product:** This is the key insight. BrutScript is not just a paper — it is the compliance infrastructure for any organisation deploying AI in health sensing. An enterprise buyer does not just want accurate predictions. They want a system where a compliance officer can open a log file and see exactly what the AI did, why, and what the physical basis was. BrutScript is the tool that produces that log file. No other sensing product in the space has this.
- **B2B deployment modes:**
  - License BrutScript as a pipeline auditing SDK to health AI teams who already have models but need explainability for enterprise/regulatory sales
  - Offer BrutScript-as-a-service: customers write pipeline scripts, results are signed and timestamped — audit-ready by design
  - White-label the verification layer to clinical trial CROs who need objective physiological endpoints with traceable methodology
- **Target journals:** *npj Digital Medicine*, *Journal of Biomedical Informatics*, *PLOS Digital Health*
- **This paper should be finished and submitted before the company launches** — it establishes the DSL as prior art and ties the scientific framework to a deployable tool

---

### Submission Priority Order

| Priority | Paper | Effort to submission | Strategic reason |
|----------|-------|---------------------|-----------------|
| 1 | Layered Optical PPG (#2) | 1–2 weeks polish | Most concrete, most defensible, 6 validation experiments already done |
| 2 | Cardiac EOS (#3) | 1–2 weeks polish | Fully compiled PDF, figures done, supports the core product metric |
| 3 | BrutScript DSL (#6) | 2–3 weeks (finish skeleton) | Establishes B2B tool as prior art; most immediately productisable |
| 4 | Cardiovascular Derivation (#1) | 2–3 weeks (section completion) | Foundation paper — establishes the framework everything else references |
| 5 | Sensor Disambiguation (#5) | 3–4 weeks | B2B enabler, supports sensor-agnostic enterprise pitch |
| 6 | Cardio-Neural Integration (#4) | 2–3 weeks | Highest ambition, highest impact, highest review bar — submit last |

### In the interview — how to say this

> "We have five near-complete scientific publications grounded in the same unified theoretical framework. Journals have already approached us asking to publish. Most startups raise a Series A without a single paper. We have the scientific legitimacy infrastructure built before the product has even launched. That changes the regulatory conversation, the clinical sales conversation, and the credibility conversation with enterprise buyers — all at once."

If they ask what the papers cover:

> "The stack goes from first principles to product. One paper derives the cardiovascular architecture from partition theory — that establishes why our constants are right. One derives the optical inversion that justifies camera-based sensing. One derives the cardiac equations of state that underpin our health score. One unifies cardiac, neural, and metabolic dynamics to define what a healthy physiological regime actually is, mathematically. And one describes the sensor disambiguation framework that makes the whole system hardware-agnostic. Together they are a complete scientific basis for the product."
