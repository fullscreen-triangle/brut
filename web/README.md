# BRUT Observatory

A camera-driven, WebGPU-shader-based instantiation of the BRUT framework for
in-browser physiological observation. The shader pipeline is the measurement,
not a visualization of one (per `publications/sources/observation-computation.tex`).

## v1 scope (current)

Cardiac channel only, end-to-end:

- **Input**: front-facing camera (`getUserMedia`)
- **ROI**: MediaPipe FaceLandmarker → forehead + bilateral cheek bbox
- **GPU pass A** (`sample-roi.wgsl`): downsample face ROI to a 64×64 patch
  grid, append per-patch green-channel mean to a circular history buffer
  (256 frames ≈ 8.5 s @ 30 fps)
- **GPU pass B** (`coherence-field.wgsl`): per-patch detrended autocorrelation
  in the cardiac band (0.7–3 Hz), output `R_c = exp(-σ_φ²/2)` per pixel as a
  64×64 `rgba16float` field (R_c, BVP amplitude, period, SNR)
- **GPU pass C** (`heatmap.wgsl`): composite the per-pixel R_c field over the
  video as a viridis heatmap inside the face bbox
- **CPU readback**: spatial mean BVP fed into HR / RMSSD / R_c / regime;
  spatial std reported as the novel "spatial coherence spread" channel

All five `sensor-disambiguation` framework outputs surfaced live:
HR, RMSSD, **R_c**, **S_k / S_t / S_e**, regime classification (turbulent /
aperture / cascade / coherent / phase-locked), failure-mode flag (rigidity vs
decoherence per `cardio-neural-integration` Theorem 11).

No backend. No persistence. The session leaves no trace beyond the page load
(matching the O(1) memory claim of `observation-computation.tex`).

## Requirements

- A WebGPU-capable browser (Chrome ≥ 113, Edge ≥ 113 on Windows/macOS;
  Chrome on Linux behind `--enable-unsafe-webgpu`; Safari Tech Preview)
- A camera and permission to use it
- Node ≥ 20 for development

## Run

```bash
cd web
npm install
npm run dev
```

Then open the URL printed by Vite (default `http://127.0.0.1:5173`) and
click **start**. Allow camera access.

## Build

```bash
npm run build      # writes dist/
npm run preview    # serves dist/ for local sanity check
npm run typecheck  # tsc --noEmit
```

## What you should see

1. Mirrored video of yourself.
2. A blue bbox around your face plus three green sub-rectangles (forehead,
   left cheek, right cheek).
3. After ~8 seconds of buffer fill, a translucent green-yellow heatmap
   appears over your face — the per-pixel R_c field.
4. The right panel shows HR, RMSSD, R_c, regime, S_k/S_t/S_e, SNR, and the
   spatial R_c spread (the new observation channel).

If the heatmap stays dark / R_c is always 0:

- Check lighting (rPPG needs reasonably even diffuse light on the face)
- Stay still — head motion artefacts dominate at >1 Hz
- Check the FPS line in the footer log; below ~15 fps the cardiac band gets
  aliased

## Files

```
web/
├── index.html
├── src/
│   ├── main.ts                 entry; camera/GPU lifecycle + RAF loop
│   ├── style.css
│   ├── vite-env.d.ts
│   ├── camera/
│   │   ├── stream.ts           getUserMedia
│   │   └── landmarks.ts        MediaPipe FaceLandmarker + ROI extraction
│   ├── gpu/
│   │   ├── device.ts           WebGPU adapter/device init
│   │   ├── rppg.ts             three-pass pipeline + readback
│   │   └── shaders/
│   │       ├── sample-roi.wgsl
│   │       ├── coherence-field.wgsl
│   │       └── heatmap.wgsl
│   ├── physio/
│   │   ├── bvp.ts              CPU-side BVP buffer + HRV stack
│   │   └── regimes.ts          5-regime classifier + failure mode
│   ├── ui/
│   │   ├── panel.ts            right-rail readout
│   │   └── overlay.ts          2D bbox overlay
│   └── util/
│       └── log.ts
```

## Where this is going

Deferred to subsequent versions, in roughly this order:

1. **Postural channel** — pose landmarks → CoM proxy → rambling/trembling
   spectral split (per `rambling-trembling-sensor.tex`)
2. **Partition field substrate** — the 5-pass GPU pipeline from
   `flux-phenomena.tex` running on the camera as the spectral oscillator
   source, with ray-marched observation as the output
3. **Four-condition charge basis sweep** — guided 8-min session yielding
   Q_th / Q_mo / Q_pe / Q_ba per `orthogonal-charge-quantification.tex`
4. **Closure-phase mode** — pure ray-marched partition field with no UI;
   measure spontaneous HRV biofeedback as a test of the closed-circuit
   prediction in `neuro-muscular-derivation.tex`

The order is "easiest to validate against existing literature first, novel
prediction last" — flip if you want to lead with the most novel claim.

## Notes

- `R_c` is computed from a Kuramoto-style estimator on the autocorrelation
  peak of the BVP at the dominant cardiac lag. This differs from the RR-CV
  estimator in `cardio-neural-integration.tex` because we don't yet have
  beat-level events from rPPG; the spectral version is bounded in the same
  [0, 1] range with similar physiological interpretation.
- The face landmark indices used for the cheek/forehead ROIs are
  hand-picked from the MediaPipe canonical 478-point model; tweak in
  `camera/landmarks.ts` if you want different sub-regions.
