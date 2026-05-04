// Per-pixel cardiac coherence field.
//
// For each (px, py) in the PATCH x PATCH grid, read the green-channel time
// series across BUF_FRAMES, detrend, then estimate the dominant cardiac period
// via normalized autocorrelation peak in the cardiac lag band.
// From the resulting coefficient of variation across successive cycles we
// compute a per-pixel R_c proxy:
//   R_c = exp(-2 * pi^2 * CV^2)
// which mirrors the Kuramoto circular-dispersion estimator used in the
// cardio-neural-integration paper.
//
// Output texture (rgba16float):
//   r = R_c                      (in [0, 1])
//   g = mean BVP amplitude       (relative units; 0 if signal too weak)
//   b = dominant period seconds  (lag / sample_rate)
//   a = SNR proxy                (peak / median autocorr in band)

override PATCH: u32 = 64u;
override BUF_FRAMES: u32 = 256u;

struct Uniforms {
  sample_rate: f32,    // frames per second (camera fps)
  frame_idx: u32,      // most recent slot written
  filled: u32,         // number of frames actually populated (clamps to BUF_FRAMES)
  _pad0: u32,
  // Cardiac search band (Hz):
  hr_min: f32,         // e.g. 0.7 Hz (~42 bpm)
  hr_max: f32,         // e.g. 3.0 Hz (~180 bpm)
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> history: array<f32>;
@group(0) @binding(2) var coherence: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265358979;

fn idx_for(slot: u32, px: u32, py: u32) -> u32 {
  return slot * PATCH * PATCH + py * PATCH + px;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= PATCH || gid.y >= PATCH) { return; }
  let px = gid.x;
  let py = gid.y;

  let n = min(u.filled, BUF_FRAMES);
  if (n < 32u) {
    textureStore(coherence, vec2<i32>(i32(px), i32(py)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
    return;
  }

  // Walk the ring buffer chronologically. Oldest slot = (frame_idx + 1) mod BUF_FRAMES
  // when filled == BUF_FRAMES; otherwise frames 0..filled-1 are valid.
  let oldest: u32 = select(0u, (u.frame_idx + 1u) % BUF_FRAMES, u.filled >= BUF_FRAMES);

  // Pass 1: mean and detrend slope (least-squares linear fit to remove DC + drift).
  var sum_x: f32 = 0.0;
  var sum_y: f32 = 0.0;
  var sum_xy: f32 = 0.0;
  var sum_xx: f32 = 0.0;
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    let slot = (oldest + i) % BUF_FRAMES;
    let v = history[idx_for(slot, px, py)];
    let x = f32(i);
    sum_x = sum_x + x;
    sum_y = sum_y + v;
    sum_xy = sum_xy + x * v;
    sum_xx = sum_xx + x * x;
  }
  let nf = f32(n);
  let denom = nf * sum_xx - sum_x * sum_x;
  let slope = select(0.0, (nf * sum_xy - sum_x * sum_y) / denom, denom > 1e-6);
  let intercept = (sum_y - slope * sum_x) / nf;

  // Pass 2: detrended series statistics and accumulator for autocorrelation.
  var var_acc: f32 = 0.0;
  var amp_acc: f32 = 0.0;
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    let slot = (oldest + i) % BUF_FRAMES;
    let raw = history[idx_for(slot, px, py)];
    let d = raw - (intercept + slope * f32(i));
    var_acc = var_acc + d * d;
    amp_acc = amp_acc + abs(d);
  }
  let variance = var_acc / nf;
  if (variance < 1e-8) {
    textureStore(coherence, vec2<i32>(i32(px), i32(py)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
    return;
  }

  // Pass 3: normalized autocorrelation across the cardiac lag band.
  let lag_min: u32 = max(2u, u32(floor(u.sample_rate / u.hr_max)));
  let lag_max: u32 = min(n / 2u, u32(ceil(u.sample_rate / u.hr_min)));
  if (lag_max <= lag_min) {
    textureStore(coherence, vec2<i32>(i32(px), i32(py)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
    return;
  }

  var best_lag: u32 = lag_min;
  var best_r: f32 = -2.0;
  var sum_band: f32 = 0.0;
  var count_band: f32 = 0.0;

  for (var lag: u32 = lag_min; lag <= lag_max; lag = lag + 1u) {
    var ac: f32 = 0.0;
    let m = n - lag;
    for (var i: u32 = 0u; i < m; i = i + 1u) {
      let s0 = (oldest + i) % BUF_FRAMES;
      let s1 = (oldest + i + lag) % BUF_FRAMES;
      let d0 = history[idx_for(s0, px, py)] - (intercept + slope * f32(i));
      let d1 = history[idx_for(s1, px, py)] - (intercept + slope * f32(i + lag));
      ac = ac + d0 * d1;
    }
    let r = (ac / f32(m)) / variance;
    sum_band = sum_band + r;
    count_band = count_band + 1.0;
    if (r > best_r) {
      best_r = r;
      best_lag = lag;
    }
  }

  let mean_band = sum_band / max(1.0, count_band);
  let snr = best_r - mean_band;
  let period_s = f32(best_lag) / u.sample_rate;

  // Convert autocorrelation peak strength into a Kuramoto-style coherence.
  // The peak r_norm is bounded in [-1, 1]; clamp to [0, 1) and treat (1 - r_norm)
  // as a phase-variance proxy. This mirrors R = exp(-sigma_phi^2 / 2) with
  // sigma_phi^2 ≈ -2 ln(r_peak) for a coherent oscillator, capped to keep
  // R_c in [0, 1].
  let r_peak = clamp(best_r, 0.0, 0.999);
  let sigma2 = max(0.0, -2.0 * log(max(r_peak, 1e-3)));
  let rc = exp(-0.5 * sigma2);

  let amp = amp_acc / nf;

  textureStore(
    coherence,
    vec2<i32>(i32(px), i32(py)),
    vec4<f32>(rc, amp, period_s, snr)
  );
}
