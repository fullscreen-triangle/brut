"""
Partition Regime Validation — Local Oura Ring Dataset
======================================================
Tests the core prediction of the cardio-neural-metabolic integration paper:
that sleep stages (A/L/D/R) correspond to distinct cardiac partition regimes
with Kuramoto order parameter R_c in predicted ranges.

Predicted mapping:
  A (Awake/arousal during night) -> Coherent regime   R_c in [0.80, 0.95]
  L (Light sleep N1/N2)          -> Cascade regime    R_c in [0.50, 0.80]
  D (Deep sleep / SWS)           -> Phase-locked      R_c > 0.95
  R (REM sleep)                  -> Aperture-dom.     R_c in [0.30, 0.55]
                                    (cardiac stays active; neural goes turbulent)

R_c estimator from HRV:
  CV = RMSSD * HR / 60000   (dimensionless coefficient of variation of RR interval)
  R_c = exp(-2*pi^2 * CV^2) (circular dispersion formula from Kuramoto theory)
"""

import json
import numpy as np
import math
from collections import defaultdict

# ── Load data ────────────────────────────────────────────────────────────────
with open('c:/Users/kunda/Documents/health/brut/demo/public/sleep_ppg_records.json') as f:
    sleep_records = json.load(f)

print(f"Loaded {len(sleep_records)} sleep nights")
print()

# ── Regime boundaries (from integration paper) ────────────────────────────────
REGIME_BOUNDARIES = {
    'phase_locked':     (0.95, 1.00),
    'coherent':         (0.80, 0.95),
    'cascade':          (0.50, 0.80),
    'aperture':         (0.30, 0.50),
    'turbulent':        (0.00, 0.30),
}

STAGE_PREDICTED_REGIME = {
    'A': 'coherent',
    'L': 'cascade',
    'D': 'phase_locked',
    'R': 'aperture',   # cardiac; neural would be turbulent but we measure cardiac
}

# ── R_c estimator ─────────────────────────────────────────────────────────────
def compute_Rc(rmssd_ms, hr_bpm):
    """
    Kuramoto R_c from RMSSD and HR.
    CV = RMSSD / RR_mean = RMSSD * HR / 60000
    R_c = exp(-2*pi^2 * CV^2)  [circular dispersion approximation]
    """
    if hr_bpm <= 0 or rmssd_ms <= 0:
        return None
    cv = (rmssd_ms * hr_bpm) / 60000.0
    rc = math.exp(-2 * math.pi**2 * cv**2)
    return rc

def regime_name(rc):
    for name, (lo, hi) in REGIME_BOUNDARIES.items():
        if lo <= rc < hi:
            return name
    return 'turbulent'

# ── Per-epoch extraction ───────────────────────────────────────────────────────
# Collect (stage, R_c) tuples, filtering invalid epochs
stage_rc = defaultdict(list)   # stage -> list of R_c values
night_rc = {}                  # period_id -> {stage: [R_c]}

for rec in sleep_records:
    pid = rec['period_id']
    hyp = rec.get('hypnogram_5min', '')
    hr  = rec.get('hr_5min', [])
    rmssd = rec.get('rmssd_5min', [])

    if not hyp or not hr or not rmssd:
        continue

    n = min(len(hyp), len(hr), len(rmssd))
    night_rc[pid] = defaultdict(list)

    for i in range(n):
        stage = hyp[i]
        h = hr[i]
        r = rmssd[i]
        if h > 0 and r > 0 and stage in ('A', 'L', 'D', 'R'):
            rc = compute_Rc(r, h)
            if rc is not None:
                stage_rc[stage].append(rc)
                night_rc[pid][stage].append(rc)

# ── Stage-level statistics ─────────────────────────────────────────────────────
print("=" * 70)
print("STAGE-LEVEL R_c STATISTICS (across all 86 nights)")
print("=" * 70)
print(f"{'Stage':<6} {'N epochs':>10} {'Mean R_c':>10} {'Median':>10} {'Std':>8} "
      f"{'5th pct':>8} {'95th pct':>8} {'Predicted':>12} {'Match?':>8}")
print("-" * 70)

stage_stats = {}
for stage in ['A', 'L', 'D', 'R']:
    vals = np.array(stage_rc[stage])
    if len(vals) == 0:
        continue
    mean_rc = np.mean(vals)
    med_rc  = np.median(vals)
    std_rc  = np.std(vals)
    p5      = np.percentile(vals, 5)
    p95     = np.percentile(vals, 95)
    pred    = STAGE_PREDICTED_REGIME[stage]
    pred_lo, pred_hi = REGIME_BOUNDARIES[pred]
    match   = pred_lo <= mean_rc < pred_hi
    stage_stats[stage] = {
        'vals': vals, 'mean': mean_rc, 'median': med_rc,
        'std': std_rc, 'p5': p5, 'p95': p95,
        'predicted': pred, 'predicted_lo': pred_lo, 'predicted_hi': pred_hi,
        'match': match
    }
    print(f"{stage:<6} {len(vals):>10,} {mean_rc:>10.4f} {med_rc:>10.4f} "
          f"{std_rc:>8.4f} {p5:>8.4f} {p95:>8.4f} {pred:>12s} {'YES' if match else 'NO':>8}")

print()

# ── Regime classification accuracy ────────────────────────────────────────────
print("=" * 70)
print("REGIME CLASSIFICATION ACCURACY (per-epoch)")
print("=" * 70)
print(f"{'Stage':<6} {'Total':>8} {'In predicted regime':>20} {'Accuracy':>10}")
print("-" * 70)

overall_correct = 0
overall_total   = 0
for stage in ['A', 'L', 'D', 'R']:
    if stage not in stage_stats:
        continue
    s = stage_stats[stage]
    lo, hi = s['predicted_lo'], s['predicted_hi']
    vals   = s['vals']
    correct = np.sum((vals >= lo) & (vals < hi))
    total   = len(vals)
    overall_correct += correct
    overall_total   += total
    print(f"{stage:<6} {total:>8,} {correct:>20,} {100*correct/total:>9.1f}%")

print("-" * 70)
print(f"{'TOTAL':<6} {overall_total:>8,} {overall_correct:>20,} "
      f"{100*overall_correct/overall_total:>9.1f}%")
print()

# ── Inter-stage separability ───────────────────────────────────────────────────
print("=" * 70)
print("INTER-STAGE R_c SEPARABILITY (Cohen's d between adjacent stages)")
print("=" * 70)
pairs = [('D','A'), ('A','L'), ('L','R')]
stage_label = {'D':'Deep','A':'Awake','L':'Light','R':'REM'}
for s1, s2 in pairs:
    if s1 not in stage_stats or s2 not in stage_stats:
        continue
    m1, m2 = stage_stats[s1]['mean'], stage_stats[s2]['mean']
    sd1, sd2 = stage_stats[s1]['std'], stage_stats[s2]['std']
    pooled_sd = math.sqrt((sd1**2 + sd2**2) / 2)
    cohens_d  = abs(m1 - m2) / pooled_sd if pooled_sd > 0 else 0
    print(f"  {stage_label[s1]} vs {stage_label[s2]}: d = {cohens_d:.3f}  "
          f"(dRc = {m1-m2:+.4f})")

print()

# ── Regime sweep test: does each night follow D > A > L > R ordering? ─────────
print("=" * 70)
print("REGIME SWEEP TEST: Is R_c ordered D > A > L > R within each night?")
print("=" * 70)
n_sweep_correct = 0
n_sweep_total   = 0
violations = []
for pid, stage_dict in night_rc.items():
    # Only test nights that have all four stages with data
    if not all(s in stage_dict and len(stage_dict[s]) > 0
               for s in ['D', 'A', 'L', 'R']):
        continue
    rd = np.mean(stage_dict['D'])
    ra = np.mean(stage_dict['A'])
    rl = np.mean(stage_dict['L'])
    rr = np.mean(stage_dict['R'])
    n_sweep_total += 1
    if rd > ra > rl > rr:
        n_sweep_correct += 1
    else:
        violations.append((pid, rd, ra, rl, rr))

print(f"Nights with all 4 stages: {n_sweep_total}")
print(f"Nights with correct D>A>L>R ordering: {n_sweep_correct} "
      f"({100*n_sweep_correct/max(n_sweep_total,1):.1f}%)")
if violations:
    print(f"Violations ({len(violations)} nights):")
    for pid, rd, ra, rl, rr in violations[:5]:
        print(f"  Period {pid}: D={rd:.3f} A={ra:.3f} L={rl:.3f} R={rr:.3f}")
print()

# ── Cardiac coherence vs sleep score ──────────────────────────────────────────
print("=" * 70)
print("CARDIAC COHERENCE vs SLEEP QUALITY SCORE")
print("=" * 70)
rc_scores = []
for rec in sleep_records:
    pid   = rec['period_id']
    score = rec.get('score', None)
    if score is None or pid not in night_rc:
        continue
    all_rc_night = []
    for stage_vals in night_rc[pid].values():
        all_rc_night.extend(stage_vals)
    if all_rc_night:
        mean_rc_night = np.mean(all_rc_night)
        rc_scores.append((mean_rc_night, score))

if len(rc_scores) > 5:
    rc_arr    = np.array([x[0] for x in rc_scores])
    score_arr = np.array([x[1] for x in rc_scores])
    corr      = np.corrcoef(rc_arr, score_arr)[0, 1]
    print(f"Pearson r(mean R_c, sleep score) = {corr:.4f}  "
          f"(N={len(rc_scores)} nights)")
    print(f"Mean R_c range: [{rc_arr.min():.4f}, {rc_arr.max():.4f}]")
    print(f"Sleep score range: [{score_arr.min():.0f}, {score_arr.max():.0f}]")
print()

# ── Deep sleep R_c vs deep sleep quality score ────────────────────────────────
print("Deep sleep R_c vs score_deep:")
deep_rc_scores = []
for rec in sleep_records:
    pid       = rec['period_id']
    score_d   = rec.get('score_deep', None)
    if score_d is None or pid not in night_rc:
        continue
    deep_vals = night_rc[pid].get('D', [])
    if deep_vals:
        deep_rc_scores.append((np.mean(deep_vals), score_d))

if len(deep_rc_scores) > 5:
    rc_d  = np.array([x[0] for x in deep_rc_scores])
    sc_d  = np.array([x[1] for x in deep_rc_scores])
    corr_d = np.corrcoef(rc_d, sc_d)[0, 1]
    print(f"  Pearson r(deep R_c, score_deep) = {corr_d:.4f}  (N={len(deep_rc_scores)})")
print()

# ── S-entropy coordinates per stage ──────────────────────────────────────────
print("=" * 70)
print("S-ENTROPY COORDINATE ESTIMATES PER STAGE")
print("=" * 70)
print("(S_k = knowledge depth proxy, S_e = entropy utilization proxy)")
print()
for stage in ['D', 'A', 'L', 'R']:
    if stage not in stage_stats:
        continue
    s    = stage_stats[stage]
    rc   = s['mean']
    std  = s['std']
    # S_k proxy: normalized Rc (depth of information integration)
    sk   = rc
    # S_e proxy: 1 - normalized variance (entropy utilization)
    se   = 1.0 - (std / (rc + 1e-9))
    se   = max(0.0, min(1.0, se))
    # S_t: sleep stage temporal position (0=sleep onset, 1=morning)
    st_map = {'D': 0.6, 'L': 0.4, 'A': 0.2, 'R': 0.8}
    st   = st_map[stage]
    print(f"  Stage {stage}  S_k={sk:.4f}  S_t={st:.2f}  S_e={se:.4f}  "
          f"regime={s['predicted']}")

print()
print("=" * 70)
print("SUMMARY: KEY VALIDATION FINDINGS")
print("=" * 70)
for stage in ['D', 'A', 'L', 'R']:
    if stage not in stage_stats:
        continue
    s = stage_stats[stage]
    status = "CONFIRMED" if s['match'] else "PARTIAL / INVESTIGATE"
    print(f"  {stage}: mean R_c={s['mean']:.4f}, predicted={s['predicted']}, "
          f"status={status}")
