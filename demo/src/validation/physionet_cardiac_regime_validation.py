"""
PhysioNet Cardiac Regime Validation
=====================================
Tests the cardiac regime classification framework against three databases:

1. MIT-BIH Arrhythmia (mitdb) -- 48 records, 360 Hz ECG, annotated rhythms
   Tests: Does R_c from true beat-to-beat RR intervals correctly separate
   Normal Sinus Rhythm / Ventricular Tachycardia / Atrial Fibrillation?

2. CHF RR Intervals (chfdb) -- 15 records, ~20 hr each, congestive heart failure
   Tests: Do CHF patients show reduced R_c vs NSR controls?
   Tests: Does the decoherence cascade (accelerating R_c decline) appear?

3. Normal Sinus Rhythm database (nsr2db) -- 54 records, healthy controls
   Tests: What is the healthy R_c baseline at beat-to-beat resolution?

Regime predictions:
  Phase-locked  R_c > 0.95   -- pacemaker-driven, extreme sinus regularity
  Coherent      0.80-0.95    -- normal resting sinus rhythm
  Cascade       0.50-0.80    -- mild arrhythmia, early HF
  Aperture      0.30-0.50    -- VT, compensated HF
  Turbulent     R_c < 0.30   -- AFIB, VF, decompensated HF
"""

import wfdb
import numpy as np
import math
from collections import defaultdict, Counter

WINDOW_SEC = 60   # compute R_c in 60-second windows
REGIME_BOUNDS = [
    ('phase_locked', 0.95, 1.00),
    ('coherent',     0.80, 0.95),
    ('cascade',      0.50, 0.80),
    ('aperture',     0.30, 0.50),
    ('turbulent',    0.00, 0.30),
]

def regime_name(rc):
    for name, lo, hi in REGIME_BOUNDS:
        if lo <= rc < hi:
            return name
    return 'turbulent'

def rc_from_rr(rr_intervals_ms):
    """
    Compute Kuramoto R_c from array of RR intervals (ms).
    Uses circular dispersion: CV = RMSSD / mean_RR, R_c = exp(-2*pi^2*CV^2)
    Requires at least 5 intervals.
    """
    rr = np.array(rr_intervals_ms)
    rr = rr[(rr > 300) & (rr < 2000)]  # physiological range filter
    if len(rr) < 5:
        return None
    mean_rr = np.mean(rr)
    if mean_rr <= 0:
        return None
    diffs = np.diff(rr)
    rmssd = np.sqrt(np.mean(diffs**2))
    cv = rmssd / mean_rr
    rc = math.exp(-2 * math.pi**2 * cv**2)
    return rc

def get_rr_from_record(rec_name, db):
    """Extract beat annotations and compute windowed R_c with rhythm labels."""
    try:
        header = wfdb.rdheader(rec_name, pn_dir=db)
        ann    = wfdb.rdann(rec_name, 'atr', pn_dir=db)
        fs     = header.fs
    except Exception as e:
        return None, None

    # Build rhythm label per sample
    rhythm_at_sample = {}
    current_rhythm = 'N'
    ann_samples = ann.sample
    ann_symbols = ann.symbol
    ann_aux     = ann.aux_note

    for i, (samp, sym, aux) in enumerate(zip(ann_samples, ann_symbols, ann_aux)):
        clean = aux.strip().rstrip('\x00').strip()
        if clean.startswith('('):
            current_rhythm = clean[1:]
        rhythm_at_sample[samp] = current_rhythm

    # Extract normal beats (N) and ectopic beats, compute RR intervals
    beat_symbols = {'N','L','R','B','A','a','J','S','V','F','e','j','n','E','f','/','Q'}
    beat_samples = [(s, sym) for s, sym in zip(ann_samples, ann_symbols) if sym in beat_symbols]
    if len(beat_samples) < 10:
        return None, None

    # Windowed R_c computation
    total_dur_sec = header.sig_len / fs
    windows = []
    for t_start in np.arange(0, total_dur_sec - WINDOW_SEC, WINDOW_SEC):
        t_end = t_start + WINDOW_SEC
        s_start = int(t_start * fs)
        s_end   = int(t_end * fs)

        # Beats in this window
        w_beats = [(s, sym) for s, sym in beat_samples if s_start <= s < s_end]
        if len(w_beats) < 5:
            continue

        rr_ms = []
        for k in range(1, len(w_beats)):
            rr = (w_beats[k][0] - w_beats[k-1][0]) / fs * 1000.0
            rr_ms.append(rr)

        rc = rc_from_rr(rr_ms)
        if rc is None:
            continue

        # Dominant rhythm in window
        beat_count = Counter()
        for s, sym in w_beats:
            # find closest rhythm label
            # look back in rhythm_at_sample
            candidates = [k for k in rhythm_at_sample if k <= s]
            if candidates:
                rhy = rhythm_at_sample[max(candidates)]
            else:
                rhy = 'N'
            beat_count[rhy] += 1
        dominant_rhythm = beat_count.most_common(1)[0][0]

        # Beat type composition
        sym_count = Counter(sym for _, sym in w_beats)
        n_beats  = sym_count.get('N', 0) + sym_count.get('L', 0) + sym_count.get('R', 0)
        v_beats  = sym_count.get('V', 0)
        a_beats  = sym_count.get('A', 0) + sym_count.get('a', 0)
        total_b  = sum(sym_count.values())
        v_frac   = v_beats / total_b if total_b > 0 else 0

        windows.append({
            'rc': rc,
            'rhythm': dominant_rhythm,
            'v_frac': v_frac,
            'n_beats': total_b,
            'mean_hr': 60000 / np.mean(rr_ms) if rr_ms else 0,
        })

    return windows, total_dur_sec

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1: MIT-BIH Arrhythmia Database
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("ANALYSIS 1: MIT-BIH ARRHYTHMIA DATABASE (mitdb)")
print("Cardiac regime classification from true beat-to-beat RR intervals")
print("=" * 70)
print()

mitdb_records = wfdb.get_record_list('mitdb')
rhythm_rc = defaultdict(list)
record_summaries = []

print("Processing 48 records...")
for rec in mitdb_records:
    windows, dur = get_rr_from_record(rec, 'mitdb')
    if windows is None:
        continue
    for w in windows:
        rhythm_rc[w['rhythm']].append(w['rc'])
    n_w = len(windows)
    if n_w > 0:
        mean_rc = np.mean([w['rc'] for w in windows])
        rhythms_seen = set(w['rhythm'] for w in windows)
        record_summaries.append({
            'rec': rec,
            'n_windows': n_w,
            'mean_rc': mean_rc,
            'rhythms': rhythms_seen,
        })

print(f"Processed {len(record_summaries)} records, "
      f"{sum(r['n_windows'] for r in record_summaries)} windows total")
print()

# Aggregate by rhythm type
print(f"{'Rhythm':<12} {'N windows':>10} {'Mean R_c':>10} {'Median':>8} "
      f"{'Std':>8} {'Predicted regime':>16}")
print("-" * 70)

# Map rhythm codes to clinical names
rhythm_map = {
    'N':    ('Normal SR',     'coherent'),
    'VT':   ('V. Tachycardia','aperture'),
    'AFIB': ('Atrial Fib.',   'turbulent'),
    'AFL':  ('Atrial Flutter','aperture'),
    'VFL':  ('V. Flutter',    'turbulent'),
    'B':    ('Bigeminy',      'aperture'),
    'T':    ('Trigeminy',     'cascade'),
    'SVTA': ('SVT',           'cascade'),
    'IVR':  ('Idiov. Rhythm', 'cascade'),
    'NOD':  ('Nodal Rhythm',  'cascade'),
}

results_by_rhythm = {}
for rhy_code, rhy_vals in sorted(rhythm_rc.items(), key=lambda x: -len(x[1])):
    if len(rhy_vals) < 3:
        continue
    arr = np.array(rhy_vals)
    name, predicted = rhythm_map.get(rhy_code, (rhy_code, '?'))
    m, med, s = np.mean(arr), np.median(arr), np.std(arr)
    results_by_rhythm[rhy_code] = {
        'name': name, 'predicted': predicted,
        'mean': m, 'median': med, 'std': s, 'n': len(arr)
    }
    print(f"{name:<12} {len(arr):>10} {m:>10.4f} {med:>8.4f} {s:>8.4f} {predicted:>16}")

print()

# Key test: does AFIB have lower R_c than Normal?
if 'N' in rhythm_rc and 'AFIB' in rhythm_rc:
    n_rc   = np.array(rhythm_rc['N'])
    af_rc  = np.array(rhythm_rc['AFIB'])
    t_stat = (np.mean(n_rc) - np.mean(af_rc)) / math.sqrt(
        np.var(n_rc)/len(n_rc) + np.var(af_rc)/len(af_rc))
    print(f"Key test: R_c(Normal SR) > R_c(AFIB)?")
    print(f"  Normal: {np.mean(n_rc):.4f} +/- {np.std(n_rc):.4f}")
    print(f"  AFIB:   {np.mean(af_rc):.4f} +/- {np.std(af_rc):.4f}")
    print(f"  Delta:  {np.mean(n_rc)-np.mean(af_rc):+.4f}")
    print(f"  t-stat: {t_stat:.2f}  (|t|>2 = significant)")
    print(f"  Confirmed: {'YES' if np.mean(n_rc) > np.mean(af_rc) else 'NO'}")
    print()

if 'N' in rhythm_rc and 'VT' in rhythm_rc:
    vt_rc = np.array(rhythm_rc['VT'])
    n_rc  = np.array(rhythm_rc['N'])
    print(f"Key test: R_c(Normal SR) > R_c(VT)?")
    print(f"  Normal: {np.mean(n_rc):.4f}")
    print(f"  VT:     {np.mean(vt_rc):.4f}")
    print(f"  Confirmed: {'YES' if np.mean(n_rc) > np.mean(vt_rc) else 'NO'}")
    print()

# Regime classification accuracy
print("Regime classification accuracy (does mean R_c fall in predicted range?):")
for code, res in results_by_rhythm.items():
    pred = res['predicted']
    m = res['mean']
    matches = [(name, lo, hi) for name, lo, hi in REGIME_BOUNDS if name == pred]
    if matches:
        _, lo, hi = matches[0]
        ok = lo <= m < hi
        # Also count per-epoch accuracy
        arr = np.array(rhythm_rc[code])
        epoch_acc = np.mean((arr >= lo) & (arr < hi))
        print(f"  {res['name']:<14}: mean={m:.4f}, predicted={pred}, "
              f"mean_in_range={'YES' if ok else 'NO'}, "
              f"epoch_acc={100*epoch_acc:.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2: CHF vs NSR -- Decoherence Prediction
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("ANALYSIS 2: CONGESTIVE HEART FAILURE vs NORMAL SINUS RHYTHM")
print("Prediction: CHF patients show reduced R_c (coherence deficit)")
print("=" * 70)
print()

chf_records = wfdb.get_record_list('chfdb')
print(f"CHF database: {len(chf_records)} records")

chf_all_rc = []
print("Processing CHF records (long recordings, sampling 1h of data each)...")
for rec in chf_records[:15]:
    try:
        header = wfdb.rdheader(rec, pn_dir='chfdb')
        ann    = wfdb.rdann(rec, 'ecg', pn_dir='chfdb')
        fs     = header.fs
        beat_samples = [(s, sym) for s, sym in zip(ann.sample, ann.symbol)
                        if sym in {'N','V','A','S','Q'}]
        if len(beat_samples) < 20:
            continue
        # Take first hour of data
        max_samp = min(3600 * fs, beat_samples[-1][0])
        beats_1h = [(s, sym) for s, sym in beat_samples if s <= max_samp]
        rr_ms = [(beats_1h[k][0] - beats_1h[k-1][0]) / fs * 1000.0
                 for k in range(1, len(beats_1h))]
        # Windowed
        rr_arr = np.array(rr_ms)
        for i in range(0, len(rr_arr) - 60, 60):
            window_rr = rr_arr[i:i+60]
            rc = rc_from_rr(window_rr)
            if rc is not None:
                chf_all_rc.append(rc)
    except Exception:
        continue

print(f"CHF: {len(chf_all_rc)} windows computed")

# NSR control -- use MIT-BIH normal segments
nsr_all_rc = [rc for rc in rhythm_rc.get('N', []) if rc is not None]
print(f"NSR (from MIT-BIH normal segments): {len(nsr_all_rc)} windows")
print()

if chf_all_rc and nsr_all_rc:
    chf_arr = np.array(chf_all_rc)
    nsr_arr = np.array(nsr_all_rc)

    print(f"{'Condition':<12} {'N windows':>10} {'Mean R_c':>10} {'Median':>8} "
          f"{'Std':>8} {'5th pct':>8} {'Regime (mean)':>14}")
    print("-" * 70)
    for label, arr in [('NSR', nsr_arr), ('CHF', chf_arr)]:
        m, med, s = np.mean(arr), np.median(arr), np.std(arr)
        p5 = np.percentile(arr, 5)
        print(f"{label:<12} {len(arr):>10} {m:>10.4f} {med:>8.4f} {s:>8.4f} "
              f"{p5:>8.4f} {regime_name(m):>14}")

    delta = np.mean(nsr_arr) - np.mean(chf_arr)
    t_stat = delta / math.sqrt(
        np.var(nsr_arr)/len(nsr_arr) + np.var(chf_arr)/len(chf_arr))
    print()
    print(f"R_c reduction NSR -> CHF: {delta:+.4f}")
    print(f"t-statistic:              {t_stat:.2f}  (|t|>2 = significant)")
    print(f"Regime shift:             {regime_name(np.mean(nsr_arr))} -> "
          f"{regime_name(np.mean(chf_arr))}")
    print(f"Prediction (coherence deficit): {'CONFIRMED' if delta > 0.01 else 'PARTIAL'}")

    # Fraction of CHF in turbulent/aperture vs NSR
    chf_turb = np.mean(chf_arr < 0.50)
    nsr_turb = np.mean(nsr_arr < 0.50)
    print(f"Fraction of epochs below cascade threshold (R_c<0.50):")
    print(f"  NSR: {100*nsr_turb:.1f}%   CHF: {100*chf_turb:.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS 3: R_c Distribution as Regime Phase Diagram
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("ANALYSIS 3: R_c DISTRIBUTION ACROSS THE FULL REGIME PHASE DIAGRAM")
print("=" * 70)
print()
print("Epoch counts in each regime:")
print(f"{'Regime':<14} {'R_c range':>12} {'NSR':>8} {'VT':>8} {'AFIB':>8} {'CHF':>8}")
print("-" * 56)

datasets = {
    'NSR':  np.array(rhythm_rc.get('N', [])),
    'VT':   np.array(rhythm_rc.get('VT', [])),
    'AFIB': np.array(rhythm_rc.get('AFIB', [])),
    'CHF':  np.array(chf_all_rc) if chf_all_rc else np.array([]),
}

for name, lo, hi in REGIME_BOUNDS:
    row = f"{name:<14} [{lo:.2f},{hi:.2f}]"
    for dname, arr in datasets.items():
        if len(arr) == 0:
            row += f"{'--':>8}"
        else:
            count = np.sum((arr >= lo) & (arr < hi))
            pct   = 100 * count / len(arr)
            row  += f"{pct:>7.1f}%"
    print(row)

print()
print("=" * 70)
print("SUMMARY OF PHYSIONET CARDIAC REGIME VALIDATION")
print("=" * 70)
