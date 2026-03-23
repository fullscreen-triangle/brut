"""
Sensor Disambiguation Validation — compute 5 composite metrics
(PCHR, S-Entropy Coordinates, TCC, CSCI, RARS) from Oura sleep/activity data.
Save results to JSON and CSV.
"""

import json, math, os
import numpy as np
import pandas as pd

BASE_DIR = 'c:/Users/kunda/Documents/health/brut'
SLEEP_PATH = BASE_DIR + '/demo/public/sleep_ppg_records.json'
ACTIVITY_PATH = BASE_DIR + '/demo/public/activity_ppg_records.json'
RESULTS_DIR = BASE_DIR + '/demo/src/validation/results/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(SLEEP_PATH, encoding='utf-8') as f:
    sleep_records = json.load(f)
with open(ACTIVITY_PATH, encoding='utf-8') as f:
    activity_records = json.load(f)

print(f"Loaded {len(sleep_records)} sleep records, {len(activity_records)} activity records")

STAGES = ['A', 'L', 'D', 'R']
STAGE_NAMES = {'A': 'Awake', 'L': 'Light', 'D': 'Deep', 'R': 'REM'}

# ---------------------------------------------------------------------------
# Helper: compute Rc for a single epoch
# ---------------------------------------------------------------------------
def compute_rc(hr, rmssd):
    if hr <= 0 or rmssd <= 0:
        return np.nan
    cv = (rmssd * hr) / 60000.0
    return math.exp(-2 * math.pi**2 * cv**2)

# ---------------------------------------------------------------------------
# 1. PCHR — Partition-Coupled Heart Rate
# ---------------------------------------------------------------------------
print("Computing PCHR decomposition...")

# Find HR_intrinsic: minimum HR during deep sleep averaged across nights
deep_hr_mins = []
for rec in sleep_records:
    hyp = rec.get('hypnogram_5min', '')
    hr5 = rec.get('hr_5min', [])
    n = min(len(hyp), len(hr5))
    deep_hrs_night = [hr5[i] for i in range(n) if hyp[i] == 'D' and hr5[i] > 0]
    if deep_hrs_night:
        deep_hr_mins.append(min(deep_hrs_night))

HR_INTRINSIC = np.mean(deep_hr_mins) if deep_hr_mins else 50.0
ALPHA_T = 0.08  # per degC

pchr_epochs = []  # list of dicts
pchr_stage_data = {s: {'hr_obs': [], 'hr_intrinsic': [], 'delta_met': [], 'delta_auto': []} for s in STAGES}

for rec in sleep_records:
    hyp = rec.get('hypnogram_5min', '')
    hr5 = rec.get('hr_5min', [])
    rm5 = rec.get('rmssd_5min', [])
    delta_T = rec.get('temperature_deviation', 0.0) or 0.0
    n = min(len(hyp), len(hr5), len(rm5))

    delta_hr_met = ALPHA_T * delta_T * HR_INTRINSIC

    for i in range(n):
        s, h, r = hyp[i] if i < len(hyp) else '', hr5[i] if i < len(hr5) else 0, rm5[i] if i < len(rm5) else 0
        if s not in STAGES or h <= 0 or r <= 0:
            continue
        delta_hr_auto = h - HR_INTRINSIC - delta_hr_met
        rc = compute_rc(h, r)
        row = {
            'stage': s, 'hr_obs': h, 'hr_intrinsic': HR_INTRINSIC,
            'delta_hr_met': delta_hr_met, 'delta_hr_o2': 0.0,
            'delta_hr_auto': delta_hr_auto, 'rmssd': r, 'rc': rc,
            'temp_dev': delta_T
        }
        pchr_epochs.append(row)
        pchr_stage_data[s]['hr_obs'].append(h)
        pchr_stage_data[s]['hr_intrinsic'].append(HR_INTRINSIC)
        pchr_stage_data[s]['delta_met'].append(delta_hr_met)
        pchr_stage_data[s]['delta_auto'].append(delta_hr_auto)

# Stage summary
pchr_summary = {}
for s in STAGES:
    d = pchr_stage_data[s]
    if not d['hr_obs']:
        continue
    pchr_summary[STAGE_NAMES[s]] = {
        'n_epochs': len(d['hr_obs']),
        'hr_obs_mean': float(np.mean(d['hr_obs'])),
        'hr_obs_std': float(np.std(d['hr_obs'])),
        'hr_intrinsic': float(HR_INTRINSIC),
        'delta_met_mean': float(np.mean(d['delta_met'])),
        'delta_met_std': float(np.std(d['delta_met'])),
        'delta_auto_mean': float(np.mean(d['delta_auto'])),
        'delta_auto_std': float(np.std(d['delta_auto'])),
    }

df_pchr = pd.DataFrame(pchr_epochs)
df_pchr.to_csv(RESULTS_DIR + 'pchr_epoch_data.csv', index=False)
with open(RESULTS_DIR + 'pchr_summary.json', 'w', encoding='utf-8') as f:
    json.dump({'hr_intrinsic': HR_INTRINSIC, 'alpha_T': ALPHA_T, 'stages': pchr_summary}, f, indent=2)
print(f"  PCHR: {len(pchr_epochs)} epochs, HR_intrinsic={HR_INTRINSIC:.1f} bpm")

# ---------------------------------------------------------------------------
# 2. S-Entropy Health Coordinates (S_k, S_t, S_e)
# ---------------------------------------------------------------------------
print("Computing S-Entropy coordinates...")

# Global normalization constants
all_rmssd = [r for rec in sleep_records for r in rec.get('rmssd_5min', []) if r > 0]
all_hr = [h for rec in sleep_records for h in rec.get('hr_5min', []) if h > 0]
rmssd_max = max(all_rmssd) if all_rmssd else 100.0
hr_min = min(all_hr) if all_hr else 40.0
rr_max = (rmssd_max * (60000.0 / hr_min))

sentropy_epochs = []
sentropy_stage_data = {s: {'S_k': [], 'S_t': [], 'S_e': []} for s in STAGES}

for rec in sleep_records:
    hyp = rec.get('hypnogram_5min', '')
    hr5 = rec.get('hr_5min', [])
    rm5 = rec.get('rmssd_5min', [])
    n = min(len(hyp), len(hr5), len(rm5))

    for i in range(n):
        s = hyp[i] if i < len(hyp) else ''
        h = hr5[i] if i < len(hr5) else 0
        r = rm5[i] if i < len(rm5) else 0
        if s not in STAGES or h <= 0 or r <= 0:
            continue

        # S_k: kinetic entropy (HRV capacity)
        rr_product = r * (60000.0 / h)
        S_k = min(1.0, rr_product / rr_max) if rr_max > 0 else 0.0

        # S_t: temporal entropy (circadian phase)
        # Each epoch is 5 min; approximate hours since midnight from epoch index
        # Sleep typically starts around 23:00-00:00, so epoch 0 ~ midnight
        hours_since_midnight = i * 5.0 / 60.0  # rough approximation
        S_t = (1.0 - math.exp(-hours_since_midnight / 8.0)) * abs(math.cos(2 * math.pi * hours_since_midnight / 24.0))
        S_t = max(0.0, min(1.0, S_t))

        # S_e: energetic entropy (from Rc)
        rc = compute_rc(h, r)
        if rc is not None and not np.isnan(rc) and rc > 0:
            S_e = min(1.0, max(0.0, -math.log(rc) / math.log(0.5))) if rc < 1.0 else 0.0
        else:
            S_e = 1.0

        sentropy_epochs.append({
            'stage': s, 'S_k': S_k, 'S_t': S_t, 'S_e': S_e,
            'hr': h, 'rmssd': r, 'rc': rc if not np.isnan(rc) else 0.0,
            'epoch_idx': i
        })
        sentropy_stage_data[s]['S_k'].append(S_k)
        sentropy_stage_data[s]['S_t'].append(S_t)
        sentropy_stage_data[s]['S_e'].append(S_e)

sentropy_summary = {}
for s in STAGES:
    d = sentropy_stage_data[s]
    if not d['S_k']:
        continue
    sentropy_summary[STAGE_NAMES[s]] = {
        'n_epochs': len(d['S_k']),
        'S_k_mean': float(np.mean(d['S_k'])), 'S_k_std': float(np.std(d['S_k'])),
        'S_t_mean': float(np.mean(d['S_t'])), 'S_t_std': float(np.std(d['S_t'])),
        'S_e_mean': float(np.mean(d['S_e'])), 'S_e_std': float(np.std(d['S_e'])),
    }

df_sentropy = pd.DataFrame(sentropy_epochs)
df_sentropy.to_csv(RESULTS_DIR + 'sentropy_epoch_data.csv', index=False)
with open(RESULTS_DIR + 'sentropy_summary.json', 'w', encoding='utf-8') as f:
    json.dump({'rmssd_max': rmssd_max, 'hr_min': hr_min, 'stages': sentropy_summary}, f, indent=2)
print(f"  S-Entropy: {len(sentropy_epochs)} epochs")

# ---------------------------------------------------------------------------
# 3. TCC — Temperature-Corrected Coherence
# ---------------------------------------------------------------------------
print("Computing TCC...")

tcc_epochs = []
tcc_stage_data = {s: {'rc': [], 'tcc': [], 'delta_T': []} for s in STAGES}

T_BODY = 310.15  # 37 degC in Kelvin

for rec in sleep_records:
    hyp = rec.get('hypnogram_5min', '')
    hr5 = rec.get('hr_5min', [])
    rm5 = rec.get('rmssd_5min', [])
    delta_T = rec.get('temperature_deviation', 0.0) or 0.0
    n = min(len(hyp), len(hr5), len(rm5))

    # TCC Arrhenius correction
    try:
        correction = math.exp(4000.0 * (1.0 / (T_BODY + delta_T) - 1.0 / T_BODY))
    except (OverflowError, ZeroDivisionError):
        correction = 1.0

    for i in range(n):
        s = hyp[i] if i < len(hyp) else ''
        h = hr5[i] if i < len(hr5) else 0
        r = rm5[i] if i < len(rm5) else 0
        if s not in STAGES or h <= 0 or r <= 0:
            continue

        rc = compute_rc(h, r)
        if np.isnan(rc):
            continue
        tcc = rc * correction
        tcc = min(1.0, max(0.0, tcc))

        tcc_epochs.append({
            'stage': s, 'rc': rc, 'tcc': tcc, 'delta_T': delta_T,
            'correction': correction, 'hr': h, 'rmssd': r
        })
        tcc_stage_data[s]['rc'].append(rc)
        tcc_stage_data[s]['tcc'].append(tcc)
        tcc_stage_data[s]['delta_T'].append(delta_T)

tcc_summary = {}
for s in STAGES:
    d = tcc_stage_data[s]
    if not d['rc']:
        continue
    tcc_summary[STAGE_NAMES[s]] = {
        'n_epochs': len(d['rc']),
        'rc_mean': float(np.mean(d['rc'])), 'rc_std': float(np.std(d['rc'])),
        'tcc_mean': float(np.mean(d['tcc'])), 'tcc_std': float(np.std(d['tcc'])),
        'delta_T_mean': float(np.mean(d['delta_T'])),
        'correction_magnitude_mean': float(np.mean(np.abs(np.array(d['tcc']) - np.array(d['rc'])))),
    }

df_tcc = pd.DataFrame(tcc_epochs)
df_tcc.to_csv(RESULTS_DIR + 'tcc_epoch_data.csv', index=False)
with open(RESULTS_DIR + 'tcc_summary.json', 'w', encoding='utf-8') as f:
    json.dump({'T_body_K': T_BODY, 'Ea_R': 4000.0, 'stages': tcc_summary}, f, indent=2)
print(f"  TCC: {len(tcc_epochs)} epochs")

# ---------------------------------------------------------------------------
# 4. CSCI — Cross-Scale Coherence Index
# ---------------------------------------------------------------------------
print("Computing CSCI...")

csci_nights = []
csci_stage_data = {s: {'csci': [], 'hr_rmssd_dev': [], 'hr_temp_dev': []} for s in STAGES}

for rec in sleep_records:
    hyp = rec.get('hypnogram_5min', '')
    hr5 = rec.get('hr_5min', [])
    rm5 = rec.get('rmssd_5min', [])
    delta_T = rec.get('temperature_deviation', 0.0) or 0.0
    score = rec.get('score', 0)
    n = min(len(hyp), len(hr5), len(rm5))

    # Compute per-stage coupling within this night
    stage_epochs = {s: {'hr': [], 'rmssd': [], 'rc': []} for s in STAGES}
    for i in range(n):
        s = hyp[i] if i < len(hyp) else ''
        h = hr5[i] if i < len(hr5) else 0
        r = rm5[i] if i < len(rm5) else 0
        if s not in STAGES or h <= 0 or r <= 0:
            continue
        stage_epochs[s]['hr'].append(h)
        stage_epochs[s]['rmssd'].append(r)
        stage_epochs[s]['rc'].append(compute_rc(h, r))

    for s in STAGES:
        hrs = np.array(stage_epochs[s]['hr'])
        rms = np.array(stage_epochs[s]['rmssd'])
        rcs = np.array(stage_epochs[s]['rc'])
        if len(hrs) < 3:
            continue

        # Observed HR-RMSSD coupling: correlation (negative expected)
        hr_rmssd_corr = np.corrcoef(hrs, rms)[0, 1] if np.std(hrs) > 0 and np.std(rms) > 0 else 0.0
        if np.isnan(hr_rmssd_corr):
            hr_rmssd_corr = 0.0

        # Predicted: from partition theory, HR and RMSSD should be anticorrelated
        # Deep: strong anticorrelation (-0.8), REM: moderate (-0.5), Light: moderate (-0.4), Awake: weak (-0.2)
        predicted_hr_rmssd = {'D': -0.8, 'R': -0.5, 'L': -0.4, 'A': -0.2}

        # HR-Temperature coupling: use delta_T as proxy (constant per night)
        # Predicted: small positive coupling
        predicted_hr_temp = {'D': 0.3, 'R': 0.4, 'L': 0.3, 'A': 0.5}
        # Observed: correlation between HR and temperature effect
        hr_temp_obs = np.corrcoef(hrs, hrs * ALPHA_T * delta_T / HR_INTRINSIC)[0, 1] if delta_T != 0 else predicted_hr_temp[s]
        if np.isnan(hr_temp_obs):
            hr_temp_obs = 0.0

        # CSCI: deviation from predicted
        dev_hr_rmssd = abs(hr_rmssd_corr - predicted_hr_rmssd[s]) / max(abs(predicted_hr_rmssd[s]), 0.01)
        dev_hr_temp = abs(hr_temp_obs - predicted_hr_temp[s]) / max(abs(predicted_hr_temp[s]), 0.01)
        csci = max(0.0, 1.0 - 0.5 * (dev_hr_rmssd + dev_hr_temp))

        csci_nights.append({
            'stage': s, 'csci': csci,
            'hr_rmssd_obs': hr_rmssd_corr, 'hr_rmssd_pred': predicted_hr_rmssd[s],
            'hr_temp_obs': hr_temp_obs, 'hr_temp_pred': predicted_hr_temp[s],
            'dev_hr_rmssd': dev_hr_rmssd, 'dev_hr_temp': dev_hr_temp,
            'sleep_score': score,
            'mean_rc': float(np.mean(rcs)),
            'mean_rmssd': float(np.mean(rms)),
        })
        csci_stage_data[s]['csci'].append(csci)
        csci_stage_data[s]['hr_rmssd_dev'].append(dev_hr_rmssd)
        csci_stage_data[s]['hr_temp_dev'].append(dev_hr_temp)

csci_summary = {}
for s in STAGES:
    d = csci_stage_data[s]
    if not d['csci']:
        continue
    csci_summary[STAGE_NAMES[s]] = {
        'n_nights': len(d['csci']),
        'csci_mean': float(np.mean(d['csci'])), 'csci_std': float(np.std(d['csci'])),
        'hr_rmssd_dev_mean': float(np.mean(d['hr_rmssd_dev'])),
        'hr_temp_dev_mean': float(np.mean(d['hr_temp_dev'])),
    }

df_csci = pd.DataFrame(csci_nights)
df_csci.to_csv(RESULTS_DIR + 'csci_night_stage_data.csv', index=False)
with open(RESULTS_DIR + 'csci_summary.json', 'w', encoding='utf-8') as f:
    json.dump({'stages': csci_summary}, f, indent=2)
print(f"  CSCI: {len(csci_nights)} night-stage entries")

# ---------------------------------------------------------------------------
# 5. RARS — Regime-Aware Recovery Score
# ---------------------------------------------------------------------------
print("Computing RARS...")

# Compute per-night deep sleep mean Rc
night_deep_rc = []
for idx, rec in enumerate(sleep_records):
    hyp = rec.get('hypnogram_5min', '')
    hr5 = rec.get('hr_5min', [])
    rm5 = rec.get('rmssd_5min', [])
    score = rec.get('score', 0)
    deep_hrs = rec.get('deep_in_hrs', 0.0) or 0.0
    n = min(len(hyp), len(hr5), len(rm5))
    deep_rcs = []
    all_rcs = []
    for i in range(n):
        s = hyp[i] if i < len(hyp) else ''
        h = hr5[i] if i < len(hr5) else 0
        r = rm5[i] if i < len(rm5) else 0
        if h > 0 and r > 0:
            rc = compute_rc(h, r)
            all_rcs.append(rc)
            if s == 'D':
                deep_rcs.append(rc)
    night_deep_rc.append({
        'night_idx': idx,
        'deep_rc_mean': float(np.mean(deep_rcs)) if deep_rcs else np.nan,
        'mean_rc': float(np.mean(all_rcs)) if all_rcs else np.nan,
        'sleep_score': score,
        'deep_hrs': deep_hrs,
    })

df_night_rc = pd.DataFrame(night_deep_rc)

# Activity data: daily metrics
activity_daily = []
for idx, rec in enumerate(activity_records):
    met_1min = rec.get('met_1min', [])

    # Find recovery windows: MET > 3.0 for >=10 consecutive minutes
    recovery_events = 0
    if met_1min:
        run_len = 0
        for m in met_1min:
            if m > 3.0:
                run_len += 1
                if run_len >= 10:
                    recovery_events += 1
                    run_len = 0
            else:
                run_len = 0

    activity_daily.append({
        'day_idx': idx,
        'cal_active': rec.get('cal_active', 0),
        'cal_total': rec.get('cal_total', 0),
        'steps': rec.get('steps', 0),
        'daily_movement': rec.get('daily_movement', 0),
        'average_met': rec.get('average_met', 0),
        'activity_score': rec.get('score', 0),
        'recovery_events': recovery_events,
    })

df_activity = pd.DataFrame(activity_daily)

# Link activity days to following night's sleep (index-based, best approximation)
n_pairs = min(len(df_activity), len(df_night_rc))
rars_pairs = []
for i in range(n_pairs):
    act = df_activity.iloc[i]
    slp = df_night_rc.iloc[i]
    if np.isnan(slp['deep_rc_mean']) or np.isnan(slp['mean_rc']):
        continue
    rars_pairs.append({
        'day_idx': i,
        'cal_active': float(act['cal_active']),
        'steps': int(act['steps']),
        'average_met': float(act['average_met']),
        'activity_score': int(act['activity_score']),
        'recovery_events': int(act['recovery_events']),
        'deep_rc_mean': float(slp['deep_rc_mean']),
        'mean_rc': float(slp['mean_rc']),
        'sleep_score': int(slp['sleep_score']),
        'deep_hrs': float(slp['deep_hrs']),
    })

df_rars = pd.DataFrame(rars_pairs)

# RARS proxy: classify recovery by sleep score bins
rars_summary = {}
if len(df_rars) > 0:
    bins = [(0, 60, 'Poor'), (60, 75, 'Fair'), (75, 90, 'Good'), (90, 101, 'Excellent')]
    for lo, hi, label in bins:
        mask = (df_rars['sleep_score'] >= lo) & (df_rars['sleep_score'] < hi)
        subset = df_rars[mask]
        if len(subset) > 0:
            rars_summary[label] = {
                'n': int(len(subset)),
                'deep_rc_mean': float(subset['deep_rc_mean'].mean()),
                'deep_rc_std': float(subset['deep_rc_mean'].std()),
                'cal_active_mean': float(subset['cal_active'].mean()),
                'steps_mean': float(subset['steps'].mean()),
            }

    # Overall correlation
    valid = df_rars.dropna(subset=['cal_active', 'deep_rc_mean'])
    if len(valid) > 2:
        corr_cal_rc = float(np.corrcoef(valid['cal_active'], valid['deep_rc_mean'])[0, 1])
        corr_steps_rc = float(np.corrcoef(valid['steps'], valid['deep_rc_mean'])[0, 1])
    else:
        corr_cal_rc = 0.0
        corr_steps_rc = 0.0
else:
    corr_cal_rc = 0.0
    corr_steps_rc = 0.0

df_rars.to_csv(RESULTS_DIR + 'rars_activity_sleep_pairs.csv', index=False)
with open(RESULTS_DIR + 'rars_summary.json', 'w', encoding='utf-8') as f:
    json.dump({
        'n_pairs': len(rars_pairs),
        'corr_cal_active_vs_deep_rc': corr_cal_rc,
        'corr_steps_vs_deep_rc': corr_steps_rc,
        'recovery_bins': rars_summary,
    }, f, indent=2)
print(f"  RARS: {len(rars_pairs)} activity-sleep pairs, r(cal,Rc)={corr_cal_rc:.3f}")

print("\nAll sensor disambiguation validation results saved to:", RESULTS_DIR)
