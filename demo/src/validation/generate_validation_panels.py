"""
Validation Panel Charts — one 4-chart panel per result file.
Each panel: 4 charts in a row, white background, at least one 3D chart,
minimal text, all charts use real numerical data.
"""

import json, math, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

BASE   = 'c:/Users/kunda/Documents/health/brut/demo/src/validation/results/'
OUTDIR = 'c:/Users/kunda/Documents/health/brut/demo/src/validation/panels/'
os.makedirs(OUTDIR, exist_ok=True)

STAGE_COLORS = {'A': '#4C72B0', 'L': '#55A868', 'D': '#C44E52', 'R': '#8172B2'}
REGIME_COLORS = {
    'phase_locked': '#1a237e', 'coherent': '#1565c0',
    'cascade': '#2e7d32',      'aperture': '#e65100',
    'turbulent': '#b71c1c',
}

def panel_style(fig):
    fig.patch.set_facecolor('white')

def ax_style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)
    if title:  ax.set_title(title, fontsize=8, pad=4, fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel, fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, fontsize=7)

def ax3d_style(ax, title=''):
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.tick_params(labelsize=6)
    if title: ax.set_title(title, fontsize=8, pad=4, fontweight='bold')

def save_panel(fig, name):
    path = f'{OUTDIR}{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  saved -> {path}')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Oura Sleep Stage Rc (oura_stage_rc_statistics.csv)
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 1: Oura sleep stage Rc")
df_oura = pd.read_csv(BASE + 'oura_stage_rc_statistics.csv')
with open(BASE + 'oura_sleep_regime_validation.json') as f:
    oura_json = json.load(f)

# Rebuild per-epoch data from raw records for distributions
import sys
sys.path.insert(0, 'c:/Users/kunda/Documents/health/brut')
with open('c:/Users/kunda/Documents/health/brut/demo/public/sleep_ppg_records.json') as f:
    sleep_recs = json.load(f)

stage_rc_vals   = {'A': [], 'L': [], 'D': [], 'R': []}
stage_rmssd_vals = {'A': [], 'L': [], 'D': [], 'R': []}
stage_hr_vals   = {'A': [], 'L': [], 'D': [], 'R': []}
for rec in sleep_recs:
    hyp, hr5, rm5 = rec.get('hypnogram_5min',''), rec.get('hr_5min',[]), rec.get('rmssd_5min',[])
    n = min(len(hyp), len(hr5), len(rm5))
    for i in range(n):
        s, h, r = hyp[i], hr5[i], rm5[i]
        if s in 'ALDR' and h > 0 and r > 0:
            cv = (r * h) / 60000.0
            rc = math.exp(-2 * math.pi**2 * cv**2)
            stage_rc_vals[s].append(rc)
            stage_rmssd_vals[s].append(r)
            stage_hr_vals[s].append(h)

stage_order = ['D', 'A', 'R', 'L']
stage_names = {'A': 'Awake', 'L': 'Light', 'D': 'Deep_SWS', 'R': 'REM'}
stage_labels = {'A': 'Awake', 'L': 'Light', 'D': 'Deep', 'R': 'REM'}
cols = [STAGE_COLORS[s] for s in stage_order]

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
panel_style(fig)

# Chart 1: Violin of Rc per stage
ax = axes[0]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
parts = ax.violinplot([stage_rc_vals[s] for s in stage_order],
                       positions=range(4), widths=0.7, showmedians=True,
                       showextrema=False)
for i, (pc, s) in enumerate(zip(parts['bodies'], stage_order)):
    pc.set_facecolor(STAGE_COLORS[s]); pc.set_alpha(0.75)
parts['cmedians'].set_color('#333'); parts['cmedians'].set_linewidth(2)
ax.axhspan(0.95, 1.0, alpha=0.08, color='navy', label='Phase-locked')
ax.axhspan(0.80, 0.95, alpha=0.08, color='royalblue', label='Coherent')
ax.axhspan(0.50, 0.80, alpha=0.08, color='green')
ax.set_xticks(range(4))
ax.set_xticklabels([stage_labels[s] for s in stage_order], fontsize=8)
ax.set_ylabel('$R_c$', fontsize=9); ax.set_ylim(0.7, 1.02)
ax_style(ax, title='$R_c$ Distribution per Stage')

# Chart 2: Mean RMSSD bar chart
ax = axes[1]
rmssd_means = [np.mean(stage_rmssd_vals[s]) for s in stage_order]
rmssd_stds  = [np.std(stage_rmssd_vals[s])  for s in stage_order]
bars = ax.bar(range(4), rmssd_means, yerr=rmssd_stds, color=cols,
              alpha=0.82, capsize=4, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(4))
ax.set_xticklabels([stage_labels[s] for s in stage_order], fontsize=8)
ax_style(ax, title='Mean RMSSD per Stage', ylabel='RMSSD (ms)')
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Chart 3: 3D scatter (stage index, Rc, RMSSD)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: Stage × $R_c$ × RMSSD')
for si, s in enumerate(stage_order):
    rc_s  = np.array(stage_rc_vals[s])
    rm_s  = np.array(stage_rmssd_vals[s])
    # downsample to 400 pts for speed
    idx = np.random.choice(len(rc_s), min(400, len(rc_s)), replace=False)
    jitter = np.random.normal(0, 0.08, len(idx))
    ax3.scatter(np.full(len(idx), si) + jitter, rc_s[idx], rm_s[idx],
                c=STAGE_COLORS[s], alpha=0.4, s=6)
ax3.set_xticks(range(4))
ax3.set_xticklabels([stage_labels[s] for s in stage_order], fontsize=6)
ax3.set_xlabel('Stage', fontsize=6, labelpad=2)
ax3.set_ylabel('$R_c$', fontsize=6, labelpad=2)
ax3.set_zlabel('RMSSD', fontsize=6, labelpad=2)
ax3.view_init(elev=22, azim=225)
ax3.set_facecolor('white')

# Chart 4: Epoch accuracy vs predicted regime boundary
ax = axes[3]
stage_name_order = [stage_names[s] for s in stage_order]  # ['Deep','Awake','REM','Light']
df_s = df_oura.set_index('stage').loc[stage_name_order]
acc  = df_s['epoch_accuracy'].values * 100
x    = np.arange(4)
bar_c = [STAGE_COLORS[s] for s in stage_order]
ax.bar(x, acc, color=bar_c, alpha=0.82, edgecolor='white')
ax.axhline(50, color='#666', linewidth=1, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels([stage_labels[s] for s in stage_order], fontsize=8)
ax.set_ylim(0, 100)
ax_style(ax, title='Epoch Accuracy in Predicted Regime', ylabel='%')
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.tight_layout(w_pad=2)
save_panel(fig, 'panel1_oura_sleep_stage_rc')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Oura Transition Matrix + Score Correlations
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 2: Oura transitions and sleep scores")
df_trans = pd.read_csv(BASE + 'oura_transition_matrix.csv')
df_score = pd.read_csv(BASE + 'oura_rc_sleep_score_pairs.csv')

# Build 4x4 matrix
stages = ['A','L','D','R']
T = np.zeros((4, 4))
for _, row in df_trans.iterrows():
    i = stages.index(row['from'])
    j = stages.index(row['to'])
    T[i, j] = row['probability']

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
panel_style(fig)

# Chart 1: Transition heatmap
ax = axes[0]
im = ax.imshow(T, cmap='Blues', vmin=0, vmax=0.75)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(stages, fontsize=9); ax.set_yticklabels(stages, fontsize=9)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{T[i,j]:.2f}', ha='center', va='center',
                fontsize=8, color='white' if T[i,j] > 0.4 else '#333')
ax.set_facecolor('white')
ax_style(ax, title='Transition Probabilities')

# Chart 2: Rc vs sleep_score scatter
ax = axes[1]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
sc = ax.scatter(df_score['mean_rc'], df_score['sleep_score'],
                c=df_score['deep_hrs'], cmap='viridis', alpha=0.65, s=30)
# Trend line
m, b = np.polyfit(df_score['mean_rc'], df_score['sleep_score'], 1)
x_ = np.linspace(df_score['mean_rc'].min(), df_score['mean_rc'].max(), 50)
ax.plot(x_, m*x_+b, color='#e53935', linewidth=1.5, linestyle='--')
plt.colorbar(sc, ax=ax, label='Deep (hrs)', shrink=0.8)
ax_style(ax, title='$R_c$ vs Sleep Score', xlabel='Mean $R_c$', ylabel='Sleep Score')

# Chart 3: 3D — (Rc, deep_hrs, sleep_score)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: $R_c$ × Deep × Score')
df_s3 = df_score.dropna()
sc3 = ax3.scatter(df_s3['mean_rc'], df_s3['deep_hrs'], df_s3['sleep_score'],
                   c=df_s3['sleep_score'], cmap='plasma', alpha=0.7, s=20)
ax3.set_xlabel('$R_c$', fontsize=6, labelpad=2)
ax3.set_ylabel('Deep hrs', fontsize=6, labelpad=2)
ax3.set_zlabel('Score', fontsize=6, labelpad=2)
ax3.view_init(elev=25, azim=210)
ax3.set_facecolor('white')

# Chart 4: Stage run-length distribution (box-like: mean ± std from json)
ax = axes[3]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
run_data = oura_json['run_length_statistics']
names_map = {'Awake':'A','Light':'L','Deep_SWS':'D','REM':'R'}
# Recreate run lengths from raw data
run_lens = {'A': [], 'L': [], 'D': [], 'R': []}
for rec in sleep_recs:
    hyp = rec.get('hypnogram_5min','')
    i = 0
    while i < len(hyp):
        s = hyp[i]
        if s not in 'ALDR': i += 1; continue
        j = i
        while j < len(hyp) and hyp[j] == s: j += 1
        run_lens[s].append((j-i)*5)
        i = j
bplot = ax.boxplot([run_lens[s] for s in stage_order],
                    patch_artist=True, widths=0.5, showfliers=False,
                    medianprops={'color': '#333', 'linewidth': 2})
for patch, s in zip(bplot['boxes'], stage_order):
    patch.set_facecolor(STAGE_COLORS[s]); patch.set_alpha(0.75)
ax.set_xticks(range(1, 5))
ax.set_xticklabels([stage_labels[s] for s in stage_order], fontsize=8)
ax_style(ax, title='Stage Run Lengths (min)', ylabel='minutes')

fig.tight_layout(w_pad=2)
save_panel(fig, 'panel2_oura_transitions_scores')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — MIT-BIH Cardiac Regime Classification
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 3: MIT-BIH arrhythmia regime classification")
df_rhy = pd.read_csv(BASE + 'mitdb_rhythm_rc_statistics.csv')
df_win = pd.read_csv(BASE + 'mitdb_window_level_rc.csv')

focus_rhythms = ['Normal_SR', 'Atrial_Fib', 'V_Tachycardia', 'Bigeminy', 'Atrial_Flutter']
rhy_colors = {
    'Normal_SR': '#1565c0', 'Atrial_Fib': '#b71c1c',
    'V_Tachycardia': '#e65100', 'Bigeminy': '#4a148c',
    'Atrial_Flutter': '#880e4f', 'Trigeminy': '#2e7d32',
    'SVT': '#006064', 'Idiov_Rhythm': '#33691e',
}

df_focus = df_rhy[df_rhy['rhythm_name'].isin(focus_rhythms)].copy()
df_win_f = df_win[df_win['rhythm_name'].isin(focus_rhythms)].copy()

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
panel_style(fig)

# Chart 1: Box plot of Rc per rhythm (from window-level data)
ax = axes[0]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
grouped = [df_win_f[df_win_f['rhythm_name']==r]['rc'].values for r in focus_rhythms]
bp = ax.boxplot(grouped, patch_artist=True, widths=0.55,
                showfliers=True, flierprops={'markersize': 2, 'alpha': 0.3},
                medianprops={'color':'white','linewidth':2})
for patch, r in zip(bp['boxes'], focus_rhythms):
    patch.set_facecolor(rhy_colors.get(r, '#999')); patch.set_alpha(0.8)
short_names = ['NSR', 'AFIB', 'VT', 'BigR', 'AFL']
ax.set_xticks(range(1, len(focus_rhythms)+1))
ax.set_xticklabels(short_names, fontsize=8)
# Regime bands
for lo, hi, clr in [(0.95,1.0,'#e3f2fd'),(0.80,0.95,'#bbdefb'),
                     (0.50,0.80,'#e8f5e9'),(0.30,0.50,'#fff3e0'),(0.00,0.30,'#ffebee')]:
    ax.axhspan(lo, hi, alpha=0.15, color=clr)
ax_style(ax, title='$R_c$ by Rhythm Type', ylabel='$R_c$')

# Chart 2: KDE of Rc distributions
from scipy.stats import gaussian_kde
ax = axes[1]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
x_grid = np.linspace(0, 1, 300)
for r, sn in zip(focus_rhythms, short_names):
    data = df_win_f[df_win_f['rhythm_name']==r]['rc'].values
    if len(data) > 5:
        kde = gaussian_kde(data, bw_method=0.15)
        ax.plot(x_grid, kde(x_grid), color=rhy_colors.get(r,'#999'),
                linewidth=2.0, label=sn, alpha=0.85)
ax.legend(fontsize=7, loc='upper left', frameon=False)
for lo, hi, clr in [(0.95,1.0,'#e3f2fd'),(0.80,0.95,'#bbdefb'),
                     (0.50,0.80,'#e8f5e9'),(0.30,0.50,'#fff3e0'),(0.00,0.30,'#ffebee')]:
    ax.axvspan(lo, hi, alpha=0.1, color=clr)
ax_style(ax, title='$R_c$ Density', xlabel='$R_c$', ylabel='density')

# Chart 3: 3D scatter (rhythm_idx, Rc, mean_HR)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: Rhythm × $R_c$ × HR')
for ri, r in enumerate(focus_rhythms):
    sub = df_win_f[df_win_f['rhythm_name']==r]
    if len(sub) == 0: continue
    idx = np.random.choice(len(sub), min(300, len(sub)), replace=False)
    sub2 = sub.iloc[idx]
    jitter = np.random.normal(0, 0.12, len(sub2))
    ax3.scatter(np.full(len(sub2), ri) + jitter,
                sub2['rc'].values, sub2['mean_hr'].values,
                c=rhy_colors.get(r,'#999'), alpha=0.45, s=8)
ax3.set_xticks(range(len(focus_rhythms)))
ax3.set_xticklabels(short_names, fontsize=5)
ax3.set_xlabel('Rhythm', fontsize=6, labelpad=2)
ax3.set_ylabel('$R_c$', fontsize=6, labelpad=2)
ax3.set_zlabel('HR (bpm)', fontsize=6, labelpad=2)
ax3.view_init(elev=22, azim=210)
ax3.set_facecolor('white')

# Chart 4: Stacked bar of regime fractions
ax = axes[3]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
regime_cols_order = ['pct_phase_locked','pct_coherent','pct_cascade','pct_aperture','pct_turbulent']
regime_palette = ['#1a237e','#1565c0','#2e7d32','#e65100','#b71c1c']
df_foc = df_focus.set_index('rhythm_name').loc[focus_rhythms]
bottom = np.zeros(len(focus_rhythms))
for col, clr in zip(regime_cols_order, regime_palette):
    vals = df_foc[col].values
    ax.bar(range(len(focus_rhythms)), vals, bottom=bottom,
           color=clr, alpha=0.85, width=0.6, edgecolor='white')
    bottom += vals
ax.set_xticks(range(len(focus_rhythms)))
ax.set_xticklabels(short_names, fontsize=8)
ax.set_ylim(0, 102)
ax_style(ax, title='Regime Fractions (%)', ylabel='% of windows')

fig.tight_layout(w_pad=2)
save_panel(fig, 'panel3_mitdb_cardiac_regimes')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — CHF vs NSR Coherence Deficit
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 4: CHF vs NSR")
df_chf = pd.read_csv(BASE + 'chf_window_level_rc.csv')
with open(BASE + 'chf_vs_nsr_coherence_deficit.json') as f:
    chf_json = json.load(f)

nsr_rc = df_win[df_win['rhythm_code']=='N']['rc'].values
chf_rc = df_chf['rc'].values

REGIME_NAMES = ['phase_locked','coherent','cascade','aperture','turbulent']
REGIME_BOUNDS_V = [(0.95,1.0),(0.80,0.95),(0.50,0.80),(0.30,0.50),(0.0,0.30)]
reg_colors = ['#1a237e','#1565c0','#2e7d32','#e65100','#b71c1c']

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
panel_style(fig)

# Chart 1: Histogram overlay
ax = axes[0]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
bins = np.linspace(0, 1, 40)
ax.hist(nsr_rc, bins=bins, alpha=0.55, color='#1565c0', density=True, label='NSR')
ax.hist(chf_rc, bins=bins, alpha=0.55, color='#c62828', density=True, label='CHF')
for (lo, hi), clr in zip(REGIME_BOUNDS_V, reg_colors):
    ax.axvspan(lo, hi, alpha=0.06, color=clr)
ax.legend(fontsize=8, frameon=False)
ax_style(ax, title='$R_c$ Distribution', xlabel='$R_c$', ylabel='density')

# Chart 2: Regime fraction comparison (grouped bars)
ax = axes[1]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
nsr_frac = np.array([chf_json['nsr_regime_distribution'].get(r, 0)*100 for r in REGIME_NAMES])
chf_frac = np.array([chf_json['chf_regime_distribution'].get(r, 0)*100 for r in REGIME_NAMES])
x = np.arange(5)
ax.bar(x - 0.2, nsr_frac, 0.38, color='#1565c0', alpha=0.8, label='NSR')
ax.bar(x + 0.2, chf_frac, 0.38, color='#c62828', alpha=0.8, label='CHF')
ax.set_xticks(x)
reg_short = ['Phase\nLock','Coherent','Cascade','Aperture','Turbulent']
ax.set_xticklabels(reg_short, fontsize=7)
ax.legend(fontsize=8, frameon=False)
ax_style(ax, title='Regime Fractions (%)', ylabel='%')

# Chart 3: 3D scatter (record, window, Rc) for CHF
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: CHF — Record × Window × $R_c$')
chf_recs = sorted(df_chf['record'].unique())
rec_idx  = {r: i for i, r in enumerate(chf_recs)}
sample   = df_chf.sample(min(800, len(df_chf)), random_state=42)
ri  = sample['record'].map(rec_idx).values
wi  = sample['window'].values
rci = sample['rc'].values
sc3 = ax3.scatter(ri, wi, rci, c=rci, cmap='RdYlGn', alpha=0.5, s=6,
                  vmin=0, vmax=1)
ax3.set_xlabel('Record', fontsize=6, labelpad=2)
ax3.set_ylabel('Window', fontsize=6, labelpad=2)
ax3.set_zlabel('$R_c$', fontsize=6, labelpad=2)
ax3.view_init(elev=25, azim=225)
ax3.set_facecolor('white')

# Chart 4: KDE comparison NSR vs CHF
ax = axes[3]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
x_g = np.linspace(0, 1, 300)
kde_nsr = gaussian_kde(nsr_rc, bw_method=0.12)
kde_chf = gaussian_kde(chf_rc, bw_method=0.12)
ax.fill_between(x_g, kde_nsr(x_g), alpha=0.35, color='#1565c0')
ax.fill_between(x_g, kde_chf(x_g), alpha=0.35, color='#c62828')
ax.plot(x_g, kde_nsr(x_g), color='#1565c0', linewidth=2, label='NSR')
ax.plot(x_g, kde_chf(x_g), color='#c62828', linewidth=2, label='CHF')
ax.axvline(np.mean(nsr_rc), color='#1565c0', linestyle='--', linewidth=1.5)
ax.axvline(np.mean(chf_rc), color='#c62828', linestyle='--', linewidth=1.5)
ax.legend(fontsize=8, frameon=False)
ax_style(ax, title='$R_c$ Density (NSR vs CHF)', xlabel='$R_c$', ylabel='density')

fig.tight_layout(w_pad=2)
save_panel(fig, 'panel4_chf_nsr_coherence_deficit')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — Cardiac-Neural Decoupling
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 5: Cardiac-neural decoupling")
df_cn = pd.read_csv(BASE + 'cardiac_neural_decoupling.csv')

stage_order_cn = ['N3_SWS', 'N2', 'N1', 'Wake', 'REM']
# N1 not in df_cn; use what we have
df_cn = df_cn.set_index('stage')
stages_cn = list(df_cn.index)
stage_short = {'Wake':'Wake','N1':'N1','N2':'N2','N3_SWS':'SWS','REM':'REM'}
cn_colors = ['#1a237e','#1565c0','#2e7d32','#e65100','#b71c1c']
cn_stage_c = {s: cn_colors[i % len(cn_colors)] for i, s in enumerate(stages_cn)}

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
panel_style(fig)

# Chart 1: Scatter Rc vs Rn with diagonal
ax = axes[0]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.plot([0.5, 1.05], [0.5, 1.05], color='#ccc', linewidth=1.5, linestyle='--')
for s in stages_cn:
    rc = df_cn.loc[s, 'cardiac_rc']
    rn = df_cn.loc[s, 'neural_rn_estimate']
    ax.scatter(rc, rn, color=cn_stage_c[s], s=100, zorder=5)
    ax.annotate(stage_short.get(s, s), (rc, rn),
                textcoords='offset points', xytext=(5, 3), fontsize=7)
    ax.annotate('', xy=(rc, rn), xytext=(0.93, 0.93),
                arrowprops=dict(arrowstyle='->', color=cn_stage_c[s],
                                lw=1.2, alpha=0.5))
ax.set_xlim(0.88, 1.02); ax.set_ylim(0.40, 1.05)
ax_style(ax, title='$R_c$ vs $R_n$', xlabel='Cardiac $R_c$', ylabel='Neural $R_n$')

# Chart 2: Gap bar chart
ax = axes[1]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
gaps = [df_cn.loc[s, 'abs_gap'] for s in stages_cn]
bar_colors = [cn_stage_c[s] for s in stages_cn]
bars = ax.bar(range(len(stages_cn)), gaps, color=bar_colors, alpha=0.82, edgecolor='white')
ax.axhline(0.15, color='#e53935', linewidth=1.5, linestyle='--')
ax.set_xticks(range(len(stages_cn)))
ax.set_xticklabels([stage_short.get(s,s) for s in stages_cn], fontsize=8)
ax_style(ax, title='|$R_c - R_n$| per Stage', ylabel='Gap')

# Chart 3: 3D — (Rc, Rn, formula_error)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: $R_c$ × $R_n$ × Formula Error')
for s in stages_cn:
    rc  = df_cn.loc[s, 'cardiac_rc']
    rn  = df_cn.loc[s, 'neural_rn_estimate']
    err = df_cn.loc[s, 'formula_error']
    ax3.scatter([rc], [rn], [err], color=cn_stage_c[s], s=120, alpha=0.9)
    ax3.text(rc, rn, err, stage_short.get(s,s), fontsize=6)
ax3.set_xlabel('$R_c$', fontsize=6, labelpad=2)
ax3.set_ylabel('$R_n$', fontsize=6, labelpad=2)
ax3.set_zlabel('Formula Err.', fontsize=6, labelpad=2)
ax3.view_init(elev=25, azim=210)
ax3.set_facecolor('white')

# Chart 4: EEG band profiles (radar/parallel coords)
ax = axes[3]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
bands = ['eeg_delta','eeg_theta','eeg_alpha','eeg_sigma','eeg_beta']
band_labels = ['δ','θ','α','σ','β']
x_b = np.arange(len(bands))
for s in stages_cn:
    vals = [df_cn.loc[s, b] for b in bands]
    ax.plot(x_b, vals, color=cn_stage_c[s], linewidth=2, alpha=0.8,
            label=stage_short.get(s,s), marker='o', markersize=4)
ax.set_xticks(x_b)
ax.set_xticklabels(band_labels, fontsize=11)
ax.legend(fontsize=7, frameon=False, loc='upper right')
ax_style(ax, title='EEG Band Power by Stage', ylabel='Relative Power')

fig.tight_layout(w_pad=2)
save_panel(fig, 'panel5_cardiac_neural_decoupling')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 6 — Window-level MIT-BIH: full Rc landscape
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 6: MIT-BIH window-level Rc landscape")
df_all = pd.read_csv(BASE + 'mitdb_window_level_rc.csv')
main_rhythms = ['Normal_SR','Atrial_Fib','V_Tachycardia','Bigeminy','Atrial_Flutter','SVT']
rhy_palette  = ['#1565c0','#b71c1c','#e65100','#4a148c','#880e4f','#006064']
rhy_to_idx   = {r: i for i, r in enumerate(main_rhythms)}

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
panel_style(fig)

# Chart 1: HR vs Rc scatter coloured by regime
ax = axes[0]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
reg_col_map = {'phase_locked':'#1a237e','coherent':'#1565c0','cascade':'#2e7d32',
               'aperture':'#e65100','turbulent':'#b71c1c'}
c_list = df_all['regime'].map(reg_col_map).fillna('#aaa')
ax.scatter(df_all['mean_hr'], df_all['rc'], c=c_list, alpha=0.3, s=6)
ax_style(ax, title='HR vs $R_c$ (coloured by regime)',
         xlabel='Heart Rate (bpm)', ylabel='$R_c$')

# Chart 2: Ectopic fraction vs Rc
ax = axes[1]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.scatter(df_all['v_fraction'], df_all['rc'], c=c_list, alpha=0.25, s=5)
ax_style(ax, title='Ectopic Fraction vs $R_c$',
         xlabel='V-beat fraction', ylabel='$R_c$')
# Trend line
xf, yf = df_all['v_fraction'].values, df_all['rc'].values
m, b = np.polyfit(xf, yf, 1)
x_ = np.linspace(0, xf.max(), 50)
ax.plot(x_, m*x_+b, color='#333', linewidth=1.5, linestyle='--')

# Chart 3: 3D — (HR, v_fraction, Rc)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: HR × Ectopic × $R_c$')
sample = df_all.sample(min(800, len(df_all)), random_state=0)
c3 = sample['regime'].map(reg_col_map).fillna('#aaa')
ax3.scatter(sample['mean_hr'], sample['v_fraction'], sample['rc'],
            c=c3, alpha=0.4, s=8)
ax3.set_xlabel('HR', fontsize=6, labelpad=2)
ax3.set_ylabel('V-frac', fontsize=6, labelpad=2)
ax3.set_zlabel('$R_c$', fontsize=6, labelpad=2)
ax3.view_init(elev=22, azim=220)
ax3.set_facecolor('white')

# Chart 4: mean Rc per record coloured by dominant rhythm
ax = axes[3]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
rec_stats = df_all.groupby('record').agg(
    mean_rc=('rc', 'mean'), dominant=('rhythm_name', lambda x: x.mode()[0])).reset_index()
rec_stats = rec_stats.sort_values('mean_rc')
col_list  = [rhy_palette[main_rhythms.index(r)] if r in main_rhythms else '#aaa'
             for r in rec_stats['dominant']]
ax.bar(range(len(rec_stats)), rec_stats['mean_rc'], color=col_list, alpha=0.8, width=1.0)
ax.axhline(0.80, color='#1565c0', linewidth=1, linestyle='--')
ax.axhline(0.30, color='#b71c1c', linewidth=1, linestyle='--')
ax_style(ax, title='Mean $R_c$ per Record', xlabel='Record (sorted)', ylabel='$R_c$')

fig.tight_layout(w_pad=2)
save_panel(fig, 'panel6_mitdb_window_landscape')

print("\nAll panels saved to:", OUTDIR)
for fn in sorted(os.listdir(OUTDIR)):
    sz = os.path.getsize(OUTDIR+fn) // 1024
    print(f"  {fn}  ({sz} KB)")
