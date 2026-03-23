"""
Sensor Disambiguation Panel Charts (Panels 16-20).
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats

BASE   = 'c:/Users/kunda/Documents/health/brut/demo/src/validation/results/'
OUTDIR = 'c:/Users/kunda/Documents/health/brut/demo/src/validation/panels/'
os.makedirs(OUTDIR, exist_ok=True)

STAGE_COLORS = {'A': '#4C72B0', 'L': '#55A868', 'D': '#C44E52', 'R': '#8172B2'}
STAGE_ORDER = ['D', 'L', 'R', 'A']
STAGE_LABELS = {'A': 'Awake', 'L': 'Light', 'D': 'Deep', 'R': 'REM'}

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

# Load data
df_pchr = pd.read_csv(BASE + 'pchr_epoch_data.csv')
df_sentropy = pd.read_csv(BASE + 'sentropy_epoch_data.csv')
df_tcc = pd.read_csv(BASE + 'tcc_epoch_data.csv')
df_csci = pd.read_csv(BASE + 'csci_night_stage_data.csv')
df_rars = pd.read_csv(BASE + 'rars_activity_sleep_pairs.csv')

with open(BASE + 'pchr_summary.json', encoding='utf-8') as f:
    pchr_summary = json.load(f)
with open(BASE + 'sentropy_summary.json', encoding='utf-8') as f:
    sentropy_summary = json.load(f)
with open(BASE + 'tcc_summary.json', encoding='utf-8') as f:
    tcc_summary = json.load(f)
with open(BASE + 'csci_summary.json', encoding='utf-8') as f:
    csci_summary = json.load(f)
with open(BASE + 'rars_summary.json', encoding='utf-8') as f:
    rars_summary = json.load(f)

# ===========================================================================
# PANEL 16 -- PCHR Decomposition
# ===========================================================================
print("Panel 16: PCHR Decomposition")
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
panel_style(fig)

# Chart 1: Stacked bar of HR decomposition per stage
ax = axes[0]
stages_present = [s for s in STAGE_ORDER if s in df_pchr['stage'].values]
hr_intr = []
delta_met = []
delta_auto_pos = []
for s in stages_present:
    sub = df_pchr[df_pchr['stage'] == s]
    hr_intr.append(sub['hr_intrinsic'].mean())
    delta_met.append(abs(sub['delta_hr_met'].mean()))
    delta_auto_pos.append(max(0, sub['delta_hr_auto'].mean()))

x = np.arange(len(stages_present))
w = 0.6
ax.bar(x, hr_intr, w, label='Intrinsic', color='#2196F3', alpha=0.85, edgecolor='white')
ax.bar(x, delta_met, w, bottom=hr_intr, label='Metabolic', color='#FF9800', alpha=0.85, edgecolor='white')
ax.bar(x, delta_auto_pos, w, bottom=np.array(hr_intr) + np.array(delta_met),
       label='Autonomic', color='#E91E63', alpha=0.85, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([STAGE_LABELS[s] for s in stages_present], fontsize=8)
ax.legend(fontsize=6, framealpha=0.7)
ax_style(ax, title='HR Decomposition per Stage', ylabel='HR (bpm)')

# Chart 2: Scatter delta_HR_auto vs delta_HR_met colored by stage
ax = axes[1]
for s in stages_present:
    sub = df_pchr[df_pchr['stage'] == s]
    idx = np.random.choice(len(sub), min(300, len(sub)), replace=False)
    ax.scatter(sub['delta_hr_met'].values[idx], sub['delta_hr_auto'].values[idx],
               c=STAGE_COLORS[s], alpha=0.4, s=8, label=STAGE_LABELS[s])
ax.axhline(0, color='#999', linewidth=0.5, linestyle='--')
ax.axvline(0, color='#999', linewidth=0.5, linestyle='--')
ax.legend(fontsize=6, framealpha=0.7, markerscale=2)
ax_style(ax, title='Autonomic vs Metabolic HR Shift',
         xlabel='$\\Delta HR_{met}$ (bpm)', ylabel='$\\Delta HR_{auto}$ (bpm)')

# Chart 3: 3D scatter (HR_obs, delta_HR_met, delta_HR_auto)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
axes[2].set_visible(False)
ax3d_style(ax3, title='3D: $HR_{obs}$ x $\\Delta HR_{met}$ x $\\Delta HR_{auto}$')
for s in stages_present:
    sub = df_pchr[df_pchr['stage'] == s]
    idx = np.random.choice(len(sub), min(300, len(sub)), replace=False)
    ax3.scatter(sub['hr_obs'].values[idx], sub['delta_hr_met'].values[idx],
                sub['delta_hr_auto'].values[idx],
                c=STAGE_COLORS[s], alpha=0.35, s=6, label=STAGE_LABELS[s])
ax3.set_xlabel('$HR_{obs}$', fontsize=6, labelpad=2)
ax3.set_ylabel('$\\Delta HR_{met}$', fontsize=6, labelpad=2)
ax3.set_zlabel('$\\Delta HR_{auto}$', fontsize=6, labelpad=2)
ax3.view_init(elev=22, azim=225)
ax3.legend(fontsize=5, framealpha=0.5, loc='upper left')

# Chart 4: Violin of delta_HR_auto per stage
ax = axes[3]
data_auto = [df_pchr[df_pchr['stage'] == s]['delta_hr_auto'].values for s in stages_present]
parts = ax.violinplot(data_auto, positions=range(len(stages_present)),
                       widths=0.7, showmedians=True, showextrema=False)
for i, (pc, s) in enumerate(zip(parts['bodies'], stages_present)):
    pc.set_facecolor(STAGE_COLORS[s])
    pc.set_alpha(0.75)
parts['cmedians'].set_color('#333')
parts['cmedians'].set_linewidth(2)
ax.axhline(0, color='#999', linewidth=0.5, linestyle='--')
ax.set_xticks(range(len(stages_present)))
ax.set_xticklabels([STAGE_LABELS[s] for s in stages_present], fontsize=8)
ax_style(ax, title='$\\Delta HR_{auto}$ Distribution', ylabel='bpm')

fig.tight_layout(w_pad=2)
save_panel(fig, 'panel16_pchr_decomposition')

# ===========================================================================
# PANEL 17 -- S-Entropy Health Coordinates
# ===========================================================================
print("Panel 17: S-Entropy Health Coordinates")
fig = plt.figure(figsize=(22, 5))
panel_style(fig)

# Chart 1: 3D scatter of (S_k, S_t, S_e) colored by sleep stage -- KEY CHART
ax3 = fig.add_subplot(1, 4, 1, projection='3d')
ax3d_style(ax3, title='S-Entropy Health Coordinates')
for s in STAGE_ORDER:
    sub = df_sentropy[df_sentropy['stage'] == s]
    if len(sub) == 0:
        continue
    idx = np.random.choice(len(sub), min(400, len(sub)), replace=False)
    ax3.scatter(sub['S_k'].values[idx], sub['S_t'].values[idx], sub['S_e'].values[idx],
                c=STAGE_COLORS[s], alpha=0.35, s=8, label=STAGE_LABELS[s])
ax3.set_xlabel('$S_k$', fontsize=7, labelpad=2)
ax3.set_ylabel('$S_t$', fontsize=7, labelpad=2)
ax3.set_zlabel('$S_e$', fontsize=7, labelpad=2)
ax3.view_init(elev=25, azim=135)
ax3.legend(fontsize=6, framealpha=0.5, loc='upper left')

# Chart 2: S_k vs S_e scatter per stage with health region
ax = fig.add_subplot(1, 4, 2)
for s in STAGE_ORDER:
    sub = df_sentropy[df_sentropy['stage'] == s]
    if len(sub) == 0:
        continue
    idx = np.random.choice(len(sub), min(400, len(sub)), replace=False)
    ax.scatter(sub['S_k'].values[idx], sub['S_e'].values[idx],
               c=STAGE_COLORS[s], alpha=0.35, s=8, label=STAGE_LABELS[s])
# Health region overlay
from matplotlib.patches import Rectangle
rect = Rectangle((0.3, 0.0), 0.7, 0.4, linewidth=1.5, edgecolor='green',
                  facecolor='green', alpha=0.08, linestyle='--')
ax.add_patch(rect)
ax.text(0.65, 0.35, 'Healthy', fontsize=6, color='green', alpha=0.6)
ax.legend(fontsize=6, framealpha=0.7, markerscale=2)
ax_style(ax, title='$S_k$ vs $S_e$ per Stage', xlabel='$S_k$ (kinetic)', ylabel='$S_e$ (energetic)')

# Chart 3: S_t distribution per stage (violin)
ax = fig.add_subplot(1, 4, 3)
stages_present = [s for s in STAGE_ORDER if s in df_sentropy['stage'].values]
data_st = [df_sentropy[df_sentropy['stage'] == s]['S_t'].values for s in stages_present]
parts = ax.violinplot(data_st, positions=range(len(stages_present)),
                       widths=0.7, showmedians=True, showextrema=False)
for i, (pc, s) in enumerate(zip(parts['bodies'], stages_present)):
    pc.set_facecolor(STAGE_COLORS[s])
    pc.set_alpha(0.75)
parts['cmedians'].set_color('#333')
parts['cmedians'].set_linewidth(2)
ax.set_xticks(range(len(stages_present)))
ax.set_xticklabels([STAGE_LABELS[s] for s in stages_present], fontsize=8)
ax_style(ax, title='$S_t$ (Temporal) Distribution', ylabel='$S_t$')

# Chart 4: Trajectory over one representative night
ax = fig.add_subplot(1, 4, 4)
# Pick a night with good epoch count (use epoch_idx to find night boundaries)
# Group by finding resets in epoch_idx
night_groups = []
current_group = []
prev_idx = -1
for _, row in df_sentropy.iterrows():
    if row['epoch_idx'] < prev_idx:
        if len(current_group) > 20:
            night_groups.append(pd.DataFrame(current_group))
        current_group = []
    current_group.append(row.to_dict())
    prev_idx = row['epoch_idx']
if len(current_group) > 20:
    night_groups.append(pd.DataFrame(current_group))

# Pick middle night
if night_groups:
    rep_night = night_groups[len(night_groups) // 2]
    epochs = np.arange(len(rep_night))
    ax.plot(epochs, rep_night['S_k'].values, '-', color='#2196F3', linewidth=1.2, alpha=0.8, label='$S_k$')
    ax.plot(epochs, rep_night['S_t'].values, '-', color='#FF9800', linewidth=1.2, alpha=0.8, label='$S_t$')
    ax.plot(epochs, rep_night['S_e'].values, '-', color='#E91E63', linewidth=1.2, alpha=0.8, label='$S_e$')
    # Color background by stage
    for i, s in enumerate(rep_night['stage'].values):
        if s in STAGE_COLORS:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color=STAGE_COLORS[s])
ax.legend(fontsize=6, framealpha=0.7)
ax.set_xlim(0, len(rep_night) if night_groups else 100)
ax_style(ax, title='S-Entropy Trajectory (1 Night)', xlabel='Epoch (5 min)', ylabel='Coordinate Value')

fig.tight_layout(w_pad=2)
save_panel(fig, 'panel17_sentropy_coordinates')

# ===========================================================================
# PANEL 18 -- Temperature-Corrected Coherence
# ===========================================================================
print("Panel 18: Temperature-Corrected Coherence")
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
panel_style(fig)

stages_present = [s for s in STAGE_ORDER if s in df_tcc['stage'].values]

# Chart 1: Scatter raw Rc vs TCC per stage
ax = axes[0]
for s in stages_present:
    sub = df_tcc[df_tcc['stage'] == s]
    idx = np.random.choice(len(sub), min(300, len(sub)), replace=False)
    ax.scatter(sub['rc'].values[idx], sub['tcc'].values[idx],
               c=STAGE_COLORS[s], alpha=0.35, s=8, label=STAGE_LABELS[s])
ax.plot([0.7, 1.0], [0.7, 1.0], 'k--', linewidth=0.8, alpha=0.5)
ax.legend(fontsize=6, framealpha=0.7, markerscale=2)
ax_style(ax, title='Raw $R_c$ vs TCC', xlabel='$R_c$', ylabel='TCC')

# Chart 2: Paired bar chart mean Rc vs mean TCC per stage
ax = axes[1]
x = np.arange(len(stages_present))
w = 0.35
rc_means = [df_tcc[df_tcc['stage'] == s]['rc'].mean() for s in stages_present]
tcc_means = [df_tcc[df_tcc['stage'] == s]['tcc'].mean() for s in stages_present]
rc_stds = [df_tcc[df_tcc['stage'] == s]['rc'].std() for s in stages_present]
tcc_stds = [df_tcc[df_tcc['stage'] == s]['tcc'].std() for s in stages_present]
ax.bar(x - w/2, rc_means, w, yerr=rc_stds, label='$R_c$', color='#4C72B0',
       alpha=0.8, capsize=3, edgecolor='white')
ax.bar(x + w/2, tcc_means, w, yerr=tcc_stds, label='TCC', color='#C44E52',
       alpha=0.8, capsize=3, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([STAGE_LABELS[s] for s in stages_present], fontsize=8)
ax.legend(fontsize=6, framealpha=0.7)
ax_style(ax, title='Mean $R_c$ vs TCC per Stage', ylabel='Value')

# Chart 3: 3D scatter (delta_T, Rc, TCC)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
axes[2].set_visible(False)
ax3d_style(ax3, title='3D: $\\Delta T$ x $R_c$ x TCC')
for s in stages_present:
    sub = df_tcc[df_tcc['stage'] == s]
    idx = np.random.choice(len(sub), min(300, len(sub)), replace=False)
    ax3.scatter(sub['delta_T'].values[idx], sub['rc'].values[idx],
                sub['tcc'].values[idx],
                c=STAGE_COLORS[s], alpha=0.35, s=6)
ax3.set_xlabel('$\\Delta T$ (C)', fontsize=6, labelpad=2)
ax3.set_ylabel('$R_c$', fontsize=6, labelpad=2)
ax3.set_zlabel('TCC', fontsize=6, labelpad=2)
ax3.view_init(elev=22, azim=225)

# Chart 4: Histogram of (TCC - Rc) correction magnitude
ax = axes[3]
diff = df_tcc['tcc'].values - df_tcc['rc'].values
for s in stages_present:
    sub = df_tcc[df_tcc['stage'] == s]
    d = sub['tcc'].values - sub['rc'].values
    ax.hist(d, bins=30, alpha=0.5, color=STAGE_COLORS[s], label=STAGE_LABELS[s], density=True)
ax.axvline(0, color='#333', linewidth=1, linestyle='--')
ax.legend(fontsize=6, framealpha=0.7)
ax_style(ax, title='TCC Correction Magnitude', xlabel='TCC - $R_c$', ylabel='Density')

fig.tight_layout(w_pad=2)
save_panel(fig, 'panel18_tcc_temperature_corrected')

# ===========================================================================
# PANEL 19 -- Cross-Scale Coherence Index
# ===========================================================================
print("Panel 19: Cross-Scale Coherence Index")
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
panel_style(fig)

stages_present = [s for s in STAGE_ORDER if s in df_csci['stage'].values]

# Chart 1: CSCI distribution per stage (violin)
ax = axes[0]
data_csci = [df_csci[df_csci['stage'] == s]['csci'].values for s in stages_present]
parts = ax.violinplot(data_csci, positions=range(len(stages_present)),
                       widths=0.7, showmedians=True, showextrema=False)
for i, (pc, s) in enumerate(zip(parts['bodies'], stages_present)):
    pc.set_facecolor(STAGE_COLORS[s])
    pc.set_alpha(0.75)
parts['cmedians'].set_color('#333')
parts['cmedians'].set_linewidth(2)
ax.set_xticks(range(len(stages_present)))
ax.set_xticklabels([STAGE_LABELS[s] for s in stages_present], fontsize=8)
ax_style(ax, title='CSCI Distribution per Stage', ylabel='CSCI')

# Chart 2: CSCI vs sleep score scatter with trend line
ax = axes[1]
for s in stages_present:
    sub = df_csci[df_csci['stage'] == s]
    ax.scatter(sub['sleep_score'].values, sub['csci'].values,
               c=STAGE_COLORS[s], alpha=0.5, s=15, label=STAGE_LABELS[s])
# Overall trend
valid = df_csci.dropna(subset=['sleep_score', 'csci'])
if len(valid) > 3:
    slope, intercept, r, p, se = stats.linregress(valid['sleep_score'], valid['csci'])
    xline = np.linspace(valid['sleep_score'].min(), valid['sleep_score'].max(), 50)
    ax.plot(xline, slope * xline + intercept, 'k-', linewidth=1.2, alpha=0.6,
            label=f'r={r:.2f}')
ax.legend(fontsize=6, framealpha=0.7, markerscale=1.5)
ax_style(ax, title='CSCI vs Sleep Score', xlabel='Sleep Score', ylabel='CSCI')

# Chart 3: 3D scatter (Rc, RMSSD, CSCI)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
axes[2].set_visible(False)
ax3d_style(ax3, title='3D: $R_c$ x RMSSD x CSCI')
for s in stages_present:
    sub = df_csci[df_csci['stage'] == s]
    ax3.scatter(sub['mean_rc'].values, sub['mean_rmssd'].values, sub['csci'].values,
                c=STAGE_COLORS[s], alpha=0.45, s=12, label=STAGE_LABELS[s])
ax3.set_xlabel('$R_c$', fontsize=6, labelpad=2)
ax3.set_ylabel('RMSSD', fontsize=6, labelpad=2)
ax3.set_zlabel('CSCI', fontsize=6, labelpad=2)
ax3.view_init(elev=22, azim=225)
ax3.legend(fontsize=5, framealpha=0.5, loc='upper left')

# Chart 4: Heatmap of coupling pair deviations per stage
ax = axes[3]
dev_matrix = np.zeros((len(stages_present), 2))
for i, s in enumerate(stages_present):
    sub = df_csci[df_csci['stage'] == s]
    dev_matrix[i, 0] = sub['dev_hr_rmssd'].mean()
    dev_matrix[i, 1] = sub['dev_hr_temp'].mean()

im = ax.imshow(dev_matrix, cmap='YlOrRd', aspect='auto', vmin=0)
ax.set_xticks([0, 1])
ax.set_xticklabels(['HR-RMSSD', 'HR-Temp'], fontsize=7)
ax.set_yticks(range(len(stages_present)))
ax.set_yticklabels([STAGE_LABELS[s] for s in stages_present], fontsize=8)
for i in range(len(stages_present)):
    for j in range(2):
        ax.text(j, i, f'{dev_matrix[i, j]:.2f}', ha='center', va='center', fontsize=7, color='black')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=6)
ax_style(ax, title='Coupling Pair Deviations')
# Re-enable all spines for heatmap
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

fig.tight_layout(w_pad=2)
save_panel(fig, 'panel19_csci_cross_scale')

# ===========================================================================
# PANEL 20 -- Regime-Aware Recovery & Activity Coupling
# ===========================================================================
print("Panel 20: RARS Activity-Sleep Coupling")
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
panel_style(fig)

# Color by sleep score
if len(df_rars) > 0:
    score_norm = plt.Normalize(df_rars['sleep_score'].min(), df_rars['sleep_score'].max())
    score_cmap = cm.RdYlGn

    # Chart 1: Daily active calories vs next-night mean Rc
    ax = axes[0]
    sc = ax.scatter(df_rars['cal_active'], df_rars['mean_rc'],
                    c=df_rars['sleep_score'], cmap=score_cmap, norm=score_norm,
                    alpha=0.65, s=20, edgecolors='white', linewidths=0.3)
    if len(df_rars) > 3:
        slope, intercept, r, p, se = stats.linregress(df_rars['cal_active'], df_rars['mean_rc'])
        xline = np.linspace(df_rars['cal_active'].min(), df_rars['cal_active'].max(), 50)
        ax.plot(xline, slope * xline + intercept, 'k-', linewidth=1.2, alpha=0.5)
        ax.text(0.05, 0.95, f'r={r:.2f}', transform=ax.transAxes, fontsize=7, va='top')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Sleep Score', fontsize=6)
    ax_style(ax, title='Active Cal vs Mean $R_c$', xlabel='Active Calories', ylabel='Mean $R_c$')

    # Chart 2: Steps vs deep sleep Rc colored by sleep score
    ax = axes[1]
    sc = ax.scatter(df_rars['steps'], df_rars['deep_rc_mean'],
                    c=df_rars['sleep_score'], cmap=score_cmap, norm=score_norm,
                    alpha=0.65, s=20, edgecolors='white', linewidths=0.3)
    if len(df_rars) > 3:
        slope, intercept, r, p, se = stats.linregress(df_rars['steps'], df_rars['deep_rc_mean'])
        xline = np.linspace(df_rars['steps'].min(), df_rars['steps'].max(), 50)
        ax.plot(xline, slope * xline + intercept, 'k-', linewidth=1.2, alpha=0.5)
        ax.text(0.05, 0.95, f'r={r:.2f}', transform=ax.transAxes, fontsize=7, va='top')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Sleep Score', fontsize=6)
    ax_style(ax, title='Steps vs Deep $R_c$', xlabel='Steps', ylabel='Deep $R_c$')

    # Chart 3: 3D scatter (cal_active, deep_hrs, mean_Rc)
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    axes[2].set_visible(False)
    ax3d_style(ax3, title='3D: Cal x Deep Hrs x $R_c$')
    sc3 = ax3.scatter(df_rars['cal_active'], df_rars['deep_hrs'], df_rars['mean_rc'],
                      c=df_rars['sleep_score'], cmap=score_cmap, norm=score_norm,
                      alpha=0.55, s=15, edgecolors='white', linewidths=0.2)
    ax3.set_xlabel('Active Cal', fontsize=6, labelpad=2)
    ax3.set_ylabel('Deep Hrs', fontsize=6, labelpad=2)
    ax3.set_zlabel('Mean $R_c$', fontsize=6, labelpad=2)
    ax3.view_init(elev=22, azim=225)

    # Chart 4: Dual-axis time series of nightly mean Rc and activity score
    ax = axes[3]
    n_pts = len(df_rars)
    nights = np.arange(n_pts)
    color_rc = '#1565c0'
    color_act = '#e65100'
    ax.plot(nights, df_rars['mean_rc'].values, '-o', color=color_rc,
            markersize=3, linewidth=1, alpha=0.8, label='Mean $R_c$')
    ax.set_ylabel('Mean $R_c$', fontsize=7, color=color_rc)
    ax.tick_params(axis='y', labelcolor=color_rc, labelsize=7)
    ax_style(ax, title='Nightly $R_c$ & Activity Score', xlabel='Night Index')

    ax2 = ax.twinx()
    ax2.plot(nights, df_rars['activity_score'].values, '-s', color=color_act,
             markersize=3, linewidth=1, alpha=0.8, label='Activity Score')
    ax2.set_ylabel('Activity Score', fontsize=7, color=color_act)
    ax2.tick_params(axis='y', labelcolor=color_act, labelsize=7)
    ax2.spines['top'].set_visible(False)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, framealpha=0.7, loc='lower left')

else:
    for i in range(4):
        axes[i].text(0.5, 0.5, 'No RARS data', transform=axes[i].transAxes,
                     ha='center', va='center', fontsize=10)
        ax_style(axes[i])

fig.tight_layout(w_pad=2)
save_panel(fig, 'panel20_rars_activity_coupling')

print("\nAll sensor disambiguation panels generated.")
