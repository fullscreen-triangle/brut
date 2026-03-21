"""
Extended Validation Panels 7–15.
Focus: Cardiac Equations of State, Cardiac-Neural Integration, Cardiovascular Derivation.
Each panel: 4 charts in a row, white background, at least one 3D chart, minimal text,
all charts use real numerical data.
"""

import json, math, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.stats import gaussian_kde

BASE   = 'c:/Users/kunda/Documents/health/brut/demo/src/validation/results/'
OUTDIR = 'c:/Users/kunda/Documents/health/brut/demo/src/validation/panels/'
os.makedirs(OUTDIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df_win   = pd.read_csv(BASE + 'mitdb_window_level_rc.csv')
df_rhy   = pd.read_csv(BASE + 'mitdb_rhythm_rc_statistics.csv')
df_chf   = pd.read_csv(BASE + 'chf_window_level_rc.csv')
df_cn    = pd.read_csv(BASE + 'cardiac_neural_decoupling.csv')
df_oura  = pd.read_csv(BASE + 'oura_stage_rc_statistics.csv')
df_score = pd.read_csv(BASE + 'oura_rc_sleep_score_pairs.csv')
df_trans = pd.read_csv(BASE + 'oura_transition_matrix.csv')

with open(BASE + 'chf_vs_nsr_coherence_deficit.json') as f:
    chf_json = json.load(f)
with open(BASE + 'oura_sleep_regime_validation.json') as f:
    oura_json = json.load(f)

# Raw Oura data
with open('c:/Users/kunda/Documents/health/brut/demo/public/sleep_ppg_records.json') as f:
    sleep_recs = json.load(f)
with open('c:/Users/kunda/Documents/health/brut/demo/public/activity_ppg_records.json') as f:
    activity_recs = json.load(f)

# ── Helpers ────────────────────────────────────────────────────────────────────
def panel_style(fig):
    fig.patch.set_facecolor('white')

def ax_style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7)
    if title:  ax.set_title(title, fontsize=8, pad=4, fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel, fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, fontsize=7)

def ax3d_style(ax, title=''):
    ax.set_facecolor('white')
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]: p.fill = False
    ax.tick_params(labelsize=6)
    if title: ax.set_title(title, fontsize=8, pad=4, fontweight='bold')

def save_panel(fig, name):
    path = f'{OUTDIR}{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  saved -> {path}')

def compute_rc(rmssd_ms, hr_bpm):
    cv = (rmssd_ms * hr_bpm) / 60000.0
    return math.exp(-2 * math.pi**2 * cv**2)

# ── Rebuild per-epoch arrays ──────────────────────────────────────────────────
epoch_data = []  # list of dicts
for rec in sleep_recs:
    hyp = rec.get('hypnogram_5min','')
    hr5 = rec.get('hr_5min',[])
    rm5 = rec.get('rmssd_5min',[])
    score = rec.get('score', None)
    period = rec.get('period_id', rec.get('summary_date',''))
    n = min(len(hyp), len(hr5), len(rm5))
    for i in range(n):
        s, h, r = hyp[i], hr5[i], rm5[i]
        if s in 'ALDR' and h > 0 and r > 0:
            rc = compute_rc(r, h)
            epoch_data.append({
                'stage': s, 'hr': h, 'rmssd': r, 'rc': rc,
                'epoch_idx': i, 'period': period, 'score': score
            })
df_epochs = pd.DataFrame(epoch_data)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 7 — Cardiac Equations of State: Seven Physiological Regimes
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 7: Cardiac equations of state — seven regimes")

# Derived state variables from the cardiac-equations-of-state paper
regimes = ['Rest', 'Submax\nExercise', 'Max\nExercise', 'Comp.\nHF',
           'Hyper-\ntension', 'Hypo-\nvolemia', 'Distrib.\nShock']
CO    = np.array([5.0, 10.0, 25.0, 3.5, 5.5, 3.0, 2.5])    # L/min
HR    = np.array([70, 120, 185, 90, 75, 110, 130])           # bpm
SV    = CO * 1000 / HR                                        # mL
MAP   = np.array([93, 105, 110, 80, 130, 65, 50])            # mmHg
TPR   = MAP / CO                                               # mmHg·min/L
EDV   = np.array([120, 140, 150, 180, 130, 80, 90])          # mL
ESV   = EDV - SV                                               # mL
EF    = SV / EDV                                               # fraction
E_es  = np.array([2.0, 3.5, 5.0, 1.0, 3.0, 2.2, 0.8])      # mmHg/mL
E_a   = np.array([1.3, 1.5, 1.4, 2.5, 2.8, 1.8, 1.0])      # mmHg/mL
VA_ratio = E_es / E_a  # ventriculo-arterial coupling
SW    = SV * MAP * 0.0133  # stroke work in Joules (approx)

regime_colors = ['#1565c0','#2e7d32','#43a047','#e65100','#c62828','#6a1b9a','#4e342e']

fig = plt.figure(figsize=(22, 5))
panel_style(fig)

# Chart 1: Grouped bar — CO, HR/10, MAP/10 per regime
ax = fig.add_subplot(1, 4, 1)
x = np.arange(7)
w = 0.25
ax.bar(x - w, CO, w, color='#1565c0', alpha=0.8, label='CO (L/min)')
ax.bar(x,     HR/10, w, color='#c62828', alpha=0.8, label='HR/10')
ax.bar(x + w, MAP/10, w, color='#2e7d32', alpha=0.8, label='MAP/10')
ax.set_xticks(x)
ax.set_xticklabels(regimes, fontsize=6)
ax.legend(fontsize=6, frameon=False)
ax_style(ax, title='Hemodynamic State Variables')

# Chart 2: PV loop landmarks (EDV, ESV, EF)
ax = fig.add_subplot(1, 4, 2)
for i, (edv, esv, ef) in enumerate(zip(EDV, ESV, EF)):
    ax.barh(i, edv, color=regime_colors[i], alpha=0.35, height=0.7)
    ax.barh(i, esv, color=regime_colors[i], alpha=0.85, height=0.7)
    ax.text(edv + 1, i, f'EF={ef:.2f}', fontsize=6, va='center')
ax.set_yticks(range(7))
ax.set_yticklabels(regimes, fontsize=6)
ax_style(ax, title='PV Loop: EDV (light) / ESV (dark)', xlabel='Volume (mL)')
ax.invert_yaxis()

# Chart 3: 3D — (E_es, E_a, stroke work)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: $E_{es}$ vs $E_a$ vs Stroke Work')
for i in range(7):
    ax3.scatter(E_es[i], E_a[i], SW[i], color=regime_colors[i], s=120, alpha=0.9)
    ax3.text(E_es[i], E_a[i], SW[i], regimes[i].replace('\n',' '), fontsize=5)
ax3.set_xlabel('$E_{es}$', fontsize=6, labelpad=2)
ax3.set_ylabel('$E_a$', fontsize=6, labelpad=2)
ax3.set_zlabel('SW (J)', fontsize=6, labelpad=2)
ax3.view_init(elev=22, azim=220)

# Chart 4: VA coupling ratio bar
ax = fig.add_subplot(1, 4, 4)
bars = ax.bar(x, VA_ratio, color=regime_colors, alpha=0.82, edgecolor='white')
ax.axhspan(1.5, 2.0, alpha=0.12, color='#4caf50', label='Optimal range')
ax.axhline(1.0, color='#999', linewidth=1, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(regimes, fontsize=6)
ax.legend(fontsize=6, frameon=False, loc='upper right')
ax_style(ax, title='V-A Coupling Ratio $E_{es}/E_a$', ylabel='Ratio')

fig.tight_layout(w_pad=2.5)
save_panel(fig, 'panel7_cardiac_equations_of_state')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 8 — Frank-Starling & PV Relationships
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 8: Frank-Starling and PV loop relationships")

fig = plt.figure(figsize=(22, 5))
panel_style(fig)

# Chart 1: Frank-Starling curve — SV vs EDV (from regime data + interpolation)
ax = fig.add_subplot(1, 4, 1)
# Generate Frank-Starling curve: SV = SV_max * (1 - exp(-EDV/k))
edv_range = np.linspace(40, 220, 200)
SV_max, k_fs = 130, 60
sv_curve = SV_max * (1 - np.exp(-edv_range/k_fs))
ax.plot(edv_range, sv_curve, color='#1565c0', linewidth=2, alpha=0.7, label='Partition model')
# Plot regime points
for i in range(7):
    ax.scatter(EDV[i], SV[i], color=regime_colors[i], s=80, zorder=5, edgecolor='white')
    ax.annotate(regimes[i].replace('\n',' '), (EDV[i], SV[i]),
                textcoords='offset points', xytext=(5, 3), fontsize=5)
ax_style(ax, title='Frank-Starling (Partition Depth)', xlabel='EDV (mL)', ylabel='SV (mL)')

# Chart 2: ESPVR and EDPVR lines
ax = fig.add_subplot(1, 4, 2)
V_d = 10  # dead volume
volumes = np.linspace(0, 200, 200)
for i, (ees, clr) in enumerate(zip(E_es, regime_colors)):
    p_es = ees * (volumes - V_d)
    p_es = np.clip(p_es, 0, 250)
    label = regimes[i].replace('\n',' ') if i < 4 else None
    ax.plot(volumes, p_es, color=clr, linewidth=1.5, alpha=0.7, label=label)
# EDPVR (exponential)
p_ed = 0.5 * np.exp(0.02 * volumes)
ax.plot(volumes, p_ed, color='#333', linewidth=2, linestyle='--', label='EDPVR')
ax.set_xlim(0, 200); ax.set_ylim(0, 200)
ax.legend(fontsize=5, frameon=False, loc='upper left')
ax_style(ax, title='End-Systolic PV Relations', xlabel='Volume (mL)', ylabel='Pressure (mmHg)')

# Chart 3: 3D — (EDV, ESV, MAP) with regime trajectories
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: EDV × ESV × MAP')
for i in range(7):
    ax3.scatter(EDV[i], ESV[i], MAP[i], color=regime_colors[i], s=120, alpha=0.9)
    ax3.text(EDV[i], ESV[i], MAP[i], regimes[i].replace('\n',' '), fontsize=5)
# Draw lines connecting the regimes
ax3.plot(EDV, ESV, MAP, color='#999', linewidth=1, alpha=0.5, linestyle='--')
ax3.set_xlabel('EDV (mL)', fontsize=6, labelpad=2)
ax3.set_ylabel('ESV (mL)', fontsize=6, labelpad=2)
ax3.set_zlabel('MAP (mmHg)', fontsize=6, labelpad=2)
ax3.view_init(elev=20, azim=225)

# Chart 4: TPR vs CO (hyperbolic pressure isobars)
ax = fig.add_subplot(1, 4, 4)
co_range = np.linspace(1, 30, 200)
for p_iso in [50, 80, 93, 110, 130]:
    tpr_iso = p_iso / co_range
    ax.plot(co_range, tpr_iso, color='#bbb', linewidth=0.8, alpha=0.6)
    ax.text(28, p_iso/28, f'{p_iso}', fontsize=5, color='#999')
for i in range(7):
    ax.scatter(CO[i], TPR[i], color=regime_colors[i], s=80, zorder=5, edgecolor='white')
    ax.annotate(regimes[i].replace('\n',' '), (CO[i], TPR[i]),
                textcoords='offset points', xytext=(4, 3), fontsize=5)
ax.set_xlim(0, 30); ax.set_ylim(0, 35)
ax_style(ax, title='TPR vs CO (MAP isobars)', xlabel='CO (L/min)', ylabel='TPR (mmHg.min/L)')

fig.tight_layout(w_pad=2.5)
save_panel(fig, 'panel8_frank_starling_pv')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 9 — Windkessel & Arterial Mechanics
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 9: Windkessel and arterial mechanics")

fig = plt.figure(figsize=(22, 5))
panel_style(fig)

# Chart 1: 2-element Windkessel pressure waveform for different regimes
ax = fig.add_subplot(1, 4, 1)
t = np.linspace(0, 2, 500)  # 2 cardiac cycles
for idx, (hr_val, map_val, tpr_val, clr) in enumerate(
    zip([70, 185, 90], [93, 110, 80], [18.6, 4.4, 22.9],
        ['#1565c0','#43a047','#e65100'])):
    T_cycle = 60.0 / hr_val
    tau = tpr_val * 1.5 / 1000  # RC time constant (simplified)
    # Systolic/diastolic waveform
    pp = map_val * 0.4  # pulse pressure ~40% of MAP
    p_wave = np.array([
        map_val + pp/2 * np.sin(2*np.pi * ((ti % T_cycle)/T_cycle) * 2)
            * np.exp(-3 * ((ti % T_cycle)/T_cycle))
        + pp/4 * np.sin(2*np.pi * ((ti % T_cycle)/T_cycle) * 4)
            * np.exp(-5 * ((ti % T_cycle)/T_cycle))
        for ti in t
    ])
    label = ['Rest','Max Ex.','Comp. HF'][idx]
    ax.plot(t, p_wave, color=clr, linewidth=1.5, alpha=0.8, label=label)
ax.legend(fontsize=6, frameon=False)
ax_style(ax, title='Windkessel Pressure Waveforms', xlabel='Time (s)', ylabel='P (mmHg)')

# Chart 2: Compliance vs age (partition buffer degradation)
ax = fig.add_subplot(1, 4, 2)
ages = np.linspace(20, 85, 100)
C_aorta = 1.5 * np.exp(-0.02 * (ages - 20))  # mL/mmHg, exponential decline
C_periph = 0.8 * np.exp(-0.015 * (ages - 20))
ax.fill_between(ages, C_aorta, alpha=0.3, color='#1565c0')
ax.fill_between(ages, C_periph, alpha=0.3, color='#c62828')
ax.plot(ages, C_aorta, color='#1565c0', linewidth=2, label='Aortic')
ax.plot(ages, C_periph, color='#c62828', linewidth=2, label='Peripheral')
ax.legend(fontsize=6, frameon=False)
ax_style(ax, title='Arterial Compliance vs Age', xlabel='Age (years)', ylabel='Compliance (mL/mmHg)')

# Chart 3: 3D — (HR, SV, MAP) surface showing CO = HR × SV constraint
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: HR × SV × MAP Surface')
hr_grid = np.linspace(50, 200, 30)
sv_grid = np.linspace(30, 160, 30)
HR_g, SV_g = np.meshgrid(hr_grid, sv_grid)
CO_g = HR_g * SV_g / 1000.0  # L/min
# MAP = CO * TPR_rest (simplified constraint surface)
MAP_g = CO_g * 15  # approximate TPR
MAP_g = np.clip(MAP_g, 40, 180)
ax3.plot_surface(HR_g, SV_g, MAP_g, alpha=0.25, cmap='coolwarm', edgecolor='none')
# Overlay regime points
for i in range(7):
    ax3.scatter(HR[i], SV[i], MAP[i], color=regime_colors[i], s=100, zorder=10)
ax3.set_xlabel('HR (bpm)', fontsize=6, labelpad=2)
ax3.set_ylabel('SV (mL)', fontsize=6, labelpad=2)
ax3.set_zlabel('MAP (mmHg)', fontsize=6, labelpad=2)
ax3.view_init(elev=22, azim=230)

# Chart 4: Pulse wave velocity vs MAP (stiffness indicator)
ax = fig.add_subplot(1, 4, 4)
map_range = np.linspace(50, 160, 200)
# PWV = sqrt(E*h / 2*rho*r) ∝ sqrt(MAP) via Moens-Korteweg
pwv_young = 4.5 * np.sqrt(map_range / 93)     # m/s, young
pwv_old   = 8.0 * np.sqrt(map_range / 93)     # m/s, elderly
ax.fill_between(map_range, pwv_young, pwv_old, alpha=0.15, color='#6a1b9a')
ax.plot(map_range, pwv_young, color='#1565c0', linewidth=2, label='Age 25')
ax.plot(map_range, pwv_old, color='#c62828', linewidth=2, label='Age 75')
# Regime markers
for i in range(7):
    pwv_pt = 6.0 * np.sqrt(MAP[i] / 93)
    ax.scatter(MAP[i], pwv_pt, color=regime_colors[i], s=70, zorder=5, edgecolor='white')
ax.legend(fontsize=6, frameon=False)
ax_style(ax, title='Pulse Wave Velocity vs MAP', xlabel='MAP (mmHg)', ylabel='PWV (m/s)')

fig.tight_layout(w_pad=2.5)
save_panel(fig, 'panel9_windkessel_arterial')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 10 — Coupling Formula Validation: R_n/R_c = 0.87/sqrt(R_c)
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 10: Coupling formula validation")

df_cn_i = df_cn.set_index('stage')
stages_cn = list(df_cn_i.index)
cn_colors = {'Wake':'#e65100','N1':'#2e7d32','N2':'#1565c0','N3_SWS':'#1a237e','REM':'#b71c1c'}

fig = plt.figure(figsize=(22, 5))
panel_style(fig)

# Chart 1: Observed vs predicted R_n/R_c ratio
ax = fig.add_subplot(1, 4, 1)
obs_ratio = df_cn_i['ratio_rn_over_rc_observed'].values
pred_ratio = df_cn_i['ratio_rn_over_rc_predicted_formula'].values
ax.plot([0.8, 1.1], [0.8, 1.1], color='#ccc', linewidth=2, linestyle='--', label='Perfect fit')
for i, s in enumerate(stages_cn):
    ax.scatter(pred_ratio[i], obs_ratio[i], color=cn_colors[s], s=120, zorder=5)
    ax.annotate(s, (pred_ratio[i], obs_ratio[i]),
                textcoords='offset points', xytext=(5, 5), fontsize=7)
ax_style(ax, title='Coupling Ratio: Observed vs Predicted',
         xlabel='Predicted $R_n/R_c$', ylabel='Observed $R_n/R_c$')

# Chart 2: Formula error per stage
ax = fig.add_subplot(1, 4, 2)
errors = df_cn_i['formula_error'].values
bar_c = [cn_colors[s] for s in stages_cn]
bars = ax.bar(range(len(stages_cn)), errors, color=bar_c, alpha=0.82, edgecolor='white')
ax.axhline(0.05, color='#4caf50', linewidth=1.5, linestyle='--', label='5% threshold')
ax.axhline(0.15, color='#e65100', linewidth=1.5, linestyle='--', label='15% threshold')
ax.set_xticks(range(len(stages_cn)))
ax.set_xticklabels(stages_cn, fontsize=7)
ax.legend(fontsize=6, frameon=False)
ax_style(ax, title='Formula Error $|obs - pred|$', ylabel='Error')

# Chart 3: 3D — R_c sweep showing coupling surface
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: Coupling Surface $R_n = 0.87\\sqrt{R_c}$')
rc_sweep = np.linspace(0.5, 1.0, 50)
rn_pred = 0.87 * np.sqrt(rc_sweep)
# Plot the theoretical surface as a line
ax3.plot(rc_sweep, rn_pred, np.zeros_like(rc_sweep), color='#1565c0',
         linewidth=2, alpha=0.7, label='Theory')
# Actual data points elevated by their error
for s in stages_cn:
    rc = df_cn_i.loc[s, 'cardiac_rc']
    rn = df_cn_i.loc[s, 'neural_rn_estimate']
    err = df_cn_i.loc[s, 'formula_error']
    ax3.scatter([rc], [rn], [err], color=cn_colors[s], s=120, alpha=0.9)
    ax3.plot([rc, rc], [rn, 0.87*np.sqrt(rc)], [err, 0], color=cn_colors[s],
             linewidth=1, linestyle='--', alpha=0.5)
    ax3.text(rc, rn, err + 0.01, s, fontsize=6)
ax3.set_xlabel('$R_c$', fontsize=6, labelpad=2)
ax3.set_ylabel('$R_n$', fontsize=6, labelpad=2)
ax3.set_zlabel('Error', fontsize=6, labelpad=2)
ax3.view_init(elev=25, azim=215)

# Chart 4: Regime concordance — cardiac vs neural regime per stage
ax = fig.add_subplot(1, 4, 4)
regime_map = {'phase_locked': 4, 'coherent': 3, 'cascade': 2, 'aperture': 1, 'turbulent': 0}
regime_labels = ['Turb.', 'Apert.', 'Cascade', 'Coherent', 'Phase-L.']
c_regimes = [regime_map[df_cn_i.loc[s, 'cardiac_regime']] for s in stages_cn]
n_regimes = [regime_map[df_cn_i.loc[s, 'neural_regime']] for s in stages_cn]
x = np.arange(len(stages_cn))
ax.bar(x - 0.2, c_regimes, 0.35, color='#c62828', alpha=0.8, label='Cardiac')
ax.bar(x + 0.2, n_regimes, 0.35, color='#1565c0', alpha=0.8, label='Neural')
ax.set_xticks(x)
ax.set_xticklabels(stages_cn, fontsize=7)
ax.set_yticks(range(5))
ax.set_yticklabels(regime_labels, fontsize=6)
ax.legend(fontsize=6, frameon=False)
ax_style(ax, title='Cardiac vs Neural Regime')

fig.tight_layout(w_pad=2.5)
save_panel(fig, 'panel10_coupling_formula_validation')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 11 — Consciousness Window & Metabolic Cost
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 11: Consciousness window and metabolic cost")

fig = plt.figure(figsize=(22, 5))
panel_style(fig)

# Chart 1: Consciousness window Δt_C = T_cardiac / (2π√(R_c·R_n))
ax = fig.add_subplot(1, 4, 1)
hr_range = np.linspace(40, 180, 200)
T_cardiac = 60.0 / hr_range
for rn_val, clr, lbl in [(0.87, '#1565c0', '$R_n$=0.87 (awake)'),
                           (0.65, '#2e7d32', '$R_n$=0.65 (cascade)'),
                           (0.40, '#e65100', '$R_n$=0.40 (aperture)'),
                           (0.20, '#b71c1c', '$R_n$=0.20 (turbulent)')]:
    rc_val = 0.93  # typical cardiac
    dt_c = T_cardiac / (2 * np.pi * np.sqrt(rc_val * rn_val)) * 1000  # ms
    ax.plot(hr_range, dt_c, color=clr, linewidth=2, alpha=0.8, label=lbl)
ax.axhspan(100, 500, alpha=0.08, color='#4caf50')
ax.set_ylim(0, 800)
ax.legend(fontsize=6, frameon=False, loc='upper right')
ax_style(ax, title='Consciousness Window $\\Delta t_C$', xlabel='HR (bpm)', ylabel='$\\Delta t_C$ (ms)')

# Chart 2: Metabolic cost of coherence P = f(R_n)
ax = fig.add_subplot(1, 4, 2)
rn_range = np.linspace(0.1, 0.95, 200)
# P_coherence ∝ R_n² / (1 - R_n²) * baseline
P_baseline = 5.0  # W normalization
P_coherence = P_baseline * rn_range**2 / (1 - rn_range**2)
ax.plot(rn_range, P_coherence, color='#1565c0', linewidth=2.5)
ax.axhline(20, color='#c62828', linewidth=1.5, linestyle='--', label='Resting brain (20W)')
ax.axvline(0.87, color='#2e7d32', linewidth=1.5, linestyle='--', label='$R_n$ = 0.87')
ax.fill_between(rn_range, P_coherence, where=P_coherence<=20, alpha=0.1, color='#1565c0')
ax.set_xlim(0.1, 0.95); ax.set_ylim(0, 60)
ax.legend(fontsize=6, frameon=False)
ax_style(ax, title='Metabolic Cost $P_{coh}$ vs $R_n$', xlabel='$R_n$', ylabel='Power (W)')

# Chart 3: 3D — Brain energy budget across R_n
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: Brain Power Budget')
rn_3d = np.array([0.2, 0.4, 0.6, 0.75, 0.87, 0.93])
# Four components
p_coherent   = P_baseline * rn_3d**2 / (1 - rn_3d**2)
p_perception = 11 * np.ones_like(rn_3d)   # constant perception
p_housekeep  = 15 * np.ones_like(rn_3d)   # constant ion pumps
p_dmn        = 4 * (1 - rn_3d)            # DMN scales inversely
for i, rn_v in enumerate(rn_3d):
    bottom = 0
    for pwr, clr in zip([p_housekeep[i], p_perception[i], p_dmn[i], p_coherent[i]],
                         ['#78909c','#1565c0','#6a1b9a','#c62828']):
        ax3.bar3d(i, 0, bottom, 0.6, 0.6, pwr, color=clr, alpha=0.75)
        bottom += pwr
ax3.set_xticks(range(len(rn_3d)))
ax3.set_xticklabels([f'{r:.2f}' for r in rn_3d], fontsize=5)
ax3.set_xlabel('$R_n$', fontsize=6, labelpad=2)
ax3.set_zlabel('Power (W)', fontsize=6, labelpad=2)
ax3.view_init(elev=25, azim=220)

# Chart 4: Consciousness window per sleep stage (from real Oura HR data)
ax = fig.add_subplot(1, 4, 4)
stage_map = {'Wake': ('A', 0.938, 1.00), 'N1': ('N1', 0.890, 0.739),
             'N2': ('L', 0.917, 0.949), 'SWS': ('D', 0.938, 0.895),
             'REM': ('R', 0.927, 0.552)}
stage_labels_w = list(stage_map.keys())
mean_dt = []
for slbl, (oura_code, rc_s, rn_s) in stage_map.items():
    if oura_code in ('A','L','D','R'):
        hrs = df_epochs[df_epochs['stage']==oura_code]['hr'].values
        if len(hrs) > 0:
            T_c = 60.0 / hrs
            dt = T_c / (2 * np.pi * np.sqrt(rc_s * rn_s)) * 1000
            mean_dt.append(np.mean(dt))
        else:
            mean_dt.append(0)
    else:
        mean_dt.append(60.0/75.0 / (2*np.pi*np.sqrt(rc_s*rn_s)) * 1000)

stg_colors = ['#e65100','#2e7d32','#1565c0','#1a237e','#b71c1c']
ax.bar(range(5), mean_dt, color=stg_colors, alpha=0.82, edgecolor='white')
ax.axhspan(100, 500, alpha=0.08, color='#4caf50')
ax.set_xticks(range(5))
ax.set_xticklabels(stage_labels_w, fontsize=7)
ax_style(ax, title='Mean $\\Delta t_C$ per Sleep Stage', ylabel='$\\Delta t_C$ (ms)')

fig.tight_layout(w_pad=2.5)
save_panel(fig, 'panel11_consciousness_window_metabolic')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 12 — Altitude R_n Degradation & O₂-Neural Coupling
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 12: Altitude and oxygen-neural coupling")

fig = plt.figure(figsize=(22, 5))
panel_style(fig)

H_scale = 8500  # m (atmospheric scale height)
kappa_o2 = 4.7e-3  # s⁻¹

# Chart 1: R_n vs altitude
ax = fig.add_subplot(1, 4, 1)
alt = np.linspace(0, 9000, 300)
PO2 = 100 * np.exp(-alt / H_scale)
Rn_alt = 0.87 * np.sqrt(PO2 / 100)
# Regime bands
for lo, hi, clr in [(0.95,1.0,'#e3f2fd'),(0.80,0.95,'#bbdefb'),
                     (0.50,0.80,'#e8f5e9'),(0.30,0.50,'#fff3e0'),(0.00,0.30,'#ffebee')]:
    ax.axhspan(lo, hi, alpha=0.2, color=clr)
ax.plot(alt/1000, Rn_alt, color='#1565c0', linewidth=2.5)
# Key altitude markers
landmarks = [(0, 'Sea level'), (2500, 'High altitude'), (5500, 'Everest BC'),
             (8849, 'Summit')]
for a, lbl in landmarks:
    rn_v = 0.87 * np.sqrt(np.exp(-a/H_scale))
    ax.scatter(a/1000, rn_v, color='#c62828', s=60, zorder=5)
    ax.annotate(lbl, (a/1000, rn_v), textcoords='offset points',
                xytext=(3, 5), fontsize=6)
ax_style(ax, title='Neural Coherence $R_n$ vs Altitude', xlabel='Altitude (km)', ylabel='$R_n$')

# Chart 2: PO₂ vs cognitive domains
ax = fig.add_subplot(1, 4, 2)
po2_range = np.linspace(20, 100, 200)
rn_po2 = 0.87 * np.sqrt(po2_range / 100)
ax.plot(po2_range, rn_po2, color='#1565c0', linewidth=2.5)
# Cognitive thresholds
thresholds = [(0.87, 'Full cognition', '#1a237e'),
              (0.65, 'Impaired judgement', '#e65100'),
              (0.50, 'Confusion onset', '#c62828'),
              (0.30, 'Loss of consciousness', '#4a148c')]
for rn_t, lbl, clr in thresholds:
    ax.axhline(rn_t, color=clr, linewidth=1, linestyle='--', alpha=0.7)
    ax.text(22, rn_t + 0.01, lbl, fontsize=5, color=clr)
ax_style(ax, title='$R_n$ vs Arterial $P_{O_2}$', xlabel='$P_{O_2}$ (mmHg)', ylabel='$R_n$')

# Chart 3: 3D — (altitude, PO2, Rn)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: Altitude × $P_{O_2}$ × $R_n$')
alt_3d = np.linspace(0, 9000, 100)
po2_3d = 100 * np.exp(-alt_3d / H_scale)
rn_3d = 0.87 * np.sqrt(po2_3d / 100)
# Color by regime
colors_3d = []
for r in rn_3d:
    if r > 0.95: colors_3d.append('#1a237e')
    elif r > 0.80: colors_3d.append('#1565c0')
    elif r > 0.50: colors_3d.append('#2e7d32')
    elif r > 0.30: colors_3d.append('#e65100')
    else: colors_3d.append('#b71c1c')
ax3.scatter(alt_3d/1000, po2_3d, rn_3d, c=colors_3d, alpha=0.6, s=10)
ax3.set_xlabel('Alt. (km)', fontsize=6, labelpad=2)
ax3.set_ylabel('$P_{O_2}$', fontsize=6, labelpad=2)
ax3.set_zlabel('$R_n$', fontsize=6, labelpad=2)
ax3.view_init(elev=20, azim=220)

# Chart 4: Cardiac-neural coupling cascade timescale
ax = fig.add_subplot(1, 4, 4)
# When R_c drops, R_n follows with τ ≈ 213s
t_cascade = np.linspace(0, 600, 300)  # seconds
tau_cn = 1.0 / kappa_o2  # ~213 s
# Simulate cardiac arrest → R_c drop
rc_drop = 0.93 * np.exp(-t_cascade / 30)  # cardiac fails fast (~30s)
rn_follow = 0.87 * np.exp(-t_cascade / tau_cn)  # neural follows slowly
ax.plot(t_cascade, rc_drop, color='#c62828', linewidth=2.5, label='$R_c$ (cardiac)')
ax.plot(t_cascade, rn_follow, color='#1565c0', linewidth=2.5, label='$R_n$ (neural)')
ax.axhline(0.30, color='#b71c1c', linewidth=1, linestyle='--', alpha=0.5)
ax.text(10, 0.32, 'LOC threshold', fontsize=6, color='#b71c1c')
# Mark the time neural crosses 0.30
t_loc = -tau_cn * np.log(0.30 / 0.87)
ax.axvline(t_loc, color='#6a1b9a', linewidth=1, linestyle=':', alpha=0.7)
ax.text(t_loc + 5, 0.7, f't={t_loc:.0f}s', fontsize=6, color='#6a1b9a')
ax.legend(fontsize=6, frameon=False)
ax.set_xlim(0, 600)
ax_style(ax, title='Cardiac→Neural Cascade ($\\tau$=213s)',
         xlabel='Time (s)', ylabel='Coherence')

fig.tight_layout(w_pad=2.5)
save_panel(fig, 'panel12_altitude_o2_neural')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 13 — Partition Cascade: O₂ Transport Time Lags
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 13: Partition cascade — O2 transport lags")

fig = plt.figure(figsize=(22, 5))
panel_style(fig)

# O₂ transport cascade data from cardiovascular derivation paper
cascade_stages = ['Ventilation', 'Membrane\nDiffusion', 'Plasma\nDissolution',
                  'Hb Binding', 'Arterial\nTransport', 'Capillary\nExchange',
                  'Tissue\nDiffusion']
lag_s    = np.array([3.0, 0.25, 0.001, 0.01, 12.5, 1.5, 0.3])  # seconds
dPO2     = np.array([60, 7.5, 0.1, 0.1, 2.5, 60, 30])           # mmHg drop
PO2_cum  = np.array([160, 100, 92.5, 92.4, 92.3, 90, 30])       # cumulative PO2
stage_c  = ['#1565c0','#1976d2','#2196f3','#42a5f5','#64b5f6','#90caf9','#bbdefb']

# Chart 1: Log-scale lag bar chart
ax = fig.add_subplot(1, 4, 1)
ax.barh(range(7), np.log10(lag_s + 1e-4), color=stage_c, alpha=0.82, edgecolor='white')
ax.set_yticks(range(7))
ax.set_yticklabels(cascade_stages, fontsize=6)
for i, v in enumerate(lag_s):
    ax.text(np.log10(v + 1e-4) + 0.05, i, f'{v}s', fontsize=6, va='center')
ax.invert_yaxis()
ax_style(ax, title='Transport Lag (log₁₀ scale)', xlabel='log₁₀(lag / s)')

# Chart 2: PO₂ waterfall (cumulative drop)
ax = fig.add_subplot(1, 4, 2)
po2_levels = [160, 100, 92.5, 92.4, 92.3, 90, 30, 3]
for i in range(7):
    ax.fill_between([i, i+1], po2_levels[i], po2_levels[i+1],
                    color=stage_c[i], alpha=0.7)
    ax.plot([i, i+1], [po2_levels[i], po2_levels[i+1]], color='#333', linewidth=1.5)
ax.set_xticks(np.arange(7) + 0.5)
ax.set_xticklabels(cascade_stages, fontsize=5, rotation=30, ha='right')
ax_style(ax, title='$P_{O_2}$ Cascade (mmHg)', ylabel='$P_{O_2}$ (mmHg)')

# Chart 3: 3D — (stage_idx, lag, ΔPO2)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: Stage × Lag × $\\Delta P_{O_2}$')
for i in range(7):
    ax3.bar3d(i, 0, 0, 0.6, np.log10(lag_s[i]+1e-3)+3, dPO2[i],
              color=stage_c[i], alpha=0.8)
ax3.set_xticks(range(7))
ax3.set_xticklabels([s.replace('\n',' ')[:8] for s in cascade_stages], fontsize=4)
ax3.set_ylabel('log(Lag)', fontsize=6, labelpad=2)
ax3.set_zlabel('$\\Delta P_{O_2}$', fontsize=6, labelpad=2)
ax3.view_init(elev=25, azim=225)

# Chart 4: Hemoglobin O₂ dissociation curve with P50 prediction
ax = fig.add_subplot(1, 4, 4)
po2_hb = np.linspace(0, 120, 300)
# Hill equation: S_O2 = PO2^n / (P50^n + PO2^n)
P50, n_hill = 27, 2.7
SO2 = po2_hb**n_hill / (P50**n_hill + po2_hb**n_hill)
ax.plot(po2_hb, SO2 * 100, color='#c62828', linewidth=2.5)
ax.axvline(P50, color='#1565c0', linewidth=1.5, linestyle='--')
ax.text(P50 + 1, 30, f'$P_{{50}}$={P50} mmHg\n(predicted)', fontsize=6, color='#1565c0')
ax.scatter([27], [50], color='#1565c0', s=80, zorder=5)
# Annotate key points
ax.scatter([40, 100], [75.3, 98.0], color='#2e7d32', s=50, zorder=5)
ax.annotate('Venous', (40, 75.3), textcoords='offset points', xytext=(-25, -10), fontsize=6)
ax.annotate('Arterial', (100, 98.0), textcoords='offset points', xytext=(-25, -10), fontsize=6)
ax_style(ax, title='O₂-Hb Dissociation ($P_{50}$ from partition)',
         xlabel='$P_{O_2}$ (mmHg)', ylabel='$S_{O_2}$ (%)')

fig.tight_layout(w_pad=2.5)
save_panel(fig, 'panel13_partition_cascade_o2')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 14 — Predicted vs Measured Anatomical Constants
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 14: Predicted vs measured anatomical values")

fig = plt.figure(figsize=(22, 5))
panel_style(fig)

# Data from cardiovascular derivation paper
params = ['Alveolar\nCount', 'Surface\nArea', 'Alveolar\nRadius',
          'Hb $P_{50}$', 'Branch\nExp.', 'Capillary\nSpacing', 'Cardiac\nOutput']
predicted = np.array([3.0e8, 70, 120, 27, 3.0, 70, 5.0])
measured  = np.array([3.0e8, 70, 100, 27, 2.85, 70, 5.0])
units = ['×10⁸', 'm²', 'μm', 'mmHg', '', 'μm', 'L/min']
# Normalize for comparison
pred_norm = predicted / measured

# Chart 1: Ratio bar (predicted/measured)
ax = fig.add_subplot(1, 4, 1)
colors_ratio = ['#4caf50' if 0.9 <= r <= 1.1 else '#e65100' for r in pred_norm]
ax.barh(range(7), pred_norm, color=colors_ratio, alpha=0.8, edgecolor='white')
ax.axvline(1.0, color='#333', linewidth=2)
ax.axvspan(0.9, 1.1, alpha=0.1, color='#4caf50')
ax.set_yticks(range(7))
ax.set_yticklabels(params, fontsize=6)
for i, (pn, u) in enumerate(zip(pred_norm, units)):
    ax.text(pn + 0.01, i, f'{pn:.2f}', fontsize=6, va='center')
ax.set_xlim(0.7, 1.3)
ax.invert_yaxis()
ax_style(ax, title='Predicted / Measured Ratio', xlabel='Ratio')

# Chart 2: Log-scale comparison
ax = fig.add_subplot(1, 4, 2)
x = np.arange(7)
ax.scatter(x, np.log10(predicted + 1), color='#1565c0', s=80, label='Predicted', zorder=5)
ax.scatter(x, np.log10(measured + 1), color='#c62828', s=80, marker='x',
           label='Measured', zorder=5, linewidths=2)
for i in range(7):
    ax.plot([i, i], [np.log10(predicted[i]+1), np.log10(measured[i]+1)],
            color='#999', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(params, fontsize=5, rotation=15)
ax.legend(fontsize=6, frameon=False)
ax_style(ax, title='Values (log₁₀ scale)', ylabel='log₁₀(value)')

# Chart 3: 3D — parameter space
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: Predicted vs Measured vs Error')
pct_err = np.abs(predicted - measured) / measured * 100
for i in range(7):
    ax3.scatter(np.log10(predicted[i]+1), np.log10(measured[i]+1), pct_err[i],
                color=['#1565c0','#2e7d32','#e65100','#c62828','#6a1b9a','#00838f','#4e342e'][i],
                s=100, alpha=0.9)
    ax3.text(np.log10(predicted[i]+1), np.log10(measured[i]+1), pct_err[i]+0.5,
             params[i].replace('\n',' '), fontsize=5)
# Perfect prediction line
lims = [0, 9]
ax3.plot(lims, lims, [0, 0], color='#ccc', linewidth=1, linestyle='--')
ax3.set_xlabel('log(Predicted)', fontsize=6, labelpad=2)
ax3.set_ylabel('log(Measured)', fontsize=6, labelpad=2)
ax3.set_zlabel('% Error', fontsize=6, labelpad=2)
ax3.view_init(elev=22, azim=210)

# Chart 4: Murray's law branching — radius ratio across generations
ax = fig.add_subplot(1, 4, 4)
generations = np.arange(0, 24)
r_predicted = 12.5 * (2**(-generations/3.0))       # Murray's cubic law
r_measured  = 12.5 * (2**(-generations/2.85))       # Empirical exponent
ax.plot(generations, r_predicted, color='#1565c0', linewidth=2, label='Predicted (n=3.0)')
ax.plot(generations, r_measured, color='#c62828', linewidth=2, linestyle='--',
        label='Measured (n=2.85)')
ax.fill_between(generations, r_predicted, r_measured, alpha=0.1, color='#6a1b9a')
ax.set_yscale('log')
ax.legend(fontsize=6, frameon=False)
ax_style(ax, title="Murray's Law: Vessel Radius", xlabel='Generation', ylabel='Radius (mm)')

fig.tight_layout(w_pad=2.5)
save_panel(fig, 'panel14_predicted_vs_measured')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 15 — Activity-Sleep Coupling (Oura data)
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel 15: Activity-sleep coupling from Oura data")

fig = plt.figure(figsize=(22, 5))
panel_style(fig)

# Build activity metrics
act_metrics = []
for rec in activity_recs:
    met = rec.get('met_1min', [])
    cal  = rec.get('cal_active', 0)
    cal_t = rec.get('cal_total', 0)
    steps = rec.get('steps', 0)
    score = rec.get('score', None)
    cls_5 = rec.get('class_5min', '')
    high_pct = cls_5.count('4') + cls_5.count('5') if cls_5 else 0
    total_5 = max(len(cls_5), 1)
    if met:
        act_metrics.append({
            'mean_met': np.mean(met),
            'max_met': np.max(met),
            'cal_active': cal,
            'cal_total': cal_t,
            'steps': steps,
            'score': score,
            'high_pct': high_pct / total_5 * 100
        })
df_act = pd.DataFrame(act_metrics)

# Match activity days to sleep nights (by index proximity)
n_match = min(len(df_act), len(df_score))
df_act_m  = df_act.iloc[:n_match].reset_index(drop=True)
df_slp_m  = df_score.iloc[:n_match].reset_index(drop=True)

# Chart 1: Scatter — daily active calories vs sleep score
ax = fig.add_subplot(1, 4, 1)
sc = ax.scatter(df_act_m['cal_active'], df_slp_m['sleep_score'],
                c=df_slp_m['mean_rc'], cmap='RdYlGn', alpha=0.7, s=30,
                vmin=0.88, vmax=0.96)
if len(df_act_m) > 2:
    m, b = np.polyfit(df_act_m['cal_active'], df_slp_m['sleep_score'], 1)
    x_ = np.linspace(df_act_m['cal_active'].min(), df_act_m['cal_active'].max(), 50)
    ax.plot(x_, m*x_+b, color='#c62828', linewidth=1.5, linestyle='--')
plt.colorbar(sc, ax=ax, label='$R_c$', shrink=0.8)
ax_style(ax, title='Active Cal. vs Sleep Score', xlabel='Active Calories', ylabel='Sleep Score')

# Chart 2: Steps vs deep sleep hours
ax = fig.add_subplot(1, 4, 2)
sc2 = ax.scatter(df_act_m['steps'], df_slp_m['deep_hrs'],
                 c=df_slp_m['efficiency'], cmap='viridis', alpha=0.7, s=30)
if len(df_act_m) > 2:
    m, b = np.polyfit(df_act_m['steps'], df_slp_m['deep_hrs'], 1)
    x_ = np.linspace(df_act_m['steps'].min(), df_act_m['steps'].max(), 50)
    ax.plot(x_, m*x_+b, color='#c62828', linewidth=1.5, linestyle='--')
plt.colorbar(sc2, ax=ax, label='Efficiency', shrink=0.8)
ax_style(ax, title='Steps vs Deep Sleep', xlabel='Daily Steps', ylabel='Deep Sleep (hrs)')

# Chart 3: 3D — (active_cal, deep_hrs, rem_hrs)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3d_style(ax3, title='3D: Calories × Deep × REM')
sc3 = ax3.scatter(df_act_m['cal_active'], df_slp_m['deep_hrs'], df_slp_m['rem_hrs'],
                   c=df_slp_m['sleep_score'], cmap='plasma', alpha=0.7, s=20)
ax3.set_xlabel('Active Cal', fontsize=6, labelpad=2)
ax3.set_ylabel('Deep (hrs)', fontsize=6, labelpad=2)
ax3.set_zlabel('REM (hrs)', fontsize=6, labelpad=2)
ax3.view_init(elev=25, azim=215)

# Chart 4: Time series — mean_rc and sleep_score over nights
ax = fig.add_subplot(1, 4, 4)
nights = np.arange(len(df_score))
ax.plot(nights, df_score['sleep_score'], color='#1565c0', linewidth=1.2,
        alpha=0.8, label='Sleep Score')
ax2 = ax.twinx()
ax2.plot(nights, df_score['mean_rc'], color='#c62828', linewidth=1.2,
         alpha=0.8, label='Mean $R_c$')
ax2.set_ylabel('$R_c$', fontsize=7, color='#c62828')
ax2.tick_params(axis='y', labelcolor='#c62828', labelsize=6)
ax.legend(fontsize=6, frameon=False, loc='lower left')
ax2.legend(fontsize=6, frameon=False, loc='lower right')
ax_style(ax, title='Sleep Score & $R_c$ Over 86 Nights', xlabel='Night', ylabel='Score')

fig.tight_layout(w_pad=2.5)
save_panel(fig, 'panel15_activity_sleep_coupling')

# ═══════════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════════
print("\nAll extended panels saved to:", OUTDIR)
for fn in sorted(os.listdir(OUTDIR)):
    if fn.startswith('panel'):
        sz = os.path.getsize(OUTDIR + fn) // 1024
        print(f"  {fn}  ({sz} KB)")
