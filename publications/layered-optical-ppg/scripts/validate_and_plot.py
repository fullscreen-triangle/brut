"""
Validation experiments and panel generation for the layered optical PPG paper.

Each panel is a 1x4 row of data-driven charts on a white background. At
least one of the four sub-panels in every panel is a 3D chart. No
conceptual schematics, no tables, no in-chart text blocks.

Requires: numpy, matplotlib (no other dependencies).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
VAL_DIR = ROOT / "validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Forward model (mirrors the TypeScript implementation)
# ---------------------------------------------------------------------------

ABS = {
    "mel_R": 0.10, "mel_G": 0.36, "mel_B": 0.84,
    "hb_R": 0.012, "hb_G": 0.112, "hb_B": 0.084,
    "hb_delta_G": 0.036,
    "D": 0.55,
    "spec": 0.04,
}


@dataclass
class State:
    melanin: float
    hb: float
    spo2: float
    vaso: float


def forward(s: State) -> tuple[float, float, float]:
    spec = ABS["spec"]
    rho2_R = np.exp(-ABS["mel_R"] * s.melanin)
    rho2_G = np.exp(-ABS["mel_G"] * s.melanin)
    rho2_B = np.exp(-ABS["mel_B"] * s.melanin)
    hb_eff = s.hb * s.vaso
    rho3_R = np.exp(-ABS["hb_R"] * hb_eff)
    rho3_G_deox = np.exp(-ABS["hb_G"] * hb_eff)
    rho3_G_oxy = np.exp(-(ABS["hb_G"] + ABS["hb_delta_G"]) * hb_eff)
    rho3_G = (1 - s.spo2) * rho3_G_deox + s.spo2 * rho3_G_oxy
    rho3_B = np.exp(-ABS["hb_B"] * hb_eff)
    D = ABS["D"]
    R = spec + (1 - spec) * (rho2_R ** 2) * (rho3_R ** 2) * D
    G = spec + (1 - spec) * (rho2_G ** 2) * (rho3_G ** 2) * D
    B = spec + (1 - spec) * (rho2_B ** 2) * (rho3_B ** 2) * D
    return float(R), float(G), float(B)


def inverse(rgb, vaso, hb_prior=14.0, n_iters=2):
    spec = ABS["spec"]; D = ABS["D"]
    def lift(c): return max(0.01, (c - spec) / (1 - spec) / D)
    lR, lG, lB = map(lift, rgb)
    hb = hb_prior; melanin = 0.15; spo2 = 0.97
    for _ in range(n_iters):
        hb_atten_B = np.exp(-2 * ABS["hb_B"] * hb * vaso)
        lB_dem = max(1e-3, lB / hb_atten_B)
        melanin = float(np.clip(-np.log(lB_dem) / (2 * ABS["mel_B"]), 0.02, 0.85))
        lR_corr = lR * np.exp(2 * ABS["mel_R"] * melanin)
        hb = float(np.clip(-np.log(max(1e-3, lR_corr)) / (2 * ABS["hb_R"] * vaso), 8, 20))
        lG_corr = lG * np.exp(2 * ABS["mel_G"] * melanin)
        t_d = np.exp(-2 * ABS["hb_G"] * hb * vaso)
        t_o = np.exp(-2 * (ABS["hb_G"] + ABS["hb_delta_G"]) * hb * vaso)
        spo2 = float(np.clip((lG_corr - t_d) / max(1e-9, t_o - t_d), 0.5, 1.0))
    return State(melanin=melanin, hb=hb, spo2=spo2, vaso=vaso)


def vaso_to_temp(vaso): return float(np.clip(33.0 + 4.0 * (vaso - 1.0), 27.0, 37.0))


def pchr(hr_obs, t_skin, spo2, hr_intrinsic=60.0):
    alpha_T = 0.08; beta_O2 = 1.5; T_ref = 33.0; SpO2_ref = 0.97
    dHR_met = alpha_T * (t_skin - T_ref) * hr_intrinsic
    dHR_o2 = beta_O2 * max(0.0, SpO2_ref - spo2) * hr_intrinsic
    dHR_auto = hr_obs - hr_intrinsic - dHR_met - dHR_o2
    return {"HR_obs": hr_obs, "HR_intrinsic": hr_intrinsic,
            "dHR_met": dHR_met, "dHR_o2": dHR_o2, "dHR_auto": dHR_auto}


# ---------------------------------------------------------------------------
# Plot styling — white background, minimal text, sans serif
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 130,
})

R_COLOR = "#d24d4d"
G_COLOR = "#3aa856"
B_COLOR = "#4d7fd2"
ACCENT = "#5fafff"
ACCENT2 = "#7fd47f"
WARN = "#ffaf5f"
HOT = "#ff5f5f"
PURPLE = "#c89fff"


def panel_grid():
    fig = plt.figure(figsize=(16.0, 4.2))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.32)
    return fig, gs


def label_subpanel(ax, letter):
    # Place an A/B/C/D label above each subplot using figure-level annotation
    # so the same call works for both 2D and 3D axes.
    bbox = ax.get_position()
    fig = ax.figure
    fig.text(bbox.x0 - 0.005, bbox.y1 + 0.015, letter,
             fontsize=13, fontweight="bold", va="bottom", ha="left")


def save_panel(fig, name):
    out = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  panel: {out.relative_to(ROOT)}")


def save_json(obj, name):
    out = VAL_DIR / name
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: asdict(o))
    print(f"  json:  {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Panel 1: Forward-model parameter sweeps (4 charts, D is 3D)
# ---------------------------------------------------------------------------

def panel1_forward_sweeps():
    base = State(0.15, 14.0, 0.97, 1.0)
    n = 128
    sweeps = {
        "melanin": np.linspace(0.02, 0.85, n),
        "hb":      np.linspace(8.0, 20.0, n),
        "vaso":    np.linspace(0.6, 1.6, n),
    }
    out = {}
    for key, xs in sweeps.items():
        Rs, Gs, Bs = [], [], []
        for x in xs:
            s = State(**asdict(base))
            setattr(s, key, float(x))
            r, g, b = forward(s)
            Rs.append(r); Gs.append(g); Bs.append(b)
        out[key] = {"x": xs.tolist(), "R": Rs, "G": Gs, "B": Bs}

    # Random states for the 3D cloud
    rng = np.random.default_rng(7)
    N = 1500
    cloud = []
    for _ in range(N):
        s = State(
            melanin=float(rng.uniform(0.05, 0.85)),
            hb=float(rng.uniform(8, 20)),
            spo2=float(rng.uniform(0.85, 1.0)),
            vaso=float(rng.uniform(0.6, 1.6)),
        )
        cloud.append((*forward(s), s.spo2))
    cloud = np.array(cloud)

    fig, gs = panel_grid()

    # A: melanin sweep
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(out["melanin"]["x"], out["melanin"]["R"], color=R_COLOR, lw=1.5)
    ax.plot(out["melanin"]["x"], out["melanin"]["G"], color=G_COLOR, lw=1.5)
    ax.plot(out["melanin"]["x"], out["melanin"]["B"], color=B_COLOR, lw=1.5)
    ax.set_xlabel("melanin index")
    ax.set_ylabel("reflectance")
    ax.set_ylim(0, 0.6)
    label_subpanel(ax, "A")

    # B: Hb sweep
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(out["hb"]["x"], out["hb"]["R"], color=R_COLOR, lw=1.5, label="R")
    ax.plot(out["hb"]["x"], out["hb"]["G"], color=G_COLOR, lw=1.5, label="G")
    ax.plot(out["hb"]["x"], out["hb"]["B"], color=B_COLOR, lw=1.5, label="B")
    ax.set_xlabel("[Hb]  (g/dL)")
    ax.set_ylabel("reflectance")
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper right")
    label_subpanel(ax, "B")

    # C: vasodilation sweep
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(out["vaso"]["x"], out["vaso"]["R"], color=R_COLOR, lw=1.5)
    ax.plot(out["vaso"]["x"], out["vaso"]["G"], color=G_COLOR, lw=1.5)
    ax.plot(out["vaso"]["x"], out["vaso"]["B"], color=B_COLOR, lw=1.5)
    ax.set_xlabel("vasodilation factor η")
    ax.set_ylabel("reflectance")
    ax.set_ylim(0, 0.6)
    label_subpanel(ax, "C")

    # D: 3D RGB cloud, colored by SpO2
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    sc = ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                    c=cloud[:, 3], cmap="plasma", s=4, alpha=0.65)
    ax.set_xlabel("R", labelpad=-2)
    ax.set_ylabel("G", labelpad=-2)
    ax.set_zlabel("B", labelpad=-2)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.10)
    cbar.set_label("SpO₂", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.tick_params(labelsize=7)
    label_subpanel(ax, "D")

    save_panel(fig, "panel1_forward_sweeps.png")
    save_json(out, "exp1_forward_spectra.json")


# ---------------------------------------------------------------------------
# Panel 2: Self-consistency (4 charts, C is 3D)
# ---------------------------------------------------------------------------

def panel2_self_consistency():
    rng = np.random.default_rng(17)
    N = 1000
    rows = []
    for _ in range(N):
        s = State(
            melanin=float(rng.uniform(0.05, 0.85)),
            hb=float(rng.uniform(8, 20)),
            spo2=float(rng.uniform(0.85, 1.0)),
            vaso=float(rng.uniform(0.6, 1.6)),
        )
        rgb = forward(s)
        r = inverse(rgb, s.vaso)
        rows.append((s.melanin, r.melanin, s.hb, r.hb, s.spo2, r.spo2))
    a = np.array(rows)

    fig, gs = panel_grid()

    # A: melanin true vs recovered
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(a[:, 0], a[:, 1], s=6, color=ACCENT, alpha=0.55)
    lim = (0, 0.9)
    ax.plot(lim, lim, color="black", lw=1, alpha=0.5)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("true melanin")
    ax.set_ylabel("recovered melanin")
    label_subpanel(ax, "A")

    # B: Hb true vs recovered
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(a[:, 2], a[:, 3], s=6, color=HOT, alpha=0.55)
    lim = (6, 22)
    ax.plot(lim, lim, color="black", lw=1, alpha=0.5)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("true [Hb] (g/dL)")
    ax.set_ylabel("recovered [Hb] (g/dL)")
    label_subpanel(ax, "B")

    # C: 3D — true_mel × true_Hb × Hb error, color by Hb error
    err_hb = np.abs(a[:, 3] - a[:, 2]) / a[:, 2]
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    sc = ax.scatter(a[:, 0], a[:, 2], err_hb * 100,
                    c=err_hb * 100, cmap="viridis", s=5, alpha=0.7)
    ax.set_xlabel("true melanin", labelpad=-2)
    ax.set_ylabel("true [Hb]", labelpad=-2)
    ax.set_zlabel("Hb err (%)", labelpad=-2)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.10)
    cbar.set_label("err (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.tick_params(labelsize=7)
    label_subpanel(ax, "C")

    # D: SpO2 true vs recovered
    ax = fig.add_subplot(gs[0, 3])
    ax.scatter(a[:, 4], a[:, 5], s=6, color=ACCENT2, alpha=0.55)
    lim = (0.45, 1.05)
    ax.plot(lim, lim, color="black", lw=1, alpha=0.5)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("true SpO₂")
    ax.set_ylabel("recovered SpO₂")
    label_subpanel(ax, "D")

    save_panel(fig, "panel2_self_consistency.png")
    summary = {
        "melanin": {
            "median": float(np.median(np.abs(a[:, 1] - a[:, 0]) / a[:, 0])),
            "p90": float(np.percentile(np.abs(a[:, 1] - a[:, 0]) / a[:, 0], 90)),
            "p99": float(np.percentile(np.abs(a[:, 1] - a[:, 0]) / a[:, 0], 99)),
        },
        "hb": {
            "median": float(np.median(np.abs(a[:, 3] - a[:, 2]) / a[:, 2])),
            "p90": float(np.percentile(np.abs(a[:, 3] - a[:, 2]) / a[:, 2], 90)),
            "p99": float(np.percentile(np.abs(a[:, 3] - a[:, 2]) / a[:, 2], 99)),
        },
        "spo2": {
            "median": float(np.median(np.abs(a[:, 5] - a[:, 4]) / a[:, 4])),
            "p90": float(np.percentile(np.abs(a[:, 5] - a[:, 4]) / a[:, 4], 90)),
            "p99": float(np.percentile(np.abs(a[:, 5] - a[:, 4]) / a[:, 4], 99)),
        },
    }
    save_json(summary, "exp2_self_consistency.json")


# ---------------------------------------------------------------------------
# Panel 3: Noise sensitivity (4 charts, C is 3D surface)
# ---------------------------------------------------------------------------

def panel3_noise_sensitivity():
    rng = np.random.default_rng(23)
    snrs_db = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    N = 200
    rms = {"melanin": [], "hb": [], "spo2": []}
    grid = []  # (snr, param_idx, rms_pct)
    for snr_db in snrs_db:
        snr_lin = 10 ** (snr_db / 10)
        per = {"melanin": [], "hb": [], "spo2": []}
        for _ in range(N):
            s = State(
                melanin=float(rng.uniform(0.05, 0.85)),
                hb=float(rng.uniform(8, 20)),
                spo2=float(rng.uniform(0.85, 1.0)),
                vaso=float(rng.uniform(0.6, 1.6)),
            )
            rgb = np.array(forward(s))
            sigma = rgb / np.sqrt(snr_lin)
            noisy = np.clip(rgb + rng.normal(0, sigma), 1e-3, 1 - 1e-3)
            r = inverse(tuple(noisy.tolist()), s.vaso)
            per["melanin"].append((r.melanin - s.melanin) / s.melanin)
            per["hb"].append((r.hb - s.hb) / s.hb)
            per["spo2"].append((r.spo2 - s.spo2) / s.spo2)
        for k in rms:
            v = np.array(per[k])
            rms[k].append(float(np.sqrt(np.mean(v ** 2))))
        for k in rms:
            grid.append((snr_db, k, rms[k][-1] * 100))

    fig, gs = panel_grid()

    # A: melanin
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(snrs_db, np.array(rms["melanin"]) * 100, "o-", color=ACCENT, lw=1.5, ms=6)
    ax.axhline(5, color="black", linestyle="--", lw=1, alpha=0.4)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("melanin RMS error (%)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.18)
    label_subpanel(ax, "A")

    # B: Hb
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(snrs_db, np.array(rms["hb"]) * 100, "o-", color=HOT, lw=1.5, ms=6)
    ax.axhline(5, color="black", linestyle="--", lw=1, alpha=0.4)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("[Hb] RMS error (%)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.18)
    label_subpanel(ax, "B")

    # C: 3D surface — SNR × parameter index × RMS
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    X, Y = np.meshgrid(snrs_db, [0, 1, 2])
    Z = np.array([
        np.array(rms["melanin"]) * 100,
        np.array(rms["hb"]) * 100,
        np.array(rms["spo2"]) * 100,
    ])
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, edgecolor="white", lw=0.3)
    ax.set_xlabel("SNR (dB)", labelpad=-2)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["mel", "Hb", "SpO₂"], fontsize=7)
    ax.set_zlabel("RMS err (%)", labelpad=-2)
    ax.tick_params(labelsize=7)
    label_subpanel(ax, "C")

    # D: SpO2
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(snrs_db, np.array(rms["spo2"]) * 100, "o-", color=ACCENT2, lw=1.5, ms=6)
    ax.axhline(5, color="black", linestyle="--", lw=1, alpha=0.4)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SpO₂ RMS error (%)")
    ax.set_ylim(0, 60)
    ax.grid(True, alpha=0.18)
    label_subpanel(ax, "D")

    save_panel(fig, "panel3_noise_sensitivity.png")
    save_json({"snr_db": snrs_db, "rms_err_pct":
              {k: [v * 100 for v in rms[k]] for k in rms}},
              "exp3_noise_sensitivity.json")


# ---------------------------------------------------------------------------
# Panel 4: Vasodilation × temperature × metabolic decomposition (4 charts, C is 3D)
# ---------------------------------------------------------------------------

def panel4_vaso_temp():
    vaso_grid = np.linspace(0.6, 1.6, 100)
    temp_grid = np.array([vaso_to_temp(v) for v in vaso_grid])

    # 3D surface: vaso × HR_intrinsic × dHR_met
    vasoX = np.linspace(0.6, 1.6, 30)
    hrintY = np.linspace(50, 80, 30)
    V, H = np.meshgrid(vasoX, hrintY)
    T = 33.0 + 4.0 * (V - 1.0)
    DM = 0.08 * (T - 33.0) * H

    # ΔHR_met vs T_skin for several HR_intrinsic levels
    Tx = np.linspace(27, 37, 100)
    hr_levels = [50, 60, 70, 80]

    fig, gs = panel_grid()

    # A: vaso → T_skin line
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(vaso_grid, temp_grid, color=HOT, lw=1.8)
    ax.scatter([1.0], [33.0], s=60, color="black", zorder=4)
    ax.set_xlabel("vasodilation factor η")
    ax.set_ylabel("T_skin (°C)")
    ax.set_xlim(0.55, 1.65); ax.set_ylim(26.5, 37.5)
    ax.grid(True, alpha=0.18)
    label_subpanel(ax, "A")

    # B: histogram of T_skin assuming uniform vaso
    rng = np.random.default_rng(31)
    vaso_samples = rng.uniform(0.6, 1.6, 5000)
    t_samples = np.array([vaso_to_temp(v) for v in vaso_samples])
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(t_samples, bins=40, color=ACCENT, edgecolor="black", lw=0.5, alpha=0.85)
    ax.set_xlabel("T_skin (°C)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.18)
    label_subpanel(ax, "B")

    # C: 3D surface — η × HR_intrinsic × dHR_met
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    surf = ax.plot_surface(V, H, DM, cmap="plasma", alpha=0.9,
                            edgecolor="white", lw=0.2)
    ax.set_xlabel("η", labelpad=-2)
    ax.set_ylabel("HR intrinsic", labelpad=-2)
    ax.set_zlabel("ΔHR_met (bpm)", labelpad=-2)
    ax.tick_params(labelsize=7)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.10)
    cbar.set_label("ΔHR_met", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    label_subpanel(ax, "C")

    # D: ΔHR_met vs T_skin for several HR_intrinsic levels
    ax = fig.add_subplot(gs[0, 3])
    cmap = plt.get_cmap("plasma")
    for i, hr in enumerate(hr_levels):
        dhr = 0.08 * (Tx - 33.0) * hr
        ax.plot(Tx, dhr, color=cmap(i / max(1, len(hr_levels) - 1)),
                lw=1.5, label=f"HR_int = {hr}")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("T_skin (°C)")
    ax.set_ylabel("ΔHR_met (bpm)")
    ax.legend()
    ax.grid(True, alpha=0.18)
    label_subpanel(ax, "D")

    save_panel(fig, "panel4_vaso_temperature.png")
    save_json({"vaso": vaso_grid.tolist(), "temp_C": temp_grid.tolist(),
               "T_hist_C": t_samples.tolist()[:1000]},
              "exp4_vaso_temperature.json")


# ---------------------------------------------------------------------------
# Panel 5: PCHR scenarios (4 charts, C is 3D)
# ---------------------------------------------------------------------------

SCENARIOS = [
    {"name": "Rest",     "T_skin": 33.0, "spo2": 0.97, "HR": 70},
    {"name": "Fever",    "T_skin": 35.5, "spo2": 0.97, "HR": 85},
    {"name": "Hypoxia",  "T_skin": 33.0, "spo2": 0.85, "HR": 85},
    {"name": "Exercise", "T_skin": 34.5, "spo2": 0.96, "HR": 120},
]


def panel5_pchr():
    decomp = []
    for sc in SCENARIOS:
        d = pchr(sc["HR"], sc["T_skin"], sc["spo2"])
        d.update({"name": sc["name"], **sc})
        decomp.append(d)

    fig, gs = panel_grid()

    # A: stacked bars per scenario
    ax = fig.add_subplot(gs[0, 0])
    names = [d["name"] for d in decomp]
    hi = np.array([d["HR_intrinsic"] for d in decomp])
    dm = np.array([d["dHR_met"] for d in decomp])
    do = np.array([d["dHR_o2"] for d in decomp])
    da = np.array([d["dHR_auto"] for d in decomp])
    bot = np.zeros_like(hi)
    for label, vals, color in [("intrinsic", hi, "#888"),
                               ("Δmet", dm, WARN),
                               ("Δhypox", do, ACCENT2),
                               ("Δauto", da, PURPLE)]:
        ax.bar(names, vals, bottom=bot, color=color, edgecolor="black", lw=0.5,
               label=label)
        bot = bot + vals
    ax.set_ylabel("HR contribution (bpm)")
    ax.legend(loc="upper left")
    label_subpanel(ax, "A")

    # B: ΔHR_met vs T_skin sweep at HR_int = 60
    Tx = np.linspace(27, 37, 100)
    dhr_met = 0.08 * (Tx - 33.0) * 60
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(Tx, dhr_met, color=WARN, lw=1.8)
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    for d in decomp:
        ax.scatter([d["T_skin"]], [d["dHR_met"]], s=50, color="black", zorder=4)
        ax.text(d["T_skin"] + 0.1, d["dHR_met"] + 0.3, d["name"][:3], fontsize=8)
    ax.set_xlabel("T_skin (°C)")
    ax.set_ylabel("ΔHR_met (bpm)")
    ax.grid(True, alpha=0.18)
    label_subpanel(ax, "B")

    # C: 3D — T_skin × SpO2 × dHR_auto across grid (decomposition surface)
    Ts = np.linspace(31, 36, 30)
    Os = np.linspace(0.85, 1.00, 30)
    Tg, Og = np.meshgrid(Ts, Os)
    HR_obs_grid = 100  # demonstration HR
    dHR_auto = HR_obs_grid - 60 - 0.08 * (Tg - 33.0) * 60 - 1.5 * np.maximum(0, 0.97 - Og) * 60
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    surf = ax.plot_surface(Tg, Og, dHR_auto, cmap="coolwarm", alpha=0.88,
                            edgecolor="white", lw=0.2)
    # Overlay scenario points (with HR-dependent vertical position)
    for d in decomp:
        sc_dauto = d["HR_obs"] - 60 - d["dHR_met"] - d["dHR_o2"]
        ax.scatter([d["T_skin"]], [d["spo2"]], [sc_dauto],
                   s=40, color="black", edgecolor="white", zorder=10)
    ax.set_xlabel("T_skin", labelpad=-2)
    ax.set_ylabel("SpO₂", labelpad=-2)
    ax.set_zlabel("ΔHR_auto", labelpad=-2)
    ax.tick_params(labelsize=7)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.10)
    cbar.set_label("ΔHR_auto", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    label_subpanel(ax, "C")

    # D: ΔHR_O2 vs SpO2 sweep
    Sx = np.linspace(0.7, 1.0, 100)
    dhr_o2 = 1.5 * np.maximum(0, 0.97 - Sx) * 60
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(Sx, dhr_o2, color=ACCENT2, lw=1.8)
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    for d in decomp:
        ax.scatter([d["spo2"]], [d["dHR_o2"]], s=50, color="black", zorder=4)
        ax.text(d["spo2"] - 0.005, d["dHR_o2"] + 1.0, d["name"][:3], fontsize=8,
                ha="right")
    ax.set_xlabel("SpO₂")
    ax.set_ylabel("ΔHR_O₂ (bpm)")
    ax.grid(True, alpha=0.18)
    label_subpanel(ax, "D")

    save_panel(fig, "panel5_pchr_scenarios.png")
    save_json(decomp, "exp5_pchr_scenarios.json")


# ---------------------------------------------------------------------------
# Panel 6: Conventional comparison (4 charts, C is 3D)
# ---------------------------------------------------------------------------

def exp6_data(seed=31):
    rng = np.random.default_rng(seed)
    fs = 30.0
    minutes = 5
    n = int(fs * 60 * minutes)
    t = np.arange(n) / fs
    true_hr = 70.0
    cardiac_freq = true_hr / 60.0
    base = State(0.15, 14.0, 0.97, 1.0)
    _, G0, _ = forward(base)

    def ma(arr, w):
        return np.convolve(arr, np.ones(w) / w, mode="same")
    def bandpass(x):
        return ma(x, 4) - ma(x, 32)
    def hr_track(sig, fs_, win_s=10.0, step_s=1.0):
        win = int(win_s * fs_); step = int(step_s * fs_)
        ot, oh = [], []
        for k in range(0, len(sig) - win, step):
            seg = sig[k:k+win] - np.mean(sig[k:k+win])
            seg = seg * np.hanning(len(seg))
            spec = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(len(seg), d=1 / fs_)
            band = (freqs >= 0.7) & (freqs <= 3.0)
            if not band.any(): continue
            oh.append(60.0 * freqs[band][np.argmax(spec[band])])
            ot.append((k + win / 2) / fs_)
        return np.array(ot), np.array(oh)

    drift_amps = [0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
    cardiac_ac = 0.006 * np.sin(2 * np.pi * cardiac_freq * t)
    noise_sd = 0.0006

    var_a, var_b, bias_a, bias_b, spo2_err = [], [], [], [], []
    method_b_traces = []
    for amp in drift_amps:
        drift = 1.0 + amp * np.sin(2 * np.pi * 0.05 * t)
        G_obs = drift * G0 * (1 + cardiac_ac) + rng.normal(0, noise_sd, n)
        ga = bandpass(G_obs)
        ta, ha = hr_track(ga, fs)
        win_dc = int(4 * fs)
        G_dc = ma(G_obs, win_dc)
        G_ratio = (G_obs - G_dc) / np.maximum(G_dc, 1e-6)
        gb = bandpass(G_ratio)
        tb, hb_ = hr_track(gb, fs)
        ha = ha[5:]; hb_ = hb_[5:]; tb = tb[5:]
        var_a.append(float(np.var(ha)))
        var_b.append(float(np.var(hb_)))
        bias_a.append(float(np.mean(ha) - true_hr))
        bias_b.append(float(np.mean(hb_) - true_hr))
        method_b_traces.append((amp, tb.tolist(), hb_.tolist()))

        # SpO2 recovery from time-averaged DC of all three channels
        R_obs = drift * forward(base)[0] + rng.normal(0, noise_sd, n)
        B_obs = drift * forward(base)[2] + rng.normal(0, noise_sd, n)
        rgb_mean = (
            float(np.mean(R_obs[n//2:] / drift[n//2:])),
            float(np.mean(G_obs[n//2:] / drift[n//2:])),
            float(np.mean(B_obs[n//2:] / drift[n//2:])),
        )
        rec = inverse(rgb_mean, base.vaso)
        spo2_err.append(float(abs(rec.spo2 - base.spo2) / base.spo2))

    return {"drift_amps": drift_amps, "var_a": var_a, "var_b": var_b,
            "bias_a": bias_a, "bias_b": bias_b, "spo2_err": spo2_err,
            "method_b_traces": method_b_traces, "true_hr": true_hr}


def panel6_comparison():
    res = exp6_data()
    amps_pct = np.array(res["drift_amps"]) * 100

    fig, gs = panel_grid()

    # A: HR variance vs drift amplitude
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(amps_pct, res["var_a"], "o-", color="#888", lw=1.5, ms=6,
            label="green-only")
    ax.plot(amps_pct, res["var_b"], "o-", color=ACCENT, lw=1.5, ms=6,
            label="layered")
    ax.set_xlabel("drift amplitude (% of DC)")
    ax.set_ylabel("HR variance (bpm²)")
    ax.legend()
    ax.grid(True, alpha=0.18)
    label_subpanel(ax, "A")

    # B: HR bias vs drift amplitude
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(amps_pct, res["bias_a"], "o-", color="#888", lw=1.5, ms=6,
            label="green-only")
    ax.plot(amps_pct, res["bias_b"], "o-", color=ACCENT, lw=1.5, ms=6,
            label="layered")
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_xlabel("drift amplitude (% of DC)")
    ax.set_ylabel("HR bias (bpm)")
    ax.legend()
    ax.grid(True, alpha=0.18)
    label_subpanel(ax, "B")

    # C: 3D — drift_amp × time × HR estimate (method B), color by HR
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    cmap = plt.get_cmap("viridis")
    for i, (amp, tb, hb_) in enumerate(res["method_b_traces"]):
        if len(tb) == 0: continue
        amps = np.full(len(tb), amp * 100)
        c = cmap(i / max(1, len(res["method_b_traces"]) - 1))
        ax.plot(amps, tb, hb_, color=c, lw=1, alpha=0.85)
    ax.set_xlabel("drift (%)", labelpad=-2)
    ax.set_ylabel("time (s)", labelpad=-2)
    ax.set_zlabel("HR (bpm)", labelpad=-2)
    ax.tick_params(labelsize=7)
    label_subpanel(ax, "C")

    # D: SpO2 recovery error vs drift amplitude
    ax = fig.add_subplot(gs[0, 3])
    spo2_pct = np.array(res["spo2_err"]) * 100
    ax.bar(amps_pct, spo2_pct, width=8, color=ACCENT2, edgecolor="black", lw=0.5)
    ax.set_xlabel("drift amplitude (% of DC)")
    ax.set_ylabel("SpO₂ recovery err (%)")
    ax.set_ylim(0, max(6, max(spo2_pct) * 1.3))
    ax.grid(True, alpha=0.18)
    label_subpanel(ax, "D")

    save_panel(fig, "panel6_conventional_comparison.png")
    summary = {
        "drift_amps": res["drift_amps"],
        "method_a_var_bpm2": res["var_a"],
        "method_b_var_bpm2": res["var_b"],
        "method_a_bias_bpm": res["bias_a"],
        "method_b_bias_bpm": res["bias_b"],
        "spo2_recovery_err_method_b": res["spo2_err"],
        "true_hr_bpm": res["true_hr"],
    }
    save_json(summary, "exp6_conventional_comparison.json")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def main():
    print("Generating panels and validation JSON for layered-optical-PPG paper")
    print(f"  figures dir: {FIG_DIR.relative_to(ROOT.parent)}")
    print(f"  validation dir: {VAL_DIR.relative_to(ROOT.parent)}")
    print()
    panel1_forward_sweeps()
    panel2_self_consistency()
    panel3_noise_sensitivity()
    panel4_vaso_temp()
    panel5_pchr()
    panel6_comparison()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
