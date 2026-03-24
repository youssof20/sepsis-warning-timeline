"""
Phase 4 — Publication figures for Sepsis Early Warning Timeline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_RESULTS = PROJECT_ROOT / "outputs" / "results"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"

DPI = 150

VITALS = {"HR", "Resp", "Temp", "O2Sat", "SBP", "MAP", "DBP"}


def _color_for(var: str) -> str:
    if var in VITALS:
        return "#1f77b4"
    return "#ff7f0e"


def _top15_lead_time(tl: pd.DataFrame) -> pd.DataFrame:
    return (
        tl.dropna(subset=["earliest_warning_hours"])
        .sort_values(
            ["earliest_warning_hours", "max_effect_size"],
            ascending=[False, False],
        )
        .head(15)
    )


def _fig1_timeline(tl: pd.DataFrame) -> None:
    top = _top15_lead_time(tl)
    labels = top["variable"].tolist()
    leads = top["earliest_warning_hours"].astype(float).tolist()
    colors = [_color_for(v) for v in labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = np.arange(len(labels))
    for i, (w, c) in enumerate(zip(leads, colors)):
        ax.barh(i, w, left=-w, height=0.75, color=c, align="center")
    ax.axvline(0, color="0.35", linestyle="--", linewidth=1.2, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-12.5, 1)
    ax.set_xlabel("Hours Before Sepsis Onset")
    ax.set_title(
        "How Many Hours Before Sepsis Onset Each Variable\n"
        "Becomes a Statistically Significant Warning",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.3, linestyle="-", linewidth=0.5)
    legend_patches = [
        mpatches.Patch(color="#1f77b4", label="Vitals"),
        mpatches.Patch(color="#ff7f0e", label="Labs"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", frameon=True)
    plt.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "early_warning_timeline.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def _fig2_heatmap(tl: pd.DataFrame, temporal: pd.DataFrame) -> None:
    top = _top15_lead_time(tl)
    vars15 = top["variable"].tolist()
    hours = list(range(-12, 0))

    mat = np.full((len(vars15), len(hours)), np.nan)
    sig = np.zeros_like(mat, dtype=bool)

    for i, v in enumerate(vars15):
        sub = temporal[temporal["variable"] == v]
        for j, h in enumerate(hours):
            row = sub[sub["hour"] == h]
            if not row.empty:
                mat[i, j] = row["cles"].iloc[0]
                sig[i, j] = bool(row["significant_corrected"].iloc[0])

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap = mpl.colormaps["Reds"].copy()
    cmap.set_bad(color="0.92")
    mat = np.ma.masked_invalid(mat)

    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=cmap,
        vmin=0.5,
        vmax=1.0,
        interpolation="nearest",
        origin="upper",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CLES (effect size)")

    ax.set_xticks(np.arange(len(hours)))
    ax.set_xticklabels([str(h) for h in hours])
    ax.set_yticks(np.arange(len(vars15)))
    ax.set_yticklabels(vars15)
    ax.set_xlabel("Hour before sepsis onset")
    ax.set_title("Effect Size by Variable and Hour Before Sepsis Onset", fontsize=12)

    msk = np.ma.getmaskarray(mat)
    ny, nx = mat.shape
    for i in range(ny):
        for j in range(nx):
            if not msk[i, j] and sig[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.5,
                    )
                )

    plt.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "effect_size_heatmap.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def _fig3_shap_scatter(comp: pd.DataFrame) -> None:
    comp = comp.dropna(subset=["shap_rank", "temporal_rank"]).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    diff = (comp["shap_rank"] - comp["temporal_rank"]).abs().to_numpy()

    colors = np.where(
        diff <= 5,
        "#2ca02c",
        np.where(diff > 10, "#d62728", "#bcbd22"),
    )

    ax.scatter(
        comp["shap_rank"],
        comp["temporal_rank"],
        c=colors,
        s=40,
        zorder=3,
        edgecolors="0.3",
        linewidths=0.3,
    )
    for _, row in comp.iterrows():
        ax.annotate(
            row["variable"],
            (row["shap_rank"], row["temporal_rank"]),
            fontsize=8,
            alpha=0.85,
            xytext=(3, 3),
            textcoords="offset points",
        )

    lim = max(comp["shap_rank"].max(), comp["temporal_rank"].max())
    ax.plot([1, lim], [1, lim], color="0.5", linestyle="--", linewidth=1, zorder=1)
    ax.set_xlabel("SHAP rank (1 = most important to model)")
    ax.set_ylabel("Temporal rank (1 = earliest warning)")
    ax.set_title("Does the Model Use What Warns Earliest?", fontsize=12)
    ax.set_xlim(0.5, lim + 0.5)
    ax.set_ylim(0.5, lim + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    fig.text(
        0.5,
        0.02,
        "Variables above the diagonal warn earlier than the model weights them.",
        ha="center",
        fontsize=9,
        style="italic",
        color="0.35",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUTPUT_FIGURES / "shap_vs_temporal.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def _fig4_missingness(tl: pd.DataFrame, miss: pd.DataFrame) -> None:
    m = miss.merge(
        tl[["variable", "earliest_warning_hours"]],
        on="variable",
        how="inner",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(
        m["pct_present"],
        m["earliest_warning_hours"],
        c="0.35",
        s=28,
        zorder=3,
    )
    for _, row in m.iterrows():
        if pd.notna(row["earliest_warning_hours"]):
            ax.annotate(
                row["variable"],
                (row["pct_present"], row["earliest_warning_hours"]),
                fontsize=8,
                alpha=0.9,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("% data present (non-missing rows)")
    ax.set_ylabel("Earliest warning lead time (hours)")
    ax.set_title("Data Availability vs Early Warning Lead Time", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "missingness_impact.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def _fig5_trajectories(sepsis: pd.DataFrame, non: pd.DataFrame, tl: pd.DataFrame) -> None:
    top5 = (
        tl.dropna(subset=["earliest_warning_hours"])
        .sort_values(
            ["earliest_warning_hours", "max_effect_size"],
            ascending=[False, False],
        )
        .head(5)
    )
    vars5 = top5["variable"].tolist()

    median_onset = float(sepsis.groupby("patient_id", sort=False)["onset_hour"].first().median())
    m = int(round(median_onset))
    hbo = sepsis["hours_before_onset"].astype(float).round(0)

    hours = list(range(-12, 1))

    fig, axes = plt.subplots(1, 5, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Clinical Variable Trajectories: Sepsis vs Non-Sepsis",
        fontsize=12,
        y=1.02,
    )

    for ax, var in zip(axes, vars5):
        ax.set_facecolor("white")
        all_vals = []
        for h in hours:
            s = sepsis.loc[hbo == float(h), var].dropna()
            t_icu = m - h
            n = non.loc[non["ICULOS"] == t_icu, var].dropna()
            all_vals.extend(s.tolist())
            all_vals.extend(n.tolist())
        all_vals = np.asarray(all_vals, dtype=float)
        mu_g = float(np.nanmean(all_vals))
        sig_g = float(np.nanstd(all_vals))
        if not np.isfinite(sig_g) or sig_g == 0:
            sig_g = 1.0

        means_s, stds_s = [], []
        means_n, stds_n = [], []
        xs = []
        for h in hours:
            s = sepsis.loc[hbo == float(h), var].dropna()
            t_icu = m - h
            n = non.loc[non["ICULOS"] == t_icu, var].dropna()
            if len(s) == 0 and len(n) == 0:
                continue
            xs.append(h)
            zs = (s - mu_g) / sig_g
            zn = (n - mu_g) / sig_g
            means_s.append(float(np.nanmean(zs)) if len(zs) else np.nan)
            stds_s.append(float(np.nanstd(zs)) if len(zs) > 1 else 0.0)
            means_n.append(float(np.nanmean(zn)) if len(zn) else np.nan)
            stds_n.append(float(np.nanstd(zn)) if len(zn) > 1 else 0.0)

        xs = np.asarray(xs, dtype=float)
        ms = np.asarray(means_s, dtype=float)
        ss = np.asarray(stds_s, dtype=float)
        mn = np.asarray(means_n, dtype=float)
        sn = np.asarray(stds_n, dtype=float)

        ax.plot(xs, ms, color="C0", linestyle="-", linewidth=1.2, label="Sepsis")
        ax.fill_between(xs, ms - ss, ms + ss, color="C0", alpha=0.15)
        ax.plot(xs, mn, color="C1", linestyle="--", linewidth=1.2, label="Non-sepsis")
        ax.fill_between(xs, mn - sn, mn + sn, color="C1", alpha=0.15)
        ax.set_title(var, fontsize=10)
        ax.set_xlabel("Hours before onset")
        ax.set_ylabel("z-score")
        ax.set_xlim(-12.5, 0.5)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    axes[0].legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "variable_trajectories.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def run_all() -> None:
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    tl = pd.read_csv(OUTPUT_RESULTS / "early_warning_timeline.csv")
    temporal = pd.read_csv(OUTPUT_RESULTS / "temporal_analysis_full.csv")
    miss = pd.read_csv(OUTPUT_RESULTS / "missingness_report.csv")
    comp = pd.read_csv(OUTPUT_RESULTS / "shap_vs_temporal.csv")

    _fig1_timeline(tl)
    _fig2_heatmap(tl, temporal)
    _fig3_shap_scatter(comp)
    _fig4_missingness(tl, miss)

    sepsis = pd.read_csv(OUTPUT_RESULTS / "sepsis_patients.csv")
    non = pd.read_csv(OUTPUT_RESULTS / "nonsepsis_patients.csv")
    _fig5_trajectories(sepsis, non, tl)


if __name__ == "__main__":
    run_all()
    print("PHASE 4 COMPLETE")
