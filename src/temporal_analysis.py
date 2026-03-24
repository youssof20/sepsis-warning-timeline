"""
Phase 2 — Temporal analysis: Mann–Whitney U tests at hourly windows before sepsis onset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_RESULTS = PROJECT_ROOT / "outputs" / "results"

SEPSIS_CSV = OUTPUT_RESULTS / "sepsis_patients.csv"
NONSEPSIS_CSV = OUTPUT_RESULTS / "nonsepsis_patients.csv"

EXCLUDE = {
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
    "SepsisLabel",
    "patient_id",
    "hours_before_onset",
    "is_sepsis",
    "onset_hour",
}

HOURS_T = list(range(-12, 0))  # -12 .. -1 inclusive
BONFERRONI_ALPHA = 0.05 / 12
ALPHA_RAW = 0.05
ALWAYS_CHECK_T = [-1, -2, -3, -4, -5, -6]
MIN_N = 30


def _clinical_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in EXCLUDE]
    if len(cols) < 34:
        raise ValueError(
            f"Expected at least 34 clinical variable columns; got {len(cols)}. "
            f"Columns: {sorted(cols)}"
        )
    return cols


def _median_onset_hour(sepsis: pd.DataFrame) -> float:
    """Median across sepsis patients of first onset time (hours)."""
    per_patient = sepsis.groupby("patient_id", sort=False)["onset_hour"].first()
    return float(per_patient.median())


def _mann_whitney_cles(
    sepsis_vals: np.ndarray, non_vals: np.ndarray
) -> tuple[float, float, float]:
    """Return (U_statistic, p_value, cles)."""
    res = stats.mannwhitneyu(
        sepsis_vals, non_vals, alternative="two-sided", method="auto"
    )
    stat = float(res.statistic)
    p = float(res.pvalue)
    denom = len(sepsis_vals) * len(non_vals)
    cles = stat / denom if denom > 0 else np.nan
    return stat, p, cles


def run_temporal_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SEPSIS_CSV.is_file():
        raise FileNotFoundError(
            f"Missing {SEPSIS_CSV}. Run src/data_pipeline.py first."
        )
    if not NONSEPSIS_CSV.is_file():
        raise FileNotFoundError(
            f"Missing {NONSEPSIS_CSV}. Run src/data_pipeline.py first."
        )

    sepsis = pd.read_csv(SEPSIS_CSV)
    nonsepsis = pd.read_csv(NONSEPSIS_CSV)

    clinical_cols = _clinical_columns(sepsis)

    median_onset = _median_onset_hour(sepsis)
    median_iculos_rounded = int(round(median_onset))

    hbo_rounded = sepsis["hours_before_onset"].astype(float).round(0)

    rows_full: list[dict] = []

    for var in clinical_cols:
        for T in HOURS_T:
            s = sepsis.loc[hbo_rounded == float(T), var]
            sepsis_vals = s.dropna().to_numpy(dtype=float)

            target_iculos = median_iculos_rounded - T
            n = nonsepsis.loc[nonsepsis["ICULOS"] == target_iculos, var]
            non_vals = n.dropna().to_numpy(dtype=float)

            n_s = int(len(sepsis_vals))
            n_n = int(len(non_vals))

            if n_s < MIN_N or n_n < MIN_N:
                continue

            try:
                _, p_value, cles = _mann_whitney_cles(sepsis_vals, non_vals)
            except ValueError:
                continue

            p_value = float(p_value)
            cles = float(cles)

            sig_raw = p_value < ALPHA_RAW
            sig_corr = p_value < BONFERRONI_ALPHA

            rows_full.append(
                {
                    "variable": var,
                    "hour": T,
                    "n_sepsis": n_s,
                    "n_nonsepsis": n_n,
                    "p_value": p_value,
                    "cles": cles,
                    "significant": sig_raw,
                    "significant_corrected": sig_corr,
                }
            )

    full_df = pd.DataFrame(rows_full)

    # --- Timeline summary per variable
    timeline_rows: list[dict] = []

    for var in clinical_cols:
        sub = full_df[full_df["variable"] == var]
        if sub.empty:
            timeline_rows.append(
                {
                    "variable": var,
                    "earliest_warning_hours": np.nan,
                    "max_effect_size": np.nan,
                    "always_significant": False,
                }
            )
            continue

        sig = sub.loc[sub["significant_corrected"], "hour"]
        if sig.empty:
            earliest_t = None
            earliest_hours = np.nan
        else:
            earliest_t = int(sig.min())
            earliest_hours = float(-earliest_t)

        max_effect = float((sub["cles"] - 0.5).abs().max() * 2.0)

        always = True
        for t in ALWAYS_CHECK_T:
            r = sub[sub["hour"] == t]
            if r.empty or not bool(r["significant_corrected"].iloc[0]):
                always = False
                break

        timeline_rows.append(
            {
                "variable": var,
                "earliest_warning_hours": earliest_hours,
                "max_effect_size": max_effect,
                "always_significant": always,
            }
        )

    timeline_df = pd.DataFrame(timeline_rows)
    timeline_df = timeline_df.sort_values(
        "earliest_warning_hours",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return full_df, timeline_df


def main() -> None:
    full_df, timeline_df = run_temporal_analysis()

    OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)
    full_path = OUTPUT_RESULTS / "temporal_analysis_full.csv"
    tl_path = OUTPUT_RESULTS / "early_warning_timeline.csv"

    full_df.to_csv(full_path, index=False)
    timeline_df.to_csv(tl_path, index=False)

    print(f"Wrote {full_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {tl_path.relative_to(PROJECT_ROOT)}")

    top = timeline_df.dropna(subset=["earliest_warning_hours"]).head(10)
    print("Top 10 early warning variables (by lead time):")
    for _, row in top.iterrows():
        h = row["earliest_warning_hours"]
        print(f"  {row['variable']}: {h:.0f} hours before onset")
    print("PHASE 2 COMPLETE")


if __name__ == "__main__":
    main()
