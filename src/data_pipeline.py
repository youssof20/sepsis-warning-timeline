"""
PhysioNet Challenge 2019 — Phase 1 data pipeline.

Loads per-patient .psv files, labels sepsis onset, derives hours_before_onset,
exports cohort CSVs and a missingness report.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_RESULTS = PROJECT_ROOT / "outputs" / "results"

EXCLUDE_FROM_CLINICAL = {
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
    "SepsisLabel",
}

SEPSIS_WINDOW = (-24, 1)  # hours_before_onset inclusive


def _find_psv_files() -> list[Path]:
    patterns = [
        DATA_DIR / "training_setA" / "*.psv",
        DATA_DIR / "training_setB" / "*.psv",
    ]
    files: list[Path] = []
    for pat in patterns:
        files.extend(Path(p) for p in glob.glob(str(pat)))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No .psv files found under {DATA_DIR / 'training_setA'} or "
            f"{DATA_DIR / 'training_setB'}. Download PhysioNet 2019 training data "
            "and place training_setA/ and training_setB/ under data/."
        )
    return files


def _load_patient_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    df["patient_id"] = path.stem
    return df


def _clinical_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in EXCLUDE_FROM_CLINICAL and c != "patient_id"]
    if len(cols) != 34:
        raise ValueError(
            f"Expected 34 clinical variable columns; got {len(cols)}. "
            f"Columns: {sorted(cols)}"
        )
    return cols


def _patient_onset_info(df_patient: pd.DataFrame) -> tuple[bool, float]:
    """is_sepsis, onset_hour (NaN if not sepsis)."""
    has_sepsis = (df_patient["SepsisLabel"] == 1).any()
    if not has_sepsis:
        return False, np.nan
    onset = df_patient.loc[df_patient["SepsisLabel"] == 1, "ICULOS"].min()
    return True, float(onset)


def _sepsis_ge_6h_before_onset(df_patient: pd.DataFrame, onset_hour: float) -> bool:
    """True if patient has ICU data at least 6 hours before sepsis onset."""
    min_iculos = df_patient["ICULOS"].min()
    return (onset_hour - min_iculos) >= 6


def run_pipeline() -> None:
    psv_files = _find_psv_files()

    all_parts: list[pd.DataFrame] = []
    patient_rows: list[dict] = []

    for path in psv_files:
        df = _load_patient_table(path)
        pid = df["patient_id"].iloc[0]
        is_sepsis, onset_hour = _patient_onset_info(df)
        patient_rows.append(
            {
                "patient_id": pid,
                "is_sepsis": is_sepsis,
                "onset_hour": onset_hour,
                "max_iculos": float(df["ICULOS"].max()),
            }
        )
        all_parts.append(df)

    full = pd.concat(all_parts, ignore_index=True)
    clinical_cols = _clinical_columns(full)

    meta = pd.DataFrame(patient_rows)
    meta_by_pid = meta.set_index("patient_id")

    full["is_sepsis"] = full["patient_id"].map(meta_by_pid["is_sepsis"])
    full["onset_hour"] = full["patient_id"].map(meta_by_pid["onset_hour"])

    # hours_before_onset: sepsis only; NaN for non-sepsis
    full["hours_before_onset"] = np.where(
        full["is_sepsis"],
        full["onset_hour"] - full["ICULOS"],
        np.nan,
    )

    sepsis_ids = meta.loc[meta["is_sepsis"], "patient_id"].tolist()
    nonsepsis_ids = meta.loc[~meta["is_sepsis"], "patient_id"].tolist()

    sepsis_mask = full["patient_id"].isin(sepsis_ids)
    low, high = SEPSIS_WINDOW
    window_mask = (full["hours_before_onset"] >= low) & (full["hours_before_onset"] <= high)
    sepsis_out = full.loc[sepsis_mask & window_mask].copy()

    nonsepsis_out = full.loc[full["patient_id"].isin(nonsepsis_ids)].copy()

    # --- Stats: 6h before onset uses full sepsis series (pre row-window filter)
    sepsis_ge_6h = 0
    for pid in sepsis_ids:
        df_p = full.loc[full["patient_id"] == pid]
        if df_p.empty:
            continue
        oh = float(meta_by_pid.loc[pid, "onset_hour"])
        if _sepsis_ge_6h_before_onset(df_p, oh):
            sepsis_ge_6h += 1

    n_patients = len(meta)
    n_sepsis = int(meta["is_sepsis"].sum())
    n_nonsepsis = n_patients - n_sepsis
    pct_sepsis = 100.0 * n_sepsis / n_patients if n_patients else 0.0
    pct_nonsepsis = 100.0 * n_nonsepsis / n_patients if n_patients else 0.0

    median_stay = float(meta["max_iculos"].median())

    # --- Missingness on exported rows (union of both tables)
    combined_for_missing = pd.concat([sepsis_out, nonsepsis_out], ignore_index=True)
    n_total_rows = len(combined_for_missing)
    missing_rows = []
    for col in clinical_cols:
        n_present = int(combined_for_missing[col].notna().sum())
        pct_present = 100.0 * n_present / n_total_rows if n_total_rows else 0.0
        missing_rows.append(
            {
                "variable": col,
                "pct_present": round(pct_present, 4),
                "n_present": n_present,
                "n_total": n_total_rows,
            }
        )
    missingness_df = pd.DataFrame(missing_rows)

    OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)
    sepsis_path = OUTPUT_RESULTS / "sepsis_patients.csv"
    nonsepsis_path = OUTPUT_RESULTS / "nonsepsis_patients.csv"
    missingness_path = OUTPUT_RESULTS / "missingness_report.csv"

    sepsis_out.to_csv(sepsis_path, index=False)
    nonsepsis_out.to_csv(nonsepsis_path, index=False)
    missingness_df.to_csv(missingness_path, index=False)

    print(f"Total patients: {n_patients}")
    print(f"Sepsis patients: {n_sepsis} ({pct_sepsis:.2f}%)")
    print(f"Non-sepsis patients: {n_nonsepsis} ({pct_nonsepsis:.2f}%)")
    print(f"Median ICU stay (hours): {median_stay:.1f}")
    print(
        "Sepsis patients with at least 6hrs data before onset: "
        f"{sepsis_ge_6h}"
    )
    print("Missingness (% rows with non-null value):")
    for _, row in missingness_df.sort_values("variable").iterrows():
        print(f"  {row['variable']}: {row['pct_present']:.2f}%")
    print(f"Wrote {sepsis_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {nonsepsis_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {missingness_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run_pipeline()
    print("PHASE 1 COMPLETE")
