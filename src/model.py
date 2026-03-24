"""
Phase 3 — Binary sepsis classifier at T = -6 hours + SHAP vs temporal comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_RESULTS = PROJECT_ROOT / "outputs" / "results"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"

SEPSIS_CSV = OUTPUT_RESULTS / "sepsis_patients.csv"
NONSEPSIS_CSV = OUTPUT_RESULTS / "nonsepsis_patients.csv"
TIMELINE_CSV = OUTPUT_RESULTS / "early_warning_timeline.csv"

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

SNAPSHOT_T = -6
RANDOM_STATE = 42


def _clinical_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in EXCLUDE]
    if len(cols) < 34:
        raise ValueError(f"Expected at least 34 clinical columns; got {len(cols)}")
    return cols


def _build_snapshot_matrix() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    sepsis = pd.read_csv(SEPSIS_CSV)
    nonsepsis = pd.read_csv(NONSEPSIS_CSV)
    feat_cols = _clinical_columns(sepsis)

    median_onset = float(sepsis.groupby("patient_id", sort=False)["onset_hour"].first().median())
    median_iculos = int(round(median_onset))

    hbo = sepsis["hours_before_onset"].astype(float).round(0)
    sepsis_snap = sepsis.loc[hbo == float(SNAPSHOT_T)].drop_duplicates(
        subset=["patient_id"], keep="first"
    )

    target_iculos = median_iculos - SNAPSHOT_T
    non_snap = nonsepsis.loc[nonsepsis["ICULOS"] == target_iculos].drop_duplicates(
        subset=["patient_id"], keep="first"
    )

    X_pos = sepsis_snap[feat_cols].copy()
    y_pos = pd.Series(1, index=X_pos.index, dtype=np.int8)

    X_neg = non_snap[feat_cols].copy()
    y_neg = pd.Series(0, index=X_neg.index, dtype=np.int8)

    X = pd.concat([X_pos, X_neg], axis=0, ignore_index=True)
    y = pd.concat([y_pos, y_neg], axis=0, ignore_index=True)
    return X, y, feat_cols


def _best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    thr = np.append(thr, 1.0)
    best_f1, best_t = 0.0, 0.5
    for t in np.unique(np.concatenate([[0.0], thr, np.linspace(0, 1, 101)])):
        pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def run_model() -> None:
    if not SEPSIS_CSV.is_file() or not NONSEPSIS_CSV.is_file():
        raise FileNotFoundError("Run Phase 1 first (sepsis / nonsepsis CSVs missing).")
    if not TIMELINE_CSV.is_file():
        raise FileNotFoundError("Run Phase 2 first (early_warning_timeline.csv missing).")

    X, y, feat_cols = _build_snapshot_matrix()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)
    X_train_df = pd.DataFrame(X_train_i, columns=feat_cols)
    X_test_df = pd.DataFrame(X_test_i, columns=feat_cols)

    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    if n_pos == 0:
        raise ValueError("No positive samples in training set.")
    spw = n_neg / n_pos

    model = xgb.XGBClassifier(
        max_depth=4,
        n_estimators=200,
        learning_rate=0.05,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train_df, y_train)

    y_proba = model.predict_proba(X_test_df)[:, 1]
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)
    thr, f1_opt = _best_f1_threshold(y_test.to_numpy(), y_proba)

    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")
    print(f"F1 at optimal threshold ({thr:.3f}): {f1_opt:.4f}")

    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_df)

    vals = shap_values.values
    mean_abs = np.abs(vals).mean(axis=0)
    imp = pd.DataFrame(
        {"variable": feat_cols, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)
    imp["rank"] = np.arange(1, len(imp) + 1)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, show=False, plot_size=(10, 8))
    shap_path = OUTPUT_FIGURES / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close()

    imp_path = OUTPUT_RESULTS / "shap_importance.csv"
    imp.to_csv(imp_path, index=False)

    tl = pd.read_csv(TIMELINE_CSV)
    tl = tl.sort_values(
        ["earliest_warning_hours", "max_effect_size"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    tl["temporal_rank"] = np.arange(1, len(tl) + 1)

    comp = imp[["variable", "rank"]].merge(
        tl[["variable", "temporal_rank"]],
        on="variable",
        how="left",
    )
    comp = comp.rename(columns={"rank": "shap_rank"})
    comp["rank_difference"] = (comp["shap_rank"] - comp["temporal_rank"]).abs()
    comp["agreement_within_5"] = comp["rank_difference"] <= 5

    comp_path = OUTPUT_RESULTS / "shap_vs_temporal.csv"
    comp.to_csv(comp_path, index=False)

    print(f"Wrote {shap_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {imp_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {comp_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run_model()
    print("PHASE 3 COMPLETE")
