"""
Sepsis Early Warning Timeline — Streamlit app (Phase 5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_RESULTS = PROJECT_ROOT / "outputs" / "results"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"

VITALS = {"HR", "Resp", "Temp", "O2Sat", "SBP", "MAP", "DBP"}

# Top-of-timeline variables + other common labs — one sentence each for Deep Dive
CLINICAL_BLURBS: dict[str, str] = {
    "HR": "Heart rate rises with stress, pain, hypovolemia, and infection; sepsis often drives sustained tachycardia as compensation fails.",
    "O2Sat": "Peripheral oxygen saturation falls when oxygen delivery or uptake is impaired — common as sepsis worsens respiratory and circulatory function.",
    "Temp": "Fever or hypothermia reflects systemic inflammatory response; sepsis frequently perturbs thermoregulation.",
    "SBP": "Systolic pressure drops with vasodilation and hypovolemia — hallmarks of progressing septic shock.",
    "MAP": "Mean arterial pressure integrates perfusion pressure; hypotension signals inadequate organ perfusion during sepsis.",
    "DBP": "Diastolic pressure reflects vascular tone and afterload; it may fall as vasodilation dominates in sepsis.",
    "Resp": "Respiratory rate increases early with metabolic acidosis, hypoxia, and the systemic stress response to infection.",
    "HCO3": "Serum bicarbonate buffers acid–base status; it may fall when lactate rises or renal compensation is limited.",
    "FiO2": "Inspired oxygen fraction reflects ventilatory support; higher FiO2 often indicates worsening gas exchange.",
    "BUN": "Blood urea nitrogen rises with reduced renal perfusion, catabolism, and protein breakdown — often elevated early when kidneys are stressed.",
    "Calcium": "Ionized calcium can fall with sepsis, citrate transfusion, and critical illness — relevant to neuromuscular and cardiac function.",
    "Lactate": "Lactate rises with tissue hypoperfusion and anaerobic metabolism — a key marker of shock severity.",
    "WBC": "White blood cell count shifts with infection (leukocytosis or leukopenia) and immunosuppression in critical illness.",
    "Glucose": "Stress hyperglycemia is common in sepsis; hypoglycemia can occur with hepatic dysfunction or severe illness.",
    "Creatinine": "Serum creatinine reflects glomerular filtration; rising values signal acute kidney injury often driven by hypoperfusion and inflammation.",
    "Chloride": "Chloride tracks fluid balance and acid–base status; shifts accompany resuscitation fluids and renal handling.",
    "Bilirubin_total": "Total bilirubin rises with hepatic dysfunction or hemolysis — both seen in severe sepsis.",
    "Phosphate": "Phosphate may fall or rise in critical illness; extremes reflect cellular stress and renal function.",
    "pH": "Arterial pH summarizes acid–base balance; acidemia often accompanies lactate accumulation and shock.",
    "AST": "AST leaks from injured hepatocytes and muscle; elevation suggests organ stress during sepsis.",
    "Alkalinephos": "Alkaline phosphatase may rise with cholestasis or bone turnover in prolonged critical illness.",
    "BaseExcess": "Base excess quantifies metabolic acid–base disturbance relative to buffering capacity.",
    "EtCO2": "End-tidal CO2 reflects ventilation and perfusion; low values can appear when perfusion drops despite ventilation.",
    "PaCO2": "Arterial CO2 reflects alveolar ventilation relative to metabolic production.",
    "SaO2": "Arterial oxygen saturation from blood gas complements pulse oximetry when perfusion is poor.",
    "Magnesium": "Magnesium often falls in critical illness and with renal losses; repletion is common in ICU care.",
    "Potassium": "Potassium shifts with acid–base, renal function, and cellular injury — dangerous extremes are common in ICU.",
    "Platelets": "Platelets may fall in sepsis (consumption/DIC) or rise reactively — both carry prognostic weight.",
    "PTT": "PTT prolongs with coagulopathy; sepsis can trigger consumptive coagulopathies.",
    "Hgb": "Hemoglobin falls with bleeding, hemolysis, or hemodilution during resuscitation.",
    "Hct": "Hematocrit parallels hemoglobin concentration and intravascular volume status.",
    "TroponinI": "Troponin rises with myocardial injury; demand ischemia and sepsis-related cardiomyopathy can elevate it.",
    "Fibrinogen": "Fibrinogen is an acute-phase reactant and coagulation factor; it shifts in inflammation and DIC.",
    "Bilirubin_direct": "Direct bilirubin rises with cholestasis or hepatocellular dysfunction.",
}

GENERIC_BLURB = (
    "This measurement reflects physiology relevant to infection, organ perfusion, and critical illness; "
    "values may shift as sepsis evolves and treatments are applied."
)


@st.cache_resource
def _load_bundle() -> dict:
    """Load CSVs and precompute stats once per session."""
    early = pd.read_csv(OUTPUT_RESULTS / "early_warning_timeline.csv")
    miss = pd.read_csv(OUTPUT_RESULTS / "missingness_report.csv")
    temporal = pd.read_csv(OUTPUT_RESULTS / "temporal_analysis_full.csv")
    shap_imp = pd.read_csv(OUTPUT_RESULTS / "shap_importance.csv")
    shap_vs = pd.read_csv(OUTPUT_RESULTS / "shap_vs_temporal.csv")

    sepsis = pd.read_csv(
        OUTPUT_RESULTS / "sepsis_patients.csv",
        usecols=["patient_id", "is_sepsis", "ICULOS", "onset_hour", "hours_before_onset"],
    )
    non = pd.read_csv(
        OUTPUT_RESULTS / "nonsepsis_patients.csv",
        usecols=["patient_id", "is_sepsis", "ICULOS"],
    )

    n_sep = sepsis["patient_id"].nunique()
    n_non = non["patient_id"].nunique()
    total_patients = n_sep + n_non
    sepsis_rate = n_sep / total_patients if total_patients else 0.0

    stay_s = sepsis.groupby("patient_id")["ICULOS"].max()
    stay_n = non.groupby("patient_id")["ICULOS"].max()
    median_stay = float(pd.concat([stay_s, stay_n]).median())

    return {
        "early": early,
        "miss": miss,
        "temporal": temporal,
        "shap_imp": shap_imp,
        "shap_vs": shap_vs,
        "sepsis_meta": sepsis,
        "non_meta": non,
        "total_patients": total_patients,
        "sepsis_rate": sepsis_rate,
        "median_stay": median_stay,
    }


def _category(var: str) -> str:
    return "Vitals" if var in VITALS else "Labs"


def _lead_style_series(s: pd.Series) -> list[str]:
    out = []
    for v in s:
        if pd.isna(v):
            out.append("")
            continue
        if v > 5:
            out.append("background-color: rgba(192, 57, 43, 0.35); color: #fdf2f2")
        elif v > 3:
            out.append("background-color: rgba(230, 126, 34, 0.35); color: #fff8f0")
        elif v > 1:
            out.append("background-color: rgba(241, 196, 15, 0.35); color: #1a1a1a")
        else:
            out.append("")
    return out


@st.cache_data
def _load_full_sepsis() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_RESULTS / "sepsis_patients.csv")


@st.cache_data
def _load_full_non() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_RESULTS / "nonsepsis_patients.csv")


def _trajectory_figure(variable: str, sepsis_full: pd.DataFrame, non_full: pd.DataFrame) -> go.Figure:
    median_onset = float(
        sepsis_full.groupby("patient_id", sort=False)["onset_hour"].first().median()
    )
    m = int(round(median_onset))
    hbo = sepsis_full["hours_before_onset"].astype(float).round(0)
    hours = list(range(-12, 1))

    mean_s, mean_n, std_s, std_n = [], [], [], []
    xs = []
    for h in hours:
        s = sepsis_full.loc[hbo == float(h), variable].dropna()
        t_icu = m - h
        n = non_full.loc[non_full["ICULOS"] == t_icu, variable].dropna()
        if len(s) == 0 and len(n) == 0:
            continue
        xs.append(h)
        mean_s.append(float(s.mean()) if len(s) else np.nan)
        std_s.append(float(s.std()) if len(s) > 1 else 0.0)
        mean_n.append(float(n.mean()) if len(n) else np.nan)
        std_n.append(float(n.std()) if len(n) > 1 else 0.0)

    up_s = [a + b for a, b in zip(mean_s, std_s)]
    lo_s = [a - b for a, b in zip(mean_s, std_s)]
    up_n = [a + b for a, b in zip(mean_n, std_n)]
    lo_n = [a - b for a, b in zip(mean_n, std_n)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=up_s,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=lo_s,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(52, 152, 219, 0.22)",
            name="Sepsis ±1 SD",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=mean_s,
            mode="lines",
            name="Sepsis (mean)",
            line=dict(color="#3498db", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=up_n,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=lo_n,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(231, 76, 60, 0.18)",
            name="Non-sepsis ±1 SD",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=mean_n,
            mode="lines",
            name="Non-sepsis (mean)",
            line=dict(color="#e74c3c", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=f"{variable}: Sepsis vs Non-Sepsis Trajectory",
        xaxis_title="Hours before onset",
        yaxis_title="Value (raw units)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Sepsis Early Warning Timeline",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    bundle = _load_bundle()
    early = bundle["early"]
    miss = bundle["miss"]
    temporal = bundle["temporal"]
    shap_imp = bundle["shap_imp"]
    shap_vs = bundle["shap_vs"]

    with st.sidebar:
        st.title("Sepsis Early Warning Timeline")
        st.caption("Which clinical signals warn first — and by how many hours?")
        st.divider()
        st.metric("Total patients", f"{bundle['total_patients']:,}")
        st.metric("Sepsis rate", f"{100 * bundle['sepsis_rate']:.2f}%")
        st.metric("Median ICU stay (hours)", f"{bundle['median_stay']:.1f}")
        st.divider()
        st.link_button(
            "PhysioNet 2019 Dataset",
            "https://physionet.org/content/challenge-2019/1.0.0/",
        )
        st.link_button("GitHub (placeholder)", "https://github.com/")
        st.caption("*Research only. Not for clinical use.*")

    page = st.sidebar.radio(
        "Navigate",
        (
            "Early Warning Timeline",
            "Variable Deep Dive",
            "Model vs Timeline",
        ),
    )

    if page == "Early Warning Timeline":
        st.header("Early Warning Timeline")
        st.markdown("The variables that give you the most lead time.")

        st.image(str(OUTPUT_FIGURES / "early_warning_timeline.png"), use_container_width=True)

        df = early.copy()
        df["Lead Time (hours)"] = df["earliest_warning_hours"]
        df["Effect Size"] = df["max_effect_size"]
        df["Variable"] = df["variable"]
        df["Category"] = df["Variable"].map(_category)
        show = df[["Variable", "Lead Time (hours)", "Effect Size", "Category"]]

        cat_filter = st.selectbox("Filter by category", ["All", "Vitals", "Labs"])
        if cat_filter != "All":
            show = show[show["Category"] == cat_filter]

        styler = show.style.apply(
            lambda col: _lead_style_series(col) if col.name == "Lead Time (hours)" else [""] * len(col),
            axis=0,
        )
        st.dataframe(styler, use_container_width=True, hide_index=True)

        top_lead = early["earliest_warning_hours"].max()
        top10 = early.sort_values(
            ["earliest_warning_hours", "max_effect_size"],
            ascending=[False, False],
            na_position="last",
        ).head(10)
        n_labs_top10 = sum(1 for v in top10["variable"] if _category(v) == "Labs")

        st.info(
            f"The variable with the longest lead time gives clinicians **{top_lead:.0f}** hours to intervene "
            f"before sepsis onset. **{n_labs_top10}** of the top 10 early warnings are lab values — "
            "but lab values are ordered far less frequently than vitals."
        )

    elif page == "Variable Deep Dive":
        st.header("Variable Deep Dive")

        clinical_cols = sorted(
            [c for c in early["variable"].tolist() if isinstance(c, str)],
            key=str.lower,
        )
        variable = st.selectbox("Select variable", clinical_cols)

        mrow = miss.loc[miss["variable"] == variable]
        pct_present = float(mrow["pct_present"].iloc[0]) if not mrow.empty else float("nan")

        tsub = temporal[temporal["variable"] == variable]
        if not tsub.empty:
            sig_rows = tsub.loc[tsub["significant_corrected"]]
            earliest_h = int(sig_rows["hour"].min()) if not sig_rows.empty else None
            tsub = tsub.copy()
            tsub["_eff"] = (tsub["cles"] - 0.5).abs()
            max_row = tsub.loc[tsub["_eff"].idxmax()]
            max_cles = float(max_row["cles"])
            max_hour = int(max_row["hour"])
        else:
            earliest_h = None
            max_cles = float("nan")
            max_hour = None

        erow = early.loc[early["variable"] == variable]
        lead_h = float(erow["earliest_warning_hours"].iloc[0]) if not erow.empty else float("nan")

        left, right = st.columns(2)
        with left:
            st.subheader("Summary")
            st.write(f"**Earliest significant lead time (hours):** {lead_h:.1f}" if pd.notna(lead_h) else "**Earliest lead time:** —")
            st.write(
                f"**Max effect size (CLES distance from 0.5):** {abs(max_cles - 0.5) * 2:.3f} at hour **{max_hour}**"
                if pd.notna(max_cles) and max_hour is not None
                else "**Max effect size:** —"
            )
            st.write(f"**% data present:** {pct_present:.2f}%" if pd.notna(pct_present) else "**% data present:** —")
            st.write("**Clinical note:**")
            st.write(CLINICAL_BLURBS.get(variable, GENERIC_BLURB))

        sepsis_full = _load_full_sepsis()
        non_full = _load_full_non()

        with right:
            fig_tr = _trajectory_figure(variable, sepsis_full, non_full)
            if earliest_h is not None:
                fig_tr.add_vline(
                    x=earliest_h,
                    line_width=2,
                    line_dash="dot",
                    line_color="white",
                    annotation_text="Earliest sig. (corr.)",
                    annotation_position="top",
                )
            st.plotly_chart(fig_tr, use_container_width=True)

        st.subheader("Effect size by hour (this variable)")
        hours = list(range(-12, 0))
        cles_vals = []
        for h in hours:
            r = tsub[tsub["hour"] == h]
            cles_vals.append(float(r["cles"].iloc[0]) if not r.empty else float("nan"))
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=[str(h) for h in hours],
                    y=cles_vals,
                    marker_color="#e67e22",
                )
            ]
        )
        fig_bar.update_layout(
            template="plotly_dark",
            title=f"{variable}: CLES by hour",
            xaxis_title="Hour before onset",
            yaxis_title="CLES",
            height=360,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.header("Does the Model Use What Warns Earliest?")
        st.markdown("Comparing SHAP importance vs temporal lead time.")

        st.image(str(OUTPUT_FIGURES / "shap_vs_temporal.png"), use_container_width=True)

        c1, c2 = st.columns(2)
        top_shap = shap_imp.head(15)
        top_temp = early.sort_values(
            ["earliest_warning_hours", "max_effect_size"],
            ascending=[False, False],
            na_position="last",
        ).head(15)

        with c1:
            st.subheader("SHAP importance (top 15)")
            fig_s = go.Figure(
                go.Bar(
                    x=top_shap["mean_abs_shap"],
                    y=top_shap["variable"],
                    orientation="h",
                    marker_color="#3498db",
                )
            )
            fig_s.update_layout(
                template="plotly_dark",
                height=480,
                margin=dict(l=120, r=20, t=30, b=40),
                xaxis_title="Mean |SHAP|",
            )
            st.plotly_chart(fig_s, use_container_width=True)

        with c2:
            st.subheader("Temporal lead time (top 15)")
            fig_t = go.Figure(
                go.Bar(
                    x=top_temp["earliest_warning_hours"],
                    y=top_temp["variable"],
                    orientation="h",
                    marker_color="#e67e22",
                )
            )
            fig_t.update_layout(
                template="plotly_dark",
                height=480,
                margin=dict(l=120, r=20, t=30, b=40),
                xaxis_title="Lead time (hours)",
            )
            st.plotly_chart(fig_t, use_container_width=True)

        def _highlight_disagree(row: pd.Series) -> list[str]:
            d = row.get("rank_difference", np.nan)
            if pd.notna(d) and d > 10:
                return ["background-color: rgba(192, 57, 43, 0.35)"] * len(row)
            return [""] * len(row)

        disp = shap_vs.copy()
        if "rank_difference" not in disp.columns:
            disp["rank_difference"] = (disp["shap_rank"] - disp["temporal_rank"]).abs()
        styled = disp.style.apply(_highlight_disagree, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.info(
            "Variables the model relies on most are not always the ones that warn earliest. This matters for "
            "clinical protocol design — if a variable warns 6 hours early but the model underweights it, "
            "clinicians may not be prompted to check it in time."
        )


if __name__ == "__main__":
    main()
