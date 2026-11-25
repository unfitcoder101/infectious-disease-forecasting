"""
Public research dashboard — How Much Surveillance History Is Enough?

READ-ONLY. This app never re-runs the experiment and never writes to
data/ or results/. It only loads and displays the CSVs and PNGs already
committed by run_experiment.py, exactly as they are on disk.

Run locally:
    pip install -r requirements.txt -r requirements-dashboard.txt
    streamlit run streamlit_app.py
"""

from pathlib import Path

import altair as alt  # transitive dependency of streamlit
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (read-only). Same Path(__file__)-relative convention used throughout
# this project's src/ and run_experiment.py.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Singapore external validation: its own directory, never mixed with results/.
SG_DIR = BASE_DIR / "singapore_analysis"
SG_RESULTS_DIR = SG_DIR / "results"
SG_FIG_DIR = SG_DIR / "figures"
SG_TABLES_DIR = SG_DIR / "tables"

# Brazil external validation: separate immutable snapshot and outputs.
BR_DIR = BASE_DIR / "brazil_analysis"
BR_RESULTS_DIR = BR_DIR / "results"
BR_FIG_DIR = BR_RESULTS_DIR / "figures"

# Instructor-requested level-versus-change follow-up (read-only outputs).
FOLLOWUP_DIR = BASE_DIR / "followup_experiments"
FOLLOWUP_RESULTS_DIR = FOLLOWUP_DIR / "results"
FOLLOWUP_FIG_DIR = FOLLOWUP_DIR / "figures"

# Same categorical hues already validated and used in every existing figure
# in results/figures/ (src/visualization.py SERIES_COLORS) -- reused here,
# never re-picked, so native charts and embedded PNGs read as one system.
XGB_COLOR = "#2a78d6"
LGBM_COLOR = "#eb6834"
SERIES_COLORS = {"XGBoost": XGB_COLOR, "LightGBM": LGBM_COLOR}

CITIES = {"sj": "San Juan, Puerto Rico", "iq": "Iquitos, Peru"}


# ---------------------------------------------------------------------------
# Cached, read-only loaders. Every one just reads a file that already exists;
# none of them compute anything or write anything back.
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


@st.cache_data
def load_processed(city: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"{city}_weekly_cases.csv", parse_dates=["date"])
    return df.set_index("date")


@st.cache_data
def load_sg_csv(name: str) -> pd.DataFrame:
    """Singapore result table from singapore_analysis/results/ (read-only)."""
    return pd.read_csv(SG_RESULTS_DIR / name)


@st.cache_data
def load_br_csv(name: str) -> pd.DataFrame:
    """Brazil external-validation result table (read-only)."""
    return pd.read_csv(BR_RESULTS_DIR / name)


@st.cache_data
def load_followup_csv(name: str) -> pd.DataFrame:
    """Instructor follow-up result table (read-only)."""
    return pd.read_csv(FOLLOWUP_RESULTS_DIR / name)


@st.cache_data
def load_sg_table(name: str) -> pd.DataFrame:
    """Singapore audit/EDA/protocol table from singapore_analysis/tables/ (read-only)."""
    return pd.read_csv(SG_TABLES_DIR / name)


def br_fig(name: str):
    """Path to an existing Brazil external-validation figure."""
    p = BR_FIG_DIR / name
    if not p.exists():
        st.error(f"Expected figure not found: {p}")
    return p


def followup_fig(name: str):
    """Path to an existing level/change relative-MAE figure."""
    p = FOLLOWUP_FIG_DIR / name
    if not p.exists():
        st.error(f"Expected figure not found: {p}")
    return p


def sg_fig(name: str):
    """Path to an existing figure in singapore_analysis/figures/."""
    p = SG_FIG_DIR / name
    if not p.exists():
        st.error(f"Expected figure not found: {p}")
    return p


def fig(name: str):
    """Path to an existing figure in results/figures/. Raises if missing --
    fail loudly rather than silently skip a supposedly-existing artifact."""
    p = FIG_DIR / name
    if not p.exists():
        st.error(f"Expected figure not found: {p}")
    return p


def model_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """Grouped (side-by-side) bar chart for a metric by history_window, one bar
    per model per window, using the SAME fixed hue order as every static figure
    in this project (XGBoost blue, LightGBM orange -- never swapped, never
    re-derived from data order).

    Explicit Altair spec rather than st.bar_chart, which stacks the series,
    sorts string x labels alphabetically and binds the axes to scroll-zoom
    (zooming clips every bar taller than the visible range to the same
    height). Here windows are numeric and sorted, the y axis starts at zero
    and the chart is not zoomable."""
    d = df[df["model"].isin(SERIES_COLORS)][[x_col, "model", y_col]].copy()
    d[x_col] = d[x_col].astype(int)
    d = d.sort_values([x_col, "model"])
    windows = sorted(d[x_col].unique().tolist())
    st.altair_chart(
        alt.Chart(d).mark_bar().encode(
            x=alt.X(f"{x_col}:O", sort=windows, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("model:N", sort=list(SERIES_COLORS)),
            y=alt.Y(f"{y_col}:Q", title=None, scale=alt.Scale(domainMin=0)),
            color=alt.Color("model:N", title=None, sort=list(SERIES_COLORS),
                            scale=alt.Scale(domain=list(SERIES_COLORS),
                                            range=list(SERIES_COLORS.values())),
                            legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip(f"{x_col}:O", title="Window"),
                     alt.Tooltip("model:N", title="Model"),
                     alt.Tooltip(f"{y_col}:Q", title=y_col, format=".2f")],
        ).properties(height=320),
        width="stretch",
    )
    st.caption(title)


# ---------------------------------------------------------------------------
# Page config + light theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Surveillance History Sensitivity — Dengue Forecasting",
    page_icon="🦟",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .research-banner {
        background: #eef4fc; border-left: 4px solid #2a78d6;
        padding: 0.9rem 1.2rem; border-radius: 6px; margin-bottom: 1rem;
    }
    .finding-box {
        background: #fff7ec; border-left: 4px solid #eb6834;
        padding: 1rem 1.3rem; border-radius: 6px;
    }
    .scope-note {
        font-size: 0.85rem; color: #6b6b6b; font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🦟 How Much Surveillance History Is Enough?")
st.markdown(
    "##### A Temporal Context Sensitivity Study for Short-Term Infectious Disease Forecasting"
)

st.markdown(
    """
    <div class="research-banner">
    <b>Research question:</b> how many weeks of past dengue surveillance data are
    actually needed to forecast <i>next week's</i> case count? This dashboard compares
    history windows of <b>1, 2, 4, 8, and 12 weeks</b>, using XGBoost and LightGBM,
    on real weekly dengue surveillance data. Model settings are held fixed across
    windows on purpose — the question is about history <i>length</i>, not about
    finding the best-tuned model.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: reproduce + navigation reminder + scope disclaimer
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📎 Reproduce this research")
    st.markdown(
        """
        ```bash
        git clone https://github.com/unfitcoder101/infectious-disease-forecasting.git
        cd infectious-disease-forecasting
        python3 -m venv .venv && source .venv/bin/activate
        pip install -r requirements.txt
        python run_experiment.py            # San Juan (default)
        python run_experiment.py --city iq  # Iquitos
        python followup_experiments/run_change_forecast.py --execute  # final four-series study
        ```
        """
    )
    st.markdown(
        "Full methodology, all limitations, and every number on this page: "
        "[`research_notes.md`](https://github.com/unfitcoder101/infectious-disease-forecasting/blob/main/research_notes.md) "
        "and [`README.md`](https://github.com/unfitcoder101/infectious-disease-forecasting/blob/main/README.md)."
    )
    st.divider()
    st.warning(
        "⚠️ **No history window is claimed optimal anywhere in this study.** "
        "See the Conclusion tab.",
        icon="⚠️",
    )
    st.caption(
        "This dashboard only *displays* already-committed results. It never "
        "re-runs the experiment or writes to `data/` or `results/`."
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_single, tab_cv, tab_fine, tab_conclusion, tab_sg, tab_br, tab_followup = st.tabs(
    [
        "📋 Overview",
        "📊 Single-Split (1/2/4/8/12 wk)",
        "🔁 Rolling-Origin CV (10-fold)",
        "🎯 Targeted Follow-up (21-fold)",
        "✅ Conclusion & Limitations",
        "🌏 Singapore External Validation",
        "🇧🇷 Brazil External Validation",
        "↔ Level vs Change Follow-up",
    ]
)

# ===========================================================================
# TAB 1 — OVERVIEW
# ===========================================================================
with tab_overview:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Dataset & methodology")
        st.markdown(
            """
            - **Source:** NOAA / CDC Dengue Forecasting Project (2015), distributed via
              the DrivenData "DengAI" dataset. Real surveillance data, not synthetic.
            - **Primary series:** San Juan, Puerto Rico — 936 weekly observations,
              1990-04-30 → 2008-04-22 (18.0 years).
            - **Lag features:** for a window of *W* weeks, `X = [lag_1 … lag_W]`
              (lag_1 = most recent week), `y` = next week's cases. All windows are
              aligned to the **same 924 target weeks**, so history length is the
              only thing that differs between conditions.
            - **Splitting:** strictly chronological — never random — to avoid
              training on weeks that come after the weeks being tested.
            - **Models:** XGBoost and LightGBM, *identical fixed hyperparameters*
              across every window (300 trees, lr 0.05, max_depth 4, seed 42).
            - **Baseline:** persistence — predict next week = this week.
            """
        )
        st.caption(
            "Full provenance, checksums, and the data-quality audit are in "
            "`data/raw/SOURCE.md` and research_notes.md §2."
        )

    with col2:
        st.subheader("Raw weekly case counts")
        city_key = st.selectbox(
            "City",
            options=list(CITIES),
            format_func=lambda k: CITIES[k],
            help="Preview only. Model-result tabs are San Juan only — see note below.",
        )
        series = load_processed(city_key)
        st.line_chart(series["cases"], color=XGB_COLOR, height=280)
        st.caption(
            f"{CITIES[city_key]} — {len(series)} weekly observations, "
            f"{series.index.min().date()} → {series.index.max().date()}."
        )
        st.markdown(
            '<p class="scope-note">This selector affects ONLY this raw case-count '
            "preview. Every model-comparison tab below (single-split, rolling-origin "
            "CV, targeted follow-up) uses <b>San Juan only</b> — those are the only "
            "results persisted as structured CSVs in this repository. The Iquitos "
            "secondary check described in research_notes.md §4.4 is documented as "
            "prose there and is intentionally not reconstructed here as charts.</p>",
            unsafe_allow_html=True,
        )

# ===========================================================================
# TAB 2 — SINGLE-SPLIT RESULTS
# ===========================================================================
with tab_single:
    st.info("📍 **Scope: San Juan only**, one chronological 70/15/15 split.", icon="📍")

    results = load_csv("results_table.csv")
    baseline_mae = results.loc[results["model"] == "Persistence (baseline)", "MAE"].iloc[0]
    model_results = results[results["model"] != "Persistence (baseline)"]

    best_row = model_results.loc[model_results["MAE"].idxmin()]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lowest observed test MAE", f"{best_row['MAE']:.2f} cases",
              help="Lowest number observed in this table. NOT a claim this window is proven best.")
    k2.metric("↳ from window / model", f"{int(best_row['history_window'])} wk · {best_row['model']}")
    k3.metric("Persistence baseline MAE", f"{baseline_mae:.2f} cases")
    n_beat = int((model_results["MAE"] < baseline_mae).sum())
    k4.metric("Configs beating baseline", f"{n_beat} / {len(model_results)}")
    st.caption(
        "⚠️ \"Lowest observed\" describes this one table. It is not evidence the "
        "window is *meaningfully* better — see the Targeted Follow-up tab, which "
        "tests exactly that."
    )

    st.subheader("MAE by history window — XGBoost vs LightGBM")
    model_bar(model_results, "history_window", "MAE",
              "Dashed reference in the static figure below shows the persistence baseline.")
    st.image(str(fig("mae_vs_history_window.png")), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.image(str(fig("rmse_vs_history_window.png")), width="stretch")
    with c2:
        st.image(str(fig("r2_vs_history_window.png")), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.image(str(fig("training_time_vs_history_window.png")), width="stretch")
    with c4:
        st.image(str(fig("inference_time_vs_history_window.png")), width="stretch")

    st.subheader("Actual vs. predicted, held-out test period")
    st.image(str(fig("forecast_actual_vs_predicted.png")), width="stretch")

    with st.expander("Full results table"):
        st.dataframe(results, width="stretch", hide_index=True)

    with st.expander("Robustness: test period vs. validation period (§4.2)"):
        st.image(str(fig("robustness_two_periods.png")), width="stretch")
        st.dataframe(load_csv("robustness_check.csv"), width="stretch", hide_index=True)
        st.caption(
            "The pre-registered minimum-sufficient-window rule gives a DIFFERENT "
            "answer on the two periods for both models — the reason a single split "
            "cannot resolve this on its own, and why rolling-origin CV (next tab) "
            "was run."
        )

# ===========================================================================
# TAB 3 — ROLLING-ORIGIN CV (10-FOLD / 26-WEEK)
# ===========================================================================
with tab_cv:
    st.info(
        "📍 **Scope: San Juan only.** 10 rolling-origin (walk-forward) folds, "
        "26-week non-overlapping test blocks, expanding training window "
        "(min_train = 646 weeks, derived from the actual 924-row aligned series).",
        icon="📍",
    )

    cv_summary = load_csv("cv_summary.csv")
    cv_model_summary = cv_summary[cv_summary["model"] != "Persistence (baseline)"]
    cv_mins = load_csv("cv_minimum_sufficient_window.csv")
    cv_wins = load_csv("cv_fold_win_counts.csv")

    best_cv = cv_model_summary.loc[cv_model_summary["MAE_mean"].idxmin()]
    k1, k2, k3 = st.columns(3)
    k1.metric("Lowest observed CV mean MAE", f"{best_cv['MAE_mean']:.2f} ± {best_cv['MAE_std']:.2f}",
              help="Mean ± std across 10 folds. The std is large relative to the mean -- see below.")
    k2.metric("↳ from window / model", f"{int(best_cv['history_window'])} wk · {best_cv['model']}")
    best_win_rate = cv_wins["win_rate_pct"].max()
    k3.metric("Highest single-window fold win rate", f"{best_win_rate:.0f}%",
              help="No window wins a majority of folds for either model.")

    st.subheader("CV mean MAE ± 1 std, by window")
    st.image(str(fig("cv_mae_vs_history_window.png")), width="stretch")
    st.caption(
        "Error bars overlapping across windows means the difference between them "
        "is not distinguishable from fold-to-fold noise."
    )

    st.image(str(fig("cv_fold_detail.png")), width="stretch")
    st.caption(
        "Every individual fold's MAE (faint dots). The handful of high-MAE outlier "
        "folds are the ones containing an outbreak — the dominant source of "
        "variance is which folds catch an outbreak, not history-window length."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Mean ± std MAE per window**")
        st.dataframe(
            cv_model_summary[["history_window", "model", "MAE_mean", "MAE_std", "n_folds"]]
            .sort_values(["model", "history_window"]),
            width="stretch", hide_index=True,
        )
    with col2:
        st.markdown("**Fold win counts** (lowest MAE that fold)")
        st.dataframe(
            cv_wins.sort_values(["model", "history_window"]),
            width="stretch", hide_index=True,
        )

    with st.expander("Pre-registered minimum-sufficient-window rule, applied to CV means"):
        st.dataframe(cv_mins, width="stretch", hide_index=True)
        st.caption(
            "Windows 1, 2, and 4 all fall within the 5% tolerance band for both "
            "models, and the shortest is within one fold-level standard deviation "
            "of the best — i.e. not statistically distinguishable at this fold count."
        )

# ===========================================================================
# TAB 4 — TARGETED FOLLOW-UP (21-FOLD / 13-WEEK, WINDOWS {1,2,4})
# ===========================================================================
with tab_fine:
    st.info(
        "📍 **Scope: San Juan only**, windows **{1, 2, 4}** only (8 and 12 were "
        "already resolved as worse and were not re-tested). 21 rolling-origin "
        "folds, 13-week non-overlapping test blocks, same 924-row aligned base "
        "and 646-week min_train as the 10-fold CV.",
        icon="📍",
    )

    fine_summary = load_csv("fine_cv_summary.csv")
    fine_model_summary = fine_summary[fine_summary["model"] != "Persistence (baseline)"]
    fine_wins = load_csv("fine_cv_fold_win_counts.csv")
    paired = load_csv("fine_cv_paired_diff_summary.csv")

    lgbm_best = fine_model_summary[fine_model_summary["model"] == "LightGBM"].loc[
        fine_model_summary[fine_model_summary["model"] == "LightGBM"]["MAE_mean"].idxmin()
    ]
    xgb_best = fine_model_summary[fine_model_summary["model"] == "XGBoost"].loc[
        fine_model_summary[fine_model_summary["model"] == "XGBoost"]["MAE_mean"].idxmin()
    ]
    k1, k2, k3 = st.columns(3)
    k1.metric("LightGBM lowest observed mean", f"{lgbm_best['MAE_mean']:.2f} (window {int(lgbm_best['history_window'])})")
    k2.metric("XGBoost lowest observed mean", f"{xgb_best['MAE_mean']:.2f} (window {int(xgb_best['history_window'])})")
    k3.metric("Models agree on best window?", "No" if lgbm_best["history_window"] != xgb_best["history_window"] else "Yes")
    st.caption(
        "⚠️ These are the lowest **observed** means only. The paired comparison "
        "below — built specifically to test whether either is *meaningfully* "
        "better — found no consistent winner. See the Conclusion tab."
    )

    st.subheader("Mean ± std MAE (21-fold CV), windows {1, 2, 4}")
    st.dataframe(
        fine_model_summary[["history_window", "model", "MAE_mean", "MAE_std", "n_folds"]]
        .sort_values(["model", "history_window"]),
        width="stretch", hide_index=True,
    )

    st.subheader("Fold win counts (21 folds)")
    st.dataframe(fine_wins.sort_values(["model", "history_window"]), width="stretch", hide_index=True)

    st.subheader("Paired per-fold MAE differences (the decisive evidence)")
    st.markdown(
        "Because all three windows share **identical test weeks within every fold**, "
        "each fold's MAE for window A and window B can be subtracted directly, "
        "cancelling the noise common to both (e.g. whether that fold contains an "
        "outbreak). A negative value means the shorter window (A) had lower error "
        "that fold."
    )
    st.image(str(fig("fine_cv_paired_differences.png")), width="stretch")
    st.caption(
        "Every error bar (±1 std) crosses zero, for all three pairs and both models "
        "— the direct visual evidence that the paired comparisons remain noisy with "
        "no consistent direction."
    )
    st.dataframe(paired, width="stretch", hide_index=True)

    with st.expander("Pre-registered minimum-sufficient-window rule, applied to the 21-fold CV means"):
        st.dataframe(load_csv("fine_cv_minimum_sufficient_window.csv"), width="stretch", hide_index=True)

# ===========================================================================
# TAB 5 — CONCLUSION & LIMITATIONS
# ===========================================================================
with tab_conclusion:
    st.subheader("Final Research Finding")
    st.markdown(
        """
        <div class="finding-box">

**Confident:** short surveillance history (1–4 weeks) is consistently at least
as good as long history (8–12 weeks) for next-week dengue forecasting on this
dataset — across a single chronological test split, a 10-fold rolling-origin
CV, and a targeted 21-fold follow-up — while costing roughly 3× less to train.

**Not confident, and more clearly so after the targeted follow-up:** the exact
minimum sufficient window within {1, 2, 4} weeks. Two independent CV designs
and a paired fold-level comparison — built specifically to isolate the
window-specific signal from fold-level noise — all failed to produce a
consistent winner.

**This study therefore identifies a short-history *regime* (1–4 weeks) rather
than an exact minimum sufficient window.** No particular window — not 1, not 2,
not 4 — is described as optimal anywhere in this research, for this dataset or
in general.

</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.subheader("Lowest observed ≠ proven optimal — stated once more, explicitly")
    st.markdown(
        """
        | | Lowest **observed** mean MAE | Meaningfully better? |
        |---|---|---|
        | LightGBM | window 4 (6.01) | **No** — paired comparisons vs. windows 1 and 2 both cross zero |
        | XGBoost | window 2 (6.27) | **No** — paired comparisons vs. windows 1 and 4 both cross zero |

        The two models don't even agree with each other on which window has the
        lowest observed mean — itself evidence against a stable, shared effect,
        not just noisy estimates of the same true answer.
        """
    )

    st.markdown("")
    st.subheader("Limitations")
    with st.expander("Show all limitations (from research_notes.md §8)", expanded=False):
        st.markdown(
            """
            1. **One disease, one dataset family.** San Juan dengue 1990–2008
               (primary) plus an Iquitos secondary check. Nothing here
               generalises to other diseases or eras.
            2. **A single train/test cut was the original design; rolling-origin
               CV addresses this**, but did not fully resolve which of {1, 2, 4}
               is shortest-sufficient (fold-to-fold std ≈62–89% of mean MAE).
            3. **Fixed hyperparameters by design.** Result says "more lags did
               not help *at fixed capacity*," not "more lags cannot help."
            4. **Only one-step-ahead (t+1) forecasting** was tested. Longer
               history may matter more at longer horizons.
            5. **No climate covariates**, deliberately — only the disease's own past.
            6. **Raw counts, untransformed.** A log/sqrt transform would change
               the error profile.
            7. **R² is unreliable as a per-fold CV metric at 26-week blocks**
               (fold R² ranged −1.06 to 0.84) — a quiet block has little
               variance to divide by. MAE/RMSE are the primary metrics.
            8. **The 8–10 fold CV target was calibrated to San Juan** (924 rows);
               Iquitos (508 rows) supports at most 7 folds at this block size.
            9. **CV variance itself was measured on one series.**
            10. **R² is even less reliable at 13-week blocks** (fold R² ranged
                −1.48 to 0.82).
            11. **The 21 fine-grained folds are different slices of the same
                ~5.3-year period, not 21 independent datasets** — `min_train`
                stayed at 646, so no new historical information was added.
            12. **Seasonal-phase heterogeneity** — 13-week blocks don't reliably
                span a full rise-and-fall of the ~52-week dengue cycle the way
                26-week blocks tend to.
            13. **The targeted follow-up adds no new independent disease/city
                replication** — same San Juan series, re-parameterized.
            14. **The six paired comparisons are descriptive**, not formal
                hypothesis tests with a multiplicity correction.
            15. **This entire result is specific to San Juan dengue, this
                preprocessing, these two models/hyperparameters, this
                one-step-ahead horizon, and this evaluation design.**
            16. **Singapore external validation exists, but is not a
                replication.** (Added after §8; see `research_notes.md` §10
                and the Singapore tab.) It uses a different case definition
                (MOH Dengue Fever notifications, DHF excluded), so its errors
                are not comparable with the San Juan or Iquitos numbers. Its
                evaluation period (2019-W41 to 2022-W52) includes the 2020 and
                2022 epidemics.
            """
        )
    st.caption("Verbatim from the committed research_notes.md §8 — see the repository for the full document.")

# ===========================================================================
# TAB 6 — SINGAPORE EXTERNAL VALIDATION
# Reads ONLY from singapore_analysis/{results,figures,tables}. Nothing above
# this line is affected by anything in this block.
# ===========================================================================
with tab_sg:
    st.markdown(
        """
        <div class="research-banner">
        <b>External validation, not replication.</b> Singapore MOH dengue fever
        (DHF excluded), 2012-2022, different case definition from DengAI.
        Evaluation period 2019-W41 to 2022-W52 includes the 2020 and 2022 epidemics.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "📍 **Scope: Singapore only.** Same windows, models, fixed hyperparameters, "
        "persistence baseline and 5% rule as the other tabs. 13 rolling-origin folds "
        "of 13 weeks (the design that carries the statistical comparisons) and 6 folds "
        "of 26 weeks (descriptive only). Errors are in cases and Singapore's counts are "
        "about 11× San Juan's, so **no number on this tab is comparable with the other tabs.**",
        icon="📍",
    )

    PERSIST = "Persistence (baseline)"
    sg13 = load_sg_csv("sg_cv13_extended_summary.csv")
    sg13_models = sg13[sg13["model"] != PERSIST]
    sg13_persist_mae = sg13.loc[sg13["model"] == PERSIST, "MAE_mean"].iloc[0]
    sg13_cmp = load_sg_csv("sg_cv13_comparisons_mae.csv")
    sg13_folds = load_sg_csv("sg_cv13_results.csv")
    sg_primary = sg13_cmp[sg13_cmp["family"].str.startswith("primary")]
    sg_vs_persist = sg13_cmp[sg13_cmp["family"].str.contains("persistence")]
    sg_pairs = sg13_cmp[sg13_cmp["family"].str.contains("window pairs")]

    sg_best = sg13_models.loc[sg13_models["MAE_mean"].idxmin()]
    sg_n_beat = int((sg13_models["MAE_mean"] < sg13_persist_mae).sum())
    sg_p_lo, sg_p_hi = sg_primary["wilcoxon_p_holm"].min(), sg_primary["wilcoxon_p_holm"].max()
    sg_p_text = f"{sg_p_hi:.2f}" if round(sg_p_lo, 2) == round(sg_p_hi, 2) else f"{sg_p_lo:.2f}–{sg_p_hi:.2f}"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Lowest observed CV mean MAE", f"{sg_best['MAE_mean']:.2f} cases",
              help="Lowest mean across 13 folds. NOT a claim this window is proven best.")
    k2.metric("↳ from window / model", f"{int(sg_best['history_window'])} wk · {sg_best['model']}")
    k3.metric("Persistence baseline MAE", f"{sg13_persist_mae:.2f} cases")
    k4.metric("Configs beating persistence", f"{sg_n_beat} / {len(sg13_models)}")
    k5.metric("Short vs long, Holm-adjusted p", sg_p_text,
              help="Windows {1, 2, 4} against {8, 12}, paired by fold, Wilcoxon signed-rank, "
                   "Holm-adjusted across the two models. Not significant.")
    st.caption(
        "⚠️ \"Lowest observed\" describes these 13 folds. It is not evidence the "
        "window is *meaningfully* better — see the table further down, which keeps "
        "the two apart."
    )

    st.markdown(
        f"""
        <div class="finding-box">

**Short history was directionally similar to San Juan but NOT statistically
separable from long history (p = {sg_p_text}).** Under the pre-registered protocol,
the 5% rule selected 1 week for both models in both CV designs, and
windows 8 and 12 were never near-optimal. The paired test, however, does not
distinguish short windows from long ones.

**Persistence beat every model.** All {len(sg13_models)} model/window configurations
had higher mean MAE than "next week = this week".

**No window is claimed optimal**, for Singapore or in general.

</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.subheader("The series")
    st.image(str(sg_fig("01_weekly_time_series.png")), width="stretch")
    st.caption(
        "574 weekly observations, 2012-W01 to 2022-W52. The dashed line marks the start "
        "of the evaluation period. The 2020 and 2022 peaks are about twice the highest "
        "value in the training years, which tree models cannot predict above."
    )
    e1, e2 = st.columns(2)
    with e1:
        st.image(str(sg_fig("02_annual_totals_and_weekly_averages.png")), width="stretch")
        st.caption("Annual totals range from 2,759 (2017) to 35,261 (2020). "
                   "2020 and 2022 together hold 41% of all cases.")
    with e2:
        st.image(str(sg_fig("05_autocorrelation.png")), width="stretch")
        st.caption("Lag-1 autocorrelation is 0.976; partial autocorrelation is small beyond "
                   "lag 4–5. Orange bars mark the study's window lengths.")

    st.subheader("MAE by history window — XGBoost vs LightGBM (13-fold CV mean)")
    # Grouped (side-by-side) bars, one per model per window. Same approach as the
    # shared model_bar helper, with axis titles and unit added for this tab.
    sg_bar = sg13_models[sg13_models["model"].isin(SERIES_COLORS)][["history_window", "model", "MAE_mean"]].copy()
    sg_bar["history_window"] = sg_bar["history_window"].astype(int)
    sg_bar = sg_bar.sort_values(["history_window", "model"])
    # Explicit Altair spec (not st.bar_chart): the built-in chart sorts string x labels
    # alphabetically (1, 12, 2, 4, 8) and binds the axes to scroll-zoom, which clips every
    # bar at the same height once the y axis is zoomed. Here the window order is numeric,
    # y starts at zero and the chart is not zoomable.
    sg_windows = sorted(sg_bar["history_window"].unique().tolist())
    st.altair_chart(
        alt.Chart(sg_bar).mark_bar().encode(
            x=alt.X("history_window:O", sort=sg_windows, title="History window (weeks)",
                    axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("model:N", sort=list(SERIES_COLORS)),
            y=alt.Y("MAE_mean:Q", title="Mean MAE (cases)", scale=alt.Scale(domainMin=0)),
            color=alt.Color("model:N", title=None, sort=list(SERIES_COLORS),
                            scale=alt.Scale(domain=list(SERIES_COLORS),
                                            range=list(SERIES_COLORS.values()))),
            tooltip=[alt.Tooltip("history_window:O", title="Window (weeks)"),
                     alt.Tooltip("model:N", title="Model"),
                     alt.Tooltip("MAE_mean:Q", title="Mean MAE", format=".2f")],
        ).properties(height=320),
        width="stretch",
    )
    st.caption(f"Persistence baseline for comparison: {sg13_persist_mae:.2f} cases — below every bar.")
    st.image(str(sg_fig("results_cv13_mae_vs_history_window.png")), width="stretch")
    st.caption(
        "Error bars overlapping across windows means the difference between them "
        "is not distinguishable from fold-to-fold noise."
    )

    st.subheader("Lowest observed ≠ meaningfully better")
    sg_rows = []
    for sg_model in SERIES_COLORS:
        sub = sg13_models[sg13_models["model"] == sg_model]
        b = sub.loc[sub["MAE_mean"].idxmin()]
        label = f"window {int(b['history_window'])}"
        own = sg_pairs[(sg_pairs["model"] == sg_model) & ((sg_pairs["a"] == label) | (sg_pairs["b"] == label))]
        sg_rows.append(
            f"| {sg_model} | {label} ({b['MAE_mean']:.2f}) | **No** — none of its {len(own)} paired "
            f"comparisons against other windows is significant after Holm adjustment "
            f"(smallest adjusted p = {own['wilcoxon_p_holm'].min():.2f}) |"
        )
    st.markdown(
        "| | Lowest **observed** mean MAE | Meaningfully better? |\n|---|---|---|\n" + "\n".join(sg_rows)
    )
    st.caption(
        "Both models have their lowest observed mean at 1 week, and the 5% rule selects "
        "1 week. Neither fact establishes that 1 week is better than the alternatives: "
        "medians rank the windows less cleanly (see the table below)."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Mean, median and std of fold MAE (13 folds)**")
        st.dataframe(
            sg13[["history_window", "model", "MAE_mean", "MAE_median", "MAE_std", "n_folds"]]
            .sort_values(["model", "history_window"]),
            width="stretch", hide_index=True,
        )
    with col2:
        st.markdown("**Fold win counts** (lowest MAE that fold)")
        st.dataframe(
            load_sg_csv("sg_cv13_fold_win_counts.csv").sort_values(["model", "history_window"]),
            width="stretch", hide_index=True,
        )

    st.subheader("Short {1, 2, 4} vs long {8, 12} — the pre-specified primary comparison")
    st.dataframe(
        sg_primary[["model", "mean_diff", "median_diff", "bootstrap_ci95_low", "bootstrap_ci95_high",
                    "folds_a_better", "folds_b_better", "wilcoxon_p", "wilcoxon_p_holm"]],
        width="stretch", hide_index=True,
    )
    st.caption(
        "Per-fold mean MAE of the short windows minus the long windows. Negative means "
        "short had lower error. The direction favours short for both models; "
        "both intervals include zero, and neither test is significant."
    )

    st.subheader("Persistence beat every model")
    sg_sig_worse = int(((sg_vs_persist["wilcoxon_p_holm"] < 0.05) & (sg_vs_persist["mean_diff"] > 0)).sum())
    st.markdown(
        f"Persistence had lower mean MAE than all {len(sg_vs_persist)} configurations, and "
        f"**{sg_sig_worse} of {len(sg_vs_persist)}** were worse than it, nominally significant after Holm adjustment "
        "(folds are not independent, so indicative only). This comparison was declared exploratory before the run."
    )
    st.dataframe(
        sg_vs_persist[["model", "a", "mean_diff", "mean_diff_pct_of_b", "median_diff",
                       "folds_a_better", "folds_b_better", "wilcoxon_p_holm"]]
        .rename(columns={"a": "window", "folds_a_better": "folds_model_better",
                         "folds_b_better": "folds_persistence_better"}),
        width="stretch", hide_index=True,
    )

    st.subheader("Results are dominated by three epidemic folds")
    st.image(str(sg_fig("results_cv13_fold_detail.png")), width="stretch")
    sg_fold_mae = sg13_folds[sg13_folds["model"] != PERSIST].groupby("first_test_target_week")["MAE"].sum()
    sg_top3 = sg_fold_mae.nlargest(3)
    st.markdown(
        f"The three folds starting **{', '.join(sorted(sg_top3.index))}** — the 2020 and 2022 "
        f"epidemic peaks — account for **{100 * sg_top3.sum() / sg_fold_mae.sum():.0f}%** of all "
        "model absolute error summed across folds. Mean MAE, and therefore the 5% rule, is "
        "largely decided by those three folds; in the quiet folds all windows are within a "
        "few cases of each other."
    )
    with st.expander("Per-fold MAE: persistence vs each model at 1 and 12 weeks"):
        sg_piv = sg13_folds[sg13_folds["history_window"].isin([1, 12])].pivot_table(
            index=["fold", "first_test_target_week", "last_test_target_week"],
            columns=["model", "history_window"], values="MAE")
        sg_piv.columns = [f"{m} · {w} wk" for m, w in sg_piv.columns]
        sg_piv = sg_piv.drop(columns=[f"{PERSIST} · 12 wk"]).rename(
            columns={f"{PERSIST} · 1 wk": "Persistence"})
        st.dataframe(sg_piv.reset_index().round(1), width="stretch", hide_index=True)

    with st.expander("Pre-registered minimum-sufficient-window rule, applied to the Singapore CV means"):
        st.markdown("**13 folds of 13 weeks**")
        st.dataframe(load_sg_csv("sg_cv13_minimum_sufficient_window.csv"), width="stretch", hide_index=True)
        st.dataframe(load_sg_csv("sg_cv13_near_optimal.csv"), width="stretch", hide_index=True)
        st.markdown("**6 folds of 26 weeks** (descriptive only)")
        st.dataframe(load_sg_csv("sg_cv26_minimum_sufficient_window.csv"), width="stretch", hide_index=True)
        st.caption(
            "Unchanged rule: a window is near-optimal if its mean MAE is within 5% of the "
            "best window's; the minimum sufficient window is the shortest near-optimal one. "
            "It selects 1 week for both models in both designs. The rule is applied to mean "
            "MAE, which the epidemic folds dominate."
        )

    with st.expander("6-fold / 26-week design and single chronological split"):
        st.image(str(sg_fig("results_cv26_mae_vs_history_window.png")), width="stretch")
        st.dataframe(
            load_sg_csv("sg_cv26_extended_summary.csv")
            [["history_window", "model", "MAE_mean", "MAE_median", "MAE_std", "n_folds"]]
            .sort_values(["model", "history_window"]),
            width="stretch", hide_index=True,
        )
        st.caption("Six folds cannot reach significance after adjustment, so this design is descriptive.")
        st.markdown("**Single 70/15/15 split** (models trained once, on data ending 2019-W40)")
        st.dataframe(load_sg_csv("sg_results_table.csv")
                     [["history_window", "model", "MAE", "RMSE", "val_MAE", "n_train", "n_test"]],
                     width="stretch", hide_index=True)

    with st.expander("Fold layout and data audit"):
        st.dataframe(load_sg_table("protocol_fold_plan_13w.csv"), width="stretch", hide_index=True)
        st.dataframe(load_sg_table("eda_annual_summary.csv"), width="stretch", hide_index=True)
        st.caption(
            "All 574 weekly values match the MOH Weekly Infectious Disease Bulletin source. "
            "No missing weeks; weeks run Sunday–Saturday, with week 53 in 2014 and 2020."
        )

    st.subheader("Caveats specific to Singapore")
    st.markdown(
        """
        - **External validation, different target.** MOH dengue fever notifications with
          DHF excluded are not the same quantity as DengAI's `total_cases`.
        - **Unrepresentative evaluation period.** 2019-W41 to 2022-W52 holds two record
          epidemics and the COVID-19 years; nothing here describes typical performance.
        - **Three epidemic folds dominate** every mean on this tab.
        - **Folds overlap in time through their training data.** Test blocks do not
          overlap, but each fold's training set contains the previous fold's, so folds
          are not independent and the **p-values are likely too small**.
        - **Extrapolation.** In the folds covering the 2020 rise, targets exceed anything
          in training; tree models cannot predict above that, persistence can.
        - **Medians, intervals and tests are additions** to the original method and were
          not applied to San Juan or Iquitos.
        - **Pending:** the overlap check against the second Singapore CSV (2014–2018)
          has not been done.
        """
    )
    st.caption(
        "Full write-up: `singapore_analysis/singapore_results.md`. Protocol fixed before "
        "the run: `singapore_analysis/singapore_experiment_protocol.md`."
    )


# ===========================================================================
# TAB 7 — BRAZIL EXTERNAL VALIDATION
# Reads only from brazil_analysis/results and figures.
# ===========================================================================
with tab_br:
    st.markdown(
        """
        <div class="research-banner">
        <b>Post-hoc external validation, not a preregistered replication.</b>
        InfoDengue reported dengue cases (`casos`) for Rio de Janeiro municipality,
        2010–2025. Source snapshot, API query, and checksum are documented in
        <code>brazil_analysis/SOURCE.md</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "📍 <b>Scope: Rio de Janeiro only.</b> Same 1/2/4/8/12-week lag windows, "
        "fixed models and persistence baseline. All values on this tab are cases "
        "per week and must not be numerically compared with San Juan, Iquitos, or "
        "Singapore. The 5% rule is descriptive; it is not a significance test.",
        icon="📍",
    )
    br26 = load_br_csv("brazil_cv26_summary.csv")
    br13 = load_br_csv("brazil_cv13_summary.csv")
    br26_models = br26[br26["model"].isin(SERIES_COLORS)]
    br13_models = br13[br13["model"].isin(SERIES_COLORS)]
    br_persist26 = br26.loc[br26["model"] == "Persistence (baseline)", "MAE_mean"].iloc[0]
    br_persist13 = br13.loc[br13["model"] == "Persistence (baseline)", "MAE_mean"].iloc[0]
    br_best = br26_models.loc[br26_models["MAE_mean"].idxmin()]
    br_min26 = load_br_csv("brazil_cv26_minimum_sufficient_window.csv")
    br_min13 = load_br_csv("brazil_cv13_minimum_sufficient_window.csv")

    a, b, c, d = st.columns(4)
    a.metric("Lowest 26-week CV mean MAE", f"{br_best['MAE_mean']:.1f} cases")
    b.metric("Window / model", f"{int(br_best['history_window'])} wk · {br_best['model']}")
    c.metric("Persistence · 26-week CV", f"{br_persist26:.1f}")
    d.metric("Persistence · 13-week CV", f"{br_persist13:.1f}")
    st.caption("Lowest observed mean is descriptive, not proof of a better window.")

    st.subheader("Mean MAE by history window")
    br_mean = br26_models[["history_window", "model", "MAE_mean"]].copy()
    br_mean = br_mean.rename(columns={"history_window": "window", "MAE_mean": "mae"})
    br_mean["model"] = br_mean["model"].replace({"XGBoost": "XGBoost", "LightGBM": "LightGBM"})
    br_mean["window"] = br_mean["window"].astype(int)
    br_mean = br_mean.sort_values(["window", "model"])
    br_windows = sorted(br_mean["window"].unique().tolist())
    st.altair_chart(
        alt.Chart(br_mean).mark_bar().encode(
            x=alt.X("window:O", sort=br_windows, title="History window (weeks)", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("model:N", sort=list(SERIES_COLORS)),
            y=alt.Y("mae:Q", title="Mean MAE (cases)", scale=alt.Scale(domainMin=0)),
            color=alt.Color("model:N", title=None, sort=list(SERIES_COLORS),
                            scale=alt.Scale(domain=list(SERIES_COLORS), range=list(SERIES_COLORS.values()))),
            tooltip=[alt.Tooltip("window:O", title="Window (weeks)"), alt.Tooltip("model:N"),
                     alt.Tooltip("mae:Q", title="Mean MAE", format=".2f")],
        ).properties(height=320), width="stretch",
    )
    st.caption(f"26-week CV: 9 expanding-window folds. Persistence MAE: {br_persist26:.2f}.")
    st.image(str(br_fig("brazil_cv26_mae_vs_history_window.png")), width="stretch")

    st.subheader("Results across both rolling-origin designs")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**26-week blocks · 9 folds (primary)**")
        st.dataframe(br26[["history_window", "model", "MAE_mean", "MAE_std"]].round(2),
                     width="stretch", hide_index=True)
        st.markdown("**5% minimum-sufficient rule**")
        st.dataframe(br_min26[["model", "best_window", "minimum_sufficient_window", "near_optimal_windows"]],
                     width="stretch", hide_index=True)
    with col2:
        st.markdown("**13-week blocks · 19 folds (secondary)**")
        st.dataframe(br13[["history_window", "model", "MAE_mean", "MAE_std"]].round(2),
                     width="stretch", hide_index=True)
        st.markdown("**5% minimum-sufficient rule**")
        st.dataframe(br_min13[["model", "best_window", "minimum_sufficient_window", "near_optimal_windows"]],
                     width="stretch", hide_index=True)

    st.markdown(
        """
        <div class="finding-box">

**Four weeks had the lowest mean MAE for both models in both CV designs.**
The 5% rule selected 4 weeks for XGBoost and 1 week for LightGBM, so it does
not provide one common minimum across models. Fold winners varied, and fold MAE
standard deviations (about 289–566 cases) were much larger than many differences
between window means. Persistence was lower than every model mean in 13-week CV;
in 26-week CV it was close to the XGBoost 4-week mean.

**Interpretation:** Rio supports the possibility that 1–4 weeks is often enough
for this setup, but does not prove a unique best window. Treat Brazil as
supporting external evidence, not as universal confirmation.

</div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Single chronological split")
    st.dataframe(load_br_csv("brazil_results_table.csv").round(2), width="stretch", hide_index=True)
    st.caption("124 held-out test weeks; the persistence test MAE is 319.77 cases. Both models had higher validation error than persistence at every window.")

    with st.expander("Fold winners and detailed report"):
        st.markdown("**26-week fold wins**")
        st.dataframe(load_br_csv("brazil_cv26_fold_win_counts.csv"), width="stretch", hide_index=True)
        st.markdown("**13-week fold wins**")
        st.dataframe(load_br_csv("brazil_cv13_fold_win_counts.csv"), width="stretch", hide_index=True)
        st.markdown("Full protocol, result interpretation, caveats and references: [`brazil_analysis/brazil_results.md`](https://github.com/unfitcoder101/infectious-disease-forecasting/blob/main/brazil_analysis/brazil_results.md).")

    st.subheader("Updated conclusion across four dengue settings")
    st.markdown(
        "Across San Juan, Iquitos, Singapore and Rio, histories of 1–4 weeks were often competitive, and adding history beyond 4 weeks did not provide a consistent improvement under the experiments run. The shortest adequate window varied by setting, model, time period and evaluation design. **No single window is established as generally optimal.** These are separate surveillance series with different case definitions and scales; their absolute errors are not pooled or ranked. Brazil broadens the dengue evidence but does not establish transfer to India, COVID-19, or other diseases. See `research_notes.md` §11.4."
    )


# ===========================================================================
# TAB 8 — LEVEL VERSUS CHANGE FOLLOW-UP
# Reads only the separately generated followup_experiments outputs.
# ===========================================================================
with tab_followup:
    st.markdown(
        """
        <div class="research-banner">
        <b>Instructor-requested follow-up.</b> This compares direct case-count
        forecasts with forecasts of next week's log change, reconstructed to
        case counts before scoring. Both use the same 1/2/4/8/12-week lags,
        rolling-origin test weeks and persistence baseline.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Relative MAE = model MAE ÷ persistence MAE for the selected series. "
        "Values below 1 are lower error than persistence. Counts and errors are "
        "kept separate by dataset."
    )
    selected_series = st.selectbox(
        "Dengue series",
        ["San Juan", "Iquitos", "Singapore", "Rio de Janeiro, Brazil"],
        key="followup_series",
    )
    series_slug = {
        "San Juan": "san_juan",
        "Iquitos": "iquitos",
        "Singapore": "singapore",
        "Rio de Janeiro, Brazil": "rio_de_janeiro_brazil",
    }[selected_series]
    fsummary = load_followup_csv("all_models_relative_mae_summary.csv")
    fsummary = fsummary[fsummary["series"] == selected_series].copy()
    st.image(str(followup_fig(f"{series_slug}_relative_mae.png")), width="stretch")
    st.subheader("Case-scale errors and relative MAE")
    st.dataframe(
        fsummary[["method", "model", "history_window", "n_oos", "MAE", "RMSE",
                  "persistence_MAE", "relative_MAE"]].round(3),
        width="stretch", hide_index=True,
    )
    st.subheader("Diebold–Mariano tests on pointwise absolute errors")
    dm_file = f"dm_tests_{series_slug}.csv"
    fdm = load_followup_csv(dm_file)
    st.dataframe(
        fdm[["comparison", "method", "model", "history_window", "short_window", "long_window",
             "model_relative_MAE", "short_relative_MAE", "long_relative_MAE", "n_oos",
             "mean_abs_error_difference", "p_value", "p_holm_within_series"]].round(4),
        width="stretch", hide_index=True,
    )
    st.caption(
        "Holm correction is applied across all 28 comparisons within this series: "
        "20 model/window comparisons against persistence and 8 pre-specified "
        "history contrasts (1 vs 12 weeks and 4 vs 12 weeks). No adjusted test "
        "was significant in any of the four series. See "
        "followup_experiments/research_update.md for the test details and caveats."
    )
    st.subheader("Selected seasonality sensitivity")
    fseason = load_followup_csv("seasonality_sensitivity.csv")
    st.dataframe(
        fseason[fseason["series"] == selected_series][
            ["method", "model", "configuration", "history_window", "MAE_without_seasonality",
             "MAE_with_seasonality", "relative_MAE_without_seasonality",
             "relative_MAE_with_seasonality", "change_in_MAE"]
        ].round(3), width="stretch", hide_index=True,
    )
    st.markdown(
        "**Conclusion:** Extending history beyond the most recent week gave no "
        "detectable improvement, and no configuration significantly outperformed "
        "persistence. The lowest observed point estimate varies by series and is "
        "not a statistically confirmed winner. This does not prove that the "
        "windows are equivalent. See the full tables and limitations in "
        "`followup_experiments/research_update.md`."
    )
