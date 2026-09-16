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


def fig(name: str):
    """Path to an existing figure in results/figures/. Raises if missing --
    fail loudly rather than silently skip a supposedly-existing artifact."""
    p = FIG_DIR / name
    if not p.exists():
        st.error(f"Expected figure not found: {p}")
    return p


def model_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """A native, interactive bar chart for a metric by history_window, one
    series per model, using the SAME fixed hue order as every static figure
    in this project (XGBoost blue, LightGBM orange -- never swapped, never
    re-derived from data order)."""
    pivot = df[df["model"].isin(SERIES_COLORS)].pivot(index=x_col, columns="model", values=y_col)
    pivot = pivot[[m for m in SERIES_COLORS if m in pivot.columns]]  # fixed order
    st.bar_chart(pivot, color=[SERIES_COLORS[m] for m in pivot.columns], height=320)
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
tab_overview, tab_single, tab_cv, tab_fine, tab_conclusion = st.tabs(
    [
        "📋 Overview",
        "📊 Single-Split (1/2/4/8/12 wk)",
        "🔁 Rolling-Origin CV (10-fold)",
        "🎯 Targeted Follow-up (21-fold)",
        "✅ Conclusion & Limitations",
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
            """
        )
    st.caption("Verbatim from the committed research_notes.md §8 — see the repository for the full document.")
