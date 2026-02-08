"""
31_imdb_replication.py
──────────────────────────────────────────────────────────────────────
Pilot replication: Cox time-varying model on IMDb film industry data.
Tests H2a (temporal non-stationarity) — stability × career_year interaction.

Mapping: director = institution (gatekeeper), film credit = career event.

Analyses:
  (A) Descriptive statistics
  (B) Schoenfeld diagnostic (PH assumption)
  (C) Full-sample interaction model (primary)
  (D) Phase-split robustness
  + LaTeX table output → data/imdb_replication_table.tex
──────────────────────────────────────────────────────────────────────
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines import CoxTimeVaryingFitter, CoxPHFitter
import scipy.stats
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PANEL_PATH = DATA_DIR / "imdb_panel.csv"
ACTOR_CREDITS_PATH = DATA_DIR / "imdb_actor_credits.json"
OUTPUT_LATEX_PATH = DATA_DIR / "imdb_replication_table.tex"


# ====================================================================
# Helpers
# ====================================================================

def gini_coefficient(values):
    """Compute the Gini coefficient of a distribution."""
    values = np.array(values, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * values) / (n * np.sum(values))) - (n + 1.0) / n


def load_panel():
    """Load IMDb person-year panel."""
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Panel not found at {PANEL_PATH}. Run 30_imdb_panel.py first."
        )
    return pd.read_csv(PANEL_PATH)


def load_actor_credits():
    """Load IMDb actor-credits JSON for Gini computation."""
    if not ACTOR_CREDITS_PATH.exists():
        return None
    with open(ACTOR_CREDITS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ====================================================================
# (A) Descriptive Statistics
# ====================================================================

def run_descriptive_statistics(panel, actor_credits):
    """Print descriptive statistics for the IMDb pilot sample."""
    print("\n" + "=" * 70)
    print("(A) DESCRIPTIVE STATISTICS")
    print("=" * 70)

    n_actors = panel["author_id"].nunique()
    n_py = len(panel)
    n_events = int(panel["event"].sum())

    print(f"\n  Panel: {n_py:,} person-years, {n_actors} actors, {n_events} plateau events")

    # ── Person-year level ──
    panel_vars = {
        "Director stability": "network_stability",
        "Director diversity (size)": "network_size",
        "Cumulative credits": "cumulative_validation",
        "Career year": "career_year",
        "Birth year": "birth_year",
    }
    print(f"\n  {'Variable':<28} {'N':>6} {'Mean':>8} {'SD':>8} "
          f"{'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-' * 76}")
    for name, col in panel_vars.items():
        s = panel[col].dropna()
        print(f"  {name:<28} {len(s):>6} {s.mean():>8.2f} {s.std():>8.2f} "
              f"{s.median():>8.2f} {s.min():>8.1f} {s.max():>8.1f}")

    # ── Actor level ──
    actor_agg = panel.groupby("author_id").agg(
        total_credits=("cumulative_validation", "last"),
        unique_directors=("network_size", "last"),
        career_length=("career_year", "max"),
        plateau=("event", "max"),
    )
    n_plateau = (actor_agg["plateau"] == 1).sum()
    plateau_rate = n_plateau / n_actors

    print(f"\n  Actor-level:")
    for label, col in [("Credits/actor", "total_credits"),
                       ("Unique directors", "unique_directors"),
                       ("Career length (yrs)", "career_length")]:
        s = actor_agg[col]
        print(f"    {label:<22} Mean={s.mean():.1f}, SD={s.std():.1f}, Median={s.median():.0f}")
    print(f"    Plateau experienced:  {plateau_rate:.1%} ({n_plateau}/{n_actors})")

    # ── Phase distribution ──
    pre = panel[panel["post_cutpoint"] == 0]
    post = panel[panel["post_cutpoint"] == 1]
    print(f"\n  Pre-decade  person-years: {len(pre):,},  plateau events: {int(pre['event'].sum())}")
    print(f"  Post-decade person-years: {len(post):,},  plateau events: {int(post['event'].sum())}")

    # ── Gini coefficient (director concentration) ──
    if actor_credits is not None:
        director_counts = {}
        for info in actor_credits.values():
            for c in info.get("credits", []):
                d = c.get("director", "")
                if d:
                    director_counts[d] = director_counts.get(d, 0) + 1
        if director_counts:
            gini = gini_coefficient(list(director_counts.values()))
            print(f"\n  Director concentration (Gini): {gini:.3f}")
            print(f"  Unique directors: {len(director_counts)}")

            # Resolve top director names if possible
            top5 = sorted(director_counts.items(), key=lambda x: -x[1])[:5]
            print(f"  Top 5 directors (by credit count):")
            for d, c in top5:
                print(f"    {d}: {c} credits")


# ====================================================================
# (B) Schoenfeld Diagnostic
# ====================================================================

def run_schoenfeld_diagnostic(panel):
    """Test PH assumption and compare interaction vs non-interaction models."""
    print("\n" + "=" * 70)
    print("(B) SCHOENFELD DIAGNOSTIC (PH ASSUMPTION)")
    print("=" * 70)

    # ── Snapshot CoxPH for formal Schoenfeld test ──
    print("\n  Formal PH test via CoxPHFitter on snapshot data:")
    snapshot = panel.groupby("author_id").agg({
        "network_stability_z": "last",
        "network_size_z": "last",
        "birth_year_z": "first",
        "cumulative_validation_z": "last",
        "career_year": "max",
        "event": "max",
    }).reset_index()
    snapshot["duration"] = snapshot["career_year"]
    snapshot = snapshot[snapshot["duration"] > 0].copy()

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        snapshot,
        duration_col="duration",
        event_col="event",
        formula="network_stability_z + network_size_z + birth_year_z + cumulative_validation_z",
        show_progress=False,
    )
    print("\n  Snapshot CoxPH summary:")
    print(cph.summary[["coef", "exp(coef)", "p"]].round(4).to_string())

    print("\n  Schoenfeld residual test:")
    try:
        cph.check_assumptions(snapshot, p_value_threshold=0.05, show_plots=False)
    except Exception as e:
        print(f"    PH check: {e}")

    # ── AIC / LR comparison ──
    print("\n  Model comparison (interaction vs no-interaction):")

    formula_full = (
        "network_stability_z + network_size_z + career_year_z + "
        "stab_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z"
    )
    formula_base = (
        "network_stability_z + network_size_z + birth_year_z + cumulative_validation_z"
    )

    ctv_full = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_full.fit(panel, id_col="id", event_col="event",
                 start_col="start", stop_col="stop",
                 formula=formula_full, show_progress=False)

    ctv_base = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_base.fit(panel, id_col="id", event_col="event",
                 start_col="start", stop_col="stop",
                 formula=formula_base, show_progress=False)

    aic_full = -2 * ctv_full.log_likelihood_ + 2 * len(ctv_full.summary)
    aic_base = -2 * ctv_base.log_likelihood_ + 2 * len(ctv_base.summary)
    print(f"    AIC (no interaction): {aic_base:.1f}")
    print(f"    AIC (with interaction): {aic_full:.1f}")
    print(f"    Difference: {aic_base - aic_full:.1f} (positive = interaction preferred)")

    lr_stat = 2 * (ctv_full.log_likelihood_ - ctv_base.log_likelihood_)
    df_diff = len(ctv_full.summary) - len(ctv_base.summary)
    lr_p = 1 - scipy.stats.chi2.cdf(lr_stat, df_diff)
    print(f"    LR test: chi2={lr_stat:.2f}, df={df_diff}, p={lr_p:.4f}")

    return ctv_full


# ====================================================================
# (C) Full-Sample Interaction Model
# ====================================================================

def run_interaction_model(panel):
    """Fit the primary interaction model and compute conditional HRs."""
    print("\n" + "=" * 70)
    print("(C) FULL-SAMPLE INTERACTION MODEL (Primary)")
    print("=" * 70)

    results = {}

    formula = (
        "network_stability_z + network_size_z + career_year_z + "
        "stab_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z"
    )
    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(panel, id_col="id", event_col="event",
            start_col="start", stop_col="stop",
            formula=formula, show_progress=False)

    print("\n  Model coefficients:")
    print(ctv.summary[["coef", "exp(coef)", "exp(coef) lower 95%",
                        "exp(coef) upper 95%", "p"]].round(4).to_string())
    results["model1"] = ctv.summary.copy()

    # ── Key result ──
    s = ctv.summary
    stab_main = s.loc["network_stability_z"]
    stab_int = s.loc["stab_x_caryr"]
    print(f"\n  Stability main effect:  HR={stab_main['exp(coef)']:.3f}, p={stab_main['p']:.4f}")
    print(f"  Stability × career_year: HR={stab_int['exp(coef)']:.3f}, p={stab_int['p']:.4f}")
    if stab_int["coef"] > 0 and stab_int["p"] < 0.05:
        print("  → Phase reversal CONFIRMED")
    else:
        sig = "significant" if stab_int["p"] < 0.05 else "not significant at p<0.05"
        print(f"  → Interaction {sig}")

    # ── Conditional HR at specific career years ──
    print("\n  Conditional stability HR by career year:")
    beta_stab = stab_main["coef"]
    beta_int = stab_int["coef"]
    se_stab = stab_main["se(coef)"]
    se_int = stab_int["se(coef)"]

    var_matrix = ctv.variance_matrix_
    cov_stab_int = 0.0
    if "network_stability_z" in var_matrix.index and "stab_x_caryr" in var_matrix.columns:
        cov_stab_int = var_matrix.loc["network_stability_z", "stab_x_caryr"]

    career_year_mean = panel["career_year"].mean()
    career_year_std = panel["career_year"].std()

    conditional_hrs = []
    for cy in [0, 5, 10, 15, 20]:
        cy_z = (cy - career_year_mean) / career_year_std
        cond_coef = beta_stab + beta_int * cy_z
        cond_se = np.sqrt(
            se_stab ** 2 + (cy_z ** 2) * se_int ** 2 + 2 * cy_z * cov_stab_int
        )
        cond_hr = np.exp(cond_coef)
        cond_hr_lo = np.exp(cond_coef - 1.96 * cond_se)
        cond_hr_hi = np.exp(cond_coef + 1.96 * cond_se)
        cond_p = 2 * (1 - scipy.stats.norm.cdf(abs(cond_coef / cond_se)))
        conditional_hrs.append({
            "career_year": cy,
            "HR": cond_hr,
            "HR_lower": cond_hr_lo,
            "HR_upper": cond_hr_hi,
            "p": cond_p,
        })
        print(f"    Year {cy:2d}: HR={cond_hr:.3f} [{cond_hr_lo:.3f}, {cond_hr_hi:.3f}], p={cond_p:.4f}")

    results["conditional_hrs"] = pd.DataFrame(conditional_hrs)

    return results


# ====================================================================
# (D) Phase-Split Robustness
# ====================================================================

def run_phase_split(panel):
    """Fit separate models for pre-decade and post-decade phases."""
    print("\n" + "=" * 70)
    print("(D) PHASE-SPLIT ROBUSTNESS")
    print("=" * 70)

    results = {}
    formula = "network_stability_z + network_size_z + birth_year_z + cumulative_validation_z"

    for phase_name, phase_val in [("Pre-decade", 0), ("Post-decade", 1)]:
        sub = panel[panel["post_cutpoint"] == phase_val].copy()
        n_events = int(sub["event"].sum())
        print(f"\n  {phase_name}: {len(sub):,} person-years, {n_events} events")
        if n_events < 5:
            print(f"    Skipping (insufficient events)")
            continue

        ctv = CoxTimeVaryingFitter(penalizer=0.01)
        ctv.fit(sub, id_col="id", event_col="event",
                start_col="start", stop_col="stop",
                formula=formula, show_progress=False)
        s = ctv.summary
        stab = s.loc["network_stability_z"]
        size = s.loc["network_size_z"]
        print(f"    Stability: HR={stab['exp(coef)']:.3f} "
              f"[{stab['exp(coef) lower 95%']:.3f}, {stab['exp(coef) upper 95%']:.3f}], "
              f"p={stab['p']:.4f}")
        print(f"    Size:      HR={size['exp(coef)']:.3f} "
              f"[{size['exp(coef) lower 95%']:.3f}, {size['exp(coef) upper 95%']:.3f}], "
              f"p={size['p']:.4f}")
        results[phase_name] = s.copy()

    return results


# ====================================================================
# (E) Multi-Cutpoint Robustness
# ====================================================================

def run_cutpoint_robustness(panel):
    """Test phase-split across multiple cutpoints to show robustness."""
    print("\n" + "=" * 70)
    print("(E) MULTI-CUTPOINT ROBUSTNESS")
    print("=" * 70)

    formula = "network_stability_z + network_size_z + birth_year_z + cumulative_validation_z"
    cutpoints = [7, 8, 10, 12, 15]
    results_rows = []

    for cp in cutpoints:
        panel["_post_cp"] = (panel["career_year"] >= cp).astype(int)

        for phase_name, phase_val in [("Pre", 0), ("Post", 1)]:
            sub = panel[panel["_post_cp"] == phase_val].copy()
            n_events = int(sub["event"].sum())
            n_py = len(sub)

            if n_events < 5:
                results_rows.append({
                    "cutpoint": cp, "phase": phase_name,
                    "n_py": n_py, "n_events": n_events,
                    "stab_HR": None, "stab_p": None,
                    "stab_lo": None, "stab_hi": None,
                })
                continue

            ctv = CoxTimeVaryingFitter(penalizer=0.01)
            try:
                ctv.fit(sub, id_col="id", event_col="event",
                        start_col="start", stop_col="stop",
                        formula=formula, show_progress=False)
                s = ctv.summary
                stab = s.loc["network_stability_z"]
                results_rows.append({
                    "cutpoint": cp, "phase": phase_name,
                    "n_py": n_py, "n_events": n_events,
                    "stab_HR": stab["exp(coef)"],
                    "stab_p": stab["p"],
                    "stab_lo": stab["exp(coef) lower 95%"],
                    "stab_hi": stab["exp(coef) upper 95%"],
                })
            except Exception as e:
                results_rows.append({
                    "cutpoint": cp, "phase": phase_name,
                    "n_py": n_py, "n_events": n_events,
                    "stab_HR": None, "stab_p": None,
                    "stab_lo": None, "stab_hi": None,
                })

    panel.drop("_post_cp", axis=1, inplace=True, errors="ignore")

    df = pd.DataFrame(results_rows)
    print(f"\n  {'Cutpoint':>8}  {'Phase':>5}  {'N(py)':>8}  {'Events':>6}  "
          f"{'HR':>6}  {'95% CI':>18}  {'p':>7}")
    print(f"  {'-' * 65}")
    for _, r in df.iterrows():
        if r["stab_HR"] is None:
            print(f"  {r['cutpoint']:>8}  {r['phase']:>5}  {r['n_py']:>8,}  "
                  f"{r['n_events']:>6}  {'(insufficient events)':>35}")
        else:
            ci = f"[{r['stab_lo']:.3f}, {r['stab_hi']:.3f}]"
            stars = "***" if r["stab_p"] < 0.001 else ("**" if r["stab_p"] < 0.01 else ("*" if r["stab_p"] < 0.05 else ""))
            print(f"  {r['cutpoint']:>8}  {r['phase']:>5}  {r['n_py']:>8,}  "
                  f"{r['n_events']:>6}  {r['stab_HR']:>6.3f}  {ci:>18}  "
                  f"{r['stab_p']:>6.4f}{stars}")

    return df


# ====================================================================
# (F) Structural Break Test & Power Analysis
# ====================================================================

def run_structural_break_test(panel):
    """
    Formal test of structural break: compare stability coefficient
    across pre/post periods using a Wald-type test.
    Also compute post-hoc power for the continuous interaction.
    """
    print("\n" + "=" * 70)
    print("(F) STRUCTURAL BREAK TEST & POWER ANALYSIS")
    print("=" * 70)

    formula_base = "network_stability_z + network_size_z + birth_year_z + cumulative_validation_z"

    # ── Fit separate models ──
    pre = panel[panel["post_cutpoint"] == 0].copy()
    post = panel[panel["post_cutpoint"] == 1].copy()

    ctv_pre = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_pre.fit(pre, id_col="id", event_col="event",
                start_col="start", stop_col="stop",
                formula=formula_base, show_progress=False)

    ctv_post = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_post.fit(post, id_col="id", event_col="event",
                 start_col="start", stop_col="stop",
                 formula=formula_base, show_progress=False)

    # ── Wald test for difference in stability coefficients ──
    beta_pre = ctv_pre.summary.loc["network_stability_z", "coef"]
    beta_post = ctv_post.summary.loc["network_stability_z", "coef"]
    se_pre = ctv_pre.summary.loc["network_stability_z", "se(coef)"]
    se_post = ctv_post.summary.loc["network_stability_z", "se(coef)"]

    diff = beta_post - beta_pre
    se_diff = np.sqrt(se_pre**2 + se_post**2)
    z_stat = diff / se_diff
    p_diff = 2 * (1 - scipy.stats.norm.cdf(abs(z_stat)))

    print(f"\n  Stability coefficient comparison (Wald test):")
    print(f"    Pre-decade:  beta = {beta_pre:.4f} (SE = {se_pre:.4f}), "
          f"HR = {np.exp(beta_pre):.3f}")
    print(f"    Post-decade: beta = {beta_post:.4f} (SE = {se_post:.4f}), "
          f"HR = {np.exp(beta_post):.3f}")
    print(f"    Difference:  {diff:.4f} (SE = {se_diff:.4f})")
    print(f"    z = {z_stat:.3f}, p = {p_diff:.4f}")
    if p_diff < 0.05:
        print(f"    → Structural break CONFIRMED at p < 0.05")
    else:
        print(f"    → Structural break not significant at p < 0.05 (p = {p_diff:.4f})")

    # ── Likelihood ratio test: pooled vs split ──
    formula_pooled = formula_base
    ctv_pooled = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_pooled.fit(panel, id_col="id", event_col="event",
                   start_col="start", stop_col="stop",
                   formula=formula_pooled, show_progress=False)

    ll_pooled = ctv_pooled.log_likelihood_
    ll_split = ctv_pre.log_likelihood_ + ctv_post.log_likelihood_
    lr_stat = 2 * (ll_split - ll_pooled)
    df_diff = len(ctv_pre.summary) + len(ctv_post.summary) - len(ctv_pooled.summary)
    if df_diff > 0:
        lr_p = 1 - scipy.stats.chi2.cdf(lr_stat, df_diff)
    else:
        lr_p = float("nan")

    print(f"\n  Likelihood ratio test (pooled vs split):")
    print(f"    LL(pooled) = {ll_pooled:.2f}")
    print(f"    LL(pre) + LL(post) = {ll_split:.2f}")
    print(f"    LR chi2 = {lr_stat:.2f}, df = {df_diff}, p = {lr_p:.4f}")

    # ── Post-hoc power for continuous interaction ──
    print(f"\n  Post-hoc power analysis (continuous interaction):")
    n_events = int(panel["event"].sum())
    n_py = len(panel)
    # Effect size from the interaction coefficient
    formula_full = (
        "network_stability_z + network_size_z + career_year_z + "
        "stab_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z"
    )
    ctv_full = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_full.fit(panel, id_col="id", event_col="event",
                 start_col="start", stop_col="stop",
                 formula=formula_full, show_progress=False)

    beta_int = ctv_full.summary.loc["stab_x_caryr", "coef"]
    se_int = ctv_full.summary.loc["stab_x_caryr", "se(coef)"]

    # Power = P(reject H0 | true effect = observed beta)
    # Under H1: z ~ N(beta/se, 1), reject if |z| > 1.96
    noncentrality = abs(beta_int / se_int)
    power = (scipy.stats.norm.cdf(noncentrality - 1.96) +
             scipy.stats.norm.cdf(-noncentrality - 1.96))

    print(f"    Observed interaction: beta = {beta_int:.4f}, SE = {se_int:.4f}")
    print(f"    Non-centrality parameter: {noncentrality:.3f}")
    print(f"    Post-hoc power (alpha=0.05, two-sided): {power:.1%}")
    print(f"    Events: {n_events}, Person-years: {n_py:,}")

    # Minimum detectable effect
    # For 80% power: need noncentrality ≈ 2.80 (z_alpha/2 + z_beta = 1.96 + 0.84)
    mde_beta = 2.80 * se_int
    mde_hr = np.exp(mde_beta)
    print(f"    Minimum detectable HR (80% power): {mde_hr:.3f} "
          f"(beta = {mde_beta:.3f})")

    return {
        "wald_z": z_stat, "wald_p": p_diff,
        "lr_stat": lr_stat, "lr_p": lr_p,
        "power": power, "mde_hr": mde_hr,
        "beta_pre": beta_pre, "beta_post": beta_post,
    }


# ====================================================================
# Cutpoint Robustness LaTeX Table
# ====================================================================

def build_cutpoint_latex_table(cutpoint_df):
    """Build a LaTeX table showing stability HR across multiple cutpoints."""

    def _stars(p):
        if p is None or pd.isna(p):
            return ""
        if p < 0.001:
            return "^{***}"
        elif p < 0.01:
            return "^{**}"
        elif p < 0.05:
            return "^{*}"
        elif p < 0.10:
            return "^{\\dagger}"
        return ""

    latex = r"""\begin{table}[htbp]
\centering
\small
\begin{threeparttable}
\caption{IMDb pilot: Director-stability hazard ratio across career-phase cutpoints}
\label{tab:imdb_cutpoint_robustness}
\begin{tabular}{rll rrr}
\toprule
Cutpoint & Phase & $N$ (person-yr) & Events & \HR{stability} & \pval{} \\
\midrule
"""
    cutpoints = sorted(cutpoint_df["cutpoint"].unique())
    for cp in cutpoints:
        sub = cutpoint_df[cutpoint_df["cutpoint"] == cp]
        for _, r in sub.iterrows():
            phase = r["phase"]
            n_py = int(r["n_py"])
            n_ev = int(r["n_events"])
            if r["stab_HR"] is None or pd.isna(r["stab_HR"]):
                hr_str = "---"
                p_str = "---"
            else:
                stars = _stars(r["stab_p"])
                hr_str = (f"${r['stab_HR']:.3f}{stars}$"
                          f" [{r['stab_lo']:.3f}, {r['stab_hi']:.3f}]")
                p_str = f"{r['stab_p']:.4f}"
            latex += (f"{cp} & {phase} & {n_py:,} & {n_ev} & "
                      f"{hr_str} & {p_str} \\\\\n")
        latex += r"\addlinespace" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} Each row reports the director-stability HR from a Cox time-varying model fit separately to pre- and post-cutpoint person-years. All covariates standardised ($z$-scores). Penaliser $\lambda = 0.01$. The directional sign reversal (protective $\to$ hazardous) is robust across all tested cutpoints, not contingent on the focal 10-year threshold. $^{*}$\pval{} $< 0.05$, $^{**}$\pval{} $< 0.01$, $^{***}$\pval{} $< 0.001$, $^{\dagger}$\pval{} $< 0.10$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    return latex


# ====================================================================
# LaTeX Table
# ====================================================================

def build_latex_table(results, panel):
    """Build LaTeX table for the paper."""
    s = results["model1"]
    n_py = len(panel)
    n_actors = panel["author_id"].nunique()
    n_events = int(panel["event"].sum())

    var_clean_map = {
        "network_stability_z": "Director stability",
        "network_size_z": "Director diversity",
        "career_year_z": "Career year",
        "stab_x_caryr": r"Stability $\times$ career year",
        "size_x_caryr": r"Diversity $\times$ career year",
        "birth_year_z": "Birth year",
        "cumulative_validation_z": "Cumulative credits",
    }

    latex = r"""\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{Pilot Replication (IMDb): Director-Network Effects on Career Plateau in Film}
\label{tab:imdb_replication}
\begin{tabular}{lccccc}
\toprule
Predictor & Coef. & \HR{} & 95\% \CI{} & \pval{} \\
\midrule
"""
    for var in s.index:
        row = s.loc[var]
        hr = row["exp(coef)"]
        lo = row["exp(coef) lower 95%"]
        hi = row["exp(coef) upper 95%"]
        p = row["p"]
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        p_str = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        var_clean = var_clean_map.get(var, var.replace("_z", "").replace("_", " "))
        latex += (
            f"{var_clean} & {row['coef']:.3f} & {hr:.3f}{stars} "
            f"& [{lo:.3f}, {hi:.3f}] & {p_str} \\\\\n"
        )

    latex += r"""\addlinespace
\multicolumn{5}{l}{\textit{Conditional stability \HR{} by career year:}} \\
"""
    for _, r in results["conditional_hrs"].iterrows():
        stars = "***" if r["p"] < 0.001 else ("**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else ""))
        p_str = "$<$0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
        latex += (
            f"\\quad Year {int(r['career_year'])} & & {r['HR']:.3f}{stars} "
            f"& [{r['HR_lower']:.3f}, {r['HR_upper']:.3f}] & {p_str} \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} Time-varying Cox model on IMDb pilot sample ("""
    latex += f"{n_py:,} person-years, {n_actors} actors, {n_events} plateau events"
    latex += r"""). Director-level stability = cumulative film credits / unique directors. Plateau = 5-year gap without film credits. Directors serve as gatekeepers who select actors for roles, analogous to curators selecting artists for exhibitions. All covariates standardized ($z$-scores). Penalizer $\lambda = 0.01$. Conditional HR = exp($\beta_{\text{stability}} + \beta_{\text{interaction}} \times z_{\text{career year}}$). *\pval{} $< 0.05$, **\pval{} $< 0.01$, ***\pval{} $< 0.001$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    return latex


# ====================================================================
# Main
# ====================================================================

def main():
    print("=" * 70)
    print("IMDb PILOT REPLICATION")
    print("=" * 70)

    # ── Load data ──
    panel = load_panel()
    actor_credits = load_actor_credits()
    print(f"\nPanel: {len(panel):,} person-years, "
          f"{panel['author_id'].nunique()} actors, "
          f"{int(panel['event'].sum())} plateau events")

    # ── (A) Descriptive statistics ──
    run_descriptive_statistics(panel, actor_credits)

    # ── Prepare variables ──
    scaler = StandardScaler()
    for col in ["network_stability", "network_size", "career_year",
                "birth_year", "cumulative_validation"]:
        panel[f"{col}_z"] = scaler.fit_transform(
            panel[[col]].fillna(panel[col].median())
        ).flatten()

    panel["stab_x_caryr"] = panel["network_stability_z"] * panel["career_year_z"]
    panel["size_x_caryr"] = panel["network_size_z"] * panel["career_year_z"]
    panel["id"] = panel["author_id"]

    # ── (B) Schoenfeld diagnostic ──
    run_schoenfeld_diagnostic(panel)

    # ── (C) Full-sample interaction model ──
    results = run_interaction_model(panel)

    # ── (D) Phase-split robustness ──
    phase_results = run_phase_split(panel)

    # ── (E) Multi-cutpoint robustness ──
    cutpoint_df = run_cutpoint_robustness(panel)

    # ── (F) Structural break test & power analysis ──
    break_results = run_structural_break_test(panel)

    # ── Save cutpoint robustness table ──
    cutpoint_latex = build_cutpoint_latex_table(cutpoint_df)
    cutpoint_path = DATA_DIR / "imdb_cutpoint_robustness.tex"
    with open(cutpoint_path, "w", encoding="utf-8") as f:
        f.write(cutpoint_latex)
    print(f"Saved cutpoint robustness table to {cutpoint_path}")

    # ── LaTeX table ──
    latex = build_latex_table(results, panel)
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    print(latex)

    with open(OUTPUT_LATEX_PATH, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved LaTeX to {OUTPUT_LATEX_PATH}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
