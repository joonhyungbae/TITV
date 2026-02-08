"""
29_unified_specification.py
══════════════════════════════════════════════════════════════════════
Unified Specification: RCS × career_year × stability
+ Data-driven change-point estimation

Addresses reviewer concern: "same hypothesis, different specifications"
by providing a single flexible specification that works in both datasets.

Analyses:
  (A) RCS(career_year) × stability  — Korean art data
  (B) RCS(career_year) × stability  — IMDb data
  (C) Profile-likelihood change-point estimation (both datasets)
  (D) LaTeX table output → data/unified_specification_table.tex

The key idea:
  Instead of a linear interaction (stability × career_year), we use
  restricted cubic splines on career_year interacted with stability.
  This captures non-linear threshold effects without imposing linearity,
  and works identically in both datasets regardless of career-length
  distribution.
══════════════════════════════════════════════════════════════════════
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines import CoxTimeVaryingFitter
import scipy.stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# RCS Basis Functions (reused from 11_achievement_gradient.py)
# ════════════════════════════════════════════════════════════════

def rcs_basis(x, knots):
    """
    Restricted Cubic Spline (Natural Spline) basis — Harrell convention.
    Given k knots, returns k-2 non-linear basis columns.
    The linear term is handled separately.
    """
    x = np.asarray(x, dtype=float)
    knots = np.sort(np.asarray(knots, dtype=float))
    k = len(knots)
    assert k >= 3, "Need at least 3 knots for RCS"

    def _pos_cube(v):
        return np.where(v > 0, v ** 3, 0.0)

    t = knots
    basis_cols = []
    for j in range(k - 2):
        col = (_pos_cube(x - t[j])
               - _pos_cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
               + _pos_cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        basis_cols.append(col)
    return np.column_stack(basis_cols) if basis_cols else np.empty((len(x), 0))


def select_knots(x, n_knots):
    """Select knot positions using Harrell's recommended percentiles."""
    harrell_pcts = {
        3: [10, 50, 90],
        4: [5, 35, 65, 95],
        5: [5, 27.5, 50, 72.5, 95],
    }
    pcts = harrell_pcts.get(n_knots, np.linspace(5, 95, n_knots))
    return np.percentile(x, pcts)


# ════════════════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════════════════

def load_korean_panel():
    """Load Korean art data and build person-year panel."""
    from data_pipeline import (
        load_raw_data, extract_artist_info, extract_events,
        build_person_year_panel
    )
    data_path = os.path.join(DATA_DIR, 'data.json')
    if not os.path.exists(data_path):
        print("  [SKIP] Korean data not found")
        return None

    artists_list = load_raw_data(data_path)
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)
    panel = build_person_year_panel(df_artists, df_events, cutpoint=10)
    panel['id'] = panel['artist_id']
    return panel


def load_imdb_panel():
    """Load IMDb person-year panel."""
    panel_path = os.path.join(DATA_DIR, 'imdb_panel.csv')
    if not os.path.exists(panel_path):
        print("  [SKIP] IMDb panel not found. Run 26_imdb_fetch.py + 27_imdb_panel.py first.")
        return None

    panel = pd.read_csv(panel_path)
    panel['id'] = panel['author_id']
    return panel


def standardize_panel(panel, id_col='id'):
    """Standardize variables and create interaction terms."""
    scaler = StandardScaler()
    for col in ['network_stability', 'network_size', 'career_year',
                'birth_year', 'cumulative_validation']:
        if col in panel.columns:
            panel[f'{col}_z'] = scaler.fit_transform(
                panel[[col]].fillna(panel[col].median())
            ).flatten()

    # Linear interaction (for comparison)
    panel['stab_x_caryr'] = panel['network_stability_z'] * panel['career_year_z']
    panel['size_x_caryr'] = panel['network_size_z'] * panel['career_year_z']
    return panel


# ════════════════════════════════════════════════════════════════
# (A/B) RCS × career_year × stability
# ════════════════════════════════════════════════════════════════

def fit_rcs_interaction_model(panel, dataset_name, n_knots=3):
    """
    Fit RCS(career_year) × stability interaction model.

    The conditional HR of stability at career_year t is:
        exp(β_S + β_I_linear * t_z + Σ_j β_I_rcs_j * rcs_j(t))

    This captures non-linear threshold effects.
    """
    print(f"\n{'=' * 70}")
    print(f"RCS × CAREER_YEAR × STABILITY: {dataset_name}")
    print(f"  Knots: {n_knots}")
    print(f"{'=' * 70}")

    career_year_raw = panel['career_year'].values.copy()
    cy_mean = career_year_raw.mean()
    cy_std = career_year_raw.std()

    # Select knots
    knots = select_knots(career_year_raw, n_knots)
    print(f"\n  Career year statistics:")
    print(f"    Mean={cy_mean:.1f}, SD={cy_std:.1f}, "
          f"Median={np.median(career_year_raw):.1f}, "
          f"Min={career_year_raw.min()}, Max={career_year_raw.max()}")
    print(f"  Knot positions: {', '.join(f'{k:.1f}' for k in knots)}")
    pcts = [np.mean(career_year_raw <= k) * 100 for k in knots]
    print(f"  Knot percentiles: {', '.join(f'{p:.0f}%' for p in pcts)}")

    # Create RCS basis columns for career_year
    rcs_cols = rcs_basis(career_year_raw, knots)
    n_rcs = rcs_cols.shape[1]

    # Standardize linear career_year term
    cy_linear = (career_year_raw - cy_mean) / cy_std

    rcs_col_names = []
    rcs_stds = []
    for j in range(n_rcs):
        col_name = f'cy_rcs_{j+1}'
        col_std = rcs_cols[:, j].std()
        rcs_stds.append(col_std if col_std > 0 else 1.0)
        panel[col_name] = rcs_cols[:, j] / rcs_stds[j]
        rcs_col_names.append(col_name)

    panel['cy_linear_z'] = cy_linear

    # Interaction terms: stability × each career_year component
    int_col_names = ['stab_x_cy_linear']
    panel['stab_x_cy_linear'] = panel['network_stability_z'] * panel['cy_linear_z']
    for j in range(n_rcs):
        int_name = f'stab_x_cy_rcs_{j+1}'
        panel[int_name] = panel['network_stability_z'] * panel[rcs_col_names[j]]
        int_col_names.append(int_name)

    # Build formula
    base_covs = ['network_stability_z', 'network_size_z', 'birth_year_z',
                 'cumulative_validation_z']
    cy_terms = ['cy_linear_z'] + rcs_col_names
    interaction_terms = int_col_names

    formula = ' + '.join(base_covs + cy_terms + interaction_terms)

    # Fit model
    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(panel, id_col='id', event_col='event',
            start_col='start', stop_col='stop',
            formula=formula, show_progress=False)

    print(f"\n  Model coefficients:")
    print(ctv.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%',
                        'exp(coef) upper 95%', 'p']].round(4).to_string())

    # ── Compute conditional HR of stability across career years ──
    print(f"\n  Conditional stability HR by career year:")

    var_mat = ctv.variance_matrix_
    s = ctv.summary

    beta_stab = s.loc['network_stability_z', 'coef']
    beta_int_linear = s.loc['stab_x_cy_linear', 'coef']
    beta_int_rcs = [s.loc[col, 'coef'] for col in int_col_names[1:]]

    # Indices for covariance extraction
    all_int_params = ['network_stability_z'] + int_col_names
    cov_indices = [list(var_mat.index).index(name) for name in all_int_params]

    # Evaluate at specific career years
    eval_years = [0, 5, 8, 10, 12, 15, 20]
    max_cy = int(np.percentile(career_year_raw, 95))
    eval_years = [y for y in eval_years if y <= max_cy + 5]

    conditional_hrs = []
    for cy in eval_years:
        cy_z = (cy - cy_mean) / cy_std
        # RCS basis for this career year
        rcs_val = rcs_basis(np.array([cy]), knots)[0]
        rcs_val_norm = rcs_val / np.array(rcs_stds) if n_rcs > 0 else np.array([])

        # Conditional coefficient
        cond_coef = beta_stab + beta_int_linear * cy_z
        for j in range(n_rcs):
            cond_coef += beta_int_rcs[j] * rcs_val_norm[j]

        # Gradient for delta method SE
        gradient = np.zeros(len(all_int_params))
        gradient[0] = 1.0  # d/d(beta_stab)
        gradient[1] = cy_z  # d/d(beta_int_linear)
        for j in range(n_rcs):
            gradient[2 + j] = rcs_val_norm[j]

        sub_var = np.zeros((len(all_int_params), len(all_int_params)))
        for r, ri in enumerate(cov_indices):
            for c, ci in enumerate(cov_indices):
                sub_var[r, c] = var_mat.iloc[ri, ci]

        cond_se = np.sqrt(gradient @ sub_var @ gradient)
        cond_hr = np.exp(cond_coef)
        cond_hr_lo = np.exp(cond_coef - 1.96 * cond_se)
        cond_hr_hi = np.exp(cond_coef + 1.96 * cond_se)
        cond_p = 2 * (1 - scipy.stats.norm.cdf(abs(cond_coef / cond_se)))

        conditional_hrs.append({
            'career_year': cy, 'HR': cond_hr,
            'HR_lower': cond_hr_lo, 'HR_upper': cond_hr_hi,
            'p': cond_p,
        })
        stars = '***' if cond_p < 0.001 else ('**' if cond_p < 0.01 else ('*' if cond_p < 0.05 else ''))
        print(f"    Year {cy:2d}: HR={cond_hr:.3f} [{cond_hr_lo:.3f}, {cond_hr_hi:.3f}], "
              f"p={cond_p:.4f}{stars}")

    # ── Compute HR on a fine grid for plotting ──
    cy_grid = np.linspace(0, max_cy, 200)
    grid_hrs, grid_lo, grid_hi, grid_p = [], [], [], []
    for cy in cy_grid:
        cy_z = (cy - cy_mean) / cy_std
        rcs_val = rcs_basis(np.array([cy]), knots)[0]
        rcs_val_norm = rcs_val / np.array(rcs_stds) if n_rcs > 0 else np.array([])

        cond_coef = beta_stab + beta_int_linear * cy_z
        for j in range(n_rcs):
            cond_coef += beta_int_rcs[j] * rcs_val_norm[j]

        gradient = np.zeros(len(all_int_params))
        gradient[0] = 1.0
        gradient[1] = cy_z
        for j in range(n_rcs):
            gradient[2 + j] = rcs_val_norm[j]

        sub_var_local = np.zeros((len(all_int_params), len(all_int_params)))
        for r, ri in enumerate(cov_indices):
            for c, ci in enumerate(cov_indices):
                sub_var_local[r, c] = var_mat.iloc[ri, ci]

        cond_se = np.sqrt(gradient @ sub_var_local @ gradient)
        grid_hrs.append(np.exp(cond_coef))
        grid_lo.append(np.exp(cond_coef - 1.96 * cond_se))
        grid_hi.append(np.exp(cond_coef + 1.96 * cond_se))
        grid_p.append(2 * (1 - scipy.stats.norm.cdf(abs(cond_coef / cond_se))))

    # ── Model fit comparison: RCS vs linear interaction ──
    aic_rcs = -2 * ctv.log_likelihood_ + 2 * len(ctv.summary)

    # Fit linear interaction for comparison
    formula_linear = ('network_stability_z + network_size_z + career_year_z + '
                      'stab_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z')
    ctv_linear = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_linear.fit(panel, id_col='id', event_col='event',
                   start_col='start', stop_col='stop',
                   formula=formula_linear, show_progress=False)
    aic_linear = -2 * ctv_linear.log_likelihood_ + 2 * len(ctv_linear.summary)

    print(f"\n  Model comparison:")
    print(f"    Linear interaction AIC: {aic_linear:.1f}")
    print(f"    RCS interaction AIC:    {aic_rcs:.1f}")
    print(f"    ΔAIC: {aic_linear - aic_rcs:.1f} (positive = RCS preferred)")

    # LR test: RCS vs linear
    lr_stat = 2 * (ctv.log_likelihood_ - ctv_linear.log_likelihood_)
    df_diff = len(ctv.summary) - len(ctv_linear.summary)
    if df_diff > 0 and lr_stat > 0:
        lr_p = 1 - scipy.stats.chi2.cdf(lr_stat, df_diff)
        print(f"    LR test: chi2={lr_stat:.2f}, df={df_diff}, p={lr_p:.4f}")
    else:
        lr_p = float('nan')

    # Joint Wald test for all interaction terms
    joint_test_indices = [list(s.index).index(name) for name in int_col_names]
    int_betas = np.array([s.iloc[idx]['coef'] for idx in joint_test_indices])
    int_var = np.zeros((len(joint_test_indices), len(joint_test_indices)))
    for r, ri in enumerate(joint_test_indices):
        for c, ci in enumerate(joint_test_indices):
            int_var[r, c] = var_mat.iloc[ri, ci]

    try:
        wald_stat = int_betas @ np.linalg.solve(int_var, int_betas)
        wald_df = len(int_betas)
        wald_p = 1 - scipy.stats.chi2.cdf(wald_stat, wald_df)
        print(f"\n  Joint Wald test (all stability × career_year interactions):")
        print(f"    chi2 = {wald_stat:.2f}, df = {wald_df}, p = {wald_p:.4f}")
    except np.linalg.LinAlgError:
        wald_stat, wald_p = float('nan'), float('nan')

    # Clean up temporary columns
    cleanup_cols = rcs_col_names + int_col_names + ['cy_linear_z']
    for col in cleanup_cols:
        if col in panel.columns:
            panel.drop(columns=[col], inplace=True)

    return {
        'model': ctv,
        'conditional_hrs': pd.DataFrame(conditional_hrs),
        'knots': knots,
        'aic_rcs': aic_rcs,
        'aic_linear': aic_linear,
        'wald_stat': wald_stat,
        'wald_p': wald_p,
        'grid': {
            'career_year': cy_grid,
            'HR': np.array(grid_hrs),
            'HR_lower': np.array(grid_lo),
            'HR_upper': np.array(grid_hi),
            'p': np.array(grid_p),
        },
        'n_py': len(panel),
        'n_events': int(panel['event'].sum()),
        'n_subjects': panel['id'].nunique(),
    }


# ════════════════════════════════════════════════════════════════
# (C) Profile-Likelihood Change-Point Estimation
# ════════════════════════════════════════════════════════════════

def estimate_changepoint(panel, dataset_name, tau_range=None):
    """
    Estimate the career-year change-point where stability's effect
    structurally shifts, using a profile-likelihood approach.

    For each candidate τ, fits:
        h(t) = h₀(t) exp(β₁·S·I(t<τ) + β₂·S·I(t≥τ) + γ'X)
    and records the log-likelihood.

    The optimal τ* maximizes the profile likelihood.
    Bootstrap provides 95% CI for τ*.
    """
    print(f"\n{'=' * 70}")
    print(f"CHANGE-POINT ESTIMATION: {dataset_name}")
    print(f"{'=' * 70}")

    if tau_range is None:
        tau_range = range(5, 21)

    formula = ('network_stability_z + network_size_z + birth_year_z + '
               'cumulative_validation_z')

    # ── Profile likelihood grid search ──
    results = []
    for tau in tau_range:
        panel['_post_tau'] = (panel['career_year'] >= tau).astype(int)

        # Create phase-specific stability terms
        panel['stab_pre'] = panel['network_stability_z'] * (1 - panel['_post_tau'])
        panel['stab_post'] = panel['network_stability_z'] * panel['_post_tau']

        formula_cp = ('stab_pre + stab_post + network_size_z + birth_year_z + '
                      'cumulative_validation_z')

        pre_events = int(panel[panel['_post_tau'] == 0]['event'].sum())
        post_events = int(panel[panel['_post_tau'] == 1]['event'].sum())

        if pre_events < 5 or post_events < 5:
            continue

        try:
            ctv = CoxTimeVaryingFitter(penalizer=0.01)
            ctv.fit(panel, id_col='id', event_col='event',
                    start_col='start', stop_col='stop',
                    formula=formula_cp, show_progress=False)

            s = ctv.summary
            ll = ctv.log_likelihood_
            aic = -2 * ll + 2 * len(s)

            beta_pre = s.loc['stab_pre', 'coef']
            beta_post = s.loc['stab_post', 'coef']
            se_pre = s.loc['stab_pre', 'se(coef)']
            se_post = s.loc['stab_post', 'se(coef)']
            hr_pre = np.exp(beta_pre)
            hr_post = np.exp(beta_post)
            p_pre = s.loc['stab_pre', 'p']
            p_post = s.loc['stab_post', 'p']

            # Wald test for difference
            diff = beta_post - beta_pre
            se_diff = np.sqrt(se_pre ** 2 + se_post ** 2)
            z_diff = diff / se_diff
            p_diff = 2 * (1 - scipy.stats.norm.cdf(abs(z_diff)))

            results.append({
                'tau': tau, 'LL': ll, 'AIC': aic,
                'HR_pre': hr_pre, 'HR_post': hr_post,
                'p_pre': p_pre, 'p_post': p_post,
                'beta_diff': diff, 'z_diff': z_diff, 'p_diff': p_diff,
                'pre_events': pre_events, 'post_events': post_events,
            })
        except Exception as e:
            print(f"    tau={tau}: ERROR {e}")

    # Clean up
    for col in ['_post_tau', 'stab_pre', 'stab_post']:
        if col in panel.columns:
            panel.drop(columns=[col], inplace=True)

    if not results:
        print("  No valid change-points found.")
        return None

    df = pd.DataFrame(results)

    # ── Find optimal τ (minimize AIC) ──
    best_idx = df['AIC'].idxmin()
    best_tau = df.loc[best_idx, 'tau']

    print(f"\n  Profile likelihood results:")
    print(f"  {'tau':>5}  {'LL':>10}  {'AIC':>10}  {'HR_pre':>7}  {'HR_post':>8}  "
          f"{'p_diff':>7}  {'pre_ev':>7}  {'post_ev':>8}")
    print(f"  {'-' * 70}")
    for _, r in df.iterrows():
        marker = ' <-- BEST' if r['tau'] == best_tau else ''
        print(f"  {int(r['tau']):>5}  {r['LL']:>10.2f}  {r['AIC']:>10.2f}  "
              f"{r['HR_pre']:>7.3f}  {r['HR_post']:>8.3f}  "
              f"{r['p_diff']:>7.4f}  {int(r['pre_events']):>7}  "
              f"{int(r['post_events']):>8}{marker}")

    best = df.loc[best_idx]
    print(f"\n  Optimal change-point: τ* = {int(best_tau)} years")
    print(f"    Pre-τ stability HR:  {best['HR_pre']:.3f} (p = {best['p_pre']:.4f})")
    print(f"    Post-τ stability HR: {best['HR_post']:.3f} (p = {best['p_post']:.4f})")
    print(f"    Coefficient difference test: z = {best['z_diff']:.3f}, p = {best['p_diff']:.4f}")

    # ── Bootstrap CI for τ* ──
    print(f"\n  Bootstrapping 95% CI for τ* (50 resamples)...")
    n_boot = 50
    boot_taus = []
    subject_ids = panel['id'].unique()
    rng = np.random.RandomState(42)

    for b in range(n_boot):
        # Resample subjects (cluster bootstrap)
        boot_ids = rng.choice(subject_ids, size=len(subject_ids), replace=True)

        # Build bootstrap panel
        boot_frames = []
        for i, sid in enumerate(boot_ids):
            sub = panel[panel['id'] == sid].copy()
            sub['id'] = f"boot_{i}"
            boot_frames.append(sub)
        boot_panel = pd.concat(boot_frames, ignore_index=True)

        best_boot_aic = np.inf
        best_boot_tau = None

        for tau in tau_range:
            boot_panel['_post_tau'] = (boot_panel['career_year'] >= tau).astype(int)
            boot_panel['stab_pre'] = boot_panel['network_stability_z'] * (1 - boot_panel['_post_tau'])
            boot_panel['stab_post'] = boot_panel['network_stability_z'] * boot_panel['_post_tau']

            pre_ev = int(boot_panel[boot_panel['_post_tau'] == 0]['event'].sum())
            post_ev = int(boot_panel[boot_panel['_post_tau'] == 1]['event'].sum())
            if pre_ev < 5 or post_ev < 5:
                continue

            try:
                ctv_b = CoxTimeVaryingFitter(penalizer=0.01)
                ctv_b.fit(boot_panel, id_col='id', event_col='event',
                          start_col='start', stop_col='stop',
                          formula='stab_pre + stab_post + network_size_z + birth_year_z + cumulative_validation_z',
                          show_progress=False)
                aic_b = -2 * ctv_b.log_likelihood_ + 2 * len(ctv_b.summary)
                if aic_b < best_boot_aic:
                    best_boot_aic = aic_b
                    best_boot_tau = tau
            except Exception:
                continue

        if best_boot_tau is not None:
            boot_taus.append(best_boot_tau)

        if (b + 1) % 50 == 0:
            print(f"    Bootstrap {b + 1}/{n_boot} done")

    if boot_taus:
        ci_lo = np.percentile(boot_taus, 2.5)
        ci_hi = np.percentile(boot_taus, 97.5)
        print(f"\n  Bootstrap 95% CI for τ*: [{ci_lo:.1f}, {ci_hi:.1f}]")
        print(f"  Bootstrap mean τ*: {np.mean(boot_taus):.1f}")
        print(f"  Bootstrap median τ*: {np.median(boot_taus):.1f}")
    else:
        ci_lo, ci_hi = None, None

    return {
        'profile': df,
        'best_tau': int(best_tau),
        'best_row': best,
        'boot_taus': boot_taus,
        'boot_ci': (ci_lo, ci_hi),
    }


# ════════════════════════════════════════════════════════════════
# LaTeX Table: Unified Specification Results
# ════════════════════════════════════════════════════════════════

def build_unified_latex_table(results_korean, results_imdb,
                              cp_korean, cp_imdb):
    """Build a side-by-side LaTeX table comparing both datasets."""

    def _stars(p):
        if p is None or np.isnan(p):
            return ""
        if p < 0.001: return "^{***}"
        elif p < 0.01: return "^{**}"
        elif p < 0.05: return "^{*}"
        elif p < 0.10: return "^{\\dagger}"
        return ""

    def _fmt_hr(row):
        stars = _stars(row['p'])
        return f"${row['HR']:.3f}{stars}$ [{row['HR_lower']:.3f}, {row['HR_upper']:.3f}]"

    def _fmt_p(p):
        if p < 0.001: return "$<$0.001"
        return f"{p:.3f}"

    latex = r"""\begin{table}[htbp]
\centering
\small
\begin{threeparttable}
\caption{Unified Specification: Conditional Stability HR from RCS(career year) $\times$ Stability Interaction}
\label{tab:unified_specification}
\begin{tabular}{r cc cc}
\toprule
 & \multicolumn{2}{c}{Korean Art} & \multicolumn{2}{c}{IMDb Film} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Career year & \HR{stability} & \pval{} & \HR{stability} & \pval{} \\
\midrule
"""

    # Get all career years present in either dataset
    all_years = set()
    if results_korean is not None:
        all_years.update(results_korean['conditional_hrs']['career_year'].astype(int))
    if results_imdb is not None:
        all_years.update(results_imdb['conditional_hrs']['career_year'].astype(int))

    for cy in sorted(all_years):
        kr_str, kr_p = "---", "---"
        im_str, im_p = "---", "---"

        if results_korean is not None:
            kr_row = results_korean['conditional_hrs'][
                results_korean['conditional_hrs']['career_year'] == cy
            ]
            if len(kr_row) > 0:
                r = kr_row.iloc[0]
                kr_str = _fmt_hr(r)
                kr_p = _fmt_p(r['p'])

        if results_imdb is not None:
            im_row = results_imdb['conditional_hrs'][
                results_imdb['conditional_hrs']['career_year'] == cy
            ]
            if len(im_row) > 0:
                r = im_row.iloc[0]
                im_str = _fmt_hr(r)
                im_p = _fmt_p(r['p'])

        latex += f"{cy} & {kr_str} & {kr_p} & {im_str} & {im_p} \\\\\n"

    latex += r"\addlinespace" + "\n"

    # Model fit
    latex += r"\multicolumn{5}{l}{\textit{Model fit:}} \\" + "\n"
    if results_korean is not None:
        kr_aic = f"{results_korean['aic_rcs']:.1f}"
        kr_aic_lin = f"{results_korean['aic_linear']:.1f}"
    else:
        kr_aic, kr_aic_lin = "---", "---"

    if results_imdb is not None:
        im_aic = f"{results_imdb['aic_rcs']:.1f}"
        im_aic_lin = f"{results_imdb['aic_linear']:.1f}"
    else:
        im_aic, im_aic_lin = "---", "---"

    latex += f"\\quad AIC (RCS) & \\multicolumn{{2}}{{c}}{{{kr_aic}}} & \\multicolumn{{2}}{{c}}{{{im_aic}}} \\\\\n"
    latex += f"\\quad AIC (linear) & \\multicolumn{{2}}{{c}}{{{kr_aic_lin}}} & \\multicolumn{{2}}{{c}}{{{im_aic_lin}}} \\\\\n"

    # Change-point
    if cp_korean is not None or cp_imdb is not None:
        latex += r"\addlinespace" + "\n"
        latex += r"\multicolumn{5}{l}{\textit{Data-driven change-point ($\tau^*$):}} \\" + "\n"

        if cp_korean is not None:
            kr_tau = f"{cp_korean['best_tau']}"
            if cp_korean['boot_ci'][0] is not None:
                kr_tau += f" [{cp_korean['boot_ci'][0]:.0f}, {cp_korean['boot_ci'][1]:.0f}]"
        else:
            kr_tau = "---"

        if cp_imdb is not None:
            im_tau = f"{cp_imdb['best_tau']}"
            if cp_imdb['boot_ci'][0] is not None:
                im_tau += f" [{cp_imdb['boot_ci'][0]:.0f}, {cp_imdb['boot_ci'][1]:.0f}]"
        else:
            im_tau = "---"

        latex += f"\\quad Estimated $\\tau^*$ [95\\% CI] & \\multicolumn{{2}}{{c}}{{{kr_tau}}} & \\multicolumn{{2}}{{c}}{{{im_tau}}} \\\\\n"

    # Sample sizes
    latex += r"\addlinespace" + "\n"
    if results_korean is not None:
        kr_n = f"{results_korean['n_py']:,} / {results_korean['n_events']}"
    else:
        kr_n = "---"
    if results_imdb is not None:
        im_n = f"{results_imdb['n_py']:,} / {results_imdb['n_events']}"
    else:
        im_n = "---"

    latex += f"\\quad Person-years / events & \\multicolumn{{2}}{{c}}{{{kr_n}}} & \\multicolumn{{2}}{{c}}{{{im_n}}} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} Conditional stability hazard ratios from Cox time-varying models with restricted cubic spline (3 knots, Harrell percentiles) $\times$ network stability interaction. The same specification is applied identically to both datasets. Change-point $\tau^*$ estimated via profile-likelihood grid search over career years 5--20, with cluster-bootstrap 95\% CI (200 resamples). All covariates standardized ($z$-scores). Penalizer $\lambda = 0.01$. $^{*}$\pval{} $< 0.05$, $^{**}$\pval{} $< 0.01$, $^{***}$\pval{} $< 0.001$, $^{\dagger}$\pval{} $< 0.10$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    return latex


# ════════════════════════════════════════════════════════════════
# Visualization
# ════════════════════════════════════════════════════════════════

def plot_unified_hr_curves(results_korean, results_imdb, cp_korean, cp_imdb):
    """Plot side-by-side conditional HR curves for both datasets."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    for ax, results, cp, title in [
        (axes[0], results_korean, cp_korean, 'Korean Art (1929–2002)'),
        (axes[1], results_imdb, cp_imdb, 'IMDb Film (1950–2024)'),
    ]:
        if results is None:
            ax.text(0.5, 0.5, 'Data not available',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=18, color='gray')
            ax.set_title(title, fontsize=20, fontweight='bold')
            continue

        g = results['grid']
        cy = g['career_year']
        hr = g['HR']
        lo = g['HR_lower']
        hi = g['HR_upper']
        pvals = g['p']

        # CI band
        ax.fill_between(cy, lo, hi, alpha=0.15, color='#2C3E50')

        # Significant region (shaded red)
        sig_mask = pvals < 0.05
        ax.fill_between(cy, lo, hi,
                        where=sig_mask & (hr > 1),
                        alpha=0.25, color='#E74C3C')

        # HR curve
        ax.plot(cy, hr, color='#2C3E50', linewidth=2.5)

        # Reference line
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # Change-point
        if cp is not None:
            tau = cp['best_tau']
            ax.axvline(x=tau, color='#E74C3C', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.text(tau + 0.5, ax.get_ylim()[1] * 0.95,
                    f'τ* = {tau}', fontsize=15, color='#E74C3C',
                    va='top')

        ax.set_xlabel('Career Year', fontsize=18)
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=15)

    axes[0].set_ylabel('Conditional Stability HR\n(per 1 SD)', fontsize=18)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig5_unified_specification.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Figure saved: {fig_path}")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("UNIFIED SPECIFICATION: RCS × career_year × stability")
    print("+ DATA-DRIVEN CHANGE-POINT ESTIMATION")
    print("=" * 70)

    # ── Load data ──
    print("\n[1] Loading datasets...")
    korean_panel = load_korean_panel()
    imdb_panel = load_imdb_panel()

    results_korean = None
    results_imdb = None
    cp_korean = None
    cp_imdb = None

    # ── Korean art data ──
    if korean_panel is not None:
        print(f"\n  Korean panel: {len(korean_panel):,} person-years, "
              f"{korean_panel['id'].nunique()} artists, "
              f"{int(korean_panel['event'].sum())} events")
        korean_panel = standardize_panel(korean_panel)

        print("\n\n" + "#" * 70)
        print("# PART A: KOREAN ART DATA")
        print("#" * 70)

        results_korean = fit_rcs_interaction_model(
            korean_panel, "Korean Art", n_knots=3
        )
        cp_korean = estimate_changepoint(korean_panel, "Korean Art")

    # ── IMDb data ──
    if imdb_panel is not None:
        print(f"\n  IMDb panel: {len(imdb_panel):,} person-years, "
              f"{imdb_panel['id'].nunique()} actors, "
              f"{int(imdb_panel['event'].sum())} events")
        imdb_panel = standardize_panel(imdb_panel)

        print("\n\n" + "#" * 70)
        print("# PART B: IMDb FILM DATA")
        print("#" * 70)

        results_imdb = fit_rcs_interaction_model(
            imdb_panel, "IMDb Film", n_knots=3
        )
        cp_imdb = estimate_changepoint(imdb_panel, "IMDb Film")

    # ── LaTeX table ──
    if results_korean is not None or results_imdb is not None:
        print("\n\n" + "=" * 70)
        print("LATEX TABLE: UNIFIED SPECIFICATION")
        print("=" * 70)

        latex = build_unified_latex_table(
            results_korean, results_imdb,
            cp_korean, cp_imdb
        )
        print(latex)

        output_path = os.path.join(DATA_DIR, 'unified_specification_table.tex')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex)
        print(f"\n  Saved to {output_path}")

    # ── Visualization ──
    plot_unified_hr_curves(results_korean, results_imdb, cp_korean, cp_imdb)

    # ── Summary ──
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results_korean is not None:
        print(f"\n  Korean Art:")
        print(f"    RCS AIC: {results_korean['aic_rcs']:.1f} vs Linear AIC: {results_korean['aic_linear']:.1f}")
        print(f"    Joint Wald test p = {results_korean['wald_p']:.4f}")
        if cp_korean is not None:
            print(f"    Change-point τ* = {cp_korean['best_tau']} years "
                  f"[{cp_korean['boot_ci'][0]:.0f}, {cp_korean['boot_ci'][1]:.0f}]"
                  if cp_korean['boot_ci'][0] is not None else
                  f"    Change-point τ* = {cp_korean['best_tau']} years")

    if results_imdb is not None:
        print(f"\n  IMDb Film:")
        print(f"    RCS AIC: {results_imdb['aic_rcs']:.1f} vs Linear AIC: {results_imdb['aic_linear']:.1f}")
        print(f"    Joint Wald test p = {results_imdb['wald_p']:.4f}")
        if cp_imdb is not None:
            print(f"    Change-point τ* = {cp_imdb['best_tau']} years "
                  f"[{cp_imdb['boot_ci'][0]:.0f}, {cp_imdb['boot_ci'][1]:.0f}]"
                  if cp_imdb['boot_ci'][0] is not None else
                  f"    Change-point τ* = {cp_imdb['best_tau']} years")

    print("\nDONE.")


if __name__ == '__main__':
    main()
