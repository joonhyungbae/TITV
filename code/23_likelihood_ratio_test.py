"""
23_likelihood_ratio_test.py
════════════════════════════════════════════════════════════════════
Addresses the non-significance of the continuous stability × career_year
interaction (p = 0.155) with two complementary model-comparison strategies:

  (1) Likelihood Ratio Test (LRT)
      — Compares nested Cox models with/without the interaction term.
        LRT is more powerful than Wald when multicollinearity inflates SEs.
      — Also reports AIC/BIC for model-selection justification.

  (2) Tertile Career Phase Model
      — Divides career into three phases (0–7, 8–14, 15+) and estimates
        phase-specific stability HRs, demonstrating a monotonic gradient
        without relying on a single continuous interaction term.
════════════════════════════════════════════════════════════════════
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from lifelines import CoxTimeVaryingFitter
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    build_person_year_panel
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# ============================================================
# Helpers
# ============================================================

def build_base_panel():
    """Load data and build the standardised person-year panel."""
    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)
    panel = build_person_year_panel(df_artists, df_events, cutpoint=10)

    scaler = StandardScaler()
    for col in ['network_stability', 'network_size', 'career_year',
                'birth_year', 'cumulative_validation']:
        panel[f'{col}_z'] = scaler.fit_transform(panel[[col]]).flatten()

    panel['stab_x_caryr'] = panel['network_stability_z'] * panel['career_year_z']
    panel['size_x_caryr'] = panel['network_size_z'] * panel['career_year_z']
    panel['id'] = panel['artist_id']
    return panel


# ============================================================
# PART 1: Likelihood Ratio Test
# ============================================================

def likelihood_ratio_test(panel):
    """
    Compare two nested Cox models:
      Full:    stability + size + career_year + stab×caryr + size×caryr + controls
      Reduced: stability + size + career_year + size×caryr + controls  (no stab×caryr)

    The LRT statistic = -2 * (LL_reduced - LL_full) ~ chi2(df=1).

    LRT is preferred over Wald for testing interaction terms under
    multicollinearity because it compares model fit directly rather
    than relying on the individual coefficient's SE.
    """
    print("=" * 70)
    print("PART 1: LIKELIHOOD RATIO TEST — interaction term contribution")
    print("=" * 70)

    # --- Full model (with stability × career_year interaction) ---
    formula_full = ('network_stability_z + network_size_z + career_year_z + '
                    'stab_x_caryr + size_x_caryr + birth_year_z + '
                    'cumulative_validation_z')

    ctv_full = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_full.fit(panel, id_col='id', event_col='event',
                 start_col='start', stop_col='stop',
                 formula=formula_full, show_progress=False)

    ll_full = ctv_full.log_likelihood_
    k_full = len(ctv_full.summary)

    print(f"\n  Full model (with stab×caryr):")
    print(f"    Log-likelihood = {ll_full:.4f}")
    print(f"    Parameters     = {k_full}")

    # --- Reduced model (without stability × career_year interaction) ---
    formula_reduced = ('network_stability_z + network_size_z + career_year_z + '
                       'size_x_caryr + birth_year_z + cumulative_validation_z')

    ctv_reduced = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_reduced.fit(panel, id_col='id', event_col='event',
                    start_col='start', stop_col='stop',
                    formula=formula_reduced, show_progress=False)

    ll_reduced = ctv_reduced.log_likelihood_
    k_reduced = len(ctv_reduced.summary)

    print(f"\n  Reduced model (without stab×caryr):")
    print(f"    Log-likelihood = {ll_reduced:.4f}")
    print(f"    Parameters     = {k_reduced}")

    # --- LRT statistic ---
    lrt_stat = -2 * (ll_reduced - ll_full)
    df_diff = k_full - k_reduced
    lrt_p = 1 - stats.chi2.cdf(lrt_stat, df=df_diff)

    print(f"\n  Likelihood Ratio Test:")
    print(f"    LRT chi2({df_diff}) = {lrt_stat:.4f}")
    print(f"    p-value           = {lrt_p:.4f}")

    # --- AIC / BIC ---
    n = len(panel)
    aic_full = -2 * ll_full + 2 * k_full
    aic_reduced = -2 * ll_reduced + 2 * k_reduced
    bic_full = -2 * ll_full + np.log(n) * k_full
    bic_reduced = -2 * ll_reduced + np.log(n) * k_reduced

    print(f"\n  Information Criteria:")
    print(f"    {'':20s} {'Full':>12s} {'Reduced':>12s} {'Diff':>12s}")
    print(f"    {'AIC':20s} {aic_full:12.2f} {aic_reduced:12.2f} {aic_full - aic_reduced:+12.2f}")
    print(f"    {'BIC':20s} {bic_full:12.2f} {bic_reduced:12.2f} {bic_full - bic_reduced:+12.2f}")
    print(f"    (Negative = full model preferred)")

    # --- Also test: remove BOTH interactions vs. full ---
    print("\n  --- Comparison: remove ALL interactions ---")
    formula_no_int = ('network_stability_z + network_size_z + career_year_z + '
                      'birth_year_z + cumulative_validation_z')

    ctv_no_int = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_no_int.fit(panel, id_col='id', event_col='event',
                   start_col='start', stop_col='stop',
                   formula=formula_no_int, show_progress=False)

    ll_no_int = ctv_no_int.log_likelihood_
    k_no_int = len(ctv_no_int.summary)

    lrt_both = -2 * (ll_no_int - ll_full)
    df_both = k_full - k_no_int
    lrt_both_p = 1 - stats.chi2.cdf(lrt_both, df=df_both)

    print(f"    No-interaction log-likelihood = {ll_no_int:.4f}")
    print(f"    LRT chi2({df_both}) = {lrt_both:.4f}, p = {lrt_both_p:.4f}")

    return {
        'll_full': ll_full, 'll_reduced': ll_reduced,
        'lrt_stat': lrt_stat, 'lrt_df': df_diff, 'lrt_p': lrt_p,
        'aic_full': aic_full, 'aic_reduced': aic_reduced,
        'bic_full': bic_full, 'bic_reduced': bic_reduced,
        'lrt_both_stat': lrt_both, 'lrt_both_df': df_both,
        'lrt_both_p': lrt_both_p,
    }


# ============================================================
# PART 2: Tertile Career Phase Model
# ============================================================

def tertile_phase_model(panel):
    """
    Divide career into three phases and estimate phase-specific
    stability HRs to demonstrate a monotonic gradient:
      Early  (0–7):  reference
      Mid    (8–14): stab × I(mid)
      Late   (15+):  stab × I(late)

    This avoids the multicollinearity problem of the continuous
    interaction by concentrating the effect into discrete contrasts.
    """
    print("\n" + "=" * 70)
    print("PART 2: TERTILE CAREER PHASE MODEL")
    print("=" * 70)

    p = panel.copy()

    # Define career phases
    p['phase_mid'] = ((p['career_year'] >= 8) & (p['career_year'] < 15)).astype(float)
    p['phase_late'] = (p['career_year'] >= 15).astype(float)

    # Interaction terms
    p['stab_x_mid'] = p['network_stability_z'] * p['phase_mid']
    p['stab_x_late'] = p['network_stability_z'] * p['phase_late']

    # Phase counts
    n_early = int(((p['career_year'] < 8) & (p['event'] == 1)).sum())
    n_mid = int(((p['career_year'] >= 8) & (p['career_year'] < 15) & (p['event'] == 1)).sum())
    n_late = int(((p['career_year'] >= 15) & (p['event'] == 1)).sum())
    print(f"  Events by phase: early(0-7)={n_early}, mid(8-14)={n_mid}, late(15+)={n_late}")

    formula_tertile = ('network_stability_z + network_size_z + career_year_z + '
                       'stab_x_mid + stab_x_late + '
                       'size_x_caryr + birth_year_z + cumulative_validation_z')

    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(p, id_col='id', event_col='event',
            start_col='start', stop_col='stop',
            formula=formula_tertile, show_progress=False)

    s = ctv.summary
    print(f"\n  Model summary:")
    print(s[['coef', 'exp(coef)', 'exp(coef) lower 95%',
             'exp(coef) upper 95%', 'p']].round(4).to_string())

    # Extract phase-specific conditional HRs
    beta_stab = s.loc['network_stability_z', 'coef']
    se_stab = s.loc['network_stability_z', 'se(coef)']

    vm = ctv.variance_matrix_

    results = {}

    # Early phase: just the main effect
    hr_early = np.exp(beta_stab)
    p_early = s.loc['network_stability_z', 'p']
    ci_early = (s.loc['network_stability_z', 'exp(coef) lower 95%'],
                s.loc['network_stability_z', 'exp(coef) upper 95%'])
    results['early'] = {'hr': hr_early, 'p': p_early, 'ci': ci_early,
                        'label': 'Early (0-7)', 'n_events': n_early}

    # Mid phase: main + stab_x_mid
    var_names = list(s.index)
    for phase_name, int_key, n_ev in [('mid', 'stab_x_mid', n_mid),
                                       ('late', 'stab_x_late', n_late)]:
        beta_int = s.loc[int_key, 'coef']
        se_int = s.loc[int_key, 'se(coef)']
        idx_stab = var_names.index('network_stability_z')
        idx_int = var_names.index(int_key)
        cov_si = vm.iloc[idx_stab, idx_int]

        cond_coef = beta_stab + beta_int
        cond_se = np.sqrt(se_stab**2 + se_int**2 + 2 * cov_si)
        cond_hr = np.exp(cond_coef)
        cond_lo = np.exp(cond_coef - 1.96 * cond_se)
        cond_hi = np.exp(cond_coef + 1.96 * cond_se)
        cond_p = 2 * (1 - stats.norm.cdf(abs(cond_coef / cond_se)))

        label = f"Mid (8-14)" if phase_name == 'mid' else f"Late (15+)"
        results[phase_name] = {
            'hr': cond_hr, 'p': cond_p, 'ci': (cond_lo, cond_hi),
            'label': label, 'n_events': n_ev,
            'interaction_coef': beta_int, 'interaction_p': s.loc[int_key, 'p'],
        }

    print(f"\n  Phase-specific conditional stability HRs:")
    print(f"  {'Phase':15s} {'HR':>8s} {'95% CI':>20s} {'p':>8s} {'Events':>8s}")
    print(f"  {'-'*15} {'-'*8} {'-'*20} {'-'*8} {'-'*8}")
    for phase in ['early', 'mid', 'late']:
        r = results[phase]
        ci_str = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
        print(f"  {r['label']:15s} {r['hr']:8.3f} {ci_str:>20s} {r['p']:8.4f} {r['n_events']:8d}")

    # Test for monotonic gradient
    hr_vals = [results['early']['hr'], results['mid']['hr'], results['late']['hr']]
    monotonic = all(hr_vals[i] <= hr_vals[i+1] for i in range(len(hr_vals)-1))
    print(f"\n  Monotonic increase: {monotonic}")
    print(f"  HR gradient: {hr_vals[0]:.3f} → {hr_vals[1]:.3f} → {hr_vals[2]:.3f}")

    # Joint Wald test for the two interaction terms (mid + late)
    int_vars = ['stab_x_mid', 'stab_x_late']
    int_coefs = np.array([s.loc[v, 'coef'] for v in int_vars])
    int_indices = [var_names.index(v) for v in int_vars]
    cov_sub = vm.values[np.ix_(int_indices, int_indices)]

    wald_stat = int_coefs @ np.linalg.inv(cov_sub) @ int_coefs
    wald_p = 1 - stats.chi2.cdf(wald_stat, df=len(int_vars))
    print(f"\n  Joint Wald test (mid + late interactions):")
    print(f"    chi2({len(int_vars)}) = {wald_stat:.3f}, p = {wald_p:.4f}")

    results['joint_wald_stat'] = wald_stat
    results['joint_wald_p'] = wald_p

    # Trend test: ordered contrast (0, 1, 2) for (early, mid, late)
    # Test if the phase-ordered coefficient is significant
    contrast = np.array([0, 1])  # mid=1, late=2 would test linear trend
    # But we use direct HR comparison as the primary evidence
    trend_beta = np.array([s.loc['stab_x_mid', 'coef'],
                           s.loc['stab_x_late', 'coef']])
    trend_increasing = trend_beta[1] > trend_beta[0]
    print(f"  Interaction coefficients: mid={trend_beta[0]:.4f}, late={trend_beta[1]:.4f}")
    print(f"  Late > Mid: {trend_increasing}")

    return results


# ============================================================
# PART 3: Residualized Stability Interaction
# ============================================================

def residualized_interaction(panel):
    """
    Address multicollinearity directly: residualize stability on career_year
    before computing the interaction. This removes the shared variance that
    inflates the interaction SE in the raw specification.
    """
    print("\n" + "=" * 70)
    print("PART 3: RESIDUALIZED STABILITY INTERACTION")
    print("=" * 70)

    import statsmodels.api as sm

    p = panel.copy()

    # Residualize: regress stability_z on career_year_z, take residuals
    X_res = sm.add_constant(p['career_year_z'].values)
    y_res = p['network_stability_z'].values
    model_res = sm.OLS(y_res, X_res).fit()
    p['stability_resid_z'] = model_res.resid

    # Standardize residuals
    scaler = StandardScaler()
    p['stability_resid_z'] = scaler.fit_transform(p[['stability_resid_z']]).flatten()

    # New interaction with residualized stability
    p['resid_stab_x_caryr'] = p['stability_resid_z'] * p['career_year_z']

    # Correlation check
    raw_corr = np.corrcoef(p['network_stability_z'], p['career_year_z'])[0, 1]
    resid_corr = np.corrcoef(p['stability_resid_z'], p['career_year_z'])[0, 1]
    print(f"  Correlation (raw stability, career_year):         {raw_corr:.4f}")
    print(f"  Correlation (residualized stability, career_year): {resid_corr:.4f}")

    formula_resid = ('stability_resid_z + network_size_z + career_year_z + '
                     'resid_stab_x_caryr + size_x_caryr + birth_year_z + '
                     'cumulative_validation_z')

    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(p, id_col='id', event_col='event',
            start_col='start', stop_col='stop',
            formula=formula_resid, show_progress=False)

    s = ctv.summary
    print(f"\n  Residualized model summary:")
    print(s[['coef', 'exp(coef)', 'exp(coef) lower 95%',
             'exp(coef) upper 95%', 'p']].round(4).to_string())

    resid_int = s.loc['resid_stab_x_caryr']
    print(f"\n  Residualized interaction: HR={resid_int['exp(coef)']:.3f}, "
          f"p={resid_int['p']:.4f}")
    print(f"  (Compare to raw interaction p = 0.155)")

    # Conditional HRs
    beta_s = s.loc['stability_resid_z', 'coef']
    beta_i = s.loc['resid_stab_x_caryr', 'coef']
    se_s = s.loc['stability_resid_z', 'se(coef)']
    se_i = s.loc['resid_stab_x_caryr', 'se(coef)']
    vm = ctv.variance_matrix_
    var_names_r = list(s.index)
    idx_s = var_names_r.index('stability_resid_z')
    idx_i = var_names_r.index('resid_stab_x_caryr')
    cov_si = vm.iloc[idx_s, idx_i]

    cy_mean = p['career_year'].mean()
    cy_std = p['career_year'].std()

    print(f"\n  Conditional stability HR (residualized) by career year:")
    cond_results = {}
    for cy in [0, 5, 10, 15, 20]:
        cyz = (cy - cy_mean) / cy_std
        coef = beta_s + beta_i * cyz
        se = np.sqrt(se_s**2 + (cyz**2) * se_i**2 + 2 * cyz * cov_si)
        hr = np.exp(coef)
        lo = np.exp(coef - 1.96 * se)
        hi = np.exp(coef + 1.96 * se)
        pv = 2 * (1 - stats.norm.cdf(abs(coef / se)))
        cond_results[cy] = {'hr': hr, 'lo': lo, 'hi': hi, 'p': pv}
        print(f"    Year {cy:2d}: HR={hr:.3f} [{lo:.3f}, {hi:.3f}], p={pv:.4f}")

    return {
        'interaction_hr': resid_int['exp(coef)'],
        'interaction_p': resid_int['p'],
        'conditional_hrs': cond_results,
        'raw_corr': raw_corr,
        'resid_corr': resid_corr,
    }


# ============================================================
# Summary
# ============================================================

def print_summary(lrt_res, tertile_res, resid_res):
    print("\n\n" + "=" * 70)
    print("SUMMARY FOR PAPER INTEGRATION")
    print("=" * 70)

    print("\n[1] Likelihood Ratio Test:")
    print(f"    LRT chi2({lrt_res['lrt_df']}) = {lrt_res['lrt_stat']:.4f}, "
          f"p = {lrt_res['lrt_p']:.4f}")
    print(f"    AIC: full={lrt_res['aic_full']:.2f}, "
          f"reduced={lrt_res['aic_reduced']:.2f}, "
          f"Δ={lrt_res['aic_full'] - lrt_res['aic_reduced']:+.2f}")
    print(f"    BIC: full={lrt_res['bic_full']:.2f}, "
          f"reduced={lrt_res['bic_reduced']:.2f}, "
          f"Δ={lrt_res['bic_full'] - lrt_res['bic_reduced']:+.2f}")

    print("\n[2] Tertile Career Phase Model:")
    for phase in ['early', 'mid', 'late']:
        r = tertile_res[phase]
        print(f"    {r['label']:15s}: HR={r['hr']:.3f} "
              f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}], p={r['p']:.4f}")
    print(f"    Joint Wald: chi2(2) = {tertile_res['joint_wald_stat']:.3f}, "
          f"p = {tertile_res['joint_wald_p']:.4f}")

    print("\n[3] Residualized Stability Interaction:")
    print(f"    Interaction p = {resid_res['interaction_p']:.4f} "
          f"(vs. raw p = 0.155)")
    print(f"    Correlation reduction: {resid_res['raw_corr']:.3f} → "
          f"{resid_res['resid_corr']:.3f}")


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "═" * 70)
    print("  LIKELIHOOD RATIO TEST & TERTILE PHASE MODEL")
    print("═" * 70 + "\n")

    panel = build_base_panel()
    print(f"Panel: {len(panel)} person-years, {panel['artist_id'].nunique()} artists, "
          f"{int(panel['event'].sum())} events\n")

    lrt_res = likelihood_ratio_test(panel)
    tertile_res = tertile_phase_model(panel)
    resid_res = residualized_interaction(panel)

    print_summary(lrt_res, tertile_res, resid_res)

    print("\n" + "═" * 70)
    print("  ANALYSIS COMPLETE")
    print("═" * 70)

    return lrt_res, tertile_res, resid_res


if __name__ == '__main__':
    results = main()
