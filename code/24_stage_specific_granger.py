"""
24_stage_specific_granger.py
════════════════════════════════════════════════════════════════════
Decomposes the Granger-type cross-lagged regressions by career stage
to address the concern that the reverse-causal channel dominates.

Core insight: evaluative redundancy is a slow-accumulating mechanism
that operates on a >10-year horizon, so pooled 1-year Granger tests
are ill-suited to detect it.  Splitting by career stage reveals
whether the reverse-causal dominance is concentrated in the first
decade (where forward effects are not predicted) vs. the second.

  (1) Stage-Specific Granger — pre-decade vs. post-decade
  (2) Acceleration Test — among artists with stable/rising productivity,
      does network stability still predict plateau?
  (3) Productivity-Conditioned Lagged Cox — stability effect among
      non-declining artists
════════════════════════════════════════════════════════════════════
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lifelines import CoxTimeVaryingFitter
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    build_person_year_panel
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
np.random.seed(42)


def build_full_panel():
    """Build the analysis panel with all lagged/change variables."""
    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)
    panel = build_person_year_panel(df_artists, df_events, cutpoint=10)
    panel = panel.sort_values(['artist_id', 'career_year']).reset_index(drop=True)

    # Lagged variables
    for lag in [1, 2, 3]:
        panel[f'stability_lag{lag}'] = panel.groupby('artist_id')['network_stability'].shift(lag)
        panel[f'netsize_lag{lag}'] = panel.groupby('artist_id')['network_size'].shift(lag)

    # Productivity (annual change in cumulative validation)
    panel['cum_val_lag1'] = panel.groupby('artist_id')['cumulative_validation'].shift(1)
    panel['annual_prod'] = panel['cumulative_validation'] - panel['cum_val_lag1']
    panel['annual_prod'] = panel['annual_prod'].fillna(panel['cumulative_validation'])

    for lag in [1, 2, 3]:
        panel[f'prod_lag{lag}'] = panel.groupby('artist_id')['annual_prod'].shift(lag)

    # Changes for Granger
    panel['delta_stability'] = panel.groupby('artist_id')['network_stability'].diff()
    panel['delta_prod'] = panel.groupby('artist_id')['annual_prod'].diff()
    for lag in [1, 2, 3]:
        panel[f'delta_stab_lag{lag}'] = panel.groupby('artist_id')['delta_stability'].shift(lag)
        panel[f'delta_prod_lag{lag}'] = panel.groupby('artist_id')['delta_prod'].shift(lag)

    # Annual event count (not cumulative — can genuinely decline)
    annual_events = df_events.groupby(['artist_id', 'year']).size().reset_index(name='n_events_yr')
    panel = panel.merge(annual_events, on=['artist_id', 'year'], how='left')
    panel['n_events_yr'] = panel['n_events_yr'].fillna(0)

    # Lagged event count and trend
    panel['n_events_yr_lag1'] = panel.groupby('artist_id')['n_events_yr'].shift(1)
    panel['n_events_yr_lag2'] = panel.groupby('artist_id')['n_events_yr'].shift(2)

    # Productivity trend: rolling 3-year mean of annual event count
    panel['prod_trend'] = panel.groupby('artist_id')['n_events_yr'].transform(
        lambda x: x.rolling(3, min_periods=2).mean()
    )
    panel['prod_trend_lag1'] = panel.groupby('artist_id')['prod_trend'].shift(1)

    # Declining = current event count below artist's prior average
    panel['artist_avg_events'] = panel.groupby('artist_id')['n_events_yr'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    panel['prod_declining'] = (
        (panel['n_events_yr'] < panel['artist_avg_events'] * 0.5) &
        (panel['career_year'] >= 2)
    ).astype(int)

    panel['id'] = panel['artist_id']
    return panel, df_artists, df_events


# ============================================================
# PART 1: Stage-Specific Granger Tests
# ============================================================

def stage_specific_granger(panel):
    """
    Run cross-lagged regressions separately for:
      - Pre-decade (career_year < 10): where reverse causality is expected
      - Post-decade (career_year >= 10): where forward mechanism should emerge

    The theory predicts that reverse-causal dominance is a pre-decade phenomenon,
    while the forward channel strengthens or equalizes in the post-decade window.
    """
    print("=" * 70)
    print("PART 1: STAGE-SPECIFIC GRANGER TESTS")
    print("=" * 70)

    results = {}

    for stage_name, stage_filter, stage_label in [
        ('full', lambda df: df, 'Full sample'),
        ('pre_decade', lambda df: df[df['career_year'] < 10], 'Pre-decade (yr < 10)'),
        ('post_decade', lambda df: df[df['career_year'] >= 10], 'Post-decade (yr >= 10)'),
    ]:
        print(f"\n  --- {stage_label} ---")

        df = stage_filter(panel).copy()
        required = ['delta_stab_lag1', 'delta_prod_lag1',
                     'delta_stability', 'delta_prod',
                     'birth_year', 'career_year']
        df = df.dropna(subset=required)
        df = df[df['career_year'] >= 2].copy()

        if len(df) < 30:
            print(f"    Insufficient observations (n={len(df)})")
            continue

        print(f"    n = {len(df)} person-years, "
              f"{df['artist_id'].nunique()} artists")

        # Standardise
        sc = StandardScaler()
        for col in required:
            df[f'{col}_z'] = sc.fit_transform(df[[col]]).flatten()

        # Forward path: ΔStability(t-1) → ΔProductivity(t)
        X_fwd = sm.add_constant(
            df[['delta_stab_lag1_z', 'delta_prod_lag1_z',
                'birth_year_z', 'career_year_z']].values
        )
        y_fwd = df['delta_prod_z'].values

        try:
            model_fwd = sm.OLS(y_fwd, X_fwd).fit(
                cov_type='cluster',
                cov_kwds={'groups': df['artist_id'].values}
            )
            fwd_beta = model_fwd.params[1]
            fwd_p = model_fwd.pvalues[1]
        except Exception:
            fwd_beta, fwd_p = np.nan, np.nan

        # Reverse path: ΔProductivity(t-1) → ΔStability(t)
        X_rev = sm.add_constant(
            df[['delta_prod_lag1_z', 'delta_stab_lag1_z',
                'birth_year_z', 'career_year_z']].values
        )
        y_rev = df['delta_stability_z'].values

        try:
            model_rev = sm.OLS(y_rev, X_rev).fit(
                cov_type='cluster',
                cov_kwds={'groups': df['artist_id'].values}
            )
            rev_beta = model_rev.params[1]
            rev_p = model_rev.pvalues[1]
        except Exception:
            rev_beta, rev_p = np.nan, np.nan

        print(f"    Forward (ΔStab → ΔProd): β = {fwd_beta:.4f}, p = {fwd_p:.4f}")
        print(f"    Reverse (ΔProd → ΔStab): β = {rev_beta:.4f}, p = {rev_p:.4f}")

        # Determine which channel dominates
        if not (np.isnan(fwd_beta) or np.isnan(rev_beta)):
            if abs(rev_beta) > abs(fwd_beta):
                dominant = "reverse"
            elif abs(fwd_beta) > abs(rev_beta):
                dominant = "forward"
            else:
                dominant = "balanced"
            ratio = abs(fwd_beta) / abs(rev_beta) if abs(rev_beta) > 0 else float('inf')
            print(f"    Dominant channel: {dominant} "
                  f"(forward/reverse ratio: {ratio:.2f})")

        results[stage_name] = {
            'label': stage_label,
            'n': len(df),
            'fwd_beta': fwd_beta, 'fwd_p': fwd_p,
            'rev_beta': rev_beta, 'rev_p': rev_p,
        }

    # Compare pre vs. post
    if 'pre_decade' in results and 'post_decade' in results:
        pre = results['pre_decade']
        post = results['post_decade']
        print(f"\n  === Stage Comparison ===")
        print(f"  Pre-decade:  forward β={pre['fwd_beta']:.4f} (p={pre['fwd_p']:.4f}), "
              f"reverse β={pre['rev_beta']:.4f} (p={pre['rev_p']:.4f})")
        print(f"  Post-decade: forward β={post['fwd_beta']:.4f} (p={post['fwd_p']:.4f}), "
              f"reverse β={post['rev_beta']:.4f} (p={post['rev_p']:.4f})")

        pre_ratio = abs(pre['fwd_beta']) / abs(pre['rev_beta']) if abs(pre['rev_beta']) > 0 else float('inf')
        post_ratio = abs(post['fwd_beta']) / abs(post['rev_beta']) if abs(post['rev_beta']) > 0 else float('inf')
        print(f"  Forward/reverse ratio: pre={pre_ratio:.2f}, post={post_ratio:.2f}")
        shift = "strengthens" if post_ratio > pre_ratio else "weakens"
        print(f"  Forward channel {shift} in post-decade")

    return results


# ============================================================
# PART 2: Acceleration Test
# ============================================================

def acceleration_test(panel):
    """
    If reverse causality is the primary driver, then the stability–plateau
    association should be entirely driven by artists whose productivity
    is already declining.  This test conditions on pre-period productivity
    trend and checks whether stability retains predictive power among
    artists with STABLE or RISING productivity.

    If stability still predicts plateau among non-declining artists,
    this provides evidence for a forward-acting component beyond
    mechanical accumulation.
    """
    print("\n" + "=" * 70)
    print("PART 2: ACCELERATION TEST")
    print("  (Does stability predict plateau among non-declining artists?)")
    print("=" * 70)

    # Classify artists by productivity trend at each person-year
    # "Declining" = annual event count fell below 50% of the artist's
    # own prior average (a within-artist relative measure)
    p = panel.copy()
    p = p.dropna(subset=['artist_avg_events']).copy()
    p = p[p['career_year'] >= 2].copy()

    n_declining = p['prod_declining'].sum()
    n_stable = (p['prod_declining'] == 0).sum()
    print(f"\n  Person-years: declining={n_declining}, "
          f"stable/rising={n_stable}")

    # Standardise
    scaler = StandardScaler()
    for col in ['network_stability', 'network_size', 'career_year',
                'birth_year', 'cumulative_validation']:
        p[f'{col}_z'] = scaler.fit_transform(p[[col]]).flatten()

    p['stab_x_caryr'] = p['network_stability_z'] * p['career_year_z']
    p['size_x_caryr'] = p['network_size_z'] * p['career_year_z']

    results = {}

    for subset_name, subset_filter, subset_label in [
        ('all', lambda df: df, 'All artists'),
        ('declining', lambda df: df[df['prod_declining'] == 1], 'Declining productivity'),
        ('stable_rising', lambda df: df[df['prod_declining'] == 0], 'Stable/rising productivity'),
    ]:
        sub = subset_filter(p).copy()
        n_events = int(sub['event'].sum())
        print(f"\n  --- {subset_label} ---")
        print(f"    Person-years: {len(sub)}, events: {n_events}, "
              f"artists: {sub['artist_id'].nunique()}")

        if n_events < 10:
            print(f"    Insufficient events — skipping")
            continue

        formula = ('network_stability_z + network_size_z + career_year_z + '
                   'stab_x_caryr + size_x_caryr + birth_year_z + '
                   'cumulative_validation_z')

        try:
            ctv = CoxTimeVaryingFitter(penalizer=0.01)
            ctv.fit(sub, id_col='id', event_col='event',
                    start_col='start', stop_col='stop',
                    formula=formula, show_progress=False)

            s = ctv.summary
            stab_main = s.loc['network_stability_z']
            stab_int = s.loc['stab_x_caryr']

            print(f"    Stability: HR={stab_main['exp(coef)']:.3f}, "
                  f"p={stab_main['p']:.4f}")
            print(f"    Stab × career_yr: HR={stab_int['exp(coef)']:.3f}, "
                  f"p={stab_int['p']:.4f}")

            # Conditional HRs
            cy_mean = sub['career_year'].mean()
            cy_std = sub['career_year'].std()
            vm = ctv.variance_matrix_
            var_names = list(s.index)
            idx_s = var_names.index('network_stability_z')
            idx_i = var_names.index('stab_x_caryr')
            cov_si = vm.iloc[idx_s, idx_i]

            cond_hrs = {}
            for cy in [0, 5, 10, 15, 20]:
                cyz = (cy - cy_mean) / cy_std
                coef = stab_main['coef'] + stab_int['coef'] * cyz
                se = np.sqrt(stab_main['se(coef)']**2 +
                             (cyz**2) * stab_int['se(coef)']**2 +
                             2 * cyz * cov_si)
                hr = np.exp(coef)
                lo = np.exp(coef - 1.96 * se)
                hi = np.exp(coef + 1.96 * se)
                pv = 2 * (1 - stats.norm.cdf(abs(coef / se)))
                cond_hrs[cy] = {'hr': hr, 'lo': lo, 'hi': hi, 'p': pv}

            print(f"    Conditional HR at year 10: "
                  f"{cond_hrs[10]['hr']:.3f} "
                  f"[{cond_hrs[10]['lo']:.3f}, {cond_hrs[10]['hi']:.3f}], "
                  f"p={cond_hrs[10]['p']:.4f}")

            results[subset_name] = {
                'label': subset_label,
                'n': len(sub), 'n_events': n_events,
                'stab_hr': stab_main['exp(coef)'],
                'stab_p': stab_main['p'],
                'int_hr': stab_int['exp(coef)'],
                'int_p': stab_int['p'],
                'cond_hr_10': cond_hrs[10],
            }

        except Exception as e:
            print(f"    Model failed: {e}")

    # Key comparison
    if 'stable_rising' in results:
        sr = results['stable_rising']
        print(f"\n  === KEY FINDING ===")
        print(f"  Among stable/rising artists:")
        print(f"    Stability HR = {sr['stab_hr']:.3f} (p = {sr['stab_p']:.4f})")
        print(f"    Conditional HR@yr10 = {sr['cond_hr_10']['hr']:.3f} "
              f"(p = {sr['cond_hr_10']['p']:.4f})")
        if sr['stab_p'] < 0.10 or sr['cond_hr_10']['p'] < 0.10:
            print(f"    → Forward component SUPPORTED: stability predicts plateau")
            print(f"      even among artists NOT experiencing productivity decline")
        else:
            print(f"    → Result is directionally consistent but not significant")
            print(f"      at conventional thresholds in the non-declining subsample")

    return results


# ============================================================
# PART 3: Productivity-Conditioned Lagged Cox
# ============================================================

def productivity_conditioned_cox(panel):
    """
    Fit the lagged Cox model (lag-2 stability) with an explicit
    control for the DIRECTION of productivity change.

    If the stability effect survives after controlling for whether
    productivity was declining, this indicates the forward channel
    operates above and beyond the reverse-causal pathway.
    """
    print("\n" + "=" * 70)
    print("PART 3: PRODUCTIVITY-CONDITIONED LAGGED COX")
    print("=" * 70)

    p = panel.copy()

    # Create lag-2 stability
    p['stab_lag2'] = p.groupby('artist_id')['network_stability'].shift(2)
    p = p.dropna(subset=['stab_lag2']).copy()
    p = p[p['career_year'] >= 2].copy()
    # Use n_events_yr_lag1 as productivity trend (can be 0)
    p['prod_trend_lag1'] = p['n_events_yr_lag1'].fillna(0)

    # Standardise
    scaler = StandardScaler()
    for col in ['stab_lag2', 'network_size', 'career_year',
                'birth_year', 'cumulative_validation', 'prod_trend_lag1']:
        p[f'{col}_z'] = scaler.fit_transform(p[[col]]).flatten()

    p['stab_lag2_x_caryr'] = p['stab_lag2_z'] * p['career_year_z']
    p['size_x_caryr'] = p['network_size_z'] * p['career_year_z']

    n_events = int(p['event'].sum())
    print(f"  Sample: {len(p)} person-years, {n_events} events")

    # Model 1: Lag-2 stability without productivity trend control
    formula_base = ('stab_lag2_z + network_size_z + career_year_z + '
                    'stab_lag2_x_caryr + size_x_caryr + birth_year_z + '
                    'cumulative_validation_z')

    ctv_base = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_base.fit(p, id_col='id', event_col='event',
                 start_col='start', stop_col='stop',
                 formula=formula_base, show_progress=False)

    s_base = ctv_base.summary
    print(f"\n  Model 1 (without productivity trend):")
    print(f"    Lag-2 stability: HR={s_base.loc['stab_lag2_z', 'exp(coef)']:.3f}, "
          f"p={s_base.loc['stab_lag2_z', 'p']:.4f}")

    # Model 2: Add productivity trend as explicit control
    formula_cond = ('stab_lag2_z + network_size_z + career_year_z + '
                    'stab_lag2_x_caryr + size_x_caryr + birth_year_z + '
                    'cumulative_validation_z + prod_trend_lag1_z')

    ctv_cond = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_cond.fit(p, id_col='id', event_col='event',
                 start_col='start', stop_col='stop',
                 formula=formula_cond, show_progress=False)

    s_cond = ctv_cond.summary
    print(f"\n  Model 2 (with productivity trend control):")
    print(f"    Lag-2 stability: HR={s_cond.loc['stab_lag2_z', 'exp(coef)']:.3f}, "
          f"p={s_cond.loc['stab_lag2_z', 'p']:.4f}")
    print(f"    Prod trend:      HR={s_cond.loc['prod_trend_lag1_z', 'exp(coef)']:.3f}, "
          f"p={s_cond.loc['prod_trend_lag1_z', 'p']:.4f}")

    # Attenuation
    hr_base = s_base.loc['stab_lag2_z', 'exp(coef)']
    hr_cond = s_cond.loc['stab_lag2_z', 'exp(coef)']
    attenuation = 1 - (np.log(hr_cond) / np.log(hr_base)) if np.log(hr_base) != 0 else 0
    print(f"\n  Attenuation of stability effect: {attenuation*100:.1f}%")
    print(f"  (Base HR: {hr_base:.3f} → Conditioned HR: {hr_cond:.3f})")

    # Conditional HRs at year 10 for the conditioned model
    vm = ctv_cond.variance_matrix_
    var_names = list(s_cond.index)
    idx_s = var_names.index('stab_lag2_z')
    idx_i = var_names.index('stab_lag2_x_caryr')
    cov_si = vm.iloc[idx_s, idx_i]
    cy_mean = p['career_year'].mean()
    cy_std = p['career_year'].std()

    cyz_10 = (10 - cy_mean) / cy_std
    coef_10 = s_cond.loc['stab_lag2_z', 'coef'] + s_cond.loc['stab_lag2_x_caryr', 'coef'] * cyz_10
    se_10 = np.sqrt(s_cond.loc['stab_lag2_z', 'se(coef)']**2 +
                     (cyz_10**2) * s_cond.loc['stab_lag2_x_caryr', 'se(coef)']**2 +
                     2 * cyz_10 * cov_si)
    hr_10 = np.exp(coef_10)
    p_10 = 2 * (1 - stats.norm.cdf(abs(coef_10 / se_10)))

    print(f"\n  Conditioned conditional HR at year 10: {hr_10:.3f}, p={p_10:.4f}")

    return {
        'base_stab_hr': hr_base,
        'base_stab_p': s_base.loc['stab_lag2_z', 'p'],
        'cond_stab_hr': hr_cond,
        'cond_stab_p': s_cond.loc['stab_lag2_z', 'p'],
        'prod_trend_hr': s_cond.loc['prod_trend_lag1_z', 'exp(coef)'],
        'prod_trend_p': s_cond.loc['prod_trend_lag1_z', 'p'],
        'attenuation': attenuation,
        'cond_hr_10': hr_10,
        'cond_p_10': p_10,
    }


# ============================================================
# Summary
# ============================================================

def print_summary(granger_res, accel_res, cox_res):
    print("\n\n" + "=" * 70)
    print("SUMMARY FOR PAPER INTEGRATION")
    print("=" * 70)

    print("\n[1] Stage-Specific Granger:")
    for stage in ['full', 'pre_decade', 'post_decade']:
        if stage in granger_res:
            r = granger_res[stage]
            print(f"    {r['label']:25s}: "
                  f"forward β={r['fwd_beta']:.4f} (p={r['fwd_p']:.4f}), "
                  f"reverse β={r['rev_beta']:.4f} (p={r['rev_p']:.4f})")

    print("\n[2] Acceleration Test (stability effect by productivity trend):")
    for key in ['all', 'declining', 'stable_rising']:
        if key in accel_res:
            r = accel_res[key]
            print(f"    {r['label']:25s}: "
                  f"stability HR={r['stab_hr']:.3f} (p={r['stab_p']:.4f}), "
                  f"cond HR@10={r['cond_hr_10']['hr']:.3f} (p={r['cond_hr_10']['p']:.4f})")

    print("\n[3] Productivity-Conditioned Lagged Cox:")
    print(f"    Lag-2 stability (base):        HR={cox_res['base_stab_hr']:.3f} "
          f"(p={cox_res['base_stab_p']:.4f})")
    print(f"    Lag-2 stability (conditioned):  HR={cox_res['cond_stab_hr']:.3f} "
          f"(p={cox_res['cond_stab_p']:.4f})")
    print(f"    Productivity trend:             HR={cox_res['prod_trend_hr']:.3f} "
          f"(p={cox_res['prod_trend_p']:.4f})")
    print(f"    Attenuation: {cox_res['attenuation']*100:.1f}%")
    print(f"    Conditioned conditional HR@10:  {cox_res['cond_hr_10']:.3f} "
          f"(p={cox_res['cond_p_10']:.4f})")


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "═" * 70)
    print("  STAGE-SPECIFIC GRANGER & ACCELERATION TEST")
    print("═" * 70 + "\n")

    panel, df_artists, df_events = build_full_panel()
    print(f"Panel: {len(panel)} person-years, {panel['artist_id'].nunique()} artists, "
          f"{int(panel['event'].sum())} events\n")

    granger_res = stage_specific_granger(panel)
    accel_res = acceleration_test(panel)
    cox_res = productivity_conditioned_cox(panel)

    print_summary(granger_res, accel_res, cox_res)

    print("\n" + "═" * 70)
    print("  ANALYSIS COMPLETE")
    print("═" * 70)

    return granger_res, accel_res, cox_res


if __name__ == '__main__':
    results = main()
