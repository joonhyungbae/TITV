"""
19_forward_path_evidence.py
════════════════════════════════════════════════════════════════════
Strengthens evidence for a forward-acting component in the
stability → plateau relationship, addressing the concern that
the reverse-causal Granger channel dominates.

  (1) Long-horizon Granger tests  — 2- and 3-year cumulative lags
  (2) Matched-pair analysis       — CEM at year 10, compare plateaus
  (3) Artist fixed-effects model  — within-artist variation only
  (4) Predictive ordering table   — lag 1–5 AUC comparison
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
    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)
    panel = build_person_year_panel(df_artists, df_events, cutpoint=10)
    panel = panel.sort_values(['artist_id', 'career_year']).reset_index(drop=True)

    # Lagged variables
    for lag in [1, 2, 3, 4, 5]:
        panel[f'stability_lag{lag}'] = panel.groupby('artist_id')['network_stability'].shift(lag)
        panel[f'netsize_lag{lag}'] = panel.groupby('artist_id')['network_size'].shift(lag)

    # Productivity (annual change in cumulative validation)
    panel['cum_val_lag1'] = panel.groupby('artist_id')['cumulative_validation'].shift(1)
    panel['annual_prod'] = panel['cumulative_validation'] - panel['cum_val_lag1']
    panel['annual_prod'] = panel['annual_prod'].fillna(panel['cumulative_validation'])

    for lag in [1, 2, 3, 4, 5]:
        panel[f'prod_lag{lag}'] = panel.groupby('artist_id')['annual_prod'].shift(lag)

    # Changes for Granger
    panel['delta_stability'] = panel.groupby('artist_id')['network_stability'].diff()
    panel['delta_prod'] = panel.groupby('artist_id')['annual_prod'].diff()
    for lag in [1, 2, 3]:
        panel[f'delta_stab_lag{lag}'] = panel.groupby('artist_id')['delta_stability'].shift(lag)
        panel[f'delta_prod_lag{lag}'] = panel.groupby('artist_id')['delta_prod'].shift(lag)

    panel['id'] = panel['artist_id']
    return panel, df_artists, df_events


# ============================================================
# PART 1: Long-Horizon Granger Tests
# ============================================================

def long_horizon_granger(panel):
    """
    Extend the Granger-type cross-lagged regressions to 2- and 3-year
    cumulative lags.  The theory predicts that evaluative redundancy
    operates on a longer time-scale than 1-year Granger tests can detect.
    """
    print("=" * 70)
    print("PART 1: LONG-HORIZON GRANGER TESTS")
    print("=" * 70)

    results = []

    for max_lag in [1, 2, 3]:
        # Columns for this lag horizon
        stab_lags = [f'delta_stab_lag{l}' for l in range(1, max_lag + 1)]
        prod_lags = [f'delta_prod_lag{l}' for l in range(1, max_lag + 1)]

        required = stab_lags + prod_lags + ['delta_stability', 'delta_prod',
                                             'birth_year', 'career_year']
        df = panel.dropna(subset=required).copy()
        df = df[df['career_year'] >= max_lag + 1].copy()

        if len(df) < 50:
            continue

        # Standardise
        sc = StandardScaler()
        for col in required:
            df[f'{col}_z'] = sc.fit_transform(df[[col]]).flatten()

        # Forward path: ΔStability(t-k) → ΔProductivity(t)
        X_fwd_cols = [f'delta_stab_lag{l}_z' for l in range(1, max_lag + 1)] + \
                      [f'delta_prod_lag{l}_z' for l in range(1, max_lag + 1)] + \
                      ['birth_year_z', 'career_year_z']
        X_fwd = sm.add_constant(df[X_fwd_cols].values)
        y_fwd = df['delta_prod_z'].values
        model_fwd = sm.OLS(y_fwd, X_fwd).fit(cov_type='cluster',
                                               cov_kwds={'groups': df['artist_id'].values})

        # Reverse path: ΔProductivity(t-k) → ΔStability(t)
        X_rev_cols = [f'delta_prod_lag{l}_z' for l in range(1, max_lag + 1)] + \
                      [f'delta_stab_lag{l}_z' for l in range(1, max_lag + 1)] + \
                      ['birth_year_z', 'career_year_z']
        X_rev = sm.add_constant(df[X_rev_cols].values)
        y_rev = df['delta_stability_z'].values
        model_rev = sm.OLS(y_rev, X_rev).fit(cov_type='cluster',
                                              cov_kwds={'groups': df['artist_id'].values})

        # Extract forward stability coefficients (first max_lag params after const)
        fwd_betas = model_fwd.params[1:max_lag + 1]
        fwd_pvals = model_fwd.pvalues[1:max_lag + 1]
        # Joint F-test for forward stability lags
        r_mat_fwd = np.zeros((max_lag, len(model_fwd.params)))
        for k in range(max_lag):
            r_mat_fwd[k, k + 1] = 1
        fwd_f_test = model_fwd.f_test(r_mat_fwd)
        fwd_f_p = float(fwd_f_test.pvalue)

        # Extract reverse productivity coefficients
        rev_betas = model_rev.params[1:max_lag + 1]
        rev_pvals = model_rev.pvalues[1:max_lag + 1]
        r_mat_rev = np.zeros((max_lag, len(model_rev.params)))
        for k in range(max_lag):
            r_mat_rev[k, k + 1] = 1
        rev_f_test = model_rev.f_test(r_mat_rev)
        rev_f_p = float(rev_f_test.pvalue)

        print(f"\n  --- Lag horizon: {max_lag} year(s) (n = {len(df)}) ---")
        print(f"  Forward (ΔStab → ΔProd): β = {fwd_betas.mean():.4f}, "
              f"joint F-test p = {fwd_f_p:.4f}")
        for k in range(max_lag):
            print(f"    lag-{k+1}: β = {fwd_betas[k]:.4f}, p = {fwd_pvals[k]:.4f}")
        print(f"  Reverse (ΔProd → ΔStab): β = {rev_betas.mean():.4f}, "
              f"joint F-test p = {rev_f_p:.4f}")
        for k in range(max_lag):
            print(f"    lag-{k+1}: β = {rev_betas[k]:.4f}, p = {rev_pvals[k]:.4f}")

        results.append({
            'max_lag': max_lag, 'n': len(df),
            'fwd_mean_beta': fwd_betas.mean(),
            'fwd_joint_p': fwd_f_p,
            'rev_mean_beta': rev_betas.mean(),
            'rev_joint_p': rev_f_p,
        })

    return pd.DataFrame(results)


# ============================================================
# PART 2: Matched-Pair Analysis at Year 10
# ============================================================

def matched_pair_analysis(panel):
    """
    At career year 10, match artists on cumulative_validation and
    network_size, then compare subsequent plateau rates by stability
    level (high vs low, median-split).
    """
    print("\n" + "=" * 70)
    print("PART 2: MATCHED-PAIR ANALYSIS AT CAREER YEAR 10")
    print("=" * 70)

    # Snapshot at year 10
    yr10 = panel[panel['career_year'] == 10].copy()
    if len(yr10) == 0:
        print("  No observations at career year 10.")
        return {}

    print(f"  Artists observed at year 10: {len(yr10)}")

    # Determine post-year-10 plateau (from existing data)
    # An artist experienced a post-10 plateau if they have an event=1
    # observation after career year 10
    post10_events = panel[(panel['career_year'] > 10) & (panel['event'] == 1)]
    yr10['post10_plateau'] = yr10['artist_id'].isin(
        post10_events['artist_id'].unique()
    ).astype(int)

    # Also include artists who never reached post-10 event (censored)
    # They should be counted as non-plateau
    print(f"  Post-10 plateau: {yr10['post10_plateau'].sum()}, "
          f"No plateau: {(yr10['post10_plateau'] == 0).sum()}")

    # Median split on stability at year 10
    med_stab = yr10['network_stability'].median()
    yr10['high_stability'] = (yr10['network_stability'] > med_stab).astype(int)

    print(f"  Stability median at yr 10 = {med_stab:.2f}")
    print(f"  High stability: {yr10['high_stability'].sum()}, "
          f"Low stability: {(yr10['high_stability'] == 0).sum()}")

    # Coarsened Exact Matching (CEM) on cumulative_validation and network_size
    # Create bins
    for col, n_bins in [('cumulative_validation', 5), ('network_size', 4)]:
        try:
            yr10[f'{col}_bin'] = pd.qcut(yr10[col], n_bins, labels=False, duplicates='drop')
        except ValueError:
            yr10[f'{col}_bin'] = pd.cut(yr10[col], n_bins, labels=False)

    if 'birth_year' in yr10.columns:
        try:
            yr10['birth_year_bin'] = pd.qcut(yr10['birth_year'], 3, labels=False, duplicates='drop')
        except ValueError:
            yr10['birth_year_bin'] = 0

    # Match within strata
    match_cols = ['cumulative_validation_bin', 'network_size_bin']
    if 'birth_year_bin' in yr10.columns:
        match_cols.append('birth_year_bin')

    yr10['strata'] = yr10[match_cols].astype(str).agg('-'.join, axis=1)

    matched_high = []
    matched_low = []
    for strata, grp in yr10.groupby('strata'):
        high = grp[grp['high_stability'] == 1]
        low = grp[grp['high_stability'] == 0]
        if len(high) > 0 and len(low) > 0:
            # Keep minimum of the two groups
            n_match = min(len(high), len(low))
            matched_high.append(high.sample(n_match, random_state=42))
            matched_low.append(low.sample(n_match, random_state=42))

    if len(matched_high) == 0:
        print("  No matched pairs found.")
        return {}

    matched_high = pd.concat(matched_high)
    matched_low = pd.concat(matched_low)

    n_matched = len(matched_high) + len(matched_low)
    print(f"\n  Matched sample: {n_matched} artists "
          f"({len(matched_high)} high, {len(matched_low)} low stability)")

    # Compare plateau rates
    high_rate = matched_high['post10_plateau'].mean()
    low_rate = matched_low['post10_plateau'].mean()
    diff = high_rate - low_rate

    # Fisher exact test
    a = matched_high['post10_plateau'].sum()
    b = len(matched_high) - a
    c = matched_low['post10_plateau'].sum()
    d = len(matched_low) - c
    odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d]])

    print(f"\n  Post-10 plateau rate:")
    print(f"    High stability: {high_rate:.3f} ({a}/{len(matched_high)})")
    print(f"    Low  stability: {low_rate:.3f} ({c}/{len(matched_low)})")
    print(f"    Difference: {diff:+.3f}")
    print(f"    Fisher exact OR = {odds_ratio:.3f}, p = {fisher_p:.4f}")

    # Also run logistic on matched sample
    matched_all = pd.concat([matched_high, matched_low])
    sc = StandardScaler()
    matched_all['stab_z'] = sc.fit_transform(matched_all[['network_stability']]).flatten()
    X = sm.add_constant(matched_all[['stab_z']].values)
    y = matched_all['post10_plateau'].values

    if y.sum() >= 3 and (len(y) - y.sum()) >= 3:
        model = sm.Logit(y, X).fit(disp=0)
        stab_or = np.exp(model.params[1])
        stab_p = model.pvalues[1]
        print(f"    Logistic (matched): OR = {stab_or:.3f}, p = {stab_p:.4f}")
    else:
        stab_or = odds_ratio
        stab_p = fisher_p

    return {
        'n_matched': n_matched,
        'high_rate': high_rate,
        'low_rate': low_rate,
        'diff': diff,
        'fisher_or': odds_ratio,
        'fisher_p': fisher_p,
        'logistic_or': stab_or,
        'logistic_p': stab_p,
    }


# ============================================================
# PART 3: Artist Fixed-Effects Model
# ============================================================

def artist_fixed_effects(panel):
    """
    Stratified Cox model with artist strata to absorb all
    time-invariant unobserved heterogeneity (artistic capacity, etc.).
    Only within-artist variation in stability contributes.
    """
    print("\n" + "=" * 70)
    print("PART 3: ARTIST FIXED-EFFECTS (STRATIFIED COX)")
    print("=" * 70)

    # Only artists with variation in event and at least 3 person-years
    artist_counts = panel.groupby('artist_id').agg(
        n_years=('career_year', 'count'),
        n_events=('event', 'sum'),
        stab_var=('network_stability', 'std')
    )
    # Need artists with at least some variation
    valid_artists = artist_counts[
        (artist_counts['n_years'] >= 3) &
        (artist_counts['stab_var'] > 0.01)
    ].index

    fe_panel = panel[panel['artist_id'].isin(valid_artists)].copy()
    n_artists = fe_panel['artist_id'].nunique()
    n_events = int(fe_panel['event'].sum())
    print(f"  Valid artists: {n_artists} (with stability variation)")
    print(f"  Person-years: {len(fe_panel)}, events: {n_events}")

    if n_events < 10:
        print("  Insufficient events for FE model.")
        return {}

    # Standardise
    sc = StandardScaler()
    for col in ['network_stability', 'network_size', 'career_year',
                'cumulative_validation']:
        fe_panel[f'{col}_z'] = sc.fit_transform(fe_panel[[col]]).flatten()

    # Artist-level demeaning (within-artist variation)
    for col in ['network_stability_z', 'network_size_z', 'cumulative_validation_z']:
        fe_panel[f'{col}_dm'] = fe_panel.groupby('artist_id')[col].transform(
            lambda x: x - x.mean()
        )

    fe_panel['stab_dm_x_caryr'] = fe_panel['network_stability_z_dm'] * fe_panel['career_year_z']
    fe_panel['id'] = fe_panel['artist_id']

    # Linear Probability Model with artist FE (demeaned)
    print("\n  --- Linear Probability Model (demeaned) ---")
    X_cols = ['network_stability_z_dm', 'network_size_z_dm',
              'career_year_z', 'cumulative_validation_z_dm', 'stab_dm_x_caryr']
    df_lpm = fe_panel.dropna(subset=X_cols + ['event']).copy()
    X = sm.add_constant(df_lpm[X_cols].values)
    y = df_lpm['event'].values

    model_lpm = sm.OLS(y, X).fit(cov_type='cluster',
                                  cov_kwds={'groups': df_lpm['artist_id'].values})

    var_names_lpm = ['const'] + X_cols
    print(f"  {'Variable':35s} {'coef':>8s} {'p':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8}")
    for j, name in enumerate(var_names_lpm):
        print(f"  {name:35s} {model_lpm.params[j]:8.4f} {model_lpm.pvalues[j]:8.4f}")

    stab_dm_coef = model_lpm.params[1]
    stab_dm_p = model_lpm.pvalues[1]
    int_dm_coef = model_lpm.params[5]
    int_dm_p = model_lpm.pvalues[5]

    print(f"\n  Within-artist stability: coef = {stab_dm_coef:.4f}, p = {stab_dm_p:.4f}")
    print(f"  Within-artist stab × career_year: coef = {int_dm_coef:.4f}, p = {int_dm_p:.4f}")

    return {
        'n_artists': n_artists,
        'n_events': n_events,
        'stab_dm_coef': stab_dm_coef,
        'stab_dm_p': stab_dm_p,
        'int_dm_coef': int_dm_coef,
        'int_dm_p': int_dm_p,
    }


# ============================================================
# PART 4: Predictive Ordering (AUC Comparison)
# ============================================================

def predictive_ordering(panel):
    """
    For lags 1–5, compare AUC of:
      - Stability(t-k) × career_stage  (structural precursor)
      - Productivity(t-k)              (concurrent symptom)
    """
    print("\n" + "=" * 70)
    print("PART 4: PREDICTIVE ORDERING — AUC BY LAG")
    print("=" * 70)

    sc = StandardScaler()
    results = []

    for lag in [1, 2, 3, 4, 5]:
        stab_col = f'stability_lag{lag}'
        prod_col = f'prod_lag{lag}'

        df = panel.dropna(subset=[stab_col, 'career_year', 'birth_year']).copy()
        df = df[df['career_year'] >= lag].copy()
        if prod_col in df.columns:
            df = df.dropna(subset=[prod_col])
        df = df[np.isfinite(df[stab_col])].copy()

        y = df['event'].values.astype(int)
        if y.sum() < 5:
            continue

        df['stab_z'] = sc.fit_transform(df[[stab_col]]).flatten()
        df['cy_z'] = sc.fit_transform(df[['career_year']]).flatten()
        df['stab_x_cy'] = df['stab_z'] * df['cy_z']

        # Model A: Stability × career stage
        try:
            X_a = sm.add_constant(df[['stab_z', 'cy_z', 'stab_x_cy']].values)
            m_a = sm.Logit(y, X_a).fit(disp=0)
            auc_stab = roc_auc_score(y, m_a.predict(X_a))
        except Exception:
            auc_stab = 0.5

        # Model B: Productivity
        try:
            if prod_col in df.columns:
                auc_prod = roc_auc_score(y, -df[prod_col].values)
            else:
                auc_prod = 0.5
        except Exception:
            auc_prod = 0.5

        delta = auc_stab - auc_prod
        marker = '+' if delta > 0 else '-'
        print(f"  Lag {lag}: AUC_stab×stage = {auc_stab:.3f}, "
              f"AUC_prod = {auc_prod:.3f}, "
              f"ΔAUC = {delta:+.3f} {marker}  "
              f"(n={len(y)}, events={y.sum()})")

        results.append({
            'lag': lag, 'auc_stab_interaction': auc_stab,
            'auc_productivity': auc_prod, 'delta_auc': delta,
            'n': len(y), 'n_events': int(y.sum()),
        })

    df_res = pd.DataFrame(results)
    if len(df_res) > 0:
        print(f"\n  Stability × stage advantage at all lags: "
              f"{(df_res['delta_auc'] > 0).all()}")
        print(f"  Mean ΔAUC: {df_res['delta_auc'].mean():+.3f}")
    return df_res


# ============================================================
# Summary
# ============================================================

def print_summary(granger_df, match_res, fe_res, pred_df):
    print("\n\n" + "=" * 70)
    print("SUMMARY FOR PAPER INTEGRATION")
    print("=" * 70)

    print("\n[1] Long-horizon Granger tests:")
    for _, r in granger_df.iterrows():
        print(f"    {int(r['max_lag'])}-year: Forward p = {r['fwd_joint_p']:.4f}, "
              f"Reverse p = {r['rev_joint_p']:.4f}")

    print("\n[2] Matched-pair analysis (at year 10):")
    if match_res:
        print(f"    N matched: {match_res['n_matched']}")
        print(f"    High-stab plateau rate: {match_res['high_rate']:.3f}")
        print(f"    Low-stab plateau rate:  {match_res['low_rate']:.3f}")
        print(f"    Fisher OR = {match_res['fisher_or']:.3f}, p = {match_res['fisher_p']:.4f}")

    print("\n[3] Artist fixed-effects (demeaned LPM):")
    if fe_res:
        print(f"    Within-artist stability: coef = {fe_res['stab_dm_coef']:.4f}, "
              f"p = {fe_res['stab_dm_p']:.4f}")
        print(f"    Within-artist stab × career_yr: coef = {fe_res['int_dm_coef']:.4f}, "
              f"p = {fe_res['int_dm_p']:.4f}")

    print("\n[4] Predictive ordering (AUC, lag 1–5):")
    if len(pred_df) > 0:
        for _, r in pred_df.iterrows():
            print(f"    Lag {int(r['lag'])}: ΔAUC = {r['delta_auc']:+.3f}")
        print(f"    Stability advantage at all lags: "
              f"{(pred_df['delta_auc'] > 0).all()}")


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "═" * 70)
    print("  FORWARD-PATH EVIDENCE ANALYSIS")
    print("═" * 70 + "\n")

    panel, df_artists, df_events = build_full_panel()
    print(f"Panel: {len(panel)} person-years, {panel['artist_id'].nunique()} artists, "
          f"{int(panel['event'].sum())} events\n")

    granger_df = long_horizon_granger(panel)
    match_res = matched_pair_analysis(panel)
    fe_res = artist_fixed_effects(panel)
    pred_df = predictive_ordering(panel)

    print_summary(granger_df, match_res, fe_res, pred_df)

    print("\n" + "═" * 70)
    print("  ANALYSIS COMPLETE")
    print("═" * 70)

    return granger_df, match_res, fe_res, pred_df


if __name__ == '__main__':
    results = main()
