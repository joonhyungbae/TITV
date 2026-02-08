"""
13_compute_placeholders.py
Compute missing placeholder values for paper.tex:
1. Period-adjusted plateau definition: conditional stability HR at year 10
2. Alternative weighting scheme sensitivity: HR range and beta_I range
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines import CoxTimeVaryingFitter
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    detect_plateau, CENSOR_YEAR, SIGNIFICANT_EVENT_TYPES,
    ORIGINAL_TYPE_WEIGHTS,
    EQUAL_TYPE_WEIGHTS, BINARY_TYPE_WEIGHTS,
    COMPRESSED_TYPE_WEIGHTS, EXPANDED_TYPE_WEIGHTS,
    compute_event_weight, compute_network_size_stability,
    compute_rank_based_weights
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def compute_decade_mean_interval(df_events, df_artists):
    """Compute decade-specific mean inter-event intervals."""
    decade_intervals = {}
    for decade_start in range(1930, 2010, 10):
        decade_end = decade_start + 9
        decade_label = f"{decade_start}s"
        
        intervals = []
        for _, row in df_artists.iterrows():
            aid = row['artist_id']
            artist_events = df_events[
                (df_events['artist_id'] == aid) &
                (df_events['event_type'].isin(SIGNIFICANT_EVENT_TYPES)) &
                (df_events['year'] >= decade_start) &
                (df_events['year'] <= decade_end)
            ].sort_values('year')
            
            years = artist_events['year'].unique()
            if len(years) > 1:
                years_sorted = np.sort(years)
                for i in range(1, len(years_sorted)):
                    intervals.append(years_sorted[i] - years_sorted[i-1])
        
        if intervals:
            decade_intervals[decade_label] = np.mean(intervals)
        else:
            decade_intervals[decade_label] = np.nan
    
    return decade_intervals


def detect_plateau_period_adjusted(df_events, artist_id, career_start, 
                                     decade_intervals, censor_year=CENSOR_YEAR):
    """
    Period-adjusted plateau: scale the 5-year gap by decade-specific 
    mean inter-event interval.
    """
    artist_events = df_events[
        (df_events['artist_id'] == artist_id) &
        (df_events['year'] >= career_start) &
        (df_events['year'] <= censor_year)
    ]
    
    sig_events = artist_events[artist_events['event_type'].isin(SIGNIFICANT_EVENT_TYPES)]
    sig_years = sorted(sig_events['year'].unique())
    
    if len(sig_years) == 0:
        return True, career_start, 0
    
    end_year = min(int(artist_events['year'].max()), censor_year)
    
    # For each potential gap start, compute adjusted threshold
    for start_y in range(int(career_start), end_year + 1):
        # Determine which decade this year falls in
        decade_start = (start_y // 10) * 10
        decade_label = f"{decade_start}s"
        
        # Get decade-specific mean interval (default to 2.5 if unavailable)
        mean_interval = decade_intervals.get(decade_label, 2.5)
        if np.isnan(mean_interval):
            mean_interval = 2.5
        
        # Adjusted window: scale 5 years by ratio of decade mean to overall mean
        # Use a reference mean of ~2.0 (approximate overall mean)
        overall_mean = np.nanmean(list(decade_intervals.values()))
        if overall_mean > 0:
            adj_window = max(3, int(round(5 * (mean_interval / overall_mean))))
        else:
            adj_window = 5
        
        # Cap between 3 and 8
        adj_window = min(8, max(3, adj_window))
        
        gap_years = set(range(start_y, start_y + adj_window))
        sig_years_set = set(int(y) for y in sig_years)
        
        if not gap_years.intersection(sig_years_set):
            if start_y + adj_window - 1 <= censor_year:
                return True, start_y, start_y - career_start
    
    return False, None, end_year - career_start


def build_panel_period_adjusted(df_artists, df_events, decade_intervals):
    """Build person-year panel with period-adjusted plateau definition."""
    plateau_info = {}
    for _, row in df_artists.iterrows():
        aid = row['artist_id']
        cs = row['career_start_year']
        if pd.isna(cs) or row['num_events'] == 0:
            continue
        occurred, p_year, ttp = detect_plateau_period_adjusted(
            df_events, aid, int(cs), decade_intervals
        )
        plateau_info[aid] = (occurred, p_year, ttp)
    
    records = []
    for _, row in df_artists.iterrows():
        aid = row['artist_id']
        cs = row['career_start_year']
        if pd.isna(cs) or aid not in plateau_info:
            continue
        
        cs_int = int(cs)
        occurred, p_year, _ = plateau_info[aid]
        end_year = int(p_year) if occurred else CENSOR_YEAR
        birth_year = row.get('birth_year')
        
        artist_events = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] <= end_year) &
            (df_events['year'] <= CENSOR_YEAR)
        ].copy()
        
        for year in range(cs_int, end_year + 1):
            career_year = year - cs_int
            events_up_to = artist_events[artist_events['year'] <= year]
            net_size, net_stability = compute_network_size_stability(events_up_to)
            cum_val = sum(
                compute_event_weight(ev['event_type'], ORIGINAL_TYPE_WEIGHTS)
                for _, ev in events_up_to.iterrows()
            )
            is_last_year = (year == end_year)
            event = 1 if (is_last_year and occurred) else 0
            
            records.append({
                'artist_id': aid, 'year': year, 'career_year': career_year,
                'start': career_year, 'stop': career_year + 1,
                'event': event,
                'network_size': max(net_size, 0.5),
                'network_stability': net_stability,
                'cumulative_validation': cum_val,
                'birth_year': birth_year,
            })
    
    df = pd.DataFrame(records)
    if 'birth_year' in df.columns:
        df['birth_year'] = df['birth_year'].fillna(df['birth_year'].median())
    return df


def run_interaction_model(panel, label=""):
    """Run the full-sample interaction model and return conditional HR at year 10."""
    scaler = StandardScaler()
    for col in ['network_stability', 'network_size', 'career_year',
                'birth_year', 'cumulative_validation']:
        panel[f'{col}_z'] = scaler.fit_transform(panel[[col]]).flatten()
    
    panel['stab_x_caryr'] = panel['network_stability_z'] * panel['career_year_z']
    panel['size_x_caryr'] = panel['network_size_z'] * panel['career_year_z']
    panel['id'] = panel['artist_id']
    
    career_year_mean = panel['career_year'].mean()
    career_year_std = panel['career_year'].std()
    
    formula = ('network_stability_z + network_size_z + career_year_z + '
               'stab_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z')
    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(panel, id_col='id', event_col='event',
            start_col='start', stop_col='stop',
            formula=formula, show_progress=False)
    
    s = ctv.summary
    beta_stab = s.loc['network_stability_z', 'coef']
    beta_int = s.loc['stab_x_caryr', 'coef']
    se_stab = s.loc['network_stability_z', 'se(coef)']
    se_int = s.loc['stab_x_caryr', 'se(coef)']
    
    vm = ctv.variance_matrix_
    # Handle both named and positional indexing
    try:
        cov_si = vm.loc['network_stability_z', 'stab_x_caryr']
    except (KeyError, TypeError):
        # Try positional: find indices
        var_names = list(s.index)
        stab_idx = var_names.index('network_stability_z')
        int_idx = var_names.index('stab_x_caryr')
        cov_si = vm.iloc[stab_idx, int_idx]
    
    cy10_z = (10 - career_year_mean) / career_year_std
    cond_coef = beta_stab + beta_int * cy10_z
    cond_se = np.sqrt(se_stab**2 + (cy10_z**2) * se_int**2 + 2 * cy10_z * cov_si)
    hr10 = np.exp(cond_coef)
    p10 = 2 * (1 - norm.cdf(abs(cond_coef / cond_se)))
    
    print(f"\n  [{label}] beta_stab={beta_stab:.4f}, beta_int={beta_int:.4f}")
    print(f"  [{label}] Conditional HR at year 10: {hr10:.3f}, p={p10:.4f}")
    
    return {
        'label': label,
        'beta_stab': beta_stab,
        'beta_int': beta_int,
        'hr_year10': hr10,
        'p_year10': p10,
        'n_events': int(panel['event'].sum()),
        'n_person_years': len(panel),
    }


def build_panel_with_weights(df_artists, df_events, type_weights):
    """Build person-year panel with specified weights for cumulative validation."""
    from data_pipeline import build_person_year_panel
    return build_person_year_panel(df_artists, df_events, cutpoint=10,
                                   type_weights=type_weights)


def main():
    print("=" * 70)
    print("COMPUTING PLACEHOLDER VALUES FOR PAPER.TEX")
    print("=" * 70)
    
    # Load data
    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)
    
    # ========================================
    # PART 1: Period-Adjusted Plateau Definition
    # ========================================
    print("\n" + "=" * 70)
    print("PART 1: PERIOD-ADJUSTED PLATEAU DEFINITION")
    print("=" * 70)
    
    decade_intervals = compute_decade_mean_interval(df_events, df_artists)
    print("\nDecade-specific mean inter-event intervals:")
    for decade, interval in sorted(decade_intervals.items()):
        print(f"  {decade}: {interval:.2f}" if not np.isnan(interval) else f"  {decade}: N/A")
    
    panel_adj = build_panel_period_adjusted(df_artists, df_events, decade_intervals)
    n_events_adj = panel_adj['event'].sum()
    n_artists_adj = panel_adj['artist_id'].nunique()
    plateau_rate_adj = panel_adj.groupby('artist_id')['event'].max().mean()
    
    print(f"\nPeriod-adjusted panel: {len(panel_adj)} person-years, "
          f"{n_artists_adj} artists, {int(n_events_adj)} events")
    print(f"Plateau rate: {plateau_rate_adj:.1%}")
    
    result_adj = run_interaction_model(panel_adj.copy(), "Period-adjusted")
    
    print(f"\n>>> PLACEHOLDER 1 (Line 220):")
    print(f"    HR at year 10 = {result_adj['hr_year10']:.2f}")
    print(f"    p-value = {result_adj['p_year10']:.3f}")
    
    # ========================================
    # PART 2: REGIME-STRATIFIED MODELS
    # ========================================
    print("\n\n" + "=" * 70)
    print("PART 2: REGIME-STRATIFIED MODELS (4 institutional regimes)")
    print("=" * 70)

    # Define regimes by career start year (proxy for which regime an artist operated in)
    # Artists are assigned to the regime in which they began their career.
    regimes = [
        ("Colonial Modernity (≤1945)", lambda cs: cs <= 1945),
        ("State Formation (1946-1960)", lambda cs: 1946 <= cs <= 1960),
        ("Developmental State (1961-1987)", lambda cs: 1961 <= cs <= 1987),
        ("Democratic Pluralism (1988+)", lambda cs: cs >= 1988),
    ]

    from data_pipeline import build_person_year_panel
    panel_full = build_person_year_panel(df_artists, df_events)
    panel_full = panel_full.merge(
        df_artists[['artist_id', 'career_start_year']],
        on='artist_id', how='left'
    )

    regime_results = []
    for regime_label, regime_filter in regimes:
        regime_aids = df_artists[
            df_artists['career_start_year'].apply(
                lambda cs: regime_filter(cs) if pd.notna(cs) else False
            )
        ]['artist_id'].values
        sub_panel = panel_full[panel_full['artist_id'].isin(regime_aids)].copy()
        n_artists = sub_panel['artist_id'].nunique()
        n_events = int(sub_panel['event'].sum())
        n_py = len(sub_panel)

        if n_events < 5:
            print(f"\n  {regime_label}: {n_artists} artists, {n_events} events — SKIPPED (too few events)")
            regime_results.append({
                'regime': regime_label, 'n_artists': n_artists,
                'n_events': n_events, 'n_py': n_py,
                'stab_hr': None, 'stab_p': None,
                'stab_x_caryr_hr': None, 'stab_x_caryr_p': None,
            })
            continue

        # Standardize within regime
        scaler = StandardScaler()
        for col in ['network_stability', 'network_size', 'career_year',
                    'birth_year', 'cumulative_validation']:
            sub_panel[f'{col}_z'] = scaler.fit_transform(sub_panel[[col]]).flatten()
        sub_panel['stab_x_caryr'] = sub_panel['network_stability_z'] * sub_panel['career_year_z']
        sub_panel['size_x_caryr'] = sub_panel['network_size_z'] * sub_panel['career_year_z']
        sub_panel['id'] = sub_panel['artist_id']

        try:
            formula = ('network_stability_z + network_size_z + career_year_z + '
                       'stab_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z')
            ctv = CoxTimeVaryingFitter(penalizer=0.05)
            ctv.fit(sub_panel, id_col='id', event_col='event',
                    start_col='start', stop_col='stop',
                    formula=formula, show_progress=False)
            s = ctv.summary
            stab_hr = np.exp(s.loc['network_stability_z', 'coef'])
            stab_p = s.loc['network_stability_z', 'p']
            int_hr = np.exp(s.loc['stab_x_caryr', 'coef'])
            int_p = s.loc['stab_x_caryr', 'p']
            sig_stab = '***' if stab_p < 0.001 else '**' if stab_p < 0.01 else '*' if stab_p < 0.05 else ''
            sig_int = '***' if int_p < 0.001 else '**' if int_p < 0.01 else '*' if int_p < 0.05 else ''
            print(f"\n  {regime_label}: {n_artists} artists, {n_events} events, {n_py} person-years")
            print(f"    Stability main:    HR={stab_hr:.3f}, p={stab_p:.4f}{sig_stab}")
            print(f"    Stab×career_year:  HR={int_hr:.3f}, p={int_p:.4f}{sig_int}")
            regime_results.append({
                'regime': regime_label, 'n_artists': n_artists,
                'n_events': n_events, 'n_py': n_py,
                'stab_hr': stab_hr, 'stab_p': stab_p,
                'stab_x_caryr_hr': int_hr, 'stab_x_caryr_p': int_p,
            })
        except Exception as e:
            print(f"\n  {regime_label}: {n_artists} artists, {n_events} events — ERROR: {e}")
            regime_results.append({
                'regime': regime_label, 'n_artists': n_artists,
                'n_events': n_events, 'n_py': n_py,
                'stab_hr': None, 'stab_p': None,
                'stab_x_caryr_hr': None, 'stab_x_caryr_p': None,
            })

    print("\n  ─── REGIME SUMMARY ───")
    print(f"  {'Regime':40s} {'N':>5s} {'Events':>6s} {'Stab HR':>8s} {'p':>8s} {'Int HR':>8s} {'p':>8s}")
    print(f"  {'-'*40} {'-'*5} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    all_positive = True
    for r in regime_results:
        if r['stab_hr'] is not None:
            direction = "+" if r['stab_hr'] > 1 else "-"
            print(f"  {r['regime']:40s} {r['n_artists']:5d} {r['n_events']:6d} "
                  f"{r['stab_hr']:8.3f} {r['stab_p']:8.4f} "
                  f"{r['stab_x_caryr_hr']:8.3f} {r['stab_x_caryr_p']:8.4f}")
            if r['stab_hr'] <= 1:
                all_positive = False
        else:
            print(f"  {r['regime']:40s} {r['n_artists']:5d} {r['n_events']:6d} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s}")
    print(f"\n  All regimes show stability HR > 1: {all_positive}")

    # --- Regime-interaction model (full sample) ---
    print("\n  ─── REGIME-INTERACTION MODEL (full sample, regime dummies × stability) ───")
    panel_regime = panel_full.copy()
    # Assign regime based on career start year
    def assign_regime(cs):
        if pd.isna(cs):
            return 'unknown'
        if cs <= 1945:
            return 'colonial'
        elif cs <= 1960:
            return 'state_formation'
        elif cs <= 1987:
            return 'developmental'
        else:
            return 'democratic'
    panel_regime['regime'] = panel_regime['career_start_year'].apply(assign_regime)
    panel_regime = panel_regime[panel_regime['regime'] != 'unknown'].copy()

    # Use developmental state as reference category (largest group)
    for reg in ['colonial', 'state_formation', 'democratic']:
        panel_regime[f'regime_{reg}'] = (panel_regime['regime'] == reg).astype(float)

    scaler = StandardScaler()
    for col in ['network_stability', 'network_size', 'career_year',
                'birth_year', 'cumulative_validation']:
        panel_regime[f'{col}_z'] = scaler.fit_transform(panel_regime[[col]]).flatten()
    panel_regime['stab_x_caryr'] = panel_regime['network_stability_z'] * panel_regime['career_year_z']
    panel_regime['size_x_caryr'] = panel_regime['network_size_z'] * panel_regime['career_year_z']

    # Stability × regime interactions
    for reg in ['colonial', 'state_formation', 'democratic']:
        panel_regime[f'stab_x_{reg}'] = panel_regime['network_stability_z'] * panel_regime[f'regime_{reg}']

    panel_regime['id'] = panel_regime['artist_id']

    formula_regime = ('network_stability_z + network_size_z + career_year_z + '
                      'stab_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z + '
                      'regime_colonial + regime_state_formation + regime_democratic + '
                      'stab_x_colonial + stab_x_state_formation + stab_x_democratic')
    ctv_regime = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_regime.fit(panel_regime, id_col='id', event_col='event',
                   start_col='start', stop_col='stop',
                   formula=formula_regime, show_progress=False)
    s_r = ctv_regime.summary
    print(f"\n  {'Variable':35s} {'coef':>8s} {'HR':>8s} {'p':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for var in s_r.index:
        coef = s_r.loc[var, 'coef']
        hr = s_r.loc[var, 'exp(coef)']
        p = s_r.loc[var, 'p']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {var:35s} {coef:8.4f} {hr:8.3f} {p:8.4f}{sig}")

    # Joint Wald test for regime × stability interactions
    int_vars = ['stab_x_colonial', 'stab_x_state_formation', 'stab_x_democratic']
    int_coefs = np.array([s_r.loc[v, 'coef'] for v in int_vars])
    vm = ctv_regime.variance_matrix_
    var_names = list(s_r.index)
    int_indices = [var_names.index(v) for v in int_vars]
    cov_sub = vm.values[np.ix_(int_indices, int_indices)]
    from scipy.stats import chi2
    wald_stat = int_coefs @ np.linalg.inv(cov_sub) @ int_coefs
    wald_p = 1 - chi2.cdf(wald_stat, df=len(int_vars))
    print(f"\n  Wald test for regime × stability interactions:")
    print(f"    chi2({len(int_vars)}) = {wald_stat:.3f}, p = {wald_p:.4f}")
    if wald_p >= 0.05:
        print(f"    → The stability effect does NOT significantly vary across institutional regimes.")
    else:
        print(f"    → The stability effect DOES significantly vary across institutional regimes.")

    # ========================================
    # PART 3: Alternative Weighting Schemes
    # ========================================
    print("\n\n" + "=" * 70)
    print("PART 3: ALTERNATIVE WEIGHTING SCHEME SENSITIVITY")
    print("=" * 70)
    
    # Compute rank-based weights
    rank_weights = compute_rank_based_weights(df_events)
    print(f"\nRank-based weights: {rank_weights}")
    print(f"Compressed weights (sqrt): {COMPRESSED_TYPE_WEIGHTS}")
    print(f"Expanded weights (square): {EXPANDED_TYPE_WEIGHTS}")
    
    weight_schemes = [
        ("Equal weights", EQUAL_TYPE_WEIGHTS),
        ("Binary weights", BINARY_TYPE_WEIGHTS),
        ("Rank-order weights", rank_weights),
        ("Compressed hierarchy (sqrt)", COMPRESSED_TYPE_WEIGHTS),
        ("Expanded hierarchy (square)", EXPANDED_TYPE_WEIGHTS),
    ]
    
    weight_results = []
    for scheme_name, tw in weight_schemes:
        print(f"\n--- {scheme_name} ---")
        panel_w = build_panel_with_weights(df_artists, df_events, tw)
        result_w = run_interaction_model(panel_w.copy(), scheme_name)
        weight_results.append(result_w)
    
    # Compute ranges
    hrs = [r['hr_year10'] for r in weight_results]
    betas = [r['beta_int'] for r in weight_results]
    ps = [r['p_year10'] for r in weight_results]
    
    print(f"\n\n>>> PLACEHOLDER 3:")
    print(f"    HR range at year 10: {min(hrs):.2f} to {max(hrs):.2f}")
    print(f"    All p-values: {[f'{p:.3f}' for p in ps]}")
    print(f"    All p < 0.05: {all(p < 0.05 for p in ps)}")
    print(f"    beta_I range: {min(betas):.3f} to {max(betas):.3f}")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n\n" + "=" * 70)
    print("SUMMARY OF ALL ROBUSTNESS RESULTS")
    print("=" * 70)
    print(f"\n1. Period-adjusted plateau definition:")
    print(f"   conditional stability HR in year 10 = {result_adj['hr_year10']:.2f}")
    print(f"   p-value = {result_adj['p_year10']:.3f}")
    print(f"\n2. Regime-stratified models:")
    for r in regime_results:
        if r['stab_hr'] is not None:
            print(f"   {r['regime']}: stab HR={r['stab_hr']:.3f} (p={r['stab_p']:.4f}), "
                  f"interaction HR={r['stab_x_caryr_hr']:.3f} (p={r['stab_x_caryr_p']:.4f})")
        else:
            print(f"   {r['regime']}: N/A (insufficient events)")
    print(f"\n3. Alternative weighting schemes:")
    print(f"   HR at year 10 ranges from {min(hrs):.2f} to {max(hrs):.2f}")
    print(f"   beta_I ranges from {min(betas):.3f} to {max(betas):.3f}")
    
    print("\nDONE.")


if __name__ == '__main__':
    main()
