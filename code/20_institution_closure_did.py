"""
20_institution_closure_did.py
════════════════════════════════════════════════════════════════════
Quasi-experimental analysis using institution closures as exogenous
shocks to network structure.

Logic:
  - When an institution "closes" (disappears from the data), artists
    who relied on it are forced to diversify — a network change NOT
    caused by the artist's own productivity trajectory.
  - If evaluative redundancy (forward path) is real, forced
    diversification should REDUCE subsequent plateau risk.
  - If only reverse causality operates, forced diversification should
    have NO effect on plateau risk.

Steps:
  (1) Track institution lifecycles from event data
  (2) Define closures conservatively (last appearance ≤ 1997, etc.)
  (3) Identify treated vs. control artists
  (4) DiD regression with matched controls
════════════════════════════════════════════════════════════════════
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    build_person_year_panel, CENSOR_YEAR
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
np.random.seed(42)


# ============================================================
# PART 1: Institution Lifecycle Tracking
# ============================================================

def track_institution_lifecycles(df_events):
    """
    For each institution, compute first/last appearance year,
    total events, number of distinct artists, and annual event counts.
    """
    print("=" * 70)
    print("PART 1: INSTITUTION LIFECYCLE TRACKING")
    print("=" * 70)

    inst_stats = df_events.groupby('institution_en').agg(
        first_year=('year', 'min'),
        last_year=('year', 'max'),
        total_events=('year', 'count'),
        n_artists=('artist_id', 'nunique'),
    ).reset_index()

    inst_stats['lifespan'] = inst_stats['last_year'] - inst_stats['first_year']

    print(f"  Total unique institutions: {len(inst_stats)}")
    print(f"  Institutions with ≥ 3 events: "
          f"{(inst_stats['total_events'] >= 3).sum()}")
    print(f"  Mean lifespan: {inst_stats['lifespan'].mean():.1f} years")
    print(f"  Median lifespan: {inst_stats['lifespan'].median():.1f} years")

    return inst_stats


# ============================================================
# PART 2: Identify Closures
# ============================================================

def identify_closures(inst_stats, df_events, max_last_year=1997,
                      min_gap=3, min_pre_events=3, pre_window=5):
    """
    Identify institution closures:
      - Last appearance ≤ max_last_year (distinguish from right-censoring)
      - Gap of ≥ min_gap years after last appearance
      - At least min_pre_events events in the pre_window years before closure
    """
    print(f"\n{'=' * 70}")
    print(f"PART 2: CLOSURE IDENTIFICATION (last ≤ {max_last_year})")
    print("=" * 70)

    candidates = inst_stats[inst_stats['last_year'] <= max_last_year].copy()
    print(f"  Institutions with last appearance ≤ {max_last_year}: {len(candidates)}")

    # Check gap after last appearance
    candidates['gap_after'] = CENSOR_YEAR - candidates['last_year']
    candidates = candidates[candidates['gap_after'] >= min_gap]
    print(f"  After gap ≥ {min_gap} filter: {len(candidates)}")

    # Check pre-closure activity
    closures = []
    for _, row in candidates.iterrows():
        inst = row['institution_en']
        last_yr = row['last_year']
        pre_start = last_yr - pre_window
        pre_events = df_events[
            (df_events['institution_en'] == inst) &
            (df_events['year'] >= pre_start) &
            (df_events['year'] <= last_yr)
        ]
        if len(pre_events) >= min_pre_events:
            closures.append({
                'institution_en': inst,
                'closure_year': last_yr,
                'pre_events': len(pre_events),
                'pre_artists': pre_events['artist_id'].nunique(),
                'total_events': row['total_events'],
                'lifespan': row['lifespan'],
            })

    df_closures = pd.DataFrame(closures)
    print(f"  After pre-activity filter (≥ {min_pre_events} events "
          f"in {pre_window} yr): {len(df_closures)}")

    if len(df_closures) > 0:
        print(f"\n  Closed institutions:")
        for _, c in df_closures.iterrows():
            print(f"    {c['institution_en'][:50]:50s} "
                  f"closed={c['closure_year']}, "
                  f"pre_events={c['pre_events']}, "
                  f"pre_artists={c['pre_artists']}")

    return df_closures


# ============================================================
# PART 3: Treated vs. Control Artists
# ============================================================

def identify_treated_control(df_events, df_closures, df_artists,
                              treatment_threshold=0.20,
                              control_threshold=0.05,
                              pre_window=5):
    """
    Treated: ≥ treatment_threshold of events at a closing institution
             in the pre_window years before closure.
    Control: < control_threshold exposure to any closing institution.
    Vectorized implementation for speed.
    """
    print(f"\n{'=' * 70}")
    print("PART 3: TREATED vs. CONTROL ARTISTS")
    print("=" * 70)

    if len(df_closures) == 0:
        print("  No closures found.")
        return pd.DataFrame(), pd.DataFrame()

    closure_insts = set(df_closures['institution_en'])
    # Build a lookup: institution → closure_year
    inst_closure_yr = dict(zip(df_closures['institution_en'],
                                df_closures['closure_year']))

    # For each closure institution, filter events in its pre-window
    treated_records = []
    for inst, cl_year in inst_closure_yr.items():
        pre_start = cl_year - pre_window
        # Events at this institution in the pre-window
        mask = ((df_events['institution_en'] == inst) &
                (df_events['year'] >= pre_start) &
                (df_events['year'] <= cl_year))
        inst_pre = df_events[mask]
        if len(inst_pre) == 0:
            continue

        # Artists who visited this institution in the pre-window
        candidate_aids = inst_pre['artist_id'].unique()

        # Total events per artist in the pre-window (any institution)
        all_pre = df_events[
            (df_events['artist_id'].isin(candidate_aids)) &
            (df_events['year'] >= pre_start) &
            (df_events['year'] <= cl_year)
        ]
        total_per_artist = all_pre.groupby('artist_id').size()
        at_inst_per_artist = inst_pre.groupby('artist_id').size()

        for aid in candidate_aids:
            total = total_per_artist.get(aid, 0)
            at_inst = at_inst_per_artist.get(aid, 0)
            if total == 0:
                continue
            share = at_inst / total
            if share >= treatment_threshold:
                treated_records.append({
                    'artist_id': aid,
                    'closure_institution': inst,
                    'closure_year': cl_year,
                    'share_at_inst': share,
                    'pre_events_total': total,
                    'pre_events_at_inst': at_inst,
                    'treated': 1,
                })

    df_treated = pd.DataFrame(treated_records)
    if len(df_treated) == 0:
        print("  No treated artists found.")
        return pd.DataFrame(), pd.DataFrame()

    # Deduplicate: keep the closure with highest share for each artist
    df_treated = df_treated.sort_values('share_at_inst', ascending=False)
    df_treated = df_treated.drop_duplicates(subset='artist_id', keep='first')

    treated_ids = set(df_treated['artist_id'])
    print(f"  Treated artists (≥ {treatment_threshold*100:.0f}% at closing inst): "
          f"{len(treated_ids)}")

    # Control: artists with minimal exposure to any closing institution
    min_cl_year = df_closures['closure_year'].min()
    max_cl_year = df_closures['closure_year'].max()

    period_events = df_events[
        (df_events['year'] >= min_cl_year - pre_window) &
        (df_events['year'] <= max_cl_year + pre_window)
    ]
    total_by_artist = period_events.groupby('artist_id').size()
    closure_events = period_events[
        period_events['institution_en'].isin(closure_insts)
    ]
    closure_by_artist = closure_events.groupby('artist_id').size()

    ref_year = int(df_closures['closure_year'].median())
    control_records = []
    for aid, total in total_by_artist.items():
        if aid in treated_ids:
            continue
        if total < 3:
            continue
        exposure = closure_by_artist.get(aid, 0) / total
        if exposure < control_threshold:
            control_records.append({
                'artist_id': aid,
                'closure_year': ref_year,
                'share_at_inst': exposure,
                'pre_events_total': total,
                'treated': 0,
            })

    df_control = pd.DataFrame(control_records)
    print(f"  Potential control artists (< {control_threshold*100:.0f}% exposure): "
          f"{len(df_control)}")

    return df_treated, df_control


# ============================================================
# PART 4: DiD Analysis
# ============================================================

def did_analysis(df_treated, df_control, df_events, df_artists, panel):
    """
    Difference-in-Differences comparing treated (lost primary institution)
    vs. control artists, before vs. after closure.
    """
    print(f"\n{'=' * 70}")
    print("PART 4: DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print("=" * 70)

    if len(df_treated) == 0 or len(df_control) == 0:
        print("  Insufficient data for DiD.")
        return {}

    results = {}

    # ---- 4a: Network stability change ----
    print("\n  --- 4a: Network Stability Change (Pre vs. Post Closure) ---")

    stability_records = []
    for source, df_src, is_treated in [
        ('treated', df_treated, 1), ('control', df_control, 0)
    ]:
        for _, row in df_src.iterrows():
            aid = row['artist_id']
            cl_yr = row['closure_year']

            # Pre-closure stability (3 years before)
            pre_events = df_events[
                (df_events['artist_id'] == aid) &
                (df_events['year'] >= cl_yr - 3) &
                (df_events['year'] <= cl_yr)
            ]
            # Post-closure stability (3 years after)
            post_events = df_events[
                (df_events['artist_id'] == aid) &
                (df_events['year'] > cl_yr) &
                (df_events['year'] <= cl_yr + 3)
            ]

            if len(pre_events) < 1 or len(post_events) < 1:
                continue

            # Stability = events / unique institutions
            pre_stab = len(pre_events) / max(pre_events['institution_en'].nunique(), 1)
            post_stab = len(post_events) / max(post_events['institution_en'].nunique(), 1)
            delta_stab = post_stab - pre_stab

            # Get artist birth year
            artist_info = df_artists[df_artists['artist_id'] == aid]
            by = artist_info['birth_year'].values[0] if len(artist_info) > 0 else np.nan
            cs = artist_info['career_start_year'].values[0] if len(artist_info) > 0 else np.nan
            career_yr = cl_yr - cs if pd.notna(cs) else np.nan

            stability_records.append({
                'artist_id': aid,
                'treated': is_treated,
                'pre_stability': pre_stab,
                'post_stability': post_stab,
                'delta_stability': delta_stab,
                'birth_year': by,
                'career_year_at_closure': career_yr,
            })

    df_stab = pd.DataFrame(stability_records)
    if len(df_stab) < 10:
        print(f"  Only {len(df_stab)} observations — too few for regression.")
        return {}

    print(f"  Observations: {len(df_stab)} "
          f"(treated={df_stab['treated'].sum()}, "
          f"control={(df_stab['treated']==0).sum()})")

    # DiD on stability change
    t_mean = df_stab[df_stab['treated'] == 1]['delta_stability'].mean()
    c_mean = df_stab[df_stab['treated'] == 0]['delta_stability'].mean()
    did_stab = t_mean - c_mean

    print(f"  Treated Δstability: {t_mean:+.3f}")
    print(f"  Control Δstability: {c_mean:+.3f}")
    print(f"  DiD (stability): {did_stab:+.3f}")

    # Regression with controls
    df_reg = df_stab.dropna(subset=['birth_year', 'career_year_at_closure']).copy()
    if len(df_reg) >= 10:
        y = df_reg['delta_stability'].values
        X = df_reg[['treated']].copy()
        X['birth_year_z'] = StandardScaler().fit_transform(
            df_reg[['birth_year']])
        X['career_year_z'] = StandardScaler().fit_transform(
            df_reg[['career_year_at_closure']])
        X = sm.add_constant(X.values)

        model = sm.OLS(y, X).fit(cov_type='HC1')
        did_coef = model.params[1]
        did_p = model.pvalues[1]
        print(f"  Regression: DiD coef = {did_coef:+.3f}, p = {did_p:.4f}")
        results['stab_did_coef'] = did_coef
        results['stab_did_p'] = did_p

    # ---- 4b: Plateau outcome ----
    print("\n  --- 4b: Post-Closure Plateau Risk ---")

    plateau_records = []
    for source, df_src, is_treated in [
        ('treated', df_treated, 1), ('control', df_control, 0)
    ]:
        for _, row in df_src.iterrows():
            aid = row['artist_id']
            cl_yr = row['closure_year']

            # Check if artist experienced a plateau within 5 years post-closure
            post_events = df_events[
                (df_events['artist_id'] == aid) &
                (df_events['year'] > cl_yr) &
                (df_events['year'] <= cl_yr + 5)
            ]

            # Check for plateau: any 3-year gap without significant events
            sig_types = {'solo_exhibition', 'award', 'biennale',
                        'collection', 'honor'}
            sig_post = post_events[post_events['event_type'].isin(sig_types)]
            sig_years = set(sig_post['year'].unique()) if len(sig_post) > 0 else set()

            had_plateau = 0
            for start_y in range(cl_yr + 1, cl_yr + 4):
                gap = set(range(start_y, start_y + 3))
                if not gap.intersection(sig_years):
                    had_plateau = 1
                    break

            # Get artist info
            artist_info = df_artists[df_artists['artist_id'] == aid]
            by = artist_info['birth_year'].values[0] if len(artist_info) > 0 else np.nan
            cs = artist_info['career_start_year'].values[0] if len(artist_info) > 0 else np.nan
            career_yr = cl_yr - cs if pd.notna(cs) else np.nan

            # Pre-closure cumulative validation
            pre_events = df_events[
                (df_events['artist_id'] == aid) &
                (df_events['year'] <= cl_yr)
            ]
            cum_val = len(pre_events)

            plateau_records.append({
                'artist_id': aid,
                'treated': is_treated,
                'plateau_post': had_plateau,
                'birth_year': by,
                'career_year_at_closure': career_yr,
                'cum_validation': cum_val,
            })

    df_plat = pd.DataFrame(plateau_records)
    if len(df_plat) < 10:
        print(f"  Only {len(df_plat)} observations — too few.")
        return results

    t_rate = df_plat[df_plat['treated'] == 1]['plateau_post'].mean()
    c_rate = df_plat[df_plat['treated'] == 0]['plateau_post'].mean()
    diff = t_rate - c_rate

    print(f"  Observations: {len(df_plat)} "
          f"(treated={df_plat['treated'].sum()}, "
          f"control={(df_plat['treated']==0).sum()})")
    print(f"  Treated plateau rate: {t_rate:.3f}")
    print(f"  Control plateau rate: {c_rate:.3f}")
    print(f"  Difference: {diff:+.3f}")

    # Logistic regression
    df_reg2 = df_plat.dropna(
        subset=['birth_year', 'career_year_at_closure', 'cum_validation']
    ).copy()
    if len(df_reg2) >= 10 and df_reg2['plateau_post'].sum() >= 3:
        y2 = df_reg2['plateau_post'].values
        X2 = df_reg2[['treated']].copy()
        for col in ['birth_year', 'career_year_at_closure', 'cum_validation']:
            X2[f'{col}_z'] = StandardScaler().fit_transform(df_reg2[[col]])
        X2 = X2.drop(columns=['birth_year', 'career_year_at_closure',
                                'cum_validation'], errors='ignore')
        X2 = sm.add_constant(X2.values)

        try:
            model2 = sm.Logit(y2, X2).fit(disp=0)
            treat_or = np.exp(model2.params[1])
            treat_p = model2.pvalues[1]
            print(f"  Logistic: treated OR = {treat_or:.3f}, p = {treat_p:.4f}")
            results['plat_or'] = treat_or
            results['plat_p'] = treat_p
        except Exception as e:
            print(f"  Logistic failed: {e}")

    # Fisher exact test
    if len(df_plat) > 0:
        a = int(df_plat[(df_plat['treated']==1) & (df_plat['plateau_post']==1)].shape[0])
        b = int(df_plat[(df_plat['treated']==1) & (df_plat['plateau_post']==0)].shape[0])
        c = int(df_plat[(df_plat['treated']==0) & (df_plat['plateau_post']==1)].shape[0])
        d = int(df_plat[(df_plat['treated']==0) & (df_plat['plateau_post']==0)].shape[0])
        if min(a+b, c+d) > 0:
            fisher_or, fisher_p = stats.fisher_exact([[a, b], [c, d]])
            print(f"  Fisher exact: OR = {fisher_or:.3f}, p = {fisher_p:.4f}")
            results['fisher_or'] = fisher_or
            results['fisher_p'] = fisher_p

    results['n_treated'] = int(df_plat['treated'].sum())
    results['n_control'] = int((df_plat['treated'] == 0).sum())
    results['t_rate'] = t_rate
    results['c_rate'] = c_rate

    return results


# ============================================================
# PART 5: Sensitivity to Closure Definition
# ============================================================

def sensitivity_analysis(df_events, inst_stats, df_artists, panel):
    """
    Vary the closure definition (max_last_year) and check robustness.
    """
    print(f"\n{'=' * 70}")
    print("PART 5: SENSITIVITY TO CLOSURE DEFINITION")
    print("=" * 70)

    sensitivity_results = []
    for max_yr in [1993, 1995, 1997, 1999]:
        print(f"\n  --- max_last_year = {max_yr} ---")
        df_closures = identify_closures(
            inst_stats, df_events, max_last_year=max_yr,
            min_gap=3, min_pre_events=3
        )
        if len(df_closures) == 0:
            continue
        df_treated, df_control = identify_treated_control(
            df_events, df_closures, df_artists
        )
        if len(df_treated) > 0 and len(df_control) > 0:
            res = did_analysis(df_treated, df_control, df_events, df_artists, panel)
            res['max_last_year'] = max_yr
            res['n_closures'] = len(df_closures)
            sensitivity_results.append(res)

    return sensitivity_results


# ============================================================
# Summary
# ============================================================

def print_summary(main_results, sens_results):
    print("\n\n" + "=" * 70)
    print("SUMMARY FOR PAPER INTEGRATION")
    print("=" * 70)

    if main_results:
        print("\n[Main DiD Analysis (default: last ≤ 1997)]:")
        print(f"  N treated: {main_results.get('n_treated', 'N/A')}")
        print(f"  N control: {main_results.get('n_control', 'N/A')}")
        if 'stab_did_coef' in main_results:
            print(f"  Stability DiD: coef = {main_results['stab_did_coef']:+.3f}, "
                  f"p = {main_results['stab_did_p']:.4f}")
        print(f"  Treated plateau rate: {main_results.get('t_rate', 'N/A'):.3f}")
        print(f"  Control plateau rate: {main_results.get('c_rate', 'N/A'):.3f}")
        if 'fisher_or' in main_results:
            print(f"  Fisher OR: {main_results['fisher_or']:.3f}, "
                  f"p = {main_results['fisher_p']:.4f}")
        if 'plat_or' in main_results:
            print(f"  Logistic OR (treated): {main_results['plat_or']:.3f}, "
                  f"p = {main_results['plat_p']:.4f}")

    if sens_results:
        print("\n[Sensitivity Analysis]:")
        for sr in sens_results:
            yr = sr.get('max_last_year', '?')
            nc = sr.get('n_closures', 0)
            nt = sr.get('n_treated', 0)
            tr = sr.get('t_rate', float('nan'))
            cr = sr.get('c_rate', float('nan'))
            print(f"  max_yr={yr}: closures={nc}, treated={nt}, "
                  f"treated_rate={tr:.3f}, control_rate={cr:.3f}", end="")
            if 'fisher_or' in sr:
                print(f", Fisher OR={sr['fisher_or']:.3f} p={sr['fisher_p']:.4f}")
            else:
                print()


# ============================================================
# PART 6: Dose-Response Analysis (Continuous Exposure)
# ============================================================

def dose_response_analysis(df_treated, df_control, df_events, df_artists):
    """
    Instead of binary treated/control, use CONTINUOUS exposure share
    as the independent variable. If evaluative redundancy is real,
    higher pre-closure reliance on the closing institution should
    predict LOWER post-closure plateau risk (forced diversification
    is stronger for higher-exposure artists).
    """
    print(f"\n{'=' * 70}")
    print("PART 6: DOSE-RESPONSE ANALYSIS (Continuous Exposure)")
    print("=" * 70)

    if len(df_treated) == 0:
        print("  No treated artists.")
        return {}

    # Combine treated and control with their exposure shares
    df_all = pd.concat([df_treated, df_control], ignore_index=True)

    # Compute post-closure plateau for each artist
    plateau_records = []
    sig_types = {'solo_exhibition', 'award', 'biennale', 'collection', 'honor'}

    for _, row in df_all.iterrows():
        aid = row['artist_id']
        cl_yr = row['closure_year']
        share = row.get('share_at_inst', 0)

        # Post-closure events (5-year window)
        post_events = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] > cl_yr) &
            (df_events['year'] <= cl_yr + 5)
        ]
        sig_post = post_events[post_events['event_type'].isin(sig_types)]
        sig_years = set(sig_post['year'].unique()) if len(sig_post) > 0 else set()

        had_plateau = 0
        for start_y in range(int(cl_yr) + 1, int(cl_yr) + 4):
            gap = set(range(start_y, start_y + 3))
            if not gap.intersection(sig_years):
                had_plateau = 1
                break

        # Pre-closure characteristics
        pre_events = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] <= cl_yr)
        ]
        cum_val = len(pre_events)
        n_institutions = pre_events['institution_en'].nunique() if len(pre_events) > 0 else 0

        # Pre-closure productivity trend (events in 3 years before vs. 3 years before that)
        recent = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] > cl_yr - 3) &
            (df_events['year'] <= cl_yr)
        ]
        earlier = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] > cl_yr - 6) &
            (df_events['year'] <= cl_yr - 3)
        ]
        prod_trend = len(recent) - len(earlier)

        artist_info = df_artists[df_artists['artist_id'] == aid]
        by = artist_info['birth_year'].values[0] if len(artist_info) > 0 else np.nan
        cs = artist_info['career_start_year'].values[0] if len(artist_info) > 0 else np.nan
        career_yr = cl_yr - cs if pd.notna(cs) else np.nan

        plateau_records.append({
            'artist_id': aid,
            'exposure_share': share,
            'treated': int(row.get('treated', 0)),
            'plateau_post': had_plateau,
            'cum_validation': cum_val,
            'n_institutions': n_institutions,
            'prod_trend': prod_trend,
            'birth_year': by,
            'career_year_at_closure': career_yr,
        })

    df_dr = pd.DataFrame(plateau_records)
    df_dr = df_dr.dropna(subset=['birth_year', 'career_year_at_closure'])

    print(f"  N = {len(df_dr)} artists")
    print(f"  Exposure share: mean={df_dr['exposure_share'].mean():.3f}, "
          f"median={df_dr['exposure_share'].median():.3f}")
    print(f"  Plateau rate: {df_dr['plateau_post'].mean():.3f}")

    results = {}

    # Model 1: Binary (original)
    if df_dr['plateau_post'].sum() >= 3 and len(df_dr) >= 20:
        y = df_dr['plateau_post'].values
        X1 = df_dr[['treated']].copy()
        for col in ['birth_year', 'career_year_at_closure', 'cum_validation']:
            X1[f'{col}_z'] = StandardScaler().fit_transform(df_dr[[col]])
        X1 = sm.add_constant(X1[['treated', 'birth_year_z',
                                   'career_year_at_closure_z',
                                   'cum_validation_z']].values)
        try:
            m1 = sm.Logit(y, X1).fit(disp=0)
            print(f"\n  Model 1 (binary): OR = {np.exp(m1.params[1]):.3f}, "
                  f"p = {m1.pvalues[1]:.4f}")
            results['binary_or'] = np.exp(m1.params[1])
            results['binary_p'] = m1.pvalues[1]
        except Exception as e:
            print(f"  Model 1 failed: {e}")

    # Model 2: Continuous exposure (dose-response)
    if df_dr['plateau_post'].sum() >= 3 and len(df_dr) >= 20:
        y = df_dr['plateau_post'].values
        sc = StandardScaler()
        df_dr['exposure_z'] = sc.fit_transform(df_dr[['exposure_share']]).flatten()
        for col in ['birth_year', 'career_year_at_closure', 'cum_validation']:
            df_dr[f'{col}_z'] = sc.fit_transform(df_dr[[col]]).flatten()
        X2 = sm.add_constant(df_dr[['exposure_z', 'birth_year_z',
                                     'career_year_at_closure_z',
                                     'cum_validation_z']].values)
        try:
            m2 = sm.Logit(y, X2).fit(disp=0)
            print(f"  Model 2 (dose-response): OR = {np.exp(m2.params[1]):.3f}, "
                  f"p = {m2.pvalues[1]:.4f}")
            results['dose_or'] = np.exp(m2.params[1])
            results['dose_p'] = m2.pvalues[1]
        except Exception as e:
            print(f"  Model 2 failed: {e}")

    # Model 3: With productivity trend control
    if df_dr['plateau_post'].sum() >= 3 and len(df_dr) >= 20:
        y = df_dr['plateau_post'].values
        df_dr['prod_trend_z'] = sc.fit_transform(df_dr[['prod_trend']]).flatten()
        X3 = sm.add_constant(df_dr[['exposure_z', 'birth_year_z',
                                     'career_year_at_closure_z',
                                     'cum_validation_z',
                                     'prod_trend_z']].values)
        try:
            m3 = sm.Logit(y, X3).fit(disp=0)
            print(f"  Model 3 (+ prod trend): OR = {np.exp(m3.params[1]):.3f}, "
                  f"p = {m3.pvalues[1]:.4f}")
            print(f"    Prod trend OR = {np.exp(m3.params[5]):.3f}, "
                  f"p = {m3.pvalues[5]:.4f}")
            results['cond_or'] = np.exp(m3.params[1])
            results['cond_p'] = m3.pvalues[1]
        except Exception as e:
            print(f"  Model 3 failed: {e}")

    # Exposure tertile comparison
    if len(df_dr) >= 30:
        try:
            df_dr['exp_tertile'] = pd.qcut(df_dr['exposure_share'], 3,
                                            labels=['Low', 'Mid', 'High'],
                                            duplicates='drop')
            print(f"\n  Plateau rate by exposure tertile:")
            for t in ['Low', 'Mid', 'High']:
                sub = df_dr[df_dr['exp_tertile'] == t]
                if len(sub) > 0:
                    rate = sub['plateau_post'].mean()
                    print(f"    {t:5s}: {rate:.3f} (n={len(sub)})")
        except Exception:
            pass

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "═" * 70)
    print("  INSTITUTION CLOSURE QUASI-EXPERIMENT")
    print("═" * 70 + "\n")

    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)
    panel = build_person_year_panel(df_artists, df_events, cutpoint=10)

    print(f"Events: {len(df_events)}, Artists: {df_artists.shape[0]}, "
          f"Institutions: {df_events['institution_en'].nunique()}\n")

    # Part 1: Lifecycle tracking
    inst_stats = track_institution_lifecycles(df_events)

    # Part 2: Default closure identification
    df_closures = identify_closures(inst_stats, df_events, max_last_year=1997)

    # Part 3: Treatment/control
    df_treated, df_control = identify_treated_control(
        df_events, df_closures, df_artists
    )

    # Part 4: DiD
    main_results = {}
    if len(df_treated) > 0 and len(df_control) > 0:
        main_results = did_analysis(
            df_treated, df_control, df_events, df_artists, panel
        )

    # Part 5: Sensitivity
    sens_results = sensitivity_analysis(
        df_events, inst_stats, df_artists, panel
    )

    # Part 6: Dose-response
    dr_results = {}
    if len(df_treated) > 0 and len(df_control) > 0:
        dr_results = dose_response_analysis(
            df_treated, df_control, df_events, df_artists
        )

    print_summary(main_results, sens_results)

    if dr_results:
        print("\n[Dose-Response Analysis]:")
        if 'binary_or' in dr_results:
            print(f"  Binary OR: {dr_results['binary_or']:.3f} "
                  f"(p={dr_results['binary_p']:.4f})")
        if 'dose_or' in dr_results:
            print(f"  Dose-response OR: {dr_results['dose_or']:.3f} "
                  f"(p={dr_results['dose_p']:.4f})")
        if 'cond_or' in dr_results:
            print(f"  Conditioned OR: {dr_results['cond_or']:.3f} "
                  f"(p={dr_results['cond_p']:.4f})")

    print("\n" + "═" * 70)
    print("  ANALYSIS COMPLETE")
    print("═" * 70)

    return main_results, sens_results, dr_results


if __name__ == '__main__':
    results = main()
