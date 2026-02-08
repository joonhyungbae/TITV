"""
25_closure_event_study.py
════════════════════════════════════════════════════════════════════
Strengthens the quasi-experimental institution closure analysis with:

  (1) Event Study Design
      — Plots relative-time coefficients for treated vs. control artists
        from t-5 to t+5 around institution closure, verifying pre-trend
        parallelism and visualising the post-closure divergence.

  (2) Propensity Score Matching (PSM)
      — Matches treated artists (≥20% exposure to a closing institution)
        to controls (<5% exposure) on birth_year, career_year_at_closure,
        cumulative_validation, and pre-closure network stability.
      — Estimates ATT on post-closure plateau risk.

  (3) Combined Figure
      — Generates figures/fig4_closure_event_study.png for inclusion
        in the paper.
════════════════════════════════════════════════════════════════════
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    build_person_year_panel, CENSOR_YEAR, SIGNIFICANT_EVENT_TYPES
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
np.random.seed(42)


# ============================================================
# Reuse closure identification from script 20
# ============================================================

def identify_closures(df_events, max_last_year=1997, min_gap=3,
                      min_pre_events=3, pre_window=5):
    """Identify institution closures (same logic as script 20)."""
    inst_stats = df_events.groupby('institution_en').agg(
        first_year=('year', 'min'),
        last_year=('year', 'max'),
        total_events=('year', 'count'),
        n_artists=('artist_id', 'nunique'),
    ).reset_index()

    candidates = inst_stats[inst_stats['last_year'] <= max_last_year].copy()
    candidates['gap_after'] = CENSOR_YEAR - candidates['last_year']
    candidates = candidates[candidates['gap_after'] >= min_gap]

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
            })

    return pd.DataFrame(closures)


def identify_treated_control(df_events, df_closures, df_artists,
                             treatment_threshold=0.20,
                             control_threshold=0.05,
                             pre_window=5):
    """Identify treated and control artists (same logic as script 20)."""
    if len(df_closures) == 0:
        return pd.DataFrame(), pd.DataFrame()

    inst_closure_yr = dict(zip(df_closures['institution_en'],
                               df_closures['closure_year']))
    closure_insts = set(df_closures['institution_en'])

    treated_records = []
    for inst, cl_year in inst_closure_yr.items():
        pre_start = cl_year - pre_window
        mask = ((df_events['institution_en'] == inst) &
                (df_events['year'] >= pre_start) &
                (df_events['year'] <= cl_year))
        inst_pre = df_events[mask]
        if len(inst_pre) == 0:
            continue

        candidate_aids = inst_pre['artist_id'].unique()
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
        return pd.DataFrame(), pd.DataFrame()

    df_treated = df_treated.sort_values('share_at_inst', ascending=False)
    df_treated = df_treated.drop_duplicates(subset='artist_id', keep='first')
    treated_ids = set(df_treated['artist_id'])

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
    return df_treated, df_control


# ============================================================
# PART 1: Event Study Design
# ============================================================

def event_study(df_treated, df_control, df_events, df_artists):
    """
    Construct relative-time panel around closure year and estimate
    event-study coefficients (treated × relative_year dummies).
    """
    print("=" * 70)
    print("PART 1: EVENT STUDY DESIGN")
    print("=" * 70)

    sig_types = SIGNIFICANT_EVENT_TYPES
    window = 5  # years before and after

    records = []
    for source, df_src, is_treated in [
        ('treated', df_treated, 1), ('control', df_control, 0)
    ]:
        for _, row in df_src.iterrows():
            aid = row['artist_id']
            cl_yr = int(row['closure_year'])

            for rel_t in range(-window, window + 1):
                abs_yr = cl_yr + rel_t
                if abs_yr > CENSOR_YEAR or abs_yr < 1929:
                    continue

                # Count significant events in this year
                yr_events = df_events[
                    (df_events['artist_id'] == aid) &
                    (df_events['year'] == abs_yr)
                ]
                sig_count = yr_events[yr_events['event_type'].isin(sig_types)].shape[0]
                any_sig = int(sig_count > 0)

                # Total events this year
                total_yr = yr_events.shape[0]

                records.append({
                    'artist_id': aid,
                    'treated': is_treated,
                    'rel_year': rel_t,
                    'abs_year': abs_yr,
                    'sig_events': sig_count,
                    'any_sig': any_sig,
                    'total_events': total_yr,
                })

    df_es = pd.DataFrame(records)
    print(f"  Event study panel: {len(df_es)} obs, "
          f"{df_es['artist_id'].nunique()} artists")

    # Aggregate: mean significant event rate by treated × rel_year
    agg = df_es.groupby(['treated', 'rel_year']).agg(
        mean_sig=('any_sig', 'mean'),
        mean_count=('sig_events', 'mean'),
        mean_total=('total_events', 'mean'),
        n=('artist_id', 'count'),
    ).reset_index()

    # Print pre-trend comparison
    print("\n  Pre-trend (mean significant event rate):")
    for rt in range(-window, 0):
        t_rate = agg[(agg['treated'] == 1) & (agg['rel_year'] == rt)]['mean_sig'].values
        c_rate = agg[(agg['treated'] == 0) & (agg['rel_year'] == rt)]['mean_sig'].values
        t_val = t_rate[0] if len(t_rate) > 0 else float('nan')
        c_val = c_rate[0] if len(c_rate) > 0 else float('nan')
        print(f"    t={rt:+d}: treated={t_val:.3f}, control={c_val:.3f}, "
              f"diff={t_val - c_val:+.3f}")

    # Regression: DiD event study with relative-year dummies
    # Omit t=-1 as reference period
    df_reg = df_es.copy()
    df_reg['post'] = (df_reg['rel_year'] > 0).astype(int)

    # Create relative year dummies (excluding -1)
    rel_years = sorted(df_reg['rel_year'].unique())
    ref_year_dummy = -1
    for rt in rel_years:
        if rt != ref_year_dummy:
            df_reg[f'ry_{rt}'] = (df_reg['rel_year'] == rt).astype(float)
            df_reg[f'treat_ry_{rt}'] = df_reg['treated'] * df_reg[f'ry_{rt}']

    # Fit OLS for event study coefficients
    treat_ry_cols = [f'treat_ry_{rt}' for rt in rel_years if rt != ref_year_dummy]
    ry_cols = [f'ry_{rt}' for rt in rel_years if rt != ref_year_dummy]

    X_cols = ['treated'] + ry_cols + treat_ry_cols
    X = sm.add_constant(df_reg[X_cols].values)
    y = df_reg['any_sig'].values

    model = sm.OLS(y, X).fit(cov_type='cluster',
                             cov_kwds={'groups': df_reg['artist_id'].values})

    # Extract treatment × rel_year coefficients
    coef_names = ['const', 'treated'] + ry_cols + treat_ry_cols
    es_coefs = {}
    for rt in rel_years:
        if rt == ref_year_dummy:
            es_coefs[rt] = {'coef': 0.0, 'se': 0.0, 'p': 1.0}
        else:
            idx = coef_names.index(f'treat_ry_{rt}')
            es_coefs[rt] = {
                'coef': model.params[idx],
                'se': model.bse[idx],
                'p': model.pvalues[idx],
            }

    print("\n  Event study coefficients (treated × rel_year, ref=t-1):")
    for rt in sorted(es_coefs.keys()):
        c = es_coefs[rt]
        sig = '*' if c['p'] < 0.05 else ('†' if c['p'] < 0.10 else '')
        print(f"    t={rt:+d}: β={c['coef']:+.4f} (se={c['se']:.4f}, p={c['p']:.3f}) {sig}")

    # Pre-trend test: joint significance of pre-treatment interactions
    pre_indices = [coef_names.index(f'treat_ry_{rt}')
                   for rt in rel_years if rt < -1]
    if len(pre_indices) >= 2:
        pre_coefs = model.params[pre_indices]
        pre_cov = model.cov_params()[np.ix_(pre_indices, pre_indices)]
        try:
            wald = pre_coefs @ np.linalg.inv(pre_cov) @ pre_coefs
            wald_p = 1 - stats.chi2.cdf(wald, df=len(pre_indices))
            print(f"\n  Pre-trend joint Wald test: chi2({len(pre_indices)}) = {wald:.3f}, "
                  f"p = {wald_p:.4f}")
        except np.linalg.LinAlgError:
            wald_p = np.nan
            print(f"\n  Pre-trend Wald: singular covariance matrix")
    else:
        wald_p = np.nan

    return agg, es_coefs, wald_p


# ============================================================
# PART 2: Propensity Score Matching
# ============================================================

def psm_analysis(df_treated, df_control, df_events, df_artists):
    """
    Nearest-neighbor propensity score matching on pre-closure
    covariates, then estimate ATT on post-closure plateau risk.
    """
    print("\n" + "=" * 70)
    print("PART 2: PROPENSITY SCORE MATCHING")
    print("=" * 70)

    sig_types = SIGNIFICANT_EVENT_TYPES

    # Build covariate matrix for all artists
    all_artists = pd.concat([df_treated, df_control], ignore_index=True)
    cov_records = []

    for _, row in all_artists.iterrows():
        aid = row['artist_id']
        cl_yr = int(row['closure_year'])
        is_treated = int(row['treated'])
        share = row.get('share_at_inst', 0)

        # Get artist info
        artist_info = df_artists[df_artists['artist_id'] == aid]
        by = artist_info['birth_year'].values[0] if len(artist_info) > 0 else np.nan
        cs = artist_info['career_start_year'].values[0] if len(artist_info) > 0 else np.nan
        career_yr = cl_yr - cs if pd.notna(cs) else np.nan

        # Pre-closure events
        pre_events = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] <= cl_yr)
        ]
        cum_val = len(pre_events)
        n_inst = pre_events['institution_en'].nunique() if len(pre_events) > 0 else 0
        pre_stability = cum_val / max(n_inst, 1)

        # Pre-closure productivity (events in 3 years before closure)
        recent = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] > cl_yr - 3) &
            (df_events['year'] <= cl_yr)
        ]
        pre_prod = len(recent)

        # Post-closure plateau
        post_events = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] > cl_yr) &
            (df_events['year'] <= cl_yr + 5)
        ]
        sig_post = post_events[post_events['event_type'].isin(sig_types)]
        sig_years = set(sig_post['year'].unique()) if len(sig_post) > 0 else set()

        had_plateau = 0
        for start_y in range(cl_yr + 1, cl_yr + 4):
            gap = set(range(start_y, start_y + 3))
            if not gap.intersection(sig_years):
                had_plateau = 1
                break

        cov_records.append({
            'artist_id': aid,
            'treated': is_treated,
            'birth_year': by,
            'career_year': career_yr,
            'cum_validation': cum_val,
            'pre_stability': pre_stability,
            'pre_prod': pre_prod,
            'n_institutions': n_inst,
            'plateau_post': had_plateau,
        })

    df_psm = pd.DataFrame(cov_records).dropna()
    print(f"  Sample: {len(df_psm)} artists "
          f"(treated={df_psm['treated'].sum()}, "
          f"control={(df_psm['treated']==0).sum()})")

    # Estimate propensity score
    covs = ['birth_year', 'career_year', 'cum_validation',
            'pre_stability', 'pre_prod']
    X_ps = sm.add_constant(df_psm[covs].values)
    y_ps = df_psm['treated'].values

    ps_model = sm.Logit(y_ps, X_ps).fit(disp=0)
    df_psm['pscore'] = ps_model.predict(X_ps)

    print(f"\n  Propensity score distribution:")
    for grp, label in [(1, 'Treated'), (0, 'Control')]:
        sub = df_psm[df_psm['treated'] == grp]
        print(f"    {label}: mean={sub['pscore'].mean():.3f}, "
              f"sd={sub['pscore'].std():.3f}, "
              f"range=[{sub['pscore'].min():.3f}, {sub['pscore'].max():.3f}]")

    # Nearest-neighbor matching (1:1 without replacement)
    treated_df = df_psm[df_psm['treated'] == 1].copy()
    control_df = df_psm[df_psm['treated'] == 0].copy()

    if len(control_df) < 5:
        print("  Insufficient controls for matching.")
        return {}

    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control_df[['pscore']].values)
    distances, indices = nn.kneighbors(treated_df[['pscore']].values)

    # Match
    matched_control_idx = control_df.index[indices.flatten()]
    matched_control = control_df.loc[matched_control_idx].copy()
    matched_treated = treated_df.copy()

    # Caliper check (0.1 SD of pscore)
    caliper = 0.1 * df_psm['pscore'].std()
    good_matches = distances.flatten() <= caliper
    n_good = good_matches.sum()
    print(f"\n  Caliper ({caliper:.4f}): {n_good}/{len(matched_treated)} "
          f"matches within caliper")

    # Use caliper-filtered matches
    matched_treated = matched_treated[good_matches].copy()
    matched_control_filtered = matched_control[good_matches].copy()

    if len(matched_treated) < 10:
        print("  Too few matched pairs. Using all matches.")
        matched_treated = treated_df.copy()
        matched_control_filtered = matched_control.copy()

    print(f"\n  Matched sample: {len(matched_treated)} pairs")

    # Covariate balance check
    print("\n  Covariate balance (standardized mean difference):")
    print(f"  {'Covariate':20s} {'Before':>10s} {'After':>10s}")
    for cov in covs:
        # Before matching
        t_mean = treated_df[cov].mean()
        c_mean = control_df[cov].mean()
        pooled_sd = np.sqrt((treated_df[cov].var() + control_df[cov].var()) / 2)
        smd_before = (t_mean - c_mean) / pooled_sd if pooled_sd > 0 else 0

        # After matching
        mt_mean = matched_treated[cov].mean()
        mc_mean = matched_control_filtered[cov].mean()
        pooled_sd_m = np.sqrt((matched_treated[cov].var() +
                               matched_control_filtered[cov].var()) / 2)
        smd_after = (mt_mean - mc_mean) / pooled_sd_m if pooled_sd_m > 0 else 0

        print(f"  {cov:20s} {smd_before:+10.3f} {smd_after:+10.3f}")

    # ATT estimation
    t_rate = matched_treated['plateau_post'].mean()
    c_rate = matched_control_filtered['plateau_post'].mean()
    att = t_rate - c_rate
    print(f"\n  ATT (matched):")
    print(f"    Treated plateau rate: {t_rate:.3f}")
    print(f"    Control plateau rate: {c_rate:.3f}")
    print(f"    ATT = {att:+.3f}")

    # McNemar's test for matched pairs
    a = int(((matched_treated['plateau_post'].values == 1) &
             (matched_control_filtered['plateau_post'].values == 0)).sum())
    b = int(((matched_treated['plateau_post'].values == 0) &
             (matched_control_filtered['plateau_post'].values == 1)).sum())
    if a + b > 0:
        mcnemar_stat = (abs(a - b) - 1)**2 / (a + b) if (a + b) > 0 else 0
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        print(f"    McNemar test: chi2 = {mcnemar_stat:.3f}, p = {mcnemar_p:.4f}")
    else:
        mcnemar_p = np.nan

    # Logistic regression on matched sample
    matched_all = pd.concat([matched_treated, matched_control_filtered],
                            ignore_index=True)
    if matched_all['plateau_post'].sum() >= 3 and len(matched_all) >= 20:
        y_m = matched_all['plateau_post'].values
        X_m = sm.add_constant(matched_all[['treated']].values)
        try:
            m_logit = sm.Logit(y_m, X_m).fit(disp=0)
            matched_or = np.exp(m_logit.params[1])
            matched_p = m_logit.pvalues[1]
            print(f"    Matched logistic OR = {matched_or:.3f}, p = {matched_p:.4f}")
        except Exception as e:
            matched_or, matched_p = np.nan, np.nan
            print(f"    Matched logistic failed: {e}")
    else:
        matched_or, matched_p = np.nan, np.nan

    return {
        'n_matched': len(matched_treated),
        'att': att,
        't_rate': t_rate,
        'c_rate': c_rate,
        'mcnemar_p': mcnemar_p,
        'matched_or': matched_or,
        'matched_p': matched_p,
    }


# ============================================================
# PART 3: Event Study Figure
# ============================================================

def plot_event_study(agg, es_coefs, psm_results, wald_p):
    """Generate the event study figure."""
    print("\n" + "=" * 70)
    print("PART 3: GENERATING EVENT STUDY FIGURE")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Raw event rates
    ax = axes[0]
    for grp, label, color, marker in [
        (1, 'Treated (high exposure)', '#c0392b', 'o'),
        (0, 'Control (low exposure)', '#2980b9', 's')
    ]:
        sub = agg[agg['treated'] == grp].sort_values('rel_year')
        ax.plot(sub['rel_year'], sub['mean_sig'],
                color=color, marker=marker, markersize=5,
                linewidth=1.5, label=label)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvspan(0.5, 5.5, alpha=0.08, color='gray')
    ax.set_xlabel('Years relative to institution closure', fontsize=11)
    ax.set_ylabel('P(significant event)', fontsize=11)
    ax.set_title('(A) Significant event rates', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(-5.5, 5.5)

    # Panel B: Event study coefficients
    ax = axes[1]
    rel_years = sorted(es_coefs.keys())
    coefs = [es_coefs[rt]['coef'] for rt in rel_years]
    ses = [es_coefs[rt]['se'] for rt in rel_years]
    ci_lo = [c - 1.96 * s for c, s in zip(coefs, ses)]
    ci_hi = [c + 1.96 * s for c, s in zip(coefs, ses)]

    ax.fill_between(rel_years, ci_lo, ci_hi, alpha=0.15, color='#2c3e50')
    ax.plot(rel_years, coefs, color='#2c3e50', marker='o', markersize=5,
            linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # Mark reference period
    ax.scatter([-1], [0], color='red', marker='D', s=60, zorder=5,
               label='Reference (t = -1)')

    # Annotate pre-trend test
    if not np.isnan(wald_p):
        ax.text(0.02, 0.98,
                f'Pre-trend Wald p = {wald_p:.3f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.8))

    ax.set_xlabel('Years relative to institution closure', fontsize=11)
    ax.set_ylabel('DiD coefficient (treated x rel. year)', fontsize=11)
    ax.set_title('(B) Event study coefficients', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_xlim(-5.5, 5.5)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'fig4_closure_event_study.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Summary
# ============================================================

def print_summary(es_coefs, wald_p, psm_results):
    print("\n\n" + "=" * 70)
    print("SUMMARY FOR PAPER INTEGRATION")
    print("=" * 70)

    print("\n[1] Event Study:")
    print(f"  Pre-trend Wald test: p = {wald_p:.4f}")
    pre_coefs = {rt: es_coefs[rt] for rt in es_coefs if rt < -1}
    max_pre = max(abs(c['coef']) for c in pre_coefs.values()) if pre_coefs else 0
    print(f"  Max pre-treatment |coef| = {max_pre:.4f}")
    print(f"  Post-treatment coefficients:")
    for rt in sorted(es_coefs.keys()):
        if rt > 0:
            c = es_coefs[rt]
            print(f"    t+{rt}: β = {c['coef']:+.4f}, p = {c['p']:.3f}")

    print(f"\n[2] Propensity Score Matching:")
    if psm_results:
        print(f"  Matched pairs: {psm_results['n_matched']}")
        print(f"  Treated plateau rate: {psm_results['t_rate']:.3f}")
        print(f"  Control plateau rate: {psm_results['c_rate']:.3f}")
        print(f"  ATT = {psm_results['att']:+.3f}")
        if not np.isnan(psm_results.get('matched_or', np.nan)):
            print(f"  Matched logistic OR = {psm_results['matched_or']:.3f}, "
                  f"p = {psm_results['matched_p']:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  INSTITUTION CLOSURE: EVENT STUDY & PSM")
    print("=" * 70 + "\n")

    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)

    print(f"Events: {len(df_events)}, Artists: {df_artists.shape[0]}, "
          f"Institutions: {df_events['institution_en'].nunique()}\n")

    # Identify closures and treatment groups
    df_closures = identify_closures(df_events, max_last_year=1997)
    print(f"Closures identified: {len(df_closures)}")

    df_treated, df_control = identify_treated_control(
        df_events, df_closures, df_artists
    )
    print(f"Treated: {len(df_treated)}, Control: {len(df_control)}\n")

    # Part 1: Event study
    agg, es_coefs, wald_p = event_study(
        df_treated, df_control, df_events, df_artists
    )

    # Part 2: PSM
    psm_results = psm_analysis(
        df_treated, df_control, df_events, df_artists
    )

    # Part 3: Figure
    plot_event_study(agg, es_coefs, psm_results, wald_p)

    # Summary
    print_summary(es_coefs, wald_p, psm_results)

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)

    return es_coefs, wald_p, psm_results


if __name__ == '__main__':
    results = main()
