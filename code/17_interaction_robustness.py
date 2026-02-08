"""
17_interaction_robustness.py
════════════════════════════════════════════════════════════════════
Addresses the non-significance of the stability × career_year
interaction term (p = 0.155) with four complementary analyses:

  (1) Permutation test — model-free p-value for the interaction
  (2) Joint Wald test  — tests both interaction terms simultaneously
  (3) Binary phase interaction — stability × I(career_year >= 10)
  (4) Cluster bootstrap — artist-level resampled CIs for conditional HRs
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
np.random.seed(42)


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


FORMULA_FULL = ('network_stability_z + network_size_z + career_year_z + '
                'stab_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z')


def compute_conditional_hrs(ctv, panel, years=(0, 5, 10, 15, 20),
                             stab_key='network_stability_z',
                             int_key='stab_x_caryr'):
    """Return conditional stability HR at specified career years."""
    s = ctv.summary
    beta_s = s.loc[stab_key, 'coef']
    beta_i = s.loc[int_key, 'coef']
    se_s = s.loc[stab_key, 'se(coef)']
    se_i = s.loc[int_key, 'se(coef)']
    vm = ctv.variance_matrix_
    cov_si = vm.loc[stab_key, int_key] if stab_key in vm.index else 0.0

    cy_mean = panel['career_year'].mean()
    cy_std = panel['career_year'].std()

    results = {}
    for cy in years:
        cyz = (cy - cy_mean) / cy_std
        coef = beta_s + beta_i * cyz
        se = np.sqrt(se_s**2 + (cyz**2) * se_i**2 + 2 * cyz * cov_si)
        hr = np.exp(coef)
        lo = np.exp(coef - 1.96 * se)
        hi = np.exp(coef + 1.96 * se)
        p = 2 * (1 - stats.norm.cdf(abs(coef / se)))
        results[cy] = {'hr': hr, 'lo': lo, 'hi': hi, 'p': p}
    return results


# ============================================================
# PART 1: Permutation Test (1 000 iterations)
# ============================================================

def permutation_test(panel, n_perm=1000):
    """
    Permutation test for the interaction term using event-timing permutation.

    Strategy: for each artist, randomly reassign which person-year receives
    the plateau event (if any), preserving the marginal event rate and
    at-risk structure but breaking the specific temporal association between
    network structure and plateau timing.
    """
    print("=" * 70)
    print("PART 1: PERMUTATION TEST FOR INTERACTION TERM")
    print(f"  (n_perm = {n_perm})")
    print("=" * 70)

    # Observed model
    ctv_obs = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_obs.fit(panel, id_col='id', event_col='event',
                start_col='start', stop_col='stop',
                formula=FORMULA_FULL, show_progress=False)
    observed_beta = ctv_obs.summary.loc['stab_x_caryr', 'coef']
    print(f"  Observed beta_interaction = {observed_beta:.4f}")

    # Pre-compute artist groups for efficiency
    artist_groups = {}
    for aid, grp in panel.groupby('artist_id'):
        has_event = grp['event'].sum() > 0
        n_rows = len(grp)
        artist_groups[aid] = {'has_event': has_event, 'n_rows': n_rows,
                              'indices': grp.index.values}

    null_betas = []
    for i in range(n_perm):
        p = panel.copy()
        # Permute event timing: for each artist who experienced a plateau,
        # randomly assign the event to a different person-year
        new_events = np.zeros(len(p), dtype=int)
        for aid, info in artist_groups.items():
            if info['has_event']:
                # Pick a random person-year for this artist's event
                chosen = np.random.choice(info['indices'])
                new_events[chosen] = 1
        p['event'] = new_events

        try:
            ctv_p = CoxTimeVaryingFitter(penalizer=0.01)
            ctv_p.fit(p, id_col='id', event_col='event',
                      start_col='start', stop_col='stop',
                      formula=FORMULA_FULL, show_progress=False)
            null_betas.append(ctv_p.summary.loc['stab_x_caryr', 'coef'])
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"    ... {i+1}/{n_perm} done")

    null_betas = np.array(null_betas)
    perm_p = np.mean(null_betas >= observed_beta)
    print(f"\n  Permutation p-value (one-sided): {perm_p:.4f}")
    print(f"  Null distribution: mean={null_betas.mean():.4f}, "
          f"sd={null_betas.std():.4f}, "
          f"95th pctile={np.percentile(null_betas, 95):.4f}")
    print(f"  Observed beta ({observed_beta:.4f}) exceeds "
          f"{(1 - perm_p)*100:.1f}% of null distribution")

    return {
        'observed_beta': observed_beta,
        'null_betas': null_betas,
        'perm_p': perm_p,
    }


# ============================================================
# PART 2: Joint Wald Test
# ============================================================

def joint_wald_test(panel):
    """
    Joint significance test for both interaction terms
    (stab × career_year and size × career_year).
    """
    print("\n" + "=" * 70)
    print("PART 2: JOINT WALD TEST FOR INTERACTION TERMS")
    print("=" * 70)

    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(panel, id_col='id', event_col='event',
            start_col='start', stop_col='stop',
            formula=FORMULA_FULL, show_progress=False)

    s = ctv.summary
    int_vars = ['stab_x_caryr', 'size_x_caryr']
    int_coefs = np.array([s.loc[v, 'coef'] for v in int_vars])
    vm = ctv.variance_matrix_
    var_names = list(s.index)
    int_indices = [var_names.index(v) for v in int_vars]
    cov_sub = vm.values[np.ix_(int_indices, int_indices)]

    wald_stat = int_coefs @ np.linalg.inv(cov_sub) @ int_coefs
    wald_p = 1 - stats.chi2.cdf(wald_stat, df=len(int_vars))

    print(f"  Interaction coefficients: {dict(zip(int_vars, int_coefs))}")
    print(f"  Joint Wald chi2({len(int_vars)}) = {wald_stat:.3f}, p = {wald_p:.4f}")

    # Also test stability interaction alone
    stab_coef = np.array([s.loc['stab_x_caryr', 'coef']])
    stab_idx = [var_names.index('stab_x_caryr')]
    stab_var = vm.values[np.ix_(stab_idx, stab_idx)]
    wald_stab = stab_coef @ np.linalg.inv(stab_var) @ stab_coef
    wald_stab_p = 1 - stats.chi2.cdf(float(wald_stab), df=1)
    print(f"  Stability interaction alone: Wald chi2(1) = {float(wald_stab):.3f}, p = {wald_stab_p:.4f}")

    return {
        'joint_wald_stat': wald_stat,
        'joint_wald_p': wald_p,
        'stab_wald_stat': float(wald_stab),
        'stab_wald_p': wald_stab_p,
    }


# ============================================================
# PART 3: Binary Phase Interaction
# ============================================================

def binary_phase_interaction(panel):
    """
    Replace the continuous career_year interaction with a single
    binary indicator: stability × I(career_year >= 10).
    This concentrates the interaction effect into one parameter,
    improving statistical power.
    """
    print("\n" + "=" * 70)
    print("PART 3: BINARY PHASE INTERACTION — stability × I(yr >= 10)")
    print("=" * 70)

    p = panel.copy()
    p['post10'] = (p['career_year'] >= 10).astype(float)
    p['stab_x_post10'] = p['network_stability_z'] * p['post10']

    formula_bin = ('network_stability_z + network_size_z + career_year_z + '
                   'stab_x_post10 + birth_year_z + cumulative_validation_z')

    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(p, id_col='id', event_col='event',
            start_col='start', stop_col='stop',
            formula=formula_bin, show_progress=False)

    s = ctv.summary
    print(s[['coef', 'exp(coef)', 'exp(coef) lower 95%',
             'exp(coef) upper 95%', 'p']].round(4).to_string())

    stab_main = s.loc['network_stability_z']
    stab_post = s.loc['stab_x_post10']

    # Pre-decade effect: just the main effect
    pre_hr = stab_main['exp(coef)']
    pre_p = stab_main['p']

    # Post-decade effect: main + interaction
    post_coef = stab_main['coef'] + stab_post['coef']
    vm = ctv.variance_matrix_
    var_names = list(s.index)
    idx_stab = var_names.index('network_stability_z')
    idx_post = var_names.index('stab_x_post10')
    cov_val = vm.iloc[idx_stab, idx_post]
    post_se = np.sqrt(stab_main['se(coef)']**2 + stab_post['se(coef)']**2 +
                       2 * cov_val)
    post_hr = np.exp(post_coef)
    post_lo = np.exp(post_coef - 1.96 * post_se)
    post_hi = np.exp(post_coef + 1.96 * post_se)
    post_p = 2 * (1 - stats.norm.cdf(abs(post_coef / post_se)))

    print(f"\n  Pre-decade stability HR  = {pre_hr:.3f}, p = {pre_p:.4f}")
    print(f"  Post-decade stability HR = {post_hr:.3f} [{post_lo:.3f}, {post_hi:.3f}], p = {post_p:.4f}")
    print(f"  Interaction term (stab × post10): HR = {stab_post['exp(coef)']:.3f}, "
          f"p = {stab_post['p']:.4f}")

    return {
        'interaction_hr': stab_post['exp(coef)'],
        'interaction_p': stab_post['p'],
        'pre_hr': pre_hr, 'pre_p': pre_p,
        'post_hr': post_hr, 'post_lo': post_lo, 'post_hi': post_hi, 'post_p': post_p,
    }


# ============================================================
# PART 4: Cluster Bootstrap
# ============================================================

def cluster_bootstrap(panel, n_boot=500):
    """
    Resample artists (clusters) with replacement and re-estimate
    conditional HRs at career years 0, 5, 10, 15, 20.
    """
    print("\n" + "=" * 70)
    print(f"PART 4: CLUSTER BOOTSTRAP (n_boot = {n_boot})")
    print("=" * 70)

    artist_ids = panel['artist_id'].unique()
    all_hrs = {cy: [] for cy in [0, 5, 10, 15, 20]}

    # Pre-group panels by artist for efficiency
    artist_panels = {aid: grp.copy() for aid, grp in panel.groupby('artist_id')}

    for i in range(n_boot):
        # Resample artists with replacement
        boot_ids = np.random.choice(artist_ids, size=len(artist_ids), replace=True)

        # Build bootstrap panel (handle duplicate IDs by appending suffix)
        frames = []
        for j, aid in enumerate(boot_ids):
            chunk = artist_panels[aid].copy()
            new_id = f"{aid}__{j}"
            chunk['id'] = new_id
            chunk['artist_id'] = new_id
            frames.append(chunk)
        boot_panel = pd.concat(frames, ignore_index=True)

        # Re-standardise within bootstrap sample
        scaler = StandardScaler()
        for col in ['network_stability', 'network_size', 'career_year',
                    'birth_year', 'cumulative_validation']:
            vals = boot_panel[[col]].values
            if np.std(vals) < 1e-10:
                boot_panel[f'{col}_z'] = 0.0
            else:
                boot_panel[f'{col}_z'] = scaler.fit_transform(vals).flatten()
        boot_panel['stab_x_caryr'] = boot_panel['network_stability_z'] * boot_panel['career_year_z']
        boot_panel['size_x_caryr'] = boot_panel['network_size_z'] * boot_panel['career_year_z']

        if boot_panel['event'].sum() < 10:
            continue

        try:
            ctv_b = CoxTimeVaryingFitter(penalizer=0.01)
            ctv_b.fit(boot_panel, id_col='id', event_col='event',
                      start_col='start', stop_col='stop',
                      formula=FORMULA_FULL, show_progress=False)

            cond = compute_conditional_hrs(ctv_b, boot_panel)
            for cy in all_hrs:
                all_hrs[cy].append(cond[cy]['hr'])
        except Exception:
            continue

        if (i + 1) % 100 == 0:
            print(f"    ... {i+1}/{n_boot} done ({len(all_hrs[10])} successful)")

    print(f"\n  Successfully completed {len(all_hrs[10])} / {n_boot} bootstrap iterations")
    print(f"\n  {'Year':>6s} {'Median HR':>10s} {'95% CI (bootstrap)':>25s}")
    print(f"  {'-'*6} {'-'*10} {'-'*25}")

    boot_results = {}
    for cy in [0, 5, 10, 15, 20]:
        hrs = np.array(all_hrs[cy])
        if len(hrs) == 0:
            continue
        med = np.median(hrs)
        lo = np.percentile(hrs, 2.5)
        hi = np.percentile(hrs, 97.5)
        print(f"  {cy:6d} {med:10.3f} [{lo:.3f}, {hi:.3f}]")
        boot_results[cy] = {'median': med, 'ci_lo': lo, 'ci_hi': hi,
                             'n_valid': len(hrs)}

    return boot_results


# ============================================================
# Summary
# ============================================================

def print_summary(perm_res, wald_res, binary_res, boot_res):
    print("\n\n" + "=" * 70)
    print("SUMMARY FOR PAPER INTEGRATION")
    print("=" * 70)

    print("\n[1] Permutation test:")
    print(f"    p_perm = {perm_res['perm_p']:.4f} (one-sided, {len(perm_res['null_betas'])} iterations)")
    print(f"    Observed beta = {perm_res['observed_beta']:.4f}")

    print("\n[2] Joint Wald test:")
    print(f"    Joint chi2(2) = {wald_res['joint_wald_stat']:.3f}, p = {wald_res['joint_wald_p']:.4f}")

    print("\n[3] Binary phase interaction (stability × I(yr >= 10)):")
    print(f"    Interaction HR = {binary_res['interaction_hr']:.3f}, p = {binary_res['interaction_p']:.4f}")
    print(f"    Pre-decade HR  = {binary_res['pre_hr']:.3f}, p = {binary_res['pre_p']:.4f}")
    print(f"    Post-decade HR = {binary_res['post_hr']:.3f} [{binary_res['post_lo']:.3f}, "
          f"{binary_res['post_hi']:.3f}], p = {binary_res['post_p']:.4f}")

    print("\n[4] Cluster bootstrap (conditional HRs):")
    for cy in [0, 5, 10, 15, 20]:
        if cy in boot_res:
            r = boot_res[cy]
            print(f"    Year {cy:2d}: median HR = {r['median']:.3f} "
                  f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "═" * 70)
    print("  INTERACTION TERM ROBUSTNESS ANALYSIS")
    print("═" * 70 + "\n")

    panel = build_base_panel()
    print(f"Panel: {len(panel)} person-years, {panel['artist_id'].nunique()} artists, "
          f"{int(panel['event'].sum())} events\n")

    # Run all four analyses
    perm_res = permutation_test(panel, n_perm=1000)
    wald_res = joint_wald_test(panel)
    binary_res = binary_phase_interaction(panel)
    boot_res = cluster_bootstrap(panel, n_boot=500)

    print_summary(perm_res, wald_res, binary_res, boot_res)

    print("\n" + "═" * 70)
    print("  ANALYSIS COMPLETE")
    print("═" * 70)

    return perm_res, wald_res, binary_res, boot_res


if __name__ == '__main__':
    results = main()
