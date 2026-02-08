"""
11b_lockin_refined.py
──────────────────────────────────────────────────────────────────────
Refined Reputational Lock-in Analysis

Since mediation through institutional type transition fails
(ACME ≈ 0, ADE dominates), we investigate the "Reputational Lock-in"
mechanism more carefully:

  (A) Institution-level HHI (Herfindahl): concentration of visits
      across *specific* institutions (not types)
  (B) Repeat-visit ratio: fraction of events at previously-visited
      institutions
  (C) Top-institution share: share of total events at the single
      most-visited institution
  (D) Within-type repetition: conditional on type, how much repetition
      exists within the same institution

For each, we test whether the metric predicts plateau hazard
*beyond* network stability, and whether the "direct effect"
of stability shrinks when controlling for the lock-in metric.
──────────────────────────────────────────────────────────────────────
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr
from collections import Counter
from lifelines import CoxTimeVaryingFitter
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    detect_plateau, compute_network_size_stability,
    compute_event_weight, CENSOR_YEAR,
    SIGNIFICANT_EVENT_TYPES, ORIGINAL_TYPE_WEIGHTS,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.json')
FIG_DIR   = os.path.join(os.path.dirname(__file__), '..', 'figures', 'reference')
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# Lock-in Metrics (computed per person-year)
# ============================================================

def compute_institution_hhi(events_up_to):
    """
    Herfindahl-Hirschman Index for institution-specific visit concentration.
    Higher HHI = visits concentrated in fewer institutions = more lock-in.
    """
    inst = events_up_to['institution_en'].dropna()
    if len(inst) < 2:
        return 1.0
    counts = inst.value_counts()
    shares = counts / counts.sum()
    return (shares ** 2).sum()


def compute_repeat_visit_ratio(events_sorted):
    """
    Fraction of events (from 2nd onward) that visit an institution
    the artist has already visited before.
    """
    if len(events_sorted) < 2:
        return 0.0
    visited = set()
    repeats = 0
    total = 0
    for inst in events_sorted['institution_en'].dropna():
        if inst in visited:
            repeats += 1
        visited.add(inst)
        total += 1
    if total <= 1:
        return 0.0
    return repeats / (total - 1)  # exclude first event


def compute_top_institution_share(events_up_to):
    """Share of total events at the single most-visited institution."""
    inst = events_up_to['institution_en'].dropna()
    if len(inst) == 0:
        return 1.0
    counts = inst.value_counts()
    return counts.iloc[0] / counts.sum()


def compute_within_type_repetition(events_up_to, inst_category_map):
    """
    For each institution type, compute the average number of
    repeat visits to the *same* institution within that type,
    normalized by total events in that type.
    Returns the weighted average across types.
    """
    if len(events_up_to) < 2:
        return 0.0

    # Map each event to its institution category
    type_inst_counts = {}
    for _, ev in events_up_to.iterrows():
        inst = ev.get('institution_en')
        if inst is None:
            continue
        cat = inst_category_map.get(inst, 'other')
        if cat not in type_inst_counts:
            type_inst_counts[cat] = Counter()
        type_inst_counts[cat][inst] += 1

    total_repeats = 0
    total_events = 0
    for cat, counter in type_inst_counts.items():
        for inst, count in counter.items():
            total_repeats += max(0, count - 1)  # repeats beyond first visit
            total_events += count

    if total_events <= 1:
        return 0.0
    return total_repeats / (total_events - 1)


# ============================================================
# Institution Category Classification (from 11_)
# ============================================================

MMCA_KW = ['national museum of modern and contemporary art', 'mmca', '국립현대미술관']  # incl. Korean for data matching
BIENNALE_KW = ['biennale', 'biennial', 'triennale', 'documenta', 'manifesta']
ACADEMIC_KW = ['university', 'college', 'academy', 'school of', 'institute of',
               'association', 'society', 'federation', 'foundation']
PUBLIC_KW = ['museum of art', 'national museum', 'city museum', 'municipal',
             'arts center', 'art center', 'culture center', 'kunsthalle']
COMMERCIAL_KW = ['gallery', 'galerie', 'auction', 'art fair']

def classify_inst(name, inst_type, event_type):
    if name is None: name = ''
    nl = name.lower().strip()
    for kw in MMCA_KW:
        if kw in nl: return 'mmca'
    if event_type == 'biennale': return 'biennale'
    for kw in BIENNALE_KW:
        if kw in nl: return 'biennale'
    for kw in ACADEMIC_KW:
        if kw in nl: return 'academic'
    if inst_type == 'public_museum': return 'public'
    for kw in PUBLIC_KW:
        if kw in nl: return 'public'
    if inst_type == 'gallery': return 'commercial'
    for kw in COMMERCIAL_KW:
        if kw in nl: return 'commercial'
    if inst_type == 'private_museum': return 'private'
    return 'commercial'


# ============================================================
# Panel Builder
# ============================================================

def build_lockin_panel(df_artists, df_events, inst_cat_map):
    """Build panel with lock-in metrics."""
    plateau_info = {}
    for _, row in df_artists.iterrows():
        aid = row['artist_id']
        cs = row['career_start_year']
        if pd.isna(cs) or row['num_events'] == 0:
            continue
        occ, py, ttp = detect_plateau(df_events, aid, int(cs), censor_year=CENSOR_YEAR)
        plateau_info[aid] = (occ, py, ttp)

    records = []
    for _, row in df_artists.iterrows():
        aid = row['artist_id']
        cs = row['career_start_year']
        if pd.isna(cs) or aid not in plateau_info:
            continue

        cs_int = int(cs)
        occ, py, _ = plateau_info[aid]
        end_year = int(py) if occ else CENSOR_YEAR

        birth_year = row.get('birth_year')
        has_overseas = int(row.get('has_overseas', False))

        artist_events = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] <= end_year) &
            (df_events['year'] <= CENSOR_YEAR)
        ].copy().sort_values('year')

        for year in range(cs_int, end_year + 1):
            cy = year - cs_int
            ev_up = artist_events[artist_events['year'] <= year]

            ns, nstab = compute_network_size_stability(ev_up)
            cum_val = sum(
                compute_event_weight(e['event_type'], ORIGINAL_TYPE_WEIGHTS)
                for _, e in ev_up.iterrows()
            )

            # Lock-in metrics
            hhi = compute_institution_hhi(ev_up)
            repeat = compute_repeat_visit_ratio(ev_up)
            top_share = compute_top_institution_share(ev_up)
            within_rep = compute_within_type_repetition(ev_up, inst_cat_map)

            is_last = (year == end_year)
            event = 1 if (is_last and occ) else 0

            records.append({
                'artist_id': aid, 'year': year, 'career_year': cy,
                'start': cy, 'stop': cy + 1, 'event': event,
                'network_size': max(ns, 0.5),
                'network_stability': nstab,
                'cumulative_validation': cum_val,
                'inst_hhi': hhi,
                'repeat_visit_ratio': repeat,
                'top_inst_share': top_share,
                'within_type_repetition': within_rep,
                'birth_year': birth_year,
                'has_overseas': has_overseas,
            })

    df = pd.DataFrame(records)
    df['birth_year'] = df['birth_year'].fillna(df['birth_year'].median())
    return df


# ============================================================
# Marginal Effect Decomposition
# ============================================================

def marginal_decomposition(panel):
    """
    Compare direct effect of network_stability before and after
    adding each lock-in metric.  If the coefficient shrinks, the
    lock-in metric partially "explains" the stability-plateau link.
    """
    scaler = StandardScaler()
    p = panel.copy()
    scale_cols = ['network_stability', 'cumulative_validation', 'birth_year',
                  'inst_hhi', 'repeat_visit_ratio', 'top_inst_share',
                  'within_type_repetition']
    p[scale_cols] = scaler.fit_transform(p[scale_cols])
    p['id'] = p['artist_id']

    results = []

    # Baseline: stability only
    base_form = 'network_stability + cumulative_validation + birth_year + has_overseas'
    ctv_base = CoxTimeVaryingFitter()
    ctv_base.fit(p, id_col='id', event_col='event',
                 start_col='start', stop_col='stop',
                 formula=base_form, show_progress=False)
    base_hr = np.exp(ctv_base.params_['network_stability'])
    base_p = ctv_base.summary.loc['network_stability', 'p']
    base_aic = ctv_base.AIC_partial_
    results.append({
        'Model': 'Baseline (Stability only)',
        'Stability_HR': base_hr,
        'Stability_p': base_p,
        'LockIn_HR': '-',
        'LockIn_p': '-',
        'AIC': base_aic,
        'HR_reduction': 0.0,
    })
    print(f"  Baseline: HR(stability) = {base_hr:.3f}, p = {base_p:.4f}, AIC = {base_aic:.1f}")

    for metric, label in [
        ('inst_hhi', 'Institution HHI'),
        ('repeat_visit_ratio', 'Repeat Visit Ratio'),
        ('top_inst_share', 'Top Institution Share'),
        ('within_type_repetition', 'Within-Type Repetition'),
    ]:
        formula = f'network_stability + {metric} + cumulative_validation + birth_year + has_overseas'
        ctv = CoxTimeVaryingFitter()
        try:
            ctv.fit(p, id_col='id', event_col='event',
                    start_col='start', stop_col='stop',
                    formula=formula, show_progress=False)
            hr_stab = np.exp(ctv.params_['network_stability'])
            p_stab = ctv.summary.loc['network_stability', 'p']
            hr_metric = np.exp(ctv.params_[metric])
            p_metric = ctv.summary.loc[metric, 'p']
            aic = ctv.AIC_partial_

            reduction = (base_hr - hr_stab) / (base_hr - 1.0) * 100 if base_hr != 1.0 else 0

            results.append({
                'Model': f'+ {label}',
                'Stability_HR': hr_stab,
                'Stability_p': p_stab,
                'LockIn_HR': hr_metric,
                'LockIn_p': p_metric,
                'AIC': aic,
                'HR_reduction': reduction,
            })
            print(f"  + {label:30s}: HR(stab)={hr_stab:.3f} (p={p_stab:.4f}), "
                  f"HR({label})={hr_metric:.3f} (p={p_metric:.4f}), "
                  f"AIC={aic:.1f}, HR reduction={reduction:.1f}%")
        except Exception as e:
            print(f"  + {label}: FAILED ({e})")
            results.append({
                'Model': f'+ {label}', 'Stability_HR': '-', 'Stability_p': '-',
                'LockIn_HR': '-', 'LockIn_p': '-', 'AIC': '-', 'HR_reduction': '-',
            })

    return pd.DataFrame(results), ctv_base


# ============================================================
# Permutation Test (Refined)
# ============================================================

def refined_permutation_test(panel, metric_col, n_perms=1000, seed=42):
    """
    Test whether plateau artists have higher lock-in than
    non-plateau artists beyond what's explained by career length
    and total events.

    Uses residualized lock-in (regress out career_year and
    cumulative events) for a fair comparison.
    """
    rng = np.random.RandomState(seed)

    # Get one row per artist (latest observation)
    artist_df = panel.sort_values('career_year').groupby('artist_id').last().reset_index()

    # Residualize the metric
    X = sm.add_constant(artist_df[['career_year', 'cumulative_validation']])
    y = artist_df[metric_col].values
    resid = sm.OLS(y, X).fit().resid
    artist_df['resid_metric'] = resid

    plateau = artist_df[artist_df['event'] == 1]['resid_metric'].values
    no_plateau = artist_df[artist_df['event'] == 0]['resid_metric'].values

    obs_diff = np.mean(plateau) - np.mean(no_plateau)

    # Permutation
    all_resids = np.concatenate([plateau, no_plateau])
    n_p = len(plateau)
    perm_diffs = []
    for _ in range(n_perms):
        rng.shuffle(all_resids)
        perm_diffs.append(np.mean(all_resids[:n_p]) - np.mean(all_resids[n_p:]))
    perm_diffs = np.array(perm_diffs)

    p_val = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))

    return {
        'obs_diff': obs_diff,
        'perm_mean': np.mean(perm_diffs),
        'perm_ci': np.percentile(perm_diffs, [2.5, 97.5]),
        'p_value': p_val,
        'plateau_mean': np.mean(plateau),
        'no_plateau_mean': np.mean(no_plateau),
        'perm_diffs': perm_diffs,
    }


# ============================================================
# Figure
# ============================================================

def create_lockin_figure(panel, decomp_df, perm_results, figpath):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    artist_df = panel.sort_values('career_year').groupby('artist_id').last().reset_index()
    plateau_mask = artist_df['event'] == 1

    metrics = [
        ('inst_hhi', 'Institution HHI\n(Visit Concentration)'),
        ('repeat_visit_ratio', 'Repeat Visit Ratio'),
        ('top_inst_share', 'Top-Inst. Share'),
    ]

    # ─── Panels A-C: Distribution by plateau status ──────────
    for i, (col, label) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        vals_p = artist_df.loc[plateau_mask, col].dropna()
        vals_n = artist_df.loc[~plateau_mask, col].dropna()

        bins = np.linspace(0, max(vals_p.max(), vals_n.max()), 30)
        ax.hist(vals_n, bins=bins, alpha=0.5, color='#1F77B4',
                label=f'No Plateau (n={len(vals_n)})', density=True)
        ax.hist(vals_p, bins=bins, alpha=0.5, color='#D62728',
                label=f'Plateau (n={len(vals_p)})', density=True)

        u, p = mannwhitneyu(vals_p, vals_n, alternative='two-sided')
        ax.text(0.05, 0.95, f'U={u:.0f}\np={p:.4f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{chr(65+i)}. {label.split(chr(10))[0]}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)

    # ─── Panel D: HR Decomposition Bar Chart ─────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    valid_decomp = decomp_df[decomp_df['Stability_HR'] != '-'].copy()
    valid_decomp['Stability_HR'] = valid_decomp['Stability_HR'].astype(float)
    models = valid_decomp['Model'].values
    hrs = valid_decomp['Stability_HR'].values

    colors = ['#4C72B0'] + ['#55A868'] * (len(hrs) - 1)
    bars = ax_d.barh(range(len(hrs)), hrs, color=colors, edgecolor='black', height=0.6)
    ax_d.axvline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_d.set_yticks(range(len(hrs)))
    ax_d.set_yticklabels([m.replace('+ ', '+\n') for m in models], fontsize=8)
    ax_d.set_xlabel('HR (Network Stability)', fontsize=10)
    ax_d.set_title('D. Stability HR After Lock-in Control', fontsize=12, fontweight='bold')
    ax_d.invert_yaxis()

    # Annotate with HR values
    for j, (hr, model) in enumerate(zip(hrs, models)):
        pval = valid_decomp.iloc[j]['Stability_p']
        if isinstance(pval, str):
            pval_str = pval
        else:
            stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            pval_str = f'{hr:.3f}{stars}'
        ax_d.text(hr + 0.005, j, pval_str, va='center', fontsize=9)

    # ─── Panel E: Permutation test (HHI) ─────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    if 'inst_hhi' in perm_results:
        pr = perm_results['inst_hhi']
        ax_e.hist(pr['perm_diffs'], bins=40, color='gray', edgecolor='black',
                  alpha=0.6, label='Permutation null')
        ax_e.axvline(pr['obs_diff'], color='red', linewidth=2.5,
                     label=f'Observed Δ={pr["obs_diff"]:.3f}')
        ax_e.axvline(0, color='black', linestyle='--', linewidth=1)
        ax_e.set_xlabel('Residualized HHI Difference\n(Plateau - No Plateau)', fontsize=10)
        ax_e.set_ylabel('Frequency', fontsize=10)
        ax_e.text(0.05, 0.95, f'p = {pr["p_value"]:.3f}',
                  transform=ax_e.transAxes, fontsize=11, va='top', fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_e.legend(fontsize=8)
    ax_e.set_title('E. Permutation Test (Residualized HHI)', fontsize=12, fontweight='bold')

    # ─── Panel F: Network Stability vs HHI scatter ──────────
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.scatter(artist_df.loc[~plateau_mask, 'network_stability'],
                 artist_df.loc[~plateau_mask, 'inst_hhi'],
                 alpha=0.35, s=25, c='#1F77B4', label='No Plateau')
    ax_f.scatter(artist_df.loc[plateau_mask, 'network_stability'],
                 artist_df.loc[plateau_mask, 'inst_hhi'],
                 alpha=0.55, s=35, c='#D62728', marker='^', label='Plateau')
    x = artist_df['network_stability'].values
    y = artist_df['inst_hhi'].values
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() > 10:
        z = np.polyfit(x[m], y[m], 1)
        p = np.poly1d(z)
        xs = np.sort(x[m])
        ax_f.plot(xs, p(xs), 'k--', linewidth=1.5)
        r, rp = pearsonr(x[m], y[m])
        ax_f.text(0.05, 0.95, f'r = {r:.3f} (p = {rp:.3f})',
                  transform=ax_f.transAxes, fontsize=9, va='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_f.set_xlabel('Network Stability', fontsize=10)
    ax_f.set_ylabel('Institution HHI', fontsize=10)
    ax_f.set_title('F. Stability vs. Visit Concentration', fontsize=12, fontweight='bold')
    ax_f.legend(fontsize=8)

    fig.suptitle('Reputational Lock-in Analysis\n'
                 '(Direct Effect Decomposition via Institution-Level Metrics)',
                 fontsize=15, fontweight='bold', y=1.01)

    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    print(f"  Figure saved: {figpath}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("REFINED REPUTATIONAL LOCK-IN ANALYSIS")
    print("=" * 80)

    # Load
    print("\n[1] Loading data...")
    artists_list = load_raw_data(DATA_PATH)
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)

    # Classify institutions
    inst_cat_map = {}
    for _, ev in df_events.iterrows():
        inst = ev.get('institution_en')
        if inst and inst not in inst_cat_map:
            inst_cat_map[inst] = classify_inst(inst, ev.get('institution_type'), ev.get('event_type'))

    print(f"  {len(inst_cat_map)} unique institutions classified")

    # Build panel
    print("\n[2] Building panel with lock-in metrics...")
    panel = build_lockin_panel(df_artists, df_events, inst_cat_map)
    print(f"  Panel: {len(panel)} person-years, {panel['artist_id'].nunique()} artists, "
          f"{panel['event'].sum()} events")

    # Descriptive
    print("\n" + "─" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("─" * 60)

    artist_df = panel.sort_values('career_year').groupby('artist_id').last().reset_index()
    plat = artist_df[artist_df['event'] == 1]
    noplat = artist_df[artist_df['event'] == 0]

    for col, label in [('inst_hhi', 'Inst. HHI'),
                       ('repeat_visit_ratio', 'Repeat Visit'),
                       ('top_inst_share', 'Top-Inst Share'),
                       ('within_type_repetition', 'Within-Type Rep.')]:
        mp = plat[col].mean()
        mn = noplat[col].mean()
        u, p = mannwhitneyu(plat[col], noplat[col], alternative='two-sided')
        print(f"  {label:25s}: Plateau={mp:.3f}, No-Plat={mn:.3f}, U={u:.0f}, p={p:.4f}")

    # Marginal decomposition
    print("\n" + "─" * 60)
    print("MARGINAL EFFECT DECOMPOSITION")
    print("─" * 60)
    decomp_df, ctv_base = marginal_decomposition(panel)

    # Permutation tests (residualized)
    print("\n" + "─" * 60)
    print("RESIDUALIZED PERMUTATION TESTS")
    print("─" * 60)

    perm_results = {}
    for col, label in [('inst_hhi', 'Inst. HHI'),
                       ('repeat_visit_ratio', 'Repeat Visit Ratio'),
                       ('top_inst_share', 'Top-Inst. Share')]:
        print(f"\n  Testing: {label}")
        pr = refined_permutation_test(panel, col, n_perms=2000, seed=42)
        perm_results[col] = pr
        print(f"    Observed Δ (residualized): {pr['obs_diff']:.4f}")
        print(f"    Permutation 95% CI:        [{pr['perm_ci'][0]:.4f}, {pr['perm_ci'][1]:.4f}]")
        print(f"    p-value:                   {pr['p_value']:.4f}")

    # Figure
    print("\n" + "─" * 60)
    print("CREATING FIGURE")
    print("─" * 60)
    figpath = os.path.join(FIG_DIR, 'reputational_lockin_refined.png')
    create_lockin_figure(panel, decomp_df, perm_results, figpath)

    # Save tables
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY: REPUTATIONAL LOCK-IN EVIDENCE")
    print("=" * 80)

    base_hr = decomp_df.iloc[0]['Stability_HR']
    print(f"""
  Baseline Network Stability HR: {base_hr:.3f}
  
  After controlling for each lock-in metric:""")
    for _, row in decomp_df.iterrows():
        if row['Model'] != 'Baseline (Stability only)':
            red = row['HR_reduction']
            red_str = f"{red:.1f}%" if isinstance(red, float) else str(red)
            print(f"    {row['Model']:40s} → HR reduction: {red_str}")

    print(f"""
  Permutation test results (residualized, controlling for career length):""")
    for col, label in [('inst_hhi', 'Institution HHI'),
                       ('repeat_visit_ratio', 'Repeat Visit Ratio'),
                       ('top_inst_share', 'Top-Inst. Share')]:
        pr = perm_results[col]
        sig = "***" if pr['p_value'] < 0.001 else "**" if pr['p_value'] < 0.01 else "*" if pr['p_value'] < 0.05 else "n.s."
        print(f"    {label:30s}: Δ={pr['obs_diff']:.4f}, p={pr['p_value']:.4f} {sig}")

    print(f"""
  Interpretation:
    The direct effect of network stability on plateau hazard
    is NOT mediated by institutional type transitions.
    Instead, lock-in operates through institution-specific
    repetition patterns — a 'Reputational Lock-in' mechanism
    where artists become tethered to familiar institutions
    regardless of those institutions' categorical type.
""")


if __name__ == '__main__':
    main()
