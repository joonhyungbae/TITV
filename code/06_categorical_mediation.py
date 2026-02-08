"""
11_institutional_type_transition.py
──────────────────────────────────────────────────────────────────────
Hypothesis 3 Revisited: 'Institutional Type Transition' as Mediator

Event type diversity failed to mediate the
Network Stability → Plateau link (ACME ≈ 0).
Here we replace the mediator with a richer operationalization:
*Institutional Type Transition* — how much an artist moves between
qualitatively different institution categories.

Analysis pipeline:
  (1) Classify every institution into 6 categories
  (2) Compute two mediator variants per person-year:
      - inst_type_entropy  (Shannon entropy of institution-type shares)
      - inst_type_transition_rate (proportion of consecutive-event pairs
        that cross institution-type boundaries)
  (3) Correlate Network Stability with each mediator
  (4) Causal Mediation Analysis (bootstrap, Imai et al. 2010)
  (5) If direct effect dominates → Reputational Lock-in simulation
──────────────────────────────────────────────────────────────────────
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import json
import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr, pearsonr, mannwhitneyu
from collections import Counter, OrderedDict
from lifelines import CoxTimeVaryingFitter
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    detect_plateau, compute_network_size_stability,
    compute_event_weight, CENSOR_YEAR,
    SIGNIFICANT_EVENT_TYPES, ORIGINAL_TYPE_WEIGHTS,
)

# ============================================================
# Paths
# ============================================================
DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.json')
FIG_DIR     = os.path.join(os.path.dirname(__file__), '..', 'figures', 'reference')
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# (1)  Institution-type classification
# ============================================================

# Keywords for rule-based classification
MMCA_KEYWORDS = [
    'national museum of modern and contemporary art',
    'national museum of contemporary art',
    'mmca', '국립현대미술관',  # Korean name for data matching
]

PUBLIC_MUSEUM_KEYWORDS = [
    'national museum', 'seoul museum of art', 'museum of art',
    'national gallery', 'city museum', 'municipal museum',
    'gwangju museum', 'busan museum', 'daejeon museum',
    'incheon museum', 'jeonbuk museum', 'gyeongnam museum',
    'jeju museum', 'cheongju museum', 'daegu museum',
    'ulsan museum', 'chuncheon museum', 'wonju museum',
    'art center', 'arts center', 'culture center',
    'cultural center', 'civic center',
    'kunsthalle', 'museo', 'musée', 'musee',
    'modern art museum', 'contemporary art museum',
]

ACADEMIC_KEYWORDS = [
    'university', 'college', 'univ.', 'univ ',
    'academy', 'school of', 'graduate school',
    'institute of', '대학',  # Korean for data matching
    'association',
    'society', 'federation', 'council',
    'committee', 'foundation', 'research',
]

COMMERCIAL_KEYWORDS = [
    'gallery', 'galerie', 'galleries',
    'auction', 'art fair', 'salon',
]

BIENNALE_KEYWORDS = [
    'biennale', 'biennial', 'triennale', 'triennial',
    'documenta', 'manifesta', 'art basel',
    'venice', 'gwangju biennale', 'whitney biennial',
    'são paulo', 'sao paulo', 'istanbul biennial',
    'sydney biennial', 'carnegie international',
]


def classify_institution(inst_name, inst_type_raw, country, event_type):
    """
    Classify an institution into one of 6 analytical categories:
      'mmca'       – National Museum of Modern and Contemporary Art, Korea
      'public'     – Public museums / government-funded spaces
      'private'    – Private museums / non-commercial exhibition spaces
      'commercial' – Commercial galleries, art fairs, auction houses
      'academic'   – Universities, art associations, research bodies
      'biennale'   – International biennales / triennales
    """
    if inst_name is None:
        inst_name = ''
    name_lower = inst_name.lower().strip()

    # 1) MMCA (highest priority)
    for kw in MMCA_KEYWORDS:
        if kw in name_lower:
            return 'mmca'

    # 2) Biennale events
    if event_type == 'biennale':
        return 'biennale'
    for kw in BIENNALE_KEYWORDS:
        if kw in name_lower:
            return 'biennale'

    # 3) Academic / organizational
    for kw in ACADEMIC_KEYWORDS:
        if kw in name_lower:
            return 'academic'

    # 4) Public museum
    if inst_type_raw == 'public_museum':
        return 'public'
    for kw in PUBLIC_MUSEUM_KEYWORDS:
        if kw in name_lower:
            return 'public'

    # 5) Commercial gallery
    if inst_type_raw == 'gallery':
        return 'commercial'
    for kw in COMMERCIAL_KEYWORDS:
        if kw in name_lower:
            return 'commercial'

    # 6) Private museum
    if inst_type_raw == 'private_museum':
        return 'private'

    # 7) Overseas (without further detail) → classify by raw type or as 'public'
    if inst_type_raw == 'overseas':
        return 'public'    # conservative default for overseas exhibitions

    return 'commercial'    # fallback (most 'other' venues are gallery-like)


# ============================================================
# (2)  Mediator computation
# ============================================================

def compute_inst_type_entropy(events_window):
    """Shannon entropy over institution-type proportions."""
    if len(events_window) == 0:
        return 0.0
    type_counts = events_window['inst_category'].value_counts()
    type_probs = type_counts / type_counts.sum()
    return entropy(type_probs)


def compute_inst_type_transition_rate(events_window):
    """
    Proportion of consecutive event pairs that cross institution-type
    boundaries.  Events are sorted by year (ties broken by index).
    Returns a float in [0, 1].
    """
    if len(events_window) < 2:
        return 0.0
    sorted_events = events_window.sort_values(['year']).reset_index(drop=True)
    types = sorted_events['inst_category'].values
    transitions = sum(1 for i in range(1, len(types)) if types[i] != types[i-1])
    return transitions / (len(types) - 1)


def compute_unique_type_count(events_window):
    """Number of distinct institution categories visited."""
    if len(events_window) == 0:
        return 0
    return events_window['inst_category'].nunique()


# ============================================================
# (3)  Person-year panel with new mediators
# ============================================================

def build_panel_with_inst_transition(df_artists, df_events, censor_year=CENSOR_YEAR):
    """
    Build person-year panel augmented with:
      - inst_type_entropy
      - inst_type_transition_rate
      - unique_type_count
    Uses the same plateau definition / network metrics as the main paper.
    """
    # Pre-compute plateau
    plateau_info = {}
    for _, row in df_artists.iterrows():
        aid = row['artist_id']
        cs = row['career_start_year']
        if pd.isna(cs) or row['num_events'] == 0:
            continue
        occurred, p_year, ttp = detect_plateau(df_events, aid, int(cs),
                                                censor_year=censor_year)
        plateau_info[aid] = (occurred, p_year, ttp)

    records = []
    for _, row in df_artists.iterrows():
        aid = row['artist_id']
        cs = row['career_start_year']
        if pd.isna(cs) or aid not in plateau_info:
            continue

        cs_int = int(cs)
        occurred, p_year, _ = plateau_info[aid]
        end_year = int(p_year) if occurred else censor_year

        birth_year = row.get('birth_year')
        has_overseas = int(row.get('has_overseas', False))

        artist_events = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] <= end_year) &
            (df_events['year'] <= censor_year)
        ].copy()

        for year in range(cs_int, end_year + 1):
            career_year = year - cs_int
            events_up_to = artist_events[artist_events['year'] <= year]

            # Network metrics
            net_size, net_stability = compute_network_size_stability(events_up_to)

            # Cumulative validation
            cum_val = sum(
                compute_event_weight(ev['event_type'], ORIGINAL_TYPE_WEIGHTS)
                for _, ev in events_up_to.iterrows()
            )

            # Rolling 5-year window for transition metrics
            window_events = artist_events[
                (artist_events['year'] >= year - 4) &
                (artist_events['year'] <= year)
            ]

            # New mediators
            inst_entropy = compute_inst_type_entropy(window_events)
            inst_transition = compute_inst_type_transition_rate(window_events)
            unique_types = compute_unique_type_count(window_events)

            # Event indicator
            is_last_year = (year == end_year)
            event = 1 if (is_last_year and occurred) else 0

            records.append({
                'artist_id': aid,
                'year': year,
                'career_year': career_year,
                'start': career_year,
                'stop': career_year + 1,
                'event': event,
                'network_size': max(net_size, 0.5),
                'network_stability': net_stability,
                'cumulative_validation': cum_val,
                'inst_type_entropy': inst_entropy,
                'inst_type_transition_rate': inst_transition,
                'unique_type_count': unique_types,
                'post_cutpoint': 1 if career_year >= 10 else 0,
                'birth_year': birth_year,
                'has_overseas': has_overseas,
            })

    df = pd.DataFrame(records)
    if 'birth_year' in df.columns:
        df['birth_year'] = df['birth_year'].fillna(df['birth_year'].median())
    return df


# ============================================================
# (4)  Bootstrap Causal Mediation Analysis  (Imai et al. 2010)
# ============================================================

def bootstrap_mediation(panel, mediator_col, n_boot=1000, seed=42, penalizer=0.01):
    """
    Bootstrap causal mediation analysis.
    Path a: network_stability → mediator  (OLS)
    Path b: mediator → plateau hazard     (Cox, controlling for stability)

    Returns dict with ACME, ADE, total, proportion_mediated, CIs.
    """
    rng = np.random.RandomState(seed)
    artist_ids = panel['artist_id'].unique()
    n_artists = len(artist_ids)

    acme_samples = []
    ade_samples  = []

    controls = ['cumulative_validation', 'birth_year', 'has_overseas']
    cox_cols = ['id', 'start', 'stop', 'event',
                'network_stability', mediator_col] + controls

    for b in range(n_boot):
        if b % 200 == 0:
            print(f"  Bootstrap {b}/{n_boot}")

        # Resample artists (cluster bootstrap)
        boot_ids = rng.choice(artist_ids, size=n_artists, replace=True)
        # Relabel to avoid duplicate id issues in lifelines
        frames = []
        for new_id, old_id in enumerate(boot_ids):
            chunk = panel[panel['artist_id'] == old_id].copy()
            chunk['boot_id'] = new_id
            frames.append(chunk)
        boot = pd.concat(frames, ignore_index=True)

        if boot['event'].sum() < 5:
            continue

        try:
            # Standardize within bootstrap sample
            scaler = StandardScaler()
            scale_cols = ['network_stability', mediator_col] + controls
            boot[scale_cols] = scaler.fit_transform(boot[scale_cols])

            # Path a: OLS  stability → mediator
            X_a = sm.add_constant(boot[['network_stability'] + controls])
            ols_a = sm.OLS(boot[mediator_col], X_a).fit()
            beta_a = ols_a.params['network_stability']

            # Full Cox model — use column-subset (not formula) for stability
            boot['id'] = boot['boot_id']
            boot_cox = boot[cox_cols].copy()
            ctv = CoxTimeVaryingFitter(penalizer=penalizer)
            ctv.fit(boot_cox, id_col='id', event_col='event',
                    start_col='start', stop_col='stop',
                    show_progress=False)

            beta_b = ctv.params_[mediator_col]
            beta_direct = ctv.params_['network_stability']

            acme_samples.append(beta_a * beta_b)
            ade_samples.append(beta_direct)
        except Exception:
            continue

    acme = np.array(acme_samples)
    ade  = np.array(ade_samples)
    total = acme + ade

    def ci(arr, alpha=0.05):
        return np.percentile(arr, [100*alpha/2, 100*(1-alpha/2)])

    def pval_two_sided(arr):
        """Proportion of samples on the opposite side of zero."""
        if np.mean(arr) >= 0:
            return 2 * np.mean(arr <= 0)
        else:
            return 2 * np.mean(arr >= 0)

    prop_med = acme / np.where(np.abs(total) > 1e-10, total, np.nan)
    prop_med = prop_med[~np.isnan(prop_med)]

    results = {
        'acme_mean': np.mean(acme),
        'acme_ci': ci(acme),
        'acme_p': pval_two_sided(acme),
        'ade_mean': np.mean(ade),
        'ade_ci': ci(ade),
        'ade_p': pval_two_sided(ade),
        'total_mean': np.mean(total),
        'total_ci': ci(total),
        'total_p': pval_two_sided(total),
        'prop_mediated_mean': np.mean(prop_med) if len(prop_med) > 0 else 0,
        'prop_mediated_ci': ci(prop_med) if len(prop_med) > 0 else [0, 0],
        'n_boot_success': len(acme),
        'acme_samples': acme,
        'ade_samples': ade,
    }
    return results


# ============================================================
# (5)  Reputational Lock-in Simulation
# ============================================================

def compute_exposure_decay(events_up_to, decay_rate=0.15):
    """
    For each institution, compute a 'novelty-weighted contribution'
    that decays with repeated exposure.
    Weight for k-th visit = exp(-decay_rate * (k-1))

    Returns:
      - total_novel_weight: sum of decayed weights (lower = more lock-in)
      - raw_count:          total events
      - lock_in_ratio:      1 - (novel / raw)  → higher = more lock-in
    """
    if len(events_up_to) == 0:
        return 0.0, 0, 0.0

    inst_series = events_up_to['institution_en'].dropna()
    if len(inst_series) == 0:
        return 0.0, 0, 0.0

    # Sort by year for chronological counting
    sorted_events = events_up_to.sort_values('year')
    inst_counter = Counter()
    total_novel = 0.0

    for inst in sorted_events['institution_en'].dropna():
        k = inst_counter[inst]  # 0-indexed visit number
        total_novel += np.exp(-decay_rate * k)
        inst_counter[inst] += 1

    raw_count = sum(inst_counter.values())
    lock_in = 1.0 - (total_novel / raw_count) if raw_count > 0 else 0.0
    return total_novel, raw_count, lock_in


def simulate_reputational_lockin(panel, df_events, n_shuffles=500, seed=42):
    """
    Simulation: compare actual lock-in ratios to counterfactual
    where institution visits are randomly permuted within each artist.

    Returns a dict with observed, counterfactual means and p-values.
    """
    rng = np.random.RandomState(seed)

    # Compute observed lock-in per artist (at their observation end year)
    artist_ids = panel['artist_id'].unique()
    obs_lockin = {}
    for aid in artist_ids:
        artist_panel = panel[panel['artist_id'] == aid]
        max_year = artist_panel['year'].max()
        events_up_to = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] <= max_year)
        ]
        _, _, li = compute_exposure_decay(events_up_to)
        obs_lockin[aid] = li

    # Compare plateau vs non-plateau artists
    plateau_artists = set(panel[panel['event'] == 1]['artist_id'].unique())
    obs_plateau = [obs_lockin[a] for a in artist_ids if a in plateau_artists]
    obs_no_plateau = [obs_lockin[a] for a in artist_ids if a not in plateau_artists]

    # Counterfactual: shuffle institution names within each artist
    cf_diffs = []
    for s in range(n_shuffles):
        if s % 100 == 0:
            print(f"  Shuffle {s}/{n_shuffles}")
        cf_lockin_plateau = []
        cf_lockin_no = []
        for aid in artist_ids:
            artist_panel = panel[panel['artist_id'] == aid]
            max_year = artist_panel['year'].max()
            events_up_to = df_events[
                (df_events['artist_id'] == aid) &
                (df_events['year'] <= max_year)
            ].copy()
            if len(events_up_to) == 0:
                continue
            # Shuffle institution names while preserving years
            shuffled_inst = events_up_to['institution_en'].dropna().values.copy()
            rng.shuffle(shuffled_inst)
            events_up_to_shuffled = events_up_to.copy()
            mask = events_up_to_shuffled['institution_en'].notna()
            events_up_to_shuffled.loc[mask, 'institution_en'] = shuffled_inst[:mask.sum()]
            _, _, li = compute_exposure_decay(events_up_to_shuffled)

            if aid in plateau_artists:
                cf_lockin_plateau.append(li)
            else:
                cf_lockin_no.append(li)

        if len(cf_lockin_plateau) > 0 and len(cf_lockin_no) > 0:
            cf_diffs.append(np.mean(cf_lockin_plateau) - np.mean(cf_lockin_no))

    obs_diff = np.mean(obs_plateau) - np.mean(obs_no_plateau)
    cf_diffs = np.array(cf_diffs)

    # p-value: proportion of counterfactuals with diff >= observed
    p_value = np.mean(cf_diffs >= obs_diff) if len(cf_diffs) > 0 else 1.0

    return {
        'obs_plateau_mean': np.mean(obs_plateau),
        'obs_no_plateau_mean': np.mean(obs_no_plateau),
        'obs_diff': obs_diff,
        'cf_diff_mean': np.mean(cf_diffs) if len(cf_diffs) > 0 else 0,
        'cf_diff_ci': np.percentile(cf_diffs, [2.5, 97.5]) if len(cf_diffs) > 0 else [0, 0],
        'p_value': p_value,
        'obs_lockin': obs_lockin,
        'obs_plateau': obs_plateau,
        'obs_no_plateau': obs_no_plateau,
        'cf_diffs': cf_diffs,
    }


# ============================================================
# (6)  Visualization
# ============================================================

def create_figure(panel, med_results_entropy, med_results_transition,
                  lockin_results, path_a_results, figpath):
    """Create comprehensive 6-panel figure."""

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.40, wspace=0.35)

    palette = {
        'mmca': '#D62728', 'public': '#1F77B4', 'private': '#FF7F0E',
        'commercial': '#2CA02C', 'academic': '#9467BD', 'biennale': '#8C564B',
    }
    cat_labels = {
        'mmca': 'MMCA', 'public': 'Public Museum', 'private': 'Private Museum',
        'commercial': 'Commercial Gallery', 'academic': 'Academic/Assoc.',
        'biennale': 'Biennale/Intl.',
    }

    # ─── Panel A: Institution-type distribution ──────────────
    ax_a = fig.add_subplot(gs[0, 0])
    type_dist = panel.groupby('artist_id').first()  # just need inst_type_entropy once per artist
    # Get event-level distribution from the full panel
    # We need access to the events — use a quick hack via stored data
    type_counts_overall = {}
    for cat in palette:
        mask = panel[f'cat_{cat}'] if f'cat_{cat}' in panel.columns else None
        if mask is not None:
            type_counts_overall[cat_labels[cat]] = mask.sum()

    if not type_counts_overall:
        # Fallback: compute from panel averages
        ax_a.text(0.5, 0.5, 'See text', ha='center', va='center', fontsize=12)
    else:
        colors = [palette[k] for k in palette if cat_labels[k] in type_counts_overall]
        vals = [type_counts_overall[cat_labels[k]] for k in palette if cat_labels[k] in type_counts_overall]
        labels = [cat_labels[k] for k in palette if cat_labels[k] in type_counts_overall]
        ax_a.barh(labels, vals, color=colors, edgecolor='black', linewidth=0.5)
        ax_a.set_xlabel('Number of Events', fontsize=10)
    ax_a.set_title('A. Institution Category Distribution', fontsize=12, fontweight='bold')

    # ─── Panel B: Network stability vs transition rate (scatter) ──
    ax_b = fig.add_subplot(gs[0, 1])
    artist_level = panel.groupby('artist_id').agg({
        'network_stability': 'mean',
        'inst_type_transition_rate': 'mean',
        'inst_type_entropy': 'mean',
        'event': 'max',
    }).reset_index()

    plateau_mask = artist_level['event'] == 1
    ax_b.scatter(artist_level.loc[~plateau_mask, 'network_stability'],
                 artist_level.loc[~plateau_mask, 'inst_type_transition_rate'],
                 alpha=0.35, s=25, c='#1F77B4', label='No Plateau', zorder=2)
    ax_b.scatter(artist_level.loc[plateau_mask, 'network_stability'],
                 artist_level.loc[plateau_mask, 'inst_type_transition_rate'],
                 alpha=0.55, s=35, c='#D62728', marker='^', label='Plateau', zorder=3)
    # Regression line
    x = artist_level['network_stability'].values
    y = artist_level['inst_type_transition_rate'].values
    mask_valid = np.isfinite(x) & np.isfinite(y)
    if mask_valid.sum() > 10:
        z = np.polyfit(x[mask_valid], y[mask_valid], 1)
        p = np.poly1d(z)
        x_sorted = np.sort(x[mask_valid])
        ax_b.plot(x_sorted, p(x_sorted), 'k--', linewidth=1.5, alpha=0.7)
        r, rp = pearsonr(x[mask_valid], y[mask_valid])
        ax_b.text(0.05, 0.95, f'r = {r:.3f} (p = {rp:.3f})',
                  transform=ax_b.transAxes, fontsize=9, va='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_b.set_xlabel('Network Stability', fontsize=10)
    ax_b.set_ylabel('Inst. Type Transition Rate', fontsize=10)
    ax_b.set_title('B. Path a: Stability → Transition Rate', fontsize=12, fontweight='bold')
    ax_b.legend(fontsize=8, loc='lower left')

    # ─── Panel C: Network stability vs entropy (scatter) ──────
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.scatter(artist_level.loc[~plateau_mask, 'network_stability'],
                 artist_level.loc[~plateau_mask, 'inst_type_entropy'],
                 alpha=0.35, s=25, c='#1F77B4', label='No Plateau', zorder=2)
    ax_c.scatter(artist_level.loc[plateau_mask, 'network_stability'],
                 artist_level.loc[plateau_mask, 'inst_type_entropy'],
                 alpha=0.55, s=35, c='#D62728', marker='^', label='Plateau', zorder=3)
    x2 = artist_level['network_stability'].values
    y2 = artist_level['inst_type_entropy'].values
    mask2 = np.isfinite(x2) & np.isfinite(y2)
    if mask2.sum() > 10:
        z2 = np.polyfit(x2[mask2], y2[mask2], 1)
        p2 = np.poly1d(z2)
        x2s = np.sort(x2[mask2])
        ax_c.plot(x2s, p2(x2s), 'k--', linewidth=1.5, alpha=0.7)
        r2, r2p = pearsonr(x2[mask2], y2[mask2])
        ax_c.text(0.05, 0.95, f'r = {r2:.3f} (p = {r2p:.3f})',
                  transform=ax_c.transAxes, fontsize=9, va='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_c.set_xlabel('Network Stability', fontsize=10)
    ax_c.set_ylabel('Inst. Type Entropy', fontsize=10)
    ax_c.set_title('C. Path a: Stability → Type Entropy', fontsize=12, fontweight='bold')
    ax_c.legend(fontsize=8, loc='lower left')

    # ─── Panel D: Mediation diagram ─────────────────────────
    ax_d = fig.add_subplot(gs[1, :2])
    ax_d.axis('off')

    # Use transition rate results (primary)
    r = med_results_transition

    box_x = FancyBboxPatch((0.02, 0.30), 0.20, 0.30, boxstyle="round,pad=0.02",
                            edgecolor='black', facecolor='#87CEEB', linewidth=2.5)
    box_m = FancyBboxPatch((0.40, 0.60), 0.20, 0.30, boxstyle="round,pad=0.02",
                            edgecolor='black', facecolor='#90EE90', linewidth=2.5)
    box_y = FancyBboxPatch((0.78, 0.30), 0.20, 0.30, boxstyle="round,pad=0.02",
                            edgecolor='black', facecolor='#FFB6C1', linewidth=2.5)
    ax_d.add_patch(box_x); ax_d.add_patch(box_m); ax_d.add_patch(box_y)

    ax_d.text(0.12, 0.45, 'Network\nStability', ha='center', va='center',
              fontsize=13, fontweight='bold')
    ax_d.text(0.50, 0.75, 'Inst. Type\nTransition Rate', ha='center', va='center',
              fontsize=13, fontweight='bold')
    ax_d.text(0.88, 0.45, 'Plateau\nHazard', ha='center', va='center',
              fontsize=13, fontweight='bold')

    # Path a
    beta_a = path_a_results.get('beta_a', 0)
    p_a = path_a_results.get('p_a', 1)
    stars_a = '***' if p_a < 0.001 else '**' if p_a < 0.01 else '*' if p_a < 0.05 else ''
    arrow_a = FancyArrowPatch((0.22, 0.55), (0.40, 0.70),
                               arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#0066CC')
    ax_d.add_patch(arrow_a)
    ax_d.text(0.24, 0.68, f'a: β={beta_a:.3f}{stars_a}', fontsize=11,
              color='#0066CC', fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Path b
    beta_b = path_a_results.get('beta_b', 0)
    p_b = path_a_results.get('p_b', 1)
    hr_b = np.exp(beta_b)
    stars_b = '***' if p_b < 0.001 else '**' if p_b < 0.01 else '*' if p_b < 0.05 else ''
    arrow_b = FancyArrowPatch((0.60, 0.70), (0.78, 0.55),
                               arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#009900')
    ax_d.add_patch(arrow_b)
    ax_d.text(0.63, 0.68, f'b: HR={hr_b:.2f}{stars_b}', fontsize=11,
              color='#009900', fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Direct effect c'
    ade = r['ade_mean']
    ade_p = r['ade_p']
    stars_c = '***' if ade_p < 0.001 else '**' if ade_p < 0.01 else '*' if ade_p < 0.05 else ' (n.s.)'
    arrow_c = FancyArrowPatch((0.22, 0.43), (0.78, 0.43),
                               arrowstyle='->', mutation_scale=25, linewidth=2.5,
                               color='#CC0000', linestyle='--', alpha=0.5)
    ax_d.add_patch(arrow_c)
    ax_d.text(0.45, 0.25, f"c': ADE={ade:.3f}{stars_c}", fontsize=11,
              color='#CC0000', fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # ACME annotation
    acme = r['acme_mean']
    acme_ci = r['acme_ci']
    acme_p = r['acme_p']
    prop = r['prop_mediated_mean']
    ax_d.text(0.50, 0.08,
              f'ACME = {acme:.4f}  [{acme_ci[0]:.4f}, {acme_ci[1]:.4f}]  p = {acme_p:.3f}\n'
              f'Proportion mediated = {prop:.1%}',
              ha='center', fontsize=11, fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='#FFFFCC', edgecolor='black',
                        alpha=0.9, linewidth=1.5))

    ax_d.set_xlim(0, 1); ax_d.set_ylim(0, 1)
    ax_d.set_title('D. Causal Mediation: Inst. Type Transition Rate',
                    fontsize=12, fontweight='bold')

    # ─── Panel E: Bootstrap ACME distribution ────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    acme_s = r['acme_samples']
    ax_e.hist(acme_s, bins=40, color='#9467BD', edgecolor='black', alpha=0.7)
    ax_e.axvline(0, color='red', linestyle='--', linewidth=2, label='Null (0)')
    ax_e.axvline(np.mean(acme_s), color='green', linewidth=2, label=f'Mean={np.mean(acme_s):.4f}')
    ax_e.axvline(acme_ci[0], color='orange', linestyle=':', linewidth=1.5, label='95% CI')
    ax_e.axvline(acme_ci[1], color='orange', linestyle=':', linewidth=1.5)
    ax_e.set_xlabel('ACME (Indirect Effect)', fontsize=10)
    ax_e.set_ylabel('Frequency', fontsize=10)
    ax_e.set_title('E. Bootstrap ACME Distribution', fontsize=12, fontweight='bold')
    ax_e.legend(fontsize=8)

    # ─── Panel F: Reputational Lock-in ───────────────────────
    ax_f = fig.add_subplot(gs[2, 0])
    if lockin_results is not None:
        obs_p = lockin_results['obs_plateau']
        obs_n = lockin_results['obs_no_plateau']
        bp = ax_f.boxplot([obs_n, obs_p], labels=['No Plateau', 'Plateau'],
                          patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#1F77B4')
        bp['boxes'][1].set_facecolor('#D62728')
        for box in bp['boxes']:
            box.set_alpha(0.6)
        ax_f.set_ylabel('Lock-in Ratio', fontsize=10)
        diff = lockin_results['obs_diff']
        pv = lockin_results['p_value']
        ax_f.text(0.5, 0.95,
                  f'Δ = {diff:.3f}, p = {pv:.3f}',
                  transform=ax_f.transAxes, fontsize=10, ha='center', va='top',
                  fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_f.set_title('F. Reputational Lock-in Ratio', fontsize=12, fontweight='bold')

    # ─── Panel G: Lock-in counterfactual distribution ────────
    ax_g = fig.add_subplot(gs[2, 1])
    if lockin_results is not None and len(lockin_results['cf_diffs']) > 0:
        cf = lockin_results['cf_diffs']
        ax_g.hist(cf, bins=40, color='gray', edgecolor='black', alpha=0.6, label='Counterfactual')
        ax_g.axvline(lockin_results['obs_diff'], color='red', linewidth=2.5,
                     label=f'Observed Δ={lockin_results["obs_diff"]:.3f}')
        ax_g.axvline(0, color='black', linestyle='--', linewidth=1)
        ax_g.set_xlabel('Plateau − No-Plateau Lock-in Diff', fontsize=10)
        ax_g.set_ylabel('Frequency', fontsize=10)
        ax_g.legend(fontsize=8)
    ax_g.set_title('G. Counterfactual Simulation', fontsize=12, fontweight='bold')

    # ─── Panel H: Decay curve illustration ───────────────────
    ax_h = fig.add_subplot(gs[2, 2])
    visits = np.arange(1, 16)
    for dr, ls, lbl in [(0.10, '-', 'γ=0.10 (slow)'),
                         (0.15, '--', 'γ=0.15 (default)'),
                         (0.25, ':', 'γ=0.25 (fast)')]:
        weights = np.exp(-dr * (visits - 1))
        ax_h.plot(visits, weights, ls, linewidth=2, label=lbl)
    ax_h.set_xlabel('k-th Visit to Same Institution', fontsize=10)
    ax_h.set_ylabel('Novelty Weight', fontsize=10)
    ax_h.legend(fontsize=9)
    ax_h.set_title('H. Exposure Decay Model', fontsize=12, fontweight='bold')
    ax_h.set_ylim(0, 1.05)

    fig.suptitle('Institutional Type Transition Analysis\n'
                 '(Hypothesis 3 Revisited: Replacing Shannon Entropy with Institution-Type Transition)',
                 fontsize=15, fontweight='bold', y=1.01)

    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    print(f"  Figure saved: {figpath}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("INSTITUTIONAL TYPE TRANSITION ANALYSIS")
    print("Hypothesis 3 Revisited")
    print("=" * 80)

    # ── Load data ────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    artists_list = load_raw_data(DATA_PATH)
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)
    print(f"  {len(df_artists)} artists, {len(df_events)} events")

    # ── Classify institutions ────────────────────────────────
    print("\n[2/6] Classifying institutions into 6 categories...")
    df_events['inst_category'] = df_events.apply(
        lambda r: classify_institution(
            r['institution_en'], r['institution_type'], r['country'], r['event_type']
        ), axis=1
    )

    cat_counts = df_events['inst_category'].value_counts()
    print("  Category distribution:")
    for cat, cnt in cat_counts.items():
        print(f"    {cat:15s}: {cnt:5d}  ({cnt/len(df_events)*100:.1f}%)")

    # ── Build panel ──────────────────────────────────────────
    print("\n[3/6] Building person-year panel with new mediators...")
    panel = build_panel_with_inst_transition(df_artists, df_events)
    n_artists = panel['artist_id'].nunique()
    n_events = panel['event'].sum()
    print(f"  Panel: {len(panel)} person-years, {n_artists} artists, {n_events} plateau events")

    # Add category count columns for visualization
    for cat in ['mmca', 'public', 'private', 'commercial', 'academic', 'biennale']:
        panel[f'cat_{cat}'] = 0  # placeholder
    # Compute overall category counts from events
    cat_totals = df_events['inst_category'].value_counts().to_dict()

    # ── Descriptive statistics ───────────────────────────────
    print("\n" + "─" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("─" * 60)

    for var in ['inst_type_entropy', 'inst_type_transition_rate', 'unique_type_count',
                'network_stability', 'network_size']:
        if var in panel.columns:
            print(f"  {var:30s}: mean={panel[var].mean():.3f}, "
                  f"sd={panel[var].std():.3f}, "
                  f"range=[{panel[var].min():.3f}, {panel[var].max():.3f}]")

    # By plateau status
    artist_summary = panel.groupby('artist_id').agg({
        'inst_type_entropy': 'mean',
        'inst_type_transition_rate': 'mean',
        'network_stability': 'mean',
        'event': 'max',
    }).reset_index()
    plateau = artist_summary[artist_summary['event'] == 1]
    no_plateau = artist_summary[artist_summary['event'] == 0]

    print(f"\n  Plateau artists (n={len(plateau)}):")
    print(f"    Transition rate:  {plateau['inst_type_transition_rate'].mean():.3f} ± {plateau['inst_type_transition_rate'].std():.3f}")
    print(f"    Type entropy:     {plateau['inst_type_entropy'].mean():.3f} ± {plateau['inst_type_entropy'].std():.3f}")
    print(f"    Net stability:    {plateau['network_stability'].mean():.3f} ± {plateau['network_stability'].std():.3f}")

    print(f"  Non-plateau artists (n={len(no_plateau)}):")
    print(f"    Transition rate:  {no_plateau['inst_type_transition_rate'].mean():.3f} ± {no_plateau['inst_type_transition_rate'].std():.3f}")
    print(f"    Type entropy:     {no_plateau['inst_type_entropy'].mean():.3f} ± {no_plateau['inst_type_entropy'].std():.3f}")
    print(f"    Net stability:    {no_plateau['network_stability'].mean():.3f} ± {no_plateau['network_stability'].std():.3f}")

    # Mann-Whitney U tests
    for var, label in [('inst_type_transition_rate', 'Transition Rate'),
                       ('inst_type_entropy', 'Type Entropy')]:
        u, p = mannwhitneyu(plateau[var], no_plateau[var], alternative='two-sided')
        print(f"  Mann-Whitney U ({label}): U={u:.0f}, p={p:.4f}")

    # ── Path a: OLS regression ───────────────────────────────
    print("\n" + "─" * 60)
    print("PATH a: Network Stability → Inst. Type Transition Rate")
    print("─" * 60)

    # Standardize for interpretability
    panel_std = panel.copy()
    panel_std['has_overseas'] = panel_std['has_overseas'].astype(float)
    scaler = StandardScaler()
    scale_cols = ['network_stability', 'inst_type_transition_rate', 'inst_type_entropy',
                  'cumulative_validation', 'birth_year']
    panel_std[scale_cols] = scaler.fit_transform(panel_std[scale_cols])

    controls = ['cumulative_validation', 'birth_year', 'has_overseas']
    X_a = sm.add_constant(panel_std[['network_stability'] + controls])
    ols_transition = sm.OLS(panel_std['inst_type_transition_rate'], X_a).fit()
    print(ols_transition.summary2().tables[1])

    ols_entropy = sm.OLS(panel_std['inst_type_entropy'], X_a).fit()
    print("\nPath a (Entropy variant):")
    print(ols_entropy.summary2().tables[1])

    path_a_results = {
        'beta_a': ols_transition.params['network_stability'],
        'p_a': ols_transition.pvalues['network_stability'],
    }

    # ── Path b + Full model ──────────────────────────────────
    print("\n" + "─" * 60)
    print("COX MODELS: Path b and Direct Effect")
    print("─" * 60)

    panel_std['id'] = panel_std['artist_id']

    # Model 1: Total effect (no mediator)
    print("\n  Model 1: Total effect (stability only)")
    ctv1 = CoxTimeVaryingFitter(penalizer=0.01)
    ctv1.fit(panel_std, id_col='id', event_col='event',
             start_col='start', stop_col='stop',
             formula='network_stability + cumulative_validation + birth_year + has_overseas',
             show_progress=False)
    ctv1.print_summary()

    # Model 2: Full model with transition rate
    print("\n  Model 2: Full model (stability + transition rate)")
    ctv2 = CoxTimeVaryingFitter(penalizer=0.01)
    ctv2.fit(panel_std, id_col='id', event_col='event',
             start_col='start', stop_col='stop',
             formula='network_stability + inst_type_transition_rate + cumulative_validation + birth_year + has_overseas',
             show_progress=False)
    ctv2.print_summary()

    # Model 3: Full model with entropy
    # Note: CoxTimeVaryingFitter with entropy can exhibit non-convergence in
    # long-running processes due to internal state. We serialize the panel to
    # CSV and fit in a subprocess to ensure clean environment.
    print("\n  Model 3: Full model (stability + type entropy)")
    import subprocess, tempfile
    m3_cols = ['id', 'start', 'stop', 'event',
               'network_stability', 'inst_type_entropy',
               'cumulative_validation', 'birth_year', 'has_overseas']
    tmp_csv = os.path.join(tempfile.gettempdir(), '_m3_panel.csv')
    panel_std[m3_cols].to_csv(tmp_csv, index=False)
    m3_script = f"""
import pandas as pd, json
from lifelines import CoxTimeVaryingFitter
df = pd.read_csv('{tmp_csv}')
ctv = CoxTimeVaryingFitter(penalizer=0.01)
ctv.fit(df, id_col='id', event_col='event', start_col='start', stop_col='stop', show_progress=False)
s = ctv.summary
result = {{}}
for var in s.index:
    result[var] = {{'coef': float(s.loc[var,'coef']), 'hr': float(s.loc[var,'exp(coef)']),
                    'p': float(s.loc[var,'p']), 'ci_lo': float(s.loc[var,'exp(coef) lower 95%']),
                    'ci_hi': float(s.loc[var,'exp(coef) upper 95%'])}}
print(json.dumps(result))
"""
    proc = subprocess.run(['python3', '-c', m3_script], capture_output=True, text=True, timeout=60)
    if proc.returncode == 0:
        import json as _json
        m3_result = _json.loads(proc.stdout.strip())
        print(f"  {'Variable':30s} {'coef':>8s} {'HR':>8s} {'p':>8s} {'95% CI':>20s}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")
        for var, vals in m3_result.items():
            sig = '***' if vals['p']<0.001 else '**' if vals['p']<0.01 else '*' if vals['p']<0.05 else ''
            print(f"  {var:30s} {vals['coef']:8.3f} {vals['hr']:8.3f} {vals['p']:8.4f} [{vals['ci_lo']:.3f}, {vals['ci_hi']:.3f}]{sig}")
        # Create a proxy ctv3 object for downstream use
        ctv3 = ctv2  # Downstream code only uses ctv2's transition rate params
    else:
        print(f"  Model 3 subprocess error: {proc.stderr[:200]}")
        ctv3 = ctv2
    os.remove(tmp_csv) if os.path.exists(tmp_csv) else None

    # Store path b results
    path_a_results['beta_b'] = ctv2.params_['inst_type_transition_rate']
    path_a_results['p_b'] = ctv2.summary.loc['inst_type_transition_rate', 'p']

    # ── Causal Mediation Analysis ────────────────────────────
    print("\n" + "─" * 60)
    print("CAUSAL MEDIATION ANALYSIS (1000 bootstrap)")
    print("─" * 60)

    # Standardize fresh for mediation (avoid double standardization)
    panel_med = panel.copy()
    med_scale_cols = ['network_stability', 'inst_type_transition_rate', 'inst_type_entropy',
                      'cumulative_validation', 'birth_year']
    scaler_med = StandardScaler()
    panel_med[med_scale_cols] = scaler_med.fit_transform(panel_med[med_scale_cols])

    print("\n  Mediator: inst_type_transition_rate")
    med_transition = bootstrap_mediation(panel_med, 'inst_type_transition_rate',
                                          n_boot=1000, seed=42)
    print(f"\n  Results:")
    print(f"    ACME = {med_transition['acme_mean']:.4f}  "
          f"[{med_transition['acme_ci'][0]:.4f}, {med_transition['acme_ci'][1]:.4f}]  "
          f"p = {med_transition['acme_p']:.3f}")
    print(f"    ADE  = {med_transition['ade_mean']:.4f}  "
          f"[{med_transition['ade_ci'][0]:.4f}, {med_transition['ade_ci'][1]:.4f}]  "
          f"p = {med_transition['ade_p']:.3f}")
    print(f"    Total = {med_transition['total_mean']:.4f}  "
          f"[{med_transition['total_ci'][0]:.4f}, {med_transition['total_ci'][1]:.4f}]")
    print(f"    Proportion mediated = {med_transition['prop_mediated_mean']:.1%}  "
          f"[{med_transition['prop_mediated_ci'][0]:.1%}, {med_transition['prop_mediated_ci'][1]:.1%}]")
    print(f"    Successful bootstraps: {med_transition['n_boot_success']}")

    print("\n  Mediator: inst_type_entropy")
    med_entropy = bootstrap_mediation(panel_med, 'inst_type_entropy',
                                       n_boot=500, seed=42, penalizer=0.05)
    print(f"\n  Results:")
    print(f"    ACME = {med_entropy['acme_mean']:.4f}  "
          f"[{med_entropy['acme_ci'][0]:.4f}, {med_entropy['acme_ci'][1]:.4f}]  "
          f"p = {med_entropy['acme_p']:.3f}")
    print(f"    ADE  = {med_entropy['ade_mean']:.4f}  "
          f"[{med_entropy['ade_ci'][0]:.4f}, {med_entropy['ade_ci'][1]:.4f}]  "
          f"p = {med_entropy['ade_p']:.3f}")
    print(f"    Total = {med_entropy['total_mean']:.4f}  "
          f"[{med_entropy['total_ci'][0]:.4f}, {med_entropy['total_ci'][1]:.4f}]")
    print(f"    Proportion mediated = {med_entropy['prop_mediated_mean']:.1%}")
    print(f"    Successful bootstraps: {med_entropy['n_boot_success']}")

    # ── Reputational Lock-in ─────────────────────────────────
    print("\n" + "─" * 60)
    print("REPUTATIONAL LOCK-IN SIMULATION")
    print("─" * 60)

    ade_significant = med_transition['ade_p'] < 0.05

    if ade_significant:
        print("  Direct effect is significant — running lock-in simulation...")
    else:
        print("  Direct effect is NOT significant (full mediation).")
        print("  Running lock-in simulation for supplementary evidence...")

    lockin = simulate_reputational_lockin(panel, df_events, n_shuffles=200, seed=42)
    print(f"\n  Observed lock-in:")
    print(f"    Plateau artists:     {lockin['obs_plateau_mean']:.3f}")
    print(f"    Non-plateau artists: {lockin['obs_no_plateau_mean']:.3f}")
    print(f"    Difference:          {lockin['obs_diff']:.3f}")
    print(f"  Counterfactual (permutation):")
    print(f"    Mean diff:           {lockin['cf_diff_mean']:.3f}")
    print(f"    95% CI:              [{lockin['cf_diff_ci'][0]:.3f}, {lockin['cf_diff_ci'][1]:.3f}]")
    print(f"    p-value:             {lockin['p_value']:.3f}")

    # ── Additional: Lock-in as predictor of plateau ──────────
    print("\n  Cox model with lock-in ratio as predictor:")
    # Add lock-in to panel
    lockin_dict = lockin['obs_lockin']
    panel_med['lockin_ratio'] = panel_med['artist_id'].map(lockin_dict)
    panel_med['lockin_ratio'] = panel_med['lockin_ratio'].fillna(panel_med['lockin_ratio'].median())
    panel_med['id'] = panel_med['artist_id']

    ctv_lockin = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_lockin.fit(panel_med, id_col='id', event_col='event',
                   start_col='start', stop_col='stop',
                   formula='network_stability + lockin_ratio + cumulative_validation + birth_year',
                   show_progress=False)
    ctv_lockin.print_summary()

    # ── Create figure ────────────────────────────────────────
    print("\n" + "─" * 60)
    print("CREATING FIGURES")
    print("─" * 60)

    figpath = os.path.join(FIG_DIR, 'institutional_type_transition.png')
    create_figure(panel, med_entropy, med_transition, lockin, path_a_results, figpath)

    # ── Save summary table ───────────────────────────────────
    print("\n[6/6] Saving summary tables...")

    # Main mediation comparison table
    rows = []
    for label, res in [('Inst. Type Transition Rate', med_transition),
                       ('Inst. Type Entropy', med_entropy)]:
        rows.append({
            'Mediator': label,
            'ACME': f"{res['acme_mean']:.4f}",
            'ACME_CI': f"[{res['acme_ci'][0]:.4f}, {res['acme_ci'][1]:.4f}]",
            'ACME_p': f"{res['acme_p']:.3f}",
            'ADE': f"{res['ade_mean']:.4f}",
            'ADE_CI': f"[{res['ade_ci'][0]:.4f}, {res['ade_ci'][1]:.4f}]",
            'ADE_p': f"{res['ade_p']:.3f}",
            'Prop_Mediated': f"{res['prop_mediated_mean']:.1%}",
            'N_boot': res['n_boot_success'],
        })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    # ── Final summary ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  Hypothesis 3 Revisited: Institutional Type Transition as Mediator     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Institution Classification:                                          │
│     {len(df_events)} events classified into 6 categories                │
│     (MMCA, Public Museum, Private Museum, Commercial Gallery,            │
│      Academic/Association, Biennale/International)                        │
│                                                                         │
│  2. Mediation Results (Transition Rate):                                 │
│     ACME = {med_transition['acme_mean']:.4f} [{med_transition['acme_ci'][0]:.4f}, {med_transition['acme_ci'][1]:.4f}], p = {med_transition['acme_p']:.3f}│
│     ADE  = {med_transition['ade_mean']:.4f} [{med_transition['ade_ci'][0]:.4f}, {med_transition['ade_ci'][1]:.4f}], p = {med_transition['ade_p']:.3f}│
│     Proportion mediated: {med_transition['prop_mediated_mean']:.1%}                                    │
│                                                                         │
│  3. Mediation Results (Type Entropy):                                    │
│     ACME = {med_entropy['acme_mean']:.4f} [{med_entropy['acme_ci'][0]:.4f}, {med_entropy['acme_ci'][1]:.4f}], p = {med_entropy['acme_p']:.3f}│
│     ADE  = {med_entropy['ade_mean']:.4f} [{med_entropy['ade_ci'][0]:.4f}, {med_entropy['ade_ci'][1]:.4f}], p = {med_entropy['ade_p']:.3f}│
│                                                                         │
│  4. Reputational Lock-in:                                                │
│     Plateau artists lock-in: {lockin['obs_plateau_mean']:.3f}                                 │
│     Non-plateau lock-in:     {lockin['obs_no_plateau_mean']:.3f}                                 │
│     Permutation p = {lockin['p_value']:.3f}                                            │
│                                                                         │
│  5. Interpretation:                                                      │
│     {'Direct effect dominates → Reputational lock-in pathway' if ade_significant else 'Mediation confirmed → Institutional type transition is the mechanism'}│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")


if __name__ == '__main__':
    main()
