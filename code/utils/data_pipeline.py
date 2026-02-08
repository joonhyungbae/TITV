"""
Shared data processing pipeline for DAARTS robustness analyses.
All functions are deterministic and reproducible.
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Constants
# ============================================================

CENSOR_YEAR = 2002

# Original cumulative validation weights (from paper Section 3.3)
ORIGINAL_TYPE_WEIGHTS = {
    'award': 5.0,
    'biennale': 4.0,
    'solo_exhibition': 3.0,
    'collection': 2.0,
    'honor': 2.0,
    'residency': 1.5,
    'position': 1.5,
    'group_exhibition': 1.0,
    'education': 0.5,
    'other': 0.5,
}

# IVS-A: Equal weights (all event types = 1.0)
EQUAL_TYPE_WEIGHTS = {k: 1.0 for k in ORIGINAL_TYPE_WEIGHTS}

# IVS-B: Binary (high-prestige vs low-prestige)
BINARY_TYPE_WEIGHTS = {
    'award': 2.0,
    'biennale': 2.0,
    'solo_exhibition': 2.0,
    'collection': 2.0,
    'honor': 2.0,
    'residency': 1.0,
    'position': 1.0,
    'group_exhibition': 1.0,
    'education': 1.0,
    'other': 1.0,
}

# IVS-D: Compressed hierarchy (sqrt of original weights — narrows the spread)
COMPRESSED_TYPE_WEIGHTS = {k: round(v ** 0.5, 2) for k, v in ORIGINAL_TYPE_WEIGHTS.items()}

# IVS-E: Expanded hierarchy (square of original weights — amplifies the spread)
EXPANDED_TYPE_WEIGHTS = {k: round(v ** 2, 2) for k, v in ORIGINAL_TYPE_WEIGHTS.items()}

# Significant event types for plateau detection
SIGNIFICANT_EVENT_TYPES = {'solo_exhibition', 'award', 'biennale', 'collection', 'honor'}


# ============================================================
# Data Loading
# ============================================================

def load_raw_data(data_path):
    """Load data.json and return the raw results list."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']


def extract_artist_info(artists_list):
    """Extract artist-level information into a DataFrame."""
    records = []
    for artist in artists_list:
        meta = artist['metadata']
        basic = artist['basic_info']
        overseas = artist['overseas_activity']
        events = artist['career_events']

        # Career start year from events
        event_years = [e['year'] for e in events if e.get('year') is not None]
        career_start = min(event_years) if event_years else None

        records.append({
            'artist_id': meta['artist_id'],
            'name_en': basic['name_en'],
            'birth_year': basic.get('birth_year'),
            'generation_cohort': basic.get('generation_cohort'),
            'career_start_year': career_start,
            'has_overseas': overseas['has_overseas_experience'],
            'num_events': len(events),
        })
    return pd.DataFrame(records)


def extract_events(artists_list):
    """Flatten all career events into a single DataFrame."""
    rows = []
    for artist in artists_list:
        aid = artist['metadata']['artist_id']
        for ev in artist['career_events']:
            if ev.get('year') is None:
                continue
            rows.append({
                'artist_id': aid,
                'year': ev['year'],
                'event_type': ev.get('event_type', 'other'),
                'institution_en': ev.get('institution_en'),
                'institution_type': ev.get('institution_type'),
                'country': ev.get('country'),
            })
    return pd.DataFrame(rows)


# ============================================================
# Cumulative Validation Score
# ============================================================

def compute_event_weight(event_type, type_weights):
    """Compute weight for a single event based on event type."""
    return type_weights.get(event_type, 0.5)


def compute_rank_based_weights(events_df):
    """IVS-C: Rank-based weights = inverse frequency of event types."""
    type_counts = events_df['event_type'].value_counts()
    total = type_counts.sum()
    # Inverse frequency, normalized so max = 5.0
    inv_freq = {etype: total / count for etype, count in type_counts.items()}
    max_w = max(inv_freq.values())
    return {etype: (w / max_w) * 5.0 for etype, w in inv_freq.items()}


# ============================================================
# Plateau Detection
# ============================================================

def detect_plateau(events_df, artist_id, career_start, window=5,
                   sig_types=None, censor_year=CENSOR_YEAR):
    """
    Detect first career plateau for an artist.
    Plateau = `window` consecutive years with no significant events.
    Returns (plateau_occurred, plateau_year, time_to_plateau).
    """
    if sig_types is None:
        sig_types = SIGNIFICANT_EVENT_TYPES

    artist_events = events_df[
        (events_df['artist_id'] == artist_id) &
        (events_df['year'] >= career_start) &
        (events_df['year'] <= censor_year)
    ]

    # Years with significant events
    sig_events = artist_events[artist_events['event_type'].isin(sig_types)]
    sig_years = set(sig_events['year'].unique())

    end_year = min(int(artist_events['year'].max()) if len(artist_events) > 0 else censor_year, censor_year)

    # Scan for first window-length gap
    for start_y in range(int(career_start), end_year - window + 2):
        gap_years = set(range(start_y, start_y + window))
        if not gap_years.intersection(sig_years):
            return True, start_y, start_y - career_start

    return False, None, end_year - career_start


# ============================================================
# Network Metrics
# ============================================================

def compute_network_size_stability(events_up_to_year):
    """Compute network size and stability from cumulative events."""
    if len(events_up_to_year) == 0:
        return 0, 0.0

    institutions = events_up_to_year['institution_en'].dropna().unique()
    network_size = len(institutions)
    total_events = len(events_up_to_year)

    if network_size > 0:
        stability = total_events / network_size
    else:
        stability = 0.0

    return network_size, stability


def compute_burt_constraint(events_up_to_year):
    """
    Compute Burt's network constraint from artist-institution bipartite projection.
    Ego-centric: each institution's share of events is p_ij.
    Indirect path assumed uniform: 1/(n-1).
    Isolates (size <= 1): constraint = 1.0.
    """
    if len(events_up_to_year) == 0:
        return 1.0

    inst_series = events_up_to_year['institution_en'].dropna()
    if len(inst_series) == 0:
        return 1.0

    inst_counts = inst_series.value_counts().to_dict()
    institutions = list(inst_counts.keys())
    n = len(institutions)
    total = sum(inst_counts.values())

    if n <= 1:
        return 1.0

    constraint = 0.0
    for inst in institutions:
        p_ij = inst_counts[inst] / total
        indirect = sum(
            (inst_counts[other] / total) * (1.0 / (n - 1))
            for other in institutions if other != inst
        )
        constraint += (p_ij + indirect) ** 2

    return constraint / n


def compute_event_type_diversity(events_window):
    """
    Shannon entropy of event types within a window of events.
    """
    if len(events_window) == 0:
        return 0.0
    type_counts = events_window['event_type'].value_counts()
    type_probs = type_counts / type_counts.sum()
    return entropy(type_probs)


# ============================================================
# Person-Year Panel Construction
# ============================================================

def build_person_year_panel(df_artists, df_events, cutpoint=10,
                            type_weights=None,
                            include_constraint=False,
                            include_event_type_diversity=False,
                            censor_year=CENSOR_YEAR):
    """
    Build person-year panel dataset for Cox time-varying models.

    Parameters
    ----------
    df_artists : DataFrame with artist-level info (must have artist_id, career_start_year, birth_year, etc.)
    df_events : DataFrame with all events (must have artist_id, year, event_type, institution_en)
    cutpoint : int, career year boundary for phase split
    type_weights : dict, event type weights for cumulative validation
    include_constraint : bool, compute Burt's constraint (slow)
    include_event_type_diversity : bool, compute Shannon entropy of event types
    censor_year : int, right-censoring year

    Returns
    -------
    DataFrame in person-year format with start/stop columns for lifelines
    """
    if type_weights is None:
        type_weights = ORIGINAL_TYPE_WEIGHTS

    # Pre-compute plateau for each artist
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

        if occurred:
            end_year = int(p_year)
        else:
            end_year = censor_year

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
            cum_val = 0.0
            for _, ev in events_up_to.iterrows():
                cum_val += compute_event_weight(
                    ev['event_type'], type_weights
                )

            # Event indicator
            is_last_year = (year == end_year)
            event = 1 if (is_last_year and occurred) else 0

            rec = {
                'artist_id': aid,
                'year': year,
                'career_year': career_year,
                'start': career_year,
                'stop': career_year + 1,
                'event': event,
                'network_size': max(net_size, 0.5),  # avoid zero
                'network_stability': net_stability,
                'cumulative_validation': cum_val,
                'post_cutpoint': 1 if career_year >= cutpoint else 0,
                'birth_year': birth_year,
                'has_overseas': has_overseas,
            }

            if include_constraint:
                rec['network_constraint'] = compute_burt_constraint(events_up_to)

            if include_event_type_diversity:
                # 5-year rolling window
                window_events = artist_events[
                    (artist_events['year'] >= year - 4) &
                    (artist_events['year'] <= year)
                ]
                rec['event_type_diversity'] = compute_event_type_diversity(window_events)

            records.append(rec)

    df = pd.DataFrame(records)

    # Fill missing birth_year with median
    if 'birth_year' in df.columns:
        median_by = df['birth_year'].median()
        df['birth_year'] = df['birth_year'].fillna(median_by)

    return df


# ============================================================
# Cox Model Fitting Helper
# ============================================================

def fit_phase_split_cox(panel_df, cutpoint_col='post_cutpoint',
                        formula=None, extra_covariates=None):
    """
    Fit Cox time-varying models split by career phase.

    Returns
    -------
    dict with 'pre' and 'post' CoxTimeVaryingFitter results
    """
    from lifelines import CoxTimeVaryingFitter

    if formula is None:
        base_covs = ['network_size', 'network_stability',
                     'birth_year', 'cumulative_validation',
                     'has_overseas']
        if extra_covariates:
            base_covs.extend(extra_covariates)
        formula = ' + '.join(base_covs)

    results = {}
    for phase_name, phase_val in [('pre', 0), ('post', 1)]:
        subset = panel_df[panel_df[cutpoint_col] == phase_val].copy()
        if len(subset) == 0 or subset['event'].sum() == 0:
            results[phase_name] = None
            continue

        # Ensure id column
        subset['id'] = subset['artist_id']

        ctv = CoxTimeVaryingFitter()
        try:
            ctv.fit(subset, id_col='id', event_col='event',
                    start_col='start', stop_col='stop',
                    formula=formula, show_progress=False)
            results[phase_name] = ctv
        except Exception as e:
            print(f"  Warning: {phase_name} model failed: {e}")
            results[phase_name] = None

    return results


def extract_hr_table(ctv_model, label=''):
    """Extract HR, CI, p-value from a fitted CoxTimeVaryingFitter."""
    if ctv_model is None:
        return pd.DataFrame()
    summary = ctv_model.summary
    df = pd.DataFrame({
        'phase': label,
        'variable': summary.index,
        'coef': summary['coef'],
        'HR': summary['exp(coef)'],
        'HR_lower': summary['exp(coef) lower 95%'],
        'HR_upper': summary['exp(coef) upper 95%'],
        'p': summary['p'],
    })
    return df


# ============================================================
# Archetype helpers
# ============================================================

def load_sequence_archetypes(csv_path):
    """Load the 5-cluster sequence-based archetype assignments."""
    df = pd.read_csv(csv_path)
    # Ensure correct columns
    if 'cluster_name' in df.columns and 'artist_id' in df.columns:
        return df[['artist_id', 'cluster_name']].drop_duplicates('artist_id')
    return df


# ============================================================
# LaTeX Table Generation
# ============================================================

def format_hr(hr, ci_low, ci_high, p, decimals=3):
    """Format HR with CI and statistical significance stars."""
    stars = ''
    if p < 0.001:
        stars = '***'
    elif p < 0.01:
        stars = '**'
    elif p < 0.05:
        stars = '*'

    return f"{hr:.{decimals}f}{stars} [{ci_low:.{decimals}f}, {ci_high:.{decimals}f}]"


def format_p(p):
    """Format p-value for LaTeX."""
    if p < 0.001:
        return f"$<$0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"
