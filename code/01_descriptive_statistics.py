"""
01_descriptive_statistics.py
──────────────────────────────────────────────────────────────────────
Compute descriptive statistics reported in the paper.
  - Table 1 (tab:descriptives): Person-year panel summary
  - Table 2 (tab:period_infra): Period-specific plateau rates & venue counts
  - Section 3.1: Gini coefficient of institutional prestige (~0.72)
──────────────────────────────────────────────────────────────────────
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    build_person_year_panel, detect_plateau, SIGNIFICANT_EVENT_TYPES, CENSOR_YEAR
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# ============================================================
# Gini coefficient
# ============================================================

def gini_coefficient(values):
    """Compute the Gini coefficient of a distribution."""
    values = np.array(values, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * values) / (n * np.sum(values))) - (n + 1.0) / n


# ============================================================
# Period-specific plateau rates (Table 2)
# ============================================================

PERIOD_DEFINITIONS = [
    ('1930s-1940s', 1929, 1949),
    ('1950s-1960s', 1950, 1969),
    ('1970s-1980s', 1970, 1989),
    ('1990s-2002',  1990, 2002),
]


def compute_period_plateau_rates(df_artists, df_events):
    """
    Compute period-specific plateau rates and active venue counts.
    An artist is assigned to the period of their career start year.
    """
    results = []
    for period_name, start, end in PERIOD_DEFINITIONS:
        # Artists whose career started in this period
        period_artists = df_artists[
            (df_artists['career_start_year'] >= start) &
            (df_artists['career_start_year'] <= end) &
            (df_artists['num_events'] > 0)
        ]
        n_artists = len(period_artists)
        if n_artists == 0:
            continue

        # Count plateau rate for these artists
        n_plateau = 0
        for _, row in period_artists.iterrows():
            aid = row['artist_id']
            cs = row['career_start_year']
            if pd.isna(cs):
                continue
            occurred, _, _ = detect_plateau(df_events, aid, int(cs))
            if occurred:
                n_plateau += 1

        plateau_rate = n_plateau / n_artists if n_artists > 0 else 0

        # Count unique venues active in this period
        period_events = df_events[
            (df_events['year'] >= start) &
            (df_events['year'] <= end)
        ]
        active_venues = period_events['institution_en'].dropna().nunique()

        results.append({
            'period': period_name,
            'n_artists': n_artists,
            'n_plateau': n_plateau,
            'plateau_rate': plateau_rate,
            'active_venues': active_venues,
        })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("DESCRIPTIVE STATISTICS (Tables 1, 2 & Gini)")
    print("=" * 70)

    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)

    # ================================================================
    # 1. SAMPLE SIZE
    # ================================================================
    print(f"\n1. SAMPLE SIZE")
    print(f"   Total artists in data.json: {len(artists_list)}")
    n_with_events = (df_artists['num_events'] > 0).sum()
    print(f"   Artists with events (num_events > 0): {n_with_events}")
    print(f"   Total career events: {len(df_events)}")
    print(f"   Event year range: {int(df_events['year'].min())}--{int(df_events['year'].max())}")

    # ================================================================
    # 2. TABLE 1: Person-Year Panel Descriptives
    # ================================================================
    print(f"\n" + "=" * 70)
    print("TABLE 1: PERSON-YEAR PANEL DESCRIPTIVE STATISTICS")
    print("=" * 70)

    panel = build_person_year_panel(df_artists, df_events, cutpoint=10)
    n_panel_artists = panel['artist_id'].nunique()
    n_person_years = len(panel)
    n_events = int(panel['event'].sum())

    print(f"\n   Panel: {n_person_years:,} person-years, {n_panel_artists} artists, {n_events} events")

    # Panel-level variables
    panel_vars = {
        'Network stability': 'network_stability',
        'Network size': 'network_size',
        'Cumulative validation': 'cumulative_validation',
        'Career year': 'career_year',
        'Birth year': 'birth_year',
    }
    print(f"\n   {'Variable':<25} {'N':>6} {'Mean':>8} {'SD':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"   {'-'*73}")
    for name, col in panel_vars.items():
        s = panel[col].dropna()
        print(f"   {name:<25} {len(s):>6} {s.mean():>8.2f} {s.std():>8.2f} "
              f"{s.median():>8.2f} {s.min():>8.1f} {s.max():>8.1f}")

    # Artist-level variables
    artist_panel = panel.groupby('artist_id').agg(
        events_per_artist=('cumulative_validation', 'last'),
        unique_institutions=('network_size', 'last'),
        career_length=('career_year', 'max'),
        plateau=('event', 'max'),
    )

    # Recount using raw events
    events_per_artist = df_events.groupby('artist_id').size()
    unique_inst = df_events.groupby('artist_id')['institution_en'].nunique()

    active_ids = set(panel['artist_id'].unique())
    epa = events_per_artist[events_per_artist.index.isin(active_ids)]
    ui = unique_inst[unique_inst.index.isin(active_ids)]

    plateau_artists = panel.groupby('artist_id')['event'].max()
    n_plateau = (plateau_artists == 1).sum()
    plateau_rate = n_plateau / n_panel_artists

    print(f"\n   Artist-level:")
    print(f"   Events per artist:    N={len(epa)}, Mean={epa.mean():.1f}, SD={epa.std():.1f}, Median={epa.median():.0f}")
    print(f"   Unique institutions:  N={len(ui)}, Mean={ui.mean():.1f}, SD={ui.std():.1f}, Median={ui.median():.0f}")
    cl = artist_panel['career_length']
    print(f"   Career length (yrs):  N={len(cl)}, Mean={cl.mean():.1f}, SD={cl.std():.1f}, Median={cl.median():.0f}")
    print(f"   Plateau experienced:  {plateau_rate:.1%} ({n_plateau}/{n_panel_artists})")

    # Event distribution by phase
    pre = panel[panel['post_cutpoint'] == 0]
    post = panel[panel['post_cutpoint'] == 1]
    print(f"\n   Pre-decade events:  {int(pre['event'].sum())}  (of {int(pre['event'].sum()) + int(post['event'].sum())})")
    print(f"   Post-decade events: {int(post['event'].sum())}")

    # ================================================================
    # 3. TABLE 2: Period-Specific Plateau Rates
    # ================================================================
    print(f"\n" + "=" * 70)
    print("TABLE 2: PERIOD-SPECIFIC INSTITUTIONAL INFRASTRUCTURE & PLATEAU RATES")
    print("=" * 70)

    period_df = compute_period_plateau_rates(df_artists, df_events)
    print(f"\n   {'Period':<15} {'Active Venues':>14} {'Plateau Rate':>14} {'N artists':>10}")
    print(f"   {'-'*55}")
    for _, row in period_df.iterrows():
        print(f"   {row['period']:<15} {row['active_venues']:>14} "
              f"{row['plateau_rate']:>13.0%} {row['n_artists']:>10}")

    # ================================================================
    # 4. GINI COEFFICIENT OF INSTITUTIONAL PRESTIGE
    # ================================================================
    print(f"\n" + "=" * 70)
    print("GINI COEFFICIENT OF INSTITUTIONAL PRESTIGE")
    print("(Section 3.1: paper reports ~0.72)")
    print("=" * 70)

    # Institutional prestige proxy: number of events per institution
    inst_event_counts = df_events['institution_en'].dropna().value_counts()
    gini = gini_coefficient(inst_event_counts.values)
    print(f"\n   Total unique institutions: {len(inst_event_counts)}")
    print(f"   Gini coefficient (event concentration): {gini:.3f}")
    print(f"   Top 5 institutions:")
    for inst, cnt in inst_event_counts.head(5).items():
        print(f"     {inst}: {cnt} events")

    # Also compute Gini using artist-count per institution (how many artists exhibited there)
    inst_artist_counts = df_events.dropna(subset=['institution_en']).groupby(
        'institution_en')['artist_id'].nunique()
    gini_artists = gini_coefficient(inst_artist_counts.values)
    print(f"\n   Gini coefficient (artist concentration): {gini_artists:.3f}")

    print(f"\n   >>> Paper reports Gini ≈ 0.72 <<<")

    print("\nDONE.")


if __name__ == '__main__':
    main()
