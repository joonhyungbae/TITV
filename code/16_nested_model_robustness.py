"""
16_nested_model_robustness.py
Nested Model Sequence (Table 5: tab:nested_models)
Post-decade stability effect under progressive controls:
  M1 (base): Birth year only
  M2: + Cumulative validation (linear)
  M3: + Cumulative validation (splines)
  M4: + Career archetype dummies
  M5 (full): + Splines + archetype dummies
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from lifelines import CoxTimeVaryingFitter
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    build_person_year_panel, CENSOR_YEAR
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def compute_archetypes(df_artists, df_events, n_clusters=5):
    """Compute career archetypes via K-Means on trajectory features."""
    arch_feats = []
    for _, row in df_artists.iterrows():
        aid = row['artist_id']
        cs = row['career_start_year']
        if pd.isna(cs) or row['num_events'] == 0:
            continue
        a_ev = df_events[df_events['artist_id'] == aid]
        cl = a_ev['year'].max() - cs if len(a_ev) > 0 else 0
        n_tot = max(len(a_ev), 1)
        solo_yr = a_ev[a_ev['event_type'] == 'solo_exhibition']['year']
        award_yr = a_ev[a_ev['event_type'] == 'award']['year']
        arch_feats.append({
            'artist_id': aid,
            'career_length': cl,
            'has_overseas': int(row.get('has_overseas', False)),
            'solo_share': len(a_ev[a_ev['event_type'] == 'solo_exhibition']) / n_tot,
            'award_share': len(a_ev[a_ev['event_type'] == 'award']) / n_tot,
            'biennale_share': len(a_ev[a_ev['event_type'] == 'biennale']) / n_tot,
            'group_share': len(a_ev[a_ev['event_type'] == 'group_exhibition']) / n_tot,
            'first_solo_rel': min((solo_yr.min() - cs) if len(solo_yr) > 0 else 40, 40),
            'first_award_rel': min((award_yr.min() - cs) if len(award_yr) > 0 else 40, 40),
            'n_events': len(a_ev),
        })
    arch_df = pd.DataFrame(arch_feats)
    fcols = ['career_length', 'has_overseas', 'solo_share', 'award_share',
             'biennale_share', 'group_share', 'first_solo_rel',
             'first_award_rel', 'n_events']
    X = StandardScaler().fit_transform(arch_df[fcols])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    arch_df['cluster'] = km.fit_predict(X)
    for i in range(n_clusters):
        arch_df[f'arch_{i}'] = (arch_df['cluster'] == i).astype(int)
    return arch_df


def restricted_cubic_spline(x, knots):
    """Compute restricted cubic spline basis."""
    k = len(knots)
    basis = np.zeros((len(x), max(k - 2, 1)))
    if k < 3:
        basis[:, 0] = x
        return basis
    for j in range(k - 2):
        tj = knots[j]
        tkm1 = knots[k - 2]
        tk = knots[k - 1]
        h_j = np.maximum(0, (x - tj) ** 3)
        h_km1 = np.maximum(0, (x - tkm1) ** 3)
        h_k = np.maximum(0, (x - tk) ** 3)
        denom = tk - tkm1
        if abs(denom) < 1e-10:
            denom = 1e-10
        basis[:, j] = h_j - h_km1 * (tk - tj) / denom + h_k * (tkm1 - tj) / denom
    return basis


def main():
    print("=" * 70)
    print("NESTED MODEL SEQUENCE (Table 5)")
    print("=" * 70)

    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)
    panel = build_person_year_panel(df_artists, df_events, cutpoint=10)

    post = panel[panel['post_cutpoint'] == 1].copy()
    print(f"\nPost-decade: {len(post)} person-years, "
          f"{post['artist_id'].nunique()} artists, "
          f"{int(post['event'].sum())} events")

    scaler = StandardScaler()
    for col in ['network_stability', 'network_size', 'birth_year',
                'cumulative_validation']:
        post[f'{col}_z'] = scaler.fit_transform(post[[col]]).flatten()
    post['id'] = post['artist_id']

    # Archetypes
    print("\nComputing archetypes...")
    arch_df = compute_archetypes(df_artists, df_events)
    arch_df['artist_id'] = arch_df['artist_id'].astype(str)
    post['artist_id'] = post['artist_id'].astype(str)
    acols = [f'arch_{i}' for i in range(5)]
    post = post.merge(arch_df[['artist_id'] + acols], on='artist_id', how='left')
    for c in acols:
        post[c] = post[c].fillna(0)

    # Cubic spline for cumulative validation
    cv = post['cumulative_validation'].values
    knots = np.percentile(cv[cv > 0], [10, 50, 90])
    sp = restricted_cubic_spline(cv, knots)
    post['cv_sp1'] = sp[:, 0]
    post['cv_sp1_z'] = scaler.fit_transform(post[['cv_sp1']]).flatten()

    # Archetype dummy string
    arch_str = ' + '.join(acols[1:])

    specs = [
        ('M1', 'network_stability_z + network_size_z + birth_year_z',
         'Birth year'),
        ('M2', 'network_stability_z + network_size_z + birth_year_z + cumulative_validation_z',
         '+ Cumulative validation (linear)'),
        ('M3', 'network_stability_z + network_size_z + birth_year_z + cumulative_validation_z + cv_sp1_z',
         '+ Cumulative validation (cubic splines)'),
        ('M4', 'network_stability_z + network_size_z + birth_year_z + cumulative_validation_z + ' + arch_str,
         '+ Career archetype dummies'),
        ('M5', 'network_stability_z + network_size_z + birth_year_z + cumulative_validation_z + cv_sp1_z + ' + arch_str,
         '+ Splines + archetype dummies'),
    ]

    models = {}
    for name, formula, controls in specs:
        print(f"\n--- {name}: {controls} ---")
        try:
            ctv = CoxTimeVaryingFitter(penalizer=0.01)
            ctv.fit(post, id_col='id', event_col='event',
                    start_col='start', stop_col='stop',
                    formula=formula, show_progress=False)
            s = ctv.summary.loc['network_stability_z']
            models[name] = {'hr': s['exp(coef)'], 'p': s['p'], 'controls': controls}
            print(f"  Stability HR={s['exp(coef)']:.3f}, p={s['p']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY: Table 5 (Nested Model Sequence)")
    print("=" * 70)
    print(f"\n  {'Model':<8} {'HR':>8} {'p':>10}   Controls")
    print(f"  {'-'*60}")
    for name, r in models.items():
        p_str = "<0.001" if r['p'] < 0.001 else f"{r['p']:.3f}"
        print(f"  {name:<8} {r['hr']:>8.3f} {p_str:>10}   {r['controls']}")

    print(f"\n  >>> Paper: M1=1.39, M2=1.39, M3=1.39, M4=1.35, M5=1.39 <<<")
    print("\n\nDONE.")


if __name__ == '__main__':
    main()
