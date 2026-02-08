"""
15_career_plateau_dynamics.py
──────────────────────────────────────────────────────────────────────
Compute career plateau dynamics reported in Section 4.1:

(1) Milestone protection analysis:
    - Solo exhibition: β(t) = -0.74 + 0.15·log(t)
      → HR at 1yr=0.48, 3yr=0.56, 10yr=0.67; half-life ≈ 20.5yr
    - Award: β(t) = -0.46 + 0.09·log(t)
      → HR at 1yr=0.63, 3yr=0.70, 10yr=0.78; half-life ≈ 17.5yr
    - Biennale: not significant

(2) Recovery HR:
    - Early career recovery HR = 1.87 vs late career HR = 1.02
    - Interaction p < 0.001
──────────────────────────────────────────────────────────────────────
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
    build_person_year_panel, detect_plateau,
    CENSOR_YEAR, SIGNIFICANT_EVENT_TYPES
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# ============================================================
# Milestone protection: time-varying Cox with log(t) decay
# ============================================================

def build_milestone_panel(df_artists, df_events, milestone_type='solo_exhibition'):
    """
    Build person-year panel with time-since-last-milestone variable.
    Uses the full panel (including artists without milestones) and
    caps years_since at a reasonable maximum for model stability.
    """
    panel = build_person_year_panel(df_artists, df_events, cutpoint=10)
    milestone_events = df_events[df_events['event_type'] == milestone_type].copy()

    years_since_list = []
    had_milestone_list = []

    for _, row in panel.iterrows():
        aid = row['artist_id']
        year = row['year']
        past = milestone_events[
            (milestone_events['artist_id'] == aid) &
            (milestone_events['year'] <= year)
        ]
        if len(past) > 0:
            years_since_list.append(year - past['year'].max())
            had_milestone_list.append(1)
        else:
            years_since_list.append(row['career_year'] + 1)  # career time as proxy
            had_milestone_list.append(0)

    panel['years_since_milestone'] = years_since_list
    panel['had_milestone'] = had_milestone_list
    # Cap and compute log
    panel['years_since_milestone'] = panel['years_since_milestone'].clip(upper=45)
    panel['log_years_since'] = np.log1p(panel['years_since_milestone'])
    return panel


def compute_milestone_protection(panel_ms, milestone_name):
    """
    Fit time-varying Cox model for milestone protection:
      β(t) = β₀ + β₁·log(years_since_milestone)
    where β₀ captures the initial protective effect and β₁ captures decay.
    """
    sub = panel_ms.copy()
    sub = sub[sub['log_years_since'].notna() & np.isfinite(sub['log_years_since'])].copy()

    if len(sub) < 50 or sub['event'].sum() < 10:
        print(f"  {milestone_name}: insufficient data")
        return None

    scaler = StandardScaler()
    for col in ['network_stability', 'network_size', 'career_year',
                'birth_year', 'cumulative_validation',
                'years_since_milestone', 'log_years_since']:
        vals = sub[[col]].values
        if np.any(~np.isfinite(vals)):
            sub[col] = sub[col].fillna(sub[col].median())
        sub[f'{col}_z'] = scaler.fit_transform(sub[[col]]).flatten()

    sub['id'] = sub['artist_id']

    # Model: use log_years_since as direct predictor (protective if negative coef)
    formula = ('log_years_since_z + '
               'network_stability_z + network_size_z + '
               'birth_year_z + cumulative_validation_z')

    try:
        ctv = CoxTimeVaryingFitter(penalizer=0.01)
        ctv.fit(sub, id_col='id', event_col='event',
                start_col='start', stop_col='stop',
                formula=formula, show_progress=False)

        s = ctv.summary
        beta_log = s.loc['log_years_since_z', 'coef']
        se_log = s.loc['log_years_since_z', 'se(coef)']
        p_log = s.loc['log_years_since_z', 'p']
        hr_log = s.loc['log_years_since_z', 'exp(coef)']

        print(f"\n  {milestone_name}:")
        print(f"    log(years_since) coef = {beta_log:.3f}, HR = {hr_log:.3f}, p = {p_log:.4f}")

        # Map to paper's β(t) = β₀ + β₁·log(t) form
        # Our standardized coef maps: the effect per SD of log(years_since)
        lys_mean = sub['log_years_since'].mean()
        lys_std = sub['log_years_since'].std()

        # Unstandardized: beta_raw = beta_log / lys_std (per unit log-year)
        beta_raw = beta_log / lys_std if lys_std > 0 else 0

        # The "main effect" is the HR when years_since = 0 (just had milestone)
        # β(t=0) = β₀, β(t) = β₀ + β₁·log(t)
        # In our model: coef at mean log_years ≈ 0 (standardized)
        # HR at 1 year since: exp(beta_raw * log(1)) = exp(0) → baseline
        # Reconstruct: β₀ = full model intercept effect, β₁ = beta_raw
        beta_0 = -beta_log * lys_mean / lys_std if lys_std > 0 else 0
        beta_1 = beta_raw

        print(f"    Reconstructed: β(t) ≈ {beta_0:.3f} + {beta_1:.3f}·log(t)")

        # Conditional HRs at specific years since milestone
        for yrs in [1, 3, 10]:
            log_t = np.log1p(yrs)
            z_val = (log_t - lys_mean) / lys_std if lys_std > 0 else 0
            hr_t = np.exp(beta_log * z_val)
            print(f"    HR at {yrs:2d} yr since milestone: {hr_t:.3f}")

        # Half-life
        if beta_1 > 0 and beta_0 < 0:
            half_life = np.exp(-beta_0 / (2 * beta_1))
            print(f"    Protection half-life ≈ {half_life:.1f} years")
        else:
            half_life = None
            print(f"    Half-life: n/a (sign pattern differs)")

        return {
            'beta_0': beta_0, 'beta_1': beta_1,
            'p_log': p_log, 'half_life': half_life,
            'beta_log_z': beta_log,
        }
    except Exception as e:
        print(f"  {milestone_name}: model failed — {e}")
        return None


# ============================================================
# Recovery analysis: stage-contingent recovery from plateau
# ============================================================

def compute_recovery_hr(df_artists, df_events):
    """
    Compute recovery hazard ratios by career stage.

    Approach: among artists who experienced a plateau, model the
    hazard of 'recovery' (resumption of significant activity)
    as a function of career stage at plateau onset.
    """
    print("\n" + "=" * 70)
    print("RECOVERY HR BY CAREER STAGE")
    print("(Paper: HR=1.87 early career vs 1.02 late career, interaction p<0.001)")
    print("=" * 70)

    recovery_records = []
    for _, row in df_artists.iterrows():
        aid = row['artist_id']
        cs = row['career_start_year']
        if pd.isna(cs) or row['num_events'] == 0:
            continue

        cs_int = int(cs)
        occurred, p_year, _ = detect_plateau(df_events, aid, cs_int)
        if not occurred or p_year is None:
            continue

        plateau_career_year = p_year - cs_int

        # Check for recovery: any significant event after plateau onset + 5 years
        a_events = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] > p_year + 4) &
            (df_events['year'] <= CENSOR_YEAR) &
            (df_events['event_type'].isin(SIGNIFICANT_EVENT_TYPES))
        ]

        recovered = len(a_events) > 0
        if recovered:
            recovery_year = a_events['year'].min()
            time_to_recovery = recovery_year - p_year
        else:
            time_to_recovery = CENSOR_YEAR - p_year

        recovery_records.append({
            'artist_id': aid,
            'plateau_year': p_year,
            'plateau_career_year': plateau_career_year,
            'recovered': int(recovered),
            'time_to_recovery': time_to_recovery,
            'birth_year': row.get('birth_year'),
            'early_career': 1 if plateau_career_year < 10 else 0,
        })

    rec_df = pd.DataFrame(recovery_records)
    if len(rec_df) == 0:
        print("  No plateau artists found.")
        return None

    n_total = len(rec_df)
    n_recovered = rec_df['recovered'].sum()
    print(f"\n  Plateau artists: {n_total}")
    print(f"  Recovered: {n_recovered} ({n_recovered/n_total:.1%})")

    # Early vs late career plateau
    early = rec_df[rec_df['early_career'] == 1]
    late = rec_df[rec_df['early_career'] == 0]
    print(f"\n  Early-career plateau (< yr 10): {len(early)} artists, "
          f"recovered {early['recovered'].sum()} ({early['recovered'].mean():.1%})")
    print(f"  Late-career plateau (≥ yr 10):  {len(late)} artists, "
          f"recovered {late['recovered'].sum()} ({late['recovered'].mean():.1%})")

    # Cox model for recovery: model the HAZARD of recovery (= good event)
    # Early-career plateau artists should have HIGHER recovery hazard (HR > 1)
    # Late-career plateau artists should have LOWER recovery hazard (HR ~ 1)
    recovery_py = []
    for _, r in rec_df.iterrows():
        max_follow = min(r['time_to_recovery'], 30)
        if max_follow < 1:
            max_follow = 1
        for t in range(1, int(max_follow) + 1):
            is_event = 1 if (t == int(r['time_to_recovery']) and r['recovered']) else 0
            recovery_py.append({
                'artist_id': r['artist_id'],
                'start': t - 1,
                'stop': t,
                'event': is_event,
                'early_career': r['early_career'],
                'plateau_career_year': r['plateau_career_year'],
                'birth_year': r['birth_year'],
            })

    rpy = pd.DataFrame(recovery_py)
    if len(rpy) == 0 or rpy['event'].sum() < 5:
        print("  Insufficient recovery events for Cox model.")
        return None

    rpy['id'] = rpy['artist_id']
    median_by = rpy['birth_year'].median()
    rpy['birth_year'] = rpy['birth_year'].fillna(median_by)

    scaler = StandardScaler()
    rpy['plateau_cy_z'] = scaler.fit_transform(rpy[['plateau_career_year']]).flatten()
    rpy['birth_year_z'] = scaler.fit_transform(rpy[['birth_year']]).flatten()

    # Model 1: binary early/late
    try:
        formula_rec = 'early_career + birth_year_z'
        ctv_rec = CoxTimeVaryingFitter(penalizer=0.01)
        ctv_rec.fit(rpy, id_col='id', event_col='event',
                    start_col='start', stop_col='stop',
                    formula=formula_rec, show_progress=False)
        s = ctv_rec.summary
        early_row = s.loc['early_career']
        print(f"\n  Recovery Cox model (event = recovery):")
        print(f"    Early career indicator: HR={early_row['exp(coef)']:.3f} "
              f"[{early_row['exp(coef) lower 95%']:.3f}, {early_row['exp(coef) upper 95%']:.3f}], "
              f"p={early_row['p']:.4f}")
        print(f"    (HR>1 = early-career plateaus recover faster)")
    except Exception as e:
        print(f"  Binary model failed: {e}")

    # Model 2: continuous career year at plateau + interaction
    try:
        # Interaction with follow-up time to test stage-contingency
        rpy['follow_year'] = rpy['stop']
        rpy['follow_year_z'] = scaler.fit_transform(rpy[['follow_year']]).flatten()
        rpy['pcy_x_follow'] = rpy['plateau_cy_z'] * rpy['follow_year_z']

        formula_int = 'plateau_cy_z + follow_year_z + pcy_x_follow + birth_year_z'
        ctv_int = CoxTimeVaryingFitter(penalizer=0.01)
        ctv_int.fit(rpy, id_col='id', event_col='event',
                    start_col='start', stop_col='stop',
                    formula=formula_int, show_progress=False)
        s_int = ctv_int.summary
        print(f"\n  Continuous interaction model:")
        for var in s_int.index:
            r = s_int.loc[var]
            print(f"    {var}: HR={r['exp(coef)']:.3f}, p={r['p']:.4f}")

        # Conditional recovery HR: early (cy=2) vs late (cy=15)
        cy_mean = rec_df['plateau_career_year'].mean()
        cy_std = rec_df['plateau_career_year'].std()
        beta_pcy = s_int.loc['plateau_cy_z', 'coef']
        beta_int = s_int.loc['pcy_x_follow', 'coef']

        for cy_val, label in [(2, 'Early (yr 2)'), (15, 'Late (yr 15)')]:
            cy_z = (cy_val - cy_mean) / cy_std
            # At mean follow-up time (z=0)
            cond_coef = beta_pcy * cy_z
            cond_hr = np.exp(cond_coef)
            print(f"    Conditional recovery HR at {label}: {cond_hr:.3f}")

        print(f"\n    >>> Paper: early HR=1.87, late HR=1.02, interaction p<0.001 <<<")

    except Exception as e:
        print(f"  Interaction model failed: {e}")

    return rec_df


def main():
    print("=" * 70)
    print("CAREER PLATEAU DYNAMICS")
    print("(Section 4.1: Milestone Protection + Recovery)")
    print("=" * 70)

    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)

    # ================================================================
    # PART 1: MILESTONE PROTECTION
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 1: MILESTONE PROTECTION ANALYSIS")
    print("(Paper: solo β(t) = -0.74 + 0.15·log(t), award β(t) = -0.46 + 0.09·log(t))")
    print("=" * 70)

    milestone_types = [
        ('solo_exhibition', 'Solo Exhibition'),
        ('award', 'Award'),
        ('biennale', 'Biennale'),
    ]

    results = {}
    for mtype, mname in milestone_types:
        print(f"\n  Building panel for {mname}...")
        panel_ms = build_milestone_panel(df_artists, df_events, milestone_type=mtype)
        result = compute_milestone_protection(panel_ms, mname)
        if result:
            results[mtype] = result

    # Summary
    print("\n" + "=" * 70)
    print("MILESTONE PROTECTION SUMMARY")
    print("=" * 70)
    for mtype, mname in milestone_types:
        if mtype in results:
            r = results[mtype]
            print(f"\n  {mname}:")
            print(f"    β(t) ≈ {r['beta_0']:.3f} + {r['beta_1']:.3f}·log(t)")
            print(f"    p = {r['p_log']:.4f}")
            if r.get('half_life'):
                print(f"    Half-life ≈ {r['half_life']:.1f} years")

    # ================================================================
    # PART 2: RECOVERY HR
    # ================================================================
    recovery = compute_recovery_hr(df_artists, df_events)

    print("\n\nDONE.")


if __name__ == '__main__':
    main()
