"""
01_full_sample_interaction.py
Full-sample interaction model: career_year * network_stability
Solves the 37-event problem by using ALL 363 events instead of post-decade only.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines import CoxTimeVaryingFitter
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    build_person_year_panel
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def main():
    print("=" * 70)
    print("FULL-SAMPLE INTERACTION MODEL")
    print("=" * 70)

    # --- Load and build panel ---
    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)
    panel = build_person_year_panel(df_artists, df_events, cutpoint=10)

    print(f"Panel: {len(panel)} person-years, {panel['artist_id'].nunique()} artists, "
          f"{panel['event'].sum()} events")
    print(f"  Pre-decade events: {panel[panel['post_cutpoint']==0]['event'].sum()}")
    print(f"  Post-decade events: {panel[panel['post_cutpoint']==1]['event'].sum()}")

    # --- Standardize ---
    scaler = StandardScaler()
    for col in ['network_stability', 'network_size', 'career_year',
                'birth_year', 'cumulative_validation']:
        panel[f'{col}_z'] = scaler.fit_transform(panel[[col]]).flatten()

    # --- Create interaction terms ---
    panel['stab_x_caryr'] = panel['network_stability_z'] * panel['career_year_z']
    panel['size_x_caryr'] = panel['network_size_z'] * panel['career_year_z']
    panel['id'] = panel['artist_id']

    # --- Add lagged stability ---
    panel = panel.sort_values(['artist_id', 'career_year'])
    panel['network_stability_lag2'] = panel.groupby('artist_id')['network_stability'].shift(2)
    panel['network_stability_lag2_z'] = np.nan
    mask = panel['network_stability_lag2'].notna()
    if mask.sum() > 10:
        panel.loc[mask, 'network_stability_lag2_z'] = scaler.fit_transform(
            panel.loc[mask, ['network_stability_lag2']]).flatten()
    panel['stab_lag2_x_caryr'] = panel['network_stability_lag2_z'] * panel['career_year_z']

    results = {}

    # ========================================
    # Model 1: Full-sample interaction (primary)
    # ========================================
    print("\n--- Model 1: Full-Sample Interaction (Primary) ---")
    formula1 = ('network_stability_z + network_size_z + career_year_z + '
                'stab_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z')
    ctv1 = CoxTimeVaryingFitter(penalizer=0.01)
    ctv1.fit(panel, id_col='id', event_col='event',
             start_col='start', stop_col='stop',
             formula=formula1, show_progress=False)
    print(ctv1.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%',
                         'exp(coef) upper 95%', 'p']].round(4).to_string())
    results['model1'] = ctv1.summary.copy()

    # Key result: stab_x_caryr
    s = ctv1.summary
    stab_main = s.loc['network_stability_z']
    stab_int = s.loc['stab_x_caryr']
    print(f"\n  Stability main effect: HR={stab_main['exp(coef)']:.3f}, p={stab_main['p']:.4f}")
    print(f"  Stability x career_year interaction: HR={stab_int['exp(coef)']:.3f}, p={stab_int['p']:.4f}")
    print(f"  Interpretation: {'Phase reversal CONFIRMED' if stab_int['coef'] > 0 and stab_int['p'] < 0.05 else 'Phase reversal pattern present but interaction ' + ('significant' if stab_int['p'] < 0.05 else 'not significant at p<0.05')}")

    # --- Conditional HR at specific career years ---
    print("\n  Conditional stability HR by career year:")
    beta_stab = stab_main['coef']
    beta_int = stab_int['coef']
    se_stab = stab_main['se(coef)']
    se_int = stab_int['se(coef)']

    # Get covariance from variance matrix
    var_matrix = ctv1.variance_matrix_
    cov_stab_int = 0.0
    if 'network_stability_z' in var_matrix.index and 'stab_x_caryr' in var_matrix.columns:
        cov_stab_int = var_matrix.loc['network_stability_z', 'stab_x_caryr']

    career_year_mean = panel['career_year'].mean()
    career_year_std = panel['career_year'].std()

    conditional_hrs = []
    for cy in [0, 5, 10, 15, 20]:
        cy_z = (cy - career_year_mean) / career_year_std
        cond_coef = beta_stab + beta_int * cy_z
        cond_se = np.sqrt(se_stab**2 + (cy_z**2) * se_int**2 + 2 * cy_z * cov_stab_int)
        cond_hr = np.exp(cond_coef)
        cond_hr_lo = np.exp(cond_coef - 1.96 * cond_se)
        cond_hr_hi = np.exp(cond_coef + 1.96 * cond_se)
        cond_p = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(cond_coef / cond_se)))
        conditional_hrs.append({
            'career_year': cy, 'HR': cond_hr,
            'HR_lower': cond_hr_lo, 'HR_upper': cond_hr_hi, 'p': cond_p
        })
        print(f"    Year {cy:2d}: HR={cond_hr:.3f} [{cond_hr_lo:.3f}, {cond_hr_hi:.3f}], p={cond_p:.4f}")

    results['conditional_hrs'] = pd.DataFrame(conditional_hrs)

    # ========================================
    # Model 2: Lagged stability interaction
    # ========================================
    print("\n--- Model 2: Lagged Stability (t-2) Interaction ---")
    lag_data = panel.dropna(subset=['network_stability_lag2_z', 'stab_lag2_x_caryr']).copy()
    print(f"  Sample: {len(lag_data)} person-years, {lag_data['event'].sum()} events")

    formula2 = ('network_stability_lag2_z + network_size_z + career_year_z + '
                'stab_lag2_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z')
    ctv2 = CoxTimeVaryingFitter(penalizer=0.01)
    ctv2.fit(lag_data, id_col='id', event_col='event',
             start_col='start', stop_col='stop',
             formula=formula2, show_progress=False)
    s2 = ctv2.summary
    print(s2[['coef', 'exp(coef)', 'exp(coef) lower 95%',
              'exp(coef) upper 95%', 'p']].round(4).to_string())
    results['model2_lag'] = s2.copy()

    lag_main = s2.loc['network_stability_lag2_z']
    lag_int = s2.loc['stab_lag2_x_caryr']
    print(f"\n  Lag-2 stability main: HR={lag_main['exp(coef)']:.3f}, p={lag_main['p']:.4f}")
    print(f"  Lag-2 stability x career_year: HR={lag_int['exp(coef)']:.3f}, p={lag_int['p']:.4f}")

    # ========================================
    # Model 3: Phase-split (for comparison / robustness)
    # ========================================
    print("\n--- Model 3: Phase-Split (Robustness) ---")
    for phase_name, phase_val in [('Pre-decade', 0), ('Post-decade', 1)]:
        sub = panel[panel['post_cutpoint'] == phase_val].copy()
        n_events = sub['event'].sum()
        print(f"  {phase_name}: {len(sub)} person-years, {n_events} events")
        if n_events < 5:
            print(f"    Skipping (insufficient events)")
            continue

        formula3 = 'network_stability_z + network_size_z + birth_year_z + cumulative_validation_z'
        ctv3 = CoxTimeVaryingFitter(penalizer=0.01)
        ctv3.fit(sub, id_col='id', event_col='event',
                 start_col='start', stop_col='stop',
                 formula=formula3, show_progress=False)
        s3 = ctv3.summary
        stab = s3.loc['network_stability_z']
        size = s3.loc['network_size_z']
        print(f"    Stability: HR={stab['exp(coef)']:.3f} [{stab['exp(coef) lower 95%']:.3f}, "
              f"{stab['exp(coef) upper 95%']:.3f}], p={stab['p']:.4f}")
        print(f"    Size:      HR={size['exp(coef)']:.3f} [{size['exp(coef) lower 95%']:.3f}, "
              f"{size['exp(coef) upper 95%']:.3f}], p={size['p']:.4f}")
        results[f'phase_split_{phase_name}'] = s3.copy()

    # ========================================
    # LaTeX table output
    # ========================================
    print("\n\n" + "=" * 70)
    print("LATEX TABLE: Full-Sample Interaction Model")
    print("=" * 70)

    s = results['model1']
    latex = r"""
\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{Full-Sample Interaction Model: Network Effects on Career Plateau}
\label{tab:interaction}
\begin{tabular}{lccccc}
\toprule
Predictor & Coef. & \HR{} & 95\% \CI{} & \pval{} \\
\midrule
"""
    for var in s.index:
        row = s.loc[var]
        hr = row['exp(coef)']
        lo = row['exp(coef) lower 95%']
        hi = row['exp(coef) upper 95%']
        p = row['p']
        stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        p_str = f"$<$0.001" if p < 0.001 else f"{p:.3f}"
        var_clean = var.replace('_z', '').replace('_', ' ').replace('stab x caryr', 'Stability $\\times$ career year').replace('size x caryr', 'Size $\\times$ career year').replace('network stability', 'Network stability').replace('network size', 'Network size').replace('career year', 'Career year').replace('birth year', 'Birth year').replace('cumulative validation', 'Cumulative validation')
        latex += f"{var_clean} & {row['coef']:.3f} & {hr:.3f}{stars} & [{lo:.3f}, {hi:.3f}] & {p_str} \\\\\n"

    latex += r"""\addlinespace
\multicolumn{5}{l}{\textit{Conditional stability HR by career year:}} \\
"""
    for _, r in results['conditional_hrs'].iterrows():
        stars = '***' if r['p'] < 0.001 else ('**' if r['p'] < 0.01 else ('*' if r['p'] < 0.05 else ''))
        p_str = f"$<$0.001" if r['p'] < 0.001 else f"{r['p']:.3f}"
        latex += f"\\quad Year {int(r['career_year'])} & & {r['HR']:.3f}{stars} & [{r['HR_lower']:.3f}, {r['HR_upper']:.3f}] & {p_str} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} *\pval{} $< 0.05$, **\pval{} $< 0.01$, ***\pval{} $< 0.001$.
Time-varying Cox model on full sample (""" + f"{len(panel)}" + r""" person-years, """ + f"{int(panel['event'].sum())}" + r""" events).
All covariates standardized ($z$-scores). Penalizer $\lambda = 0.01$.
Conditional HR = exp($\beta_{\text{stability}} + \beta_{\text{interaction}} \times z_{\text{career year}}$).
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    print(latex)

    print("\nDONE.")

    return results

if __name__ == '__main__':
    main()
