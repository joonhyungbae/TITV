"""
05_plateau_sensitivity.py
Sensitivity analysis for plateau definition (3yr, 5yr, 7yr windows).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines import CoxTimeVaryingFitter
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    build_person_year_panel, detect_plateau, CENSOR_YEAR, SIGNIFICANT_EVENT_TYPES,
    ORIGINAL_TYPE_WEIGHTS, compute_event_weight,
    compute_network_size_stability
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def build_panel_with_window(df_artists, df_events, plateau_window=5):
    """Build person-year panel with custom plateau window."""
    plateau_info = {}
    for _, row in df_artists.iterrows():
        aid = row['artist_id']
        cs = row['career_start_year']
        if pd.isna(cs) or row['num_events'] == 0:
            continue
        occurred, p_year, ttp = detect_plateau(
            df_events, aid, int(cs), window=plateau_window, censor_year=CENSOR_YEAR
        )
        plateau_info[aid] = (occurred, p_year, ttp)

    records = []
    for _, row in df_artists.iterrows():
        aid = row['artist_id']
        cs = row['career_start_year']
        if pd.isna(cs) or aid not in plateau_info:
            continue

        cs_int = int(cs)
        occurred, p_year, _ = plateau_info[aid]
        end_year = int(p_year) if occurred else CENSOR_YEAR
        birth_year = row.get('birth_year')
        has_overseas = int(row.get('has_overseas', False))

        artist_events = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] <= end_year) &
            (df_events['year'] <= CENSOR_YEAR)
        ].copy()

        for year in range(cs_int, end_year + 1):
            career_year = year - cs_int
            events_up_to = artist_events[artist_events['year'] <= year]
            net_size, net_stability = compute_network_size_stability(events_up_to)
            cum_val = sum(
                compute_event_weight(ev['event_type'], ORIGINAL_TYPE_WEIGHTS)
                for _, ev in events_up_to.iterrows()
            )
            is_last_year = (year == end_year)
            event = 1 if (is_last_year and occurred) else 0

            records.append({
                'artist_id': aid, 'year': year, 'career_year': career_year,
                'start': career_year, 'stop': career_year + 1,
                'event': event,
                'network_size': max(net_size, 0.5),
                'network_stability': net_stability,
                'cumulative_validation': cum_val,
                'birth_year': birth_year,
            })

    df = pd.DataFrame(records)
    if 'birth_year' in df.columns:
        df['birth_year'] = df['birth_year'].fillna(df['birth_year'].median())
    return df


def main():
    print("=" * 70)
    print("PLATEAU DEFINITION SENSITIVITY (3yr, 5yr, 7yr)")
    print("=" * 70)

    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)

    results = []

    for window in [3, 5, 7]:
        print(f"\n--- Plateau window = {window} years ---")
        panel = build_panel_with_window(df_artists, df_events, plateau_window=window)
        n_events = panel['event'].sum()
        n_artists = panel['artist_id'].nunique()
        plateau_rate = panel.groupby('artist_id')['event'].max().mean()

        print(f"  Panel: {len(panel)} person-years, {n_artists} artists, {n_events} events")
        print(f"  Plateau rate: {plateau_rate:.1%}")

        # Standardize
        scaler = StandardScaler()
        for col in ['network_stability', 'network_size', 'career_year',
                    'birth_year', 'cumulative_validation']:
            panel[f'{col}_z'] = scaler.fit_transform(panel[[col]]).flatten()

        panel['stab_x_caryr'] = panel['network_stability_z'] * panel['career_year_z']
        panel['size_x_caryr'] = panel['network_size_z'] * panel['career_year_z']
        panel['id'] = panel['artist_id']

        career_year_mean = panel['career_year'].mean()
        career_year_std = panel['career_year'].std()

        # Fit full-sample interaction model
        formula = ('network_stability_z + network_size_z + career_year_z + '
                   'stab_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z')
        ctv = CoxTimeVaryingFitter(penalizer=0.01)
        ctv.fit(panel, id_col='id', event_col='event',
                start_col='start', stop_col='stop',
                formula=formula, show_progress=False)

        s = ctv.summary
        stab = s.loc['network_stability_z']
        interaction = s.loc['stab_x_caryr']
        size = s.loc['network_size_z']

        # Conditional HR at year 10
        beta_stab = stab['coef']
        beta_int = interaction['coef']
        se_stab = stab['se(coef)']
        se_int = interaction['se(coef)']
        vm = ctv.variance_matrix_
        stab_idx = list(vm.index).index('network_stability_z')
        int_idx = list(vm.index).index('stab_x_caryr')
        cov_si = vm.iloc[stab_idx, int_idx]
        cy10_z = (10 - career_year_mean) / career_year_std
        cond_coef = beta_stab + beta_int * cy10_z
        cond_se = np.sqrt(se_stab**2 + (cy10_z**2) * se_int**2 + 2 * cy10_z * cov_si)
        hr10 = np.exp(cond_coef)
        hr10_lo = np.exp(cond_coef - 1.96 * cond_se)
        hr10_hi = np.exp(cond_coef + 1.96 * cond_se)
        p10 = 2 * (1 - norm.cdf(abs(cond_coef / cond_se)))

        print(f"  Stability main: HR={stab['exp(coef)']:.3f}, p={stab['p']:.4f}")
        print(f"  Stab x career_year: HR={interaction['exp(coef)']:.3f}, p={interaction['p']:.4f}")
        print(f"  Size main: HR={size['exp(coef)']:.3f}, p={size['p']:.4f}")
        print(f"  Conditional HR at year 10: {hr10:.3f} [{hr10_lo:.3f}, {hr10_hi:.3f}], p={p10:.4f}")

        results.append({
            'window': window,
            'n_artists': n_artists,
            'n_events': int(n_events),
            'plateau_rate': plateau_rate,
            'stability_hr': stab['exp(coef)'],
            'stability_p': stab['p'],
            'interaction_hr': interaction['exp(coef)'],
            'interaction_p': interaction['p'],
            'size_hr': size['exp(coef)'],
            'size_p': size['p'],
            'hr_at_year10': hr10,
            'hr10_ci': f'[{hr10_lo:.3f}, {hr10_hi:.3f}]',
            'hr10_p': p10,
        })

    df_results = pd.DataFrame(results)
    print("\n\n=== SENSITIVITY SUMMARY ===")
    print(df_results.to_string(index=False))

    # LaTeX
    print("\n\n=== LATEX TABLE ===")
    print(r"""
\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{Sensitivity of Network Phase Reversal to Plateau Definition}
\label{tab:plateau_sensitivity}
\begin{tabular}{lcccccc}
\toprule
Window & Events & Plateau & Stability & Stab $\times$ year & Size & Cond.\ HR \\
(years) & & rate & \HR{} & \HR{} & \HR{} & (year 10) \\
\midrule""")
    for _, r in df_results.iterrows():
        stars_s = '***' if r['stability_p'] < 0.001 else ('**' if r['stability_p'] < 0.01 else ('*' if r['stability_p'] < 0.05 else ''))
        stars_i = '***' if r['interaction_p'] < 0.001 else ('**' if r['interaction_p'] < 0.01 else ('*' if r['interaction_p'] < 0.05 else ''))
        stars_sz = '***' if r['size_p'] < 0.001 else ('**' if r['size_p'] < 0.01 else ('*' if r['size_p'] < 0.05 else ''))
        stars_10 = '***' if r['hr10_p'] < 0.001 else ('**' if r['hr10_p'] < 0.01 else ('*' if r['hr10_p'] < 0.05 else ''))
        w = int(r['window'])
        bold = r'\\textbf{' if w == 5 else ''
        bold_end = '}' if w == 5 else ''
        print(f"{bold}{w}{bold_end} & {int(r['n_events'])} & {r['plateau_rate']:.1%} & "
              f"{r['stability_hr']:.3f}{stars_s} & {r['interaction_hr']:.3f}{stars_i} & "
              f"{r['size_hr']:.3f}{stars_sz} & {r['hr_at_year10']:.3f}{stars_10} \\\\")

    print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} *\pval{} $< 0.05$, **\pval{} $< 0.01$, ***\pval{} $< 0.001$.
Full-sample interaction Cox models with standardized covariates. Bold = primary specification.
\end{tablenotes}
\end{threeparttable}
\end{table}""")

    print("\nDONE.")

if __name__ == '__main__':
    main()
