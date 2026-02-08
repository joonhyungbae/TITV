"""
02_continuous_hr_plot.py
Visualize how the stability HR changes continuously across career years.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines import CoxTimeVaryingFitter
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    build_person_year_panel
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

def main():
    print("Building panel and fitting model...")
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

    career_year_mean = panel['career_year'].mean()
    career_year_std = panel['career_year'].std()

    formula = ('network_stability_z + network_size_z + career_year_z + '
               'stab_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z')
    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(panel, id_col='id', event_col='event',
            start_col='start', stop_col='stop',
            formula=formula, show_progress=False)

    beta_stab = ctv.summary.loc['network_stability_z', 'coef']
    beta_int = ctv.summary.loc['stab_x_caryr', 'coef']
    se_stab = ctv.summary.loc['network_stability_z', 'se(coef)']
    se_int = ctv.summary.loc['stab_x_caryr', 'se(coef)']

    var_matrix = ctv.variance_matrix_
    # Columns may be integer-indexed; use positional lookup
    stab_idx = list(var_matrix.index).index('network_stability_z')
    int_idx = list(var_matrix.index).index('stab_x_caryr')
    cov_stab_int = var_matrix.iloc[stab_idx, int_idx]

    # Compute conditional HR across career years 0-30
    career_years = np.arange(0, 31, 0.5)
    hrs = []
    hr_lows = []
    hr_highs = []
    pvals = []

    for cy in career_years:
        cy_z = (cy - career_year_mean) / career_year_std
        cond_coef = beta_stab + beta_int * cy_z
        cond_se = np.sqrt(se_stab**2 + (cy_z**2) * se_int**2 + 2 * cy_z * cov_stab_int)
        hrs.append(np.exp(cond_coef))
        hr_lows.append(np.exp(cond_coef - 1.96 * cond_se))
        hr_highs.append(np.exp(cond_coef + 1.96 * cond_se))
        pvals.append(2 * (1 - norm.cdf(abs(cond_coef / cond_se))))

    hrs = np.array(hrs)
    hr_lows = np.array(hr_lows)
    hr_highs = np.array(hr_highs)
    pvals = np.array(pvals)

    # --- Plot ---
    C_MAIN = '#2C3E50'
    C_SIG = '#E74C3C'

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Significant region shading (draw first so it's behind everything)
    sig_mask = pvals < 0.05
    sig_years = career_years[sig_mask]
    if len(sig_years) > 0:
        ax.axvspan(sig_years.min(), sig_years.max(), alpha=0.08, color=C_SIG,
                   label=f'p < 0.05 (year {sig_years.min():.0f}+)')

    # CI band and main curve
    ax.fill_between(career_years, hr_lows, hr_highs, alpha=0.18, color=C_MAIN)
    ax.plot(career_years, hrs, color=C_MAIN, linewidth=3,
            label='Stability HR (per 1 SD)')

    # Reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axvline(x=10, color=C_SIG, linestyle=':', linewidth=1.5, alpha=0.5,
               label='Year 10 cutpoint')

    ax.set_xlabel('Career Year', fontsize=18)
    ax.set_ylabel('Hazard Ratio\n(per 1 SD stability)', fontsize=18)
    ax.legend(loc='upper left', fontsize=15, framealpha=0.95)
    ax.set_xlim(0, 30)
    ax.set_ylim(0.5, 2.2)
    ax.tick_params(labelsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotations — use reported Table 2 (Panel A) values for consistency
    # with the paper text; arrow targets use the model curve position.
    ax.annotate('HR = 1.04\n(n.s.)',
                xy=(0, hrs[career_years==0][0]), xytext=(3, 0.65),
                fontsize=15, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray', lw=2),
                color='gray',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='gray', alpha=0.9))
    idx10 = np.argmin(np.abs(career_years - 10))
    ax.annotate('HR = 1.20\n(p = 0.002)',
                xy=(10, hrs[idx10]), xytext=(13, 0.75),
                fontsize=15, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_MAIN, lw=2),
                color=C_MAIN,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=C_MAIN, alpha=0.9))
    idx20 = np.argmin(np.abs(career_years - 20))
    ax.annotate('HR = 1.39\n(p = 0.011)',
                xy=(20, hrs[idx20]), xytext=(22, 1.6),
                fontsize=15, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_MAIN, lw=2),
                color=C_MAIN,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=C_MAIN, alpha=0.9))

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, 'fig2_stability_hr.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {out_path}")
    plt.close()

    print("DONE.")

if __name__ == '__main__':
    main()
