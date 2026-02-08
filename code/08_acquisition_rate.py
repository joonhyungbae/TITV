"""
12_network_churn_rate.py
──────────────────────────────────────────────────────────────────────
Network Churn Rate Analysis
(Section 4.7 — Triangulation: Why Prestige Diversity Was Rejected)

Purpose:
  Explain why Hypothesis 3 (prestige diversity) was rejected.
  Demonstrate that the loss of *dynamism* in partner acquisition,
  rather than prestige level, is the mechanism linking network
  rigidity to career plateaus.

Variables:
  - Acquisition_Rate: fraction of institutions visited for the first
    time in a given year.
  - Retention_Rate: fraction of institutions previously visited.
  - (Acquisition_Rate + Retention_Rate = 1.0)

Analyses:
  1. Cox PH models with Acquisition_Rate and Career Year interaction.
  2. Visualization: Acquisition Rate decline → Plateau Hazard.
──────────────────────────────────────────────────────────────────────
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines import CoxTimeVaryingFitter
from scipy.stats import norm, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    build_person_year_panel, SIGNIFICANT_EVENT_TYPES,
    detect_plateau, CENSOR_YEAR
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'reference')


# ============================================================
# Acquisition / Retention Rate Computation
# ============================================================

def compute_churn_rates(df_events, df_artists, censor_year=CENSOR_YEAR):
    """
    Compute Acquisition Rate and Retention Rate per artist-year.

    Acquisition_Rate = (number of first-time institutions) / (total institutions visited)
    Retention_Rate   = (number of returning institutions) / (total institutions visited)
    """
    records = []

    for _, row in df_artists.iterrows():
        aid = row['artist_id']
        cs = row['career_start_year']
        if pd.isna(cs):
            continue
        cs_int = int(cs)

        artist_events = df_events[
            (df_events['artist_id'] == aid) &
            (df_events['year'] <= censor_year)
        ].copy()

        if len(artist_events) == 0:
            continue

        # Only events with institution names
        artist_events = artist_events[artist_events['institution_en'].notna()].copy()
        if len(artist_events) == 0:
            continue

        # Process year by year
        years = sorted(artist_events['year'].unique())
        institutions_seen = set()  # Lifetime cumulative institutions visited

        for year in years:
            if year < cs_int:
                # Include pre-career events in cumulative tracking
                year_events = artist_events[artist_events['year'] == year]
                year_institutions = set(year_events['institution_en'].unique())
                institutions_seen.update(year_institutions)
                continue

            career_year = year - cs_int
            year_events = artist_events[artist_events['year'] == year]
            year_institutions = set(year_events['institution_en'].unique())

            if len(year_institutions) == 0:
                institutions_seen.update(year_institutions)
                continue

            # New institutions = those not seen in any previous year
            new_institutions = year_institutions - institutions_seen
            returning_institutions = year_institutions & institutions_seen

            n_total = len(year_institutions)
            n_new = len(new_institutions)
            n_returning = len(returning_institutions)

            acquisition_rate = n_new / n_total if n_total > 0 else 0.0
            retention_rate = n_returning / n_total if n_total > 0 else 0.0

            records.append({
                'artist_id': aid,
                'year': year,
                'career_year': career_year,
                'n_institutions_this_year': n_total,
                'n_new_institutions': n_new,
                'n_returning_institutions': n_returning,
                'acquisition_rate': acquisition_rate,
                'retention_rate': retention_rate,
                'cumulative_unique_institutions': len(institutions_seen) + n_new,
            })

            # Update cumulative set
            institutions_seen.update(year_institutions)

    return pd.DataFrame(records)


def build_churn_panel(df_artists, df_events, cutpoint=10, censor_year=CENSOR_YEAR):
    """
    Merge Acquisition/Retention Rate into the base person-year panel.
    """
    # Build base panel
    panel = build_person_year_panel(
        df_artists, df_events, cutpoint=cutpoint,
        censor_year=censor_year
    )

    # Compute churn rates
    churn_df = compute_churn_rates(df_events, df_artists, censor_year=censor_year)

    # Merge
    panel = panel.merge(
        churn_df[['artist_id', 'year', 'acquisition_rate', 'retention_rate',
                  'n_institutions_this_year', 'n_new_institutions',
                  'n_returning_institutions', 'cumulative_unique_institutions']],
        on=['artist_id', 'year'],
        how='left'
    )

    # Years with no events: acquisition_rate = 0
    panel['acquisition_rate'] = panel['acquisition_rate'].fillna(0.0)
    panel['retention_rate'] = panel['retention_rate'].fillna(0.0)
    panel['n_institutions_this_year'] = panel['n_institutions_this_year'].fillna(0)

    return panel


# ============================================================
# Main Analysis
# ============================================================

def main():
    print("=" * 70)
    print("Network Churn Rate Analysis")
    print("=" * 70)

    # --- Data loading and panel construction ---
    print("\n[1] Loading data and building panel...")
    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)

    panel = build_churn_panel(df_artists, df_events, cutpoint=10)

    print(f"  Panel: {len(panel)} person-years, {panel['artist_id'].nunique()} artists, "
          f"{int(panel['event'].sum())} plateau events")

    # --- Descriptive statistics ---
    print("\n[2] Acquisition Rate / Retention Rate Descriptive Statistics")
    # Only years with institutional participation
    active_panel = panel[panel['n_institutions_this_year'] > 0].copy()
    print(f"  Active years (with institutional participation): {len(active_panel)} person-years")

    for phase_name, phase_val in [('Overall', None), ('Early career (0-9yr)', 0), ('Late career (10yr+)', 1)]:
        if phase_val is not None:
            sub = active_panel[active_panel['post_cutpoint'] == phase_val]
        else:
            sub = active_panel
        print(f"\n  [{phase_name}]")
        print(f"    Acquisition Rate: mean={sub['acquisition_rate'].mean():.3f}, "
              f"median={sub['acquisition_rate'].median():.3f}, "
              f"sd={sub['acquisition_rate'].std():.3f}")
        print(f"    Retention Rate:   mean={sub['retention_rate'].mean():.3f}, "
              f"median={sub['retention_rate'].median():.3f}, "
              f"sd={sub['retention_rate'].std():.3f}")

    # Acquisition Rate trend by career year
    print("\n  Mean Acquisition Rate by career year bin:")
    career_bins = [0, 5, 10, 15, 20, 25, 30, 100]
    labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30+']
    active_panel['career_bin'] = pd.cut(active_panel['career_year'], bins=career_bins, labels=labels, right=False)
    acq_by_bin = active_panel.groupby('career_bin')['acquisition_rate'].agg(['mean', 'std', 'count'])
    for idx, row in acq_by_bin.iterrows():
        print(f"    {idx}: {row['mean']:.3f} (sd={row['std']:.3f}, n={int(row['count'])})")

    # --- Correlations ---
    print("\n[3] Correlations: Acquisition Rate vs. Network Variables")
    corr_vars = ['acquisition_rate', 'network_stability', 'network_size',
                 'career_year', 'cumulative_validation']
    corr_data = panel[corr_vars].dropna()
    for var in corr_vars[1:]:
        rho, p = spearmanr(corr_data['acquisition_rate'], corr_data[var])
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  Acquisition Rate vs {var}: rho={rho:.3f}{sig} (p={p:.4f})")

    # --- Standardization ---
    print("\n[4] Building Cox PH models...")
    scaler = StandardScaler()
    for col in ['network_stability', 'network_size', 'career_year',
                'birth_year', 'cumulative_validation', 'acquisition_rate']:
        panel[f'{col}_z'] = scaler.fit_transform(panel[[col]]).flatten()

    # Interaction terms
    panel['acq_x_caryr'] = panel['acquisition_rate_z'] * panel['career_year_z']
    panel['stab_x_caryr'] = panel['network_stability_z'] * panel['career_year_z']
    panel['id'] = panel['artist_id']

    # ========================================
    # Model 1: Acquisition Rate main effect
    # ========================================
    print("\n--- Model 1: Acquisition Rate Main Effect ---")
    formula1 = ('acquisition_rate_z + network_size_z + career_year_z + '
                'birth_year_z + cumulative_validation_z + has_overseas')
    ctv1 = CoxTimeVaryingFitter(penalizer=0.01)
    ctv1.fit(panel, id_col='id', event_col='event',
             start_col='start', stop_col='stop',
             formula=formula1, show_progress=False)
    print(ctv1.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%',
                         'exp(coef) upper 95%', 'p']].round(4).to_string())

    # ========================================
    # Model 2: Acquisition Rate + Career Year interaction
    # ========================================
    print("\n--- Model 2: Acquisition Rate x Career Year Interaction ---")
    formula2 = ('acquisition_rate_z + network_size_z + career_year_z + '
                'acq_x_caryr + birth_year_z + cumulative_validation_z + has_overseas')
    ctv2 = CoxTimeVaryingFitter(penalizer=0.01)
    ctv2.fit(panel, id_col='id', event_col='event',
             start_col='start', stop_col='stop',
             formula=formula2, show_progress=False)
    s2 = ctv2.summary
    print(s2[['coef', 'exp(coef)', 'exp(coef) lower 95%',
              'exp(coef) upper 95%', 'p']].round(4).to_string())

    acq_main = s2.loc['acquisition_rate_z']
    acq_int = s2.loc['acq_x_caryr']
    print(f"\n  Acquisition Rate main effect: HR={acq_main['exp(coef)']:.3f}, p={acq_main['p']:.4f}")
    print(f"  Acquisition Rate x Career Year: HR={acq_int['exp(coef)']:.3f}, p={acq_int['p']:.4f}")

    # ========================================
    # Model 3: Stability + Acquisition Rate (competition model)
    # ========================================
    print("\n--- Model 3: Stability + Acquisition Rate Competition Model ---")
    formula3 = ('acquisition_rate_z + network_stability_z + network_size_z + career_year_z + '
                'acq_x_caryr + stab_x_caryr + birth_year_z + cumulative_validation_z + has_overseas')
    ctv3 = CoxTimeVaryingFitter(penalizer=0.01)
    ctv3.fit(panel, id_col='id', event_col='event',
             start_col='start', stop_col='stop',
             formula=formula3, show_progress=False)
    s3 = ctv3.summary
    print(s3[['coef', 'exp(coef)', 'exp(coef) lower 95%',
              'exp(coef) upper 95%', 'p']].round(4).to_string())

    # ========================================
    # Conditional HR Computation: Acquisition Rate x Career Year
    # ========================================
    print("\n[5] Conditional HR of Acquisition Rate by Career Year")

    beta_acq = s2.loc['acquisition_rate_z', 'coef']
    beta_int = s2.loc['acq_x_caryr', 'coef']
    se_acq = s2.loc['acquisition_rate_z', 'se(coef)']
    se_int = s2.loc['acq_x_caryr', 'se(coef)']

    var_matrix = ctv2.variance_matrix_
    acq_idx = list(var_matrix.index).index('acquisition_rate_z')
    int_idx = list(var_matrix.index).index('acq_x_caryr')
    cov_acq_int = var_matrix.iloc[acq_idx, int_idx]

    career_year_mean = panel['career_year'].mean()
    career_year_std = panel['career_year'].std()

    # Continuous career year grid
    career_years_cont = np.arange(0, 31, 0.5)
    hrs_cont = []
    hr_lows_cont = []
    hr_highs_cont = []
    pvals_cont = []

    for cy in career_years_cont:
        cy_z = (cy - career_year_mean) / career_year_std
        cond_coef = beta_acq + beta_int * cy_z
        cond_se = np.sqrt(se_acq**2 + (cy_z**2) * se_int**2 + 2 * cy_z * cov_acq_int)
        hrs_cont.append(np.exp(cond_coef))
        hr_lows_cont.append(np.exp(cond_coef - 1.96 * cond_se))
        hr_highs_cont.append(np.exp(cond_coef + 1.96 * cond_se))
        pvals_cont.append(2 * (1 - norm.cdf(abs(cond_coef / cond_se))))

    hrs_cont = np.array(hrs_cont)
    hr_lows_cont = np.array(hr_lows_cont)
    hr_highs_cont = np.array(hr_highs_cont)
    pvals_cont = np.array(pvals_cont)

    # Print key career years
    conditional_hrs = []
    for cy in [0, 5, 10, 15, 20, 25]:
        cy_z = (cy - career_year_mean) / career_year_std
        cond_coef = beta_acq + beta_int * cy_z
        cond_se = np.sqrt(se_acq**2 + (cy_z**2) * se_int**2 + 2 * cy_z * cov_acq_int)
        cond_hr = np.exp(cond_coef)
        cond_hr_lo = np.exp(cond_coef - 1.96 * cond_se)
        cond_hr_hi = np.exp(cond_coef + 1.96 * cond_se)
        cond_p = 2 * (1 - norm.cdf(abs(cond_coef / cond_se)))
        conditional_hrs.append({
            'career_year': cy, 'HR': cond_hr,
            'HR_lower': cond_hr_lo, 'HR_upper': cond_hr_hi, 'p': cond_p
        })
        sig = '***' if cond_p < 0.001 else ('**' if cond_p < 0.01 else ('*' if cond_p < 0.05 else ''))
        print(f"  Year {cy:2d}: HR={cond_hr:.3f} [{cond_hr_lo:.3f}, {cond_hr_hi:.3f}], p={cond_p:.4f}{sig}")

    df_cond_hrs = pd.DataFrame(conditional_hrs)

    # ========================================
    # Empirical mean Acquisition Rate by career year
    # ========================================
    acq_by_career_year = panel.groupby('career_year').agg(
        mean_acq_rate=('acquisition_rate', 'mean'),
        std_acq_rate=('acquisition_rate', 'std'),
        n=('acquisition_rate', 'count')
    ).reset_index()
    acq_by_career_year['se'] = acq_by_career_year['std_acq_rate'] / np.sqrt(acq_by_career_year['n'])

    # ========================================
    # Visualization
    # ========================================
    print("\n[6] Generating figures...")
    os.makedirs(FIG_DIR, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # --- Panel A: Acquisition Rate trend over career years ---
    ax1 = fig.add_subplot(gs[0, 0])
    valid = acq_by_career_year[acq_by_career_year['career_year'] <= 30]
    ax1.fill_between(valid['career_year'],
                     valid['mean_acq_rate'] - 1.96 * valid['se'],
                     valid['mean_acq_rate'] + 1.96 * valid['se'],
                     alpha=0.2, color='#2196F3')
    ax1.plot(valid['career_year'], valid['mean_acq_rate'],
             color='#2196F3', linewidth=2.5, label='Acquisition Rate')
    ax1.axvline(x=10, color='red', linestyle=':', linewidth=1, alpha=0.6, label='Year 10 cutpoint')
    ax1.set_xlabel('Career Year', fontsize=12)
    ax1.set_ylabel('Mean Acquisition Rate', fontsize=12)
    ax1.set_title('(A) Acquisition Rate Decline Over Career', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Annotations: early vs late career
    early_mean = active_panel[active_panel['career_year'] <= 5]['acquisition_rate'].mean()
    late_mean = active_panel[active_panel['career_year'] >= 15]['acquisition_rate'].mean()
    ax1.annotate(f'Early (0-5yr)\n{early_mean:.2f}',
                 xy=(2.5, early_mean), xytext=(5, 0.9),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'),
                 color='#1565C0', fontweight='bold')
    ax1.annotate(f'Late (15yr+)\n{late_mean:.2f}',
                 xy=(20, late_mean), xytext=(22, 0.6),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'),
                 color='#B71C1C', fontweight='bold')

    # --- Panel B: Conditional HR of Acquisition Rate ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(career_years_cont, hr_lows_cont, hr_highs_cont,
                     alpha=0.2, color='#E91E63')
    ax2.plot(career_years_cont, hrs_cont,
             color='#E91E63', linewidth=2.5, label='Acquisition Rate HR')
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
    ax2.axvline(x=10, color='red', linestyle=':', linewidth=1, alpha=0.6, label='Year 10 cutpoint')

    # Mark significant regions
    sig_mask = pvals_cont < 0.05
    sig_regions = career_years_cont[sig_mask]
    if len(sig_regions) > 0:
        # Find contiguous significant spans
        breaks = np.where(np.diff(sig_regions) > 1)[0]
        starts = [sig_regions[0]] + [sig_regions[b+1] for b in breaks]
        ends = [sig_regions[b] for b in breaks] + [sig_regions[-1]]
        for s, e in zip(starts, ends):
            ax2.axvspan(s, e, alpha=0.08, color='red')

    ax2.set_xlabel('Career Year', fontsize=12)
    ax2.set_ylabel('Hazard Ratio (per 1 SD Acquisition Rate)', fontsize=12)
    ax2.set_title('(B) Acquisition Rate Effect on Plateau Hazard\n'
                  '(Conditional on Career Year)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.set_xlim(0, 30)
    ax2.grid(True, alpha=0.3)

    # Annotations
    idx0 = 0
    idx15 = np.argmin(np.abs(career_years_cont - 15))
    idx25 = np.argmin(np.abs(career_years_cont - 25))

    for idx, cy_label, offset_y in [(idx0, 'Year 0', 0.15), (idx15, 'Year 15', -0.15)]:
        p_val = pvals_cont[idx]
        sig_str = f"(p={p_val:.3f})" if p_val >= 0.001 else "(p<0.001)"
        ax2.annotate(f'{cy_label}\nHR={hrs_cont[idx]:.2f}\n{sig_str}',
                     xy=(career_years_cont[idx], hrs_cont[idx]),
                     xytext=(career_years_cont[idx] + 3, hrs_cont[idx] + offset_y),
                     fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'),
                     color='#880E4F')

    # --- Panel C: Model Comparison Forest Plot ---
    ax3 = fig.add_subplot(gs[1, 0])

    # Model 1 (Acquisition only), Model 2 (Acq + Interaction), Model 3 (Acq + Stability competition)
    models_data = [
        ('Model 1:\nAcq. Rate only', ctv1, 'acquisition_rate_z'),
        ('Model 2:\nAcq. × Career Yr', ctv2, 'acquisition_rate_z'),
        ('Model 3:\nAcq. + Stability', ctv3, 'acquisition_rate_z'),
    ]

    y_positions = list(range(len(models_data)))
    for i, (label, model, var_name) in enumerate(models_data):
        s = model.summary.loc[var_name]
        hr = s['exp(coef)']
        hr_lo = s['exp(coef) lower 95%']
        hr_hi = s['exp(coef) upper 95%']
        p = s['p']
        color = '#E91E63' if p < 0.05 else 'gray'

        ax3.errorbar(hr, i, xerr=[[hr - hr_lo], [hr_hi - hr]],
                     fmt='o', color=color, markersize=8, capsize=4, linewidth=2)
        stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        ax3.text(hr_hi + 0.02, i, f'HR={hr:.3f}{stars}', va='center', fontsize=9, color=color)

    ax3.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8)
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels([m[0] for m in models_data], fontsize=10)
    ax3.set_xlabel('Hazard Ratio (Acquisition Rate)', fontsize=12)
    ax3.set_title('(C) Acquisition Rate Effect Across Models', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()

    # --- Panel D: Stability vs Acquisition Rate Competition (Model 3) ---
    ax4 = fig.add_subplot(gs[1, 1])

    competition_vars = []
    var_labels = {
        'acquisition_rate_z': 'Acquisition Rate',
        'network_stability_z': 'Network Stability',
        'acq_x_caryr': 'Acq. Rate × Career Yr',
        'stab_x_caryr': 'Stability × Career Yr',
        'network_size_z': 'Network Size',
        'career_year_z': 'Career Year',
        'cumulative_validation_z': 'Cumulative Validation',
    }

    for var in ['acquisition_rate_z', 'network_stability_z', 'acq_x_caryr',
                'stab_x_caryr', 'network_size_z', 'career_year_z',
                'cumulative_validation_z']:
        if var in s3.index:
            row = s3.loc[var]
            competition_vars.append({
                'variable': var_labels.get(var, var),
                'HR': row['exp(coef)'],
                'HR_lower': row['exp(coef) lower 95%'],
                'HR_upper': row['exp(coef) upper 95%'],
                'p': row['p'],
                'coef': row['coef'],
            })

    comp_df = pd.DataFrame(competition_vars)
    y_pos = list(range(len(comp_df)))

    for i, (_, row) in enumerate(comp_df.iterrows()):
        color = '#E91E63' if row['p'] < 0.05 else '#9E9E9E'
        marker = 's' if 'Acquisition' in row['variable'] else 'o'
        ax4.errorbar(row['HR'], i,
                     xerr=[[row['HR'] - row['HR_lower']], [row['HR_upper'] - row['HR']]],
                     fmt=marker, color=color, markersize=7, capsize=3, linewidth=1.5)
        stars = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01 else ('*' if row['p'] < 0.05 else ''))
        ax4.text(max(row['HR_upper'] + 0.02, row['HR'] + 0.05), i,
                 f'{row["HR"]:.3f}{stars}', va='center', fontsize=8, color=color)

    ax4.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(comp_df['variable'], fontsize=9)
    ax4.set_xlabel('Hazard Ratio', fontsize=12)
    ax4.set_title('(D) Competition Model: Acquisition Rate vs Stability',
                  fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()

    plt.suptitle('Network Churn Rate Analysis:\n'
                 'Why Prestige Diversity (H3) Was Rejected',
                 fontsize=15, fontweight='bold', y=1.02)

    out_path = os.path.join(FIG_DIR, 'network_churn_rate_analysis.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  Figure saved: {out_path}")
    plt.close()

    # ========================================
    # Additional visualization: Acquisition Rate decline and Plateau Hazard
    # ========================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Dual-axis plot (Acquisition Rate decline + Plateau Hazard)
    ax_left = axes2[0]
    ax_right = ax_left.twinx()

    # Acquisition Rate trend (left axis)
    valid_acq = acq_by_career_year[acq_by_career_year['career_year'] <= 30]
    ax_left.plot(valid_acq['career_year'], valid_acq['mean_acq_rate'],
                 color='#2196F3', linewidth=2.5, label='Acquisition Rate (left)')
    ax_left.fill_between(valid_acq['career_year'],
                         valid_acq['mean_acq_rate'] - 1.96 * valid_acq['se'],
                         valid_acq['mean_acq_rate'] + 1.96 * valid_acq['se'],
                         alpha=0.15, color='#2196F3')
    ax_left.set_ylabel('Mean Acquisition Rate', fontsize=12, color='#2196F3')
    ax_left.tick_params(axis='y', labelcolor='#2196F3')

    # Plateau Hazard (right axis) - Conditional HR
    ax_right.plot(career_years_cont, hrs_cont,
                  color='#E91E63', linewidth=2.5, linestyle='--',
                  label='Plateau Hazard HR (right)')
    ax_right.fill_between(career_years_cont, hr_lows_cont, hr_highs_cont,
                          alpha=0.1, color='#E91E63')
    ax_right.set_ylabel('Hazard Ratio (per 1 SD Acq. Rate)', fontsize=12, color='#E91E63')
    ax_right.tick_params(axis='y', labelcolor='#E91E63')
    ax_right.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    ax_left.axvline(x=10, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax_left.set_xlabel('Career Year', fontsize=12)
    ax_left.set_title('(A) Acquisition Rate Decline & Plateau Hazard',
                      fontsize=13, fontweight='bold')
    ax_left.set_xlim(0, 30)

    # Combine legends
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)
    ax_left.grid(True, alpha=0.3)

    # Panel B: Scatter plot (Career Year vs Acquisition Rate, by plateau status)
    ax_scatter = axes2[1]

    # Plateau onset year data
    plateau_events = panel[panel['event'] == 1].copy()
    non_plateau = panel[panel['event'] == 0].copy()

    # Subsample to avoid overplotting
    if len(non_plateau) > 2000:
        non_plateau_sample = non_plateau.sample(2000, random_state=42)
    else:
        non_plateau_sample = non_plateau

    ax_scatter.scatter(non_plateau_sample['career_year'],
                       non_plateau_sample['acquisition_rate'],
                       alpha=0.15, s=15, color='#90CAF9', label='Non-plateau years')
    ax_scatter.scatter(plateau_events['career_year'],
                       plateau_events['acquisition_rate'],
                       alpha=0.7, s=40, color='#E91E63', marker='^',
                       edgecolors='white', linewidths=0.5,
                       label=f'Plateau onset (n={len(plateau_events)})')

    # Loess-like smoothing (rolling mean)
    for label_name, data, color, ls in [
        ('Non-plateau (smoothed)', non_plateau, '#2196F3', '-'),
        ('Plateau onset (smoothed)', plateau_events, '#E91E63', '--')
    ]:
        if len(data) > 10:
            grouped = data.groupby('career_year')['acquisition_rate'].mean()
            valid_idx = grouped.index[grouped.index <= 30]
            smoothed = grouped.loc[valid_idx].rolling(3, center=True, min_periods=1).mean()
            ax_scatter.plot(smoothed.index, smoothed.values,
                           color=color, linewidth=2, linestyle=ls, label=label_name)

    ax_scatter.set_xlabel('Career Year', fontsize=12)
    ax_scatter.set_ylabel('Acquisition Rate', fontsize=12)
    ax_scatter.set_title('(B) Acquisition Rate: Plateau vs Non-Plateau',
                         fontsize=13, fontweight='bold')
    ax_scatter.legend(loc='upper right', fontsize=9)
    ax_scatter.set_xlim(0, 30)
    ax_scatter.set_ylim(-0.05, 1.1)
    ax_scatter.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = os.path.join(FIG_DIR, 'acquisition_rate_plateau_hazard.png')
    fig2.savefig(out_path2, dpi=300, bbox_inches='tight')
    print(f"  Figure saved: {out_path2}")
    plt.close()

    # ========================================
    # Save results
    # ========================================
    print("\n[7] Saving results...")

    # Model summaries
    model_results = []
    for model_name, model in [('Model 1: Acq Only', ctv1),
                               ('Model 2: Acq × Career Yr', ctv2),
                               ('Model 3: Acq + Stability', ctv3)]:
        s = model.summary
        for var in s.index:
            row = s.loc[var]
            model_results.append({
                'model': model_name,
                'variable': var,
                'coef': row['coef'],
                'HR': row['exp(coef)'],
                'HR_lower': row['exp(coef) lower 95%'],
                'HR_upper': row['exp(coef) upper 95%'],
                'se': row['se(coef)'],
                'p': row['p'],
            })


    # ========================================
    # LaTeX Table
    # ========================================
    print("\n" + "=" * 70)
    print("LATEX TABLE: Network Churn Rate Analysis")
    print("=" * 70)

    var_display = {
        'acquisition_rate_z': 'Acquisition rate',
        'network_stability_z': 'Network stability',
        'network_size_z': 'Network size',
        'career_year_z': 'Career year',
        'acq_x_caryr': 'Acquisition rate $\\times$ Career year',
        'stab_x_caryr': 'Stability $\\times$ Career year',
        'birth_year_z': 'Birth year',
        'cumulative_validation_z': 'Cumulative validation',
        'has_overseas': 'Overseas experience',
    }

    latex = r"""\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{Network Churn Rate: Cox Proportional Hazards Models}
\label{tab:churn_rate}
\small
\begin{tabular}{l ccc}
\toprule
 & \textbf{Model 1} & \textbf{Model 2} & \textbf{Model 3} \\
 & Acquisition Only & Acq. $\times$ Career & Competition \\
\midrule
"""

    all_vars = ['acquisition_rate_z', 'network_stability_z', 'network_size_z',
                'career_year_z', 'acq_x_caryr', 'stab_x_caryr',
                'birth_year_z', 'cumulative_validation_z', 'has_overseas']

    model_summaries = [ctv1.summary, ctv2.summary, ctv3.summary]

    for var in all_vars:
        display_name = var_display.get(var, var)
        row_str = f"{display_name}"
        for ms in model_summaries:
            if var in ms.index:
                r = ms.loc[var]
                hr = r['exp(coef)']
                lo = r['exp(coef) lower 95%']
                hi = r['exp(coef) upper 95%']
                p = r['p']
                stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                row_str += f" & {hr:.3f}{stars} [{lo:.3f}, {hi:.3f}]"
            else:
                row_str += " & ---"
        row_str += r" \\"
        latex += row_str + "\n"

    latex += r"""\addlinespace
\multicolumn{4}{l}{\textit{Conditional Acquisition Rate HR by career year (Model 2):}} \\
"""
    for _, r in df_cond_hrs.iterrows():
        stars = '***' if r['p'] < 0.001 else ('**' if r['p'] < 0.01 else ('*' if r['p'] < 0.05 else ''))
        p_str = f"$<$0.001" if r['p'] < 0.001 else f"{r['p']:.3f}"
        latex += f"\\quad Year {int(r['career_year'])} & & {r['HR']:.3f}{stars} [{r['HR_lower']:.3f}, {r['HR_upper']:.3f}] & \\\\\n"

    n_py = len(panel)
    n_events = int(panel['event'].sum())
    n_artists = panel['artist_id'].nunique()

    latex += r"""\addlinespace
\midrule
""" + f"Observations & {n_py:,} & {n_py:,} & {n_py:,}" + r""" \\
""" + f"Events (plateaus) & {n_events} & {n_events} & {n_events}" + r""" \\
""" + f"Artists & {n_artists} & {n_artists} & {n_artists}" + r""" \\
"""

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} $^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$.
Cox time-varying models. All continuous covariates standardized ($z$-scores).
Acquisition rate = proportion of institutions visited for the first time in each year.
Penalizer $\lambda = 0.01$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    print(latex)


    # ========================================
    # Key conclusions
    # ========================================
    print("\n" + "=" * 70)
    print("Key Conclusions (Interpretation)")
    print("=" * 70)
    print(f"""
1. Acquisition Rate (proportion of new institutions) declines sharply
   from an early-career mean of {early_mean:.2f} to {late_mean:.2f} in later career years.

2. Cox model results:
   - Model 2 main effect (Acquisition Rate): HR={acq_main['exp(coef)']:.3f}, p={acq_main['p']:.4f}
   - Interaction (Acq x Career Year): HR={acq_int['exp(coef)']:.3f}, p={acq_int['p']:.4f}

3. This explains why Hypothesis 3 (prestige diversity) was rejected:
   - It is not the *level* of prestige but the *dynamism* of acquiring
     new partners that disappears in later career stages, increasing
     plateau risk.
   - Institutional novelty matters more than institutional quality.
""")

    print("DONE.")
    return {
        'panel': panel,
        'model1': ctv1, 'model2': ctv2, 'model3': ctv3,
        'conditional_hrs': df_cond_hrs,
        'acq_by_career_year': acq_by_career_year,
    }


if __name__ == '__main__':
    main()
