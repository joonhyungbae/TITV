"""
12_institutional_logic_analysis.py
──────────────────────────────────────────────────────────────────────
Institutional Logic Analysis
— Commercial Gallery vs. Symbolic Museum —

Purpose:
  Investigate whether the *site* of entrenchment matters: does
  confinement to commercial circuits drive plateau risk more than
  confinement to symbolic institutions?

Analysis Pipeline:
  (1) Classify events as Commercial / Symbolic based on event_type
      and institution_type.
  (2) Construct per-artist, per-year Commercial_Ratio variable.
  (3) Cox time-varying model with Network_Stability × Commercial_Ratio
      interaction.
  (4) Compare conditional hazard ratios for commercially- vs.
      symbolically-entrenched artists.

Classification Rules:
  Commercial = gallery, corporate, private_museum exhibitions
  Symbolic   = public_museum, biennale, award, honor, collection
──────────────────────────────────────────────────────────────────────
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, ttest_ind
from collections import Counter
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import (
    load_raw_data, extract_artist_info, extract_events,
    detect_plateau, compute_network_size_stability,
    compute_event_weight, build_person_year_panel,
    CENSOR_YEAR, SIGNIFICANT_EVENT_TYPES,
    ORIGINAL_TYPE_WEIGHTS,
)

# ============================================================
# Paths
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR  = os.path.join(os.path.dirname(__file__), '..', 'figures', 'reference')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# (1) Institutional Logic Classification: Commercial vs Symbolic
# ============================================================
# Classification rules:
#   Commercial = exhibitions (solo/group) at gallery, corporate,
#                or private_museum institutions
#   Symbolic   = public_museum events, biennale, award, honor,
#                collection
#   Neutral    = education, position, residency, other (unclassified)

def classify_event_logic(event_type, institution_type):
    """
    Classify an event as Commercial, Symbolic, or Neutral.

    Returns: 'commercial', 'symbolic', or 'neutral'
    """
    et = event_type if event_type else 'other'
    it = institution_type if institution_type else 'other'
    
    # ---- Symbolic capital ----
    # 1. All public museum events are symbolic
    if it == 'public_museum':
        return 'symbolic'
    
    # 2. Biennale events are symbolic
    if et == 'biennale':
        return 'symbolic'
    
    # 3. Awards and honors are symbolic
    if et in ('award', 'honor'):
        return 'symbolic'
    
    # 4. Collections (museum acquisitions) are symbolic
    if et == 'collection':
        return 'symbolic'
    
    # ---- Commercial capital ----
    # 1. Gallery exhibitions are commercial
    if it == 'gallery' and et in ('solo_exhibition', 'group_exhibition'):
        return 'commercial'
    
    # 2. Corporate-sponsored exhibitions are commercial
    if it == 'corporate' and et in ('solo_exhibition', 'group_exhibition'):
        return 'commercial'
    
    # 3. Private museum exhibitions classified as commercial
    if it == 'private_museum' and et in ('solo_exhibition', 'group_exhibition'):
        return 'commercial'
    
    # ---- Neutral (unclassified) ----
    # education, position, residency, other, etc.
    return 'neutral'


# ============================================================
# (2) Person-Year Panel with Commercial_Ratio
# ============================================================
def build_panel_with_commercial_ratio(df_artists, df_events, censor_year=CENSOR_YEAR):
    """
    Build person-year panel augmented with Commercial_Ratio.
    Commercial_Ratio = cumulative commercial events / (cumulative commercial + symbolic events)
    """
    # Add logic classification to events
    df_events = df_events.copy()
    df_events['logic'] = df_events.apply(
        lambda r: classify_event_logic(r['event_type'], r['institution_type']),
        axis=1
    )
    
    # Pre-compute plateau info per artist
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
                    ev['event_type'], ORIGINAL_TYPE_WEIGHTS
                )
            
            # Commercial Ratio (cumulative)
            logic_counts = events_up_to['logic'].value_counts()
            n_commercial = logic_counts.get('commercial', 0)
            n_symbolic = logic_counts.get('symbolic', 0)
            total_classifiable = n_commercial + n_symbolic
            commercial_ratio = n_commercial / total_classifiable if total_classifiable > 0 else 0.5
            
            # 5-year rolling window commercial ratio
            window_events = artist_events[
                (artist_events['year'] >= year - 4) &
                (artist_events['year'] <= year)
            ]
            wl = window_events['logic'].value_counts()
            w_com = wl.get('commercial', 0)
            w_sym = wl.get('symbolic', 0)
            w_total = w_com + w_sym
            commercial_ratio_5yr = w_com / w_total if w_total > 0 else 0.5
            
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
                'network_size': max(net_size, 0.5),
                'network_stability': net_stability,
                'cumulative_validation': cum_val,
                'post_cutpoint': 1 if career_year >= 10 else 0,
                'birth_year': birth_year,
                'has_overseas': has_overseas,
                'commercial_ratio': commercial_ratio,
                'commercial_ratio_5yr': commercial_ratio_5yr,
                'n_commercial': n_commercial,
                'n_symbolic': n_symbolic,
            }
            records.append(rec)
    
    df = pd.DataFrame(records)
    
    # Fill missing birth_year with median
    if 'birth_year' in df.columns:
        median_by = df['birth_year'].median()
        df['birth_year'] = df['birth_year'].fillna(median_by)
    
    return df, df_events


# ============================================================
# Color palette
# ============================================================
COLORS = {
    'commercial': '#E74C3C',     # Red — commercial circuit
    'symbolic':   '#2980B9',     # Blue — symbolic/institutional
    'mixed':      '#8E44AD',     # Purple — mixed
    'neutral':    '#95A5A6',     # Gray
    'accent1':    '#27AE60',     # Green
    'accent2':    '#F39C12',     # Orange
    'dark':       '#2C3E50',     # Dark blue-gray
}


def main():
    print("=" * 70)
    print("Institutional Logic Analysis")
    print("Commercial Gallery vs Symbolic Museum")
    print("=" * 70)
    
    # ============================================================
    # Load data
    # ============================================================
    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)
    
    print(f"\n  Total artists: {len(df_artists)}")
    print(f"  Total events: {len(df_events)}")
    
    # ============================================================
    # (1) Event Classification Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: Institutional Logic Classification (Commercial vs Symbolic)")
    print("=" * 70)
    
    df_events['logic'] = df_events.apply(
        lambda r: classify_event_logic(r['event_type'], r['institution_type']),
        axis=1
    )
    
    logic_dist = df_events['logic'].value_counts()
    print("\n  Event classification summary:")
    for logic, cnt in logic_dist.items():
        pct = cnt / len(df_events) * 100
        print(f"    {logic:12s}: {cnt:5d} ({pct:.1f}%)")
    
    # Detailed breakdown by classification
    print("\n  --- Commercial events detail ---")
    commercial = df_events[df_events['logic'] == 'commercial']
    print(f"    event_type distribution:")
    for et, cnt in commercial['event_type'].value_counts().items():
        print(f"      {et:20s}: {cnt:5d}")
    print(f"    institution_type distribution:")
    for it, cnt in commercial['institution_type'].value_counts().items():
        print(f"      {str(it):20s}: {cnt:5d}")
    
    print("\n  --- Symbolic events detail ---")
    symbolic = df_events[df_events['logic'] == 'symbolic']
    print(f"    event_type distribution:")
    for et, cnt in symbolic['event_type'].value_counts().items():
        print(f"      {et:20s}: {cnt:5d}")
    print(f"    institution_type distribution:")
    for it, cnt in symbolic['institution_type'].value_counts().items():
        print(f"      {str(it):20s}: {cnt:5d}")
    
    # ============================================================
    # (2) Person-Year Panel with Commercial_Ratio
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2: Person-Year Panel Construction (with Commercial_Ratio)")
    print("=" * 70)
    
    panel, df_events_classified = build_panel_with_commercial_ratio(
        df_artists, df_events
    )
    
    print(f"\n  Panel size: {len(panel):,} person-year observations")
    print(f"  Artists: {panel['artist_id'].nunique()}")
    print(f"  Plateau events: {panel['event'].sum()}")
    
    print(f"\n  Commercial_Ratio descriptive statistics:")
    print(f"    Mean: {panel['commercial_ratio'].mean():.3f}")
    print(f"    Std:  {panel['commercial_ratio'].std():.3f}")
    print(f"    Min:  {panel['commercial_ratio'].min():.3f}")
    print(f"    Max:  {panel['commercial_ratio'].max():.3f}")
    print(f"    Q25:  {panel['commercial_ratio'].quantile(0.25):.3f}")
    print(f"    Q50:  {panel['commercial_ratio'].quantile(0.50):.3f}")
    print(f"    Q75:  {panel['commercial_ratio'].quantile(0.75):.3f}")
    
    # ============================================================
    # (3) Cox Model: Network_Stability × Commercial_Ratio
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 3: Cox Time-Varying Model")
    print("Network_Stability × Commercial_Ratio Interaction")
    print("=" * 70)
    
    # Standardize variables
    scaler = StandardScaler()
    for col in ['network_stability', 'network_size', 'cumulative_validation',
                'birth_year', 'commercial_ratio', 'commercial_ratio_5yr']:
        panel[f'{col}_z'] = scaler.fit_transform(panel[[col]]).flatten()
    
    # Create interaction terms
    panel['stab_x_comratio'] = panel['network_stability_z'] * panel['commercial_ratio_z']
    panel['stab_x_comratio5'] = panel['network_stability_z'] * panel['commercial_ratio_5yr_z']
    panel['id'] = panel['artist_id']
    
    # --- Model 1: Main effects only ---
    print("\n  --- Model 1: Main Effects Only ---")
    formula_m1 = ('network_stability_z + network_size_z + '
                  'commercial_ratio_z + '
                  'cumulative_validation_z + birth_year_z + has_overseas')
    
    ctv_m1 = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_m1.fit(panel, id_col='id', event_col='event',
               start_col='start', stop_col='stop',
               formula=formula_m1, show_progress=False)
    
    print(ctv_m1.summary.to_string())
    
    # --- Model 2: With Interaction ---
    print("\n\n  --- Model 2: With Stability × Commercial_Ratio Interaction ---")
    formula_m2 = ('network_stability_z + network_size_z + '
                  'commercial_ratio_z + stab_x_comratio + '
                  'cumulative_validation_z + birth_year_z + has_overseas')
    
    ctv_m2 = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_m2.fit(panel, id_col='id', event_col='event',
               start_col='start', stop_col='stop',
               formula=formula_m2, show_progress=False)
    
    print(ctv_m2.summary.to_string())
    
    # --- Model 3: Phase-split (Early vs Late career) ---
    print("\n\n  --- Model 3: Phase-Split (Career Year < 10 vs ≥ 10) ---")
    
    results_phase = {}
    for phase_name, phase_val in [('Early (< 10yr)', 0), ('Late (≥ 10yr)', 1)]:
        subset = panel[panel['post_cutpoint'] == phase_val].copy()
        if len(subset) == 0 or subset['event'].sum() == 0:
            print(f"    {phase_name}: insufficient data")
            continue
        
        subset['id'] = subset['artist_id']
        ctv_ph = CoxTimeVaryingFitter(penalizer=0.01)
        try:
            ctv_ph.fit(subset, id_col='id', event_col='event',
                       start_col='start', stop_col='stop',
                       formula=formula_m2, show_progress=False)
            results_phase[phase_name] = ctv_ph
            print(f"\n    {phase_name}:")
            print(ctv_ph.summary.to_string())
        except Exception as e:
            print(f"    {phase_name} failed: {e}")
    
    # --- Model 4: 5-year rolling commercial ratio ---
    print("\n\n  --- Model 4: 5-year Rolling Commercial Ratio ---")
    formula_m4 = ('network_stability_z + network_size_z + '
                  'commercial_ratio_5yr_z + stab_x_comratio5 + '
                  'cumulative_validation_z + birth_year_z + has_overseas')
    
    ctv_m4 = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_m4.fit(panel, id_col='id', event_col='event',
               start_col='start', stop_col='stop',
               formula=formula_m4, show_progress=False)
    
    print(ctv_m4.summary.to_string())
    
    # ============================================================
    # (4) Commercial vs Symbolic Entrenchment: Hazard Ratio Comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 4: Commercial vs Symbolic Entrenchment HR Comparison")
    print("=" * 70)
    
    # Classify artists by Commercial_Ratio at plateau onset
    # Use commercial_ratio from last observed person-year
    artist_last_obs = panel.groupby('artist_id').last().reset_index()
    plateau_artists = artist_last_obs[artist_last_obs['event'] == 1].copy()
    non_plateau = artist_last_obs[artist_last_obs['event'] == 0].copy()
    
    # Median split for commercial/symbolic entrenchment
    median_cr = plateau_artists['commercial_ratio'].median()
    plateau_artists['entrenchment_type'] = plateau_artists['commercial_ratio'].apply(
        lambda x: 'Commercial Entrenchment' if x > median_cr else 'Symbolic Entrenchment'
    )
    
    # Tercile split for finer analysis
    q33 = plateau_artists['commercial_ratio'].quantile(1/3)
    q66 = plateau_artists['commercial_ratio'].quantile(2/3)
    plateau_artists['entrenchment_tercile'] = plateau_artists['commercial_ratio'].apply(
        lambda x: 'High Commercial' if x > q66 else ('Mixed' if x > q33 else 'High Symbolic')
    )
    
    print(f"\n  Plateau artists: {len(plateau_artists)}")
    print(f"  Non-plateau artists: {len(non_plateau)}")
    print(f"  Median Commercial Ratio (plateau): {median_cr:.3f}")
    print(f"  Q33: {q33:.3f}, Q66: {q66:.3f}")
    
    print("\n  --- Entrenchment Type Distribution ---")
    for etype in ['Commercial Entrenchment', 'Symbolic Entrenchment']:
        sub = plateau_artists[plateau_artists['entrenchment_type'] == etype]
        print(f"    {etype}: n={len(sub)}, "
              f"avg_CR={sub['commercial_ratio'].mean():.3f}, "
              f"avg_stability={sub['network_stability'].mean():.2f}, "
              f"avg_career_year={sub['career_year'].mean():.1f}")
    
    print("\n  --- Entrenchment Tercile Distribution ---")
    for trc in ['High Symbolic', 'Mixed', 'High Commercial']:
        sub = plateau_artists[plateau_artists['entrenchment_tercile'] == trc]
        print(f"    {trc:20s}: n={len(sub)}, "
              f"avg_CR={sub['commercial_ratio'].mean():.3f}, "
              f"avg_stability={sub['network_stability'].mean():.2f}")
    
    # --- Conditional HRs from Model 2 ---
    s2 = ctv_m2.summary
    beta_stab = s2.loc['network_stability_z', 'coef']
    beta_cr = s2.loc['commercial_ratio_z', 'coef']
    beta_int = s2.loc['stab_x_comratio', 'coef']
    
    se_stab = s2.loc['network_stability_z', 'se(coef)']
    se_cr = s2.loc['commercial_ratio_z', 'se(coef)']
    se_int = s2.loc['stab_x_comratio', 'se(coef)']
    
    # Variance-covariance for CI computation
    var_matrix = ctv_m2.variance_matrix_
    idx_stab = list(var_matrix.index).index('network_stability_z')
    idx_cr = list(var_matrix.index).index('commercial_ratio_z')
    idx_int = list(var_matrix.index).index('stab_x_comratio')
    
    cov_stab_int = var_matrix.iloc[idx_stab, idx_int]
    cov_cr_int = var_matrix.iloc[idx_cr, idx_int]
    
    print(f"\n  --- Model 2 Key Coefficients ---")
    print(f"    Network_Stability: β={beta_stab:.4f} (SE={se_stab:.4f}), "
          f"HR={np.exp(beta_stab):.3f}, p={s2.loc['network_stability_z', 'p']:.4f}")
    print(f"    Commercial_Ratio:  β={beta_cr:.4f} (SE={se_cr:.4f}), "
          f"HR={np.exp(beta_cr):.3f}, p={s2.loc['commercial_ratio_z', 'p']:.4f}")
    print(f"    Interaction:       β={beta_int:.4f} (SE={se_int:.4f}), "
          f"HR={np.exp(beta_int):.3f}, p={s2.loc['stab_x_comratio', 'p']:.4f}")
    
    # Conditional effects of stability at different CR levels
    # At CR = -1SD (symbolic-heavy): total stab effect = beta_stab + beta_int * (-1)
    # At CR = 0 (average):           total stab effect = beta_stab
    # At CR = +1SD (commercial-heavy): total stab effect = beta_stab + beta_int * (+1)
    cr_levels = {'Symbolic-heavy\n(CR = -1 SD)': -1, 
                 'Average\n(CR = mean)': 0, 
                 'Commercial-heavy\n(CR = +1 SD)': 1}
    
    print(f"\n  --- Conditional Stability Effect at Different Commercial Ratios ---")
    for label, cr_z in cr_levels.items():
        cond_beta = beta_stab + beta_int * cr_z
        cond_se = np.sqrt(se_stab**2 + (cr_z**2) * se_int**2 + 2 * cr_z * cov_stab_int)
        cond_hr = np.exp(cond_beta)
        cond_hr_lo = np.exp(cond_beta - 1.96 * cond_se)
        cond_hr_hi = np.exp(cond_beta + 1.96 * cond_se)
        z_val = cond_beta / cond_se
        p_val = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(z_val)))
        stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
        label_clean = label.replace('\n', ' ')
        print(f"    {label_clean:35s}: HR={cond_hr:.3f}{stars} [{cond_hr_lo:.3f}, {cond_hr_hi:.3f}], "
              f"β={cond_beta:.4f}, p={p_val:.4f}")
    
    # ============================================================
    # FIGURE: 4-Panel Visualization
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 5: Visualization")
    print("=" * 70)
    
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                           left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    # ──────────────────────────────────────────────
    # Panel A: Commercial Ratio Distribution (Plateau vs Non-plateau)
    # ──────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Based on each artist's last observation
    bins = np.linspace(0, 1, 21)
    
    ax_a.hist(plateau_artists['commercial_ratio'], bins=bins, alpha=0.6,
              color=COLORS['commercial'], edgecolor='white', linewidth=0.8,
              density=True, label=f'Plateau (n={len(plateau_artists)})')
    ax_a.hist(non_plateau['commercial_ratio'], bins=bins, alpha=0.5,
              color=COLORS['symbolic'], edgecolor='white', linewidth=0.8,
              density=True, label=f'Non-plateau (n={len(non_plateau)})')
    
    # Median lines
    ax_a.axvline(plateau_artists['commercial_ratio'].mean(), color=COLORS['commercial'],
                 linestyle='--', linewidth=2, alpha=0.8)
    ax_a.axvline(non_plateau['commercial_ratio'].mean(), color=COLORS['symbolic'],
                 linestyle='--', linewidth=2, alpha=0.8)
    
    # Mann-Whitney test
    u_stat, p_mw = mannwhitneyu(
        plateau_artists['commercial_ratio'],
        non_plateau['commercial_ratio'],
        alternative='two-sided'
    )
    
    # Effect size (rank-biserial correlation)
    n1, n2 = len(plateau_artists), len(non_plateau)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)
    
    ax_a.set_xlabel('Commercial Ratio', fontsize=12)
    ax_a.set_ylabel('Density', fontsize=12)
    ax_a.set_title('A. Commercial Ratio Distribution\n'
                   'Plateau vs. Non-Plateau Artists', fontsize=13, fontweight='bold')
    ax_a.legend(fontsize=10, loc='upper right', framealpha=0.9)
    
    stars_mw = '***' if p_mw < 0.001 else ('**' if p_mw < 0.01 else ('*' if p_mw < 0.05 else 'n.s.'))
    ax_a.text(0.03, 0.85, f'Mann-Whitney U = {u_stat:.0f}\np = {p_mw:.4f} {stars_mw}\nr = {r_rb:.3f}',
              transform=ax_a.transAxes, fontsize=9,
              bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                       edgecolor='gray', alpha=0.9))
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    
    # ──────────────────────────────────────────────
    # Panel B: Conditional HR Surface Plot
    # Network_Stability effect at different Commercial_Ratio levels
    # ──────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    
    stab_range = np.linspace(-2, 2, 100)
    cr_z_vals = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
    cmap = plt.cm.RdBu_r
    
    for i, cr_z in enumerate(cr_z_vals):
        total_beta = beta_stab + beta_int * cr_z
        hr_line = np.exp(total_beta * stab_range)
        
        # SE for CI
        se_cond = np.sqrt(se_stab**2 + (cr_z**2) * se_int**2 + 2 * cr_z * cov_stab_int)
        hr_lo = np.exp((total_beta - 1.96 * se_cond) * stab_range)
        hr_hi = np.exp((total_beta + 1.96 * se_cond) * stab_range)
        
        color = cmap((cr_z + 2) / 4)  # normalize to [0,1]
        alpha = 0.9 if abs(cr_z) >= 1 else 0.5
        lw = 2.5 if abs(cr_z) >= 1 else 1.5
        
        label = f'CR z={cr_z:+.1f}'
        if cr_z == -1.5:
            label = 'Symbolic-heavy (z=-1.5)'
        elif cr_z == 0:
            label = 'Average (z=0)'
        elif cr_z == 1.5:
            label = 'Commercial-heavy (z=+1.5)'
        else:
            label = None
        
        ax_b.plot(stab_range, hr_line, color=color, linewidth=lw, alpha=alpha,
                  label=label)
        if abs(cr_z) in [1, 1.5]:
            ax_b.fill_between(stab_range, hr_lo, hr_hi, color=color, alpha=0.08)
    
    ax_b.axhline(y=1.0, color='#2C3E50', linestyle=':', linewidth=1.2, alpha=0.6)
    ax_b.set_xlabel('Network Stability (z-score)', fontsize=12)
    ax_b.set_ylabel('Plateau Hazard Ratio', fontsize=12)
    ax_b.set_title('B. Conditional Stability Effect on Plateau Hazard\n'
                   'by Commercial Ratio Level', fontsize=13, fontweight='bold')
    ax_b.legend(fontsize=9, loc='best', framealpha=0.9)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    
    # Annotation for interaction interpretation
    ax_b.annotate('Entrenchment\nzone (HR > 1)',
                  xy=(1.5, 1.0), xycoords='data',
                  xytext=(1.8, 0.6), textcoords='data',
                  arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2),
                  fontsize=8, color='#C0392B', ha='center',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='#FDEDEC',
                           edgecolor='#C0392B', alpha=0.6))
    
    # ──────────────────────────────────────────────
    # Panel C: Commercial vs Symbolic Entrenchment - Hazard Comparison
    # ──────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    
    # Conditional HR at tercile-specific CR values
    tercile_labels = ['High Symbolic\n(Bottom tercile)', 'Mixed\n(Middle tercile)', 
                      'High Commercial\n(Top tercile)']
    
    # Get the actual mean CR z-score for each tercile group
    # Map from panel (all observations) to get actual z-scores
    tercile_cr_means = []
    for trc_name, trc_label in [('High Symbolic', tercile_labels[0]),
                                ('Mixed', tercile_labels[1]),
                                ('High Commercial', tercile_labels[2])]:
        sub = plateau_artists[plateau_artists['entrenchment_tercile'] == trc_name]
        # Get mean commercial_ratio for this group
        mean_cr = sub['commercial_ratio'].mean()
        # Convert to z-score using panel statistics
        cr_mean_panel = panel['commercial_ratio'].mean()
        cr_std_panel = panel['commercial_ratio'].std()
        cr_z = (mean_cr - cr_mean_panel) / cr_std_panel
        tercile_cr_means.append(cr_z)
    
    # Compute conditional HR of stability for each group
    conditional_hrs = []
    conditional_cis = []
    for cr_z in tercile_cr_means:
        cond_beta = beta_stab + beta_int * cr_z
        cond_se = np.sqrt(se_stab**2 + (cr_z**2) * se_int**2 + 2 * cr_z * cov_stab_int)
        cond_hr = np.exp(cond_beta)
        cond_hr_lo = np.exp(cond_beta - 1.96 * cond_se)
        cond_hr_hi = np.exp(cond_beta + 1.96 * cond_se)
        conditional_hrs.append(cond_hr)
        conditional_cis.append((cond_hr_lo, cond_hr_hi))
    
    colors_tercile = [COLORS['symbolic'], COLORS['mixed'], COLORS['commercial']]
    x_pos = np.arange(len(tercile_labels))
    
    bars = ax_c.bar(x_pos, conditional_hrs,
                    color=colors_tercile, alpha=0.85,
                    edgecolor='white', linewidth=1.5, width=0.55)
    
    # Error bars
    for i in range(len(tercile_labels)):
        lo, hi = conditional_cis[i]
        ax_c.plot([i, i], [lo, hi], color='black', linewidth=2, zorder=3)
        ax_c.plot([i-0.08, i+0.08], [lo, lo], color='black', linewidth=1.5, zorder=3)
        ax_c.plot([i-0.08, i+0.08], [hi, hi], color='black', linewidth=1.5, zorder=3)
    
    # HR values on bars
    for i, (hr, (lo, hi)) in enumerate(zip(conditional_hrs, conditional_cis)):
        z_val = (np.log(hr)) / np.sqrt(se_stab**2 + (tercile_cr_means[i]**2) * se_int**2 + 
                                       2 * tercile_cr_means[i] * cov_stab_int)
        p_val = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(z_val)))
        stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
        ax_c.text(i, hi + 0.02, f'HR={hr:.3f}{stars}\n[{lo:.3f}, {hi:.3f}]',
                  ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax_c.axhline(y=1.0, color='#C0392B', linestyle=':', linewidth=1.5, alpha=0.7)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(tercile_labels, fontsize=10)
    ax_c.set_ylabel('Conditional Stability → Plateau HR', fontsize=12)
    ax_c.set_title('C. Network Stability Effect on Plateau Risk\n'
                   'by Artist\'s Commercial Ratio Tercile', fontsize=13, fontweight='bold')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    
    # Theory annotation
    ax_c.text(0.98, 0.05,
              'Higher bars = stronger\nstability → plateau link\n(entrenchment effect)',
              transform=ax_c.transAxes, ha='right', va='bottom',
              fontsize=8, fontstyle='italic',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.8))
    
    # ──────────────────────────────────────────────
    # Panel D: Forest Plot — Model Comparison
    # ──────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Collect key results from all models
    forest_data = []
    
    # Model 1: Main effects
    s1 = ctv_m1.summary
    for var in ['network_stability_z', 'commercial_ratio_z']:
        hr = s1.loc[var, 'exp(coef)']
        ci_lo = s1.loc[var, 'exp(coef) lower 95%']
        ci_hi = s1.loc[var, 'exp(coef) upper 95%']
        p = s1.loc[var, 'p']
        forest_data.append({
            'label': f'M1: {var.replace("_z", "")}',
            'hr': hr, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'p': p,
            'model': 'M1'
        })
    
    # Model 2: Interaction
    for var in ['network_stability_z', 'commercial_ratio_z', 'stab_x_comratio']:
        hr = s2.loc[var, 'exp(coef)']
        ci_lo = s2.loc[var, 'exp(coef) lower 95%']
        ci_hi = s2.loc[var, 'exp(coef) upper 95%']
        p = s2.loc[var, 'p']
        forest_data.append({
            'label': f'M2: {var.replace("_z", "")}',
            'hr': hr, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'p': p,
            'model': 'M2'
        })
    
    # Model 4: 5yr rolling
    s4 = ctv_m4.summary
    for var in ['network_stability_z', 'commercial_ratio_5yr_z', 'stab_x_comratio5']:
        hr = s4.loc[var, 'exp(coef)']
        ci_lo = s4.loc[var, 'exp(coef) lower 95%']
        ci_hi = s4.loc[var, 'exp(coef) upper 95%']
        p = s4.loc[var, 'p']
        clean_label = var.replace('_z', '').replace('5yr', '(5yr)')
        forest_data.append({
            'label': f'M4: {clean_label}',
            'hr': hr, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'p': p,
            'model': 'M4'
        })
    
    # Phase-split results
    for phase_name, ctv_ph in results_phase.items():
        if ctv_ph is None:
            continue
        s_ph = ctv_ph.summary
        for var in ['network_stability_z', 'stab_x_comratio']:
            if var not in s_ph.index:
                continue
            hr = s_ph.loc[var, 'exp(coef)']
            ci_lo = s_ph.loc[var, 'exp(coef) lower 95%']
            ci_hi = s_ph.loc[var, 'exp(coef) upper 95%']
            p = s_ph.loc[var, 'p']
            short_phase = 'Early' if 'Early' in phase_name else 'Late'
            clean_label = var.replace('_z', '')
            forest_data.append({
                'label': f'M3-{short_phase}: {clean_label}',
                'hr': hr, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'p': p,
                'model': f'M3-{short_phase}'
            })
    
    # Plot forest
    forest_data = forest_data[::-1]  # reverse for bottom-to-top
    
    model_colors = {
        'M1': '#34495E',
        'M2': COLORS['commercial'],
        'M4': COLORS['accent2'],
        'M3-Early': COLORS['accent1'],
        'M3-Late': COLORS['dark'],
    }
    
    for i, fd in enumerate(forest_data):
        color = model_colors.get(fd['model'], 'black')
        stars = '***' if fd['p'] < 0.001 else ('**' if fd['p'] < 0.01 else 
                ('*' if fd['p'] < 0.05 else ''))
        
        ax_d.errorbar(fd['hr'], i,
                      xerr=[[fd['hr'] - fd['ci_lo']], [fd['ci_hi'] - fd['hr']]],
                      marker='D', markersize=7, color=color, linewidth=1.8,
                      capsize=4, capthick=1.2)
        
        ax_d.text(max(fd['ci_hi'], fd['hr']) + 0.03, i,
                  f'{fd["hr"]:.3f}{stars}',
                  va='center', fontsize=8, fontweight='bold', color=color)
    
    ax_d.axvline(x=1.0, color='#C0392B', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_d.set_yticks(range(len(forest_data)))
    ax_d.set_yticklabels([fd['label'] for fd in forest_data], fontsize=9)
    ax_d.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
    ax_d.set_title('D. Forest Plot: Model Comparison\n'
                   'Stability × Commercial_Ratio Effects', fontsize=13, fontweight='bold')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    
    # Overall title
    fig.suptitle(
        'Institutional Logic Analysis: Commercial Entrenchment vs. Symbolic Network Effects\n'
        'on Career Plateau Risk in the Korean Art World',
        fontsize=15, fontweight='bold', y=0.98
    )
    
    fig_path = os.path.join(FIG_DIR, 'institutional_logic_analysis.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Figure saved: {fig_path}")
    
    # ============================================================
    # LaTeX Table
    # ============================================================
    print("\n" + "=" * 70)
    print("LaTeX TABLE: Cox Model Results")
    print("=" * 70)
    
    def format_hr_cell(summary_df, var):
        hr = summary_df.loc[var, 'exp(coef)']
        ci_lo = summary_df.loc[var, 'exp(coef) lower 95%']
        ci_hi = summary_df.loc[var, 'exp(coef) upper 95%']
        p = summary_df.loc[var, 'p']
        stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        return f"{hr:.3f}{stars} [{ci_lo:.3f}, {ci_hi:.3f}]"
    
    latex = r"""\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{Cox Time-Varying Models: Network Stability $\times$ Commercial Ratio Interaction}
\label{tab:institutional_logic}
\begin{tabular}{lcccc}
\toprule
 & \textbf{Model 1} & \textbf{Model 2} & \textbf{Model 3-Early} & \textbf{Model 3-Late} \\
 & Main Effects & Interaction & ($<$ 10 yr) & ($\geq$ 10 yr) \\
\midrule
"""
    
    key_vars = [
        ('network_stability_z', 'Network Stability (z)'),
        ('network_size_z', 'Network Size (z)'),
        ('commercial_ratio_z', 'Commercial Ratio (z)'),
        ('stab_x_comratio', 'Stability $\\times$ Comm. Ratio'),
        ('cumulative_validation_z', 'Cumulative Validation (z)'),
        ('birth_year_z', 'Birth Year (z)'),
        ('has_overseas', 'Overseas Experience'),
    ]
    
    for var, label in key_vars:
        row = f"{label}"
        
        # Model 1
        if var in s1.index:
            row += f" & {format_hr_cell(s1, var)}"
        else:
            row += " & ---"
        
        # Model 2
        if var in s2.index:
            row += f" & {format_hr_cell(s2, var)}"
        else:
            row += " & ---"
        
        # Model 3 Early
        if 'Early (< 10yr)' in results_phase and results_phase['Early (< 10yr)'] is not None:
            s_early = results_phase['Early (< 10yr)'].summary
            if var in s_early.index:
                row += f" & {format_hr_cell(s_early, var)}"
            else:
                row += " & ---"
        else:
            row += " & ---"
        
        # Model 3 Late
        if 'Late (≥ 10yr)' in results_phase and results_phase['Late (≥ 10yr)'] is not None:
            s_late = results_phase['Late (≥ 10yr)'].summary
            if var in s_late.index:
                row += f" & {format_hr_cell(s_late, var)}"
            else:
                row += " & ---"
        else:
            row += " & ---"
        
        row += r" \\"
        latex += row + "\n"
    
    # Add model fit statistics
    latex += r"""\addlinespace
\midrule
"""
    n_obs = f"Observations & {len(panel[panel['post_cutpoint']==0]) + len(panel[panel['post_cutpoint']==1])}"
    n_events_total = panel['event'].sum()
    
    latex += f"$N$ (person-years) & {len(panel):,} & {len(panel):,}"
    if 'Early (< 10yr)' in results_phase:
        n_early = len(panel[panel['post_cutpoint'] == 0])
        latex += f" & {n_early:,}"
    else:
        latex += " & ---"
    if 'Late (≥ 10yr)' in results_phase:
        n_late = len(panel[panel['post_cutpoint'] == 1])
        latex += f" & {n_late:,}"
    else:
        latex += " & ---"
    latex += r" \\" + "\n"
    
    latex += f"Events (plateaus) & {int(n_events_total)} & {int(n_events_total)}"
    if 'Early (< 10yr)' in results_phase:
        n_ev_e = panel[panel['post_cutpoint'] == 0]['event'].sum()
        latex += f" & {int(n_ev_e)}"
    else:
        latex += " & ---"
    if 'Late (≥ 10yr)' in results_phase:
        n_ev_l = panel[panel['post_cutpoint'] == 1]['event'].sum()
        latex += f" & {int(n_ev_l)}"
    else:
        latex += " & ---"
    latex += r" \\" + "\n"
    
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} HR [95\% CI] reported. *$p < 0.05$, **$p < 0.01$, ***$p < 0.001$.
All continuous predictors standardized (z-scored).
Commercial Ratio = proportion of commercial (gallery, corporate, private museum exhibitions)
vs.\ symbolic events (public museum, biennale, award, honor, collection) in cumulative career events.
Model 1: Main effects only. Model 2: Includes Stability $\times$ Commercial Ratio interaction.
Model 3: Phase-split (career year $<$ 10 vs.\ $\geq$ 10).
Penalizer = 0.01 for regularization.
\end{tablenotes}
\end{threeparttable}
\end{table}"""
    
    print(latex)

    # ============================================================
    # ARCHETYPE-SPECIFIC STABILITY ESTIMATES  (Section 4.4)
    # Paper: Late Recognition/Award HR=2.50, overseas HR=0.66,
    #        Wald chi2=7.60, df=4, p=0.107
    # ============================================================
    print("\n\n" + "=" * 70)
    print("ARCHETYPE-SPECIFIC STABILITY ESTIMATES")
    print("(Section 4.4: Career Archetype Heterogeneity)")
    print("=" * 70)

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler as SS2

    # Build archetype features per artist
    arch_feats = []
    for _, row in df_artists.iterrows():
        aid = row['artist_id']
        cs = row['career_start_year']
        if pd.isna(cs) or row['num_events'] == 0:
            continue
        a_ev = df_events[df_events['artist_id'] == aid]
        cl = a_ev['year'].max() - cs if len(a_ev) > 0 else 0
        n_tot = len(a_ev)
        n_solo = len(a_ev[a_ev['event_type'] == 'solo_exhibition'])
        n_award = len(a_ev[a_ev['event_type'] == 'award'])
        n_bien = len(a_ev[a_ev['event_type'] == 'biennale'])
        n_grp = len(a_ev[a_ev['event_type'] == 'group_exhibition'])
        solo_yr = a_ev[a_ev['event_type'] == 'solo_exhibition']['year']
        award_yr = a_ev[a_ev['event_type'] == 'award']['year']
        arch_feats.append({
            'artist_id': aid,
            'career_length': cl,
            'has_overseas': int(row.get('has_overseas', False)),
            'solo_share': n_solo / max(n_tot, 1),
            'award_share': n_award / max(n_tot, 1),
            'biennale_share': n_bien / max(n_tot, 1),
            'group_share': n_grp / max(n_tot, 1),
            'first_solo_rel': min((solo_yr.min() - cs) if len(solo_yr) > 0 else 40, 40),
            'first_award_rel': min((award_yr.min() - cs) if len(award_yr) > 0 else 40, 40),
            'n_total_events': n_tot,
        })

    arch_df = pd.DataFrame(arch_feats)
    fcols = ['career_length', 'has_overseas', 'solo_share', 'award_share',
             'biennale_share', 'group_share', 'first_solo_rel',
             'first_award_rel', 'n_total_events']
    X_arch = SS2().fit_transform(arch_df[fcols])
    km = KMeans(n_clusters=5, random_state=42, n_init=20)
    arch_df['cluster'] = km.fit_predict(X_arch)

    # Name clusters
    cnames = {}
    for cl_id in range(5):
        sub = arch_df[arch_df['cluster'] == cl_id]
        ov = sub['has_overseas'].mean()
        ar = sub['first_award_rel'].mean()
        ash = sub['award_share'].mean()
        clen = sub['career_length'].mean()
        if ov > 0.4:
            cnames[cl_id] = 'Overseas-Oriented'
        elif ar < 10 and ash > 0.05:
            cnames[cl_id] = 'Early Recognition'
        elif ar > 15 and ash > 0.03:
            cnames[cl_id] = 'Late Recognition/Award'
        elif clen < 5:
            cnames[cl_id] = 'Short Career'
        else:
            cnames[cl_id] = 'Exhibition-Focused'
    arch_df['archetype'] = arch_df['cluster'].map(cnames)

    for cl_id in sorted(cnames.keys()):
        sub = arch_df[arch_df['cluster'] == cl_id]
        print(f"  {cnames[cl_id]}: N={len(sub)}, career={sub['career_length'].mean():.1f}yr")

    # Archetype-specific Cox (post-decade)
    arch_df['artist_id'] = arch_df['artist_id'].astype(str)
    panel['artist_id'] = panel['artist_id'].astype(str)
    pa = panel.merge(arch_df[['artist_id', 'archetype']], on='artist_id', how='left')
    post_pa = pa[pa['post_cutpoint'] == 1].copy()
    sc_a = StandardScaler()
    for c in ['network_stability', 'network_size', 'birth_year', 'cumulative_validation']:
        post_pa[f'{c}_z'] = sc_a.fit_transform(post_pa[[c]]).flatten()
    post_pa['id'] = post_pa['artist_id']

    f_arch = 'network_stability_z + network_size_z + birth_year_z + cumulative_validation_z'
    arch_res = {}
    for aname in sorted(post_pa['archetype'].dropna().unique()):
        sub = post_pa[post_pa['archetype'] == aname].copy()
        ne = int(sub['event'].sum())
        if ne < 3:
            print(f"  {aname}: {ne} events — SKIPPED")
            continue
        try:
            ctv_a = CoxTimeVaryingFitter(penalizer=0.01)
            ctv_a.fit(sub, id_col='id', event_col='event',
                      start_col='start', stop_col='stop',
                      formula=f_arch, show_progress=False)
            sa = ctv_a.summary.loc['network_stability_z']
            arch_res[aname] = {'coef': sa['coef'], 'se': sa['se(coef)'],
                               'hr': sa['exp(coef)'], 'p': sa['p'],
                               'ci_lo': sa['exp(coef) lower 95%'],
                               'ci_hi': sa['exp(coef) upper 95%']}
            print(f"  {aname}: HR={sa['exp(coef)']:.3f} "
                  f"[{sa['exp(coef) lower 95%']:.3f},{sa['exp(coef) upper 95%']:.3f}] "
                  f"p={sa['p']:.3f} ({ne} events)")
        except Exception as e:
            print(f"  {aname}: FAILED — {e}")

    # Wald test
    if len(arch_res) >= 2:
        coefs = np.array([r['coef'] for r in arch_res.values()])
        ses = np.array([r['se'] for r in arch_res.values()])
        wt = 1.0 / (ses ** 2)
        pooled = np.sum(wt * coefs) / np.sum(wt)
        wald_chi2 = np.sum(((coefs - pooled) / ses) ** 2)
        wald_df = len(coefs) - 1
        from scipy.stats import chi2 as chi2_dist
        wald_p = 1 - chi2_dist.cdf(wald_chi2, wald_df)
        print(f"\n  Wald test: chi2={wald_chi2:.2f}, df={wald_df}, p={wald_p:.3f}")
        print(f"  >>> Paper: chi2=7.60, df=4, p=0.107 <<<")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n\n" + "=" * 70)
    print("COMPLETE SUMMARY")
    print("=" * 70)
    
    print(f"""
[1. Event Classification]
  - Commercial events: {len(commercial):,} ({len(commercial)/len(df_events)*100:.1f}%)
  - Symbolic events: {len(symbolic):,} ({len(symbolic)/len(df_events)*100:.1f}%)
  - Neutral events: {logic_dist.get('neutral', 0):,} ({logic_dist.get('neutral', 0)/len(df_events)*100:.1f}%)

[2. Commercial_Ratio]
  - Plateau artists mean: {plateau_artists['commercial_ratio'].mean():.3f}
  - Non-plateau artists mean: {non_plateau['commercial_ratio'].mean():.3f}
  - Mann-Whitney p = {p_mw:.4f}

[3. Cox Model Key Results (Model 2)]
  - Network_Stability: HR={np.exp(beta_stab):.3f}, p={s2.loc['network_stability_z', 'p']:.4f}
  - Commercial_Ratio: HR={np.exp(beta_cr):.3f}, p={s2.loc['commercial_ratio_z', 'p']:.4f}
  - Stability x Comm.Ratio Interaction: HR={np.exp(beta_int):.3f}, p={s2.loc['stab_x_comratio', 'p']:.4f}

[4. Conditional Stability Effects]
  - Symbolic-heavy (CR=-1SD): HR={np.exp(beta_stab + beta_int*(-1)):.3f}
  - Average (CR=mean): HR={np.exp(beta_stab):.3f}
  - Commercial-heavy (CR=+1SD): HR={np.exp(beta_stab + beta_int*(1)):.3f}

[5. Theoretical Interpretation]
  - If interaction coef > 0: commercially-entrenched artists show stronger
    stability-to-plateau link (commercial entrenchment effect)
  - If interaction coef < 0: symbolically-entrenched artists show stronger
    stability-to-plateau link (symbolic entrenchment effect)
  - If interaction is non-significant: network rigidity itself drives
    plateau risk regardless of institutional type (structural, not logic-specific)
""")
    
    print("\nDONE.")


if __name__ == '__main__':
    main()
