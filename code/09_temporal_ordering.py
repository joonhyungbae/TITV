"""
07_lagged_iv_and_evalue.py
──────────────────────────────────────────────────────────────────────
(1) Lagged Independent Variable Strategy:
    Full-sample interaction model with t-1, t-2, t-3 lagged stability
    to establish temporal precedence (network → plateau, not reverse).
(2) E-value Calculation:
    Sensitivity analysis for unmeasured confounding (VanderWeele & Ding 2017).
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
    build_person_year_panel
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# ============================================================
# E-value computation (VanderWeele & Ding, 2017)
# ============================================================

def compute_evalue(hr):
    """
    Compute the E-value for a hazard ratio (or risk ratio).
    For HR >= 1: E = HR + sqrt(HR * (HR - 1))
    For HR < 1:  Convert to 1/HR, compute, report both.
    Returns (E_point, E_ci).
    """
    if hr < 1:
        hr = 1.0 / hr
    evalue = hr + np.sqrt(hr * (hr - 1))
    return evalue


def compute_evalue_with_ci(hr, ci_lower, ci_upper):
    """
    Compute E-value for point estimate and lower CI bound.
    The E-value for the CI bound uses the CI limit closest to 1.0.
    """
    e_point = compute_evalue(hr)

    # CI limit closest to null (1.0)
    if hr >= 1.0:
        ci_closest = ci_lower
        if ci_closest <= 1.0:
            e_ci = 1.0  # CI crosses null → E-value = 1
        else:
            e_ci = compute_evalue(ci_closest)
    else:
        ci_closest = ci_upper
        if ci_closest >= 1.0:
            e_ci = 1.0
        else:
            e_ci = compute_evalue(1.0 / ci_closest)

    return e_point, e_ci


# ============================================================
# Main analysis
# ============================================================

def main():
    print("=" * 70)
    print("LAGGED IV STRATEGY + E-VALUE SENSITIVITY ANALYSIS")
    print("=" * 70)

    # --- Load data and build panel ---
    artists_list = load_raw_data(os.path.join(DATA_DIR, 'data.json'))
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)
    panel = build_person_year_panel(df_artists, df_events, cutpoint=10)

    print(f"Base panel: {len(panel)} person-years, "
          f"{panel['artist_id'].nunique()} artists, "
          f"{int(panel['event'].sum())} events")

    # --- Sort and create lagged variables ---
    panel = panel.sort_values(['artist_id', 'career_year']).copy()
    for lag in [1, 2, 3]:
        panel[f'network_stability_lag{lag}'] = (
            panel.groupby('artist_id')['network_stability'].shift(lag)
        )
        panel[f'network_size_lag{lag}'] = (
            panel.groupby('artist_id')['network_size'].shift(lag)
        )

    # --- Standardize base variables ---
    scaler = StandardScaler()
    for col in ['network_stability', 'network_size', 'career_year',
                'birth_year', 'cumulative_validation']:
        panel[f'{col}_z'] = scaler.fit_transform(panel[[col]]).flatten()

    # Career year stats (for conditional HR computation)
    cy_mean = panel['career_year'].mean()
    cy_std = panel['career_year'].std()

    panel['id'] = panel['artist_id']

    # ================================================================
    # PART 1: LAGGED IV MODELS (Full-sample interaction)
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 1: LAGGED INDEPENDENT VARIABLE MODELS")
    print("=" * 70)

    results_lag = {}

    # --- Model 0: Contemporaneous (baseline) ---
    panel['stab_x_caryr'] = panel['network_stability_z'] * panel['career_year_z']
    panel['size_x_caryr'] = panel['network_size_z'] * panel['career_year_z']

    formula_base = ('network_stability_z + network_size_z + career_year_z + '
                    'stab_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z')

    print("\n--- Model 0: Contemporaneous (t) ---")
    ctv0 = CoxTimeVaryingFitter(penalizer=0.01)
    ctv0.fit(panel, id_col='id', event_col='event',
             start_col='start', stop_col='stop',
             formula=formula_base, show_progress=False)
    results_lag['t'] = extract_model_results(ctv0, panel, cy_mean, cy_std, 'Contemporaneous ($t$)')
    print_model_summary(results_lag['t'])

    # --- Models 1-3: Lagged stability ---
    for lag in [1, 2, 3]:
        print(f"\n--- Model {lag}: Lagged Stability (t-{lag}) ---")

        lag_col = f'network_stability_lag{lag}'
        lag_col_z = f'{lag_col}_z'
        lag_size_col = f'network_size_lag{lag}'
        lag_size_col_z = f'{lag_size_col}_z'

        # Standardize lagged variables within their non-missing subsample
        lag_data = panel.dropna(subset=[lag_col, lag_size_col]).copy()

        lag_data[lag_col_z] = scaler.fit_transform(lag_data[[lag_col]]).flatten()
        lag_data[lag_size_col_z] = scaler.fit_transform(lag_data[[lag_size_col]]).flatten()

        # Interaction terms with lagged variables
        lag_data[f'stab_lag{lag}_x_caryr'] = lag_data[lag_col_z] * lag_data['career_year_z']
        lag_data[f'size_lag{lag}_x_caryr'] = lag_data[lag_size_col_z] * lag_data['career_year_z']

        n_events = int(lag_data['event'].sum())
        print(f"  Sample: {len(lag_data)} person-years, {n_events} events")

        formula_lag = (f'{lag_col_z} + {lag_size_col_z} + career_year_z + '
                       f'stab_lag{lag}_x_caryr + size_lag{lag}_x_caryr + '
                       f'birth_year_z + cumulative_validation_z')

        ctv = CoxTimeVaryingFitter(penalizer=0.01)
        ctv.fit(lag_data, id_col='id', event_col='event',
                start_col='start', stop_col='stop',
                formula=formula_lag, show_progress=False)

        label = f'Lag-{lag} ($t-{lag}$)'
        results_lag[f't-{lag}'] = extract_model_results(
            ctv, lag_data, cy_mean, cy_std, label,
            stab_key=lag_col_z, int_key=f'stab_lag{lag}_x_caryr'
        )
        results_lag[f't-{lag}']['n_events'] = n_events
        results_lag[f't-{lag}']['n_personyears'] = len(lag_data)
        print_model_summary(results_lag[f't-{lag}'])

    # ================================================================
    # PART 1b: BURT CONSTRAINT (Section 4.4)
    # Paper: HR = 1.05, p = 0.617 — "does not show independent association"
    # ================================================================
    print("\n\n" + "=" * 70)
    print("PART 1b: BURT CONSTRAINT")
    print("(Paper: HR = 1.05, p = 0.617 once size and stability controlled)")
    print("=" * 70)

    # Rebuild panel with Burt constraint included
    panel_burt = build_person_year_panel(
        df_artists, df_events, cutpoint=10, include_constraint=True
    )
    panel_burt = panel_burt.sort_values(['artist_id', 'career_year']).copy()

    scaler_b = StandardScaler()
    for col in ['network_stability', 'network_size', 'career_year',
                'birth_year', 'cumulative_validation', 'network_constraint']:
        panel_burt[f'{col}_z'] = scaler_b.fit_transform(panel_burt[[col]]).flatten()

    panel_burt['stab_x_caryr'] = panel_burt['network_stability_z'] * panel_burt['career_year_z']
    panel_burt['size_x_caryr'] = panel_burt['network_size_z'] * panel_burt['career_year_z']
    panel_burt['id'] = panel_burt['artist_id']

    formula_burt = ('network_stability_z + network_size_z + career_year_z + '
                    'stab_x_caryr + size_x_caryr + network_constraint_z + '
                    'birth_year_z + cumulative_validation_z')

    ctv_burt = CoxTimeVaryingFitter(penalizer=0.01)
    ctv_burt.fit(panel_burt, id_col='id', event_col='event',
                 start_col='start', stop_col='stop',
                 formula=formula_burt, show_progress=False)

    s_burt = ctv_burt.summary
    constr_row = s_burt.loc['network_constraint_z']
    print(f"  Burt constraint: HR={constr_row['exp(coef)']:.3f} "
          f"[{constr_row['exp(coef) lower 95%']:.3f}, {constr_row['exp(coef) upper 95%']:.3f}], "
          f"p={constr_row['p']:.3f}")
    print(f"  >>> Paper reports: HR = 1.05, p = 0.617 <<<")

    # ================================================================
    # PART 2: E-VALUE SENSITIVITY ANALYSIS
    # ================================================================
    print("\n\n" + "=" * 70)
    print("PART 2: E-VALUE SENSITIVITY ANALYSIS")
    print("=" * 70)

    evalue_results = []

    # Key HRs to compute E-values for:
    key_hrs = [
        # From contemporaneous full-sample interaction model
        {'label': 'Stability main effect (contemporaneous)',
         'hr': results_lag['t']['stab_main_hr'],
         'ci_lo': results_lag['t']['stab_main_ci'][0],
         'ci_hi': results_lag['t']['stab_main_ci'][1]},
        # Conditional HR at year 10
        {'label': 'Conditional stability HR at year 10',
         'hr': results_lag['t']['conditional_hrs'][10]['hr'],
         'ci_lo': results_lag['t']['conditional_hrs'][10]['ci_lo'],
         'ci_hi': results_lag['t']['conditional_hrs'][10]['ci_hi']},
        # Conditional HR at year 15
        {'label': 'Conditional stability HR at year 15',
         'hr': results_lag['t']['conditional_hrs'][15]['hr'],
         'ci_lo': results_lag['t']['conditional_hrs'][15]['ci_lo'],
         'ci_hi': results_lag['t']['conditional_hrs'][15]['ci_hi']},
        # Conditional HR at year 20
        {'label': 'Conditional stability HR at year 20',
         'hr': results_lag['t']['conditional_hrs'][20]['hr'],
         'ci_lo': results_lag['t']['conditional_hrs'][20]['ci_lo'],
         'ci_hi': results_lag['t']['conditional_hrs'][20]['ci_hi']},
        # Network size (protective)
        {'label': 'Network size (protective)',
         'hr': results_lag['t']['size_hr'],
         'ci_lo': results_lag['t']['size_ci'][0],
         'ci_hi': results_lag['t']['size_ci'][1]},
        # Lag-2 stability main effect
        {'label': 'Lag-2 stability main effect',
         'hr': results_lag['t-2']['stab_main_hr'],
         'ci_lo': results_lag['t-2']['stab_main_ci'][0],
         'ci_hi': results_lag['t-2']['stab_main_ci'][1]},
        # Lag-2 conditional HR at year 10
        {'label': 'Lag-2 conditional stability HR at year 10',
         'hr': results_lag['t-2']['conditional_hrs'][10]['hr'],
         'ci_lo': results_lag['t-2']['conditional_hrs'][10]['ci_lo'],
         'ci_hi': results_lag['t-2']['conditional_hrs'][10]['ci_hi']},
    ]

    print(f"\n{'Finding':<50} {'HR':>6} {'E-point':>8} {'E-CI':>6}")
    print("-" * 75)

    for item in key_hrs:
        e_pt, e_ci = compute_evalue_with_ci(item['hr'], item['ci_lo'], item['ci_hi'])
        evalue_results.append({
            'Finding': item['label'],
            'HR': item['hr'],
            'CI_lower': item['ci_lo'],
            'CI_upper': item['ci_hi'],
            'E_point': e_pt,
            'E_ci': e_ci,
        })
        print(f"  {item['label']:<48} {item['hr']:6.3f} {e_pt:8.2f} {e_ci:6.2f}")

    evalue_df = pd.DataFrame(evalue_results)

    # ================================================================
    # PART 3: LaTeX OUTPUT
    # ================================================================
    print("\n\n" + "=" * 70)
    print("LATEX TABLE: Lagged IV Models")
    print("=" * 70)

    # Table: Lagged IV comparison
    latex_lag = generate_lag_table(results_lag)
    print(latex_lag)

    # Table: E-values
    print("\n" + "=" * 70)
    print("LATEX TABLE: E-value Sensitivity Analysis")
    print("=" * 70)
    latex_evalue = generate_evalue_table(evalue_df)
    print(latex_evalue)

    # Print summary for paper.tex paragraph
    print("\n\n" + "=" * 70)
    print("SUMMARY FOR PAPER INTEGRATION")
    print("=" * 70)

    print("\n[Lagged IV Key Results]")
    for key in ['t', 't-1', 't-2', 't-3']:
        r = results_lag[key]
        print(f"  {r['label']:25s}: Main stab HR={r['stab_main_hr']:.3f} (p={r['stab_main_p']:.4f}), "
              f"Interaction HR={r['int_hr']:.3f} (p={r['int_p']:.4f})")
        if 10 in r['conditional_hrs']:
            c = r['conditional_hrs'][10]
            print(f"    → Conditional HR@y10: {c['hr']:.3f} [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}], p={c['p']:.4f}")

    print("\n[E-value Key Results]")
    for _, row in evalue_df.iterrows():
        print(f"  {row['Finding']:50s}: E-point={row['E_point']:.2f}, E-CI={row['E_ci']:.2f}")

    print("\nDONE.")
    return results_lag, evalue_df


# ============================================================
# Helper functions
# ============================================================

def extract_model_results(ctv, data, cy_mean, cy_std, label,
                          stab_key='network_stability_z',
                          int_key='stab_x_caryr'):
    """Extract key results from a fitted CoxTimeVaryingFitter."""
    s = ctv.summary
    stab = s.loc[stab_key]
    interaction = s.loc[int_key]

    # Network size key (find it)
    size_keys = [k for k in s.index if 'network_size' in k and 'caryr' not in k]
    size_key = size_keys[0] if size_keys else None
    size_row = s.loc[size_key] if size_key else None

    result = {
        'label': label,
        'model': ctv,
        'summary': s,
        'stab_main_hr': stab['exp(coef)'],
        'stab_main_p': stab['p'],
        'stab_main_ci': (stab['exp(coef) lower 95%'], stab['exp(coef) upper 95%']),
        'stab_main_coef': stab['coef'],
        'stab_main_se': stab['se(coef)'],
        'int_hr': interaction['exp(coef)'],
        'int_p': interaction['p'],
        'int_coef': interaction['coef'],
        'int_se': interaction['se(coef)'],
        'size_hr': size_row['exp(coef)'] if size_row is not None else None,
        'size_p': size_row['p'] if size_row is not None else None,
        'size_ci': (size_row['exp(coef) lower 95%'], size_row['exp(coef) upper 95%']) if size_row is not None else (None, None),
    }

    # Covariance for conditional HR computation
    vm = ctv.variance_matrix_
    stab_idx = list(vm.index).index(stab_key)
    int_idx = list(vm.index).index(int_key)
    cov_stab_int = vm.iloc[stab_idx, int_idx]

    # Conditional HRs at specific career years
    cond_hrs = {}
    for cy in [0, 5, 10, 15, 20]:
        cy_z = (cy - cy_mean) / cy_std
        cond_coef = stab['coef'] + interaction['coef'] * cy_z
        cond_se = np.sqrt(
            stab['se(coef)']**2 +
            (cy_z**2) * interaction['se(coef)']**2 +
            2 * cy_z * cov_stab_int
        )
        cond_hr = np.exp(cond_coef)
        cond_hr_lo = np.exp(cond_coef - 1.96 * cond_se)
        cond_hr_hi = np.exp(cond_coef + 1.96 * cond_se)
        cond_p = 2 * (1 - stats.norm.cdf(abs(cond_coef / cond_se)))
        cond_hrs[cy] = {
            'hr': cond_hr, 'ci_lo': cond_hr_lo,
            'ci_hi': cond_hr_hi, 'p': cond_p
        }

    result['conditional_hrs'] = cond_hrs
    return result


def print_model_summary(r):
    """Print summary of a model result."""
    print(f"  Stability main: HR={r['stab_main_hr']:.3f} "
          f"[{r['stab_main_ci'][0]:.3f}, {r['stab_main_ci'][1]:.3f}], "
          f"p={r['stab_main_p']:.4f}")
    print(f"  Stab × career_year: HR={r['int_hr']:.3f}, p={r['int_p']:.4f}")
    if r['size_hr'] is not None:
        print(f"  Network size: HR={r['size_hr']:.3f} "
              f"[{r['size_ci'][0]:.3f}, {r['size_ci'][1]:.3f}], "
              f"p={r['size_p']:.4f}")
    print("  Conditional stability HR by career year:")
    for cy in [0, 5, 10, 15, 20]:
        c = r['conditional_hrs'][cy]
        sig = '***' if c['p'] < 0.001 else ('**' if c['p'] < 0.01 else ('*' if c['p'] < 0.05 else ''))
        print(f"    Year {cy:2d}: HR={c['hr']:.3f} [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}], "
              f"p={c['p']:.4f} {sig}")


def generate_lag_table(results_lag):
    """Generate LaTeX table comparing contemporaneous and lagged models."""

    def fmt_stars(p):
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        return ''

    def fmt_p(p):
        if p < 0.001: return '$<$0.001'
        return f'{p:.3f}'

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\begin{threeparttable}')
    lines.append(r'\caption{Lagged Independent Variable Strategy: Network Stability Predicts Future Plateau Onset}')
    lines.append(r'\label{tab:lagged_iv}')
    lines.append(r'\begin{tabular}{lcccccc}')
    lines.append(r'\toprule')
    lines.append(r' & \multicolumn{3}{c}{Stability main effect} & \multicolumn{3}{c}{Conditional \HR{} at year 10} \\')
    lines.append(r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}')
    lines.append(r'Specification & \HR{} & 95\% \CI{} & \pval{} & \HR{} & 95\% \CI{} & \pval{} \\')
    lines.append(r'\midrule')

    for key, lag_label in [('t', r'Contemporaneous ($t$)'),
                           ('t-1', r'Lag-1 ($t-1$)'),
                           ('t-2', r'Lag-2 ($t-2$)'),
                           ('t-3', r'Lag-3 ($t-3$)')]:
        r = results_lag[key]
        # Main effect
        hr_main = r['stab_main_hr']
        ci_main = r['stab_main_ci']
        p_main = r['stab_main_p']
        stars_main = fmt_stars(p_main)

        # Conditional at year 10
        c10 = r['conditional_hrs'][10]
        stars_c10 = fmt_stars(c10['p'])

        bold_start = r'\textbf{' if key == 't-2' else ''
        bold_end = '}' if key == 't-2' else ''

        line = (f'{bold_start}{lag_label}{bold_end} & '
                f'{hr_main:.3f}{stars_main} & '
                f'[{ci_main[0]:.2f}, {ci_main[1]:.2f}] & '
                f'{fmt_p(p_main)} & '
                f'{c10["hr"]:.3f}{stars_c10} & '
                f'[{c10["ci_lo"]:.2f}, {c10["ci_hi"]:.2f}] & '
                f'{fmt_p(c10["p"])} \\\\')
        lines.append(line)

    lines.append(r'\addlinespace')

    # Add interaction term row
    lines.append(r'\multicolumn{7}{l}{\textit{Stability $\times$ career year interaction coefficient:}} \\')
    for key, lag_label in [('t', r'$t$'), ('t-1', r'$t-1$'),
                           ('t-2', r'$t-2$'), ('t-3', r'$t-3$')]:
        r = results_lag[key]
        stars = fmt_stars(r['int_p'])
        lines.append(f'\\quad {lag_label} & '
                     f'$\\beta$ = {r["int_coef"]:.3f}{stars} & '
                     f'\\multicolumn{{2}}{{c}}{{(\\pval{{}} = {fmt_p(r["int_p"])})}} & '
                     f'& & \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\begin{tablenotes}')
    lines.append(r'\small')
    lines.append(r'\item \textit{Note.} *\pval{} $< 0.05$, **\pval{} $< 0.01$, '
                 r'***\pval{} $< 0.001$.')
    lines.append(r'Full-sample time-varying Cox models with career year $\times$ stability interactions.')
    lines.append(r'All covariates standardized ($z$-scores). Penalizer $\lambda = 0.01$.')
    lines.append(r'Controls: network size (lagged to match), career year, birth year, cumulative validation.')
    lines.append(r'Bold row = preferred specification (two-year lag provides temporal separation')
    lines.append(r'while retaining adequate sample size).')
    lines.append(r'\end{tablenotes}')
    lines.append(r'\end{threeparttable}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def generate_evalue_table(evalue_df):
    """Generate LaTeX table for E-value results."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\begin{threeparttable}')
    lines.append(r'\caption{E-value Sensitivity Analysis for Unmeasured Confounding}')
    lines.append(r'\label{tab:evalue}')
    lines.append(r'\begin{tabular}{p{7.5cm}cccc}')
    lines.append(r'\toprule')
    lines.append(r'Finding & \HR{} & 95\% \CI{} & E-value & E-value \\')
    lines.append(r' & & & (point) & (CI) \\')
    lines.append(r'\midrule')

    for _, row in evalue_df.iterrows():
        label = row['Finding']
        # LaTeX-safe label
        label = label.replace('$', r'\$') if '$' not in label else label
        hr = row['HR']
        ci_str = f'[{row["CI_lower"]:.2f}, {row["CI_upper"]:.2f}]'
        e_pt = row['E_point']
        e_ci = row['E_ci']
        lines.append(f'{label} & {hr:.3f} & {ci_str} & {e_pt:.2f} & {e_ci:.2f} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\begin{tablenotes}')
    lines.append(r'\small')
    lines.append(r'\item \textit{Note.} E-values computed following \citet{vanderweele2017sensitivity}.')
    lines.append(r'The E-value (point) is the minimum strength of association on the risk ratio scale')
    lines.append(r'that an unmeasured confounder would need to have with both the treatment and the')
    lines.append(r'outcome to fully explain away the observed association, conditional on measured covariates.')
    lines.append(r'The E-value (CI) applies to the confidence interval limit closest to the null.')
    lines.append(r'For protective effects (\HR{} $< 1$), the reciprocal $1/\HR{}$ is used.')
    lines.append(r'\end{tablenotes}')
    lines.append(r'\end{threeparttable}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


if __name__ == '__main__':
    main()
