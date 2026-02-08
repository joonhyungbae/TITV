"""
30_power_and_restriction.py
Simulation-Based Power Analysis + Career Length Restriction Experiment

Addresses reviewer concern: "Is the null continuous interaction in IMDb
evidence of no effect, or insufficient power?"

Analyses:
  (A) Analytical power: transplant Korean effect size into IMDb SE
  (B) Career length restriction: subset IMDb to <=20 year careers
  (C) LaTeX table output
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines import CoxTimeVaryingFitter
import scipy.stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_korean_panel():
    from data_pipeline import (
        load_raw_data, extract_artist_info, extract_events,
        build_person_year_panel
    )
    data_path = os.path.join(DATA_DIR, 'data.json')
    if not os.path.exists(data_path):
        print("  [SKIP] Korean data not found")
        return None
    artists_list = load_raw_data(data_path)
    df_artists = extract_artist_info(artists_list)
    df_events = extract_events(artists_list)
    panel = build_person_year_panel(df_artists, df_events, cutpoint=10)
    panel['id'] = panel['artist_id']
    return panel


def load_imdb_panel():
    panel_path = os.path.join(DATA_DIR, 'imdb_panel.csv')
    if not os.path.exists(panel_path):
        print("  [SKIP] IMDb panel not found")
        return None
    panel = pd.read_csv(panel_path)
    panel['id'] = panel['author_id']
    return panel


def standardize_panel(panel):
    scaler = StandardScaler()
    for col in ['network_stability', 'network_size', 'career_year',
                'birth_year', 'cumulative_validation']:
        if col in panel.columns:
            panel[f'{col}_z'] = scaler.fit_transform(
                panel[[col]].fillna(panel[col].median())
            ).flatten()
    panel['stab_x_caryr'] = panel['network_stability_z'] * panel['career_year_z']
    panel['size_x_caryr'] = panel['network_size_z'] * panel['career_year_z']
    return panel


def fit_interaction_and_extract(panel, label=""):
    formula = ('network_stability_z + network_size_z + career_year_z + '
               'stab_x_caryr + size_x_caryr + birth_year_z + cumulative_validation_z')
    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(panel, id_col='id', event_col='event',
            start_col='start', stop_col='stop',
            formula=formula, show_progress=False)

    s = ctv.summary
    stab_main = s.loc['network_stability_z']
    stab_int = s.loc['stab_x_caryr']

    beta_stab = stab_main['coef']
    beta_int = stab_int['coef']
    se_stab = stab_main['se(coef)']
    se_int = stab_int['se(coef)']

    var_matrix = ctv.variance_matrix_
    cov_stab_int = 0.0
    if ('network_stability_z' in var_matrix.index and
            'stab_x_caryr' in var_matrix.columns):
        cov_stab_int = var_matrix.loc['network_stability_z', 'stab_x_caryr']

    cy_mean = panel['career_year'].mean()
    cy_std = panel['career_year'].std()

    cond_hrs = []
    for cy in [0, 5, 10, 15, 20]:
        cy_z = (cy - cy_mean) / cy_std
        cond_coef = beta_stab + beta_int * cy_z
        cond_se = np.sqrt(se_stab**2 + (cy_z**2) * se_int**2 + 2 * cy_z * cov_stab_int)
        cond_hr = np.exp(cond_coef)
        cond_p = 2 * (1 - scipy.stats.norm.cdf(abs(cond_coef / cond_se)))
        cond_hrs.append({
            'career_year': cy, 'HR': cond_hr, 'p': cond_p,
            'HR_lower': np.exp(cond_coef - 1.96 * cond_se),
            'HR_upper': np.exp(cond_coef + 1.96 * cond_se),
        })

    return {
        'model': ctv,
        'summary': s,
        'beta_stab': beta_stab,
        'beta_int': beta_int,
        'se_stab': se_stab,
        'se_int': se_int,
        'stab_main_hr': stab_main['exp(coef)'],
        'stab_main_p': stab_main['p'],
        'stab_int_hr': stab_int['exp(coef)'],
        'stab_int_p': stab_int['p'],
        'conditional_hrs': pd.DataFrame(cond_hrs),
        'n_py': len(panel),
        'n_events': int(panel['event'].sum()),
        'n_subjects': panel['id'].nunique(),
    }


# ================================================================
# (A) Analytical Power Analysis
# ================================================================

def run_analytical_power_analysis(korean_panel, imdb_panel):
    print("\n" + "=" * 70)
    print("(A) ANALYTICAL POWER ANALYSIS")
    print("=" * 70)

    kr_results = fit_interaction_and_extract(korean_panel, "Korean")
    im_results = fit_interaction_and_extract(imdb_panel, "IMDb")

    beta_kr = kr_results['beta_int']
    se_imdb = im_results['se_int']

    print(f"\n  Korean interaction beta: {beta_kr:.4f}")
    print(f"  IMDb interaction SE:    {se_imdb:.4f}")
    print(f"  Observed IMDb beta:     {im_results['beta_int']:.4f}")
    print(f"  Observed IMDb p:        {im_results['stab_int_p']:.4f}")

    print(f"\n  Power analysis (alpha = 0.05, two-sided):")
    print(f"  {'Effect size':>15}  {'beta':>8}  {'NCP':>8}  {'Power':>8}")
    print(f"  {'-' * 45}")

    results = []
    for multiplier in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        beta_test = beta_kr * multiplier
        ncp = abs(beta_test / se_imdb)
        power = (scipy.stats.norm.cdf(ncp - 1.96) +
                 scipy.stats.norm.cdf(-ncp - 1.96))
        label = f"{multiplier:.2f}x Korean"
        if multiplier == 1.0:
            label += " <-"
        print(f"  {label:>15}  {beta_test:>8.4f}  {ncp:>8.3f}  {power:>8.1%}")
        results.append({
            'multiplier': multiplier,
            'beta': beta_test,
            'ncp': ncp,
            'power': power,
        })

    mde_beta = 2.80 * se_imdb
    mde_hr = np.exp(mde_beta)
    mde_multiplier = mde_beta / beta_kr if beta_kr != 0 else float('inf')
    print(f"\n  Minimum detectable effect (80% power):")
    print(f"    beta = {mde_beta:.4f} (HR = {mde_hr:.3f})")
    print(f"    = {mde_multiplier:.1f}x the Korean effect size")

    return {
        'kr_results': kr_results,
        'im_results': im_results,
        'power_table': pd.DataFrame(results),
        'mde_beta': mde_beta,
        'mde_hr': mde_hr,
    }


# ================================================================
# (B) Monte Carlo Simulation Power Analysis
# ================================================================

def run_simulation_power(korean_panel, imdb_panel, n_sim=500):
    """
    Monte Carlo: generate events under Korean effect size
    but with IMDb covariate structure. Count detection rate.
    """
    print("\n" + "=" * 70)
    print("(B) MONTE CARLO SIMULATION POWER")
    print("=" * 70)

    kr_results = fit_interaction_and_extract(korean_panel, "Korean")
    beta_stab_kr = kr_results['beta_stab']
    beta_int_kr = kr_results['beta_int']
    baseline_rate = imdb_panel['event'].mean()

    print(f"\n  True effect: beta_stab={beta_stab_kr:.4f}, beta_int={beta_int_kr:.4f}")
    print(f"  IMDb baseline event rate: {baseline_rate:.4f}")
    print(f"  Running {n_sim} simulations...")

    rng = np.random.RandomState(42)
    sig_count = 0
    sim_pvals = []
    subjects = imdb_panel['id'].unique()

    for sim_i in range(n_sim):
        sim_panel = imdb_panel.copy()
        sim_panel['event'] = 0

        for sid in subjects:
            mask = sim_panel['id'] == sid
            sub = sim_panel.loc[mask]

            for j, (idx, row) in enumerate(sub.iterrows()):
                stab_z = row['network_stability_z']
                caryr_z = row['career_year_z']
                log_hr = beta_stab_kr * stab_z + beta_int_kr * stab_z * caryr_z
                hazard = baseline_rate * np.exp(log_hr)
                if rng.random() < min(hazard, 0.5):
                    sim_panel.loc[idx, 'event'] = 1
                    # Drop remaining years for this subject
                    remaining = sub.index[j+1:]
                    sim_panel = sim_panel.drop(remaining, errors='ignore')
                    break

        n_ev = int(sim_panel['event'].sum())
        if n_ev < 10:
            sim_pvals.append(1.0)
            continue

        try:
            formula = ('network_stability_z + network_size_z + career_year_z + '
                       'stab_x_caryr + size_x_caryr + birth_year_z + '
                       'cumulative_validation_z')
            ctv = CoxTimeVaryingFitter(penalizer=0.01)
            ctv.fit(sim_panel, id_col='id', event_col='event',
                    start_col='start', stop_col='stop',
                    formula=formula, show_progress=False)
            p = ctv.summary.loc['stab_x_caryr', 'p']
            b = ctv.summary.loc['stab_x_caryr', 'coef']
            sim_pvals.append(p)
            if p < 0.05 and b > 0:
                sig_count += 1
        except Exception:
            sim_pvals.append(1.0)

        if (sim_i + 1) % 100 == 0:
            print(f"    Sim {sim_i+1}/{n_sim}: power = {sig_count/(sim_i+1):.1%}")

    power = sig_count / n_sim
    print(f"\n  SIMULATION POWER: {power:.1%} ({sig_count}/{n_sim})")
    return {'power': power, 'n_sim': n_sim, 'pvals': sim_pvals}


# ================================================================
# (C) Career Length Restriction
# ================================================================

def run_career_restriction(imdb_panel, max_career_years=20):
    print(f"\n{'=' * 70}")
    print(f"(C) CAREER LENGTH RESTRICTION (max={max_career_years} years)")
    print(f"{'=' * 70}")

    actor_career = imdb_panel.groupby('id')['career_year'].max()
    print(f"\n  Full sample:")
    print(f"    Actors: {len(actor_career)}")
    print(f"    Career year: mean={actor_career.mean():.1f}, median={actor_career.median():.0f}")
    print(f"    Person-years: {len(imdb_panel):,}, Events: {int(imdb_panel['event'].sum())}")

    restriction_results = []
    for max_cy in [15, 20, 25, 30, 9999]:
        if max_cy == 9999:
            sub = imdb_panel.copy()
            label = "Full"
        else:
            short_ids = actor_career[actor_career <= max_cy].index
            sub = imdb_panel[
                (imdb_panel['id'].isin(short_ids)) &
                (imdb_panel['career_year'] <= max_cy)
            ].copy()
            label = f"<={max_cy}"

        n_ev = int(sub['event'].sum())
        n_actors = sub['id'].nunique()
        pre_pct = len(sub[sub['career_year'] < 10]) / max(len(sub), 1) * 100

        if n_ev < 10:
            restriction_results.append({
                'max_career': max_cy, 'label': label,
                'n_actors': n_actors, 'n_py': len(sub), 'n_events': n_ev,
                'pre_dec_pct': pre_pct,
                'int_HR': None, 'int_p': None,
                'cond_hr_10': None, 'cond_p_10': None,
            })
            continue

        sub = standardize_panel(sub)
        try:
            r = fit_interaction_and_extract(sub, label)
            chr10 = r['conditional_hrs'][r['conditional_hrs']['career_year'] == 10]
            cond_hr_10 = chr10['HR'].iloc[0] if len(chr10) > 0 else None
            cond_p_10 = chr10['p'].iloc[0] if len(chr10) > 0 else None

            restriction_results.append({
                'max_career': max_cy, 'label': label,
                'n_actors': r['n_subjects'], 'n_py': r['n_py'],
                'n_events': r['n_events'],
                'pre_dec_pct': pre_pct,
                'int_HR': r['stab_int_hr'], 'int_p': r['stab_int_p'],
                'cond_hr_10': cond_hr_10, 'cond_p_10': cond_p_10,
            })
        except Exception as e:
            print(f"    {label}: ERROR {e}")
            restriction_results.append({
                'max_career': max_cy, 'label': label,
                'n_actors': n_actors, 'n_py': len(sub), 'n_events': n_ev,
                'pre_dec_pct': pre_pct,
                'int_HR': None, 'int_p': None,
                'cond_hr_10': None, 'cond_p_10': None,
            })

    df = pd.DataFrame(restriction_results)
    print(f"\n  {'Label':>8}  {'Actors':>7}  {'PY':>8}  {'Events':>7}  "
          f"{'Pre%':>6}  {'Int HR':>7}  {'Int p':>7}  {'HR@10':>7}  {'p@10':>7}")
    print(f"  {'-' * 72}")
    for _, r in df.iterrows():
        int_hr = f"{r['int_HR']:.3f}" if r['int_HR'] is not None else "---"
        int_p = f"{r['int_p']:.4f}" if r['int_p'] is not None else "---"
        cond_hr = f"{r['cond_hr_10']:.3f}" if r['cond_hr_10'] is not None else "---"
        cond_p = f"{r['cond_p_10']:.4f}" if r['cond_p_10'] is not None else "---"
        print(f"  {r['label']:>8}  {int(r['n_actors']):>7}  {int(r['n_py']):>8}  "
              f"{int(r['n_events']):>7}  {r['pre_dec_pct']:>5.1f}%  "
              f"{int_hr:>7}  {int_p:>7}  {cond_hr:>7}  {cond_p:>7}")

    return {'restriction_table': df}


# ================================================================
# LaTeX Table
# ================================================================

def build_latex_table(power_results, restriction_results):
    latex = r"""\begin{table}[htbp]
\centering
\small
\begin{threeparttable}
\caption{Why the Continuous Interaction is Null in IMDb: Power Analysis and Career-Length Restriction}
\label{tab:power_restriction}
\begin{tabular}{lcccc}
\toprule
"""
    latex += r"\multicolumn{5}{l}{\textit{Panel A: Analytical power for detecting the Korean effect in IMDb}} \\" + "\n"
    latex += r"\midrule" + "\n"
    latex += r"Effect size & $\beta_{\text{int}}$ & Non-centrality & Power (\%) & \\" + "\n"
    latex += r"\midrule" + "\n"

    if power_results is not None:
        for _, r in power_results['power_table'].iterrows():
            mult = f"{r['multiplier']:.2f}$\\times$ Korean"
            if r['multiplier'] == 1.0:
                mult = r"1.00$\times$ Korean $\leftarrow$"
            latex += f"{mult} & {r['beta']:.4f} & {r['ncp']:.3f} & {r['power']*100:.1f} & \\\\\n"

    latex += r"\addlinespace" + "\n"
    latex += r"\multicolumn{5}{l}{\textit{Panel B: Continuous interaction under career-length restriction (IMDb)}} \\" + "\n"
    latex += r"\midrule" + "\n"
    latex += r"Max career (yr) & Pre-decade (\%) & Interaction \HR{} & \pval{} & Cond.\ \HR{} yr 10 \\" + "\n"
    latex += r"\midrule" + "\n"

    if restriction_results is not None:
        for _, r in restriction_results['restriction_table'].iterrows():
            mc = "Full" if r['max_career'] == 9999 else str(int(r['max_career']))
            pre_pct = f"{r['pre_dec_pct']:.1f}" if r['pre_dec_pct'] is not None else "---"
            int_hr = f"{r['int_HR']:.3f}" if r['int_HR'] is not None else "---"
            int_p = f"{r['int_p']:.3f}" if r['int_p'] is not None else "---"
            cond_hr = f"{r['cond_hr_10']:.3f}" if r['cond_hr_10'] is not None else "---"
            latex += f"{mc} & {pre_pct} & {int_hr} & {int_p} & {cond_hr} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} Panel A: Analytical power to detect the Korean interaction effect in the IMDb data structure at $\alpha = 0.05$. Panel B: IMDb data restricted to actors with career $\leq$ N years, re-standardized, with the same continuous interaction model. As the career distribution narrows, the pre-decade proportion increases and the continuous interaction gains leverage.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    return latex


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("POWER ANALYSIS + CAREER RESTRICTION EXPERIMENT")
    print("=" * 70)

    korean_panel = load_korean_panel()
    imdb_panel = load_imdb_panel()

    power_results = None
    restriction_results = None

    if korean_panel is not None and imdb_panel is not None:
        korean_panel = standardize_panel(korean_panel)
        imdb_panel = standardize_panel(imdb_panel)
        power_results = run_analytical_power_analysis(korean_panel, imdb_panel)

    if imdb_panel is not None:
        if 'network_stability_z' not in imdb_panel.columns:
            imdb_panel = standardize_panel(imdb_panel)
        restriction_results = run_career_restriction(imdb_panel)

    if power_results is not None or restriction_results is not None:
        print("\n\n" + "=" * 70)
        print("LATEX TABLE")
        print("=" * 70)
        latex = build_latex_table(power_results, restriction_results)
        print(latex)

        output_path = os.path.join(DATA_DIR, 'power_and_restriction_table.tex')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex)
        print(f"\n  Saved to {output_path}")

    print("\nDONE.")


if __name__ == '__main__':
    main()
