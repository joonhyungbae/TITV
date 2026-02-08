# Time-Invariant Models, Time-Varying Effects: The Reversal of Network Embeddedness across Professional Careers

**Replication Package**

This repository contains (1) a web crawler for collecting artist career data from the DA-Arts database, (2) analysis scripts that reproduce every table and figure in the paper, and (3) an external replication pipeline using IMDb data.

---

## Repository Structure

```
.
├── main.py                  # DA-Arts web crawler entry point
├── crawler/                 # Crawler modules (artist list & detail)
├── code/
│   ├── utils/               # Shared data pipeline & constants
│   ├── 00_build_dataset.py  # Raw data → analysis-ready dataset
│   ├── 01–25_*.py           # Main analyses, robustness checks, figures
│   ├── 26–28_imdb_*.py      # IMDb external replication pipeline
│   ├── 29_unified_specification.py
│   └── 30_power_and_restriction.py
├── data/                    # Datasets and generated tables (.tex)
├── figures/                 # Generated figures (.png)
├── requirements.txt
└── setup.sh                 # Conda environment setup script
```

## Prerequisites

- Python 3.11+ (recommended)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (for automated environment setup)

## Setup

```bash
# Option A: Automated setup (creates a conda env named "daarts")
chmod +x setup.sh
./setup.sh
conda activate daarts

# Option B: Manual setup with pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Key Dependencies

| Category | Packages |
|---|---|
| Crawler | `requests`, `beautifulsoup4`, `lxml`, `aiohttp`, `tqdm` |
| Data & Analysis | `pandas`, `numpy`, `scipy`, `statsmodels`, `patsy` |
| Survival Analysis | `lifelines` |
| ML & Stats | `scikit-learn` |
| Plotting | `matplotlib` |
---

## Reproducing the Paper

### Step 0: Data Collection (optional)

The repository ships with pre-collected data (`data/data.json`, `data/artist_details.json`). To re-crawl from DA-Arts:

```bash
python main.py                # Full crawl (~505 artists, ~15 min)
python main.py --test         # Test mode (5 artists only)
python main.py --resume       # Resume an interrupted crawl
```

### Step 1: Build the Analysis Dataset

```bash
python code/00_build_dataset.py
```

Transforms raw crawled data into the enriched, analysis-ready `data/data.json` using deterministic regex-based parsing with pre-built mapping tables (`data/institution_mappings.json`, `data/translation_mappings.json`).

### Step 2: Run Analysis Scripts

Scripts are numbered in execution order. Each script prints results to stdout and writes outputs (tables, figures) to `data/` or `figures/`.

```bash
# Descriptive statistics (Tables 1–2)
python code/01_descriptive_statistics.py

# Cox regression & Schoenfeld diagnostics
python code/02_schoenfeld_diagnostic.py
python code/03_full_sample_interaction.py

# Figures
python code/04_stability_hr_plot.py          # Figure 2
python code/10_causal_dag.py                 # Figure 1
python code/11_achievement_gradient.py       # Figure 3

# Mechanism & robustness analyses
python code/05_plateau_sensitivity.py
python code/06_categorical_mediation.py
python code/07_reputational_lockin.py
python code/08_acquisition_rate.py
python code/09_temporal_ordering.py
python code/12_institutional_logic.py
python code/13_triangulation.py

# Robustness checks
python code/14_robustness_placeholders.py
python code/15_career_plateau_dynamics.py
python code/16_nested_model_robustness.py
python code/17_interaction_robustness.py
python code/18_small_sample_robustness.py

# Forward path & causal evidence
python code/19_forward_path_evidence.py
python code/20_institution_closure_did.py
python code/21_absolute_risk.py
python code/22_field_portable_metrics.py
python code/23_likelihood_ratio_test.py
python code/24_stage_specific_granger.py
python code/25_closure_event_study.py        # Figure 4
```

Or run all main analyses at once:

```bash
for f in code/[0-2][0-9]_*.py; do echo "=== $f ===" && python "$f"; done
```

### Step 3: External Replication — IMDb

IMDb raw data files are **not** included in the repository (gitignored, ~1 GB). Run the following scripts in order to download, build, and replicate:

```bash
python code/26_imdb_fetch.py        # Download IMDb TSV dumps → data/imdb_actor_credits.json
python code/27_imdb_panel.py        # Build person-year panel → data/imdb_panel.csv
python code/28_imdb_replication.py  # Run replication analysis → data/imdb_replication_table.tex
```

Source: [IMDb Non-Commercial Datasets](https://datasets.imdbws.com)

### Step 4: Unified Specification & Power Analysis

```bash
python code/29_unified_specification.py   # Unified RCS specification (Figure 5)
python code/30_power_and_restriction.py   # Simulation-based power analysis
```

---

## Output Summary

| Output | Location |
|---|---|
| Descriptive statistics | stdout |
| Cox regression tables | `data/*.tex` |
| Figure 1 — Causal DAG | `figures/fig1_causal_dag.png` |
| Figure 2 — Stability HR | `figures/fig2_stability_hr.png` |
| Figure 3 — Achievement gradient | `figures/fig3_achievement_gradient.png` |
| Figure 4 — Closure event study | `figures/fig4_closure_event_study.png` |
| Figure 5 — Unified specification | `figures/fig5_unified_specification.png` |

---

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
