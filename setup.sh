#!/usr/bin/env bash
#
# Setup script for the daarts project.
# Creates a conda environment and installs dependencies from requirements.txt.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#

set -e

# Project root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Python version for the conda environment (3.10+ recommended for compatibility)
PYTHON_VERSION="3.11"

echo "=============================================="
echo "DA-Arts environment setup"
echo "=============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Remove existing environment if it exists (optional; comment out to preserve)
# conda env remove -n daarts -y 2>/dev/null || true

# Create conda environment with Python
echo ""
echo "Creating conda environment 'daarts' with Python ${PYTHON_VERSION}..."
conda create -n daarts python="${PYTHON_VERSION}" -y

# Activate environment and install pip packages
echo ""
echo "Activating environment and installing dependencies from requirements.txt..."
eval "$(conda shell.bash hook)"
conda activate daarts

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Activate the environment with:"
echo "  conda activate daarts"
echo ""
echo "Then run the project:"
echo "  python main.py           # Crawler"
echo "  python code/01_full_sample_interaction.py   # Analysis scripts"
echo ""
echo "=============================================="
echo "IMDb dataset (external replication)"
echo "=============================================="
echo ""
echo "IMDb raw TSV files are NOT included in this repository (gitignored)."
echo "To download and build the IMDb replication dataset, run the following"
echo "scripts in order:"
echo ""
echo "  1. python code/26_imdb_fetch.py     # Download IMDb TSV dumps (~1 GB)"
echo "                                       # and parse → data/imdb_actor_credits.json"
echo "  2. python code/27_imdb_panel.py     # Build person-year panel"
echo "                                       # → data/imdb_panel.csv"
echo "  3. python code/28_imdb_replication.py  # Run replication analysis"
echo "                                       # → data/imdb_replication_table.tex"
echo ""
echo "Source: https://datasets.imdbws.com (IMDb Non-Commercial Datasets)"
echo "Files downloaded: title.basics, title.crew, title.principals, name.basics"
echo ""
