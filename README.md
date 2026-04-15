# Motor Planning Langevin Sampling

This repository contains the analysis code and figure generation scripts for the paper:

**"Secondary Motor Cortex Dynamics Are Consistent with Langevin Diffusion and Somatosensory Cortex Shows a Weak Correlation with Plan Certainty in Mice"**  
Stamelos Loutsos (Independent Researcher)

## Data availability

All raw electrophysiological data are publicly available from DANDI:
- Mouse data: [DANDI:000139](https://dandiarchive.org/dandiset/000139) (Steinmetz et al., 2019)
- Human data: [DANDI:000469](https://dandiarchive.org/dandiset/000469) (Daume et al., 2024)

Processed CSV files (MSD curves, summary statistics) are included in the `data/` folder.

## Code structure

- `scripts/generate_paper_figures.py` – generates Figures 1-4
- `scripts/optogenetic_complete_analysis.py` – in silico optogenetic predictions (Supplementary Figure S4)

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
