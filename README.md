Motor Planning Langevin Sampling
This repository contains the analysis code and figure generation scripts for the paper:
"Secondary Motor Cortex Dynamics Are Consistent with Langevin Diffusion and Somatosensory Cortex Shows a Weak Correlation with Plan Certainty in Mice"  
Stamelos Loutsos (Independent Researcher)
Data availability
All raw electrophysiological data are publicly available from DANDI:
Mouse data: DANDI:000139 (Steinmetz et al., 2019)
Human data: DANDI:000469 (Daume et al., 2024)
Processed CSV files (MSD curves, summary statistics) are included in the `data/` folder.
Code structure
`scripts/generate_paper_figures.py` – generates Figures 1-4
`scripts/optogenetic_complete_analysis.py` – in silico optogenetic predictions (Supplementary Figure S4)
Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```
Clinical Implications: Dissociating Apraxia from Hemiplegia
The Langevin sampling framework in M2/PMd provides a measurable, local criterion for the anatomical level of stroke damage.
1. Motor Apraxia
If ischemia/hemorrhage occurs within PFC or PMd, the Langevin sampler is destroyed. The mean-squared displacement (MSD) of the neural trajectory is no longer linear and `max(p)` becomes noise. Result: motor plans cannot be generated. The patient cannot conceive or simulate the action, regardless of preserved muscle strength.
2. Hemiplegia / Failed Execution
If damage occurs outside PFC/PMd – e.g. internal capsule, corticospinal tract, basal ganglia, cerebellum – the Langevin sampler remains intact. MSD stays linear and `max(p)` is computed normally. Result: motor plans are generated correctly but fail during execution. The patient subjectively "knows" what they want to do, but the limb does not obey.
Mechanistic Biomarker
The correlation `r(S1 ~ max(p))` acts as a diagnostic readout of M2 function:
If weak correlation `r ~ 0.16` persists → M2 is functioning → lesion is outside PFC/PMd → execution deficit
If correlation vanishes → M2 is compromised → lesion is within PFC/PMd → planning deficit
Schematic
```mermaid
flowchart LR
    A[Stroke Location] --> B{Is M2/PMd damaged?}
    B -->|Yes| C[Langevin Sampler Destroyed<br>MSD non-linear<br>max(p) = noise]
    B -->|No| D[Langevin Sampler Intact<br>MSD linear<br>max(p) normal]
    C --> E[No motor plans generated<br><b>Motor Apraxia</b>]
    D --> F[Plans generated but fail<br><b>Hemiplegia</b>]
    E --> G[r(S1~max(p)) ≈ 0]
    F --> H[r(S1~max(p)) ~ 0.16]
```
Keywords: `apraxia` `stroke` `motor-planning` `Langevin-dynamics` `biomarker` `M2` `PMd` `internal-capsule`
Code: `max(p)` computed in `src/analysis/langevin_sampler.py` [line 87]  
Archived Code: https://doi.org/10.5281/zenodo.19593096  
Preprint: bioRxiv [DOI pending]
