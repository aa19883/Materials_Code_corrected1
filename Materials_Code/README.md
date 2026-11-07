# Materials_Code: reproducible aggregate audit

Everything needed to regenerate the numbers, table inputs, figures and audit records of the manuscript from printed aggregates. No detector was trained, so no training code exists or is needed.

## Repository layout
The repository root is the folder `Materials_Code`. All commands below are run from `analysis_code/`.

    README.md
    source_provenance.csv          locations of every source number
    analysis_code/                 scripts, inputs and generated outputs
        run_all.sh  aggregate_audit.py  requirements.txt  inputs.json
        core.py part2.py part3.py part4.py part5.py controlled.py figs.py figs_col.py eqs.py
        cases/caseA.json caseB.json caseC.json
        results*.json  record_A/B/C.json     reference outputs shipped with the code
    digitised/                     digitised curves of Fig. 6 and the extraction script
    screening/screen.json          case-selection searches and exclusions
    figures_col/  figures_wide/    final figures used in the manuscript
    orig_media/                    source images of Fig. 6

## Requirements
Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2 (the tested versions). `pip install -r requirements.txt` accepts NumPy >= 2.4 and SciPy >= 1.17. Figures use the font Liberation Serif. pandoc (3.1.3 tested) is needed only for the optional script `eqs.py`; `run_all.sh` does not call it. No network access is needed to run.

## Run
    cd analysis_code
    pip install -r requirements.txt
    bash run_all.sh                                  # about 60 s; creates figs/ and figs_wide/ itself

Outputs written in `analysis_code/`: `results_core.json`, `results.json`, `results_rev.json`, `results_controlled.json`, `record_A.json`, `record_B.json`, `record_C.json`, `figs/` (column-width figures) and `figs_wide/` (full-width figures). Optional: `python3 eqs.py` writes `eq.md`, `eq.docx` and `eq_omml.json`.

Audit of one report (from `analysis_code/`):

    python3 aggregate_audit.py cases/caseA.json

This prints the step table and writes `record_A.json` (the default output name is `record_<case id>.json`). Options: `--prev 0.001 0.01 0.05 --k 10 100 --out FILE`.

## Verified reproduction
Both commands were run from a fresh copy of this layout (Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2). `results.json`, `results_rev.json`, `results_controlled.json` and `record_A/B/C.json` were identical to the shipped reference files (maximum absolute difference 0). The full-width figures and Figs. 2, 7 and 10 were byte-identical to `figures_wide/` and `figures_col/`. Six single-column figures (Figs. 1, 3, 4, 5, 8, 9) are regenerated with the same content but differ in 1 to 3 pixels of width and in about 1 to 3 percent of pixels from the shipped copies, because of rendering differences; the shipped copies are the ones in the manuscript.

## Files
- analysis_code/inputs.json: all source counts, printed rates, AUCs, baseline accuracies, status codes (V printed, D digitised, H hypothetical, C derived). cases/case{A,B,C}.json: the same per case plus provenance fields and locations in the sources. ../source_provenance.csv: the same locations in one table.
- core.py: metrics, Wilson/Clopper-Pearson/Jeffreys intervals, Dirichlet posterior (200,000 draws, seed 20261021). part2.py: ROC families, fitted parameters, optimisation, prior-shift thresholds, accuracy bounds, baseline scenarios. part3.py: simulations (40,000 replications), exact coverage, clustered beta-binomial design (200,000 replications, seed 20261027). part4.py: credible intervals (adds to results.json). part5.py: limiting-case p-values, joint precision limits (seed 20261026), uncertainty layers, Monte Carlo checks (results_rev.json). controlled.py: controlled example (seed 20261028).
- aggregate_audit.py: executable audit with evidential labels, unresolved inputs and stop rules (seed 20261021 + 5, 200,000 draws).
- figs.py, figs_col.py: figures; eqs.py: equation XML (needs pandoc).
- digitised/: digitised_curves.csv, digitised_threshold_curve.csv, digitisation_calibration.csv, digitise_fig6.py, README_digitised.md.
- screening/screen.json: record of the case-selection re-screen of 2 November 2026.

## Note on versions
Monte Carlo outputs depend on the NumPy version at the level of Monte Carlo error (below 0.1 percentage points here). Numbers in the manuscript come from the versions listed above.
