#!/bin/bash
# Reproduces every number, table input and figure from inputs.json (seeded), the revision analyses and the audit records.
set -e
python3 core.py > /dev/null
python3 part2.py > /dev/null
python3 part3.py > /dev/null
python3 part4.py
python3 part5.py > /dev/null          # limiting-case p-values, joint precision limits, uncertainty layers, Monte Carlo checks -> results_rev.json
python3 controlled.py > /dev/null     # controlled example with known score distributions -> results_controlled.json
python3 figs.py                       # full-width figures -> figs_wide/
python3 figs_col.py                   # single-column figures -> figs/
for c in A B C; do python3 aggregate_audit.py cases/case$c.json --out record_$c.json; done
