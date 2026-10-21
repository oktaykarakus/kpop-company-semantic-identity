# Environment record

`requirements.txt` and `environment.yml` were built from static evidence only: the third-party imports found in the scripts, and the lower bounds in `src/visualisation/requirements.txt`. **No environment was created or tested from them.** The package versions used for the original runs were not recorded. The versions noted in the comments come from a local environment inspected on 2026-10-21. That environment runs Python 3.10, so it cannot be the one that ran `09a2`.

| Requirement | Evidence |
|---|---|
| Python ≥ 3.12 | `09a2_cross_workpackage_consistency.py` uses same-quote nested f-strings (PEP 701). All scripts parse under Python 3.12.3 |
| pandas, numpy, pyarrow, scipy, scikit-learn | Analysis scripts |
| statsmodels | `05b` only |
| matplotlib | `09e`, `09f` only |
| plotly ≥ 6, kaleido ≥ 1, Pillow ≥ 10 | Figure scripts (static export; kaleido may need a local Chrome/Chromium) |

These files are not lockfiles. They do not establish that the results can be reproduced on another platform.
