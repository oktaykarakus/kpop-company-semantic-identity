# Provenance

## Code and data files

`file_provenance.csv` maps each file under `src/`, `data/` and `environments/` to its project source, its SHA-256 and its status. 47 source files are byte-identical copies of the project scripts. One file was changed:

| File | Original SHA-256 | Released SHA-256 | Change |
|---|---|---|---|
| `src/analysis/07d_prediction_summary.py` | `31fe7b951089c40c15b9bc7423693c95d986df015cd5a5fa3bffbd1983a3f94c` | `027f0653a3b2e43c84dd31f51bf7cb1178cd66cf3f29605e5b303aaf34e7d1cd` | Lines 53–56: a hard-coded absolute path under a user home directory was replaced by `Path(__file__).resolve().parents[2]`, the convention the other scripts use. No logic or parameter changed |

`src/tools/derive_text_free_dataset.py` was written for this release. It produced `data/` from the private analysis table `outputs/tables/company_analysis_dataset.parquet` (SHA-256 `9cbff1c7…e284f029`). It was the only script executed during release preparation.

The original project copies are unchanged. Their hashes match the project's recovery inventory.

## Result tables

`results_hashes.csv` lists the original and released SHA-256, the shape and the change (if any) for each of the 20 tables. 18 are byte-identical. In two, the absolute private path prefix in text columns was replaced by `<PROJECT_ROOT>`, and numeric cells were verified identical. See `../results/MANIFEST.md`.

## Script versions behind the saved outputs

| Stage | Included version | `SCRIPT_VERSION` | Evidence |
|---|---|---|---|
| 08c | `08c_semantic_transition_explanation_v9.py` | `2026-08-01-v9-wp5-window-aligned` | Saved `semantic_transition_metadata.json` |
| 08f | `08f_integrated_organisational_identity_v4.py` | `2026-08-01-v4-wp5-aligned` | Saved `integrated_identity_metadata.json` (names the script) |

The static audit `09d` (saved summary dated 2026-07-20) covered the earlier unversioned `08c` (v7) and `08f` (v3). It does not cover the included versions. The scripts that wrote the earlier WP9 output folders (`09d`/`09d2`, `09e`, `09f`) were not recovered.

## Upstream source

EX3exp/Kpop-lyric-datasets, commit `a7d141f5431fffb61c52c8ec1e48553975fa6350` (2023-12-03). Each of the 1,289 analysed songs maps to a file in that commit. 1,288 of them are token-identical to the source lyrics; one differs by 4 of 554 tokens, a preprocessing difference. The record-to-source key is not released.

## Checksums

`SHA256SUMS` covers every file in the repository except itself and the `.git` directory. Check with `shasum -a 256 -c provenance/SHA256SUMS` from the repository root.
