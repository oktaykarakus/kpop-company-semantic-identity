# Code and Text-Free Data for Organisation-Level Semantic Identity in K-pop

This package contains the analysis code, a text-free derived song-level dataset and selected numeric result tables associated with *A Computational Framework for Modelling Organisation-Level Semantic Identity from Longitudinal Textual Data*, by Brinda Murali Krishna, Oktay Karakuş and Can Eyüpoğlu ([arXiv:2609.29584](https://arxiv.org/abs/2609.29584)). The code in `src/` is licensed under MIT, Copyright (c) 2026 Oktay Karakuş. The derived dataset and result tables are licensed under CC BY 4.0 (see [Licensing](#licensing)). Version `v1.0.0`, distributed through GitHub only; no DOI has been assigned.

It is not a complete end-to-end reproduction archive. It contains no lyrics, song titles or source metadata.

## Scientific objective

The study models *organisation-level semantic identity*: whether the lyrics released by artists of the same entertainment company share a measurable, stable and distinctive semantic profile. Building on a song-level semantic similarity graph and its semantic communities, the framework:

1. aggregates song-level community membership and graph metrics into company-level semantic profiles (WP2–WP3);
2. tracks how those profiles change over overlapping five-year rolling windows, including drift and change points (WP5);
3. tests robustness to similarity metric, bootstrap resampling, leave-one-artist-out, balanced sampling, temporal window and community-detection choices (WP6);
4. assesses how far company membership and future profiles can be predicted from the semantic representation (WP7);
5. integrates the evidence into a four-dimensional semantic fingerprint (exploration, integration, breadth, adaptation) and company archetypes (WP8);
6. synthesises cross-work-package consistency, traceability, transferability, computational audit and threats to validity (WP9).

## Study population

| | |
|---|---|
| Songs | 1,289 |
| Artists | 126 |
| Companies | 4 (HYBE, JYP, SM, YG), used as fixed analytical portfolios |
| Release years | 1997–2023 |
| Semantic communities | 15 |

Company labels come from a static artist-to-company lookup. Affiliation was not checked at each release date.

## Repository contents

| Path | Contents |
|---|---|
| `data/` | Text-free dataset (`company_text_free_v1.parquet`, plus the same data as CSV) and `data_dictionary.csv` |
| `results/` | 20 numeric result tables, at their original paths relative to the project `outputs/` folder; `results/MANIFEST.md` links each to the manuscript |
| `src/analysis/` | 33 analysis scripts (`01`–`09f`) |
| `src/visualisation/` | Plot scripts for PLOS Figs 4–11, a shared `config.py` and the original plotting `requirements.txt` |
| `src/utils/statistics.py` | Helper imported by `02_company_statistics.py` |
| `src/build_company_dataset.py` | Builds the company dataset from upstream corpus files (provenance only; its inputs are not included) |
| `src/tools/derive_text_free_dataset.py` | The script that produced `data/` from the private analysis table |
| `environments/` | Untested environment specification and the evidence behind it |
| `provenance/` | Script provenance, result-table hashes, release-safety record and `SHA256SUMS` |
| `CITATION.cff` | Citation metadata |
| `LICENSE` | MIT licence for the code only |
| `LICENSE-DATA.md` | CC BY 4.0 licence for `data/` and `results/` |

There is no `figures/` folder. The submitted figures are part of the manuscript, and Fig 5 and Table 4 display lyric-derived community vocabulary that has not been cleared for redistribution.

## Dataset

`data/company_text_free_v1.parquet` has one row per analysed song: 1,289 rows and 18 columns, sorted by `record_id`. The CSV copy is exact when read with `pd.read_csv(..., float_precision="round_trip")`.

| Group | Columns |
|---|---|
| Identifiers (pseudonymous) | `record_id` (S0001–S1289), `artist_id` (A001–A126) |
| Grouping | `company`, `release_year`, `community` (15 labels), `analysis_eligible` |
| Graph and novelty metrics | `degree`, `weighted_degree`, `external_strength`, `external_strength_fraction`, `participation_coefficient`, `community_entropy`, `external_communities_reached`, `local_novelty`, `max_centroid_cosine`, `centroid_margin`, `centroid_novelty`, `bridge_affinity_score` |

Definitions are in `data/data_dictionary.csv`.

**Identifiers.** `record_id` is the rank of the original song identifier in ascending sort order. `artist_id` is the rank of the normalised artist name in ascending sort order. No crosswalk is included. These are pseudonyms, not anonymisation: the artists are public performers and could be re-identified from public chart data combined with company, years and song counts.

**Dropped fields.** Lyrics, lyric-derived text, titles, full dates, Melon identifiers, chart positions and source file paths were removed. Four constant fields were also dropped. Their values were `mapping_status = matched_student_lookup`, `mapping_basis = static_student_lookup`, `company_at_release_verified = False` and `mapping_review_required = False`. Four further columns can be recomputed exactly from retained columns:

- `internal_strength = weighted_degree − external_strength`
- `communities_reached = external_communities_reached + 1`
- `mean_neighbour_cosine = 1 − local_novelty`
- `second_centroid_cosine = max_centroid_cosine − centroid_margin`

The derived values were checked against the private analysis table: row count, per-company counts, cross-tabulations and all retained values agree, with a maximum absolute difference of 0.

## Analysis pipeline

Scripts write to `outputs/` under the repository root, because every script resolves the root as `Path(__file__).resolve().parents[2]`.

| Step | Scripts | Main input | Runs from this package? |
|---|---|---|---|
| 0 | `build_company_dataset.py` | Upstream corpus files and the artist–company lookup | No: inputs not included |
| 1 | `01_build_master_dataset` | Step 0 output, including lyrics | No: reads lyrics |
| 2–8 | `02` → `03` → `04` → `05a`–`05e` → `06a`–`06g` → `07a`–`07d` → `08a` → `08b` → `08b1` → `08c_…_v9` → `08d` → `08e` → `08f_…_v4` | `outputs/tables/company_master_analysis_v1.parquet` and earlier outputs | Partly, see below |
| 9 | `09a2` → `09b` → `09c` → `09d` → `09e` → `09f` | Outputs of steps 1–8 | Only after steps 1–8 |
| Figures | see table below | Result tables | Inputs released for 7 of 8 figures (not executed) |

**Steps 2–8 and the derived dataset.** The scripts expect `company_master_analysis_v1.parquet`, which contains lyrics and uses the original column names. No adapter is supplied. Using the derived dataset would need `song_uid` ← `record_id`, `artist`/`artist_normalised` ← `artist_id` and `year` ← `release_year`.

**Steps that cannot run from this package** because they read lyrics or titles: `01`, `06f`, `08b` and `08b1`. Their outputs feed `06g`, `08d`, `08e` and `08f`. `08c_…_v9` stops with a `ValueError` when there is no song-title column, so it cannot run from the derived dataset either. Embedding generation, the similarity graph and community detection are upstream of this package. The graph and novelty metrics are supplied as computed values only.

**Script versions behind the saved outputs.** The saved output metadata identify `08c_semantic_transition_explanation_v9.py` (`2026-08-01-v9-wp5-window-aligned`) and `08f_integrated_organisational_identity_v4.py` (`2026-08-01-v4-wp5-aligned`). Earlier versions of `08c`, `08f` and `09a` are not included.

### Figure scripts

| PLOS figure | Script | Inputs |
|---|---|---|
| Fig 4 | `figure03_dashboard.py` | `table_company_semantic_profiles.csv` |
| Fig 5 | `figure04_landscape.py` | `community_company_concentration.csv`, `community_company_enrichment.csv`, **`community_semantic_dictionary.csv` (not released)** |
| Fig 6 | `figure05_statistics.py` | `artist_level_global_metric_tests.csv`, `artist_level_pairwise_metric_tests.csv` |
| Fig 7 | `figure06_temporal_evolution_v3.py` | `semantic_drift_summary_consecutive.csv`, `temporal_trajectory_summary.csv` |
| Fig 8 | `figure07_temporal_stability_v2.py` | `semantic_change_point_summary.csv`, `temporal_stability_summary.csv` |
| Fig 9 | `figure08_integrated_identity_v1.py` | `integrated_company_identity.csv` |
| Fig 10 | `figure09_robustness_v2.py` | Six WP6 robustness summaries |
| Fig 11 | `figure10_predictive_validation_v3.py` | `prediction_results.csv`, `forecast_results.csv`, `catalogue_recovery_curves.csv` |

To run a figure script against the released tables, copy `results/` to `outputs/` at the repository root (`cp -R results outputs`) and run it from the repository root, for example `python src/visualisation/figure03_dashboard.py`. Figs 1–3 are LaTeX or vector drawings and have no script. Fig 5 cannot be regenerated because its community-vocabulary input is withheld. The figure scripts were not executed against this package during preparation.

## What can be verified with the released materials

| Manuscript item | Verifiable here? | Source |
|---|---|---|
| Table 1 (dataset summary) | Yes, recomputable | `data/` |
| Table 2 (fingerprint components) | Not applicable (method definition) | — |
| Table 3 (company fingerprint summary) | Values available | `table_company_semantic_profiles.csv`, `integrated_company_identity.csv` |
| Table 4 (semantic communities) | Partly: community sizes only | `data/`; the representative vocabulary is lyric-derived and withheld |
| Table 5 (artist-level inferential tests) | Values available | `artist_level_*_metric_tests.csv`; the inputs are in `data/`, but `04` would need the column renaming described above (not tested) |
| Table 6 (validation synthesis) | No: WP9 synthesis tables not released | — |
| Figs 4, 6–11 | Values available; plot scripts should run from them (not tested) | See `results/MANIFEST.md` |
| Fig 5 | Partly | Concentration and enrichment tables only |

"Values available" means the released table holds the values reported. It does not mean they were regenerated independently. The correspondence between manuscript numbers and these tables has been spot-checked, not verified line by line.

## Reproducibility coverage

| Evidence level | Status |
|---|---|
| Analysis and figure source code included | Yes, 33 analysis and 8 figure scripts |
| Text-free dataset and numeric result tables included | Yes |
| Scripts executed for this release | No. Only the dataset derivation script was run |
| Lyric-dependent stages (embeddings, graph, communities, `01`, `06f`, `08b`, `08b1`, `08c`) reproducible from this package | No |
| Published results independently reproduced from this package | No |

**Static computational audit.** `09d_computational_reproducibility_audit.py` inspects source code and does not execute anything. Its saved summary (`results/framework_validation/09d3_…/reproducibility_run_summary.csv`, generated 2026-07-20) reports 27/27 scripts passing, 6/6 stochastic stages seed-controlled, no unresolved dependencies and no upstream rerun. That audit predates `08c` v9 and `08f` v4 (2026-08-01). It covered the earlier, unversioned `08c` and `08f`, so its PASS does not extend to the included versions. Run in this repository, `09d` would report those two files as missing because it looks for the unversioned names.

## Known limitations and unavailable dependencies

- The lyric corpus, `company_master_analysis_v1.parquet`, song embeddings, the similarity graph and the community partition are not included.
- The artist-to-company lookup (`metadata/big4_lookup.csv`, 142 artists) is not included. Its provenance and redistribution rights are unconfirmed.
- Intermediate tables that contain song titles or lyric-derived keywords are withheld: `community_semantic_dictionary.csv`, `community_contributions.csv`, `community_rankings.csv`, `semantic_transition_explanations.csv` and `representative_songs_by_company_community.csv`.
- The preprocessing from the upstream JSON files to the project corpus has not been independently reconstructed.
- Package versions used for the original runs were not recorded. `environments/` lists untested lower bounds. `09a2` requires Python ≥ 3.12.
- The scripts that produced the earlier WP9 output folders (`09d`/`09d2`, `09e`, `09f`) were not recovered; the included scripts write the latest folders.

## Data source and copyright

The lyrics analysed in the study come from the public GitHub repository [EX3exp/Kpop-lyric-datasets](https://github.com/EX3exp/Kpop-lyric-datasets), commit `a7d141f` (`a7d141f5431fffb61c52c8ec1e48553975fa6350`, 2023-12-03). It compiles songs from the Melon monthly Top 100 charts (2000–2023). Melon is a music-streaming service operated by Kakao Entertainment. All 1,289 analysed songs map to files in that commit.

**No lyrics are included in this package.** Song titles, Melon song identifiers, chart positions, release dates and source file paths are also excluded, as is any lookup from records to source files. The lyrics are copyright works of their lyricists and other rights holders. The upstream repository's BSD 3-Clause licence covers only what its maintainer owns. That repository states that it does not claim ownership of the songs, so the BSD licence does not grant any right to the lyrics. Researchers who need the text should obtain it from the upstream repository, check its terms and the applicable law, and must not redistribute it.

To inspect the upstream snapshot:

```sh
git clone https://github.com/EX3exp/Kpop-lyric-datasets.git
cd Kpop-lyric-datasets
git checkout a7d141f5431fffb61c52c8ec1e48553975fa6350
```

Rebuilding the exact 1,289-song set would also need the artist-to-company lookup and the upstream preprocessing, neither of which is released.

## Licensing

- **Code** (`src/`): MIT, Copyright (c) 2026 Oktay Karakuş (see `LICENSE`). This follows the arrangement used for the authors' companion code release.
- **Derived dataset and result tables** (`data/`, `results/`): CC BY 4.0, Copyright (c) 2026 Brinda Murali Krishna, Oktay Karakuş and Can Eyüpoğlu (see `LICENSE-DATA.md`).
- **Institutional IP**: ownership has not been independently verified, including against Cardiff University's IP policy. The copyright lines record the authors' assertion, and both licences are granted on that basis.
- **Third-party material**: neither licence applies to lyrics, Melon metadata or any other third-party material, none of which is included. Dependencies keep their own licences.

## How to Cite

If you use this code or data, please cite the paper:

> Brinda Murali Krishna, Oktay Karakuş and Can Eyüpoğlu. *A Computational Framework for Modelling Organisation-Level Semantic Identity from Longitudinal Textual Data.* arXiv:2609.29584, 2026. https://arxiv.org/abs/2609.29584

```bibtex
@misc{muralikrishna2026organisational,
  title         = {A Computational Framework for Modelling Organisation-Level Semantic Identity from Longitudinal Textual Data},
  author        = {Murali Krishna, Brinda and Karaku{\c{s}}, Oktay and Ey{\"u}po{\u{g}}lu, Can},
  year          = {2026},
  eprint        = {2609.29584},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.29584}
}
```

Please also cite the lyric source, [EX3exp/Kpop-lyric-datasets](https://github.com/EX3exp/Kpop-lyric-datasets). Machine-readable citation metadata are in `CITATION.cff`.
