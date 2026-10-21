# Result tables

Each file sits at its original path relative to the project's `outputs/` folder, so the figure scripts can read it after `cp -R results outputs`. All tables are author-generated numeric summaries. None contains lyrics, lyric-derived keywords, song titles or source identifiers. Text cells hold only company names, metric names, window labels, community numbers and generated classification labels.

No table was recomputed. 18 files are byte-identical to the project outputs. In the other two (marked †), only the absolute private path prefix in text columns was replaced by `<PROJECT_ROOT>`; every numeric cell is unchanged. Original and released SHA-256 values are in `../provenance/results_hashes.csv`.

| File | Work package / script | Manuscript use |
|---|---|---|
| `company_profiles/tables/table_company_semantic_profiles.csv` | WP3 / `03` | Table 3 (diversity rows spot-checked: exact); Fig 4 |
| `company_profiles/tables/community_company_concentration.csv` | WP3 / `03` | Fig 5 (B) |
| `company_significance/tables/community_company_enrichment.csv` | WP4 / `04` | Fig 5 |
| `company_significance/tables/artist_level_global_metric_tests.csv` | WP4 / `04` | Table 5 (Holm significance spot-checked: consistent); Fig 6 |
| `company_significance/tables/artist_level_pairwise_metric_tests.csv` | WP4 / `04` | Fig 6 |
| `temporal/rolling5/semantic_drift_summary_consecutive.csv` | WP5 / `05b` | Fig 7 |
| `temporal/rolling5/temporal_trajectory_summary.csv` | WP5 / `05c` | Fig 7 |
| `temporal/rolling5/semantic_change_point_summary.csv` | WP5 / `05d` | Fig 8 |
| `temporal/rolling5/temporal_stability_summary.csv` | WP5 / `05e` | Fig 8 |
| `explainability/integrated_company_identity.csv` | WP8 / `08f_…_v4` | Table 3 (fingerprint dimensions); Fig 9 |
| `robustness/community_detection_sensitivity/community_detection_global_summary.csv` | WP6 / `06f` | Fig 10 |
| `robustness/metric_sensitivity/metric_sensitivity_summary.csv` | WP6 / `06a` | Fig 10 |
| `robustness/temporal_window_sensitivity/temporal_window_publication_summary.csv` | WP6 / `06e` | Fig 10 |
| `robustness/temporal_window_sensitivity/window_sensitivity.csv` | WP6 / `06e` | Fig 10 |
| `robustness/summary_robustness/wp6_global_robustness_summary.csv` | WP6 / `06g` | Fig 10 |
| `robustness/summary_robustness/wp6_primary_analysis_summary.csv` † | WP6 / `06g` | Fig 10 |
| `tables/prediction_results.csv` | WP7 / `07a` | Fig 11 (A) |
| `tables/forecast_results.csv` | WP7 / `07b` | Fig 11 |
| `tables/catalogue_recovery_curves.csv` | WP7 / `07c` | Fig 11 |
| `framework_validation/09d3_computational_reproducibility_audit/reproducibility_run_summary.csv` † | WP9 / `09d` | Computational audit (static; predates `08c` v9 and `08f` v4) |

## Not released

| Item | Reason |
|---|---|
| `explainability/community_semantic_dictionary.csv`, `community_contributions.csv`, `community_rankings.csv`, `semantic_transition_explanations.csv` | Contain lyric-derived keywords (input to Fig 5 and Table 4 vocabulary) |
| `company_profiles/tables/representative_songs_by_company_community.csv` | Contains song titles |
| `company_master*.parquet`, `company_analysis_dataset.parquet`, `company_mapping_manual_review.csv` | Contain lyrics; replaced by `data/company_text_free_v1` |
| Other WP2–WP9 intermediate tables, including the WP9 synthesis behind Table 6 | Not reviewed field by field for release |

The table-to-script attributions follow the folder each script writes to. The correspondence between manuscript numbers and these tables was spot-checked, not verified line by line.
