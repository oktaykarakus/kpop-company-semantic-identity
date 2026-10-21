# Release-safety record

Prepared 2026-10-21. These are static checks of the repository contents. They are not legal advice.

| Check | Result |
|---|---|
| Lyrics or lyric excerpts | None. The dataset has no free-text column. No Hangul or CJK characters appear in the data, results or code |
| Song titles, Melon IDs, chart positions, release dates, source paths, record-to-source key | None |
| Lyric-derived keywords | None. The five tables that contain them are withheld (see `../results/MANIFEST.md`) |
| Artist–company lookup | Not included |
| Credentials or tokens | None found by pattern scan (API keys, secrets, passwords, bearer, AWS, Hugging Face, OpenAI and GitHub tokens) |
| Private absolute paths | None. One script path and two result-table path prefixes were redacted; see `manifest.md` |
| E-mail addresses | None |
| Personal data | Artists appear only as pseudonymous `artist_id`, which could be re-identified from public chart data. `src/build_company_dataset.py` names three public stage names in its collision rules; these are public performers and the code is needed for provenance |
| Dataset shape | 1,289 × 18; 126 artists; 4 companies; 1997–2023; 15 communities |

## Rights basis (summary of the source-rights review)

| Material | Status |
|---|---|
| Lyrics | Restricted: copyright of the rights holders. Not included |
| Melon chart and song metadata | Unclear: no permission found. Not included |
| Computed per-song metrics, pseudonymous IDs, company, community, release year | Author-generated or factual; judged releasable |
| Numeric aggregate result tables | Author-generated; judged releasable |
| Author-generated code | Releasable under MIT at the copyright claimant's direction |

## Open items

- Licence for `data/` and `results/` (author decision).
- Cardiff University's IP policy on code and data release has not been checked.
- Korean database and copyright law were not reviewed. This matters mainly for `release_year`, which originates in Melon's release-date field.
