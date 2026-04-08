# WR Data Pipeline

## Quick start

Run everything from this folder:

```powershell
cd wr_data
python -m pip install pandas numpy scikit-learn joblib
```

Minimum package set used by the active scripts:

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

The project no longer depends on all CSVs living beside the scripts. Paths are centralized in `paths.py`, and the data is now organized under `data/`, `outputs/`, and `docs/`.

## Input locations

Put files in these folders before rebuilding:

- `data/raw/receiving/`: yearly PFF receiving summary files like `receiving_summary25.csv`
- `data/raw/fantasypros_wr/`: yearly FantasyPros WR files like `FantasyPros_Fantasy_Football_Statistics_WR2025.csv`
- `data/raw/combine/`: combine and pro day files
- `data/reference/`: draft orders and `my_players_with_measurables.csv`

## Full rebuild

Run the full pipeline in this order:

```powershell
python build_receiving_summary_wide.py
python transform_rec_sum.py
python transform_final_season.py
python concat_and_add_fpts.py
python merge_draft_orders.py
python split_holdout_by_season.py
```

This rebuilds the processed tables in `data/processed/`.

## Run the main models

Current primary scripts:

```powershell
python RF2_binary_top40.py
python RF2_binary_top40_recent_hits.py
python RF2_binary_600yd_first2yrs.py
```

Optional / secondary scripts:

```powershell
python RF2.py
python RF2_binary_600yd_first2yrs_recent_hits.py
python binary_model_tuning_sweeps.py
python export_bio_collection_templates.py
```

What each main binary model does:

- `RF2_binary_top40.py`: predicts whether a drafted WR records at least one top-40 FantasyPros WR `FPTS/G` season within his first 3 NFL seasons
- `RF2_binary_top40_recent_hits.py`: same target, but includes recent early hits from incomplete NFL windows
- `RF2_binary_600yd_first2yrs.py`: predicts whether a drafted WR records at least 600 receiving yards in year 1 or year 2

## Where outputs go

- `outputs/predictions/`: projected incoming-class boards
- `outputs/backtests/`: labeled historical backtests
- `outputs/feature_importance/`: model feature importance exports
- `outputs/models/`: local `.joblib` artifacts
- `outputs/sweeps/`: draft-blend and feature-reduction tuning runs
- `outputs/reports/`: draft/combine matching reports
- `outputs/templates/`: measurable / bio collection templates

## Current WR pipeline

1. `build_receiving_summary_wide.py`
   Reads the season-level `receiving_summaryYY.csv` files and pivots them into one row per `player_id`.
2. `transform_rec_sum.py`
   Aggregates multi-season PFF receiving stats into career-level features.
3. `transform_final_season.py`
   Adds `final_position` and `final_franchise_id`.
4. `concat_and_add_fpts.py`
   Merges FantasyPros NFL fantasy output and creates `career_fpts_per_game`.
5. `merge_draft_orders.py`
   Matches historical drafted WRs back to the college feature table and adds `ROUND` and `SELECTION`.
6. `split_holdout_by_season.py`
   Splits the final table into training and holdout sets by `final_season`.
7. `RF2.py`
   Trains the current WR random forest and scores the holdout class.

## Typical workflow

If you add new raw college or FantasyPros data:

```powershell
python build_receiving_summary_wide.py
python transform_rec_sum.py
python transform_final_season.py
python concat_and_add_fpts.py
python merge_draft_orders.py
python split_holdout_by_season.py
```

If you only changed model logic and want fresh boards from existing processed data:

```powershell
python RF2_binary_top40.py
python RF2_binary_top40_recent_hits.py
python RF2_binary_600yd_first2yrs.py
```

## Repo layout

- `data/raw/receiving/`: season-level PFF receiving summaries like `receiving_summary25.csv`
- `data/raw/fantasypros_wr/`: yearly FantasyPros WR output used for labels/history
- `data/raw/combine/`: combine and pro day input files
- `data/reference/`: draft orders and `my_players_with_measurables.csv`
- `data/processed/`: rebuildable intermediate and final training tables
- `outputs/`: published prediction boards plus other generated artifacts
- `docs/`: model summaries and supporting writeups
- `archive/`: local archives and old bundles that are not published to Git

## Current model results

- See `docs/MODEL_RESULTS_SUMMARY.md` for the latest snapshot of the three main binary WR model variants, their targets, output files, metrics, and top 15 projected WRs.
- The current summary also includes the April 3, 2026 tuning update with before/after improvement tables.
- Current preferred overall model: `RF2_binary_600yd_first2yrs.py`
- Current preferred ceiling-hit model: `RF2_binary_top40_recent_hits.py`

## Position scope

- The raw `receiving_summaryYY.csv` files may contain more than WRs.
- `transform_rec_sum.py` is now configured to keep only players who were WR in at least one season.
- `RF2.py` filters again to `final_position == 'WR'`.
- If you want a strict WR-only pipeline from the start, the raw input files should be WR-only before the wide build step.

## Current feature families

- Career PFF receiving aggregates: sums, averages, and maxima across seasons.
- Final team and final position metadata.
- Draft capital when available: `ROUND`, `SELECTION`, and `OVERALL_PICK`.
- NFL fantasy outcome target: `career_fpts_per_game`.

## Not currently active in the model

- The legacy `my_players_with_hw.csv` fallback is no longer part of the active pipeline.
- The current binary models only use `HEIGHT` and `WEIGHT` from the measurable side unless a script explicitly adds more fields.
