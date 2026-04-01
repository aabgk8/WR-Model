# WR Data Pipeline

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

## Repo layout

- Keep the active pipeline inputs and scripts in this folder.
- Generated predictions, model artifacts, match reports, and legacy one-off files are archived under `archive/` and ignored by Git.
- `my_players_with_measurables.csv` is the current measurable / draft-capital source of truth.
- `receiving_summary25.csv` should live alongside the other `receiving_summaryYY.csv` files here so the raw input set stays in one place.

## Position scope

- The raw `receiving_summaryYY.csv` files may contain more than WRs.
- `transform_rec_sum.py` is now configured to keep only players who were WR in at least one season.
- `RF2.py` filters again to `final_position == 'WR'`.
- If you want a strict WR-only pipeline from the start, the raw input files should be WR-only before the wide build step.

## Current feature families

- Career PFF receiving aggregates: sums, averages, and maxima across seasons.
- Final team and final position metadata.
- Draft capital when available: `ROUND` and `SELECTION`.
- NFL fantasy outcome target: `career_fpts_per_game`.

## Not currently active in the model

- The legacy `my_players_with_hw.csv` fallback is no longer part of the active pipeline.
- The current binary models only use `HEIGHT` and `WEIGHT` from the measurable side unless a script explicitly adds more fields.
