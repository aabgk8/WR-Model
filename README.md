# Fantasy PPG WR Model

This repo contains the WR college-to-NFL fantasy modeling pipeline and the current draft-class prediction work.

## Main folders

- `wr_data/`: WR modeling project root with scripts, raw/reference data, processed tables, outputs, and docs

## Key docs

- `wr_data/README.md`: pipeline overview and current layout
- `wr_data/docs/MODEL_RESULTS_SUMMARY.md`: latest model variants, metrics, and top 15 WR boards

## Quick start

From the repo root:

```powershell
cd wr_data
python -m pip install pandas numpy scikit-learn joblib
python build_receiving_summary_wide.py
python transform_rec_sum.py
python transform_final_season.py
python concat_and_add_fpts.py
python merge_draft_orders.py
python split_holdout_by_season.py
python RF2_binary_top40.py
python RF2_binary_top40_recent_hits.py
python RF2_binary_600yd_first2yrs.py
```

That rebuilds the processed tables and reruns the three main binary WR boards.

## Main outputs

- Predictions: `wr_data/outputs/predictions/`
- Backtests: `wr_data/outputs/backtests/`
- Feature importance: `wr_data/outputs/feature_importance/`
- Full model summary: `wr_data/docs/MODEL_RESULTS_SUMMARY.md`

