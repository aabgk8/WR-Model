# WR Model Results Summary

This file tracks two snapshots for the incoming April 2026 WR class:

- April 1, 2026 baseline models
- April 3, 2026 tuned models after teammate-adjusted features, draft-prior blending, and feature reduction

## Primary model variants

| Model | Script | Target | Main output files | Latest read |
|---|---|---|---|---|
| Top-40 PPG, strict | `RF2_binary_top40.py` | At least one top-40 FantasyPros WR `FPTS/G` season within first 3 NFL seasons | `rf_binary_top40_within3yrs.joblib`, `wr_holdout_2025_binary_top40_within3yrs_predictions.csv`, `wr_binary_top40_within3yrs_backtest.csv`, `wr_binary_top40_within3yrs_feature_importance.csv` | Solid ceiling-hit baseline |
| Top-40 PPG, recent-hit inclusive | `RF2_binary_top40_recent_hits.py` | Same top-40 target, plus recent positives from incomplete windows | `rf_binary_top40_within3yrs_recent_hits.joblib`, `wr_holdout_2025_binary_top40_within3yrs_recent_hits_predictions.csv`, `wr_binary_top40_within3yrs_recent_hits_backtest.csv`, `wr_binary_top40_within3yrs_recent_hits_feature_importance.csv` | Best current ceiling-hit board |
| 600 receiving yards in first 2 years | `RF2_binary_600yd_first2yrs.py` | At least 600 receiving yards in NFL year 1 or year 2 | `rf_binary_600yd_first2yrs.joblib`, `wr_holdout_2025_binary_600yd_first2yrs_predictions.csv`, `wr_binary_600yd_first2yrs_backtest.csv`, `wr_binary_600yd_first2yrs_feature_importance.csv` | Best current overall model |

## High-level takeaway

- Best overall current board: `RF2_binary_600yd_first2yrs.py`
- Best upside / ceiling-hit board: `RF2_binary_top40_recent_hits.py`
- April 3 tuning materially improved validation on all three main binary models.
- The biggest changes were teammate-adjusted context features, a draft-prior blend, and top-20 feature reduction for the 600-yard model.
- Current measurable contribution is still intentionally limited to `HEIGHT` and `WEIGHT`.

## Latest tuning update

Current tuned settings:

- Shared draft-prior blend weight: `0.25`
- Shared binary feature set now includes teammate-adjusted ratio/gap and top-2 share features
- `RF2_binary_600yd_first2yrs.py` now uses the top `20` features selected by importance

## Improvement Summary

| Model | Main tuning change | Random AP | Random ROC-AUC | Backtest AP | Backtest ROC-AUC | Read |
|---|---|---:|---:|---:|---:|---|
| Top-40 PPG, strict | Teammate-adjusted features + draft blend `0.25` | `0.5038 -> 0.7206` | `0.7202 -> 0.8075` | `0.7988 -> 0.8634` | `0.8278 -> 0.8565` | Clear improvement |
| Top-40 PPG, recent-hit inclusive | Teammate-adjusted features + draft blend `0.25` | `0.5872 -> 0.6943` | `0.7216 -> 0.7821` | `0.7988 -> 0.8634` | `0.8278 -> 0.8565` | Better current ceiling board |
| 600 receiving yards in year 1 or 2 | Draft blend `0.25` + top `20` features | `0.6120 -> 0.7246` | `0.8176 -> 0.8784` | `0.9405 -> 0.9034` | `0.9735 -> 0.9524` | Better validation, slightly weaker backtest |

## Current tuned model comparison

| Target | Train rows | Positives | Random split ROC-AUC | Random split AP | Random split Brier | Draft-year backtest | Backtest ROC-AUC | Backtest AP | Backtest Brier |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Top-40 PPG in first 3 years | 268 | 60 | 0.8075 | 0.7206 | 0.1325 | 2023 | 0.8565 | 0.8634 | 0.1936 |
| Top-40 PPG in first 3 years, recent-hit inclusive | 275 | 67 | 0.7821 | 0.6943 | 0.1365 | 2023 | 0.8565 | 0.8634 | 0.1936 |
| 600 receiving yards in year 1 or 2 | 302 | 68 | 0.8784 | 0.7246 | 0.1196 | 2024 | 0.9524 | 0.9034 | 0.0847 |

## April 1 baseline comparison

| Target | Train rows | Positives | Random split ROC-AUC | Random split AP | Random split Brier | Draft-year backtest | Backtest ROC-AUC | Backtest AP | Backtest Brier |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Top-40 PPG in first 3 years | 268 | 60 | 0.7202 | 0.5038 | 0.1515 | 2023 | 0.8278 | 0.7988 | 0.2019 |
| Top-40 PPG in first 3 years, recent-hit inclusive | 275 | 67 | 0.7216 | 0.5872 | 0.1521 | 2023 | 0.8278 | 0.7988 | 0.2019 |
| 600 receiving yards in year 1 or 2 | 302 | 68 | 0.8176 | 0.6120 | 0.1383 | 2024 | 0.9735 | 0.9405 | 0.0838 |

## Top 15 boards

### 1. Top-40 PPG in first 3 years

Source output: `wr_holdout_2025_binary_top40_within3yrs_predictions.csv`

1. Omar Cooper Jr. `(Pick 26, 0.588)`
2. Kevin Concepcion `(Pick 27, 0.573)`
3. Jordyn Tyson `(Pick 16, 0.546)`
4. Makai Lemon `(Pick 11, 0.535)`
5. Carnell Tate `(Pick 6, 0.503)`
6. Denzel Boston `(Pick 29, 0.416)`
7. Elijah Sarratt `(Pick 94, 0.415)`
8. Antonio Williams `(Pick 54, 0.393)`
9. Germie Bernard `(Pick 40, 0.370)`
10. Zachariah Branch `(Pick 42, 0.316)`
11. Chris Bell `(Pick 66, 0.315)`
12. Skyler Bell `(Pick 89, 0.313)`
13. Ted Hurst `(Pick 97, 0.298)`
14. Malachi Fields `(Pick 71, 0.279)`
15. De'Zhaun Stribling `(Pick 93, 0.252)`

### 2. Top-40 PPG in first 3 years, recent-hit inclusive

Source output: `wr_holdout_2025_binary_top40_within3yrs_recent_hits_predictions.csv`

1. Omar Cooper Jr. `(Pick 26, 0.646)`
2. Carnell Tate `(Pick 6, 0.605)`
3. Makai Lemon `(Pick 11, 0.601)`
4. Kevin Concepcion `(Pick 27, 0.600)`
5. Jordyn Tyson `(Pick 16, 0.572)`
6. Denzel Boston `(Pick 29, 0.473)`
7. Antonio Williams `(Pick 54, 0.436)`
8. Elijah Sarratt `(Pick 94, 0.416)`
9. Germie Bernard `(Pick 40, 0.378)`
10. Zachariah Branch `(Pick 42, 0.334)`
11. De'Zhaun Stribling `(Pick 93, 0.321)`
12. Skyler Bell `(Pick 89, 0.310)`
13. Malachi Fields `(Pick 71, 0.308)`
14. Ted Hurst `(Pick 97, 0.303)`
15. Chris Bell `(Pick 66, 0.303)`

### 3. 600 receiving yards in year 1 or 2

Source output: `wr_holdout_2025_binary_600yd_first2yrs_predictions.csv`

1. Makai Lemon `(Pick 11, 0.683)`
2. Antonio Williams `(Pick 54, 0.594)`
3. Omar Cooper Jr. `(Pick 26, 0.562)`
4. Carnell Tate `(Pick 6, 0.521)`
5. Elijah Sarratt `(Pick 94, 0.493)`
6. Zachariah Branch `(Pick 42, 0.452)`
7. Kevin Concepcion `(Pick 27, 0.443)`
8. Jordyn Tyson `(Pick 16, 0.387)`
9. Germie Bernard `(Pick 40, 0.350)`
10. Chris Bell `(Pick 66, 0.332)`
11. Ted Hurst `(Pick 97, 0.263)`
12. Malachi Fields `(Pick 71, 0.258)`
13. Kevin Coleman `(Pick 149, 0.219)`
14. Eric McAlister `(Pick 152, 0.201)`
15. Deion Burks `(Pick 79, 0.195)`

## Additional variant tested

`RF2_binary_600yd_first2yrs_recent_hits.py` was also tested.

- Target: same 600-yard threshold, plus recent year-1 positives from the 2025 draft class
- Training rows: `305`
- Positives: `71`
- Random split: `ROC-AUC 0.7720`, `AP 0.5485`, `Brier 0.1488`
- Backtest: same 2024 backtest as the strict 600-yard model
- Read: valid experiment, but weaker than the strict 600-yard model, so it is not the preferred primary version right now

## April 3 tuning notes

- Carnell Tate moved much closer to expectation after the teammate-adjusted features and draft-prior blend:
  - Top-40 strict: from `#7` to `#5`
  - Top-40 recent-hit inclusive: from `#7` to `#2`
  - 600 yards in first 2 years: from `#9` to `#4`
- The draft-prior sweep suggested that `0.25` is the best compromise between better split validation and avoiding too much damage to historical backtests.
- The 600-yard target benefited most from feature reduction, with the top `20` features outperforming the full feature set.

## Notes

- Output CSVs, backtests, feature-importance files, and joblib artifacts are ignored by Git in the main repo, but they are regenerated automatically when the corresponding script is run.
- The current binary models use college PFF production/efficiency features, draft capital, breakout timing features, teammate-context features, and only `HEIGHT` / `WEIGHT` from the measurable side.
