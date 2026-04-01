# WR Model Results Summary

Results below reflect the latest reruns completed on April 1, 2026 for the incoming April 2026 WR class.

## Primary model variants

| Model | Script | Target | Main output files | Latest read |
|---|---|---|---|---|
| Top-40 PPG, strict | `RF2_binary_top40.py` | At least one top-40 FantasyPros WR `FPTS/G` season within first 3 NFL seasons | `rf_binary_top40_within3yrs.joblib`, `wr_holdout_2025_binary_top40_within3yrs_predictions.csv`, `wr_binary_top40_within3yrs_backtest.csv`, `wr_binary_top40_within3yrs_feature_importance.csv` | Solid ceiling-hit baseline |
| Top-40 PPG, recent-hit inclusive | `RF2_binary_top40_recent_hits.py` | Same top-40 target, plus recent positives from incomplete windows | `rf_binary_top40_within3yrs_recent_hits.joblib`, `wr_holdout_2025_binary_top40_within3yrs_recent_hits_predictions.csv`, `wr_binary_top40_within3yrs_recent_hits_backtest.csv`, `wr_binary_top40_within3yrs_recent_hits_feature_importance.csv` | Best current ceiling-hit board |
| 600 receiving yards in first 2 years | `RF2_binary_600yd_first2yrs.py` | At least 600 receiving yards in NFL year 1 or year 2 | `rf_binary_600yd_first2yrs.joblib`, `wr_holdout_2025_binary_600yd_first2yrs_predictions.csv`, `wr_binary_600yd_first2yrs_backtest.csv`, `wr_binary_600yd_first2yrs_feature_importance.csv` | Best current overall model |

## High-level takeaway

- Best overall current board: `RF2_binary_600yd_first2yrs.py`
- Best upside / ceiling-hit board: `RF2_binary_top40_recent_hits.py`
- Draft capital remains the strongest feature in all binary variants, with `OVERALL_PICK` clearly leading.
- Current measurable contribution is intentionally limited to `HEIGHT` and `WEIGHT`.

## Model comparison

| Target | Train rows | Positives | Random split ROC-AUC | Random split AP | Random split Brier | Draft-year backtest | Backtest ROC-AUC | Backtest AP | Backtest Brier |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Top-40 PPG in first 3 years | 268 | 60 | 0.7202 | 0.5038 | 0.1515 | 2023 | 0.8278 | 0.7988 | 0.2019 |
| Top-40 PPG in first 3 years, recent-hit inclusive | 275 | 67 | 0.7216 | 0.5872 | 0.1521 | 2023 | 0.8278 | 0.7988 | 0.2019 |
| 600 receiving yards in year 1 or 2 | 302 | 68 | 0.8176 | 0.6120 | 0.1383 | 2024 | 0.9735 | 0.9405 | 0.0838 |

## Top 15 boards

### 1. Top-40 PPG in first 3 years

Source output: `wr_holdout_2025_binary_top40_within3yrs_predictions.csv`

1. Omar Cooper Jr. `(Pick 26, 0.558)`
2. Kevin Concepcion `(Pick 27, 0.519)`
3. Elijah Sarratt `(Pick 94, 0.496)`
4. Makai Lemon `(Pick 11, 0.483)`
5. Jordyn Tyson `(Pick 16, 0.447)`
6. Antonio Williams `(Pick 54, 0.425)`
7. Carnell Tate `(Pick 6, 0.410)`
8. Eric McAlister `(Pick 152, 0.343)`
9. Skyler Bell `(Pick 89, 0.315)`
10. CJ Daniels `(Pick 208, 0.313)`
11. Eric Rivers `(Pick 195, 0.304)`
12. Kevin Coleman `(Pick 149, 0.302)`
13. Denzel Boston `(Pick 29, 0.296)`
14. Chase Roberts `(Pick 218, 0.289)`
15. Chris Bell `(Pick 66, 0.279)`

### 2. Top-40 PPG in first 3 years, recent-hit inclusive

Source output: `wr_holdout_2025_binary_top40_within3yrs_recent_hits_predictions.csv`

1. Omar Cooper Jr. `(Pick 26, 0.648)`
2. Makai Lemon `(Pick 11, 0.542)`
3. Elijah Sarratt `(Pick 94, 0.525)`
4. Kevin Concepcion `(Pick 27, 0.521)`
5. Antonio Williams `(Pick 54, 0.518)`
6. Jordyn Tyson `(Pick 16, 0.457)`
7. Carnell Tate `(Pick 6, 0.451)`
8. Eric McAlister `(Pick 152, 0.397)`
9. Chase Roberts `(Pick 218, 0.358)`
10. Denzel Boston `(Pick 29, 0.332)`
11. Skyler Bell `(Pick 89, 0.331)`
12. CJ Daniels `(Pick 208, 0.324)`
13. Kevin Coleman `(Pick 149, 0.323)`
14. Chris Bell `(Pick 66, 0.305)`
15. Eric Rivers `(Pick 195, 0.305)`

### 3. 600 receiving yards in year 1 or 2

Source output: `wr_holdout_2025_binary_600yd_first2yrs_predictions.csv`

1. Elijah Sarratt `(Pick 94, 0.551)`
2. Kevin Concepcion `(Pick 27, 0.519)`
3. Eric McAlister `(Pick 152, 0.507)`
4. Omar Cooper Jr. `(Pick 26, 0.473)`
5. Makai Lemon `(Pick 11, 0.435)`
6. Antonio Williams `(Pick 54, 0.412)`
7. Eric Rivers `(Pick 195, 0.386)`
8. Jordyn Tyson `(Pick 16, 0.381)`
9. Carnell Tate `(Pick 6, 0.334)`
10. Malachi Fields `(Pick 71, 0.306)`
11. Germie Bernard `(Pick 40, 0.294)`
12. De'Zhaun Stribling `(Pick 93, 0.291)`
13. Chase Roberts `(Pick 218, 0.289)`
14. Skyler Bell `(Pick 89, 0.287)`
15. Josh Cameron `(Pick 124, 0.274)`

## Additional variant tested

`RF2_binary_600yd_first2yrs_recent_hits.py` was also tested.

- Target: same 600-yard threshold, plus recent year-1 positives from the 2025 draft class
- Training rows: `305`
- Positives: `71`
- Random split: `ROC-AUC 0.7720`, `AP 0.5485`, `Brier 0.1488`
- Backtest: same 2024 backtest as the strict 600-yard model
- Read: valid experiment, but weaker than the strict 600-yard model, so it is not the preferred primary version right now

## Notes

- Output CSVs, backtests, feature-importance files, and joblib artifacts are ignored by Git in the main repo, but they are regenerated automatically when the corresponding script is run.
- The current binary models use college PFF production/efficiency features, draft capital, breakout timing features, teammate-context features, and only `HEIGHT` / `WEIGHT` from the measurable side.
