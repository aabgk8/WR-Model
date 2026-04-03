from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

import RF2_binary_top40 as base
import RF2_binary_top40_recent_hits as top40_recent
import RF2_binary_600yd_first2yrs as yards600


SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHT_SWEEP_FILE = SCRIPT_DIR / "binary_model_draft_blend_sweep_results.csv"
FEATURE_SWEEP_FILE = SCRIPT_DIR / "binary_model_feature_reduction_sweep_600yd.csv"

WEIGHT_GRID = [0.0, 0.15, 0.25, 0.35, 0.45, 0.55]
FEATURE_LIMITS = [20, 35, 50, 75, 1000]


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    metrics = {
        "samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, (y_prob >= base.PREDICTION_THRESHOLD).astype(int))),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }

    if pd.Series(y_true).nunique(dropna=True) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
    return metrics


def select_top_features(train_df, target_col, feature_cols, limit):
    if limit >= len(feature_cols):
        return list(feature_cols)

    X_train, imputer = base.prepare_feature_matrix(train_df, feature_cols, fit_imputer=True)
    model = base.build_classifier()
    model.fit(X_train, train_df[target_col].values)

    feat_imp = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values(["importance", "feature"], ascending=[False, True])
    return feat_imp.head(limit)["feature"].tolist()


def evaluate_weight_sweep(dataset_name, df, target_col, feature_cols, backtest_year):
    rows = []

    train_split_df, val_split_df = train_test_split(
        df,
        test_size=0.20,
        random_state=base.RANDOM_STATE,
        stratify=df[target_col],
    )

    X_train_split, split_imputer = base.prepare_feature_matrix(
        train_split_df, feature_cols, fit_imputer=True
    )
    X_val_split, _ = base.prepare_feature_matrix(
        val_split_df, feature_cols, imputer=split_imputer, fit_imputer=False
    )
    split_model = base.build_classifier()
    split_model.fit(X_train_split, train_split_df[target_col].values)
    split_model_prob = split_model.predict_proba(X_val_split)[:, 1]
    split_prior_state = base.fit_draft_prior(train_split_df, target_col)
    split_prior_prob = base.score_draft_prior(val_split_df, split_prior_state)

    backtest_df = df[df["draft_year"] == backtest_year].copy()
    earlier_df = df[df["draft_year"] < backtest_year].copy()

    backtest_model_prob = None
    backtest_prior_prob = None
    if not backtest_df.empty and not earlier_df.empty:
        X_backtest_train, backtest_imputer = base.prepare_feature_matrix(
            earlier_df, feature_cols, fit_imputer=True
        )
        X_backtest_holdout, _ = base.prepare_feature_matrix(
            backtest_df, feature_cols, imputer=backtest_imputer, fit_imputer=False
        )
        backtest_model = base.build_classifier()
        backtest_model.fit(X_backtest_train, earlier_df[target_col].values)
        backtest_model_prob = backtest_model.predict_proba(X_backtest_holdout)[:, 1]
        backtest_prior_state = base.fit_draft_prior(earlier_df, target_col)
        backtest_prior_prob = base.score_draft_prior(backtest_df, backtest_prior_state)

    for weight in WEIGHT_GRID:
        split_prob = base.blend_model_with_draft_prior(split_model_prob, split_prior_prob, blend_weight=weight)
        split_metrics = compute_metrics(val_split_df[target_col].values, split_prob)
        rows.append(
            {
                "dataset": dataset_name,
                "evaluation": "random_split",
                "weight": weight,
                **split_metrics,
            }
        )

        if backtest_model_prob is not None:
            backtest_prob = base.blend_model_with_draft_prior(
                backtest_model_prob,
                backtest_prior_prob,
                blend_weight=weight,
            )
            backtest_metrics = compute_metrics(backtest_df[target_col].values, backtest_prob)
            rows.append(
                {
                    "dataset": dataset_name,
                    "evaluation": f"draft_year_backtest_{backtest_year}",
                    "weight": weight,
                    **backtest_metrics,
                }
            )

    return rows


def evaluate_feature_sweep_600yd(df, target_col, feature_cols, backtest_year, blend_weight):
    rows = []

    train_split_df, val_split_df = train_test_split(
        df,
        test_size=0.20,
        random_state=base.RANDOM_STATE,
        stratify=df[target_col],
    )

    earlier_df = df[df["draft_year"] < backtest_year].copy()
    backtest_df = df[df["draft_year"] == backtest_year].copy()

    for limit in FEATURE_LIMITS:
        selected_split_features = select_top_features(train_split_df, target_col, feature_cols, limit)
        X_train_split, split_imputer = base.prepare_feature_matrix(
            train_split_df, selected_split_features, fit_imputer=True
        )
        X_val_split, _ = base.prepare_feature_matrix(
            val_split_df, selected_split_features, imputer=split_imputer, fit_imputer=False
        )
        split_model = base.build_classifier()
        split_model.fit(X_train_split, train_split_df[target_col].values)
        split_model_prob = split_model.predict_proba(X_val_split)[:, 1]
        split_prior_state = base.fit_draft_prior(train_split_df, target_col)
        split_prior_prob = base.score_draft_prior(val_split_df, split_prior_state)
        split_blended_prob = base.blend_model_with_draft_prior(
            split_model_prob,
            split_prior_prob,
            blend_weight=blend_weight,
        )
        split_metrics = compute_metrics(val_split_df[target_col].values, split_blended_prob)
        rows.append(
            {
                "feature_limit": limit,
                "selected_feature_count": len(selected_split_features),
                "evaluation": "random_split",
                "blend_weight": blend_weight,
                **split_metrics,
            }
        )

        if not backtest_df.empty and not earlier_df.empty:
            selected_backtest_features = select_top_features(earlier_df, target_col, feature_cols, limit)
            X_backtest_train, backtest_imputer = base.prepare_feature_matrix(
                earlier_df, selected_backtest_features, fit_imputer=True
            )
            X_backtest_holdout, _ = base.prepare_feature_matrix(
                backtest_df, selected_backtest_features, imputer=backtest_imputer, fit_imputer=False
            )
            backtest_model = base.build_classifier()
            backtest_model.fit(X_backtest_train, earlier_df[target_col].values)
            backtest_model_prob = backtest_model.predict_proba(X_backtest_holdout)[:, 1]
            backtest_prior_state = base.fit_draft_prior(earlier_df, target_col)
            backtest_prior_prob = base.score_draft_prior(backtest_df, backtest_prior_state)
            backtest_blended_prob = base.blend_model_with_draft_prior(
                backtest_model_prob,
                backtest_prior_prob,
                blend_weight=blend_weight,
            )
            backtest_metrics = compute_metrics(backtest_df[target_col].values, backtest_blended_prob)
            rows.append(
                {
                    "feature_limit": limit,
                    "selected_feature_count": len(selected_backtest_features),
                    "evaluation": f"draft_year_backtest_{backtest_year}",
                    "blend_weight": blend_weight,
                    **backtest_metrics,
                }
            )

    return rows


def prepare_top40_strict():
    train_df = base.load_model_frame(base.TRAIN_FILE)
    breakout_df = base.build_breakout_features(base.WIDE_FILE)
    long_receiving_df = base.load_long_receiving_history(base.SCRIPT_DIR)
    teammate_df = base.build_teammate_context_features(long_receiving_df)
    fantasy_df = base.load_fantasy_ppg_history(base.SCRIPT_DIR)

    train_df = base.merge_breakout_features(train_df, breakout_df)
    train_df = base.merge_breakout_features(train_df, teammate_df)
    train_df, latest_fantasy_year = base.add_binary_target(train_df, fantasy_df)
    train_df = base.add_draft_capital_features(train_df)

    complete_train_df = train_df[
        (train_df["final_position"] == "WR")
        & train_df["ROUND"].notna()
        & train_df["target_top40_ppg_within_3yrs"].notna()
    ].copy()
    complete_train_df["target_top40_ppg_within_3yrs"] = complete_train_df[
        "target_top40_ppg_within_3yrs"
    ].astype(int)

    return {
        "dataset_name": "top40_strict",
        "df": complete_train_df,
        "target_col": "target_top40_ppg_within_3yrs",
        "feature_cols": base.build_feature_list(complete_train_df),
        "backtest_year": latest_fantasy_year - base.TARGET_WINDOW_YEARS + 1,
    }


def prepare_top40_recent():
    train_df = base.load_model_frame(base.TRAIN_FILE)
    breakout_df = base.build_breakout_features(base.WIDE_FILE)
    long_receiving_df = base.load_long_receiving_history(base.SCRIPT_DIR)
    teammate_df = base.build_teammate_context_features(long_receiving_df)
    fantasy_df = base.load_fantasy_ppg_history(base.SCRIPT_DIR)

    train_df = base.merge_breakout_features(train_df, breakout_df)
    train_df = base.merge_breakout_features(train_df, teammate_df)
    train_df, latest_fantasy_year = top40_recent.add_recent_hit_inclusive_target(train_df, fantasy_df)
    train_df = base.add_draft_capital_features(train_df)

    trainable_df = train_df[
        (train_df["final_position"] == "WR")
        & train_df["OVERALL_PICK"].notna()
        & train_df[top40_recent.TARGET_COL].notna()
    ].copy()
    trainable_df[top40_recent.TARGET_COL] = trainable_df[top40_recent.TARGET_COL].astype(int)
    full_window_df = trainable_df[trainable_df[top40_recent.INCLUSION_COL] == "full_window"].copy()

    return {
        "dataset_name": "top40_recent",
        "df": trainable_df,
        "target_col": top40_recent.TARGET_COL,
        "feature_cols": top40_recent.build_feature_list_recent(trainable_df),
        "backtest_year": latest_fantasy_year - base.TARGET_WINDOW_YEARS + 1,
        "full_window_df": full_window_df,
    }


def prepare_yards600():
    train_df = base.load_model_frame(base.TRAIN_FILE)
    breakout_df = base.build_breakout_features(base.WIDE_FILE)
    long_receiving_df = base.load_long_receiving_history(base.SCRIPT_DIR)
    teammate_df = base.build_teammate_context_features(long_receiving_df)
    receiving_df = yards600.load_receiving_yards_history(base.SCRIPT_DIR)

    train_df = base.merge_breakout_features(train_df, breakout_df)
    train_df = base.merge_breakout_features(train_df, teammate_df)
    train_df, latest_fantasy_year = yards600.add_binary_target(train_df, receiving_df)
    train_df = base.add_draft_capital_features(train_df)

    complete_train_df = train_df[
        (train_df["final_position"] == "WR")
        & train_df["OVERALL_PICK"].notna()
        & train_df[yards600.TARGET_COL].notna()
    ].copy()
    complete_train_df[yards600.TARGET_COL] = complete_train_df[yards600.TARGET_COL].astype(int)

    return {
        "dataset_name": "yards600_strict",
        "df": complete_train_df,
        "target_col": yards600.TARGET_COL,
        "feature_cols": yards600.build_feature_list_yards(complete_train_df),
        "backtest_year": latest_fantasy_year - yards600.TARGET_WINDOW_YEARS + 1,
    }


def main():
    dataset_specs = [
        prepare_top40_strict(),
        prepare_top40_recent(),
        prepare_yards600(),
    ]

    weight_rows = []
    for spec in dataset_specs:
        if spec["dataset_name"] == "top40_recent":
            weight_rows.extend(
                evaluate_weight_sweep(
                    spec["dataset_name"],
                    spec["df"],
                    spec["target_col"],
                    spec["feature_cols"],
                    spec["backtest_year"],
                )
            )
        else:
            weight_rows.extend(
                evaluate_weight_sweep(
                    spec["dataset_name"],
                    spec["df"],
                    spec["target_col"],
                    spec["feature_cols"],
                    spec["backtest_year"],
                )
            )

    weight_df = pd.DataFrame(weight_rows)
    weight_df.sort_values(["dataset", "evaluation", "weight"], inplace=True)
    weight_df.to_csv(WEIGHT_SWEEP_FILE, index=False)
    print(f"Saved draft blend sweep results to '{WEIGHT_SWEEP_FILE.name}'.")
    print("\nDraft blend sweep summary:")
    print(weight_df[["dataset", "evaluation", "weight", "roc_auc", "average_precision", "brier"]].to_string(index=False))

    yards_spec = next(spec for spec in dataset_specs if spec["dataset_name"] == "yards600_strict")
    feature_rows = evaluate_feature_sweep_600yd(
        yards_spec["df"],
        yards_spec["target_col"],
        yards_spec["feature_cols"],
        yards_spec["backtest_year"],
        blend_weight=0.25,
    )
    feature_df = pd.DataFrame(feature_rows)
    feature_df.sort_values(["evaluation", "feature_limit"], inplace=True)
    feature_df.to_csv(FEATURE_SWEEP_FILE, index=False)
    print(f"\nSaved 600-yard feature reduction sweep results to '{FEATURE_SWEEP_FILE.name}'.")
    print("\n600-yard feature reduction summary (blend weight 0.25):")
    print(feature_df[["evaluation", "feature_limit", "selected_feature_count", "roc_auc", "average_precision", "brier"]].to_string(index=False))


if __name__ == "__main__":
    main()
