from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import RF2_binary_top40 as base


TARGET_COL = "target_top40_ppg_within3yrs_recent_hit_inclusive"
INCLUSION_COL = "label_inclusion_type"
RECENT_TARGET_EXCLUDED_COLS = {
    TARGET_COL,
    INCLUSION_COL,
    "available_nfl_years_for_target",
    "early_observed_top40_hit",
}

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_FILE = SCRIPT_DIR / "rf_binary_top40_within3yrs_recent_hits.joblib"
HOLDOUT_PREDICTIONS_FILE = (
    SCRIPT_DIR / "wr_holdout_2025_binary_top40_within3yrs_recent_hits_predictions.csv"
)
BACKTEST_FILE = SCRIPT_DIR / "wr_binary_top40_within3yrs_recent_hits_backtest.csv"
FEATURE_IMPORTANCE_FILE = SCRIPT_DIR / "wr_binary_top40_within3yrs_recent_hits_feature_importance.csv"


def add_recent_hit_inclusive_target(df, fantasy_df):
    df = base.add_draft_year(df)
    df["player_clean"] = df["player"].map(base.clean_player_name)

    latest_fantasy_year = int(fantasy_df["fantasy_year"].max())
    df["has_complete_target_window"] = df["draft_year"].notna() & (
        df["draft_year"] + base.TARGET_WINDOW_YEARS - 1 <= latest_fantasy_year
    )

    fantasy_lookup = fantasy_df.set_index(["player_clean", "fantasy_year"])[["ppg_rank", "top_n_ppg_hit"]]

    target_values = []
    best_ppg_ranks = []
    first_hit_years = []
    available_year_counts = []
    early_observed_hits = []
    inclusion_types = []

    for _, row in df.iterrows():
        if pd.isna(row["draft_year"]):
            target_values.append(np.nan)
            best_ppg_ranks.append(np.nan)
            first_hit_years.append(np.nan)
            available_year_counts.append(0)
            early_observed_hits.append(np.nan)
            inclusion_types.append(np.nan)
            continue

        draft_year = int(row["draft_year"])
        max_observed_year = min(draft_year + base.TARGET_WINDOW_YEARS - 1, latest_fantasy_year)
        if draft_year > max_observed_year:
            observed_years = []
        else:
            observed_years = list(range(draft_year, max_observed_year + 1))

        available_year_counts.append(len(observed_years))

        ranks = []
        hit_year = np.nan
        for year in observed_years:
            key = (row["player_clean"], year)
            if key not in fantasy_lookup.index:
                continue
            ppg_rank = fantasy_lookup.loc[key, "ppg_rank"]
            top_hit = fantasy_lookup.loc[key, "top_n_ppg_hit"]
            if pd.notna(ppg_rank):
                ranks.append(float(ppg_rank))
            if bool(top_hit) and pd.isna(hit_year):
                hit_year = year

        best_ppg_ranks.append(min(ranks) if ranks else np.nan)
        first_hit_years.append(hit_year)
        early_hit = int(pd.notna(hit_year)) if observed_years and len(observed_years) < base.TARGET_WINDOW_YEARS else 0
        early_observed_hits.append(early_hit)

        if bool(row["has_complete_target_window"]):
            target_values.append(int(pd.notna(hit_year)))
            inclusion_types.append("full_window")
        elif len(observed_years) in {1, 2} and pd.notna(hit_year):
            target_values.append(1)
            inclusion_types.append(f"recent_positive_years_{len(observed_years)}")
        else:
            target_values.append(np.nan)
            inclusion_types.append(np.nan)

    df["best_ppg_rank_within_window"] = best_ppg_ranks
    df["first_top40_ppg_year"] = first_hit_years
    df["available_nfl_years_for_target"] = available_year_counts
    df["early_observed_top40_hit"] = early_observed_hits
    df[INCLUSION_COL] = inclusion_types
    df[TARGET_COL] = target_values
    return df, latest_fantasy_year


def save_feature_importances(model, feature_cols):
    feat_imp = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feat_imp.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
    print(f"\nSaved feature importances to '{FEATURE_IMPORTANCE_FILE.name}'.")
    print("\nTop 25 feature importances:")
    print(feat_imp.head(25))
    return feat_imp


def build_feature_list_recent(df):
    return [col for col in base.build_feature_list(df) if col not in RECENT_TARGET_EXCLUDED_COLS]


def main():
    train_df = base.load_model_frame(base.TRAIN_FILE)
    holdout_df = base.load_model_frame(base.HOLDOUT_FILE)
    breakout_df = base.build_breakout_features(base.WIDE_FILE)
    long_receiving_df = base.load_long_receiving_history(base.SCRIPT_DIR)
    teammate_df = base.build_teammate_context_features(long_receiving_df)
    fantasy_df = base.load_fantasy_ppg_history(base.SCRIPT_DIR)

    train_df = base.merge_breakout_features(train_df, breakout_df)
    holdout_df = base.merge_breakout_features(holdout_df, breakout_df)
    train_df = base.merge_breakout_features(train_df, teammate_df)
    holdout_df = base.merge_breakout_features(holdout_df, teammate_df)

    train_df, latest_fantasy_year = add_recent_hit_inclusive_target(train_df, fantasy_df)
    holdout_df, _ = add_recent_hit_inclusive_target(holdout_df, fantasy_df)

    trainable_df = train_df[
        (train_df["final_position"] == "WR")
        & train_df["OVERALL_PICK"].notna()
        & train_df[TARGET_COL].notna()
    ].copy()
    trainable_df[TARGET_COL] = trainable_df[TARGET_COL].astype(int)

    full_window_df = trainable_df[trainable_df[INCLUSION_COL] == "full_window"].copy()
    recent_positive_df = trainable_df[trainable_df[INCLUSION_COL] != "full_window"].copy()

    print(f"Latest fantasy season available: {latest_fantasy_year}")
    print(f"Training rows with full 3-year windows: {full_window_df.shape[0]}")
    print(f"Recent early-hit positive additions: {recent_positive_df.shape[0]}")
    print(f"Total training rows in recent-hit-inclusive set: {trainable_df.shape[0]}")
    print(
        "Positive class count: "
        f"{int(trainable_df[TARGET_COL].sum())} / {trainable_df.shape[0]}"
    )

    feature_cols = build_feature_list_recent(trainable_df)

    train_split_df, val_split_df = train_test_split(
        trainable_df,
        test_size=0.20,
        random_state=base.RANDOM_STATE,
        stratify=trainable_df[TARGET_COL],
    )
    X_train_split, split_imputer = base.prepare_feature_matrix(
        train_split_df, feature_cols, fit_imputer=True
    )
    X_val_split, _ = base.prepare_feature_matrix(
        val_split_df, feature_cols, imputer=split_imputer, fit_imputer=False
    )

    split_model = base.build_classifier()
    split_model.fit(X_train_split, train_split_df[TARGET_COL].values)
    split_val_prob = split_model.predict_proba(X_val_split)[:, 1]
    base.print_metrics(
        "Random split validation (recent-hit-inclusive)",
        val_split_df[TARGET_COL].values,
        split_val_prob,
    )

    latest_complete_draft_year = latest_fantasy_year - base.TARGET_WINDOW_YEARS + 1
    backtest_df = full_window_df[full_window_df["draft_year"] == latest_complete_draft_year].copy()
    earlier_df = full_window_df[full_window_df["draft_year"] < latest_complete_draft_year].copy()

    if not backtest_df.empty and not earlier_df.empty:
        X_backtest_train, backtest_imputer = base.prepare_feature_matrix(
            earlier_df, feature_cols, fit_imputer=True
        )
        X_backtest_holdout, _ = base.prepare_feature_matrix(
            backtest_df, feature_cols, imputer=backtest_imputer, fit_imputer=False
        )

        backtest_model = base.build_classifier()
        backtest_model.fit(X_backtest_train, earlier_df[TARGET_COL].values)
        backtest_prob = backtest_model.predict_proba(X_backtest_holdout)[:, 1]
        base.print_metrics(
            f"Draft-year backtest ({latest_complete_draft_year})",
            backtest_df[TARGET_COL].values,
            backtest_prob,
        )

        backtest_output = backtest_df[
            [
                "player",
                "draft_year",
                "OVERALL_PICK",
                "ROUND",
                "SELECTION",
                "HEIGHT",
                "WEIGHT",
                TARGET_COL,
                "best_ppg_rank_within_window",
                "first_top40_ppg_year",
                "college_seasons_played",
                "first_breakout_season_index",
                "early_breakout_flag",
                "final_team_target_share",
                "final_top_teammate_target_share",
                "final_season_with_dominant_teammate",
            ]
        ].copy()
        backtest_output["prob_top40_ppg_within_3yrs"] = backtest_prob
        backtest_output["predicted_top40_ppg_within_3yrs"] = (
            backtest_output["prob_top40_ppg_within_3yrs"] >= base.PREDICTION_THRESHOLD
        ).astype(int)
        backtest_output.sort_values("prob_top40_ppg_within_3yrs", ascending=False, inplace=True)
        backtest_output.to_csv(BACKTEST_FILE, index=False)
        print(f"Saved backtest predictions to '{BACKTEST_FILE.name}'.")

    X_full, final_imputer = base.prepare_feature_matrix(trainable_df, feature_cols, fit_imputer=True)
    final_model = base.build_classifier()
    final_model.fit(X_full, trainable_df[TARGET_COL].values)
    save_feature_importances(final_model, feature_cols)

    model_artifact = {
        "model": final_model,
        "imputer": final_imputer,
        "feature_names": feature_cols,
        "config": {
            "target_variant": "recent_hit_inclusive",
            "target_top_n": base.TARGET_TOP_N,
            "target_window_years": base.TARGET_WINDOW_YEARS,
            "min_games_for_ppg_rank": base.MIN_GAMES_FOR_PPG_RANK,
            "prediction_threshold": base.PREDICTION_THRESHOLD,
            "holdout_year": base.HOLDOUT_YEAR,
            "latest_fantasy_year": latest_fantasy_year,
        },
    }
    joblib.dump(model_artifact, MODEL_FILE)
    print(f"Saved binary model artifact to '{MODEL_FILE.name}'.")

    prospective_df = holdout_df[
        (holdout_df["final_position"] == "WR")
        & holdout_df["OVERALL_PICK"].notna()
    ].copy()

    X_holdout, _ = base.prepare_feature_matrix(
        prospective_df, feature_cols, imputer=final_imputer, fit_imputer=False
    )
    prospective_prob = final_model.predict_proba(X_holdout)[:, 1]

    prospective_output = prospective_df[
        [
            "player",
            "OVERALL_PICK",
            "ROUND",
            "SELECTION",
            "HEIGHT",
            "WEIGHT",
            "draft_year",
            "college_seasons_played",
            "first_breakout_season_index",
            "early_breakout_flag",
            "breakout_700yd_season_index",
            "breakout_2yprr_season_index",
            "final_team_target_share",
            "final_team_yard_share",
            "final_top_teammate_target_share",
            "final_top_teammate_yard_share",
            "final_teammate_800yd_count",
            "final_season_with_dominant_teammate",
        ]
    ].copy()
    prospective_output["prob_top40_ppg_within_3yrs"] = prospective_prob
    prospective_output["predicted_top40_ppg_within_3yrs"] = (
        prospective_output["prob_top40_ppg_within_3yrs"] >= base.PREDICTION_THRESHOLD
    ).astype(int)
    prospective_output.sort_values("prob_top40_ppg_within_3yrs", ascending=False, inplace=True)
    prospective_output.to_csv(HOLDOUT_PREDICTIONS_FILE, index=False)

    print(f"\nSaved prospective predictions to '{HOLDOUT_PREDICTIONS_FILE.name}'.")
    print("\nTop projected drafted WRs by probability:")
    print(prospective_output.head(base.TOP_N_PRINT))


if __name__ == "__main__":
    main()
