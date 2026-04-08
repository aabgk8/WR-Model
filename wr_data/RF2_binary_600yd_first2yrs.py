from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import RF2_binary_top40 as base
import paths


TARGET_COL = "target_600yd_receiving_within_2yrs"
TARGET_WINDOW_YEARS = 2
TARGET_RECEIVING_YARDS = 600
FEATURE_LIMIT = 20
TARGET_GAMES_COL = "games_in_best_receiving_season_within_window"
BEST_YARDS_COL = "best_receiving_yards_within_window"
FIRST_HIT_YEAR_COL = "first_600yd_receiving_year"
COMPLETE_WINDOW_COL = "has_complete_600yd_target_window"
EXTRA_EXCLUDED_COLS = {
    TARGET_COL,
    TARGET_GAMES_COL,
    BEST_YARDS_COL,
    FIRST_HIT_YEAR_COL,
    COMPLETE_WINDOW_COL,
}

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_FILE = paths.MODELS_DIR / "rf_binary_600yd_first2yrs.joblib"
HOLDOUT_PREDICTIONS_FILE = paths.PREDICTIONS_DIR / "wr_holdout_2025_binary_600yd_first2yrs_predictions.csv"
BACKTEST_FILE = paths.BACKTESTS_DIR / "wr_binary_600yd_first2yrs_backtest.csv"
FEATURE_IMPORTANCE_FILE = paths.FEATURE_IMPORTANCE_DIR / "wr_binary_600yd_first2yrs_feature_importance.csv"


def parse_numeric(series):
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def load_receiving_yards_history(script_dir):
    frames = []
    for file_path in sorted(paths.FANTASY_WR_RAW_DIR.glob("FantasyPros_Fantasy_Football_Statistics_WR*.csv")):
        fantasy_year = base.extract_year_from_filename(file_path)
        if fantasy_year is None:
            continue

        df_part = pd.read_csv(file_path, low_memory=False)
        required_cols = {"Player", "YDS", "G"}
        if not required_cols.issubset(df_part.columns):
            continue

        df_part["fantasy_year"] = fantasy_year
        df_part["player_clean"] = df_part["Player"].map(base.clean_player_name)
        df_part["receiving_yards"] = parse_numeric(df_part["YDS"])
        df_part["games"] = parse_numeric(df_part["G"])
        df_part["hit_600yd"] = df_part["receiving_yards"] >= TARGET_RECEIVING_YARDS

        frames.append(
            df_part[
                [
                    "fantasy_year",
                    "player_clean",
                    "games",
                    "receiving_yards",
                    "hit_600yd",
                ]
            ]
        )

    if not frames:
        raise FileNotFoundError("No FantasyPros WR history files were found for the 600-yard target.")

    receiving_df = pd.concat(frames, ignore_index=True)
    receiving_df = receiving_df.groupby(["player_clean", "fantasy_year"], as_index=False).agg(
        {
            "games": "max",
            "receiving_yards": "max",
            "hit_600yd": "max",
        }
    )
    return receiving_df


def add_binary_target(df, receiving_df):
    df = base.add_draft_year(df)
    df["player_clean"] = df["player"].map(base.clean_player_name)

    latest_fantasy_year = int(receiving_df["fantasy_year"].max())
    df[COMPLETE_WINDOW_COL] = df["draft_year"].notna() & (
        df["draft_year"] + TARGET_WINDOW_YEARS - 1 <= latest_fantasy_year
    )

    receiving_lookup = receiving_df.set_index(["player_clean", "fantasy_year"])[
        ["games", "receiving_yards", "hit_600yd"]
    ]

    target_values = []
    best_yards = []
    best_games = []
    first_hit_years = []

    for _, row in df.iterrows():
        if pd.isna(row["draft_year"]):
            target_values.append(np.nan)
            best_yards.append(np.nan)
            best_games.append(np.nan)
            first_hit_years.append(np.nan)
            continue

        draft_year = int(row["draft_year"])
        years = range(draft_year, draft_year + TARGET_WINDOW_YEARS)
        yearly_rows = []
        hit_year = np.nan

        for year in years:
            key = (row["player_clean"], year)
            if key not in receiving_lookup.index:
                continue
            lookup_row = receiving_lookup.loc[key]
            yards = lookup_row["receiving_yards"]
            games = lookup_row["games"]
            hit_600yd = lookup_row["hit_600yd"]
            yearly_rows.append((year, yards, games))
            if bool(hit_600yd) and pd.isna(hit_year):
                hit_year = year

        if yearly_rows:
            best_year = max(
                yearly_rows,
                key=lambda season_row: (
                    -np.inf if pd.isna(season_row[1]) else season_row[1],
                    -np.inf if pd.isna(season_row[2]) else season_row[2],
                ),
            )
            best_yards.append(best_year[1])
            best_games.append(best_year[2])
        else:
            best_yards.append(np.nan)
            best_games.append(np.nan)

        first_hit_years.append(hit_year)

        if not bool(row[COMPLETE_WINDOW_COL]):
            target_values.append(np.nan)
        else:
            target_values.append(int(pd.notna(hit_year)))

    df[BEST_YARDS_COL] = best_yards
    df[TARGET_GAMES_COL] = best_games
    df[FIRST_HIT_YEAR_COL] = first_hit_years
    df[TARGET_COL] = target_values
    return df, latest_fantasy_year


def build_feature_list_yards(df):
    return [col for col in base.build_feature_list(df) if col not in EXTRA_EXCLUDED_COLS]


def save_feature_importances(model, feature_cols):
    feat_imp = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feat_imp.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
    print(f"\nSaved feature importances to '{FEATURE_IMPORTANCE_FILE.name}'.")
    print("\nTop 25 feature importances:")
    print(feat_imp.head(25))
    return feat_imp


def main():
    paths.ensure_directories()
    train_df = base.load_model_frame(base.TRAIN_FILE)
    holdout_df = base.load_model_frame(base.HOLDOUT_FILE)
    breakout_df = base.build_breakout_features(base.WIDE_FILE)
    long_receiving_df = base.load_long_receiving_history(base.SCRIPT_DIR)
    teammate_df = base.build_teammate_context_features(long_receiving_df)
    receiving_df = load_receiving_yards_history(base.SCRIPT_DIR)

    train_df = base.merge_breakout_features(train_df, breakout_df)
    holdout_df = base.merge_breakout_features(holdout_df, breakout_df)
    train_df = base.merge_breakout_features(train_df, teammate_df)
    holdout_df = base.merge_breakout_features(holdout_df, teammate_df)

    train_df, latest_fantasy_year = add_binary_target(train_df, receiving_df)
    holdout_df, _ = add_binary_target(holdout_df, receiving_df)
    train_df = base.add_draft_capital_features(train_df)
    holdout_df = base.add_draft_capital_features(holdout_df)

    complete_train_df = train_df[
        (train_df["final_position"] == "WR")
        & train_df["OVERALL_PICK"].notna()
        & train_df[TARGET_COL].notna()
    ].copy()
    complete_train_df[TARGET_COL] = complete_train_df[TARGET_COL].astype(int)

    print(f"Latest fantasy season available: {latest_fantasy_year}")
    print(f"Training rows with complete {TARGET_WINDOW_YEARS}-year target windows: {complete_train_df.shape[0]}")
    print(
        "Positive class count: "
        f"{int(complete_train_df[TARGET_COL].sum())} / {complete_train_df.shape[0]}"
    )

    feature_cols = build_feature_list_yards(complete_train_df)
    print(f"Candidate feature count before reduction: {len(feature_cols)}")

    train_split_df, val_split_df = train_test_split(
        complete_train_df,
        test_size=0.20,
        random_state=base.RANDOM_STATE,
        stratify=complete_train_df[TARGET_COL],
    )
    selected_split_features = base.select_top_features_by_importance(
        train_split_df,
        TARGET_COL,
        feature_cols,
        FEATURE_LIMIT,
    )
    print(f"Random split selected feature count: {len(selected_split_features)}")
    X_train_split, split_imputer = base.prepare_feature_matrix(
        train_split_df, selected_split_features, fit_imputer=True
    )
    X_val_split, _ = base.prepare_feature_matrix(
        val_split_df, selected_split_features, imputer=split_imputer, fit_imputer=False
    )

    split_model = base.build_classifier()
    split_model.fit(X_train_split, train_split_df[TARGET_COL].values)
    split_val_prob = split_model.predict_proba(X_val_split)[:, 1]
    split_prior_state = base.fit_draft_prior(train_split_df, TARGET_COL)
    split_val_draft_prior = base.score_draft_prior(val_split_df, split_prior_state)
    split_val_blended_prob = base.blend_model_with_draft_prior(split_val_prob, split_val_draft_prior)
    base.print_metrics(
        "Random split validation (600 receiving yards within 2 years, model only)",
        val_split_df[TARGET_COL].values,
        split_val_prob,
    )
    base.print_metrics(
        f"Random split validation (600 receiving yards within 2 years, draft-blended, weight={base.DRAFT_PRIOR_BLEND_WEIGHT:.2f})",
        val_split_df[TARGET_COL].values,
        split_val_blended_prob,
    )

    latest_complete_draft_year = latest_fantasy_year - TARGET_WINDOW_YEARS + 1
    backtest_df = complete_train_df[complete_train_df["draft_year"] == latest_complete_draft_year].copy()
    earlier_df = complete_train_df[complete_train_df["draft_year"] < latest_complete_draft_year].copy()

    if not backtest_df.empty and not earlier_df.empty:
        selected_backtest_features = base.select_top_features_by_importance(
            earlier_df,
            TARGET_COL,
            feature_cols,
            FEATURE_LIMIT,
        )
        print(f"Backtest selected feature count: {len(selected_backtest_features)}")
        X_backtest_train, backtest_imputer = base.prepare_feature_matrix(
            earlier_df, selected_backtest_features, fit_imputer=True
        )
        X_backtest_holdout, _ = base.prepare_feature_matrix(
            backtest_df, selected_backtest_features, imputer=backtest_imputer, fit_imputer=False
        )

        backtest_model = base.build_classifier()
        backtest_model.fit(X_backtest_train, earlier_df[TARGET_COL].values)
        backtest_prob = backtest_model.predict_proba(X_backtest_holdout)[:, 1]
        backtest_prior_state = base.fit_draft_prior(earlier_df, TARGET_COL)
        backtest_draft_prior = base.score_draft_prior(backtest_df, backtest_prior_state)
        backtest_blended_prob = base.blend_model_with_draft_prior(backtest_prob, backtest_draft_prior)
        base.print_metrics(
            f"Draft-year backtest ({latest_complete_draft_year}, model only)",
            backtest_df[TARGET_COL].values,
            backtest_prob,
        )
        base.print_metrics(
            f"Draft-year backtest ({latest_complete_draft_year}, draft-blended)",
            backtest_df[TARGET_COL].values,
            backtest_blended_prob,
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
                "draft_pick_bucket_ordinal",
                "is_round1_pick",
                "is_top10_pick",
                TARGET_COL,
                BEST_YARDS_COL,
                TARGET_GAMES_COL,
                FIRST_HIT_YEAR_COL,
                "college_seasons_played",
                "first_breakout_season_index",
                "early_breakout_flag",
                "final_team_target_share",
                "final_top_teammate_target_share",
                "final_season_with_dominant_teammate",
            ]
        ].copy()
        backtest_output["model_prob_600yd_within_2yrs"] = backtest_prob
        backtest_output["draft_prior_prob"] = backtest_draft_prior
        backtest_output["prob_600yd_within_2yrs"] = backtest_blended_prob
        backtest_output["draft_blend_weight"] = base.DRAFT_PRIOR_BLEND_WEIGHT
        backtest_output["predicted_600yd_within_2yrs"] = (
            backtest_output["prob_600yd_within_2yrs"] >= base.PREDICTION_THRESHOLD
        ).astype(int)
        backtest_output.sort_values("prob_600yd_within_2yrs", ascending=False, inplace=True)
        backtest_output.to_csv(BACKTEST_FILE, index=False)
        print(f"Saved backtest predictions to '{BACKTEST_FILE.name}'.")

    selected_final_features = base.select_top_features_by_importance(
        complete_train_df,
        TARGET_COL,
        feature_cols,
        FEATURE_LIMIT,
    )
    print(f"Final model selected feature count: {len(selected_final_features)}")
    X_full, final_imputer = base.prepare_feature_matrix(
        complete_train_df,
        selected_final_features,
        fit_imputer=True,
    )
    final_model = base.build_classifier()
    final_model.fit(X_full, complete_train_df[TARGET_COL].values)
    final_prior_state = base.fit_draft_prior(complete_train_df, TARGET_COL)
    save_feature_importances(final_model, selected_final_features)

    model_artifact = {
        "model": final_model,
        "imputer": final_imputer,
        "draft_prior_state": final_prior_state,
        "feature_names": selected_final_features,
        "config": {
            "target_variant": "600_receiving_yards_within_2yrs",
            "target_receiving_yards": TARGET_RECEIVING_YARDS,
            "target_window_years": TARGET_WINDOW_YEARS,
            "feature_limit": FEATURE_LIMIT,
            "prediction_threshold": base.PREDICTION_THRESHOLD,
            "draft_prior_blend_weight": base.DRAFT_PRIOR_BLEND_WEIGHT,
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
        prospective_df, selected_final_features, imputer=final_imputer, fit_imputer=False
    )
    prospective_prob = final_model.predict_proba(X_holdout)[:, 1]
    prospective_draft_prior = base.score_draft_prior(prospective_df, final_prior_state)
    prospective_blended_prob = base.blend_model_with_draft_prior(prospective_prob, prospective_draft_prior)

    prospective_output = prospective_df[
        [
            "player",
            "OVERALL_PICK",
            "ROUND",
            "SELECTION",
            "HEIGHT",
            "WEIGHT",
            "draft_pick_bucket_ordinal",
            "is_round1_pick",
            "is_top10_pick",
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
    prospective_output["model_prob_600yd_within_2yrs"] = prospective_prob
    prospective_output["draft_prior_prob"] = prospective_draft_prior
    prospective_output["prob_600yd_within_2yrs"] = prospective_blended_prob
    prospective_output["draft_blend_weight"] = base.DRAFT_PRIOR_BLEND_WEIGHT
    prospective_output["predicted_600yd_within_2yrs"] = (
        prospective_output["prob_600yd_within_2yrs"] >= base.PREDICTION_THRESHOLD
    ).astype(int)
    prospective_output.sort_values("prob_600yd_within_2yrs", ascending=False, inplace=True)
    prospective_output.to_csv(HOLDOUT_PREDICTIONS_FILE, index=False)

    print(f"\nSaved prospective predictions to '{HOLDOUT_PREDICTIONS_FILE.name}'.")
    print("\nTop projected drafted WRs by probability:")
    print(prospective_output.head(base.TOP_N_PRINT))


if __name__ == "__main__":
    main()
