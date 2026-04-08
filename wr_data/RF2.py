from pathlib import Path
import re
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import paths

TARGET_COL = "career_fpts_per_game"
TOP_N_FEATURES = 40
TOP_N_PRINT = 100
HOLDOUT_YEAR = 2025
FILTER_ZERO_TARGET = True
FILTER_TO_HISTORICAL_DRAFTED_WRS = True
FILTER_HOLDOUT_TO_PROJECTED_DRAFTED_WRS = True

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = paths.processed_train_file(HOLDOUT_YEAR)
HOLDOUT_FILE = paths.processed_holdout_file(HOLDOUT_YEAR)
PREDICTIONS_FILE = paths.PREDICTIONS_DIR / f"wr_holdout_{HOLDOUT_YEAR}_predictions_3.csv"

def get_latest_fantasy_year(script_dir=SCRIPT_DIR):
    latest_year = None
    for file_path in paths.FANTASY_WR_RAW_DIR.glob("FantasyPros_Fantasy_Football_Statistics_*.csv"):
        match = re.search(r"(\d{4})\.csv$", file_path.name)
        if not match:
            continue
        year = int(match.group(1))
        if latest_year is None or year > latest_year:
            latest_year = year
    return latest_year

def train_rf_for_wr_all_features(train_file=TRAIN_FILE):
    df = pd.read_csv(train_file)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from '{Path(train_file).name}'.")

    if "final_position" not in df.columns:
        print("Error: 'final_position' column not found. Aborting.")
        return None, None, None

    df = df[df["final_position"] == "WR"].copy()
    print(f"After filtering final_position == 'WR': {df.shape[0]} rows remain.")

    if FILTER_TO_HISTORICAL_DRAFTED_WRS and "is_historical_drafted_wr" in df.columns:
        before_rows = len(df)
        df = df[df["is_historical_drafted_wr"] == True].copy()
        print(f"After filtering to historically drafted WRs: {df.shape[0]} rows remain (dropped {before_rows - len(df)}).")

    if TARGET_COL not in df.columns:
        print(f"Error: target_col '{TARGET_COL}' not in data. Aborting.")
        return None, None, None

    drop_cols = [
        "player", "player_id", "final_team_name", "career_Season_range",
        "final_position", "final_franchise_id", "matched_draft_year",
        "draft_match_method", "draft_match_score", "is_historical_drafted_wr",
    ]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    df = df.dropna(subset=[TARGET_COL])
    if FILTER_ZERO_TARGET:
        before_rows = len(df)
        df = df[df[TARGET_COL] > 0].copy()
        print(f"After filtering {TARGET_COL} > 0: {df.shape[0]} rows remain (dropped {before_rows - len(df)}).")

    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_imputed = imputer.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    score = rf.score(X_val, y_val)
    print(f"Random Forest R^2 on validation set (all features, WR only): {score:.4f}")

    feat_imp = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nTop 50 Features by Random Forest Importance (all features, WR):")
    print(feat_imp.head(50))

    return rf, X.columns, df

def retrain_top_features(df_preprocessed, original_rf, top_n=TOP_N_FEATURES):
    drop_cols = [
        "player", "player_id", "final_team_name", "career_Season_range",
        "final_position", "final_franchise_id", "matched_draft_year",
        "draft_match_method", "draft_match_score", "is_historical_drafted_wr",
    ]
    temp_df = df_preprocessed.drop(columns=drop_cols, errors="ignore")

    if TARGET_COL not in temp_df.columns:
        print(f"Error: '{TARGET_COL}' not in preprocessed DF. Aborting retrain.")
        return None, None

    temp_df = temp_df.dropna(subset=[TARGET_COL])
    y = temp_df[TARGET_COL].values
    X = temp_df.drop(columns=[TARGET_COL])

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    imputer = SimpleImputer(strategy="constant", fill_value=0)
    importances = original_rf.feature_importances_
    feat_imp_df = pd.DataFrame({"feature": X.columns, "importance": importances})
    feat_imp_df.sort_values("importance", ascending=False, inplace=True)

    top_features = feat_imp_df.head(top_n)["feature"].tolist()
    print(f"\nSelected top {top_n} features:\n{top_features}")

    X_top = X[top_features].copy()
    X_top_array = imputer.fit_transform(X_top)

    X_tr, X_val, y_tr, y_val = train_test_split(X_top_array, y, test_size=0.2, random_state=42)
    rf_top = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_top.fit(X_tr, y_tr)

    score_top = rf_top.score(X_val, y_val)
    print(f"Random Forest R^2 on validation set (top {top_n} features): {score_top:.4f}")

    return rf_top, top_features

def predict_holdout_top_features(rf_top, top_features, holdout_file=HOLDOUT_FILE, top_print=TOP_N_PRINT):
    df_holdout = pd.read_csv(holdout_file)
    print(f"\nLoaded holdout shape: {df_holdout.shape} from '{Path(holdout_file).name}'.")

    if "final_position" not in df_holdout.columns:
        print("Error: 'final_position' not found in holdout data.")
        return

    df_wr = df_holdout[df_holdout["final_position"] == "WR"].copy()
    print(f"After WR filter: {df_wr.shape[0]} rows remain in holdout set.")
    original_holdout_wr = df_wr.copy()

    if FILTER_HOLDOUT_TO_PROJECTED_DRAFTED_WRS:
        expected_draft_year = HOLDOUT_YEAR + 1
        if "matched_draft_year" in df_wr.columns:
            before_rows = len(df_wr)
            matched_year = pd.to_numeric(df_wr["matched_draft_year"], errors="coerce")
            filtered_df = df_wr[matched_year == expected_draft_year].copy()
            if not filtered_df.empty:
                df_wr = filtered_df
                print(
                    f"After filtering holdout to projected drafted WRs for {expected_draft_year}: "
                    f"{df_wr.shape[0]} rows remain (dropped {before_rows - len(df_wr)})."
                )

        if "ROUND" in df_wr.columns and (
            not FILTER_HOLDOUT_TO_PROJECTED_DRAFTED_WRS
            or len(df_wr) == 0
            or df_wr["ROUND"].notna().sum() < len(df_wr)
        ):
            before_rows = len(original_holdout_wr)
            round_col = pd.to_numeric(original_holdout_wr["ROUND"], errors="coerce")
            df_wr = original_holdout_wr[round_col.notna()].copy()
            print(
                "After filtering holdout to rows with ROUND populated: "
                f"{df_wr.shape[0]} rows remain (dropped {before_rows - len(df_wr)})."
            )

    if df_wr.empty:
        print("No holdout WR rows remain after filtering. Aborting holdout prediction step.")
        return

    if "player" not in df_wr.columns:
        print("Error: 'player' column not found in holdout data.")
        return

    holdout_names = df_wr["player"].copy()
    actual_target = df_wr[TARGET_COL].copy() if TARGET_COL in df_wr.columns else None
    latest_fantasy_year = get_latest_fantasy_year()
    if actual_target is not None and latest_fantasy_year is not None and "final_season" in df_wr.columns:
        eligible_for_actuals = pd.to_numeric(df_wr["final_season"], errors="coerce") + 1 <= latest_fantasy_year
        actual_target = actual_target.where(eligible_for_actuals)

    drop_cols = [
        "player_id", "final_team_name", "career_Season_range",
        "final_position", "final_franchise_id", TARGET_COL, "matched_draft_year",
        "draft_match_method", "draft_match_score", "is_historical_drafted_wr",
    ]
    for col in drop_cols:
        if col in df_wr.columns:
            df_wr.drop(columns=col, inplace=True)

    used_feats = [feature for feature in top_features if feature in df_wr.columns]
    for col in used_feats:
        if df_wr[col].dtype == object:
            df_wr[col] = pd.to_numeric(df_wr[col], errors="coerce")

    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_holdout = imputer.fit_transform(df_wr[used_feats])
    preds = rf_top.predict(X_holdout)

    df_wr["player"] = holdout_names
    if actual_target is not None:
        df_wr["actual_fpts_per_game"] = actual_target.values
    df_wr["predicted_fpts_per_game"] = preds
    df_wr.sort_values("predicted_fpts_per_game", ascending=False, inplace=True)

    if actual_target is not None:
        eval_df = df_wr.dropna(subset=["actual_fpts_per_game"]).copy()
        if not eval_df.empty:
            full_r2 = r2_score(eval_df["actual_fpts_per_game"], eval_df["predicted_fpts_per_game"])
            full_mae = mean_absolute_error(eval_df["actual_fpts_per_game"], eval_df["predicted_fpts_per_game"])
            print(f"\nHoldout metrics on all labeled WRs: R^2={full_r2:.4f}, MAE={full_mae:.4f}")

            positive_eval_df = eval_df[eval_df["actual_fpts_per_game"] > 0].copy()
            if not positive_eval_df.empty:
                pos_r2 = r2_score(
                    positive_eval_df["actual_fpts_per_game"],
                    positive_eval_df["predicted_fpts_per_game"],
                )
                pos_mae = mean_absolute_error(
                    positive_eval_df["actual_fpts_per_game"],
                    positive_eval_df["predicted_fpts_per_game"],
                )
                print(f"Holdout metrics on labeled WRs with actual_fpts_per_game > 0: R^2={pos_r2:.4f}, MAE={pos_mae:.4f}")

    top_df = df_wr.head(top_print)
    print(f"\nTop {top_print} WRs by predicted_fpts_per_game:")
    display_cols = ["player", "predicted_fpts_per_game"]
    if "actual_fpts_per_game" in top_df.columns:
        display_cols.append("actual_fpts_per_game")
    print(top_df[display_cols])

    df_wr.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Saved predictions to '{PREDICTIONS_FILE.name}', shape={df_wr.shape}.")

def main():
    paths.ensure_directories()
    rf_all, _, df_prep = train_rf_for_wr_all_features(TRAIN_FILE)
    if rf_all is None:
        print("Error training on all features. Exiting.")
        return

    rf_top, top_feats = retrain_top_features(df_prep, rf_all, top_n=TOP_N_FEATURES)
    if rf_top is None:
        print("Error retraining on top features. Exiting.")
        return

    predict_holdout_top_features(rf_top, top_feats, HOLDOUT_FILE, top_print=TOP_N_PRINT)

if __name__ == "__main__":
    main()
