from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

import paths

HOLDOUT_YEAR = 2025
TARGET_TOP_N = 40
TARGET_WINDOW_YEARS = 3
MIN_GAMES_FOR_PPG_RANK = 8
PREDICTION_THRESHOLD = 0.50
RANDOM_STATE = 42
TOP_N_PRINT = 20
RAW_FILE_RE = re.compile(r"receiving_summary(\d{2})\.csv$", re.IGNORECASE)

BREAKOUT_YARDS_THRESHOLD = 700
BREAKOUT_ELITE_YARDS_THRESHOLD = 900
BREAKOUT_TARGETS_THRESHOLD = 100
BREAKOUT_TOUCHDOWNS_THRESHOLD = 8
BREAKOUT_YPRR_THRESHOLD = 2.0
BREAKOUT_MIN_ROUTES = 100
DOMINANT_TEAMMATE_TARGET_SHARE_THRESHOLD = 0.20
DOMINANT_TEAMMATE_YARD_SHARE_THRESHOLD = 0.25
STRONG_TEAMMATE_500_YARDS = 500
STRONG_TEAMMATE_800_YARDS = 800
HEIGHT_WEIGHT_FEATURES = {"HEIGHT", "WEIGHT"}
DRAFT_PRIOR_BLEND_WEIGHT = 0.25
DRAFT_PRIOR_SMOOTHING = 12.0
DRAFT_PICK_BUCKET_BINS = [0, 10, 20, 32, 50, 75, 100, 150, np.inf]
EXCLUDED_EXTRA_MEASURABLES = {
    "AGE_ON_DRAFT_DAY",
    "HAND_SIZE",
    "ARM_LENGTH",
    "WINGSPAN",
    "FORTY_TIME",
    "TEN_SPLIT",
    "TWENTY_SPLIT",
    "VERTICAL",
    "BROAD",
    "SHUTTLE",
    "THREE_CONE",
    "BENCH",
    "SPEED",
}

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = paths.processed_train_file(HOLDOUT_YEAR)
HOLDOUT_FILE = paths.processed_holdout_file(HOLDOUT_YEAR)
WIDE_FILE = paths.RECEIVING_SUMMARY_WIDE_FILE
MODEL_FILE = paths.MODELS_DIR / f"rf_binary_top{TARGET_TOP_N}_within{TARGET_WINDOW_YEARS}yrs.joblib"
HOLDOUT_PREDICTIONS_FILE = (
    paths.PREDICTIONS_DIR
    / f"wr_holdout_{HOLDOUT_YEAR}_binary_top{TARGET_TOP_N}_within{TARGET_WINDOW_YEARS}yrs_predictions.csv"
)
BACKTEST_FILE = (
    paths.BACKTESTS_DIR / f"wr_binary_top{TARGET_TOP_N}_within{TARGET_WINDOW_YEARS}yrs_backtest.csv"
)
FEATURE_IMPORTANCE_FILE = (
    paths.FEATURE_IMPORTANCE_DIR
    / f"wr_binary_top{TARGET_TOP_N}_within{TARGET_WINDOW_YEARS}yrs_feature_importance.csv"
)


def clean_player_name(name):
    if not isinstance(name, str):
        return name
    return re.sub(r"\(.*?\)", "", name).strip()


def extract_year_from_filename(file_path):
    match = re.search(r"(\d{4})\.csv$", Path(file_path).name)
    return int(match.group(1)) if match else None


def extract_college_year_from_filename(filename):
    match = RAW_FILE_RE.fullmatch(Path(filename).name)
    if not match:
        return None
    return 2000 + int(match.group(1))


def find_raw_summary_files(search_dirs):
    discovered = {}
    for base_dir in search_dirs:
        for path in sorted(base_dir.glob("receiving_summary*.csv")):
            if not RAW_FILE_RE.fullmatch(path.name):
                continue
            discovered.setdefault(path.name, path)
    return sorted(discovered.values(), key=lambda path: extract_college_year_from_filename(path.name) or 0)


def merge_duplicate_named_columns(df, base_name):
    candidate_cols = [col for col in df.columns if col == base_name or col.startswith(f"{base_name}.")]
    if len(candidate_cols) <= 1:
        return df

    merged = None
    for col in candidate_cols:
        series = df[col]
        if merged is None:
            merged = series.copy()
            continue

        if pd.api.types.is_numeric_dtype(series):
            use_series = merged.isna()
        else:
            use_series = merged.isna() | merged.astype(str).str.strip().isin(["", "nan", "None"])
        merged = merged.where(~use_series, series)

    df[base_name] = merged
    for col in candidate_cols:
        if col != base_name:
            df.drop(columns=[col], inplace=True)
    return df


def load_model_frame(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    for base_name in ["matched_draft_year", "draft_match_method", "draft_match_score"]:
        df = merge_duplicate_named_columns(df, base_name)
    return df


def numeric_value(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return float(numerator) / float(denominator)


def first_matching_season_index(season_rows, predicate):
    for season in season_rows:
        if predicate(season):
            return season["season_index"]
    return np.nan


def peak_season_index(season_rows, key):
    valid_rows = [season for season in season_rows if pd.notna(season[key])]
    if not valid_rows:
        return np.nan
    return max(valid_rows, key=lambda season: season[key])["season_index"]


def build_breakout_features(wide_file):
    wide_df = pd.read_csv(wide_file, low_memory=False)
    season_indices = sorted(
        {
            int(col.split("_", 1)[0])
            for col in wide_df.columns
            if col.endswith("_Season") and col.split("_", 1)[0].isdigit()
        }
    )

    records = []
    for _, row in wide_df.iterrows():
        season_rows = []
        for season_index in season_indices:
            season_year = numeric_value(row.get(f"{season_index}_Season"))
            if np.isnan(season_year):
                continue

            season_rows.append(
                {
                    "season_index": season_index,
                    "season_year": int(season_year),
                    "yards": numeric_value(row.get(f"{season_index}_yards")),
                    "targets": numeric_value(row.get(f"{season_index}_targets")),
                    "touchdowns": numeric_value(row.get(f"{season_index}_touchdowns")),
                    "yprr": numeric_value(row.get(f"{season_index}_yprr")),
                    "routes": numeric_value(row.get(f"{season_index}_routes")),
                }
            )

        breakout_yards_index = first_matching_season_index(
            season_rows, lambda season: season["yards"] >= BREAKOUT_YARDS_THRESHOLD
        )
        breakout_elite_yards_index = first_matching_season_index(
            season_rows, lambda season: season["yards"] >= BREAKOUT_ELITE_YARDS_THRESHOLD
        )
        breakout_targets_index = first_matching_season_index(
            season_rows, lambda season: season["targets"] >= BREAKOUT_TARGETS_THRESHOLD
        )
        breakout_touchdowns_index = first_matching_season_index(
            season_rows, lambda season: season["touchdowns"] >= BREAKOUT_TOUCHDOWNS_THRESHOLD
        )
        breakout_yprr_index = first_matching_season_index(
            season_rows,
            lambda season: season["routes"] >= BREAKOUT_MIN_ROUTES and season["yprr"] >= BREAKOUT_YPRR_THRESHOLD,
        )

        breakout_candidates = [
            breakout_yards_index,
            breakout_targets_index,
            breakout_yprr_index,
        ]
        breakout_candidates = [value for value in breakout_candidates if pd.notna(value)]
        first_breakout_index = min(breakout_candidates) if breakout_candidates else np.nan

        records.append(
            {
                "player_id": row["player_id"],
                "college_seasons_played": len(season_rows),
                "breakout_700yd_season_index": breakout_yards_index,
                "breakout_900yd_season_index": breakout_elite_yards_index,
                "breakout_100target_season_index": breakout_targets_index,
                "breakout_8td_season_index": breakout_touchdowns_index,
                "breakout_2yprr_season_index": breakout_yprr_index,
                "first_breakout_season_index": first_breakout_index,
                "early_breakout_flag": int(pd.notna(first_breakout_index) and first_breakout_index <= 2),
                "peak_yards_season_index": peak_season_index(season_rows, "yards"),
                "peak_targets_season_index": peak_season_index(season_rows, "targets"),
                "peak_yprr_season_index": peak_season_index(season_rows, "yprr"),
            }
        )

    return pd.DataFrame(records)


def load_long_receiving_history(script_dir):
    files = find_raw_summary_files([paths.RECEIVING_RAW_DIR])
    if not files:
        raise FileNotFoundError("No raw receiving summary files were found.")

    frames = []
    base_columns = [
        "player_id",
        "player",
        "position",
        "team_name",
        "targets",
        "yards",
        "routes",
        "yprr",
    ]
    for file_path in files:
        season_year = extract_college_year_from_filename(file_path.name)
        df_part = pd.read_csv(file_path, low_memory=False)
        available_cols = [col for col in base_columns if col in df_part.columns]
        if "player_id" not in available_cols or "team_name" not in available_cols:
            continue

        df_part = df_part[available_cols].copy()
        df_part["Season"] = season_year
        frames.append(df_part)

    if not frames:
        raise ValueError("No usable receiving summary data was available for teammate-context features.")

    long_df = pd.concat(frames, ignore_index=True)
    long_df["Season"] = pd.to_numeric(long_df["Season"], errors="coerce")
    for col in ["targets", "yards", "routes", "yprr"]:
        if col in long_df.columns:
            long_df[col] = pd.to_numeric(long_df[col], errors="coerce")
    return long_df.dropna(subset=["player_id", "team_name", "Season"]).copy()


def summarize_teammate_group(group):
    group = group.copy()
    team_targets = group["targets"].fillna(0).sum()
    team_yards = group["yards"].fillna(0).sum()

    group["team_target_share"] = (
        group["targets"] / team_targets if team_targets > 0 else np.nan
    )
    group["team_yard_share"] = (
        group["yards"] / team_yards if team_yards > 0 else np.nan
    )
    group["team_target_rank"] = (
        group["targets"].fillna(-1).rank(method="min", ascending=False)
    )
    group["team_yard_rank"] = (
        group["yards"].fillna(-1).rank(method="min", ascending=False)
    )

    target_shares = group["team_target_share"].fillna(0).to_numpy()
    yard_shares = group["team_yard_share"].fillna(0).to_numpy()
    yards = group["yards"].fillna(0).to_numpy()

    teammate_top_target_share = []
    teammate_top_yard_share = []
    teammate_500yd_count = []
    teammate_800yd_count = []
    teammate_20pct_target_share_count = []

    for idx in range(len(group)):
        other_mask = np.ones(len(group), dtype=bool)
        other_mask[idx] = False

        teammate_top_target_share.append(float(target_shares[other_mask].max()) if other_mask.any() else 0.0)
        teammate_top_yard_share.append(float(yard_shares[other_mask].max()) if other_mask.any() else 0.0)
        teammate_500yd_count.append(int((yards[other_mask] >= STRONG_TEAMMATE_500_YARDS).sum()))
        teammate_800yd_count.append(int((yards[other_mask] >= STRONG_TEAMMATE_800_YARDS).sum()))
        teammate_20pct_target_share_count.append(
            int((target_shares[other_mask] >= DOMINANT_TEAMMATE_TARGET_SHARE_THRESHOLD).sum())
        )

    group["top_teammate_target_share"] = teammate_top_target_share
    group["top_teammate_yard_share"] = teammate_top_yard_share
    group["teammate_500yd_count"] = teammate_500yd_count
    group["teammate_800yd_count"] = teammate_800yd_count
    group["teammate_20pct_target_share_count"] = teammate_20pct_target_share_count
    group["target_share_gap_vs_top_teammate"] = (
        group["team_target_share"] - group["top_teammate_target_share"]
    )
    group["yard_share_gap_vs_top_teammate"] = (
        group["team_yard_share"] - group["top_teammate_yard_share"]
    )
    group["target_share_to_top_teammate_ratio"] = group.apply(
        lambda row: safe_ratio(row["team_target_share"], row["top_teammate_target_share"]),
        axis=1,
    )
    group["yard_share_to_top_teammate_ratio"] = group.apply(
        lambda row: safe_ratio(row["team_yard_share"], row["top_teammate_yard_share"]),
        axis=1,
    )
    group["top2_target_share"] = (
        group["team_target_share"] + group["top_teammate_target_share"]
    )
    group["top2_yard_share"] = (
        group["team_yard_share"] + group["top_teammate_yard_share"]
    )
    group["share_of_top2_targets"] = group.apply(
        lambda row: safe_ratio(row["team_target_share"], row["top2_target_share"]),
        axis=1,
    )
    group["share_of_top2_yards"] = group.apply(
        lambda row: safe_ratio(row["team_yard_share"], row["top2_yard_share"]),
        axis=1,
    )
    group["dominant_teammate_flag"] = (
        (group["top_teammate_target_share"] >= DOMINANT_TEAMMATE_TARGET_SHARE_THRESHOLD)
        | (group["top_teammate_yard_share"] >= DOMINANT_TEAMMATE_YARD_SHARE_THRESHOLD)
        | (group["teammate_800yd_count"] >= 1)
    ).astype(int)
    return group


def build_teammate_context_features(long_df):
    enriched = (
        long_df.groupby(["Season", "team_name"], group_keys=False)
        .apply(summarize_teammate_group)
        .reset_index(drop=True)
    )
    enriched.sort_values(["player_id", "Season"], inplace=True)

    records = []
    metric_cols = [
        "team_target_share",
        "team_yard_share",
        "top_teammate_target_share",
        "top_teammate_yard_share",
        "teammate_500yd_count",
        "teammate_800yd_count",
        "teammate_20pct_target_share_count",
        "team_target_rank",
        "team_yard_rank",
    ]

    for player_id, group in enriched.groupby("player_id"):
        group = group.sort_values("Season").copy()
        final_row = group.iloc[-1]
        dominant_group = group[group["dominant_teammate_flag"] == 1].copy()

        record = {
            "player_id": player_id,
            "career_avg_team_target_share": group["team_target_share"].mean(),
            "career_max_team_target_share": group["team_target_share"].max(),
            "career_avg_team_yard_share": group["team_yard_share"].mean(),
            "career_max_team_yard_share": group["team_yard_share"].max(),
            "career_avg_target_share_gap_vs_top_teammate": group["target_share_gap_vs_top_teammate"].mean(),
            "career_max_target_share_gap_vs_top_teammate": group["target_share_gap_vs_top_teammate"].max(),
            "career_avg_yard_share_gap_vs_top_teammate": group["yard_share_gap_vs_top_teammate"].mean(),
            "career_max_yard_share_gap_vs_top_teammate": group["yard_share_gap_vs_top_teammate"].max(),
            "career_avg_target_share_to_top_teammate_ratio": group["target_share_to_top_teammate_ratio"].mean(),
            "career_max_target_share_to_top_teammate_ratio": group["target_share_to_top_teammate_ratio"].max(),
            "career_avg_yard_share_to_top_teammate_ratio": group["yard_share_to_top_teammate_ratio"].mean(),
            "career_max_yard_share_to_top_teammate_ratio": group["yard_share_to_top_teammate_ratio"].max(),
            "career_avg_top_teammate_target_share": group["top_teammate_target_share"].mean(),
            "career_max_top_teammate_target_share": group["top_teammate_target_share"].max(),
            "career_avg_top_teammate_yard_share": group["top_teammate_yard_share"].mean(),
            "career_max_top_teammate_yard_share": group["top_teammate_yard_share"].max(),
            "career_avg_top2_target_share": group["top2_target_share"].mean(),
            "career_max_top2_target_share": group["top2_target_share"].max(),
            "career_avg_top2_yard_share": group["top2_yard_share"].mean(),
            "career_max_top2_yard_share": group["top2_yard_share"].max(),
            "career_avg_share_of_top2_targets": group["share_of_top2_targets"].mean(),
            "career_max_share_of_top2_targets": group["share_of_top2_targets"].max(),
            "career_avg_share_of_top2_yards": group["share_of_top2_yards"].mean(),
            "career_max_share_of_top2_yards": group["share_of_top2_yards"].max(),
            "career_max_teammate_500yd_count": group["teammate_500yd_count"].max(),
            "career_max_teammate_800yd_count": group["teammate_800yd_count"].max(),
            "career_max_teammate_20pct_target_share_count": group["teammate_20pct_target_share_count"].max(),
            "career_best_team_target_rank": group["team_target_rank"].min(),
            "career_best_team_yard_rank": group["team_yard_rank"].min(),
            "career_seasons_with_dominant_teammate": group["dominant_teammate_flag"].sum(),
            "final_team_target_share": final_row["team_target_share"],
            "final_team_yard_share": final_row["team_yard_share"],
            "final_target_share_gap_vs_top_teammate": final_row["target_share_gap_vs_top_teammate"],
            "final_yard_share_gap_vs_top_teammate": final_row["yard_share_gap_vs_top_teammate"],
            "final_target_share_to_top_teammate_ratio": final_row["target_share_to_top_teammate_ratio"],
            "final_yard_share_to_top_teammate_ratio": final_row["yard_share_to_top_teammate_ratio"],
            "final_top_teammate_target_share": final_row["top_teammate_target_share"],
            "final_top_teammate_yard_share": final_row["top_teammate_yard_share"],
            "final_top2_target_share": final_row["top2_target_share"],
            "final_top2_yard_share": final_row["top2_yard_share"],
            "final_share_of_top2_targets": final_row["share_of_top2_targets"],
            "final_share_of_top2_yards": final_row["share_of_top2_yards"],
            "final_teammate_500yd_count": final_row["teammate_500yd_count"],
            "final_teammate_800yd_count": final_row["teammate_800yd_count"],
            "final_teammate_20pct_target_share_count": final_row["teammate_20pct_target_share_count"],
            "final_team_target_rank": final_row["team_target_rank"],
            "final_team_yard_rank": final_row["team_yard_rank"],
            "final_season_with_dominant_teammate": final_row["dominant_teammate_flag"],
        }

        if dominant_group.empty:
            record.update(
                {
                    "career_avg_team_target_share_with_dominant_teammate": np.nan,
                    "career_max_team_target_share_with_dominant_teammate": np.nan,
                    "career_avg_team_yard_share_with_dominant_teammate": np.nan,
                    "career_max_team_yard_share_with_dominant_teammate": np.nan,
                    "career_avg_top_teammate_target_share_when_dominant": np.nan,
                    "career_avg_top_teammate_yard_share_when_dominant": np.nan,
                    "career_avg_target_share_to_top_teammate_ratio_when_dominant": np.nan,
                    "career_max_target_share_to_top_teammate_ratio_when_dominant": np.nan,
                    "career_avg_yard_share_to_top_teammate_ratio_when_dominant": np.nan,
                    "career_max_yard_share_to_top_teammate_ratio_when_dominant": np.nan,
                    "career_avg_target_share_gap_vs_top_teammate_when_dominant": np.nan,
                    "career_max_target_share_gap_vs_top_teammate_when_dominant": np.nan,
                    "career_avg_share_of_top2_targets_when_dominant": np.nan,
                    "career_max_share_of_top2_targets_when_dominant": np.nan,
                }
            )
        else:
            record.update(
                {
                    "career_avg_team_target_share_with_dominant_teammate": dominant_group["team_target_share"].mean(),
                    "career_max_team_target_share_with_dominant_teammate": dominant_group["team_target_share"].max(),
                    "career_avg_team_yard_share_with_dominant_teammate": dominant_group["team_yard_share"].mean(),
                    "career_max_team_yard_share_with_dominant_teammate": dominant_group["team_yard_share"].max(),
                    "career_avg_top_teammate_target_share_when_dominant": dominant_group["top_teammate_target_share"].mean(),
                    "career_avg_top_teammate_yard_share_when_dominant": dominant_group["top_teammate_yard_share"].mean(),
                    "career_avg_target_share_to_top_teammate_ratio_when_dominant": dominant_group["target_share_to_top_teammate_ratio"].mean(),
                    "career_max_target_share_to_top_teammate_ratio_when_dominant": dominant_group["target_share_to_top_teammate_ratio"].max(),
                    "career_avg_yard_share_to_top_teammate_ratio_when_dominant": dominant_group["yard_share_to_top_teammate_ratio"].mean(),
                    "career_max_yard_share_to_top_teammate_ratio_when_dominant": dominant_group["yard_share_to_top_teammate_ratio"].max(),
                    "career_avg_target_share_gap_vs_top_teammate_when_dominant": dominant_group["target_share_gap_vs_top_teammate"].mean(),
                    "career_max_target_share_gap_vs_top_teammate_when_dominant": dominant_group["target_share_gap_vs_top_teammate"].max(),
                    "career_avg_share_of_top2_targets_when_dominant": dominant_group["share_of_top2_targets"].mean(),
                    "career_max_share_of_top2_targets_when_dominant": dominant_group["share_of_top2_targets"].max(),
                }
            )

        records.append(record)

    return pd.DataFrame(records)


def load_fantasy_ppg_history(script_dir):
    frames = []
    for file_path in sorted(paths.FANTASY_WR_RAW_DIR.glob("FantasyPros_Fantasy_Football_Statistics_WR*.csv")):
        fantasy_year = extract_year_from_filename(file_path)
        if fantasy_year is None:
            continue

        df_part = pd.read_csv(file_path, low_memory=False)
        if {"Player", "G", "FPTS", "FPTS/G"} - set(df_part.columns):
            continue

        df_part["fantasy_year"] = fantasy_year
        df_part["player_clean"] = df_part["Player"].map(clean_player_name)
        df_part["games"] = pd.to_numeric(df_part["G"], errors="coerce")
        df_part["fpts"] = pd.to_numeric(df_part["FPTS"], errors="coerce")
        df_part["fpts_per_game"] = pd.to_numeric(df_part["FPTS/G"], errors="coerce")
        df_part["ppg_rank"] = np.nan

        eligible = df_part["games"] >= MIN_GAMES_FOR_PPG_RANK
        ranked = df_part.loc[eligible].sort_values(
            ["fpts_per_game", "fpts", "games"],
            ascending=[False, False, False],
        )
        df_part.loc[ranked.index, "ppg_rank"] = np.arange(1, len(ranked) + 1)
        df_part["top_n_ppg_hit"] = df_part["ppg_rank"] <= TARGET_TOP_N

        frames.append(
            df_part[
                [
                    "fantasy_year",
                    "player_clean",
                    "games",
                    "fpts_per_game",
                    "ppg_rank",
                    "top_n_ppg_hit",
                ]
            ]
        )

    if not frames:
        raise FileNotFoundError("No FantasyPros WR history files were found.")

    fantasy_df = pd.concat(frames, ignore_index=True)
    fantasy_df = fantasy_df.groupby(["player_clean", "fantasy_year"], as_index=False).agg(
        {
            "games": "max",
            "fpts_per_game": "max",
            "ppg_rank": "min",
            "top_n_ppg_hit": "max",
        }
    )
    return fantasy_df


def add_draft_year(df):
    df = df.copy()
    df["matched_draft_year"] = pd.to_numeric(df.get("matched_draft_year"), errors="coerce")
    df["final_season"] = pd.to_numeric(df.get("final_season"), errors="coerce")
    df["ROUND"] = pd.to_numeric(df.get("ROUND"), errors="coerce")
    df["SELECTION"] = pd.to_numeric(df.get("SELECTION"), errors="coerce")
    df["OVERALL_PICK"] = pd.to_numeric(df.get("OVERALL_PICK"), errors="coerce")
    df["draft_year"] = df["matched_draft_year"]

    fallback_mask = df["draft_year"].isna() & df["final_season"].notna() & df["ROUND"].notna()
    df.loc[fallback_mask, "draft_year"] = df.loc[fallback_mask, "final_season"] + 1
    return df


def add_draft_capital_features(df):
    df = df.copy()
    df["OVERALL_PICK"] = pd.to_numeric(df.get("OVERALL_PICK"), errors="coerce")
    df["ROUND"] = pd.to_numeric(df.get("ROUND"), errors="coerce")

    pick = df["OVERALL_PICK"]
    valid_pick = pick > 0

    df["draft_pick_inverse"] = np.where(valid_pick, 1.0 / pick, np.nan)
    df["draft_pick_sqrt_inverse"] = np.where(valid_pick, 1.0 / np.sqrt(pick), np.nan)
    df["draft_pick_log"] = np.where(valid_pick, np.log1p(pick), np.nan)
    df["draft_pick_bucket_ordinal"] = pd.cut(
        pick,
        bins=DRAFT_PICK_BUCKET_BINS,
        labels=False,
        include_lowest=True,
    )
    if "draft_pick_bucket_ordinal" in df.columns:
        df["draft_pick_bucket_ordinal"] = pd.to_numeric(df["draft_pick_bucket_ordinal"], errors="coerce")

    round_series = df["ROUND"]
    df["is_round1_pick"] = (round_series == 1).astype(int)
    df["is_day2_pick"] = round_series.isin([2, 3]).astype(int)
    df["is_top10_pick"] = valid_pick & (pick <= 10)
    df["is_top20_pick"] = valid_pick & (pick <= 20)
    df["is_top32_pick"] = valid_pick & (pick <= 32)
    df["is_top100_pick"] = valid_pick & (pick <= 100)

    for col in ["is_top10_pick", "is_top20_pick", "is_top32_pick", "is_top100_pick"]:
        df[col] = df[col].astype(int)

    return df


def add_binary_target(df, fantasy_df):
    df = add_draft_year(df)
    df["player_clean"] = df["player"].map(clean_player_name)

    latest_fantasy_year = int(fantasy_df["fantasy_year"].max())
    df["has_complete_target_window"] = df["draft_year"].notna() & (
        df["draft_year"] + TARGET_WINDOW_YEARS - 1 <= latest_fantasy_year
    )

    fantasy_lookup = fantasy_df.set_index(["player_clean", "fantasy_year"])[["ppg_rank", "top_n_ppg_hit"]]

    target_values = []
    best_ppg_ranks = []
    first_hit_years = []

    for _, row in df.iterrows():
        if pd.isna(row["draft_year"]):
            target_values.append(np.nan)
            best_ppg_ranks.append(np.nan)
            first_hit_years.append(np.nan)
            continue

        draft_year = int(row["draft_year"])
        years = range(draft_year, draft_year + TARGET_WINDOW_YEARS)
        ranks = []
        hit_year = np.nan

        for year in years:
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

        if not bool(row["has_complete_target_window"]):
            target_values.append(np.nan)
        else:
            target_values.append(int(pd.notna(hit_year)))

    df["best_ppg_rank_within_window"] = best_ppg_ranks
    df["first_top40_ppg_year"] = first_hit_years
    df["target_top40_ppg_within_3yrs"] = target_values
    return df, latest_fantasy_year


def merge_breakout_features(df, breakout_df):
    breakout_df = breakout_df.drop_duplicates(subset=["player_id"])
    return df.merge(breakout_df, on="player_id", how="left")


def build_feature_list(df):
    drop_cols = {
        "player",
        "player_id",
        "player_clean",
        "final_team_name",
        "career_Season_range",
        "final_position",
        "final_franchise_id",
        "career_fpts_per_game",
        "matched_draft_year",
        "draft_match_method",
        "draft_match_score",
        "is_historical_drafted_wr",
        "draft_year",
        "has_complete_target_window",
        "best_ppg_rank_within_window",
        "first_top40_ppg_year",
        "target_top40_ppg_within_3yrs",
        "ROUND",
        "SELECTION",
    }
    drop_cols.update(EXCLUDED_EXTRA_MEASURABLES)
    return [col for col in df.columns if col not in drop_cols]


def fit_draft_prior(df, target_col):
    prior_df = df[["OVERALL_PICK", target_col]].copy()
    prior_df = prior_df.dropna(subset=["OVERALL_PICK", target_col])
    if prior_df.empty:
        return {
            "global_rate": 0.0,
            "bucket_rates": {},
        }

    global_rate = float(prior_df[target_col].mean())
    bucket_codes = pd.cut(
        prior_df["OVERALL_PICK"],
        bins=DRAFT_PICK_BUCKET_BINS,
        labels=False,
        include_lowest=True,
    )
    prior_df["draft_pick_bucket_ordinal"] = pd.to_numeric(bucket_codes, errors="coerce")
    grouped = prior_df.dropna(subset=["draft_pick_bucket_ordinal"]).groupby("draft_pick_bucket_ordinal")[target_col]

    bucket_rates = {}
    for bucket, series in grouped:
        bucket = int(bucket)
        count = int(series.count())
        positives = float(series.sum())
        smoothed_rate = (positives + DRAFT_PRIOR_SMOOTHING * global_rate) / (count + DRAFT_PRIOR_SMOOTHING)
        bucket_rates[bucket] = float(smoothed_rate)

    return {
        "global_rate": global_rate,
        "bucket_rates": bucket_rates,
    }


def select_top_features_by_importance(train_df, target_col, feature_cols, limit):
    if limit is None or limit >= len(feature_cols):
        return list(feature_cols)

    X_train, _ = prepare_feature_matrix(train_df, feature_cols, fit_imputer=True)
    model = build_classifier()
    model.fit(X_train, train_df[target_col].values)

    feat_imp = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values(["importance", "feature"], ascending=[False, True])
    return feat_imp.head(limit)["feature"].tolist()


def score_draft_prior(df, prior_state):
    bucket_codes = pd.cut(
        pd.to_numeric(df["OVERALL_PICK"], errors="coerce"),
        bins=DRAFT_PICK_BUCKET_BINS,
        labels=False,
        include_lowest=True,
    )
    bucket_codes = pd.to_numeric(bucket_codes, errors="coerce")
    global_rate = float(prior_state["global_rate"])
    bucket_rates = prior_state["bucket_rates"]

    return bucket_codes.map(lambda bucket: bucket_rates.get(int(bucket), global_rate) if pd.notna(bucket) else global_rate).astype(float).to_numpy()


def blend_model_with_draft_prior(model_prob, draft_prior_prob, blend_weight=DRAFT_PRIOR_BLEND_WEIGHT):
    model_prob = np.asarray(model_prob, dtype=float)
    draft_prior_prob = np.asarray(draft_prior_prob, dtype=float)
    return (1.0 - blend_weight) * model_prob + blend_weight * draft_prior_prob


def prepare_feature_matrix(df, feature_cols, imputer=None, fit_imputer=False):
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    if fit_imputer:
        mean_fill_values = {}
        for col in HEIGHT_WEIGHT_FEATURES:
            if col in X.columns:
                mean_value = X[col].mean(skipna=True)
                mean_fill_values[col] = 0 if pd.isna(mean_value) else float(mean_value)

        X_filled = X.copy()
        for col, mean_value in mean_fill_values.items():
            X_filled[col] = X_filled[col].fillna(mean_value)

        base_imputer = SimpleImputer(strategy="constant", fill_value=0)
        X_array = base_imputer.fit_transform(X_filled)
        imputer = {
            "base_imputer": base_imputer,
            "mean_fill_values": mean_fill_values,
        }
        return X_array, imputer

    X_filled = X.copy()
    for col, mean_value in imputer["mean_fill_values"].items():
        if col in X_filled.columns:
            X_filled[col] = X_filled[col].fillna(mean_value)

    X_array = imputer["base_imputer"].transform(X_filled)
    return X_array, imputer


def build_classifier():
    return RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
        n_jobs=1,
    )


def safe_metric(metric_fn, y_true, y_prob):
    unique = pd.Series(y_true).nunique(dropna=True)
    if unique < 2:
        return np.nan
    return metric_fn(y_true, y_prob)


def print_metrics(label, y_true, y_prob):
    y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)
    roc_auc = safe_metric(roc_auc_score, y_true, y_prob)
    average_precision = safe_metric(average_precision_score, y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)

    print(f"\n{label}")
    print(f"  Samples: {len(y_true)}")
    print(f"  Positive rate: {np.mean(y_true):.4f}")
    if pd.notna(roc_auc):
        print(f"  ROC-AUC: {roc_auc:.4f}")
    if pd.notna(average_precision):
        print(f"  Average precision: {average_precision:.4f}")
    print(f"  Accuracy @ {PREDICTION_THRESHOLD:.2f}: {accuracy:.4f}")
    print(f"  Brier score: {brier:.4f}")


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
    train_df = load_model_frame(TRAIN_FILE)
    holdout_df = load_model_frame(HOLDOUT_FILE)
    breakout_df = build_breakout_features(WIDE_FILE)
    long_receiving_df = load_long_receiving_history(SCRIPT_DIR)
    teammate_df = build_teammate_context_features(long_receiving_df)
    fantasy_df = load_fantasy_ppg_history(SCRIPT_DIR)

    train_df = merge_breakout_features(train_df, breakout_df)
    holdout_df = merge_breakout_features(holdout_df, breakout_df)
    train_df = merge_breakout_features(train_df, teammate_df)
    holdout_df = merge_breakout_features(holdout_df, teammate_df)

    train_df, latest_fantasy_year = add_binary_target(train_df, fantasy_df)
    holdout_df, _ = add_binary_target(holdout_df, fantasy_df)
    train_df = add_draft_capital_features(train_df)
    holdout_df = add_draft_capital_features(holdout_df)

    complete_train_df = train_df[
        (train_df["final_position"] == "WR")
        & train_df["ROUND"].notna()
        & train_df["target_top40_ppg_within_3yrs"].notna()
    ].copy()
    complete_train_df["target_top40_ppg_within_3yrs"] = complete_train_df[
        "target_top40_ppg_within_3yrs"
    ].astype(int)

    print(f"Latest fantasy season available: {latest_fantasy_year}")
    print(f"Training rows with complete 3-year target windows: {complete_train_df.shape[0]}")
    print(
        "Positive class count: "
        f"{int(complete_train_df['target_top40_ppg_within_3yrs'].sum())} / {complete_train_df.shape[0]}"
    )

    feature_cols = build_feature_list(complete_train_df)

    train_split_df, val_split_df = train_test_split(
        complete_train_df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=complete_train_df["target_top40_ppg_within_3yrs"],
    )
    X_train_split, split_imputer = prepare_feature_matrix(
        train_split_df, feature_cols, fit_imputer=True
    )
    X_val_split, _ = prepare_feature_matrix(
        val_split_df, feature_cols, imputer=split_imputer, fit_imputer=False
    )

    split_model = build_classifier()
    split_model.fit(X_train_split, train_split_df["target_top40_ppg_within_3yrs"].values)
    split_val_prob = split_model.predict_proba(X_val_split)[:, 1]
    split_prior_state = fit_draft_prior(train_split_df, "target_top40_ppg_within_3yrs")
    split_val_draft_prior = score_draft_prior(val_split_df, split_prior_state)
    split_val_blended_prob = blend_model_with_draft_prior(split_val_prob, split_val_draft_prior)
    print_metrics(
        "Random split validation (model only)",
        val_split_df["target_top40_ppg_within_3yrs"].values,
        split_val_prob,
    )
    print_metrics(
        f"Random split validation (draft-blended, weight={DRAFT_PRIOR_BLEND_WEIGHT:.2f})",
        val_split_df["target_top40_ppg_within_3yrs"].values,
        split_val_blended_prob,
    )

    latest_complete_draft_year = latest_fantasy_year - TARGET_WINDOW_YEARS + 1
    backtest_df = complete_train_df[complete_train_df["draft_year"] == latest_complete_draft_year].copy()
    earlier_df = complete_train_df[complete_train_df["draft_year"] < latest_complete_draft_year].copy()

    if not backtest_df.empty and not earlier_df.empty:
        X_backtest_train, backtest_imputer = prepare_feature_matrix(
            earlier_df, feature_cols, fit_imputer=True
        )
        X_backtest_holdout, _ = prepare_feature_matrix(
            backtest_df, feature_cols, imputer=backtest_imputer, fit_imputer=False
        )

        backtest_model = build_classifier()
        backtest_model.fit(X_backtest_train, earlier_df["target_top40_ppg_within_3yrs"].values)
        backtest_prob = backtest_model.predict_proba(X_backtest_holdout)[:, 1]
        backtest_prior_state = fit_draft_prior(earlier_df, "target_top40_ppg_within_3yrs")
        backtest_draft_prior = score_draft_prior(backtest_df, backtest_prior_state)
        backtest_blended_prob = blend_model_with_draft_prior(backtest_prob, backtest_draft_prior)
        print_metrics(
            f"Draft-year backtest ({latest_complete_draft_year}, model only)",
            backtest_df["target_top40_ppg_within_3yrs"].values,
            backtest_prob,
        )
        print_metrics(
            f"Draft-year backtest ({latest_complete_draft_year}, draft-blended)",
            backtest_df["target_top40_ppg_within_3yrs"].values,
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
                "target_top40_ppg_within_3yrs",
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
        backtest_output["model_prob_top40_ppg_within_3yrs"] = backtest_prob
        backtest_output["draft_prior_prob"] = backtest_draft_prior
        backtest_output["prob_top40_ppg_within_3yrs"] = backtest_blended_prob
        backtest_output["draft_blend_weight"] = DRAFT_PRIOR_BLEND_WEIGHT
        backtest_output["predicted_top40_ppg_within_3yrs"] = (
            backtest_output["prob_top40_ppg_within_3yrs"] >= PREDICTION_THRESHOLD
        ).astype(int)
        backtest_output.sort_values("prob_top40_ppg_within_3yrs", ascending=False, inplace=True)
        backtest_output.to_csv(BACKTEST_FILE, index=False)
        print(f"Saved backtest predictions to '{BACKTEST_FILE.name}'.")

    X_full, final_imputer = prepare_feature_matrix(complete_train_df, feature_cols, fit_imputer=True)
    final_model = build_classifier()
    final_model.fit(X_full, complete_train_df["target_top40_ppg_within_3yrs"].values)
    final_prior_state = fit_draft_prior(complete_train_df, "target_top40_ppg_within_3yrs")
    save_feature_importances(final_model, feature_cols)

    model_artifact = {
        "model": final_model,
        "imputer": final_imputer,
        "draft_prior_state": final_prior_state,
        "feature_names": feature_cols,
        "config": {
            "target_top_n": TARGET_TOP_N,
            "target_window_years": TARGET_WINDOW_YEARS,
            "min_games_for_ppg_rank": MIN_GAMES_FOR_PPG_RANK,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "draft_prior_blend_weight": DRAFT_PRIOR_BLEND_WEIGHT,
            "holdout_year": HOLDOUT_YEAR,
            "latest_fantasy_year": latest_fantasy_year,
        },
    }
    joblib.dump(model_artifact, MODEL_FILE)
    print(f"Saved binary model artifact to '{MODEL_FILE.name}'.")

    prospective_df = holdout_df[
        (holdout_df["final_position"] == "WR")
        & holdout_df["ROUND"].notna()
    ].copy()

    X_holdout, _ = prepare_feature_matrix(
        prospective_df, feature_cols, imputer=final_imputer, fit_imputer=False
    )
    prospective_prob = final_model.predict_proba(X_holdout)[:, 1]
    prospective_draft_prior = score_draft_prior(prospective_df, final_prior_state)
    prospective_blended_prob = blend_model_with_draft_prior(prospective_prob, prospective_draft_prior)

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
    prospective_output["model_prob_top40_ppg_within_3yrs"] = prospective_prob
    prospective_output["draft_prior_prob"] = prospective_draft_prior
    prospective_output["prob_top40_ppg_within_3yrs"] = prospective_blended_prob
    prospective_output["draft_blend_weight"] = DRAFT_PRIOR_BLEND_WEIGHT
    prospective_output["predicted_top40_ppg_within_3yrs"] = (
        prospective_output["prob_top40_ppg_within_3yrs"] >= PREDICTION_THRESHOLD
    ).astype(int)
    prospective_output.sort_values("prob_top40_ppg_within_3yrs", ascending=False, inplace=True)
    prospective_output.to_csv(HOLDOUT_PREDICTIONS_FILE, index=False)

    print(f"\nSaved prospective predictions to '{HOLDOUT_PREDICTIONS_FILE.name}'.")
    print("\nTop projected drafted WRs by probability:")
    print(prospective_output.head(TOP_N_PRINT))


if __name__ == "__main__":
    main()
