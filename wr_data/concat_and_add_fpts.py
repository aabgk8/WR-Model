from pathlib import Path
import glob
import pandas as pd
import re

import paths

def extract_year_from_filename(fn):
    """
    Capture the year from a filename like 'FantasyPros_Fantasy_Football_Statistics_WR2015.csv'.
    """
    match = re.search(r"(\d{4})\.csv$", Path(fn).name)
    if match:
        return int(match.group(1))
    return None

def clean_player_name(name):
    """
    Remove any '(...)' substring from the player name and strip whitespace.
    """
    if not isinstance(name, str):
        return name
    return re.sub(r"\(.*?\)", "", name).strip()

def parse_final_season(rng_str):
    """
    Parse the final year from career_Season_range.
    """
    if not isinstance(rng_str, str):
        return None
    rng_str = rng_str.strip()
    if not rng_str:
        return None
    parts = rng_str.split("-")
    try:
        return int(parts[-1])
    except ValueError:
        return None

def merge_measurables(df):
    """
    Merge measurable / draft-capital columns by player_id from the current source file.
    """
    candidate_files = [paths.MEASURABLES_FILE]
    target_cols = [
        "player_id",
        "AGE_ON_DRAFT_DAY",
        "HEIGHT",
        "WEIGHT",
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
        "ROUND",
        "SELECTION",
    ]

    for file_path in candidate_files:
        if not file_path.exists():
            continue

        df_meas = pd.read_csv(file_path)
        available_cols = [col for col in target_cols if col in df_meas.columns]
        if "player_id" not in available_cols:
            continue

        df_meas = df_meas[available_cols].drop_duplicates(subset=["player_id"])
        df = df.merge(df_meas, on="player_id", how="left", suffixes=("", "_meas"))

        for col in [
            "AGE_ON_DRAFT_DAY",
            "HEIGHT",
            "WEIGHT",
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
            "ROUND",
            "SELECTION",
        ]:
            meas_col = f"{col}_meas"
            if meas_col not in df.columns:
                continue
            if col in df.columns:
                df[col] = df[col].fillna(df[meas_col])
            else:
                df[col] = df[meas_col]
            df.drop(columns=[meas_col], inplace=True)

        return df

    return df

def main():
    paths.ensure_directories()

    pattern_wr = str(paths.FANTASY_WR_RAW_DIR / "FantasyPros_Fantasy_Football_Statistics_WR*.csv")
    pattern_te = str(paths.FANTASY_WR_RAW_DIR / "FantasyPros_Fantasy_Football_Statistics_TE*.csv")
    files_wr = glob.glob(pattern_wr)
    files_te = glob.glob(pattern_te)

    all_files = files_wr + files_te
    if not all_files:
        print("No WR/TE fantasy CSV files found. Check your patterns.")
        return

    df_list = []
    fantasy_years = []
    for file_path in all_files:
        df_part = pd.read_csv(file_path)
        year = extract_year_from_filename(file_path)
        if year:
            df_part["Year"] = year
            fantasy_years.append(year)
        if "Player" in df_part.columns:
            df_part["Player"] = df_part["Player"].apply(clean_player_name)
        df_list.append(df_part)

    df_all = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {df_all.shape[0]} rows from {len(all_files)} files.")

    if "Player" not in df_all.columns or "FPTS" not in df_all.columns or "G" not in df_all.columns:
        print("Error: The data must have columns 'Player', 'FPTS', 'G'. Adjust script accordingly.")
        return

    grouped = df_all.groupby("Player", dropna=True)
    fantasy_agg = grouped.agg({"FPTS": "sum", "G": "sum"}).reset_index()
    fantasy_agg["career_fpts_per_game"] = fantasy_agg.apply(
        lambda row: row["FPTS"] / row["G"] if row["G"] != 0 else 0,
        axis=1,
    )

    wr_te_file = paths.FINAL_FEATURES_FILE
    df_wr_te = pd.read_csv(wr_te_file)

    if "player" in df_wr_te.columns:
        df_wr_te["player"] = df_wr_te["player"].apply(clean_player_name)
    else:
        print("Warning: 'player' column not found in wr_te_final.csv, adapt script.")

    merged_df = pd.merge(
        df_wr_te,
        fantasy_agg[["Player", "career_fpts_per_game"]],
        left_on="player",
        right_on="Player",
        how="left",
    )

    if "Player" in merged_df.columns:
        merged_df.drop(columns="Player", inplace=True)

    merged_df["final_season"] = merged_df["career_Season_range"].apply(parse_final_season)
    max_fantasy_year = max(fantasy_years)

    merged_df = merge_measurables(merged_df)

    # Fill missing targets with 0 only for players whose first NFL season should already
    # exist in the available fantasy data.
    zero_fill_mask = (
        merged_df["career_fpts_per_game"].isna()
        & merged_df["final_season"].notna()
        & ((merged_df["final_season"] + 1) <= max_fantasy_year)
    )
    merged_df.loc[zero_fill_mask, "career_fpts_per_game"] = 0

    out_file = paths.FINAL_WITH_FANTASY_FILE
    merged_df.to_csv(out_file, index=False)
    print(f"Merged fantasy data saved to '{out_file.name}' with shape={merged_df.shape}")

if __name__ == "__main__":
    main()
