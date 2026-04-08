from pathlib import Path

import pandas as pd

import RF2_binary_top40 as binary_model
import paths


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_TEMPLATE_FILE = paths.TEMPLATES_DIR / "wr_binary_top40_train_bio_collection_template.csv"
HOLDOUT_TEMPLATE_FILE = paths.TEMPLATES_DIR / "wr_binary_top40_holdout_2025_bio_collection_template.csv"
ALL_TEMPLATE_FILE = paths.TEMPLATES_DIR / "wr_binary_top40_all_bio_collection_template.csv"


def add_blank_bio_columns(df):
    df = df.copy()
    blank_columns = [
        "BIRTHDATE",
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
        "SOURCE",
        "NOTES",
    ]
    for col in blank_columns:
        if col not in df.columns:
            df[col] = ""
    return df


def prepare_train_population():
    train_df = binary_model.load_model_frame(binary_model.TRAIN_FILE)
    fantasy_df = binary_model.load_fantasy_ppg_history(binary_model.SCRIPT_DIR)
    train_df, latest_fantasy_year = binary_model.add_binary_target(train_df, fantasy_df)

    train_df = train_df[
        (train_df["final_position"] == "WR")
        & train_df["OVERALL_PICK"].notna()
        & train_df["target_top40_ppg_within_3yrs"].notna()
    ].copy()
    train_df["target_top40_ppg_within_3yrs"] = train_df["target_top40_ppg_within_3yrs"].astype(int)
    train_df["dataset_split"] = "train"
    train_df["latest_fantasy_year"] = latest_fantasy_year
    return train_df


def prepare_holdout_population():
    holdout_df = binary_model.load_model_frame(binary_model.HOLDOUT_FILE)
    holdout_df, _ = binary_model.add_binary_target(
        holdout_df, binary_model.load_fantasy_ppg_history(binary_model.SCRIPT_DIR)
    )
    holdout_df = holdout_df[
        (holdout_df["final_position"] == "WR")
        & holdout_df["OVERALL_PICK"].notna()
    ].copy()
    holdout_df["dataset_split"] = "holdout"
    return holdout_df


def select_output_columns(df):
    preferred_cols = [
        "dataset_split",
        "player_id",
        "player",
        "final_team_name",
        "final_season",
        "draft_year",
        "OVERALL_PICK",
        "ROUND",
        "SELECTION",
        "target_top40_ppg_within_3yrs",
        "BIRTHDATE",
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
        "SOURCE",
        "NOTES",
    ]
    existing_cols = [col for col in preferred_cols if col in df.columns]
    return df[existing_cols].copy()


def main():
    paths.ensure_directories()
    train_df = prepare_train_population()
    holdout_df = prepare_holdout_population()

    train_out = select_output_columns(add_blank_bio_columns(train_df))
    holdout_out = select_output_columns(add_blank_bio_columns(holdout_df))
    all_out = pd.concat([train_out, holdout_out], ignore_index=True)

    train_out.sort_values(["draft_year", "OVERALL_PICK", "player"], inplace=True)
    holdout_out.sort_values(["OVERALL_PICK", "player"], inplace=True)
    all_out.sort_values(["dataset_split", "draft_year", "OVERALL_PICK", "player"], inplace=True)

    train_out.to_csv(TRAIN_TEMPLATE_FILE, index=False)
    holdout_out.to_csv(HOLDOUT_TEMPLATE_FILE, index=False)
    all_out.to_csv(ALL_TEMPLATE_FILE, index=False)

    print(f"Training template rows: {len(train_out)}")
    print(f"Holdout template rows: {len(holdout_out)}")
    print(f"Combined template rows: {len(all_out)}")
    print(f"Wrote '{TRAIN_TEMPLATE_FILE.name}'")
    print(f"Wrote '{HOLDOUT_TEMPLATE_FILE.name}'")
    print(f"Wrote '{ALL_TEMPLATE_FILE.name}'")


if __name__ == "__main__":
    main()
