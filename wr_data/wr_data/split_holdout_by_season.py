from pathlib import Path
import pandas as pd

HOLDOUT_YEAR = 2025

def get_final_season(rng_str):
    """
    Given a string like "2020-2024" or "2022" in 'career_Season_range',
    return the final year as an integer.
    """
    if not isinstance(rng_str, str) or not rng_str.strip():
        return None
    if "-" not in rng_str:
        try:
            return int(rng_str)
        except ValueError:
            return None
    parts = rng_str.split("-")
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None

def main():
    script_dir = Path(__file__).resolve().parent
    input_file = script_dir / "wr_te_final_with_fantasy.csv"
    df = pd.read_csv(input_file)
    print(f"Loaded {df.shape[0]} rows from '{input_file.name}'.")

    if "career_Season_range" not in df.columns:
        print("Error: 'career_Season_range' not found. Adjust script or aggregator logic.")
        return

    if "final_season" not in df.columns:
        df["final_season"] = df["career_Season_range"].apply(get_final_season)

    df_holdout = df[df["final_season"] == HOLDOUT_YEAR].copy()
    print(f"Players with final season = {HOLDOUT_YEAR}: {df_holdout.shape[0]}")

    holdout_file = script_dir / f"wr_te_holdout_{HOLDOUT_YEAR}.csv"
    df_holdout.to_csv(holdout_file, index=False)
    print(f"Wrote holdout file '{holdout_file.name}'.")

    df_train = df[df["final_season"] != HOLDOUT_YEAR].copy()
    train_file = script_dir / f"wr_te_final_no_{HOLDOUT_YEAR}.csv"
    df_train.to_csv(train_file, index=False)
    print(f"Wrote training file '{train_file.name}' with shape={df_train.shape}.")

if __name__ == "__main__":
    main()
