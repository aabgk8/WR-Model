from pathlib import Path
import pandas as pd

import paths

def get_final_value(row, base_col, max_seasons=20):
    """
    Return the last non-NaN value among 1_<base_col>, 2_<base_col>, ..., up to max_seasons_<base_col>.
    """
    val_final = None
    for season_num in range(1, max_seasons + 1):
        coln = f"{season_num}_{base_col}"
        if coln in row.index and pd.notna(row[coln]):
            val_final = row[coln]
    return val_final

def main():
    paths.ensure_directories()
    input_file = paths.TRANSFORMED_FEATURES_FILE
    output_file = paths.FINAL_FEATURES_FILE
    max_seasons = 20

    df = pd.read_csv(input_file)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns from '{input_file.name}'.")

    df["final_position"] = df.apply(lambda row: get_final_value(row, "position", max_seasons), axis=1)
    df["final_franchise_id"] = df.apply(lambda row: get_final_value(row, "franchise_id", max_seasons), axis=1)

    drop_cols = []
    for base_col in ["position", "franchise_id"]:
        for season_num in range(1, max_seasons + 1):
            prefix_col = f"{season_num}_{base_col}"
            if prefix_col in df.columns:
                drop_cols.append(prefix_col)

    drop_cols = list(set(drop_cols))
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    cols = df.columns.tolist()
    if "player_id" in cols and "final_position" in cols and "final_franchise_id" in cols:
        cols.remove("player_id")
        cols.remove("final_position")
        cols.remove("final_franchise_id")
        new_order = ["player_id", "final_position", "final_franchise_id"] + cols
        df = df[new_order]

    df.to_csv(output_file, index=False)
    print(f"Saved final file '{output_file.name}' with shape={df.shape}.")

if __name__ == "__main__":
    main()
