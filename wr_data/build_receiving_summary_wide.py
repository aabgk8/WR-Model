from pathlib import Path
import pandas as pd
import re

import paths

RAW_FILE_RE = re.compile(r"receiving_summary(\d{2})\.csv$", re.IGNORECASE)

def extract_year_from_filename(filename):
    """
    Given a filename like 'receiving_summary14.csv', extract '14' -> 2014.
    """
    base = Path(filename).name
    match = RAW_FILE_RE.fullmatch(base)
    if not match:
        return None
    return 2000 + int(match.group(1))

def find_raw_summary_files(search_dirs):
    """
    Find only season-level receiving summary files and ignore derived wide files.
    If duplicate basenames exist, keep the first one found.
    """
    discovered = {}
    for base_dir in search_dirs:
        for path in sorted(base_dir.glob("receiving_summary*.csv")):
            if not RAW_FILE_RE.fullmatch(path.name):
                continue
            discovered.setdefault(path.name, path)
    return sorted(discovered.values(), key=lambda p: extract_year_from_filename(p.name) or 0)

def pivot_seasons_for_player_id(group):
    """
    Pivot multiple rows (seasons) for a single player_id into one wide row,
    enumerating each row with numeric prefixes: '1_', '2_', etc.,
    sorted by 'Season' if available.
    """
    group = group.copy()
    if "Season" in group.columns:
        group["Season_int"] = pd.to_numeric(group["Season"], errors="coerce")
        group.sort_values(by="Season_int", inplace=True)

    out_dict = {"player_id": group["player_id"].iloc[0]}
    if "player" in group.columns:
        out_dict["player"] = group["player"].iloc[0]

    for i, (_, row) in enumerate(group.iterrows(), start=1):
        for col in group.columns:
            if col in ["player_id", "player", "Season_int"]:
                continue
            out_dict[f"{i}_{col}"] = row[col]

    return pd.DataFrame([out_dict])

def main():
    paths.ensure_directories()

    files = find_raw_summary_files([paths.RECEIVING_RAW_DIR])
    if not files:
        print("No season-level receiving summary files found.")
        return

    df_list = []
    years = []
    for file_path in files:
        df = pd.read_csv(file_path)
        year4 = extract_year_from_filename(file_path.name)
        df["Season"] = year4
        df_list.append(df)
        years.append(year4)

    df_all = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df_all)} rows from {len(files)} files spanning {min(years)}-{max(years)}.")

    if "player_id" not in df_all.columns:
        print("No 'player_id' column found in the data. Please adjust script.")
        return

    pivoted_rows = []
    for _, group in df_all.groupby("player_id"):
        pivoted_rows.append(pivot_seasons_for_player_id(group))

    final_df = pd.concat(pivoted_rows, ignore_index=True)

    cols = final_df.columns.tolist()
    if "player_id" in cols:
        cols.remove("player_id")
        cols.insert(0, "player_id")
    if "player" in cols:
        cols.remove("player")
        cols.insert(1, "player")
    final_df = final_df[cols]

    year_range_output = (
        paths.PROCESSED_DIR / f"receiving_summary_{min(years)}_{max(years)}_wide_by_id.csv"
    )
    stable_output = paths.RECEIVING_SUMMARY_WIDE_FILE
    final_df.to_csv(year_range_output, index=False)
    final_df.to_csv(stable_output, index=False)

    print(f"Saved wide file '{year_range_output.name}' (grouped by 'player_id').")
    print(f"Saved stable wide file alias '{stable_output.name}'.")

if __name__ == "__main__":
    main()
