from pathlib import Path
import pandas as pd

######################################
# CONFIG: HOW TO HANDLE EACH BASE COL
######################################
AGGREGATOR_CONFIG = {
    "player_id": ["keep"],
    "player": ["keep"],
    "position": "filter_position_wr_te",
    "team_name": "final",
    "player_game_count": ["sum", "avg"],
    "avg_depth_of_target": ["avg", "max"],
    "avoided_tackles": ["avg", "max"],
    "caught_percent": ["avg", "max"],
    "contested_catch_rate": ["avg", "max"],
    "contested_receptions": ["sum", "avg"],
    "contested_targets": ["sum", "avg"],
    "declined_penalties": "drop",
    "drop_rate": ["avg", "max"],
    "drops": ["sum", "avg"],
    "first_downs": ["sum", "max"],
    "franchise_id": ["keep"],
    "fumbles": "sum",
    "grades_hands_drop": ["avg", "max"],
    "grades_hands_fumble": ["avg", "max"],
    "grades_offense": ["avg", "max"],
    "grades_pass_block": ["avg", "max"],
    "grades_pass_route": ["avg", "max"],
    "inline_rate": ["avg", "max"],
    "inline_snaps": ["sum", "max"],
    "interceptions": "drop",
    "longest": "max",
    "pass_block_rate": ["avg", "max"],
    "pass_blocks": "sum",
    "pass_plays": "sum",
    "penalties": "drop",
    "receptions": ["sum", "avg"],
    "route_rate": ["avg", "max"],
    "routes": ["sum", "avg"],
    "slot_rate": ["avg", "max"],
    "slot_snaps": "sum",
    "targeted_qb_rating": ["avg", "max"],
    "targets": ["sum", "avg"],
    "touchdowns": ["sum", "avg"],
    "wide_rate": ["avg", "max"],
    "wide_snaps": ["sum", "avg"],
    "yards": ["sum", "avg"],
    "yards_after_catch": ["sum", "avg"],
    "yards_after_catch_per_reception": ["avg", "max"],
    "yards_per_reception": ["avg", "max"],
    "yprr": ["avg", "max"],
    "Season": "career_range",
}

ALLOWED_POSITIONS = {"WR"}

def has_allowed_position_any_season(row, allowed_positions=None, max_seasons=20):
    """
    Check columns like '1_position', '2_position', etc.
    If ANY season matches an allowed position, return True.
    """
    allowed_positions = allowed_positions or ALLOWED_POSITIONS
    for season_num in range(1, max_seasons + 1):
        pos_col = f"{season_num}_position"
        if pos_col in row.index:
            val = str(row[pos_col]).upper().strip()
            if val in allowed_positions:
                return True
    return False

def main():
    script_dir = Path(__file__).resolve().parent
    input_file = script_dir / "receiving_summary_wide_by_id.csv"
    if not input_file.exists():
        legacy_candidates = sorted(script_dir.glob("receiving_summary_*_wide_by_id.csv"))
        if not legacy_candidates:
            print("No wide receiving summary file found. Run untitled61.py first.")
            return
        input_file = legacy_candidates[-1]

    output_file = script_dir / "wr_te_transformed.csv"
    max_seasons = 20

    df = pd.read_csv(input_file)
    print(f"Loaded shape: {df.shape}")

    keep_mask = df.apply(lambda row: has_allowed_position_any_season(row, ALLOWED_POSITIONS, max_seasons), axis=1)
    df_filtered = df[keep_mask].copy()
    print(f"After allowed-position filter {sorted(ALLOWED_POSITIONS)}: {df_filtered.shape} rows remain.")

    df_agg = pd.DataFrame(index=df_filtered.index)
    drop_cols = []

    def get_career_season_range(row):
        years = []
        for season_num in range(1, max_seasons + 1):
            col = f"{season_num}_Season"
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    years.append(int(val))
        if not years:
            return ""
        if len(years) == 1:
            return str(years[0])
        return f"{min(years)}-{max(years)}"

    if AGGREGATOR_CONFIG.get("Season") == "career_range":
        df_agg["career_Season_range"] = df_filtered.apply(get_career_season_range, axis=1)
        for season_num in range(1, max_seasons + 1):
            pcol = f"{season_num}_Season"
            if pcol in df_filtered.columns:
                drop_cols.append(pcol)

    for base_col, instructions_raw in AGGREGATOR_CONFIG.items():
        instructions = [instructions_raw] if isinstance(instructions_raw, str) else instructions_raw

        if "filter_position_wr_te" in instructions or base_col == "Season":
            continue
        if "drop" in instructions or "remove" in instructions:
            for season_num in range(1, max_seasons + 1):
                col = f"{season_num}_{base_col}"
                if col in df_filtered.columns:
                    drop_cols.append(col)
            continue
        if "keep" in instructions:
            continue
        if "final" in instructions:
            newcol = f"final_{base_col}"

            def get_final_value(row):
                val_final = None
                for season_num in range(1, max_seasons + 1):
                    coln = f"{season_num}_{base_col}"
                    if coln in row.index:
                        this_val = row[coln]
                        if pd.notna(this_val):
                            val_final = this_val
                return val_final

            df_agg[newcol] = df_filtered.apply(get_final_value, axis=1)
            for season_num in range(1, max_seasons + 1):
                col = f"{season_num}_{base_col}"
                if col in df_filtered.columns:
                    drop_cols.append(col)
            continue

        def gather_prefix_values(row, base):
            vals = []
            for season_num in range(1, max_seasons + 1):
                coln = f"{season_num}_{base}"
                if coln in row.index:
                    value = row[coln]
                    if pd.notna(value):
                        vals.append(value)
            return vals

        def aggregator(row):
            vals = gather_prefix_values(row, base_col)
            if not vals:
                return {}
            outd = {}
            total = sum(vals)
            count = len(vals)
            if "sum" in instructions:
                outd[f"career_sum_{base_col}"] = total
            if "avg" in instructions:
                outd[f"career_avg_{base_col}"] = total / count
            if "max" in instructions:
                outd[f"career_max_{base_col}"] = max(vals)
            return outd

        tmp_series = df_filtered.apply(aggregator, axis=1)
        tmp_df = pd.DataFrame(tmp_series.tolist(), index=df_filtered.index)
        df_agg = pd.concat([df_agg, tmp_df], axis=1)

        for season_num in range(1, max_seasons + 1):
            col = f"{season_num}_{base_col}"
            if col in df_filtered.columns:
                drop_cols.append(col)

    drop_cols = list(set(drop_cols))
    df_filtered.drop(columns=[c for c in drop_cols if c in df_filtered.columns], inplace=True)

    df_final = pd.concat([df_filtered, df_agg], axis=1)
    df_final.to_csv(output_file, index=False)
    print(f"Final file saved: '{output_file.name}', shape={df_final.shape}")

if __name__ == "__main__":
    main()
