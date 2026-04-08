from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd
import re

import paths

NICKNAMES = {
    "nathaniel": "tank",
    "gabriel": "gabe",
    "scotty": "scott",
    "josh": "joshua",
}

COLLEGE_ALIASES = {
    "louisiana state": "lsu",
    "southern california": "usc",
    "southern methodist": "smu",
    "central florida": "ucf",
    "texas christian": "tcu",
    "brigham young": "byu",
    "mississippi": "ole miss",
    "bowling green": "bowl green",
    "south carolina": "s carolina",
    "north carolina": "n carolina",
    "western kentucky": "w kentucky",
    "washington state": "wash state",
    "oregon state": "oregon st",
    "new mexico state": "new mex st",
    "georgia southern": "ga southrn",
}

def clean_tokens(text):
    if pd.isna(text):
        return []
    text = str(text).lower().replace("'", "").replace(".", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [token for token in text.split() if token]

def norm_name(text, drop_suffix=False, nickname_map=False, compact=False):
    tokens = clean_tokens(text)
    if drop_suffix:
        suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
        tokens = [token for token in tokens if token not in suffixes]
    if nickname_map:
        tokens = [NICKNAMES.get(token, token) for token in tokens]
    return "".join(tokens) if compact else " ".join(tokens)

def norm_school(text):
    school = " ".join(clean_tokens(text))
    return COLLEGE_ALIASES.get(school, school)

def score_candidates(draft_row, candidates):
    name_scores = []
    school_scores = []
    for _, row in candidates.iterrows():
        ratios = [
            SequenceMatcher(None, draft_row["name_nosuffix"], row["name_nosuffix"]).ratio(),
            SequenceMatcher(None, draft_row["name_compact_nosuffix"], row["name_compact_nosuffix"]).ratio(),
            SequenceMatcher(None, draft_row["name_nickname"], row["name_nickname"]).ratio(),
        ]
        name_scores.append(max(ratios))
        school_scores.append(
            SequenceMatcher(None, draft_row["school_norm"], row["school_norm"]).ratio()
            if draft_row["school_norm"] else 0
        )

    scored = candidates.copy()
    scored["name_score"] = name_scores
    scored["school_score"] = school_scores
    scored["combined_score"] = scored["name_score"] * 0.85 + scored["school_score"] * 0.15
    return scored.sort_values(["combined_score", "name_score", "school_score"], ascending=False)

def build_matches(features_df, draft_df):
    matches = []

    for _, draft_row in draft_df.iterrows():
        candidates = features_df[
            (features_df["final_season"] >= draft_row["draft_year"] - 3)
            & (features_df["final_season"] <= draft_row["draft_year"])
        ].copy()

        matched_row = None
        match_method = None
        match_score = None

        for col, label in [
            ("name_norm", "exact"),
            ("name_nosuffix", "nosuffix"),
            ("name_compact", "compact"),
            ("name_compact_nosuffix", "compact_nosuffix"),
            ("name_nickname", "nickname"),
        ]:
            possible = candidates[candidates[col] == draft_row[col]]
            if len(possible) == 1:
                matched_row = possible.iloc[0]
                match_method = label
                match_score = 1.0
                break
            if len(possible) > 1 and draft_row["school_norm"]:
                school_possible = possible[possible["school_norm"] == draft_row["school_norm"]]
                if len(school_possible) == 1:
                    matched_row = school_possible.iloc[0]
                    match_method = f"{label}_school"
                    match_score = 1.0
                    break

        if matched_row is None and not candidates.empty:
            scored = score_candidates(draft_row, candidates)
            top = scored.iloc[0]
            second_score = scored.iloc[1]["combined_score"] if len(scored) > 1 else 0
            gap = float(top["combined_score"] - second_score)

            if (top["combined_score"] >= 0.91) or (
                top["combined_score"] >= 0.86 and top["school_score"] >= 0.95 and gap >= 0.03
            ):
                matched_row = top
                match_method = "fuzzy"
                match_score = float(top["combined_score"])

        matches.append({
            "draft_year": draft_row["draft_year"],
            "draft_name": draft_row["draft_name"],
            "draft_college": draft_row["draft_college"],
            "ROUND": draft_row["ROUND"],
            "SELECTION": draft_row["SELECTION"],
            "OVERALL_PICK": draft_row["OVERALL_PICK"],
            "matched_player_id": None if matched_row is None else matched_row["player_id"],
            "matched_player": None if matched_row is None else matched_row["player"],
            "matched_final_season": None if matched_row is None else matched_row["final_season"],
            "matched_college": None if matched_row is None else matched_row["final_team_name"],
            "match_method": match_method,
            "match_score": match_score,
        })

    return pd.DataFrame(matches)

def main():
    paths.ensure_directories()
    features_file = paths.FINAL_WITH_FANTASY_FILE
    draft_file = paths.DRAFT_ORDERS_FILE

    features_df = pd.read_csv(features_file)
    features_df = features_df[features_df["final_position"] == "WR"].copy()

    draft_df = pd.read_csv(draft_file)
    draft_df = draft_df.rename(columns={
        "Year": "draft_year",
        "Name": "draft_name",
        "College": "draft_college",
        "Round": "ROUND",
        "Pick": "SELECTION",
        "Player": "OVERALL_PICK",
    })

    for frame, name_col, school_col in [
        (features_df, "player", "final_team_name"),
        (draft_df, "draft_name", "draft_college"),
    ]:
        frame["name_norm"] = frame[name_col].map(norm_name)
        frame["name_nosuffix"] = frame[name_col].map(lambda x: norm_name(x, drop_suffix=True))
        frame["name_compact"] = frame[name_col].map(lambda x: norm_name(x, compact=True))
        frame["name_compact_nosuffix"] = frame[name_col].map(
            lambda x: norm_name(x, drop_suffix=True, compact=True)
        )
        frame["name_nickname"] = frame[name_col].map(
            lambda x: norm_name(x, drop_suffix=True, nickname_map=True)
        )
        frame["school_norm"] = frame[school_col].map(norm_school)

    match_report = build_matches(features_df, draft_df)
    match_report_file = paths.REPORTS_DIR / "draft_order_match_report.csv"
    match_report.to_csv(match_report_file, index=False)

    unmatched_file = paths.REPORTS_DIR / "draft_order_unmatched.csv"
    match_report[match_report["matched_player_id"].isna()].to_csv(unmatched_file, index=False)

    matched = match_report[match_report["matched_player_id"].notna()].copy()
    matched["matched_player_id"] = matched["matched_player_id"].astype(int)

    merged_df = pd.read_csv(features_file)
    existing_merge_cols = [
        "ROUND",
        "SELECTION",
        "OVERALL_PICK",
        "matched_draft_year",
        "draft_match_method",
        "draft_match_score",
        "is_historical_drafted_wr",
    ]
    duplicate_merge_cols = [col for col in merged_df.columns if col.endswith(".1")]
    cols_to_drop = [col for col in existing_merge_cols + duplicate_merge_cols if col in merged_df.columns]
    if cols_to_drop:
        merged_df.drop(columns=cols_to_drop, inplace=True)

    merged_df = merged_df.merge(
        matched[
            [
                "matched_player_id",
                "ROUND",
                "SELECTION",
                "OVERALL_PICK",
                "draft_year",
                "match_method",
                "match_score",
            ]
        ],
        left_on="player_id",
        right_on="matched_player_id",
        how="left",
    )
    merged_df.drop(columns=["matched_player_id"], inplace=True)
    merged_df.rename(columns={
        "draft_year": "matched_draft_year",
        "match_method": "draft_match_method",
        "match_score": "draft_match_score",
    }, inplace=True)
    merged_df["is_historical_drafted_wr"] = merged_df["ROUND"].notna()

    out_file = paths.FINAL_WITH_FANTASY_FILE
    merged_df.to_csv(out_file, index=False)

    print(f"Match report saved to '{match_report_file.name}'.")
    print(f"Unmatched report saved to '{unmatched_file.name}'.")
    print(f"Matched {matched.shape[0]} of {draft_df.shape[0]} draft rows.")
    print(f"Updated '{out_file.name}' with ROUND/SELECTION for matched drafted WRs.")

if __name__ == "__main__":
    main()
