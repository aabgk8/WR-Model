from pathlib import Path
from difflib import SequenceMatcher
import math
import re

import pandas as pd

import paths

RAW_2014_COLUMNS = [
    "Name",
    "Pos",
    "School",
    "SourceText",
    "Height",
    "Weight",
    "40 Yd",
    "Vertical",
    "Bench",
    "Broad",
    "3-Cone",
    "Shuttle",
    "DraftInfo",
]

NICKNAMES = {
    "josh": "joshua",
    "gabe": "gabriel",
    "scotty": "scott",
}

FULL_NAME_CANONICAL = {
    "kc concepcion": "kevin concepcion",
    "kevin concepcion": "kevin concepcion",
    "nathaniel dell": "tank dell",
    "tank dell": "tank dell",
    "mike woods": "michael woods",
    "michael woods": "michael woods",
    "michael woods ii": "michael woods",
    "bisi johnson": "olabisi johnson",
    "olabisi johnson": "olabisi johnson",
}

COLLEGE_ALIASES = {
    "southern cal": "usc",
    "usc": "usc",
    "southern california": "usc",
    "ohio st": "ohio state",
    "ohio state": "ohio state",
    "arizona st": "arizona state",
    "arizona state": "arizona state",
    "mississippi": "ole miss",
    "ole miss": "ole miss",
    "uconn": "uconn",
    "connecticut": "uconn",
    "ga state": "georgia state",
    "georgia state": "georgia state",
    "miss state": "mississippi state",
    "mississippi st": "mississippi state",
    "mississippi state": "mississippi state",
    "notre dame": "notre dame",
    "texas a m": "texas a&m",
    "texas a&m": "texas a&m",
    "texas am": "texas a&m",
    "north dakota st": "north dakota state",
    "north dakota state": "north dakota state",
    "louisiana state": "lsu",
    "lsu": "lsu",
    "texas christian": "tcu",
    "tcu": "tcu",
    "miami fl": "miami",
    "miami (fl)": "miami",
    "miami": "miami",
    "southern methodist": "smu",
    "smu": "smu",
    "unlv": "unlv",
    "washington st": "washington state",
    "washington state": "washington state",
    "florida intl": "fiu",
    "florida international": "fiu",
    "central michigan": "central michigan",
    "northern illinois": "northern illinois",
    "west virginia": "west virginia",
    "pittsburg state": "pittsburg state",
    "boston college": "boston college",
    "cincinnati": "cincinnati",
    "kentucky": "kentucky",
    "baylor": "baylor",
    "indiana": "indiana",
    "alabama": "alabama",
    "georgia": "georgia",
    "clemson": "clemson",
    "tennessee": "tennessee",
    "louisville": "louisville",
    "washington": "washington",
    "florida": "florida",
    "arkansas": "arkansas",
    "texas": "texas",
    "virginia tech": "virginia tech",
    "minnesota": "minnesota",
    "stanford": "stanford",
    "nebraska": "nebraska",
    "auburn": "auburn",
    "duke": "duke",
    "maryland": "maryland",
    "michigan": "michigan",
}

SCRIPT_DIR = Path(__file__).resolve().parent
MEASURABLES_FILE = paths.MEASURABLES_FILE
MATCH_REPORT_FILE = paths.REPORTS_DIR / "combine_measurable_match_report.csv"
UNMATCHED_TARGET_FILE = paths.REPORTS_DIR / "combine_measurable_unmatched_targets.csv"
UNUSED_SOURCE_FILE = paths.REPORTS_DIR / "combine_measurable_unused_source_rows.csv"


def extract_year_from_filename(path):
    name = Path(path).name
    if name == "2014_combine_results.csv":
        return 2014
    if name == "2015_combine_results.csv":
        return 2015
    match = re.search(r"combine-results-(\d{4})-WR\.csv$", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def clean_tokens(text):
    if pd.isna(text):
        return []
    text = str(text).lower().replace("&", " and ")
    text = text.replace("'", "").replace(".", " ").replace("-", " ")
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


def canonical_name(text):
    normalized = norm_name(text, drop_suffix=True)
    return FULL_NAME_CANONICAL.get(normalized, normalized)


def norm_school(text):
    cleaned = " ".join(clean_tokens(text))
    return COLLEGE_ALIASES.get(cleaned, cleaned)


def is_blank(value):
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none"}


def fill_blank_series(base_series, fill_series):
    result = base_series.copy()
    mask = result.apply(is_blank)
    result.loc[mask] = fill_series.loc[mask]
    return result


def parse_fractional_number(value):
    if pd.isna(value):
        return None
    text = str(value).strip().replace('"', "")
    if not text:
        return None
    text = text.replace("–", "-")
    try:
        return float(text)
    except ValueError:
        pass

    if " " in text:
        whole, frac = text.split(" ", 1)
        try:
            whole_val = float(whole)
        except ValueError:
            return None
        if "/" in frac:
            numerator, denominator = frac.split("/", 1)
            try:
                return whole_val + float(numerator) / float(denominator)
            except ValueError:
                return None
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            return float(numerator) / float(denominator)
        except ValueError:
            return None
    return None


def parse_height_2014(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None

    month_map = {"may": 5, "jun": 6}
    match = re.match(r"^(\d+)-([A-Za-z]{3})$", text)
    if match:
        inches = int(match.group(1))
        feet = month_map.get(match.group(2).lower())
        if feet is not None:
            return feet * 12 + inches

    match = re.match(r"^([A-Za-z]{3})-(\d+)$", text)
    if match:
        feet = month_map.get(match.group(1).lower())
        inches = int(match.group(2))
        if feet is not None:
            return feet * 12 + inches

    return parse_height_general(text)


def parse_height_general(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace('"', "").replace("”", "").replace("′", "'").replace("″", "")
    text = text.replace(" ", " ").strip()

    match = re.match(r"^(\d+)'(\d+(?: \d+/\d+)?)$", text)
    if match:
        feet = int(match.group(1))
        inches = parse_fractional_number(match.group(2))
        if inches is not None:
            return feet * 12 + inches

    match = re.match(r"^(\d+)-(\d+(?: \d+/\d+)?)$", text)
    if match:
        feet = int(match.group(1))
        inches = parse_fractional_number(match.group(2))
        if inches is not None:
            return feet * 12 + inches

    return parse_fractional_number(text)


def parse_broad(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace('"', "").replace("”", "").replace("′", "'").replace("″", "")
    match = re.match(r"^(\d+)'(\d+)?$", text)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2)) if match.group(2) else 0
        return feet * 12 + inches
    return parse_fractional_number(text)


def to_numeric(value):
    number = parse_fractional_number(value)
    if number is None:
        return None
    if isinstance(number, float) and math.isnan(number):
        return None
    return number


def build_2014_frame(path):
    df = pd.read_csv(path, header=None)
    df.columns = RAW_2014_COLUMNS
    df["draft_year"] = 2014
    out = pd.DataFrame(index=df.index)
    out["draft_year"] = df["draft_year"]
    out["combine_name"] = df["Name"]
    out["combine_school"] = df["School"]
    out["HEIGHT"] = df["Height"].apply(parse_height_2014)
    out["WEIGHT"] = pd.to_numeric(df["Weight"], errors="coerce")
    out["HAND_SIZE"] = pd.Series([None] * len(df))
    out["ARM_LENGTH"] = pd.Series([None] * len(df))
    out["WINGSPAN"] = pd.Series([None] * len(df))
    out["FORTY_TIME"] = pd.to_numeric(df["40 Yd"], errors="coerce")
    out["TEN_SPLIT"] = pd.Series([None] * len(df))
    out["TWENTY_SPLIT"] = pd.Series([None] * len(df))
    out["VERTICAL"] = pd.to_numeric(df["Vertical"], errors="coerce")
    out["BROAD"] = pd.to_numeric(df["Broad"], errors="coerce")
    out["SHUTTLE"] = pd.to_numeric(df["Shuttle"], errors="coerce")
    out["THREE_CONE"] = pd.to_numeric(df["3-Cone"], errors="coerce")
    out["BENCH"] = pd.to_numeric(df["Bench"], errors="coerce")
    out["SPEED"] = pd.Series([None] * len(df))
    out["SOURCE"] = path.name
    out["NOTES"] = "2014 source normalized from Excel-style height dates"
    return out


def build_2015_frame(path):
    df = pd.read_csv(path)
    out = pd.DataFrame(index=df.index)
    out["draft_year"] = 2015
    out["combine_name"] = df["Player"]
    out["combine_school"] = df["School"]
    out["HEIGHT"] = df["Height"].apply(parse_height_general)
    out["WEIGHT"] = pd.to_numeric(df["Weight"], errors="coerce")
    out["HAND_SIZE"] = df["Hand"].apply(to_numeric)
    out["ARM_LENGTH"] = df["Arm"].apply(to_numeric)
    out["WINGSPAN"] = pd.Series([None] * len(df))
    out["FORTY_TIME"] = pd.to_numeric(df["40"], errors="coerce")
    out["TEN_SPLIT"] = pd.to_numeric(df["10 Yard"], errors="coerce")
    out["TWENTY_SPLIT"] = pd.Series([None] * len(df))
    out["VERTICAL"] = df["Vertical"].apply(to_numeric)
    out["BROAD"] = df["Broad"].apply(parse_broad)
    out["SHUTTLE"] = pd.to_numeric(df["Short Shuttle"], errors="coerce")
    out["THREE_CONE"] = pd.to_numeric(df["3-Cone"], errors="coerce")
    out["BENCH"] = pd.to_numeric(df["Bench"], errors="coerce")
    out["SPEED"] = pd.Series([None] * len(df))
    out["SOURCE"] = path.name
    out["NOTES"] = ""
    return out


def build_modern_frame(path):
    df = pd.read_csv(path)
    draft_year = extract_year_from_filename(path)
    out = pd.DataFrame(index=df.index)
    out["draft_year"] = draft_year
    out["combine_name"] = df["Name"]
    out["combine_school"] = df["School"]
    out["HEIGHT"] = df["Height"].apply(parse_height_general)
    out["WEIGHT"] = pd.to_numeric(df["Weight"], errors="coerce")
    out["HAND_SIZE"] = df["Hands"].apply(to_numeric)
    out["ARM_LENGTH"] = df["Arms"].apply(to_numeric)
    out["WINGSPAN"] = df["Wingspan"].apply(to_numeric)
    out["FORTY_TIME"] = pd.to_numeric(df["40 Yd"], errors="coerce")
    out["TEN_SPLIT"] = pd.to_numeric(df["10 Split"], errors="coerce")
    out["TWENTY_SPLIT"] = pd.Series([None] * len(df))
    out["VERTICAL"] = df["Vertical"].apply(to_numeric)
    out["BROAD"] = df["Broad"].apply(parse_broad)
    out["SHUTTLE"] = pd.to_numeric(df["Shuttle"], errors="coerce")
    out["THREE_CONE"] = pd.to_numeric(df["3-Cone"], errors="coerce")
    out["BENCH"] = pd.to_numeric(df["Bench"], errors="coerce")
    out["SPEED"] = pd.Series([None] * len(df))
    out["SOURCE"] = path.name
    out["NOTES"] = ""
    return out


def load_combine_sources():
    files = sorted(paths.COMBINE_RAW_DIR.glob("*.csv"))
    frames = []
    for path in files:
        if path.name == "2014_combine_results.csv":
            frames.append(build_2014_frame(path))
        elif path.name == "2015_combine_results.csv":
            frames.append(build_2015_frame(path))
        elif "combine-results-" in path.name:
            frames.append(build_modern_frame(path))

    if not frames:
        raise FileNotFoundError("No combine result files were found.")

    combined = pd.concat(frames, ignore_index=True)
    combined["draft_year"] = pd.to_numeric(combined["draft_year"], errors="coerce").astype("Int64")

    for frame, name_col, school_col in [
        (combined, "combine_name", "combine_school"),
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
        frame["name_canonical"] = frame[name_col].map(canonical_name)
        frame["school_norm"] = frame[school_col].map(norm_school)

    return combined


def score_candidates(target_row, candidates):
    name_scores = []
    school_scores = []
    for _, row in candidates.iterrows():
        ratios = [
            SequenceMatcher(None, target_row["name_norm"], row["name_norm"]).ratio(),
            SequenceMatcher(None, target_row["name_nosuffix"], row["name_nosuffix"]).ratio(),
            SequenceMatcher(None, target_row["name_compact_nosuffix"], row["name_compact_nosuffix"]).ratio(),
            SequenceMatcher(None, target_row["name_nickname"], row["name_nickname"]).ratio(),
        ]
        name_scores.append(max(ratios))
        school_scores.append(
            SequenceMatcher(None, target_row["school_norm"], row["school_norm"]).ratio()
            if target_row["school_norm"] else 0
        )

    scored = candidates.copy()
    scored["name_score"] = name_scores
    scored["school_score"] = school_scores
    scored["combined_score"] = scored["name_score"] * 0.88 + scored["school_score"] * 0.12
    return scored.sort_values(["combined_score", "name_score", "school_score"], ascending=False)


def build_matches(target_df, source_df):
    matches = []
    used_source_indices = set()

    for idx, target_row in target_df.iterrows():
        draft_year = target_row["draft_year"]
        candidates = source_df[source_df["draft_year"] == draft_year].copy()

        matched_row = None
        match_method = None
        match_score = None

        for col, label in [
            ("name_norm", "exact"),
            ("name_nosuffix", "nosuffix"),
            ("name_compact", "compact"),
            ("name_compact_nosuffix", "compact_nosuffix"),
            ("name_nickname", "nickname"),
            ("name_canonical", "canonical"),
        ]:
            possible = candidates[candidates[col] == target_row[col]]
            if len(possible) == 1:
                matched_row = possible.iloc[0]
                match_method = label
                match_score = 1.0
                break
            if len(possible) > 1 and target_row["school_norm"]:
                school_possible = possible[possible["school_norm"] == target_row["school_norm"]]
                if len(school_possible) == 1:
                    matched_row = school_possible.iloc[0]
                    match_method = f"{label}_school"
                    match_score = 1.0
                    break

        if matched_row is None and not candidates.empty:
            scored = score_candidates(target_row, candidates)
            top = scored.iloc[0]
            second_score = scored.iloc[1]["combined_score"] if len(scored) > 1 else 0
            gap = float(top["combined_score"] - second_score)
            if (top["combined_score"] >= 0.91) or (
                top["combined_score"] >= 0.86 and top["school_score"] >= 0.95 and gap >= 0.03
            ):
                matched_row = top
                match_method = "fuzzy"
                match_score = float(top["combined_score"])

        if matched_row is not None:
            used_source_indices.add(int(matched_row.name))

        matches.append(
            {
                "target_index": idx,
                "player_id": target_row["player_id"],
                "player": target_row["player"],
                "final_team_name": target_row["final_team_name"],
                "draft_year": draft_year,
                "matched_source_index": None if matched_row is None else int(matched_row.name),
                "matched_name": None if matched_row is None else matched_row["combine_name"],
                "matched_school": None if matched_row is None else matched_row["combine_school"],
                "match_method": match_method,
                "match_score": match_score,
            }
        )

    match_report = pd.DataFrame(matches)
    unused_sources = source_df[~source_df.index.isin(used_source_indices)].copy()
    return match_report, unused_sources


def main():
    paths.ensure_directories()
    if not MEASURABLES_FILE.exists():
        raise FileNotFoundError(f"Could not find '{MEASURABLES_FILE.name}'.")

    target_df = pd.read_csv(MEASURABLES_FILE, low_memory=False)
    target_df["target_index"] = target_df.index
    source_df = load_combine_sources()

    target_df["draft_year"] = pd.to_numeric(target_df["draft_year"], errors="coerce").astype("Int64")

    for frame, name_col, school_col in [
        (target_df, "player", "final_team_name"),
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
        frame["name_canonical"] = frame[name_col].map(canonical_name)
        frame["school_norm"] = frame[school_col].map(norm_school)

    match_report, unused_sources = build_matches(target_df, source_df)
    match_report.to_csv(MATCH_REPORT_FILE, index=False)

    matched = match_report[match_report["matched_source_index"].notna()].copy()
    unmatched_targets = match_report[match_report["matched_source_index"].isna()].copy()
    unmatched_targets.to_csv(UNMATCHED_TARGET_FILE, index=False)

    if not unused_sources.empty:
        report_cols = [
            "draft_year",
            "combine_name",
            "combine_school",
            "HEIGHT",
            "WEIGHT",
            "FORTY_TIME",
            "SOURCE",
        ]
        unused_sources[report_cols].to_csv(UNUSED_SOURCE_FILE, index=False)
    else:
        pd.DataFrame(columns=["draft_year", "combine_name", "combine_school"]).to_csv(
            UNUSED_SOURCE_FILE, index=False
        )

    updated_df = target_df.copy()
    merge_cols = [
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

    if not matched.empty:
        matched["matched_source_index"] = matched["matched_source_index"].astype(int)
        source_subset = source_df.loc[matched["matched_source_index"], merge_cols].copy()
        source_subset["target_index"] = matched["target_index"].values
        updated_df = updated_df.merge(source_subset, on="target_index", how="left", suffixes=("", "_combine"))

        for col in merge_cols:
            combine_col = f"{col}_combine"
            if combine_col not in updated_df.columns:
                continue
            if col not in updated_df.columns:
                updated_df[col] = updated_df[combine_col]
            else:
                updated_df[col] = fill_blank_series(updated_df[col], updated_df[combine_col])
            updated_df.drop(columns=[combine_col], inplace=True)

    helper_cols = [
        "target_index",
        "name_norm",
        "name_nosuffix",
        "name_compact",
        "name_compact_nosuffix",
        "name_nickname",
        "name_canonical",
        "school_norm",
    ]
    updated_df.drop(columns=[col for col in helper_cols if col in updated_df.columns], inplace=True)
    updated_df.to_csv(MEASURABLES_FILE, index=False)

    print(f"Loaded {len(target_df)} target rows from '{MEASURABLES_FILE.name}'.")
    print(f"Loaded {len(source_df)} combine rows across 2014-2026 WR files.")
    print(f"Matched {len(matched)} target rows.")
    print(f"Unmatched target rows: {len(unmatched_targets)}")
    print(f"Unused combine rows: {len(unused_sources)}")
    print(f"Updated '{MEASURABLES_FILE.name}'.")
    print(f"Saved match report to '{MATCH_REPORT_FILE.name}'.")
    print(f"Saved unmatched targets to '{UNMATCHED_TARGET_FILE.name}'.")
    print(f"Saved unused source rows to '{UNUSED_SOURCE_FILE.name}'.")


if __name__ == "__main__":
    main()
