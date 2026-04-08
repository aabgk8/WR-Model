from pathlib import Path


WR_DATA_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WR_DATA_ROOT.parent

DATA_DIR = WR_DATA_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PROCESSED_DIR = DATA_DIR / "processed"

RECEIVING_RAW_DIR = RAW_DIR / "receiving"
FANTASY_WR_RAW_DIR = RAW_DIR / "fantasypros_wr"
COMBINE_RAW_DIR = RAW_DIR / "combine"

OUTPUTS_DIR = WR_DATA_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
BACKTESTS_DIR = OUTPUTS_DIR / "backtests"
FEATURE_IMPORTANCE_DIR = OUTPUTS_DIR / "feature_importance"
SWEEPS_DIR = OUTPUTS_DIR / "sweeps"
REPORTS_DIR = OUTPUTS_DIR / "reports"
TEMPLATES_DIR = OUTPUTS_DIR / "templates"

DOCS_DIR = WR_DATA_ROOT / "docs"
ARCHIVE_DIR = WR_DATA_ROOT / "archive"

RECEIVING_SUMMARY_WIDE_FILE = PROCESSED_DIR / "receiving_summary_wide_by_id.csv"
TRANSFORMED_FEATURES_FILE = PROCESSED_DIR / "wr_te_transformed.csv"
FINAL_FEATURES_FILE = PROCESSED_DIR / "wr_te_final.csv"
FINAL_WITH_FANTASY_FILE = PROCESSED_DIR / "wr_te_final_with_fantasy.csv"
MEASURABLES_FILE = REFERENCE_DIR / "my_players_with_measurables.csv"
DRAFT_ORDERS_FILE = REFERENCE_DIR / "2025-2014_receiver_draft_orders.csv"
PIPELINE_README_FILE = WR_DATA_ROOT / "README.md"
MODEL_RESULTS_SUMMARY_FILE = DOCS_DIR / "MODEL_RESULTS_SUMMARY.md"


def processed_train_file(holdout_year):
    return PROCESSED_DIR / f"wr_te_final_no_{holdout_year}.csv"


def processed_holdout_file(holdout_year):
    return PROCESSED_DIR / f"wr_te_holdout_{holdout_year}.csv"


def ensure_directories():
    for directory in [
        DATA_DIR,
        RAW_DIR,
        REFERENCE_DIR,
        PROCESSED_DIR,
        RECEIVING_RAW_DIR,
        FANTASY_WR_RAW_DIR,
        COMBINE_RAW_DIR,
        OUTPUTS_DIR,
        MODELS_DIR,
        PREDICTIONS_DIR,
        BACKTESTS_DIR,
        FEATURE_IMPORTANCE_DIR,
        SWEEPS_DIR,
        REPORTS_DIR,
        TEMPLATES_DIR,
        DOCS_DIR,
        ARCHIVE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
