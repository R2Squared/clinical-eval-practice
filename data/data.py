import pandas as pd

REQUIRED_COLUMNS = ["encounter_id", "patient_id", "specialty",
                    "transcript", "ai_note", "error_type", "has_error"]


def load_encounters(path: str) -> pd.DataFrame:
    """Load the encounters CSV and check that every required column is present."""
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df