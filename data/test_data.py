from data.data import load_encounters

def test_dataset_shape_and_labels():
    df = load_encounters("data/encounters.csv")
    assert len(df) == 20
    assert df["has_error"].isna().sum() == 0
    assert set(df["has_error"]) <= {0, 1}


def test_label_matches_error_type():
    df = load_encounters("data/encounters.csv")
    assert ((df["error_type"] == "none") == (df["has_error"] == 0)).all()