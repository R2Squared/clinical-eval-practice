from data import load_encounters
from checks import any_flag
from metrics import confusion_counts, sensitivity, ppv


def main() -> None:
    df = load_encounters("data/encounters.csv")
    df["rule_flag"] = [int(any_flag(t, n)) for t, n in zip(df["transcript"], df["ai_note"])]

    y_true = list(df["has_error"])
    y_pred = list(df["rule_flag"])
    print(confusion_counts(y_true, y_pred))
    print(f"Sensitivity: {sensitivity(y_true, y_pred):.2f}")
    print(f"PPV: {ppv(y_true, y_pred):.2f}")

    print("\nCaught per error type:")
    print(df.groupby("error_type")["rule_flag"].agg(["sum", "count"]))

    cols = ["encounter_id", "error_type"]
    print("\nFalse positives:")
    print(df[(df["has_error"] == 0) & (df["rule_flag"] == 1)][cols])
    print("\nFalse negatives:")
    print(df[(df["has_error"] == 1) & (df["rule_flag"] == 0)][cols])


if __name__ == "__main__":
    main()