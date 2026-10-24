def confusion_counts(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    """Count true/false positives/negatives from paired labels and flags."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            counts["tp"] += 1
        elif truth == 0 and pred == 1:
            counts["fp"] += 1
        elif truth == 1 and pred == 0:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def sensitivity(y_true: list[int], y_pred: list[int]) -> float:
    """Proportion of true errors that were flagged.

    Raises ValueError if y_true contains no true errors, since sensitivity
    is undefined in that case.
    """
    c = confusion_counts(y_true, y_pred)
    positives = c["tp"] + c["fn"]
    if positives == 0:
        raise ValueError("sensitivity is undefined: no true positive cases in y_true")
    return c["tp"] / positives


def ppv(y_true: list[int], y_pred: list[int]) -> float:
    """Proportion of flags that were true errors.

    Raises ValueError if y_pred contains no flags, since PPV is undefined
    in that case.
    """
    c = confusion_counts(y_true, y_pred)
    flagged = c["tp"] + c["fp"]
    if flagged == 0:
        raise ValueError("ppv is undefined: no positive predictions in y_pred")
    return c["tp"] / flagged