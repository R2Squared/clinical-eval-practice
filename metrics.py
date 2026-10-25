def confusion_counts(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    """Count true/false positives/negatives from paired labels and flags.

    Raises ValueError if the inputs differ in length or contain anything
    other than 0 or 1 (including missing values).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} items, y_pred has {len(y_pred)}"
        )
    for name, labels in (("y_true", y_true), ("y_pred", y_pred)):
        for i, value in enumerate(labels):
            if value not in (0, 1):
                raise ValueError(f"Invalid value {value!r} at index {i} in {name}; expected 0 or 1")
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
        raise ValueError("No positive cases in y_true: sensitivity is undefined")
    return c["tp"] / positives


def ppv(y_true: list[int], y_pred: list[int]) -> float:
    """Proportion of flags that were true errors.

    Raises ValueError if y_pred contains no flags, since PPV is undefined
    in that case.
    """
    c = confusion_counts(y_true, y_pred)
    flagged = c["tp"] + c["fp"]
    if flagged == 0:
        raise ValueError("No flagged cases in y_pred: ppv is undefined")
    return c["tp"] / flagged