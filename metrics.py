from math import sqrt
from statistics import NormalDist
from sklearn.metrics import cohen_kappa_score

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

def specificity(y_true: list[int], y_pred: list[int]) -> float:
    """Proportion of correct notes that were NOT flagged."""
    c = confusion_counts(y_true, y_pred)
    negatives = c["tn"] + c["fp"]
    if negatives == 0:
        raise ValueError("No negative cases: specificity is undefined")
    return c["tn"] / negatives


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Proportion of all notes classified correctly. Misleading when errors are rare."""
    c = confusion_counts(y_true, y_pred)
    return (c["tp"] + c["tn"]) / len(y_true)


def kappa(y_true: list[int], y_pred: list[int]) -> float:
    """Cohen's kappa: agreement beyond chance, via scikit-learn."""
    confusion_counts(y_true, y_pred)  # validates lengths and 0/1 values
    return float(cohen_kappa_score(y_true, y_pred))


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if total <= 0:
        raise ValueError("Total must be positive")
    if not 0 <= successes <= total:
        raise ValueError(f"Successes {successes} must be between 0 and {total}")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(y_true: list[int], y_pred: list[int]) -> dict[str, tuple[float, float, float]]:
    """Sensitivity, specificity and PPV, each as (estimate, ci_low, ci_high)."""
    c = confusion_counts(y_true, y_pred)
    parts = {
        "sensitivity": (c["tp"], c["tp"] + c["fn"]),
        "specificity": (c["tn"], c["tn"] + c["fp"]),
        "ppv": (c["tp"], c["tp"] + c["fp"]),
    }
    summary = {}
    for name, (x, n) in parts.items():
        if n == 0:
            raise ValueError(f"{name} is undefined: no cases in its denominator")
        low, high = wilson_ci(x, n)
        summary[name] = (x / n, low, high)
    return summary