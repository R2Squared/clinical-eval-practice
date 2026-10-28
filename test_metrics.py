import pytest
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint
from metrics import (confusion_counts, sensitivity, ppv, specificity,
                     accuracy, kappa, wilson_ci, summarise)

Y_TRUE = [1, 1, 1, 0, 0, 0, 1, 0]
Y_PRED = [1, 0, 1, 1, 0, 1, 1, 0]


def test_confusion_counts():
    assert confusion_counts(Y_TRUE, Y_PRED) == {"tp": 3, "fp": 2, "fn": 1, "tn": 2}


def test_sensitivity():
    assert sensitivity(Y_TRUE, Y_PRED) == pytest.approx(0.75)


def test_ppv():
    assert ppv(Y_TRUE, Y_PRED) == pytest.approx(0.6)

def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="Length mismatch"):
        confusion_counts([1, 1, 1], [1])


def test_invalid_label_raises():
    with pytest.raises(ValueError, match="Invalid value"):
        confusion_counts([2, 1], [1, 1])


def test_missing_label_raises():
    with pytest.raises(ValueError, match="Invalid value"):
        confusion_counts([float("nan"), 1], [1, 1])


def test_sensitivity_no_positives_raises():
    with pytest.raises(ValueError, match="No positive cases"):
        sensitivity([0, 0, 0], [0, 1, 0])


def test_ppv_no_flags_raises():
    with pytest.raises(ValueError, match="No flagged cases"):
        ppv([1, 0, 1], [0, 0, 0])

def test_counts_match_sklearn():
    tn, fp, fn, tp = confusion_matrix(Y_TRUE, Y_PRED).ravel()
    assert confusion_counts(Y_TRUE, Y_PRED) == {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def test_specificity():
    assert specificity(Y_TRUE, Y_PRED) == pytest.approx(0.5)


def test_accuracy():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(0.625)


def test_kappa_known_table():
    assert kappa(Y_TRUE, Y_PRED) == pytest.approx(0.25)


def test_specificity_no_negatives_raises():
    with pytest.raises(ValueError, match="No negative cases"):
        specificity([1, 1], [1, 0])


def test_wilson_matches_statsmodels():
    low, high = wilson_ci(3, 4)
    ref_low, ref_high = proportion_confint(3, 4, alpha=0.05, method="wilson")
    assert low == pytest.approx(ref_low)
    assert high == pytest.approx(ref_high)


def test_wilson_zero_successes_starts_at_zero():
    low, high = wilson_ci(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high > 0


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_ci(5, 4)


def test_summarise_agrees_with_point_estimates():
    s = summarise(Y_TRUE, Y_PRED)
    estimate, low, high = s["sensitivity"]
    assert estimate == pytest.approx(sensitivity(Y_TRUE, Y_PRED))
    assert low <= estimate <= high