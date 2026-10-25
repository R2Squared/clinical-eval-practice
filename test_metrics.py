import pytest
from metrics import confusion_counts, sensitivity, ppv

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