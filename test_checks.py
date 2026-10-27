from checks import current_drugs, medication_consistency, laterality_consistency


def test_finds_drugs_case_insensitive():
    assert current_drugs("Takes Amlodipine daily.") == {"amlodipine"}


def test_omission_flagged():
    result = medication_consistency("Takes amlodipine and metformin.", "Medications: amlodipine.")
    assert result == {"omitted": {"metformin"}, "invented": set()}


def test_invention_flagged():
    result = medication_consistency("Takes amlodipine.", "Medications: amlodipine, warfarin.")
    assert result == {"omitted": set(), "invented": {"warfarin"}}


def test_past_drug_not_flagged_as_omission():
    result = medication_consistency("I used to take ramipril. Now on amlodipine.", "Medications: amlodipine.")
    assert result == {"omitted": set(), "invented": set()}


def test_past_drug_listed_as_current_is_flagged():
    result = medication_consistency("Now takes apixaban. Switched from warfarin last year.", "Medications: warfarin.")
    assert result == {"omitted": {"apixaban"}, "invented": {"warfarin"}}


def test_laterality_swap_flagged():
    assert laterality_consistency("Pain in the left knee.", "Pain in the right knee.") == {("right", "knee")}


def test_same_side_not_flagged():
    assert laterality_consistency("Pain in the left knee.", "Left knee pain.") == set()


def test_side_as_last_word_does_not_crash():
    assert laterality_consistency("Pain on the left", "Pain on the left") == set()