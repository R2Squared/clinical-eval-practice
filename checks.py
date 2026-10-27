import re

DRUGS = {
    "apixaban",
    "warfarin",
    "metformin",
    "ramipril",
    "amlodipine",
    "penicillin",
    "cetirizine",
    "metoprolol succinate",
    "lisinopril",
    "levothyroxine",
    "insulin glargine",
    "hydroxychloroquine",
    "sulfasalazine",
    "folic acid",
    "omeprazole",
    "clopidogrel",
    "aspirin",
    "methotrexate",
    "salbutamol",
}

PAST_CUES = ("used to", "stopped", "previously", "switched from", "no longer")
PAST_WINDOW = 3  # words before a drug name to search for a past cue
SIDES = {"left", "right"}


def tokenize(text: str) -> list[str]:
    """Lowercase the text and split it into words, dropping punctuation and numbers."""
    return re.findall(r"[a-z]+", text.lower())


def current_drugs(text: str) -> set[str]:
    """Drug names mentioned in the text, skipping any preceded by a past-tense cue."""
    words = tokenize(text)
    found = set()
    for i, word in enumerate(words):
        if word in DRUGS:
            preceding = " ".join(words[max(0, i - PAST_WINDOW):i])
            if not any(cue in preceding for cue in PAST_CUES):
                found.add(word)
    return found


def medication_consistency(transcript: str, note: str) -> dict[str, set[str]]:
    """Drugs in the source but missing from the note, and drugs in the note but absent from the source."""
    source = current_drugs(transcript)
    in_note = current_drugs(note)
    return {"omitted": source - in_note, "invented": in_note - source}


def side_mentions(text: str) -> set[tuple[str, str]]:
    """Pairs of (side, the word after it), e.g. ('left', 'knee')."""
    words = tokenize(text)
    pairs = set()
    for i, word in enumerate(words):
        if word in SIDES and i + 1 < len(words):
            pairs.add((word, words[i + 1]))
    return pairs


def laterality_consistency(transcript: str, note: str) -> set[tuple[str, str]]:
    """Side mentions that appear in the note but not in the transcript."""
    return side_mentions(note) - side_mentions(transcript)


def any_flag(transcript: str, note: str) -> bool:
    """True if any check finds a problem."""
    meds = medication_consistency(transcript, note)
    return bool(meds["omitted"] or meds["invented"] or laterality_consistency(transcript, note))