"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import re

import pytest

from app.core import anonymize, corpus_stats, load_corpus, tokenize


# Milestone 1: Curate an anonymized corpus of administrative medical text.
def test_milestone_1():
    text = ("Mr. Daniel Reyes (MRN 4482913) called (212) 555-0142 on 03/14/2026 "
            "at 9:30 am, email dan@example.com.")
    cleaned = anonymize(text)
    for secret in ["Daniel", "Reyes", "4482913", "555-0142", "03/14/2026", "9:30", "dan@example.com"]:
        assert secret not in cleaned
    assert tokenize(text) == ["mr", "<name>", "mrn", "<mrn>", "called", "<phone>", "on", "<date>",
                              "at", "<time>", "email", "<email>"]


def test_milestone_1_the_corpus_is_clean():
    sentences = load_corpus()
    assert corpus_stats(sentences)["sentences"] >= 50
    text = " ".join(" ".join(sentence) for sentence in sentences)
    assert text == text.lower()
    assert "@" not in text and not re.search(r"\d{3,}", text)  # no emails, phones, dates or IDs left
    assert "<name>" in text and "<date>" in text                # identifiers were replaced, not lost


def test_milestone_1_comments_and_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("# a comment\n\nPatient checked in.\n   \nReferral sent to Dr. Lee.\n", encoding="utf-8")
    assert load_corpus(path) == [["patient", "checked", "in"], ["referral", "sent", "to", "dr", "<name>"]]
    assert corpus_stats(load_corpus(path)) == {"sentences": 2, "words": 8, "vocabulary": 8}


# Milestone 2: Implement the n-gram language model and apply smoothing.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Develop a text generation feature that suggests words based on input.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Create a CLI for users to interact with the suggestion engine.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")

