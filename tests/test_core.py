"""Tests for each milestone in README.md, in order: corpus cleaning, n-gram model, suggestions, command line.
"""
import re

import pytest

from app.cli import main, run_interactive
from app.core import (END, START, UNK, NGramModel, anonymize, complete, corpus_stats, load_corpus, suggest,
                      tokenize, train_model)


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
def test_milestone_2():
    model = NGramModel(n=2, k=1).fit([["a", "b"], ["a", "c"]])
    assert model.vocab == {"a", "b", "c", END, UNK}                    # V = 5
    assert model.prob("b", ["a"]) == pytest.approx((1 + 1) / (2 + 5))   # seen after "a"
    assert model.prob("a", ["a"]) == pytest.approx((0 + 1) / (2 + 5))   # never seen after "a", still > 0
    assert model.prob("a", []) == pytest.approx((2 + 1) / (2 + 5))      # first word of a sentence
    # "zebra" is unknown, so the context backs off to plain word counts: a=2 b=1 c=1 </s>=2
    assert model.prob("a", ["zebra"]) == pytest.approx((2 + 1) / (6 + 5))


def test_milestone_2_probabilities_sum_to_one():
    model = train_model()
    for history in ([], ["referral", "sent", "to"], ["completely", "unknown", "words"]):
        assert sum(model.prob(word, history) for word in model.vocab) == pytest.approx(1.0)


def test_milestone_2_the_model_prefers_what_it_has_seen():
    model = train_model()
    assert model.prob("appointment", ["a", "follow-up"]) > model.prob("cardiology", ["a", "follow-up"])
    assert model.prob("cardiology", ["a", "follow-up"]) > 0


# Milestone 3: Develop a text generation feature that suggests words based on input.
def test_milestone_3():
    model = train_model()
    words = [word for word, _ in suggest(model, "Patient called to", top_n=3)]
    assert len(words) == 3
    assert set(words) <= {"ask", "cancel", "reschedule", "schedule"}   # the only words that follow it in the corpus
    assert suggest(model, "Patient called TO") == suggest(model, "patient called to")  # input is cleaned


def test_milestone_3_suggestions_are_ranked_and_clean():
    model = train_model()
    result = suggest(model, "referral sent", top_n=5)
    probabilities = [p for _, p in result]
    assert probabilities == sorted(probabilities, reverse=True) and all(p > 0 for p in probabilities)
    assert not {START, END, UNK} & {word for word, _ in result}
    assert result[0][0] == "to"                                  # "referral sent" is always followed by "to"
    assert not any(word.startswith("<") for word, _ in result)   # no never-seen placeholders as filler
    assert suggest(model, "referral sent", top_n=0) == []
    assert suggest(model, "") != []                              # an empty box suggests sentence starters


def test_milestone_3_unknown_words_still_get_suggestions():
    assert len(suggest(train_model(), "zebra quantum", top_n=3)) == 3


def test_milestone_3_complete_finishes_a_sentence():
    model = train_model()
    sentence = complete(model, "referral sent to")
    assert sentence.startswith("referral sent to") and len(sentence.split()) > 3
    assert complete(model, "referral sent to") == sentence       # same input, same output
    assert len(complete(model, "patient", max_words=2).split()) <= 3
    assert complete(model, "patient", max_words=0) == "patient"


# Milestone 4: Create a CLI for users to interact with the suggestion engine.
def test_milestone_4(capsys):
    main(["suggest", "referral", "sent"])
    assert "1) to" in capsys.readouterr().out
    main(["complete", "referral", "sent", "to"])
    assert capsys.readouterr().out.startswith("referral sent to")
    main(["stats"])
    assert "Sentences:" in capsys.readouterr().out


def test_milestone_4_options_are_validated(capsys):
    for bad in (["--order", "0", "stats"], ["suggest", "--top", "0"], ["--smoothing", "-1", "stats"]):
        with pytest.raises(SystemExit):
            main(bad)
    with pytest.raises(SystemExit):
        main(["--corpus", "no-such-file.txt", "stats"])
    assert "corpus file not found" in capsys.readouterr().err


def test_milestone_4_interactive_session():
    model = train_model()
    lines = iter(["patient", "1", "undo", "undo", "undo", "referral sent", "clear", "help", "quit"])
    output = []
    final = run_interactive(model, top_n=3, input_fn=lambda prompt: next(lines), output_fn=output.append)
    text = "\n".join(output)
    assert final == ""                                         # the draft was cleared before quitting
    assert "Removed 'patient'." in text and "Nothing to undo." in text
    assert "Draft: patient" in text and "Draft: referral sent" in text


def test_milestone_4_interactive_pick_and_end_of_input():
    model = train_model()
    lines = iter(["\ufeffpatient", "1", "9", "0", "²", "in 2 weeks"])   # then input runs out, like Ctrl+D
    def fake_input(prompt):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError
    output = []
    final = run_interactive(model, top_n=3, input_fn=fake_input, output_fn=output.append)
    words = final.split()
    assert words[0] == "patient" and words[2:] == ["²", "in", "2", "weeks"]   # "1" picked a word; 9 and 0 were refused
    assert len(words) == 6                          # "²" doesn't crash (it is plain text); "in 2 weeks" stays text
    assert "There is no suggestion 9 right now." in output
    assert "Final draft:" in output[-1]

