"""Tests for the held-out evaluation in app/evaluate.py."""
import pytest

from app.cli import main
from app.core import NGramModel, load_corpus
from app.evaluate import compare, evaluate_model, split_corpus


def test_split_keeps_every_sentence_exactly_once():
    sentences = [[str(i)] for i in range(10)]
    train, test = split_corpus(sentences, every=5)
    assert test == [["0"], ["5"]] and len(train) == 8
    assert sorted(train + test) == sorted(sentences)         # nothing lost, nothing in both


def test_a_model_that_has_memorized_the_text_scores_perfectly():
    sentences = [["referral", "sent", "to", "cardiology"]] * 3
    result = evaluate_model(NGramModel(3, 1).fit(sentences), sentences[:1], top_n=3)
    assert result["top1"] == 1.0 and result["topn"] == 1.0 and result["predictions"] == 4
    assert result["perplexity"] < 3


def test_the_ngram_model_beats_the_simple_baseline():
    baseline, order2, order3 = compare(load_corpus())
    assert baseline["order"] == 1
    for row in (order2, order3):
        assert row["topn"] > baseline["topn"] and row["top1"] > baseline["top1"]
        assert row["perplexity"] < baseline["perplexity"]


def test_the_evaluate_command_prints_a_table(capsys):
    main(["evaluate"])
    out = capsys.readouterr().out
    assert "held-out" in out and "order 1 (baseline)" in out and "order 3" in out
    with pytest.raises(SystemExit):
        main(["--corpus", "no-such-file.txt", "evaluate"])
