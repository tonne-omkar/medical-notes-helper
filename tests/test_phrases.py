"""Tests for saving and using the user's own abbreviations (app/phrases.py)."""
import pytest

from app.cli import main, run_interactive
from app.core import train_model
from app.phrases import PhraseBook


def test_abbreviations_are_saved_and_loaded_again(tmp_path):
    path = tmp_path / "sub" / "phrases.json"                   # the folder doesn't exist yet
    book = PhraseBook(path)
    assert book.items == {}                                    # no file yet is fine
    assert book.add("  FU ", "follow-up   appointment") == "fu"
    assert PhraseBook(path).items == {"fu": "follow-up appointment"}   # survives a restart
    assert book.remove("FU") is True and book.remove("fu") is False
    assert PhraseBook(path).items == {}


def test_expand_replaces_whole_words_only():
    book = PhraseBook("does-not-exist.json")
    book.items = {"fu": "follow-up appointment", "ins": "insurance"}
    assert book.expand(["Patient", "needs", "FU", "and", "ins", "check", "fund"]) == [
        "Patient", "needs", "follow-up", "appointment", "and", "insurance", "check", "fund"]


@pytest.mark.parametrize("abbreviation, expansion", [
    ("", "x"), ("two words", "x"), ("12", "x"), ("quit", "x"), ("Undo", "x"), ("fu", "   ")])
def test_bad_abbreviations_are_refused(tmp_path, abbreviation, expansion):
    book = PhraseBook(tmp_path / "p.json")
    with pytest.raises(ValueError):
        book.add(abbreviation, expansion)
    assert book.items == {} and not (tmp_path / "p.json").exists()    # nothing was saved


@pytest.mark.parametrize("content", ["{not json", '["a", "b"]', '{"fu": 5}'])
def test_a_broken_phrases_file_gives_a_clear_error(tmp_path, content):
    path = tmp_path / "p.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Fix it or delete it"):
        PhraseBook(path)


def test_the_phrases_command_end_to_end(tmp_path, capsys):
    path = str(tmp_path / "p.json")
    main(["--phrases", path, "phrases", "list"])
    assert "No saved abbreviations yet" in capsys.readouterr().out
    main(["--phrases", path, "phrases", "add", "ref", "referral", "sent", "to"])
    assert "Saved: ref = referral sent to" in capsys.readouterr().out
    main(["--phrases", path, "phrases", "list"])
    assert "ref = referral sent to" in capsys.readouterr().out
    main(["--phrases", path, "suggest", "ref"])                # "ref" is expanded before suggesting
    assert capsys.readouterr().out.startswith("1) ")
    main(["--phrases", path, "complete", "ref"])
    assert capsys.readouterr().out.startswith("referral sent to")
    main(["--phrases", path, "phrases", "remove", "ref"])
    assert "Removed: ref" in capsys.readouterr().out
    main(["--phrases", path, "phrases", "remove", "ref"])
    assert "No saved abbreviation" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--phrases", path, "phrases", "add", "quit", "leave"])
    assert "interactive mode" in capsys.readouterr().err


def test_interactive_mode_expands_abbreviations(tmp_path):
    book = PhraseBook(tmp_path / "p.json")
    book.add("fu", "follow-up appointment")
    lines = iter(["patient needs a FU", "quit"])
    output = []
    final = run_interactive(train_model(), 3, input_fn=lambda prompt: next(lines), output_fn=output.append,
                            phrases=book)
    assert final == "patient needs a follow-up appointment"
    assert "Expanded: patient needs a follow-up appointment" in output
