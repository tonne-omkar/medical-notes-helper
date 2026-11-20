"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import sys
from pathlib import Path

from app import evaluate
from app.core import CORPUS_PATH, corpus_stats, complete, load_corpus, suggest, train_model
from app.phrases import DEFAULT_PATH as PHRASES_PATH, PhraseBook

HELP_TEXT = """Type words to build your draft. Then:
  1, 2, 3 ...   add that suggestion to the draft (a line with only a number picks;
                a number inside a longer line, like "in 2 weeks", is just text)
  undo          remove the last word
  clear         start a new draft
  help          show this message
  quit          leave (Ctrl+D or Ctrl+C also work)
Saved abbreviations (see the 'phrases' command) are expanded as you type them."""


def _format_suggestions(suggestions: list[tuple[str, float]]) -> str:
    if not suggestions:
        return "(no suggestions)"
    return "   ".join(f"{i}) {word} ({p:.1%})" for i, (word, p) in enumerate(suggestions, start=1))


def run_interactive(model, top_n: int = 3, input_fn=input, output_fn=print, phrases: PhraseBook | None = None) -> str:
    """Let the user build a draft word by word. Returns the final draft.

    input_fn and output_fn are parameters so tests can feed in lines and read the output.
    `phrases` holds the user's saved abbreviations, which are expanded as they are typed.
    """
    draft: list[str] = []
    output_fn("Medical Notes Helper. Type 'help' for commands, 'quit' to leave.")
    while True:
        output_fn(f"\nDraft: {' '.join(draft) or '(empty)'}")
        output_fn("Next: " + _format_suggestions(suggest(model, " ".join(draft), top_n)))
        try:
            line = input_fn("> ").lstrip("\ufeff").strip()   # a byte-order mark can sneak in from piped input
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            break
        command = line.lower()
        if command in ("quit", "exit"):
            break
        elif command == "help":
            output_fn(HELP_TEXT)
        elif command == "clear":
            draft = []
        elif command == "undo":
            if draft:
                output_fn(f"Removed '{draft.pop()}'.")
            else:
                output_fn("Nothing to undo.")
        elif line.isascii() and line.isdigit():          # a lone number picks a suggestion
            choices = suggest(model, " ".join(draft), top_n)
            if 1 <= int(line) <= len(choices):
                draft.append(choices[int(line) - 1][0])
            else:
                output_fn(f"There is no suggestion {line} right now.")
        elif line:
            words = line.split()
            expanded = phrases.expand(words) if phrases else words
            if expanded != words:
                output_fn("Expanded: " + " ".join(expanded))
            draft.extend(expanded)
    output_fn(f"\nFinal draft: {' '.join(draft) or '(empty)'}")
    return " ".join(draft)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medical-notes-helper", description="Clinic staff use this CLI tool to draft administrative sections of medical notes by generating next-word suggestions, ensuring consistency and reducing manual typing.")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH, help="text file with one sentence per line (default: corpus/admin_notes.txt)")
    parser.add_argument("--order", type=_positive_int, default=3, help="n-gram order: how many words the model looks at, including the next one (default: 3)")
    parser.add_argument("--smoothing", type=_positive_float, default=1.0, help="add-k smoothing value (default: 1.0, which is add-one smoothing)")
    parser.add_argument("--phrases", type=Path, default=PHRASES_PATH, help="JSON file holding your saved abbreviations (default: data/phrases.json)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    p = sub.add_parser("phrases", help="save, list or remove your own abbreviations, e.g. fu = follow-up appointment")
    actions = p.add_subparsers(dest="action", required=True)
    q = actions.add_parser("add", help="save an abbreviation (replaces it if it exists)")
    q.add_argument("abbreviation", help="a single word, e.g. fu")
    q.add_argument("expansion", nargs="+", help="what it stands for, e.g. follow-up appointment")
    actions.add_parser("list", help="show your saved abbreviations")
    q = actions.add_parser("remove", help="delete a saved abbreviation")
    q.add_argument("abbreviation")
    p = sub.add_parser("suggest", help="show the most likely next words for the text you give")
    p.add_argument("text", nargs="*", help="the draft so far; leave empty to see how sentences usually start")
    p.add_argument("--top", type=_positive_int, default=3, help="how many suggestions to show (default: 3)")
    p = sub.add_parser("complete", help="finish a sentence with the most likely words")
    p.add_argument("text", nargs="*", help="the start of the sentence")
    p.add_argument("--max-words", type=_positive_int, default=12, help="stop after adding this many words (default: 12)")
    sub.add_parser("stats", help="show how big the corpus is")
    sub.add_parser("evaluate", help="test the suggestions on held-out sentences and compare with a simple baseline")
    p = sub.add_parser("interactive", help="build a draft by typing words and picking suggestions")
    p.add_argument("--top", type=_positive_int, default=3, help="how many suggestions to show (default: 3)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Medical Notes Helper is ready. Open README.md for your milestones.")
        return
    try:
        phrases = PhraseBook(args.phrases)
        if args.command == "phrases":
            _run_phrases(phrases, args)
            return
        if args.command == "stats":
            stats = corpus_stats(load_corpus(args.corpus))
            print(f"Corpus: {args.corpus}")
            print(f"Sentences: {stats['sentences']}   Words: {stats['words']}   Distinct words: {stats['vocabulary']}")
            return
        if args.command == "evaluate":
            print(evaluate.run(args.corpus, args.smoothing))
            return
        model = train_model(args.order, args.smoothing, args.corpus)
    except FileNotFoundError:
        parser.exit(1, f"{parser.prog}: error: corpus file not found: {args.corpus}\n")
    except ValueError as error:                      # a bad phrases file or an invalid abbreviation
        parser.exit(1, f"{parser.prog}: error: {error}\n")
    text = " ".join(phrases.expand(" ".join(getattr(args, "text", [])).split()))
    if args.command == "suggest":
        print(_format_suggestions(suggest(model, text, args.top)))
    elif args.command == "complete":
        print(complete(model, text, args.max_words))
    elif args.command == "interactive":
        run_interactive(model, args.top, phrases=phrases)


def _run_phrases(phrases: PhraseBook, args) -> None:
    if args.action == "add":
        key = phrases.add(args.abbreviation, " ".join(args.expansion))
        print(f"Saved: {key} = {phrases.items[key]}")
    elif args.action == "remove":
        print(f"Removed: {args.abbreviation.lower()}" if phrases.remove(args.abbreviation)
              else f"No saved abbreviation called '{args.abbreviation}'.")
    elif phrases.items:
        for key, expansion in sorted(phrases.items.items()):
            print(f"{key} = {expansion}")
    else:
        print(f"No saved abbreviations yet. Add one with: phrases add fu follow-up appointment  (file: {phrases.path})")


if __name__ == "__main__":
    main()
