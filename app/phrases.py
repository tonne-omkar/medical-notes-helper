"""Save and load a clinic's own abbreviations, e.g. `fu` -> `follow-up appointment`.

They live in one small JSON file (default data/phrases.json, which git ignores because it is personal to each
user). In the interactive mode and in `suggest` / `complete`, a word that matches an abbreviation is replaced by its
expansion before the model sees it.
"""
import json
from pathlib import Path

DEFAULT_PATH = Path("data") / "phrases.json"
RESERVED = {"quit", "exit", "help", "clear", "undo"}   # interactive commands: an abbreviation can't be one of these


class PhraseBook:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)
        self.items: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{self.path} is not valid JSON ({error.msg}). Fix it or delete it.") from None
        if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
            raise ValueError(f"{self.path} should hold an object of abbreviation: expansion pairs. Fix it or delete it.")
        return data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(sorted(self.items.items())), indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")

    def add(self, abbreviation: str, expansion: str) -> str:
        """Save (or replace) an abbreviation and return the cleaned-up abbreviation."""
        key, expansion = abbreviation.strip().lower(), " ".join(expansion.split())
        if not key or len(key.split()) != 1:
            raise ValueError("An abbreviation must be a single word.")
        if key.isascii() and key.isdigit():
            raise ValueError("An abbreviation can't be only digits (a lone number picks a suggestion).")
        if key in RESERVED:
            raise ValueError(f"'{key}' is a command in interactive mode, so it can't be an abbreviation.")
        if not expansion:
            raise ValueError("The expansion can't be empty.")
        self.items[key] = expansion
        self._save()
        return key

    def remove(self, abbreviation: str) -> bool:
        """Delete an abbreviation. Returns False if there was no such abbreviation."""
        if self.items.pop(abbreviation.strip().lower(), None) is None:
            return False
        self._save()
        return True

    def expand(self, words: list[str]) -> list[str]:
        """Replace every word that is a saved abbreviation (any capitalization) by its expansion."""
        result: list[str] = []
        for word in words:
            result.extend(self.items[word.lower()].split() if word.lower() in self.items else [word])
        return result
