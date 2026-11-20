"""Your code. One stub per milestone; build them in order and keep each one small."""
import re
from collections import Counter, defaultdict
from pathlib import Path

# --- Milestone 1: Curate an anonymized corpus of administrative medical text.
# Hint: Use text processing techniques to clean and prepare your corpus.

CORPUS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "admin_notes.txt"

# (pattern, replacement) pairs, applied in this order. Emails and dates go first so their
# digits are not mistaken for phone numbers or IDs later on.
_PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "<EMAIL>"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "<DATE>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "<DATE>"),
    (re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}"
                r"(?:st|nd|rd|th)?,?\s+\d{4}\b", re.IGNORECASE), "<DATE>"),
    (re.compile(r"\b\d{1,2}:\d{2}(?:\s?[ap]\.?m\b\.?)?", re.IGNORECASE), "<TIME>"),
    (re.compile(r"(?<!\d)(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.]\d{4}(?!\d)"), "<PHONE>"),
    (re.compile(r"\b(MRN|medical record number)(\s*[:#]?\s*)\d+", re.IGNORECASE), r"\1\2<MRN>"),
    (re.compile(r"\b\d{6,}\b"), "<ID>"),
    # A capitalized name after a title: "Dr. Alan Brooks" -> "Dr. <NAME>".
    (re.compile(r"\b(Mr|Mrs|Ms|Miss|Dr)(\.?\s+)[A-Z][a-z]+(?:[ '-][A-Z][a-z]+)?"), r"\1\2<NAME>"),
]

# A placeholder like <date>, or a word that may contain digits, hyphens or apostrophes.
_TOKEN_RE = re.compile(r"<[a-z]+>|[a-z0-9]+(?:['-][a-z0-9]+)*")


def anonymize(text: str) -> str:
    """Replace emails, dates, times, phone numbers, record numbers and titled names with placeholders.

    These are simple patterns for synthetic practice text. They are not a substitute for a proper
    de-identification process on real patient data.
    """
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def tokenize(text: str) -> list[str]:
    """Anonymize, lowercase and split into words. Punctuation is dropped."""
    return _TOKEN_RE.findall(anonymize(text).lower())


def load_corpus(path: str | Path = CORPUS_PATH) -> list[list[str]]:
    """Read one sentence per line, skip blank lines and # comments, and return each line cleaned into words."""
    sentences = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words = tokenize(line)
        if words:
            sentences.append(words)
    return sentences


def corpus_stats(sentences: list[list[str]]) -> dict:
    """Sentence count, word count and number of distinct words."""
    words = [w for sentence in sentences for w in sentence]
    return {"sentences": len(sentences), "words": len(words), "vocabulary": len(set(words))}


# --- Milestone 2: Implement the n-gram language model and apply smoothing.
# Hint: Apply the n-gram model and smoothing techniques from 'Build a Next-Word Predictor from Scratch'.

START, END, UNK = "<s>", "</s>", "<unk>"   # sentence start padding, sentence end, unknown word


class NGramModel:
    """An n-gram language model with add-k smoothing (k=1 is Laplace, or add-one, smoothing).

    It counts every order from 1 up to n. To score a word it uses the longest context it saw
    in training: if the last n-1 words never appeared together, it backs off to a shorter
    context, down to plain word counts (the empty context), which always exist after fit().
    """

    def __init__(self, n: int = 3, k: float = 1.0):
        if n < 1:
            raise ValueError("n must be at least 1")
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.n, self.k = n, k
        self.counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)  # context -> next-word counts
        self.totals: dict[tuple[str, ...], int] = {}                       # context -> sum of those counts
        self.vocab: set[str] = set()

    def fit(self, sentences: list[list[str]]) -> "NGramModel":
        for sentence in sentences:
            words = [START] * (self.n - 1) + list(sentence) + [END]
            self.vocab.update(sentence)
            for i in range(self.n - 1, len(words)):
                for size in range(self.n):                 # context sizes 0 .. n-1
                    self.counts[tuple(words[i - size:i])][words[i]] += 1
        self.vocab.update({END, UNK})
        self.totals = {context: sum(c.values()) for context, c in self.counts.items()}
        return self

    def context(self, history: list[str]) -> tuple[str, ...]:
        """The last n-1 words of history, with unknown words as <unk> and <s> padding at the start."""
        if self.n == 1:
            return ()
        words = [START] * (self.n - 1) + [w if w in self.vocab else UNK for w in history]
        return tuple(words[-(self.n - 1):])

    def seen_context(self, history: list[str]) -> tuple[str, ...]:
        """The longest end part of the context that appeared in training (backoff)."""
        context = self.context(history)
        while context and context not in self.counts:
            context = context[1:]
        return context

    def prob(self, word: str, history: list[str] = ()) -> float:
        """P(word | history) = (count + k) / (total + k * vocabulary size)."""
        if not self.totals:
            raise RuntimeError("Call fit() before asking for probabilities.")
        context = self.seen_context(list(history))
        count = self.counts[context][word if word in self.vocab else UNK]
        return (count + self.k) / (self.totals[context] + self.k * len(self.vocab))


def train_model(n: int = 3, k: float = 1.0, path: str | Path = CORPUS_PATH) -> NGramModel:
    """Load and clean the corpus, then fit an n-gram model on it."""
    return NGramModel(n, k).fit(load_corpus(path))


# --- Milestone 3: Develop a text generation feature that suggests words based on input.
# Hint: Implement the text generation logic, leveraging your language model to predict words.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Create a CLI for users to interact with the suggestion engine.
# Hint: Use command-line parsing libraries to create an interactive user interface.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
