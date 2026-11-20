"""Measure how good the suggestions are on sentences the model was not trained on.

Every 5th corpus sentence is held out for testing and the model is trained on the rest. We compare
n-gram orders, where order 1 is the simple baseline: it ignores what you typed and always suggests the
most common words. Run it with `python -m app.cli evaluate`.
"""
import math

from app.core import END, NGramModel, load_corpus, suggest


def split_corpus(sentences: list[list[str]], every: int = 5) -> tuple[list[list[str]], list[list[str]]]:
    """(train, test): every `every`-th sentence goes to the test set, the rest to training."""
    train = [s for i, s in enumerate(sentences) if i % every != 0]
    test = [s for i, s in enumerate(sentences) if i % every == 0]
    return train, test


def evaluate_model(model: NGramModel, test: list[list[str]], top_n: int = 3) -> dict:
    """Perplexity (lower is better) and how often the real next word is the top suggestion or in the top_n."""
    log_prob_sum, predictions = 0.0, 0
    top1 = topn = scored = 0
    for sentence in test:
        for i, word in enumerate(list(sentence) + [END]):
            history = list(sentence[:i])
            log_prob_sum += math.log(model.prob(word, history))
            predictions += 1
            if word == END:
                continue                      # suggestions are words, so the sentence end is not scored here
            suggestions = [w for w, _ in suggest(model, " ".join(history), top_n)]
            scored += 1
            top1 += suggestions[:1] == [word]
            topn += word in suggestions
    return {"perplexity": math.exp(-log_prob_sum / predictions), "top1": top1 / scored, "topn": topn / scored,
            "predictions": scored}


def compare(sentences: list[list[str]], orders=(1, 2, 3), k: float = 1.0, top_n: int = 3, every: int = 5) -> list[dict]:
    train, test = split_corpus(sentences, every)
    rows = []
    for n in orders:
        result = evaluate_model(NGramModel(n, k).fit(train), test, top_n)
        rows.append({"order": n, "train": len(train), "test": len(test), **result})
    return rows


def format_table(rows: list[dict], top_n: int = 3) -> str:
    first = rows[0]
    lines = [f"Trained on {first['train']} sentences, tested on {first['test']} held-out sentences "
             f"({first['predictions']} next-word predictions).",
             f"{'model':<26}{'perplexity':>11}{'top-1':>9}{'top-' + str(top_n):>9}"]
    for row in rows:
        name = f"order {row['order']}" + (" (baseline)" if row["order"] == 1 else "")
        lines.append(f"{name:<26}{row['perplexity']:>11.1f}{row['top1']:>9.1%}{row['topn']:>9.1%}")
    return "\n".join(lines)


def run(corpus_path, k: float = 1.0, top_n: int = 3) -> str:
    return format_table(compare(load_corpus(corpus_path), k=k, top_n=top_n), top_n)
