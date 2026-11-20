# Medical Notes Helper

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/f8m2fghn7d/badge.svg)](https://ask.thili.ai/b/f8m2fghn7d)

The Medical Notes Helper is a CLI tool designed for clinic staff in the Healthcare domain. It assists in drafting administrative sections of medical notes by providing next-word suggestions, aiming to improve consistency and reduce manual typing effort.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that helps clinic staff draft administrative sections of medical notes by generating next-word suggestions, ensuring consistency and reducing manual typing.

**Done when.** The tool successfully generates next-word suggestions based on an anonymized corpus of administrative medical text, and users can interact with this suggestion engine through a CLI.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Curate an anonymized corpus of administrative medical text.**  
      Hint: Use text processing techniques to clean and prepare your corpus.
- [x] **2. Implement the n-gram language model and apply smoothing.**  
      Hint: Apply the n-gram model and smoothing techniques from 'Build a Next-Word Predictor from Scratch'.
- [x] **3. Develop a text generation feature that suggests words based on input.**  
      Hint: Implement the text generation logic, leveraging your language model to predict words.
- [x] **4. Create a CLI for users to interact with the suggestion engine.**  
      Hint: Use command-line parsing libraries to create an interactive user interface.

## Run it

You need Python 3.12 and git.

macOS / Linux:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli hello
pytest
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.cli hello
pytest
```

No `.env` file, model or API key is needed: the suggestion engine is a plain-Python n-gram model.

## Using the tool

```bash
python -m app.cli interactive                 # build a draft word by word (recommended)
python -m app.cli suggest patient called to   # top 3 next words
python -m app.cli suggest --top 5 referral sent
python -m app.cli complete referral sent to   # finish the sentence
python -m app.cli stats                       # size of the corpus
```

Options go before the command: `--order 2` (n-gram order, default 3), `--smoothing 0.5` (add-k value, default 1),
`--corpus my_notes.txt` (use your own sentences, one per line). In `interactive` mode, type words to build a draft, then
`1`, `2` or `3` to add a suggestion, `undo` to remove the last word, `clear` to start over, `quit` to leave.

## How it works

- **Corpus** (`corpus/admin_notes.txt`): about 85 made-up administrative note sentences (scheduling, insurance,
  referrals, records, refills). It contains **no real patient data**. A few lines include fake names, dates, phone
  numbers, emails and record numbers on purpose, to test the cleaning step.
- **Cleaning** (`anonymize`, `tokenize` in `app/core.py`): replaces emails, dates, times, phone numbers, record numbers
  and titled names (like "Dr. Smith") with placeholders such as `<name>` and `<date>`, lowercases, and splits into
  words. These are simple patterns for practice text, **not** a de-identification process for real patient records.
- **Model** (`NGramModel`): counts n-grams of every order up to n. A word's probability is
  `(count + k) / (context total + k * vocabulary size)` (add-k smoothing), so a word never seen in a context still gets
  a small probability. If the whole context was never seen, it backs off to a shorter one.
- **Suggestions** (`suggest`, `complete`): suggests only words seen after the context, filling spare slots by backing
  off to shorter contexts. Probabilities shown are the smoothed ones. `complete` always picks the top word, so the same
  input gives the same output.
- **Limits:** the corpus is small, so suggestions are only as good as it is. Add your own clinic's wording to the corpus
  file to improve them.

## Tests

`pytest` runs everything (the same command runs on GitHub Actions). `tests/test_smoke.py` checks the starter plumbing;
`tests/test_core.py` has the tests for each milestone.

## Stretch ideas

- [x] Allow the user to select from a list of top N suggestions instead of just the single best one (`--top N`, and
      pick by number in interactive mode).
- [ ] Implement a feature to save and load user-specific common phrases or abbreviations.
- [x] Add a simple 'undo' functionality to remove the last suggested word (`undo` in interactive mode; it removes the
      last word, whether you typed it or picked it).
