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

No `.env` file, model or API key is needed: the suggestion engine is a plain-Python n-gram model. The only dependency
is `pytest` (pinned to `pytest==9.2.0` in `requirements.txt`); the tool itself uses just the Python standard library.
It was developed and tested on Python 3.12.4 (Windows), and the same tests run on Ubuntu with Python 3.12 in GitHub
Actions.

## Using the tool

```bash
python -m app.cli interactive                 # build a draft word by word (recommended)
python -m app.cli suggest patient called to   # top 3 next words
python -m app.cli suggest --top 5 referral sent
python -m app.cli complete referral sent to   # finish the sentence
python -m app.cli stats                       # size of the corpus
python -m app.cli evaluate                    # test on held-out sentences vs. a simple baseline
python -m app.cli phrases add fu follow-up appointment   # save your own abbreviation
python -m app.cli phrases list                           # show saved abbreviations
python -m app.cli phrases remove fu                      # delete one
```

Options go before the command: `--order 2` (n-gram order, default 3), `--smoothing 0.5` (add-k value, default 1),
`--corpus my_notes.txt` (use your own sentences, one per line), `--phrases my_phrases.json` (where abbreviations are
saved; default `data/phrases.json`, which git ignores because it is personal). A saved abbreviation is a single word,
case-insensitive, that is replaced by its expansion in `interactive`, `suggest` and `complete`: with `fu` saved above,
`python -m app.cli complete patient needs a fu` starts from "patient needs a follow-up appointment". In `interactive` mode, type words to build a draft, then
`1`, `2` or `3` to add a suggestion, `undo` to remove the last word, `clear` to start over, `quit` to leave.

## Results

Real output from the tool (corpus: 85 sentences, 632 words, 186 distinct words):

```text
$ python -m app.cli suggest patient called to
1) ask (1.0%)   2) cancel (1.0%)   3) reschedule (1.0%)

$ python -m app.cli complete referral sent to
referral sent to the patient

$ python -m app.cli interactive      # typed: patient, 1, 9, undo, quit
Draft: (empty)
Next: 1) patient (11.4%)   2) referral (2.6%)   3) appointment (1.8%)
> patient
Draft: patient
Next: 1) was (6.4%)   2) called (2.8%)   3) arrived (1.4%)
> 1
Draft: patient was
Next: 1) notified (2.5%)   2) asked (2.0%)   3) given (1.5%)
> 9
There is no suggestion 9 right now.
> undo
Removed 'was'.
```

**How good are the suggestions?** `python -m app.cli evaluate` holds out every 5th sentence (17 of 85), trains on the
other 68, and checks, for each word in the held-out sentences, whether the model's suggestions contain the word that
was really written (125 predictions). Order 1 is a simple baseline that ignores what you typed and always suggests the
most common words.

| Model | Perplexity (lower is better) | Real next word is the top suggestion | Real next word is in the top 3 |
|---|---|---|---|
| Order 1 (baseline) | 87.0 | 11.2% | 24.8% |
| Order 2 | 76.0 | 36.8% | 49.6% |
| Order 3 (default) | 84.4 | 38.4% | 51.2% |

What this shows, and what it doesn't:

- Looking at the previous words roughly **triples** top-1 accuracy and **doubles** top-3 accuracy compared with the
  baseline (11% to 38%, 25% to 51%). A test checks this, so it can't silently regress.
- Order 3 is only slightly more accurate than order 2 and has a *worse* perplexity. With just 68 training sentences,
  three-word contexts are rarely repeated, and add-one smoothing spreads probability thinly. I did not tune the
  settings to hide this; a bigger corpus would be the real fix.
- These numbers are optimistic for real clinics: the corpus was written by one person using repeating sentence
  patterns, so held-out sentences resemble training ones. Treat them as a comparison between models, not as a promise
  of accuracy on real notes. One split of 17 sentences is also a small sample.

## How it works

- **Corpus** (`corpus/admin_notes.txt`): where the data comes from: I wrote these 85 sentences myself (with AI
  assistance) for this project. They are synthetic administrative note lines (scheduling, insurance, referrals,
  records, refills), not copied from any dataset, website or patient record. It contains **no real patient data**. A
  few lines include fake names, dates, phone numbers (reserved 555-01xx style), emails (example.com) and record
  numbers on purpose, to test the cleaning step.
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

`pytest` runs everything (the same command runs on GitHub Actions). `tests/test_core.py` has the tests for each
milestone, `tests/test_evaluate.py` tests the evaluation (including that the n-gram model beats the baseline),
`tests/test_phrases.py` tests saved abbreviations, and `tests/test_smoke.py` checks that the command line starts.
`tests/conftest.py` keeps your own saved abbreviations out of the tests.

## Stretch ideas

- [x] Allow the user to select from a list of top N suggestions instead of just the single best one (`--top N`, and
      pick by number in interactive mode).
- [x] Implement a feature to save and load user-specific common phrases or abbreviations (`phrases add/list/remove`;
      saved in `data/phrases.json` and expanded as you type).
- [x] Add a simple 'undo' functionality to remove the last suggested word (`undo` in interactive mode; it removes the
      last word, whether you typed it or picked it).
