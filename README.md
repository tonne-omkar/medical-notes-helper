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
- [ ] **2. Implement the n-gram language model and apply smoothing.**  
      Hint: Apply the n-gram model and smoothing techniques from 'Build a Next-Word Predictor from Scratch'.
- [ ] **3. Develop a text generation feature that suggests words based on input.**  
      Hint: Implement the text generation logic, leveraging your language model to predict words.
- [ ] **4. Create a CLI for users to interact with the suggestion engine.**  
      Hint: Use command-line parsing libraries to create an interactive user interface.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the user to select from a list of top N suggestions instead of just the single best one.
- Implement a feature to save and load user-specific common phrases or abbreviations.
- Add a simple 'undo' functionality to remove the last suggested word.
