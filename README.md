# TSW-NL

TSW-NL is a Dutch LLM benchmark. Each puzzle provides three clues that refer to different meanings or uses of the same word. The task is to produce that word in Dutch, without a list of candidate answers or external tools.

For example, the puzzle shown below: 

> Het is een voetbaltrainer, een alcoholische drank, en een beroep. Wat is het?

Has as an answer **Advocaat**.

## Setup

Use Python 3.10 or newer. Clone the repository, create and activate a virtual environment, and install the requirements:

```bash
git clone https://github.com/RikVN/TSW-NL.git
cd TSW-NL
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

We release 2 sets: a dev set of 22 instances and a test set of 200 instances.

- `data/dev.jsonl`
- `data/test.jsonl`

Each instance contains `id`, `clue1`, `clue2`, `clue3`, and `answer`. There are 7 instances that additionally contain `accepted_answers`, listing the primary answer and an accepted alternative.

The puzzles were constructed from Dutch Wikipedia disambiguation pages and related-page descriptions, using automatic filtering followed by manual clue selection and editing.

## Run a model

Run from the main dir:

```bash
python run.py --model HUGGING_FACE_MODEL_ID
python evaluate.py outputs/predictions.jsonl
```

This runner always uses **zero-shot prompting and clue order 123**. It loads the exact instructions from `prompts/system.txt` and `prompts/user.txt`.

To test your setup run:

```bash
python run.py --model YOUR_HUGGING_FACE_MODEL_ID --limit 3 --output outputs/test.jsonl
```

## Evaluate your own predictions

Evaluate your model like this:

```bash
python evaluate.py path/to/predictions.jsonl
```

## License

The repository includes the Apache License 2.0 in [LICENSE](LICENSE). The benchmark's source provenance is described above.

## Paper and citation

The corresponding paper was accepted at EACL 2027 (Findings).

Citation will be added when available.
