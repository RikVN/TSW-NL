#!/usr/bin/env python3

"""Score predictions by item ID using strict, case-insensitive answer matching."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


PLACEHOLDER_ANSWERS = {
    "",
    "...",
    "<antwoord>",
    "<answer>",
    "<antwoord hier>",
    "<your answer>",
    "<het antwoord>",
}

def extract_json_objects(text: str) -> list[str]:
    return re.findall(r"\{.*?\}", text, flags=re.DOTALL)

def text_after_final_reasoning_block(text: str) -> str:
    matches = list(re.finditer(r"</think>", text, flags=re.IGNORECASE))
    if not matches:
        return text
    return text[matches[-1].end() :].strip()

def is_placeholder_answer(answer: str) -> bool:
    normalized = answer.strip().lower()
    return normalized in PLACEHOLDER_ANSWERS

def extract_last_json_answer(text: str) -> str | None:
    for obj in reversed(extract_json_objects(text)):
        try:
            parsed = json.loads(obj)
        except json.JSONDecodeError:
            continue
        if "answer" not in parsed:
            continue
        answer = str(parsed.get("answer", "")).strip()
        if not is_placeholder_answer(answer):
            return answer
    return None

def cleanup_answer_text(text: str) -> str:
    answer = text.strip()
    answer = answer.strip().strip("`*_ ")
    answer = answer.strip("\"'“”‘’.:;,- ")
    return answer

def extract_labeled_answer(text: str) -> str | None:
    patterns = [
        r"^\s*(?:answer|antwoord|final answer|eindantwoord)\s*:\s*(.+?)\s*$",
        r"^\s*(?:het\s+antwoord\s+is|the\s+answer\s+is)\s*:?\s*(.+?)\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        for match in reversed(matches):
            answer = cleanup_answer_text(match)
            if answer:
                return answer
    return None

def strip_reasoning_blocks(text: str) -> str:
    without_blocks = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    without_blocks = re.sub(r"^\s*(?:analysis|reasoning)\b.*?(?:final|answer)\s*:?","", without_blocks, flags=re.DOTALL | re.IGNORECASE)
    return without_blocks.strip()

def cleanup_plain_answer(text: str) -> str:
    stripped = strip_reasoning_blocks(text)
    if not stripped:
        return ""
    first_line = stripped.splitlines()[0].strip()
    if first_line.lower().startswith(("<think>", "analysis", "reasoning")):
        return ""
    first_line = re.sub(
        r"^(antwoord|answer|final answer|eindantwoord)\s*:\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    )
    return cleanup_answer_text(first_line)

def parse_model_answer(text: str, answer_format: str) -> tuple[str, str | None]:
    final_text = text_after_final_reasoning_block(text)
    answer = extract_last_json_answer(final_text)
    if answer:
        return answer, None

    labeled_answer = extract_labeled_answer(final_text)
    if labeled_answer:
        return labeled_answer, "response was not valid JSON - used labeled answer fallback"

    answer = cleanup_plain_answer(final_text)
    if answer:
        if answer_format == "json":
            return answer, "response was not valid JSON - used first-line answer fallback"
        return answer, None
    return "", "response did not contain valid JSON with an answer key"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    gold = {row["id"]: row for row in load_jsonl(ROOT / "data/test.jsonl")}
    predictions = {}
    for row in load_jsonl(args.predictions):
        item_id = row["id"]
        if item_id not in gold or item_id in predictions:
            parser.error(f"Unknown or duplicate prediction ID: {item_id}")
        answer = row.get("answer")
        if answer is None:
            answer, _ = parse_model_answer(row.get("raw_response", ""), "json")
        if not isinstance(answer, str):
            parser.error(f"Answer must be a string: {item_id}")
        predictions[item_id] = answer.strip().lower()
    correct = sum(
        predictions.get(item_id, "") in {answer.strip().lower() for answer in row.get("accepted_answers", [row["answer"]])}
        for item_id, row in gold.items()
    )
    print(json.dumps({"correct": correct, "total": len(gold), "predictions": len(predictions),
                      "missing": len(gold) - len(predictions), "accuracy": correct / len(gold)}, indent=2))


if __name__ == "__main__":
    main()
