"""Prepare the headline dataset for Stage 1."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random


DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCE = DATA_DIR.parent / "Sarcasm_Headlines_Dataset_v2.json"
SPLIT_DIR = DATA_DIR / "splits"
SEED = 42
TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
CLASS_NAMES = {0: "non-sarcastic", 1: "sarcastic"}


def load_headlines(path):
    """Read one JSON record per line and keep the text and its label."""
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not {
                "headline", "is_sarcastic", "article_link"
            }.issubset(record):
                raise ValueError(f"Missing expected fields on line {line_number}")
            headline = record["headline"]
            label = record["is_sarcastic"]
            if not isinstance(headline, str) or not headline.strip():
                raise ValueError(f"Invalid headline on line {line_number}")
            if type(label) is not int or label not in CLASS_NAMES:
                raise ValueError(f"Invalid sarcasm label on line {line_number}")
            rows.append({"headline": headline, "is_sarcastic": label})
    return rows


def deduplicate(rows):
    """Keep the first copy of each headline and drop conflicting label groups."""
    groups = defaultdict(list)
    for row in rows:
        key = " ".join(row["headline"].casefold().split())
        groups[key].append(row)

    unique = []
    conflicting_rows = 0
    for copies in groups.values():
        labels = {row["is_sarcastic"] for row in copies}
        if len(labels) == 1:
            unique.append(copies[0])
        else:
            conflicting_rows += len(copies)
    return unique, conflicting_rows


def stratified_split(rows):
    """Shuffle and split each class separately to preserve its proportion."""
    rng = Random(SEED)
    splits = {"train": [], "validation": [], "test": []}
    for label in CLASS_NAMES:
        class_rows = [row for row in rows if row["is_sarcastic"] == label]
        rng.shuffle(class_rows)
        train_end = round(len(class_rows) * TRAIN_FRACTION)
        validation_end = train_end + round(len(class_rows) * VALIDATION_FRACTION)
        splits["train"].extend(class_rows[:train_end])
        splits["validation"].extend(class_rows[train_end:validation_end])
        splits["test"].extend(class_rows[validation_end:])
    for split in splits.values():
        rng.shuffle(split)
    return splits


def print_counts(name, rows):
    counts = Counter(row["is_sarcastic"] for row in rows)
    print(f"{name}: {len(rows):,} total | "
          f"{counts[0]:,} non-sarcastic | {counts[1]:,} sarcastic")


def print_examples(training_rows):
    rng = Random(SEED)
    for label, name in CLASS_NAMES.items():
        headlines = [row["headline"] for row in training_rows
                     if row["is_sarcastic"] == label]
        examples = rng.sample(headlines, min(20, len(headlines)))
        print(f"\n{name.capitalize()} training examples")
        for number, headline in enumerate(examples, start=1):
            print(f"{number:2}. {headline}")


def main():
    rows = load_headlines(SOURCE)
    print_counts("Raw data", rows)
    unique_rows, conflicting_rows = deduplicate(rows)
    print_counts("After deduplication", unique_rows)
    print(f"Removed {len(rows) - len(unique_rows):,} rows, "
          f"including {conflicting_rows:,} rows with conflicting labels.")

    splits = stratified_split(unique_rows)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        path = SPLIT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for row in split:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        print_counts(name.capitalize(), split)

    print_examples(splits["train"])
    print(f"\nSaved splits to {SPLIT_DIR}")


if __name__ == "__main__":
    main()
