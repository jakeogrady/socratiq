import csv
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

RESULTS_CSV_HEADER = [
    "model",
    "dataset",
    "few_shot",
    "accuracy",
    "correct",
    "total",
    "date_last_modified",
]


def summarize_results(filename: str) -> dict:
    """Summarize results of model evaluation files."""
    path = Path(filename)
    if not path.exists():
        msg = f"{filename} not found"
        raise FileNotFoundError(msg)

    df = pd.read_csv(path)

    total = len(df)
    correct = int(df["is_correct"].sum())
    accuracy = round((correct / total) * 100, 2)

    model_name = path.stem.replace("_evaluation_results", "").replace("-", "/")

    return {
        "model": model_name,
        "dataset": "GSM8K",
        "few_shot": 4,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "date_last_modified": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
    }


if __name__ == "__main__":
    evaluation_file = "../mlx-community-Qwen3-0.6B-bf16_evaluation_results.csv"
    summary_file = "../evaluation_summary.csv"

    results = summarize_results(evaluation_file)

    file_exists = Path(summary_file).exists()

    with Path(summary_file).open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_HEADER, quoting=csv.QUOTE_ALL)

        if not file_exists:
            writer.writeheader()

        writer.writerow(results)
