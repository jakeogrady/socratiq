import json
import re
from pathlib import Path

INPUT_DIR = Path("new_data")
OUTPUT_DIR = Path("new_data_clean")
SPLITS = ["train", "valid", "test"]


def clean_answer(answer: str) -> str:
    """
    Remove trailing filler sentences before the #### marker.
    Targets:
    - Questions: "Can you check...?"
    - Rhetorical confirmations: "Does X? Yes, it does."
    - Any sentence ending in ? or similar filler immediately before ####
    """
    # Remove any sentence(s) between the last calculation and ####
    # Pattern: one or more sentences that aren't calculations, right before ####
    answer = answer.strip()

    # Remove patterns like "Can you verify X?" or "Does X? Yes." before ####
    answer = re.sub(
        r'\s+[A-Z][^#]{0,200}?\?\s*(?:[A-Z][^#]{0,100}?\.)?\s*(####)',
        r' \1',
        answer
    )

    # Remove any remaining standalone question sentences before ####
    answer = re.sub(
        r'\s+[A-Z][^#.!]*\?[^#]*?(####)',
        r' \1',
        answer
    )

    return answer.strip()


def format_example(question: str, answer: str) -> str:
    question = question.strip()
    answer = clean_answer(answer.strip())
    return f"Question: {question}\nAnswer: {answer}"


def convert_split(split: str) -> tuple:
    input_path = INPUT_DIR / f"{split}.jsonl"
    output_path = OUTPUT_DIR / f"{split}.jsonl"

    if not input_path.exists():
        print(f"  Skipping {split} — {input_path} not found")
        return 0, 0, 0

    converted = 0
    skipped = 0
    cleaned = 0

    with input_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8") as f_out:

        for i, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Skipping line {i} in {split}: {e}")
                skipped += 1
                continue

            question = example.get("question", "").strip()
            answer = example.get("answer", "").strip()

            if not question or not answer:
                skipped += 1
                continue

            cleaned_answer = clean_answer(answer)
            if cleaned_answer != answer:
                cleaned += 1

            text = f"Question: {question}\nAnswer: {cleaned_answer}"
            json.dump({"text": text}, f_out)
            f_out.write("\n")
            converted += 1

    return converted, skipped, cleaned


def verify_sample(n: int = 5) -> None:
    train_path = OUTPUT_DIR / "train.jsonl"
    if not train_path.exists():
        return

    print("\n--- Sample converted examples ---")
    with train_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            example = json.loads(line)
            print(f"\n[Example {i}]")
            print(example["text"])
            print("-" * 60)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Converting {INPUT_DIR} -> {OUTPUT_DIR}\n")

    total_cleaned = 0
    for split in SPLITS:
        converted, skipped, cleaned = convert_split(split)
        total_cleaned += cleaned
        print(f"  {split}: {converted} converted, {skipped} skipped, {cleaned} trailing questions removed")

    print(f"\nTotal trailing questions removed: {total_cleaned}")

    verify_sample(n=5)

    print("\nDone. Update your config:")
    print('  data: "new_data_clean"')
    print("  # remove prompt_feature and completion_feature")