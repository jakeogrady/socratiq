import json
import logging
import re
from pathlib import Path

INPUT_DIR = Path("new_data")
OUTPUT_DIR = Path("new_data_clean")
SPLITS: list[str] = ["train", "valid", "test"]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def clean_answer(answer: str) -> str:
    """Remove trailing rhetorical questions appearing before the answer delimiter."""
    answer = answer.strip()

    answer = re.sub(
        r"\s+[A-Z][^#]{0,200}?\?\s*(?:[A-Z][^#]{0,100}?\.)?\s*(####)", r" \1", answer
    )

    answer = re.sub(r"\s+[A-Z][^#.!]*\?[^#]*?(####)", r" \1", answer)

    return answer.strip()


def format_example(question: str, answer: str) -> str:
    """Format a question and cleaned answer into a single training text string."""
    question = question.strip()
    answer = clean_answer(answer.strip())
    return f"Question: {question}\nAnswer: {answer}"


def convert_split(split: str) -> tuple[int, int, int]:
    """Convert a single dataset split, returning counts of converted, skipped and cleaned examples."""
    input_path = INPUT_DIR / f"{split}.jsonl"
    output_path = OUTPUT_DIR / f"{split}.jsonl"

    if not input_path.exists():
        logger.info("  Skipping %s — %s not found", split, input_path)
        return 0, 0, 0

    converted: int = 0
    skipped: int = 0
    cleaned: int = 0

    with (
        input_path.open("r", encoding="utf-8") as file_in,
        output_path.open("w", encoding="utf-8") as file_out,
    ):
        for line_index, line in enumerate(file_in):
            if not line.strip():
                continue

            try:
                example: dict = json.loads(line)
            except json.JSONDecodeError as error:
                logger.info("  Skipping line %d in %s: %s", line_index, split, error)
                skipped += 1
                continue

            question: str = example.get("question", "").strip()
            answer: str = example.get("answer", "").strip()

            if not question or not answer:
                skipped += 1
                continue

            cleaned_answer: str = clean_answer(answer)
            if cleaned_answer != answer:
                cleaned += 1

            text: str = f"Question: {question}\nAnswer: {cleaned_answer}"
            json.dump({"text": text}, file_out)
            file_out.write("\n")
            converted += 1

    return converted, skipped, cleaned


def verify_sample(num_examples: int = 5) -> None:
    """Log the first num_examples entries from the converted training split."""
    train_path = OUTPUT_DIR / "train.jsonl"
    if not train_path.exists():
        return

    logger.info("\n--- Sample converted examples ---")
    with train_path.open("r", encoding="utf-8") as train_file:
        for example_index, line in enumerate(train_file):
            if example_index >= num_examples:
                break
            example: dict = json.loads(line)
            logger.info("\n[Example %d]", example_index)
            logger.info("%s", example["text"])
            logger.info("-" * 60)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info("Converting %s -> %s\n", INPUT_DIR, OUTPUT_DIR)

    total_cleaned: int = 0
    for split in SPLITS:
        num_converted, num_skipped, num_cleaned = convert_split(split)
        total_cleaned += num_cleaned
        logger.info(
            "  %s: %d converted, %d skipped, %d trailing questions removed",
            split,
            num_converted,
            num_skipped,
            num_cleaned,
        )

    logger.info("\nTotal trailing questions removed: %d", total_cleaned)

    verify_sample(num_examples=5)

    logger.info("\nDone. Update your config:")
    logger.info('  data: "new_data_clean"')
    logger.info("  # remove prompt_feature and completion_feature")
