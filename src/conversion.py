import argparse
import json
import logging
import re
import time
from pathlib import Path

from datasets import Dataset
from mlx_lm import generate, load

from src.baseline_evaluation import validate_answer
from src.constants import (
    ANSWER_REGEX,
    QUESTION_REGEX,
    MISTRAL_7B_Q4,
    DATASET_CONVERSION_PROMPT,
)
from src.models import load_and_process_gsm8k

# -----------------------------
# Setup logging
# -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Constants
# -----------------------------
MAX_ATTEMPTS = 3
ANSWER_MARKER_REGEX = r"####\s*(\d+)"

# -----------------------------
# Extract question + answer
# -----------------------------
def extract_qa(text: str) -> tuple[str, str]:
    """Extracts question and numeric answer from a GSM8K example."""
    q_match = re.search(r"Question:\s*(.*?)\s*Answer:", text, re.DOTALL)
    a_match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text, re.DOTALL)
    if not q_match or not a_match:
        raise ValueError("Could not extract question/answer")
    return q_match.group(1).strip(), a_match.group(1).strip()

# -----------------------------
# Model request
# -----------------------------
def model_request_conversion(model, tokenizer, task: str) -> str:
    """Convert a question+answer into a Socratic step-by-step version."""
    prompt = DATASET_CONVERSION_PROMPT.format(task=task)
    return generate(model, tokenizer, prompt=prompt, max_tokens=500).strip()

# -----------------------------
# Resume logic
# -----------------------------
def get_resume_index(output_path: Path) -> int:
    """Return the next train index to process if the file exists."""
    if not output_path.exists():
        return 0

    last_index = -1
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                last_index = max(last_index, record.get("train_index", -1))
            except Exception:
                continue
    return last_index + 1

# -----------------------------
# Main
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="train.jsonl")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = load_and_process_gsm8k()
    train_set: Dataset = dataset.train
    output_path = Path(args.output_file)

    # Determine start index
    if args.overwrite:
        start_index = args.start_index
        mode = "w"
        logger.warning("Overwrite enabled — starting fresh.")
    else:
        resume_index = get_resume_index(output_path)
        start_index = max(args.start_index, resume_index)
        mode = "a"
        logger.info("Resuming from index %d", start_index)

    # Load teacher model
    logger.info("Loading teacher model...")
    model, tokenizer = load(MISTRAL_7B_Q4)

    logger.info("Processing examples %d to %d", start_index, len(train_set) - 1)

    # Open output file
    with output_path.open(mode, encoding="utf-8") as f:
        for i in range(start_index, len(train_set)):
            raw_text = train_set[i]["text"]

            try:
                question, answer = extract_qa(raw_text)
                logger.info(f"Question {i}: {question}")
            except Exception:
                logger.warning("Skipping index %d: extraction failed", i)
                continue

            task = f"Question: {question}\nAnswer: {answer}"

            success = False
            start_time = time.time()

            socratic_version = model_request_conversion(model, tokenizer, task)
            match = re.search(ANSWER_MARKER_REGEX, socratic_version)
            extracted_answer = match.group(1) if match else None

            if extracted_answer and validate_answer(extracted_answer, answer):
                success = True
                logger.info("Successfully converted example %d", i)

            if not success:
                logger.error("Conversion failed for index %d: original answer=%s", i, answer)

            elapsed = time.time() - start_time

            # Save result
            example = {
                "train_index": i,
                "original_question": question,
                "original_answer": answer,
                "socratic_rewrite": socratic_version,
                "success": success,
                "time_taken": elapsed
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            f.flush()

            logger.info(
                "Processed index %d in %.2f seconds, success=%s", i, elapsed, success
            )
