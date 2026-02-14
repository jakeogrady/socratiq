import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import openai
from datasets import Dataset

from src.baseline_evaluation import validate_answer
from src.constants import ANSWER_REGEX, QUESTION_REGEX
from src.models import load_and_process_gsm8k
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_qa(text: str) -> tuple[str, str]:
    """Extract question and numeric answer from GSM8K example."""
    import re

    # Use DOTALL so question can be multi-line
    q_match = re.search(r"Question:\s*(.*?)\s*Answer:", text, re.DOTALL)
    a_match = re.search(r"####\s*(-?\d+)", text)

    if not q_match:
        logger.warning(f"Failed to extract question from text:\n{text[:200]}...")
    if not a_match:
        logger.warning(f"Failed to extract answer from text:\n{text[:200]}...")

    if not q_match or not a_match:
        raise ValueError("Could not extract question/answer")

    question = q_match.group(1).strip()
    answer = a_match.group(1).strip()
    return question, answer


def model_request_conversion(question: str, answer: str, max_retries=5) -> str:
    prompt = f"""
Rewrite the following question and answer in a Socratic step-by-step style.
Keep the final answer exactly the same. End the solution with "#### <final answer>".
Do NOT add extra solutions or commentary.

Original:
Q: {question}
A: {answer}
"""

    for attempt in range(1, max_retries + 1):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.6,
            )

            socratic_version = response.choices[0].message.content
            print(type(socratic_version), repr(socratic_version[:200]))

            match = re.search(ANSWER_REGEX, socratic_version)
            if match and validate_answer(match.group(1), answer):
                return socratic_version
            else:
                logger.warning(f"Attempt {attempt}: output missing or wrong final answer.")
        except Exception as e:
            logger.error(f"Attempt {attempt}: OpenAI API error: {e}")
        time.sleep(1)

    raise RuntimeError(f"Failed to convert question after {max_retries} attempts.")

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="gsm8k_socratic.jsonl")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Load dataset
    dataset = load_and_process_gsm8k()
    train_set: Dataset = dataset.train
    output_path = Path(args.output_file)

    # Resume or overwrite
    if args.overwrite:
        start_index = args.start_index
        mode = "w"
        logger.warning("Overwrite enabled — starting fresh.")
    else:
        resume_index = get_resume_index(output_path)
        start_index = max(args.start_index, resume_index)
        mode = "a"
        logger.info(f"Resuming from index {start_index}")

    logger.info(f"Processing examples {start_index} to {len(train_set) - 1}")

    with output_path.open(mode, encoding="utf-8") as f:
        for i in range(start_index, len(train_set)):
            raw_text = train_set[i]["text"]

            try:
                question, answer = extract_qa(raw_text)
            except Exception:
                logger.warning(f"Skipping index {i}: extraction failed")
                continue

            print(f"\nProcessing train_index: {i}")
            start_time = time.time()

            try:
                socratic_version = model_request_conversion(question, answer)
            except RuntimeError:
                logger.error(f"Skipping index {i}: failed to convert after retries")
                continue

            elapsed = time.time() - start_time
            print(f"Time taken: {elapsed:.2f} seconds")

            example = {
                "train_index": i,
                "original_question": question,
                "original_answer": answer,
                "socratic_rewrite": socratic_version,
            }

            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            f.flush()

            print("\nSocratic Rewrite:\n", socratic_version)
