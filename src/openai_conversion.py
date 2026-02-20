import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types import Batch

from src.constants import (
    ANSWER_REGEX,
    DATASET_CONVERSION_PROMPT,
    MAX_CONVERSION_OUTPUT_TOKENS,
)
from src.models import load_and_process_gsm8k

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_qa(text: str) -> tuple[str, str]:
    """Extract Question-Answer pair from Dataset Text."""
    q_match = re.search(r"Question:\s*(.*?)\s*Answer:", text, re.DOTALL)
    a_match = re.search(ANSWER_REGEX, text)

    if not q_match or not a_match:
        msg = "Could not extract question/answer"
        raise ValueError(msg)

    return q_match.group(1).strip(), a_match.group(1).strip()


def build_batch_file(output_jsonl: Path, limit: int = 50) -> None:
    """Build batch file from GSM8K Dataset (for testing, limit entries)."""
    logger.info("Building batch JSONL file...")

    dataset = load_and_process_gsm8k()

    with output_jsonl.open("w", encoding="utf-8") as f:
        for i in range(min(limit, len(dataset.train))):
            raw_text = dataset.train[i]["text"]
            try:
                question, answer = extract_qa(raw_text)
            except Exception:
                logger.exception("Could not extract Q-A pair from entry %d", i)
                continue

            entry = {
                "custom_id": f"gsm8k_{i}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "gpt-5-mini-2025-08-07",
                    "input": [
                        {"role": "system", "content": DATASET_CONVERSION_PROMPT},
                        {
                            "role": "user",
                            "content": f"Problem:\n{question}\n\nSolution:\n{answer}\nConvert the solution into Socratic questions.",
                        },
                    ],
                    "max_output_tokens": MAX_CONVERSION_OUTPUT_TOKENS,
                    "reasoning": {"effort": "low"},
                },
            }

            f.write(json.dumps(entry) + "\n")

    logger.info("Batch file created at %s", output_jsonl)


def submit_batch(batch_file: Path) -> str:
    """Submit batch to OpenAI."""
    logger.info("Uploading batch file...")

    file = client.files.create(file=batch_file.open("rb"), purpose="batch")

    logger.info("Creating batch job...")

    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )

    logger.info("Batch submitted: %s", batch.id)
    return batch.id


def wait_for_batch(batch_id: str) -> Batch | None:
    """Wait for batch to return."""
    logger.info("Waiting for batch completion...")

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        logger.info("Status: %s", status)

        if status in ["completed", "failed", "cancelled"]:
            logger.info("Status achieved %s", status)
            return batch

        time.sleep(15)


def download_results(output_file_id: str, save_path: Path) -> None:
    """Download results of conversion."""
    logger.info("Downloading results...")
    content = client.files.content(output_file_id)
    save_path.write_text(content.text, encoding="utf-8")
    logger.info("Results saved to %s", save_path)


def batch_prompt(output_file: str = "gsm8k_socratic_results.jsonl") -> None:
    """Full batch pipeline: build, submit, wait, download."""
    batch_input_path = Path("batch_input.jsonl")
    build_batch_file(batch_input_path, limit=50)

    batch_id = submit_batch(batch_input_path)
    batch = wait_for_batch(batch_id)

    logger.info("Batch %s", batch)

    if batch.status != "completed":
        logger.error("Batch failed with status: %s", batch.status)
        return

    if batch.output_file_id:
        download_results(batch.output_file_id, Path(output_file))
    else:
        logger.error("No output_file_id found in batch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", type=str, default="gsm8k_socratic_results.jsonl"
    )
    args = parser.parse_args()

    batch_prompt(output_file=args.output_file)
