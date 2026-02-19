import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI
from openai.types import Batch

from src.models import load_and_process_gsm8k

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SYSTEM_PROMPT = """You convert worked solutions into Socratic questions.
Each question must include the calculation for that step.
Generate 3-5 concise questions.
Do not add explanations outside the questions.
"""


def extract_qa(text: str) -> tuple[str, str]:
    """Extract Question-Answer pair from Dataset Text."""
    q_match = re.search(r"Question:\s*(.*?)\s*Answer:", text, re.DOTALL)
    a_match = re.search(r"####\s*(-?\d+)", text)

    if not q_match or not a_match:
        msg = "Could not extract question/answer"
        raise ValueError(msg)

    return q_match.group(1).strip(), a_match.group(1).strip()


def build_batch_file(dataset: Dataset, output_jsonl: Path) -> None:
    """Build batch file from GSM8K Dataset."""
    logger.info("Building batch JSONL file...")

    with output_jsonl.open("w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            raw_text = dataset[i]["text"]
            try:
                question, answer = extract_qa(raw_text)
            except Exception:
                logger.exception("Could not extract Q-A pair from %s", raw_text)

            user_prompt = f"""Problem:
                {question}

                Solution:
                {answer}

                Convert the solution into Socratic questions.
                """

            entry = {
                "custom_id": f"gsm8k_{i}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "gpt-5-mini",
                    "input": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_output_tokens": 400,
                },
            }

            f.write(json.dumps(entry) + "\n")

    logger.info("Batch file created.")


def submit_batch(batch_file: Path) -> str:
    """Submit branch to OpenAI."""
    logger.info("Uploading batch file...")

    file = client.files.create(file=batch_file.open("rb"), purpose="batch")

    logger.info("Creating batch job...")

    batch = client.batches.create(
        input_file_id=file.id, endpoint="/v1/responses", completion_window="24h"
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
            return batch

        time.sleep(15)


def download_results(output_file_id: str, save_path: Path) -> None:
    """Download results of conversion."""
    logger.info("Downloading results...")

    content = client.files.content(output_file_id)
    text = content.text

    save_path.write_text(text, encoding="utf-8")
    logger.info("Results saved to %s", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", type=str, default="gsm8k_socratic_results.jsonl"
    )
    args = parser.parse_args()

    dataset = load_and_process_gsm8k()
    train_set: Dataset = dataset.train

    batch_input_path = Path("batch_input.jsonl")

    build_batch_file(train_set, batch_input_path)

    batch_id = submit_batch(batch_input_path)

    batch = wait_for_batch(batch_id)

    if batch.status == "completed":
        download_results(batch.output_file_id, Path(args.output_file))
    else:
        logger.error("Batch failed.")
