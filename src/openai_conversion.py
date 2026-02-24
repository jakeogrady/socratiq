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


def build_batch_file(
    output_jsonl: Path,
    limit: int = 8000,
    max_tokens: int = MAX_CONVERSION_OUTPUT_TOKENS,
) -> None:
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
                    "max_output_tokens": max_tokens,
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


def batch_prompt(
    batch_input_path: str,
    output_file: str,
) -> None:
    """Submit batching to OpenAI."""
    batch_input_path = Path(batch_input_path)

    if not batch_input_path.exists():
        msg = "Batch input file not found"
        raise ValueError(msg)

    batch_id = submit_batch(batch_input_path)
    batch = wait_for_batch(batch_id)

    if batch.status != "completed":
        logger.error("Batch failed with status: %s", batch.status)
        return

    if batch.output_file_id:
        download_results(batch.output_file_id, Path(output_file))
    else:
        logger.error("No output_file_id found")


def rerun_truncated_requests() -> None:
    """Rerun previous requests that were truncated."""
    failed_ids = []

    with Path("gsm8k_socratic_results.jsonl").open() as f:
        for line in f:
            item = json.loads(line)

            response = item.get("response", {})
            body = response.get("body", {})

            if body.get("status") == "incomplete":
                reason = body.get("incomplete_details", {}).get("reason")

                if reason == "max_output_tokens":
                    failed_ids.append(item["custom_id"])

    logger.info("Found %d truncated samples", len(failed_ids))

    rerun_items = []

    if not Path("batch_input.jsonl").exists():
        msg = "Original batch_input.jsonl not found"
        raise ValueError(msg)

    new_max_tokens = 2000

    with Path("batch_input.jsonl").open() as f:
        for line in f:
            item = json.loads(line)

            if item.get("custom_id") in failed_ids:
                item["body"]["max_output_tokens"] = new_max_tokens

                rerun_items.append(item)

    logger.info("Preparing %d rerun requests", len(rerun_items))

    rerun_path = Path("rerun_requests.jsonl")

    with rerun_path.open("w") as f:
        for item in rerun_items:
            f.write(json.dumps(item) + "\n")

    if len(rerun_items) == 0:
        logger.warning("No rerun items found")
        return

    logger.info("Submitting rerun batch...")

    batch_prompt(
        batch_input_path=str(rerun_path),
        output_file="reran_socratic_results.jsonl",
    )


def load_completed(path: str) -> list[dict]:
    """Find lines in a .jsonl file that have been completed."""
    completed = []
    with Path(path).open() as f:
        for line in f:
            obj = json.loads(line)
            if obj["response"]["body"].get("status") == "completed":
                completed.append(obj)
    return completed


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--output_file", type=str, default="gsm8k_socratic_results.jsonl"
    # )
    # args = parser.parse_args()
    #
    # batch_prompt(output_file=args.output_file)
    # rerun_truncated_requests()

    data1 = load_completed("gsm8k_socratic_results.jsonl")
    data2 = load_completed("reran_socratic_results.jsonl")

    final = data1 + data2

    with Path("socratic_results_final.jsonl").open("w") as f:
        f.writelines(json.dumps(obj) + "\n" for obj in final)
