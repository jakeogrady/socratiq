import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from openai.types import Batch

from src.constants import (
    ANSWER_REGEX,
    DATASET_CONVERSION_PROMPT2,
    MAX_CONVERSION_OUTPUT_TOKENS,
)
from src.models import load_and_process_gsm8k

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

INPUT_FILE = "qa_pairs.jsonl"
TRAIN_OUT = "new_data/train.jsonl"
VAL_OUT = "new_data/valid.jsonl"

MIN_ANSWER_CHARS = 120
MAX_ANSWER_CHARS = 2000
NGRAM_N = 5
SIM_THRESHOLD = 0.85
VAL_SPLIT = 0.10
SEED = 42

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
                        {"role": "system", "content": DATASET_CONVERSION_PROMPT2},
                        {
                            "role": "user",
                            "content": f"Problem:\n{question}\n\nSolution:\n{answer}\nConvert the solution into Socratic question-solution pairs.",
                        },
                    ],
                    "max_output_tokens": max_tokens,
                    "reasoning": {"effort": "low"},
                },
            }

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

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


def load_completed(path: str) -> list[dict]:
    """Find lines in a .jsonl file that have been completed."""
    completed = []
    with Path(path).open() as f:
        for line in f:
            obj = json.loads(line)
            if obj["response"]["body"].get("status") == "completed":
                completed.append(obj)
    return completed


def chunk_dataset(dataset, chunk_size=1000, start_index: int = 0):
    """Yield dataset indices in chunks."""
    for i in range(start_index, len(dataset.train), chunk_size):
        yield range(i, min(i + chunk_size, len(dataset.train)))


from pathlib import Path


def merge_files():
    path = Path("socratic_results")

    if not path.exists():
        print("Results directory not found")
        return

    chunk_files = sorted(path.glob("*.jsonl"))

    output_file = Path("merged_results.jsonl")

    with output_file.open("w", encoding="utf-8") as out_f:
        for chunk_file in chunk_files:
            print("Processing", chunk_file)

            with chunk_file.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        obj = json.loads(line)

                        # Only keep successful responses
                        if (
                            "response" in obj
                            and obj["response"]["body"].get("status") == "completed"
                        ):
                            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

                    except json.JSONDecodeError:
                        continue

    print("Merged dataset written to", output_file)


def extract_qa_pairs(input_file="merged_results.jsonl",
                     output_file="qa_pairs.jsonl"):

    input_path = Path(input_file)
    output_path = Path(output_file)

    qa_pairs = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)

                # Extract generated text
                text_output = ""
                body = obj.get("response", {}).get("body", {})

                if "output" in body:
                    for item in body["output"]:
                        if item.get("type") == "message":
                            for content in item.get("content", []):
                                if content.get("type") == "output_text":
                                    text_output = content.get("text", "")

                if not text_output:
                    continue

                # Split into individual QA blocks
                blocks = text_output.split("<|endofsolution|>")

                for block in blocks:
                    block = block.strip()
                    if not block:
                        continue

                    # Extract question
                    q_match = re.search(
                        r"Question:\s*(.*?)\s*Solution:",
                        block,
                        re.DOTALL
                    )

                    # Extract solution (everything after "Solution:")
                    s_match = re.search(
                        r"Solution:\s*(.*)",
                        block,
                        re.DOTALL
                    )

                    if q_match and s_match:
                        question = q_match.group(1).strip()
                        solution = s_match.group(1).strip()

                        qa_pairs.append({
                            "question": question,
                            "answer": solution
                        })

            except Exception:
                continue

    # Write JSONL output
    with output_path.open("w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved {len(qa_pairs)} QA pairs → {output_file}")


def clean_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def valid_numeric_answer(answer):
    match = re.search(r"####\s*(-?\d+)", answer)
    return match is not None


def generate_ngrams(text, n):
    tokens = clean_text(text).split()
    return set(
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    )


def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0
    return len(set1 & set2) / len(set1 | set2)

def filter_and_deduplicate(data):

    filtered = []
    seen_questions = set()
    ngram_index = []

    for item in data:
        q = item["question"].strip()
        a = item["answer"].strip()

        if not q or not a:
            continue

        if not valid_numeric_answer(a):
            continue

        if len(a) < MIN_ANSWER_CHARS or len(a) > MAX_ANSWER_CHARS:
            continue

        q_clean = clean_text(q)
        if q_clean in seen_questions:
            continue

        q_ngrams = generate_ngrams(q, NGRAM_N)

        seen_questions.add(q_clean)
        ngram_index.append(q_ngrams)

        filtered.append(item)

    return filtered


def train_val_split(data):
    random.seed(SEED)
    random.shuffle(data)

    split_idx = int(len(data) * (1 - VAL_SPLIT))

    train = data[:split_idx]
    val = data[split_idx:]

    return train, val

def create_split_files():

    print("Loading data...")
    data = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    print(f"Original samples: {len(data)}")

    print("Filtering + deduplicating...")
    data = filter_and_deduplicate(data)

    print(f"After filtering: {len(data)}")

    print("Splitting train/validation...")
    train, val = train_val_split(data)

    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")

    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(VAL_OUT, "w", encoding="utf-8") as f:
        for item in val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Done.")

def append_gsm8k_main_to_jsonl(output_path="qa_pairs.jsonl"):
    """
    Loads the main train split of GSM8K from Hugging Face
    and appends it to a JSONL file in question/answer format.
    """

    dataset = load_dataset("gsm8k", "main", split="train")

    with open(output_path, "a", encoding="utf-8") as f:
        for example in dataset:
            record = {
                "question": example["question"].strip(),
                "answer": example["answer"].strip(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Appended {len(dataset)} GSM8K main examples to {output_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--output_file", type=str, default="gsm8k_socratic_results11.jsonl"
    # )
    # parser.add_argument(
    #     "--start_index",
    #     type=int,
    #     default=2000,
    #     help="Resume batching from dataset index",
    # )
    # args = parser.parse_args()
    #
    # dataset = load_and_process_gsm8k()
    #
    # logger.info("Starting sequential batching...")
    #
    # for chunk_id, chunk_indices in enumerate(
    #     chunk_dataset(dataset, chunk_size=1000, start_index=args.start_index)
    # ):
    #     print(chunk_indices)
    #     logger.info("Processing chunk %d", chunk_id)
    #
    #     chunk_file = Path(f"batch_input_chunk_{chunk_id}.jsonl")
    #
    #     with chunk_file.open("w", encoding="utf-8") as f:
    #         for i in chunk_indices:
    #             raw_text = dataset.train[i]["text"]
    #
    #             try:
    #                 question, answer = extract_qa(raw_text)
    #             except Exception:
    #                 logger.exception("Skipping entry %d", i)
    #                 continue
    #
    #             entry = {
    #                 "custom_id": f"gsm8k_{i}",
    #                 "method": "POST",
    #                 "url": "/v1/responses",
    #                 "body": {
    #                     "model": "gpt-5-mini-2025-08-07",
    #                     "input": [
    #                         {"role": "system", "content": DATASET_CONVERSION_PROMPT2},
    #                         {
    #                             "role": "user",
    #                             "content": f"Problem:\n{question}\n\nSolution:\n{answer}\nConvert the solution into Socratic question-solution pairs.",
    #                         },
    #                     ],
    #                     "max_output_tokens": MAX_CONVERSION_OUTPUT_TOKENS,
    #                     "reasoning": {"effort": "low"},
    #                 },
    #             }
    #
    #             f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    #
    #     # Submit chunk batch
    #     logger.info("Submitting chunk batch...")
    #
    #     batch_id = submit_batch(chunk_file)
    #     batch = wait_for_batch(batch_id)
    #
    #     if batch.status != "completed":
    #         logger.error("Batch failed. Stopping pipeline.")
    #         break
    #
    #     if batch.output_file_id:
    #         download_results(
    #             batch.output_file_id,
    #             Path(f"socratic_results/{args.output_file}_{chunk_id}.jsonl"),
    #         )
    #
    #     logger.info("Chunk %d finished.", chunk_id)
    #     time.sleep(10)

    # merge_files()
    # extract_qa_pairs()
    append_gsm8k_main_to_jsonl()
    create_split_files()
