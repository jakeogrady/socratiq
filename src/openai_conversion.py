import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import openai
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI
from openai.types import Batch

from src.constants import ANSWER_REGEX
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
    a_match = re.search(ANSWER_REGEX, text)

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


def immediate_prompt() -> None:
    """Prompt GPT5-Mini with immediate to improve wording."""
    prompt = """
    Developer: # Role and Objective
    - Serve as an expert Socratic tutor, transforming math problems and their solutions into a series of clear, step-by-step Socratic questions.

    # Instructions
    - Begin with a concise checklist (3-7 bullets) outlining the conceptual breakdown of the problem before drafting the Socratic questions; keep items high-level and not implementation-specific.
    - Guide learners only through questions, not direct answers.
    - Do not perform or verify the final answer; always assume it is correct.
    - Decompose the solution into micro-steps, each prompted by a question.
    - Encourage learner reflection with periodic prompts to check reasoning, such as "Does this make sense?" or "Why does this step work?"
    - Ensure each question follows logically from the previous one with no gaps or skipped steps.
    - Maintain a neutral tone throughout: avoid instructions, commentary, or evaluative language like "obviously" or "clearly."
    - Reproduce the original answer at the end in the prescribed format: `#### <final answer>`
    - Avoid verbosity: do not include extraneous explanations, derivations, or text outside what is required for reasoning at each step.
    - When a step involves a calculation, include the operation in parentheses after the question.
    - Set reasoning_effort = low: guide the decomposition but minimize unnecessary internal computation.

    # Output Format
    - Present the initial checklist, followed by each step as a numbered Socratic question,
     including any associated calculation in parentheses.
    - End with the original final answer in the exact format: `#### <original final answer>`

    # Example Format
    Checklist:
    - Identify quantities given
    - Determine operation to combine values
    - Calculate result after subtraction
    - Check answer alignment with problem statement
    1) Question prompting the first step (calculation)
    2) Question prompting the next step (calculation)
    ...
    N) Synthesis or check question (calculation)
    #### <original final answer>

    # Example
    Checklist:
    - Find the total quantity
    - Decide what is being removed
    - Calculate how many are left
    - Assess if final value is consistent
    1) What is the total number of apples? (3+2)
    2) How many are left after giving some away? (5-2)
    3) Does this total make sense compared to the problem?
    #### 3

    """

    question = (
        "Natalia sold clips to 48 of her friends in April,"
        " and then she sold half as many clips in May."
        " How many clips did Natalia sell altogether in April and May?"
    )
    answer = (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May. "
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May."
        "#### 72"
    )

    user_input = (
        f"Problem:\n{question}\n\nSolution:\n{answer}\n"
        f"Convert the solution into Socratic questions."
    )

    response = openai.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
        reasoning={"effort": "low"},
        max_output_tokens=700,
    )

    logger.info("Full response:")
    logger.info(response.model_dump())

    with Path("test.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(response.output_text) + "\n")
        logger.info("Response Text %s", response.output_text)


def batch_prompt() -> None:
    """Prompt GPT5-Mini with batching to improve wording."""
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


if __name__ == "__main__":
    immediate_prompt()
