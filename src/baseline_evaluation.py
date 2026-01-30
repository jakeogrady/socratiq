import argparse
import csv
import logging
import re
import time
from pathlib import Path

from mlx_lm import generate, load

from src.constants import (
    ANSWER_REGEX,
    FEW_SHOT_NUM,
    MISTRAL_7B_Q4,
    TEST_CASES,
)
from src.models import generate_prompt, load_and_process_gsm8k

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CSV_COLUMNS = [
    "test_index",
    "correct_answer",
    "generated_answer",
    "is_correct",
    "raw_response",
]


def validate_answer(generated_answer: str, correct_answer: str) -> bool:
    """Validate if the generated answer matches the correct answer."""
    try:
        logger.info(
            "Validating Generated Answer: %s against Correct Answer: %s",
            generated_answer,
            correct_answer,
        )
        return int(generated_answer) == int(correct_answer)
    except Exception:  # noqa: BLE001
        return False


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the evaluation script."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, default=MISTRAL_7B_Q4, help="Model name or path"
    )
    parser.add_argument(
        "--test_cases",
        type=int,
        default=TEST_CASES,
        help="Number of test cases to evaluate",
    )
    parser.add_argument(
        "--few_shot_num",
        type=int,
        default=FEW_SHOT_NUM,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start evaluation from this test case index (0-based)",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="evaluation_results.csv",
        help="File to append results to (CSV)",
    )
    parser.add_argument(
        "--print_answer",
        action="store_true",
        help="Whether to print the generated answers",
    )
    parser.add_argument(
        "--print_prompt", action="store_true", help="Whether to print the prompts"
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    logger.info("Model Name: %s", args.model_name)
    model, tokenizer = load(args.model_name)

    answer_correct = 0

    dataset = load_and_process_gsm8k()

    start = args.start_index
    end = start + args.test_cases
    eval_filename = f"{args.model_name.replace('/', '-')}_{args.results_file}"

    logger.info("Evaluating test cases from %d to %d", start, end - 1)

    for i in range(start, end):
        text_prompt = generate_prompt(
            dataset.test,
            few_shot_num=args.few_shot_num,
            target_question_index=i,
        )

        if args.print_prompt:
            logger.info("Prompt generated: %s", text_prompt)

        logger.info("Generating response for text index %s ...", i)
        generation_start = time.time()

        response = generate(
            model,
            tokenizer,
            prompt=text_prompt,
            max_tokens=256,
        )

        if args.print_answer:
            logger.info("\n===== Generated Response =====")
            logger.info(response)
            logger.info("==============================\n")

        logger.info("Response generated in %ss", time.time() - generation_start)

        match = re.search(ANSWER_REGEX, response)
        extracted_answer = ""
        is_correct = False

        if match:
            extracted_answer = match.group(1)
            logger.info("Extracted Answer: %s", extracted_answer)
            correct_answer = dataset.get_test_case_answer(i + args.few_shot_num)

            if validate_answer(extracted_answer, correct_answer):
                answer_correct += 1
                is_correct = True
                logger.info("Answer is correct!")
        else:
            logger.info("No answer found in the response.")
            correct_answer = dataset.get_test_case_answer(i + args.few_shot_num)

        with Path(eval_filename).open("a", newline="", encoding="utf-8") as f:
            if (
                not Path(eval_filename).exists()
                or Path(eval_filename).stat().st_size == 0
            ):
                writer = csv.DictWriter(
                    f,
                    fieldnames=CSV_COLUMNS,
                    quoting=csv.QUOTE_ALL,
                )
                writer.writeheader()

            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
            writer.writerow(
                {
                    "test_index": i,
                    "correct_answer": correct_answer,
                    "generated_answer": extracted_answer
                    if extracted_answer
                    else "No Answer",
                    "is_correct": is_correct,
                    "raw_response": response,
                }
            )

    logger.info("Total Correct Answers: %d out of %d", answer_correct, args.test_cases)
    accuracy = (answer_correct / args.test_cases) * 100
    logger.info("Accuracy: %.2f%%", accuracy)
