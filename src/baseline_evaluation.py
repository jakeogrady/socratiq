import argparse
import logging
import re
import time

from datasets import Dataset

from constants import ANSWER_REGEX, MODEL_NAME, QUESTION_REGEX
from dataset import load_and_process_gsm8k
from src.constants import FEW_SHOT_NUM, TEST_CASES
from train import Model, Tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_prompt(
    test_set: Dataset, few_shot_num: int = 4, target_question_index: int = 1
) -> str:
    """Generate a few-shot prompt for the model."""
    few_shot_texts = test_set[:few_shot_num]["text"]
    few_shot_block = "\n\n".join(few_shot_texts)

    match = re.search(
        QUESTION_REGEX,
        test_set[few_shot_num + target_question_index]["text"],
        re.DOTALL,
    )
    target_question = match.group(1).strip()

    logger.info("Target Question: %s", target_question)

    return few_shot_block + "\n\nQuestion: " + target_question + "\nAnswer:"


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, default=MODEL_NAME, help="Model name or path"
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
        "--print_answer",
        action="store_true",
        help="Whether to print the generated answers",
    )
    parser.add_argument(
        "--print_prompt", action="store_true", help="Whether to print the prompts"
    )
    args = parser.parse_args()

    answer_correct = 0

    logger.info("Loading dataset...")
    start = time.time()
    dataset = load_and_process_gsm8k()
    logger.info("Dataset loaded in %ss", time.time() - start)

    model = Model.create_model(name=args.model_name)
    tokenizer = Tokenizer.load_tokenizer(args.model_name)

    for i in range(args.test_cases):
        text_prompt = generate_prompt(
            dataset.test,
            few_shot_num=args.few_shot_num,
            target_question_index=i,
        )

        if args.print_prompt:
            logger.info("Prompt generated: %s", text_prompt)

        logger.info("Generating response...")
        generation_start = time.time()
        response = model.generate_response(tokenizer, text_prompt)

        if args.print_answer:
            logger.info("\n===== Generated Response =====")
            logger.info(response)
            logger.info("==============================\n")

        logger.info("Response generated in %ss", time.time() - generation_start)

        match = re.search(ANSWER_REGEX, response)
        if match:
            answer = match.group(1)
            logger.info("Extracted Answer: %s", answer)
            correct_answer = dataset.get_test_case_answer(i + args.few_shot_num)
            if validate_answer(answer, correct_answer):
                answer_correct += 1
                logger.info("Answer is correct!")
        else:
            logger.info("No answer found in the response.")

    logger.info("Total Correct Answers: %d out of %d", answer_correct, args.test_cases)
    accuracy = (answer_correct / args.test_cases) * 100
    logger.info("Accuracy: %.2f%%", accuracy)
