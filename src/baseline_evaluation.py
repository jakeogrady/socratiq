import argparse
import logging
import re
import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import ANSWER_REGEX, MODEL_NAME, QUESTION_PARSE_REGEX
from dataset import GSM8KDataset, load_and_process_gsm8k
from src.constants import FEW_SHOT_NUM, TEST_CASES
from train import Model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_model(model_name: str) -> Model:
    """Load the specified model in FP16 on CPU."""
    logger.info("Loading model %s in FP16 on CPU...", model_name)
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,
    )
    logger.info("Model loaded in %ss", time.time() - start)
    wrapper = Model(name=model_name, model=model)

    if not hasattr(wrapper, "device"):
        wrapper.device = torch.device("cpu")
    return wrapper


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load the tokenizer for the specified model."""
    logger.info("Loading tokenizer %ss ...", model_name)
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Tokenizer loaded in %ss ...", time.time() - start)
    return tokenizer


def generate_prompt(
    test_set: Dataset, few_shot_num: int = 4, target_question_index: int = 1
) -> str:
    """Generate a few-shot prompt for the model."""
    few_shot_texts = test_set[:few_shot_num]["text"]
    few_shot_block = "\n\n".join(few_shot_texts)

    match = re.search(
        QUESTION_PARSE_REGEX,
        test_set[few_shot_num + target_question_index]["text"],
        re.DOTALL,
    )
    target_question = match.group(1).strip()

    logger.info("Target Question: %s", target_question)

    return few_shot_block + "\n\nQuestion: " + target_question + "\nAnswer:"


def generate_response(
    model_wrapper: Model,
    tokenizer: AutoTokenizer,
    text_prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate a response from the model given a text prompt."""
    inputs = tokenizer(
        text_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1500,
    )
    inputs = {k: v.to(model_wrapper.device) for k, v in inputs.items()}

    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model_wrapper.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    match = re.search(r"(####\s*-?\d+)", generated_text)
    if match:
        return generated_text[: match.end()].strip()
    return generated_text.strip()


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


def get_test_case_answer(dataset: GSM8KDataset, index: int) -> str | None:
    """Extract the correct answer from the dataset for a given test case index."""
    answer_match = re.search(ANSWER_REGEX, dataset.test[index]["text"])
    return answer_match.group(1).strip() if answer_match else None


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

    model_wrapper = load_model(args.model_name)
    tokenizer = load_tokenizer(args.model_name)

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
        response = generate_response(model_wrapper, tokenizer, text_prompt)

        if args.print_answer:
            logger.info("\n===== Generated Response =====")
            logger.info(response)
            logger.info("==============================\n")

        logger.info("Response generated in %ss", time.time() - generation_start)

        match = re.search(ANSWER_REGEX, response)
        if match:
            answer = match.group(1)
            logger.info("Extracted Answer: %s", answer)
            correct_answer = get_test_case_answer(dataset, i + args.few_shot_num)
            if validate_answer(answer, correct_answer):
                answer_correct += 1
                logger.info("Answer is correct!")
        else:
            logger.info("No answer found in the response.")

    logger.info("Total Correct Answers: %d out of %d", answer_correct, args.test_cases)
    accuracy = (answer_correct / args.test_cases) * 100
    logger.info("Accuracy: %.2f%%", accuracy)
