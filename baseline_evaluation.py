import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from constants import QUESTION_PARSE_REGEX
from dataset import load_and_process_gsm8k
from train import Model


MODEL_NAME = "meta-llama/Llama-3.2-3B"


def load_model(model_name: str):
    print(f"[DEBUG] Loading model {model_name} in FP16 on CPU...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,
    )
    print(f"[DEBUG] Model loaded in {time.time() - start:.2f}s")
    wrapper = Model(name=model_name, model=model)

    if not hasattr(wrapper, "device"):
        wrapper.device = torch.device("cpu")
    return wrapper


def load_tokenizer(model_name: str):
    print(f"[DEBUG] Loading tokenizer {model_name}...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[DEBUG] Tokenizer loaded in {time.time() - start:.2f}s")
    return tokenizer

def generate_prompt(test_set, few_shot_num: int = 4, target_question_index: int = 1):
    few_shot_texts = test_set[:few_shot_num]["text"]
    few_shot_block = "\n\n".join(few_shot_texts)

    match = re.search(QUESTION_PARSE_REGEX, test_set[few_shot_num + target_question_index]["text"], re.DOTALL)
    target_question = match.group(1).strip()

    print(f"[DEBUG] Target Question: {target_question}")

    prompt = (
        few_shot_block
        + "\n\nQuestion: "
        + target_question
        + "\nAnswer:"
    )

    return prompt

def generate_response(model_wrapper, tokenizer, text_prompt, max_new_tokens=256):
    inputs = tokenizer(
        text_prompt, return_tensors="pt", truncation=True, max_length=1500
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


if __name__ == "__main__":
    print_answer = True
    print_prompt = False

    print("[DEBUG] Loading dataset...")
    start = time.time()
    dataset = load_and_process_gsm8k()
    print(f"[DEBUG] Dataset loaded in {time.time() - start:.2f}s")

    model_wrapper = load_model(MODEL_NAME)
    tokenizer = load_tokenizer(MODEL_NAME)

    for i in range(3):
        text_prompt = generate_prompt(dataset.test, few_shot_num=4, target_question_index=i)

        if print_prompt:
            print(f"[DEBUG] Prompt generated: {text_prompt}")

        print("[DEBUG] Generating response...")
        generation_start = time.time()
        response = generate_response(model_wrapper, tokenizer, text_prompt)

        if print_answer:
            print("\n===== Generated Response =====")
            print(response)
            print("==============================\n")

        print(f"[DEBUG] Response generated in {time.time() - generation_start:.2f}s")

        number_regex = r"####\s*(-?\d+)"

        match = re.search(number_regex, response)
        if match:
            answer = match.group(1)
            print(f"[DEBUG] Extracted Answer: {answer}")
        else:
            print("[DEBUG] No answer found in the response.")