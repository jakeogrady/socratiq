import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from constants import QUESTION_PARSE_REGEX
from dataset import load_and_process_gsm8k
from train import Model


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def load_model(model_name: str):
    print(f"[DEBUG] Loading model {model_name} in FP32 on CPU...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,
    )
    print(f"[DEBUG] Model loaded in {time.time() - start:.2f}s")
    return Model(name=model_name, model=model)


def load_tokenizer(model_name: str):
    print(f"[DEBUG] Loading tokenizer {model_name}...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[DEBUG] Tokenizer loaded in {time.time() - start:.2f}s")
    return tokenizer

def generate_prompt(test_set, few_shot_num: int = 8):
    few_shot_texts = test_set[:few_shot_num]["text"]
    system_content = "".join(
        f"{ex}" for ex in few_shot_texts
    )

    # Extract target question
    match = re.search(
        QUESTION_PARSE_REGEX, test_set[few_shot_num + 1]["text"], re.DOTALL
    )
    if not match:
        raise Exception("Failed to extract target question for CoT prompt.")

    target_question_text = match.group(1).strip()
    user_content = f"Question: {target_question_text}\n"

    return "Here are some example problems and solutions:\n\n " + system_content + "=====\nAnswer the following question\n" + user_content

def generate_response(model_wrapper, tokenizer, text_prompt, max_new_tokens=256):
    inputs = tokenizer(
        text_prompt, return_tensors="pt", truncation=True, max_length=512
    )
    inputs = {k: v.to(model_wrapper.device) for k, v in inputs.items()}

    print("[DEBUG] Generating response...")
    start = time.time()
    with torch.no_grad():
        outputs = model_wrapper.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    print(f"[DEBUG] Generation completed in {time.time() - start:.2f}s")

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "===" in decoded:
        assistant_reply = decoded.split("===", 1)[1].strip()
    else:
        assistant_reply = decoded.strip()

    return assistant_reply

if __name__ == "__main__":

    print("[DEBUG] Loading dataset...")
    start = time.time()
    dataset = load_and_process_gsm8k()
    print(f"[DEBUG] Dataset loaded in {time.time() - start:.2f}s")

    model_wrapper = load_model(MODEL_NAME)
    tokenizer = load_tokenizer(MODEL_NAME)

    user_prompt = generate_prompt(dataset.test, few_shot_num=3)

    messages = [
        {
            "role": "system",
            "content": f"You are a helpful maths tutor. Answer the question step-by-step"
                       f" and finish with ### <number>.\n",
        },
        {"role": "user", "content": f"{user_prompt}"},
    ]

    print(f"[DEBUG] Prompt generated: {user_prompt}")

    text_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    print(f"[DEBUG] Text prompt generated: {text_prompt}")

    response = generate_response(model_wrapper, tokenizer, text_prompt)
    print("\n===== Generated Response =====")
    print(response)
    print("==============================\n")
