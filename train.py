from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

from constants import PHI_2
from dataset import load_and_process_gsm8k


class Model(BaseModel):
    name: str
    model: PreTrainedModel
    device: torch.device | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, /, **data: Any):
        super().__init__(**data)

        # Use Apple GPU if available
        # self.device = (
        #     torch.device("mps")
        #     if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        #     else torch.device("cpu")
        # )
        self.device = torch.device("cpu")
        self.model.to(self.device)
        torch.set_num_threads(4)

        print(f"Using device: {self.device}")

    def enable_gradient_checkpointing(self):
        self.model.gradient_checkpointing_enable()

    def generate_response(self, inputs: Any, tokenizer):
        return self.model.generate(**inputs, max_new_tokens=256)


class Tokenizer(BaseModel):
    name: str
    model: PreTrainedTokenizer

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def print_chat_template(self):
        print(self.model.chat_template)


def finetune():
    # Load a small model for Mac-friendly training
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    phi_2 = Model(name=model_name, model=model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_and_process_gsm8k()
    train_dataset = dataset.train
    eval_dataset = dataset.validation

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # SFT training config
    sft_config = SFTConfig(
        output_dir="./smollm_lora_test",
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        max_steps=100,
        logging_steps=10,
        gradient_checkpointing=False,
        bf16=False,
        fp16=False,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=phi_2.model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        peft_config=lora_config,
    )

    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    trainer.train()


if __name__ == "__main__":
    # finetune()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import time

    torch.set_num_threads(4)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    start = time.time()
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded!", time.time() - start)

    start = time.time()
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=None
    )
    print("Model loaded!", time.time() - start)

    start = time.time()
    print("Moving model to CPU...")
    model = model.to("cpu")
    print("Model on CPU!", time.time() - start)

    start = time.time()
    prompt = "Hello from my Mac!"
    print("Tokenizing input...")
    inputs = tokenizer(prompt, return_tensors="pt")
    print("Tokenized!", time.time() - start)

    start = time.time()
    print("Generating output...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64)
    print("Generation done!", time.time() - start)

    print(tokenizer.decode(output[0], skip_special_tokens=True))
