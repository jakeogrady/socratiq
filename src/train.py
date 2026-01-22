import logging
import re
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from pydantic import BaseModel, ConfigDict, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import SFTConfig, SFTTrainer

from src.constants import MODEL_NAME, QUESTION_REGEX
from src.dataset import load_and_process_gsm8k

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Model(BaseModel):
    """Model wrapper class."""

    name: str
    model: PreTrainedModel
    device: torch.device | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, /, **data: dict) -> None:
        super().__init__(**data)

        self.device = torch.device("cpu")
        self.model.to(self.device)
        torch.set_num_threads(4)

        logger.info("Using device: %s", self.device)

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory-efficient training."""
        self.model.gradient_checkpointing_enable()

    @staticmethod
    def create_model(name: str = MODEL_NAME) -> "Model":
        """Create a Model instance."""
        logger.info("Loading model %s in FP16 on CPU...", name)
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16,
            device_map=None,
        )
        logger.info("Model loaded in %ss", time.time() - start)
        wrapper = Model(name=name, model=model)

        if not hasattr(wrapper, "device"):
            wrapper.device = torch.device("cpu")
        return wrapper

    def generate_prompt(
        self, test_set: Dataset, few_shot_num: int = 4, target_question_index: int = 1
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

    def generate_response(
        self,
        tokenizer: AutoTokenizer,
        text_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
    ) -> str:
        """Generate a response from the model given a text prompt."""
        inputs = tokenizer(
            text_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1500,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        match = re.search(r"(####\s*-?\d+)", generated_text)
        if match:
            return generated_text[: match.end()].strip()
        return generated_text.strip()


class Tokenizer(BaseModel):
    """Tokenizer wrapper class."""

    name: str
    model: PreTrainedTokenizer

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def print_chat_template(self) -> None:
        """Print the chat template of the tokenizer, if available."""
        logger.info(self.model.chat_template)

    @staticmethod
    def load_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
        """Load the tokenizer for the specified model."""
        logger.info("Loading tokenizer %ss ...", model_name)
        start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        logger.info("Tokenizer loaded in %ss ...", time.time() - start)
        return tokenizer


def finetune() -> None:
    """Load a small model for Mac-friendly training."""
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
    torch.set_num_threads(4)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    start = time.time()
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer loaded: %s", time.time() - start)

    start = time.time()
    logger.info("Loading model...")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,
    )
    logger.info("Model loaded: %s", time.time() - start)

    start = time.time()
    logger.info("Moving model to CPU...")
    model.to("cpu")
    logger.info("Model on CPU: %s", time.time() - start)

    start = time.time()
    prompt = "Hello from my Mac!"
    logger.info("Tokenizing input...")
    inputs = tokenizer(prompt, return_tensors="pt")
    logger.info("Tokenized: %s", time.time() - start)

    start = time.time()
    logger.info("Generating output...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64)
    logger.info("Generation done! %s", time.time() - start)

    logger.info(tokenizer.decode(output[0], skip_special_tokens=True))
