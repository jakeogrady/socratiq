import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import LoraConfig

from constants import PHI_2
from dataset import load_and_process_gsm8k, GSM8KDataset

if __name__ == "__main__":
    # Use Apple GPU if available
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Load a small model for Mac-friendly training
    model = AutoModelForCausalLM.from_pretrained(
        PHI_2,
        torch_dtype=torch.float16,
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(PHI_2)
    tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    dataset = GSM8KDataset(**load_and_process_gsm8k())
    train_dataset = dataset.train

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
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
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=sft_config,
        peft_config=lora_config,
    )

    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    trainer.train()
