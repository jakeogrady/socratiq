import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import LoraConfig

# Use Apple GPU if available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load a small model for Mac-friendly training
model_name = "microsoft/phi-2"  # smaller than 3B
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16  # MPS supports fp16
)
model.to(device)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # ensure padding token

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Load dataset (can use a small subset for testing)
dataset = load_dataset("HuggingFaceTB/smoltalk2_everyday_convs_think", split="train[:200]")  # first 200 examples

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj"],  # typical attention projections
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# SFT training config
sft_config = SFTConfig(
    output_dir="./smollm_lora_test",
    per_device_train_batch_size=1,  # keep small for MPS
    learning_rate=5e-5,
    max_steps=100,  # quick test run
    logging_steps=10,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    peft_config=lora_config,
)

# Free memory before training
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Start training
trainer.train()
