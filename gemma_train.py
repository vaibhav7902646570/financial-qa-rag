from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from datasets import load_dataset
import torch

# === Load Gemma 13B tokenizer and model (4-bit QLoRA) ===
model_id = "google/gemma-13b"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# === Prepare model for QLoRA training ===
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# === Load unstructured raw text dataset ===
data_path = "/users/sgvanil/Test_Data-20250711T161830Z-1-001/Test_Data/*.txt"
dataset = load_dataset("text", data_files={"train": data_path})

# === Tokenization ===
def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./gemma13b_finetuned",
    per_device_train_batch_size=1,  # Gemma 13B is huge — keep this low
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=250,
    save_total_limit=2,
    fp16=True,
    bf16=False,  # set to True if your GPU supports BF16
    gradient_checkpointing=True,
    optim="paged_adamw_8bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer
)

trainer.train()

# === Save fine-tuned model ===
model.save_pretrained("./gemma13b_finetuned_model")
tokenizer.save_pretrained("./gemma13b_finetuned_model")
