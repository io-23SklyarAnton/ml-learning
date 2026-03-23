import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

model_name = "HuggingFaceTB/SmolLM2-135M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)

dataset = load_dataset(
    "text",
    data_files={"train": "../../datasets/Shakespeare_Dataset/alllines.txt"}
)


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

block_size = 256


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(group_texts, batched=True)

training_args = TrainingArguments(
    output_dir="./shakespeare_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    optim="adamw_torch",
    remove_unused_columns=False
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=lm_datasets["train"],
)

trainer.train()

peft_model.save_pretrained("./shakespeare_lora_final")
