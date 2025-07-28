from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os
from collections import Counter


# 1. Define file paths
data_files = {
    'train': './data_cat/data_cat/Train_ttc_T_output.json',
    'test': './data_cat/data_cat/Test_ttc_T_output.json'
}


# Verify files exist
for split, file_path in data_files.items():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")

# 2. Load the datasets
dataset = load_dataset('json', data_files=data_files)

# 3. Assign train and test datasets
train_dataset = dataset['train']
eval_dataset = dataset['test']



# 3. Count and print label distribution for train and test sets
train_labels = [example["output"] for example in train_dataset]
eval_labels = [example["output"] for example in eval_dataset]

train_label_counts = Counter(train_labels)
eval_label_counts = Counter(eval_labels)

print("训练集类别分布：")
for label, count in train_label_counts.items():
    print(f"{label}: {count}")

print("\n测试集类别分布：")
for label, count in eval_label_counts.items():
    print(f"{label}: {count}")

# 4. Save train and test datasets to JSON files
output_dir = ".data_cat/data_cat/split_datasets_svi"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
train_dataset.to_json(os.path.join(output_dir, "train_dataset.json"))
eval_dataset.to_json(os.path.join(output_dir, "test_dataset.json"))
print(f"\n训练集已保存至 {os.path.join(output_dir, 'train_dataset.json')}")
print(f"测试集已保存至 {os.path.join(output_dir, 'test_dataset.json')}")

# 5. Initialize the tokenizer
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 6. Preprocessing function
def preprocess_function(examples):
    max_length = 512  # Adjust based on memory
    label_map = {"Vigor": "Vigor", "Non": "Non"}  # Binary labels
    texts = [
        f"Input: {text} Output: {label_map[label]}" 
        for text, label in zip(examples["instruction"], examples["output"])  # Use instruction and output
    ]

    # Tokenize
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Create labels, ignoring loss on input part
    labels = tokenized["input_ids"].clone()
    for i, text in enumerate(texts):
        output_start = text.find("Output:") + len("Output:")
        tokenized_text = tokenizer(text, add_special_tokens=False)["input_ids"]
        output_start_token = len(tokenizer(text[:output_start], add_special_tokens=False)["input_ids"])
        labels[i, :output_start_token] = -100

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

# Apply preprocessing
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["instruction", "output"]  # Updated to match dataset columns
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["instruction", "output"]
)

# 7. Load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for PEFT training
model = prepare_model_for_kbit_training(model)

# Configure LoRA parameters
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA adapter
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 8. Set up data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# 9. Configure training arguments
training_args = TrainingArguments(
    output_dir="./qwen3_vigor_classifier",
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    num_train_epochs=100,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=50,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
    save_total_limit=40,
)

# 10. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 11. Start training
trainer.train()

# 12. Save model and tokenizer
model.save_pretrained("./qwen3_vigor_classifier_final")
tokenizer.save_pretrained("./qwen3_vigor_classifier_final")