import wandb

import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)

# Initialize wandb
wandb.init(
    project="LLM-Recipe",
    entity="sayannath235",
    name="Roberta:base-AgNews",
)

model_id = "FacebookAI/roberta-base"
dataset_id = "ag_news"
repository_id = "roberta-base-ag_news"

# Load dataset
dataset = load_dataset(dataset_id)

# Training and testing datasets
train_dataset = dataset['train']
test_dataset = dataset["test"].shard(num_shards=2, index=0)

# Validation dataset
val_dataset = dataset['test'].shard(num_shards=2, index=1)

# Preprocessing
tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

# This function tokenizes the input text using the RoBERTa tokenizer. 
# It applies padding and truncation to ensure that all sequences have the same length (256 tokens).
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

# Set dataset format
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# We will need this to directly output the class names when using the pipeline without mapping the labels later.
# Extract the number of classes and their names
num_labels = dataset['train'].features['label'].num_classes
class_names = dataset["train"].features["label"].names
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

# Create an id2label mapping
id2label = {i: label for i, label in enumerate(class_names)}

# Update the model's configuration with the id2label mapping
config = AutoConfig.from_pretrained(model_id)
config.update({"id2label": id2label})

model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)

# TrainingArguments
training_args = TrainingArguments(
    output_dir=repository_id,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",         
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="wandb",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()