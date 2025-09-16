from unsloth import FastLanguageModel
import torch
import json
from datasets import Dataset,DatasetDict
import pandas as pd
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import Trainer,TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np
import os
import config
from huggingface_hub import login
import wandb
import sys

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("All parametars not provided!")
        sys.exit(1)

    dataset_name = sys.argv[1]
    signal_type=sys.argv[2]
    num_tokens=int(sys.argv[3])
    
    
    huggingface_api_key=os.environ.get("HUGGINGFACE_API_KEY")
    wandb_api_key=os.environ.get("WANDB_API_KEY")
    
    if not huggingface_api_key or not wandb_api_key:
        print("API keys not set in environment variables")
        sys.exit(1)

    login(huggingface_api_key)
    wandb.login(key=wandb_api_key)

    # config.get_available_models() to get all models that are used in the project
    cfg=config.get_config(signal_type,dataset_name,"unsloth/Meta-Llama-3.1-8B-bnb-4bit",token_len=num_tokens,num_epochs=20,batch_size=32)

    with open(os.path.join(cfg["dataset"],"train.jsonl"), "r") as file:
        train_dataset = [json.loads(line) for line in file]

    with open(os.path.join(cfg["dataset"],"val.jsonl"), "r") as file:
        val_dataset = [json.loads(line) for line in file]

    with open(os.path.join(cfg["dataset"],"test.jsonl"), "r") as file:
        test_dataset = [json.loads(line) for line in file]


    train_dataset=Dataset.from_list(train_dataset)
    val_dataset=Dataset.from_list(val_dataset)
    test_dataset=Dataset.from_list(test_dataset)

    dataset=DatasetDict({
        "train":train_dataset,
        "val":val_dataset,
        "test":test_dataset
    })
    def rename_features(example):
        return {
            "text": example["prompt"],
            "target": cfg["mapping_function"](example)
        }

    dataset = dataset.map(rename_features, remove_columns=["prompt", "completion"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    col_to_delete = ['text']
    def preprocessing_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=cfg["token_len"])

    tokenized_datasets = dataset.map(preprocessing_function, batched=True, remove_columns=col_to_delete)
    tokenized_datasets = tokenized_datasets.rename_column("target", "label")
    tokenized_datasets.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    model =  AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=cfg["model"],
    num_labels=cfg["num_classes"],
    device_map="auto",
    offload_folder="offload",
    trust_remote_code=True
    )
    model.config.pad_token_id = model.config.eos_token_id

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
        ],
    )

    model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
    model = model.to("cuda")
    training_args = TrainingArguments(
        output_dir="lora-token-classification",
        learning_rate=cfg["lr"],
        lr_scheduler_type= "constant",
        warmup_ratio= 0.1,
        max_grad_norm= 0.3,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["num_epochs"],
        weight_decay=0.001,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        fp16=True,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
        compute_metrics=cfg["compute_metrics"]
    )

    trainer.train()

    results=trainer.evaluate(tokenized_datasets["test"])
    results["dataset_name"]=dataset_name
    print(results)
    print("\n")
    with open(f"results/{dataset_name}.json", 'w') as f:
            json.dump(results, f, indent=2)

    sys.exit(0)
