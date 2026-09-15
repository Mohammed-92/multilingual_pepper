# src/finetune_inkubalm.py
import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

def load_data(filepath):
    """Load prepared fine-tuning data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_example(example):
    """Format example for training."""
    if isinstance(example, dict):
        return f"### Instruction: {example['instruction']}\n### Input: {example['input']}\n### Response: {example['output']}"
    else:
        return str(example)

def finetune_inkubalm(lang_code, lang_name):
    """
    Fine-tune InkubaLM on the prepared data.
    """
    print(f"\n{'='*60}")
    print(f"FINE-TUNING INKUBALM FOR {lang_name}")
    print('='*60)
    
    # Load data
    data_file = f"data/finetune/{lang_code}.json"
    data = load_data(data_file)
    
    if not data:
        print(f"❌ No data loaded from {data_file}")
        return
    
    print(f"✅ Loaded {len(data)} examples")
    
    # Use a subset for faster training
    num_examples = min(5000, len(data))  # Reduce for faster training
    data = data[:num_examples]
    print(f"✅ Using {num_examples} examples")
    
    # Create dataset
    def format_examples(examples):
        texts = []
        for i in range(len(examples['instruction'])):
            ex = {
                'instruction': examples['instruction'][i],
                'input': examples['input'][i],
                'output': examples['output'][i]
            }
            texts.append(format_example(ex))
        return {'text': texts}
    
    # Create dataset from list
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_examples, batched=True, remove_columns=dataset.column_names)
    
    # Load model
    model_name = "lelapa/InkubaLM-0.4B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=False, max_length=256)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Prepare model for LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./inkubalm-finetuned-{lang_code}",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_steps=500,
        logging_steps=50,
        learning_rate=2e-4,
        warmup_steps=100,
        fp16=True,
        report_to="none",
        remove_unused_columns=False  # Important for causal LM
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save model
    model.save_pretrained(f"./inkubalm-finetuned-{lang_code}")
    tokenizer.save_pretrained(f"./inkubalm-finetuned-{lang_code}")
    print(f"✅ Model saved to ./inkubalm-finetuned-{lang_code}")
    
    return model, tokenizer

if __name__ == "__main__":
    languages = {
        "zu": "isiZulu",
        "nso": "Northern Sotho"
    }
    
    # Start with isiZulu first
    for lang_code, lang_name in languages.items():
        finetune_inkubalm(lang_code, lang_name)