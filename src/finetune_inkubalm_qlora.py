# src/finetune_inkubalm_qlora.py
import os
import json
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
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
    return f"### Instruction: {example['instruction']}\n### Input: {example['input']}\n### Response: {example['output']}"

def finetune_inkubalm_qlora(lang_code, lang_name):
    """
    Fine-tune InkubaLM using QLoRA (4-bit quantisation).
    """
    print(f"\n{'='*60}")
    print(f"FINE-TUNING INKUBALM FOR {lang_name} (QLoRA)")
    print('='*60)
    
    # Load data
    data_file = f"data/finetune/{lang_code}.json"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please run prepare_finetune_data.py first.")
        return
    
    data = load_data(data_file)
    print(f"✅ Loaded {len(data)} examples")
    
    # Use a subset for faster training (adjust based on time)
    num_examples = min(10000, len(data))
    data = data[:num_examples]
    print(f"✅ Using {num_examples} examples for training")
    
    # Create dataset
    dataset = Dataset.from_list(data)
    
    # Quantisation config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
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
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize dataset
    def tokenize_function(examples):
        texts = [format_example(ex) for ex in examples]
        return tokenizer(texts, truncation=True, padding=False, max_length=256)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./inkubalm-finetuned-{lang_code}",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=500,
        logging_steps=50,
        learning_rate=2e-4,
        warmup_steps=100,
        fp16=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
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
    
    for lang_code, lang_name in languages.items():
        finetune_inkubalm_qlora(lang_code, lang_name)