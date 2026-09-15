# src/prepare_finetune_data.py
import os
import json

def prepare_finetune_data(lang_code, lang_name):
    """
    Prepare fine-tuning data in instruction format.
    """
    train_file = f"data/processed/{lang_code}/train.txt"
    
    if not os.path.exists(train_file):
        print(f"❌ Train file not found: {train_file}")
        return
    
    examples = []
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                english, target = parts
                # Format as instruction
                example = {
                    "instruction": f"Translate the following English sentence to {lang_name}.",
                    "input": english,
                    "output": target
                }
                examples.append(example)
    
    # Save as JSON
    output_file = f"data/finetune/{lang_code}.json"
    os.makedirs("data/finetune", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(examples)} examples to {output_file}")
    return output_file

if __name__ == "__main__":
    languages = {
        "zu": "isiZulu",
        "nso": "Northern Sotho"
    }
    
    for lang_code, lang_name in languages.items():
        prepare_finetune_data(lang_code, lang_name)