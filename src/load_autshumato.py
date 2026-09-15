# src/load_autshumato.py
import os
import random
from typing import List, Tuple

def load_autshumato_pair(eng_file: str, target_file: str) -> List[Tuple[str, str]]:
    """
    Load an Autshumato bilingual corpus from two text files.
    Each file has one sentence per line, aligned by line number.
    """
    pairs = []
    
    if not os.path.exists(eng_file):
        print(f"❌ English file not found: {eng_file}")
        return pairs
    if not os.path.exists(target_file):
        print(f"❌ Target file not found: {target_file}")
        return pairs
    
    with open(eng_file, 'r', encoding='utf-8') as f_en:
        eng_lines = [line.strip() for line in f_en if line.strip()]
    
    with open(target_file, 'r', encoding='utf-8') as f_tgt:
        tgt_lines = [line.strip() for line in f_tgt if line.strip()]
    
    # Ensure both files have the same number of lines
    if len(eng_lines) != len(tgt_lines):
        print(f"⚠️ Warning: Files have different lengths!")
        print(f"   English: {len(eng_lines)} lines")
        print(f"   Target: {len(tgt_lines)} lines")
        print(f"   Taking the smaller length...")
    
    min_len = min(len(eng_lines), len(tgt_lines))
    pairs = list(zip(eng_lines[:min_len], tgt_lines[:min_len]))
    
    print(f"✅ Loaded {len(pairs)} pairs from {os.path.basename(eng_file)}")
    return pairs

def load_all_languages(data_dir: str):
    """
    Load all three language pairs from the Autshumato corpus.
    Uses your exact folder names.
    """
    print("=" * 50)
    print("LOADING AUTSHUMATO CORPUS")
    print("=" * 50)
    
    # Your actual folder names from the directory listing
    languages = {
        "af": {
            "folder": "Autshumato.BilingualCorpus(English-Afrikaans).v2.0",
            "code": "af"
        },
        "zu": {
            "folder": "Autshumato.BilingualCorpus(English-isiZulu).v2.0",
            "code": "zu"
        },
        "nso": {
            "folder": "Autshumato.BilingualCorpus(English-Sesotho).v2.0",
            "code": "nso"
        }
    }
    
    all_data = {}
    
    for lang_key, lang_info in languages.items():
        folder_name = lang_info["folder"]
        lang_code = lang_info["code"]
        
        print(f"\n📂 Loading {lang_key.upper()}...")
        
        # Try multiple naming patterns for the files
        folder_path = os.path.join(data_dir, folder_name)
        
        # Pattern 1: Full folder name as filename
        eng_file = os.path.join(folder_path, f"{folder_name}.en.txt")
        tgt_file = os.path.join(folder_path, f"{folder_name}.{lang_code}.txt")
        
        # Pattern 2: Simplified format (for Sesotho)
        if not os.path.exists(eng_file):
            # Try without .v2.0
            base_name = folder_name.replace(".v2.0", "")
            eng_file = os.path.join(folder_path, f"{base_name}.en.txt")
            tgt_file = os.path.join(folder_path, f"{base_name}.{lang_code}.txt")
        
        # Pattern 3: For Afrikaans (the one that worked)
        if not os.path.exists(eng_file):
            eng_file = os.path.join(folder_path, f"Autshumato.BilingualCorpus(English-{lang_code.upper()}).en.txt")
            tgt_file = os.path.join(folder_path, f"Autshumato.BilingualCorpus(English-{lang_code.upper()}).{lang_code}.txt")
        
        # Pattern 4: For isiZulu with v2.0 in filename
        if not os.path.exists(eng_file):
            eng_file = os.path.join(folder_path, f"Autshumato.BilingualCorpus(English-{lang_code.upper()}).v2.0.en.txt")
            tgt_file = os.path.join(folder_path, f"Autshumato.BilingualCorpus(English-{lang_code.upper()}).v2.0.{lang_code}.txt")
        
        if os.path.exists(eng_file) and os.path.exists(tgt_file):
            pairs = load_autshumato_pair(eng_file, tgt_file)
            if pairs:
                all_data[lang_key] = pairs
                print(f"   Sample: {pairs[0][0][:50]}... -> {pairs[0][1][:50]}...")
        else:
            print(f"❌ Could not find files for {lang_key}")
            print(f"   Tried: {eng_file}")
            print(f"   and: {tgt_file}")
    
    print("\n" + "=" * 50)
    print(f"✅ Loaded {len(all_data)} languages")
    return all_data

def split_data(pairs: List[Tuple[str, str]], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split data into train, validation, and test sets.
    """
    random.seed(42)  # For reproducibility
    random.shuffle(pairs)
    
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = pairs[:train_end]
    val = pairs[train_end:val_end]
    test = pairs[val_end:]
    
    print(f"   Split: {len(train)} train, {len(val)} val, {len(test)} test")
    return train, val, test

if __name__ == "__main__":
    # Set the path to your data folder
    DATA_PATH = r"C:\Users\user\Desktop\multilingual_pepper\venv\data\raw"
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data path not found: {DATA_PATH}")
        print("Please update DATA_PATH to your actual folder location.")
        exit()
    
    # Load all languages
    all_data = load_all_languages(DATA_PATH)
    
    # Create processed folder
    os.makedirs("data/processed", exist_ok=True)
    
    # Split each language pair
    for lang_code, pairs in all_data.items():
        print(f"\n📊 Splitting {lang_code.upper()} data:")
        train, val, test = split_data(pairs)
        
        # Save splits to files
        lang_dir = f"data/processed/{lang_code}"
        os.makedirs(lang_dir, exist_ok=True)
        
        with open(f"{lang_dir}/train.txt", 'w', encoding='utf-8') as f:
            for eng, tgt in train:
                f.write(f"{eng}\t{tgt}\n")
        
        with open(f"{lang_dir}/val.txt", 'w', encoding='utf-8') as f:
            for eng, tgt in val:
                f.write(f"{eng}\t{tgt}\n")
        
        with open(f"{lang_dir}/test.txt", 'w', encoding='utf-8') as f:
            for eng, tgt in test:
                f.write(f"{eng}\t{tgt}\n")
        
        print(f"   ✅ Saved splits to data/processed/{lang_code}/")
    
    print("\n" + "=" * 50)
    print("🎉 DATA LOADING COMPLETE!")