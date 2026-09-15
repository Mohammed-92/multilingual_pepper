# src/load_autshumato_simple.py
import os
import random

def load_autshumato_files(data_dir: str):
    """
    Load Autshumato files using your exact filenames.
    """
    print("=" * 60)
    print("LOADING AUTSHUMATO CORPUS (SIMPLE VERSION)")
    print("=" * 60)
    
    # Your exact file paths based on the directory listing
    lang_configs = {
        "af": {
            "eng_file": "Autshumato.BilingualCorpus(English-Afrikaans).v2.0/Autshumato.BilingualCorpus(English-Afrikaans).v2.0.en.txt",
            "tgt_file": "Autshumato.BilingualCorpus(English-Afrikaans).v2.0/Autshumato.BilingualCorpus(English-Afrikaans).v2.0.af.txt"
        },
        "zu": {
            "eng_file": "Autshumato.BilingualCorpus(English-isiZulu).v2.0/Autshumato.BilingualCorpus(English-Xitsonga).v2.0.en.txt",
            "tgt_file": "Autshumato.BilingualCorpus(English-isiZulu).v2.0/Autshumato.BilingualCorpus(English-isiZulu).v2.0.zu.txt"
        },
        "nso": {
            "eng_file": "Autshumato.BilingualCorpus(English-Sesotho).v2.0/Autshumato.BilingualCorpus(English-Sesotho).v2.0.en.txt",
            "tgt_file": "Autshumato.BilingualCorpus(English-Sesotho).v2.0/Autshumato.BilingualCorpus(English-Sesotho).v2.0.st.txt"
        }
    }
    
    all_data = {}
    
    for lang_code, paths in lang_configs.items():
        print(f"\n📂 Loading {lang_code.upper()}...")
        
        eng_full = os.path.join(data_dir, paths["eng_file"])
        tgt_full = os.path.join(data_dir, paths["tgt_file"])
        
        if not os.path.exists(eng_full):
            print(f"   ❌ English file not found: {eng_full}")
            continue
        if not os.path.exists(tgt_full):
            print(f"   ❌ Target file not found: {tgt_full}")
            continue
        
        # Read files
        with open(eng_full, 'r', encoding='utf-8') as f:
            eng_lines = [line.strip() for line in f if line.strip()]
        
        with open(tgt_full, 'r', encoding='utf-8') as f:
            tgt_lines = [line.strip() for line in f if line.strip()]
        
        # Pair them up
        min_len = min(len(eng_lines), len(tgt_lines))
        pairs = list(zip(eng_lines[:min_len], tgt_lines[:min_len]))
        
        print(f"   ✅ Loaded {len(pairs)} pairs")
        print(f"   Sample: {pairs[0][0][:50]}...")
        print(f"         -> {pairs[0][1][:50]}...")
        
        all_data[lang_code] = pairs
    
    print("\n" + "=" * 60)
    print(f"✅ Loaded {len(all_data)} languages: {list(all_data.keys())}")
    return all_data

def split_and_save(pairs, lang_code, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split and save data."""
    random.seed(42)
    random.shuffle(pairs)
    
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = pairs[:train_end]
    val = pairs[train_end:val_end]
    test = pairs[val_end:]
    
    # Create directory
    os.makedirs(f"data/processed/{lang_code}", exist_ok=True)
    
    # Save files
    with open(f"data/processed/{lang_code}/train.txt", 'w', encoding='utf-8') as f:
        for eng, tgt in train:
            f.write(f"{eng}\t{tgt}\n")
    
    with open(f"data/processed/{lang_code}/val.txt", 'w', encoding='utf-8') as f:
        for eng, tgt in val:
            f.write(f"{eng}\t{tgt}\n")
    
    with open(f"data/processed/{lang_code}/test.txt", 'w', encoding='utf-8') as f:
        for eng, tgt in test:
            f.write(f"{eng}\t{tgt}\n")
    
    print(f"   ✅ Saved: {len(train)} train, {len(val)} val, {len(test)} test")
    return train, val, test

if __name__ == "__main__":
    DATA_PATH = r"C:\Users\user\Desktop\multilingual_pepper\venv\data\raw"
    
    # Load all data
    all_data = load_autshumato_files(DATA_PATH)
    
    # Split and save each language
    print("\n" + "=" * 60)
    print("SPLITTING AND SAVING DATA")
    print("=" * 60)
    
    for lang_code, pairs in all_data.items():
        print(f"\n📊 {lang_code.upper()}:")
        split_and_save(pairs, lang_code)
    
    print("\n" + "=" * 60)
    print("✅ ALL DONE!")
    print("   Files saved in: data/processed/")
    print("   - af/train.txt, val.txt, test.txt")
    print("   - zu/train.txt, val.txt, test.txt")
    print("   - nso/train.txt, val.txt, test.txt")
