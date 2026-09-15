# src/generate_user_study_sentences.py
import os
import random
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================================================================
# NLLB Model Loading (Built-in)
# =========================================================================

def load_nllb():
    """Load the NLLB-200 model."""
    model_name = "facebook/nllb-200-distilled-600M"
    
    print(f"Loading NLLB from {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("✅ NLLB loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading NLLB: {e}")
        return None, None

def translate_nllb(tokenizer, model, text, src_lang="eng_Latn", tgt_lang="zul_Latn"):
    """Translate using NLLB."""
    if tokenizer is None or model is None:
        return "Model not loaded"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=200
        )
    
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translation

# =========================================================================
# Main Script
# =========================================================================

def load_test_data(lang_code: str):
    """Load test data from processed folder."""
    test_file = f"data/processed/{lang_code}/test.txt"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return []
    
    pairs = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    
    print(f"✅ Loaded {len(pairs)} English-{lang_code} pairs")
    return pairs

def generate_user_study_sentences():
    """
    Generate 50 sentences for the user study.
    """
    print("=" * 60)
    print("GENERATING USER STUDY SENTENCES (NLLB)")
    print("=" * 60)
    
    # Load NLLB
    print("\n📂 Loading NLLB model...")
    tokenizer, model = load_nllb()
    
    if tokenizer is None or model is None:
        print("❌ Model not loaded. Exiting.")
        return
    
    # Load isiZulu test data
    print("\n📂 Loading isiZulu test data...")
    pairs = load_test_data("zu")
    
    if not pairs:
        print("❌ No test data found. Exiting.")
        return
    
    # Select 50 random sentences
    random.seed(42)  # For reproducibility
    selected = random.sample(pairs, min(50, len(pairs)))
    print(f"✅ Selected {len(selected)} random sentences")
    
    # Translate each using NLLB
    print("\n🔄 Generating translations...")
    results = []
    
    for i, (eng, ref) in enumerate(selected, 1):
        translation = translate_nllb(tokenizer, model, eng, "eng_Latn", "zul_Latn")
        results.append({
            "number": i,
            "english": eng,
            "reference": ref,
            "nllb_translation": translation
        })
        
        # Show progress
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(selected)}")
    
    print(f"✅ Completed {len(results)} translations")
    
    # Save to CSV
    output_file = "user_study_sentences.csv"
    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["number", "english", "reference", "nllb_translation"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Saved to {output_file}")
    
    # Show some examples
    print("\n📝 Example translations:")
    for i in range(min(3, len(results))):
        print(f"\n{i+1}. English: {results[i]['english'][:80]}...")
        print(f"   Reference: {results[i]['reference'][:80]}...")
        print(f"   NLLB:      {results[i]['nllb_translation'][:80]}...")
    
    print("\n" + "=" * 60)
    print("✅ User study sentences generated successfully!")
    print(f"   File saved: {output_file}")
    print(f"   Total sentences: {len(results)}")
    print("\nNext steps:")
    print("   1. Open the CSV file")
    print("   2. Create a Google Form with the NLLB translations")
    print("   3. Recruit 3 native isiZulu speakers to rate them")
    print("=" * 60)

if __name__ == "__main__":
    generate_user_study_sentences()