# src/evaluate_nllb.py
# Purpose: Evaluate NLLB-200 on Autshumato test sets for isiZulu and Northern Sotho
# Author: Mohammed Gori
# Date: August 2026
# This script loads NLLB-200, translates 100 test sentences per language,
# and computes BLEU scores using sacrebleu.


import os
import sacrebleu
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_nllb():
    """Load NLLB-200 model."""
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

def load_test_data(lang_code: str):
    """Load test data from processed folder."""
    test_file = f"data/processed/{lang_code}/test.txt"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return [], []
    
    english_sentences = []
    reference_sentences = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                english_sentences.append(parts[0])
                reference_sentences.append(parts[1])
    
    print(f"✅ Loaded {len(english_sentences)} test sentences for {lang_code}")
    return english_sentences, reference_sentences

def evaluate_language(lang_code, lang_name, tokenizer, model, nllb_lang_code, num_samples=200):
    """
    Evaluate BLEU for one language using NLLB.
    """
    print("\n" + "=" * 60)
    print(f"EVALUATING {lang_name} ({lang_code.upper()}) with NLLB")
    print("=" * 60)
    
    # Load test data
    english, references = load_test_data(lang_code)
    
    if not english:
        print(f"❌ No test data for {lang_code}")
        return None
    
    # Limit samples if specified
    if num_samples and num_samples < len(english):
        english = english[:num_samples]
        references = references[:num_samples]
        print(f"   Using first {num_samples} sentences for faster evaluation")
    
    # Translate each sentence
    print(f"   Translating {len(english)} sentences...")
    translations = []
    
    for i, sentence in enumerate(english):
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(english)}")
        
        translation = translate_nllb(tokenizer, model, sentence, "eng_Latn", nllb_lang_code)
        translations.append(translation)
    
    # Compute BLEU
    print(f"   Computing BLEU score...")
    bleu = sacrebleu.corpus_bleu(translations, [references])
    
    print(f"\n📊 Results for {lang_name}:")
    print(f"   BLEU Score: {bleu.score:.2f}")
    print(f"   Sentences evaluated: {len(translations)}")
    
    # Show some examples
    print(f"\n   Example translations:")
    for i in range(min(3, len(translations))):
        print(f"   {i+1}. English: {english[i][:60]}...")
        print(f"      Reference: {references[i][:60]}...")
        print(f"      Generated: {translations[i][:60]}...")
        print()
    
    return bleu.score

if __name__ == "__main__":
    print("=" * 60)
    print("BLEU EVALUATION - NLLB BASELINE")
    print("=" * 60)
    
    # Load the model
    print("\n📂 Loading NLLB model...")
    tokenizer, model = load_nllb()
    
    if tokenizer is None or model is None:
        print("❌ Model not loaded. Exiting.")
        exit()
    
    # Language mapping for NLLB
    languages = {
        "af": {"name": "Afrikaans", "code": "afr_Latn"},
        "zu": {"name": "isiZulu", "code": "zul_Latn"},
        "nso": {"name": "Northern Sotho", "code": "nso_Latn"}
    }
    
    # Evaluate each language
    results = {}
    
    for lang_code, lang_info in languages.items():
        score = evaluate_language(
            lang_code, 
            lang_info["name"], 
            tokenizer, 
            model, 
            lang_info["code"],
            num_samples=200  # Change to None for full evaluation
        )
        if score is not None:
            results[lang_code] = score
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - BLEU SCORES (NLLB)")
    print("=" * 60)
    for lang_code, lang_info in languages.items():
        if lang_code in results:
            print(f"   {lang_info['name']}: {results[lang_code]:.2f}")
        else:
            print(f"   {lang_info['name']}: ❌ Not evaluated")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")