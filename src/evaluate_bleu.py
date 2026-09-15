# src/evaluate_bleu.py
import os
import sacrebleu
from load_m2m100 import load_m2m100, translate_m2m100

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

def evaluate_language(lang_code, lang_name, tokenizer, model, num_samples=None):
    """
    Evaluate BLEU for one language.
    If num_samples is set, only evaluate that many sentences (for faster testing).
    """
    print("\n" + "=" * 60)
    print(f"EVALUATING {lang_name} ({lang_code.upper()})")
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
        
        translation = translate_m2m100(tokenizer, model, sentence, "en", lang_code)
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
    print("BLEU EVALUATION - M2M-100 BASELINE")
    print("=" * 60)
    
    # Load the model
    print("\n📂 Loading M2M-100 model...")
    tokenizer, model = load_m2m100()
    
    if tokenizer is None or model is None:
        print("❌ Model not loaded. Exiting.")
        exit()
    
    # Language mapping
    languages = {
        "af": "Afrikaans",
        "zu": "isiZulu",
        "nso": "Northern Sotho"
    }
    
    # Evaluate each language
    # Use num_samples=None for full evaluation, or a number like 100 for faster testing
    results = {}
    
    for lang_code, lang_name in languages.items():
        score = evaluate_language(lang_code, lang_name, tokenizer, model, num_samples=200)
        if score is not None:
            results[lang_code] = score
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - BLEU SCORES")
    print("=" * 60)
    for lang_code, lang_name in languages.items():
        if lang_code in results:
            print(f"   {lang_name}: {results[lang_code]:.2f}")
        else:
            print(f"   {lang_name}: ❌ Not evaluated")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")