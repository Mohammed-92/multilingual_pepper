# src/evaluate_inkubalm_final.py
import os
import sacrebleu
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_inkubalm():
    """Load InkubaLM with minimal settings."""
    model_name = "lelapa/InkubaLM-0.4B"
    
    print(f"Loading InkubaLM from {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("✅ InkubaLM loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading InkubaLM: {e}")
        return None, None

def translate_inkubalm(tokenizer, model, text, target_lang="isiZulu"):
    """Translate using InkubaLM with a simpler prompt."""
    if tokenizer is None or model is None:
        return "Model not loaded"
    
    # Simpler prompt format
    prompt = f"English: {text}\n{target_lang}:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.5,  # Try higher temperature
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2  # Add this to reduce repetition
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract translation
    if f"{target_lang}:" in response:
        translation = response.split(f"{target_lang}:")[-1].strip()
        # Remove any remaining English text
        if "English:" in translation:
            translation = translation.split("English:")[0].strip()
        return translation
    return response

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

def evaluate_language(lang_code, lang_name, tokenizer, model, num_samples=50):
    """
    Evaluate BLEU for one language using InkubaLM.
    """
    print("\n" + "=" * 60)
    print(f"EVALUATING {lang_name} ({lang_code.upper()}) with InkubaLM")
    print("=" * 60)
    
    # Load test data
    english, references = load_test_data(lang_code)
    
    if not english:
        print(f"❌ No test data for {lang_code}")
        return None
    
    # Limit samples
    if num_samples and num_samples < len(english):
        english = english[:num_samples]
        references = references[:num_samples]
        print(f"   Using first {num_samples} sentences for evaluation")
    
    # Translate each sentence
    print(f"   Translating {len(english)} sentences...")
    translations = []
    
    for i, sentence in enumerate(english):
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{len(english)}")
        
        translation = translate_inkubalm(tokenizer, model, sentence, lang_name)
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
    print("BLEU EVALUATION - INKUBALM")
    print("=" * 60)
    
    # Load the model
    print("\n📂 Loading InkubaLM model...")
    tokenizer, model = load_inkubalm()
    
    if tokenizer is None or model is None:
        print("❌ Model not loaded. Exiting.")
        exit()
    
    # Language mapping
    languages = {
        "zu": "isiZulu",
        "nso": "Northern Sotho"
    }
    
    # Evaluate each language
    results = {}
    
    for lang_code, lang_name in languages.items():
        score = evaluate_language(lang_code, lang_name, tokenizer, model, num_samples=50)
        if score is not None:
            results[lang_code] = score
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY - BLEU SCORES COMPARISON")
    print("=" * 60)
    print(f"   {'Language':<20} | {'NLLB':<10} | {'InkubaLM':<10} | {'Improvement':<10}")
    print("   " + "-" * 65)
    
    nllb_scores = {"zu": 10.02, "nso": 5.47}
    
    for lang_code, lang_name in languages.items():
        if lang_code in results:
            inkubalm_score = results[lang_code]
            nllb_score = nllb_scores.get(lang_code, 0)
            improvement = inkubalm_score - nllb_score
            arrow = "📈" if improvement > 2 else ("📉" if improvement < -2 else "➡️")
            print(f"   {lang_name:<20} | {nllb_score:<10.2f} | {inkubalm_score:<10.2f} | {improvement:<+10.2f} {arrow}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")