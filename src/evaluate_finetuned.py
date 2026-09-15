# src/evaluate_finetuned.py
import os
import sacrebleu
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_finetuned_model(lang_code):
    """Load the fine-tuned InkubaLM model."""
    base_model_name = "lelapa/InkubaLM-0.4B"
    adapter_path = f"./inkubalm-finetuned-{lang_code}"
    
    print(f"Loading fine-tuned model from {adapter_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            use_fast=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        print("✅ Fine-tuned model loaded!")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        return None, None

def translate_finetuned(tokenizer, model, text, lang_name):
    """Translate using the fine-tuned model with exact training prompt."""
    if tokenizer is None or model is None:
        return "Model not loaded"
    
    # Use EXACTLY the same format as training
    prompt = f"### Instruction: Translate the following English sentence to {lang_name}.\n### Input: {text}\n### Response:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.1,  # Lower temperature for more deterministic output
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Try to extract just the Response part
    if "### Response:" in response:
        translation = response.split("### Response:")[-1].strip()
        # Stop at any new instruction if it appears
        if "###" in translation:
            translation = translation.split("###")[0].strip()
        return translation
    return response
    
    # Extract the response part
    if "### Response:" in response:
        translation = response.split("### Response:")[-1].strip()
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

def evaluate_language(lang_code, lang_name, tokenizer, model, num_samples=100):
    """
    Evaluate BLEU for one language using the fine-tuned model.
    """
    print("\n" + "=" * 60)
    print(f"EVALUATING {lang_name} ({lang_code.upper()}) with Fine-tuned InkubaLM")
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
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{len(english)}")
        
        translation = translate_finetuned(tokenizer, model, sentence, lang_name)
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
    print("BLEU EVALUATION - FINE-TUNED INKUBALM")
    print("=" * 60)
    
    # Language mapping
    languages = {
        "zu": "isiZulu",
        "nso": "Northern Sotho"
    }
    
    results = {}
    
    for lang_code, lang_name in languages.items():
        # Load fine-tuned model
        print(f"\n📂 Loading fine-tuned model for {lang_name}...")
        tokenizer, model = load_finetuned_model(lang_code)
        
        if tokenizer is None or model is None:
            print(f"❌ Could not load fine-tuned model for {lang_name}")
            continue
        
        # Evaluate
        score = evaluate_language(lang_code, lang_name, tokenizer, model, num_samples=100)
        if score is not None:
            results[lang_code] = score
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("FINAL SUMMARY - BLEU SCORES COMPARISON")
    print("=" * 60)
    print(f"   {'Language':<20} | {'NLLB':<10} | {'Base InkubaLM':<13} | {'Fine-tuned':<10} | {'Improvement':<10}")
    print("   " + "-" * 85)
    
    nllb_scores = {"zu": 10.02, "nso": 5.47}
    base_scores = {"zu": 0.18, "nso": 0.27}
    
    for lang_code, lang_name in languages.items():
        if lang_code in results:
            fine_tuned = results[lang_code]
            nllb = nllb_scores.get(lang_code, 0)
            base = base_scores.get(lang_code, 0)
            improvement = fine_tuned - nllb
            arrow = "📈" if improvement > 2 else ("📉" if improvement < -2 else "➡️")
            print(f"   {lang_name:<20} | {nllb:<10.2f} | {base:<13.2f} | {fine_tuned:<10.2f} | {improvement:<+10.2f} {arrow}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")