# src/test_nllb.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

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

if __name__ == "__main__":
    # Language codes for NLLB
    languages = {
        "zu": {"name": "isiZulu", "code": "zul_Latn"},
        "nso": {"name": "Northern Sotho", "code": "nso_Latn"},
        "af": {"name": "Afrikaans", "code": "afr_Latn"}
    }
    
    tokenizer, model = load_nllb()
    
    if tokenizer is None or model is None:
        print("❌ Model not loaded. Exiting.")
        exit()
    
    # Test translations
    test_sentences = [
        "Hello, how are you?",
        "Please remind the patient to take their medication.",
        "The doctor will see you now."
    ]
    
    for lang_code, lang_info in languages.items():
        print(f"\n=== {lang_info['name']} ({lang_code}) ===")
        for sentence in test_sentences:
            translation = translate_nllb(tokenizer, model, sentence, "eng_Latn", lang_info['code'])
            print(f"English: {sentence}")
            print(f"{lang_info['name']}: {translation}")
            print()