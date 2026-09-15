# src/load_m2m100.py
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

def load_m2m100():
    """
    Load the M2M-100 multilingual translation model.
    This is faster to download and works well for testing.
    """
    model_name = "facebook/m2m100_418M"
    
    print(f"Loading M2M-100 from {model_name}...")
    
    try:
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        print("✅ M2M-100 loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading M2M-100: {e}")
        return None, None

def translate_m2m100(tokenizer, model, text, src_lang="en", tgt_lang="zu"):
    """
    Translate text using M2M-100.
    """
    if tokenizer is None or model is None:
        return "Model not loaded"
    
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            max_new_tokens=50
        )
    
    translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translation

if __name__ == "__main__":
    # Test the loading
    tokenizer, model = load_m2m100()
    
    if tokenizer is not None and model is not None:
        # Test translation
        text = "Hello, how are you?"
        translation = translate_m2m100(tokenizer, model, text, "en", "zu")
        print(f"English: {text}")
        print(f"isiZulu: {translation}")