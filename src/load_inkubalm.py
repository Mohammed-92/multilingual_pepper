# src/load_inkubalm.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_inkubalm():
    """
    Load the InkubaLM model from Hugging Face.
    """
    model_name = "lelapa/InkubaLM-0.4B"
    
    print(f"Loading InkubaLM from {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16  # Use half precision to save memory
        )
        print("✅ InkubaLM loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading InkubaLM: {e}")
        return None, None

def generate_text(tokenizer, model, prompt, max_length=50):
    """
    Generate text using InkubaLM (simplified version).
    """
    if tokenizer is None or model is None:
        return "Model not loaded"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Use a simpler generation approach
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    tokenizer, model = load_inkubalm()
    
    if tokenizer is not None and model is not None:
        prompt = "Hello, how are you?"
        response = generate_text(tokenizer, model, prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")