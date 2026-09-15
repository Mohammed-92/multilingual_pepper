# src/test_inkubalm_minimal.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("=" * 60)
print("INKUBALM - MINIMAL LOAD TEST")
print("=" * 60)

model_name = "lelapa/InkubaLM-0.4B"
print(f"\nLoading {model_name}...")

try:
    # Load tokenizer with minimal settings
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,  # Try fast tokenizer this time
        resume_download=True
    )
    print("✅ Tokenizer loaded")
    
    # Load model with minimal settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        resume_download=True
    )
    print("✅ Model loaded!")
    
    # Test generation
    print("\n📝 Testing generation...")
    prompt = "Today I planned to"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("\n✅ InkubaLM is working!")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nIf this fails, we can use NLLB as our primary model.")
    print("InkubaLM is still a valuable topic for discussion/future work.")