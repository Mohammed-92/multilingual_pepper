# src/test_inkubalm_pipeline.py
from transformers import pipeline

print("Loading InkubaLM...")

# Use the pipeline with the model
pipe = pipeline(
    "text-generation",
    model="lelapa/InkubaLM-0.4B",
    trust_remote_code=True,
    device_map="auto"  # Automatically uses GPU if available
)

print("✅ InkubaLM loaded through pipeline!")

# Test generation
prompt = "Hello, how are you?"
result = pipe(prompt, max_new_tokens=50)
print(f"Prompt: {prompt}")
print(f"Response: {result[0]['generated_text']}")

# Test in isiZulu
prompt_zulu = "Sawubona, unjani?"  # Hello, how are you?
result_zulu = pipe(prompt_zulu, max_new_tokens=50)
print(f"\nPrompt (isiZulu): {prompt_zulu}")
print(f"Response: {result_zulu[0]['generated_text']}")