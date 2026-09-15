# src/test_baseline_simple.py
from load_m2m100 import load_m2m100, translate_m2m100

print("=" * 50)
print("MULTILINGUAL PEPPER - BASELINE TEST")
print("=" * 50)

# Load the model
tokenizer, model = load_m2m100()

if tokenizer is None or model is None:
    print("❌ Model not loaded. Exiting.")
    exit()

# Test sentences (pharmacy/medical commands)
test_sentences = [
    "Please remind the patient to take their medication.",
    "The doctor will see you now.",
    "Thank you for your help.",
    "Good morning, how can I assist you?",
    "Please wait here for your appointment."
]

print("\n📝 Testing translations:\n")

# Test isiZulu (zu)
print("=== isiZulu ===")
for i, sentence in enumerate(test_sentences, 1):
    print(f"{i}. English: {sentence}")
    translation = translate_m2m100(tokenizer, model, sentence, "en", "zu")
    print(f"   isiZulu: {translation}")
    print()

# Test Afrikaans (af)
print("=== Afrikaans ===")
for i, sentence in enumerate(test_sentences, 1):
    print(f"{i}. English: {sentence}")
    translation = translate_m2m100(tokenizer, model, sentence, "en", "af")
    print(f"   Afrikaans: {translation}")
    print()

print("=" * 50)
print("✅ Test complete!")