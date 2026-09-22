# evaluate_quantised_bleu.py
# Purpose: Compute BLEU for quantised NLLB on all 3 languages
# Author: Mohammed Gori
# Date: September 2026

import os
import statistics
import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_nllb_quantised():
    """Load NLLB-200 with int8 quantisation."""
    model_name = "facebook/nllb-200-distilled-600M"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    )

    torch.backends.quantized.engine = 'qnnpack'
    print("Applying int8 quantisation...")
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    model.eval()
    print("NLLB quantised successfully.\n")
    return tokenizer, model


def translate(tokenizer, model, text, tgt_lang):
    """Translate one sentence."""
    target_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            forced_bos_token_id=target_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_test_set(lang_code):
    """Load the small Jetson test set."""
    path = f"data/jetson_test/{lang_code}.txt"

    if not os.path.exists(path):
        print(f"Missing: {path}")
        return [], []

    english, references = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                english.append(parts[0])
                references.append(parts[1])

    return english, references


def evaluate_language(tokenizer, model, lang_code, lang_name, nllb_code):
    """Compute BLEU for one language."""
    print("=" * 70)
    print(f"BLEU EVALUATION (quantised): {lang_name}")
    print("=" * 70)

    english, references = load_test_set(lang_code)

    if not english:
        print(f"No test data for {lang_code}")
        return None

    print(f"Translating {len(english)} sentences...")
    translations = []
    for i, sent in enumerate(english):
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(english)}")
        translations.append(translate(tokenizer, model, sent, nllb_code))

    bleu = sacrebleu.corpus_bleu(translations, [references])

    print(f"\nBLEU score ({lang_name}, quantised): {bleu.score:.2f}\n")

    return {
        "lang_code": lang_code,
        "lang_name": lang_name,
        "bleu": bleu.score,
        "num_sentences": len(translations),
        "examples": list(zip(english[:3], references[:3], translations[:3]))
    }


def main():
    tokenizer, model = load_nllb_quantised()

    languages = [
        ("zu", "isiZulu", "zul_Latn"),
        ("af", "Afrikaans", "afr_Latn"),
        ("nso", "Northern Sotho", "nso_Latn")
    ]

    all_results = []

    for lang_code, lang_name, nllb_code in languages:
        result = evaluate_language(tokenizer, model, lang_code, lang_name, nllb_code)
        if result:
            all_results.append(result)

    # Save
    with open("bleu_results_quantised.txt", "w") as f:
        f.write("NLLB-200 Quantised (int8) BLEU Results\n")
        f.write("=" * 70 + "\n\n")
        for r in all_results:
            f.write(f"LANGUAGE: {r['lang_name']}\n")
            f.write(f"  BLEU: {r['bleu']:.2f}\n")
            f.write(f"  Sentences: {r['num_sentences']}\n\n")
            f.write("  Examples:\n")
            for eng, ref, gen in r["examples"]:
                f.write(f"    EN:  {eng[:70]}\n")
                f.write(f"    REF: {ref[:70]}\n")
                f.write(f"    GEN: {gen[:70]}\n\n")
            f.write("-" * 70 + "\n\n")

        f.write("=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Language':<20} | {'BLEU':<10}\n")
        f.write("-" * 70 + "\n")
        for r in all_results:
            f.write(f"{r['lang_name']:<20} | {r['bleu']:<10.2f}\n")

    print("Results saved to bleu_results_quantised.txt")


if __name__ == "__main__":
    main()