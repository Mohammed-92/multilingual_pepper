# measure_latency_quantised.py
# Purpose: Test NLLB-200 in 8-bit quantisation on the Jetson
# Author: Mohammed Gori
# Date: September 2026
#
# This script uses bitsandbytes to load NLLB in 8-bit precision.
# It measures latency for isiZulu and compares with full-precision results.

import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig


def load_nllb_8bit():
    """Load NLLB-200 in 8-bit quantisation."""
    model_name = "facebook/nllb-200-distilled-600M"
    print(f"Loading {model_name} in 8-bit...")

    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    model.eval()
    print("NLLB loaded in 8-bit.\n")
    return tokenizer, model


def measure_latency(tokenizer, model, sentences, tgt_lang="zul_Latn", repeats=3):
    """Measure translation latency per sentence."""
    results = []
    target_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    for sentence in sentences:
        latencies = []
        for _ in range(repeats):
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            start = time.perf_counter()
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=50,
                    forced_bos_token_id=target_token_id
                )
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

        results.append({
            "sentence": sentence[:55],
            "mean_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies)
        })
    return results


def main():
    test_sentences = [
        "Hello, how are you?",
        "Please remind the patient to take their medication.",
        "The doctor will see you now.",
        "Thank you for your help.",
        "Good morning, how can I assist you?",
        "Please wait here for your appointment.",
        "The hospital is open 24 hours a day.",
        "You can collect your medication at the pharmacy.",
        "The nurse will check your blood pressure.",
        "Please fill out this form before you proceed."
    ]

    try:
        tokenizer, model = load_nllb_8bit()
    except Exception as e:
        print(f"❌ Failed to load 8-bit model: {e}")
        print("\nbitsandbytes may not be installed correctly on Jetson.")
        print("Try: pip3 install bitsandbytes")
        print("If that fails, skip this test and report the full-precision results instead.")
        return

    print("Running warm-up...")
    _ = measure_latency(tokenizer, model, [test_sentences[0]], repeats=1)

    print(f"\nMeasuring latency for {len(test_sentences)} sentences (3 repeats each)...\n")
    results = measure_latency(tokenizer, model, test_sentences, tgt_lang="zul_Latn", repeats=3)

    print("=" * 70)
    print(f"{'Sentence':<57} | {'Mean (ms)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['sentence']:<57} | {r['mean_ms']:.2f}")

    all_means = [r["mean_ms"] for r in results]
    avg = sum(all_means) / len(all_means)
    below_2s = sum(1 for x in all_means if x < 2000)

    print("=" * 70)
    print(f"Average latency (8-bit): {avg:.2f} ms ({avg/1000:.3f} s)")
    print(f"Min: {min(all_means):.2f} ms | Max: {max(all_means):.2f} ms")
    print(f"Sentences below 2s: {below_2s}/{len(all_means)}")
    print("=" * 70)

    # Save results
    with open("latency_results_quantised.txt", "w") as f:
        f.write("NLLB-200 8-bit Latency Results (Jetson)\n")
        f.write("=" * 60 + "\n")
        f.write("Precision: 8-bit (bitsandbytes)\n")
        f.write("Target language: isiZulu\n\n")
        for r in results:
            f.write(f"{r['sentence']:<57} | {r['mean_ms']:.2f}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Average: {avg:.2f} ms\n")
        f.write(f"Min: {min(all_means):.2f} ms | Max: {max(all_means):.2f} ms\n")
        f.write(f"Below 2s: {below_2s}/{len(all_means)}\n")

    print("\nResults saved to latency_results_quantised.txt")


if __name__ == "__main__":
    main()