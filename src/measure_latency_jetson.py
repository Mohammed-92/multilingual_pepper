# measure_latency_jetson.py
# Purpose: Measure inference latency of NLLB-200 on the NVIDIA Jetson
# Author: Mohammed Gori
# Date: September 2026
#
# This script loads NLLB-200, runs inference on a set of English sentences,
# and measures the time taken to generate isiZulu translations. It reports
# mean, min, and max latency per sentence, and the overall average.

import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_nllb():
    """Load the NLLB-200 model and tokenizer."""
    model_name = "facebook/nllb-200-distilled-600M"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("NLLB loaded successfully.")
    return tokenizer, model


def measure_latency(tokenizer, model, sentences, tgt_lang="zul_Latn", repeats=5):
    """
    Measure the latency of translation for each sentence.
    
    Args:
        tokenizer: NLLB tokenizer
        model: NLLB model
        sentences: list of English strings
        tgt_lang: target language code (e.g., 'zul_Latn' for isiZulu)
        repeats: number of times to run each sentence (for averaging)
    
    Returns:
        list of dicts with sentence, mean/min/max latency in milliseconds
    """
    results = []
    for sentence in sentences:
        latencies = []
        for _ in range(repeats):
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            start = time.perf_counter()
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=50,
                    forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
                )
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # convert to milliseconds

        results.append({
            "sentence": sentence[:60],
            "mean_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies)
        })
    return results


def main():
    # --- Test sentences (English) ---
    # These are the same style of sentences from the Autshumato test set
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

    # --- Load model ---
    tokenizer, model = load_nllb()

    # --- Move to GPU if available ---
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU (CUDA)")
    else:
        print("Using CPU (no GPU detected — this will be slower)")

    # --- Warm-up run (not counted) ---
    print("\nRunning warm-up...")
    _ = measure_latency(tokenizer, model, [test_sentences[0]], repeats=1)

    # --- Measure latency ---
    print(f"\nMeasuring latency for {len(test_sentences)} sentences (5 repeats each)...")
    results = measure_latency(tokenizer, model, test_sentences, tgt_lang="zul_Latn", repeats=5)

    # --- Print results ---
    print("\n" + "=" * 70)
    print(f"{'Sentence':<45} | {'Mean (ms)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['sentence']:<45} | {r['mean_ms']:.2f}")

    # --- Summary ---
    all_means = [r["mean_ms"] for r in results]
    avg = sum(all_means) / len(all_means)
    print("=" * 70)
    print(f"Overall average latency: {avg:.2f} ms ({avg/1000:.3f} seconds)")
    print(f"Min latency: {min(all_means):.2f} ms")
    print(f"Max latency: {max(all_means):.2f} ms")
    print("=" * 70)

    # --- Save results to file ---
    with open("latency_results_jetson.txt", "w") as f:
        f.write("NLLB-200 Latency Results (Jetson)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Target language: isiZulu (zul_Latn)\n\n")
        f.write(f"{'Sentence':<45} | {'Mean (ms)':<12}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['sentence']:<45} | {r['mean_ms']:.2f}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Overall average: {avg:.2f} ms\n")
        f.write(f"Min: {min(all_means):.2f} ms\n")
        f.write(f"Max: {max(all_means):.2f} ms\n")

    print("\nResults saved to latency_results_jetson.txt")


if __name__ == "__main__":
    main()