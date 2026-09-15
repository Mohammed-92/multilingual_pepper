# measure_latency_jetson.py
# Purpose: Measure inference latency of NLLB-200 on the NVIDIA Jetson (CPU mode)
# Author: Mohammed Gori
# Date: September 2026
#
# Notes:
# - Running on CPU because Pepper has no GPU, so this is a more realistic proxy.
# - The GPU on the Jetson has insufficient memory for the full NLLB model.

import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_nllb():
    """Load the NLLB-200 model and tokenizer (CPU only)."""
    model_name = "facebook/nllb-200-distilled-600M"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    )
    model.eval()
    print("NLLB loaded successfully (CPU).")
    return tokenizer, model


def measure_latency(tokenizer, model, sentences, tgt_lang="zul_Latn", repeats=3):
    """
    Measure translation latency per sentence.
    """
    results = []
    for sentence in sentences:
        latencies = []
        for _ in range(repeats):
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)

            start = time.perf_counter()
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=50,
                    forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
                )
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

        results.append({
            "sentence": sentence[:60],
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

    tokenizer, model = load_nllb()

    print("\nRunning warm-up (CPU)...")
    _ = measure_latency(tokenizer, model, [test_sentences[0]], repeats=1)

    print(f"\nMeasuring latency for {len(test_sentences)} sentences (3 repeats each)...")
    results = measure_latency(tokenizer, model, test_sentences, tgt_lang="zul_Latn", repeats=3)

    print("\n" + "=" * 70)
    print(f"{'Sentence':<45} | {'Mean (ms)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['sentence']:<45} | {r['mean_ms']:.2f}")

    all_means = [r["mean_ms"] for r in results]
    avg = sum(all_means) / len(all_means)
    print("=" * 70)
    print(f"Overall average latency: {avg:.2f} ms ({avg/1000:.3f} seconds)")
    print(f"Min latency: {min(all_means):.2f} ms")
    print(f"Max latency: {max(all_means):.2f} ms")
    print("=" * 70)

    with open("latency_results_jetson.txt", "w") as f:
        f.write("NLLB-200 Latency Results (Jetson CPU)\n")
        f.write("=" * 60 + "\n")
        f.write("Device: CPU (GPU memory insufficient for NLLB-600M)\n")
        f.write("Target language: isiZulu (zul_Latn)\n\n")
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