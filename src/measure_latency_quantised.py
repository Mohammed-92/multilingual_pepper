# measure_latency_quantised.py
# Purpose: Test NLLB-200 with PyTorch dynamic quantisation (CPU) on the Jetson
# Author: Mohammed Gori
# Date: September 2026
#
# Note: Uses torch.quantization.quantize_dynamic which works on CPU.
# Requires torch.backends.quantized.engine = 'qnnpack' on ARM/Jetson.

import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_nllb_quantised():
    """Load NLLB-200 and apply dynamic quantisation to linear layers."""
    model_name = "facebook/nllb-200-distilled-600M"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    )

    # Set the quantization engine to qnnpack (required for ARM/Jetson)
    print("Setting quantization engine to qnnpack...")
    torch.backends.quantized.engine = 'qnnpack'
    print(f"Supported engines: {torch.backends.quantized.supported_engines}")

    # Apply dynamic quantisation to Linear layers
    print("Applying dynamic quantisation...")
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    model.eval()
    print("NLLB loaded with dynamic quantisation (CPU).\n")
    return tokenizer, model


def measure_latency(tokenizer, model, sentences, tgt_lang="zul_Latn", repeats=5):
    """Measure translation latency per sentence."""
    results = []
    target_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    for sentence in sentences:
        latencies = []
        for _ in range(repeats):
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)

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
            "median_ms": statistics.median(latencies),
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
        tokenizer, model = load_nllb_quantised()
    except Exception as e:
        print(f"Failed to load quantised model: {e}")
        print("\nQuantisation may not be supported on this Jetson build.")
        return

    print("Running warm-up...")
    _ = measure_latency(tokenizer, model, [test_sentences[0]], repeats=1)

    print(f"\nMeasuring latency for {len(test_sentences)} sentences (5 repeats each)...\n")
    results = measure_latency(tokenizer, model, test_sentences, tgt_lang="zul_Latn", repeats=5)

    print("=" * 70)
    print(f"{'Sentence':<57} | {'Median (ms)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['sentence']:<57} | {r['median_ms']:.2f}")

    medians = [r["median_ms"] for r in results]
    avg = statistics.median(medians)
    below_2s = sum(1 for x in medians if x < 2000)

    print("=" * 70)
    print(f"Median latency (dynamic quantisation): {avg:.2f} ms ({avg/1000:.3f} s)")
    print(f"Min: {min(medians):.2f} ms | Max: {max(medians):.2f} ms")
    print(f"Sentences below 2s: {below_2s}/{len(medians)}")
    print("=" * 70)

    with open("latency_results_quantised.txt", "w") as f:
        f.write("NLLB-200 Dynamic Quantisation Latency Results (Jetson CPU)\n")
        f.write("=" * 60 + "\n")
        f.write("Precision: Dynamic int8 (PyTorch qnnpack)\n")
        f.write("Repeats: 5 | Metric: Median\n")
        f.write("Target language: isiZulu\n\n")
        for r in results:
            f.write(f"{r['sentence']:<57} | {r['median_ms']:.2f}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Median: {avg:.2f} ms\n")
        f.write(f"Min: {min(medians):.2f} ms | Max: {max(medians):.2f} ms\n")
        f.write(f"Below 2s: {below_2s}/{len(medians)}\n")

    print("\nResults saved to latency_results_quantised.txt")


if __name__ == "__main__":
    main()