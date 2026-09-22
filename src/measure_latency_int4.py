# measure_latency_int4.py
# Purpose: Test NLLB-200 with torchao int4 weight-only quantisation on Jetson CPU
# Author: Mohammed Gori
# Date: September 2026
#
# Note: torchao int4 is the recommended quantisation for Jetson Orin by NVIDIA.
# bitsandbytes int4 was avoided due to aarch64 compatibility issues and
# dequantization overhead that negates latency gains.

import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TorchAoConfig
from torchao.quantization import Int4WeightOnlyConfig


def load_nllb_int4():
    """Load NLLB-200 with torchao int4 weight-only quantisation."""
    model_name = "facebook/nllb-200-distilled-600M"
    print(f"Loading {model_name} with int4 weight-only quantisation...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")

    # Configure int4 weight-only quantisation with group size 128
    quant_config = Int4WeightOnlyConfig(group_size=128)
    quantization_config = TorchAoConfig(quant_type=quant_config)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    model.eval()
    print("NLLB loaded with torchao int4 quantisation.\n")
    return tokenizer, model


def measure_latency(tokenizer, model, sentences, tgt_lang, repeats=5):
    """Measure translation latency per sentence (median across repeats)."""
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
            "min_ms": min(latencies),
            "max_ms": max(latencies)
        })
    return results


def run_language(tokenizer, model, lang_code, lang_name, sentences, repeats=5):
    """Run int4 latency test for one language."""
    print("=" * 70)
    print(f"LANGUAGE: {lang_name} ({lang_code}) - int4 | {repeats} repeats")
    print("=" * 70)

    _ = measure_latency(tokenizer, model, [sentences[0]], lang_code, repeats=1)

    results = measure_latency(tokenizer, model, sentences, lang_code, repeats=repeats)

    print(f"\n{'Sentence':<57} | {'Median (ms)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['sentence']:<57} | {r['median_ms']:.2f}")

    medians = [r["median_ms"] for r in results]
    overall_median = statistics.median(medians)
    below_2s = sum(1 for x in medians if x < 2000)

    print("-" * 70)
    print(f"Overall median: {overall_median:.2f} ms ({overall_median/1000:.3f} s)")
    print(f"Min: {min(medians):.2f} ms | Max: {max(medians):.2f} ms")
    print(f"Sentences below 2s: {below_2s}/{len(medians)}")
    print("=" * 70 + "\n")

    return {
        "lang_code": lang_code,
        "lang_name": lang_name,
        "results": results,
        "median_ms": overall_median,
        "min_ms": min(medians),
        "max_ms": max(medians),
        "below_2s": below_2s
    }


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
        tokenizer, model = load_nllb_int4()
    except Exception as e:
        print(f"Failed to load int4 model: {e}")
        print("\ntorchao int4 may not be supported on this build.")
        return

    languages = [
        ("zul_Latn", "isiZulu"),
        ("afr_Latn", "Afrikaans"),
        ("nso_Latn", "Northern Sotho")
    ]

    all_results = []

    for lang_code, lang_name in languages:
        result = run_language(tokenizer, model, lang_code, lang_name, test_sentences, repeats=5)
        all_results.append(result)

    with open("latency_results_int4_all_languages.txt", "w") as f:
        f.write("NLLB-200 int4 (torchao) Latency Results (Jetson CPU)\n")
        f.write("Repeats: 5 | Metric: Median\n")
        f.write("Quantisation: Int4WeightOnlyConfig(group_size=128)\n")
        f.write("=" * 70 + "\n\n")
        for r in all_results:
            f.write(f"LANGUAGE: {r['lang_name']} ({r['lang_code']})\n")
            f.write("-" * 70 + "\n")
            for item in r["results"]:
                f.write(f"{item['sentence']:<57} | median {item['median_ms']:.2f} ms\n")
            f.write(f"\nOverall median: {r['median_ms']:.2f} ms\n")
            f.write(f"Min: {r['min_ms']:.2f} ms | Max: {r['max_ms']:.2f} ms\n")
            f.write(f"Below 2s: {r['below_2s']}/{len(r['results'])}\n\n")

        f.write("=" * 70 + "\n")
        f.write("SUMMARY - INT4\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Language':<20} | {'Median (ms)':<14} | {'Below 2s':<10}\n")
        f.write("-" * 70 + "\n")
        for r in all_results:
            f.write(f"{r['lang_name']:<20} | {r['median_ms']:<14.2f} | "
                    f"{r['below_2s']}/{len(r['results'])}\n")

    print("Results saved to latency_results_int4_all_languages.txt")


if __name__ == "__main__":
    main()