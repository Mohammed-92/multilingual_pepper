# measure_latency_multilang.py
# Purpose: Measure NLLB-200 inference latency on the Jetson (CPU) for all 3 languages
# Author: Mohammed Gori
# Date: September 2026

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
    print("NLLB loaded successfully (CPU).\n")
    return tokenizer, model


def measure_latency(tokenizer, model, sentences, tgt_lang, repeats=3):
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
            "mean_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies)
        })
    return results


def run_language(tokenizer, model, lang_code, lang_name, sentences, repeats=3):
    """Run latency test for one language and print results."""
    print("=" * 70)
    print(f"LANGUAGE: {lang_name} ({lang_code})")
    print("=" * 70)

    # Warm-up
    _ = measure_latency(tokenizer, model, [sentences[0]], lang_code, repeats=1)

    results = measure_latency(tokenizer, model, sentences, lang_code, repeats=repeats)

    print(f"\n{'Sentence':<57} | {'Mean (ms)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['sentence']:<57} | {r['mean_ms']:.2f}")

    all_means = [r["mean_ms"] for r in results]
    avg = sum(all_means) / len(all_means)
    below_2s = sum(1 for x in all_means if x < 2000)

    print("-" * 70)
    print(f"Average latency: {avg:.2f} ms ({avg/1000:.3f} s)")
    print(f"Min: {min(all_means):.2f} ms | Max: {max(all_means):.2f} ms")
    print(f"Sentences below 2s: {below_2s}/{len(all_means)}")
    print("=" * 70 + "\n")

    return {
        "lang_code": lang_code,
        "lang_name": lang_name,
        "results": results,
        "avg_ms": avg,
        "min_ms": min(all_means),
        "max_ms": max(all_means),
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

    tokenizer, model = load_nllb()

    languages = [
        ("zul_Latn", "isiZulu"),
        ("afr_Latn", "Afrikaans"),
        ("nso_Latn", "Northern Sotho")
    ]

    all_results = []

    for lang_code, lang_name in languages:
        result = run_language(tokenizer, model, lang_code, lang_name, test_sentences, repeats=3)
        all_results.append(result)

    # Save to file
    with open("latency_results_all_languages.txt", "w") as f:
        f.write("NLLB-200 Latency Results (Jetson CPU) — All Languages\n")
        f.write("=" * 70 + "\n\n")
        for r in all_results:
            f.write(f"LANGUAGE: {r['lang_name']} ({r['lang_code']})\n")
            f.write("-" * 70 + "\n")
            for item in r["results"]:
                f.write(f"{item['sentence']:<57} | {item['mean_ms']:.2f}\n")
            f.write(f"\nAverage: {r['avg_ms']:.2f} ms\n")
            f.write(f"Min: {r['min_ms']:.2f} ms | Max: {r['max_ms']:.2f} ms\n")
            f.write(f"Below 2s: {r['below_2s']}/{len(r['results'])}\n\n")

        # Summary table
        f.write("=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Language':<20} | {'Avg (ms)':<12} | {'Avg (s)':<10} | {'Below 2s':<10}\n")
        f.write("-" * 70 + "\n")
        for r in all_results:
            f.write(f"{r['lang_name']:<20} | {r['avg_ms']:<12.2f} | "
                    f"{r['avg_ms']/1000:<10.3f} | {r['below_2s']}/{len(r['results'])}\n")

    print("Results saved to latency_results_all_languages.txt")


if __name__ == "__main__":
    main()