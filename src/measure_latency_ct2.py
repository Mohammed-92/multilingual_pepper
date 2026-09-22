# measure_latency_ct2.py
# Purpose: Test NLLB-200 with CTranslate2 int8 on Jetson CPU
# Author: Mohammed Gori
# Date: September 2026

import time
import statistics
import os
import ctranslate2
import sentencepiece as spm


def load_ct2_translator():
    """Load the CTranslate2 translator and SentencePiece tokenizer."""
    model_path = "./nllb-ct2-int8"

    print(f"Loading CTranslate2 model from {model_path}...")
    translator = ctranslate2.Translator(
        model_path,
        device="cpu",
        compute_type="int8"
    )

    # Locate the SentencePiece model
    spm_path = os.path.join(model_path, "sentencepiece.bpe.model")
    if not os.path.exists(spm_path):
        spm_path = os.path.join(model_path, "shared_vocabulary.txt")

    sp = spm.SentencePieceProcessor()
    if spm_path.endswith(".model"):
        sp.load(spm_path)
    else:
        print("Using shared vocabulary in pieces mode (already tokenised)")

    print("CTranslate2 model loaded successfully.\n")
    return translator, sp, spm_path


def translate_ct2(translator, sp, text, tgt_lang="zul_Latn", use_spm=True):
    """Translate using CTranslate2."""
    if use_spm:
        source_pieces = sp.encode_as_pieces(text)
    else:
        # Fallback: split on whitespace
        source_pieces = text.split()

    # Add language tokens
    source_pieces = ["eng_Latn"] + source_pieces + ["</s>"]

    results = translator.translate_batch(
        [source_pieces],
        target_prefix=[[tgt_lang]],
        max_batch_size=1024,
        beam_size=1,
        max_decoding_length=50
    )

    output_pieces = results[0].hypotheses[0]

    if use_spm:
        translation = sp.decode(output_pieces)
    else:
        translation = " ".join(output_pieces)

    return translation


def measure_latency(translator, sp, sentences, tgt_lang, repeats=5, use_spm=True):
    results = []
    for sentence in sentences:
        latencies = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = translate_ct2(translator, sp, sentence, tgt_lang, use_spm)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        results.append({
            "sentence": sentence[:55],
            "median_ms": statistics.median(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies)
        })
    return results


def run_language(translator, sp, lang_code, lang_name, sentences, repeats=5, use_spm=True):
    print("=" * 70)
    print(f"LANGUAGE: {lang_name} ({lang_code}) - CTranslate2 int8 | {repeats} repeats")
    print("=" * 70)

    _ = measure_latency(translator, sp, [sentences[0]], lang_code, repeats=1, use_spm=use_spm)
    results = measure_latency(translator, sp, sentences, lang_code, repeats=repeats, use_spm=use_spm)

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
        translator, sp, spm_path = load_ct2_translator()
        use_spm = spm_path.endswith(".model")
    except Exception as e:
        print(f"Failed to load CTranslate2 model: {e}")
        return

    # Test one translation
    print("Test translation:")
    print(f"  EN: Hello, how are you?")
    try:
        print(f"  ZU: {translate_ct2(translator, sp, 'Hello, how are you?', use_spm=use_spm)}")
    except Exception as e:
        print(f"  Translation failed: {e}")
    print()

    languages = [
        ("zul_Latn", "isiZulu"),
        ("afr_Latn", "Afrikaans"),
        ("nso_Latn", "Northern Sotho")
    ]

    all_results = []

    for lang_code, lang_name in languages:
        result = run_language(translator, sp, lang_code, lang_name, test_sentences,
                              repeats=5, use_spm=use_spm)
        all_results.append(result)

    with open("latency_results_ct2_all_languages.txt", "w") as f:
        f.write("NLLB-200 CTranslate2 int8 Latency Results (Jetson CPU)\n")
        f.write("Repeats: 5 | Metric: Median\n")
        f.write("Compute type: int8 | Beam size: 1\n")
        f.write("=" * 70 + "\n\n")
        for r in all_results:
            f.write(f"LANGUAGE: {r['lang_name']} ({r['lang_code']})\n")
            f.write("-" * 70 + "\n")
            for item in r["results"]:
                f.write(f"{item['sentence']:<57} | median {item['median_ms']:.2f} ms\n")
            f.write(f"\nOverall median: {r['median_ms']:.2f} ms\n")
            f.write(f"Below 2s: {r['below_2s']}/{len(r['results'])}\n\n")

        f.write("=" * 70 + "\n")
        f.write("SUMMARY - CTranslate2 int8\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Language':<20} | {'Median (ms)':<14} | {'Below 2s':<10}\n")
        f.write("-" * 70 + "\n")
        for r in all_results:
            f.write(f"{r['lang_name']:<20} | {r['median_ms']:<14.2f} | "
                    f"{r['below_2s']}/{len(r['results'])}\n")

    print("Results saved to latency_results_ct2_all_languages.txt")


if __name__ == "__main__":
    main()
