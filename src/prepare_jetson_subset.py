# prepare_jetson_subset.py
# Purpose: Create small 100-sentence test sets for Jetson BLEU evaluation
# Author: Mohammed Gori
# Date: September 2026

import os
import random

def create_subset(lang_code, num_sentences=100):
    """Extract random subset from the test set."""
    input_file = f"data/processed/{lang_code}/test.txt"
    output_file = f"data/jetson_test/{lang_code}.txt"

    if not os.path.exists(input_file):
        print(f"Missing: {input_file}")
        return

    os.makedirs("data/jetson_test", exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    random.seed(42)
    subset = random.sample(lines, min(num_sentences, len(lines)))

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in subset:
            f.write(line + '\n')

    print(f"Created {output_file} with {len(subset)} sentences")


if __name__ == "__main__":
    for lang in ["zu", "af", "nso"]:
        create_subset(lang, 100)
    print("\nDone. Files ready in data/jetson_test/")