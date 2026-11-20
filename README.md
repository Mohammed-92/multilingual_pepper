# Multilingual Pepper: Real-Time NMT for South African Languages

Code and evaluation scripts for deploying **NLLB-200** on CPU-only edge hardware (NVIDIA Jetson Nano, a proxy for Pepper's Intel Atom E3845) for **isiZulu**, **Afrikaans**, and **Northern Sotho** translation.

This repository accompanies the Honours research paper:

> **Real-Time Multilingual Generation for Pepper-Class Social Robots: A Quantised Inference Study for South African Languages**  
> Mohammed Gori (2026)  
> Supervisors: Dr. Branden Ingram & Dr. Pravesh Ranchod  
> School of Computer Science & Applied Mathematics, University of the Witwatersrand

---

## 🎯 Key Results

| Metric | Result |
|---|---|
| **CTranslate2 int8 latency** | **0.991–1.218 s** median (100% below 2 s threshold) |
| **Speedup vs PyTorch fp32** | **up to 3.2×** |
| **Speedup vs PyTorch int8** | **up to 1.4×** |
| **Native isiZulu fluency rating** | **3.85 / 5** (13 speakers, 650 ratings) |
| **BLEU vs human correlation** | ρ = 0.34 (weak — BLEU alone is insufficient) |

**Bottom line:** Real-time multilingual generation on CPU-only social robots is computationally feasible with CTranslate2 int8, but a lightweight grammar checker is needed for safe deployment.

---

## 📁 Repository Structure

```
multilingual_pepper/
├── src/                      # Translation, evaluation, and benchmarking scripts
│   ├── evaluate_nllb.py      # BLEU evaluation for NLLB-200
│   ├── measure_latency_jetson.py  # Latency benchmarking on Jetson Nano
│   ├── finetune_inkubalm.py  # InkubaLM fine-tuning experiments
│   └── ...
├── data/                     # Data download & preprocessing instructions
├── notebooks/                # Exploratory analysis
├── RQ3_Results_Slides.pptx   # Native-speaker evaluation summary
├── User Study Sentences.csv  # 50 isiZulu sentences used for human eval
├── README.md
├── LICENSE
├── CITATION.cff
└── requirements.txt
```

---

## ⚙️ Hardware & Environment

- **Device:** NVIDIA Jetson Nano (4GB), CPU-only
- **CPU:** Quad-core ARM Cortex-A57
- **OS:** Ubuntu 18.04 LTS
- **PyTorch:** 1.13.1
- **CTranslate2:** 3.24.0
- **Model:** `facebook/nllb-200-distilled-600M`

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Mohammed-92/multilingual_pepper.git
cd multilingual_pepper

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a translation (example — adjust script name as needed)
python src/translate.py --config ctranslate2_int8 --text "Hello, how are you?"
```

---

## 📊 Data

This project uses the **Autshumato parallel corpus** of South African government documents.

| Language pair | Available sentences | Test sample used |
|---|---|---|
| English → isiZulu | 23,370 | 200 (BLEU) + 50 (human eval) |
| English → Afrikaans | 136,787 | 200 (BLEU) |
| English → Northern Sotho | 17,130 | 200 (BLEU) |

**Raw data is not redistributed** due to licensing. See `data/README.md` for download and preprocessing instructions.

**Note on language selection:** The Autshumato corpus contains Northern Sotho (Sepedi) rather than Sesotho, so Northern Sotho is used as a proxy low-resource Sotho language for evaluation.

---

## 🧪 User Study & Ethics

The human evaluation of isiZulu translations was approved by the **Wits Human Research Ethics Committee (HREC)** [CSAM-2026-05].

- 19 respondents, of whom **13 native isiZulu speakers** completed all 50 ratings → **650 valid ratings**
- Raw survey responses are **not shared** to protect participant privacy
- The blank questionnaire is available at `https://forms.gle/bJ4NEN7ssWu6q3bj9`
- Results summary: see `RQ3_Results_Slides.pptx`

---

## 🔬 Reproducing the Results

**Latency benchmark:**

```bash
python src/measure_latency_jetson.py --config {fp32, pytorch_int8, ctranslate2_int8}
```

**BLEU evaluation:**

```bash
python src/evaluate_nllb.py --language {zul, afr, nso} --n-samples 200
```

All scores are reported on a 0–100 scale using SacreBLEU v2.0 (default `13a` tokenizer, consistent casing). Latency was measured over 5 timed repetitions per sentence after 1 warm-up inference, using greedy decoding (beam size = 1, max output length 50 tokens).

---

## 🧠 Method Summary

Three inference configurations were compared:

1. **PyTorch fp32** — full-precision baseline
2. **PyTorch int8** — dynamic quantisation via `qnnpack`
3. **CTranslate2 int8** — optimised C++ engine with hand-tuned ARM int8 kernels

CTranslate2 wins because it avoids Python interpreter overhead and the dequantise/re-quantise cycles that limit PyTorch's dynamic quantisation.

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{gori2026multilingualpepper,
  title  = {Multilingual Pepper: Real-Time NMT for South African Languages},
  author = {Gori, Mohammed},
  year   = {2026},
  note   = {Honours research project, University of the Witwatersrand},
  url    = {https://github.com/Mohammed-92/multilingual_pepper}
}
```

A `CITATION.cff` file is included, so GitHub will offer a **"Cite this repository"** button.

---

## 📄 License

Released under the **MIT License** — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- **Supervisors:** Dr. Branden Ingram & Dr. Pravesh Ranchod
- **Institution:** University of the Witwatersrand, School of Computer Science & Applied Mathematics
- **Models:** Meta's [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M), [InkubaLM](https://huggingface.co/sarahkumal/InkubaLM-0.4B)
- **Inference engine:** [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- **Data:** [Autshumato](https://autshumato.sourceforge.net/)
- **Ethics:** Wits Human Research Ethics Committee