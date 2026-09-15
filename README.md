# Multilingual Language Generation on the Pepper Robot

Honours Research Project – Mohammed Gori (2580962)
Supervisors: Branden Ingram & Pravesh Ranchod
University of the Witwatersrand

## Overview

This repository contains code for evaluating multilingual language generation
for isiZulu, Afrikaans, and Northern Sotho, with the goal of deploying on
the Pepper robot.

## Project Structure

- `src/` – Python scripts for data processing, model evaluation, and latency testing
- `data/` – Processed Autshumato parallel corpus (not included in repo due to size)
- `notebooks/` – Exploratory analysis

## Scripts

- `load_nllb.py` – Loads NLLB-200 model
- `evaluate_nllb.py` – Computes BLEU scores for all three languages
- `load_autshumato_simple.py` – Loads and splits the Autshumato dataset
- `finetune_inkubalm.py` – Fine-tunes InkubaLM on Autshumato data
- `generate_user_study_sentences.py` – Generates sentences for user study
- `measure_latency_jetson.py` – Measures inference latency on Jetson

## Setup

```bash
pip install torch transformers sacrebleu datasets accelerate
=======
# multilingual_pepper

