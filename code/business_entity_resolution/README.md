# Business Entity Resolution Pipeline

High-performance, large-scale Business Entity Resolution system designed for the Amazon ML Challenge 2026.

## 🚀 Key Innovations & Architecture
1. **Zero-Leakage Country Partitioning**: Empirically verified 0.00% cross-border matching. The problem is decomposed into three isolated streams (`US`, `India`, `France`), reducing comparison space by ~65% and keeping peak RAM under 4 GB.
2. **Unified Brahmi Multilingual Indic Normalization**: Phonetizes Devanagari, Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam, Oriya, and Gurmukhi scripts into Roman characters, solving the ~27.5% Indic-script match failure mode.
3. **Diacritic Folding & Zero-Stripped Address Anchors**: Strips leading zeros in addresses (`0684` <-> `684`, `0029` <-> `29`) to connect noisy street and plot numbers.
4. **Multi-Index Inverted Index Blocker**: Dual-channel inverted index indexing core brand words (IDF-weighted), 4-character brand prefixes, and `(number, locality)` pairs.
5. **C++ Accelerated RapidFuzz Pairwise Feature Extraction**: 15 domain-specific pairwise features spanning token sort ratios, brand ratios, address Jaccard, and numeric matches.
6. **LightGBM Classifier with Asymmetric Macro F_0.5 Calibration**: Optimizes threshold directly against macro F_0.5 on held-out validation queries.

## 📦 Setup & Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃 Reproduction & Execution
Run the end-to-end pipeline from `student_resource/`:
```bash
python code/business_entity_resolution/src/pipeline.py
```
This script will:
1. Train the LightGBM classifier on representative training pairs and calibrate the macro F_0.5 decision threshold.
2. Run candidate generation and scoring across the test set for France, US, and India.
3. Output `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
4. Automatically run `utils/validate_submission.py` to confirm submission validity.
