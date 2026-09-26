# Business Entity Resolution Pipeline

High-performance, large-scale Business Entity Resolution system designed for the Amazon ML Challenge 2026.

## 🚀 Key Innovations & Architecture
1. **Zero-Leakage Country Partitioning**: Empirically verified 0.00% cross-border matching. The problem is decomposed into three isolated streams (`US`, `India`, `France`), reducing comparison space by ~65% and keeping peak RAM under 4 GB.
2. **Unified Brahmi Multilingual Indic Normalization**: Phonetizes Devanagari, Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam, Oriya, and Gurmukhi scripts into Roman characters, solving the ~27.5% Indic-script match failure mode.
3. **Diacritic Folding & Zero-Stripped Address Anchors**: Strips leading zeros in addresses (`0684` <-> `684`, `0029` <-> `29`) to connect noisy street and plot numbers.
4. **Multi-Channel Inverted Index Blocker**: High-recall blocker indexing core brand words (IDF-weighted), 4-char prefixes, character 3-grams, acronyms, and `(number, locality)` / `(postal, number)` anchors.
5. **C++ Accelerated RapidFuzz Pairwise Feature Extraction**: 26 domain-specific pairwise features spanning token sort ratios, brand ratios, Jaro-Winkler, acronym cross-matching, geo-state matches/conflicts, and composite confidence.
6. **LightGBM Classifier with Hard Negative Geo-Veto & Per-Country Calibration**: Eliminates cross-state false positive merges with hard negative veto gates and optimizes thresholds directly against macro F_0.5 per country.

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
1. Load the trained LightGBM model (`model_checkpoint.joblib`) with calibrated per-country decision thresholds.
2. Run candidate generation and scoring across the test set for France, US, and India.
3. Stream predictions to `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
4. Automatically run `utils/validate_submission.py` to confirm submission validity.
