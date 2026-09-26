# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Evaluation Metric:** Macro $F_{0.5}$ Score (Precision-Weighted)  

---

## 1. Executive Summary
We present a high-performance, multi-stage Business Entity Resolution framework designed to resolve noisy business records across three independent sources (`S1`, `S2`, `S3`) spanning over 11 million records. Our approach combines **strict zero-leakage country partitioning**, a **unified Brahmi Indic transliteration engine**, and a **high-recall multi-channel inverted index blocker** that prunes 17.2 trillion pairwise comparisons by **99.999%**. Using a 26-dimensional C++ accelerated RapidFuzz feature pipeline, a **LightGBM gradient-boosted decision tree** with **Hard Negative Geo-Veto** scores candidate pairs with country-calibrated decision thresholds ($\tau_{\text{US}} = 0.760, \tau_{\text{India}} = 0.780$) optimized directly for macro $F_{0.5}$, achieving an outstanding **validation macro $F_{0.5}$ score of 0.8654** (US: 0.8869, India: 0.8332).

---

## 2. Methodology

### 2.1 Problem Analysis
During comprehensive exploratory data analysis (EDA) across the 22 million training and test records, four key insights were discovered:
1. **Zero Cross-Country Leakage**: Across all 2,206,821 ground truth links in the training data, exactly **0.00%** crossed country borders. Matches are strictly intra-national.
2. **Multilingual Indic Script Barrier**: In Indian records, **27.5% of Source 2 and Source 3 business names are written in native Indic scripts** (Devanagari, Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam, Oriya, Gurmukhi), whereas Source 1 names are **100% Romanized/English**. Unnormalized string matching completely fails on over a quarter of Indian records.
3. **Leading-Zero Address Variations**: Addresses frequently differ by leading zeros in street, house, or plot numbers (e.g., `AF-0684` vs `Af-684`, `Office No. 0404` vs `404`, `B-0029` vs `B-29`).
4. **Asymmetric Precision Penalty**: The evaluation metric is macro $F_{0.5}$, where precision is weighted twice as heavily as recall ($\beta = 0.5$). False merges severely penalize the score, and singletons (5.58% of entities with no match) earn a full 1.0 if predicted empty but drop to 0.0 if even a single false match is made.

### 2.2 Solution Strategy
- **Approach Type**: Multi-Stage Hybrid: Country Partitioning $\rightarrow$ Multilingual Normalization $\rightarrow$ Inverted Index Blocking $\rightarrow$ C++ Pairwise Feature Extraction $\rightarrow$ LightGBM Ranking with Hard Negative Geo-Veto & Asymmetric $F_{0.5}$ Threshold Optimization.
- **Core Innovations**:
  - **Unified Brahmi Transliteration**: Exploits the identical relative Unicode offset architecture of Indic alphabets to phonetically map Hindi, Tamil, Telugu, Bengali, etc., into phonetic Latin script at microsecond speeds.
  - **Zero-Stripped Numeric Address Anchors**: Normalizes house numbers and PIN codes, connecting noisy street references even when business names differ (trade names / DBAs).
  - **Hard Negative Geo-Veto**: Explicit state and postal contradiction gates vetoing cross-state false positive merges.
  - **Country-Specific $F_{0.5}$ Calibration**: Precision-biased threshold tuning specifically targeted at maximizing singleton credit and eliminating false merges per geography.

---

## 3. Candidate Generation (Blocking)

### 3.1 Blocking Strategy & Keys Used
To reduce the $1.73 \times 10^6 \times 9.97 \times 10^6 \approx 17.2 \text{ Trillion}$ comparison space to a tractable candidate set, we construct an inverted index blocker per country partition:
1. **Core Brand Name Words (Channel A)**: Distinctive business name tokens (excluding legal suffixes like `LLC`, `Pvt Ltd`, `SARL`) weighted by Inverse Document Frequency ($\text{IDF} = \frac{100}{\text{DF} + 10}$).
2. **Brand 4-gram Prefix (Channel B)**: First 4 characters of the core brand name to capture typos, suffixes, and inflections (e.g., `payn`, `dahl`, `maur`).
3. **Character 3-Grams (Channel C)**: Interior tri-grams of brand names to withstand heavy transliteration variations and typos.
4. **Acronym & Initials (Channel D)**: First letters of multi-word business names (e.g., `KEC` $\leftrightarrow$ `Kamla Engineering Corp`).
5. **Address Anchors (Channel E)**: Tuple keys combining normalized numbers and locality words: `(normalized_number, locality_word)` and `(postal_code, normalized_number)` to capture trade names/acronyms (e.g., `EC` $\leftrightarrow$ `East Consultants` via `(1-13-14, vinayaknagar)`).

### 3.2 Candidate Set Statistics
- **Candidate space reduction ratio**: $> 99.999\%$
- **Average candidate pairs per query**: $\approx 22 - 25$
- **Candidate retrieval latency**: $\approx 0.2 - 0.5 \text{ ms}$ per query
- **Blocking Recall**: $> 92\%$ on true matches

---

## 4. Matching Model

### 4.1 Feature Engineering (26 Pairwise Features)
For each candidate pair `(Query, Target)`, we compute a 26-dimensional feature vector using C++ accelerated RapidFuzz:
- **Name Features**:
  1. `name_token_sort_ratio`: Word-order invariant similarity.
  2. `name_token_set_ratio`: Substring / set overlap similarity.
  3. `name_ratio`: Full string Levenshtein ratio.
  4. `core_ratio`: Ratio on core brand name (excluding legal suffixes).
  5. `core_token_set_ratio`: Token set ratio on core brand name.
  6. `jaro_winkler_sim`: Prefix-sensitive string similarity.
  7. `core_first_word_ratio`: Ratio of the primary brand anchor token.
  8. `name_length_diff`: Relative length disparity.
  9. `prefix_match`: Binary indicator if first 4 characters match.
  10. `acronym_match`: Binary indicator of acronym / initials equivalence.
  11. `suffix_compatible`: Tristate compatibility score for legal forms (LLC vs Corp vs Pvt Ltd).
- **Address & Geo Features**:
  12. `target_addr_empty`: Indicator if target address is missing.
  13. `addr_token_set_ratio`: RapidFuzz ratio on address strings.
  14. `addr_token_jaccard`: Jaccard similarity of distinctive address words.
  15. `addr_token_overlap`: Integer count of shared address words.
  16. `locality_match`: Match indicator between parsed city/locality tokens.
  17. `addr_num_jaccard`: Jaccard similarity of address numeric tokens.
  18. `exact_number_match`: Binary indicator if house number or PIN code matches exactly.
  19. `postal_code_match`: Exact postal code match flag.
  20. `postal_code_conflict`: Contradictory postal codes indicator.
  21. `geo_state_match`: State code equality indicator.
  22. `geo_state_conflict`: Conflicting state code indicator (Hard Veto trigger).
- **Composite & Metadata Features**:
  23. `name_in_addr`: Binary indicator if brand name appears within the address field.
  24. `composite_confidence`: Geometric mean of name similarity and address similarity.
  25. `num_locality_hit`: Joint numeric and locality co-occurrence score.
  26. `target_source`: Categorical indicator of candidate origin (Source 2 vs Source 3).

### 4.2 Model Architecture & Threshold Selection
- **Model Type**: LightGBM Classifier (400 trees, max depth 7, 45 leaves, learning rate 0.05, L1/L2 regularization).
- **Hard Negative Geo-Veto**: Candidate pairs with confirmed state conflicts (`geo_state_conflict == 1`) whose name ratio is below 0.95 are immediately vetoed ($\text{score} \leftarrow 0.0$), eliminating false cross-state merges.
- **Threshold Selection**: Calibrated per country on a stratified 20,000-query validation set:
  - $\tau_{\text{US}} = 0.760$
  - $\tau_{\text{India}} = 0.780$
  - $\tau_{\text{France}} = 0.760$

---

## 5. Results & Error Analysis

### 5.1 Validation Performance
- **Validation Dataset**: 19,995 stratified Source 1 entities with ground truth (covering US and India, with natural singleton distribution).
- **Overall Macro $F_{0.5}$ Score**: **0.8654**
  - **US Macro $F_{0.5}$**: **0.8869**
  - **India Macro $F_{0.5}$**: **0.8332**
- **Optimal Decision Thresholds**: $\tau_{\text{US}} = 0.760, \tau_{\text{India}} = 0.780$

### 5.2 Error Analysis
- **False Positives (Wrong Merges)**: Dramatically reduced by Hard Negative Geo-Veto. Remaining false positives occur in national retail chains that share exact street addresses in multi-tenant commercial malls.
- **False Negatives (Missed Matches)**: Occur predominantly when Source 2/3 records have missing addresses combined with phonetic drift in regional Indian brand transliterations.

---

## 6. Conclusion
By systematically addressing the underlying domain challenges — multilingual Indic scripts, zero-leakage country partitioning, numeric address zero-padding, and cross-state geo-vetoing — our hybrid blocking and gradient boosting pipeline scales seamlessly to over 10 million records while keeping peak memory under 4 GB. The precision-weighted thresholding achieves an outstanding **0.8654 validation macro $F_{0.5}$**, delivering both industrial scalability and competitive leaderboard performance.

---

## Appendix: Code Artefacts
All reproduction code is packaged under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── README.md               # End-to-end execution guide
├── requirements.txt        # Pinned dependencies
├── model_checkpoint.joblib # Trained LightGBM model & calibrated thresholds
└── src/
    ├── __init__.py
    ├── normalizer.py       # Multilingual text cleaning & Brahmi transliteration
    ├── blocking.py         # MultiIndexBlocker inverted index candidate generator
    ├── feature_extractor.py# 26-dimensional RapidFuzz pairwise feature extractor
    ├── model.py            # LightGBM classifier & threshold optimizer with Geo-Veto
    ├── pipeline.py         # End-to-end streaming orchestrator
    └── train_evaluate.py   # Training and validation benchmark script
```
**Entry Point to Reproduce**:
```bash
python code/business_entity_resolution/src/pipeline.py
```
This automatically produces `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs `utils/validate_submission.py`.
