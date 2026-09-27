# Amazon ML Challenge 2026: Business Entity Resolution
## Comprehensive Agent Trajectory & System Architecture Documentation

**Challenge:** Amazon ML Challenge 2026 — Multi-Source Business Entity Resolution  
**Target:** Link reference Source 1 (`S1`) entities to zero, one, or multiple records in Source 2 (`S2`) and Source 3 (`S3`).  
**Primary Metric:** Macro-averaged $F_{0.5}$ score per Source 1 entity (Precision weighted $2\times$ over Recall). Singletons (0 matches) score 1.0 if predicted empty, but drop to 0.0 on any false match.

---

## 1. Executive Summary & Timeline of Work

From the inception of the project to the current state-of-the-art iteration, the system evolved through five major architectural milestones:

| Phase | Core Objective | Key Deliverables & Changes | Benchmark / Score |
|---|---|---|---|
| **Phase 1** | Exploratory Data Analysis & Discovery | Zero-leakage country check (0.00%), Indic script discovery (27.5%), leading-zero address variations, singleton analysis (5.58%). | EDA scripts & report |
| **Phase 2** | Validation Benchmarking Framework | Stratified 19,995-query validation split (`val_source1.tsv`, `val_ground_truth.tsv`), local macro $F_{0.5}$ evaluator. | Ground truth benchmark |
| **Phase 3** | Baseline Multi-Channel Pipeline | Brahmi Indic phonetizer, inverted index blocker (IDF + brand prefixes + address anchors), 15 RapidFuzz features, LightGBM model. | Val $F_{0.5} \approx 0.8460$ |
| **Phase 4** | 26-Feature Deep Model & Geo-Veto | Hard Negative Geo-Veto (`geo_state_conflict`), 26 pairwise features, per-country thresholds. Full 1.73M test inference run in 9,764s. | Val $F_{0.5} = 0.8654$<br>LB 1: $0.830$ |
| **Phase 5** | Initial Co-Reference & French Engine | French administrative regions/dept engine, Anti-Mall brand veto, 50k balanced training pairs. | Val $F_{0.5} = 0.8682$<br>LB 2: $\mathbf{0.834}$ |
| **Phase 6** | 1-to-1 Source Cardinality & Dual-Anchor Reranking | Fix for the "Over-matching Precision Leak": Enforce $\le 1$ match per target source ($S_2, S_3$), eliminating false-positive multi-merges (5–11 matches) that collapse $F_{0.5}$. | In Progress (Targeting $>0.92+$) |

---

## 2. Phase 1: Exploratory Data Analysis (EDA) & Discoveries

Before training any model, comprehensive analysis was conducted on all 22 million training and test records:

1. **Zero Cross-Country Leakage (0.00%)**:
   - Analysis of all 2,206,821 ground truth links proved that **no entity in Source 1 ever matches a record from a different country**.
   - **Architectural Decision**: Country-based hard partitioning (`US`, `India`, `France`). Completely isolates the search space, reducing comparison space by ~65% and keeping peak memory under 4 GB.

2. **The Multilingual Indic Script Barrier**:
   - In Indian records, **27.5% of Source 2 and Source 3 names are written in native Indic scripts** (Devanagari, Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam, Oriya, Gurmukhi), whereas Source 1 names are **100% Romanized/English**.
   - Unnormalized string matchers fail completely on these records.
   - **Architectural Decision**: Implemented a unified relative Unicode offset transliteration engine (`normalizer.py`), mapping phonetic phonemes across all Brahmi-derived Indic scripts to Latin characters at microsecond speeds.

3. **Leading-Zero Numeric Variations**:
   - Real-world addresses in India and the US frequently differ by zero padding (`AF-0684` vs `Af-684`, `Office No. 0404` vs `404`, `B-0029` vs `B-29`).
   - **Architectural Decision**: Implemented regular-expression-based leading-zero stripping on all numeric tokens in addresses.

4. **Singleton Distribution & Metric Penalty**:
   - 5.58% of Source 1 entities have zero true matches in Sources 2 and 3.
   - Under Macro $F_{0.5}$, correctly predicting an empty match list for a singleton yields a perfect 1.0. A single false positive match drops the score to 0.0.
   - **Architectural Decision**: Calibrate high singleton decision thresholds ($\tau \ge 0.76$) to protect singletons from false merges.

5. **Unseen Country in Test**:
   - The training set contains only `US` and `India`.
   - The test set introduces an unseen country: `France` (259,452 entities, 15% of test data).

---

## 3. Phase 2: Validation Benchmarking Setup

To ensure strict zero-leakage local validation:
1. Created `benchmark/val/val_source1.tsv` and `benchmark/val/val_ground_truth.tsv`:
   - Sampled 19,995 Source 1 entities stratified by country and match count (singletons, 1-match, 2-match, 3+ match).
   - Validation entities were strictly excluded from the training set.
2. Built `utils/evaluate_local.py` to calculate exact official Macro $F_{0.5}$ scores.

---

## 4. Phase 3 & 4: Core Pipeline Architecture & Initial Results

### 4.1 Candidate Generation (Blocking) — `src/blocking.py`
To reduce the 17.2 trillion pairwise comparison space to a tractable candidate set:
- **Channel 1 (Core Brand IDF Tokens)**: Distinctive business tokens (excluding legal suffixes like `LLC`, `Pvt Ltd`, `SARL`) weighted by Inverse Document Frequency ($\text{IDF} = \frac{120}{\text{DF} + 10}$).
- **Channel 2 (Brand 4-gram Prefix)**: First 4 characters of core brand names.
- **Channel 3 (Acronym & Initials)**: Acronym generation for multi-word business names (e.g., `KEC` $\leftrightarrow$ `Kamla Engineering Corp`).
- **Channel 4 (Character 3-Grams)**: Substring shingles of brand names to tolerate character transposition and typos.
- **Channel 5 (Address Numeric & Locality Anchors)**:
  - `(postal_code, normalized_number)`
  - `(normalized_number, locality_word)`
  - `(normalized_number, brand_word)`

### 4.2 Deep Pairwise Feature Engineering (26 Features) — `src/feature_extractor.py`
Computed for each candidate pair using C++ accelerated RapidFuzz:
- **Name Signals**: Token sort ratio, token set ratio, full Levenshtein ratio, core brand ratio, core token set ratio, Jaro-Winkler similarity, core first word ratio, relative length difference, brand prefix match, acronym/initials cross-match, legal suffix compatibility.
- **Address & Geo Signals**: Target address empty indicator, address token set ratio, address word Jaccard similarity, address word overlap count, city/locality match, numeric token Jaccard similarity, exact number match, postal code match, postal code conflict, geographic state/region match, geographic state/region conflict.
- **Composite Signals**: Brand in address indicator, composite confidence (geometric mean of brand similarity and address similarity), joint number + locality co-occurrence score, target source origin (`S2` vs `S3`).

### 4.3 Classification Model with Hard Negative Geo-Veto — `src/model.py`
- LightGBM Gradient-Boosted Decision Tree classifier.
- **Hard Negative Geo-Veto**: If candidate pair has `geo_state_conflict == 1` and `name_ratio < 0.95`, prediction probability is immediately clamped to `0.0`.
- **Validation Benchmark**: Achieved **0.8654 Macro $F_{0.5}$** (US: 0.8869, India: 0.8332).

### 4.4 Full Test Set Inference Run
- Ran streaming chunked inference across 1,732,544 test queries.
- Completed in 9,764.0s (~2.7 hours).
- Passed official submission validator `validate_submission.py` with zero errors.

---

## 5. Phase 5: Leaderboard Feedback (0.83 Score) & Root Cause Analysis

When evaluated on the competition leaderboard, the submission scored **0.83**. A deep diagnostic identified three primary causes:

### 1. The "Missing Second Match" Mathematical Ceiling ($F_{0.5} = \mathbf{0.8333}$)
- In the ground truth, **over 84% of non-singleton entities match to BOTH Source 2 and Source 3**.
- Evaluating candidates independently against Source 1 with a high threshold ($\tau = 0.76$) resulted in:
  - $S_1 \leftrightarrow S_2$ scored $0.85 \ge 0.76 \implies$ Accepted.
  - $S_1 \leftrightarrow S_3$ had minor address/brand noise and scored $0.68 < 0.76 \implies$ Rejected.
- When an entity has 2 true matches and only 1 is predicted:
  $$\text{Precision} = \frac{1}{1} = 1.0, \quad \text{Recall} = \frac{1}{2} = 0.5$$
  $$F_{0.5} = \frac{1.25 \times 1.0 \times 0.5}{0.25 \times 1.0 + 0.5} = \frac{0.625}{0.75} = \mathbf{0.8333}$$
- Independent candidate scoring artificially caps the score at **0.8333** across all dual-match entities.

### 2. The France Geographic Blind Spot (15% of Test Data)
- France accounts for 259,452 entities (15.0% of test queries).
- The earlier normalizer lacked French administrative regions (`Nouvelle-Aquitaine`, `Hauts-de-France`, `Île-de-France`, etc.) and French department codes (`75` Paris, `69` Lyon, `13` Marseille, etc.).
- Because `state_code` was empty for France, `geo_state_conflict` was always 0. The Hard Negative Geo-Veto never activated for France, allowing cross-department false positive merges that damaged precision.

### 3. Commercial Complexes / Shopping Malls
- Unrelated businesses in commercial centers share identical house numbers and PIN codes.
- Without an explicit brand contradiction veto, identical addresses with different brands produced false merges.

---

## 6. The Refactored Solution: Precision-Anchor Graph Co-Reference Engine

To break past the 0.83 ceiling and target $>0.92+$, the codebase was upgraded with:

### 1. Full French Geographic & Lexical Normalization (`src/normalizer.py`)
- Integrated all 18 French Administrative Regions (`auvergne rhone alpes`, `bourgogne franche comte`, `bretagne`, `centre val de loire`, `corse`, `grand est`, `hauts de france`, `ile de france`, `normandie`, `nouvelle aquitaine`, `occitanie`, `pays de la loire`, `provence alpes cote d azur`, and overseas regions).
- Extracted French department codes from 5-digit postal codes (`postal_code[:2]`, e.g., `d_75`, `d_69`, `d_13`, `d_33`).
- Added French legal suffixes (`sarl`, `sasu`, `sas`, `eurl`, `sci`, `sa`, `snc`, `gie`, `sca`, `scop`, `cie`, `ets`, `ste`) and French address stopwords (`rue`, `boulevard`, `chemin`, `impasse`, `cedex`, `bp`, `bis`, `ter`).
- Cross-department false merges in France are now actively vetoed by the Hard Negative Geo-Veto.

### 2. Transitive Dual-Match Co-Reference Propagation (`src/model.py`)
- Implemented `resolve_entity_matches()`:
  - If an $S_1$ entity matches an $S_2$ candidate with high confidence ($P \ge \tau_{\text{base}}$), it searches the remaining $S_3$ candidates.
  - If an $S_3$ candidate shares the exact address number or postal code with the matched $S_2$ entity and has brand similarity $\ge 65\%$, it is admitted into the matched set.
  - Symmetrically propagates from $S_3 \rightarrow S_2$.
  - Recovers the missing twin match for non-singletons, pushing scores on those entities from 0.833 directly to 1.000, while preserving 1.0 on singletons.

### 3. Anti-Mall Brand Contradiction Veto (`src/model.py`)
- Added gate: If a pair has `core_ratio < 0.45` and `core_token_set_ratio < 0.50` with no acronym match, probability is clamped to `0.0`.
- Eliminates false positive merges from unrelated stores sharing a building or mall address.

### 4. Balanced 50,000-Pair Training & Higher Recall Ceiling (`src/blocking.py`, `src/train_evaluate.py`)
- `max_candidates` increased from 30 to 35 to lift the candidate recall ceiling above 95%.
- Training set expanded to 50,000 stratified entities (25,000 US + 25,000 India).

---

## 7. Phase 6: Submission 2 Diagnostic (0.834 Score) & The Cardinality Collapse Discovery

Following the deployment of the initial co-reference and French geo-engine, the second submission scored **0.834** on the competition leaderboard (an increase from 0.830). A forensic error analysis of the generated prediction file `output/matching_results.tsv` revealed the exact structural bottleneck preventing the model from breaking into 0.90+:

### 1. The "Cardinality Explosion" (Over-matching Precision Collapse)
Inspection of `output/matching_results.tsv` revealed that dozens of thousands of queries were predicting **5, 6, 8, 10, or even 11 matches**:
- `S1-717749279` $\rightarrow$ 10 matches (`S3-926674871,S2-456139444,S3-188429503,S2-768360552,S3-206088667...`)
- `S1-819898451` $\rightarrow$ 11 matches (`S2-324032891,S3-86970259,S2-283043147,S3-751470254...`)
- `S1-628750886` $\rightarrow$ 5 matches (`S2-406813553,S2-696378102,S3-159214046,S3-361673112...`)

### 2. The Mathematical Catastrophe under Macro $F_{0.5}$
In the ground truth:
- A real-world business entity in Source 1 links to **at most ONE record in Source 2** and **at most ONE record in Source 3** (except in $<1\%$ of data where true duplicate records exist).
- When an entity with 2 true matches is predicted with 10 matches (2 true + 8 false positives):
  $$\text{Precision} = \frac{2}{10} = 0.20, \quad \text{Recall} = \frac{2}{2} = 1.0$$
  $$F_{0.5} = \frac{1.25 \times 0.20 \times 1.0}{0.25 \times 0.20 + 1.0} = \frac{0.25}{0.05 + 1.0} = \frac{0.25}{1.05} = \mathbf{0.238}$$
- **A single entity's score collapses from 1.000 down to 0.238**!
- Because Macro $F_{0.5}$ weights Precision **twice as heavily as Recall**, admitting multiple false candidates across common French words (`ecole`, `fils`, `club`, `grain`, `auto`) or Indian trade names completely destroys overall macro precision.

### 3. The 1-to-1 Source Cardinality Solution ($\le 1 \text{ from } S_2, \le 1 \text{ from } S_3$)
To guarantee high precision and eliminate the multi-merge leak:
1. **Source-Partitioned Selection**:
   - For any query $S_1$, separate candidates into $S_2$ candidates and $S_3$ candidates.
   - Select **at most the single best candidate from $S_2$** (highest probability $P \ge \tau_{\text{base}}$).
   - Select **at most the single best candidate from $S_3$** (highest probability $P \ge \tau_{\text{base}}$).
2. **Mutual Co-Reference Twin Recovery**:
   - If an $S_2$ anchor is selected with high confidence ($P \ge \tau_{\text{base}}$) and no $S_3$ candidate passed $\tau_{\text{base}}$, evaluate the top $S_3$ candidate:
     - If it shares the exact address number or postal code and brand similarity $\ge 65\%$, admit **only that single best $S_3$ candidate**.
   - Symmetrically for $S_3 \rightarrow S_2$.
3. **Cardinality Bound**:
   - Every non-singleton entity predicts **either 1 or 2 matches** (never 5, 8, or 11).
   - Precision on non-singletons immediately surges above 95%, pushing Macro $F_{0.5}$ toward **$>0.92+$**.

---

## 7b. CORRECTION (2026-09-27): the Phase 6 premise above is FALSE — superseded by Phase 7

Read-only verification against the full TSVs showed:

- **Multiple links per source are the norm.** Validation link counts are 0:1116, 1:1079, 2:3399, 3:4732, 4:4466, 5:2946, ≥6:2257. That is 72% with ≥3 links. 62.5% of non-singletons have ≥2 S2 links.
  - A ≤1-S2 + ≤1-S3 cap limits even a perfect model to about 0.86.
  - The "cardinality explosion" examples above are mostly TRUE copies: all 4 `Grain & Fils` records sit at 329 Av. de Dunkerque.
- **The real precision leaks:**
  - **Missing target exclusivity.** Each S2/S3 record belongs to at most one S1, yet many S1 share a name, e.g. `Clinique de la Team` at 12 Rue de Vincennes vs `Clinique de la Lest` at 12 Rue Buffon (itself an S1).
  - **Deliberate decoys.** These are unlinked clones with a shifted house number plus a legal-form swap or added word; they cluster on singletons. Test has ~1.9× more unlinked targets per S1 than train.
- **Parsing bugs that zeroed out true matches through the hard vetoes:**
  - state taken by substring (`West Virginia`→VA, `HADLEY CT`, `Michigan City, IN`);
  - 5-digit US/India house numbers treated as postal codes;
  - France `d_33` vs region conflicts;
  - Telangana/Andhra Pradesh swaps;
  - Indic `ल`→`ll`.

## Phase 7: focused rebuild (current code)

The design is summarised in `code/business_entity_resolution/README.md`:

- component-based parsing;
- a CSR hash-index blocker with 13 channels and top-K per source;
- 71 vectorised features, including house-number relations for decoys;
- LightGBM with unlinked negatives weighted by the density ratio r;
- a decision rule of global exclusive assignment plus τ/δ, with sibling/lonely rules and France gates, and no per-source cap;
- thresholds tuned on a density-adjusted F0.5, with exclusivity simulated by model-free, forward-checked competitor pairs;
- `pipeline.py` with a per-country subprocess, atomic chunk cache, `--reassign` and `--smoke`.

Run order: `train_evaluate.py --smoke` → `pipeline.py --smoke` (twice) → `train_evaluate.py` → `pipeline.py` → validator.

---

## 8. Repository Structure & Artifacts

```
student_resource/
├── AGENTS.md                          # Full trajectory and architectural documentation
├── README.md                          # Challenge rules and problem definition
├── Documentation_template.md          # Technical report for submission package
├── requirements.txt                   # Dependency definitions
├── benchmark/
│   ├── check_output_distribution.py   # Diagnostic script for match count distributions
│   ├── benchmark_co_reference.py     # Standalone validation script for co-reference boost
│   └── val/
│       ├── val_source1.tsv           # 19,995 stratified S1 validation entities
│       └── val_ground_truth.tsv      # Validation ground truth match lists
├── code/
│   └── business_entity_resolution/
│       ├── README.md                  # Execution instructions
│       ├── requirements.txt           # Code-specific dependencies
│       ├── model_checkpoint.joblib    # Trained LightGBM model and calibrated thresholds
│       ├── run_in_colab.ipynb         # Google Colab reproduction notebook
│       └── src/
│           ├── __init__.py
│           ├── normalizer.py          # Multilingual text cleaning, Brahmi transliteration & French geo engine
│           ├── blocking.py            # High-recall MultiIndexBlocker candidate generator
│           ├── feature_extractor.py   # 26-dimensional RapidFuzz pairwise feature extractor
│           ├── model.py               # LightGBM classifier with Geo-Veto, Brand-Veto & Co-Reference Propagation
│           ├── pipeline.py            # End-to-end streaming test inference orchestrator
│           └── train_evaluate.py      # Stratified training and validation benchmark
├── output/
│   ├── matching_results.tsv           # Official submission matching file (1,732,544 rows)
│   └── candidate_pairs.tsv            # Official submission candidate file (1,732,544 rows)
└── utils/
    ├── evaluate_local.py              # Local Macro F_0.5 evaluator
    └── validate_submission.py         # Official submission format and ID validator
```

---

## 9. Operational Execution Guide

### Step 1: Model Training & Threshold Optimization
```powershell
.\.venv\Scripts\python.exe -u code/business_entity_resolution/src/train_evaluate.py
```
*Trains on 50,000 balanced pairs, applies Anti-Mall and Geo-Vetoes, tunes country thresholds with transitive propagation, and outputs the updated validation score.*

### Step 2: Full Test Inference (with 1-to-1 Source Cardinality Constraint)
```powershell
Remove-Item output/matching_results.tsv, output/candidate_pairs.tsv -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -u code/business_entity_resolution/src/pipeline.py
```
*Processes France, US, and India queries sequentially in 50k streaming chunks with automatic crash-proof resume.*

### Step 3: Submission Verification
```powershell
.\.venv\Scripts\python.exe utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

### Step 4: Final Submission Packaging
```powershell
Compress-Archive -Path output, code, Documentation_template.md -DestinationPath team_submission.zip -Force
```
