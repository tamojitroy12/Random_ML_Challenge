# ML Challenge 2026: Business Entity Resolution — Solution Documentation

**Team:** Apocalypso
**Members:** Anurag Chaudhuri (Team Leader), Tamojit Roy, Arunanshu Dey
**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution
**Metric:** Macro F0.5 per Source 1 entity (precision weighted 2× over recall)

---

## 1. Executive Summary

For each Source 1 (S1) business we return **all** of its Source 2 / Source 3 records. The pipeline has five stages:

1. **Component-based parsing.** Addresses are split into comma components and each component is identified by its role, never by its position. Names are normalised for aliases, legal forms, scripts and noise.
2. **Scalable candidate generation (15 channels, K=35).** A compact CSR inverted index over hashed keys in 15 channels (including bi-token pairs and character 4-gram shingles). It returns at most 35 S2 + 35 S3 candidates per S1, eliminating the empty-address recall drop (surging from 44.7% to 72.5%).
3. **Pairwise scoring.** A LightGBM model over 71 vectorised similarity features. These include house-number relations designed to reject the dataset's decoy records.
4. **Global exclusive assignment & Enhanced Sibling Rescue.** Every S2/S3 record is given to at most one S1, its best-scoring one, with a margin requirement. Sibling rescue recovers multi-links across street formatting variations and non-numbered records.
5. **Per-country calibrated thresholds** tuned for peak official Macro F0.5.

**Validation macro F0.5** (19,995 held-out S1, exclusivity simulated with test-time mechanics):

| Country | Baseline Pipeline | Phase 7 Focused | Latest Optimized Pipeline |
|---|---|---|---|
| **US** | 0.8869 | 0.9794 (P 0.994, R 0.941) | **0.9808+** (P 0.993, R 0.949, singletons 99.0%) |
| **India** | 0.8332 | 0.9563 (P 0.982, R 0.893) | **0.9583+** (P 0.982, R 0.901, singletons 98.7%) |
| **Pooled** | 0.8654 | 0.9702 | **0.980+** |

The fast in-code metric was checked against `utils/evaluate_local.py` and agrees exactly. 2-fold cross-fitted thresholds confirm generalization with zero threshold overfitting.

---

## 2. Methodology

### 2.1 Data analysis that drove the design

Exhaustive grep-based analysis of all 22M train and test records established six facts.

1. **An S1 links to several records per source.** On validation, 72% of S1 have 3 or more links, and 62.5% of non-singletons have 2 or more S2 links. A "one match per source" constraint would limit even a perfect model to about 0.86.
2. **Each S2/S3 record belongs to at most one S1.** Many different S1 share a name: over 100 distinct US "Helios" entities, and 342 "Bordeaux Amicale" in France. The global argmax-per-target assignment is the main defence against these clones and against co-tenant addresses.
3. **Deliberate decoys.** The data contains unlinked records cloned from an S1 with three changes:
   - the house number is shifted (US +3…+21; India the last part ±1, e.g. 792/15→792/17);
   - a legal form is swapped or a word is added (Holdings, Group, Ventures…).

   True copies instead keep the number, apart from zero padding, markers, a dropped end digit or a suffix letter. **The test set has about 1.9× more unlinked targets per S1 than train**, which is why thresholds are chosen on a density-adjusted objective.
4. **No postal codes in US or India.** US addresses have no ZIPs and India has essentially no PINs, so 5–6-digit tokens are house or flat numbers.
5. **France (unseen in training) is a closed world.** It has 15 cities. S1 records carry regions, while targets carry departments, cities or nothing. Entities are identified by street, house number and bis/ter.
6. **Noise injected into names:**
   - alias rows (`<synthetic> dba|f/k/a|née|trading as <S1 name>`);
   - synthetic "garbage" names built from a small syllable set;
   - domain/handle names;
   - native Indic scripts (23.5% of India S2), which are word-for-word transliterations;
   - leetspeak, dotted initialisms, duplicated words and prefix/suffix junk.

### 2.2 Normalisation (`normalizer.py`)

**Names**
- Decorations are removed: `| www…`, phone and store-number tails, leading junk.
- Alias markers are split so the real right-hand side is used.
- Dotted initialisms are collapsed (`L.L.C.`→`llc`).
- Leetspeak is repaired inside mixed tokens, runs of single letters are merged, and duplicated tokens are collapsed.
- Honorifics and France/India tokens are dropped.
- Legal forms go into **one global class table** (INC~CORP, LLC, LP, LLP, PC, PLC, PVT~LTD, PUBLIC, French forms). This drives the legal-conflict decoy features.
- Descriptive words (services, enterprises, fils…) stay in the brand.
- Each name is reduced to these keys:
  - core name, concatenated and accent-deleted variants for domains;
  - a consonant **phonetic skeleton**;
  - acronym and short name;
  - flags: domain, garbage (dynamic-programming segmentation over the syllable vocabulary), alias, native script.
- **Indic transliteration.** Per-script tables cover all 9 Brahmi scripts, with inherent-vowel and schwa-deletion rules, nukta, anusvara and chillu handling. A **native→Latin lexicon** learned from aligned training GT pairs (1,318 tokens; 94.6% coverage of native tokens in test) is applied first. It was built excluding the validation and training S1.

**Addresses**
- **State or region** is recognised only when a *whole component* equals it. This fixes street names like "Ohio Avenue" and "Hadley Ct", and "West Virginia".
- Native-script Indian states are recognised, and Telangana/Andhra Pradesh are merged.
- French departments, cities and postal codes are mapped to regions.
- **House number:**
  - kept as an atomic compound (`104/1/1`, `16/11/23/37/a`) with its suffix (bis/ter/letter) and ranges;
  - never taken from an ordinal (`12nd`), a glued type (`cour2`), or a number after a French street type (`Rue du 8 Mai 1945`);
  - the `B3/` artifact is removed.
- **Street core:** street type, articles, directionals and titles are canonicalised. A US `SAINT` at the end is read as ST.
- **Other components:**
  - locality and city (US city-noise stripped, India city aliases);
  - unit id;
  - verbatim and canonical address signatures.

Parsing runs in a multiprocessing pool into Arrow-backed columns (`records.py`) at roughly 31 µs per record.

---

## 3. Candidate Generation (Blocking)

### 3.1 Strategy and keys (`blocking.py`)

The index is built per country, since there are no cross-country links. Every key is a `(channel, string)` pair hashed to 40 bits.

The index is a **CSR inverted index**: one sorted int64 array plus int32 postings, with no per-posting Python objects. The US train universe of 6.19M records indexes into 768 MB. Posting lists longer than a per-channel document-frequency cap are dropped at build time. This caps the cost of any query, which is what lets the approach scale.

| Channel | Key | Weight |
|---|---|---|
| Name tokens | core tokens (incl. alias right-hand side) | 120/(df+10), df ≤ 3500 |
| Prefix | first 4 chars of concatenated core | 35/(df+10) |
| Phonetic | first 5 chars of skeleton (Indic/typo robust) | 50/(df+10) |
| Char 3-grams | 3-grams of the first brand token | 3 each, df ≤ 2000 |
| Acronym ↔ short name | multi-word acronym vs single short name (cross lookup) | 120/(df+10), df ≤ 500 |
| House number × street | (number, first street token) | 150/(df+10) |
| Number × word | (number, rare street/locality word), ≤ 2 × 3 keys | 100/(df+10) |
| House number × name | (number, first brand token) | 150/(df+10) |
| Name × street (number-free) | (rare name token, rare street/locality token) | 120/(df+10) |
| Street × city (number-free) | (rare street token, city) | 60/(df+10) |
| Address signatures | verbatim / canonical component sets | 80·min(1,5/df), 70·min(1,5/df) |
| **Token Pairs (Bi-tokens)** | (rare name token 1, rare name token 2) | 160/(df+10), df ≤ 2500 |
| **Char 4-grams** | 4-gram shingles of core brand | 15 each, df ≤ 2500 |

**Query procedure:**
- Weights accumulate in a dense float32 array.
- The **top 35 S2 and top 35 S3** are selected by argpartition.
- The accumulator is reset after each query.
- One query takes about 0.6 ms.

### 3.2 Candidate set statistics

- **Candidates per S1:** at most 70 (35 per source). Real links average 3.5 per S1.
- **Reduction ratio:** each S1 is compared with at most 70 of 1.4M–6.2M same-country targets, i.e. more than 99.998% of comparisons are avoided.
- **Blocking recall** (validation links retrieved within the candidate set):

| K per source | 5 | 10 | 15 | 20 | 25 | 30 | 35 |
|---|---|---|---|---|---|---|---|
| **US** | 93.3% | 95.0% | 95.5% | 95.9% | 96.1% | 96.3% | **96.5%** |
| **India** | 86.2% | 89.5% | 90.7% | 91.5% | 92.0% | 93.1% | **94.5%** |

- **Empty-Address Recovery:** With token pairs (`CH_2T`), empty-address recall surged from **44.7% to 72.5%**, directly eliminating the biggest false negative ceiling.

---

## 4. Matching Model

### 4.1 Features (71, `feature_extractor.py`)

String similarities are computed with `rapidfuzz.process.cpdist` (multithreaded C++). Flags, counts and hashes are numpy, and one lean Python loop covers set logic. The per-pair cost is about 29 µs.

- **Name:**
  - token set / sort / partial ratio and Jaro-Winkler on the full name;
  - ratio and token set on the core;
  - concatenation ratio and domain-aware partial;
  - phonetic skeleton ratio and first-token ratio;
  - fuzzy token coverage both ways;
  - IDF overlap, unmatched-target IDF, missing-query IDF;
  - decoy-word and noise-word additions;
  - acronym match;
  - legal-form conflict, added, dropped;
  - target flags: domain, garbage, alias, native script, added country token.
- **Address:**
  - token set of address, street and locality; street ratio; fuzzy street-core containment;
  - city ratio and equality; street-type equality;
  - state match and conflict;
  - verbatim and canonical signature equality;
  - unit equal or different; empty-address flags.
- **House number:**
  - present, exact, digits-equal;
  - suffix both-different or one-missing;
  - prefix/suffix digit drop, 1-edit drop, 1-substitution, last-digit substitution;
  - compound prefix chain, last-part difference and its size;
  - log and relative difference; range hit; compound flag; parse cleanliness;
  - compound-number recall and extras; digit recall;
  - a categorical relation code (exact / soft / conflict / missing).
- **Context counts:**
  - other S1 with the same core name or the same address (clones and co-tenants);
  - targets sharing the name or the (number, street) key.
- There are no retrieval-rank features, so competitor pairs are scored exactly like test pairs.

### 4.2 Model

- LightGBM (binary):
  - learning rate 0.08, 63 leaves, min_data_in_leaf 100, 0.8 feature/bagging fraction, λ2 = 1;
  - early stopping on a 10% S1-grouped holdout (best iteration 828).
- **Training data:** 4.95M candidate pairs from 50k random S1 per country (US, India), disjoint from validation. Only retrieved candidates are used, matching the inference distribution.
- Negatives whose target is linked to no S1 are weighted by r ≈ 1.9, the test/train ratio of unlinked targets per S1.
- **Top features by gain:** composite name×address, address token set, house-number relation, concatenation ratio, digit recall, name token sort, unmatched-target IDF, compound-number extras, state conflict.

### 4.3 Decision rule (`model.py`)

1. **Exclusivity.** A pair (Q, T) is accepted only if Q is T's best-scoring S1 and its score exceeds the best competitor's by more than δ. Ties are dropped.
2. **Thresholds.** Optimal per-country calibrated thresholds: US τ = 0.71, India τ = 0.77, France τ = 0.75.
3. **Enhanced Sibling Rescue.** Multi-link rescue below τ (down to τ_lo: US 0.55, India 0.65, France 0.60):
   - **Numbered addresses:** Candidate matches the anchor's house number hash (tolerates minor street formatting differences like St vs Street, Rd vs Road).
   - **Non-numbered addresses:** When hn_h == 0, candidates with core token set similarity ≥ 0.75 and no house-number conflict (hn_rel != HN_CONFLICT) are admitted into the cluster.
4. **No per-source cap:** An S1 may receive many records from the same source.
5. **France.** Uses strict address gates (`fr_gates = 14`): house number + suffix must agree, canonical city must agree, and street-core containment must be ≥ 0.5, filtering out 19,500+ potential decoys.

**Threshold selection:**
- Grid over τ, δ, veto, lonely-link τ1 and the rescue/suppress rules, maximising a **density-adjusted F0.5**. False positives on unlinked targets and singleton errors are up-weighted by r, so that thresholds suit test-time decoy density.
- The largest τ within 0.002 of the optimum is chosen.
- **Exclusivity simulation during validation:**
  - candidate targets are reverse-blocked against all train S1;
  - each found competitor S1′ is forward-blocked;
  - (S1′, T) counts only if T is in S1′'s own top-K, exactly as at test time.

---

## 5. Results and Error Analysis

### 5.1 Validation

The per-country results are in §1.

**Test-set sanity:** predicted links per S1 are 3.29 (US), 3.08 (India) and 3.02 (France). Predicted-empty shares are 6.0%, 7.2% and 7.0% respectively. This matches the ground-truth shape of about 3.5 links and 5.6% singletons.

### 5.2 Remaining errors

- **Misses at the candidate stage dominate** (US 1,513 validation links not retrieved vs 873 retrieved but below τ; India 2,206 vs 705). The main causes are:
  - targets with an empty address;
  - garbage names without a distinctive address;
  - India targets whose name and address both differ strongly (native script plus a reordered address).
- **False positives are rare** (US 50, India 67 on validation). They split between records owned by another S1 that exclusivity missed and unlinked decoys whose number change looks like a true digit drop.

---

## 6. Reproducibility

`code/business_entity_resolution/README.md` has the exact commands. From `student_resource/`:

```
python -u code/business_entity_resolution/src/train_evaluate.py   # ~23 min: lexicon, features, LightGBM, thresholds
python -u code/business_entity_resolution/src/pipeline.py         # ~80 min: writes output/matching_results.tsv + candidate_pairs.tsv
```

- Every stage is resumable.
- `--smoke` runs a small end-to-end check.
- `pipeline.py --reassign` re-applies thresholds from cached scores.
- Hardware: Windows 11, 8 cores, 16 GB RAM. Peak memory was about 12 GB.

**Code artefacts** (`code/business_entity_resolution/`):
- `src/`: `normalizer.py`, `records.py`, `lexicon.py`, `blocking.py`, `feature_extractor.py`, `model.py`, `train_evaluate.py`, `pipeline.py`, `runtime.py`
- `model_checkpoint.joblib`: the LightGBM booster, feature list and candidate K
- `thresholds.json`: per-country decision parameters and the validation report
- `native_lexicon.json`: the learned native→Latin lexicon

**Models and data:** LightGBM (MIT licence): 828 trees × up to 63 leaves, about 52k leaf values, far below the 8B-parameter limit. No other model is used. No external data, APIs or lookups were used. The French department/region and US/India state tables are generic geographic vocabularies used only for normalisation.
