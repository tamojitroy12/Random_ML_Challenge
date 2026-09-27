# Business Entity Resolution Pipeline

**Team Apocalypso:** Anurag Chaudhuri (Team Leader), Tamojit Roy, Arunanshu Dey

Links every Source 1 business to **all** of its Source 2 / Source 3 records. The score is macro F0.5 per S1 entity.

**Expected layout.** Run everything from `student_resource/`, with the challenge data in `dataset/train/` and `dataset/test/` and the official helpers in `utils/`. Our validation split (19,995 train S1 held out, stratified) ships in `val/` next to this README; it is used automatically when `benchmark/val/` is absent. Tested on Windows 11 with Python 3.12, 8 cores and 16 GB RAM: training takes about 23 min and inference about 80 min, with a peak of about 12 GB.

**Shipped artefacts.** `model_checkpoint.joblib`, `thresholds.json` and `native_lexicon.json` are the exact files that produced the submitted outputs. To regenerate `output/` without retraining, run step 3 only (`pipeline.py`).

## Approach
1. **Parse once, compactly** (`normalizer.py`, `records.py`). Names and addresses are parsed in a multiprocessing pool into Arrow-backed columns.
   - **Addresses** are parsed per comma component:
     - State/region is recognised only when a whole component equals it.
     - France departments and cities map to regions.
     - Telangana and Andhra Pradesh are merged.
     - US and India have no postal codes.
   - **House numbers** keep compounds (`104/1/1`), bis/ter/letter suffixes and ranges.
   - **Names:**
     - Alias markers are handled (`X dba Y` → `Y`), as are dotted initialisms, a global legal-form class table, honorifics, leetspeak and domains.
     - Synthetic garbage names are detected.
     - Indic scripts are transliterated with inherent-vowel handling, plus a native→Latin lexicon learned from the training ground truth (`lexicon.py`).
2. **Blocking** (`blocking.py`). A compact CSR inverted index over hashed keys in 13 channels:
   - name tokens, prefixes, phonetic skeletons and 3-grams;
   - acronym↔short name;
   - house number × street/name;
   - number-free name × street and street × city;
   - verbatim and canonical address signatures.

   Weights are IDF-style and capped by document frequency. It returns the top-K per source (S2/S3).
3. **Features** (`feature_extractor.py`). 71 pairwise features computed with `rapidfuzz.process.cpdist` (multithreaded) plus numpy. They cover:
   - house-number relations (exact / digit drop / substitution / compound prefix chain / last-part change / range);
   - street containment and legal-form conflicts;
   - decoy words, duplicate-name and co-tenant counts;
   - alias, garbage, domain and native-script flags.
4. **Model** (`model.py`). LightGBM. Negatives whose target is unlinked are up-weighted to reflect the higher decoy density in test.
5. **Decision rule**:
   - **Exclusivity:** each target goes to at most one S1, its best-scoring one, with a margin δ; ties are dropped.
   - **Threshold:** a per-country τ.
   - **Optional rules:** sibling rescue, sibling suppress and a lonely-link threshold.
   - **France gates:** number+suffix, city and street.
   - There is no per-source cap: an S1 can match several records from the same source.
   - Thresholds are tuned per country on a density-adjusted F0.5, with exclusivity simulated using model-free competitor pairs that mirror test mechanics.

## Setup
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

## Reproduce (run from `student_resource/`)
```powershell
$py = ".\.venv\Scripts\python.exe"
# 1. smoke tests (~10 min total) - catch problems before the long runs
& $py -u code/business_entity_resolution/src/train_evaluate.py --smoke
& $py -u code/business_entity_resolution/src/pipeline.py --smoke        # run twice: 2nd run must resume from cache
# 2. training + validation report (writes model_checkpoint.joblib, thresholds.json, native_lexicon.json)
& $py -u code/business_entity_resolution/src/train_evaluate.py
# 3. test inference (resumable; writes output/matching_results.tsv and output/candidate_pairs.tsv)
& $py -u code/business_entity_resolution/src/pipeline.py
# optional: new thresholds from cached scores in minutes, e.g. a different France tau
& $py -u code/business_entity_resolution/src/pipeline.py --reassign --tau France=0.85
```
Useful flags:
- `train_evaluate.py`:
  - `--stage select|lexicon|passA|train|tune` runs one stage (all stages resume from `output/_work/train`).
  - `--fresh` recomputes everything.
  - `--n-train N` sets the TRAIN_S1 size per country.
  - `--workers N` sets the number of parser processes.
- `pipeline.py`:
  - `--countries India,US` limits the run to those countries.
  - `--fresh` discards cached scores.
  - `--tau/--delta/--tau1` take `US=..,India=..,France=..` or a single value.
  - `--fr-gates on|off` switches the France gates.
