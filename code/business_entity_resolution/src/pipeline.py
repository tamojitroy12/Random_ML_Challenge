"""
Optimized, Resumable End-to-End Pipeline for Business Entity Resolution.
Supports:
  1. Crash-proof auto-resume (detects already scored entities and skips them)
  2. Multi-core parallel feature extraction
  3. Strict country partitioning and streaming TSV writes
  4. Automatic validation via utils/validate_submission.py
"""

import os
import sys
import time
import subprocess
import polars as pl
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.normalizer import clean_string, extract_name_components, extract_addr_components
from src.blocking import MultiIndexBlocker
from src.feature_extractor import compute_pair_features, FEATURE_NAMES
from src.model import MatchingModel, compute_macro_f05


class EntityResolutionPipeline:
    def __init__(self, model_path="code/business_entity_resolution/model_checkpoint.joblib", default_threshold=0.76):
        self.model_path = model_path
        self.default_threshold = default_threshold
        self.model = MatchingModel()
        if os.path.isfile(model_path):
            self.model.load(model_path)
            print(f"Loaded existing model from {model_path} with threshold = {self.model.best_threshold:.3f}")
        else:
            self.model.best_threshold = default_threshold

    def run_inference(self, test_dir="dataset/test", output_dir="output", chunk_size=50000):
        """
        Runs full test set inference country by country with auto-resume support.
        Streams output directly to:
          - output/candidate_pairs.tsv
          - output/matching_results.tsv
        """
        os.makedirs(output_dir, exist_ok=True)
        cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
        match_path = os.path.join(output_dir, "matching_results.tsv")
        
        # Check for existing completed queries (Crash-Proof Resume)
        existing_matched_qids = set()
        if os.path.isfile(match_path):
            print(f"Checking existing output file for resume...")
            with open(match_path, "r", encoding="utf-8") as f:
                header = f.readline()
                for line in f:
                    parts = line.split("\t", 1)
                    if parts and parts[0].strip():
                        existing_matched_qids.add(parts[0].strip())
            print(f"Found {len(existing_matched_qids):,} already processed entities. Resuming...")
        else:
            with open(match_path, "w", encoding="utf-8") as f_match:
                f_match.write("source1_entity_id\tmatched_entity_ids\n")

        if not os.path.isfile(cand_path):
            with open(cand_path, "w", encoding="utf-8") as f_cand:
                f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        print("=" * 60)
        print("INFERENCE STAGE: Running Test Set Predictions...")
        print(f"Decision Threshold: tau = {self.model.best_threshold:.3f}")
        print("=" * 60)
        t_start = time.time()
        
        test_s1_scan = pl.scan_csv(os.path.join(test_dir, "test_source1.tsv"), separator="\t")
        test_countries = [
            row["country"] for row in
            test_s1_scan.group_by("country").len().collect().sort("len").to_dicts()
        ]
        print(f"Test Countries to process: {test_countries}")
        
        total_queries_processed = len(existing_matched_qids)

        for country in test_countries:
            print(f"\n==========================================")
            print(f"Processing Test Country: {country}")
            print(f"==========================================")
            t_country = time.time()
            
            # Load queries for this country
            country_s1 = test_s1_scan.filter(pl.col("country") == country).collect()
            all_qids = country_s1["entity_id"].to_list()
            all_names = country_s1["business_name"].to_list()
            all_addrs = country_s1["business_address"].to_list()
            
            # Filter out queries already processed
            remaining_indices = [i for i, qid in enumerate(all_qids) if qid not in existing_matched_qids]
            
            if not remaining_indices:
                print(f"All {len(all_qids):,} queries for {country} ALREADY COMPLETED! Skipping.")
                continue
                
            qids_to_process = [all_qids[i] for i in remaining_indices]
            names_to_process = [all_names[i] for i in remaining_indices]
            addrs_to_process = [all_addrs[i] for i in remaining_indices]
            num_queries = len(qids_to_process)
            print(f"Resuming {country}: {num_queries:,} remaining queries to process (out of {len(all_qids):,} total).")
            
            # 1. Load Country Target Records
            s2_scan = pl.scan_csv(os.path.join(test_dir, "test_source2.tsv"), separator="\t").filter(pl.col("country") == country)
            s3_scan = pl.scan_csv(os.path.join(test_dir, "test_source3.tsv"), separator="\t").filter(pl.col("country") == country)
            target_df = pl.concat([s2_scan, s3_scan]).select(["entity_id", "business_name", "business_address"]).collect()
            
            target_ids = target_df["entity_id"].to_list()
            target_names = target_df["business_name"].to_list()
            target_addrs = target_df["business_address"].to_list()
            print(f"Loaded {len(target_df):,} target records for {country} in {time.time()-t_country:.2f}s")
            
            # 2. Build Inverted Index Blocker
            blocker = MultiIndexBlocker(max_candidates=25)
            blocker.fit_target_universe(target_ids, target_names, target_addrs)
            tid_to_idx = {tid: i for i, tid in enumerate(target_ids)}
            
            def get_target_rec(idx):
                cn = blocker.cleaned_names[idx]
                _, _, suf = extract_name_components(cn)
                return {
                    'target_id': target_ids[idx],
                    'cleaned_name': cn,
                    'cleaned_addr': blocker.cleaned_addrs[idx],
                    'core_name': blocker.core_names[idx],
                    'name_tokens': blocker.name_tokens_list[idx],
                    'legal_suffixes': suf,
                    'addr_nums': blocker.addr_nums_list[idx],
                    'addr_words': blocker.addr_words_list[idx]
                }

            # 3. Process remaining queries in chunks
            with open(cand_path, "a", encoding="utf-8") as f_cand, open(match_path, "a", encoding="utf-8") as f_match:
                for c_start in range(0, num_queries, chunk_size):
                    c_end = min(c_start + chunk_size, num_queries)
                    qids_chunk = qids_to_process[c_start:c_end]
                    names_chunk = names_to_process[c_start:c_end]
                    addrs_chunk = addrs_to_process[c_start:c_end]
                    
                    t_chunk = time.time()
                    cand_map = blocker.generate_candidates(qids_chunk, names_chunk, addrs_chunk)
                    
                    # Extract features
                    chunk_pairs = []
                    chunk_feats = []
                    
                    for q_idx, qid in enumerate(qids_chunk):
                        cn = clean_string(names_chunk[q_idx])
                        ca = clean_string(addrs_chunk[q_idx])
                        core_name, ntoks, suffixes = extract_name_components(cn)
                        nums, awords = extract_addr_components(ca)
                        q_rec = {
                            'cleaned_name': cn, 'cleaned_addr': ca, 'core_name': core_name,
                            'name_tokens': ntoks, 'legal_suffixes': suffixes,
                            'addr_nums': nums, 'addr_words': awords
                        }
                        
                        cands = cand_map.get(qid, [])
                        for tid in cands:
                            if tid in tid_to_idx:
                                t_rec = get_target_rec(tid_to_idx[tid])
                                feat = compute_pair_features(q_rec, t_rec)
                                chunk_pairs.append((qid, tid))
                                chunk_feats.append(feat)
                                
                    # Score batch with model
                    matched_map = {qid: [] for qid in qids_chunk}
                    if chunk_feats:
                        X_chunk = np.array(chunk_feats, dtype=np.float32)
                        probs = self.model.predict_proba(X_chunk)
                        
                        for i in range(len(chunk_pairs)):
                            score = float(probs[i])
                            if score >= self.model.best_threshold:
                                qid, tid = chunk_pairs[i]
                                matched_map[qid].append(tid)
                                
                    # Write to TSVs
                    for qid in qids_chunk:
                        cand_list = cand_map.get(qid, [])
                        match_list = matched_map.get(qid, [])
                        f_cand.write(f"{qid}\t{','.join(cand_list)}\n")
                        f_match.write(f"{qid}\t{','.join(match_list)}\n")
                        
                    total_queries_processed += len(qids_chunk)
                    print(f"  Processed {c_end:,}/{num_queries:,} {country} queries (chunk time: {time.time()-t_chunk:.1f}s)")
                    
            print(f"Finished {country} in {time.time()-t_country:.1f}s")
            
        print("\n" + "=" * 60)
        print(f"INFERENCE FINISHED in {time.time()-t_start:.1f}s")
        print(f"Total S1 entities processed: {total_queries_processed:,}")
        print("=" * 60)
        
        # 4. Deduplication safeguard for clean validation
        for fpath in [match_path, cand_path]:
            if os.path.isfile(fpath):
                df_chk = pl.read_csv(fpath, separator="\t")
                if df_chk["source1_entity_id"].is_duplicated().any():
                    df_chk = df_chk.unique(subset=["source1_entity_id"], keep="last")
                    df_chk.write_csv(fpath, separator="\t")

        # 5. Run Submission Validator
        print("\nRunning submission validator...")
        val_cmd = [
            sys.executable,
            "utils/validate_submission.py",
            "--matching", match_path,
            "--candidate", cand_path,
            "--test-dir", test_dir
        ]

        res = subprocess.run(val_cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        return res.returncode


def main():
    pipeline = EntityResolutionPipeline()
    pipeline.run_inference()


if __name__ == "__main__":
    main()
