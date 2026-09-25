"""
Training and Validation Pipeline for Business Entity Resolution.
Trains LightGBM on generated candidate pairs, optimizes the F_0.5 decision threshold,
and evaluates full macro F_0.5 performance on held-out validation queries.
"""

import os
import sys
import time
import polars as pl
import numpy as np

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.normalizer import clean_string, extract_name_components, extract_addr_components
from src.blocking import MultiIndexBlocker
from src.feature_extractor import compute_pair_features, FEATURE_NAMES
from src.model import MatchingModel, compute_macro_f05


def run_training_and_validation(train_dir="dataset/train", val_dir="benchmark/val", model_save_path="model_checkpoint.joblib"):
    print("=" * 60)
    print("STAGE 1: Loading Datasets...")
    print("=" * 60)
    t0 = time.time()
    
    # 1. Load validation queries and ground truth
    val_s1 = pl.read_csv(os.path.join(val_dir, "val_source1.tsv"), separator="\t")
    val_gt = pl.read_csv(os.path.join(val_dir, "val_ground_truth.tsv"), separator="\t")
    
    val_gt_dict = {}
    for r in val_gt.iter_rows(named=True):
        mids = {m.strip() for m in r["matched_entity_ids"].split(",") if m.strip()} if r["matched_entity_ids"] else set()
        val_gt_dict[r["source1_entity_id"]] = mids
        
    print(f"Validation Set: {len(val_s1):,} S1 queries loaded.")
    
    # 2. Select a high-quality training sample (e.g. 25,000 distinct S1 queries from train)
    val_s1_ids = set(val_s1["entity_id"])
    train_gt_full = pl.scan_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t")
    train_s1_full = pl.scan_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t")
    
    train_sample_s1 = (
        train_s1_full
        .filter(~pl.col("entity_id").is_in(val_s1_ids))
        .head(25000)
        .collect()
    )
    train_sample_ids = set(train_sample_s1["entity_id"])
    
    train_gt_sample = (
        train_gt_full
        .filter(pl.col("source1_entity_id").is_in(train_sample_ids))
        .collect()
    )
    
    train_gt_dict = {}
    for r in train_gt_sample.iter_rows(named=True):
        mids = {m.strip() for m in r["matched_entity_ids"].split(",") if m.strip()} if r["matched_entity_ids"] else set()
        train_gt_dict[r["source1_entity_id"]] = mids
        
    print(f"Training Sample: {len(train_sample_s1):,} S1 queries selected.")
    
    # 3. Load Target Universe partitioned by Country (US and India)
    countries = ["US", "India"]
    
    train_X = []
    train_y = []
    
    val_candidates_map = {}
    val_pair_records = [] # (qid, tid, features)
    
    for country in countries:
        print("\n" + "-" * 50)
        print(f"Processing Country: {country}")
        print("-" * 50)
        
        # Load S2 and S3 for this country
        t_country = time.time()
        s2_scan = pl.scan_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t").filter(pl.col("country") == country)
        s3_scan = pl.scan_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t").filter(pl.col("country") == country)
        target_df = pl.concat([s2_scan, s3_scan]).select(["entity_id", "business_name", "business_address"]).collect()
        
        target_ids = target_df["entity_id"].to_list()
        target_names = target_df["business_name"].to_list()
        target_addrs = target_df["business_address"].to_list()
        print(f"Loaded {len(target_df):,} target records for {country} in {time.time()-t_country:.2f}s")
        
        # Build Blocker
        blocker = MultiIndexBlocker(max_candidates=25)
        build_time = blocker.fit_target_universe(target_ids, target_names, target_addrs)
        print(f"Multi-index blocker built in {build_time:.2f}s")
        
        # Build target record lookup cache for feature extraction
        target_cache = {}
        for i in range(len(target_ids)):
            tid = target_ids[i]
            target_cache[tid] = {
                'target_id': tid,
                'cleaned_name': blocker.cleaned_names[i],
                'cleaned_addr': blocker.cleaned_addrs[i],
                'core_name': blocker.core_names[i],
                'name_tokens': blocker.name_tokens_list[i],
                'legal_suffixes': set(extract_name_components(blocker.cleaned_names[i])[2]),
                'addr_nums': blocker.addr_nums_list[i],
                'addr_words': blocker.addr_words_list[i]
            }

        # --- A. Generate Training Pairs ---
        country_train_s1 = train_sample_s1.filter(pl.col("country") == country)
        c_train_qids = country_train_s1["entity_id"].to_list()
        c_train_names = country_train_s1["business_name"].to_list()
        c_train_addrs = country_train_s1["business_address"].to_list()
        
        print(f"Generating training candidates for {len(c_train_qids):,} queries...")
        train_cand_map = blocker.generate_candidates(c_train_qids, c_train_names, c_train_addrs)
        
        for q_idx, qid in enumerate(c_train_qids):
            cn = clean_string(c_train_names[q_idx])
            ca = clean_string(c_train_addrs[q_idx])
            core_name, ntoks, suffixes = extract_name_components(cn)
            nums, awords = extract_addr_components(ca)
            q_rec = {
                'cleaned_name': cn,
                'cleaned_addr': ca,
                'core_name': core_name,
                'name_tokens': ntoks,
                'legal_suffixes': suffixes,
                'addr_nums': nums,
                'addr_words': awords
            }
            
            true_targets = train_gt_dict.get(qid, set())
            cands = train_cand_map.get(qid, [])
            
            # Ensure true targets are included in training if in target_cache
            candidate_pool = set(cands)
            for tt in true_targets:
                if tt in target_cache:
                    candidate_pool.add(tt)
                    
            for tid in candidate_pool:
                if tid in target_cache:
                    t_rec = target_cache[tid]
                    feat = compute_pair_features(q_rec, t_rec)
                    label = 1 if tid in true_targets else 0
                    train_X.append(feat)
                    train_y.append(label)

        # --- B. Generate Validation Candidates ---
        country_val_s1 = val_s1.filter(pl.col("country") == country)
        c_val_qids = country_val_s1["entity_id"].to_list()
        c_val_names = country_val_s1["business_name"].to_list()
        c_val_addrs = country_val_s1["business_address"].to_list()
        
        print(f"Generating validation candidates for {len(c_val_qids):,} queries...")
        val_cand_submap = blocker.generate_candidates(c_val_qids, c_val_names, c_val_addrs)
        val_candidates_map.update(val_cand_submap)
        
        for q_idx, qid in enumerate(c_val_qids):
            cn = clean_string(c_val_names[q_idx])
            ca = clean_string(c_val_addrs[q_idx])
            core_name, ntoks, suffixes = extract_name_components(cn)
            nums, awords = extract_addr_components(ca)
            q_rec = {
                'cleaned_name': cn,
                'cleaned_addr': ca,
                'core_name': core_name,
                'name_tokens': ntoks,
                'legal_suffixes': suffixes,
                'addr_nums': nums,
                'addr_words': awords
            }
            
            cands = val_cand_submap.get(qid, [])
            for tid in cands:
                if tid in target_cache:
                    t_rec = target_cache[tid]
                    feat = compute_pair_features(q_rec, t_rec)
                    val_pair_records.append((qid, tid, feat))

    print("\n" + "=" * 60)
    print(f"STAGE 2: Training LightGBM Model on {len(train_X):,} candidate pairs...")
    print(f"Positive pairs: {sum(train_y):,} ({sum(train_y)/len(train_y)*100:.2f}%)")
    print("=" * 60)
    
    train_X = np.array(train_X, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.int32)
    
    model = MatchingModel(n_estimators=250, learning_rate=0.07, max_depth=6, num_leaves=31)
    model.fit(train_X, train_y)
    print("LightGBM training complete!")
    
    # Feature importances
    importances = model.clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop Feature Importances:")
    for idx in sorted_idx[:8]:
        print(f"  {FEATURE_NAMES[idx]:25}: {importances[idx]:.1f}")
        
    print("\n" + "=" * 60)
    print(f"STAGE 3: Scoring Validation Pairs and Optimizing Threshold...")
    print("=" * 60)
    
    val_X = np.array([r[2] for r in val_pair_records], dtype=np.float32)
    val_scores = model.predict_proba(val_X)
    
    val_cand_scores = [
        (val_pair_records[i][0], val_pair_records[i][1], float(val_scores[i]))
        for i in range(len(val_pair_records))
    ]
    
    best_tau, best_f05 = model.optimize_threshold(val_cand_scores, val_gt_dict, val_candidates_map)
    print(f"\nOPTIMAL DECISION THRESHOLD: tau = {best_tau:.3f}")
    print(f"VALIDATION MACRO F_0.5 SCORE: {best_f05:.4f}")
    
    # Save trained model checkpoint
    model.save(model_save_path)
    print(f"Trained model checkpoint saved to: {model_save_path}")
    print(f"Total time elapsed: {time.time()-t0:.2f}s")
    return model, best_f05


if __name__ == "__main__":
    run_training_and_validation()
