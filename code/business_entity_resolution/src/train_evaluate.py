"""
High-Accuracy Training & Benchmark for Business Entity Resolution.
Trains LightGBM on multi-channel candidate pairs with 26-dimensional features,
applies Hard Negative Geo-Veto, and optimizes per-country decision thresholds.
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


def run_training_and_validation(train_dir="dataset/train", val_dir="benchmark/val", model_save_path="code/business_entity_resolution/model_checkpoint.joblib", n_sample=25000):
    print("=" * 65)
    print("STAGE 1: Loading Validation & Training Data...")
    print("=" * 65)
    t0 = time.time()
    
    val_s1 = pl.read_csv(os.path.join(val_dir, "val_source1.tsv"), separator="\t")
    val_gt = pl.read_csv(os.path.join(val_dir, "val_ground_truth.tsv"), separator="\t")
    val_s1_ids = set(val_s1["entity_id"])
    
    val_gt_dict = {}
    for r in val_gt.iter_rows(named=True):
        mids = {m.strip() for m in r["matched_entity_ids"].split(",") if m.strip()} if r["matched_entity_ids"] else set()
        val_gt_dict[r["source1_entity_id"]] = mids
        
    print(f"Validation Set: {len(val_s1):,} S1 entities loaded.")
    
    train_gt_full = pl.scan_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t")
    train_s1_full = pl.scan_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t")
    
    train_sample_s1 = (
        train_s1_full
        .filter(~pl.col("entity_id").is_in(val_s1_ids))
        .head(n_sample)
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
        
    print(f"Training Sample: {len(train_sample_s1):,} S1 entities selected.")
    
    train_X = []
    train_y = []
    val_cand_scores_by_country = {"US": [], "India": []}
    val_gt_by_country = {"US": {}, "India": {}}
    
    for country in ["US", "India"]:
        print("\n" + "-" * 55)
        print(f"Processing Country: {country}")
        print("-" * 55)
        
        t_country = time.time()
        s2_scan = pl.scan_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t").filter(pl.col("country") == country)
        s3_scan = pl.scan_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t").filter(pl.col("country") == country)
        target_df = pl.concat([s2_scan, s3_scan]).select(["entity_id", "business_name", "business_address"]).collect()
        
        target_ids = target_df["entity_id"].to_list()
        target_names = target_df["business_name"].to_list()
        target_addrs = target_df["business_address"].to_list()
        print(f"Loaded {len(target_df):,} target records for {country} in {time.time()-t_country:.2f}s")
        
        blocker = MultiIndexBlocker(max_candidates=30)
        build_time = blocker.fit_target_universe(target_ids, target_names, target_addrs, country)
        print(f"High-recall multi-channel blocker built in {build_time:.2f}s")
        tid_to_idx = {tid: i for i, tid in enumerate(target_ids)}
        
        def get_target_rec(idx):
            cn = blocker.cleaned_names[idx]
            _, _, suf, acr = extract_name_components(cn)
            return {
                'target_id': target_ids[idx],
                'cleaned_name': cn,
                'cleaned_addr': blocker.cleaned_addrs[idx],
                'core_name': blocker.core_names[idx],
                'name_tokens': blocker.name_tokens_list[idx],
                'acronym': acr,
                'legal_suffixes': suf,
                'addr_nums': blocker.addr_nums_list[idx],
                'addr_words': blocker.addr_words_list[idx],
                'postal_code': blocker.postal_codes_list[idx],
                'state_code': blocker.state_codes_list[idx]
            }

        # --- A. Training Candidates ---
        c_train_s1 = train_sample_s1.filter(pl.col("country") == country)
        qids = c_train_s1["entity_id"].to_list()
        qnames = c_train_s1["business_name"].to_list()
        qaddrs = c_train_s1["business_address"].to_list()
        
        print(f"Generating training candidates for {len(qids):,} queries...")
        cand_map = blocker.generate_candidates(qids, qnames, qaddrs, country)
        
        for q_idx, qid in enumerate(qids):
            cn = clean_string(qnames[q_idx])
            ca = clean_string(qaddrs[q_idx])
            core_name, ntoks, suffixes, acr = extract_name_components(cn)
            nums, awords, post, state_code = extract_addr_components(ca, country)
            q_rec = {
                'cleaned_name': cn, 'cleaned_addr': ca, 'core_name': core_name,
                'name_tokens': ntoks, 'acronym': acr, 'legal_suffixes': suffixes,
                'addr_nums': nums, 'addr_words': awords,
                'postal_code': post, 'state_code': state_code
            }
            
            true_tids = train_gt_dict.get(qid, set())
            cand_pool = set(cand_map.get(qid, [])) | (true_tids & tid_to_idx.keys())
            
            for tid in cand_pool:
                if tid in tid_to_idx:
                    t_rec = get_target_rec(tid_to_idx[tid])
                    feat = compute_pair_features(q_rec, t_rec)
                    train_X.append(feat)
                    train_y.append(1 if tid in true_tids else 0)

        # --- B. Validation Candidates ---
        c_val_s1 = val_s1.filter(pl.col("country") == country)
        val_qids = c_val_s1["entity_id"].to_list()
        val_qnames = c_val_s1["business_name"].to_list()
        val_qaddrs = c_val_s1["business_address"].to_list()
        
        for qid in val_qids:
            val_gt_by_country[country][qid] = val_gt_dict.get(qid, set())
            
        print(f"Generating validation candidates for {len(val_qids):,} queries...")
        val_cand_submap = blocker.generate_candidates(val_qids, val_qnames, val_qaddrs, country)
        
        val_country_pairs = []
        val_country_feats = []
        for q_idx, qid in enumerate(val_qids):
            cn = clean_string(val_qnames[q_idx])
            ca = clean_string(val_qaddrs[q_idx])
            core_name, ntoks, suffixes, acr = extract_name_components(cn)
            nums, awords, post, state_code = extract_addr_components(ca, country)
            q_rec = {
                'cleaned_name': cn, 'cleaned_addr': ca, 'core_name': core_name,
                'name_tokens': ntoks, 'acronym': acr, 'legal_suffixes': suffixes,
                'addr_nums': nums, 'addr_words': awords,
                'postal_code': post, 'state_code': state_code
            }
            for tid in val_cand_submap.get(qid, []):
                if tid in tid_to_idx:
                    t_rec = get_target_rec(tid_to_idx[tid])
                    feat = compute_pair_features(q_rec, t_rec)
                    val_country_pairs.append((qid, tid))
                    val_country_feats.append(feat)
                    
        val_cand_scores_by_country[country] = (val_country_pairs, val_country_feats)

    print("\n" + "=" * 65)
    print(f"STAGE 2: Training LightGBM on {len(train_X):,} candidate pairs (26 features)...")
    print(f"Positive pairs: {sum(train_y):,} ({sum(train_y)/len(train_y)*100:.2f}%)")
    print("=" * 65)
    
    train_X = np.array(train_X, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.int32)
    
    model = MatchingModel(n_estimators=400, learning_rate=0.05, max_depth=7, num_leaves=45)
    model.fit(train_X, train_y)
    print("LightGBM training complete!")
    
    # Feature Importances
    importances = model.clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 10 Feature Importances:")
    for idx in sorted_idx[:10]:
        print(f"  {FEATURE_NAMES[idx]:25}: {importances[idx]:.1f}")
        
    print("\n" + "=" * 65)
    print("STAGE 3: Scoring Validation Pairs with Hard Negative Geo-Veto...")
    print("=" * 65)
    
    scored_by_country = {}
    for country in ["US", "India"]:
        pairs, feats = val_cand_scores_by_country[country]
        if feats:
            X_val = np.array(feats, dtype=np.float32)
            scores = model.predict_proba(X_val)
            scored_list = [(pairs[i][0], pairs[i][1], float(scores[i])) for i in range(len(pairs))]
            scored_by_country[country] = scored_list
        else:
            scored_by_country[country] = []

    print("\nOptimizing Thresholds per Country:")
    country_taus = model.optimize_threshold_per_country(scored_by_country, val_gt_by_country)
    
    # Combined Global Macro F_0.5 Evaluation
    all_preds = {}
    for country in ["US", "India"]:
        tau = country_taus.get(country, 0.770)
        gt_dict = val_gt_by_country[country]
        grouped = {qid: [] for qid in gt_dict}
        for qid, tid, s in scored_by_country[country]:
            if qid in grouped:
                grouped[qid].append((tid, s))
        for qid, t_scores in grouped.items():
            all_preds[qid] = {tid for tid, s in t_scores if s >= tau}
            
    global_f05 = compute_macro_f05(val_gt_dict, all_preds)
    print("-" * 55)
    print(f"⭐ GLOBAL COMBINED VALIDATION MACRO F_0.5 SCORE: {global_f05:.4f}")
    print("-" * 55)
    
    model.save(model_save_path)
    print(f"Model checkpoint saved to: {model_save_path}")
    print(f"Total time elapsed: {time.time()-t0:.2f}s")
    return model, global_f05


if __name__ == "__main__":
    run_training_and_validation()
