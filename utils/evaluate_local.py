"""
Local Evaluation Script for Amazon ML Challenge 2026.
Computes the official Macro F_0.5 metric, Precision, Recall, and Singleton Accuracy
given a predictions TSV and a ground truth TSV.

Usage:
    python utils/evaluate_local.py \
        --predictions output/val_predictions.tsv \
        --ground-truth benchmark/val/val_ground_truth.tsv
"""

import argparse
import sys
import os
import polars as pl
import numpy as np


def load_mapping(tsv_path, id_col, match_col):
    """Loads a TSV into a mapping {s1_id: set of target_ids}."""
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"File not found: {tsv_path}")
    df = pl.read_csv(tsv_path, separator="\t")
    mapping = {}
    for r in df.iter_rows(named=True):
        s1 = str(r[id_col]).strip()
        val = r[match_col]
        if val is None or val == "" or val == "None":
            mids = set()
        else:
            mids = {m.strip() for m in str(val).split(",") if m.strip()}
        mapping[s1] = mids
    return mapping


def evaluate(ground_truth_dict, predictions_dict):
    """
    Computes exact Macro F_0.5, Macro Precision, Macro Recall, and Singleton Accuracy.
    """
    f05_scores = []
    precisions = []
    recalls = []
    singleton_scores = []
    non_singleton_f05 = []

    for s1_id, y_true in ground_truth_dict.items():
        y_pred = predictions_dict.get(s1_id, set())

        # Case 1: True Singleton (0 true matches)
        if len(y_true) == 0:
            if len(y_pred) == 0:
                f05_scores.append(1.0)
                singleton_scores.append(1.0)
            else:
                f05_scores.append(0.0)
                singleton_scores.append(0.0)
            continue

        # Case 2: True matches exist, but predicted empty
        if len(y_pred) == 0:
            f05_scores.append(0.0)
            non_singleton_f05.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        # Case 3: Both have entities
        tp = len(y_true.intersection(y_pred))
        precision = tp / len(y_pred)
        recall = tp / len(y_true)
        precisions.append(precision)
        recalls.append(recall)

        if tp == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        
        f05_scores.append(f05)
        non_singleton_f05.append(f05)

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    singleton_acc = float(np.mean(singleton_scores)) if singleton_scores else 0.0
    non_sing_f05 = float(np.mean(non_singleton_f05)) if non_singleton_f05 else 0.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "singleton_accuracy": singleton_acc,
        "non_singleton_f05": non_sing_f05,
        "total_queries": len(ground_truth_dict),
        "total_singletons": len(singleton_scores),
        "total_non_singletons": len(ground_truth_dict) - len(singleton_scores)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Macro F_0.5 Score for Business Entity Resolution.")
    parser.add_argument("--predictions", "-p", default="output/val_predictions.tsv", help="Path to predictions TSV")
    parser.add_argument("--ground-truth", "-g", default="benchmark/val/val_ground_truth.tsv", help="Path to ground truth TSV")
    parser.add_argument("--pred-id-col", default="source1_entity_id", help="Source 1 ID column in predictions")
    parser.add_argument("--pred-match-col", default="matched_entity_ids", help="Match list column in predictions")
    parser.add_argument("--gt-id-col", default="source1_entity_id", help="Source 1 ID column in ground truth")
    parser.add_argument("--gt-match-col", default="matched_entity_ids", help="Match list column in ground truth")

    args = parser.parse_args()

    print(f"Loading Ground Truth: {args.ground_truth}")
    gt_map = load_mapping(args.ground_truth, args.gt_id_col, args.gt_match_col)
    print(f"Loading Predictions:  {args.predictions}")
    pred_map = load_mapping(args.predictions, args.pred_id_col, args.pred_match_col)

    results = evaluate(gt_map, pred_map)

    print("\n" + "=" * 55)
    print("           LOCAL EVALUATION RESULTS")
    print("=" * 55)
    print(f"⭐ OFFICIAL MACRO F_0.5 SCORE : {results['macro_f05']:.4f}")
    print("-" * 55)
    print(f"• Macro Precision (weight 2x) : {results['macro_precision'] * 100:.2f}%")
    print(f"• Macro Recall    (weight 1x) : {results['macro_recall'] * 100:.2f}%")
    print(f"• Singleton Accuracy (0 links): {results['singleton_accuracy'] * 100:.2f}% ({results['total_singletons']:,} entities)")
    print(f"• Non-Singleton F_0.5 Score   : {results['non_singleton_f05']:.4f} ({results['total_non_singletons']:,} entities)")
    print(f"• Total Evaluated Entities    : {results['total_queries']:,}")
    print("=" * 55)


if __name__ == "__main__":
    main()
