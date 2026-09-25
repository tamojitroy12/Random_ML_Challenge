"""
LightGBM Matching Classifier and Macro F_0.5 Threshold Optimizer.
Trains a gradient boosted decision tree classifier and searches for the optimal
precision-weighted threshold to maximize macro F_0.5 on validation data.
"""

import numpy as np
import lightgbm as lgb
import os
import joblib
from .feature_extractor import FEATURE_NAMES


def compute_macro_f05(ground_truth_dict, predictions_dict):
    """
    Computes the macro-averaged F_0.5 score per Source 1 entity.
    """
    scores = []
    for s1_id, y_true in ground_truth_dict.items():
        y_pred = predictions_dict.get(s1_id, set())
        
        # Case 1: True singleton
        if len(y_true) == 0:
            scores.append(1.0 if len(y_pred) == 0 else 0.0)
            continue
            
        # Case 2: Predicted singleton when true matches exist
        if len(y_pred) == 0:
            scores.append(0.0)
            continue
            
        # Case 3: Both non-empty
        tp = len(y_true.intersection(y_pred))
        if tp == 0:
            scores.append(0.0)
            continue
            
        precision = tp / len(y_pred)
        recall = tp / len(y_true)
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        scores.append(f05)
        
    return float(np.mean(scores)) if scores else 0.0


class MatchingModel:
    def __init__(self, n_estimators=200, learning_rate=0.08, max_depth=6, num_leaves=31):
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=42,
            n_jobs=-1,
            importance_type="gain",
            verbose=-1
        )
        self.best_threshold = 0.65 # Default precision-biased threshold

    def fit(self, X, y):
        """Fits LightGBM on training candidate pairs."""
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        """Returns positive class match probabilities."""
        return self.clf.predict_proba(X)[:, 1]

    def optimize_threshold(self, query_cand_scores, ground_truth_dict, candidate_pairs_map):
        """
        Performs a fine grid search for threshold tau in [0.40, 0.95]
        to directly maximize Macro F_0.5 score on validation data.
        
        query_cand_scores: [(qid, tid, score), ...]
        candidate_pairs_map: {qid: [tids]}
        """
        # Group by qid
        grouped = {}
        for qid in candidate_pairs_map:
            grouped[qid] = []
        for qid, tid, score in query_cand_scores:
            grouped[qid].append((tid, score))

        thresholds = np.linspace(0.40, 0.92, 27)
        best_f05 = -1.0
        best_tau = 0.65

        for tau in thresholds:
            preds = {}
            for qid, t_scores in grouped.items():
                matched_tids = {tid for tid, score in t_scores if score >= tau}
                preds[qid] = matched_tids
                
            score = compute_macro_f05(ground_truth_dict, preds)
            if score > best_f05:
                best_f05 = score
                best_tau = tau

        self.best_threshold = best_tau
        return best_tau, best_f05

    def save(self, filepath):
        """Saves model and threshold to disk."""
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        joblib.dump({"model": self.clf, "threshold": self.best_threshold}, filepath)


    def load(self, filepath):
        """Loads model and threshold from disk."""
        data = joblib.load(filepath)
        self.clf = data["model"]
        self.best_threshold = data["threshold"]
        return self
