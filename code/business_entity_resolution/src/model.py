"""
High-Precision LightGBM Classifier with Hard Negative Geo-Veto,
Per-Country Threshold Optimization, and Transitive Match Propagation.
"""

import numpy as np
import lightgbm as lgb
import os
import joblib
from .feature_extractor import FEATURE_NAMES

STATE_CONFLICT_IDX = FEATURE_NAMES.index("geo_state_conflict")
NAME_RATIO_IDX = FEATURE_NAMES.index("name_ratio")
POSTAL_CONFLICT_IDX = FEATURE_NAMES.index("postal_code_conflict")
EXACT_NUM_IDX = FEATURE_NAMES.index("exact_number_match")


def compute_macro_f05(ground_truth_dict, predictions_dict):
    """
    Computes official macro-averaged F_0.5 score per Source 1 entity.
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
    def __init__(self, n_estimators=400, learning_rate=0.05, max_depth=7, num_leaves=45):
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            importance_type="gain",
            verbose=-1
        )
        self.best_threshold = 0.770
        self.country_thresholds = {
            "US": 0.760,
            "India": 0.780,
            "France": 0.760
        }

    def fit(self, X, y):
        """Fits LightGBM on pairwise training candidate pairs."""
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Returns positive class match probabilities with Hard Negative Geo-Veto.
        """
        raw_probs = self.clf.predict_proba(X)[:, 1]
        
        # Hard Negative Geo-Veto
        veto_mask = (X[:, STATE_CONFLICT_IDX] > 0.5) & (X[:, NAME_RATIO_IDX] < 0.95)
        raw_probs[veto_mask] = 0.0
        
        # Postal conflict penalty
        postal_mask = (X[:, POSTAL_CONFLICT_IDX] > 0.5) & (X[:, NAME_RATIO_IDX] < 0.95)
        raw_probs[postal_mask] = raw_probs[postal_mask] * 0.3
        
        return raw_probs

    def optimize_threshold_per_country(self, val_cand_scores_by_country, val_gt_by_country):
        """
        Optimizes separate thresholds per country to maximize global macro F_0.5.
        val_cand_scores_by_country: {"US": [(qid, tid, score), ...], "India": [...]}
        val_gt_by_country: {"US": {qid: set(mids)}, "India": {...}}
        """
        thresholds = np.linspace(0.65, 0.92, 28)
        
        for country, scores_list in val_cand_scores_by_country.items():
            gt_dict = val_gt_by_country[country]
            grouped = {qid: [] for qid in gt_dict}
            for qid, tid, score in scores_list:
                if qid in grouped:
                    grouped[qid].append((tid, score))
                    
            best_c_f05 = -1.0
            best_c_tau = 0.770
            
            for tau in thresholds:
                preds = {}
                for qid, t_scores in grouped.items():
                    preds[qid] = {tid for tid, s in t_scores if s >= tau}
                f05 = compute_macro_f05(gt_dict, preds)
                if f05 > best_c_f05:
                    best_c_f05 = f05
                    best_c_tau = tau
                    
            self.country_thresholds[country] = best_c_tau
            print(f"  Country {country:6}: Optimal tau = {best_c_tau:.3f} | Macro F_0.5 = {best_c_f05:.4f}")

        # France default
        if "France" not in self.country_thresholds:
            self.country_thresholds["France"] = self.country_thresholds.get("US", 0.760)
            
        return self.country_thresholds

    def save(self, filepath):
        """Saves model and per-country thresholds to disk."""
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        joblib.dump({
            "model": self.clf,
            "threshold": self.best_threshold,
            "country_thresholds": self.country_thresholds
        }, filepath)

    def load(self, filepath):
        """Loads model and thresholds from disk."""
        data = joblib.load(filepath)
        self.clf = data["model"]
        self.best_threshold = data.get("threshold", 0.770)
        self.country_thresholds = data.get("country_thresholds", {
            "US": 0.760,
            "India": 0.780,
            "France": 0.760
        })
        return self
