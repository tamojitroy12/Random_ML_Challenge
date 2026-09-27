"""
Matching model and decision rule.

* MatchingModel: LightGBM (native API) over FEATURE_NAMES; checkpoint stores the booster string.
* exclusivity(): every target goes to at most one S1 (its best-scoring one), ties dropped.
* assign(): tau threshold + margin delta over the best competitor, plus optional
  sibling-rescue, sibling-suppress and lonely-link rules. No per-S1 / per-source caps.
* macro F0.5 (official) and a test-density-adjusted F_adj used for threshold selection.
"""

import os

import joblib
import lightgbm as lgb
import numpy as np

from .feature_extractor import FEATURE_NAMES, G_FR_CITY, G_FR_NUM, G_FR_STREET, G_GEO, HN_CONFLICT, HN_EXACT
from .runtime import log, safe_div

FR_GATES = G_FR_NUM | G_FR_CITY | G_FR_STREET

LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "num_threads": 8, "force_col_wise": True, "verbose": -1, "seed": 42, "metric": "binary_logloss",
}


class MatchingModel:
    def __init__(self, booster=None, meta=None):
        self.booster = booster
        self.meta = meta or {}

    def fit(self, X, y, w, holdout, max_rounds=1000):
        assert X.shape[1] == len(FEATURE_NAMES)
        tr = ~holdout
        dtr = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], feature_name=FEATURE_NAMES,
                          free_raw_data=True)
        dho = lgb.Dataset(X[holdout], label=y[holdout], weight=w[holdout], reference=dtr,
                          free_raw_data=True)
        self.booster = lgb.train(LGB_PARAMS, dtr, num_boost_round=max_rounds, valid_sets=[dho],
                                 valid_names=["holdout"],
                                 callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)])
        best = self.booster.best_iteration or self.booster.current_iteration()
        self.meta["best_iteration"] = int(best)
        return self

    def predict(self, X, batch=2_000_000):
        if X.shape[0] == 0:
            return np.zeros(0, np.float32)
        out = np.empty(X.shape[0], np.float32)
        for s in range(0, X.shape[0], batch):
            out[s:s + batch] = self.booster.predict(X[s:s + batch], num_threads=8)
        return out

    def importance(self, top=25):
        gain = self.booster.feature_importance(importance_type="gain")
        order = np.argsort(-gain)[:top]
        return [(FEATURE_NAMES[i], float(gain[i])) for i in order]

    def save(self, path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        best = self.meta.get("best_iteration", -1)
        payload = {"booster_str": self.booster.model_to_string(num_iteration=best if best > 0 else None),
                   "feature_names": list(FEATURE_NAMES), **self.meta}
        tmp = path + ".tmp"
        joblib.dump(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        if list(d.get("feature_names", [])) != list(FEATURE_NAMES):
            raise RuntimeError("Checkpoint feature list does not match feature_extractor.FEATURE_NAMES; "
                               "retrain with the current code.")
        booster = lgb.Booster(model_str=d["booster_str"])
        meta = {k: v for k, v in d.items() if k not in ("booster_str",)}
        return cls(booster, meta)


# ---------------------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------------------
def assert_unique_pairs(q, t):
    key = q.astype(np.int64) * np.int64(1 << 31) + t.astype(np.int64)
    if np.unique(key).size != key.size:
        raise RuntimeError("duplicate (query, target) pairs before assignment")


def effective_p(p, gate, veto_bits):
    pe = p.astype(np.float32, copy=True)
    if veto_bits:
        pe[(gate & np.uint8(veto_bits)) != 0] = 0.0
    return pe


def exclusivity(t, p_eff):
    """-> (is_best, margin): margin = p - best competing p for the same target (ties -> 0)."""
    n = t.size
    best = np.zeros(n, bool)
    margin = np.full(n, -1.0, np.float32)
    if n == 0:
        return best, margin
    order = np.lexsort((-p_eff, t))
    ts, ps = t[order], p_eff[order]
    first = np.ones(n, bool)
    first[1:] = ts[1:] != ts[:-1]
    nxt_same = np.zeros(n, bool)
    nxt_same[:-1] = ts[1:] == ts[:-1]
    second = np.full(n, -1.0, np.float32)
    fi = np.flatnonzero(first & nxt_same)
    second[fi] = ps[fi + 1]
    fidx = np.flatnonzero(first)
    best[order[fidx]] = True
    margin[order[fidx]] = ps[fidx] - second[fidx]
    return best, margin


_P1 = np.uint64(0x9E3779B97F4A7C15)
_P2 = np.uint64(0xC2B2AE3D27D4EB4F)


def _k2(q, h):
    return q.astype(np.uint64) * _P1 ^ h.astype(np.uint64)


def _k3(q, h, s):
    return (q.astype(np.uint64) * _P1) ^ h.astype(np.uint64) ^ (s.astype(np.uint64) * _P2)


def assign(q, p_eff, best, margin, hn_rel, hn_h, st_h, core_set, n_q, tau, delta,
           tau1=None, tau_lo=None, suppress=False):
    base = best & (margin > delta) & (p_eff > 0)
    acc = base & (p_eff >= tau)
    with np.errstate(over="ignore"):
        if tau_lo is not None and tau_lo < tau:
            anc = acc
            anc_q = np.unique(q[anc]) if anc.any() else np.zeros(0, np.int64)
            # Case 1: Same house number hash (tolerates minor street variations)
            c1 = base & ~acc & (p_eff >= tau_lo) & (core_set >= 0.55) & (hn_h != 0)
            anc_hn = anc & (hn_h != 0)
            if c1.any() and anc_hn.any():
                k_anc_hn = np.unique(q[anc_hn].astype(np.uint64) * _P1 ^ hn_h[anc_hn].astype(np.uint64))
                c1_idx = np.flatnonzero(c1)
                hit_c1 = np.isin(q[c1_idx].astype(np.uint64) * _P1 ^ hn_h[c1_idx].astype(np.uint64), k_anc_hn)
                acc[c1_idx[hit_c1]] = True

            # Case 2: Non-numbered entities (hn_h == 0) with high brand similarity and no conflict
            if anc_q.size:
                c2 = base & ~acc & (p_eff >= tau_lo) & (core_set >= 0.75) & (hn_rel != HN_CONFLICT) & np.isin(q, anc_q)
                if c2.any():
                    acc[c2] = True
        if suppress:
            ex = acc & (hn_rel == HN_EXACT)
            if ex.any():
                has_exact = np.zeros(n_q, bool)
                has_exact[q[ex]] = True
                conf = acc & (hn_rel == HN_CONFLICT) & has_exact[q]
                if conf.any():
                    u, cnt = np.unique(_k2(q[acc], hn_h[acc]), return_counts=True)
                    ci = np.flatnonzero(conf)
                    c = cnt[np.searchsorted(u, _k2(q[ci], hn_h[ci]))]
                    acc[ci[c <= 1]] = False
    if tau1 is not None and tau1 > tau:
        cnt = np.bincount(q[acc], minlength=n_q)
        acc &= ~((cnt[q] == 1) & (p_eff < tau1))
    return acc


def assign_with_params(q, t, p, gate, hn_rel, hn_h, st_h, core_set, n_q, prm):
    pe = effective_p(p, gate, int(prm.get("veto", 0)) | int(prm.get("fr_gates", 0)))
    best, margin = exclusivity(t, pe)
    return assign(q, pe, best, margin, hn_rel, hn_h, st_h, core_set, n_q, prm["tau"], prm["delta"],
                  prm.get("tau1"), prm.get("tau_lo"), bool(prm.get("suppress", False)))


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------
def f05_stats(acc, qv, label, linked, n_true, r=1.0):
    """acc/qv/label/linked are per pair (qv = val position or -1); n_true per val S1."""
    nv = n_true.size
    m = acc & (qv >= 0)
    qm = qv[m]
    lab = label[m]
    tp = np.bincount(qm, weights=lab.astype(np.float64), minlength=nv)
    npred = np.bincount(qm, minlength=nv).astype(np.float64)
    fp_owned = np.bincount(qm, weights=(~lab & linked[m]).astype(np.float64), minlength=nv)
    fp_unl = npred - tp - fp_owned
    nt = n_true.astype(np.float64)
    sing = nt == 0
    f = np.zeros(nv)
    f[sing] = (npred[sing] == 0).astype(np.float64)
    ns = ~sing
    f[ns] = np.where(tp[ns] > 0, 1.25 * tp[ns] / np.maximum(0.25 * nt[ns] + npred[ns], 1e-9), 0.0)
    fadj_ns = np.where(tp[ns] > 0, 1.25 * tp[ns] /
                       np.maximum(0.25 * nt[ns] + tp[ns] + fp_owned[ns] + r * fp_unl[ns], 1e-9), 0.0)
    s_acc = float(np.mean(npred[sing] == 0)) if sing.any() else 1.0
    w_s = float(sing.mean()) if nv else 0.0
    f_adj = w_s * (s_acc ** r) + (1 - w_s) * (float(fadj_ns.mean()) if ns.any() else 0.0)
    prec = np.where(npred[ns] > 0, tp[ns] / np.maximum(npred[ns], 1), 0.0)
    rec = tp[ns] / np.maximum(nt[ns], 1)
    one = ns & (nt == 1)
    return {
        "F": float(f.mean()) if nv else 0.0, "F_adj": f_adj, "singleton_acc": s_acc,
        "nonsingleton_F": float(f[ns].mean()) if ns.any() else 0.0,
        "one_link_F": float(f[one].mean()) if one.any() else float("nan"),
        "macro_P": float(prec.mean()) if ns.any() else 0.0,
        "macro_R": float(rec.mean()) if ns.any() else 0.0,
        "fp_owned": float(fp_owned.sum()), "fp_unlinked": float(fp_unl.sum()),
        "tp": float(tp.sum()), "n_pred": float(npred.sum()), "n_true": float(nt.sum()),
        "per_query_f": f,
    }


TAUS = np.round(np.arange(0.30, 0.9701, 0.01), 2)
DELTAS = [0.0, 0.02, 0.05, 0.1, 0.2]


def tune_country(d, r, sel=None, verbose=True):
    """d: dict of per-pair arrays (q, t, p, gate, hn_rel, hn_h, st_h, core_set, qv, label, linked)
    plus n_true, n_q. sel: optional bool mask over val positions to tune on (cross-fit)."""
    n_true = d["n_true"]
    qv = d["qv"]
    if sel is not None:
        qv = np.where((qv >= 0) & sel[np.maximum(qv, 0)], qv, -1)
        remap = -np.ones(n_true.size, np.int64)
        remap[np.flatnonzero(sel)] = np.arange(int(sel.sum()))
        qv = np.where(qv >= 0, remap[np.maximum(qv, 0)], -1)
        n_true = n_true[sel]
    best_cfg, best_val = None, -1.0
    table = {}
    for veto in (0, G_GEO):
        pe = effective_p(d["p"], d["gate"], veto)
        bst, mg = exclusivity(d["t"], pe)
        for delta in DELTAS:
            base = bst & (mg > delta) & (pe > 0)
            for tau in TAUS:
                acc = base & (pe >= tau)
                s = f05_stats(acc, qv, d["label"], d["linked"], n_true, r)
                table[(veto, delta, float(tau))] = s["F_adj"]
                if s["F_adj"] > best_val:
                    best_val, best_cfg = s["F_adj"], (veto, delta, float(tau))
    veto, delta, _ = best_cfg
    # Precision-leaning choice: largest tau within 0.002 of the best F_adj (same veto/delta)
    cands = [tau for (v, dl, tau), val in table.items() if v == veto and dl == delta and val >= best_val - 0.002]
    tau = max(cands)
    pe = effective_p(d["p"], d["gate"], veto)
    bst, mg = exclusivity(d["t"], pe)

    def run(tau1=None, tau_lo=None, suppress=False):
        acc = assign(d["q"], pe, bst, mg, d["hn_rel"], d["hn_h"], d["st_h"], d["core_set"], d["n_q"],
                     tau, delta, tau1, tau_lo, suppress)
        return f05_stats(acc, qv, d["label"], d["linked"], n_true, r)

    base_s = run()
    best_tau1, best_v = None, base_s["F_adj"]
    for tau1 in np.round(np.arange(tau + 0.01, 0.9901, 0.01), 2):
        v = run(tau1=float(tau1))["F_adj"]
        if v > best_v + 1e-5:
            best_tau1, best_v = float(tau1), v
    best_rule = (None, False)
    for tau_lo in (None, tau - 0.1, tau - 0.2, tau - 0.3):
        for sup in (False, True):
            v = run(tau1=best_tau1, tau_lo=tau_lo, suppress=sup)["F_adj"]
            if v > best_v + 1e-5:
                best_v, best_rule = v, (tau_lo, sup)
    prm = {"tau": float(tau), "delta": float(delta), "veto": int(veto), "tau1": best_tau1,
           "tau_lo": None if best_rule[0] is None else float(round(best_rule[0], 2)),
           "suppress": bool(best_rule[1]), "fr_gates": 0}
    return prm, base_s


def evaluate_params(d, prm, r):
    acc = assign_with_params(d["q"], d["t"], d["p"], d["gate"], d["hn_rel"], d["hn_h"], d["st_h"],
                             d["core_set"], d["n_q"], prm)
    return acc, f05_stats(acc, d["qv"], d["label"], d["linked"], d["n_true"], r)


def no_exclusivity_stats(d, prm, r):
    """Same thresholds but without exclusivity/rules and without competitor pairs."""
    pe = effective_p(d["p"], d["gate"], prm.get("veto", 0))
    acc = (pe >= prm["tau"]) & (d["qv"] >= 0)
    return f05_stats(acc, d["qv"], d["label"], d["linked"], d["n_true"], r)


def summarize(s):
    return (f"F={s['F']:.4f} F_adj={s['F_adj']:.4f} P={s['macro_P']:.4f} R={s['macro_R']:.4f} "
            f"single_acc={s['singleton_acc']:.4f} nonsingle_F={s['nonsingleton_F']:.4f} "
            f"1link_F={s['one_link_F']:.4f} FP_owned={s['fp_owned']:.0f} FP_unl={s['fp_unlinked']:.0f}")


def safe_ratio(a, b):
    return safe_div(a, b, float("nan"))
