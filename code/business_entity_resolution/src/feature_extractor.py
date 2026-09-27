"""
Vectorised pairwise features for (S1 query, S2/S3 target) candidate pairs.

compute_features(qs, ts, qi, ti, ctx) -> (X float32 [n, F], aux dict)

* string similarities: rapidfuzz.process.cpdist (C++, multithreaded), scaled to [0, 1];
* flags / counts / signature equality / geo: numpy on precomputed store columns;
* one lean Python loop: token coverage, IDF overlap, decoy/noise words, acronyms,
  house-number relations (drop/substitution/prefix-chain/last-part), compound-number recall,
  street-core containment.
aux carries per-pair values needed by the decision rules (gates, hn relation, hashes, core_set).

No retrieval-rank or per-query context features: competitor pairs must score exactly like
test pairs.
"""

import math

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .normalizer import DECOY_WORDS, NOISE_WORDS

FEATURE_NAMES = [
    "name_set", "name_sort", "name_partial", "name_jw", "core_ratio", "core_set", "concat_ratio",
    "fconcat_best", "skel_ratio", "tok0_ratio", "addr_set", "street_set", "street_ratio", "loc_set",
    "city_ratio",
    "t_is_domain", "t_is_garbage", "t_has_alias", "t_non_latin", "country_tok_added",
    "legal_conflict", "legal_added", "legal_dropped", "raw_sig_eq", "clean_sig_eq",
    "state_match", "state_conflict", "city_eq", "stype_eq",
    "q_name_s1dup", "t_name_freq", "q_core_in_t", "q_addr_s1dup", "t_addr_key_freq",
    "hn_q_present", "hn_t_present", "hn_exact", "hn_digits_eq", "hn_sfx_both_diff",
    "hn_sfx_one_missing", "hn_prefix_suffix", "hn_drop1", "hn_sub1", "hn_sub1_last",
    "hn_prefix_chain", "hn_lastpart_diff", "hn_lastpart_absdelta", "hn_logdiff", "hn_reldiff",
    "hn_range_hit", "hn_is_compound", "hn_clean_both", "unit_eq", "unit_diff",
    "q_addr_empty", "t_addr_empty", "target_source", "composite",
    "cov_q", "cov_t", "idf_overlap", "extra_t_idf", "missing_q_idf", "t_extra_decoy",
    "t_extra_noise_only", "acronym_match", "q_cnum_recall", "t_cnum_extra", "q_digit_recall",
    "street_cont", "hn_rel",
]
NF = len(FEATURE_NAMES)
FI = {f: i for i, f in enumerate(FEATURE_NAMES)}

# Gate bits (computed here, applied by model.assign_matches)
G_GEO, G_FR_NUM, G_FR_CITY, G_FR_STREET = 1, 2, 4, 8

# hn relation codes
HN_MISSING, HN_EXACT, HN_SOFT, HN_CONFLICT = 0, 1, 2, 3

_STR_FIELDS = ["name", "core", "concat", "fconcat", "fconcat_del", "skel", "tok0", "acr", "short",
               "addr", "street", "loc", "city", "hnum", "hn_digits", "hn_sfx", "cnums", "digits"]
_ratio = fuzz.ratio


def _legal_group(m):
    # LTD alone is compatible with PVT and PUBLIC; INC~CORP, PC~PLLC~PA already merged.
    return m & ~np.int64(128)


def _gather_obj(store, fields, idx):
    u, inv = np.unique(np.asarray(idx, dtype=np.int64), return_inverse=True)
    cols = store.take_str(fields, u)
    return u, inv, {f: np.array(cols[f], dtype=object) for f in fields}


def _parts(h):
    return h.split("/") if h else []


def _chain(a, b):
    """a, b part lists: shorter is a prefix or suffix of the longer (and strictly shorter)."""
    if not a or not b or len(a) == len(b):
        return False
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return l[:len(s)] == s or l[-len(s):] == s


def _contain(x, y):
    """Share of tokens of x found (exactly or ratio>=80) in token list y."""
    hit = 0
    for tok in x:
        if tok in y or (len(tok) >= 3 and any(len(z) >= 3 and _ratio(tok, z) >= 80 for z in y)):
            hit += 1
    return hit / len(x)


def _cnum_match(c, other):
    if c in other:
        return True
    for o in other:
        if o.startswith(c + "/") or o.endswith("/" + c) or c.startswith(o + "/") or c.endswith("/" + o):
            return True
    return False


def compute_features(qs, ts, qi, ti, ctx, workers=-1):
    qi = np.asarray(qi, dtype=np.int64)
    ti = np.asarray(ti, dtype=np.int64)
    n = qi.size
    X = np.zeros((n, NF), np.float32)
    aux = {"gate": np.zeros(n, np.uint8), "hn_rel": np.zeros(n, np.int8),
           "hn_h": np.zeros(n, np.uint64), "st_h": np.zeros(n, np.uint64),
           "core_set": np.zeros(n, np.float32)}
    if n == 0:
        return X, aux
    country = ctx.get("country", "")
    idf_df = ctx.get("idf", {})
    n_docs = float(ctx.get("n_docs", 1))

    uq, qinv, Qu = _gather_obj(qs, _STR_FIELDS, qi)
    ut, tinv, Tu = _gather_obj(ts, _STR_FIELDS, ti)
    Q = {f: Qu[f][qinv] for f in _STR_FIELDS}
    T = {f: Tu[f][tinv] for f in _STR_FIELDS}

    def empty(arr):
        return np.fromiter((len(s) == 0 for s in arr), dtype=bool, count=arr.size)

    def cp(scorer, a, b, scale=100.0):
        r = process.cpdist(a.tolist(), b.tolist(), scorer=scorer, workers=workers, dtype=np.float32)
        return np.asarray(r, dtype=np.float32) / scale

    def put(fname, vals, mask=None):
        v = np.asarray(vals, dtype=np.float32)
        if mask is not None:
            v = np.where(mask, np.float32(0.0), v)
        X[:, FI[fname]] = v

    # ---------------- string similarities (cpdist) ----------------
    ne = empty(Q["name"]) | empty(T["name"])
    put("name_set", cp(fuzz.token_set_ratio, Q["name"], T["name"]), ne)
    put("name_sort", cp(fuzz.token_sort_ratio, Q["name"], T["name"]), ne)
    put("name_partial", cp(fuzz.partial_ratio, Q["name"], T["name"]), ne)
    put("name_jw", cp(JaroWinkler.normalized_similarity, Q["name"], T["name"], 1.0), ne)
    ce = empty(Q["core"]) | empty(T["core"])
    put("core_ratio", cp(fuzz.ratio, Q["core"], T["core"]), ce)
    core_set = np.where(ce, 0.0, cp(fuzz.token_set_ratio, Q["core"], T["core"])).astype(np.float32)
    put("core_set", core_set)
    cce = empty(Q["concat"]) | empty(T["concat"])
    put("concat_ratio", cp(fuzz.ratio, Q["concat"], T["concat"]), cce)
    fe = empty(Q["fconcat"]) | empty(T["fconcat"])
    fb = np.maximum(cp(fuzz.partial_ratio, Q["fconcat"], T["fconcat"]),
                    cp(fuzz.ratio, Q["fconcat_del"], T["fconcat"]))
    put("fconcat_best", fb, fe)
    put("skel_ratio", cp(fuzz.ratio, Q["skel"], T["skel"]), empty(Q["skel"]) | empty(T["skel"]))
    put("tok0_ratio", cp(fuzz.ratio, Q["tok0"], T["tok0"]), empty(Q["tok0"]) | empty(T["tok0"]))
    ae = empty(Q["addr"]) | empty(T["addr"])
    addr_set = np.where(ae, 0.0, cp(fuzz.token_set_ratio, Q["addr"], T["addr"])).astype(np.float32)
    put("addr_set", addr_set)
    se = empty(Q["street"]) | empty(T["street"])
    put("street_set", cp(fuzz.token_set_ratio, Q["street"], T["street"]), se)
    put("street_ratio", cp(fuzz.ratio, Q["street"], T["street"]), se)
    put("loc_set", cp(fuzz.token_set_ratio, Q["loc"], T["loc"]), empty(Q["loc"]) | empty(T["loc"]))
    put("city_ratio", cp(fuzz.ratio, Q["city"], T["city"]), empty(Q["city"]) | empty(T["city"]))
    hde = empty(Q["hn_digits"]) | empty(T["hn_digits"])
    lev = np.asarray(process.cpdist(Q["hn_digits"].tolist(), T["hn_digits"].tolist(),
                                    scorer=Levenshtein.distance, workers=workers, dtype=np.int32))
    qdl = np.fromiter((len(s) for s in Q["hn_digits"]), dtype=np.int64, count=n)
    tdl = np.fromiter((len(s) for s in T["hn_digits"]), dtype=np.int64, count=n)

    # ---------------- numpy features ----------------
    qn, tn = qs.num, ts.num

    def g(store_num, key, idx):
        return store_num[key][idx]

    put("t_is_domain", g(tn, "is_domain", ti))
    put("t_is_garbage", g(tn, "is_garbage", ti))
    put("t_has_alias", g(tn, "has_alias", ti))
    put("t_non_latin", g(tn, "non_latin", ti))
    put("country_tok_added", (g(tn, "ctok", ti) > 0) & (g(qn, "ctok", qi) == 0))
    lq = _legal_group(g(qn, "legal", qi).astype(np.int64))
    lt = _legal_group(g(tn, "legal", ti).astype(np.int64))
    put("legal_conflict", (lq != 0) & (lt != 0) & ((lq & lt) == 0))
    put("legal_added", (lq == 0) & (lt != 0))
    put("legal_dropped", (lq != 0) & (lt == 0))
    rq, rt = g(qn, "raw_sig_h", qi), g(tn, "raw_sig_h", ti)
    raw_eq = (rq != 0) & (rq == rt)
    put("raw_sig_eq", raw_eq)
    cq, ct = g(qn, "clean_sig_h", qi), g(tn, "clean_sig_h", ti)
    put("clean_sig_eq", (cq != 0) & (cq == ct))
    sq, stt = g(qn, "state_h", qi), g(tn, "state_h", ti)
    both_state = (sq != 0) & (stt != 0)
    state_conflict = both_state & (sq != stt)
    put("state_match", both_state & (sq == stt))
    put("state_conflict", state_conflict)
    cyq, cyt = g(qn, "city_h", qi), g(tn, "city_h", ti)
    both_city = (cyq != 0) & (cyt != 0)
    put("city_eq", both_city & (cyq == cyt))
    tyq, tyt = g(qn, "stype_h", qi), g(tn, "stype_h", ti)
    put("stype_eq", np.where((tyq != 0) & (tyt != 0), (tyq == tyt).astype(np.float32), 0.5))
    put("q_name_s1dup", np.log1p(g(qn, "s1_name_other", qi)))
    put("t_name_freq", np.log1p(g(tn, "t_name_other", ti)))
    put("q_core_in_t", np.log1p(g(qn, "core_in_targets", qi)))
    put("q_addr_s1dup", np.log1p(g(qn, "s1_addr_other", qi)))
    put("t_addr_key_freq", np.log1p(g(tn, "t_akey_other", ti)))
    hq, ht = g(qn, "hnum_h", qi), g(tn, "hnum_h", ti)
    hq_p, ht_p = hq != 0, ht != 0
    both_hn = hq_p & ht_p
    hn_exact = both_hn & (hq == ht)
    put("hn_q_present", hq_p)
    put("hn_t_present", ht_p)
    put("hn_exact", hn_exact)
    dq, dt = g(qn, "hn_digits_h", qi), g(tn, "hn_digits_h", ti)
    digits_eq = both_hn & (dq != 0) & (dq == dt)
    put("hn_digits_eq", digits_eq)
    fq, ft = g(qn, "hn_sfx_h", qi), g(tn, "hn_sfx_h", ti)
    put("hn_sfx_both_diff", digits_eq & (fq != 0) & (ft != 0) & (fq != ft))
    put("hn_sfx_one_missing", digits_eq & ((fq != 0) ^ (ft != 0)))
    drop1 = both_hn & ~hde & (lev == 1) & (np.abs(qdl - tdl) == 1)
    sub1 = both_hn & ~hde & (lev == 1) & (qdl == tdl)
    put("hn_drop1", drop1)
    put("hn_sub1", sub1)
    npq, npt = g(qn, "hn_nparts", qi), g(tn, "hn_nparts", ti)
    single = both_hn & (npq == 1) & (npt == 1)
    a = g(qn, "hn_d", qi).astype(np.float64)
    b = g(tn, "hn_d", ti).astype(np.float64)
    ok = single & (a >= 0) & (b >= 0)
    diff = np.abs(a - b)
    put("hn_logdiff", np.where(ok, np.log1p(diff), -1.0))
    put("hn_reldiff", np.where(ok, diff / np.maximum(np.maximum(a, b), 1.0), -1.0))
    put("hn_is_compound", (npq >= 2) | (npt >= 2))
    put("hn_clean_both", (g(qn, "hn_clean", qi) > 0) & (g(tn, "hn_clean", ti) > 0))
    uq_h, ut_h = g(qn, "unit_h", qi), g(tn, "unit_h", ti)
    both_unit = (uq_h != 0) & (ut_h != 0)
    put("unit_eq", both_unit & (uq_h == ut_h))
    put("unit_diff", both_unit & (uq_h != ut_h))
    put("q_addr_empty", g(qn, "addr_empty", qi))
    put("t_addr_empty", g(tn, "addr_empty", ti))
    put("target_source", g(tn, "source", ti))
    name_set = X[:, FI["name_set"]]
    put("composite", np.sqrt(np.maximum(name_set, 0) * np.maximum(addr_set, 0)))

    # ---------------- per-pair Python loop ----------------
    def idf(tok):
        return math.log((n_docs + 1.0) / (idf_df.get(tok, 0) + 1.0))

    def tokinfo(cores):
        out = []
        for s in cores:
            toks = s.split()
            idfs = [idf(t) for t in toks]
            out.append((toks, set(toks), idfs, sum(idfs)))
        return out

    qinfo, tinfo = tokinfo(Qu["core"]), tokinfo(Tu["core"])
    q_cn = [set(s.split()) for s in Qu["cnums"]]
    t_cn = [set(s.split()) for s in Tu["cnums"]]
    q_dg = [set(s.split()) for s in Qu["digits"]]
    t_dg = [set(s.split()) for s in Tu["digits"]]
    q_st = [s.split() for s in Qu["street"]]
    t_st = [s.split() for s in Tu["street"]]
    q_hp = [_parts(s) for s in Qu["hnum"]]
    t_hp = [_parts(s) for s in Tu["hnum"]]
    q_hi = qn["hn_hi"][uq].tolist()
    t_hi = tn["hn_hi"][ut].tolist()
    q_hd = qn["hn_d"][uq].tolist()
    t_hd = tn["hn_d"][ut].tolist()

    cov_q = np.zeros(n, np.float32)
    cov_t = np.zeros(n, np.float32)
    idf_ov = np.zeros(n, np.float32)
    extra = np.zeros(n, np.float32)
    missing = np.zeros(n, np.float32)
    decoy = np.zeros(n, np.float32)
    noise_only = np.zeros(n, np.float32)
    acro = np.zeros(n, np.float32)
    cn_rec = np.zeros(n, np.float32)
    cn_extra = np.zeros(n, np.float32)
    dg_rec = np.zeros(n, np.float32)
    st_cont = np.full(n, -1.0, np.float32)
    prefix_suffix = np.zeros(n, np.float32)
    sub1_last = np.zeros(n, np.float32)
    chain = np.zeros(n, np.float32)
    last_diff = np.zeros(n, np.float32)
    last_delta = np.zeros(n, np.float32)
    range_hit = np.zeros(n, np.float32)
    rel = np.zeros(n, np.int8)

    qacr, qshort, tacr, tshort = (Q["acr"].tolist(), Q["short"].tolist(), T["acr"].tolist(),
                                  T["short"].tolist())
    qdig, tdig, qsfx, tsfx = (Q["hn_digits"].tolist(), T["hn_digits"].tolist(), Q["hn_sfx"].tolist(),
                              T["hn_sfx"].tolist())
    qinv_l, tinv_l = qinv.tolist(), tinv.tolist()
    hn_exact_l, digits_eq_l, drop1_l = hn_exact.tolist(), digits_eq.tolist(), drop1.tolist()

    for i in range(n):
        a_, b_ = qinv_l[i], tinv_l[i]
        qt, qset, qidf, qsum = qinfo[a_]
        tt, tset, tidf, tsum = tinfo[b_]
        # token coverage / IDF
        if qt:
            mq = 0
            ex = 0.0
            miss = 0.0
            for k, tok in enumerate(qt):
                if tok in tset:
                    mq += 1
                    ex += qidf[k]
                elif len(tok) >= 3 and any(len(x) >= 3 and _ratio(tok, x) >= 80 for x in tt):
                    mq += 1
                else:
                    miss += qidf[k]
            cov_q[i] = mq / len(qt)
            idf_ov[i] = ex / qsum if qsum > 0 else 0.0
            missing[i] = miss
        if tt:
            unmatched = [tok for tok in tt if tok not in qset and not (
                len(tok) >= 3 and any(len(x) >= 3 and _ratio(tok, x) >= 80 for x in qt))]
            cov_t[i] = 1.0 - len(unmatched) / len(tt)
            if unmatched:
                e = 0.0
                d = 0.0
                allnoise = True
                for tok in unmatched:
                    if tok in NOISE_WORDS:
                        continue
                    allnoise = False
                    e += idf(tok)
                    if tok in DECOY_WORDS:
                        d = 1.0
                extra[i] = min(e, 40.0)
                decoy[i] = d
                noise_only[i] = 1.0 if allnoise else 0.0
        qa, ts_, ta, qs_ = qacr[i], tshort[i], tacr[i], qshort[i]
        if (qa and qa == ts_) or (qs_ and qs_ == ta):
            acro[i] = 1.0
        # compound numbers
        qc, tc = q_cn[a_], t_cn[b_]
        if qc:
            m = sum(1 for c in qc if _cnum_match(c, tc))
            cn_rec[i] = m / len(qc)
        if tc:
            cn_extra[i] = sum(1 for c in tc if not _cnum_match(c, qc))
        qd_, td_ = q_dg[a_], t_dg[b_]
        if qd_:
            dg_rec[i] = len(qd_ & td_) / len(qd_)
        # street containment
        sa, sb = q_st[a_], t_st[b_]
        if sa and sb:
            st_cont[i] = max(_contain(sa, sb), _contain(sb, sa))
        # house-number relation
        pa, pb = q_hp[a_], t_hp[b_]
        if not pa or not pb:
            rel[i] = HN_MISSING
            continue
        if hn_exact_l[i]:
            rel[i] = HN_EXACT
            continue
        dqs, dts = qdig[i], tdig[i]
        if digits_eq_l[i]:
            if qsfx[i] and tsfx[i] and qsfx[i] != tsfx[i]:
                rel[i] = HN_CONFLICT
            else:
                rel[i] = HN_EXACT
            continue
        qh_, th_, qhd, thd = q_hi[a_], t_hi[b_], q_hd[a_], t_hd[b_]
        if (qh_ >= 0 and thd >= 0 and qhd <= thd <= qh_) or (th_ >= 0 and qhd >= 0 and thd <= qhd <= th_):
            range_hit[i] = 1.0
            rel[i] = HN_EXACT
            continue
        if _chain(pa, pb):
            chain[i] = 1.0
        if len(pa) == len(pb) and len(pa) >= 2 and pa[:-1] == pb[:-1] and pa[-1] != pb[-1]:
            last_diff[i] = 1.0
            if pa[-1].isdigit() and pb[-1].isdigit():
                last_delta[i] = math.log1p(abs(int(pa[-1][:9]) - int(pb[-1][:9])))
        if dqs and dts and len(dqs) != len(dts) and (
                dqs.startswith(dts) or dqs.endswith(dts) or dts.startswith(dqs) or dts.endswith(dqs)):
            prefix_suffix[i] = 1.0
        if dqs and dts and len(dqs) == len(dts) and dqs[:-1] == dts[:-1] and dqs[-1] != dts[-1]:
            sub1_last[i] = 1.0
        if chain[i] or prefix_suffix[i] or drop1_l[i]:
            rel[i] = HN_SOFT
        else:
            rel[i] = HN_CONFLICT

    put("cov_q", cov_q)
    put("cov_t", cov_t)
    put("idf_overlap", idf_ov)
    put("extra_t_idf", extra)
    put("missing_q_idf", missing)
    put("t_extra_decoy", decoy)
    put("t_extra_noise_only", noise_only)
    put("acronym_match", acro)
    put("q_cnum_recall", cn_rec)
    put("t_cnum_extra", cn_extra)
    put("q_digit_recall", dg_rec)
    put("street_cont", st_cont)
    put("hn_prefix_suffix", prefix_suffix)
    put("hn_sub1_last", sub1_last)
    put("hn_prefix_chain", chain)
    put("hn_lastpart_diff", last_diff)
    put("hn_lastpart_absdelta", last_delta)
    put("hn_range_hit", range_hit)
    put("hn_rel", rel)

    # ---------------- gates ----------------
    gate = np.zeros(n, np.uint8)
    geo = state_conflict & (core_set < 0.9) & ~hn_exact & ~raw_eq
    gate[geo] |= G_GEO
    if country == "France":
        clean_both = X[:, FI["hn_clean_both"]] > 0
        gate[both_hn & clean_both & (rel != HN_EXACT)] |= G_FR_NUM
        gate[both_city & (cyq != cyt)] |= G_FR_CITY
        gate[(st_cont >= 0) & (st_cont < 0.5)] |= G_FR_STREET
    aux["gate"] = gate
    aux["hn_rel"] = rel
    aux["hn_h"] = ht.astype(np.uint64)
    aux["st_h"] = g(tn, "street_h", ti).astype(np.uint64)
    aux["core_set"] = core_set
    return X, aux


def check_ranges(X):
    """Smoke assertion: similarity features must lie in [0, 1]."""
    sims = ["name_set", "name_sort", "name_partial", "name_jw", "core_ratio", "core_set",
            "concat_ratio", "fconcat_best", "skel_ratio", "tok0_ratio", "addr_set", "street_set",
            "street_ratio", "loc_set", "city_ratio", "cov_q", "cov_t", "idf_overlap",
            "q_cnum_recall", "q_digit_recall"]
    bad = []
    for f in sims:
        col = X[:, FI[f]]
        if col.size and (np.nanmin(col) < -1e-6 or np.nanmax(col) > 1 + 1e-6):
            bad.append((f, float(np.nanmin(col)), float(np.nanmax(col))))
    if np.isnan(X).any():
        bad.append(("NaN present", float(np.isnan(X).sum()), 0.0))
    return bad
