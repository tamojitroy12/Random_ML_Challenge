"""
Training + validation for business entity resolution.

Stages (each resumable from output/_work/train, or output/_work_smoke/train with --smoke):
  select   choose TRAIN_S1 (random, disjoint from VAL_S1) per country; density ratios r
  lexicon  native-script -> Latin lexicon from train GT (excluding TRAIN_S1 and VAL_S1)
  passA    per country, in a fresh subprocess: parse all S1 + targets, index, forward-block
           TRAIN_S1 and VAL_S1, features, recall@K, model-free competitor pairs that mirror
           test-time exclusivity (reverse-block -> forward-check), save npz
  train    LightGBM on TRAIN_S1 pairs (checkpoint saved immediately)
  tune     predict val + competitor pairs, grid-search tau/delta/veto/rules on the
           density-adjusted F0.5, report, write thresholds.json + val_predictions.tsv

Run from student_resource/:
  .\\.venv\\Scripts\\python.exe -u code/business_entity_resolution/src/train_evaluate.py [--smoke]
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np
import polars as pl

BER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BER_DIR, "..", ".."))
if BER_DIR not in sys.path:
    sys.path.insert(0, BER_DIR)

from src import lexicon as lexmod  # noqa: E402
from src import normalizer  # noqa: E402
from src.blocking import HashIndex  # noqa: E402
from src.feature_extractor import FEATURE_NAMES, check_ranges, compute_features  # noqa: E402
from src.model import (FR_GATES, G_GEO, MatchingModel, evaluate_params, f05_stats,  # noqa: E402
                       no_exclusivity_stats, summarize, tune_country)
from src.records import RecordStore, add_counts, read_tsv  # noqa: E402
from src.runtime import (Timer, atomic_savez, atomic_write_json, fmt_secs, load_npz, log,  # noqa: E402
                         read_json, setup_console, sha1_file)

TRAIN_COUNTRIES = ["US", "India"]
K_LIST = (5, 10, 15, 20, 25)
DENSE_RE = {
    "US": r"(?i)(^|,)\s*(vt|vermont|wy|wyoming|de|delaware)\s*(,|$)",
    "India": r"(?i)(^|,)\s*(kerala|keralam|kl|punjab|pb|കേരളം|ਪੰਜਾਬ)\s*(,|$)",
}
EDGE_NAME_RE = r'(?i)^\s*(limited|llc|inc|sas|sarl|ltd|pvt ltd|private limited)\s*$|"'
EDGE_ADDR_RE = r"\d{10,}|0000|\bb3/"


def paths(args):
    tag = "_smoke" if args.smoke else ""
    work = os.path.join(ROOT, "output", f"_work{tag}", "train")
    os.makedirs(work, exist_ok=True)
    pre = "smoke_" if args.smoke else ""
    val_dir = os.path.join(ROOT, "benchmark", "val")
    if not os.path.isfile(os.path.join(val_dir, "val_source1.tsv")):
        val_dir = os.path.join(BER_DIR, "val")  # copy shipped inside the submission package
    return {
        "work": work,
        "select": os.path.join(work, "select.json"),
        "model": os.path.join(BER_DIR, f"{pre}model_checkpoint.joblib"),
        "thr": os.path.join(BER_DIR, f"{pre}thresholds.json"),
        "lex": os.path.join(BER_DIR, f"{pre}native_lexicon.json"),
        "val_s1": os.path.join(val_dir, "val_source1.tsv"),
        "val_gt": os.path.join(val_dir, "val_ground_truth.tsv"),
        "train_dir": os.path.join(ROOT, "dataset", "train"),
        "test_dir": os.path.join(ROOT, "dataset", "test"),
        "val_pred": os.path.join(val_dir, f"{pre}val_predictions.tsv"),
    }


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def index_of(ids_series, wanted):
    """Positions of `wanted` ids (in wanted order) inside ids_series; -1 when missing."""
    base = pl.DataFrame({"id": ids_series, "pos": np.arange(len(ids_series), dtype=np.int64)})
    w = pl.DataFrame({"id": pl.Series(list(wanted), dtype=pl.String),
                      "ord": np.arange(len(wanted), dtype=np.int64)})
    j = w.join(base, on="id", how="left").sort("ord")
    return j.get_column("pos").fill_null(-1).to_numpy().astype(np.int64)


def read_gt(path):
    return read_tsv(path, columns=("source1_entity_id", "matched_entity_ids"))


def explode_gt(gt):
    return (gt.filter(pl.col("matched_entity_ids").str.len_bytes() > 0)
            .with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids")
            .rename({"matched_entity_ids": "tid"})
            .with_columns(pl.col("tid").str.strip_chars())
            .filter(pl.col("tid").str.len_bytes() > 0))


def country_counts(path):
    lf = pl.scan_csv(path, separator="\t", quote_char='"', infer_schema=False)
    df = lf.group_by("country").len().collect()
    return dict(zip(df.get_column("country").to_list(), df.get_column("len").to_list()))


def target_frame(train_dir, country, prefix="train"):
    parts = []
    for i, src in ((1, "source2"), (2, "source3")):
        df = read_tsv(os.path.join(train_dir, f"{prefix}_{src}.tsv"), country=country)
        parts.append(df.with_columns(pl.lit(i, dtype=pl.Int8).alias("src")))
    return pl.concat(parts)


def recall_table(gq, gt_t, q, t, rank, src_of_t, n_t, flags=None):
    """Recall@K of GT links (gq, gt_t) within retrieved pairs (q, t, per-source rank)."""
    out = {}
    if gq.size == 0:
        return out
    kr = q.astype(np.int64) * n_t + t.astype(np.int64)
    order = np.argsort(kr)
    ks, rk = kr[order], rank[order]
    kg = gq.astype(np.int64) * n_t + gt_t.astype(np.int64)
    if ks.size:
        pos = np.minimum(np.searchsorted(ks, kg), ks.size - 1)
        hit = ks[pos] == kg
        rg = np.where(hit, rk[pos], 9999)
    else:
        rg = np.full(kg.size, 9999)
    src = src_of_t[gt_t]
    for K in K_LIST:
        out[f"all@{K}"] = float(np.mean(rg < K))
        for s, nm in ((1, "S2"), (2, "S3")):
            m = src == s
            if m.any():
                out[f"{nm}@{K}"] = float(np.mean(rg[m] < K))
    if flags:
        for nm, arr in flags.items():
            m = arr[gt_t] > 0
            if m.any():
                out[f"slice:{nm}@25"] = float(np.mean(rg[m] < 25))
                out[f"slice:{nm}_n"] = int(m.sum())
    return out


# ---------------------------------------------------------------------------------------
# stage: select
# ---------------------------------------------------------------------------------------
def stage_select(args, P):
    if os.path.isfile(P["select"]) and not args.fresh:
        log("select: exists, skipping")
        return
    val = read_tsv(P["val_s1"])
    val_gt = read_gt(P["val_gt"])
    s1_all = read_tsv(os.path.join(P["train_dir"], "train_source1.tsv"))
    sel = {"countries": {}, "r": {}, "counts": {}}
    val_ids_all = set(val.get_column("entity_id").to_list())
    for c in TRAIN_COUNTRIES:
        cv = val.filter(pl.col("country") == c)
        pool = s1_all.filter((pl.col("country") == c) & ~pl.col("entity_id").is_in(list(val_ids_all)))
        if args.smoke:
            dense = pl.col("business_address").str.contains(DENSE_RE[c])
            pool = pool.filter(dense)
            cv = cv.filter(dense)
            if cv.height < 200:
                extra = val.filter(pl.col("country") == c).head(300)
                cv = pl.concat([cv, extra]).unique(subset=["entity_id"], keep="first")
        n_take = min(args.n_train, pool.height)
        tr = pool.sample(n=n_take, seed=42, shuffle=True)
        sel["countries"][c] = {"train": tr.get_column("entity_id").to_list(),
                               "val": cv.get_column("entity_id").to_list()}
        log(f"select {c}: TRAIN_S1={n_take:,} VAL_S1={cv.height:,}")
    # density ratio r = unlinked targets per S1 (test) / (train)
    gt = read_gt(os.path.join(P["train_dir"], "train_ground_truth.tsv"))
    links = explode_gt(gt).join(s1_all.select(pl.col("entity_id").alias("source1_entity_id"), "country"),
                                on="source1_entity_id", how="inner").group_by("country").len()
    L = dict(zip(links.get_column("country").to_list(), links.get_column("len").to_list()))
    cnt = {}
    for split in ("train", "test"):
        for s in ("source1", "source2", "source3"):
            cnt[f"{split}_{s}"] = country_counts(os.path.join(ROOT, "dataset", split, f"{split}_{s}.tsv"))
    for c in TRAIN_COUNTRIES:
        n1_tr = cnt["train_source1"].get(c, 1)
        nt_tr = cnt["train_source2"].get(c, 0) + cnt["train_source3"].get(c, 0)
        n1_te = cnt["test_source1"].get(c, 1)
        nt_te = cnt["test_source2"].get(c, 0) + cnt["test_source3"].get(c, 0)
        l_per = L.get(c, 0) / n1_tr
        r = (nt_te / n1_te - l_per) / max(nt_tr / n1_tr - l_per, 1e-6)
        sel["r"][c] = float(min(max(r, 1.0), 3.0))
        log(f"density {c}: train targets/S1={nt_tr / n1_tr:.2f} test={nt_te / n1_te:.2f} "
            f"links/S1={l_per:.2f} -> r={sel['r'][c]:.2f}")
    sel["counts"] = cnt
    atomic_write_json(P["select"], sel)


# ---------------------------------------------------------------------------------------
# stage: lexicon
# ---------------------------------------------------------------------------------------
def stage_lexicon(args, P):
    if os.path.isfile(P["lex"]) and not args.fresh:
        log(f"lexicon: exists ({P['lex']}), skipping")
        return
    sel = read_json(P["select"])
    exclude = set()
    for c in sel["countries"].values():
        exclude.update(c["train"])
        exclude.update(c["val"])
    val_all = read_tsv(P["val_s1"]).get_column("entity_id").to_list()
    exclude.update(val_all)
    with Timer("lexicon build"):
        lexmod.build_lexicon(P["train_dir"], exclude, P["lex"],
                             s1_limit=150000 if args.smoke else None)


# ---------------------------------------------------------------------------------------
# stage: passA (runs in its own process per country)
# ---------------------------------------------------------------------------------------
def _features_in_chunks(S1, T, tidx, ctx, qidx, k, chunk, label):
    Xs, qs, ts, rks, auxs = [], [], [], [], []
    t_block = t_feat = 0.0
    n_pairs = 0
    for s in range(0, qidx.size, chunk):
        idx = qidx[s:s + chunk]
        t0 = time.time()
        kp, ka = S1.keys_for(idx)
        qrow, t, _, rk = tidx.query_many(kp, ka, range(idx.size), k)
        q = idx[qrow]
        t1 = time.time()
        X, aux = compute_features(S1, T, q, t, ctx)
        t2 = time.time()
        t_block += t1 - t0
        t_feat += t2 - t1
        n_pairs += q.size
        Xs.append(X)
        qs.append(q)
        ts.append(t)
        rks.append(rk)
        auxs.append(aux)
        log(f"  {label}: {min(s + chunk, qidx.size):,}/{qidx.size:,} queries, pairs={n_pairs:,}")
    cat = (lambda a: np.concatenate(a) if a else np.zeros(0))
    X = np.concatenate(Xs) if Xs else np.zeros((0, len(FEATURE_NAMES)), np.float32)
    aux = {key: np.concatenate([a[key] for a in auxs]) if auxs else np.zeros(0)
           for key in ("gate", "hn_rel", "hn_h", "st_h", "core_set")}
    timing = {"block_ms_per_query": 1000 * t_block / max(qidx.size, 1),
              "feat_us_per_pair": 1e6 * t_feat / max(n_pairs, 1), "pairs": n_pairs,
              "queries": int(qidx.size)}
    return X, cat(qs).astype(np.int64), cat(ts).astype(np.int64), cat(rks).astype(np.int16), aux, timing


def stage_passA_worker(args, P, country):
    out_npz = os.path.join(P["work"], f"passA_{country}.npz")
    out_json = os.path.join(P["work"], f"passA_{country}.json")
    normalizer.STRICT = bool(args.smoke)
    sel = read_json(P["select"])
    csel = sel["countries"][country]
    r = sel["r"][country]
    lex = lexmod.load_lexicon(P["lex"])
    normalizer.set_lexicon(lex)
    stats = {"country": country, "timing": {}}

    with Timer(f"[{country}] read TSVs"):
        s1_df = read_tsv(os.path.join(P["train_dir"], "train_source1.tsv"), country=country)
        t_df = target_frame(P["train_dir"], country)
        gt = read_gt(os.path.join(P["train_dir"], "train_ground_truth.tsv"))
        gt_links = explode_gt(gt)
        if args.smoke:
            keep_ids = set(csel["train"]) | set(csel["val"])
            s1_df = s1_df.filter(pl.col("business_address").str.contains(DENSE_RE[country])
                                 | pl.col("entity_id").is_in(list(keep_ids))
                                 | pl.col("business_address").str.contains(EDGE_ADDR_RE))
            need_t = gt_links.filter(pl.col("source1_entity_id").is_in(list(keep_ids))).get_column("tid")
            null_addr = t_df.filter(pl.col("business_address").str.len_bytes() == 0).head(20000)
            t_df = pl.concat([
                t_df.filter(pl.col("business_address").str.contains(DENSE_RE[country])
                            | pl.col("entity_id").is_in(need_t)
                            | pl.col("business_name").str.contains(EDGE_NAME_RE)
                            | pl.col("business_address").str.contains(EDGE_ADDR_RE)),
                null_addr]).unique(subset=["entity_id"], keep="first")
    log(f"[{country}] S1 rows={s1_df.height:,} target rows={t_df.height:,}")

    t0 = time.time()
    S1 = RecordStore.build(s1_df, country, 0, lex, args.workers, want_keys=True, keep_keys=True,
                           strict=bool(args.smoke), label=f"{country} S1")
    t_s1 = time.time() - t0
    del s1_df
    t0 = time.time()
    T = RecordStore.build(t_df, country, t_df.get_column("src").to_numpy(), lex, args.workers,
                          want_keys=True, keep_keys=False, strict=bool(args.smoke), label=f"{country} T")
    t_t = time.time() - t0
    del t_df
    gc.collect()
    stats["timing"]["parse_us_per_record"] = 1e6 * (t_s1 + t_t) / max(S1.n + T.n, 1)
    stats["parse_errors"] = {"S1": S1.errors, "T": T.errors}
    log(f"[{country}] parsed S1={S1.n:,} T={T.n:,} ({stats['timing']['parse_us_per_record']:.0f} us/record)")

    add_counts(S1, T)
    t0 = time.time()
    tidx = HashIndex.build(T.key_batches, T.n, T.num["source"], consume=True)
    T.drop_key_batches()
    gc.collect()
    stats["timing"]["index_us_per_record"] = 1e6 * (time.time() - t0) / max(T.n, 1)
    log(f"[{country}] target index {tidx.stats()}")
    ctx = {"country": country, "idf": T.token_df(), "n_docs": T.n}

    # id maps
    train_q = index_of(S1.ids, csel["train"])
    val_q_all = index_of(S1.ids, csel["val"])
    assert (train_q >= 0).all(), "some TRAIN_S1 ids missing from S1 store"
    val_keep = val_q_all >= 0
    val_ids = [v for v, ok in zip(csel["val"], val_keep) if ok]
    val_q = val_q_all[val_keep]
    assert np.intersect1d(train_q, val_q).size == 0, "TRAIN_S1 and VAL_S1 overlap"
    t_pos = pl.DataFrame({"tid": T.ids, "tpos": np.arange(T.n, dtype=np.int64)})
    s1_pos = pl.DataFrame({"source1_entity_id": S1.ids, "qpos": np.arange(S1.n, dtype=np.int64)})
    linked_ids = gt_links.select("tid").unique()
    linked = np.zeros(T.n, bool)
    lp = t_pos.join(linked_ids, on="tid", how="inner").get_column("tpos").to_numpy()
    linked[lp] = True
    sub_gt = gt_links.filter(pl.col("source1_entity_id").is_in(csel["train"] + val_ids))
    gj = sub_gt.join(s1_pos, on="source1_entity_id", how="inner").join(t_pos, on="tid", how="inner")
    gq = gj.get_column("qpos").to_numpy().astype(np.int64)
    gtt = gj.get_column("tpos").to_numpy().astype(np.int64)
    gt_key = np.unique(gq * T.n + gtt)
    del gt, gt_links, sub_gt, gj
    gc.collect()

    # --- TRAIN_S1 pairs
    Xtr, qtr, ttr, rktr, _, tim_tr = _features_in_chunks(S1, T, tidx, ctx, train_q, args.k, args.chunk,
                                                         f"{country} train")
    ytr = np.isin(qtr * T.n + ttr, gt_key)
    wtr = np.where(~ytr & ~linked[ttr], r, 1.0).astype(np.float32)
    stats["timing"]["train"] = tim_tr
    # --- VAL_S1 pairs
    Xv, qv, tv, rkv, auxv, tim_v = _features_in_chunks(S1, T, tidx, ctx, val_q, args.k, args.chunk,
                                                       f"{country} val")
    yv = np.isin(qv * T.n + tv, gt_key)
    stats["timing"]["val"] = tim_v
    if args.smoke:
        bad = check_ranges(Xv)
        log(f"[{country}] feature range check: {'OK' if not bad else bad}")
        assert not bad, f"feature range problems: {bad}"

    flags = {k: T.num[k] for k in ("addr_empty", "is_domain", "is_garbage", "has_alias", "non_latin")}
    is_val_gt = np.isin(gq, val_q)
    stats["recall_val"] = recall_table(gq[is_val_gt], gtt[is_val_gt], qv, tv, rkv, T.num["source"], T.n, flags)
    stats["recall_train"] = recall_table(gq[~is_val_gt], gtt[~is_val_gt], qtr, ttr, rktr, T.num["source"], T.n)
    log(f"[{country}] val recall: " + ", ".join(f"{k}={v:.4f}" for k, v in stats["recall_val"].items()
                                                if not k.endswith("_n")))

    # --- competitors: reverse-block val candidates, keep S1' only if T is in S1' forward top-K
    t0 = time.time()
    t_sel = np.unique(tv[rkv < 8])
    excl = np.zeros(S1.n, bool)
    excl[train_q] = True
    sidx = HashIndex.build(S1.key_batches, S1.n, np.zeros(S1.n, np.int8), exclude=excl, consume=True)
    S1.drop_key_batches()
    kp, ka = T.keys_for(t_sel)
    rrow, s1p, _, _ = sidx.query_many(kp, ka, range(t_sel.size), 6)
    del sidx
    rev_q = s1p.astype(np.int64)
    rev_t = t_sel[rrow]
    uq = np.unique(rev_q)
    kp2, ka2 = S1.keys_for(uq)
    frow, ft, _, _ = tidx.query_many(kp2, ka2, range(uq.size), args.k)
    fwd_key = np.unique(uq[frow] * T.n + ft.astype(np.int64))
    rev_key = rev_q * T.n + rev_t
    keep = np.isin(rev_key, fwd_key) & ~np.isin(rev_key, qv * T.n + tv)
    cq, ct = rev_q[keep], rev_t[keep]
    ckey, cu = np.unique(cq * T.n + ct, return_index=True)
    cq, ct = cq[cu], ct[cu]
    log(f"[{country}] competitors: T_sel={t_sel.size:,} reverse pairs={rev_key.size:,} "
        f"S1'={uq.size:,} kept={cq.size:,}")
    Xc_parts, auxc_parts = [], []
    for s in range(0, cq.size, 200000):
        Xc, auxc = compute_features(S1, T, cq[s:s + 200000], ct[s:s + 200000], ctx)
        Xc_parts.append(Xc)
        auxc_parts.append(auxc)
    Xc = np.concatenate(Xc_parts) if Xc_parts else np.zeros((0, len(FEATURE_NAMES)), np.float32)
    auxc = {key: np.concatenate([a[key] for a in auxc_parts]) if auxc_parts else np.zeros(0)
            for key in ("gate", "hn_rel", "hn_h", "st_h", "core_set")}
    stats["timing"]["competitors_s"] = time.time() - t0

    val_gt = read_gt(P["val_gt"])
    ntrue = dict(zip(val_gt.get_column("source1_entity_id").to_list(),
                     [len([x for x in s.split(",") if x.strip()]) if s else 0
                      for s in val_gt.get_column("matched_entity_ids").to_list()]))
    n_true = np.array([ntrue.get(v, 0) for v in val_ids], np.int64)

    atomic_savez(
        out_npz,
        X_train=Xtr, y_train=ytr.astype(np.int8), w_train=wtr, g_train=qtr,
        X_val=Xv, q_val=qv, t_val=tv, rank_val=rkv, y_val=yv.astype(np.int8),
        linked_val=linked[tv].astype(np.int8), gate_val=auxv["gate"], hnrel_val=auxv["hn_rel"],
        hnh_val=auxv["hn_h"], sth_val=auxv["st_h"], coreset_val=auxv["core_set"],
        tid_val=np.array(T.ids.gather(tv).to_list(), dtype="U"),
        X_comp=Xc, q_comp=cq, t_comp=ct, linked_comp=linked[ct].astype(np.int8),
        gate_comp=auxc["gate"], hnrel_comp=auxc["hn_rel"], hnh_comp=auxc["hn_h"],
        sth_comp=auxc["st_h"], coreset_comp=auxc["core_set"],
        val_q=val_q, val_ids=np.array(val_ids, dtype="U"), n_true=n_true,
        train_q=train_q, n_q=np.array([S1.n]), n_t=np.array([T.n]),
    )
    stats["n"] = {"S1": S1.n, "T": T.n, "train_pairs": int(qtr.size), "val_pairs": int(qv.size),
                  "comp_pairs": int(cq.size), "train_pos": int(ytr.sum()), "val_pos": int(yv.sum())}
    atomic_write_json(out_json, stats)
    log(f"[{country}] passA saved -> {out_npz}")


def stage_passA(args, P):
    for c in TRAIN_COUNTRIES:
        out_npz = os.path.join(P["work"], f"passA_{c}.npz")
        if os.path.isfile(out_npz) and not args.fresh:
            log(f"passA {c}: exists, skipping")
            continue
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--stage", "passA_worker", "--country", c,
               "--workers", str(args.workers), "--chunk", str(args.chunk), "--k", str(args.k),
               "--n-train", str(args.n_train)]
        if args.smoke:
            cmd.append("--smoke")
        log(f"passA {c}: launching subprocess")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise SystemExit(f"passA worker for {c} failed with exit code {rc}")


# ---------------------------------------------------------------------------------------
# stage: train
# ---------------------------------------------------------------------------------------
def _choose_k(P):
    rec = {}
    for c in TRAIN_COUNTRIES:
        st = read_json(os.path.join(P["work"], f"passA_{c}.json"), {})
        for K in K_LIST:
            rec.setdefault(K, []).append(st.get("recall_val", {}).get(f"all@{K}", 0.0))
    avg = {K: float(np.mean(v)) for K, v in rec.items()}
    k = 25
    for K in (15, 20, 25):
        if avg.get(25, 0) - avg.get(K, 0) <= 0.001:
            k = K
            break
    return k, avg


def stage_train(args, P):
    Xs, ys, ws, gs = [], [], [], []
    for ci, c in enumerate(TRAIN_COUNTRIES):
        d = load_npz(os.path.join(P["work"], f"passA_{c}.npz"))
        if d is None:
            raise SystemExit(f"missing passA output for {c}")
        assert np.intersect1d(d["g_train"], d["val_q"]).size == 0, "VAL_S1 leaked into training pairs"
        Xs.append(d["X_train"])
        ys.append(d["y_train"])
        ws.append(d["w_train"])
        gs.append(d["g_train"] + ci * 10_000_000)
        del d
        gc.collect()
    X = np.concatenate(Xs)
    y = np.concatenate(ys).astype(np.float32)
    w = np.concatenate(ws)
    g = np.concatenate(gs)
    del Xs, ys, ws, gs
    holdout = ((g * 2654435761) % 1000) < 100
    log(f"train: pairs={X.shape[0]:,} positives={int(y.sum()):,} holdout={int(holdout.sum()):,} "
        f"features={X.shape[1]}")
    k_infer, rec = _choose_k(P)
    model = MatchingModel()
    with Timer("LightGBM training"):
        model.fit(X, y, w, holdout)
    model.meta.update({
        "k_infer": int(k_infer), "recall_by_k": rec, "lexicon_sha1": sha1_file(P["lex"]),
        "parser_version": normalizer.PARSER_VERSION, "n_train_pairs": int(X.shape[0]),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": bool(args.smoke),
    })
    model.save(P["model"])
    log(f"checkpoint saved -> {P['model']} (k_infer={k_infer}, best_iter={model.meta['best_iteration']})")
    log("top features by gain:")
    for f, v in model.importance(25):
        print(f"    {f:24s} {v:14.1f}")


# ---------------------------------------------------------------------------------------
# stage: tune
# ---------------------------------------------------------------------------------------
def _country_data(model, P, c):
    d = load_npz(os.path.join(P["work"], f"passA_{c}.npz"))
    t0 = time.time()
    pv = model.predict(d["X_val"])
    pc = model.predict(d["X_comp"])
    pred_us = 1e6 * (time.time() - t0) / max(pv.size + pc.size, 1)
    n_q = int(d["n_q"][0])
    val_pos = -np.ones(n_q, np.int64)
    val_pos[d["val_q"]] = np.arange(d["val_q"].size)
    q = np.concatenate([d["q_val"], d["q_comp"]]).astype(np.int64)
    t = np.concatenate([d["t_val"], d["t_comp"]]).astype(np.int64)
    out = {
        "q": q, "t": t, "p": np.concatenate([pv, pc]).astype(np.float32),
        "gate": np.concatenate([d["gate_val"], d["gate_comp"]]).astype(np.uint8),
        "hn_rel": np.concatenate([d["hnrel_val"], d["hnrel_comp"]]).astype(np.int8),
        "hn_h": np.concatenate([d["hnh_val"], d["hnh_comp"]]).astype(np.uint64),
        "st_h": np.concatenate([d["sth_val"], d["sth_comp"]]).astype(np.uint64),
        "core_set": np.concatenate([d["coreset_val"], d["coreset_comp"]]).astype(np.float32),
        "qv": val_pos[q],
        "label": np.concatenate([d["y_val"] > 0, np.zeros(d["q_comp"].size, bool)]),
        "linked": np.concatenate([d["linked_val"] > 0, d["linked_comp"] > 0]),
        "n_true": d["n_true"].astype(np.int64), "n_q": n_q,
        "val_ids": d["val_ids"], "tid_val": d["tid_val"], "n_val_pairs": d["q_val"].size,
        "pred_us_per_row": pred_us,
    }
    from src.model import assert_unique_pairs
    assert_unique_pairs(out["q"], out["t"])
    return out


def _halves(ids):
    h = np.array([hash_str(s) % 2 for s in ids])
    return h == 0


def hash_str(s):
    v = 0
    for ch in s:
        v = (v * 131 + ord(ch)) & 0xFFFFFFFF
    return v


def stage_tune(args, P):
    model = MatchingModel.load(P["model"])
    sel = read_json(P["select"])
    report = {"countries": {}}
    params = {}
    per_query = {}
    for c in TRAIN_COUNTRIES:
        d = _country_data(model, P, c)
        r = sel["r"][c]
        with Timer(f"[{c}] threshold grid"):
            prm, _ = tune_country(d, r)
        acc, st = evaluate_params(d, prm, r)
        _, st_r1 = evaluate_params(d, prm, 1.0)
        noex = no_exclusivity_stats(d, prm, r)
        # 2-fold cross-fit (tune on one half, score the other)
        half = _halves(d["val_ids"].tolist())
        cf = []
        for hsel in (half, ~half):
            prm_h, _ = tune_country(d, r, sel=hsel)
            acc_h, _ = evaluate_params(d, prm_h, r)
            s_h = f05_stats(acc_h, np.where((d["qv"] >= 0) & ~hsel[np.maximum(d["qv"], 0)], d["qv"], -1),
                            d["label"], d["linked"], d["n_true"], r)
            f_other = s_h["per_query_f"][~hsel]
            cf.append(float(f_other.mean()) if f_other.size else float("nan"))
        # error breakdown
        vmask = d["qv"] >= 0
        lab = d["label"] & vmask
        fn_retrieved_below = int((lab & ~acc).sum())
        n_true_total = int(d["n_true"].sum())
        fn_not_retrieved = n_true_total - int(lab.sum())
        params[c] = prm
        per_query[c] = st["per_query_f"]
        rep = {
            "params": prm, "r": r,
            "F": st_r1["F"], "F_adj": st["F_adj"], "singleton_acc": st_r1["singleton_acc"],
            "one_link_F": st_r1["one_link_F"], "nonsingleton_F": st_r1["nonsingleton_F"],
            "macro_P": st_r1["macro_P"], "macro_R": st_r1["macro_R"],
            "F_without_exclusivity": noex["F"], "crossfit_F": cf,
            "FN_not_retrieved": fn_not_retrieved, "FN_below_threshold": fn_retrieved_below,
            "FP_owned_by_other_S1": st_r1["fp_owned"], "FP_unlinked_decoy": st_r1["fp_unlinked"],
            "predict_us_per_row": d["pred_us_per_row"],
        }
        report["countries"][c] = rep
        log(f"[{c}] params={prm}")
        log(f"[{c}] {summarize(st_r1)}")
        log(f"[{c}] F_adj(r={r:.2f})={st['F_adj']:.4f} | without exclusivity F={noex['F']:.4f} | "
            f"cross-fit F={cf} | FN not retrieved={fn_not_retrieved:,} FN below tau={fn_retrieved_below:,}")
        _write_val_predictions(d, acc, c, P, first=(c == TRAIN_COUNTRIES[0]))
        del d
        gc.collect()

    # France / default: conservative combination
    us, ind = params["US"], params["India"]

    def mx(a, b):
        vals = [v for v in (a, b) if v is not None]
        return max(vals) if vals else None

    fr = {"tau": max(us["tau"], ind["tau"]), "delta": max(us["delta"], ind["delta"]),
          "tau1": mx(us["tau1"], ind["tau1"]),
          "tau_lo": (max(us["tau_lo"], ind["tau_lo"]) if (us["tau_lo"] is not None and ind["tau_lo"] is not None)
                     else None),
          "suppress": bool(us["suppress"] or ind["suppress"]),
          "veto": G_GEO if (us["veto"] and ind["veto"]) else 0, "fr_gates": FR_GATES}
    default = dict(fr)
    default["fr_gates"] = 0
    params["France"] = fr
    thr = {"countries": params, "default": default, "r": sel["r"],
           "k_infer": model.meta.get("k_infer", 25), "val_report": report,
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_write_json(P["thr"], thr)

    f_us = report["countries"]["US"]["F"]
    f_in = report["countries"]["India"]["F"]
    allf = np.concatenate([per_query["US"], per_query["India"]])
    print("\n" + "=" * 78)
    print("VALIDATION REPORT (exclusivity simulated with model-free forward-checked competitors)")
    print("=" * 78)
    for c in TRAIN_COUNTRIES:
        rp = report["countries"][c]
        print(f"{c:6s} F={rp['F']:.4f}  F_adj={rp['F_adj']:.4f}  P={rp['macro_P']:.4f}  R={rp['macro_R']:.4f}  "
              f"single_acc={rp['singleton_acc']:.4f}  noExcl={rp['F_without_exclusivity']:.4f}  "
              f"crossfit={['%.4f' % x for x in rp['crossfit_F']]}")
    print(f"pooled val F (official metric over all val S1) = {allf.mean():.4f}")
    mix = 0.45 * f_us + 0.55 * f_in
    print(f"test-mix F (0.45 US + 0.55 India) = {mix:.4f}")
    for ffr in (0.7, 0.8, 0.9):
        print(f"  projected LB if France F={ffr:.1f}: {0.85 * mix + 0.15 * ffr:.4f}")
    print(f"previous per-country val: US 0.8869, India 0.8332 (go/no-go gate)")
    print(f"thresholds -> {P['thr']}")
    print("=" * 78)
    _crosscheck_official(P, allf.mean())


def _write_val_predictions(d, acc, c, P, first):
    nv = d["n_true"].size
    vm = acc & (d["qv"] >= 0)
    lists = [[] for _ in range(nv)]
    idx = np.flatnonzero(vm)
    tids = d["tid_val"]
    for i in idx:
        if i < d["n_val_pairs"]:
            lists[d["qv"][i]].append(str(tids[i]))
    mode = "w" if first else "a"
    with open(P["val_pred"], mode, encoding="utf-8", newline="\n") as f:
        if first:
            f.write("source1_entity_id\tmatched_entity_ids\n")
        for vid, ls in zip(d["val_ids"].tolist(), lists):
            f.write(f"{vid}\t{','.join(ls)}\n")


def _crosscheck_official(P, fast_f):
    try:
        sys.path.insert(0, os.path.join(ROOT, "utils"))
        import evaluate_local
        gt_map = evaluate_local.load_mapping(P["val_gt"], "source1_entity_id", "matched_entity_ids")
        pred_map = evaluate_local.load_mapping(P["val_pred"], "source1_entity_id", "matched_entity_ids")
        written = set(pred_map)
        gt_sub = {k: v for k, v in gt_map.items() if k in written}
        res = evaluate_local.evaluate(gt_sub, pred_map)
        diff = abs(res["macro_f05"] - fast_f)
        log(f"official evaluator on written val S1: F={res['macro_f05']:.6f} (fast={fast_f:.6f}, |diff|={diff:.2e})")
        if diff > 1e-6:
            log("WARNING: fast metric and official evaluator disagree")
    except Exception as exc:
        log(f"official cross-check skipped: {exc!r}")


# ---------------------------------------------------------------------------------------
def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stage", default="all",
                    choices=["all", "select", "lexicon", "passA", "passA_worker", "train", "tune"])
    ap.add_argument("--country", default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if args.n_train is None:
        args.n_train = 2000 if args.smoke else 50000
    if args.workers is None:
        args.workers = 2 if args.smoke else 6
    if args.chunk is None:
        args.chunk = 500 if args.smoke else 20000
    P = paths(args)
    log(f"train_evaluate stage={args.stage} smoke={args.smoke} n_train={args.n_train} "
        f"workers={args.workers} chunk={args.chunk} k={args.k}")
    t0 = time.time()
    if args.stage == "passA_worker":
        stage_passA_worker(args, P, args.country)
        return
    if args.stage in ("all", "select"):
        stage_select(args, P)
    if args.stage in ("all", "lexicon"):
        stage_lexicon(args, P)
    if args.stage in ("all", "passA"):
        stage_passA(args, P)
    if args.stage in ("all", "train"):
        stage_train(args, P)
    if args.stage in ("all", "tune"):
        stage_tune(args, P)
    if args.smoke:
        _smoke_projection(P)
    log(f"done in {fmt_secs(time.time() - t0)}")


def _smoke_projection(P):
    """Project full-run times from smoke measurements."""
    print("\nSMOKE PROJECTIONS (rough):")
    full = {"US": (1_323_633, 6_186_873), "India": (883_188, 4_133_346)}
    total = 0.0
    for c in TRAIN_COUNTRIES:
        st = read_json(os.path.join(P["work"], f"passA_{c}.json"), {})
        tm = st.get("timing", {})
        n1, nt = full[c]
        parse = tm.get("parse_us_per_record", 0) * (n1 + nt) / 1e6
        index = tm.get("index_us_per_record", 0) * nt / 1e6
        q = 50000 + 10000
        block = tm.get("train", {}).get("block_ms_per_query", 0) * q / 1e3
        feat = tm.get("train", {}).get("feat_us_per_pair", 0) * q * 45 / 1e6
        comp = tm.get("competitors_s", 0) * 10
        est = parse + index + block + feat + comp
        total += est
        print(f"  {c}: parse~{fmt_secs(parse)} index~{fmt_secs(index)} block~{fmt_secs(block)} "
              f"features~{fmt_secs(feat)} competitors~{fmt_secs(comp)} => ~{fmt_secs(est)}")
    print(f"  full training pass A total ~{fmt_secs(total)} (+ LightGBM ~5-10 min, tune ~2 min)")
    t_s = read_json(os.path.join(P["work"], "passA_India.json"), {}).get("timing", {})
    pairs = 1_732_544 * 40
    inf = (t_s.get("parse_us_per_record", 0) * 11_700_000 + t_s.get("val", {}).get("feat_us_per_pair", 0) * pairs) / 1e6 \
        + t_s.get("val", {}).get("block_ms_per_query", 0) * 1_732_544 / 1e3
    print(f"  full test inference ~{fmt_secs(inf)} (+ prediction ~{fmt_secs(pairs * 2e-6)})")


if __name__ == "__main__":
    main()
