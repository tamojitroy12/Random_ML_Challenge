"""
Test inference for business entity resolution.

Per country (India, US, France, then any other label) in a fresh subprocess:
  parse all S1 + S2/S3 -> index -> 50k-query chunks: block, features, predict -> atomic npz
  cache (resumable; signature excludes thresholds) -> global exclusive assignment -> per-country
  matching/candidate TSVs + diagnostics.
Then: assemble output/matching_results.tsv + output/candidate_pairs.tsv (every test S1 exactly
once, LF endings), streaming matches-subset-of-candidates check, validator, France tau hint.

  --reassign   re-run the assignment from cached scores with new thresholds (minutes)
  --smoke      small but realistic run into output_smoke/ using the smoke checkpoint

Run from student_resource/:
  .\\.venv\\Scripts\\python.exe -u code/business_entity_resolution/src/pipeline.py [--smoke]
"""

import argparse
import gc
import glob
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
from src.feature_extractor import (FEATURE_NAMES, G_FR_CITY, G_FR_NUM, G_FR_STREET, G_GEO,  # noqa: E402
                                   HN_CONFLICT, check_ranges, compute_features)
from src.model import MatchingModel, assert_unique_pairs, assign_with_params  # noqa: E402
from src.records import RecordStore, add_counts, read_tsv  # noqa: E402
from src.runtime import (atomic_savez, atomic_write_json, fmt_secs, load_npz, log, read_json,  # noqa: E402
                         setup_console, sha1_bytes, sha1_file)

SCORING_FILES = ["normalizer.py", "records.py", "blocking.py", "feature_extractor.py", "lexicon.py"]
ORDER = ["India", "US", "France"]
DENSE_RE = {
    "US": r"(?i)(^|,)\s*(vt|vermont|wy|wyoming|de|delaware)\s*(,|$)",
    "India": r"(?i)(^|,)\s*(kerala|keralam|kl|punjab|pb|കേരളം|ਪੰਜਾਬ)\s*(,|$)",
}
EDGE_NAME_RE = r'(?i)^\s*(limited|llc|inc|sas|sarl|ltd|pvt ltd|private limited)\s*$|"'
EDGE_ADDR_RE = r"\d{10,}|0000|\bb3/"


def paths(args):
    out = args.output_dir or os.path.join(ROOT, "output_smoke" if args.smoke else "output")
    pre = "smoke_" if args.smoke else ""
    return {
        "out": out,
        "work": os.path.join(out, "_work"),
        "model": args.model or os.path.join(BER_DIR, f"{pre}model_checkpoint.joblib"),
        "thr": args.thresholds or os.path.join(BER_DIR, f"{pre}thresholds.json"),
        "lex": os.path.join(BER_DIR, f"{pre}native_lexicon.json"),
        "test_dir": os.path.join(ROOT, "dataset", "test"),
        "match": os.path.join(out, "matching_results.tsv"),
        "cand": os.path.join(out, "candidate_pairs.tsv"),
    }


def safe_name(c):
    return "".join(ch if ch.isalnum() else "_" for ch in (c or "unknown"))


def parse_override(s):
    """'0.8' -> {'*': 0.8}; 'US=0.8,France=0.85' -> {...}."""
    if s is None:
        return {}
    s = s.strip()
    if "=" not in s:
        return {"*": float(s)}
    out = {}
    for part in s.split(","):
        k, v = part.split("=", 1)
        out[k.strip()] = None if v.strip().lower() in ("none", "off") else float(v)
    return out


def params_for(thr, c, args):
    prm = dict(thr.get("countries", {}).get(c) or thr.get("default"))
    for name, ov in (("tau", parse_override(args.tau)), ("delta", parse_override(args.delta)),
                     ("tau1", parse_override(args.tau1))):
        if c in ov:
            prm[name] = ov[c]
        elif "*" in ov:
            prm[name] = ov["*"]
    if args.fr_gates == "off":
        prm["fr_gates"] = 0
    elif args.fr_gates == "on" and c == "France":
        prm["fr_gates"] = G_FR_NUM | G_FR_CITY | G_FR_STREET
    return prm


def static_signature(model, P, k, chunk, smoke):
    parts = [sha1_bytes(model.booster.model_to_string().encode("utf-8")), ",".join(FEATURE_NAMES),
             sha1_file(P["lex"]), normalizer.PARSER_VERSION, str(k), str(chunk), str(bool(smoke))]
    parts += [sha1_file(os.path.join(BER_DIR, "src", f)) for f in SCORING_FILES]
    return sha1_bytes("|".join(parts).encode("utf-8"))


def target_frame(test_dir, country):
    parts = []
    for i, src in ((1, "source2"), (2, "source3")):
        df = read_tsv(os.path.join(test_dir, f"test_{src}.tsv"), country=country)
        parts.append(df.with_columns(pl.lit(i, dtype=pl.Int8).alias("src")))
    return pl.concat(parts)


def smoke_filter(c, s1_df, t_df):
    edge_s1 = s1_df.filter(pl.col("business_address").str.contains(EDGE_ADDR_RE)
                           | pl.col("business_name").str.contains(EDGE_NAME_RE)).head(50)
    if c == "France":
        s1 = pl.concat([s1_df.head(3000), edge_s1]).unique(subset=["entity_id"], keep="first")
        return s1, t_df
    if c in DENSE_RE:
        dense_s1 = s1_df.filter(pl.col("business_address").str.contains(DENSE_RE[c])).head(1500)
        s1 = pl.concat([dense_s1, edge_s1]).unique(subset=["entity_id"], keep="first")
        t = pl.concat([
            t_df.filter(pl.col("business_address").str.contains(DENSE_RE[c])
                        | pl.col("business_name").str.contains(EDGE_NAME_RE)
                        | pl.col("business_address").str.contains(EDGE_ADDR_RE)),
            t_df.filter(pl.col("business_address").str.len_bytes() == 0).head(20000),
        ]).unique(subset=["entity_id"], keep="first")
        return s1, t
    return s1_df.head(1000), t_df


# ---------------------------------------------------------------------------------------
# country worker (subprocess)
# ---------------------------------------------------------------------------------------
def country_worker(args, P, c):
    normalizer.STRICT = bool(args.smoke)
    t_start = time.time()
    model = MatchingModel.load(P["model"])
    thr = read_json(P["thr"])
    if thr is None:
        raise SystemExit(f"thresholds file missing: {P['thr']} (run train_evaluate.py first)")
    lex = lexmod.load_lexicon(P["lex"])
    normalizer.set_lexicon(lex)
    k = int(args.k or model.meta.get("k_infer", 35))
    wd = os.path.join(P["work"], safe_name(c))
    os.makedirs(wd, exist_ok=True)

    s1_df = read_tsv(os.path.join(P["test_dir"], "test_source1.tsv"), country=c)
    t_df = target_frame(P["test_dir"], c)
    if args.smoke:
        s1_df, t_df = smoke_filter(c, s1_df, t_df)
    log(f"[{c}] S1={s1_df.height:,} targets={t_df.height:,} k={k} chunk={args.chunk}")
    if lex and c == "India":
        cov, ntok = lexmod.coverage(t_df.get_column("business_name").head(300000).to_list(), lex)
        log(f"[{c}] lexicon coverage of native tokens in test targets: {cov:.3f} ({ntok:,} tokens)")

    t0 = time.time()
    S1 = RecordStore.build(s1_df, c, 0, lex, args.workers, want_keys=True, keep_keys=True,
                           strict=bool(args.smoke), label=f"{c} S1")
    T = RecordStore.build(t_df, c, t_df.get_column("src").to_numpy(), lex, args.workers,
                          want_keys=True, keep_keys=False, strict=bool(args.smoke), label=f"{c} T")
    del s1_df, t_df
    gc.collect()
    log(f"[{c}] parsed in {fmt_secs(time.time() - t0)}")
    add_counts(S1, T)
    tidx = HashIndex.build(T.key_batches, T.n, T.num["source"], consume=True)
    T.drop_key_batches()
    S1.drop_key_batches()
    gc.collect()
    log(f"[{c}] index {tidx.stats()}")
    ctx = {"country": c, "idf": T.token_df(), "n_docs": T.n}

    sig = static_signature(model, P, k, args.chunk, args.smoke)
    n_chunks = (S1.n + args.chunk - 1) // args.chunk
    old = read_json(os.path.join(wd, "signature.json"))
    if old is None or old.get("static") != sig or old.get("n1") != S1.n or old.get("nt") != T.n:
        for f in glob.glob(os.path.join(wd, "chunk_*.npz")) + glob.glob(os.path.join(wd, "*.tsv")) + \
                glob.glob(os.path.join(wd, ".done")):
            os.remove(f)
        if old is not None:
            log(f"[{c}] signature changed -> cache cleared")
    atomic_write_json(os.path.join(wd, "signature.json"),
                      {"static": sig, "n1": S1.n, "nt": T.n, "n_chunks": n_chunks, "k": k,
                       "chunk": args.chunk})
    pl.DataFrame({"id": S1.ids}).write_ipc(os.path.join(wd, "s1_ids.arrow"))
    pl.DataFrame({"id": T.ids}).write_ipc(os.path.join(wd, "t_ids.arrow"))

    computed = resumed = 0
    t_loop = time.time()
    for ci in range(n_chunks):
        path = os.path.join(wd, f"chunk_{ci:05d}.npz")
        if os.path.isfile(path) and load_npz(path) is not None:
            resumed += 1
            continue
        t0 = time.time()
        idx = np.arange(ci * args.chunk, min((ci + 1) * args.chunk, S1.n), dtype=np.int64)
        kp, ka = S1.keys_for(idx)
        qrow, t, _, _ = tidx.query_many(kp, ka, range(idx.size), k)
        q = idx[qrow]
        t_b = time.time() - t0
        X, aux = compute_features(S1, T, q, t, ctx)
        if args.smoke and computed == 0:
            bad = check_ranges(X)
            log(f"[{c}] feature range check: {'OK' if not bad else bad}")
            assert not bad, bad
        t_f = time.time() - t0 - t_b
        p = model.predict(X)
        atomic_savez(path, q=q.astype(np.int32), t=t.astype(np.int32), p=p.astype(np.float32),
                     gate=aux["gate"], hn_rel=aux["hn_rel"], hn_h=aux["hn_h"], st_h=aux["st_h"],
                     core_set=aux["core_set"])
        computed += 1
        el = time.time() - t0
        msg = (f"[{c}] chunk {ci + 1}/{n_chunks}: {idx.size:,} queries, {q.size:,} pairs, "
               f"block {t_b:.1f}s ({1000 * t_b / max(idx.size, 1):.2f} ms/q), features {t_f:.1f}s "
               f"({1e6 * t_f / max(q.size, 1):.1f} us/pair), total {el:.1f}s")
        if computed == 1:
            msg += f" | projected remaining for {c}: ~{fmt_secs(el * (n_chunks - ci - 1))}"
        log(msg)
    if resumed:
        log(f"[{c}] resumed {resumed}/{n_chunks} chunks from cache")
    log(f"[{c}] scoring done in {fmt_secs(time.time() - t_loop)}")
    prm = params_for(thr, c, args)
    finalize_country(c, wd, prm, S1.ids.to_list(), T.ids.to_list(), n_chunks, sig)
    log(f"[{c}] worker finished in {fmt_secs(time.time() - t_start)}")


def load_country_scores(wd, n_chunks):
    parts = []
    for ci in range(n_chunks):
        d = load_npz(os.path.join(wd, f"chunk_{ci:05d}.npz"))
        if d is None:
            raise RuntimeError(f"missing/corrupt chunk {ci} in {wd}")
        parts.append(d)
    keys = ["q", "t", "p", "gate", "hn_rel", "hn_h", "st_h", "core_set"]
    out = {k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0) for k in keys}
    out["q"] = out["q"].astype(np.int64)
    out["t"] = out["t"].astype(np.int64)
    return out


def finalize_country(c, wd, prm, s1_ids, t_ids, n_chunks, sig):
    d = load_country_scores(wd, n_chunks)
    n_q = len(s1_ids)
    assert_unique_pairs(d["q"], d["t"])
    acc = assign_with_params(d["q"], d["t"], d["p"], d["gate"], d["hn_rel"], d["hn_h"], d["st_h"],
                             d["core_set"], n_q, prm)
    q, t = d["q"], d["t"]
    order = np.lexsort((t, q))
    qs, ts, accs = q[order], t[order], acc[order]
    bounds = np.searchsorted(qs, np.arange(n_q + 1))
    pm, pc = os.path.join(wd, "matching.tsv"), os.path.join(wd, "candidates.tsv")
    with open(pm + ".tmp", "w", encoding="utf-8", newline="\n") as fm, \
            open(pc + ".tmp", "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in range(n_q):
            a, b = bounds[i], bounds[i + 1]
            tt = ts[a:b]
            fc.write(f"{s1_ids[i]}\t{','.join(t_ids[j] for j in tt)}\n")
            fm.write(f"{s1_ids[i]}\t{','.join(t_ids[j] for j, ok in zip(tt, accs[a:b]) if ok)}\n")
    os.replace(pm + ".tmp", pm)
    os.replace(pc + ".tmp", pc)

    links = np.bincount(q[acc], minlength=n_q)
    hist = np.bincount(np.minimum(links, 6), minlength=7)
    acc_n = int(acc.sum())
    decoy_share = float(((d["hn_rel"] == HN_CONFLICT) & acc).sum() / max(acc_n, 1))
    above = d["p"] >= prm["tau"]
    gates = {name: int((((d["gate"] & np.uint8(bit)) != 0) & above).sum())
             for name, bit in (("geo", G_GEO), ("fr_num", G_FR_NUM), ("fr_city", G_FR_CITY),
                               ("fr_street", G_FR_STREET))}
    stats = {"country": c, "n_s1": n_q, "pairs": int(q.size), "accepted": acc_n,
             "mean_links": float(links.mean()) if n_q else 0.0,
             "empty_share": float((links == 0).mean()) if n_q else 0.0,
             "hist_0_to_6plus": hist.tolist(), "accepted_with_hn_conflict_share": decoy_share,
             "gate_vetoes_among_p_ge_tau": gates, "params": prm, "signature": sig}
    atomic_write_json(os.path.join(wd, ".done"), stats)
    log(f"[{c}] params={prm}")
    log(f"[{c}] mean links/S1={stats['mean_links']:.3f} empty={stats['empty_share']:.3%} "
        f"hist(0..6+)={hist.tolist()} hn-conflict share={decoy_share:.3%} gate vetoes={gates}")
    return stats


def reassign_country(c, P, thr, args):
    wd = os.path.join(P["work"], safe_name(c))
    sig = read_json(os.path.join(wd, "signature.json"))
    if sig is None or not os.path.isfile(os.path.join(wd, "s1_ids.arrow")):
        log(f"[{c}] no cache to reassign; skipping")
        return None
    s1_ids = pl.read_ipc(os.path.join(wd, "s1_ids.arrow")).get_column("id").to_list()
    t_ids = pl.read_ipc(os.path.join(wd, "t_ids.arrow")).get_column("id").to_list()
    prm = params_for(thr, c, args)
    return finalize_country(c, wd, prm, s1_ids, t_ids, sig["n_chunks"], sig["static"])


# ---------------------------------------------------------------------------------------
# assemble / validate / France hint
# ---------------------------------------------------------------------------------------
def assemble(P, countries):
    all_ids = read_tsv(os.path.join(P["test_dir"], "test_source1.tsv"),
                       columns=("entity_id",)).get_column("entity_id").to_list()
    written = set()
    viol = 0
    os.makedirs(P["out"], exist_ok=True)
    with open(P["match"] + ".tmp", "w", encoding="utf-8", newline="\n") as fm, \
            open(P["cand"] + ".tmp", "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for c in countries:
            wd = os.path.join(P["work"], safe_name(c))
            pm, pc = os.path.join(wd, "matching.tsv"), os.path.join(wd, "candidates.tsv")
            if not (os.path.isfile(pm) and os.path.isfile(pc)):
                log(f"assemble: WARNING no outputs for {c}")
                continue
            with open(pm, encoding="utf-8") as a, open(pc, encoding="utf-8") as b:
                next(a)
                next(b)
                for la, lb in zip(a, b):
                    ia, ra = la.rstrip("\n").split("\t", 1)
                    ib, rb = lb.rstrip("\n").split("\t", 1)
                    if ia != ib:
                        raise RuntimeError(f"row mismatch in {c}: {ia} vs {ib}")
                    if ia in written:
                        continue
                    if ra and not set(ra.split(",")) <= set(rb.split(",")):
                        viol += 1
                    fm.write(la if la.endswith("\n") else la + "\n")
                    fc.write(lb if lb.endswith("\n") else lb + "\n")
                    written.add(ia)
        missing = 0
        for sid in all_ids:
            if sid not in written:
                fm.write(f"{sid}\t\n")
                fc.write(f"{sid}\t\n")
                written.add(sid)
                missing += 1
    os.replace(P["match"] + ".tmp", P["match"])
    os.replace(P["cand"] + ".tmp", P["cand"])
    log(f"assembled {len(all_ids):,} S1 rows ({missing:,} filled empty, not scored); "
        f"matches-not-in-candidates rows: {viol}")
    log(f"sha1 matching={sha1_file(P['match'])[:12]} candidates={sha1_file(P['cand'])[:12]}")


def run_validator(P):
    cmd = [sys.executable, os.path.join(ROOT, "utils", "validate_submission.py"), "--matching", P["match"],
           "--candidate", os.path.join(P["out"], "__skip_candidate_check__.tsv"), "--test-dir", P["test_dir"]]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    return res.returncode


def france_hint(P, thr, args):
    ref = []
    for c in ("US", "India"):
        st = read_json(os.path.join(P["work"], safe_name(c), ".done"))
        if st:
            ref.append((st["empty_share"], st["mean_links"]))
    wd = os.path.join(P["work"], "France")
    sig = read_json(os.path.join(wd, "signature.json"))
    if not ref or sig is None or not os.path.isfile(os.path.join(wd, ".done")):
        return
    tgt_empty = float(np.mean([r[0] for r in ref]))
    tgt_links = float(np.mean([r[1] for r in ref]))
    d = load_country_scores(wd, sig["n_chunks"])
    n_q = int(sig["n1"])
    base = params_for(thr, "France", args)
    best = None
    for tau in np.round(np.arange(0.30, 0.9701, 0.02), 2):
        prm = dict(base)
        prm["tau"] = float(tau)
        if prm.get("tau1") is not None and prm["tau1"] < tau:
            prm["tau1"] = float(tau)
        acc = assign_with_params(d["q"], d["t"], d["p"], d["gate"], d["hn_rel"], d["hn_h"], d["st_h"],
                                 d["core_set"], n_q, prm)
        links = np.bincount(d["q"][acc], minlength=n_q)
        e, m = float((links == 0).mean()), float(links.mean())
        score = abs(e - tgt_empty) + abs(m - tgt_links) / 4.0
        if best is None or score < best[0]:
            best = (score, float(tau), e, m)
    log(f"France label-free hint: US/India test give empty={tgt_empty:.3%}, links/S1={tgt_links:.2f}; "
        f"France tau={best[1]:.2f} would give empty={best[2]:.3%}, links/S1={best[3]:.2f} "
        f"(current France tau={base['tau']:.2f}). Try: pipeline.py --reassign --tau France={best[1]:.2f}")


# ---------------------------------------------------------------------------------------
def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stage", default="all", choices=["all", "country"])
    ap.add_argument("--country", default=None)
    ap.add_argument("--countries", default=None, help="comma list; default: all test countries")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--reassign", action="store_true")
    ap.add_argument("--tau", default=None)
    ap.add_argument("--delta", default=None)
    ap.add_argument("--tau1", default=None)
    ap.add_argument("--fr-gates", default=None, choices=["on", "off"])
    args = ap.parse_args()
    if args.workers is None:
        args.workers = 2 if args.smoke else 6
    if args.chunk is None:
        args.chunk = 500 if args.smoke else 50000
    P = paths(args)
    os.makedirs(P["work"], exist_ok=True)
    if args.stage == "country":
        country_worker(args, P, args.country)
        return

    t0 = time.time()
    thr = read_json(P["thr"])
    if thr is None:
        raise SystemExit(f"thresholds file missing: {P['thr']} (run train_evaluate.py first)")
    cdf = (pl.scan_csv(os.path.join(P["test_dir"], "test_source1.tsv"), separator="\t", quote_char='"',
                       infer_schema=False).group_by("country").len().collect())
    found = [c for c in cdf.get_column("country").to_list() if c is not None]
    countries = [c for c in ORDER if c in found] + sorted(c for c in found if c not in ORDER)
    if args.countries:
        want = [x.strip() for x in args.countries.split(",")]
        countries = [c for c in countries if c in want]
    log(f"pipeline countries={countries} smoke={args.smoke} reassign={args.reassign} out={P['out']}")

    if args.reassign:
        for c in countries:
            reassign_country(c, P, thr, args)
    else:
        model = MatchingModel.load(P["model"])
        for c in countries:
            wd = os.path.join(P["work"], safe_name(c))
            done = read_json(os.path.join(wd, ".done"))
            sigf = read_json(os.path.join(wd, "signature.json"))
            k = int(args.k or model.meta.get("k_infer", 25))
            sig = static_signature(model, P, k, args.chunk, args.smoke)
            if done and sigf and not args.fresh and sigf.get("static") == sig and done.get("signature") == sig:
                log(f"[{c}] already complete with the same signature -> reassigning only")
                reassign_country(c, P, thr, args)
                continue
            if args.fresh:
                for f in glob.glob(os.path.join(wd, "*")) + glob.glob(os.path.join(wd, ".done")):
                    if os.path.isfile(f):
                        os.remove(f)
            cmd = [sys.executable, "-u", os.path.abspath(__file__), "--stage", "country", "--country", c,
                   "--workers", str(args.workers), "--chunk", str(args.chunk)]
            for flag, val in (("--k", args.k), ("--model", args.model), ("--thresholds", args.thresholds),
                              ("--output-dir", args.output_dir), ("--tau", args.tau), ("--delta", args.delta),
                              ("--tau1", args.tau1), ("--fr-gates", args.fr_gates)):
                if val is not None:
                    cmd += [flag, str(val)]
            if args.smoke:
                cmd.append("--smoke")
            log(f"[{c}] launching worker subprocess")
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                raise SystemExit(f"country worker {c} failed with exit code {rc}; rerun to resume")
    assemble(P, countries)
    rc = run_validator(P)
    try:
        france_hint(P, thr, args)
    except Exception as exc:
        log(f"France hint skipped: {exc!r}")
    log(f"pipeline finished in {fmt_secs(time.time() - t0)} (validator exit code {rc})")


if __name__ == "__main__":
    main()
