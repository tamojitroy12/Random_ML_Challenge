"""
Native-script -> Latin token lexicon learned from train ground truth.

India S2/S3 names in Indic scripts are word-for-word transliterations of the Latin S1 name
(same token count). Aligning GT pairs positionally and keeping majority mappings gives a
dictionary that fixes names the phonetic transliterator gets wrong. VAL_S1 and TRAIN_S1 are
excluded from the build so their features are not in-sample.
"""

import json
import os
import re
from collections import Counter, defaultdict

import polars as pl

from .normalizer import fold_ascii, has_indic, lexicon_key
from .records import read_tsv
from .runtime import log

_INDIC_CLASS = "[ऀ-ൿ]"
_RE_LAT = re.compile(r"[^a-z0-9]+")


def _latin_tok(t):
    return _RE_LAT.sub("", fold_ascii(t).lower())


def build_lexicon(train_dir, exclude_ids, out_path, min_count=3, min_share=0.6, s1_limit=None):
    gt = read_tsv(os.path.join(train_dir, "train_ground_truth.tsv"),
                  columns=("source1_entity_id", "matched_entity_ids"))
    s1 = read_tsv(os.path.join(train_dir, "train_source1.tsv"),
                  columns=("entity_id", "business_name", "country"), country="India")
    s1 = s1.filter(~pl.col("entity_id").is_in(list(exclude_ids)))
    if s1_limit:
        s1 = s1.head(s1_limit)
    s1 = s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("s1_name"))
    links = (gt.filter(pl.col("matched_entity_ids").str.len_bytes() > 0)
             .with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids")
             .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("tid"))
             .join(s1, on="s1", how="inner"))
    tparts = []
    for src in ("train_source2.tsv", "train_source3.tsv"):
        t = read_tsv(os.path.join(train_dir, src), columns=("entity_id", "business_name", "country"),
                     country="India")
        t = t.filter(pl.col("business_name").str.contains(_INDIC_CLASS))
        tparts.append(t.select(pl.col("entity_id").alias("tid"), pl.col("business_name").alias("t_name")))
    tn = pl.concat(tparts)
    pairs = links.join(tn, on="tid", how="inner").select("s1_name", "t_name")
    log(f"lexicon: {pairs.height:,} native-script GT pairs")

    counts = defaultdict(Counter)
    for s1_name, t_name in pairs.iter_rows():
        lt = [_latin_tok(x) for x in s1_name.split()]
        nt = [lexicon_key(x) for x in t_name.split()]
        lt = [x for x in lt if x]
        nt = [x for x in nt if x]
        if not lt or len(lt) != len(nt):
            continue
        for a, b in zip(nt, lt):
            if a and not a.isascii() and has_indic(a):
                counts[a][b] += 1
    lex = {}
    for tok, c in counts.items():
        best, n_best = c.most_common(1)[0]
        tot = sum(c.values())
        if n_best >= min_count and n_best / tot >= min_share:
            lex[tok] = best
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(lex, f, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, out_path)
    log(f"lexicon: {len(lex):,} native tokens mapped (from {len(counts):,} seen) -> {out_path}")
    return lex


def load_lexicon(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coverage(names, lex):
    """Share of native-script tokens in `names` that the lexicon maps."""
    tot = hit = 0
    for nm in names:
        if not nm or nm.isascii():
            continue
        for t in nm.split():
            k = lexicon_key(t)
            if k and has_indic(k):
                tot += 1
                hit += k in lex
    return hit / tot if tot else float("nan"), tot
