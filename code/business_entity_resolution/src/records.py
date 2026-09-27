"""
RecordStore: parse raw (id, name, address) rows once into compact columns.

* Parsing runs in a spawn multiprocessing pool (Windows-safe) over 50k-row batches; workers
  also emit hashed blocking keys, so the main process never holds per-key Python objects.
* String columns live in a polars DataFrame (Arrow memory, no per-string Python overhead);
  flags/numbers/hashes live in numpy arrays.
* Hashes use polars' seeded hash, which is deterministic across processes of one version.
"""

import multiprocessing as mp
import os
import sys

import numpy as np
import polars as pl

from . import normalizer
from .blocking import (CH_C, CH_R, hash_strings, key_ints, make_keys, sig_key_ints)
from .runtime import log

STR_FIELDS = ["name", "core", "concat", "fconcat", "fconcat_del", "skel", "acr", "short", "tok0",
              "tokN", "addr", "street", "stype", "loc", "city", "state", "hnum", "hn_digits",
              "hn_sfx", "cnums", "digits", "unit"]
INT_FIELDS = {"legal": np.int32, "is_domain": np.int8, "is_garbage": np.int8, "has_alias": np.int8,
              "non_latin": np.int8, "ctok": np.int8, "hn_d": np.int64, "hn_nparts": np.int8,
              "hn_hi": np.int64, "hn_clean": np.int8, "addr_empty": np.int8}
HASHED_FIELDS = ["core", "hnum", "hn_digits", "hn_sfx", "street", "state", "city", "stype", "unit"]
_NAME_IDX = {f: i for i, f in enumerate(normalizer.NAME_FIELDS)}
_ADDR_IDX = {f: i for i, f in enumerate(normalizer.ADDR_FIELDS)}


def _init_worker(lexicon, src_path):
    if src_path and src_path not in sys.path:
        sys.path.insert(0, src_path)
    normalizer.set_lexicon(lexicon)


def _parse_batch(task):
    names, addrs, country, want_keys, strict = task
    normalizer.STRICT = strict
    e0 = dict(normalizer.ERRORS)
    n0 = len(normalizer.ERROR_EXAMPLES)
    cols = {f: [] for f in STR_FIELDS}
    ints = {f: [] for f in INT_FIELDS}
    raw_sig, clean_sig, akey = [], [], []
    kch, kstr, kcnt = [], [], []
    ni, ai = _NAME_IDX, _ADDR_IDX
    for nm, ad in zip(names, addrs):
        nf = normalizer.parse_name_safe(nm)
        af = normalizer.parse_address_safe(ad, country)
        for f in STR_FIELDS:
            cols[f].append(nf[ni[f]] if f in ni else af[ai[f]])
        for f in INT_FIELDS:
            ints[f].append(nf[ni[f]] if f in ni else af[ai[f]])
        rs, cs = af[ai["raw_sig"]], af[ai["clean_sig"]]
        raw_sig.append(rs)
        clean_sig.append(cs)
        street = af[ai["street"]]
        hn_d = af[ai["hn_d"]]
        st0 = street.split()[0] if street else ""
        akey.append(f"{hn_d}|{st0}" if (hn_d >= 0 and st0) else "")
        if want_keys:
            ks = make_keys(nf[ni["core"]], nf[ni["concat"]], nf[ni["skel"]], nf[ni["tok0"]],
                           nf[ni["acr"]], nf[ni["short"]], street, af[ai["loc"]], af[ai["city"]],
                           hn_d, af[ai["digits"]])
            if rs:
                ks.append((CH_R, rs))
            if cs:
                ks.append((CH_C, cs))
            kcnt.append(len(ks))
            for c, s in ks:
                kch.append(c)
                kstr.append(s)
    out_int = {f: np.asarray(v, dtype=INT_FIELDS[f]) for f, v in ints.items()}
    sig = {"raw_sig_h": hash_strings(raw_sig), "clean_sig_h": hash_strings(clean_sig),
           "akey_h": hash_strings(akey)}
    keys = key_ints(kch, kstr) if want_keys else None
    kc = np.asarray(kcnt, dtype=np.int32) if want_keys else None
    errs = {k: normalizer.ERRORS[k] - e0.get(k, 0) for k in normalizer.ERRORS}
    examples = normalizer.ERROR_EXAMPLES[n0:n0 + 5]
    return cols, out_int, sig, keys, kc, errs, examples


def _bounded_imap(pool, fn, task_iter, window):
    """Ordered imap with at most `window` tasks in flight (bounds memory for big inputs)."""
    from collections import deque
    pending = deque()
    for task in task_iter:
        pending.append(pool.apply_async(fn, (task,)))
        if len(pending) >= window:
            yield pending.popleft().get()
    while pending:
        yield pending.popleft().get()


def count_other(h):
    """For each record: number of OTHER records with the same (non-zero) hash."""
    if h.size == 0:
        return np.zeros(0, np.int32)
    _, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
    c = (cnt[inv] - 1).astype(np.int32)
    c[h == 0] = 0
    return c


def count_in(h_query, h_ref):
    """For each query hash: number of reference records with the same (non-zero) hash."""
    ref = h_ref[h_ref != 0]
    if ref.size == 0 or h_query.size == 0:
        return np.zeros(h_query.size, np.int32)
    u, cnt = np.unique(ref, return_counts=True)
    pos = np.searchsorted(u, h_query)
    pos = np.minimum(pos, u.size - 1)
    hit = (u[pos] == h_query) & (h_query != 0)
    return np.where(hit, cnt[pos], 0).astype(np.int32)


def add_counts(s1, tg):
    """Name/address duplicate counts. s1 must hold ALL S1 of the country, tg all its targets."""
    s1.num["s1_name_other"] = count_other(s1.num["core_h"])
    s1.num["s1_addr_other"] = count_other(s1.num["clean_sig_h"])
    s1.num["core_in_targets"] = count_in(s1.num["core_h"], tg.num["core_h"])
    tg.num["t_name_other"] = count_other(tg.num["core_h"])
    tg.num["t_akey_other"] = count_other(tg.num["akey_h"])


class RecordStore:
    """Parsed records of one universe (one country, one role: S1 queries or S2+S3 targets)."""

    def __init__(self):
        self.n = 0
        self.ids = None          # polars Series (String)
        self.df = None           # polars DataFrame of STR_FIELDS
        self.num = {}            # numpy arrays (INT_FIELDS, hashes, counts, flags)
        self.key_batches = []    # list of (keys int64, first record, per-record key counts)
        self.key_ptr = None      # CSR of this store's own keys (kept only if keep_keys)
        self.key_arr = None
        self.errors = {"name": 0, "addr": 0}
        self.error_examples = []

    # ----------------------------------------------------------------------------------
    @classmethod
    def build(cls, frame, country, source, lexicon=None, workers=6, batch=50000, want_keys=True,
              keep_keys=False, strict=False, label=""):
        """frame: polars DataFrame with entity_id, business_name, business_address (strings)."""
        st = cls()
        n = frame.height
        st.n = n
        st.ids = frame.get_column("entity_id")
        src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def tasks():
            for i in range(0, n, batch):
                sl = frame.slice(i, batch)
                yield (sl.get_column("business_name").to_list(),
                       sl.get_column("business_address").to_list(), country, want_keys, strict)

        frames, ints, sigs = [], {f: [] for f in INT_FIELDS}, {"raw_sig_h": [], "clean_sig_h": [],
                                                               "akey_h": []}
        kparts, kcounts = [], []
        pool = None
        if workers and workers > 0 and n > batch:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(workers, initializer=_init_worker, initargs=(lexicon or {}, src_path))
            it = _bounded_imap(pool, _parse_batch, tasks(), 2 * workers)
        else:
            normalizer.set_lexicon(lexicon or {})
            it = map(_parse_batch, tasks())
        off = 0
        ok = False
        try:
            for cols, out_int, sig, keys, kc, errs, examples in it:
                nb = len(cols["name"])
                frames.append(pl.DataFrame(cols, schema={f: pl.String for f in STR_FIELDS}))
                for f in INT_FIELDS:
                    ints[f].append(out_int[f])
                for f in sigs:
                    sigs[f].append(sig[f])
                if want_keys:
                    kparts.append((keys, off, kc))  # records off..off+nb-1, kc keys each
                    kcounts.append(kc)
                for k, v in errs.items():
                    st.errors[k] = st.errors.get(k, 0) + v
                if len(st.error_examples) < 5:
                    st.error_examples.extend(examples[:5 - len(st.error_examples)])
                off += nb
            ok = True
        finally:
            if pool is not None:
                if ok:
                    pool.close()
                else:
                    pool.terminate()
                pool.join()
        assert off == n, f"parsed {off} of {n} rows"
        st.df = pl.concat(frames, rechunk=True) if frames else pl.DataFrame(
            {f: [] for f in STR_FIELDS}, schema={f: pl.String for f in STR_FIELDS})
        del frames
        for f in INT_FIELDS:
            st.num[f] = np.concatenate(ints[f]) if ints[f] else np.zeros(0, INT_FIELDS[f])
        for f in sigs:
            st.num[f] = np.concatenate(sigs[f]) if sigs[f] else np.zeros(0, np.uint64)
        for f in HASHED_FIELDS:
            st.num[f + "_h"] = st._hash_col(f)
        st.num["source"] = np.full(n, source, np.int8) if np.isscalar(source) else np.asarray(source, np.int8)
        if want_keys:
            st.key_batches = kparts
            if keep_keys:
                cnt = np.concatenate(kcounts) if kcounts else np.zeros(0, np.int32)
                st.key_ptr = np.zeros(n + 1, np.int64)
                np.cumsum(cnt, out=st.key_ptr[1:])
                st.key_arr = np.concatenate([k for k, _, _ in kparts]) if kparts else np.zeros(0, np.int64)
        if st.errors["name"] or st.errors["addr"]:
            log(f"  {label}: parse errors name={st.errors['name']} addr={st.errors['addr']} "
                f"examples={st.error_examples[:3]}")
        return st

    def _hash_col(self, f):
        s = self.df.get_column(f)
        h = s.hash(seed=0).to_numpy().astype(np.uint64, copy=True)
        empty = (s.str.len_bytes() == 0).to_numpy()
        h[(h == 0) & ~empty] = np.uint64(1)
        h[empty] = np.uint64(0)
        return h

    def drop_key_batches(self):
        self.key_batches = []

    # ----------------------------------------------------------------------------------
    def take_str(self, fields, idx):
        idx = np.asarray(idx, dtype=np.int64)
        return {f: self.df.get_column(f).gather(idx).to_list() for f in fields}

    def subset(self, idx):
        """Compact copy of selected records (strings + numeric arrays, no keys)."""
        idx = np.asarray(idx, dtype=np.int64)
        sub = RecordStore()
        sub.n = idx.size
        sub.ids = self.ids.gather(idx)
        sub.df = self.df[idx]
        sub.num = {k: v[idx] for k, v in self.num.items()}
        return sub

    def keys_for(self, idx):
        """Index-form key CSR for selected records -> (key_ptr, key_arr) aligned with idx order."""
        idx = np.asarray(idx, dtype=np.int64)
        if self.key_ptr is not None:
            ptr = self.key_ptr
            cnt = (ptr[idx + 1] - ptr[idx]).astype(np.int64)
            out_ptr = np.zeros(idx.size + 1, np.int64)
            np.cumsum(cnt, out=out_ptr[1:])
            if out_ptr[-1] == 0:
                return out_ptr, np.zeros(0, np.int64)
            sel = np.repeat(ptr[idx], cnt) + (np.arange(out_ptr[-1]) - np.repeat(out_ptr[:-1], cnt))
            return out_ptr, self.key_arr[sel]
        f = self.take_str(["core", "concat", "skel", "tok0", "acr", "short", "street", "loc", "city",
                           "digits"], idx)
        hn_d = self.num["hn_d"][idx]
        rsig = self.num["raw_sig_h"][idx]
        csig = self.num["clean_sig_h"][idx]
        kch, kstr, cnt = [], [], np.zeros(idx.size, np.int64)
        for i in range(idx.size):
            ks = make_keys(f["core"][i], f["concat"][i], f["skel"][i], f["tok0"][i], f["acr"][i],
                           f["short"][i], f["street"][i], f["loc"][i], f["city"][i], int(hn_d[i]),
                           f["digits"][i])
            cnt[i] = len(ks)
            for c, s in ks:
                kch.append(c)
                kstr.append(s)
        str_keys = key_ints(kch, kstr)
        has_r = rsig != 0
        has_c = csig != 0
        tot = cnt + has_r + has_c
        out_ptr = np.zeros(idx.size + 1, np.int64)
        np.cumsum(tot, out=out_ptr[1:])
        out = np.zeros(out_ptr[-1], np.int64)
        # string keys first, then R, then C, per record
        str_start = np.zeros(idx.size + 1, np.int64)
        np.cumsum(cnt, out=str_start[1:])
        pos = np.repeat(out_ptr[:-1], cnt) + (np.arange(str_start[-1]) - np.repeat(str_start[:-1], cnt))
        out[pos] = str_keys
        r_pos = out_ptr[:-1] + cnt
        out[r_pos[has_r]] = sig_key_ints(CH_R, rsig[has_r])
        c_pos = out_ptr[:-1] + cnt + has_r
        out[c_pos[has_c]] = sig_key_ints(CH_C, csig[has_c])
        return out_ptr, out

    # ----------------------------------------------------------------------------------
    def token_df(self):
        """Document frequency of core tokens (dict) for IDF."""
        vc = (self.df.select(pl.col("core").str.split(" ").alias("t")).explode("t")
              .filter(pl.col("t").str.len_bytes() > 0).group_by("t").len())
        return dict(zip(vc.get_column("t").to_list(), vc.get_column("len").to_list()))


def read_tsv(path, columns=("entity_id", "business_name", "business_address", "country"),
             country=None):
    """Robust TSV read: RFC-4180 quoting, all strings, nulls -> ''."""
    lf = pl.scan_csv(path, separator="\t", quote_char='"', infer_schema=False)
    lf = lf.select(list(columns))
    if country is not None:
        lf = lf.filter(pl.col("country") == country)
    df = lf.collect()
    return df.with_columns([pl.col(c).fill_null("") for c in df.columns])
