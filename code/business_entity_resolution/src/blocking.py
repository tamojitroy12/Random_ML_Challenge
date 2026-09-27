"""
Candidate generation: compact CSR inverted index over hashed multi-channel keys.

Keys are (channel, string) pairs hashed to int64 as (channel << 36) | (polars_hash(string) & 2^36-1).
The index is built from packed (key << 23 | record) int64 arrays: one in-place sort, no Python
objects per posting. Posting lists longer than a per-channel df cap are dropped at build time.

Query: dense float32 accumulator over the universe, IDF-style per-channel weights, top-K per
source (S2 / S3). The same class indexes S1 records for reverse blocking (source = 0).
"""

import numpy as np
import polars as pl

from .normalizer import GENERIC_NAME_WORDS, STREET_GENERIC

# Channels
CH_T, CH_P, CH_K, CH_G, CH_A, CH_S, CH_H, CH_N, CH_B, CH_X, CH_Y, CH_R, CH_C, CH_2T, CH_G4 = range(15)
CHANNEL_NAMES = ["token", "prefix4", "skel5", "gram3", "acronym", "short", "hn_street", "num_word",
                 "hn_tok0", "name_x_street", "street_x_city", "raw_sig", "clean_sig", "tok_pair", "gram4"]
HASH_BITS = 36
MASK36 = (1 << HASH_BITS) - 1
REC_BITS = 23
MAX_RECS = 1 << REC_BITS

# weight mode: 0 -> base/(df+10); 1 -> constant base; 2 -> base*min(1, 5/df)
CH_BASE = np.array([120, 35, 50, 3, 120, 120, 150, 100, 150, 120, 60, 80, 70, 160, 15, 0], np.float64)
CH_MODE = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 1, 1], np.int64)
CH_BUILD_CAP = np.array([20000, 2500, 3000, 2000, 500, 500, 1000, 1000, 1000, 1000, 1000, 200, 200,
                         3000, 3000, 0], np.int64)
CH_QUERY_CAP = np.array([3500, 2500, 3000, 2000, 500, 500, 1000, 1000, 1000, 1000, 1000, 200, 200,
                         2500, 2500, 0], np.int64)
T_FALLBACK_CAP = 20000


def _rare(tokens, generic, k):
    cand = sorted({t for t in tokens if len(t) >= 3 and not t.isdigit() and t not in generic},
                  key=lambda t: (-len(t), t))
    return cand[:k]


def make_keys(core, concat, skel, tok0, acr, short, street, loc, city, hn_d, digits):
    """String blocking keys (channels T..G4). Signature channels are added by the caller."""
    keys = []
    ctoks = [t for t in core.split() if len(t) >= 2]
    seen = set()
    for t in ctoks:
        if t not in seen:
            seen.add(t)
            keys.append((CH_T, t))
    if len(concat) >= 4:
        keys.append((CH_P, concat[:4]))
    if len(skel) >= 3:
        keys.append((CH_K, skel[:5]))
    if len(tok0) >= 3:
        g = "^" + tok0 + "$"
        grams = []
        for i in range(len(g) - 2):
            gg = g[i:i + 3]
            if gg not in grams:
                grams.append(gg)
        for gg in grams[:12]:
            keys.append((CH_G, gg))
    if len(acr) >= 2:
        keys.append((CH_A, acr))
    if short:
        keys.append((CH_S, short))
    stoks = street.split()
    st0 = next((t for t in stoks if t not in STREET_GENERIC and not t.isdigit()), stoks[0] if stoks else "")
    if hn_d >= 0 and st0:
        keys.append((CH_H, f"{hn_d}|{st0}"))
    ltoks = loc.split()
    words = _rare(stoks + ltoks, STREET_GENERIC, 3)
    for d in digits.split()[:2]:
        for w in words:
            keys.append((CH_N, f"{d}|{w}"))
    if hn_d >= 0 and tok0:
        keys.append((CH_B, f"{hn_d}|{tok0}"))
    rn = _rare(ctoks, GENERIC_NAME_WORDS, 2) or ([max(ctoks, key=len)] if ctoks else [])
    rs = _rare(stoks + ltoks, STREET_GENERIC, 2)
    for a in rn:
        for b in rs:
            keys.append((CH_X, f"{a}|{b}"))
    rs_street = _rare(stoks, STREET_GENERIC, 1)
    cty = city or (ltoks[-1] if ltoks else "")
    if rs_street and cty:
        keys.append((CH_Y, f"{rs_street[0]}|{cty}"))

    # Channel 13 (CH_2T): Distinctive Name Token Pairs (High IDF for empty/sparse addresses)
    rn_pair = _rare(ctoks, GENERIC_NAME_WORDS, 4)
    if len(rn_pair) >= 2:
        for i in range(len(rn_pair)):
            for j in range(i + 1, len(rn_pair)):
                w1, w2 = (rn_pair[i], rn_pair[j]) if rn_pair[i] < rn_pair[j] else (rn_pair[j], rn_pair[i])
                keys.append((CH_2T, f"{w1}|{w2}"))

    # Channel 14 (CH_G4): Name character 4-gram shingles for single or short names
    if len(concat) >= 5:
        g4s = []
        for i in range(len(concat) - 3):
            g4 = concat[i:i + 4]
            if g4 not in g4s:
                g4s.append(g4)
        for g4 in g4s[:8]:
            keys.append((CH_G4, g4))

    return keys


def hash_strings(strs):
    """Deterministic (cross-process) uint64 hashes; 0 is reserved for the empty string."""
    if len(strs) == 0:
        return np.zeros(0, np.uint64)
    h = pl.Series(strs, dtype=pl.String).hash(seed=0).to_numpy().astype(np.uint64, copy=True)
    empty = np.fromiter((len(s) == 0 for s in strs), dtype=bool, count=len(strs))
    h[(h == 0) & ~empty] = np.uint64(1)
    h[empty] = np.uint64(0)
    return h


def key_ints(channels, strs):
    """(channel list, string list) -> int64 keys."""
    if len(strs) == 0:
        return np.zeros(0, np.int64)
    h = pl.Series(strs, dtype=pl.String).hash(seed=0).to_numpy().astype(np.uint64, copy=False)
    low = (h & np.uint64(MASK36)).astype(np.int64)
    return low | (np.asarray(channels, dtype=np.int64) << HASH_BITS)


def sig_key_ints(channel, sig_hashes):
    """Signature hash (uint64, 0 = absent) -> int64 key for channel R/C."""
    low = (sig_hashes & np.uint64(MASK36)).astype(np.int64)
    return low | (np.int64(channel) << HASH_BITS)


class HashIndex:
    def __init__(self, ukeys, indptr, postings, source, n_records):
        self.ukeys = ukeys
        self.indptr = indptr
        self.postings = postings
        self.source = source
        self.n = n_records
        self.nu = len(ukeys)
        self.acc = np.zeros(n_records, np.float32)

    @classmethod
    def build(cls, key_batches, n_records, source, exclude=None, consume=False):
        """key_batches: list of (keys int64, first_record, per-record key counts).
        exclude: bool mask of records to skip. consume=True pops batches as they are packed
        (frees memory progressively; the caller's list ends up empty)."""
        assert n_records < MAX_RECS, "universe too large for 23-bit record ids"
        parts = []
        batches = key_batches if consume else list(key_batches)
        while batches:
            keys, off, kc = batches.pop(0)
            recs = np.repeat(np.arange(off, off + kc.size, dtype=np.int64), kc)
            if exclude is not None and recs.size:
                m = ~exclude[recs]
                keys, recs = keys[m], recs[m]
            parts.append((keys.astype(np.int64) << REC_BITS) | recs)
            del keys, recs
        packed = np.concatenate(parts) if parts else np.zeros(0, np.int64)
        del parts
        packed.sort()
        if packed.size:
            keep = np.empty(packed.size, bool)
            keep[0] = True
            np.not_equal(packed[1:], packed[:-1], out=keep[1:])
            if not keep.all():
                packed = packed[keep]
            del keep
        if packed.size == 0:
            return cls(np.zeros(0, np.int64), np.zeros(1, np.int64), np.zeros(0, np.int32),
                       source, n_records)
        keys = packed >> REC_BITS
        recs = (packed & (MAX_RECS - 1)).astype(np.int32)
        del packed
        brk = np.empty(keys.size, bool)
        brk[0] = True
        np.not_equal(keys[1:], keys[:-1], out=brk[1:])
        starts = np.flatnonzero(brk)
        del brk
        ukeys = keys[starts]
        del keys
        lengths = np.diff(np.append(starts, recs.size)).astype(np.int64)
        del starts
        ch = ukeys >> HASH_BITS
        keep_run = lengths <= CH_BUILD_CAP[ch]
        if not keep_run.all():
            recs = recs[np.repeat(keep_run, lengths)]
            ukeys = ukeys[keep_run]
            lengths = lengths[keep_run]
        indptr = np.zeros(ukeys.size + 1, np.int64)
        np.cumsum(lengths, out=indptr[1:])
        return cls(ukeys, indptr, recs, source, n_records)

    def stats(self):
        return {"keys": int(self.nu), "postings": int(self.postings.size),
                "mb": (self.ukeys.nbytes + self.indptr.nbytes + self.postings.nbytes) / 1e6}

    def _weights(self, ch, df):
        base = CH_BASE[ch]
        mode = CH_MODE[ch]
        dff = df.astype(np.float64)
        w = np.where(mode == 0, base / (dff + 10.0),
                     np.where(mode == 1, base, base * np.minimum(1.0, 5.0 / np.maximum(dff, 1.0))))
        w[df > CH_QUERY_CAP[ch]] = 0.0
        tmask = ch == CH_T
        if tmask.any() and not (w[tmask] > 0).any():
            ti = np.flatnonzero(tmask)
            j = ti[np.argmin(df[ti])]
            if df[j] <= T_FALLBACK_CAP:
                w[j] = 100.0 / (df[j] + 10.0)
        return w

    def query(self, qk, k):
        """One query's index-form keys -> (targets int32, scores float32, ranks int16), top-k per source."""
        empty = (np.zeros(0, np.int32), np.zeros(0, np.float32), np.zeros(0, np.int16))
        if qk.size == 0 or self.nu == 0:
            return empty
        ch = qk >> HASH_BITS
        is_a, is_s = ch == CH_A, ch == CH_S
        if is_a.any() or is_s.any():  # acronym <-> short-name cross lookup
            qk = qk.copy()
            low = qk & MASK36
            qk[is_a] = low[is_a] | (CH_S << HASH_BITS)
            qk[is_s] = low[is_s] | (CH_A << HASH_BITS)
        pos = np.searchsorted(self.ukeys, qk)
        ok = pos < self.nu
        pos = np.where(ok, pos, 0)
        ok &= self.ukeys[pos] == qk
        if not ok.any():
            return empty
        pos = pos[ok]
        chq = qk[ok] >> HASH_BITS
        s = self.indptr[pos]
        e = self.indptr[pos + 1]
        w = self._weights(chq, e - s)
        sel = np.flatnonzero(w > 0)
        if sel.size == 0:
            return empty
        acc, post = self.acc, self.postings
        parts = []
        for i in sel:
            seg = post[s[i]:e[i]]
            acc[seg] += w[i]
            parts.append(seg)
        touched = np.concatenate(parts)
        ktot = 2 * k
        if touched.size <= 4096:
            u = np.unique(touched)
        else:
            m = 16 * ktot
            if m < touched.size:
                part = np.argpartition(-acc[touched], m)[:m]
                u = np.unique(touched[part])
                if u.size < ktot:
                    u = np.unique(touched)
            else:
                u = np.unique(touched)
        sc = acc[u]
        acc[touched] = 0.0
        src = self.source[u]
        outs_t, outs_s, outs_r = [], [], []
        for sv in np.unique(src):
            msk = src == sv
            us, ss = u[msk], sc[msk]
            if us.size > k:
                top = np.argpartition(-ss, k - 1)[:k]
                us, ss = us[top], ss[top]
            order = np.lexsort((us, -ss))
            outs_t.append(us[order].astype(np.int32))
            outs_s.append(ss[order].astype(np.float32))
            outs_r.append(np.arange(order.size, dtype=np.int16))
        if len(outs_t) == 1:
            return outs_t[0], outs_s[0], outs_r[0]
        return np.concatenate(outs_t), np.concatenate(outs_s), np.concatenate(outs_r)

    def query_many(self, key_ptr, key_arr, rows, k):
        """rows: iterable of query row ids (into key_ptr). -> (qrow, t, score, rank) arrays."""
        qs, ts, ss, rs = [], [], [], []
        for r in rows:
            t, s, rk = self.query(key_arr[key_ptr[r]:key_ptr[r + 1]], k)
            if t.size:
                qs.append(np.full(t.size, r, np.int32))
                ts.append(t)
                ss.append(s)
                rs.append(rk)
        if not qs:
            return (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32),
                    np.zeros(0, np.int16))
        return np.concatenate(qs), np.concatenate(ts), np.concatenate(ss), np.concatenate(rs)
