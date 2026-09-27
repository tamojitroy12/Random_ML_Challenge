"""
Multilingual name/address parsing for business entity resolution.

parse_name(raw)              -> tuple in NAME_FIELDS order
parse_address(raw, country)  -> tuple in ADDR_FIELDS order

Design notes (from data mining of the train/test TSVs):
  * addresses are parsed per comma component, never by position (order is shuffled);
  * a state/region is only recognised when a WHOLE component equals it;
  * US has no ZIP codes and India no PIN codes: 5-6 digit tokens are house numbers;
  * France S1 uses regions while targets use departments/cities: everything maps to regions;
  * house numbers keep compounds (104/1/1), suffixes (bis/ter/letters) and never come from
    ordinals (12nd), glued tokens (cour2) or numbers after a French street type (Rue du 8 Mai);
  * names: alias markers (X dba Y -> Y), dotted initialisms, legal forms (one global class
    table), honorifics, leetspeak, domains, garbage syllable names, Indic transliteration.
"""

import re
import unicodedata

from rapidfuzz import fuzz, process

PARSER_VERSION = "2.0.3"

# ----------------------------------------------------------------------------------------
# Lexicon (native-script token -> Latin token), learned from train GT by lexicon.py
# ----------------------------------------------------------------------------------------
_LEXICON = {}


def set_lexicon(lex):
    global _LEXICON
    _LEXICON = dict(lex or {})


_ZWNJ, _ZWJ = chr(0x200C), chr(0x200D)


def strip_zw(s):
    return s.replace(_ZWNJ, "").replace(_ZWJ, "")


def lexicon_key(tok):
    """Normalisation used for lexicon keys at build AND lookup time."""
    tok = unicodedata.normalize("NFC", strip_zw(tok))
    return tok.strip(" .,;:()[]{}|'\"-_/\\")


# ----------------------------------------------------------------------------------------
# Indic transliteration (Brahmi blocks share a relative code-point layout)
# ----------------------------------------------------------------------------------------
_BASE = {
    0x01: "n", 0x02: "n", 0x03: "h",
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
    0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
    0x1D: "jh", 0x1E: "ny", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
    0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x3C: "", 0x3D: "",
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x4D: "", 0x50: "om", 0x51: "", 0x52: "", 0x53: "", 0x54: "", 0x55: "", 0x56: "", 0x57: "au",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
    0x60: "ri", 0x61: "li", 0x62: "li", 0x63: "li", 0x64: " ", 0x65: " ",
}
for _d in range(10):
    _BASE[0x66 + _d] = str(_d)

_OVERRIDES = {
    "beng": {0x4E: "t", 0x70: "r", 0x71: "w"},
    "guru": {0x70: "n", 0x71: "", 0x72: "", 0x73: "", 0x74: "om"},
    "orya": {0x71: "w"},
    "taml": {0x03: ""},
    "telu": {0x58: "ts", 0x59: "dz", 0x56: "ai"},
    "knda": {0x5E: "l"},
    "mlym": {0x4E: "", 0x54: "m", 0x55: "y", 0x56: "l", 0x7A: "n", 0x7B: "n", 0x7C: "r",
             0x7D: "l", 0x7E: "l", 0x7F: "k"},
}
_BLOCKS = {}
for _key, (_script, _start) in {
    0x12: ("deva", 0x0900), 0x13: ("beng", 0x0980), 0x14: ("guru", 0x0A00), 0x15: ("gujr", 0x0A80),
    0x16: ("orya", 0x0B00), 0x17: ("taml", 0x0B80), 0x18: ("telu", 0x0C00), 0x19: ("knda", 0x0C80),
    0x1A: ("mlym", 0x0D00),
}.items():
    _tbl = dict(_BASE)
    _tbl.update(_OVERRIDES.get(_script, {}))
    _anus = {0x01, 0x02} | ({0x70} if _script == "guru" else set())
    _BLOCKS[_key] = (_start, _tbl, _anus)

_CONS = set(range(0x15, 0x3A)) | set(range(0x58, 0x60))
_MATRA = set(range(0x3E, 0x4D)) | {0x55, 0x56, 0x57, 0x62, 0x63}
_VIRAMA, _NUKTA, _VISARGA = 0x4D, 0x3C, 0x03
_LABIAL = {0x2A, 0x2B, 0x2C, 0x2D, 0x2E}
_NUKTA_MAP = {"j": "z", "ph": "f", "d": "r", "dh": "rh", "k": "q", "g": "g", "kh": "kh"}
_PRE_REPL = [("ന്റ", "nt"), ("റ്റ", "tt"), ("ஃப", "f"), ("ஃஜ", "z")]


def _is_indic(ch):
    return (ord(ch) >> 7) in _BLOCKS


def has_indic(s):
    return any((ord(c) >> 7) in _BLOCKS for c in s) if s and not s.isascii() else False


def _off_in(text, i, start):
    """Offset of text[i] inside the block starting at `start`, else None."""
    if i < 0 or i >= len(text):
        return None
    cp = ord(text[i])
    if start <= cp < start + 0x80:
        return cp - start
    return None


def transliterate(text):
    """Indic scripts -> phonetic Latin (inherent vowel + schwa deletion). Other chars pass."""
    if not text or text.isascii():
        return text
    for a, b in _PRE_REPL:
        if a in text:
            text = text.replace(a, b)
    out = []
    n = len(text)
    for i in range(n):
        ch = text[i]
        info = _BLOCKS.get(ord(ch) >> 7)
        if info is None:
            out.append(ch)
            continue
        start, tbl, anus = info
        off = ord(ch) - start
        if off in _CONS:
            out.append(tbl.get(off, ""))
            nxt = _off_in(text, i + 1, start)
            if nxt is None:
                # word end: keep 'a' only after a conjunct (e.g. rashtra, mishra)
                if _off_in(text, i - 1, start) == _VIRAMA:
                    out.append("a")
            elif nxt in _MATRA or nxt == _VIRAMA or nxt == _NUKTA:
                pass
            elif nxt in _CONS or nxt in anus or nxt == _VISARGA:
                out.append("a")
        elif off == _NUKTA:
            if out:
                out[-1] = _NUKTA_MAP.get(out[-1], out[-1])
            nxt = _off_in(text, i + 1, start)
            if nxt is not None and (nxt in _CONS or nxt in anus):
                out.append("a")
        elif off in anus:
            nxt = _off_in(text, i + 1, start)
            out.append("m" if (nxt is None or nxt in _LABIAL) else "n")
        else:
            out.append(tbl.get(off, ""))
    return "".join(out)


def apply_lexicon(text):
    if not _LEXICON or not text or text.isascii():
        return text
    toks = text.split()
    changed = False
    for i, t in enumerate(toks):
        if t.isascii():
            continue
        v = _LEXICON.get(lexicon_key(t))
        if v:
            toks[i] = v
            changed = True
    return " ".join(toks) if changed else text


def fold_ascii(s):
    if s.isascii():
        return s
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def delete_non_ascii(s):
    return "".join(c for c in s if ord(c) < 128)


# ----------------------------------------------------------------------------------------
# Phonetic skeleton and garbage-name detection
# ----------------------------------------------------------------------------------------
_SK_DIGRAPHS = [("tion", "shn"), ("sion", "shn"), ("ture", "chr"), ("tech", "tek"), ("ck", "k"),
                ("qu", "k"), ("ph", "f"), ("sh", "s"), ("ch", "9"), ("th", "t"), ("dh", "d"),
                ("kh", "k"), ("gh", "g"), ("bh", "b"), ("jh", "j"), ("x", "ks")]
_SK_TRANS = str.maketrans({"c": "k", "q": "k", "z": "j", "w": "v", "b": "p", "d": "t", "g": "k",
                           "9": "c"})
_SK_DROP = set("aeiouyh")


def skeleton(s):
    if not s:
        return ""
    s = s.lower()
    for a, b in _SK_DIGRAPHS:
        if a in s:
            s = s.replace(a, b)
    s = s.translate(_SK_TRANS)
    out = []
    prev = ""
    for ch in s:
        if ch in _SK_DROP or not ch.isalnum():
            continue
        if ch != prev:
            out.append(ch)
        prev = ch
    return "".join(out)


_GARB_SYL = set("""
ecto wex orbi avi vio syn veo cira quo xylo sol nex lum flux delta mira gild arc lyra vantage umbra
calo pyra tavo jax yuma zeta nyla dova vera drex zeph brix kor korz eta riza visol belo halo kelo
faye evo aria onyx io ium ari dyne kel novi iri lumx avid synio rizano kelo veloz vio evoi solx
beli irix jaxx oni nyx zeta aria delt faye quol dexa miraz
""".split())


def is_garbage_word(w):
    w = w.lower()
    n = len(w)
    if n < 5 or not w.isalpha():
        return False
    best = [99] * (n + 1)
    best[0] = 0
    for i in range(n):
        if best[i] == 99:
            continue
        for L in range(2, 9):
            j = i + L
            if j > n:
                break
            if best[i] + 1 < best[j] and w[i:j] in _GARB_SYL:
                best[j] = best[i] + 1
    return best[n] != 99 and best[n] >= 2


# ----------------------------------------------------------------------------------------
# Name vocabularies
# ----------------------------------------------------------------------------------------
# One GLOBAL legal-class table (the generator mixes legal forms across countries).
L_INC, L_LLC, L_LP, L_LLP, L_PC, L_PLC, L_PVT, L_LTD, L_PUBLIC = 1, 2, 4, 8, 16, 32, 64, 128, 256
_FR_LEGAL = ["sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei", "selarl", "earl", "gie", "sca",
             "scop", "gmbh", "selas", "eirl", "scm", "scp"]
LEGAL_BITS = {
    "inc": L_INC, "incorporated": L_INC, "corp": L_INC, "corporation": L_INC, "inc.": L_INC,
    "llc": L_LLC, "lc": L_LLC, "l1c": L_LLC,
    "lp": L_LP, "llp": L_LLP, "pc": L_PC, "pllc": L_PC, "pa": L_PC, "plc": L_PLC,
    "pvt": L_PVT, "private": L_PVT, "praivet": L_PVT, "praibhet": L_PVT, "piraivet": L_PVT,
    "privet": L_PVT, "prayivet": L_PVT, "praivat": L_PVT, "prvt": L_PVT,
    "ltd": L_LTD, "limited": L_LTD, "limitet": L_LTD, "limitad": L_LTD, "limtid": L_LTD,
    "limitid": L_LTD, "limited.": L_LTD, "lmtd": L_LTD,
    "public": L_PUBLIC,
    "elelpi": L_LLP, "elelpee": L_LLP, "elalpi": L_LLP,
}
for _i, _f in enumerate(_FR_LEGAL):
    LEGAL_BITS[_f] = 1 << (9 + _i)
LEGAL_BITS["5arl"] = LEGAL_BITS["sarl"]
_NEUTRAL_LEGAL = {"co", "company", "cos", "lnc"}
_LEET_WORDS = {"lnc": "inc", "l1c": "llc", "1lc": "llc"}

HONORIFICS = {"mr", "mrs", "ms", "messrs", "shri", "sri", "smt", "dr", "the"}
CORE_STOP = {"and", "of", "the", "de", "du", "des", "la", "le", "les", "d", "l", "et", "a", "an",
             "for", "in", "at", "by", "on", "to", "s", "y", "da", "del", "di"}
NOISE_WORDS = {"center", "centre", "services", "service", "trading", "partners", "enterprises",
               "enterprise", "co", "corp", "corporation", "company", "inc", "llc", "ltd", "limited",
               "private", "pvt", "lp", "llp", "council", "federation", "trust", "district", "board",
               "associates", "sons", "solutions", "consultants", "consulting", "and", "sys", "one"}
DECOY_WORDS = {"holding", "holdings", "group", "groupe", "international", "participations",
               "developpement", "development", "distribution", "ventures", "exports", "export",
               "infratech", "medicals", "industries", "electronics", "buildtech", "constructions",
               "infracon", "marketing", "public", "north", "south", "east", "west", "coastal",
               "eastgate", "associes", "foundation", "global", "metal", "works", "france", "india"}
GENERIC_NAME_WORDS = NOISE_WORDS | DECOY_WORDS | set("""
clinic clinique hospital hopital school ecole college lycee club association society trust care health
medical pharmacy pharmacie pharma solutions systems technologies technology tech national united general
new first best city home auto garage sante sport sportive sports amicale comite union maison societe
consulting consultants traders industries agency academy church primaire maternelle elementaire amis
anciens jeunes parents culture culturelle loisirs fetes section foyer ets etablissements institut atelier
ateliers residence theatre danse musique patrimoine compagnie cie fils freres conseil agricole collectif
notre dame saint st sainte ehpad centre hospitalier centre medical infra builders developers projects
foods food finance investments investment properties realty estate constructions construction logistics
digital power energy agro exim impex overseas trading store shop mart bakery restaurant hotel hotels
law office dental family church ministries fellowship baptist apex prime summit helios alliance pacific
""".split())

# ----------------------------------------------------------------------------------------
# Name regexes
# ----------------------------------------------------------------------------------------
_RE_PIPE_TAIL = re.compile(r"\s*\|.*$")
_RE_PHONE_TAIL = re.compile(r"\s+-\s*\+?\d[\d\s\-]{6,}\s*$")
_RE_STORE_TAIL = re.compile(r"\s*#\s?\d{3,}\s*$")
_RE_WWW = re.compile(r"(?i)\bwww\.")
_RE_LEAD_JUNK = re.compile(r"^[\s\-><.*#@~_=+|:;,!?]+")
_RE_TRAIL_JUNK = re.compile(r"[\s\-><*#@~_=+|:;,!?]+$")
_RE_DOMAIN = re.compile(r"^[#@]?([a-z0-9][a-z0-9&\-_]*)\.(?:com|co\.in|in|net|org|fr|biz|info|co|io|us)$")
_RE_ALIAS = re.compile(
    r"\s+(doing\s+business\s+as|d\s*/\s*b\s*/\s*a|d\.\s*b\.\s*a\.?|dba:?|a\s*/\s*k\s*/\s*a|"
    r"a\.\s*k\.\s*a\.?|aka|f\s*/\s*k\s*/\s*a|f\.\s*k\.\s*a\.?|fka|formerly\s+known\s+as|formerly:?|"
    r"trading\s+as|t/a|née)\s+", re.IGNORECASE)
_WEAK_ALIAS = {"aka", "a/k/a", "a.k.a", "a.k.a.", "t/a"}
_ALIAS_LHS_EXTRA = {"sys", "one", "co", "labs", "lab", "inc", "llc", "+", "&"}
_RE_DOTTED = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]\.){2,}(?:[A-Za-z](?![A-Za-z]))?")
_RE_NONALNUM = re.compile(r"[^a-z0-9]+")
_RE_COUNTRY_TOK = re.compile(r"^(?:f[rn]{1,2}a?n?ce|india)$")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"})

NAME_FIELDS = ("name", "core", "concat", "fconcat", "fconcat_del", "skel", "acr", "short", "tok0",
               "tokN", "legal", "is_domain", "is_garbage", "has_alias", "non_latin", "ctok")
EMPTY_NAME = ("", "", "", "", "", "", "", "", "", "", 0, 0, 0, 0, 0, 0)


def _leet_fix(t):
    if t in _LEET_WORDS:
        return _LEET_WORDS[t]
    nd = sum(ch.isdigit() for ch in t)
    if nd == 0 or nd > 2:
        return t
    na = sum(ch.isalpha() for ch in t)
    if na < 2:
        return t
    return t.translate(_LEET)


def _tokens_from(s):
    """Lowercase ASCII string -> cleaned token list (leet, single-letter merge, dedupe)."""
    toks = _RE_NONALNUM.sub(" ", s).split()
    toks = [_leet_fix(t) for t in toks]
    merged, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            merged.append("".join(run) if len(run) > 1 else run[0])
            run = []
        merged.append(t)
    if run:
        merged.append("".join(run) if len(run) > 1 else run[0])
    out = []
    for t in merged:
        if not out or out[-1] != t:
            out.append(t)
    return out


def parse_name(raw):
    if raw is None:
        return EMPTY_NAME
    s = str(raw)
    if not s.strip():
        return EMPTY_NAME
    if not s.isascii():
        s = unicodedata.normalize("NFC", strip_zw(s))
    non_latin = 1 if has_indic(s) else 0

    s = _RE_PIPE_TAIL.sub("", s)
    s = _RE_PHONE_TAIL.sub("", s)
    s = _RE_STORE_TAIL.sub("", s)
    s = _RE_WWW.sub("", s).strip()
    is_handle = s[:1] in ("#", "@") and " " not in s and len(s) > 2
    s = _RE_LEAD_JUNK.sub("", s)
    s = _RE_TRAIL_JUNK.sub("", s).strip()
    if not s:
        return EMPTY_NAME

    # Domain / handle names
    is_domain = 1 if is_handle else 0
    low = s.lower()
    if " " not in low:
        m = _RE_DOMAIN.match(low)
        if m:
            s, is_domain = m.group(1), 1
        elif len(low) >= 10 and low.endswith("com") and low.isalnum():
            s, is_domain = low[:-3], 1

    # Alias markers: "<garbage> dba <real name>" -> real name
    has_alias = 0
    m = _RE_ALIAS.search(s)
    if m:
        lhs, rhs = s[: m.start()].strip(), s[m.end():].strip()
        marker = re.sub(r"\s+", " ", m.group(1).lower()).rstrip(":")
        lhs_toks = [t for t in (re.sub(r"[^\w]+", " ", lhs.lower()).split()) if t not in _ALIAS_LHS_EXTRA]
        garbage_lhs = bool(lhs_toks) and all(is_garbage_word(t) for t in lhs_toks)
        if rhs and len(lhs_toks) <= 2 and (garbage_lhs or marker not in _WEAK_ALIAS):
            s, has_alias = rhs, 1

    s = _RE_DOTTED.sub(lambda mm: mm.group(0).replace(".", ""), s)
    s = s.replace("&", " and ").replace("+", " and ")
    s_del = None
    if not s.isascii():
        if non_latin:
            s = transliterate(apply_lexicon(s))
        else:
            s_del = delete_non_ascii(s.lower())
        s = fold_ascii(s)
    s = s.lower()

    toks = _tokens_from(s)
    if not toks:
        return EMPTY_NAME
    while len(toks) > 1 and toks[0] in HONORIFICS:
        toks.pop(0)
    ctok = 0
    if len(toks) > 1:
        kept = [t for t in toks if not _RE_COUNTRY_TOK.match(t)]
        if kept and len(kept) < len(toks):
            toks, ctok = kept, 1
    # translit legal abbreviation "pra li" (pvt ltd)
    legal = 0
    if "pra" in toks and "li" in toks and len(toks) > 2:
        i = toks.index("pra")
        if i + 1 < len(toks) and toks[i + 1] == "li":
            legal |= L_PVT | L_LTD
            toks = toks[:i] + toks[i + 2:]
    body = []
    for t in toks:
        bit = LEGAL_BITS.get(t)
        if bit:
            legal |= bit
        elif t not in _NEUTRAL_LEGAL:
            body.append(t)
    core_toks = [t for t in body if t not in CORE_STOP] or body or toks

    name = " ".join(toks)
    core = " ".join(core_toks)
    concat = "".join(core_toks)
    fconcat = "".join(toks)
    fconcat_del = _RE_NONALNUM.sub("", s_del) if s_del is not None else fconcat
    acr = "".join(t[0] for t in core_toks if t[0].isalpha()) if len(core_toks) >= 2 else ""
    short = core_toks[0] if (len(core_toks) == 1 and 2 <= len(core_toks[0]) <= 5
                             and core_toks[0].isalpha()) else ""
    is_garbage = 1 if (len(core_toks) == 1 and is_garbage_word(core_toks[0])) else 0
    return (name, core, concat, fconcat, fconcat_del, skeleton(concat), acr, short,
            core_toks[0], core_toks[-1], legal, is_domain, is_garbage, has_alias, non_latin, ctok)


# ----------------------------------------------------------------------------------------
# Geography tables
# ----------------------------------------------------------------------------------------
_US_STATE_NAMES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "dc": "district of columbia",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana",
    "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio",
    "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "pr": "puerto rico",
}
US_LOOKUP = {}
for _c, _n in _US_STATE_NAMES.items():
    US_LOOKUP[_c] = _c
    US_LOOKUP[_n] = _c
US_LOOKUP["d c"] = "dc"
US_LOOKUP["washington dc"] = "dc"

IN_LOOKUP = {
    "maharashtra": "mh", "mh": "mh", "rajasthan": "rj", "rj": "rj", "madhya pradesh": "mp", "mp": "mp",
    "uttar pradesh": "up", "up": "up", "u p": "up", "bihar": "br", "br": "br", "delhi": "dl", "dl": "dl",
    "nct of delhi": "dl", "haryana": "hr", "hr": "hr", "west bengal": "wb", "wb": "wb", "gujarat": "gj",
    "gj": "gj", "tamil nadu": "tn", "tamilnadu": "tn", "tn": "tn", "karnataka": "ka", "ka": "ka",
    "kerala": "kl", "keralam": "kl", "kl": "kl", "telangana": "tgap", "telengana": "tgap", "tg": "tgap",
    "ts": "tgap", "andhra pradesh": "tgap", "ap": "tgap", "punjab": "pb", "pb": "pb", "orissa": "od",
    "odisha": "od", "od": "od", "or": "od", "goa": "ga", "ga": "ga", "assam": "as", "as": "as",
    "jharkhand": "jh", "jh": "jh", "chhattisgarh": "cg", "chattisgarh": "cg", "cg": "cg", "ct": "cg",
    "uttarakhand": "uk", "uttaranchal": "uk", "uk": "uk", "himachal pradesh": "hp", "hp": "hp",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "jk": "jk", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "py": "py", "arunachal pradesh": "ar", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "sikkim": "sk", "tripura": "tr", "ladakh": "lk",
    "andaman and nicobar islands": "an", "andaman nicobar islands": "an", "lakshadweep": "ld",
    "dadra and nagar haveli": "dn", "daman and diu": "dn",
}
_IN_NATIVE_RAW = {
    "महाराष्ट्र": "mh", "राजस्थान": "rj", "मध्य प्रदेश": "mp", "उत्तर प्रदेश": "up", "बिहार": "br",
    "दिल्ली": "dl", "हरियाणा": "hr", "পশ্চিমবঙ্গ": "wb", "ગુજરાત": "gj", "தமிழ்நாடு": "tn",
    "ಕರ್ನಾಟಕ": "ka", "കേരളം": "kl", "తెలంగాణ": "tgap", "ఆంధ్రప్రదేశ్": "tgap",
    "ఆంధ్ర ప్రదేశ్": "tgap", "ਪੰਜਾਬ": "pb", "ଓଡ଼ିଶା": "od", "ଓଡିଶା": "od",
}
IN_NATIVE_STATES = {unicodedata.normalize("NFC", k): v for k, v in _IN_NATIVE_RAW.items()}
IN_STATE_SKEL = {}
for _n, _c in IN_LOOKUP.items():
    if len(_n) >= 4:
        _sk = skeleton(_n.replace(" ", ""))
        if len(_sk) >= 3:
            IN_STATE_SKEL.setdefault(_sk, _c)

FR_REGIONS = {
    "auvergne rhone alpes": "ara", "bourgogne franche comte": "bfc", "bretagne": "bre",
    "centre val de loire": "cvl", "corse": "cor", "grand est": "ges", "hauts de france": "hdf",
    "ile de france": "idf", "normandie": "nor", "nouvelle aquitaine": "naq", "occitanie": "occ",
    "pays de la loire": "pdl", "provence alpes cote d azur": "pac", "provence alpes cote dazur": "pac",
    "guadeloupe": "glp", "martinique": "mtq", "guyane": "guf", "la reunion": "reu", "reunion": "reu",
    "mayotte": "myt",
}
_DEPT_TABLE = """
01 ain|02 aisne|03 allier|04 alpes de haute provence|05 hautes alpes|06 alpes maritimes|07 ardeche|
08 ardennes|09 ariege|10 aube|11 aude|12 aveyron|13 bouches du rhone|14 calvados|15 cantal|16 charente|
17 charente maritime|18 cher|19 correze|2a corse du sud|2b haute corse|21 cote d or|21 cote dor|
22 cotes d armor|22 cotes darmor|23 creuse|24 dordogne|25 doubs|26 drome|27 eure|28 eure et loir|
29 finistere|30 gard|31 haute garonne|32 gers|33 gironde|34 herault|35 ille et vilaine|36 indre|
37 indre et loire|38 isere|39 jura|40 landes|41 loir et cher|42 loire|43 haute loire|44 loire atlantique|
45 loiret|46 lot|47 lot et garonne|48 lozere|49 maine et loire|50 manche|51 marne|52 haute marne|
53 mayenne|54 meurthe et moselle|55 meuse|56 morbihan|57 moselle|58 nievre|59 nord|60 oise|61 orne|
62 pas de calais|63 puy de dome|64 pyrenees atlantiques|65 hautes pyrenees|66 pyrenees orientales|
67 bas rhin|68 haut rhin|69 rhone|70 haute saone|71 saone et loire|72 sarthe|73 savoie|74 haute savoie|
75 paris|76 seine maritime|77 seine et marne|78 yvelines|79 deux sevres|80 somme|81 tarn|
82 tarn et garonne|83 var|84 vaucluse|85 vendee|86 vienne|87 haute vienne|88 vosges|89 yonne|
90 territoire de belfort|91 essonne|92 hauts de seine|93 seine saint denis|94 val de marne|95 val d oise|
95 val doise|971 guadeloupe|972 martinique|973 guyane|974 la reunion|976 mayotte
"""
_REGION_DEPTS = {
    "ara": "01 03 07 15 26 38 42 43 63 69 73 74", "bfc": "21 25 39 58 70 71 89 90",
    "bre": "22 29 35 56", "cvl": "18 28 36 37 41 45", "cor": "2a 2b 20",
    "ges": "08 10 51 52 54 55 57 67 68 88", "hdf": "02 59 60 62 80",
    "idf": "75 77 78 91 92 93 94 95", "nor": "14 27 50 61 76",
    "naq": "16 17 19 23 24 33 40 47 64 79 86 87", "occ": "09 11 12 30 31 32 34 46 48 65 66 81 82",
    "pdl": "44 49 53 72 85", "pac": "04 05 06 13 83 84",
    "glp": "971", "mtq": "972", "guf": "973", "reu": "974", "myt": "976",
}
DEPT_REGION = {}
for _r, _ds in _REGION_DEPTS.items():
    for _d in _ds.split():
        DEPT_REGION[_d] = _r
FR_DEPTS = {}
for _entry in _DEPT_TABLE.replace("\n", "").split("|"):
    _entry = _entry.strip()
    if _entry:
        _code, _name = _entry.split(" ", 1)
        FR_DEPTS[_name.strip()] = DEPT_REGION.get(_code, "")
FR_CITIES = {
    "bordeaux": "naq", "merignac": "naq", "pessac": "naq", "la teste de buch": "naq",
    "lege cap ferret": "naq", "lille": "hdf", "tourcoing": "hdf", "roubaix": "hdf",
    "dunkerque": "hdf", "calais": "hdf", "nantes": "pdl", "saint nazaire": "pdl",
    "saint herblain": "pdl", "la baule escoublac": "pdl", "pornic": "pdl",
}
FR_CITY_ALIAS = {"la baule": "la baule escoublac", "la teste": "la teste de buch",
                 "cap ferret": "lege cap ferret", "dunkirk": "dunkerque", "saint herblin": "saint herblain"}
_FR_CITY_NAMES = list(FR_CITIES)

IN_CITY_ALIASES = {
    "bombay": "mumbai", "greater bombay": "mumbai", "mumbai city": "mumbai", "mumbai suburban": "mumbai",
    "mumbai ii": "mumbai", "poona": "pune", "pune city": "pune", "madras": "chennai", "calcutta": "kolkata",
    "bangalore": "bengaluru", "bangalore urban": "bengaluru", "bengaluru urban": "bengaluru",
    "bangalore north": "bengaluru", "bangalore south": "bengaluru", "banaglore": "bengaluru",
    "gurgaon": "gurugram", "trivandrum": "thiruvananthapuram", "calicut": "kozhikode",
    "vizag": "visakhapatnam", "vishakhapatnam": "visakhapatnam", "allahabad": "prayagraj",
    "ahmadabad": "ahmedabad", "ahemedabad": "ahmedabad", "ahmadabad city": "ahmedabad",
    "secunderabad": "hyderabad", "secunderbad": "hyderabad", "rangareddi": "rangareddy",
    "k v rangareddy": "rangareddy", "kv rangareddy": "rangareddy", "baroda": "vadodara",
    "mysore": "mysuru", "mangalore": "mangaluru", "belgaum": "belagavi", "gulbarga": "kalaburagi",
    "shimoga": "shivamogga", "bellary": "ballari", "tumkur": "tumakuru", "cuddapah": "kadapa",
    "nasik": "nashik", "cochin": "kochi", "kutch": "kachchh", "morvi": "morbi",
    "ahmed nagar": "ahmednagar", "ahmadnagar": "ahmednagar",
}

# ----------------------------------------------------------------------------------------
# Street vocabularies
# ----------------------------------------------------------------------------------------
US_STYPES = {
    "street": "st", "st": "st", "str": "st", "avenue": "ave", "ave": "ave", "av": "ave", "aven": "ave",
    "avn": "ave", "avenoe": "ave", "avenre": "ave", "drive": "dr", "dr": "dr", "drv": "dr",
    "road": "rd", "rd": "rd", "rad": "rd", "roda": "rd", "lane": "ln", "ln": "ln", "court": "ct",
    "ct": "ct", "circle": "cir", "cir": "cir", "circ": "cir", "place": "pl", "pl": "pl",
    "terrace": "ter", "ter": "ter", "terr": "ter", "trail": "trl", "trl": "trl", "highway": "hwy",
    "hwy": "hwy", "hiway": "hwy", "boulevard": "blvd", "blvd": "blvd", "boul": "blvd", "bd": "blvd",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy", "square": "sq", "sq": "sq", "way": "way",
    "wy": "way", "loop": "loop", "pike": "pike", "turnpike": "tpke", "tpke": "tpke",
    "expressway": "expy", "expy": "expy", "freeway": "fwy", "fwy": "fwy", "crossing": "xing",
    "xing": "xing", "point": "pt", "pt": "pt", "heights": "hts", "hts": "hts", "ridge": "rdg",
    "rdg": "rdg", "trace": "trce", "trce": "trce", "garden": "gdn", "gardens": "gdn", "gdn": "gdn",
    "gdns": "gdn", "run": "run", "alley": "aly", "aly": "aly", "bend": "bnd", "bnd": "bnd",
    "cove": "cv", "cv": "cv", "creek": "crk", "crk": "crk", "crescent": "cres", "cres": "cres",
    "grove": "grv", "grv": "grv", "hollow": "holw", "holw": "holw", "junction": "jct", "jct": "jct",
    "landing": "lndg", "lndg": "lndg", "meadow": "mdw", "meadows": "mdw", "mdw": "mdw", "mdws": "mdw",
    "plaza": "plz", "plz": "plz", "row": "row", "station": "sta", "sta": "sta", "valley": "vly",
    "vly": "vly", "view": "vw", "vw": "vw", "walk": "walk", "path": "path", "route": "rte",
    "rte": "rte", "rt": "rte",
}
FR_STYPES = {
    "rue": "rue", "r": "rue", "ruelle": "rue", "avenue": "av", "av": "av", "ave": "av", "aveneu": "av",
    "boulevard": "bd", "bd": "bd", "blvd": "bd", "bld": "bd", "bouleavrd": "bd", "buolevard": "bd",
    "boulevrad": "bd", "boulevad": "bd", "bouievard": "bd", "bouoevard": "bd", "impasse": "imp",
    "imp": "imp", "impsase": "imp", "ipasse": "imp", "allee": "all", "alle": "all", "all": "all",
    "allees": "all", "place": "pl", "pl": "pl", "route": "rte", "rte": "rte", "quai": "quai",
    "q": "quai", "quia": "quai", "cours": "crs", "crs": "crs", "cour2": "crs", "cour": "cour",
    "chemin": "ch", "ch": "ch", "che": "ch", "square": "sq", "sq": "sq", "passage": "pass",
    "psg": "pass", "pass": "pass", "residence": "res", "res": "res", "rs": "res", "esplanade": "espl",
    "espl": "espl", "espl1": "espl", "cite": "cite", "lotissement": "lot", "domaine": "dom",
    "parvis": "parvis", "chaussee": "chaus", "hameau": "ham", "za": "za", "voie": "voie",
    "sentier": "sent", "promenade": "prom", "villa": "villa", "carrefour": "carr", "corniche": "corn",
}
IN_STYPES = {"road": "rd", "rd": "rd", "street": "st", "st": "st", "marg": "marg", "mg": "marg",
             "lane": "ln", "salai": "salai", "highway": "hwy", "path": "path"}
_STYPES = {"US": US_STYPES, "France": FR_STYPES, "India": IN_STYPES,
           "generic": {**US_STYPES, **FR_STYPES}}
ARTICLES = {"de", "du", "des", "la", "le", "les", "d", "l", "of", "the", "au", "aux", "a"}
DIRECTIONALS = {"n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south", "east", "west",
                "northeast", "northwest", "southeast", "southwest"}
FR_TITLES = {"president", "general", "docteur", "dr", "capitaine", "marechal", "professeur", "lt",
             "colonel", "commandant", "abbe", "lieutenant", "sergent", "soldat", "cardinal"}
SPELLED_ORD = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
               "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
               "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
               "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19",
               "twentieth": "20", "thirtieth": "30", "fortieth": "40", "fiftieth": "50"}
UNIT_WORDS = {"unit", "apt", "apartment", "suite", "fl", "floor", "flr", "bldg", "building", "rm",
              "room", "appt", "app", "appart", "appartement", "bat", "batiment", "etage", "esc",
              "escalier", "log", "logement", "porte", "chambre", "entree", "bsmt", "basement",
              "trlr", "spc", "dept", "ofc", "rez", "rdc", "cond", "condo"}
_UNIT_WORDS_US = UNIT_WORDS | {"ste", "lot", "space", "office"}
_UNIT_DESIGNATORS = UNIT_WORDS | {"ste", "lot", "space", "office", "no", "number"}
_DISCARD_FIRST = {"po", "pmb", "bp", "cs", "cedex", "box"}
_COUNTRY_COMPS = {"france", "india", "usa", "us", "united states", "united states of america",
                  "bharat", "u s a", "u s"}
_NULL_COMPS = {"<null>", "null", "n/a", "na", "none", "nan", "-", "--", ".", "n.a.", "nil", "<na>"}
_CITY_NOISE = {"city", "cty", "ciy", "icty", "town", "township", "twp", "towship", "tonwship",
               "townshiip", "village", "borough", "cdp", "cddp", "county", "carrier", "annex",
               "charter", "municipality"}
STREET_GENERIC = set(US_STYPES) | set(FR_STYPES) | set(IN_STYPES) | ARTICLES | DIRECTIONALS | {
    "unit", "null", "city", "county", "building", "block", "sector", "floor", "near", "opp",
    "opposite", "nagar", "colony", "main", "cross", "phase", "layout", "area", "market", "complex",
    "tower", "house", "society", "saint", "st", "village", "post", "dist", "district", "taluka",
    "tehsil", "ward", "west", "east", "north", "south", "new", "old", "ground", "first", "second",
    "shop", "plot", "flat", "office", "apartment", "apartments", "estate", "industrial", "chs"}

_RE_B3 = re.compile(r"\bb3/(?=[a-z]?\d)")
_RE_NUM_MARK = re.compile(
    r"(?:\bn\s*[°º]\s*|\bn[o°º]\s*\.\s*|\bno\s+(?=[#(]?\s*[a-z]?\d)|\bno(?=\d)|\bnum(?:ero)?\b\.?|#+|"
    r"\bh\s*\.?\s*no\b\.?|\bhno\b\.?|\bhn\b\.?|\bho\s*\.?\s*no\b\.?|\bdoo?ro?\s*no\b\.?|"
    r"\bd\s*\.\s*no\b\.?|\bplot\s*no\b\.?|\bplto\s*no\b|\bpgot\s*no\b|\bpl\s*\.\s*no\b\.?|"
    r"\bflat\s*no\b\.?|\bf\s*\.\s*no\b\.?|\bshop\s*no\b\.?|\bshp\s*no\b\.?|\bsy\s*\.?\s*no\b\.?|"
    r"\bs\s*\.\s*no\b\.?|\bsr\s*\.?\s*no\b\.?|\bsurvey\s*no\b\.?|\bkh\s*\.?\s*no\b\.?|"
    r"\bkhata\s*no\b\.?|\bgat\s*no\b\.?|\bward\s*no\b\.?|\bhouse\s*no\b\.?|\bbuilding\s*no\b\.?|"
    r"\bunit\s*no\b\.?|\broom\s*no\b\.?|\boffice\s*no\b\.?)")
_RE_FR_A_AVENUE = re.compile(r"(?<=\d)\s+a\.\s+(?=[a-z])")
_RE_BOULEVARD_SPLIT = re.compile(r"\bb\s+oulevard\b")
_RE_WORD = re.compile(r"[a-z0-9/\-]+")
_RE_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th|e|er|eme|ere|ieme)$")
_RE_ORD_SUFFIX = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
_RE_GLUED_ALPHA_NUM = re.compile(r"^[a-z]{2,}\d+$")
_RE_NUM_SFX = re.compile(r"^(\d+)(bis|ter|quater|[a-z])$")
_RE_REGION_ADMIN = re.compile(r"^[a-z ]+ (?:hq |city )?region$|regional office")
_RE_DIGITS = re.compile(r"\d+")
_SFX_MAP = {"bis": "b", "b": "b", "ter": "t", "t": "t", "quater": "q", "q": "q"}
_SFX_WORDS = {"bis", "ter", "quater"}
# Single-letter suffixes after a house number ("13 t Rue Dade", "5 B Rue ..."); US excludes
# directionals (N/S/E/W) so "5559 E Orville Ave" is not read as 5559-E.
_SFX_LETTERS = {"France": "abcdefghjkt", "US": "abcdfghjk"}

ADDR_FIELDS = ("addr", "street", "stype", "loc", "city", "state", "hnum", "hn_digits", "hn_sfx",
               "cnums", "digits", "unit", "raw_sig", "clean_sig", "hn_d", "hn_nparts", "hn_hi",
               "hn_clean", "addr_empty")
EMPTY_ADDR = ("", "", "", "", "", "", "", "", "", "", "", "", "", "", -1, 0, -1, 0, 1)

_MODES = {"US": "US", "India": "India", "France": "France"}


def _has_digit(t):
    return any(c.isdigit() for c in t)


def _strip0(p):
    if p.isdigit():
        return p.lstrip("0") or "0"
    m = re.match(r"^(\d+)([a-z]+)$", p)
    if m:
        return (m.group(1).lstrip("0") or "0") + m.group(2)
    return p


def _norm_compound(t):
    parts = [p for p in re.split(r"[/\-]+", t) if p]
    return [_strip0(p)[:12] for p in parts]


def _norm_component(c, mode):
    """Raw component -> (tokens, native_state_code)."""
    if not c.isascii():
        st = IN_NATIVE_STATES.get(c.strip())
        if st and mode in ("India", "generic"):
            return [], st
        c = transliterate(c)
    lc = c.lower()
    lc = _RE_B3.sub("", lc)
    lc = _RE_NUM_MARK.sub(" ", lc)
    if mode == "France":
        lc = _RE_FR_A_AVENUE.sub(" avenue ", lc)
    lc = fold_ascii(lc)
    lc = _RE_BOULEVARD_SPLIT.sub("boulevard", lc)
    out = []
    for w in _RE_WORD.findall(lc):
        w = w.strip("/-")
        if not w:
            continue
        if _has_digit(w):
            out.append(w)
        else:
            out.extend(p for p in re.split(r"[/\-]+", w) if p)
    return out, ""


def _fr_city(words):
    ws = [("saint" if t == "st" else "sainte" if t == "ste" else t) for t in words
          if t != "cedex" and not (t.isdigit() and len(t) == 5)]
    cj = " ".join(ws)
    if not cj:
        return ""
    cj = FR_CITY_ALIAS.get(cj, cj)
    if cj in FR_CITIES:
        return cj
    if len(ws) <= 4 and not any(_has_digit(t) for t in ws):
        m = process.extractOne(cj, _FR_CITY_NAMES, scorer=fuzz.ratio, score_cutoff=90)
        if m is not None:
            return m[0]
    return ""


def _state_of(words, mode):
    cj = " ".join(words)
    if mode == "US":
        st = US_LOOKUP.get(cj)
        if st:
            return st
        if len(words) == 2 and words[0] in US_LOOKUP and words[1].isdigit() and len(words[1]) == 5:
            return US_LOOKUP[words[0]]
        return ""
    if mode == "India":
        return IN_LOOKUP.get(cj, "")
    if mode == "France":
        cj2 = " ".join("saint" if t == "st" else t for t in words)
        return FR_REGIONS.get(cj2) or FR_DEPTS.get(cj2) or ""
    return ""


def _extract_hn(words, mode):
    """-> (index of house-number token, index of separate suffix token or -1) or None."""
    stypes = _STYPES.get(mode, US_STYPES)
    type_idx = None
    if mode in ("US", "France"):
        for j, t in enumerate(words):
            if t in stypes:
                type_idx = j
                break
    for j, t in enumerate(words):
        if not _has_digit(t):
            continue
        if _RE_ORDINAL.match(t) or _RE_GLUED_ALPHA_NUM.match(t):
            continue
        if type_idx is not None and j > type_idx:
            return None
        sfx_j = -1
        if j + 1 < len(words):
            nxt = words[j + 1]
            letters = _SFX_LETTERS.get(mode, "")
            if t.isdigit() and (nxt in _SFX_WORDS or (
                    len(nxt) == 1 and nxt in letters and j + 2 < len(words))):
                sfx_j = j + 1
        return j, sfx_j
    return None


def _hn_fields(tok, sfx_tok, mode):
    """-> hnum, hn_digits, hn_sfx, hn_d, hn_nparts, hn_hi"""
    sfx = ""
    m = _RE_NUM_SFX.match(tok)
    if m:
        parts = [m.group(1).lstrip("0") or "0"]
        sfx = _SFX_MAP.get(m.group(2), m.group(2))
    else:
        parts = _norm_compound(tok)
        if len(parts) == 2 and parts[0].isdigit() and len(parts[1]) == 1 and parts[1].isalpha():
            sfx = _SFX_MAP.get(parts[1], parts[1])
            parts = parts[:1]
    if sfx_tok:
        sfx = _SFX_MAP.get(sfx_tok, sfx_tok)
    hn_hi = -1
    num_parts = [p for p in parts if _has_digit(p)]
    if (mode in ("US", "France") and len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()):
        a, b = int(parts[0][:9]), int(parts[1][:9])
        if a < b <= a + 20:
            hn_hi = b
            parts = parts[:1]
            num_parts = parts
    hnum = "/".join(parts) + sfx
    hn_digits = "".join(_RE_DIGITS.findall("/".join(parts)))[:15]
    m2 = _RE_DIGITS.search(hnum)
    hn_d = int(m2.group(0)[:9]) if m2 else -1
    return hnum, hn_digits, sfx, hn_d, len(num_parts), hn_hi


def _clean_other(words, mode):
    """Locality component canonicalisation."""
    if mode == "US":
        cj = " ".join(words)
        for pre in ("city of ", "town of ", "village of ", "borough of ", "township of "):
            if cj.startswith(pre):
                cj = cj[len(pre):]
        ws = [t for t in cj.split() if t not in _CITY_NOISE]
        ws = [re.sub(r"(?<=[a-z]{3})township$", "", t) for t in ws]
        ws = ["saint" if t == "st" else "mount" if t == "mt" else "fort" if t == "ft" else t for t in ws]
        return [t for t in ws if t]
    if mode == "India":
        cj = " ".join(words)
        if cj in IN_CITY_ALIASES:
            return IN_CITY_ALIASES[cj].split()
        return [IN_CITY_ALIASES.get(t, t) for t in words]
    if mode == "France":
        return [("saint" if t == "st" else "sainte" if t == "ste" else t) for t in words
                if t != "cedex" and not (t.isdigit() and len(t) == 5)]
    return list(words)


def parse_address(raw, country):
    if raw is None:
        return EMPTY_ADDR
    s = str(raw)
    if not s.strip():
        return EMPTY_ADDR
    mode = _MODES.get(country, "generic")
    if not s.isascii():
        s = unicodedata.normalize("NFC", strip_zw(s))
    comps = [re.sub(r"\s+", " ", c).strip() for c in s.split(",")]
    comps = [c for c in comps if c and c.lower() not in _NULL_COMPS]
    if not comps:
        return EMPTY_ADDR
    raw_sig = "|".join(sorted(comps))
    stypes = _STYPES.get(mode, US_STYPES)
    unit_words = _UNIT_WORDS_US if mode == "US" else UNIT_WORDS

    words_l, kinds = [], []
    state, state_i, city, fr_region, fr_postal_region = "", -1, "", "", ""
    for ci, c in enumerate(comps):
        w, native = _norm_component(c, mode)
        if native:
            words_l.append([])
            kinds.append("state")
            state, state_i = native, ci
            continue
        words_l.append(w)
        if not w:
            kinds.append("skip")
            continue
        cj = " ".join(w)
        st = _state_of(w, mode)
        if st:
            kinds.append("state")
            state, state_i = st, ci  # right-most wins (later components overwrite)
            continue
        if mode == "India" and not c.isascii() and has_indic(c):
            sk = skeleton(cj.replace(" ", ""))
            if sk in IN_STATE_SKEL:
                kinds.append("state")
                state, state_i = IN_STATE_SKEL[sk], ci
                continue
        if cj in _COUNTRY_COMPS:
            kinds.append("skip")
            continue
        if mode == "France":
            if w[0].isdigit() and len(w[0]) == 5:
                dep = w[0][:3] if w[0].startswith("97") else w[0][:2]
                fr_postal_region = fr_postal_region or DEPT_REGION.get(dep, "")
                rest = w[1:]
                cc = _fr_city(rest) if rest else ""
                if cc:
                    city = city or cc
                    fr_region = fr_region or FR_CITIES[cc]
                kinds.append("skip")
                continue
            cc = _fr_city(w)
            if cc:
                city = city or cc
                fr_region = fr_region or FR_CITIES[cc]
                kinds.append("skip")
                continue
        first = w[0]
        if first in _DISCARD_FIRST or cj.startswith(("p o box", "po box", "post box", "p o b")):
            kinds.append("skip")
            continue
        if mode == "India" and _RE_REGION_ADMIN.search(cj):
            kinds.append("skip")
            continue
        if len(w) == 1 and w[0].isdigit() and len(w[0]) in (5, 6) and mode != "US":
            kinds.append("skip")  # bare postal code
            continue
        if first in unit_words or (len(w) <= 3 and any(t in ("etage", "floor", "fl") for t in w)):
            kinds.append("unit")
            continue
        if len(w) == 1 and _has_digit(w[0]):
            kinds.append("bare")
            continue
        if any(_has_digit(t) for t in w) or any(t in stypes for t in w):
            kinds.append("street")
            continue
        kinds.append("other")

    if mode == "France" and not state:
        state = fr_region or fr_postal_region
    # Only the right-most state component is the state; earlier ones are locality.
    for ci, k in enumerate(kinds):
        if k == "state" and ci != state_i and words_l[ci]:
            kinds[ci] = "other"

    n = len(kinds)
    street_i, hn_src = -1, None
    for ci in range(n):
        if kinds[ci] == "street" and words_l[ci] and _has_digit(words_l[ci][0]) and \
                not _RE_ORDINAL.match(words_l[ci][0]) and not _RE_GLUED_ALPHA_NUM.match(words_l[ci][0]):
            street_i = ci
            break
    if street_i < 0:
        for ci in range(n):
            if kinds[ci] == "bare" and ci + 1 < n and kinds[ci + 1] == "street":
                street_i, hn_src = ci + 1, ci
                break
    if street_i < 0:
        for ci in range(n):
            if kinds[ci] == "street":
                street_i = ci
                break

    hnum = hn_digits = hn_sfx = ""
    hn_d, hn_nparts, hn_hi, hn_clean = -1, 0, -1, 0
    hn_j = sfx_j = -1
    swords = words_l[street_i] if street_i >= 0 else []
    if hn_src is not None:
        hnum, hn_digits, hn_sfx, hn_d, hn_nparts, hn_hi = _hn_fields(words_l[hn_src][0], "", mode)
        hn_clean = 1
    elif swords:
        r = _extract_hn(swords, mode)
        if r is not None:
            hn_j, sfx_j = r
            hnum, hn_digits, hn_sfx, hn_d, hn_nparts, hn_hi = _hn_fields(
                swords[hn_j], swords[sfx_j] if sfx_j >= 0 else "", mode)
            nxt = sfx_j + 1 if sfx_j >= 0 else hn_j + 1
            hn_clean = 1 if (hn_j == 0 and any(not _has_digit(t) for t in swords[nxt:])) else 0
    if not hnum:
        for ci in range(n):
            if kinds[ci] == "bare":
                hnum, hn_digits, hn_sfx, hn_d, hn_nparts, hn_hi = _hn_fields(words_l[ci][0], "", mode)
                break

    # Street core
    core, stype = [], ""
    last = len(swords) - 1
    for j, t in enumerate(swords):
        if j == hn_j or j == sfx_j:
            continue
        if mode == "US" and j == last and t in ("st", "saint", "street", "str"):
            stype = stype or "st"
            continue
        if t in stypes:
            stype = stype or stypes[t]
            continue
        if t in ARTICLES:
            continue
        if mode == "US" and t in DIRECTIONALS:
            continue
        if mode == "France" and t in FR_TITLES:
            continue
        if t in ("saint", "sainte", "st", "ste"):
            t = "st"
        m = _RE_ORD_SUFFIX.match(t)
        if m:
            t = m.group(1)
        t = SPELLED_ORD.get(t, t)
        core.append(t)
    street = " ".join(core)

    # Numbers
    cnums, digits = [], []
    for ci in range(n):
        if kinds[ci] not in ("street", "bare", "other"):
            continue
        for t in words_l[ci]:
            if not _has_digit(t) or _RE_ORDINAL.match(t) or _RE_GLUED_ALPHA_NUM.match(t):
                continue
            cn = "/".join(_norm_compound(t))
            if cn and cn not in cnums:
                cnums.append(cn)
            for d in _RE_DIGITS.findall(t):
                d = (d.lstrip("0") or "0")[:12]
                if d not in digits:
                    digits.append(d)

    # Locality / city / unit
    loc_toks, others = [], []
    for ci in range(n):
        if kinds[ci] == "other":
            cw = _clean_other(words_l[ci], mode)
            if cw:
                others.append(cw)
                for t in cw:
                    if t not in loc_toks:
                        loc_toks.append(t)
    if mode != "France" and others:
        city = " ".join(others[-1])
    unit = ""
    for ci in range(n):
        if kinds[ci] == "unit":
            rest = [t for t in words_l[ci] if t not in _UNIT_DESIGNATORS]
            if rest:
                unit = rest[-1]
    loc = " ".join(loc_toks)
    addr_toks = ([hnum] if hnum else []) + core + ([stype] if stype else []) + loc_toks
    if mode == "France" and city:
        addr_toks += city.split()
    addr = " ".join(addr_toks)
    clean_sig = f"{hnum}|{stype}|{street}|{city or loc}" if (hnum or street) else ""
    return (addr, street, stype, loc, city, state, hnum, hn_digits, hn_sfx, " ".join(cnums),
            " ".join(digits), unit, raw_sig, clean_sig, hn_d, hn_nparts, hn_hi, hn_clean, 0)


# ----------------------------------------------------------------------------------------
# Crash-safe wrappers
# ----------------------------------------------------------------------------------------
STRICT = False
ERRORS = {"name": 0, "addr": 0}
ERROR_EXAMPLES = []


def parse_name_safe(raw):
    try:
        return parse_name(raw)
    except Exception as exc:
        if STRICT:
            raise
        ERRORS["name"] += 1
        if len(ERROR_EXAMPLES) < 5:
            ERROR_EXAMPLES.append(("name", repr(raw)[:120], repr(exc)[:200]))
        return EMPTY_NAME


def parse_address_safe(raw, country):
    try:
        return parse_address(raw, country)
    except Exception as exc:
        if STRICT:
            raise
        ERRORS["addr"] += 1
        if len(ERROR_EXAMPLES) < 5:
            ERROR_EXAMPLES.append(("addr", repr(raw)[:120], repr(exc)[:200]))
        return EMPTY_ADDR
