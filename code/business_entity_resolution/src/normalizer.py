"""
Multilingual Text & Address Normalizer for Large-Scale Entity Resolution.
Supports Indic script phonetization, French accent folding, legal suffix isolation,
and zero-stripped address number extraction.
"""

import re
import unicodedata

# 1. Unified Indic transliteration table (Brahmi relative Unicode offset structure)
INDIC_BLOCKS = [
    (0x0900, 0x097F), # Devanagari (Hindi, Marathi, Sanskrit)
    (0x0980, 0x09FF), # Bengali / Assamese
    (0x0A00, 0x0A7F), # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF), # Gujarati
    (0x0B00, 0x0B7F), # Oriya
    (0x0B80, 0x0BFF), # Tamil
    (0x0C00, 0x0C7F), # Telugu
    (0x0C80, 0x0CFF), # Kannada
    (0x0D00, 0x0D7F), # Malayalam
]

OFFSET_MAP = {
    0x02: "n", 0x03: "h", 0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo",
    0x0B: "ri", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "nn",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "rr", 0x32: "ll", 0x33: "ll", 0x34: "lll", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x3E: "a", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo",
    0x43: "ri", 0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x4D: "" # Virama
}

# 2. Comprehensive Legal Suffixes across US, India, and France
LEGAL_SUFFIXES = {
    # US
    "llc", "inc", "corp", "corporation", "company", "co", "ltd", "limited", "incorporated",
    # India
    "pvt", "private", "llp", "traders", "enterprises", "enterprise", "industries", "services",
    # France
    "sarl", "sasu", "sas", "eurl", "sci", "sa", "gmbh", "cie", "fils", "freres"
}

GENERIC_STOPWORDS = {
    "and", "the", "of", "in", "for", "at", "by", "from", "with", "near", "opp", "opposite",
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "nagar", "colony",
    "floor", "flr", "plot", "door", "shop", "no", "flat", "rue", "boulevard", "blvd",
    "allée", "allee", "chemin", "impasse"
}

DOMAIN_REGEX = re.compile(r'\.(com|in|org|net|co|io|fr|gov|edu)\b', re.IGNORECASE)
PUNCT_REGEX = re.compile(r'[^a-z0-9]+')
DIGIT_PREFIX_ZERO_REGEX = re.compile(r'\b0+(\d+)\b')


def transliterate_text(text: str) -> str:
    """Converts Indic script characters to phonetic Latin representation."""
    if not text or text.isascii():
        return text
    out = []
    for ch in text:
        cp = ord(ch)
        if cp <= 127:
            out.append(ch)
            continue
        mapped = False
        for start, end in INDIC_BLOCKS:
            if start <= cp <= end:
                out.append(OFFSET_MAP.get(cp - start, ""))
                mapped = True
                break
        if not mapped:
            out.append(ch)
    return "".join(out)


def clean_string(text: str) -> str:
    """Normalizes text: transliterates Indic, folds accents, strips domains, lowercases."""
    if not text or text == "None" or text == "null":
        return ""
    # Fast path for pure ASCII
    if not text.isascii():
        text = transliterate_text(text)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    text = DOMAIN_REGEX.sub('', text)
    text = DIGIT_PREFIX_ZERO_REGEX.sub(r'\1', text)
    text = PUNCT_REGEX.sub(' ', text)
    return text.strip()



def extract_name_components(cleaned_name: str):
    """
    Splits business name into core brand name, tokens, and legal suffixes.
    Returns (core_brand_name, informative_tokens, legal_suffixes_found).
    """
    words = cleaned_name.split()
    core_words = []
    suffixes_found = []
    
    for w in words:
        if w in LEGAL_SUFFIXES:
            suffixes_found.append(w)
        else:
            core_words.append(w)
            
    core_name = " ".join(core_words) if core_words else cleaned_name
    tokens = [w for w in core_words if w not in GENERIC_STOPWORDS and len(w) >= 2]
    
    return core_name, tokens, set(suffixes_found)


def extract_addr_components(cleaned_addr: str):
    """
    Extracts normalized numeric anchors and distinctive address words.
    Returns (numeric_anchors, address_tokens).
    """
    words = cleaned_addr.split()
    nums = []
    tokens = []
    
    for w in words:
        if any(c.isdigit() for c in w):
            num_clean = re.sub(r'^0+', '', w)
            if num_clean:
                nums.append(num_clean)
        elif len(w) >= 3 and w not in GENERIC_STOPWORDS:
            tokens.append(w)
            
    return nums, tokens
