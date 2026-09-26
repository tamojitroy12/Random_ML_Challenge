"""
Deep 26-Dimensional Pairwise Feature Extractor for Business Entity Resolution.
Extracts RapidFuzz metrics, Jaro-Winkler, geographic compatibility, postal codes,
initials/acronyms, locality matching, and composite signals.
"""

from rapidfuzz import fuzz, distance
from .normalizer import clean_string, extract_name_components, extract_addr_components


def compute_pair_features(query_rec, target_rec):
    """
    Computes a 26-dimensional feature vector for a candidate pair (Query, Target).
    """
    q_name = query_rec['cleaned_name']
    t_name = target_rec['cleaned_name']
    
    q_core = query_rec['core_name']
    t_core = target_rec['core_name']
    
    q_addr = query_rec['cleaned_addr']
    t_addr = target_rec['cleaned_addr']
    
    # 1. Name Similarities (RapidFuzz C++)
    f_name_sort = fuzz.token_sort_ratio(q_name, t_name) / 100.0
    f_name_set = fuzz.token_set_ratio(q_name, t_name) / 100.0
    f_name_ratio = fuzz.ratio(q_name, t_name) / 100.0
    f_core_ratio = fuzz.ratio(q_core, t_core) / 100.0
    f_core_set = fuzz.token_set_ratio(q_core, t_core) / 100.0
    
    # Jaro-Winkler similarity (prefix sensitive)
    f_jw = distance.JaroWinkler.similarity(q_name, t_name)
    
    # First word ratio
    q_words = query_rec['name_tokens']
    t_words = target_rec['name_tokens']
    if q_words and t_words:
        f_first_word = fuzz.ratio(q_words[0], t_words[0]) / 100.0
    else:
        f_first_word = 0.0
        
    # Relative length difference
    max_len = max(len(q_name), len(t_name), 1)
    f_len_diff = abs(len(q_name) - len(t_name)) / max_len
    
    # Brand prefix match
    q_prefix = q_core[:4] if len(q_core) >= 4 else q_core
    t_prefix = t_core[:4] if len(t_core) >= 4 else t_core
    f_prefix_match = 1.0 if (q_prefix and t_prefix and q_prefix == t_prefix) else 0.0
    
    # Initials / Acronym Cross-Match
    q_acr = query_rec.get('acronym', '')
    t_acr = target_rec.get('acronym', '')
    q_inits = "".join(w[0] for w in q_name.split() if w and w[0].isalnum())
    t_inits = "".join(w[0] for w in t_name.split() if w and w[0].isalnum())
    
    f_acronym = 0.0
    if q_acr and (q_acr == t_core or q_acr == t_acr or q_acr == t_inits):
        f_acronym = 1.0
    elif t_acr and (t_acr == q_core or t_acr == q_acr or t_acr == q_inits):
        f_acronym = 1.0
    elif q_inits and t_inits and (q_inits == t_inits or q_inits == t_core or t_inits == q_core):
        f_acronym = 1.0

    # Legal suffix compatibility
    q_suf = query_rec['legal_suffixes']
    t_suf = target_rec['legal_suffixes']
    if not q_suf or not t_suf:
        f_suf_compat = 0.5
    elif q_suf.intersection(t_suf):
        f_suf_compat = 1.0
    else:
        f_suf_compat = 0.0
        
    # 2. Address & Geographic Similarities
    t_addr_empty = 1.0 if not t_addr else 0.0
    
    if t_addr_empty or not q_addr:
        f_addr_set = 0.0
        f_addr_jaccard = 0.0
        f_addr_overlap = 0.0
        f_num_jaccard = 0.0
        f_exact_num = 0.0
        f_locality_match = 0.0
        f_post_match = 0.0
        f_post_conflict = 0.0
        f_state_match = 0.0
        f_state_conflict = 0.0
    else:
        f_addr_set = fuzz.token_set_ratio(q_addr, t_addr) / 100.0
        
        q_aw = set(query_rec['addr_words'])
        t_aw = set(target_rec['addr_words'])
        union_aw = q_aw.union(t_aw)
        intersect_aw = q_aw.intersection(t_aw)
        f_addr_jaccard = (len(intersect_aw) / len(union_aw)) if union_aw else 0.0
        f_addr_overlap = float(len(intersect_aw))
        
        # Locality match: share distinctive words of length >= 4
        distinctive_loc = [w for w in intersect_aw if len(w) >= 4]
        f_locality_match = 1.0 if distinctive_loc else 0.0
        
        q_nums = set(query_rec['addr_nums'])
        t_nums = set(target_rec['addr_nums'])
        union_nums = q_nums.union(t_nums)
        f_num_jaccard = (len(q_nums.intersection(t_nums)) / len(union_nums)) if union_nums else 0.0
        f_exact_num = 1.0 if q_nums.intersection(t_nums) else 0.0

        # Postal code check
        q_post = query_rec.get('postal_code', '')
        t_post = target_rec.get('postal_code', '')
        if q_post and t_post:
            f_post_match = 1.0 if q_post == t_post else 0.0
            f_post_conflict = 1.0 if q_post != t_post else 0.0
        else:
            f_post_match = 0.0
            f_post_conflict = 0.0

        # State check
        q_state = query_rec.get('state_code', '')
        t_state = target_rec.get('state_code', '')
        if q_state and t_state:
            f_state_match = 1.0 if q_state == t_state else 0.0
            f_state_conflict = 1.0 if q_state != t_state else 0.0
        else:
            f_state_match = 0.0
            f_state_conflict = 0.0

    # 3. Cross-field Signals
    f_name_in_addr = 1.0 if (q_core and len(q_core) >= 4 and q_core in t_addr) or (t_core and len(t_core) >= 4 and t_core in q_addr) else 0.0
    f_composite = (f_name_sort * (f_addr_set if not t_addr_empty else f_name_ratio)) ** 0.5
    f_num_locality_hit = 1.0 if (f_exact_num > 0.5 and f_locality_match > 0.5) else 0.0
    
    # 4. Target Source Indicator
    target_id = target_rec.get('target_id', '')
    f_source = 1.0 if target_id.startswith('S2-') else (2.0 if target_id.startswith('S3-') else 0.0)

    return [
        f_name_sort,
        f_name_set,
        f_name_ratio,
        f_core_ratio,
        f_core_set,
        f_jw,
        f_first_word,
        f_len_diff,
        f_prefix_match,
        f_acronym,
        f_suf_compat,
        t_addr_empty,
        f_addr_set,
        f_addr_jaccard,
        f_addr_overlap,
        f_locality_match,
        f_num_jaccard,
        f_exact_num,
        f_post_match,
        f_post_conflict,
        f_state_match,
        f_state_conflict,
        f_name_in_addr,
        f_composite,
        f_num_locality_hit,
        f_source
    ]


FEATURE_NAMES = [
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_ratio",
    "core_ratio",
    "core_token_set_ratio",
    "jaro_winkler_sim",
    "core_first_word_ratio",
    "name_length_diff",
    "prefix_match",
    "acronym_match",
    "suffix_compatible",
    "target_addr_empty",
    "addr_token_set_ratio",
    "addr_token_jaccard",
    "addr_token_overlap",
    "locality_match",
    "addr_num_jaccard",
    "exact_number_match",
    "postal_code_match",
    "postal_code_conflict",
    "geo_state_match",
    "geo_state_conflict",
    "name_in_addr",
    "composite_confidence",
    "num_locality_hit",
    "target_source"
]
