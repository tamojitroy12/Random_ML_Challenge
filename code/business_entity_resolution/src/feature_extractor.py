"""
High-Performance Pairwise Feature Extraction for Business Entity Resolution.
Computes RapidFuzz C++ string metrics, address token overlaps, numeric anchor matches,
and domain-specific compatibility signals.
"""

from rapidfuzz import fuzz
from .normalizer import clean_string, extract_name_components, extract_addr_components


def compute_pair_features(query_rec, target_rec):
    """
    Computes a vector of 15 informative features for a candidate pair.
    
    query_rec: dict with keys 'cleaned_name', 'cleaned_addr', 'core_name',
                         'name_tokens', 'legal_suffixes', 'addr_nums', 'addr_words'
    target_rec: dict with same keys + 'target_id'
    """
    q_name = query_rec['cleaned_name']
    t_name = target_rec['cleaned_name']
    
    q_core = query_rec['core_name']
    t_core = target_rec['core_name']
    
    q_addr = query_rec['cleaned_addr']
    t_addr = target_rec['cleaned_addr']
    
    # 1. Name Similarities (RapidFuzz C++ accelerated)
    f_name_sort = fuzz.token_sort_ratio(q_name, t_name) / 100.0
    f_name_set = fuzz.token_set_ratio(q_name, t_name) / 100.0
    f_name_ratio = fuzz.ratio(q_name, t_name) / 100.0
    f_core_ratio = fuzz.ratio(q_core, t_core) / 100.0
    f_core_set = fuzz.token_set_ratio(q_core, t_core) / 100.0
    
    # Brand prefix match
    q_prefix = q_core[:4] if len(q_core) >= 4 else q_core
    t_prefix = t_core[:4] if len(t_core) >= 4 else t_core
    f_prefix_match = 1.0 if (q_prefix and t_prefix and q_prefix == t_prefix) else 0.0
    
    # Legal suffix compatibility
    q_suf = query_rec['legal_suffixes']
    t_suf = target_rec['legal_suffixes']
    if not q_suf or not t_suf:
        f_suf_compat = 0.5 # Neutral
    elif q_suf.intersection(t_suf):
        f_suf_compat = 1.0 # Compatible
    else:
        f_suf_compat = 0.0 # Contradictory
        
    # 2. Address Similarities
    t_addr_empty = 1.0 if not t_addr else 0.0
    
    if t_addr_empty or not q_addr:
        f_addr_set = 0.0
        f_addr_jaccard = 0.0
        f_addr_overlap = 0.0
        f_num_jaccard = 0.0
        f_exact_num = 0.0
    else:
        f_addr_set = fuzz.token_set_ratio(q_addr, t_addr) / 100.0
        
        q_aw = set(query_rec['addr_words'])
        t_aw = set(target_rec['addr_words'])
        union_aw = q_aw.union(t_aw)
        f_addr_jaccard = (len(q_aw.intersection(t_aw)) / len(union_aw)) if union_aw else 0.0
        f_addr_overlap = float(len(q_aw.intersection(t_aw)))
        
        q_nums = set(query_rec['addr_nums'])
        t_nums = set(target_rec['addr_nums'])
        union_nums = q_nums.union(t_nums)
        f_num_jaccard = (len(q_nums.intersection(t_nums)) / len(union_nums)) if union_nums else 0.0
        f_exact_num = 1.0 if q_nums.intersection(t_nums) else 0.0

    # 3. Cross-field Signals
    f_name_in_addr = 1.0 if (q_core and len(q_core) >= 4 and q_core in t_addr) or (t_core and len(t_core) >= 4 and t_core in q_addr) else 0.0
    
    # 4. Target Source indicator (S2 vs S3)
    target_id = target_rec.get('target_id', '')
    f_source = 1.0 if target_id.startswith('S2-') else (2.0 if target_id.startswith('S3-') else 0.0)

    return [
        f_name_sort,
        f_name_set,
        f_name_ratio,
        f_core_ratio,
        f_core_set,
        f_prefix_match,
        f_suf_compat,
        t_addr_empty,
        f_addr_set,
        f_addr_jaccard,
        f_addr_overlap,
        f_num_jaccard,
        f_exact_num,
        f_name_in_addr,
        f_source
    ]


FEATURE_NAMES = [
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_ratio",
    "core_ratio",
    "core_token_set_ratio",
    "prefix_match",
    "suffix_compatible",
    "target_addr_empty",
    "addr_token_set_ratio",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_num_jaccard",
    "exact_number_match",
    "name_in_addr",
    "target_source"
]
