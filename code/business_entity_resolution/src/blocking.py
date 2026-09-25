"""
Scalable Multi-Channel Inverted Index Candidate Generation (Blocking).
Partitions candidate search space by country, builds multi-key inverted indexes,
and generates high-recall candidate sets with minimal memory overhead.
"""

from collections import defaultdict, Counter
import time
from .normalizer import clean_string, extract_name_components, extract_addr_components


class MultiIndexBlocker:
    """
    High-recall, ultra-fast inverted index blocker for entity resolution.
    Indexed keys:
      1. Informative brand name tokens (weighted by inverse document frequency)
      2. 4-character brand name prefix (catches spelling variations and typos)
      3. (normalized_number, addr_locality_word) anchors (catches DBA / trade names)
      4. (normalized_number, first_brand_word) anchors
    """
    def __init__(self, max_token_df=3500, max_prefix_df=2500, max_candidates=30):
        self.max_token_df = max_token_df
        self.max_prefix_df = max_prefix_df
        self.max_candidates = max_candidates
        
        self.target_ids = []
        self.cleaned_names = []
        self.cleaned_addrs = []
        self.core_names = []
        self.name_tokens_list = []
        self.addr_nums_list = []
        self.addr_words_list = []
        
        self.token_to_targets = defaultdict(list)
        self.prefix_to_targets = defaultdict(list)
        self.addr_anchor_to_targets = defaultdict(list)
        self.num_brand_to_targets = defaultdict(list)
        self.token_freq = Counter()
        self.prefix_freq = Counter()

    def fit_target_universe(self, target_ids, names, addrs):
        """
        Builds inverted indices over all target records (Source 2 and Source 3 for a country).
        """
        t0 = time.time()
        self.target_ids = target_ids
        n = len(target_ids)
        
        # Pre-allocate lists
        self.cleaned_names = [None] * n
        self.cleaned_addrs = [None] * n
        self.core_names = [None] * n
        self.name_tokens_list = [None] * n
        self.addr_nums_list = [None] * n
        self.addr_words_list = [None] * n
        
        # First pass: clean and count frequencies
        for i in range(n):
            cn = clean_string(names[i])
            ca = clean_string(addrs[i])
            self.cleaned_names[i] = cn
            self.cleaned_addrs[i] = ca
            
            core_name, ntoks, _ = extract_name_components(cn)
            self.core_names[i] = core_name
            self.name_tokens_list[i] = ntoks
            
            nums, awords = extract_addr_components(ca)
            self.addr_nums_list[i] = nums
            self.addr_words_list[i] = awords
            
            for tok in set(ntoks):
                self.token_freq[tok] += 1
                
            if len(core_name) >= 4:
                prefix = core_name[:4]
                self.prefix_freq[prefix] += 1

        # Second pass: populate inverted indexes
        for i in range(n):
            ntoks = self.name_tokens_list[i]
            core_name = self.core_names[i]
            nums = self.addr_nums_list[i]
            awords = self.addr_words_list[i]
            
            # 1. Name tokens
            for tok in set(ntoks):
                if self.token_freq[tok] <= self.max_token_df:
                    self.token_to_targets[tok].append(i)
                    
            # 2. Brand prefix
            if len(core_name) >= 4:
                prefix = core_name[:4]
                if self.prefix_freq[prefix] <= self.max_prefix_df:
                    self.prefix_to_targets[prefix].append(i)
                    
            # 3. Numeric anchors: (num, addr_word)
            if nums and awords:
                for num in nums[:2]:
                    for w in awords[:3]:
                        self.addr_anchor_to_targets[(num, w)].append(i)
                        
            # 4. Numeric + Brand word: (num, brand_tok)
            if nums and ntoks:
                primary_brand = ntoks[0]
                for num in nums[:2]:
                    self.num_brand_to_targets[(num, primary_brand)].append(i)

        elapsed = time.time() - t0
        return elapsed

    def generate_candidates(self, query_ids, query_names, query_addrs):
        """
        Retrieves top candidate target IDs for each query record.
        Returns: {query_id: [candidate_target_ids]}
        """
        candidates_map = {}
        
        for q_idx in range(len(query_ids)):
            qid = query_ids[q_idx]
            cn = clean_string(query_names[q_idx])
            ca = clean_string(query_addrs[q_idx])
            
            core_name, ntoks, _ = extract_name_components(cn)
            nums, awords = extract_addr_components(ca)
            
            scores = Counter()
            
            # 1. Score by name tokens (IDF-weighted)
            for tok in set(ntoks):
                if tok in self.token_to_targets:
                    df = self.token_freq[tok]
                    weight = 100.0 / (df + 10)
                    for tid_idx in self.token_to_targets[tok]:
                        scores[tid_idx] += weight
                        
            # 2. Score by brand prefix (catches minor typos/suffixes)
            if len(core_name) >= 4:
                prefix = core_name[:4]
                if prefix in self.prefix_to_targets:
                    p_df = self.prefix_freq[prefix]
                    p_weight = 30.0 / (p_df + 10)
                    for tid_idx in self.prefix_to_targets[prefix]:
                        scores[tid_idx] += p_weight
                        
            # 3. Score by numeric address anchors
            if nums and awords:
                for num in nums[:2]:
                    for w in awords[:3]:
                        anchor = (num, w)
                        if anchor in self.addr_anchor_to_targets:
                            for tid_idx in self.addr_anchor_to_targets[anchor]:
                                scores[tid_idx] += 45.0
                                
            # 4. Score by numeric + brand anchor
            if nums and ntoks:
                primary_brand = ntoks[0]
                for num in nums[:2]:
                    nbrand = (num, primary_brand)
                    if nbrand in self.num_brand_to_targets:
                        for tid_idx in self.num_brand_to_targets[nbrand]:
                            scores[tid_idx] += 60.0
                            
            if not scores:
                candidates_map[qid] = []
            else:
                top_targets = [self.target_ids[idx] for idx, _ in scores.most_common(self.max_candidates)]
                candidates_map[qid] = top_targets
                
        return candidates_map
