from collections import defaultdict
from time import time
from ._string_distance import jamo_levenshtein
from ._string_distance import levenshtein
from ._hangle import decompose
from ._hangle import character_is_korean

class LevenshteinIndex:
    def __init__(self, word_counter=None, levenshtein_distance=levenshtein, verbose=False):
        self._words = {}
        self._index = {} # character to words
        self._cho_index, self._jung_index, self._jong_index = {}, {}, {}
        if word_counter:
            self.indexing(word_counter)
        self.verbose = verbose
        self.jamo_score = 1/3
        
    def indexing(self, word_counter):
        self._words = word_counter if type(word_counter) == dict else {w:1 for w in word_counter if w}
        self._index = defaultdict(lambda: set())
        self._cho_index = defaultdict(lambda: set())
        self._jung_index = defaultdict(lambda: set())
        self._jong_index = defaultdict(lambda: set())
        
        for word in word_counter:
            # Indexing for levenshtein
            for c in word:
                self._index[c].add(word)
            # Indexing for jamo_levenshtein
            for c in word:
                if not character_is_korean(c):
                    continue
                cho, jung, jong = decompose(c)
                self._cho_index[cho].add(word)
                self._jung_index[jung].add(word)
                self._jong_index[jong].add(word)
                
        self._index = dict(self._index)
        self._cho_index = dict(self._cho_index)
        self._jung_index = dict(self._jung_index)
        self._jong_index = dict(self._jong_index)
    
    def levenshtein_search(self, word, max_distance=1):
        search_time = time()
        candidates = defaultdict(int)
        (n, nc) = (len(word), len(set(word)))
        for c in set(word):
            for item in self._index.get(c, {}):
                candidates[item] += 1
                
        if self.verbose:
            print('query={}, candidates={} '.format(word, len(candidates)), end='')
            
        candidates = {c for c,f in candidates.items() if (abs(n-len(c)) <= max_distance) and (abs(nc - f) <= max_distance)}
        if self.verbose:
            print('-> {}'.format(len(candidates)), end='')
        
        dist = {}
        for c in candidates:
            dist[c] = levenshtein(c, word)
        
        if self.verbose:
            search_time = time() - search_time
            print(', time={:.3} sec.'.format(search_time))
        
        return sorted(filter(lambda x:x[1] <= max_distance, dist.items()), key=lambda x:x[1])
    
    def jamo_levenshtein_search(self, word, max_distance=1):
        search_time = time()
        candidates = defaultdict(lambda: 0)
        (n, nc) = (len(word), len(set(word)))
        for c in set(word):
            if not character_is_korean(c):
                for item in self._index.get(c, {}):
                    candidates[item] += 1
                continue
            cho, jung, jong = decompose(c)
            for item in self._cho_index.get(cho, {}):
                candidates[item] += self.jamo_score
            for item in self._jung_index.get(jung, {}):
                candidates[item] += self.jamo_score
            for item in self._jong_index.get(jong, {}):
                candidates[item] += self.jamo_score
                
        if self.verbose:
            print('query={}, candidates={} '.format(word, len(candidates)), end='')
            
        candidates = {c for c,f in candidates.items() if (abs(n-len(c)) <= max_distance) and (abs(nc - f) <= max_distance)}
        if self.verbose:
            print('-> {}'.format(len(candidates)), end='')
        
        dist = {}
        for c in candidates:
            dist[c] = jamo_levenshtein(c, word)
        
        if self.verbose:
            search_time = time() - search_time
            print(', time={:.3} sec.'.format(search_time))
        
        return sorted(filter(lambda x:x[1] <= max_distance, dist.items()), key=lambda x:x[1])