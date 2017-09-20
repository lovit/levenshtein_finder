from collections import defaultdict
from time import time
from ._string_distance import jamo_levenshtein
from ._string_distance import levenshtein

class LevenshteinIndex:
    def __init__(self, word_counter=None, levenshtein_distance=levenshtein, verbose=False):
        self._words = {}
        self._index = {} # character to words
        if word_counter:
            self.indexing(word_counter)
        self.verbose = verbose
        self._levenshtein_distance = levenshtein_distance
        
    def indexing(self, word_counter):
        self._words = word_counter if type(word_counter) == dict else {w:1 for w in word_counter if w}
        self._index = defaultdict(lambda: set())
        for word in word_counter:
            for c in word:
                self._index[c].add(word)
        self._index = dict(self._index)
    
    def levenshtein_search(self, word, max_distance=1):
        process_time = time()
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
            dist[c] = self._levenshtein_distance(c, word)
        
        if self.verbose:
            process_time = time() - process_time
            print(', time={:.3} sec.'.format(process_time))
        
        return sorted(dist.items(), key=lambda x:x[1])
    
    def jamo_levenshtein_search(self, word):
        raise NotImplemented