import os
import time
from array import array
from collections import defaultdict

from .distance import levenshtein
from .tokenizer import CharacterTokenizer


class LevenshteinFinder:
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = CharacterTokenizer()
        self.tokenizer = tokenizer

    @property
    def is_trained(self):
        return hasattr(self, "data") and len(getattr(self, "data")) > 0

    def indexing(self, strings, pretokenized=False):
        if not self.tokenizer.is_trained:
            self.tokenizer.train(strings)

        if isinstance(strings, str) and os.path.isfile(strings):
            with open(strings, encoding="utf-8") as f:
                strings = [line.strip() for line in f]

        # prepare tokens
        if not pretokenized:
            tokens = [self.tokenizer.tokenize(string) for string in strings]
        else:
            tokens = [string.split() for string in strings]
        token_ids = [
            self.tokenizer.convert_tokens_to_ids(tokens_in_string)
            for tokens_in_string in tokens
        ]

        # create inverted index of length
        lengths = [len(tokens_in_string) for tokens_in_string in tokens]
        self.lengths = lengths

        # create inverted index of tokens
        inverted_index = [[] for _ in range(len(self.tokenizer))]
        for string_idx, (string, token_ids_in_string) in enumerate(
            zip(strings, token_ids)
        ):
            for idx in token_ids_in_string:
                inverted_index[idx].append(string_idx)
        inverted_index = [
            array("I", set(string_indices)) for string_indices in inverted_index
        ]
        self.inverted_index = inverted_index
        self.token_ids = [
            array("I", token_ids_in_string) for token_ids_in_string in token_ids
        ]
        self.data = strings

    def search(self, query, max_distance=1, pretokenized=False, verbose=False):
        t = time.time()
        # convert query to token indices
        if not pretokenized:
            tokens = self.tokenizer.tokenize(query)
        else:
            tokens = query.split()
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        unique_token_ids = set(token_ids)

        # token-matching
        candidates = defaultdict(int)
        for unique_token_id in set(token_ids):
            for string_idx in self.inverted_index[unique_token_id]:
                candidates[string_idx] += 1
        n_token_matched = len(candidates)

        # filtering using num unique token & length
        n = len(token_ids)
        len_min, len_max = n - max_distance, n + max_distance
        unique_n = len(unique_token_ids)
        candidates = {
            string_idx: count
            for string_idx, count in candidates.items()
            if (
                (len_min <= self.lengths[string_idx] <= len_max)
                and (abs(count - unique_n) <= max_distance)
            )
        }
        n_filtered = len(candidates)

        # calculate levenshtein distance
        distances = [
            levenshtein(self.token_ids[string_idx], token_ids)
            for string_idx in candidates
        ]
        similars = [
            {"idx": string_idx, "data": self.data[string_idx], "distance": distance}
            for string_idx, distance in zip(candidates, distances)
            if distance <= max_distance
        ]
        similars = sorted(similars, key=lambda x: (x["distance"], x["idx"]))
        t = time.time() - t

        if verbose:
            print(f"query               : {query}")
            print(f"tokens              : {tokens}")
            print(f"num data            : {len(self.data)}")
            print(f"num 1st candidates  : {n_token_matched}")
            print(f"num final candidates: {n_filtered}")
            print(f"num similars        : {len(similars)}")
            print(f"elapsed time        : {t:.6} sec")

        return similars
