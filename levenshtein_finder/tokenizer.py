import os
from typing import List, Union
from unicodedata import normalize
from .normalizer import Normalizers


class Tokenizer:
    def __init__(self):
        self.vocab_size = 0
        pass

    def __len__(self):
        return self.vocab_size

    @property
    def is_trained(self):
        raise NotImplementedError("Implement Tokenizer.is_trained")

    def train(self, strings: Union[str, List[str]]):
        raise NotImplementedError("Implement Tokenizer.train")

    def save_model(self, directory: str, prefix: str = None):
        raise NotImplementedError("Implement Tokenizer.save_model")

    def tokenize(self, string: str) -> List[str]:
        raise NotImplementedError("Implement Tokenizer.tokenize")

    def detokenize(self, tokens: List[str]) -> str:
        raise NotImplementedError("Implement Tokenizer.detokenize")

    def encode(self, string: str) -> List[int]:
        raise NotImplementedError("Implement Tokenizer.encode")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        raise NotImplementedError("Implement Tokenizer.convert_tokens_to_ids")


class WordpieceTokenizersWrapper(Tokenizer):
    """
    Args:
        tokenizer (tokenizers.BertWordPieceTokenizer)
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    @property
    def is_trained(self) -> bool:
        return self.tokenizer.get_vocab_size() > 0

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        special_tokens: List[str] = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
        ],
        show_progress: bool = True,
        wordpieces_prefix: str = "##",
    ):
        if not isinstance(files, str) or not os.path.isfile(files):
            raise ValueError(
                "`WordpieceTokenizersWrapper.train` input argument must be file path"
            )
        self.tokenizer.train(
            files=files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            special_tokens=special_tokens,
            show_progress=show_progress,
            wordpieces_prefix=wordpieces_prefix,
        )
        self.vocab_size = self.tokenizer.get_vocab_size() + 1

    def save_model(self, directory: str, prefix: str = None):
        self.tokenizer.save_model(directory=directory, prefix=prefix)

    def tokenize(self, string: str) -> List[str]:
        return self.tokenizer.encode(string).tokens

    def detokenize(self, tokens: List[str]) -> str:
        prefix = self.tokenizer._parameters["wordpieces_prefix"]
        n = len(prefix)
        detok = []
        for token in tokens:
            if token[:n] == prefix:
                detok.append(token[n:])
            else:
                detok.append(f" {token}")
        return normalize("NFKC", "".join(detok).strip())

    def encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string).ids

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.tokenizer.token_to_id(token) for token in tokens]


class CharacterTokenizer(Tokenizer):
    def __init__(self, vocab=None, normalizer=None):
        if vocab is None:
            vocab = []
        self.vocab = vocab
        self.vocab_to_idx = {v: idx for idx, v in enumerate(vocab)}
        if vocab:
            self.unk_ids = len(vocab) + 1
        else:
            self.unk_ids = 0
        if normalizer is None:
            normalizer = Normalizers.create_normalizer()
        if not callable(normalizer):
            raise ValueError("Normalizer must be `callable` or None")
        self.normalizer = normalizer

    @property
    def is_trained(self) -> bool:
        return len(self.vocab) > 0

    def train(self, strings: List[str]):
        if isinstance(strings, str) and os.path.isfile(strings):
            with open(strings, encoding="utf-8") as f:
                strings = [line.strip() for line in f]
        charset = {char for string in strings for char in self.normalizer(string)}
        vocab = sorted(charset)
        vocab_to_idx = {v: idx for idx, v in enumerate(vocab)}
        self.vocab = vocab
        self.vocab_to_idx = vocab_to_idx
        self.unk_ids = len(vocab) + 1
        self.vocab_size = len(vocab) + 1

    def tokenize(self, string: str) -> List[str]:
        return list(self.normalizer(string))

    def detokenize(self, tokens: List[str]) -> str:
        return self.normalizer.denormalize("".join(tokens))

    def encode(self, string: str) -> List[int]:
        return self.convert_tokens_to_ids(self.tokenize(string))

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab_to_idx.get(token, self.unk_ids) for token in tokens]


class BigramTokenizer(Tokenizer):
    def __init__(self, vocab=None, normalizer=None):
        if vocab is None:
            vocab = []
        self.vocab = vocab
        self.vocab_to_idx = {v: idx for idx, v in enumerate(vocab)}
        if vocab:
            self.unk_ids = len(vocab) + 1
        else:
            self.unk_ids = 0
        if normalizer is None:
            normalizer = Normalizers.create_normalizer()
        if not callable(normalizer):
            raise ValueError("Normalizer must be `callable` or None")
        self.normalizer = normalizer

    @property
    def is_trained(self) -> bool:
        return len(self.vocab) > 0

    def _to_bigram(self, string):
        n = max(len(string), 2)
        return [string[i : i + 2] for i in range(n - 1)]

    def train(self, strings: List[str]):
        if isinstance(strings, str) and os.path.isfile(strings):
            with open(strings, encoding="utf-8") as f:
                strings = [line.strip() for line in f]
        strings = [self.normalizer(string) for string in strings]
        bigrams = {bigram for string in strings for bigram in self._to_bigram(string)}
        vocab = sorted(bigrams)
        vocab_to_idx = {v: idx for idx, v in enumerate(vocab)}
        self.vocab = vocab
        self.vocab_to_idx = vocab_to_idx
        self.unk_ids = len(vocab) + 1
        self.vocab_size = len(vocab) + 1

    def tokenize(self, string: str) -> List[str]:
        return self._to_bigram(self.normalizer(string))

    def detokenize(self, tokens: List[str]) -> str:
        string = "".join(token[0] for token in tokens[:-1]) + tokens[-1]
        return self.normalizer.denormalize(string)

    def encode(self, string: str) -> List[int]:
        return self.convert_tokens_to_ids(self.tokenize(string))

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab_to_idx.get(token, self.unk_ids) for token in tokens]
