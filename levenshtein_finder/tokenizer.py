from typing import List, Union


class Tokenizer:
    def __init__(self):
        pass

    @property
    def is_trained(self):
        raise NotImplementedError("Implement Tokenizer.is_trained")

    def train(self, strings: Union[str, List[str]]):
        raise NotImplementedError("Implement Tokenizer.train")

    def save_model(self, directory: str, prefix: str = None):
        raise NotImplementedError("Implement Tokenizer.save_model")

    def tokenize(self, string: str) -> List[str]:
        raise NotImplementedError("Implement Tokenizer.tokenize")

    def encode(self, string: str) -> List[int]:
        raise NotImplementedError("Implement Tokenizer.encode")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        raise NotImplementedError("Implement Tokenizer.convert_tokens_to_ids")


class HuggingfaceTokenizersWrapper(Tokenizer):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    @property
    def is_trained(self):
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

    def save_model(self, directory: str, prefix: str = None):
        self.tokenizer.save_model(directory=directory, prefix=prefix)

    def tokenize(self, string):
        return self.tokenizer.encode(string).tokens

    def encode(self, string):
        return self.tokenizer.encode(string).ids

    def convert_tokens_to_ids(self, tokens):
        return [self.tokenizer.token_to_id(token) for token in tokens]


class CharacterTokenizer(Tokenizer):
    def __init__(self, vocab=None):
        if vocab is None:
            vocab = []
        self.vocab = vocab
        self.vocab_to_idx = {v: idx for idx, v in enumerate(vocab)}
        if vocab:
            self.unk_ids = len(vocab) + 1
        else:
            self.unk_ids = 0

    @property
    def is_trained(self):
        return len(self.vocab) > 0

    def train(self, strings: List[str]):
        charset = {char for string in strings for char in string}
        vocab = sorted(charset)
        vocab_to_idx = {v: idx for idx, v in enumerate(vocab)}
        self.vocab = vocab
        self.vocab_to_idx = vocab_to_idx
        self.unk_ids = len(vocab) + 1

    def tokenize(self, string):
        return list(string)

    def encode(self, string):
        return self.convert_tokens_to_ids(self.tokenize(string))

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab_to_idx.get(token, self.unk_ids) for token in tokens]
