import re
from unicodedata import normalize as unicode_normalize


class Normalizer:
    def __call__(self, string):
        return self.normalize(string)

    def __repr__(self):
        return "Normalizer"

    def __str__(self):
        return self.__repr__()

    def normalize(self, string):
        raise NotImplementedError("Implement Normalizer.normalize")

    def denormalize(self, string):
        raise NotImplementedError("Implement Normalizer.denormalize")


class Normalizers(Normalizer):
    def __init__(self, normalizers):
        self.normalizers = normalizers

    def __call__(self, string):
        return self.normalize(string)

    def __repr__(self):
        return " -> ".join(str(normalizer) for normalizer in self.normalizers)

    def normalize(self, string):
        if not self.normalizers:
            return string
        for normalizer in self.normalizers:
            string = normalizer(string)
        return string

    def denormalize(self, string):
        if not self.normalizers:
            return string
        for normalizer in self.normalizers:
            string = normalizer.denormalize(string)
        return string

    @classmethod
    def create_normalizer(
        cls,
        unicodedata: bool = False,
        lowercase: bool = False,
        number: bool = False,
        customs=None,
    ):
        normalizers = []
        if unicodedata:
            normalizers.append(UnicodedataNormalizer())
        if lowercase:
            normalizers.append(LowercaseNormalizer())
        if number:
            normalizers.append(NumberNormalizer())
        if customs is not None:
            for normalizer in customs:
                if isinstance(normalizer, Normalizer):
                    normalizers.append(normalizer)
        return Normalizers(normalizers)


class UnicodedataNormalizer(Normalizer):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "UnicodedataNormalizer(NFKD, NFKC)"

    def normalize(self, string):
        return unicode_normalize("NFKD", string)

    def denormalize(self, string):
        return unicode_normalize("NFKC", string)


class LowercaseNormalizer(Normalizer):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "LowercaseNormalizer(Abc > abc)"

    def normalize(self, string):
        return string.lower()

    def denormalize(self, string):
        """Irreversible normalization"""
        return string


class NumberNormalizer(Normalizer):
    def __init__(self):
        super().__init__()
        self.pattern = re.compile(r"[\d]+")

    def __repr__(self):
        return "NumberNormalizer(123 > [NUM])"

    def normalize(self, string):
        return self.pattern.sub("1", string)

    def denormalize(self, string):
        """Irreversible normalization"""
        return string.replace("1", "[NUM]")
