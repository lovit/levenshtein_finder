# Similar string search in Levenshtein-distance

## Install

from source
```
git clone https://github.com/lovit/levenshtein_finder
cd levenshtein_finder
python setup.py install
```

from PyPI
```
pip install levenshtein_finder
```

## Usage

### Define Tokenizer

`levenshtein_finder` provides two basic tokenizers.

In [1]
```python
from levenshtein_finder import CharacterTokenizer, BigramTokenizer

char_tokenizer = CharacterTokenizer()
bigram_tokenizer = BigramTokenizer()

char_tokenizer.tokenize("abc de")  # ['a', 'b', 'c', ' ', 'd', 'e']
bigram_tokenizer.tokenize("abc de")  # ['ab', 'bc', 'c ', ' d', 'de']

char_tokenizer.train("path/to/text")  # or
char_tokenizer.train(["list", "of", "words"])
```

When you use word piece tokenizer in Huggingface's [`tokenizers`](https://github.com/huggingface/tokenizers), wrap the class instance with `WordpieceTokenizersWrapper`

In [2]
```
from tokenizers import BertWordPieceTokenizer
from levenshtein_finder import WordpieceTokenizersWrapper

tokenizer = WordpieceTokenizersWrapper(BertWordPieceTokenizer())
tokenizer.train("path/to/text")
tokenizer.tokenize("abc")  # ["ab", "#c"]
tokenizer.encode("abc"  # [53, 2]
tokenizer.decode(tokenizer.tokenize("abc"))  # "abc"
```

Set tokenizer which Levenshtein Finder uses. Default is `CharacterTokenizer`

In [3]
```python
from levenshtein_finder import LevenshteinFinder()

finder = LevenshteinFinder()
finder = LevenshteinFinder(BigramTokenizer())
finder = LevenshteinFinder(WordpieceTokenizersWrapper(BertWordPieceTokenizer()))
```

### Indexing

In [4]
```python
finder = LevenshteinFinder()
finder.indexing("path/to/text")

# when you use CharacterTokenizer or BigramTokenizer, you can index with list of words
finder.indexing(["list", "of", "words"])
```

### Searching

Basic search. The form of `similars` is `dict{term: distance}`.

In [5]
```python
finder.search("query")
```

If you want to check details in searching, set `verbose=True`

In [6]
```python
finder.search("분식회계", verbose=True)
```

Out [6]
```
query               : 분식회계
tokens              : ['분', '식', '회', '계']
num data            : 132864
num 1st candidates  : 10137
num final candidates: 7
num similars        : 4
elapsed time        : 0.00542903 sec
[{'idx': 84243, 'data': '분식회계', 'distance': 0},
 {'idx': 36211, 'data': '분식회', 'distance': 1},
 {'idx': 113942, 'data': '분식회계설', 'distance': 1},
 {'idx': 114122, 'data': '분석회계', 'distance': 1}]
```

### For decomposing Korean to Jamo (NFKD)

To decompose Korean character to jamo components (`감` -> [`ㄱ`, `ㅏ`, `ㅁ`]), use `Normalizers`.

In [7]
```python
normalizer = Normalizers.create_normalizer(unicodedata=True)
tokenizer = CharacterTokenizer(normalizer=normalizer)
finder = LevenshteinFinder(tokenizer=tokenizer)
finder.indexing("text.txt")
finder.search("분식회계", verbose=True)
```

Out [7]
```
query               : 분식회계
tokens              : ['ᄇ', 'ᅮ', 'ᆫ', 'ᄉ', 'ᅵ', 'ᆨ', 'ᄒ', 'ᅬ', 'ᄀ', 'ᅨ']
num data            : 132864
num 1st candidates  : 125925
num final candidates: 4
num similars        : 2
elapsed time        : 0.0863643 sec
[{'idx': 84243, 'data': '분식회계', 'distance': 0},
 {'idx': 114122, 'data': '분석회계', 'distance': 1}]
```

## Demo

### Searching similar words

`LevenshteinFinder` needs ` seconds`.

In [8]
```python
import json
with open("data/nouns_from_financial_news.json") as f:
    words = list(json.load(f).keys())

finder = LevenshteinFinder()
finder.indexing(words)
finder.search("분식회계", verbose=True)
```

Out [8]
```
elapsed time        : 0.00542903 sec
```

Brute-force distance computation needs `1.495 seconds`

In [9]
```python
import time
from levenshtein_finder import levenshtein

query = '분식회계'

begin_time = time.time()
distance = {word:levenshtein(word, query) for word in words}
search_time = time.time() - begin_time
print('search time = {} sec'.format('%.3f'%search_time))

similars = sorted(filter(lambda x:x[1] <= 1, distance.items()), key=lambda x:x[1])
elapsed_time = time.time() - begin_time
print('total elapsed time = {} sec'.format('%.3f'%elapsed_time))
print(similars)
```

Out [9]
```
search time = 1.480 sec
total elapsed time = 1.495 sec
[('분식회계', 0), ('분식회', 1), ('분식회계설', 1), ('분석회계', 1)]
```

### Searching similar strings

Find similar sentence from NSMC corpus (20M).

In [10]
```python
finder = LevenshteinFinder()
finder.indexing("~/Korpora/nsmc/corpus.txt")
finder.search("음악이 주가 된 최고의 음악영화", max_distance=1, verbose=True)
```

Out [10]
```
query               : 음악이 주가 된 최고의 음악영화
tokens              : ['음', '악', '이', ' ', '주', '가', ' ', '된', ' ', '최', '고', '의', ' ', '음', '악', '영', '화']
num data            : 200002
num 1st candidates  : 189663
num final candidates: 1
num similars        : 1
elapsed time        : 0.123322 sec
[{'idx': 6, 'data': '음악이 주가 된, 최고의 음악영화', 'distance': 1}]
```