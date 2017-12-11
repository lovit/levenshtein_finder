# 빠른 한글 수정 거리 검색을 위한 inverted index

### Indexing

    from fast_hangle_levenshtein import LevenshteinIndex
    indexer = LevenshteinIndex(verbose=True)

    indexer.indexing('아이고 어이고 아이고야 아이고야야야야 어이구야 지화자 징화자 쟝화장'.split())

### Searching

1. Levenshtein distance

        indexer.levenshtein_search('아이코', max_distance=1)
    
1. Jamo Levenshtein distance

        indexer.jamo_levenshtein_search('아이코', max_distance=4/3)
