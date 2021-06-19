def levenshtein(s1, s2):
    """
    based on Wikipedia/Levenshtein_distance#Python

    Args:
        s1 (list of any)
        s2 (list of any)

    Returns:
        distance (int) : edit distance value

    Examples:
        >>> levenshtein([0, 1, 2, 3], [1, 2, 5])
        $ 2

        >>> levenshtein([0, 1, 2, 3], [0, 1, 2, 5])
        $ 1
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
