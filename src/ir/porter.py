"""Porter stemming algorithm, implemented from the published 1980 rules.

Reference: M.F. Porter, "An algorithm for suffix stripping", Program 14(3),
pp. 130-137, 1980.

This is a from-scratch implementation. No stemming library is used here.
NLTK's PorterStemmer appears only in the test suite, as an independent oracle
to measure this implementation against.

A word is treated as a sequence of consonants (C) and vowels (V) and written
[C](VC){m}[V], where m is the "measure" of the word. Most rules only fire when
the surviving stem has a large enough measure, which is what stops the stemmer
from eating short words down to nothing.
"""

VOWELS = "aeiou"


def _is_consonant(word: str, i: int) -> bool:
    """y is the awkward one: a consonant at the start or after a vowel."""
    ch = word[i]
    if ch in VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def measure(stem: str) -> int:
    """Count the VC pairs in [C](VC){m}[V]."""
    n = len(stem)
    i = 0
    while i < n and _is_consonant(stem, i):
        i += 1
    m = 0
    while i < n:
        while i < n and not _is_consonant(stem, i):
            i += 1
        if i >= n:
            break
        m += 1
        while i < n and _is_consonant(stem, i):
            i += 1
    return m


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return (
        len(word) >= 2
        and word[-1] == word[-2]
        and _is_consonant(word, len(word) - 1)
    )


def _ends_cvc(word: str) -> bool:
    """The *o condition: ends consonant-vowel-consonant, last not w, x or y."""
    n = len(word)
    if n < 3:
        return False
    return (
        _is_consonant(word, n - 3)
        and not _is_consonant(word, n - 2)
        and _is_consonant(word, n - 1)
        and word[-1] not in "wxy"
    )


def _step1a(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b_cleanup(stem: str) -> str:
    if stem.endswith(("at", "bl", "iz")):
        return stem + "e"
    if _ends_double_consonant(stem) and not stem.endswith(("l", "s", "z")):
        return stem[:-1]
    if measure(stem) == 1 and _ends_cvc(stem):
        return stem + "e"
    return stem


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        return word[:-1] if measure(word[:-3]) > 0 else word
    if word.endswith("ed") and _contains_vowel(word[:-2]):
        return _step1b_cleanup(word[:-2])
    if word.endswith("ing") and _contains_vowel(word[:-3]):
        return _step1b_cleanup(word[:-3])
    return word


def _step1c(word: str) -> str:
    if word.endswith("y") and _contains_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


# Longest suffixes first where one is a suffix of another.
_STEP2 = [
    ("ational", "ate"), ("tional", "tion"),
    ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"),
    ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
    ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
]

_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"),
    ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
]

_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
    "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
    "ous", "ive", "ize",
]


def _apply_first(word: str, rules: list[tuple[str, str]], min_measure: int) -> str:
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if measure(stem) > min_measure:
                return stem + replacement
            return word
    return word


def _step2(word: str) -> str:
    return _apply_first(word, _STEP2, 0)


def _step3(word: str) -> str:
    return _apply_first(word, _STEP3, 0)


def _step4(word: str) -> str:
    for suffix in _STEP4:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if suffix == "ion":
                if measure(stem) > 1 and stem.endswith(("s", "t")):
                    return stem
                return word
            return stem if measure(stem) > 1 else word
    return word


def _step5a(word: str) -> str:
    if word.endswith("e"):
        stem = word[:-1]
        m = measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            return stem
    return word


def _step5b(word: str) -> str:
    if measure(word) > 1 and _ends_double_consonant(word) and word.endswith("l"):
        return word[:-1]
    return word


def stem(word: str) -> str:
    """Reduce a lowercase word to its Porter stem."""
    word = word.lower()
    if len(word) <= 2:
        return word
    for step in (_step1a, _step1b, _step1c, _step2, _step3, _step4, _step5a, _step5b):
        word = step(word)
    return word


def trace(word: str) -> list[tuple[str, str]]:
    """Return the word after each step, for the report's worked examples."""
    word = word.lower()
    if len(word) <= 2:
        return [("input", word)]
    history = [("input", word)]
    steps = [
        ("1a", _step1a), ("1b", _step1b), ("1c", _step1c),
        ("2", _step2), ("3", _step3), ("4", _step4),
        ("5a", _step5a), ("5b", _step5b),
    ]
    for name, fn in steps:
        word = fn(word)
        history.append((name, word))
    return history
