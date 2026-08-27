"""Tolerant retrieval: k-gram wildcard lookup and edit-distance correction.

A single k-gram index over the vocabulary serves both requirements. Wildcards
are answered by intersecting the k-gram postings of the pattern's literal runs
and then filtering survivors with a regex; misspellings are answered by using
k-gram overlap to shortlist candidates and Levenshtein distance to rank them.

Shortlisting matters. Computing edit distance against all ~30k vocabulary terms
for every query term is wasteful, and a term sharing no 3-grams with the query
is almost never within edit distance 2 of it.
"""

import re
from collections import defaultdict

# Below this collection frequency a candidate is treated as an implausible
# repair target, however close it looks.
MIN_CREDIBLE_FREQUENCY = 3


def levenshtein(a: str, b: str, cap: int | None = None) -> int:
    """Edit distance with the usual two-row optimisation.

    `cap` allows early exit: if every value in a row already exceeds the cap,
    no later row can come back under it, so we abandon and report cap + 1.
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        if cap is not None and min(current) > cap:
            return cap + 1
        previous = current
    return previous[-1]


def levenshtein_matrix(a: str, b: str) -> list[list[int]]:
    """Full DP table, kept for the report and the visual companion."""
    rows = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        rows[i][0] = i
    for j in range(len(b) + 1):
        rows[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            rows[i][j] = min(
                rows[i - 1][j] + 1,
                rows[i][j - 1] + 1,
                rows[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )
    return rows


def kgrams(term: str, k: int = 3) -> list[str]:
    """k-grams of $term$; the sentinels let a k-gram anchor to a word boundary."""
    padded = f"${term}$"
    if len(padded) < k:
        return [padded]
    return [padded[i : i + k] for i in range(len(padded) - k + 1)]


class KGramIndex:
    def __init__(self, terms, k: int = 3) -> None:
        self.k = k
        self._index: dict[str, set[str]] = defaultdict(set)
        self.terms = list(terms)
        for term in self.terms:
            for gram in kgrams(term, k):
                self._index[gram].add(term)

    def __len__(self) -> int:
        return len(self._index)

    def lookup(self, gram: str) -> set[str]:
        return self._index.get(gram, set())

    # ------------------------------------------------------------- wildcards

    def wildcard(self, pattern: str) -> list[str]:
        """Terms matching a pattern containing one or more '*'.

        Two stages, and both are necessary. The k-gram intersection is fast but
        over-generates: 're*val' and 'reval' share k-grams with terms that put
        the pieces in the wrong order. The regex pass removes those.
        """
        if "*" not in pattern:
            return [pattern] if pattern in self._term_set else []

        candidates = self._candidates_for(pattern)
        matcher = re.compile("^" + ".*".join(re.escape(p) for p in pattern.split("*")) + "$")
        return sorted(t for t in candidates if matcher.match(t))

    def _candidates_for(self, pattern: str) -> set[str]:
        padded = f"${pattern}$"
        grams: list[str] = []
        for run in padded.split("*"):
            if len(run) >= self.k:
                grams.extend(
                    run[i : i + self.k] for i in range(len(run) - self.k + 1)
                )
        if not grams:
            return set(self.terms)
        candidates = self.lookup(grams[0])
        for gram in grams[1:]:
            candidates = candidates & self.lookup(gram)
            if not candidates:
                break
        return set(candidates)

    # ------------------------------------------------------- spelling repair

    def nearest(
        self, term: str, max_distance: int = 2, limit: int = 5
    ) -> list[tuple[str, int]]:
        """Vocabulary terms within max_distance, closest first."""
        if term in self._term_set:
            return [(term, 0)]

        shortlist: set[str] = set()
        for gram in kgrams(term, self.k):
            shortlist |= self.lookup(gram)
        if not shortlist:
            shortlist = set(self.terms)

        scored = []
        for candidate in shortlist:
            if abs(len(candidate) - len(term)) > max_distance:
                continue
            d = levenshtein(term, candidate, cap=max_distance)
            if d <= max_distance:
                frequency = self._frequency(candidate)
                # Pure edit distance is frequency-blind, which lets a rare word
                # one edit away beat the common word the user obviously meant
                # ('pirce' -> 'pire' rather than 'price'). Discount candidates
                # the collection barely attests by one edit.
                penalty = 0 if frequency >= MIN_CREDIBLE_FREQUENCY else 1
                scored.append((candidate, d, d + penalty, frequency))
        scored.sort(key=lambda row: (row[2], -row[3], row[0]))
        return [(candidate, d) for candidate, d, _, _ in scored[:limit]]

    def correct(self, term: str, max_distance: int = 2) -> str:
        matches = self.nearest(term, max_distance, limit=1)
        return matches[0][0] if matches else term

    # ------------------------------------------------------------- internals

    @property
    def _term_set(self) -> set[str]:
        if not hasattr(self, "_cached_term_set"):
            self._cached_term_set = set(self.terms)
        return self._cached_term_set

    def _frequency(self, term: str) -> int:
        return getattr(self, "_freq", {}).get(term, 0)

    def attach_frequencies(self, frequencies: dict[str, int]) -> None:
        """Lets ties in edit distance be broken toward the commoner word."""
        self._freq = frequencies
