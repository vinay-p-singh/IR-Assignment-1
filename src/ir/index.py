"""Inverted index construction and sorted postings lists."""

from collections import defaultdict

from .corpus import Document
from .dictionary import Vocabulary
from .preprocess import Preprocessor


class InvertedIndex:
    """Maps each term to the ascending list of doc IDs that contain it.

    Postings are stored sorted and duplicate-free. That single invariant is what
    lets the Boolean merges in boolean.py run in linear time with one pass over
    each list; break it and every merge silently degrades or returns nonsense.
    """

    def __init__(
        self,
        name: str,
        postings: dict[str, list[int]],
        vocabulary: Vocabulary,
        num_docs: int,
    ) -> None:
        self.name = name
        self._postings = postings
        self.vocabulary = vocabulary
        self.num_docs = num_docs
        self.all_doc_ids: list[int] = list(range(num_docs))

    @classmethod
    def build(
        cls, docs: list[Document], preprocessor: Preprocessor
    ) -> "InvertedIndex":
        doc_sets: dict[str, set[int]] = defaultdict(set)
        collection_freq: dict[str, int] = defaultdict(int)

        for doc in docs:
            for token in preprocessor(doc.text):
                doc_sets[token].add(doc.doc_id)
                collection_freq[token] += 1

        postings = {term: sorted(ids) for term, ids in doc_sets.items()}
        terms = tuple(sorted(postings))
        vocabulary = Vocabulary(
            terms=terms,
            document_frequency={t: len(postings[t]) for t in terms},
            collection_frequency=dict(collection_freq),
        )
        return cls(preprocessor.name, postings, vocabulary, len(docs))

    def postings(self, term: str) -> list[int]:
        return self._postings.get(term, [])

    def df(self, term: str) -> int:
        return self.vocabulary.df(term)

    def __contains__(self, term: str) -> bool:
        return term in self._postings

    @property
    def terms(self) -> tuple[str, ...]:
        return self.vocabulary.terms

    def stats(self) -> dict:
        return {"index": self.name, "num_docs": self.num_docs, **self.vocabulary.stats()}

    def sample_postings(self, term: str, limit: int = 10) -> str:
        """One-line rendering for the report's sample postings list."""
        ids = self.postings(term)
        shown = ", ".join(str(i) for i in ids[:limit])
        tail = f", ... (+{len(ids) - limit} more)" if len(ids) > limit else ""
        return f"{term} -> df={len(ids)} -> [{shown}{tail}]"

    def check_invariants(self) -> None:
        """Raise if any postings list is unsorted, duplicated, or out of range."""
        for term, ids in self._postings.items():
            if any(b <= a for a, b in zip(ids, ids[1:])):
                raise AssertionError(f"postings for {term!r} not strictly ascending")
            if ids and (ids[0] < 0 or ids[-1] >= self.num_docs):
                raise AssertionError(f"postings for {term!r} out of doc-id range")
            if len(ids) != self.vocabulary.df(term):
                raise AssertionError(f"df mismatch for {term!r}")
