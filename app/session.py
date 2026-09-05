"""Shared, lazily-built IR state for the chat app.

The chat app is a thin shell over src/ir. Nothing here re-implements retrieval;
it only holds the objects that are expensive to construct (corpus, indexes,
k-gram indexes, qrels) so a chat turn stays interactive.

Indexes are built on first use rather than at boot. Building all five costs tens
of seconds and most sessions only ever touch one.
"""

import pathlib

from ir import boolean, preprocess, queryset, tolerant
from ir.corpus import corpus_stats, load_corpus
from ir.index import InvertedIndex

from . import ROOT

CORPUS_ROOT = ROOT / "data" / "raw" / "bbc"
DEFAULT_INDEX = "stemmed"


class Session:
    def __init__(self, corpus_root: pathlib.Path = CORPUS_ROOT) -> None:
        self.docs = load_corpus(corpus_root)
        self.by_id = {d.doc_id: d for d in self.docs}
        self.corpus_stats = corpus_stats(self.docs)
        self.active = DEFAULT_INDEX
        self._indexes: dict[str, InvertedIndex] = {}
        self._kgrams: dict[str, tolerant.KGramIndex] = {}
        self._qrels: dict[str, set[int]] | None = None

    @property
    def config_names(self) -> list[str]:
        return [c.name for c in preprocess.CONFIGURATIONS]

    def index(self, name: str | None = None) -> InvertedIndex:
        name = name or self.active
        if name not in preprocess.BY_NAME:
            raise KeyError(name)
        if name not in self._indexes:
            index = InvertedIndex.build(self.docs, preprocess.BY_NAME[name])
            index.check_invariants()
            self._indexes[name] = index
        return self._indexes[name]

    def is_built(self, name: str) -> bool:
        return name in self._indexes

    def normalizer(self, name: str | None = None) -> boolean.TermNormalizer:
        return boolean.normalizer_for(preprocess.BY_NAME[name or self.active])

    def kgram(self, name: str | None = None) -> tolerant.KGramIndex:
        name = name or self.active
        if name not in self._kgrams:
            index = self.index(name)
            kg = tolerant.KGramIndex(index.terms, k=3)
            kg.attach_frequencies(index.vocabulary.collection_frequency)
            self._kgrams[name] = kg
        return self._kgrams[name]

    @property
    def qrels(self) -> dict[str, set[int]]:
        if self._qrels is None:
            self._qrels = queryset.build_qrels(self.docs)
        return self._qrels

    def document(self, doc_id: int):
        return self.by_id.get(doc_id)

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.docs:
            counts[d.category] = counts.get(d.category, 0) + 1
        return counts
