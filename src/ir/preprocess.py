"""Text processing: tokenization, normalization, stop-words, stemming, lemmas.

Five preprocessing configurations are exposed so the report can compare them
directly. They are cumulative:

    raw        -> tokenize only, case preserved
    normalized -> + lowercase
    nostop     -> + stop-word removal
    stemmed    -> + Porter stemming (our own implementation)
    lemmatized -> normalized + stop-word removal + WordNet lemmatization

Library boundary: the stop-word list and the WordNet lemmatizer come from NLTK.
Tokenization and stemming are ours.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from . import porter

# Keeps intra-word apostrophes and hyphens so "don't" and "Netherlands-based"
# survive as single tokens; splits everything else on non-word characters.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")

_lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=1)
def stop_words() -> frozenset[str]:
    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=200_000)
def _cached_stem(token: str) -> str:
    return porter.stem(token)


@lru_cache(maxsize=200_000)
def _cached_lemma(token: str) -> str:
    return _lemmatizer.lemmatize(token)


def tokenize(text: str) -> list[str]:
    """Split raw text into word tokens. No case folding at this stage."""
    return _TOKEN_RE.findall(text)


@dataclass(frozen=True)
class Preprocessor:
    """A named, reproducible preprocessing configuration."""

    name: str
    lowercase: bool = False
    remove_stopwords: bool = False
    stem: bool = False
    lemmatize: bool = False

    def __call__(self, text: str) -> list[str]:
        tokens = tokenize(text)
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        if self.remove_stopwords:
            stops = stop_words()
            tokens = [t for t in tokens if t.lower() not in stops]
        if self.stem:
            tokens = [_cached_stem(t) for t in tokens]
        if self.lemmatize:
            tokens = [_cached_lemma(t) for t in tokens]
        return tokens


RAW = Preprocessor("raw")
NORMALIZED = Preprocessor("normalized", lowercase=True)
NOSTOP = Preprocessor("nostop", lowercase=True, remove_stopwords=True)
STEMMED = Preprocessor("stemmed", lowercase=True, remove_stopwords=True, stem=True)
LEMMATIZED = Preprocessor(
    "lemmatized", lowercase=True, remove_stopwords=True, lemmatize=True
)

CONFIGURATIONS = [RAW, NORMALIZED, NOSTOP, STEMMED, LEMMATIZED]
BY_NAME = {p.name: p for p in CONFIGURATIONS}
