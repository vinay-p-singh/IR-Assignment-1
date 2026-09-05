"""Text processing: tokenization, normalization, stop-words, stemming, lemmas.

Five preprocessing configurations are exposed so the report can compare them
directly. They are cumulative:

    raw        -> tokenize only, case preserved
    normalized -> + lowercase
    nostop     -> + stop-word removal
    stemmed    -> + Porter stemming (our own implementation)
    lemmatized -> normalized + POS-tagged WordNet lemmatization + stop-word removal

Library boundary: the stop-word list, the POS tagger and the WordNet lemmatizer
come from NLTK. Tokenization and stemming are ours.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from . import porter

# Keeps intra-word apostrophes and hyphens so "don't" and "Netherlands-based"
# survive as single tokens; splits everything else on non-word characters.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")

_lemmatizer = WordNetLemmatizer()

# WordNet only knows four parts of speech; everything else is treated as a noun.
_PENN_TO_WORDNET = {"J": "a", "V": "v", "N": "n", "R": "r"}
# Verb first for context-free lookups only. See context_free_lemma.
_WORDNET_POS = ("v", "n", "a", "r")


@lru_cache(maxsize=1)
def stop_words() -> frozenset[str]:
    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=200_000)
def _cached_stem(token: str) -> str:
    return porter.stem(token)


@lru_cache(maxsize=400_000)
def _cached_lemma(token: str, pos: str = "n") -> str:
    return _lemmatizer.lemmatize(token, pos)


@lru_cache(maxsize=200_000)
def context_free_lemma(token: str) -> str:
    """Best-effort lemma for a vocabulary term seen without a sentence around it.

    WordNet needs a part of speech and an isolated term has none. morphy is used
    rather than the lemmatizer because it reports whether the token was found at
    all: the lemmatizer returns the input unchanged both for a word it already
    recognises ('news') and for one it has never seen, and those two cases have
    to be told apart.

    Verbs are tried before nouns. The noun rule is the one that mis-fires on an
    isolated form -- it strips the final 's' of 'was' and lands on 'wa', an
    ethnic group WordNet really does list -- while the verb paradigm carries the
    irregulars worth catching, was/be and ran/run among them. The cost is that
    a deverbal noun such as 'meeting' is reported as 'meet'. Document text is
    not affected either way; it goes through lemmatize_tokens, which has a real
    tag from a real sentence.
    """
    for pos in _WORDNET_POS:
        found = wordnet.morphy(token, pos)
        if found:
            return found
    return token


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """POS-tag the sequence, then lemmatize each token under its own tag.

    Tagging is what makes this different from stemming. Without it NLTK assumes
    every word is a noun, so 'announced' and 'winning' come back untouched and
    the lemmatized index collapses into little more than plural-stripping.
    """
    return [
        _cached_lemma(token, _PENN_TO_WORDNET.get(tag[:1], "n"))
        for token, tag in pos_tag(tokens)
    ]


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
        if self.lemmatize:
            # Runs before stop-word removal because the tagger needs the
            # function words around a token to decide its part of speech.
            tokens = lemmatize_tokens(tokens)
        if self.remove_stopwords:
            stops = stop_words()
            tokens = [t for t in tokens if t.lower() not in stops]
        if self.stem:
            tokens = [_cached_stem(t) for t in tokens]
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
