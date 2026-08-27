"""The 30 evaluation queries, their relevance rules, and 20 misspelled variants.

Relevance judgments are defined as explicit predicates over the *raw* article
text, using whole-word matching. Retrieval, by contrast, runs over a *processed*
index. Keeping those two on different footings is what makes the evaluation
informative: a stemmed index can find 'investors' for the query 'investment'
and be rewarded for it, while an unstemmed index cannot and is penalised.

The obvious limitation, which the report must state: these judgments are
lexical, not semantic. An article genuinely about the oil market that never
writes the word 'oil' is counted as non-relevant. That inflates precision and
deflates recall relative to human judgment, uniformly across all systems.
"""

import re
from dataclasses import dataclass
from typing import Callable

Predicate = Callable[[str], bool]


def w(word: str) -> Predicate:
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    return lambda text: pattern.search(text) is not None


def all_of(*predicates: Predicate) -> Predicate:
    return lambda text: all(p(text) for p in predicates)


def any_of(*predicates: Predicate) -> Predicate:
    return lambda text: any(p(text) for p in predicates)


def none_of(predicate: Predicate) -> Predicate:
    return lambda text: not predicate(text)


def word_family(*words: str) -> Predicate:
    return any_of(*(w(word) for word in words))


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    kind: str          # simple | boolean | morphological | negation
    gold: Predicate


@dataclass(frozen=True)
class MisspelledQuery:
    qid: str
    text: str
    base_qid: str
    corruption: str    # transposition | omission | doubling | substitution


QUERIES: list[Query] = [
    # ---------------------------------------------------------- simple terms
    Query("q01", "oil", "simple", w("oil")),
    Query("q02", "election", "simple", w("election")),
    Query("q03", "broadband", "simple", w("broadband")),
    Query("q04", "chelsea", "simple", w("chelsea")),
    Query("q05", "oscar", "simple", w("oscar")),
    Query("q06", "economy", "simple", w("economy")),
    Query("q07", "software", "simple", w("software")),
    Query("q08", "album", "simple", w("album")),
    # -------------------------------------------------------- boolean shapes
    Query("q09", "oil AND price", "boolean", all_of(w("oil"), w("price"))),
    Query("q10", "mobile AND phone", "boolean", all_of(w("mobile"), w("phone"))),
    Query("q11", "football OR rugby", "boolean", any_of(w("football"), w("rugby"))),
    Query(
        "q12",
        "(oil OR gas) AND prices",
        "boolean",
        all_of(any_of(w("oil"), w("gas")), w("prices")),
    ),
    Query(
        "q13",
        "government AND economy",
        "boolean",
        all_of(w("government"), w("economy")),
    ),
    Query(
        "q14",
        "(microsoft OR apple) AND software",
        "boolean",
        all_of(any_of(w("microsoft"), w("apple")), w("software")),
    ),
    Query("q15", "shares AND profits", "boolean", all_of(w("shares"), w("profits"))),
    Query("q16", "election AND labour", "boolean", all_of(w("election"), w("labour"))),
    Query(
        "q17",
        "(chelsea OR arsenal) AND league",
        "boolean",
        all_of(any_of(w("chelsea"), w("arsenal")), w("league")),
    ),
    Query("q18", "digital AND music", "boolean", all_of(w("digital"), w("music"))),
    Query(
        "q19",
        "internet AND security",
        "boolean",
        all_of(w("internet"), w("security")),
    ),
    Query("q20", "film AND awards", "boolean", all_of(w("film"), w("awards"))),
    # ------------------------------------------------------------- negations
    Query(
        "q21",
        "bank AND NOT football",
        "negation",
        all_of(w("bank"), none_of(w("football"))),
    ),
    Query(
        "q22",
        "music AND NOT film",
        "negation",
        all_of(w("music"), none_of(w("film"))),
    ),
    Query(
        "q23",
        "minister AND NOT chelsea",
        "negation",
        all_of(w("minister"), none_of(w("chelsea"))),
    ),
    # --------------------------------------------------------- morphological
    # Gold covers the whole surface family, so only a stemming or lemmatising
    # index can reach full recall. This is the point of these seven.
    Query(
        "q24",
        "investment",
        "morphological",
        word_family("invest", "invests", "invested", "investing", "investment",
                    "investments"),
    ),
    Query(
        "q25",
        "manager",
        "morphological",
        word_family("manage", "manages", "managed", "managing", "manager",
                    "managers", "management"),
    ),
    Query(
        "q26",
        "technology",
        "morphological",
        word_family("technology", "technologies", "technological"),
    ),
    Query(
        "q27",
        "players",
        "morphological",
        word_family("play", "plays", "played", "playing", "player", "players"),
    ),
    Query(
        "q28",
        "announced",
        "morphological",
        word_family("announce", "announces", "announced", "announcing",
                    "announcement", "announcements"),
    ),
    Query(
        "q29",
        "computing",
        "morphological",
        word_family("compute", "computes", "computed", "computing", "computer",
                    "computers"),
    ),
    Query(
        "q30",
        "winning",
        "morphological",
        word_family("win", "wins", "winning", "winner", "winners"),
    ),
]


MISSPELLED: list[MisspelledQuery] = [
    MisspelledQuery("m01", "oill", "q01", "doubling"),
    MisspelledQuery("m02", "electon", "q02", "omission"),
    MisspelledQuery("m03", "brodband", "q03", "omission"),
    MisspelledQuery("m04", "chelsae", "q04", "transposition"),
    MisspelledQuery("m05", "oscer", "q05", "substitution"),
    MisspelledQuery("m06", "ecnomy", "q06", "omission"),
    MisspelledQuery("m07", "softwear", "q07", "substitution"),
    MisspelledQuery("m08", "albbum", "q08", "doubling"),
    MisspelledQuery("m09", "oil AND pirce", "q09", "transposition"),
    MisspelledQuery("m10", "moblie AND phone", "q10", "transposition"),
    MisspelledQuery("m11", "futball OR rugby", "q11", "substitution"),
    MisspelledQuery("m12", "(oil OR gass) AND prices", "q12", "doubling"),
    MisspelledQuery("m13", "goverment AND economy", "q13", "omission"),
    MisspelledQuery("m14", "(microsft OR apple) AND software", "q14", "omission"),
    MisspelledQuery("m15", "sahres AND profits", "q15", "transposition"),
    MisspelledQuery("m16", "election AND labuor", "q16", "transposition"),
    MisspelledQuery("m17", "digital AND musci", "q18", "transposition"),
    MisspelledQuery("m18", "internett AND security", "q19", "doubling"),
    MisspelledQuery("m19", "film AND awrds", "q20", "omission"),
    MisspelledQuery("m20", "managment", "q25", "omission"),
]

BY_QID = {q.qid: q for q in QUERIES}


def build_qrels(documents) -> dict[str, set[int]]:
    """Apply every gold predicate to every document. This is the qrels file."""
    return {
        query.qid: {d.doc_id for d in documents if query.gold(d.text)}
        for query in QUERIES
    }
