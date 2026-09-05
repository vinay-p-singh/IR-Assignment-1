"""Probes that expose where stemming and lemmatization disagree.

Assignment section 4.2 asks for undesirable transformations, not just a count of
how much each method conflates. Two named failure modes are measured here:

    over-stemming   two words with unrelated senses collapse to one stem, so a
                    query for one of them retrieves documents about the other
    under-stemming  two surface forms of the same word survive as separate
                    terms, so a query for one of them misses the other

The probe pairs below are chosen by hand; the verdicts are not. Every row in the
output is produced by running the real stemmer and the real lemmatizer over the
pair, so the table changes if the implementation changes.
"""

from dataclasses import dataclass

from .preprocess import _cached_stem, context_free_lemma


@dataclass(frozen=True)
class Probe:
    a: str
    b: str
    related: bool  # do the two words share a sense a searcher would want merged?
    note: str


PROBES: list[Probe] = [
    # --- pairs a searcher would NOT want merged; merging them is over-stemming
    Probe("communism", "community", False, "political system vs social group"),
    Probe("organisation", "organ", False, "institution vs body part"),
    Probe("university", "universe", False, "institution vs cosmos"),
    Probe("generic", "generation", False, "unspecific vs a cohort of people"),
    Probe("dominance", "dominic", False, "abstract noun vs a personal name"),
    Probe("news", "new", False, "reports vs the adjective"),
    Probe("policy", "police", False, "a plan vs a law-enforcement body"),
    Probe("marketing", "market", True, "same commercial sense; merging is wanted"),
    # --- pairs a searcher WOULD want merged; not merging them is under-stemming
    Probe("mice", "mouse", True, "irregular plural"),
    Probe("ran", "run", True, "irregular past tense"),
    Probe("better", "good", True, "suppletive comparative"),
    Probe("was", "be", True, "irregular copula"),
    Probe("children", "child", True, "irregular plural"),
    Probe("europe", "european", True, "place and its adjective"),
    Probe("analysis", "analyses", True, "Greek-origin plural"),
    Probe("company", "companies", True, "regular y-plural"),
]


def _verdict(probe: Probe, same_stem: bool, same_lemma: bool) -> str:
    if probe.related:
        if same_stem and same_lemma:
            return "both merge (wanted)"
        if same_stem:
            return "stemmer merges, lemmatizer does not"
        if same_lemma:
            return "under-stemming: only the lemmatizer merges"
        return "under-stemming: neither merges"
    if same_stem and same_lemma:
        return "both over-merge"
    if same_stem:
        return "over-stemming: only the stemmer merges"
    if same_lemma:
        return "over-lemmatizing: only the lemmatizer merges"
    return "both keep apart (wanted)"


def probe_rows() -> list[dict]:
    rows = []
    for probe in PROBES:
        stem_a, stem_b = _cached_stem(probe.a), _cached_stem(probe.b)
        lemma_a, lemma_b = context_free_lemma(probe.a), context_free_lemma(probe.b)
        same_stem = stem_a == stem_b
        same_lemma = lemma_a == lemma_b
        rows.append(
            {
                "word_a": probe.a,
                "word_b": probe.b,
                "should_merge": "yes" if probe.related else "no",
                "stem_a": stem_a,
                "stem_b": stem_b,
                "same_stem": "yes" if same_stem else "no",
                "lemma_a": lemma_a,
                "lemma_b": lemma_b,
                "same_lemma": "yes" if same_lemma else "no",
                "verdict": _verdict(probe, same_stem, same_lemma),
                "note": probe.note,
            }
        )
    return rows


def undesirable(rows: list[dict]) -> list[dict]:
    """The subset a reader should look at: every row where a method got it wrong."""
    return [r for r in rows if "wanted" not in r["verdict"]]
