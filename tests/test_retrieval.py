import pathlib

import pytest

from ir import boolean, evaluate, morphology, preprocess, tolerant
from ir.corpus import load_corpus
from ir.index import InvertedIndex
from ir.queryset import MISSPELLED, QUERIES, build_qrels

ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "bbc"


@pytest.fixture(scope="module")
def docs():
    return load_corpus(ROOT)


@pytest.fixture(scope="module")
def stemmed(docs):
    return InvertedIndex.build(docs, preprocess.STEMMED)


@pytest.fixture(scope="module")
def normalize():
    return boolean.normalizer_for(preprocess.STEMMED)


# ------------------------------------------------------------- preprocessing


def test_tokenizer_edge_cases():
    assert preprocess.tokenize("don't stop") == ["don't", "stop"]
    assert preprocess.tokenize("Netherlands-based firm") == [
        "Netherlands-based",
        "firm",
    ]
    assert preprocess.tokenize("$4.5bn and £16.7bn") == ["4", "5bn", "and", "16", "7bn"]
    assert preprocess.tokenize("U.S. investors") == ["U", "S", "investors"]
    assert preprocess.tokenize("45,000 jobs") == ["45", "000", "jobs"]
    assert preprocess.tokenize("") == []


def test_configurations_are_cumulative():
    text = "The Investors were INVESTING heavily"
    assert "The" in preprocess.RAW(text)
    assert "the" in preprocess.NORMALIZED(text)
    assert "the" not in preprocess.NOSTOP(text)
    assert preprocess.STEMMED(text) == ["investor", "invest", "heavili"]


def test_case_normalization_collides_us_with_stopword():
    """A real, reportable precision loss.

    Lowercasing maps the country 'US' onto the pronoun 'us'. NLTK's English
    stop-word list does not contain 'us', so the collision is not even hidden by
    stop-word removal -- both senses survive merged into a single postings list.
    A query for the country therefore retrieves articles that merely used the
    pronoun.
    """
    assert "us" not in preprocess.stop_words()
    assert preprocess.NOSTOP("US investors") == ["us", "investors"]
    assert preprocess.NOSTOP("between us") == ["us"]


def test_lemmatizer_is_pos_aware():
    """Without a tag NLTK assumes every word is a noun and verbs pass through.

    That default is what makes an untagged 'lemmatized' index little more than
    plural stripping, and it would make the stemming-vs-lemmatization
    comparison meaningless.
    """
    assert preprocess._cached_lemma("announced") == "announced"
    assert preprocess._cached_lemma("announced", "v") == "announce"
    tagged = preprocess.lemmatize_tokens(
        ["they", "announced", "the", "bids"]
    )
    assert tagged == ["they", "announce", "the", "bid"]


def test_lemmatizer_keeps_irregular_forms_the_stemmer_misses():
    assert preprocess.context_free_lemma("children") == "child"
    assert preprocess.context_free_lemma("mice") == "mouse"
    assert preprocess.context_free_lemma("was") == "be"
    assert preprocess._cached_stem("children") == "children"


def test_context_free_lemma_leaves_known_lemmas_alone():
    """'news' is a noun in its own right, not the plural of 'new'."""
    assert preprocess.context_free_lemma("news") == "news"
    assert preprocess.context_free_lemma("analysis") == "analysis"


def test_morphology_probes_report_both_failure_modes():
    rows = morphology.probe_rows()
    verdicts = {r["verdict"] for r in rows}
    assert any("over-stemming" in v for v in verdicts)
    assert any("under-stemming" in v for v in verdicts)
    assert morphology.undesirable(rows), "expected at least one bad transformation"
    communism = next(r for r in rows if r["word_a"] == "communism")
    assert communism["same_stem"] == "yes" and communism["same_lemma"] == "no"


# -------------------------------------------------------------------- index


def test_index_invariants_hold(stemmed):
    stemmed.check_invariants()


def test_vocabulary_shrinks_monotonically(docs):
    sizes = [
        len(InvertedIndex.build(docs, c).terms) for c in preprocess.CONFIGURATIONS[:4]
    ]
    assert sizes[0] > sizes[1] > sizes[2] > sizes[3], sizes


def test_df_matches_postings_length(stemmed):
    for term in list(stemmed.terms)[:500]:
        assert stemmed.df(term) == len(stemmed.postings(term))


def test_postings_are_sorted_and_unique(stemmed):
    for term in list(stemmed.terms)[:500]:
        postings = stemmed.postings(term)
        assert postings == sorted(set(postings))


def test_unknown_term_returns_empty(stemmed):
    assert stemmed.postings("zzzznotaterm") == []
    assert stemmed.df("zzzznotaterm") == 0


# ------------------------------------------------------------------ parsing


def test_parses_precedence():
    assert boolean.render(boolean.parse("a AND b OR c")) == "((a AND b) OR c)"
    assert boolean.render(boolean.parse("a OR b AND c")) == "(a OR (b AND c))"
    assert boolean.render(boolean.parse("(a OR b) AND c")) == "((a OR b) AND c)"
    assert boolean.render(boolean.parse("NOT a AND b")) == "(NOT a AND b)"


def test_implicit_and():
    assert boolean.render(boolean.parse("oil price")) == "(oil AND price)"


@pytest.mark.parametrize("bad", ["", "(a", "a)", "AND b", "a AND"])
def test_syntax_errors(bad):
    with pytest.raises(boolean.QuerySyntaxError):
        boolean.parse(bad)


# ---------------------------------------------------------------- retrieval


def test_merge_primitives():
    stats = boolean.QueryStats()
    assert boolean.intersect([1, 3, 5, 7], [3, 4, 5], stats) == [3, 5]
    assert boolean.union([1, 3], [2, 3, 4], stats) == [1, 2, 3, 4]
    assert boolean.difference([1, 2, 3, 4], [2, 4], stats) == [1, 3]
    assert stats.comparisons > 0


def test_and_matches_set_intersection(stemmed, normalize):
    docs_found, _ = boolean.execute("oil AND price", stemmed, normalize)
    expected = set(stemmed.postings(normalize("oil"))) & set(
        stemmed.postings(normalize("price"))
    )
    assert set(docs_found) == expected


def test_not_excludes(stemmed, normalize):
    with_football, _ = boolean.execute("football", stemmed, normalize)
    without, _ = boolean.execute("bank AND NOT football", stemmed, normalize)
    assert not (set(without) & set(with_football))


def test_strategies_agree_on_every_query(stemmed, normalize):
    """The correctness gate for the optimisation experiment."""
    for query in QUERIES:
        naive, _ = boolean.execute(query.text, stemmed, normalize, optimize=False)
        optimized, _ = boolean.execute(query.text, stemmed, normalize, optimize=True)
        assert set(naive) == set(optimized), query.qid


def test_optimization_never_costs_more_overall(stemmed, normalize):
    naive_total = optimized_total = 0
    for query in QUERIES:
        _, n = boolean.execute(query.text, stemmed, normalize, optimize=False)
        _, o = boolean.execute(query.text, stemmed, normalize, optimize=True)
        naive_total += n.comparisons
        optimized_total += o.comparisons
    assert optimized_total <= naive_total


# ------------------------------------------------------------------ tolerant


@pytest.fixture(scope="module")
def kgram(docs):
    """Built over the UNSTEMMED surface vocabulary, matching the experiment."""
    surface = InvertedIndex.build(docs, preprocess.NOSTOP)
    idx = tolerant.KGramIndex(surface.terms, k=3)
    idx.attach_frequencies(surface.vocabulary.collection_frequency)
    return idx


def test_levenshtein_known_pairs():
    assert tolerant.levenshtein("kitten", "sitting") == 3
    assert tolerant.levenshtein("flaw", "lawn") == 2
    assert tolerant.levenshtein("oil", "oil") == 0
    assert tolerant.levenshtein("", "abc") == 3
    assert tolerant.levenshtein("pirce", "price") == 2


def test_levenshtein_cap_short_circuits():
    assert tolerant.levenshtein("abcdefgh", "zzzzzzzz", cap=2) == 3


def test_levenshtein_matrix_corner():
    matrix = tolerant.levenshtein_matrix("kitten", "sitting")
    assert matrix[-1][-1] == 3


def test_kgrams_have_boundaries():
    assert tolerant.kgrams("oil", 3) == ["$oi", "oil", "il$"]


def test_wildcard_superset_of_exact(kgram, stemmed):
    matches = kgram.wildcard("comput*")
    assert matches, "expected some comput* terms"
    assert all(m.startswith("comput") for m in matches)


def test_wildcard_middle(kgram):
    for match in kgram.wildcard("gov*ment"):
        assert match.startswith("gov") and match.endswith("ment")


def test_wildcard_parses_as_its_own_node():
    node = boolean.parse("comput* AND security")
    assert isinstance(node.children[0], boolean.Wildcard)
    assert node.children[0].pattern == "comput*"
    with pytest.raises(boolean.QuerySyntaxError):
        boolean.parse("* AND security")


def test_wildcard_without_an_expander_is_rejected(stemmed, normalize):
    """The '*' must never be silently discarded.

    Dropping it turns 'comput*' into a lookup for the literal term 'comput',
    which is not in the dictionary, so the query would return zero documents
    and look like a legitimate empty result rather than a mistake.
    """
    with pytest.raises(boolean.QuerySyntaxError):
        boolean.execute("comput* AND security", stemmed, normalize)


def test_wildcard_query_dominates_the_literal_prefix(kgram, stemmed, normalize):
    """'invest*' also reaches investigate/investigation, which stem elsewhere."""
    expanded, _ = boolean.execute(
        "invest* AND NOT football", stemmed, normalize, expand=kgram.wildcard
    )
    exact, _ = boolean.execute("investment AND NOT football", stemmed, normalize)
    assert set(exact) < set(expanded)


def test_wildcard_strategies_agree(kgram, stemmed, normalize):
    for query in ["comput* AND security", "invest* AND NOT football"]:
        naive, _ = boolean.execute(
            query, stemmed, normalize, optimize=False, expand=kgram.wildcard
        )
        opt, _ = boolean.execute(
            query, stemmed, normalize, optimize=True, expand=kgram.wildcard
        )
        assert set(naive) == set(opt)


def test_correction_repairs_a_clear_misspelling(kgram):
    assert kgram.correct("goverment") == "government"
    assert kgram.correct("brodband") == "broadband"
    assert kgram.correct("chelsae") == "chelsea"


def test_correction_must_run_before_stemming(kgram, stemmed):
    """Why the k-gram index is built over surface forms, not stems.

    Porter reduces the misspelling 'goverment' to 'gover', which exists in the
    stemmed dictionary as a term in its own right. A corrector consulting the
    stemmed vocabulary therefore sees a known term, declines to repair it, and
    the query silently returns nothing. Repairing the surface form first and
    stemming afterwards avoids the trap entirely.
    """
    from ir import porter

    assert porter.stem("goverment") == "gover"
    assert "gover" in stemmed


def test_edit_distance_is_frequency_blind(kgram):
    """The report's headline false-positive mechanism.

    Edit distance ranks candidates by form alone, so a common word two edits
    away loses to a rarer word one edit away. 'oill' is one edit from both the
    intended 'oil' and the unrelated 'bill'. Resolving this properly needs a
    noisy-channel model weighting the candidate prior by term frequency, which
    a pure edit-distance corrector has no way to express.
    """
    assert tolerant.levenshtein("oill", "oil") == 1
    assert tolerant.levenshtein("oill", "bill") == 1
    assert tolerant.levenshtein("pirce", "price") == 2


def test_every_misspelled_query_runs(kgram):
    for mis in MISSPELLED:
        node = boolean.parse(mis.text)
        for term in boolean.all_terms(node):
            kgram.correct(term.lower())


# ---------------------------------------------------------------- evaluation


def test_scores_arithmetic():
    s = evaluate.score("q", retrieved={1, 2, 3}, relevant={2, 3, 4})
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == pytest.approx(2 / 3)
    assert s.f1 == pytest.approx(2 / 3)


def test_degenerate_cases_do_not_divide_by_zero():
    assert evaluate.score("q", set(), {1}).precision == 0.0
    assert evaluate.score("q", {1}, set()).recall == 0.0
    assert evaluate.score("q", set(), set()).f1 == 0.0


def test_perfect_and_empty_retrieval():
    assert evaluate.score("q", {1, 2}, {1, 2}).f1 == 1.0
    assert evaluate.score("q", {9}, {1, 2}).f1 == 0.0


def test_macro_and_micro_differ_when_sizes_differ():
    scores = [
        evaluate.score("a", {1}, {1}),
        evaluate.score("b", set(range(100)), set(range(50))),
    ]
    assert evaluate.macro_average(scores) != evaluate.micro_average(scores)


# ------------------------------------------------------------------- qrels


def test_every_query_has_relevant_documents(docs):
    qrels = build_qrels(docs)
    empty = [qid for qid, rel in qrels.items() if not rel]
    assert not empty, f"queries with no relevant docs: {empty}"


def test_query_counts_meet_assignment_minimums():
    assert len(QUERIES) >= 30
    assert len(MISSPELLED) >= 20
