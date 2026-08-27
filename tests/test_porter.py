"""Our Porter implementation is checked against NLTK's as an independent oracle.

NLTK is used here only as a reference to measure against; the stemmer under
test is entirely our own. NLTK's default mode adds post-1980 extensions, so the
comparison uses ORIGINAL_ALGORITHM to match the published paper.
"""

import pytest
from nltk.stem.porter import PorterStemmer

from ir import porter
from ir.preprocess import tokenize

reference = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)

# Worked examples from Porter's paper and its standard test material.
KNOWN = [
    ("caresses", "caress"),
    ("ponies", "poni"),
    ("ties", "ti"),
    ("caress", "caress"),
    ("cats", "cat"),
    ("feed", "feed"),
    ("agreed", "agre"),
    ("plastered", "plaster"),
    ("motoring", "motor"),
    ("sing", "sing"),
    ("conflated", "conflat"),
    ("troubling", "troubl"),
    ("hopping", "hop"),
    ("falling", "fall"),
    ("hissing", "hiss"),
    ("relational", "relat"),
    ("conditional", "condit"),
    ("rational", "ration"),
    ("callousness", "callous"),
    ("formality", "formal"),
    ("sensitivity", "sensit"),
    ("triplicate", "triplic"),
    ("formative", "form"),
    ("electricity", "electr"),
    ("hopeful", "hope"),
    ("goodness", "good"),
    ("revival", "reviv"),
    ("allowance", "allow"),
    ("inference", "infer"),
    ("adjustable", "adjust"),
    ("defensible", "defens"),
    ("irritant", "irrit"),
    ("replacement", "replac"),
    ("adjustment", "adjust"),
    ("dependent", "depend"),
    ("adoption", "adopt"),
    ("homologou", "homolog"),
    ("communism", "commun"),
    ("activate", "activ"),
    ("angulariti", "angular"),
    ("homologous", "homolog"),
    ("effective", "effect"),
    ("bowdlerize", "bowdler"),
    ("probate", "probat"),
    ("rate", "rate"),
    ("cease", "ceas"),
    ("controll", "control"),
    ("roll", "roll"),
]


@pytest.mark.parametrize("word,expected", KNOWN)
def test_published_examples(word, expected):
    assert porter.stem(word) == expected


def test_measure_examples():
    assert porter.measure("tr") == 0
    assert porter.measure("tree") == 0
    assert porter.measure("trouble") == 1
    assert porter.measure("oats") == 1
    assert porter.measure("trees") == 1
    assert porter.measure("troubles") == 2
    assert porter.measure("private") == 2


def test_short_words_untouched():
    for word in ["a", "an", "be", "go", "us"]:
        assert porter.stem(word) == word


def test_morphological_family_collapses():
    """The property the retrieval experiments actually depend on."""
    for family in [
        ["invest", "invests", "invested", "investing"],
        ["manage", "manages", "managed", "managing", "manager", "managers",
         "management"],
        ["announce", "announced", "announcing", "announcement"],
    ]:
        stems = {porter.stem(w) for w in family}
        assert len(stems) == 1, f"{family} produced {stems}"


def test_known_porter_failure_technology_family():
    """A documented limitation of the 1980 algorithm, not a bug in this code.

    Step 1c rewrites a trailing 'y' to 'i', so 'technology' becomes
    'technologi'. 'technological' instead loses '-ical' in step 3 and '-ic' in
    step 4, landing on 'technolog'. The 1980 paper has no rule bridging the two,
    so the family splits and query q26 cannot reach full recall. Later Porter
    revisions add a 'logi -> log' rule that repairs exactly this case.
    """
    assert porter.stem("technology") == "technologi"
    assert porter.stem("technologies") == "technologi"
    assert porter.stem("technological") == "technolog"


def test_known_porter_failure_play_family():
    """Measure gates block the 'player' -> 'play' reduction.

    'play' has measure 1 (the 'y' counts as a consonant after a vowel), and
    step 4's '-er' rule needs measure > 1, so 'player' survives intact. Meanwhile
    step 1c turns 'play' itself into 'plai'. Query q27 pays for this in recall.
    """
    assert porter.stem("play") == "plai"
    assert porter.stem("playing") == "plai"
    assert porter.stem("player") == "player"
    assert porter.measure("play") == 1


def test_two_letter_words_match_porters_reference_not_nltk():
    """Porter's own C code returns early when the word is 2 letters or fewer
    ('if (k <= k0+1) return;'). NLTK's ORIGINAL_ALGORITHM mode does not, and
    stems 'as' to 'a'. We follow the reference implementation.
    """
    for word in ["as", "is", "ms", "us"]:
        assert porter.stem(word) == word


def test_agreement_with_nltk_on_corpus_vocabulary(vocabulary_sample):
    """Report the agreement rate; every disagreement must be inspectable."""
    disagreements = [
        (w, porter.stem(w), reference.stem(w))
        for w in vocabulary_sample
        if porter.stem(w) != reference.stem(w)
    ]
    rate = 1 - len(disagreements) / len(vocabulary_sample)
    print(
        f"\nPorter agreement with NLTK ORIGINAL_ALGORITHM: {rate:.4%} "
        f"({len(disagreements)} of {len(vocabulary_sample)} differ)"
    )
    for word, ours, theirs in disagreements[:20]:
        print(f"  {word:<20} ours={ours:<16} nltk={theirs}")
    assert rate > 0.99


@pytest.fixture(scope="module")
def vocabulary_sample():
    from ir.corpus import load_corpus
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "bbc"
    docs = load_corpus(root)[:400]
    words = {t.lower() for d in docs for t in tokenize(d.text) if t.isalpha()}
    return sorted(words)


def test_trace_records_every_step():
    history = porter.trace("relational")
    assert history[0] == ("input", "relational")
    assert history[-1][1] == "relat"
    assert [name for name, _ in history[1:]] == [
        "1a", "1b", "1c", "2", "3", "4", "5a", "5b"
    ]
