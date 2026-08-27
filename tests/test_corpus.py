from pathlib import Path

import pytest

from ir.corpus import Document, corpus_stats, load_corpus

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "bbc"
CATEGORIES = {"business", "entertainment", "politics", "sport", "tech"}


@pytest.fixture(scope="module")
def docs() -> list[Document]:
    return load_corpus(ROOT)


def test_document_count(docs):
    assert len(docs) == 2225


def test_readme_excluded(docs):
    assert all(d.filename.upper() != "README.TXT" for d in docs)


def test_categories(docs):
    assert {d.category for d in docs} == CATEGORIES


def test_doc_ids_are_dense_and_ordered(docs):
    assert [d.doc_id for d in docs] == list(range(len(docs)))


def test_known_document_position(docs):
    """business sorts first and filenames are zero-padded, so 027.txt is index 26."""
    assert docs[26].category == "business"
    assert docs[26].filename == "027.txt"
    assert docs[26].text.startswith("Steel firm 'to cut' 45,000 jobs")


def test_encoding_canary(docs):
    """647 files carry a pound sign; cp1252 decoding would turn every one into 'Â£'."""
    assert sum(1 for d in docs if "\u00a3" in d.text) == 647
    assert not any("\u00c2\u00a3" in d.text for d in docs)


def test_load_is_reproducible():
    first = load_corpus(ROOT)
    second = load_corpus(ROOT)
    assert first == second


def test_stats_shape(docs):
    stats = corpus_stats(docs)
    assert stats["num_docs"] == 2225
    assert stats["total_whitespace_tokens"] > 0
    assert stats["unique_whitespace_types"] > 0
    assert stats["doc_length"]["min"] <= stats["doc_length"]["median"]
    assert stats["doc_length"]["median"] <= stats["doc_length"]["max"]


def test_stats_rejects_empty_corpus():
    with pytest.raises(ValueError):
        corpus_stats([])
