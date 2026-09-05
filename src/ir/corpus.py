# src/ir/corpus.py
"""Loading and profiling of the BBC news corpus.

Deliberately does no text processing. It answers two questions: which files are
in the corpus, and what shape is the corpus. Tokenization, normalization and
stemming all live in preprocess.py.
"""

import statistics
from dataclasses import dataclass
from pathlib import Path

# Assignment section 2 requires source, attribution and domain to be reported.
# They live here so every generated artifact carries them.
DATASET = {
    "name": "BBC News full-text corpus",
    "domain": "news (single domain, five sub-topics)",
    "source": "UCD Machine Learning Group - http://mlg.ucd.ie/datasets/bbc.html",
    "download": "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip",
    "attribution": (
        "D. Greene and P. Cunningham, 'Practical Solutions to the Problem of "
        "Diagonal Dominance in Kernel Document Clustering', Proc. ICML 2006."
    ),
    "licence": (
        "Article content copyright BBC; released for non-commercial research "
        "use only."
    ),
    "categories": ("business", "entertainment", "politics", "sport", "tech"),
}


@dataclass(frozen=True)
class Document:
    doc_id: int  # dense, 0-based, stable across runs
    category: str  # parent folder name
    filename: str  # e.g. '027.txt'
    text: str  # raw article, byte-for-byte as stored


def load_corpus(root: Path) -> list[Document]:
    """Load every BBC article under root, ordered by (category, filename)."""
    # Suffix compared case-insensitively: pathlib globbing is case-insensitive
    # on Windows but not on Linux, and the Virtual Lab is Linux.
    paths = sorted(
        (
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() == ".txt"
            and p.name.upper() != "README.TXT"
        ),
        key=lambda p: (p.parent.name, p.name),
    )
    return [
        Document(
            doc_id=i,
            category=p.parent.name,
            filename=p.name,
            text=p.read_text(encoding="utf-8"),
        )
        for i, p in enumerate(paths)
    ]


def corpus_stats(docs: list[Document]) -> dict:
    """Whitespace-level profile of the corpus, for report section 2.

    This is the pre-preprocessing baseline: it splits on whitespace only, so
    'Steel' and 'steel' count as different types and '45,000' keeps its comma.
    The gap between these numbers and the real tokenizer's is itself a result.
    """
    if not docs:
        raise ValueError("corpus is empty")

    lengths = [len(d.text.split()) for d in docs]
    types = {token for d in docs for token in d.text.split()}

    return {
        "dataset": DATASET["name"],
        "domain": DATASET["domain"],
        "source": DATASET["source"],
        "attribution": DATASET["attribution"],
        "num_docs": len(docs),
        "raw_bytes": sum(len(d.text.encode("utf-8")) for d in docs),
        "total_whitespace_tokens": sum(lengths),
        "unique_whitespace_types": len(types),
        "doc_length": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "max": max(lengths),
            "mean": round(statistics.fmean(lengths), 2),
        },
    }
