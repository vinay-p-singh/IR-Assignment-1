"""Single-command demonstration for the BITS Virtual Lab.

    python demo.py

Prints one section per assignment requirement, in the order the assignment
states them, small enough to fit in a screenshot. Everything shown is computed
live from data/raw/bbc -- nothing is read back from a saved results file.

Run experiments/run_experiments.py instead if you want the full result set
written to data/derived.
"""

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import nltk

nltk.data.path.insert(0, str(ROOT / "nltk_data"))

from ir import boolean, evaluate, morphology, preprocess, queryset, tolerant
from ir.corpus import DATASET, corpus_stats, load_corpus
from ir.index import InvertedIndex
from ir.porter import trace

WIDTH = 78


def heading(number: str, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}  {title}")
    print("=" * WIDTH)


def row(cells: list, widths: list[int]) -> None:
    print("  " + "".join(str(c).ljust(w) for c, w in zip(cells, widths)))


def main() -> None:
    print("=" * WIDTH)
    print("Boolean Information Retrieval System - BBC News corpus".center(WIDTH))
    print("=" * WIDTH)

    # ------------------------------------------------------------ section 2
    heading("[2]", "DATASET")
    docs = load_corpus(ROOT / "data" / "raw" / "bbc")
    stats = corpus_stats(docs)
    print(f"  source       {DATASET['source']}")
    print(f"  domain       {DATASET['domain']}")
    print(f"  attribution  {DATASET['attribution'][:60]}...")
    print(f"  licence      {DATASET['licence'][:60]}...")
    print(f"  documents    {stats['num_docs']:,}")
    print(f"  corpus size  {stats['raw_bytes'] / 1e6:.1f} MB")
    print(f"  tokens       {stats['total_whitespace_tokens']:,} (whitespace split)")
    print(f"  unique types {stats['unique_whitespace_types']:,} (before processing)")
    d = stats["doc_length"]
    print(
        f"  doc length   min={d['min']}  median={d['median']}  "
        f"mean={d['mean']}  max={d['max']} words"
    )

    # ------------------------------------------------------------ section 3A
    heading("[3A]", "TEXT PROCESSING")
    sample = "The Investors were ANNOUNCING new investments in US technologies."
    print(f'  input: "{sample}"\n')
    for config in preprocess.CONFIGURATIONS:
        tokens = config(sample)
        print(f"  {config.name:<11} ({len(tokens):>2}) {' '.join(tokens)}")

    print("\n  Porter stemmer, step by step (our own implementation):")
    for word in ["relational", "investments", "technologies", "announced"]:
        steps = " -> ".join(f"{s}:{v}" for s, v in trace(word))
        print(f"    {word:<14} {steps}")

    # ------------------------------------------------------------ section 3B
    heading("[3B]", "VOCABULARY, DICTIONARY AND INVERTED INDEX")
    indexes = {}
    row(["config", "vocabulary", "postings", "tokens", "hapax", "mean df"],
        [13, 13, 12, 12, 10, 10])
    print("  " + "-" * (WIDTH - 4))
    for config in preprocess.CONFIGURATIONS:
        index = InvertedIndex.build(docs, config)
        index.check_invariants()
        indexes[config.name] = index
        s = index.stats()
        row([config.name, f"{s['vocabulary_size']:,}", f"{s['total_postings']:,}",
             f"{s['total_tokens']:,}", f"{s['hapax_legomena']:,}", s["mean_df"]],
            [13, 13, 12, 12, 10, 10])

    stemmed = indexes["stemmed"]
    normalize = boolean.normalizer_for(preprocess.STEMMED)
    print("\n  Sorted postings lists (stemmed index):")
    for term in ["oil", "elect", "broadband", "chelsea", "technolog"]:
        if term in stemmed:
            print(f"    {stemmed.sample_postings(term)}")

    # ------------------------------------------------------------ section 3C
    heading("[3C]", "BOOLEAN RETRIEVAL AND QUERY OPTIMIZATION")
    demo_queries = [
        "oil AND price",
        "football OR rugby",
        "bank AND NOT football",
        "(microsoft OR apple) AND software",
        "market AND prices AND oil AND gas",
    ]
    row(["query", "hits", "naive cmp", "opt cmp", "saving"], [38, 8, 12, 12, 8])
    print("  " + "-" * (WIDTH - 4))
    total_naive = total_opt = 0
    for q in demo_queries:
        naive, n_stats = boolean.execute(q, stemmed, normalize, optimize=False)
        opt, o_stats = boolean.execute(q, stemmed, normalize, optimize=True)
        assert set(naive) == set(opt), f"strategies disagree on {q!r}"
        total_naive += n_stats.comparisons
        total_opt += o_stats.comparisons
        saving = n_stats.comparisons - o_stats.comparisons
        pct = 100 * saving / n_stats.comparisons if n_stats.comparisons else 0.0
        row([q, len(naive), f"{n_stats.comparisons:,}", f"{o_stats.comparisons:,}",
             f"{pct:.0f}%"], [38, 8, 12, 12, 8])
    print(f"\n  total comparisons  naive={total_naive:,}  optimized={total_opt:,}")

    trace_query = "market AND prices AND oil AND gas"
    _, naive_stats = boolean.execute(trace_query, stemmed, normalize, optimize=False)
    _, opt_stats = boolean.execute(trace_query, stemmed, normalize, optimize=True)
    print(f"\n  Merge order trace for '{trace_query}':")
    print(f"    as written      {' '.join(naive_stats.order)}")
    print(f"    smallest first  {' '.join(opt_stats.order)}")

    # ------------------------------------------------------------ section 3D
    heading("[3D]", "TOLERANT RETRIEVAL")
    surface = indexes["nostop"]
    kgram = tolerant.KGramIndex(surface.terms, k=3)
    kgram.attach_frequencies(surface.vocabulary.collection_frequency)
    print(f"  3-gram index over {len(surface.terms):,} surface terms "
          f"({len(kgram):,} distinct k-grams)")

    print("\n  Wildcard expansion:")
    for pattern in ["comput*", "gov*ment", "*ology"]:
        matches = kgram.wildcard(pattern)
        print(f"    {pattern:<10} {len(matches):>3} terms: {' '.join(matches[:6])}")

    print("\n  Wildcards inside Boolean queries:")
    for q in ["comput* AND security", "invest* AND NOT football"]:
        hits, _ = boolean.execute(
            q, stemmed, normalize, optimize=True, expand=kgram.wildcard
        )
        print(f"    {q:<28} -> {len(hits)} documents")

    print("\n  Edit-distance spelling repair:")
    for wrong in ["goverment", "brodband", "recieve", "managment"]:
        near = kgram.nearest(wrong, limit=3)
        shown = ", ".join(f"{t} (d={d})" for t, d in near)
        print(f"    {wrong:<12} -> {shown}")

    # ------------------------------------------------------------ section 3E
    heading("[3E]", "EVALUATION")
    started = time.perf_counter()
    qrels = queryset.build_qrels(docs)
    print(f"  {len(queryset.QUERIES)} test queries, "
          f"{len(queryset.MISSPELLED)} misspelled variants, "
          f"{sum(len(v) for v in qrels.values()):,} relevance judgments")

    row(["index", "precision", "recall", "F1"], [14, 14, 14, 14])
    print("  " + "-" * (WIDTH - 4))
    for name, index in indexes.items():
        norm = boolean.normalizer_for(preprocess.BY_NAME[name])
        scores = [
            evaluate.score(q.qid, set(boolean.execute(q.text, index, norm)[0]),
                           qrels[q.qid])
            for q in queryset.QUERIES
        ]
        macro = evaluate.macro_average(scores)
        row([name, f"{macro['precision']:.4f}", f"{macro['recall']:.4f}",
             f"{macro['f1']:.4f}"], [14, 14, 14, 14])

    exact, repaired = [], []
    for mis in queryset.MISSPELLED:
        relevant = qrels[mis.base_qid]
        hits, _ = boolean.execute(mis.text, stemmed, normalize)
        exact.append(evaluate.score(mis.qid, set(hits), relevant))

        fixed = mis.text
        for term in set(boolean.all_terms(boolean.parse(mis.text))):
            if term.lower() not in surface:
                fixed = fixed.replace(term, kgram.correct(term.lower()))
        hits, _ = boolean.execute(fixed, stemmed, normalize)
        repaired.append(evaluate.score(mis.qid, set(hits), relevant))

    print("\n  Misspelled queries, exact vs tolerant retrieval:")
    row(["mode", "precision", "recall", "F1"], [14, 14, 14, 14])
    print("  " + "-" * (WIDTH - 4))
    for label, scores in [("exact", exact), ("tolerant", repaired)]:
        m = evaluate.macro_average(scores)
        row([label, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
             f"{m['f1']:.4f}"], [14, 14, 14, 14])

    # ------------------------------------------------------------- section 4
    heading("[4]", "UNDESIRABLE MORPHOLOGICAL TRANSFORMATIONS")
    rows = morphology.probe_rows()
    row(["word a", "word b", "stem", "lemma", "verdict"], [14, 13, 13, 13, 25])
    print("  " + "-" * (WIDTH - 4))
    for r in morphology.undesirable(rows)[:10]:
        merged_stem = "merged" if r["same_stem"] == "yes" else "kept apart"
        merged_lemma = "merged" if r["same_lemma"] == "yes" else "kept apart"
        row([r["word_a"], r["word_b"], merged_stem, merged_lemma,
             r["verdict"].split(":")[0]], [14, 13, 13, 13, 25])

    print(f"\n  Completed in {time.perf_counter() - started:.1f}s of scoring.")
    print("=" * WIDTH)


if __name__ == "__main__":
    main()
