"""Runs every experiment the assignment requires and writes results to data/derived.

    python experiments/run_experiments.py

Everything downstream -- the report tables, the plots, the visual companion --
reads the files this script produces. Nothing is hand-copied.
"""

import csv
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nltk  # noqa: E402

nltk.data.path.append(str(ROOT / "nltk_data"))

from ir import boolean, evaluate, preprocess, queryset, tolerant  # noqa: E402
from ir.corpus import corpus_stats, load_corpus  # noqa: E402
from ir.index import InvertedIndex  # noqa: E402
from ir.porter import trace  # noqa: E402

DATA = ROOT / "data" / "raw" / "bbc"
DERIVED = ROOT / "data" / "derived"
QUERIES_DIR = ROOT / "queries"


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


def write_json(path: pathlib.Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    docs = load_corpus(DATA)
    stats = corpus_stats(docs)
    write_json(DERIVED / "corpus_stats.json", stats)
    print(f"  {stats['num_docs']} documents, {stats['total_whitespace_tokens']} tokens")

    # ------------------------------------------------ experiment 1: indexes
    print("\nExperiment 1: preprocessing effect on vocabulary and index")
    indexes: dict[str, InvertedIndex] = {}
    index_rows = []
    for config in preprocess.CONFIGURATIONS:
        index = InvertedIndex.build(docs, config)
        index.check_invariants()
        indexes[config.name] = index
        index_rows.append(index.stats())
        print(f"  {config.name:<11} vocabulary={len(index.terms):>7}")
    write_csv(DERIVED / "index_stats.csv", index_rows)

    category_counts = defaultdict(int)
    for d in docs:
        category_counts[d.category] += 1

    # ------------------------------------------------ qrels and query files
    print("\nBuilding relevance judgments")
    qrels = queryset.build_qrels(docs)
    empty = [qid for qid, rel in qrels.items() if not rel]
    if empty:
        print(f"  WARNING: queries with no relevant documents: {empty}")
    print(f"  {sum(len(v) for v in qrels.values())} judgments over {len(qrels)} queries")

    with (QUERIES_DIR / "queries.tsv").open("w", encoding="utf-8", newline="") as fh:
        fh.write("qid\tkind\tquery\n")
        for q in queryset.QUERIES:
            fh.write(f"{q.qid}\t{q.kind}\t{q.text}\n")
    with (QUERIES_DIR / "misspelled.tsv").open("w", encoding="utf-8", newline="") as fh:
        fh.write("qid\tbase_qid\tcorruption\tquery\n")
        for m in queryset.MISSPELLED:
            fh.write(f"{m.qid}\t{m.base_qid}\t{m.corruption}\t{m.text}\n")
    with (QUERIES_DIR / "qrels.tsv").open("w", encoding="utf-8", newline="") as fh:
        fh.write("qid\tdoc_id\n")
        for qid in sorted(qrels):
            for doc_id in sorted(qrels[qid]):
                fh.write(f"{qid}\t{doc_id}\n")
    print("  wrote queries/queries.tsv, misspelled.tsv, qrels.tsv")

    # ------------------------- experiment 2 + 3: retrieval and optimisation
    print("\nExperiment 2: retrieval quality per preprocessing configuration")
    print("Experiment 3: naive vs postings-length-ordered query processing")
    eval_rows, opt_rows, summary_by_index = [], [], {}

    for name, index in indexes.items():
        config = preprocess.BY_NAME[name]
        normalize = boolean.normalizer_for(config)
        scores = []

        for query in queryset.QUERIES:
            naive_docs, naive_stats = boolean.execute(
                query.text, index, normalize, optimize=False
            )
            opt_docs, opt_stats = boolean.execute(
                query.text, index, normalize, optimize=True
            )
            if set(naive_docs) != set(opt_docs):
                raise AssertionError(
                    f"strategies disagree on {query.qid} in index {name}"
                )

            s = evaluate.score(query.qid, set(naive_docs), qrels[query.qid])
            scores.append(s)
            eval_rows.append({"index": name, "kind": query.kind, **s.as_row()})

            opt_rows.append(
                {
                    "index": name,
                    "query_id": query.qid,
                    "query": query.text,
                    "results": len(naive_docs),
                    "naive_comparisons": naive_stats.comparisons,
                    "optimized_comparisons": opt_stats.comparisons,
                    "comparison_saving": naive_stats.comparisons
                    - opt_stats.comparisons,
                    "naive_ms": round(naive_stats.elapsed_ms, 4),
                    "optimized_ms": round(opt_stats.elapsed_ms, 4),
                    "naive_order": " ".join(naive_stats.order),
                    "optimized_order": " ".join(opt_stats.order),
                }
            )

        summary_by_index[name] = evaluate.summarise(scores)
        macro = summary_by_index[name]["macro"]
        print(
            f"  {name:<11} P={macro['precision']:.3f} "
            f"R={macro['recall']:.3f} F1={macro['f1']:.3f}"
        )

    write_csv(DERIVED / "evaluation.csv", eval_rows)
    write_csv(DERIVED / "boolean_optimization.csv", opt_rows)
    write_json(DERIVED / "evaluation_summary.json", summary_by_index)

    total_naive = sum(r["naive_comparisons"] for r in opt_rows)
    total_opt = sum(r["optimized_comparisons"] for r in opt_rows)
    print(
        f"  comparisons: naive={total_naive}  optimized={total_opt}  "
        f"saving={100 * (total_naive - total_opt) / max(total_naive, 1):.1f}%"
    )

    # ------------------------------ experiment 4: stemming vs lemmatisation
    print("\nExperiment 4: stemming vs lemmatisation conflations")
    normalized = indexes["nostop"]
    stem_groups: dict[str, set[str]] = defaultdict(set)
    lemma_groups: dict[str, set[str]] = defaultdict(set)
    for term in normalized.terms:
        stem_groups[preprocess._cached_stem(term)].add(term)
        lemma_groups[preprocess._cached_lemma(term)].add(term)

    conflation_rows = []
    for stem_key, members in sorted(
        stem_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])
    )[:40]:
        if len(members) < 2:
            continue
        conflation_rows.append(
            {
                "stem": stem_key,
                "surface_forms": len(members),
                "members": " ".join(sorted(members)[:12]),
                "lemma_groups_covering_them": len(
                    {preprocess._cached_lemma(m) for m in members}
                ),
            }
        )
    write_csv(DERIVED / "stem_vs_lemma.csv", conflation_rows)
    print(f"  {len(conflation_rows)} multi-member stem groups recorded")

    # ---------------------------------------- experiment 5: tolerant retrieval
    print("\nExperiment 5: exact vs tolerant retrieval on misspelled queries")
    stem_index = indexes["stemmed"]
    # Spelling repair runs against the UNSTEMMED surface vocabulary. Correcting
    # against stemmed terms is backwards: 'goverment' stems to 'gover', which
    # exists, so the corrector would conclude it is spelled correctly and never
    # reach 'government'. Repair the surface form first, then let the normalizer
    # stem the repaired term for the index lookup.
    surface_index = indexes["nostop"]
    kgram = tolerant.KGramIndex(surface_index.terms, k=3)
    kgram.attach_frequencies(surface_index.vocabulary.collection_frequency)
    normalize = boolean.normalizer_for(preprocess.STEMMED)

    tolerant_rows, exact_scores, tolerant_scores = [], [], []
    for mis in queryset.MISSPELLED:
        base = queryset.BY_QID[mis.base_qid]
        relevant = qrels[base.qid]

        exact_docs, _ = boolean.execute(mis.text, stem_index, normalize)

        corrections = {}
        for raw_term in set(boolean.all_terms(boolean.parse(mis.text))):
            surface = raw_term.lower()
            if surface not in surface_index:
                corrections[raw_term] = kgram.correct(surface)

        repaired = mis.text
        for original, replacement in corrections.items():
            repaired = repaired.replace(original, replacement)
        tolerant_docs, _ = boolean.execute(repaired, stem_index, normalize)

        e = evaluate.score(mis.qid, set(exact_docs), relevant)
        t = evaluate.score(mis.qid, set(tolerant_docs), relevant)
        exact_scores.append(e)
        tolerant_scores.append(t)

        tolerant_rows.append(
            {
                "qid": mis.qid,
                "base_qid": mis.base_qid,
                "corruption": mis.corruption,
                "misspelled_query": mis.text,
                "repaired_query": repaired,
                "corrections": "; ".join(f"{k}->{v}" for k, v in corrections.items()),
                "exact_results": e.retrieved,
                "tolerant_results": t.retrieved,
                "relevant": len(relevant),
                "exact_precision": round(e.precision, 4),
                "exact_recall": round(e.recall, 4),
                "tolerant_precision": round(t.precision, 4),
                "tolerant_recall": round(t.recall, 4),
            }
        )
    write_csv(DERIVED / "tolerant.csv", tolerant_rows)

    tolerant_summary = {
        "exact": evaluate.summarise(exact_scores),
        "tolerant": evaluate.summarise(tolerant_scores),
    }
    write_json(DERIVED / "tolerant_summary.json", tolerant_summary)
    print(
        f"  exact    macro R={tolerant_summary['exact']['macro']['recall']:.3f}\n"
        f"  tolerant macro R={tolerant_summary['tolerant']['macro']['recall']:.3f}"
    )

    # ----------------------------------------------- wildcard demonstrations
    wildcard_rows = []
    for pattern in ["re*val", "comput*", "*ology", "invest*", "gov*ment", "f*ball"]:
        matches = kgram.wildcard(pattern)
        wildcard_rows.append(
            {
                "pattern": pattern,
                "matches": len(matches),
                "sample": " ".join(matches[:12]),
            }
        )
    write_csv(DERIVED / "wildcards.csv", wildcard_rows)

    # --------------------------------------------------- qualitative samples
    sample_terms = ["oil", "elect", "broadband", "chelsea", "technolog"]
    postings_text = "\n".join(
        stem_index.sample_postings(t) for t in sample_terms if t in stem_index
    )
    (DERIVED / "postings_samples.txt").write_text(postings_text, encoding="utf-8")

    trace_words = [
        "relational", "conditional", "investments", "managing", "technologies",
        "announced", "winning", "universities", "agreed", "feed",
    ]
    trace_text = "\n".join(
        f"{word:<14} " + " -> ".join(f"{s}:{v}" for s, v in trace(word))
        for word in trace_words
    )
    (DERIVED / "porter_traces.txt").write_text(trace_text, encoding="utf-8")
    print("\n  wrote postings_samples.txt, porter_traces.txt")

    # ------------------------------------------------ payload for the visual
    write_json(
        DERIVED / "summary.json",
        {
            "corpus": stats,
            "categories": dict(sorted(category_counts.items())),
            "indexes": index_rows,
            "evaluation": summary_by_index,
            "tolerant": tolerant_summary,
            "comparisons": {
                "naive_total": total_naive,
                "optimized_total": total_opt,
                "saving_pct": round(
                    100 * (total_naive - total_opt) / max(total_naive, 1), 2
                ),
            },
            "sample_postings": {
                t: stem_index.postings(t)[:15]
                for t in sample_terms
                if t in stem_index
            },
            "top_terms": stem_index.vocabulary.most_frequent(15),
            "wildcards": wildcard_rows,
            "porter_traces": {w: trace(w) for w in trace_words},
        },
    )
    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
