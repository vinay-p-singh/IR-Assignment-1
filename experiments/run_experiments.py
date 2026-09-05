"""Runs every experiment the assignment requires and writes results to data/derived.

    python experiments/run_experiments.py

Everything downstream -- the report tables, the plots, the visual companion --
reads the files this script produces. Nothing is hand-copied.
"""

import csv
import json
import pathlib
import statistics
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nltk  # noqa: E402

nltk.data.path.insert(0, str(ROOT / "nltk_data"))

from ir import boolean, evaluate, morphology, preprocess, queryset, tolerant  # noqa: E402
from ir.corpus import DATASET, corpus_stats, load_corpus  # noqa: E402
from ir.index import InvertedIndex  # noqa: E402
from ir.porter import trace  # noqa: E402

DATA = ROOT / "data" / "raw" / "bbc"
DERIVED = ROOT / "data" / "derived"
QUERIES_DIR = ROOT / "queries"
FIGURES = ROOT / "docs" / "report" / "figures"

# Every timing is the median of this many runs. A single perf_counter reading is
# dominated by cache warm-up, which is why the first version of this experiment
# showed a "speed-up" on single-term queries that do no merging at all.
TIMING_REPEATS = 9


def timed_pair(query: str, index: InvertedIndex, normalize):
    """Run both strategies repeatedly, alternating which one goes first.

    Alternating matters: whichever strategy runs first pays for pulling the
    postings lists into cache, so a fixed order silently hands the second one a
    free advantage.
    """
    boolean.execute(query, index, normalize, optimize=False)
    boolean.execute(query, index, normalize, optimize=True)

    naive_samples, opt_samples = [], []
    for i in range(TIMING_REPEATS):
        if i % 2 == 0:
            n_docs, n_stats = boolean.execute(query, index, normalize, optimize=False)
            o_docs, o_stats = boolean.execute(query, index, normalize, optimize=True)
        else:
            o_docs, o_stats = boolean.execute(query, index, normalize, optimize=True)
            n_docs, n_stats = boolean.execute(query, index, normalize, optimize=False)
        naive_samples.append(n_stats.elapsed_ms)
        opt_samples.append(o_stats.elapsed_ms)

    n_stats.elapsed_ms = statistics.median(naive_samples)
    o_stats.elapsed_ms = statistics.median(opt_samples)
    return n_docs, n_stats, o_docs, o_stats


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
            naive_docs, naive_stats, opt_docs, opt_stats = timed_pair(
                query.text, index, normalize
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
                    "timing_repeats": TIMING_REPEATS,
                    "naive_ms_median": round(naive_stats.elapsed_ms, 4),
                    "optimized_ms_median": round(opt_stats.elapsed_ms, 4),
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

    # Aggregating by query shape is what separates a real effect from an
    # artefact: the morphological queries are the only ones a stemmer can help,
    # so a headline average hides where the gain actually comes from.
    by_kind_scores: dict[tuple[str, str], list] = defaultdict(list)
    for row in eval_rows:
        by_kind_scores[(row["index"], row["kind"])].append(
            evaluate.Scores(
                row["query_id"], row["retrieved"], row["relevant"],
                row["true_positives"],
            )
        )
    kind_rows = [
        {
            "index": index_name,
            "kind": kind,
            "queries": len(scores),
            **evaluate.macro_average(scores),
        }
        for (index_name, kind), scores in sorted(by_kind_scores.items())
    ]
    write_csv(DERIVED / "evaluation_by_kind.csv", kind_rows)

    total_naive = sum(r["naive_comparisons"] for r in opt_rows)
    total_opt = sum(r["optimized_comparisons"] for r in opt_rows)
    print(
        f"  comparisons: naive={total_naive}  optimized={total_opt}  "
        f"saving={100 * (total_naive - total_opt) / max(total_naive, 1):.1f}%"
    )

    # Where the saving comes from. Reordering two postings lists cannot help --
    # intersection is symmetric -- so any saving on a two-term AND would be a
    # bug. The gain is concentrated in negations and in three-or-more-way ANDs.
    saving_by_shape: dict[str, dict] = defaultdict(
        lambda: {"queries": 0, "naive": 0, "optimized": 0}
    )
    for row in opt_rows:
        query_text = row["query"]
        if "NOT" in query_text:
            shape = "contains NOT"
        elif "OR" in query_text and "AND" in query_text:
            shape = "mixed AND/OR"
        elif query_text.count("AND") >= 2:
            shape = "3+ term AND"
        elif "AND" in query_text:
            shape = "2 term AND"
        elif "OR" in query_text:
            shape = "OR only"
        else:
            shape = "single term"
        bucket = saving_by_shape[shape]
        bucket["queries"] += 1
        bucket["naive"] += row["naive_comparisons"]
        bucket["optimized"] += row["optimized_comparisons"]

    shape_rows = [
        {
            "shape": shape,
            "queries": b["queries"],
            "naive_comparisons": b["naive"],
            "optimized_comparisons": b["optimized"],
            "saving_pct": round(
                100 * (b["naive"] - b["optimized"]) / b["naive"], 2
            )
            if b["naive"]
            else 0.0,
        }
        for shape, b in sorted(saving_by_shape.items())
    ]
    write_csv(DERIVED / "optimization_by_shape.csv", shape_rows)

    # ------------------------------ experiment 4: stemming vs lemmatisation
    print("\nExperiment 4: stemming vs lemmatisation conflations")
    normalized = indexes["nostop"]
    stem_groups: dict[str, set[str]] = defaultdict(set)
    lemma_groups: dict[str, set[str]] = defaultdict(set)
    for term in normalized.terms:
        stem_groups[preprocess._cached_stem(term)].add(term)
        lemma_groups[preprocess.context_free_lemma(term)].add(term)

    conflation_rows = []
    for stem_key, members in sorted(
        stem_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])
    )[:40]:
        if len(members) < 2:
            continue
        lemmas = {preprocess.context_free_lemma(m) for m in members}
        conflation_rows.append(
            {
                "stem": stem_key,
                "surface_forms": len(members),
                "members": " ".join(sorted(members)[:12]),
                "distinct_lemmas": len(lemmas),
                # How much more aggressively the stemmer merges than the
                # lemmatizer. A high ratio is where over-stemming lives.
                "conflation_ratio": round(len(members) / len(lemmas), 2),
            }
        )
    write_csv(DERIVED / "stem_vs_lemma.csv", conflation_rows)
    print(f"  {len(conflation_rows)} multi-member stem groups recorded")

    probe_rows = morphology.probe_rows()
    write_csv(DERIVED / "morphology_probes.csv", probe_rows)
    bad = morphology.undesirable(probe_rows)
    write_csv(DERIVED / "undesirable_transformations.csv", bad)
    print(f"  {len(bad)} of {len(probe_rows)} probes are undesirable transformations")

    lemma_sizes = [len(m) for m in lemma_groups.values()]
    stem_sizes = [len(m) for m in stem_groups.values()]
    print(
        f"  stem groups={len(stem_groups)} (mean {statistics.fmean(stem_sizes):.2f} "
        f"forms)  lemma groups={len(lemma_groups)} "
        f"(mean {statistics.fmean(lemma_sizes):.2f} forms)"
    )

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

    # Wildcards inside real Boolean queries, evaluated against the same
    # postings lists. Without the expander the parser rejects the '*' outright,
    # which is the behaviour being demonstrated on the last row.
    print("\nWildcard retrieval inside Boolean queries")
    wildcard_query_rows = []
    for text_query in [
        "comput* AND security",
        "invest* AND NOT football",
        "(broadband OR mobile) AND *phone",
        "elect* AND labour",
    ]:
        expanded, wc_stats = boolean.execute(
            text_query, stem_index, normalize, optimize=True, expand=kgram.wildcard
        )
        plain, _ = boolean.execute(
            text_query.replace("*", ""), stem_index, normalize, optimize=True
        )
        wildcard_query_rows.append(
            {
                "query": text_query,
                "expanded_results": len(expanded),
                "results_if_star_ignored": len(plain),
                "extra_documents": len(set(expanded) - set(plain)),
                "merge_order": " ".join(wc_stats.order),
                "comparisons": wc_stats.comparisons,
            }
        )
        print(f"  {text_query:<34} -> {len(expanded)} docs")

    try:
        boolean.execute("comput* AND security", stem_index, normalize)
        raise AssertionError("a wildcard without an expander must be rejected")
    except boolean.QuerySyntaxError as exc:
        print(f"  without an expander the parser refuses: {exc}")
    write_csv(DERIVED / "wildcard_queries.csv", wildcard_query_rows)

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
            "dataset": DATASET,
            "categories": dict(sorted(category_counts.items())),
            "indexes": index_rows,
            "evaluation": summary_by_index,
            "evaluation_by_kind": kind_rows,
            "tolerant": tolerant_summary,
            "comparisons": {
                "naive_total": total_naive,
                "optimized_total": total_opt,
                "saving_pct": round(
                    100 * (total_naive - total_opt) / max(total_naive, 1), 2
                ),
                "timing_repeats": TIMING_REPEATS,
                "by_shape": shape_rows,
            },
            "morphology_probes": probe_rows,
            "sample_postings": {
                t: stem_index.postings(t)[:15]
                for t in sample_terms
                if t in stem_index
            },
            "top_terms": stem_index.vocabulary.most_frequent(15),
            "wildcards": wildcard_rows,
            "wildcard_queries": wildcard_query_rows,
            "porter_traces": {w: trace(w) for w in trace_words},
        },
    )

    print("\nGenerating figures")
    make_figures(index_rows, summary_by_index, kind_rows, shape_rows, tolerant_summary)
    print("\nAll experiments complete.")


def make_figures(index_rows, summary_by_index, kind_rows, shape_rows, tolerant_summary):
    """Four figures for the report. Every value is read from the run above."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    names = [r["index"] for r in index_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, [r["vocabulary_size"] for r in index_rows], color="#4c72b0")
    for i, r in enumerate(index_rows):
        ax.text(i, r["vocabulary_size"], f"{r['vocabulary_size']:,}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("Vocabulary size by preprocessing configuration")
    ax.set_ylabel("distinct terms")
    fig.tight_layout()
    fig.savefig(FIGURES / "vocabulary_by_config.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.26
    positions = range(len(names))
    for offset, metric, colour in [
        (-width, "precision", "#4c72b0"),
        (0.0, "recall", "#dd8452"),
        (width, "f1", "#55a868"),
    ]:
        ax.bar(
            [p + offset for p in positions],
            [summary_by_index[n]["macro"][metric] for n in names],
            width=width, label=metric, color=colour,
        )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Macro-averaged retrieval effectiveness (30 queries)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "effectiveness_by_config.png", dpi=150)
    plt.close(fig)

    kinds = sorted({r["kind"] for r in kind_rows})
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        values = [
            next(r["recall"] for r in kind_rows
                 if r["index"] == name and r["kind"] == kind)
            for kind in kinds
        ]
        ax.bar([x + i * width for x in range(len(kinds))], values,
               width=width, label=name)
    ax.set_xticks([x + 0.4 - width / 2 for x in range(len(kinds))])
    ax.set_xticklabels(kinds)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("macro recall")
    ax.set_title("Recall by query shape: where stemming actually pays")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "recall_by_query_kind.png", dpi=150)
    plt.close(fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    shapes = [r["shape"] for r in shape_rows]
    left.barh(shapes, [r["saving_pct"] for r in shape_rows], color="#4c72b0")
    left.set_xlabel("comparisons saved (%)")
    left.set_title("Query optimization saving by query shape")

    metrics = ["precision", "recall", "f1"]
    right.bar([m + "\nexact" for m in metrics],
              [tolerant_summary["exact"]["macro"][m] for m in metrics],
              color="#c44e52")
    right.bar([m + "\ntolerant" for m in metrics],
              [tolerant_summary["tolerant"]["macro"][m] for m in metrics],
              color="#55a868")
    right.set_ylim(0, 1.05)
    right.set_title("Exact vs tolerant retrieval (20 misspelled queries)")
    right.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "optimization_and_tolerance.png", dpi=150)
    plt.close(fig)
    print(f"  wrote 4 figures to {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
