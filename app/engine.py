"""Chat turn handling: message in, list of render blocks out.

Every reply is a list of typed blocks (text, table, bar, docs, matrix, ...).
The browser knows how to draw each type and nothing else, so adding a command
here never requires touching the rendering code.

The interesting command is the default one. Running a Boolean query returns not
just the hits but a *validation*: the same query is evaluated a second time
directly over the raw article text with whole-word matching, and the two answers
are scored against each other. That second evaluation is the oracle -- it is
what the index is supposed to approximate -- so the precision/recall gap it
exposes is exactly the effect of the preprocessing pipeline under test.

A query written as a sentence is handled differently from one written with
explicit operators, because the two mean different things. See _query.
"""

import re
import time
from collections import Counter
from functools import lru_cache

from ir import boolean, evaluate, porter, preprocess, queryset, tolerant

from .session import Session

MAX_DOCS_SHOWN = 40
SNIPPET_WIDTH = 220

# A query is treated as free text unless it says otherwise. Operators must be
# upper case to count, which is how the assignment writes them; parentheses and
# wildcards also mark a query as deliberately structured.
_EXPLICIT_SYNTAX = re.compile(r"\bAND\b|\bOR\b|\bNOT\b|[()*]")


# ---------------------------------------------------------------- blocks


def text(body: str, tone: str = "plain") -> dict:
    return {"type": "text", "text": body, "tone": tone}


def kv(title: str, items: list[tuple[str, object]]) -> dict:
    return {"type": "kv", "title": title, "items": [[k, str(v)] for k, v in items]}


def table(title: str, columns: list[str], rows: list[list]) -> dict:
    return {
        "type": "table",
        "title": title,
        "columns": columns,
        "rows": [[str(c) for c in row] for row in rows],
    }


def bar(title: str, items: list[tuple[str, float, str]], unit: str = "") -> dict:
    return {
        "type": "bar",
        "title": title,
        "unit": unit,
        "items": [{"label": a, "value": b, "note": c} for a, b, c in items],
    }


def chips(title: str, items: list[str], note: str = "") -> dict:
    return {"type": "chips", "title": title, "items": items, "note": note}


def steps(title: str, items: list[tuple[str, str]]) -> dict:
    return {
        "type": "steps",
        "title": title,
        "items": [{"label": a, "value": b} for a, b in items],
    }


# ---------------------------------------------------------------- oracle


@lru_cache(maxsize=4096)
def _word_re(word: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)


@lru_cache(maxsize=1024)
def _wildcard_re(pattern: str) -> re.Pattern:
    body = r"[A-Za-z0-9]*".join(re.escape(p) for p in pattern.split("*"))
    return re.compile(rf"\b{body}\b", re.IGNORECASE)


def _lexical_gold(node, docs, all_ids: set[int]) -> set[int]:
    """Evaluate a parsed query over raw article text, whole-word, no stemming.

    This is deliberately naive and deliberately slow. It is the reference answer
    a reader would produce by hand with Ctrl-F, which is what makes it a fair
    yardstick for the index.
    """
    match node:
        case boolean.Term(value):
            pattern = _word_re(value)
            return {d.doc_id for d in docs if pattern.search(d.text)}
        case boolean.Wildcard(pattern_text):
            pattern = _wildcard_re(pattern_text)
            return {d.doc_id for d in docs if pattern.search(d.text)}
        case boolean.Not(child):
            return all_ids - _lexical_gold(child, docs, all_ids)
        case boolean.And(children):
            result = _lexical_gold(children[0], docs, all_ids)
            for child in children[1:]:
                result &= _lexical_gold(child, docs, all_ids)
            return result
        case boolean.Or(children):
            result = set()
            for child in children:
                result |= _lexical_gold(child, docs, all_ids)
            return result
    raise TypeError(f"unknown node {node!r}")


def _snippet(raw: str, stems: list[str]) -> str:
    """Window of the article around the first term hit.

    Matched on a prefix rather than a whole word because the query terms arrive
    stemmed, and a stem is a prefix of the surface forms it stands for.
    """
    lowered = raw.lower()
    pos = -1
    for stem in stems:
        if not stem:
            continue
        found = re.search(rf"\b{re.escape(stem)}", lowered)
        if found:
            pos = found.start()
            break
    if pos < 0:
        pos = 0
    start = max(0, pos - SNIPPET_WIDTH // 3)
    end = min(len(raw), start + SNIPPET_WIDTH)
    body = " ".join(raw[start:end].split())
    return ("... " if start else "") + body + (" ..." if end < len(raw) else "")


# ---------------------------------------------------------------- engine


class ChatEngine:
    def __init__(self, session: Session) -> None:
        self.s = session

    # -------------------------------------------------------- entry point

    def handle(self, message: str) -> dict:
        message = (message or "").strip()
        if not message:
            return self._reply([text("Type a query, or :help for the command list.")])

        head, _, rest = message.partition(" ")
        rest = rest.strip()
        command = head.lower().lstrip("/:") if head[0] in ":/" else None

        nullary = {"help": self._help, "stats": self._stats, "queries": self._queries}
        unary = {
            "index": self._index,
            "term": self._term,
            "doc": self._doc,
            "stem": self._stem,
            "wild": self._wildcard,
            "spell": self._spell,
            "compare": self._compare,
            "eval": self._eval,
        }
        try:
            if command is None:
                return self._reply(self._query(message))
            if command in nullary:
                return self._reply(nullary[command]())
            if command in unary:
                return self._reply(unary[command](rest))
            return self._reply(
                [text(f"Unknown command :{command}. Try :help.", "warn")]
            )
        except boolean.QuerySyntaxError as exc:
            return self._reply([text(f"Query syntax error: {exc}", "error")])
        except KeyError as exc:
            return self._reply([text(f"Not found: {exc}", "error")])
        except Exception as exc:  # surfaced in the transcript, not the console
            return self._reply([text(f"{type(exc).__name__}: {exc}", "error")])

    def _reply(self, blocks: list[dict]) -> dict:
        return {"blocks": blocks, "state": self.state()}

    def state(self) -> dict:
        return {
            "active": self.s.active,
            "configs": [
                {"name": n, "built": self.s.is_built(n)} for n in self.s.config_names
            ],
            "num_docs": len(self.s.docs),
        }

    # ------------------------------------------------------------ commands

    def _help(self) -> list[dict]:
        return [
            text(
                "Anything that is not a command is run as a query. Write AND, OR "
                "and NOT in capitals to get strict Boolean retrieval; write a "
                "plain sentence and the stop words are dropped for you."
            ),
            table(
                "Commands",
                ["command", "what it does"],
                [
                    [":index <name>", "switch preprocessing config, or list them"],
                    [":compare <query>", "naive vs postings-ordered execution"],
                    [":eval <qid|all>", "score a benchmark query against its qrels"],
                    [":queries", "the 30 benchmark queries"],
                    [":term <term>", "df, cf and the postings list"],
                    [":doc <id>", "read one article"],
                    [":stem <word>", "Porter algorithm, step by step"],
                    [":wild <pat*ern>", "k-gram wildcard expansion"],
                    [":spell <word>", "nearest vocabulary terms + edit matrix"],
                    [":stats", "corpus and index statistics"],
                ],
            ),
            chips(
                "Try",
                [
                    "oil AND price",
                    "show me discussion on oil and gas prices",
                    "(microsoft OR apple) AND software",
                    "bank AND NOT football",
                    ":compare (oil OR gas) AND prices",
                    ":spell recieve",
                    ":wild econom*",
                    ":stem nationalization",
                    ":eval all",
                ],
            ),
        ]

    def _stats(self) -> list[dict]:
        cs = self.s.corpus_stats
        index = self.s.index()
        stats = index.stats()
        blocks = [
            kv(
                "Corpus",
                [
                    ("documents", cs["num_docs"]),
                    ("raw bytes", f"{cs['raw_bytes']:,}"),
                    ("whitespace tokens", f"{cs['total_whitespace_tokens']:,}"),
                    ("whitespace types", f"{cs['unique_whitespace_types']:,}"),
                    ("doc length min/median/max", 
                     f"{cs['doc_length']['min']} / {cs['doc_length']['median']} / {cs['doc_length']['max']}"),
                ],
            ),
            bar(
                "Documents per category",
                [(k, v, str(v)) for k, v in sorted(self.s.category_counts().items())],
            ),
            kv(
                f"Index '{index.name}'",
                [
                    ("vocabulary", f"{stats['vocabulary_size']:,}"),
                    ("total postings", f"{stats['total_postings']:,}"),
                    ("total tokens", f"{stats['total_tokens']:,}"),
                    ("hapax legomena", f"{stats['hapax_legomena']:,}"),
                    ("mean df", stats["mean_df"]),
                ],
            ),
            bar(
                "Most frequent terms",
                [
                    (t, c, f"{c:,}")
                    for t, c in index.vocabulary.most_frequent(12)
                ],
            ),
        ]
        built = [n for n in self.s.config_names if self.s.is_built(n)]
        if len(built) > 1:
            blocks.append(
                bar(
                    "Vocabulary size by configuration (built so far)",
                    [
                        (n, len(self.s.index(n).terms), f"{len(self.s.index(n).terms):,}")
                        for n in built
                    ],
                )
            )
        return blocks

    def _index(self, name: str) -> list[dict]:
        if not name:
            return [
                table(
                    "Preprocessing configurations",
                    ["name", "lowercase", "stopwords", "stem", "lemma", "built"],
                    [
                        [
                            c.name,
                            "yes" if c.lowercase else "-",
                            "removed" if c.remove_stopwords else "-",
                            "Porter" if c.stem else "-",
                            "WordNet" if c.lemmatize else "-",
                            "yes" if self.s.is_built(c.name) else "on demand",
                        ]
                        for c in preprocess.CONFIGURATIONS
                    ],
                ),
                text(f"Active: {self.s.active}. Switch with :index <name>."),
            ]
        if name not in preprocess.BY_NAME:
            return [text(f"No configuration named {name!r}.", "error")]
        started = time.perf_counter()
        index = self.s.index(name)
        self.s.active = name
        elapsed = (time.perf_counter() - started) * 1000
        return [
            text(f"Active index is now '{name}' ({elapsed:.0f} ms).", "ok"),
            kv(
                f"Index '{name}'",
                [(k, f"{v:,}" if isinstance(v, int) else v) for k, v in index.stats().items()],
            ),
        ]

    def _term(self, raw: str) -> list[dict]:
        if not raw:
            return [text("Usage: :term <term>", "warn")]
        index = self.s.index()
        surface = raw.split()[0]
        term = self.s.normalizer()(surface)
        postings = index.postings(term)
        blocks = [
            kv(
                f"Term '{surface}'",
                [
                    ("normalised to", term),
                    ("in vocabulary", "yes" if term in index else "no"),
                    ("document frequency", index.df(term)),
                    ("collection frequency", index.vocabulary.cf(term)),
                    ("index", index.name),
                ],
            )
        ]
        if not postings:
            near = self.s.kgram().nearest(term, limit=6)
            if near:
                blocks.append(
                    chips(
                        "Did you mean",
                        [f"{t} (d={d})" for t, d in near],
                        "ranked by edit distance, then collection frequency",
                    )
                )
            return blocks
        blocks.append(text(index.sample_postings(term, limit=25), "mono"))
        blocks.append(self._docs_block(postings, [term], "Documents"))
        blocks.append(self._category_bar(postings))
        return blocks

    def _doc(self, raw: str) -> list[dict]:
        if not raw.strip().isdigit():
            return [text("Usage: :doc <doc_id>", "warn")]
        doc = self.s.document(int(raw.strip()))
        if doc is None:
            return [text(f"No document with id {raw}.", "error")]
        tokens = self.s.index().name
        return [
            kv(
                f"Document {doc.doc_id}",
                [
                    ("category", doc.category),
                    ("file", doc.filename),
                    ("characters", f"{len(doc.text):,}"),
                    ("whitespace tokens", f"{len(doc.text.split()):,}"),
                    (f"tokens under '{tokens}'",
                     f"{len(preprocess.BY_NAME[tokens](doc.text)):,}"),
                ],
            ),
            {"type": "article", "title": doc.filename, "text": doc.text},
        ]

    def _stem(self, raw: str) -> list[dict]:
        if not raw:
            return [text("Usage: :stem <word>", "warn")]
        word = raw.split()[0]
        history = porter.trace(word)
        changed = [
            (label, value)
            for (label, value), (_, previous) in zip(history[1:], history)
            if value != previous
        ]
        return [
            kv(
                f"Porter stem of '{word}'",
                [
                    ("stem", porter.stem(word)),
                    ("measure m", porter.measure(word.lower())),
                    ("steps that fired", ", ".join(l for l, _ in changed) or "none"),
                ],
            ),
            steps("Step by step", [(l, v) for l, v in history]),
        ]

    def _wildcard(self, pattern: str) -> list[dict]:
        if not pattern:
            return [text("Usage: :wild <pat*ern>", "warn")]
        pattern = pattern.split()[0].lower()
        index = self.s.index()
        kg = self.s.kgram()
        started = time.perf_counter()
        matches = kg.wildcard(pattern)
        elapsed = (time.perf_counter() - started) * 1000
        expanded = sorted(matches, key=lambda t: (-index.df(t), t))
        docs = set()
        for term in expanded:
            docs.update(index.postings(term))
        blocks = [
            kv(
                f"Wildcard '{pattern}'",
                [
                    ("matching terms", len(expanded)),
                    ("documents reached", len(docs)),
                    ("k-grams in index", f"{len(kg):,}"),
                    ("time", f"{elapsed:.2f} ms"),
                ],
            )
        ]
        if not expanded:
            return blocks + [text("No vocabulary term matches that pattern.", "warn")]
        blocks.append(
            bar(
                "Expansion, by document frequency",
                [(t, index.df(t), str(index.df(t))) for t in expanded[:15]],
            )
        )
        if len(expanded) > 15:
            blocks.append(chips("Remaining terms", expanded[15:80]))
        blocks.append(self._docs_block(sorted(docs), expanded[:5], "Documents reached"))
        return blocks

    def _spell(self, raw: str) -> list[dict]:
        if not raw:
            return [text("Usage: :spell <word>", "warn")]
        word = raw.split()[0].lower()
        index = self.s.index()
        kg = self.s.kgram()
        started = time.perf_counter()
        near = kg.nearest(word, max_distance=2, limit=8)
        elapsed = (time.perf_counter() - started) * 1000
        blocks = [
            kv(
                f"Spelling candidates for '{word}'",
                [
                    ("in vocabulary", "yes" if word in index else "no"),
                    ("candidates within distance 2", len(near)),
                    ("best correction", near[0][0] if near else "none"),
                    ("time", f"{elapsed:.2f} ms"),
                ],
            )
        ]
        if not near:
            return blocks + [text("Nothing within edit distance 2.", "warn")]
        blocks.append(
            table(
                "Ranked candidates",
                ["term", "edit distance", "df", "collection freq"],
                [
                    [t, d, index.df(t), index.vocabulary.cf(t)]
                    for t, d in near
                ],
            )
        )
        best = near[0][0]
        if best != word:
            blocks.append(
                {
                    "type": "matrix",
                    "title": f"Levenshtein table: {word} to {best}",
                    "rowLabels": [""] + list(word),
                    "colLabels": [""] + list(best),
                    "rows": tolerant.levenshtein_matrix(word, best),
                }
            )
        return blocks

    def _queries(self) -> list[dict]:
        return [
            table(
                "Benchmark queries",
                ["qid", "kind", "query", "relevant docs"],
                [
                    [q.qid, q.kind, q.text, len(self.s.qrels[q.qid])]
                    for q in queryset.QUERIES
                ],
            ),
            text("Score one with :eval q09, or the whole set with :eval all."),
        ]

    def _eval(self, raw: str) -> list[dict]:
        if not raw:
            return [text("Usage: :eval <qid> | :eval all", "warn")]
        index = self.s.index()
        normalize = self.s.normalizer()
        qrels = self.s.qrels

        if raw.lower() == "all":
            scores = []
            by_kind: dict[str, list] = {}
            for q in queryset.QUERIES:
                hits, _ = boolean.execute(q.text, index, normalize, optimize=True)
                s = evaluate.score(q.qid, set(hits), qrels[q.qid])
                scores.append(s)
                by_kind.setdefault(q.kind, []).append(s)
            summary = evaluate.summarise(scores)
            return [
                text(
                    f"All 30 benchmark queries on index '{index.name}', "
                    f"scored against the assignment qrels.",
                ),
                bar(
                    "Macro average",
                    [
                        ("precision", summary["macro"]["precision"], f"{summary['macro']['precision']:.3f}"),
                        ("recall", summary["macro"]["recall"], f"{summary['macro']['recall']:.3f}"),
                        ("F1", summary["macro"]["f1"], f"{summary['macro']['f1']:.3f}"),
                    ],
                    unit="ratio",
                ),
                bar(
                    "Micro average",
                    [
                        ("precision", summary["micro"]["precision"], f"{summary['micro']['precision']:.3f}"),
                        ("recall", summary["micro"]["recall"], f"{summary['micro']['recall']:.3f}"),
                        ("F1", summary["micro"]["f1"], f"{summary['micro']['f1']:.3f}"),
                    ],
                    unit="ratio",
                ),
                table(
                    "By query kind (macro)",
                    ["kind", "queries", "precision", "recall", "F1"],
                    [
                        [
                            kind,
                            len(group),
                            f"{evaluate.macro_average(group)['precision']:.3f}",
                            f"{evaluate.macro_average(group)['recall']:.3f}",
                            f"{evaluate.macro_average(group)['f1']:.3f}",
                        ]
                        for kind, group in sorted(by_kind.items())
                    ],
                ),
                table(
                    "Per query",
                    ["qid", "query", "retrieved", "relevant", "TP", "P", "R", "F1"],
                    [
                        [
                            s.query_id,
                            queryset.BY_QID[s.query_id].text,
                            s.retrieved,
                            s.relevant,
                            s.true_positives,
                            f"{s.precision:.3f}",
                            f"{s.recall:.3f}",
                            f"{s.f1:.3f}",
                        ]
                        for s in scores
                    ],
                ),
            ]

        qid = raw.split()[0].lower()
        if qid not in queryset.BY_QID:
            return [text(f"No benchmark query {qid!r}. See :queries.", "error")]
        query = queryset.BY_QID[qid]
        hits, stats = boolean.execute(query.text, index, normalize, optimize=True)
        s = evaluate.score(qid, set(hits), qrels[qid])
        relevant = qrels[qid]
        retrieved = set(hits)
        terms = [normalize(t) for t in boolean.all_terms(boolean.parse(query.text))]
        return [
            kv(
                f"{qid} — {query.text}",
                [("kind", query.kind), ("index", index.name),
                 ("merge order", " ".join(stats.order)), ("time", f"{stats.elapsed_ms:.2f} ms")],
            ),
            self._metrics_block(s),
            bar(
                "Set breakdown",
                [
                    ("true positives", len(retrieved & relevant), str(len(retrieved & relevant))),
                    ("false positives", len(retrieved - relevant), str(len(retrieved - relevant))),
                    ("false negatives", len(relevant - retrieved), str(len(relevant - retrieved))),
                ],
            ),
            self._docs_block(sorted(retrieved & relevant)[:12], terms, "Correct hits"),
            self._docs_block(sorted(relevant - retrieved)[:12], terms, "Missed (false negatives)"),
        ]

    def _compare(self, query: str) -> list[dict]:
        if not query:
            return [text("Usage: :compare <query>", "warn")]
        index = self.s.index()
        normalize = self.s.normalizer()
        naive, naive_stats = boolean.execute(
            query, index, normalize, optimize=False, expand=self.s.kgram().wildcard
        )
        opt, opt_stats = boolean.execute(
            query, index, normalize, optimize=True, expand=self.s.kgram().wildcard
        )
        agree = set(naive) == set(opt)
        saved = naive_stats.comparisons - opt_stats.comparisons
        pct = (saved / naive_stats.comparisons * 100) if naive_stats.comparisons else 0.0
        return [
            text(
                "Both strategies returned the same document set."
                if agree
                else "STRATEGIES DISAGREE — this is a bug, not a result.",
                "ok" if agree else "error",
            ),
            table(
                f"Execution of {boolean.render(boolean.parse(query))}",
                ["strategy", "results", "comparisons", "postings touched", "ms", "term order"],
                [
                    [
                        "naive (as written)",
                        len(naive),
                        f"{naive_stats.comparisons:,}",
                        f"{naive_stats.postings_touched:,}",
                        f"{naive_stats.elapsed_ms:.3f}",
                        " ".join(naive_stats.order),
                    ],
                    [
                        "optimised (shortest first)",
                        len(opt),
                        f"{opt_stats.comparisons:,}",
                        f"{opt_stats.postings_touched:,}",
                        f"{opt_stats.elapsed_ms:.3f}",
                        " ".join(opt_stats.order),
                    ],
                ],
            ),
            bar(
                "Comparisons performed",
                [
                    ("naive", naive_stats.comparisons, f"{naive_stats.comparisons:,}"),
                    ("optimised", opt_stats.comparisons, f"{opt_stats.comparisons:,}"),
                ],
            ),
            text(
                f"Ordering AND arguments by postings length saved {saved:,} comparisons "
                f"({pct:.1f}%).",
                "ok" if saved >= 0 else "warn",
            ),
        ]

    # -------------------------------------------------------- default query

    def _rewrite_free_text(self, query: str) -> tuple[str, list[str], list[str]]:
        """Turn a sentence into a Boolean query the index can actually answer.

        Words the active pipeline deletes are dropped rather than passed through.
        That matters more than it looks: boolean.normalizer_for falls back to the
        raw lower-cased word when the pipeline returns nothing, so a stop word
        left in a conjunction either has df=0 and empties the whole query, or
        collides with a stem and matches the wrong thing.
        """
        config = preprocess.BY_NAME[self.s.active]
        kept: list[str] = []
        dropped: list[str] = []
        for token in boolean.tokenize_query(query):
            (kept if config(token) else dropped).append(token)
        return " AND ".join(kept), kept, dropped

    def _query(self, query: str) -> list[dict]:
        index = self.s.index()
        normalize = self.s.normalizer()

        free_text = _EXPLICIT_SYNTAX.search(query) is None
        preamble: list[dict] = []
        if free_text:
            rewritten, kept, dropped = self._rewrite_free_text(query)
            if not kept:
                return [
                    text(
                        "Every word in that query is a stop word, so the "
                        f"'{index.name}' pipeline deletes all of them. Nothing is "
                        "left to search for.",
                        "warn",
                    )
                ]
            if dropped or len(kept) > 1:
                preamble.append(
                    kv(
                        "Read as free text",
                        [
                            ("you typed", query),
                            ("searched for", rewritten),
                            ("dropped as stop words", ", ".join(dropped) or "none"),
                        ],
                    )
                )
            query = rewritten

        node = boolean.parse(query)
        surface_terms = list(boolean.all_terms(node))
        expand = self.s.kgram().wildcard
        hits, stats = boolean.execute(
            query, index, normalize, optimize=True, expand=expand
        )

        started = time.perf_counter()
        gold = _lexical_gold(node, self.s.docs, set(index.all_doc_ids))
        gold_ms = (time.perf_counter() - started) * 1000

        retrieved = set(hits)
        score = evaluate.score("adhoc", retrieved, gold)
        stems = [normalize(t) for t in surface_terms]
        unknown = [t for t, st in zip(surface_terms, stems) if st not in index]

        blocks = preamble + [
            kv(
                "Query",
                [
                    ("parsed as", boolean.render(node)),
                    ("index", index.name),
                    ("terms after normalisation", ", ".join(stems)),
                    ("merge order (smallest first)", " ".join(stats.order)),
                    ("results", len(hits)),
                    ("comparisons", f"{stats.comparisons:,}"),
                    ("index time", f"{stats.elapsed_ms:.2f} ms"),
                    ("raw-text oracle time", f"{gold_ms:.0f} ms"),
                ],
            )
        ]

        if unknown:
            suggestions = []
            kg = self.s.kgram()
            for term in unknown:
                near = kg.nearest(normalize(term), limit=3)
                if near:
                    suggestions.append(
                        f"{term} → " + ", ".join(f"{t} (d={d})" for t, d in near)
                    )
            blocks.append(
                text(
                    f"Not in the '{index.name}' vocabulary: {', '.join(unknown)}.",
                    "warn",
                )
            )
            if suggestions:
                blocks.append(chips("Nearest vocabulary terms", suggestions))

        if retrieved or gold:
            blocks.append(
                text(
                    "Validation compares the index answer against the same query run "
                    "directly over the raw article text with whole-word matching.",
                )
            )
            blocks.append(self._metrics_block(score, title="Index vs raw-text oracle"))
            blocks.append(
                bar(
                    "Agreement",
                    [
                        ("both agree", len(retrieved & gold), str(len(retrieved & gold))),
                        ("index only", len(retrieved - gold), str(len(retrieved - gold))),
                        ("oracle only", len(gold - retrieved), str(len(gold - retrieved))),
                    ],
                )
            )
        else:
            # Scoring two empty sets would report 0.000 across the board and read
            # as a failure, when it actually means the two methods agree.
            blocks.append(
                text(
                    "The raw-text oracle finds nothing either, so the empty result "
                    "is the query, not the index.",
                )
            )
        extra = sorted(retrieved - gold)
        missing = sorted(gold - retrieved)
        if extra:
            blocks.append(
                text(
                    f"{len(extra)} document(s) matched only through the index. With a "
                    "stemmed index this is usually morphological recall, not an error.",
                )
            )
            blocks.append(self._docs_block(extra[:8], stems, "Index-only documents"))
        if missing:
            blocks.append(
                self._docs_block(missing[:8], surface_terms, "Oracle-only documents")
            )

        if hits:
            blocks.append(self._category_bar(hits))
            blocks.append(self._docs_block(hits, stems, "Results"))
        else:
            blocks.append(
                text(
                    "No document contains every one of those terms. An AND is only "
                    "as large as its smallest conjunct.",
                    "warn",
                )
            )
            blocks.append(self._term_diagnostic(surface_terms))
            blocks.extend(self._coordination_blocks(surface_terms))
        return blocks

    # ------------------------------------------------------- empty-result aid

    def _term_diagnostic(self, surface_terms: list[str]) -> dict:
        """Which conjunct emptied the intersection, and why."""
        config = preprocess.BY_NAME[self.s.active]
        normalize = self.s.normalizer()
        index = self.s.index()
        rows = []
        for term in dict.fromkeys(surface_terms):
            normalised = normalize(term)
            df = index.df(normalised)
            if not config(term):
                note = f"stop word: deleted by '{index.name}', so it can never match"
            elif df == 0:
                note = "not in the vocabulary: empties every AND it appears in"
            else:
                note = "ok"
            rows.append([term, normalised, df, note])
        return table(
            "Term by term", ["term", "normalised", "df", "note"], rows
        )

    def _coordination_blocks(self, surface_terms: list[str]) -> list[dict]:
        """Relax the AND: group documents by how many query terms they contain.

        Strict Boolean retrieval has no notion of a partial match, so a long
        query is all-or-nothing. Counting matched terms per document is the
        cheapest way to show what the query *nearly* found, and it turns an
        empty result into a diagnosis.
        """
        index = self.s.index()
        normalize = self.s.normalizer()
        stems = list(dict.fromkeys(normalize(t) for t in surface_terms))
        if len(stems) < 2:
            return []

        matched = Counter()
        for stem in stems:
            for doc_id in index.postings(stem):
                matched[doc_id] += 1
        if not matched:
            return []

        by_level: dict[int, list[int]] = {}
        for doc_id, count in matched.items():
            by_level.setdefault(count, []).append(doc_id)
        total = len(stems)
        best = max(by_level)

        return [
            bar(
                "Documents by number of query terms matched",
                [
                    (
                        f"{level} of {total} terms",
                        len(by_level[level]),
                        str(len(by_level[level])),
                    )
                    for level in sorted(by_level, reverse=True)
                ],
            ),
            text(
                f"No document reaches {total} of {total}. The closest are the "
                f"{len(by_level[best])} document(s) matching {best}.",
            ),
            self._docs_block(
                sorted(by_level[best]), stems, f"Best partial matches ({best} of {total})"
            ),
        ]

    # ------------------------------------------------------------- helpers

    def _metrics_block(self, score, title: str = "Effectiveness") -> dict:
        return {
            "type": "metrics",
            "title": title,
            "precision": round(score.precision, 4),
            "recall": round(score.recall, 4),
            "f1": round(score.f1, 4),
            "retrieved": score.retrieved,
            "relevant": score.relevant,
            "true_positives": score.true_positives,
        }

    def _category_bar(self, doc_ids) -> dict:
        counts: dict[str, int] = {}
        for doc_id in doc_ids:
            doc = self.s.document(doc_id)
            if doc:
                counts[doc.category] = counts.get(doc.category, 0) + 1
        return bar(
            "Results by category",
            [(k, v, str(v)) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])],
        )

    def _docs_block(self, doc_ids, terms: list[str], title: str) -> dict:
        shown = list(doc_ids)[:MAX_DOCS_SHOWN]
        items = []
        for doc_id in shown:
            doc = self.s.document(doc_id)
            if doc is None:
                continue
            items.append(
                {
                    "doc_id": doc.doc_id,
                    "category": doc.category,
                    "filename": doc.filename,
                    "title": doc.text.strip().splitlines()[0][:110] if doc.text.strip() else doc.filename,
                    "snippet": _snippet(doc.text, terms),
                }
            )
        return {
            "type": "docs",
            "title": title,
            "items": items,
            "total": len(list(doc_ids)),
            "highlights": [t for t in terms if t],
        }
