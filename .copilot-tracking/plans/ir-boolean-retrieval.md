# Plan — Boolean IR System + Visual Concept Companion

**Slug:** `ir-boolean-retrieval`
**Source of truth for requirements:** [docs/IR_Assignment.md](docs/IR_Assignment.md)
**Status:** §1 confirmed 2026-08-26. Stage A in progress, Stage B unblocked.

---

## 0. Two parallel deliverables

| ID | Deliverable | Audience | Submitted? |
| --- | --- | --- | --- |
| **D1** | Boolean IR system — source code + technical report + Virtual Lab screenshot | Examiner | Yes |
| **D2** | Interactive visual concept companion — beginner → expert walkthrough of every moving part | You | No |

D2 is not decoration. It is built from **your** corpus and **your** index, so every diagram
shows real postings lists and real numbers. D2 finishing first for a stage is what makes
D1 writable by hand.

---

## 1. Operating model — CONFIRMED 2026-08-26

Assignment §7 states, verbatim:

> Do not use generative AI/LLMs to generate the implementation, experimental results or
> report.

That rules out the default mode of operation for D1. The compliant split below is
**agreed and binding** for the rest of this project:

| Activity | Who | Rationale |
| --- | --- | --- |
| Dataset research, comparison, sourcing | Agent | Not implementation, results, or report |
| Task decomposition, this plan | Agent | Not implementation, results, or report |
| Explaining Porter, postings merge, k-gram, edit distance — concepts, worked examples on paper | Agent | Teaching, not generation |
| **D2 visual companion (HTML/diagrams)** | Agent | Not a submitted artifact |
| **IR algorithm code (`src/`)** | **You**, coached line-by-line | §7 |
| **Running experiments, producing numbers** | **You** | §7 — "experimental results" |
| **Technical report prose** | **You** | §7 — "report" |
| Reviewing your code for bugs, asking you comprehension questions | Agent | Review ≠ generation |

**Coaching contract for every `src/` task.** The agent supplies, per function: the
signature, the job it does, the algorithm in prose or pseudocode, the edge cases, a worked
example traced by hand, and the test assertions. You type the body. The agent then reviews
what you wrote and asks one comprehension question. The agent does not write the body,
even when asked in the moment — if you want that, say `override §1` explicitly and it goes
in the ledger.

**Solo submission.** §6's group wording does not apply; all tasks marked *you* are yours
alone. Plan unchanged otherwise — the split was never about splitting labour between
people, it was about §7.

**Pace (confirmed):** function by function. One brief → you write the body → agent
reviews → one comprehension check → next function.

### Override log

| Date | Scope | Requested | Effect |
| --- | --- | --- | --- |
| 2026-08-26 | **T1 only** — `src/ir/corpus.py` + `tests/test_corpus.py` | `override §1` | Agent wrote the function bodies. Corpus loading and profiling only. |
| 2026-08-26 | **Widened to T3–T9 + D2** — full `src/ir/`, tests, experiments, visual companion | "do all the steps, create the visual, then we go through together" | Agent implements the complete pipeline and generates experiment outputs. User reviews afterwards rather than writing bodies. §1's coaching split is suspended for implementation; the walkthrough obligation remains. |

Anything written under override must be declared in the report's §7 library-vs-own-code
table as agent-assisted, not hand-written. That is the cost of the override and it is
non-negotiable — an undeclared override is the thing §7 actually prohibits. With the
widened scope this now covers essentially the whole codebase, so the honest declaration
is: implementation agent-generated, reviewed and understood by the student.

---

## 2. Dataset decision

### Requirement recap
Substantial, publicly available, **one domain**, with variation in vocabulary, document
length and topics.

### Options considered

| Candidate | Docs | Domain | Readable by a human? | Verdict |
| --- | --- | --- | --- | --- |
| Cranfield (`cran.tar.gz`) — your sample | 1,400 | Aeronautics abstracts | Poor — dense jargon, ~100 words each | **Reject.** Ships qrels, but abstracts are near-uniform in length, vocabulary is narrow technical, and you cannot eyeball a doc to judge relevance. Kills the "document length variation" requirement and makes error analysis guesswork. |
| CISI | 1,460 | Information science | Moderate | Reject — same abstract-shaped problem as Cranfield. |
| Reuters-21578 | 21,578 | Newswire | Moderate | Reject — many stubs under 20 words, messy SGML, ~10× more than you need to reason about. |
| 20 Newsgroups | 18,846 | 20 topics | Good | Reject — explicitly **multi**-domain, violates "work within one domain". |
| **BBC News Full-Text** | **2,225** | **News** (single domain) | **Excellent — full prose articles** | **Selected.** |

### Selected: BBC News full-text corpus

- **Source:** UCD Machine Learning Group — <http://mlg.ucd.ie/datasets/bbc.html>
- **Download:** `bbc-fulltext.zip` — <http://mlg.ucd.ie/files/datasets/bbc.zip> is the
  *pre-processed* matrix form; you must take
  <http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip> instead.
- **Attribution (required in report):** D. Greene and P. Cunningham, *"Practical Solutions
  to the Problem of Diagonal Dominance in Kernel Document Clustering"*, Proc. ICML 2006.
  Copyright in article content remains with the BBC; released for non-commercial research
  use only.
- **Shape:** 2,225 plain `.txt` articles, five sub-topics within the news domain —
  business, entertainment, politics, sport, tech.

### Why this one

1. **You can read it.** Relevance judgments for 30 queries are hand-made. Judging "is this
   article about the Premier League?" takes seconds; judging a boundary-layer
   aerodynamics abstract does not.
2. **Real length variation** — short match reports next to long political analyses. That
   is the requirement Cranfield fails.
3. **Category labels give you a defensible judgment scaffold** — pool candidates by
   category, then hand-verify. Documented pooling beats ad-hoc judging in the report.
4. **Scale is honest** — a few thousand documents, index builds in seconds, and a postings
   list is short enough to print in the report.
5. **News vocabulary stresses the interesting cases** — named entities, morphology
   (`invest/investing/investment/investor`), and misspellings people actually make.

### Known trade-off, to state explicitly in the report
Cranfield ships 225 queries with official relevance judgments; BBC does not. But §3E
requires you to *author* 30 queries and judgments yourself, so no free lunch is lost —
and hand-built judgments over readable text are more defensible than borrowed ones.

### Fallback
If a marker insists on a standard test collection, Cranfield can be added as a **secondary**
corpus purely for a sanity check of P/R/F1 against published qrels. Not planned; noted.

---

## 3. Target repository layout

```
Assignment 1/
├─ docs/
│  ├─ IR_Assignment.md          # done — assignment spec
│  └─ report/                   # D1 technical report (yours)
├─ .copilot-tracking/plans/     # this ledger
├─ data/
│  ├─ raw/bbc/                  # downloaded corpus (git-ignored)
│  └─ derived/                  # index dumps, stats CSVs
├─ src/ir/
│  ├─ corpus.py                 # loading + doc IDs
│  ├─ preprocess.py             # tokenize, normalize, stopwords, stem, lemma
│  ├─ porter.py                 # Porter, hand-written
│  ├─ dictionary.py             # vocabulary + df
│  ├─ index.py                  # inverted index + sorted postings
│  ├─ boolean.py                # parser + merge algorithms + op counters
│  ├─ tolerant.py               # k-gram, wildcard, edit distance
│  └─ evaluate.py               # P / R / F1
├─ queries/
│  ├─ queries.tsv               # ≥30 test queries
│  ├─ misspelled.tsv            # ≥20 corrupted queries
│  └─ qrels.tsv                 # your relevance judgments
├─ experiments/                 # 4 required experiment runners
├─ tests/                       # verification per task
└─ visual/                      # D2 companion
```

---

## 4. Task breakdown

Every task carries **A**nalysis → **I**mplementation → **V**erification, as requested.
`Owner` reflects §1.

### Stage A — foundation (unblocked, no §7 exposure)

---

#### T0 · Environment and corpus acquisition
**Owner:** agent (mechanical) · **Depends on:** —

- **A** — Confirm Python version, choose dependency set. Candidate libraries, each to be
  declared in the report as library-provided: `nltk` (stopword list, WordNet lemmatizer,
  *reference* Porter for cross-checking your own), `matplotlib` (plots). Nothing that
  performs retrieval.
- **I** — Create venv, `pyproject.toml` / `requirements.txt`, `.gitignore`, folder
  skeleton. Download and unpack `bbc-fulltext.zip` into `data/raw/bbc/`.
- **V** — `python -c "import nltk, matplotlib"` exits 0. File count under `data/raw/bbc/`
  equals 2,225 across 5 category folders. Expected: `2225`.
- **Evidence:** command + exit code + file count.

---

#### T1 · Corpus profiling (feeds report §2)
**Owner:** you, coached · **Depends on:** T0

- **A** — Decide doc-ID scheme (`category/filename` → stable integer). Decide encoding
  handling. **Verified fact:** all 2,225 files are valid UTF-8; 647 of them contain the
  two-byte sequence `C2 A3` (`£`), 1,403 occurrences total. The trap is therefore *not* a
  decode crash — it is that Python's `open()` on Windows defaults to the **locale**
  encoding (cp1252), which decodes `£` as `Â£` **silently**. Understand why that produces a
  corrupt vocabulary entry before writing the reader.
- **I** — `corpus.py`: load all docs, assign IDs. Compute raw corpus size on disk, total
  whitespace-token count, unique-type count, and the doc-length distribution.
- **V** — Assert 2,225 docs loaded, zero decode errors, min/median/max length printed.
  Numbers recorded in `data/derived/corpus_stats.json`.
- **Report hook:** §2 Dataset table — source, domain, doc count, corpus size, tokens,
  unique terms.

---

#### T2 · D2 stage 1 — the pipeline map
**Owner:** agent · **Depends on:** T1

**Surface decision (confirmed):** self-contained HTML opened in a browser. Single file,
no network at runtime, no build step. Not Mermaid-in-Markdown.

- **A** — Choose diagram set for `Documents → Preprocessing → Dictionary → Index →
  Postings → Boolean → Tolerant → Evaluation`.
- **I** — `visual/ir-companion.html`, themed per the web-artifacts skill. Stage 1 panel:
  the whole pipeline, clickable, each node showing *what goes in, what comes out, and one
  real example from your corpus*.
- **V** — Opens offline by double-clicking; every pipeline stage from the assignment
  appears; numbers shown match `corpus_stats.json`.

---

### Stage B — core system (gated on §1 confirmation)

---

#### T3 · Text processing — 5 variants · *1.5 marks*
**Owner:** you, coached · **Depends on:** T1

- **A** — Tokenization boundary decisions: hyphens, apostrophes (`don't`), acronyms
  (`U.S.`), digits, currency. Where case normalization loses information
  (`US` → `us` collides with a stopword — a genuine failure to report). Read Porter's
  five steps and hand-trace `relational → relate`, `conditional → condit`,
  `feed → feed` vs `agreed → agree`.
- **I** — `preprocess.py` exposing five composable configurations: raw, +normalize,
  +stopwords, +stem, +lemma. `porter.py` written by hand, not imported.
- **V** — Unit tests: your Porter vs `nltk.PorterStemmer` over the full vocabulary;
  report agreement rate and inspect every disagreement. Expected: >99% agreement, each
  mismatch explained. Plus ≥15 fixed tokenizer edge-case assertions.
- **Report hook:** representative stemming examples; the `US`/`us` class of error.

---

#### T4 · Vocabulary, dictionary, inverted index, postings · *1.5 marks*
**Owner:** you, coached · **Depends on:** T3

- **A** — Dictionary vs postings separation. Why postings stay **sorted by docID** — it is
  what makes the linear merge in T5 possible. Where df lives and why.
- **I** — `dictionary.py`, `index.py`. Build one index per preprocessing variant.
- **V** — Invariants asserted: every postings list strictly ascending and duplicate-free;
  `df(t) == len(postings(t))`; sum of postings lengths equals total token count after
  filtering. Vocabulary size strictly decreases across raw → normalize → stopword → stem.
- **Report hook:** stats table before/after each preprocessing step; a printed sample
  postings list.

---

#### T5 · Boolean retrieval + optimization · *2.0 marks*
**Owner:** you, coached · **Depends on:** T4

- **A** — Grammar for `AND`/`OR`/`NOT`/parentheses and precedence. Shunting-yard or
  recursive descent — pick one, justify. The optimization: process a conjunctive query in
  **increasing order of df**, so the smallest postings list bounds the work. Predict the
  effect before measuring it.
- **I** — `boolean.py`: parser → query tree → evaluation over **your** postings lists,
  with a comparison counter threaded through every merge. Two execution strategies: naive
  left-to-right, and df-ordered.
- **V** — Both strategies return **identical** result sets across all 30 queries (this is
  the correctness gate). Comparison counts and wall-clock recorded per query. Expected:
  df-ordering strictly fewer comparisons on multi-term ANDs.
- **Report hook:** query trace tables; comparisons + time table.

---

#### T6 · Tolerant retrieval · *1.5 marks*
**Owner:** you, coached · **Depends on:** T4

- **A** — Choose the combination: **k-gram index (k=3) for candidate generation +
  Levenshtein for ranking** covers wildcard *and* spelling in one structure, and the
  assignment only demands one. Understand why a bigram index answers `re*val` via
  `$re AND val$`, and why post-filtering is still required.
- **I** — `tolerant.py`: k-gram index over the vocabulary, wildcard → k-gram boolean query
  → regex post-filter; Levenshtein with a distance cap.
- **V** — Wildcard results are a superset of exact matches and contain no term failing the
  pattern. Levenshtein validated against a fixed table of known pairs. Runs over all 20
  misspelled queries without error.

---

#### T7 · Query set and relevance judgments
**Owner:** you · **Depends on:** T5

- **A** — Design 30 queries spanning simple / compound / Boolean / morphological /
  tolerant. Pooling protocol for judgments — document it, since it is defensible only if
  written down.
- **I** — `queries.tsv`, `misspelled.tsv` (≥20 corruptions: transposition, omission,
  doubling, phonetic), `qrels.tsv`.
- **V** — Counts ≥30 and ≥20 asserted. Every query ID in qrels resolves. Judge a 10-query
  sample twice, on different days, and report self-agreement.

---

#### T8 · Evaluation · *part of 1.5*
**Owner:** you, coached · **Depends on:** T7

- **A** — P/R/F1 for **set** retrieval, not ranked. Recall denominator problem when
  judgments are pooled — state the bias honestly.
- **I** — `evaluate.py`, per-query and macro/micro averaged.
- **V** — Hand-computed check on 3 queries matches code output exactly. Degenerate cases
  (empty result, empty relevant set) handled without div-by-zero.

---

### Stage C — experiments, report, demo

---

#### T9 · The four required experiments · *1.5 marks*
**Owner:** you · **Depends on:** T5, T6, T8

One runner per experiment, each emitting a CSV and a plot:

1. Preprocessing effect on vocabulary/index stats **and** retrieval quality.
2. Stemming vs lemmatization — including a hunt for undesirable transformations
   (`university/universe → univers`, `operating system → oper system`).
3. Boolean optimization — comparisons and time.
4. Exact vs tolerant — recall gain against false-positive cost.

- **V** — Each runner reproducible from a clean state; outputs land in `data/derived/`;
  every claim in the report traces to a generated file.

---

#### T10 · D2 stages 2–4 — the deep companion
**Owner:** agent · **Depends on:** T4, T5, T6

Progressive disclosure, beginner → expert:
- **Beginner:** what a term, a posting, a df is — using your actual data.
- **Intermediate:** animated postings merge for AND/OR/NOT, step-counted; Porter applied
  step-by-step to a word you choose.
- **Advanced:** why df-ordering wins, with your measured numbers; k-gram wildcard
  resolution shown as a set intersection; the Levenshtein DP matrix rendered as a grid.
- **Expert:** the trade-off surfaces — index size vs recall, tolerance vs precision — and
  where each design decision in your code sits on them.

- **V** — Every concept in assignment §3 has a panel; all figures sourced from your
  generated CSVs; opens offline.

---

#### T11 · Technical report
**Owner:** you · **Depends on:** T9

Seven sections per §6. Must carry intermediate outputs, sample postings lists, query
traces, calculations, tables, plots. Explicit library-vs-own-code declaration table per §7.

- **V** — Checklist pass against §6 and the marks table; every number cross-checked to a
  file in `data/derived/`.

---

#### T12 · Virtual Lab demo · *1.0 mark*
**Owner:** you · **Depends on:** T11

Access confirmed; deliberately deferred until the system is complete. T0 pre-downloads
NLTK corpora to `nltk_data/` inside the repo so the system stays runnable if the Lab has
no outbound internet.

- **A** — Confirm the Lab's Python version and whether internet-free operation is needed.
- **I** — Run end-to-end in BITS Virtual Lab; capture screenshot.
- **V** — Screenshot shows the system executing successfully, with the terminal and
  visible output in frame.

---

## 5. Marks traceability

| Component | Marks | Tasks |
| --- | --- | --- |
| Text preprocessing and stemming/lemmatization | 1.5 | T3 |
| Vocabulary, dictionary, inverted index & postings | 1.5 | T4 |
| Boolean retrieval & query optimization | 2.0 | T5 |
| Tolerant retrieval | 1.5 | T6 |
| Experiments, evaluation & analysis | 1.5 | T7, T8, T9 |
| Code, report & documentation | 1.0 | T11 |
| Virtual Lab execution screenshot | 1.0 | T12 |

---

## 6. Status ledger

| Task | Status | Evidence |
| --- | --- | --- |
| PDF → Markdown | done | `docs/IR_Assignment.md`, 4 pages transcribed |
| Dataset selection | done | §2 above; `bbc-fulltext.zip` confirmed live |
| §1 checkpoint | done | Confirmed 2026-08-26 |
| T0 | done | 2225 .txt in 5 categories; imports-ok; Python 3.14.4; NLTK 5/5 vendored |
| T1 | done | `corpus_stats.json`: 2,225 docs, 854,490 whitespace tokens |
| T2 / T10 | done | Two surfaces: `visual/ir-walkthrough.html` (11-chapter linear teaching journey, 44 verbatim code blocks, 156 annotations) and `visual/ir-companion.html` (results reference) |
| T3 | done | Porter agrees with NLTK ORIGINAL_ALGORITHM on 99.95% of 9,596 corpus types; 5 disagreements all 2-letter words where we match Porter's own C reference |
| T4 | done | 5 indexes built, invariants asserted; vocabulary 38,266 -> 33,981 -> 33,805 -> 25,049 |
| T5 | done | naive and df-ordered agree on all 30 queries x 5 indexes; 77,943 -> 21,790 comparisons, 72.0% saved |
| T6 | done | k-gram (k=3) + Levenshtein; 18/20 misspellings repaired correctly |
| T7 | done | 30 queries, 20 misspellings, 5,044 judgments; every query has non-empty qrels |
| T8 | done | P/R/F1 macro and micro; hand-checked arithmetic in tests |
| T9 | done | 5 experiment outputs in `data/derived/` |
| Test suite | done | 101 passed, exit 0 |
| T11 report | **yours** | not started |
| T12 Virtual Lab | **yours** | deferred by agreement |

---

## 8. Headline results

| Configuration | Vocabulary | Macro P | Macro R | Macro F1 |
| --- | --- | --- | --- | --- |
| raw | 38,266 | 0.866 | 0.644 | 0.703 |
| normalized | 33,981 | 1.000 | 0.869 | 0.904 |
| nostop | 33,805 | 1.000 | 0.869 | 0.904 |
| stemmed | 25,049 | 0.906 | **0.956** | **0.920** |
| lemmatized | 30,821 | 0.920 | 0.876 | 0.865 |

- **Boolean optimisation:** 77,943 -> 21,790 comparisons over 150 executions, 72.0% saved.
- **Tolerant retrieval:** macro recall 0.133 -> 0.854; micro precision 0.984 -> 0.919.

### Known failures, all verified and documented

1. `technology`/`technologies` -> `technologi` but `technological` -> `technolog`. The 1980
   algorithm has no bridging rule; q26 cannot reach full recall.
2. `player` never reduces to `play`: `play` has measure 1 and step 4's `-er` rule needs
   measure > 1. Step 1c meanwhile turns `play` into `plai`.
3. Lowercasing merges the country `US` into the pronoun `us`, and NLTK's stop list does not
   contain `us`, so both senses share one postings list.
4. Edit distance is frequency-blind: `labuor` -> `labor` not `labour`, `pirce` -> `piece`
   not `price`. Damerau-Levenshtein would fix both; the blindness would remain.
5. Spelling repair must precede stemming. `goverment` stems to `gover`, which exists, so a
   stem-based corrector declines to repair it. Fixed by correcting surface forms first —
   this alone moved macro recall from 0.592 to 0.854.

---

## 7. Resolved decisions

| # | Question | Answer |
| --- | --- | --- |
| 1 | §1 operating model | Confirmed as written |
| 2 | Group submission | Solo — §6 group wording does not apply |
| 3 | Virtual Lab | Access held; T12 deferred to the end |
| 4 | D2 surface | Self-contained HTML in a browser |
