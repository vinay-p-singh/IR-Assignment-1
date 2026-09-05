# Boolean Information Retrieval System — BBC News

A Boolean retrieval system built from scratch: tokenizer, Porter stemmer, term
dictionary, inverted index with sorted postings lists, a recursive-descent
Boolean query engine with postings-length optimization, k-gram wildcard and
edit-distance tolerant retrieval, and a set-based evaluation harness.

No search engine library is used. Every retrieval operation runs over postings
lists this code builds. See the library boundary table in
[docs/report/technical_report.md](docs/report/technical_report.md#71-library-boundary).

## Dataset

BBC News full-text corpus — 2,225 articles, 5.0 MB, one domain (news) across
five sub-topics.

- Source: UCD Machine Learning Group — <http://mlg.ucd.ie/datasets/bbc.html>
- Attribution: D. Greene and P. Cunningham, *"Practical Solutions to the Problem
  of Diagonal Dominance in Kernel Document Clustering"*, Proc. ICML 2006
- Licence: article content copyright BBC; non-commercial research use only

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux
pip install -r requirements.txt
```

NLTK data (`stopwords`, `punkt`, `wordnet`, `omw-1.4`,
`averaged_perceptron_tagger`) is bundled in `nltk_data/`, so no download is
needed and the system runs offline.

### If you are working from a git clone

`data/raw/` and `nltk_data/` are both gitignored — the first because BBC
article content is copyright and must not be redistributed, the second because
it is 114 MB. A clone therefore has neither, and nothing will run until you
restore them:

```bash
# corpus: download and unzip so that data/raw/bbc/<category>/*.txt exists
curl -O http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip

# NLTK data, into the repo-local nltk_data/ that src/ir/__init__.py registers
python -c "import nltk; [nltk.download(p, download_dir='nltk_data') for p in ['stopwords','punkt','punkt_tab','wordnet','omw-1.4','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"
```

The submitted archive ships both directories already populated, so this step is
only for clones.

## Running

```bash
python demo.py                        # assignment-aligned demonstration (~2 min)
python experiments/run_experiments.py # full result set -> data/derived + figures
python -m pytest -q                   # 109 tests
python -m app.server                  # interactive chat UI at http://127.0.0.1:8765
```

`demo.py` prints one section per assignment requirement and is the script to
capture for the Virtual Lab screenshot.

`IR_Assignment1_Group35.ipynb` is the executable notebook companion: open it,
select the project `.venv` interpreter, and choose **Run All**. It calls the
same `src/ir/` modules and writes its artefacts to `data/derived/notebook/` so
it never overwrites the canonical experiment outputs.

## Layout

| Path | Contents |
| --- | --- |
| `src/ir/` | The retrieval system — corpus, preprocess, porter, dictionary, index, boolean, tolerant, morphology, evaluate, queryset |
| `experiments/run_experiments.py` | Runs every experiment, writes `data/derived/` and the report figures |
| `demo.py` | Single-command demonstration |
| `IR_Assignment1_Group35.ipynb` | Executable notebook walkthrough of the full pipeline |
| `tests/` | 109 tests, including NLTK cross-checks for the Porter implementation |
| `docs/report/` | Technical report and figures |
| `data/derived/` | Generated results (CSV/JSON) — nothing here is hand-written |
| `queries/` | Generated `queries.tsv`, `misspelled.tsv`, `qrels.tsv` |
| `app/` | Interactive chat UI over the same modules |
| `visual/` | Standalone HTML concept walkthroughs |

## Results at a glance

| | |
| --- | --- |
| Vocabulary | 38,266 raw → 25,049 stemmed |
| Best configuration | stemmed — macro F1 0.918 over 33 queries |
| Boolean optimization | 90,071 → 32,263 comparisons (64.2%), 91.9% of it from deferred `NOT` |
| Tolerant retrieval | macro recall 0.133 → 0.854 on 20 misspelled queries |

Full analysis: [docs/report/technical_report.md](docs/report/technical_report.md).
