"""Boolean IR system over the BBC News corpus.

Registers the repository's bundled `nltk_data/` on import. Every entry point
used to do this for itself, which meant importing `ir` any other way -- a
notebook, the Virtual Lab REPL, a grader's own script -- failed on a missing
`stopwords` resource even though the data was sitting in the repo.
"""

from pathlib import Path

import nltk

_BUNDLED_NLTK_DATA = Path(__file__).resolve().parents[2] / "nltk_data"
if _BUNDLED_NLTK_DATA.is_dir() and str(_BUNDLED_NLTK_DATA) not in nltk.data.path:
    nltk.data.path.insert(0, str(_BUNDLED_NLTK_DATA))
