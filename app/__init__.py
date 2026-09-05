"""Chat app package: a local browser front end over src/ir.

The path and NLTK bootstrap lives here rather than in a submodule so that
importing anything in this package makes `ir` importable, whichever entry point
was used.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import nltk  # noqa: E402

_NLTK_DATA = str(ROOT / "nltk_data")
if _NLTK_DATA not in nltk.data.path:
    nltk.data.path.append(_NLTK_DATA)
