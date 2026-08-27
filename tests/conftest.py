import pathlib

import nltk

ROOT = pathlib.Path(__file__).resolve().parents[1]
nltk.data.path.append(str(ROOT / "nltk_data"))
