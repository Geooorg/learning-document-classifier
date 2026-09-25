"""Pfade und Schwellwerte. Alles über Umgebungsvariablen überschreibbar, nichts hart kodiert."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("DOCCLS_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
"""Originale. Werden nur gelesen, nie verändert."""

PARQUET_DIR = DATA_DIR / "parquet"
GENERATED_DIR = DATA_DIR / "generated"

MIN_CHARS_PER_PAGE = int(os.environ.get("DOCCLS_MIN_CHARS_PER_PAGE", "120"))
"""Liefert eine Seite weniger Zeichen, gilt das Dokument als gescannt und braucht OCR."""
