"""Pfade und Schwellwerte. Alles über Umgebungsvariablen überschreibbar, nichts hart kodiert."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("DOCCLS_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
"""Originale. Werden nur gelesen, nie verändert."""

PARQUET_DIR = DATA_DIR / "parquet"
GENERATED_DIR = DATA_DIR / "generated"

MIN_CHARS_PER_PAGE = int(os.environ.get("DOCCLS_MIN_CHARS_PER_PAGE", "40"))
"""Liefert eine Seite weniger Zeichen, gilt das Dokument als gescannt und braucht OCR.

Der Wert trennt zwei weit auseinanderliegende Fälle, keine knappe Grenze: Ein PDF mit
Textebene liefert im Bestand minimal 112,7 Zeichen je Seite, ein Scan ohne Textebene
praktisch null. 40 liegt mit fast dreifachem Abstand unter dem Bestandsminimum und
weit über dem Scan-Fall.

Die ursprünglich geplanten 120 lagen mitten in der Verteilung der echten Dokumente
(113 bis 157 Zeichen je Seite) und markierten 30 von 230 PDF mit Textebene
fälschlich als OCR-bedürftig. ``test_kein_erzeugtes_pdf_gilt_als_ocr_beduerftig``
hält diesen Abstand fest."""

FEATURES_DIR = DATA_DIR / "features"
"""Merkmalsmatrizen je feature_version."""

MODELS_DIR = DATA_DIR / "models"
"""Modellartefakte je Lauf."""

REPORTS_DIR = DATA_DIR / "reports"
"""Diagramme und Kennzahlen je Lauf."""

E5_MODEL_NAME = os.environ.get("DOCCLS_E5_MODEL", "intfloat/multilingual-e5-base")
"""768 Dimensionen, 512 Token Kontext. Verlangt das Präfix ``passage: `` (Konzept § 6.1)."""

OLLAMA_URL = os.environ.get("DOCCLS_OLLAMA_URL", "http://localhost:11434")
BGE_M3_MODEL_NAME = os.environ.get("DOCCLS_BGE_M3_MODEL", "bge-m3:567m-fp16")
"""1024 Dimensionen, 8192 Token Kontext, keine Präfixe. Läuft über einen Ollama-Dienst."""
