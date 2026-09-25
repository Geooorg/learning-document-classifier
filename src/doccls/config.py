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
