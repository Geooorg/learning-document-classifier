"""Gemeinsame Helfer für die Tests, die auf dem erzeugten Bestand arbeiten.

Die Funktionen hier standen bis zur Einführung der Anhangszeilen im Manifest zweimal im
Baum – einmal in ``test_extraction_mail.py``, einmal in ``test_extraction_office.py``.
Als die Anhänge eine eigene Manifestzeile bekamen, schlug die eine Kopie fehl und die
andere nicht, weil ihre Vorlagen zufällig keinen Anhang tragen. Genau so sieht eine Falle
aus, die beim übernächsten Umbau zuschlägt: zwei Stellen, dieselbe Regel, eine gepflegt.
"""

from pathlib import Path

import polars as pl

from doccls.config import GENERATED_DIR, RAW_DIR

ANHANG_TRENNER = "!"
"""``pipeline.py`` bildet den Pfad eines Mailanhangs als ``<mailpfad>!<anhangname>``."""


def _manifest_ohne_anhaenge() -> pl.DataFrame:
    """Das Manifest, beschränkt auf Zeilen, die eine Datei unter ``RAW_DIR`` haben.

    Eine Anhangszeile steht für ein Dokument, das erst beim Einlesen entsteht (Konzept § 5:
    jeder Anhang wird ein eigenständiges ``Document``). Eine Datei gibt es dafür nicht –
    wer sie zu öffnen versucht, bekommt einen ``FileNotFoundError``.
    """
    manifest = pl.read_parquet(GENERATED_DIR / "manifest.parquet")
    return manifest.filter(~pl.col("source_path").str.contains(ANHANG_TRENNER, literal=True))


def pfade_fuer_vorlage(template_id: str) -> list[Path]:
    """Alle Dateien einer Vorlage, über das Manifest gefunden – nicht über den Dateinamen.

    Der ist seit der Schließung der Dateiname-Abkürzung neutral (``doc-0001.pdf``) und
    verrät weder Klasse noch Vorlage mehr; ``template_id`` steht nur noch im Manifest.
    """
    zeilen = _manifest_ohne_anhaenge().filter(pl.col("template_id") == template_id)
    zeilen = zeilen.sort("source_path")
    assert not zeilen.is_empty(), f"Keine Testdaten für Vorlage {template_id!r}"
    return [RAW_DIR / pfad for pfad in zeilen["source_path"]]


def pfade_fuer_klasse_und_format(class_key: str, format_: str) -> list[Path]:
    """Alle Dateien einer Klasse in einem Format – über mehrere Vorlagen hinweg.

    Gedacht für Eigenschaften, die an der Klasse hängen (etwa dem Betreff), nicht an der
    Vorlage.
    """
    zeilen = _manifest_ohne_anhaenge().filter(
        (pl.col("class_key") == class_key) & (pl.col("format") == format_)
    )
    zeilen = zeilen.sort("source_path")
    assert not zeilen.is_empty(), f"Keine Testdaten für {class_key}/{format_}"
    return [RAW_DIR / pfad for pfad in zeilen["source_path"]]
