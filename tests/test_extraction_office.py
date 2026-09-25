"""Extraktion arbeitet auf Bytes, nicht auf Pfaden – prüfbar ohne Dateien auf der Platte."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from doccls.config import RAW_DIR
from doccls.extraction import extract
from doccls.models import Document, SegmentKind


def dokument(pfad: Path, media_type: str) -> tuple[Document, bytes]:
    daten = pfad.read_bytes()
    return Document.create(
        source_path=pfad.name,
        media_type=media_type,
        content=daten,
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    ), daten


def erste(ordner: str) -> Path:
    dateien = sorted((RAW_DIR / ordner).glob(f"*.{ordner}"))
    assert dateien, f"Keine Testdaten in {RAW_DIR / ordner}"
    return dateien[0]


def test_pdf_ergibt_ein_segment_je_seite() -> None:
    doc, daten = dokument(erste("pdf"), "application/pdf")
    segmente = extract(doc, daten)
    assert segmente
    assert all(s.kind is SegmentKind.SEITE for s in segmente)
    assert [s.index for s in segmente] == list(range(len(segmente)))
    assert segmente[0].locator == "S. 1"


def test_pdf_erfasst_alle_seiten() -> None:
    """Ein Abbruch der Seitenschleife nach der ersten Seite läge gemessen an den übrigen
    Prüfungen unbemerkt: Sie fordern eine fortlaufende Indexfolge und ``S. 1`` als erste
    Fundstelle, beides bliebe auch bei einem einzigen erfassten Segment wahr. Deshalb wird
    hier zusätzlich gegen die tatsächliche Seitenzahl der Quelle geprüft.
    """
    import pymupdf

    pfad = erste("pdf")
    doc, daten = dokument(pfad, "application/pdf")
    segmente = extract(doc, daten)
    with pymupdf.open(stream=daten, filetype="pdf") as pdf:
        seitenzahl = len(pdf)
    assert seitenzahl > 1, "Testdatei hat nur eine Seite – Prüfung liefe leer"
    assert len(segmente) == seitenzahl


def test_docx_liefert_abschnitte_mit_ueberschriften() -> None:
    doc, daten = dokument(
        erste("docx"),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    segmente = extract(doc, daten)
    assert any(s.heading for s in segmente)
    assert all(s.kind is SegmentKind.ABSCHNITT for s in segmente)


def test_xlsx_liefert_ein_segment_je_blatt() -> None:
    doc, daten = dokument(
        erste("xlsx"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    segmente = extract(doc, daten)
    assert segmente
    assert all(s.kind is SegmentKind.BLATT for s in segmente)


def test_tabellenzeile_bleibt_als_einheit_beisammen() -> None:
    """Zerfiele eine Zeile in Zellen, verlöre man den Bezug zwischen Leistung und Betrag –
    und genau dieser Bezug unterscheidet eine Rechnung von einer Liste.

    Geprüft wird deshalb, dass Spaltenname und Wert im **selben** Segment stehen. Eine
    Suche nach "EUR" im zusammengefügten Gesamttext wäre auch dann wahr, wenn jede Zelle
    ein eigenes Segment bekäme – also genau im Fehlerfall.
    """
    import re

    pfad = RAW_DIR / "xlsx" / "RECHNUNG-tabelle-00.xlsx"
    assert pfad.exists(), f"{pfad} fehlt – erst den Generator laufen lassen"
    doc, daten = dokument(pfad, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    treffer = [s for s in extract(doc, daten) if "Einzelpreis:" in s.text]
    assert treffer, "Kein Segment enthält die Spalte 'Einzelpreis'"
    zeile = treffer[0].text
    assert "Leistung:" in zeile, "Spalten derselben Zeile liegen in verschiedenen Segmenten"
    assert re.search(r"Einzelpreis: [\d.,]+ EUR", zeile), "Spaltenname und Wert sind getrennt"


def test_text_ist_normalisiert() -> None:
    doc, daten = dokument(erste("pdf"), "application/pdf")
    segmente = extract(doc, daten)
    assert segmente, "Ohne Segmente prüft die Schleife nichts"
    for segment in segmente:
        assert "  " not in segment.text
        assert "\n" not in segment.text


def test_segment_ids_sind_eindeutig() -> None:
    doc, daten = dokument(erste("pdf"), "application/pdf")
    segmente = extract(doc, daten)
    assert len({s.segment_id for s in segmente}) == len(segmente)


def test_unbekanntes_format_wird_abgelehnt() -> None:
    doc, _ = dokument(erste("pdf"), "application/pdf")
    with pytest.raises(ValueError, match="Nicht unterstützt"):
        extract(doc, b"\x00\x01\x02 irgendwas")


# Aufgabe 3 verlangt deterministische index-Vergabe: derselbe Bytes-Input muss bei jeder
# Extraktion dieselbe Reihenfolge und Indexfolge ergeben. Bibliotheken wie openpyxl
# garantieren das nicht von sich aus für jede Konstellation – deshalb wird hier gemessen,
# nicht angenommen.
DATEIEN_JE_FORMAT: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@pytest.mark.parametrize("ordner,media_type", DATEIEN_JE_FORMAT.items())
def test_extraktion_ist_deterministisch(ordner: str, media_type: str) -> None:
    doc, daten = dokument(erste(ordner), media_type)
    erster_lauf = extract(doc, daten)
    zweiter_lauf = extract(doc, daten)
    assert erster_lauf, "Ohne Segmente prüft der Vergleich nichts"
    assert erster_lauf == zweiter_lauf
    assert [s.index for s in erster_lauf] == [s.index for s in zweiter_lauf]
