"""DOCX und XLSX.

DOCX: Ein Segment umfasst eine Überschrift und alles bis zur nächsten. Das hält
zusammenhängende Gedanken beieinander und gibt dem Segment eine sprechende Fundstelle.

XLSX: Ein Blatt wird ein Segment. Eine Tabellenzeile wird als ``Spalte: Wert``
geschrieben – zerfiele sie in einzelne Zellen, verlöre man den Bezug zwischen Position,
Menge und Betrag, und genau dieser Bezug unterscheidet eine Rechnung von einer Liste.
"""

import io

import docx
import openpyxl

from doccls.models import Document, Segment, SegmentKind, normalize_text

MIN_HEADER_CELLS = 2
"""Ab so vielen gefüllten Zellen gilt die erste Zeile eines Blattes als Kopfzeile."""


def extract_docx(document: Document, data: bytes) -> list[Segment]:
    quelle = docx.Document(io.BytesIO(data))
    segmente: list[Segment] = []
    ueberschrift: str | None = None
    absaetze: list[str] = []

    def abschliessen() -> None:
        text = normalize_text(" ".join(absaetze))
        if text:
            segmente.append(
                Segment.create(
                    document=document,
                    index=len(segmente),
                    kind=SegmentKind.ABSCHNITT,
                    locator=ueberschrift or f"Abschnitt {len(segmente) + 1}",
                    heading=ueberschrift,
                    text=text,
                )
            )

    for absatz in quelle.paragraphs:
        text = absatz.text.strip()
        if not text:
            continue
        stil = absatz.style.name if absatz.style is not None else None
        if stil is not None and stil.startswith("Heading"):
            abschliessen()
            ueberschrift, absaetze = normalize_text(text), []
        else:
            absaetze.append(text)
    abschliessen()

    for nummer, tabelle in enumerate(quelle.tables, start=1):
        zeilen = [" | ".join(zelle.text.strip() for zelle in zeile.cells) for zeile in tabelle.rows]
        text = normalize_text(" ".join(zeilen))
        if text:
            segmente.append(
                Segment.create(
                    document=document,
                    index=len(segmente),
                    kind=SegmentKind.ABSCHNITT,
                    locator=f"Tabelle {nummer}",
                    heading=None,
                    text=text,
                )
            )
    return segmente


def extract_xlsx(document: Document, data: bytes) -> list[Segment]:
    mappe = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    segmente: list[Segment] = []
    for blatt in mappe.worksheets:
        zeilen = [
            [("" if zelle is None else str(zelle)).strip() for zelle in zeile]
            for zeile in blatt.iter_rows(values_only=True)
        ]
        zeilen = [zeile for zeile in zeilen if any(zeile)]
        if not zeilen:
            continue
        kopf = zeilen[0] if sum(bool(z) for z in zeilen[0]) >= MIN_HEADER_CELLS else None
        rest = zeilen[1:] if kopf else zeilen
        if kopf:
            formatiert = [
                ", ".join(
                    f"{name}: {wert}" for name, wert in zip(kopf, zeile, strict=False) if wert
                )
                for zeile in rest
            ]
            inhalt = " | ".join([" | ".join(kopf), *formatiert])
        else:
            inhalt = " | ".join(" ".join(z for z in zeile if z) for zeile in rest)
        text = normalize_text(inhalt)
        if text:
            segmente.append(
                Segment.create(
                    document=document,
                    index=len(segmente),
                    kind=SegmentKind.BLATT,
                    locator=f"Blatt {blatt.title}",
                    heading=blatt.title,
                    text=text,
                )
            )
    mappe.close()
    return segmente
