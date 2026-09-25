"""PDF: eine Seite wird ein Segment.

Die Seite bleibt die Einheit, weil die Seitenzahl die einzige Fundstelle ist, die ein
Mensch im Original wiederfindet. Die Überschrift wird an der Schriftgröße erkannt – der
größte Text auf der Seite, sofern er deutlich größer ist als der Fließtext.
"""

import pymupdf

from doccls.models import Document, Segment, SegmentKind, normalize_text

MAX_HEADING_LENGTH = 80
"""Längerer Text ist Fließtext, keine Überschrift."""

MIN_HEADING_SIZE_RATIO = 1.15
"""So viel größer als der Fließtext muss eine Zeile sein, um als Überschrift zu gelten."""


def heading_by_font_size(page: pymupdf.Page) -> str | None:
    """Die größte Textzeile der Seite, wenn sie deutlich größer ist als der Rest."""
    zeilen: list[tuple[float, str]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            groessen = [span["size"] for span in line["spans"]]
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text and groessen:
                zeilen.append((max(groessen), text))
    if not zeilen:
        return None
    groesse, text = max(zeilen, key=lambda eintrag: eintrag[0])
    fliesstext = sorted(g for g, _ in zeilen)[len(zeilen) // 2]
    if groesse < fliesstext * MIN_HEADING_SIZE_RATIO or len(text) > MAX_HEADING_LENGTH:
        return None
    return normalize_text(text)


def extract_pdf(document: Document, data: bytes) -> list[Segment]:
    segmente: list[Segment] = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for seitennummer, seite in enumerate(pdf, start=1):
            text = normalize_text(seite.get_text("text"))
            if not text:
                continue  # Leere oder rein grafische Seiten übergehen; needs_ocr fängt sie.
            segmente.append(
                Segment.create(
                    document=document,
                    index=len(segmente),
                    kind=SegmentKind.SEITE,
                    locator=f"S. {seitennummer}",
                    heading=heading_by_font_size(seite),
                    text=text,
                )
            )
    return segmente
