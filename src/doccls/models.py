"""Datenmodelle und die zugehörigen Polars-Schemas.

Zwei Ebenen:

* ``Document`` – eine Version einer Datei. Die Identität umfasst Pfad **und** Inhalts-Hash:
  Dieselbe Datei an zwei Stellen sind zwei Dokumente, sonst würde die zweite übersprungen
  und wäre nirgends verzeichnet. Geänderter Inhalt ergibt eine neue ID; die alte Version
  bleibt erhalten.
* ``Segment``  – eine Einheit im Dokument: PDF-Seite, Abschnitt, Tabellenblatt, Mailteil.
  Ergebnis der Extraktion. Wird gespeichert, damit Merkmale neu gebildet werden können,
  ohne die Originale erneut zu lesen.

IDs sind aus dem Inhalt abgeleitet und damit stabil: Ein erneuter Lauf erzeugt dieselben
IDs und überschreibt, statt zu verdoppeln.
"""

import hashlib
import re
import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import Self

import polars as pl
from pydantic import BaseModel

ID_LENGTH = 32
"""Hex-Zeichen abgeleiteter IDs. 32 Zeichen = 128 Bit, für Kollisionen ausreichend."""

HYPHENATION = re.compile(r"(\w)-\s*\n\s*(\w)")
"""Silbentrennung am Zeilenende: „Rech-\\nnung“. Ohne Auflösung zerfällt jedes zweite
Fachwort in zwei unbekannte Bruchstücke."""

WHITESPACE = re.compile(r"\s+")


def derive_id(*parts: bytes | str) -> str:
    """ID aus dem Inhalt ableiten. Teile werden mit ``\\0`` getrennt, damit „ab“+„c“ und
    „a“+„bc“ verschiedene IDs ergeben."""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8") if isinstance(part, str) else part)
        hasher.update(b"\0")
    return hasher.hexdigest()[:ID_LENGTH]


def normalize_text(text: str) -> str:
    """NFKC, Silbentrennung auflösen, Whitespace zusammenfassen.

    NFKC vereinheitlicht Ligaturen („ﬁ“ → „fi“) und Kompatibilitätszeichen, die je nach
    PDF-Erzeuger verschieden kodiert sind. Ohne diesen Schritt wären zwei gleich aussehende
    Dokumente verschiedene Zeichenfolgen.
    """
    text = unicodedata.normalize("NFKC", text)
    text = HYPHENATION.sub(r"\1\2", text)
    return WHITESPACE.sub(" ", text).strip()


ANHANG_TRENNER = "!"
"""Trennt im ``source_path`` eines Mailanhangs den Pfad der Elternmail vom Anhangnamen
(``eml/doc-0001.eml!doc-0001.pdf``). Sowohl ``pipeline.py`` (bildet den Pfad beim
tatsächlichen Einlesen) als auch ``generation/manifest.py`` (bildet ihn vorweg für die
Wahrheit) brauchen exakt dieselbe Regel – zwei unabhängige Kopien liefen irgendwann
auseinander, und der Fehler wäre unsichtbar: eine Manifestzeile, die kein ``join`` mehr
trifft."""


def anhang_pfad(eltern_pfad: str, anhang_name: str) -> str:
    """Den ``source_path`` eines Mailanhangs aus dem Pfad der Elternmail und dem Anhangnamen
    bilden – die einzige Stelle, die das Trennzeichen kennt."""
    return f"{eltern_pfad}{ANHANG_TRENNER}{anhang_name}"


class SegmentKind(StrEnum):
    """Was für eine Einheit ein Segment ist. Bestimmt, wie ``locator`` zu lesen ist."""

    SEITE = "seite"
    ABSCHNITT = "abschnitt"
    BLATT = "blatt"
    ZEILE = "zeile"
    MAIL_KOPF = "mail_kopf"
    MAIL_KOERPER = "mail_koerper"


class Document(BaseModel):
    """Eine Version einer Datei."""

    model_config = {"frozen": True}

    document_id: str
    source_path: str
    """Pfad relativ zu ``RAW_DIR``. Bei Anhängen: ``eltern.eml!anhang.pdf``."""

    file_name: str
    media_type: str
    content_sha256: str
    size_bytes: int
    parent_document_id: str | None = None
    """Gesetzt bei E-Mail-Anhängen. Eine Mail und ihre Rechnung sind zwei Dokumente."""

    ingested_at: datetime
    needs_ocr: bool = False
    """Zu wenig Text je Seite – das Dokument ist vermutlich gescannt."""

    ocr_applied: bool = False

    @classmethod
    def create(
        cls,
        *,
        source_path: str,
        media_type: str,
        content: bytes,
        ingested_at: datetime,
        parent_document_id: str | None = None,
    ) -> Self:
        return cls(
            document_id=derive_id(source_path, content),
            source_path=source_path,
            file_name=source_path.rsplit("/", 1)[-1],
            media_type=media_type,
            content_sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            parent_document_id=parent_document_id,
            ingested_at=ingested_at,
        )


class Segment(BaseModel):
    """Eine Einheit im Dokument, Ergebnis der Extraktion."""

    model_config = {"frozen": True}

    segment_id: str
    document_id: str
    index: int
    kind: SegmentKind
    locator: str
    """Menschenlesbare Fundstelle: „S. 3“, „Blatt Positionen“, „Anhang 2“."""

    heading: str | None
    text: str

    @classmethod
    def create(
        cls,
        *,
        document: Document,
        index: int,
        kind: SegmentKind,
        locator: str,
        text: str,
        heading: str | None = None,
    ) -> Self:
        return cls(
            segment_id=derive_id(document.document_id, str(index)),
            document_id=document.document_id,
            index=index,
            kind=kind,
            locator=locator,
            heading=heading,
            text=text,
        )


SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "documents": {
        "document_id": pl.String(),
        "source_path": pl.String(),
        "file_name": pl.String(),
        "media_type": pl.String(),
        "content_sha256": pl.String(),
        "size_bytes": pl.Int64(),
        "parent_document_id": pl.String(),
        "ingested_at": pl.Datetime(time_unit="us", time_zone="UTC"),
        "needs_ocr": pl.Boolean(),
        "ocr_applied": pl.Boolean(),
    },
    "segments": {
        "segment_id": pl.String(),
        "document_id": pl.String(),
        "index": pl.Int64(),
        "kind": pl.String(),
        "locator": pl.String(),
        "heading": pl.String(),
        "text": pl.String(),
    },
}
"""Schemas explizit, nicht abgeleitet: Eine leere Tabelle bekommt sonst andere Spaltentypen
als eine gefüllte, und der zweite Lauf scheitert am ersten."""


def to_frame(rows: list[Document] | list[Segment], name: str) -> pl.DataFrame:
    """Modelle in einen DataFrame mit festem Schema überführen – auch wenn ``rows`` leer ist."""
    return pl.DataFrame([row.model_dump() for row in rows], schema=SCHEMAS[name])
