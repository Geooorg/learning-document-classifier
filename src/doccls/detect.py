"""Format am Inhalt erkennen, nicht an der Endung.

Die Endung ist eine Behauptung des Erzeugers. In echten Beständen sind die ``.xls``, die
HTML enthält, und die ``.pdf``, die ein Scan ist, der Normalfall. Die Endung entscheidet
nur dort, wo es keine Magic Bytes gibt – bei E-Mails und CSV.
"""

import io
import re
import zipfile
from enum import StrEnum


class Format(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    EML = "eml"
    MSG = "msg"
    HTML = "html"
    CSV = "csv"
    IMAGE = "image"
    UNKNOWN = "unknown"


MEDIA_TYPES: dict[Format, str] = {
    Format.PDF: "application/pdf",
    Format.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    Format.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    Format.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    Format.EML: "message/rfc822",
    Format.MSG: "application/vnd.ms-outlook",
    Format.HTML: "text/html",
    Format.CSV: "text/csv",
    Format.IMAGE: "image/*",
    Format.UNKNOWN: "application/octet-stream",
}

OOXML_MARKER: tuple[tuple[str, Format], ...] = (
    ("word/document.xml", Format.DOCX),
    ("xl/workbook.xml", Format.XLSX),
    ("ppt/presentation.xml", Format.PPTX),
)
"""DOCX, XLSX und PPTX sind alle ZIP-Archive. Nur ihr Inhalt unterscheidet sie."""

IMAGE_MAGIC: tuple[bytes, ...] = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"II*\x00", b"MM\x00*")
OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
"""OLE2 ist der gemeinsame Container von MSG, aber auch der Altformate .doc/.xls/.ppt.
Nicht trennscharf – bewusst: ``Format`` kennt diese Altformate nicht, sie kommen laut Plan
erst in Phase 4 über Apache Tika."""

BOM = b"\xef\xbb\xbf"


def _ohne_bom(data: bytes) -> bytes:
    """Führendes Byte-Order-Mark und Whitespace entfernen. ``bytes.lstrip`` kennt das BOM
    nicht, und ein BOM davor macht jede Inhaltsprüfung blind."""
    return data.lstrip().removeprefix(BOM).lstrip()


FIELD_LINE = re.compile(rb"^[A-Za-z][A-Za-z0-9-]*:[ \t]")
"""Irgendein Feld nach RFC 5322, nicht aus einer festen Liste: Eine echte Nachricht darf
mit ``Delivered-To:``, ``X-Mailer:`` oder ``DKIM-Signature:`` beginnen."""

MBOX_LINE = re.compile(rb"^From \S+")
"""Die mbox-Trennzeile – „From " ohne Doppelpunkt, deshalb kein Feld im obigen Sinn."""

KNOWN_HEADER = re.compile(
    rb"^(From|To|Cc|Bcc|Subject|Date|Message-ID|Received|Return-Path|MIME-Version"
    rb"|Content-Type):",
    re.IGNORECASE,
)
"""E-Mails haben keine Magic Bytes. Erkannt wird an den Kopfzeilen am Anfang."""

MIN_MAIL_HEADERS = 2
"""So viele **verschiedene** bekannte Felder müssen vorkommen. Verschieden, nicht bloß
viele: Eine Nachricht sammelt auf ihrem Weg mehrere ``Received:``-Zeilen, und die allein
machen noch keine Mail aus."""


def _ooxml_kind(data: bytes) -> Format:
    """ZIP-Inhalt auswerten. Enthält ein Archiv mehrere Marker, gewinnt der erste Treffer
    in ``OOXML_MARKER`` – das ist eine bewusste, deterministische Entscheidung."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archiv:
            namen = set(archiv.namelist())
    except zipfile.BadZipFile:
        return Format.UNKNOWN
    for marker, format_ in OOXML_MARKER:
        if marker in namen:
            return format_
    return Format.UNKNOWN


def _sieht_aus_wie_mail(data: bytes) -> bool:
    """Zwei Bedingungen, die Verschiedenes prüfen.

    Die erste Zeile muss **kopfzeilenförmig** sein – nach RFC 5322 beginnt eine Nachricht
    mit Kopfzeilen, und eine Textdatei mit einer „To:“-Zeile in der Mitte ist keine.
    Welches Feld es ist, bleibt offen; die Liste bekannter Felder wäre nie vollständig.

    Danach müssen mindestens zwei **verschiedene bekannte** Felder vorkommen. Sonst wäre
    jede Konfigurationsdatei, die mit „name: …“ beginnt, eine E-Mail.
    """
    kopf = _ohne_bom(data[:2048])
    zeilen = [z for z in kopf.splitlines()[:16] if z.strip()]
    if not zeilen or not (FIELD_LINE.match(zeilen[0]) or MBOX_LINE.match(zeilen[0])):
        return False
    felder = {treffer.group(1).lower() for z in zeilen if (treffer := KNOWN_HEADER.match(z))}
    return len(felder) >= MIN_MAIL_HEADERS


def detect_format(data: bytes, file_name: str) -> Format:
    """Format bestimmen. Magic Bytes zuerst, die Endung nur als letzte Auskunft."""
    endung = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if data.startswith(b"%PDF-"):
        return Format.PDF
    if data.startswith(b"PK\x03\x04"):
        art = _ooxml_kind(data)
        if art is not Format.UNKNOWN:
            return art
    if data.startswith(OLE2_MAGIC):
        return Format.MSG
    if any(data.startswith(magic) for magic in IMAGE_MAGIC):
        return Format.IMAGE
    if _ohne_bom(data[:1024])[:15].lower().startswith((b"<!doctype html", b"<html")):
        return Format.HTML
    if _sieht_aus_wie_mail(data):
        return Format.EML

    # Ohne Magic Bytes entscheidet die Endung.
    match endung:
        case "eml":
            return Format.EML
        case "csv" | "tsv":
            return Format.CSV
        case "htm" | "html":
            return Format.HTML
        case _:
            return Format.UNKNOWN
