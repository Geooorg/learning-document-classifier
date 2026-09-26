"""Extraktion: Bytes in Segmente. Neue Formate kommen in ``extract`` als weiterer Fall dazu."""

from doccls.detect import Format, detect_format
from doccls.extraction.mail import Attachment, extract_eml
from doccls.extraction.office import extract_docx, extract_xlsx
from doccls.extraction.pdf import count_pages, extract_pdf
from doccls.models import Document, Segment

__all__ = [
    "Attachment",
    "count_pages",
    "extract",
    "extract_docx",
    "extract_eml",
    "extract_pdf",
    "extract_xlsx",
]


def extract(document: Document, data: bytes) -> list[Segment]:
    """Segmente aus dem Dateiinhalt. Das Format wird am Inhalt erkannt, nicht an der Endung.

    Anhänge einer Mail liefert diese Funktion bewusst nicht mit – ``extract`` gibt nur
    Segmente zurück. ``pipeline.py`` ruft dafür ``extract_eml`` direkt auf, weil ein Anhang
    ein eigenes Dokument wird und keine Segmente der Mail (Moduldoc von ``mail.py``).
    """
    match detect_format(data, document.file_name):
        case Format.PDF:
            return extract_pdf(document, data)
        case Format.DOCX:
            return extract_docx(document, data)
        case Format.XLSX:
            return extract_xlsx(document, data)
        case Format.EML:
            segmente, _ = extract_eml(document, data)
            return segmente
        case unbekannt:
            raise ValueError(f"Nicht unterstütztes Format {unbekannt} für {document.file_name!r}")
