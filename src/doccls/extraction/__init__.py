"""Extraktion: Bytes in Segmente. Neue Formate kommen in ``extract`` als weiterer Fall dazu."""

from doccls.detect import Format, detect_format
from doccls.extraction.office import extract_docx, extract_xlsx
from doccls.extraction.pdf import extract_pdf
from doccls.models import Document, Segment

__all__ = ["extract", "extract_docx", "extract_pdf", "extract_xlsx"]


def extract(document: Document, data: bytes) -> list[Segment]:
    """Segmente aus dem Dateiinhalt. Das Format wird am Inhalt erkannt, nicht an der Endung."""
    match detect_format(data, document.file_name):
        case Format.PDF:
            return extract_pdf(document, data)
        case Format.DOCX:
            return extract_docx(document, data)
        case Format.XLSX:
            return extract_xlsx(document, data)
        case unbekannt:
            raise ValueError(f"Nicht unterstütztes Format {unbekannt} für {document.file_name!r}")
