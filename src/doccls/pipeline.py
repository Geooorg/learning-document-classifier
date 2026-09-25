"""Orchestrierung: Originale → Parquet.

Hier – und nur hier – wird auf das Dateisystem zugegriffen. Erkennung, Extraktion und
Normalisierung bleiben reine Funktionen. Für den späteren Betrieb mit Objektspeicher ist
damit genau dieses Modul die Stelle, die umgestellt werden muss.

Inkrementell über die ``document_id`` (Pfad + Inhalts-Hash): Ein bereits bekanntes Dokument
wird übersprungen. Geänderter Inhalt ergibt eine neue ID und damit eine zusätzliche
Version; die alte bleibt erhalten und bleibt bewertbar.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from doccls.config import MIN_CHARS_PER_PAGE
from doccls.detect import MEDIA_TYPES, Format, detect_format
from doccls.extraction import extract
from doccls.extraction.mail import extract_eml
from doccls.models import SCHEMAS, Document, Segment, to_frame
from doccls.normalize import strip_boilerplate

TABLES = ("documents", "segments")

IGNORED_PREFIXES = ("~$", ".")
"""Sperr- und Systemdateien. ``~$…`` sind Word-Sperrdateien geöffneter Dokumente, ``.``
u. a. ``.DS_Store`` und versteckte Dateien. Sie werden stillschweigend übersprungen –
``failed`` soll echte Probleme anzeigen, keine Artefakte des Dateisystems."""


@dataclass
class IngestResult:
    """Was ein Durchlauf bewirkt hat."""

    documents: int = 0
    skipped: int = 0
    segments: int = 0
    attachments: int = 0
    failed: list[str] = field(default_factory=list)
    """Dateien, deren Format nicht unterstützt wird oder die nicht lesbar waren."""


def read_table(parquet_dir: Path, name: str) -> pl.DataFrame:
    """Tabelle lesen; noch nicht vorhanden heißt leer, nicht Fehler."""
    pfad = parquet_dir / f"{name}.parquet"
    return pl.read_parquet(pfad) if pfad.exists() else pl.DataFrame(schema=SCHEMAS[name])


def _write_table(parquet_dir: Path, name: str, neu: pl.DataFrame) -> None:
    """Anhängen statt ersetzen. Bestehende Zeilen bleiben, damit alte Versionen erhalten sind."""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    zusammen = pl.concat([read_table(parquet_dir, name), neu], how="vertical")
    zusammen.write_parquet(parquet_dir / f"{name}.parquet")


def _needs_ocr(segments: list[Segment], format_: Format) -> bool:
    """Zu wenig Text je Seite spricht für einen Scan. Die Entscheidung wird vermerkt,
    damit später auswertbar ist, ob OCR-Dokumente systematisch schlechter abschneiden."""
    if format_ is not Format.PDF:
        return False
    if not segments:
        return True
    return sum(len(s.text) for s in segments) / len(segments) < MIN_CHARS_PER_PAGE


def _process(
    source_path: str,
    data: bytes,
    now: datetime,
    parent_document_id: str | None,
    known: set[str],
    result: IngestResult,
) -> tuple[list[Document], list[Segment]]:
    """Ein Dokument samt seiner Anhänge verarbeiten. Rekursiv, weil Mails Mails enthalten können."""
    format_ = detect_format(data, source_path.rsplit("/", 1)[-1])
    dokument = Document.create(
        source_path=source_path,
        media_type=MEDIA_TYPES[format_],
        content=data,
        ingested_at=now,
        parent_document_id=parent_document_id,
    )
    if dokument.document_id in known:
        result.skipped += 1
        return [], []

    try:
        if format_ is Format.EML:
            segmente, anhaenge = extract_eml(dokument, data)
        else:
            segmente, anhaenge = extract(dokument, data), []
    except Exception:
        # Eine kaputte oder unbekannte Datei darf den Lauf nicht beenden. Sie wird vermerkt
        # und taucht in der Zusammenfassung auf; ein stiller Abbruch wäre schlimmer.
        result.failed.append(source_path)
        return [], []

    bereinigt = strip_boilerplate([s.text for s in segmente])
    segmente = [
        s.model_copy(update={"text": text})
        for s, text in zip(segmente, bereinigt, strict=True)
        if text
    ]

    dokument = dokument.model_copy(update={"needs_ocr": _needs_ocr(segmente, format_)})
    known.add(dokument.document_id)
    result.documents += 1
    result.segments += len(segmente)

    dokumente, alle_segmente = [dokument], list(segmente)
    for anhang in anhaenge:
        result.attachments += 1
        kind_dokumente, kind_segmente = _process(
            f"{source_path}!{anhang.file_name}",
            anhang.content,
            now,
            dokument.document_id,
            known,
            result,
        )
        dokumente += kind_dokumente
        alle_segmente += kind_segmente
    return dokumente, alle_segmente


def ingest(raw_dir: Path, parquet_dir: Path) -> IngestResult:
    """Alle Dateien unterhalb ``raw_dir`` einlesen und als Parquet ablegen.

    Sperr- und Systemdateien (``~$…``, ``.…``) werden gar nicht erst geöffnet – sie sind
    keine Dokumente und gehören nicht nach ``failed``.
    """
    ergebnis = IngestResult()
    bekannt = set(read_table(parquet_dir, "documents")["document_id"])
    jetzt = datetime.now(UTC)

    dokumente: list[Document] = []
    segmente: list[Segment] = []
    dateien = sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and not p.name.startswith(IGNORED_PREFIXES)
    )
    for datei in dateien:
        neue_dokumente, neue_segmente = _process(
            str(datei.relative_to(raw_dir)), datei.read_bytes(), jetzt, None, bekannt, ergebnis
        )
        dokumente += neue_dokumente
        segmente += neue_segmente

    _write_table(parquet_dir, "documents", to_frame(dokumente, "documents"))
    _write_table(parquet_dir, "segments", to_frame(segmente, "segments"))
    return ergebnis
