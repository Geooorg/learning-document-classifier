"""IDs müssen aus dem Inhalt abgeleitet und damit über Läufe hinweg stabil sein."""

from datetime import UTC, datetime

import polars as pl

from doccls.models import (
    SCHEMAS,
    Document,
    Segment,
    SegmentKind,
    derive_id,
    normalize_text,
    to_frame,
)


def beispiel_dokument(inhalt: bytes = b"Rechnung 4711") -> Document:
    return Document.create(
        source_path="rechnungen/a.pdf",
        media_type="application/pdf",
        content=inhalt,
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_derive_id_trennt_die_teile() -> None:
    """Ohne Trenner ergäben ("ab", "c") und ("a", "bc") dieselbe ID – und zwei
    verschiedene Dokumente dieselbe Identität."""
    assert derive_id("ab", "c") != derive_id("a", "bc")


def test_derive_id_mischt_bytes_und_text() -> None:
    assert derive_id("pfad.pdf", b"inhalt") == derive_id("pfad.pdf", b"inhalt")
    assert derive_id("pfad.pdf", b"inhalt") != derive_id("pfad.pdf", b"anderes")


def test_id_ist_stabil_und_32_zeichen_lang() -> None:
    a, b = beispiel_dokument(), beispiel_dokument()
    assert a.document_id == b.document_id
    assert len(a.document_id) == 32


def test_geaenderter_inhalt_ergibt_neue_version() -> None:
    assert beispiel_dokument(b"x").document_id != beispiel_dokument(b"y").document_id


def test_gleicher_inhalt_an_zwei_pfaden_sind_zwei_dokumente() -> None:
    """Sonst würde die zweite Fundstelle übersprungen und wäre nirgends verzeichnet."""
    a = Document.create(
        source_path="a/x.pdf",
        media_type="application/pdf",
        content=b"gleich",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    b = Document.create(
        source_path="b/x.pdf",
        media_type="application/pdf",
        content=b"gleich",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert a.document_id != b.document_id
    assert a.content_sha256 == b.content_sha256


def test_segment_id_haengt_am_dokument_und_am_index() -> None:
    doc = beispiel_dokument()
    erst = Segment.create(document=doc, index=0, kind=SegmentKind.SEITE, locator="S. 1", text="a")
    zweit = Segment.create(document=doc, index=1, kind=SegmentKind.SEITE, locator="S. 2", text="a")
    assert erst.segment_id != zweit.segment_id


def test_normalize_text_loest_silbentrennung_und_whitespace_auf() -> None:
    assert normalize_text("Rech-\nnung  über\n\n  100") == "Rechnung über 100"


def test_normalize_text_vereinheitlicht_unicode() -> None:
    assert normalize_text("ﬁnal") == "final"  # Ligatur -> NFKC


def test_to_frame_haelt_sich_an_das_schema() -> None:
    rahmen = to_frame([beispiel_dokument()], "documents")
    assert rahmen.schema == pl.Schema(SCHEMAS["documents"])
    assert rahmen.height == 1


def test_leere_tabelle_hat_trotzdem_das_schema() -> None:
    """Sonst schlägt der erste Lauf beim Schreiben einer leeren Parquet-Datei fehl."""
    assert to_frame([], "segments").schema == pl.Schema(SCHEMAS["segments"])
