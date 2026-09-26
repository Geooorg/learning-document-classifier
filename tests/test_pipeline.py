"""Die Pipeline muss inkrementell sein und Anhänge als eigene Dokumente führen."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from doccls.config import MIN_CHARS_PER_PAGE, RAW_DIR
from doccls.detect import MEDIA_TYPES, Format
from doccls.extraction import extract
from doccls.generation.content import build_corpus
from doccls.generation.writers import write
from doccls.models import Document
from doccls.pipeline import ingest, read_table


@pytest.fixture
def kleiner_bestand(tmp_path: Path) -> Path:
    """Vier Dokumente, je eines pro Format, plus eine Mail mit PDF-Anhang."""
    roh = tmp_path / "raw"
    korpus = build_corpus()
    for fmt in ("pdf", "docx", "xlsx"):
        spec = next(s for s in korpus if fmt in s.formats)
        write(roh / fmt / "dok", spec, fmt)
    mail_spec = next(s for s in korpus if "eml" in s.formats)
    pdf_spec = next(s for s in korpus if "pdf" in s.formats)
    anhang = write(tmp_path / "tmp" / "anhang", pdf_spec, "pdf").read_bytes()
    write(
        roh / "eml" / "mit-anhang",
        mail_spec,
        "eml",
        attachment=("rechnung.pdf", anhang),
    )
    return roh


def _pruefe_jedes_dokument_hat_inhalt(dokumente: pl.DataFrame, segmente: pl.DataFrame) -> None:
    """Kein Zählen, sondern Inhalt binden: jedes Dokument braucht mindestens ein Segment,
    und die Segmenttexte dürfen zusammen nicht leer sein. Eine reine Anzahlprüfung (»es
    gibt Segmente«) hätte in Aufgabe 8 mehrere Mutationen mit stillem Inhaltsverlust
    überleben lassen – halbierter Text, nur die erste Zeile, nur der erste Satz."""
    for dokument_id, source_path in dokumente.select("document_id", "source_path").iter_rows():
        je_dokument = segmente.filter(pl.col("document_id") == dokument_id)
        assert je_dokument.height > 0, f"{source_path}: kein Segment geschrieben"
        gesamtlaenge = je_dokument["text"].str.len_chars().sum()
        assert gesamtlaenge and gesamtlaenge > 0, f"{source_path}: Segmente ohne Text"


def test_alle_dokumente_werden_eingelesen(kleiner_bestand: Path, tmp_path: Path) -> None:
    ziel = tmp_path / "parquet"
    ergebnis = ingest(kleiner_bestand, ziel)
    assert ergebnis.documents == 5  # 4 Dateien + 1 Anhang
    assert ergebnis.attachments == 1
    assert ergebnis.failed == []

    dokumente = read_table(ziel, "documents")
    segmente = read_table(ziel, "segments")
    assert dokumente.height == 5
    _pruefe_jedes_dokument_hat_inhalt(dokumente, segmente)


def test_zweiter_lauf_ueberspringt_alles(kleiner_bestand: Path, tmp_path: Path) -> None:
    """Inkrementell über den Inhalts-Hash – sonst verdoppelt jeder Lauf den Bestand."""
    ziel = tmp_path / "parquet"
    ingest(kleiner_bestand, ziel)
    zweiter = ingest(kleiner_bestand, ziel)
    assert zweiter.documents == 0
    assert zweiter.skipped == 4, (
        "Nur die vier Dateien werden erneut angefasst. Die übersprungene Mail wird gar nicht "
        "erst geöffnet, deshalb wird ihr Anhang nicht noch einmal besucht – genau so soll es sein."
    )
    assert read_table(ziel, "documents").height == 5


def test_geaenderte_datei_ergibt_eine_zusaetzliche_version(
    kleiner_bestand: Path, tmp_path: Path
) -> None:
    ziel = tmp_path / "parquet"
    ingest(kleiner_bestand, ziel)
    datei = next((kleiner_bestand / "pdf").glob("*.pdf"))
    spec = next(s for s in build_corpus() if "pdf" in s.formats and s.variant == 3)
    write(datei.with_suffix(""), spec, "pdf")
    ingest(kleiner_bestand, ziel)
    dokumente = read_table(ziel, "documents")
    assert dokumente.height == 6
    assert dokumente.filter(pl.col("source_path") == f"pdf/{datei.stem}.pdf").height == 2


def test_anhang_traegt_den_elternbezug(kleiner_bestand: Path, tmp_path: Path) -> None:
    """Eine Mail und ihre Rechnung sind zwei Dokumente (Konzept § 5). Gebunden wird nicht nur
    die Anzahl, sondern der tatsächliche Anhangsinhalt – sonst besteht auch eine Mutation,
    die den Anhang durch leere oder falsche Bytes ersetzt."""
    erwarteter_anhang = write(
        tmp_path / "erwartung" / "anhang",
        (next(s for s in build_corpus() if "pdf" in s.formats)),
        "pdf",
    ).read_bytes()

    ingest(kleiner_bestand, tmp_path / "parquet")
    dokumente = read_table(tmp_path / "parquet", "documents")
    kinder = dokumente.filter(pl.col("parent_document_id").is_not_null())
    assert kinder.height == 1
    assert kinder["source_path"][0].endswith("!rechnung.pdf")
    assert kinder["size_bytes"][0] == len(erwarteter_anhang)
    assert kinder["content_sha256"][0] == hashlib.sha256(erwarteter_anhang).hexdigest()
    eltern = kinder["parent_document_id"][0]
    assert eltern in set(dokumente["document_id"])


def test_segmente_verweisen_nur_auf_bekannte_dokumente(
    kleiner_bestand: Path, tmp_path: Path
) -> None:
    ingest(kleiner_bestand, tmp_path / "parquet")
    dokumente = read_table(tmp_path / "parquet", "documents")
    segmente = read_table(tmp_path / "parquet", "segments")
    assert set(segmente["document_id"]) <= set(dokumente["document_id"])


def test_unlesbare_datei_stoppt_den_lauf_nicht(kleiner_bestand: Path, tmp_path: Path) -> None:
    (kleiner_bestand / "kaputt.dat").write_bytes(b"\x00\x01\x02 kein bekanntes Format")
    ergebnis = ingest(kleiner_bestand, tmp_path / "parquet")
    assert ergebnis.failed == ["kaputt.dat"]
    # Exakte Zahl statt ">=": Sonst besteht die Prüfung auch, wenn zusätzliche Dokumente
    # auftauchen, die dort nicht hingehören (Korrektur des Steuernden zu Aufgabe 8).
    assert ergebnis.documents == 5


def test_sperrdatei_wird_uebersprungen(kleiner_bestand: Path, tmp_path: Path) -> None:
    """Word legt beim Öffnen eine Sperrdatei ``~$…`` an. ``extract`` erkennt ihr Format
    nicht und wirft, also würde sie ohne Filter dauerhaft als Fehlschlag erscheinen, obwohl
    sie keiner ist. Sie – und Systemdateien wie ``.DS_Store`` – werden deshalb schon vor
    dem Öffnen übersprungen, und zwar stillschweigend: ``failed`` bleibt leer."""
    (kleiner_bestand / "docx" / "~$OTOKOLL-top-02.docx").write_bytes(b"\x00\x00")
    (kleiner_bestand / "docx" / ".DS_Store").write_bytes(b"nicht relevant")
    ergebnis = ingest(kleiner_bestand, tmp_path / "parquet")
    assert ergebnis.failed == []
    assert ergebnis.documents == 5


def test_mehrere_dateien_im_selben_verzeichnis_werden_alle_verarbeitet(tmp_path: Path) -> None:
    """Ein Verzeichnislauf, der nur die erste Datei je Ordner anfasst, würde hier
    unbemerkt bleiben, wenn jedes Testverzeichnis nur eine Datei enthält – deshalb ein
    eigenes Verzeichnis mit drei Dateien."""
    roh = tmp_path / "raw"
    korpus = build_corpus()
    pdf_specs = [s for s in korpus if "pdf" in s.formats][:3]
    assert len(pdf_specs) == 3
    for spec in pdf_specs:
        write(roh / "pdf" / f"dok-{spec.variant}", spec, "pdf")

    ergebnis = ingest(roh, tmp_path / "parquet")
    assert ergebnis.documents == 3
    assert ergebnis.failed == []

    dokumente = read_table(tmp_path / "parquet", "documents")
    segmente = read_table(tmp_path / "parquet", "segments")
    assert dokumente.height == 3
    _pruefe_jedes_dokument_hat_inhalt(dokumente, segmente)


def test_gescanntes_pdf_wird_als_ocr_bedürftig_markiert(tmp_path: Path) -> None:
    """Ein PDF mit einer leeren Seite hat zu wenig Text – needs_ocr, aber kein Fehler."""
    import pymupdf

    roh = tmp_path / "raw"
    roh.mkdir(parents=True)
    leer = pymupdf.open()
    leer.new_page(width=595, height=842)
    leer.save(roh / "scan.pdf", no_new_id=True)
    leer.close()

    ingest(roh, tmp_path / "parquet")
    dokumente = read_table(tmp_path / "parquet", "documents")
    assert dokumente.filter(pl.col("file_name") == "scan.pdf")["needs_ocr"][0] is True


def test_teilgescanntes_pdf_wird_als_ocr_beduerftig_markiert(tmp_path: Path) -> None:
    """K1: ``extract_pdf`` überspringt textlose Seiten mit dem Kommentar „needs_ocr fängt
    sie" – aber ``_needs_ocr`` teilte bisher durch die Zahl der übriggebliebenen Segmente,
    nicht durch die tatsächliche Seitenzahl. Genau die Seiten, die die Schwelle fangen
    soll, waren vorher aus dem Nenner entfernt: Ein PDF mit einer Textseite und neun
    Scanseiten lieferte ein Segment und blieb bei vollem Zeichenschnitt unter der Schwelle
    – needs_ocr wäre ``False`` geblieben, obwohl 90 % der Seiten gescannt sind.
    """
    import pymupdf

    roh = tmp_path / "raw"
    roh.mkdir(parents=True)
    dok = pymupdf.open()
    seite = dok.new_page(width=595, height=842)
    seite.insert_text(
        (72, 100),
        "Diese Seite enthaelt ausreichend echten Fliesstext fuer die Pruefung des Zeichenschnitts.",
        fontsize=12,
    )
    for _ in range(9):
        dok.new_page(width=595, height=842)
    dok.save(roh / "teilgescannt.pdf", no_new_id=True)
    dok.close()

    ingest(roh, tmp_path / "parquet")
    dokumente = read_table(tmp_path / "parquet", "documents")
    zeile = dokumente.filter(pl.col("file_name") == "teilgescannt.pdf")
    assert zeile.height == 1
    assert zeile["needs_ocr"][0] is True, (
        "Ein Segment auf zehn Seiten (nur die getippte Seite) darf nicht als "
        "vollwertiges Textdokument durchgehen"
    )


def test_kein_erzeugtes_pdf_gilt_als_ocr_beduerftig() -> None:
    """Die OCR-Schwelle trennt Scan von Textebene – sie darf nicht in die Verteilung der
    echten Dokumente hineinschneiden.

    Mit den ursprünglich geplanten 120 Zeichen je Seite wurden 30 von 230 PDF mit
    vollständiger Textebene als OCR-bedürftig markiert, weil der Bestand nur von 113 bis
    157 Zeichen je Seite reicht. Kein anderer Test bemerkte das: alle prüften die
    Gegenrichtung (ein leeres PDF *wird* markiert). Schlägt dieser Test fehl, liegt die
    Schwelle wieder zu hoch – oder die Vorlagen sind kürzer geworden.
    """
    pdfs = sorted(RAW_DIR.glob("pdf/*.pdf"))
    assert pdfs, "Kein Bestand vorhanden – erst scripts/generate_documents.py laufen lassen"

    knappste: list[tuple[float, str]] = []
    for pfad in pdfs:
        daten = pfad.read_bytes()
        dokument = Document.create(
            source_path=pfad.name,
            media_type=MEDIA_TYPES[Format.PDF],
            content=daten,
            ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        segmente = extract(dokument, daten)
        assert segmente, f"{pfad.name} liefert keine Segmente"
        knappste.append((sum(len(s.text) for s in segmente) / len(segmente), pfad.name))

    zeichen, name = min(knappste)
    assert zeichen >= MIN_CHARS_PER_PAGE, (
        f"{name} hat nur {zeichen:.1f} Zeichen je Seite und gälte als gescannt, "
        f"obwohl es eine Textebene hat (Schwelle {MIN_CHARS_PER_PAGE})"
    )
    assert zeichen >= 2 * MIN_CHARS_PER_PAGE, (
        f"Der Abstand zur Schwelle ist auf {zeichen / MIN_CHARS_PER_PAGE:.1f}-fach "
        f"geschrumpft ({name}). Unter dem doppelten Abstand ist die Trennung zwischen "
        "Scan und Textebene nicht mehr robust."
    )
