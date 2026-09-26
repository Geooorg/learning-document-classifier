"""Ein ``DocumentSpec`` in PDF, DOCX, XLSX oder EML ausschreiben.

Alles Zeitabhängige wird fixiert: ZIP-Zeitstempel (DOCX, XLSX), PDF-Erzeugungsdatum,
Mail-Datum und MIME-Grenze. Sonst hätte dieselbe Datei bei jedem Lauf einen anderen
SHA-256 – und damit eine andere ``document_id``.
"""

import re
import unicodedata
import zipfile
from datetime import UTC, datetime
from email.message import EmailMessage
from email.policy import SMTP
from pathlib import Path

import docx
import openpyxl
import pymupdf
from docx.document import Document as DocxDocument

from doccls.generation.content import Block, DocumentSpec

FIXED_TIMESTAMP = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
BOUNDARY = "===============doccls-fixed-boundary=="

PDF_CSS = """
* { font-family: sans-serif; font-size: 10pt; }
h1 { font-size: 15pt; margin-bottom: 4pt; }
h2 { font-size: 12pt; margin-top: 10pt; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #999999; padding: 3pt; text-align: left; }
th { background-color: #e6e6e6; }
"""

TAG = re.compile(r"<[^>]+>")
ENTITIES = (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"))
MIN_RENDERED_SHARE = 0.99
"""So viel des Eingabetexts muss auf der Seite ankommen. Beide Seiten werden NFKC-normalisiert,
weil die Textextraktion „fl"/„fi" als Ligatur ﬂ/ﬁ zurückgibt und sonst zwei Zeichen fehlten.
Nach der Normalisierung liegt der Anteil bei allen heutigen Vorlagen bei exakt 1,0; der
Spielraum ist Sicherheitsabstand, keine Krücke."""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def plain_text(html: str) -> str:
    """Den sichtbaren Text aus dem erzeugten HTML – ohne Tags, ohne Whitespace, NFKC."""
    text = TAG.sub(" ", html)
    for entity, zeichen in ENTITIES:
        text = text.replace(entity, zeichen)
    return "".join(unicodedata.normalize("NFKC", text).split())


def block_html(block: Block) -> str:
    """Einen Block als HTML – die Zwischenform für die PDF-Ausgabe."""
    teile: list[str] = []
    if block.heading:
        teile.append(f"<h2>{_escape(block.heading)}</h2>")
    teile += [f"<p>{_escape(absatz)}</p>" for absatz in block.paragraphs]
    if block.table:
        kopf, zeilen = block.table
        kopf_html = "".join(f"<th>{_escape(z)}</th>" for z in kopf)
        leib = "".join(
            "<tr>" + "".join(f"<td>{_escape(z)}</td>" for z in zeile) + "</tr>" for zeile in zeilen
        )
        teile.append(f"<table><tr>{kopf_html}</tr>{leib}</table>")
    return "".join(teile)


def block_text(block: Block) -> str:
    """Einen Block als Klartext – für die Mail-Ausgabe."""
    zeilen: list[str] = []
    if block.heading:
        zeilen += [block.heading, "-" * len(block.heading)]
    zeilen += list(block.paragraphs)
    if block.table:
        kopf, tabelle = block.table
        zeilen.append(" | ".join(kopf))
        zeilen += [" | ".join(zeile) for zeile in tabelle]
    return "\n".join(zeilen)


def _group_starts(groups: tuple[int, ...]) -> list[bool]:
    """Je Block: beginnt hier eine neue Gruppe (Seite/Segment) oder gehört er noch zur
    vorigen? Aus ``spec.page_groups`` (Blockanzahl je Gruppe) abgeleitet – dieselbe
    Aufteilung, die ``write_pdf`` in Seiten übersetzt, steuert hier, ob eine Überschrift
    ohne eigene Vorlagen-Tabelle als Word-Überschrift geschrieben wird (DOCX) bzw. ob eine
    Tabelle ein eigenes Blatt bekommt (XLSX). So hängt die Segmentzahl auch in diesen
    Formaten nicht mehr starr an der Vorlage."""
    starts: list[bool] = []
    for groesse in groups:
        starts += [True] + [False] * (groesse - 1)
    return starts


def normalize_zip(path: Path) -> None:
    """ZIP-Zeitstempel und das Änderungsdatum in ``docProps/core.xml`` fixieren.

    Ohne das ergibt derselbe Inhalt bei jedem Lauf einen anderen SHA-256. Übernommen aus
    ``bauprojekt-ai-pipeline``.
    """
    with zipfile.ZipFile(path) as quelle:
        eintraege = [(info.filename, quelle.read(info)) for info in quelle.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as ziel:
        for name, daten in eintraege:
            if name == "docProps/core.xml":
                daten = re.sub(
                    rb"(<dcterms:(?:modified|created)[^>]*>)[^<]*",
                    rb"\g<1>2026-03-01T12:00:00Z",
                    daten,
                )
            info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # immer „Unix", sonst hinge der Hash am Betriebssystem
            ziel.writestr(info, daten)


def write_pdf(path: Path, spec: DocumentSpec) -> None:
    """Seiten nach ``spec.page_groups`` füllen – mehrere Blöcke können dieselbe Seite teilen.

    Die Seite bleibt die Fundstelle (Seitenzahlen stabil), aber nicht mehr zwingend genau
    ein Block: eine je Vorlage feste Seitenzahl wäre sonst ein Merkmal der Herkunft, nicht
    des Inhalts (``_page_groups`` in ``generation/content.py``). Der Text selbst – welcher
    Block was sagt – ändert sich dadurch nicht, nur seine Verteilung auf Seiten.
    """
    doc = pymupdf.open()
    seiten: list[str] = []
    index = 0
    for seitennummer, groesse in enumerate(spec.page_groups):
        gruppe = spec.blocks[index : index + groesse]
        index += groesse
        teile = ["<h1>" + _escape(spec.title) + "</h1>"] if seitennummer == 0 else []
        teile += [block_html(block) for block in gruppe]
        seiten.append("".join(teile))
    for nummer, html in enumerate(seiten, start=1):
        seite = doc.new_page(width=595, height=842)  # A4
        rest, _ = seite.insert_htmlbox(
            seite.rect + (56, 56, -56, -56), html, css=PDF_CSS, scale_low=1
        )
        erwartet = plain_text(html)
        angekommen = "".join(unicodedata.normalize("NFKC", seite.get_text("text")).split())
        if rest < 0 or len(angekommen) < len(erwartet) * MIN_RENDERED_SHARE:
            raise ValueError(
                f"{path.name}: Seite {nummer} verliert Inhalt "
                f"({len(angekommen)} von {len(erwartet)} Zeichen angekommen, rest={rest:.1f}). "
                "Ursache ist meist eine Tabellenzelle mit langem, nicht umbrechbarem Inhalt: "
                "insert_htmlbox meldet horizontalen Überlauf nicht über den Rückgabewert."
            )
    datum = FIXED_TIMESTAMP.strftime("D:%Y%m%d%H%M%S")
    doc.set_metadata(
        {
            "title": spec.title,
            "author": "Synthetische Testdaten",
            "creationDate": datum,
            "modDate": datum,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path, garbage=4, deflate=True, no_new_id=True)
    doc.close()


def write_docx(path: Path, spec: DocumentSpec) -> None:
    """Überschriften werden zu Word-„Heading“-Absätzen – außer eine Überschrift ohne eigene
    Tabelle liegt laut ``spec.page_groups`` in derselben Gruppe wie ihr Vorgänger; dann wird
    sie ein gewöhnlicher Absatz. ``extract_docx`` zieht Segmentgrenzen nur an „Heading“-
    Absätzen, eine Tabelle dagegen immer an ihrer eigenen Segmentgrenze – deshalb bleiben
    Tabellenüberschriften unangetastet. Der Text selbst ändert sich nicht, nur ob er als
    eigenes Segment zählt."""
    dokument: DocxDocument = docx.Document()
    dokument.add_heading(spec.title, level=1)
    for beginnt_gruppe, block in zip(_group_starts(spec.page_groups), spec.blocks, strict=True):
        if block.heading:
            if beginnt_gruppe or block.table:
                dokument.add_heading(block.heading, level=2)
            else:
                dokument.add_paragraph(block.heading)
        for absatz in block.paragraphs:
            dokument.add_paragraph(absatz)
        if block.table:
            kopf, zeilen = block.table
            tabelle = dokument.add_table(rows=1, cols=len(kopf))
            tabelle.style = "Table Grid"
            for zelle, text in zip(tabelle.rows[0].cells, kopf, strict=True):
                zelle.text = text
            for zeile in zeilen:
                for zelle, text in zip(tabelle.add_row().cells, zeile, strict=True):
                    zelle.text = text
    eigenschaften = dokument.core_properties
    eigenschaften.title = spec.title
    eigenschaften.author = "Synthetische Testdaten"
    eigenschaften.created = eigenschaften.modified = FIXED_TIMESTAMP.replace(tzinfo=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    dokument.save(str(path))
    normalize_zip(path)


def write_xlsx(path: Path, spec: DocumentSpec) -> None:
    """Fließtext in einem oder mehreren „Dokument“-Blättern, je Tabelle ein eigenes Blatt.

    Eine Tabelle bekommt immer ihr eigenes Blatt – daran rührt die Streuung nicht, sonst
    verlöre eine Zeile ihre Kopfzeile und damit die „Spalte: Wert“-Zuordnung, um die es
    ``extract_xlsx`` gerade geht. Gestreut wird stattdessen die Zahl der Fließtext-Blätter:
    Beginnt laut ``spec.page_groups`` an einem Block ohne Tabelle eine neue Gruppe, geht der
    Fließtext ab hier in einem neuen „Dokument N“-Blatt weiter. Ohne das stünde die Blattzahl
    einer Vorlage fest (ein Blatt je Tabelle plus ein Fließtextblatt) und verriete die
    Klasse."""
    mappe = openpyxl.Workbook()
    erstes_blatt = mappe.active
    if erstes_blatt is None:
        raise ValueError("Neue Arbeitsmappe hat kein aktives Blatt")
    erstes_blatt.title = "Dokument"
    haupt = erstes_blatt
    haupt.append([spec.title])
    hauptblattnummer = 1
    tabellennummer = 0
    for index, (beginnt_gruppe, block) in enumerate(
        zip(_group_starts(spec.page_groups), spec.blocks, strict=True)
    ):
        if index > 0 and beginnt_gruppe and not block.table:
            hauptblattnummer += 1
            haupt = mappe.create_sheet(f"Dokument {hauptblattnummer}"[:31])
        if block.heading:
            haupt.append([block.heading])
        for absatz in block.paragraphs:
            haupt.append([absatz])
        if block.table:
            tabellennummer += 1
            kopf, zeilen = block.table
            tabellenblatt = mappe.create_sheet(f"Positionen {tabellennummer}"[:31])
            tabellenblatt.append(list(kopf))
            for zeile in zeilen:
                tabellenblatt.append(list(zeile))
    mappe.properties.title = spec.title
    mappe.properties.creator = "Synthetische Testdaten"
    mappe.properties.created = mappe.properties.modified = FIXED_TIMESTAMP.replace(tzinfo=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    mappe.save(str(path))
    normalize_zip(path)


def write_eml(path: Path, spec: DocumentSpec, attachment: tuple[str, bytes] | None = None) -> None:
    """Mail mit Klartextkörper, optional mit Anhang.

    Der Anhang wird später ein **eigenständiges Dokument** mit Elternbezug (Konzept § 5).
    Die MIME-Grenze wird fixiert, weil sie sonst zufällig ist.
    """
    nachricht = EmailMessage(policy=SMTP)
    nachricht["Subject"] = spec.title
    nachricht["From"] = "buchhaltung@muster-handels.example"
    nachricht["To"] = "eingang@kunde.example"
    nachricht["Date"] = "Sun, 01 Mar 2026 12:00:00 +0000"
    nachricht["Message-ID"] = f"<{spec.template_id}.{spec.variant}@muster-handels.example>"
    nachricht.set_content("\n\n".join(block_text(block) for block in spec.blocks))
    if attachment is not None:
        name, daten = attachment
        nachricht.add_attachment(
            daten, maintype="application", subtype="octet-stream", filename=name
        )
        nachricht.set_boundary(BOUNDARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(nachricht.as_bytes())


def write(
    path_without_suffix: Path,
    spec: DocumentSpec,
    fmt: str,
    attachment: tuple[str, bytes] | None = None,
) -> Path:
    """Ausgabeformat wählen und schreiben. Liefert den geschriebenen Pfad."""
    pfad = path_without_suffix.with_suffix(f".{fmt}")
    match fmt:
        case "pdf":
            write_pdf(pfad, spec)
        case "docx":
            write_docx(pfad, spec)
        case "xlsx":
            write_xlsx(pfad, spec)
        case "eml":
            write_eml(pfad, spec, attachment)
        case _:
            raise ValueError(f"Unbekanntes Format: {fmt!r}")
    return pfad
