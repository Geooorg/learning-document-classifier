"""DOCX und XLSX.

DOCX: Ein Segment umfasst eine Überschrift und alles bis zur nächsten. Das hält
zusammenhängende Gedanken beieinander und gibt dem Segment eine sprechende Fundstelle.
Absätze und Tabellen werden in der tatsächlichen Reihenfolge des Dokuments durchlaufen
(``iter_inner_content``), nicht getrennt in zwei Durchläufen – nur so lässt sich einer
Tabelle die davorstehende Überschrift zuordnen.

XLSX: Ein Blatt wird ein Segment. Eine Tabellenzeile wird als ``Spalte: Wert``
geschrieben – zerfiele sie in einzelne Zellen, verlöre man den Bezug zwischen Position,
Menge und Betrag, und genau dieser Bezug unterscheidet eine Rechnung von einer Liste.
"""

import io

import docx
import openpyxl
from docx.table import _Cell
from docx.text.paragraph import Paragraph

from doccls.models import Document, Segment, SegmentKind, normalize_text

MIN_HEADER_CELLS = 2
"""Ab so vielen gefüllten Zellen gilt die erste Zeile eines Blattes als Kopfzeile."""


def _zellentext(zelle: _Cell) -> str:
    """Text einer Zelle einschließlich darin verschachtelter Tabellen.

    ``zelle.text`` sieht nur die eigenen Absätze einer Zelle; eine in die Zelle
    eingebettete Tabelle – etwa eine Positionsliste innerhalb einer äußeren Rahmentabelle –
    bliebe sonst vollständig unsichtbar. ``iter_inner_content`` läuft rekursiv über
    ``_Cell``, deshalb genügt der Aufruf hier für beliebig tief verschachtelte Tabellen.
    """
    teile: list[str] = []
    for inhalt in zelle.iter_inner_content():
        if isinstance(inhalt, Paragraph):
            text = inhalt.text.strip()
            if text:
                teile.append(text)
        else:
            for zeile in inhalt.rows:
                zeilentext = " | ".join(_zellentext(z) for z in zeile.cells)
                if zeilentext.strip(" |"):
                    teile.append(zeilentext)
    return " ".join(teile)


def extract_docx(document: Document, data: bytes) -> list[Segment]:
    quelle = docx.Document(io.BytesIO(data))
    segmente: list[Segment] = []
    ueberschrift: str | None = None
    absaetze: list[str] = []

    def abschliessen() -> None:
        """Segment aus den bisher gesammelten Absätzen der aktuellen Überschrift schließen.

        Eine Überschrift ohne eigenen Folgeabsatz bekommt hier trotzdem ein Segment – mit
        sich selbst als Text, statt kommentarlos zu verschwinden. Ein Dokumenttitel, dem
        sofort die nächste Überschrift oder eine Tabelle folgt, ist im echten Bestand der
        Normalfall (Rechnungskopf, Abschnittsüberschrift vor einer Positionstabelle).
        Die Alternative – die Überschrift an das nächstfolgende Segment „weiterreichen“ –
        wurde verworfen: Folgen zwei Überschriften direkt aufeinander (Titel, dann
        Unterabschnitt), gäbe es keinen eindeutigen Empfänger, ohne Überschriften
        ineinander zu verschachteln oder eine von ihnen doch wieder zu verwerfen. Ein
        eigenes Segment verliert nie Information. Die Zuordnung Überschrift → Tabelle wird
        unabhängig davon beim Antreffen der Tabelle vorgenommen (siehe unten).
        """
        text = normalize_text(" ".join(absaetze)) if absaetze else (ueberschrift or "")
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
        absaetze.clear()

    tabellennummer = 0
    for inhalt in quelle.iter_inner_content():
        if isinstance(inhalt, Paragraph):
            text = inhalt.text.strip()
            if not text:
                continue
            stil = inhalt.style.name if inhalt.style is not None else None
            if stil is not None and stil.startswith("Heading"):
                abschliessen()
                ueberschrift, absaetze = normalize_text(text), []
            else:
                absaetze.append(text)
        else:
            tabellennummer += 1
            zellen = [[_zellentext(zelle) for zelle in zeile.cells] for zeile in inhalt.rows]
            if not any(zelle for zeile in zellen for zelle in zeile):
                continue  # Layout-Tabelle ohne Inhalt – "" | "" ergäbe sonst nur Trennzeichen.
            tabellentext = normalize_text(" ".join(" | ".join(zeile) for zeile in zellen))
            if tabellentext:
                segmente.append(
                    Segment.create(
                        document=document,
                        index=len(segmente),
                        kind=SegmentKind.ABSCHNITT,
                        locator=f"Tabelle {tabellennummer}",
                        heading=ueberschrift,
                        text=tabellentext,
                    )
                )
    abschliessen()
    return segmente


def extract_xlsx(document: Document, data: bytes) -> list[Segment]:
    mappe = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        segmente: list[Segment] = []
        for blatt in mappe.worksheets:
            # Das <dimension ref>-Attribut der Blatt-XML ist oft nur ein Platzhalter
            # (z. B. "A1") bei Erzeugern, die die Ausdehnung beim Schreiben nicht kennen.
            # openpyxl vertraut ihm im read_only-Modus blind und verwirft sonst Spalten
            # und Zeilen kommentarlos; reset_dimensions() erzwingt das Nachmessen. Ohne
            # bekannte Blattbreite polstert openpyxl jede Zeile aber nur noch auf ihre
            # eigene letzte gefüllte Zelle auf, nicht mehr auf eine einheitliche Breite –
            # das verschöbe kopf/zeile in der Zuordnung unten, sobald eine Zeile (etwa die
            # Kopfzeile) kürzer ist als eine andere. Deshalb wird hier selbst auf die
            # größte tatsächlich gelesene Zeilenbreite aufgefüllt, statt sich auf eine
            # interne openpyxl-Neuberechnung zu verlassen.
            blatt.reset_dimensions()
            rohzeilen = list(blatt.iter_rows(values_only=True))
            breite = max((len(zeile) for zeile in rohzeilen), default=0)
            zeilen = [
                [
                    ("" if i >= len(zeile) or zeile[i] is None else str(zeile[i])).strip()
                    for i in range(breite)
                ]
                for zeile in rohzeilen
            ]
            zeilen = [zeile for zeile in zeilen if any(zeile)]
            if not zeilen:
                continue
            kopf = zeilen[0] if sum(bool(z) for z in zeilen[0]) >= MIN_HEADER_CELLS else None
            rest = zeilen[1:] if kopf else zeilen
            if kopf:
                # openpyxl polstert jede gelesene Zeile auf die Spaltenzahl des Blattes auf;
                # ist die Kopfzeile selbst kürzer als eine Datenzeile, blieben die
                # aufgefüllten Leerzellen sonst als zusätzliche "|"-Trennzeichen im Text
                # stehen. Gefiltert wird nur die Anzeigezeichenkette – ``kopf`` selbst bleibt
                # unverändert, weil ``zip(kopf, zeile, ...)`` unten positionsweise arbeitet.
                kopf_anzeige = [name for name in kopf if name]
                formatiert = [
                    ", ".join(
                        f"{name}: {wert}" for name, wert in zip(kopf, zeile, strict=False) if wert
                    )
                    for zeile in rest
                ]
                inhalt = " | ".join([" | ".join(kopf_anzeige), *formatiert])
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
        return segmente
    finally:
        mappe.close()
