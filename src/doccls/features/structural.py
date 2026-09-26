"""Rund 40 billige, erklärbare Zahlen je Dokument (Konzept § 6.3).

Sie sind **Merkmale, keine Regeln**: Niemand schreibt ``if IBAN then Rechnung``, das
Modell entscheidet über das Gewicht. Ihr Wert liegt darin, zu sehen, was ein gemitteltes
Embedding übersieht – ein Vorzeichen, eine IBAN, eine Unterschriftenzeile.

Die Reihenfolge in ``STRUCTURAL_NAMES`` ist die Reihenfolge im Vektor. Sie darf sich nur
zusammen mit der ``feature_version`` ändern: ``coef_`` gegen die Merkmalsnamen zu halten
(Konzept § 7.1) ist die wichtigste Diagnose dieses Projekts, und sie wird stillschweigend
falsch, wenn Namen und Spalten auseinanderlaufen.

Nicht umgesetzt aus § 6.3, mit Begründung:

* **Dateinamen-Tokens** – die Dateinamen wurden in Phase 1 neutralisiert (``doc-0001.pdf``),
  weil sich aus ihnen 480 von 560 Klassen raten ließen. Ein Merkmal daraus trüge jetzt null
  Information.
* **Absenderdomain und Betreff-Tokens der Elternmail** – brauchen einen Rückgriff aufs
  Elterndokument, den die Merkmalsbildung nicht hat. Vertagt auf Phase 3, zusammen mit der
  Überarbeitung der Mailvorlagen.
* **``ocr_applied``** – auf dem gesamten erzeugten Bestand für alle 570 Dokumente ``False``:
  Der Generator erzeugt keine gescannten bzw. OCR-pflichtigen Dokumente (siehe
  ``test_kein_erzeugtes_pdf_gilt_als_ocr_beduerftig`` und die ``MIN_CHARS_PER_PAGE``-Doku in
  ``config.py``). Ein Merkmal ohne jede Streuung im Bestand trägt null Information und
  verwässert nur die Gewichte – dieselbe Begründung wie bei den Dateinamen-Tokens, hier aber
  gemessen statt vorhergesagt (``test_kein_merkmal_ist_auf_dem_ganzen_bestand_konstant``
  schlägt mit dem Feld sofort fehl). Nachzutragen, sobald der Bestand tatsächlich
  OCR-pflichtige Dokumente enthält – das ``Document``-Feld existiert bereits und die
  Ergänzung ist dann eine Zeile in ``MERKMALE``.
* **``digit_ratio``** (Konzept § 6.3 nennt „Anteil Ziffern" als Beispiel) – gemessen über den
  ganzen Bestand ist der Ziffernanteil des Dokuments allein ein Entscheidungsstumpf mit
  74,2 % Trefferquote über die Klasse (Grundrate 14 %, Schranke aus
  ``test_kein_einzelnes_strukturmerkmal_verraet_die_klasse`` 60 %; die zehn Quantil-Körbe
  sortieren fast lückenlos AGB → VERTRAG → STATUSBERICHT → PROTOKOLL → SONSTIGES →
  GUTSCHRIFT → RECHNUNG). Anders als eine IBAN – ein "legitim starker Hinweis" laut
  Testdoc – ist das kein inhaltlicher Hinweis, sondern eine Eigenschaft der 56 Vorlagen
  dieses synthetischen Bestands: Jede Klasse hat eine so eng bemessene, feste
  Zahlendichte, dass der rohe Anteil fast zur Klassensignatur wird – dieselbe Art
  Abkürzung wie die Dateinamen-Tokens, nur über die Vorlage statt über den Pfad. Gemäß
  Auftrag nicht durch Anheben der 60-%-Schranke "gelöst", sondern das Merkmal
  weggelassen.
* **``date_count``** (rohe Anzahl, zusätzlich zu ``date_density``) – aus demselben Grund:
  73,7 % Trefferquote über die Klasse. Jede Vorlage nennt eine feste kleine Anzahl Daten
  (Statusbericht immer 1, Rechnung immer 3–4, Protokoll immer 5–6 …), und bei nur 56
  Vorlagen wird die rohe Anzahl fast zur Vorlagen-, damit Klassensignatur.
  ``date_density`` (längenbezogen, von ``test_datumsdichte_bezieht_sich_auf_die_textlaenge``
  verbindlich verlangt) bleibt – mit 68,1 % selbst über der 60-%-Schranke, aber als
  einziges Merkmal dieses Blocks per Testauftrag nicht entfernbar; siehe die begründete
  Ausnahme in ``tests/test_features_structural.py``.
"""

import re
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from doccls.detect import MEDIA_TYPES, Format
from doccls.models import Document, Segment, SegmentKind

BETRAG = re.compile(r"\d{1,3}(?:\.\d{3})*,\d{2}\s*(?:€|EUR\b)")
"""``\\b`` steht bewusst nur hinter ``EUR``, nicht hinter ``€``: ``€`` ist kein
Wortzeichen, ein ``\\b`` unmittelbar danach kann also nie eine Grenze finden (weder vor
Leerraum noch am Textende) und liesse jeden Betrag in Euro-Zeichen-Schreibweise
stillschweigend durchfallen – mit der wörtlichen Fassung aus dem Brief zählt
``test_betraege_werden_gezaehlt`` nur 2 statt 3 Treffer."""
BETRAGSZAHL = re.compile(r"\d{1,3}(?:\.\d{3})*,\d{2}")
"""Nur der Zahlenteil eines Treffers aus ``BETRAG`` – gebraucht, um den Wert zu parsen
(``max_amount_log``), ohne die Waehrungsangabe mit einzuschliessen."""
PROZENT = re.compile(r"\d{1,3}(?:,\d+)?\s*%")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b")
"""Reale IBAN werden in Vierergruppen mit Leerzeichen geschrieben
(``DE02 1203 0000 0000 2020 51``, siehe ``src/doccls/generation/content.py``), nicht als
eine zusammenhaengende Zeichenkette. Die wörtliche Fassung aus dem Brief
(``[A-Z0-9]{11,30}`` ohne Leerraum) trifft deshalb auf dem gesamten erzeugten Bestand kein
einziges Mal – ``iban_count`` waere dort konstant null (gemessen, nicht vermutet). Je
optionalem Leerzeichen vor einem Alphanumerikzeichen zaehlt das Muster weiterhin genau wie
zuvor auf einer zusammenhaengenden IBAN wie im Testbeispiel."""
USTID = re.compile(r"\bDE\s?\d{9}\b")
STEUERNUMMER = re.compile(r"\b\d{2,3}/\d{3}/\d{4,5}\b")
BELEGNUMMER = re.compile(r"\b[A-Z]{2,4}-\d{4}-\d{3,6}\b")
DATUM = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b")
PARAGRAF = re.compile(r"§\s*\d+")
TOP = re.compile(r"\bTOP\b(?:\s*\d+)?", re.IGNORECASE)
"""Die wörtliche Fassung aus dem Brief (``TOP\\s*\\d+``, eine Ziffer zwingend) geht von
Fließtext wie „TOP 1 Begrüßung" aus. Der reale Generator schreibt Tagesordnungspunkte als
Tabelle (Kopfzeile ``TOP | Thema | Beschluss | ...``, die Nummern erst in den Datenzeilen);
mit der strengen Fassung waere ``top_count`` auf dem ganzen Bestand konstant null. Die
Ziffer ist deshalb optional – „TOP" allein zaehlt bereits als Treffer."""
UNTERSCHRIFT = re.compile(r"_{5,}|\bUnterschrift\b", re.IGNORECASE)
NEGATIVBETRAG = re.compile(r"-\s*\d{1,3}(?:\.\d{3})*,\d{2}\s*(?:€|EUR\b)")

SHORT_SEGMENT_CHARS = 80
"""Unter dieser Zeichenzahl gilt ein Segment als kurz – eine Unterschriftenzeile oder eine
Zwischenueberschrift ohne eigenen Absatz, kein normaler Flieszttextabschnitt."""

SCHLUESSELWOERTER = (
    "zahlbar",
    "gutschrift",
    "vertragsgegenstand",
    "kündigung",
    "teilnehmer",
    "rechnungsdatum",
    "erstattung",
    "geltungsbereich",
    "protokoll",
    "ampel",
    "fortschritt",
    "leistungsverzeichnis",
)
"""Startliste. Konzept § 6.3 will sie **aus den Labels gelernt** haben – nach jeder
Trainingsrunde die Terme mit höchster punktweiser Transinformation je Klasse ausgeben und
ergänzen. Das Ausgeben kommt in Phase 3; hier steht der Anfang, bewusst als Liste und
nicht als Regel.

Sechs der ursprünglich im Brief wörtlich vorgegebenen Wörter (``zahlungsziel``,
``vertragspartner``, ``kuendigungsfrist``, ``anwesend``, ``rechnungsnummer``,
``leistungsbeschreibung``) kommen im gesamten erzeugten Bestand kein einziges Mal vor
(``src/doccls/generation/content.py`` verwendet andere Formulierungen: „Zahlbar bis …“,
„§ 1 Vertragsgegenstand“, „§ 4 Kündigung“, „Teilnehmer: …“, „Rechnungsdatum: …“,
„Leistungsverzeichnis“). Mit den Originalwörtern wären die zugehörigen ``kw_*``-Merkmale
auf dem ganzen Bestand konstant null – genau der Fall, den
``test_kein_merkmal_ist_auf_dem_ganzen_bestand_konstant`` fangen soll. Ersetzt durch
Wörter aus derselben semantischen Gruppe, die im Bestand tatsächlich vorkommen (gemessen
über ``data/parquet/segments.parquet``, nicht vermutet)."""


def _text(segments: list[Segment]) -> str:
    return " ".join(s.text for s in segments)


def _je_tausend(treffer: int, laenge: int) -> float:
    """Dichte statt Anzahl: Eine reine Anzahl waere nur ein verkapptes Laengenmass."""
    return 1000.0 * treffer / laenge if laenge else 0.0


def _betrag_zu_zahl(treffer: str) -> float:
    """Den in einem ``BETRAG``-Treffer enthaltenen Zahlenwert als ``float`` – deutsches
    Format (Tausenderpunkt, Komma als Dezimaltrennzeichen) in das uebliche umgerechnet."""
    ziffern = BETRAGSZAHL.match(treffer)
    assert ziffern is not None, f"BETRAG-Treffer ohne Zahlenanteil: {treffer!r}"
    return float(ziffern.group(0).replace(".", "").replace(",", "."))


def _max_amount_log(document: Document, segments: list[Segment], text: str) -> float:
    """``log1p`` des groessten gefundenen Betrags – roh gingen Betraege ueber vier
    Groessenordnungen und erschluegen jedes andere Merkmal."""
    betraege = [_betrag_zu_zahl(treffer) for treffer in BETRAG.findall(text)]
    return float(np.log1p(max(betraege))) if betraege else 0.0


def _anteil(segments: list[Segment], passt: Callable[[Segment], bool]) -> float:
    """Anteil der Segmente, auf die ``passt`` zutrifft – 0.0 bei einem leeren Dokument,
    statt durch null zu teilen."""
    return sum(1 for s in segments if passt(s)) / len(segments) if segments else 0.0


def _kw_merkmal(wort: str) -> Callable[[Document, list[Segment], str], float]:
    """Eine Abschlussfunktion je Schluesselwort statt einer Schleifenvariable, die in
    der Lambda-Closure spaeter ausgewertet wuerde und dann bei allen Woertern dasselbe,
    letzte Wort traefe."""

    def merkmal(document: Document, segments: list[Segment], text: str) -> float:
        return _je_tausend(text.lower().count(wort), len(text))

    return merkmal


MERKMALE: dict[str, Callable[[Document, list[Segment], str], float]] = {
    # Umfang
    "segment_count": lambda d, s, t: float(len(s)),
    "chars_total": lambda d, s, t: float(len(t)),
    "chars_per_segment": lambda d, s, t: float(len(t)) / len(s) if s else 0.0,
    "distinct_headings": lambda d, s, t: float(len({x.heading for x in s if x.heading})),
    # Zahlen: `digit_ratio` bewusst nicht enthalten, siehe Moduldoc.
    "amount_count": lambda d, s, t: float(len(BETRAG.findall(t))),
    "amount_density": lambda d, s, t: _je_tausend(len(BETRAG.findall(t)), len(t)),
    "percent_count": lambda d, s, t: float(len(PROZENT.findall(t))),
    "has_negative_amount": lambda d, s, t: 1.0 if NEGATIVBETRAG.search(t) else 0.0,
    "max_amount_log": _max_amount_log,
    # Muster
    "iban_count": lambda d, s, t: float(len(IBAN.findall(t))),
    "ustid_count": lambda d, s, t: float(len(USTID.findall(t))),
    "steuernummer_count": lambda d, s, t: float(len(STEUERNUMMER.findall(t))),
    "belegnummer_count": lambda d, s, t: float(len(BELEGNUMMER.findall(t))),
    # `date_count` (rohe Anzahl) bewusst nicht enthalten, siehe Moduldoc.
    "date_density": lambda d, s, t: _je_tausend(len(DATUM.findall(t)), len(t)),
    "paragraph_count": lambda d, s, t: float(len(PARAGRAF.findall(t))),
    "top_count": lambda d, s, t: float(len(TOP.findall(t))),
    # Layout
    # `table_segment_share` und `sheet_segment_share` laufen ueber SegmentKind, nicht
    # ueber den locator-Text. `SegmentKind` unterscheidet Tabellen aktuell nur als ganzes
    # Blatt (`BLATT`, siehe extraction/office.py) – eine Tabelle innerhalb eines DOCX
    # bekommt keine eigene Art und teilt `ABSCHNITT` mit gewoehnlichem Flieszttext; eine
    # zeilenweise Aufloesung (`SegmentKind.ZEILE`) ist im Bestand nie erzeugt (siehe
    # docs/phase-1-offene-punkte.md). `table_segment_share` ist deshalb bewusst weiter
    # gefasst (`BLATT` oder `ZEILE`) als `sheet_segment_share` (nur `BLATT`) – auf dem
    # heutigen Bestand identisch, aber nicht redundant, sobald Zeilen-Segmente entstehen.
    "table_segment_share": lambda d, s, t: _anteil(
        s, lambda x: x.kind in (SegmentKind.BLATT, SegmentKind.ZEILE)
    ),
    "sheet_segment_share": lambda d, s, t: _anteil(s, lambda x: x.kind == SegmentKind.BLATT),
    "short_segment_ratio": lambda d, s, t: _anteil(s, lambda x: len(x.text) < SHORT_SEGMENT_CHARS),
    "avg_segment_len": lambda d, s, t: sum(len(x.text) for x in s) / len(s) if s else 0.0,
    "has_signature_line": lambda d, s, t: 1.0 if UNTERSCHRIFT.search(t) else 0.0,
    # Schluesselwoerter
    **{f"kw_{wort}": _kw_merkmal(wort) for wort in SCHLUESSELWOERTER},
    # Herkunft
    "is_pdf": lambda d, s, t: 1.0 if d.media_type == MEDIA_TYPES[Format.PDF] else 0.0,
    "is_docx": lambda d, s, t: 1.0 if d.media_type == MEDIA_TYPES[Format.DOCX] else 0.0,
    "is_xlsx": lambda d, s, t: 1.0 if d.media_type == MEDIA_TYPES[Format.XLSX] else 0.0,
    "is_eml": lambda d, s, t: 1.0 if d.media_type == MEDIA_TYPES[Format.EML] else 0.0,
    "is_attachment": lambda d, s, t: 1.0 if d.parent_document_id is not None else 0.0,
    # Verarbeitung: `ocr_applied` bewusst nicht enthalten, siehe Moduldoc.
}

STRUCTURAL_NAMES: tuple[str, ...] = tuple(MERKMALE)


def structural_features(document: Document, segments: list[Segment]) -> npt.NDArray[np.float32]:
    """Ein Wert je Eintrag in ``MERKMALE``, in genau dieser Reihenfolge."""
    text = _text(segments)
    return np.array(
        [funktion(document, segments, text) for funktion in MERKMALE.values()],
        dtype=np.float32,
    )
