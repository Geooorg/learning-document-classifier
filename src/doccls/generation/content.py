"""Inhalte der synthetischen Testdokumente – reine Funktionen, kein Dateisystem.

Aufbau: ``TEMPLATES`` beschreibt je Klasse mehrere Vorlagen; ``build_corpus`` erzeugt je
Vorlage zehn Varianten mit anderen Zahlen, Namen und Daten. Der Gold-Set-Schnitt läuft
später über ``template_id``, nicht über Dokumente – zwei Varianten derselben Vorlage sind
fast dasselbe Dokument und dürfen nicht auf beide Seiten des Schnitts fallen
(Konzept § 9.2).

Die Wahrheit, die hier konstruiert wird, ist in ``docs/testdaten.md`` beschrieben. Beides
muss gemeinsam gepflegt werden.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass

VARIANTS_PER_TEMPLATE = 10

FOOTER = (
    "Muster Handels GmbH · Industriestraße 17 · 40211 Düsseldorf · "
    "HRB 44821 Amtsgericht Düsseldorf · USt-IdNr. DE812345678"
)
"""Gemeinsame Fußzeile über alle Klassen. Wäre sie klassenspezifisch, würde das Modell sie
statt des Inhalts lernen (Konzept § 8.5, Abkürzungen)."""

FIRMEN = [
    "Nordwind Logistik GmbH",
    "Steinbach Elektro KG",
    "Vogel & Partner mbB",
    "Auric Systems AG",
    "Lindner Bau GmbH",
    "Kessler Datentechnik e.K.",
]
PERSONEN = ["A. Albers", "B. Vogt", "C. Yilmaz", "D. Brandt", "E. Kessler", "F. Radtke"]
STAEDTE = ["Düsseldorf", "Leipzig", "Bremen", "Augsburg", "Rostock", "Kassel"]


@dataclass(frozen=True)
class Block:
    """Ein Abschnitt: optionale Überschrift, Absätze, optional eine Tabelle."""

    heading: str | None = None
    paragraphs: tuple[str, ...] = ()
    table: tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] | None = None


@dataclass(frozen=True)
class DocumentSpec:
    """Ein zu erzeugendes Dokument, unabhängig vom Ausgabeformat."""

    class_key: str
    template_id: str
    variant: int
    title: str
    blocks: tuple[Block, ...]
    formats: tuple[str, ...]


@dataclass(frozen=True)
class Template:
    """Eine Dokumentvorlage. ``build`` bekommt einen geseedeten Zufall und liefert die Blöcke."""

    template_id: str
    class_key: str
    formats: tuple[str, ...]
    build: Callable[[random.Random], tuple[str, tuple[Block, ...]]]
    """Liefert (Titel, Blöcke). Der Titel entsteht erst beim Bauen, weil er die Nummer der
    Variante nennt – „Rechnung RE-2026-4711“."""


def _betrag(rnd: random.Random) -> tuple[float, float, float]:
    """Netto, Umsatzsteuer, Brutto – konsistent gerundet, damit sie zueinander passen."""
    netto = round(rnd.uniform(180, 24_000), 2)
    ust = round(netto * 0.19, 2)
    return netto, ust, round(netto + ust, 2)


def _eur(wert: float) -> str:
    return f"{wert:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".") + " EUR"


def _kopf(rnd: random.Random) -> Block:
    """Briefkopf – für alle Klassen gleich aufgebaut."""
    return Block(
        paragraphs=(
            "Muster Handels GmbH, Industriestraße 17, 40211 Düsseldorf",
            f"{rnd.choice(FIRMEN)}, {rnd.choice(STAEDTE)}",
        )
    )


def _fuss() -> Block:
    return Block(paragraphs=(FOOTER,))


# --------------------------------------------------------------------------- Vorlagen


def _rechnung(rnd: random.Random, *, storniert: bool = False) -> tuple[str, tuple[Block, ...]]:
    nummer = f"RE-2026-{rnd.randint(1000, 9999)}"
    netto, ust, brutto = _betrag(rnd)
    tag = rnd.randint(1, 28)
    zeilen = tuple(
        (str(i), leistung, str(rnd.randint(1, 20)), _eur(round(netto / 3, 2)))
        for i, leistung in enumerate(
            rnd.sample(
                [
                    "Wartung Anlagentechnik",
                    "Materiallieferung Profilschienen",
                    "Montagestunden",
                    "Transport und Verpackung",
                    "Softwarepflege Jahreslizenz",
                ],
                3,
            ),
            start=1,
        )
    )
    hinweis = (
        (f"Diese Rechnung ersetzt die stornierte Rechnung RE-2026-{rnd.randint(1000, 9999)}.")
        if storniert
        else ""
    )
    return (
        f"Rechnung {nummer}",
        (
            _kopf(rnd),
            Block(
                heading=f"Rechnung {nummer}",
                paragraphs=tuple(
                    p
                    for p in (
                        f"Rechnungsdatum: {tag:02d}.03.2026",
                        f"Leistungszeitraum: 01.02.2026 bis {tag:02d}.02.2026",
                        f"Kundennummer: K-{rnd.randint(10000, 99999)}",
                        hinweis,
                    )
                    if p
                ),
            ),
            Block(
                heading="Positionen", table=(("Pos.", "Leistung", "Menge", "Einzelpreis"), zeilen)
            ),
            Block(
                heading="Zahlung",
                paragraphs=(
                    f"Nettobetrag: {_eur(netto)}",
                    f"zzgl. 19 % Umsatzsteuer: {_eur(ust)}",
                    f"Rechnungsbetrag: {_eur(brutto)}",
                    f"Zahlbar bis {tag:02d}.04.2026 ohne Abzug.",
                    "Bankverbindung: IBAN DE02 1203 0000 0000 2020 51, BIC BYLADEM1001",
                ),
            ),
            _fuss(),
        ),
    )


def _gutschrift(rnd: random.Random, *, grund: str) -> tuple[str, tuple[Block, ...]]:
    """Bewusst in Rechnungsaufmachung – der harte Verwechslungsfall (Konzept Anhang D)."""
    nummer = f"GU-2026-{rnd.randint(100, 999)}"
    ursprung = f"RE-2026-{rnd.randint(1000, 9999)}"
    netto, ust, brutto = _betrag(rnd)
    return (
        f"Gutschrift {nummer}",
        (
            _kopf(rnd),
            Block(
                heading=f"Gutschrift {nummer}",
                paragraphs=(
                    f"Gutschriftdatum: {rnd.randint(1, 28):02d}.04.2026",
                    f"Bezug: Rechnung {ursprung} vom {rnd.randint(1, 28):02d}.03.2026",
                    f"Grund der Gutschrift: {grund}",
                ),
            ),
            Block(
                heading="Positionen",
                table=(
                    ("Pos.", "Bezug", "Menge", "Betrag"),
                    (("1", f"Teilstorno zu {ursprung}", "1", _eur(-netto)),),
                ),
            ),
            Block(
                heading="Erstattung",
                paragraphs=(
                    f"Nettobetrag: {_eur(-netto)}",
                    f"zzgl. 19 % Umsatzsteuer: {_eur(-ust)}",
                    f"Gutschriftbetrag: {_eur(-brutto)}",
                    "Der Betrag wird auf das uns bekannte Konto erstattet. Eine Zahlung Ihrerseits "
                    "ist nicht erforderlich.",
                ),
            ),
            _fuss(),
        ),
    )


def _vertrag(
    rnd: random.Random, *, art: str, mit_agb_anhang: bool
) -> tuple[str, tuple[Block, ...]]:
    partner, vertreter = rnd.choice(FIRMEN), rnd.choice(PERSONEN)
    bloecke = [
        _kopf(rnd),
        Block(
            heading=f"{art} zwischen Muster Handels GmbH und {partner}",
            paragraphs=(
                f"Die Muster Handels GmbH, Industriestraße 17, 40211 Düsseldorf, vertreten durch "
                f"{rnd.choice(PERSONEN)} – nachfolgend Auftragnehmer –",
                f"und {partner}, {rnd.choice(STAEDTE)}, vertreten durch {vertreter} "
                "– nachfolgend Auftraggeber – schließen folgenden Vertrag:",
            ),
        ),
        Block(
            heading="§ 1 Vertragsgegenstand",
            paragraphs=(
                f"Der Auftragnehmer erbringt die Leistungen gemäß Leistungsverzeichnis "
                f"vom {rnd.randint(1, 28):02d}.01.2026.",
            ),
        ),
        Block(
            heading="§ 2 Laufzeit",
            paragraphs=(
                f"Der Vertrag beginnt am 01.{rnd.randint(1, 9):02d}.2026 und läuft "
                f"{rnd.choice(['12', '24', '36'])} Monate.",
            ),
        ),
        Block(
            heading="§ 3 Vergütung",
            paragraphs=(
                f"Die Vergütung beträgt {_eur(round(rnd.uniform(1200, 9000), 2))} monatlich "
                "zuzüglich gesetzlicher Umsatzsteuer.",
            ),
        ),
        Block(
            heading="§ 4 Kündigung",
            paragraphs=(
                "Der Vertrag kann mit einer Frist von drei Monaten zum Quartalsende "
                "schriftlich gekündigt werden.",
            ),
        ),
    ]
    if mit_agb_anhang:
        bloecke.append(
            Block(
                heading="Anlage 1: Allgemeine Geschäftsbedingungen",
                paragraphs=(
                    "Es gelten ergänzend die AGB des Auftragnehmers in der Fassung vom 01.01.2026.",
                ),
            )
        )
    bloecke.append(
        Block(
            heading="Unterschriften",
            paragraphs=(
                "Düsseldorf, den ______________    ______________________ (Auftragnehmer)",
                f"{rnd.choice(STAEDTE)}, den ______________    ______________________ "
                "(Auftraggeber)",
            ),
        )
    )
    bloecke.append(_fuss())
    return (f"{art} {partner}", tuple(bloecke))


def _agb(rnd: random.Random, *, bereich: str) -> tuple[str, tuple[Block, ...]]:
    fassung = (
        f"Fassung vom 01.{rnd.randint(1, 9):02d}.2026, "
        f"Version {rnd.randint(2, 9)}.{rnd.randint(0, 9)}"
    )
    zahlungsziel = rnd.choice(("14", "30", "45", "60"))
    gewaehrleistung = rnd.choice(("zwölf", "vierundzwanzig", "sechsunddreißig"))
    paragraphen = [
        (
            "§ 1 Geltungsbereich",
            "Diese Bedingungen gelten für alle Verträge zwischen dem Anbieter und Unternehmern "
            "im Sinne des § 14 BGB. Abweichende Bedingungen des Kunden werden nicht Vertrags"
            "bestandteil, auch wenn ihnen nicht ausdrücklich widersprochen wird.",
        ),
        (
            "§ 2 Vertragsschluss",
            "Angebote des Anbieters sind freibleibend. Ein Vertrag kommt erst mit schriftlicher "
            "Auftragsbestätigung zustande.",
        ),
        (
            "§ 3 Preise und Zahlung",
            "Alle Preise verstehen sich netto zuzüglich Umsatzsteuer. Rechnungen sind innerhalb "
            f"von {zahlungsziel} Tagen ohne Abzug zur Zahlung fällig.",
        ),
        (
            "§ 4 Gewährleistung",
            f"Die Gewährleistungsfrist beträgt {gewaehrleistung} Monate ab Gefahrübergang.",
        ),
        (
            "§ 5 Haftung",
            "Der Anbieter haftet unbeschränkt bei Vorsatz und grober Fahrlässigkeit. Im Übrigen "
            "ist die Haftung auf den vertragstypischen, vorhersehbaren Schaden begrenzt.",
        ),
        (
            "§ 6 Gerichtsstand",
            f"Gerichtsstand für alle Streitigkeiten ist {rnd.choice(STAEDTE)}. Es gilt "
            "ausschließlich deutsches Recht.",
        ),
    ]
    return (
        f"Allgemeine Geschäftsbedingungen {bereich}",
        (
            _kopf(rnd),
            Block(heading=f"Allgemeine Geschäftsbedingungen – {bereich}", paragraphs=(fassung,)),
            *[Block(heading=titel, paragraphs=(text,)) for titel, text in paragraphen],
            _fuss(),
        ),
    )


def _statusbericht(rnd: random.Random, *, ampel: str) -> tuple[str, tuple[Block, ...]]:
    kw = rnd.randint(10, 40)
    return (
        f"Statusbericht KW {kw}",
        (
            _kopf(rnd),
            Block(
                heading=f"Statusbericht – Berichtszeitraum KW {kw - 1} bis KW {kw}/2026",
                paragraphs=(
                    f"Gesamtstatus: {ampel}",
                    f"Fortschritt: {rnd.randint(15, 95)} % der geplanten Leistung.",
                ),
            ),
            Block(
                heading="Erreichte Meilensteine",
                table=(
                    ("Meilenstein", "Geplant", "Erreicht"),
                    (
                        ("Anforderungsaufnahme", "KW 08", "KW 08"),
                        ("Teillieferung 1", f"KW {kw - 2}", f"KW {kw - 1}"),
                        ("Abnahmetest", f"KW {kw + 4}", "offen"),
                    ),
                ),
            ),
            Block(
                heading="Offene Punkte und Risiken",
                paragraphs=(
                    f"Die Lieferung der Bauteile verzögert sich um {rnd.randint(1, 6)} Wochen.",
                    "Gegenmaßnahme: Ersatzlieferant angefragt, Entscheidung in der kommenden "
                    "Woche.",
                ),
            ),
            Block(
                heading="Ausblick", paragraphs=(f"In KW {kw + 1} beginnt die Integrationsphase.",)
            ),
            _fuss(),
        ),
    )


def _protokoll(rnd: random.Random, *, berichtsform: bool) -> tuple[str, tuple[Block, ...]]:
    """``berichtsform`` erzeugt Protokolle, die wie Statusberichte aussehen – Verwechslungsfall."""
    nummer, tag = rnd.randint(1, 30), rnd.randint(1, 28)
    tops = (
        ("1", "Freigabe Lastenheft", "Freigegeben", rnd.choice(PERSONEN), f"{tag:02d}.05.2026"),
        ("2", "Budgetnachtrag", "Vertagt", rnd.choice(PERSONEN), f"{tag:02d}.06.2026"),
        ("3", "Termine Abnahme", "Beschlossen", rnd.choice(PERSONEN), f"{tag:02d}.07.2026"),
    )
    kopf = Block(
        heading=f"Protokoll der {nummer}. Projektbesprechung",
        paragraphs=(
            f"Datum: {tag:02d}.04.2026    Ort: {rnd.choice(STAEDTE)}, Besprechungsraum 2",
            "Teilnehmer: " + ", ".join(rnd.sample(PERSONEN, 4)),
            "Protokollführung: " + rnd.choice(PERSONEN),
        ),
    )
    inhalt = (
        Block(
            heading="Zusammenfassung des Sitzungsverlaufs",
            paragraphs=(
                "Der Stand der Arbeitspakete wurde besprochen; die Ampel steht auf gelb.",
                f"Der Nachtrag über {_eur(round(rnd.uniform(2000, 40000), 2))} wurde vertagt.",
            ),
        )
        if berichtsform
        else Block(
            heading="Tagesordnungspunkte",
            table=(("TOP", "Thema", "Beschluss", "Verantwortlich", "Termin"), tops),
        )
    )
    return (
        f"Protokoll Besprechung Nr. {nummer}",
        (
            _kopf(rnd),
            kopf,
            inhalt,
            Block(
                heading="Nächste Sitzung",
                paragraphs=(
                    f"Die nächste Besprechung findet am {tag:02d}.05.2026 statt.",
                    "Einwände gegen dieses Protokoll sind binnen fünf Werktagen mitzuteilen.",
                ),
            ),
            _fuss(),
        ),
    )


def _sonstiges(rnd: random.Random, *, art: str) -> tuple[str, tuple[Block, ...]]:
    """Die Restklasse ist mit Absicht heterogen – sie hat kein gemeinsames Merkmal (Konzept § 2)."""
    texte = {
        "Anschreiben": (
            "Sehr geehrte Damen und Herren,",
            "anbei erhalten Sie die angeforderten Unterlagen zur weiteren "
            "Verwendung. Für Rückfragen stehen wir gern zur Verfügung.",
            "Mit freundlichen Grüßen",
        ),
        "Lieferschein": (
            f"Lieferschein-Nr. LS-2026-{rnd.randint(100, 999)}",
            "Die nachstehend aufgeführten Waren wurden vollständig geliefert. "
            "Beträge sind diesem Beleg nicht zu entnehmen.",
        ),
        "Werbebrief": (
            "Jetzt Frühbucherrabatt sichern!",
            "Bis zum 30.06.2026 erhalten Sie auf unser gesamtes Zubehörsortiment "
            "einen Nachlass. Sprechen Sie uns an.",
        ),
        "Bedienungsanleitung": (
            "Inbetriebnahme",
            "Verbinden Sie das Gerät mit der Spannungsversorgung und "
            "warten Sie, bis die Betriebsanzeige dauerhaft leuchtet.",
        ),
        "Bescheinigung": (
            f"Bescheinigung Nr. B-{rnd.randint(100, 999)}",
            "Hiermit wird bescheinigt, dass die genannte Person im Zeitraum "
            "01.01.2026 bis 31.03.2026 an der Schulung teilgenommen hat.",
        ),
    }
    kennung = (
        f"Dokument-Nr. {rnd.choice(['A', 'B', 'K', 'W'])}-{rnd.randint(1000, 9999)} "
        f"vom {rnd.randint(1, 28):02d}.{rnd.randint(1, 12):02d}.2026"
    )
    return (art, (_kopf(rnd), Block(heading=art, paragraphs=(kennung, *texte[art])), _fuss()))


TEMPLATES: tuple[Template, ...] = (
    # RECHNUNG – in vier Formaten, damit das Format nichts verrät
    Template("RECHNUNG-standard", "RECHNUNG", ("pdf",), _rechnung),
    Template(
        "RECHNUNG-storno-ersatz", "RECHNUNG", ("pdf",), lambda r: _rechnung(r, storniert=True)
    ),
    Template("RECHNUNG-tabelle", "RECHNUNG", ("xlsx",), _rechnung),
    Template("RECHNUNG-mail", "RECHNUNG", ("eml",), _rechnung),
    Template("RECHNUNG-docx", "RECHNUNG", ("docx",), _rechnung),
    Template("RECHNUNG-mail-anhang", "RECHNUNG", ("eml",), _rechnung),
    Template("RECHNUNG-storno-xlsx", "RECHNUNG", ("xlsx",), lambda r: _rechnung(r, storniert=True)),
    Template("RECHNUNG-storno-docx", "RECHNUNG", ("docx",), lambda r: _rechnung(r, storniert=True)),
    Template(
        "GUTSCHRIFT-storno",
        "GUTSCHRIFT",
        ("pdf",),
        lambda r: _gutschrift(r, grund="Storno der Lieferung"),
    ),
    Template(
        "GUTSCHRIFT-maengel",
        "GUTSCHRIFT",
        ("pdf",),
        lambda r: _gutschrift(r, grund="Mängelrüge, Preisminderung"),
    ),
    Template(
        "GUTSCHRIFT-ruecksendung",
        "GUTSCHRIFT",
        ("docx",),
        lambda r: _gutschrift(r, grund="Rücksendung der Ware"),
    ),
    Template(
        "GUTSCHRIFT-tabelle", "GUTSCHRIFT", ("xlsx",), lambda r: _gutschrift(r, grund="Teilstorno")
    ),
    Template(
        "GUTSCHRIFT-mail",
        "GUTSCHRIFT",
        ("eml",),
        lambda r: _gutschrift(r, grund="Doppelberechnung"),
    ),
    Template(
        "GUTSCHRIFT-skonto",
        "GUTSCHRIFT",
        ("pdf",),
        lambda r: _gutschrift(r, grund="Nachträglicher Skontoabzug"),
    ),
    Template(
        "GUTSCHRIFT-bonus", "GUTSCHRIFT", ("docx",), lambda r: _gutschrift(r, grund="Jahresbonus")
    ),
    Template(
        "GUTSCHRIFT-fracht",
        "GUTSCHRIFT",
        ("xlsx",),
        lambda r: _gutschrift(r, grund="Zu viel berechnete Frachtkosten"),
    ),
    Template(
        "VERTRAG-dienst",
        "VERTRAG",
        ("pdf",),
        lambda r: _vertrag(r, art="Dienstleistungsvertrag", mit_agb_anhang=False),
    ),
    Template(
        "VERTRAG-wartung",
        "VERTRAG",
        ("docx",),
        lambda r: _vertrag(r, art="Wartungsvertrag", mit_agb_anhang=False),
    ),
    Template(
        "VERTRAG-miete",
        "VERTRAG",
        ("pdf",),
        lambda r: _vertrag(r, art="Mietvertrag", mit_agb_anhang=False),
    ),
    Template(
        "VERTRAG-mit-agb",
        "VERTRAG",
        ("pdf",),
        lambda r: _vertrag(r, art="Liefervertrag", mit_agb_anhang=True),
    ),
    Template(
        "VERTRAG-rahmen",
        "VERTRAG",
        ("docx",),
        lambda r: _vertrag(r, art="Rahmenvertrag", mit_agb_anhang=True),
    ),
    Template(
        "VERTRAG-werk",
        "VERTRAG",
        ("pdf",),
        lambda r: _vertrag(r, art="Werkvertrag", mit_agb_anhang=False),
    ),
    Template(
        "VERTRAG-lizenz",
        "VERTRAG",
        ("docx",),
        lambda r: _vertrag(r, art="Lizenzvertrag", mit_agb_anhang=True),
    ),
    Template(
        "VERTRAG-mail",
        "VERTRAG",
        ("eml",),
        lambda r: _vertrag(r, art="Beratungsvertrag", mit_agb_anhang=False),
    ),
    Template("AGB-allgemein", "AGB", ("pdf",), lambda r: _agb(r, bereich="Lieferung und Leistung")),
    Template("AGB-einkauf", "AGB", ("pdf",), lambda r: _agb(r, bereich="Einkauf")),
    Template("AGB-online", "AGB", ("docx",), lambda r: _agb(r, bereich="Onlineshop")),
    Template("AGB-vermietung", "AGB", ("pdf",), lambda r: _agb(r, bereich="Vermietung")),
    Template("AGB-service", "AGB", ("docx",), lambda r: _agb(r, bereich="Servicearbeiten")),
    Template("AGB-software", "AGB", ("pdf",), lambda r: _agb(r, bereich="Softwareüberlassung")),
    Template("AGB-transport", "AGB", ("docx",), lambda r: _agb(r, bereich="Transport")),
    Template("AGB-mail", "AGB", ("eml",), lambda r: _agb(r, bereich="Wartung")),
    Template("STATUS-gruen", "STATUSBERICHT", ("docx",), lambda r: _statusbericht(r, ampel="grün")),
    Template("STATUS-gelb", "STATUSBERICHT", ("docx",), lambda r: _statusbericht(r, ampel="gelb")),
    Template("STATUS-rot", "STATUSBERICHT", ("pdf",), lambda r: _statusbericht(r, ampel="rot")),
    Template(
        "STATUS-pdf-gruen", "STATUSBERICHT", ("pdf",), lambda r: _statusbericht(r, ampel="grün")
    ),
    Template("STATUS-xlsx", "STATUSBERICHT", ("xlsx",), lambda r: _statusbericht(r, ampel="gelb")),
    Template("STATUS-mail", "STATUSBERICHT", ("eml",), lambda r: _statusbericht(r, ampel="grün")),
    Template("STATUS-monat", "STATUSBERICHT", ("docx",), lambda r: _statusbericht(r, ampel="gelb")),
    Template("STATUS-quartal", "STATUSBERICHT", ("pdf",), lambda r: _statusbericht(r, ampel="rot")),
    Template("PROTOKOLL-top", "PROTOKOLL", ("docx",), lambda r: _protokoll(r, berichtsform=False)),
    Template(
        "PROTOKOLL-top-pdf", "PROTOKOLL", ("pdf",), lambda r: _protokoll(r, berichtsform=False)
    ),
    Template(
        "PROTOKOLL-bericht", "PROTOKOLL", ("docx",), lambda r: _protokoll(r, berichtsform=True)
    ),
    Template(
        "PROTOKOLL-bericht-pdf", "PROTOKOLL", ("pdf",), lambda r: _protokoll(r, berichtsform=True)
    ),
    Template(
        "PROTOKOLL-jour-fixe", "PROTOKOLL", ("docx",), lambda r: _protokoll(r, berichtsform=False)
    ),
    Template("PROTOKOLL-mail", "PROTOKOLL", ("eml",), lambda r: _protokoll(r, berichtsform=False)),
    Template("PROTOKOLL-xlsx", "PROTOKOLL", ("xlsx",), lambda r: _protokoll(r, berichtsform=False)),
    Template(
        "PROTOKOLL-abnahme", "PROTOKOLL", ("pdf",), lambda r: _protokoll(r, berichtsform=False)
    ),
    Template(
        "SONST-anschreiben", "SONSTIGES", ("pdf",), lambda r: _sonstiges(r, art="Anschreiben")
    ),
    Template(
        "SONST-anschreiben-mail", "SONSTIGES", ("eml",), lambda r: _sonstiges(r, art="Anschreiben")
    ),
    Template(
        "SONST-lieferschein", "SONSTIGES", ("pdf",), lambda r: _sonstiges(r, art="Lieferschein")
    ),
    Template(
        "SONST-lieferschein-xlsx",
        "SONSTIGES",
        ("xlsx",),
        lambda r: _sonstiges(r, art="Lieferschein"),
    ),
    Template("SONST-werbung", "SONSTIGES", ("docx",), lambda r: _sonstiges(r, art="Werbebrief")),
    Template(
        "SONST-anleitung", "SONSTIGES", ("pdf",), lambda r: _sonstiges(r, art="Bedienungsanleitung")
    ),
    Template(
        "SONST-bescheinigung", "SONSTIGES", ("docx",), lambda r: _sonstiges(r, art="Bescheinigung")
    ),
    Template(
        "SONST-bescheinigung-pdf",
        "SONSTIGES",
        ("pdf",),
        lambda r: _sonstiges(r, art="Bescheinigung"),
    ),
)


def build_corpus(seed: int = 42) -> list[DocumentSpec]:
    """Je Vorlage ``VARIANTS_PER_TEMPLATE`` Varianten.

    Der Zufall wird je (Vorlage, Variante) neu geseedet. Dadurch bleibt eine bestehende
    Variante gleich, auch wenn vorher eine Vorlage ergänzt wird – sonst änderten sich alle
    ``document_id`` und das Gold-Set wäre wertlos.
    """
    specs: list[DocumentSpec] = []
    for template in TEMPLATES:
        for variant in range(VARIANTS_PER_TEMPLATE):
            rnd = random.Random(f"{seed}:{template.template_id}:{variant}")
            titel, bloecke = template.build(rnd)
            specs.append(
                DocumentSpec(
                    class_key=template.class_key,
                    template_id=template.template_id,
                    variant=variant,
                    title=titel,
                    blocks=bloecke,
                    formats=template.formats,
                )
            )
    return specs
