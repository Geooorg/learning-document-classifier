"""Eine Mail ist ein Baum: Körper und jeder Anhang sind eigene Dokumente (Konzept § 5)."""

from datetime import UTC, datetime
from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import SMTP
from email.policy import default as default_policy
from pathlib import Path

from doccls.config import RAW_DIR
from doccls.extraction import extract
from doccls.extraction.mail import extract_eml
from doccls.models import Document, SegmentKind, normalize_text

MEDIA_TYPE_EML = "message/rfc822"


def mail_bytes(*, mit_anhang: bool) -> bytes:
    nachricht = EmailMessage(policy=SMTP)
    nachricht["Subject"] = "Ihre Rechnung RE-2026-4711"
    nachricht["From"] = "buchhaltung@lieferant.example"
    nachricht["To"] = "eingang@kunde.example"
    nachricht["Date"] = "Sun, 01 Mar 2026 12:00:00 +0000"
    nachricht.set_content("Guten Tag,\n\nanbei die Rechnung.\n\nMit freundlichen Grüßen")
    if mit_anhang:
        nachricht.add_attachment(
            b"%PDF-1.7\nRechnungsinhalt",
            maintype="application",
            subtype="pdf",
            filename="RE-2026-4711.pdf",
        )
    return nachricht.as_bytes()


def dokument(daten: bytes, dateiname: str = "post/a.eml") -> Document:
    return Document.create(
        source_path=dateiname,
        media_type=MEDIA_TYPE_EML,
        content=daten,
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def alle_eml_pfade() -> list[Path]:
    dateien = sorted(RAW_DIR.glob("eml/*.eml"))
    assert dateien, f"Keine Testdaten in {RAW_DIR / 'eml'}"
    return dateien


# --- Aus dem Aufgabenbrief wörtlich übernommen -----------------------------------------


def test_kopf_und_koerper_werden_getrennte_segmente() -> None:
    daten = mail_bytes(mit_anhang=False)
    segmente, anhaenge = extract_eml(dokument(daten), daten)
    arten = [s.kind for s in segmente]
    assert SegmentKind.MAIL_KOPF in arten
    assert SegmentKind.MAIL_KOERPER in arten
    assert anhaenge == []


def test_kopf_enthaelt_betreff_und_absender() -> None:
    """Beide werden später Strukturmerkmale: Mails von rechnung@… tragen oft Rechnungen."""
    daten = mail_bytes(mit_anhang=False)
    segmente, _ = extract_eml(dokument(daten), daten)
    kopf = next(s for s in segmente if s.kind is SegmentKind.MAIL_KOPF)
    assert "RE-2026-4711" in kopf.text
    assert "lieferant.example" in kopf.text


def test_anhang_wird_herausgereicht_und_nicht_in_den_mailtext_gemischt() -> None:
    daten = mail_bytes(mit_anhang=True)
    segmente, anhaenge = extract_eml(dokument(daten), daten)
    assert [a.file_name for a in anhaenge] == ["RE-2026-4711.pdf"]
    assert anhaenge[0].content.startswith(b"%PDF")
    assert "Rechnungsinhalt" not in " ".join(s.text for s in segmente)


def test_mail_ohne_körper_liefert_wenigstens_den_kopf() -> None:
    nachricht = EmailMessage(policy=SMTP)
    nachricht["Subject"] = "Ohne Text"
    nachricht["From"] = "a@b.example"
    nachricht["To"] = "c@d.example"
    daten = nachricht.as_bytes()
    segmente, _ = extract_eml(dokument(daten), daten)
    assert any(s.kind is SegmentKind.MAIL_KOPF for s in segmente)


# --- Ergänzende Tests: Inhaltsmenge statt bloßer Segmentzahl ---------------------------


def test_kopf_erfasst_alle_fuenf_kopffelder_nicht_nur_die_ersten() -> None:
    """``test_kopf_enthaelt_betreff_und_absender`` prüft nur Betreff und Absender – ein
    Abbruch der Kopfschleife nach dem zweiten Feld bliebe dort unbemerkt. Hier bekommt
    jedes der fünf übernommenen Felder (``HEADER_FIELDS``) einen eigenen, unverwechselbaren
    Marker, und alle fünf müssen im Kopftext auftauchen.
    """
    nachricht = EmailMessage(policy=SMTP)
    nachricht["Subject"] = "MARKERSUBJEKT"
    nachricht["From"] = "markerabsender@quelle.example"
    nachricht["To"] = "markerempfaenger@ziel.example"
    nachricht["Cc"] = "markerkopie@ziel.example"
    nachricht["Date"] = "Sun, 01 Mar 2026 12:00:00 +0000"
    nachricht.set_content("Rumpf")
    daten = nachricht.as_bytes()
    segmente, _ = extract_eml(dokument(daten), daten)
    kopf = next(s for s in segmente if s.kind is SegmentKind.MAIL_KOPF)
    for marker in (
        "MARKERSUBJEKT",
        "markerabsender@quelle.example",
        "markerempfaenger@ziel.example",
        "markerkopie@ziel.example",
        "01 Mar 2026",
    ):
        assert marker in kopf.text, f"{marker!r} fehlt im Kopfsegment: {kopf.text!r}"


def test_koerper_segmenttext_deckt_gesamten_koerperinhalt_ab() -> None:
    """Eine Kürzung wie ``get_content()[:20]`` änderte weder Segmentzahl noch -art. Deshalb
    wird hier die Zeichenzahl des Rumpfs unabhängig gemessen (gleiche Normalisierung wie
    die Extraktion) und mit der tatsächlichen Segmentlänge verglichen – mit einem Rumpf,
    der deutlich länger ist als jeder plausible Kürzungs-Schwellwert.
    """
    langer_rumpf = "\n\n".join(f"Absatz {i}: {'Text ' * 10}" for i in range(20))
    nachricht = EmailMessage(policy=SMTP)
    nachricht["Subject"] = "Langer Rumpf"
    nachricht["From"] = "a@b.example"
    nachricht["To"] = "c@d.example"
    nachricht.set_content(langer_rumpf)
    daten = nachricht.as_bytes()
    segmente, _ = extract_eml(dokument(daten), daten)
    koerper = next(s for s in segmente if s.kind is SegmentKind.MAIL_KOERPER)

    geparst = BytesParser(policy=default_policy).parsebytes(daten)
    referenz = geparst.get_body(preferencelist=("plain", "html"))
    assert referenz is not None
    erwartete_laenge = len(normalize_text(referenz.get_content()))
    assert erwartete_laenge > 200, "Testrumpf zu kurz – Prüfung liefe leer"
    assert len(koerper.text) == erwartete_laenge


def test_indexfolge_ist_lueckenlos_und_aufsteigend() -> None:
    daten = mail_bytes(mit_anhang=True)
    segmente, _ = extract_eml(dokument(daten), daten)
    assert [s.index for s in segmente] == list(range(len(segmente)))
    assert len(segmente) == 2, "Kopf und Körper – der Anhang erzeugt kein drittes Segment"


def test_multipart_alternative_liefert_reinen_text_ohne_html_auszeichnung() -> None:
    """Bei Text+HTML-Variante muss der Text gewinnen: HTML-Markup wäre reines Rauschen für
    die spätere Merkmalsbildung, und ein Vertauschen der ``preferencelist``-Reihenfolge
    änderte weder Segmentzahl noch -art – nur den Inhalt.
    """
    nachricht = EmailMessage(policy=SMTP)
    nachricht["Subject"] = "Alternative"
    nachricht["From"] = "a@b.example"
    nachricht["To"] = "c@d.example"
    nachricht.set_content("NURTEXTMARKER ohne Auszeichnung")
    nachricht.add_alternative("<p>NURTEXTMARKER <b>mit</b> Auszeichnung</p>", subtype="html")
    daten = nachricht.as_bytes()
    segmente, _ = extract_eml(dokument(daten), daten)
    koerper = next(s for s in segmente if s.kind is SegmentKind.MAIL_KOERPER)
    assert "NURTEXTMARKER ohne Auszeichnung" in koerper.text
    assert "<" not in koerper.text


def test_anhaenge_ohne_dateinamen_bekommen_eindeutige_ersatznamen() -> None:
    """Ein fester Ersatzname für jeden namenlosen Anhang machte zwei Anhänge derselben
    Mail ununterscheidbar (Moduldoc). Zwei Anhänge ohne Dateiname müssen deshalb
    verschiedene Ersatznamen bekommen, in ihrer Reihenfolge in der Mail.
    """
    nachricht = EmailMessage(policy=SMTP)
    nachricht["Subject"] = "Zwei namenlose Anhänge"
    nachricht["From"] = "a@b.example"
    nachricht["To"] = "c@d.example"
    nachricht.set_content("Rumpf")
    nachricht.add_attachment(b"ERSTERANHANG", maintype="application", subtype="octet-stream")
    nachricht.add_attachment(b"ZWEITERANHANG", maintype="application", subtype="octet-stream")
    daten = nachricht.as_bytes()
    _, anhaenge = extract_eml(dokument(daten), daten)
    assert len(anhaenge) == 2
    assert anhaenge[0].file_name != anhaenge[1].file_name
    assert anhaenge[0].content == b"ERSTERANHANG"
    assert anhaenge[1].content == b"ZWEITERANHANG"


def test_anhanginhalt_bleibt_bytegenau_auch_am_ende() -> None:
    """``content[:n]`` oder ein zu kleiner Lesepuffer fiele bei einem kurzen Testinhalt
    nicht auf. Deshalb hier ein Anhang mit eindeutigem Anfang **und** Ende über mehrere
    Base64-Zeilen hinweg (die Kodierung, mit der echte Mailanhänge ankommen).
    """
    inhalt = b"KOPFMARKER" + b"F" * 500 + b"ENDMARKER"
    nachricht = EmailMessage(policy=SMTP)
    nachricht["Subject"] = "Großer Anhang"
    nachricht["From"] = "a@b.example"
    nachricht["To"] = "c@d.example"
    nachricht.set_content("Rumpf")
    nachricht.add_attachment(
        inhalt, maintype="application", subtype="octet-stream", filename="datei.bin"
    )
    daten = nachricht.as_bytes()
    _, anhaenge = extract_eml(dokument(daten), daten)
    assert len(anhaenge) == 1
    assert anhaenge[0].content == inhalt


def test_alle_drei_anhaenge_werden_erfasst_nicht_nur_der_erste() -> None:
    """Ein Abbruch der Schleife nach dem ersten Treffer (oder ein ``break``/eine
    Beschränkung auf ``iter_attachments()[0]``) ließe Segmentzahl und Kopf/Körper-Prüfungen
    unberührt – nur die Anhangzahl verriete den Fehler. Drei Anhänge mit unterscheidbarem
    Inhalt müssen deshalb alle drei ankommen, in ihrer Reihenfolge in der Mail.
    """
    nachricht = EmailMessage(policy=SMTP)
    nachricht["Subject"] = "Drei Anhänge"
    nachricht["From"] = "a@b.example"
    nachricht["To"] = "c@d.example"
    nachricht.set_content("Rumpf")
    for name, inhalt in (
        ("eins.bin", b"INHALT-EINS"),
        ("zwei.bin", b"INHALT-ZWEI"),
        ("drei.bin", b"INHALT-DREI"),
    ):
        nachricht.add_attachment(
            inhalt, maintype="application", subtype="octet-stream", filename=name
        )
    daten = nachricht.as_bytes()
    _, anhaenge = extract_eml(dokument(daten), daten)
    assert [a.file_name for a in anhaenge] == ["eins.bin", "zwei.bin", "drei.bin"]
    assert [a.content for a in anhaenge] == [b"INHALT-EINS", b"INHALT-ZWEI", b"INHALT-DREI"]


def test_extract_dispatcher_liefert_fuer_eml_nur_segmente_ohne_anhaenge() -> None:
    """``extract()`` (der Verteiler in ``__init__.py``) darf für EML nichts anderes liefern
    als ``extract_eml`` an Segmenten – Anhänge holt sich ausschließlich ``pipeline.py``.
    """
    daten = mail_bytes(mit_anhang=True)
    doc = dokument(daten, dateiname="post/rechnung.eml")
    segmente_direkt, _ = extract_eml(doc, daten)
    segmente_ueber_dispatcher = extract(doc, daten)
    assert segmente_ueber_dispatcher == segmente_direkt


# --- Korpus-Tests: echte Beispielmails, nicht nur synthetische -------------------------


def test_alle_beispielmails_liefern_mindestens_das_kopfsegment() -> None:
    for pfad in alle_eml_pfade():
        daten = pfad.read_bytes()
        segmente, _ = extract_eml(dokument(daten, str(pfad)), daten)
        arten = [s.kind for s in segmente]
        assert SegmentKind.MAIL_KOPF in arten, f"{pfad}: kein Kopfsegment"


def test_mails_mit_anhang_liefern_genau_einen_pdf_anhang() -> None:
    pfade = [p for p in alle_eml_pfade() if "mail-anhang" in p.name]
    assert pfade, "Keine Testdaten mit Anhang im Korpus"
    for pfad in pfade:
        daten = pfad.read_bytes()
        _, anhaenge = extract_eml(dokument(daten, str(pfad)), daten)
        assert len(anhaenge) == 1, f"{pfad}: erwarte genau einen Anhang, habe {len(anhaenge)}"
        assert anhaenge[0].file_name.endswith(".pdf")
        assert anhaenge[0].content.startswith(b"%PDF")


def test_mimekodierter_betreff_wird_lesbar_ins_kopfsegment_uebernommen() -> None:
    """``AGB-mail-00.eml`` trägt den Betreff RFC-2047-kodiert
    (``=?utf-8?q?Gesch=C3=A4ftsbedingungen?=``). Landete die Rohkodierung unverändert im
    Kopfsegment, wäre der Umlaut für jede spätere Textauswertung unbrauchbar.
    """
    pfad = RAW_DIR / "eml" / "AGB-mail-00.eml"
    daten = pfad.read_bytes()
    segmente, _ = extract_eml(dokument(daten, str(pfad)), daten)
    kopf = next(s for s in segmente if s.kind is SegmentKind.MAIL_KOPF)
    assert "Geschäftsbedingungen" in kopf.text
    assert "=?utf-8" not in kopf.text.lower()


def test_extraktion_ist_deterministisch_fuer_echte_mail() -> None:
    pfad = RAW_DIR / "eml" / "RECHNUNG-mail-anhang-00.eml"
    daten = pfad.read_bytes()
    doc = dokument(daten, str(pfad))
    erster_lauf, erste_anhaenge = extract_eml(doc, daten)
    zweiter_lauf, zweite_anhaenge = extract_eml(doc, daten)
    assert erster_lauf, "Ohne Segmente prüft der Vergleich nichts"
    assert erster_lauf == zweiter_lauf
    assert erste_anhaenge == zweite_anhaenge


def test_betreff_steht_als_heading_am_kopfsegment() -> None:
    """``heading`` ist ein eigenes Feld, nicht nur ein Nebenprodukt von ``text``.

    Ohne diese Prüfung überlebt eine Umsetzung, die durchgängig ``heading=None`` setzt:
    der Betreff steht dann zwar weiter im Kopftext, aber das Feld, das ihn später als
    Fundstelle und als Merkmal tragen soll, bleibt leer – und kein anderer Test merkt es.
    """
    daten = mail_bytes(mit_anhang=False)
    segmente, _ = extract_eml(dokument(daten), daten)
    kopf = next(s for s in segmente if s.kind is SegmentKind.MAIL_KOPF)
    assert kopf.heading == "Ihre Rechnung RE-2026-4711"


def test_mail_ohne_betreff_hat_kein_heading() -> None:
    """Gegenstück: ``heading`` darf nicht mit einer leeren Zeichenkette belegt werden,
    sonst wäre „kein Betreff" später nicht von „leerer Betreff" zu unterscheiden."""
    nachricht = EmailMessage(policy=SMTP)
    nachricht["From"] = "a@b.example"
    nachricht["To"] = "c@d.example"
    nachricht.set_content("Ohne Betreff.")
    daten = nachricht.as_bytes()
    segmente, _ = extract_eml(dokument(daten), daten)
    kopf = next(s for s in segmente if s.kind is SegmentKind.MAIL_KOPF)
    assert kopf.heading is None
