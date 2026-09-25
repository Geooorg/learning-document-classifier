"""Die Dateiendung ist eine Behauptung. Erkannt wird am Inhalt."""

from pathlib import Path

import pytest

from doccls.detect import Format, detect_format

ECHTE_DATEIEN = {
    Format.PDF: "pdf",
    Format.DOCX: "docx",
    Format.XLSX: "xlsx",
    Format.EML: "eml",
}


@pytest.mark.parametrize("fmt,ordner", ECHTE_DATEIEN.items())
def test_erzeugte_dateien_werden_erkannt(fmt: Format, ordner: str) -> None:
    from doccls.config import RAW_DIR

    dateien = sorted((RAW_DIR / ordner).glob(f"*.{ordner}"))
    assert dateien, f"Keine Testdaten in {RAW_DIR / ordner} – erst den Generator laufen lassen"
    datei = dateien[0]
    assert detect_format(datei.read_bytes(), datei.name) is fmt


def test_falsche_endung_schlaegt_nicht_durch(tmp_path: Path) -> None:
    """Eine als .xlsx benannte PDF ist eine PDF."""
    assert detect_format(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", "quartal.xlsx") is Format.PDF


def test_ooxml_wird_am_zip_inhalt_unterschieden() -> None:
    """DOCX, XLSX und PPTX haben dieselben Magic Bytes (PK). Nur der Inhalt trennt sie."""
    import io
    import zipfile

    def zip_mit(eintrag: str) -> bytes:
        puffer = io.BytesIO()
        with zipfile.ZipFile(puffer, "w") as z:
            z.writestr("[Content_Types].xml", "<Types/>")
            z.writestr(eintrag, "x")
        return puffer.getvalue()

    assert detect_format(zip_mit("word/document.xml"), "a.bin") is Format.DOCX
    assert detect_format(zip_mit("xl/workbook.xml"), "a.bin") is Format.XLSX
    assert detect_format(zip_mit("ppt/presentation.xml"), "a.bin") is Format.PPTX


def test_eml_wird_an_den_kopfzeilen_erkannt() -> None:
    roh = b"From: a@b.example\r\nTo: c@d.example\r\nSubject: Test\r\n\r\nHallo"
    assert detect_format(roh, "nachricht.eml") is Format.EML
    assert detect_format(roh, "ohne_endung") is Format.EML


def test_unbekanntes_bleibt_unbekannt() -> None:
    assert detect_format(b"\x00\x01\x02\x03 irgendwas", "x.dat") is Format.UNKNOWN


def test_csv_wird_allein_an_der_endung_erkannt() -> None:
    """CSV hat bewusst keine Magic Bytes – Textinhalt allein ist nicht von anderem Text zu
    unterscheiden. Erkannt wird deshalb ausschließlich über die Endung."""
    assert detect_format(b"a;b;c\n1;2;3\n", "daten.csv") is Format.CSV


def test_html_wird_mit_und_ohne_doctype_erkannt() -> None:
    assert detect_format(b"<!DOCTYPE html><html><body>x</body></html>", "x.bin") is Format.HTML
    assert detect_format(b"<html><body>x</body></html>", "x.bin") is Format.HTML


def test_bildformate_werden_an_verschiedenen_magic_bytes_erkannt() -> None:
    """PNG, JPEG und beide TIFF-Byte-Reihenfolgen (Intel/Motorola) – gerade TIFF ist im
    Brief die genannte Kernmotivation: die .pdf, die eigentlich ein Scan ist."""
    assert detect_format(b"\x89PNG\r\n\x1a\n\x00\x00", "scan.bin") is Format.IMAGE
    assert detect_format(b"\xff\xd8\xff\xe0\x00\x10", "foto.bin") is Format.IMAGE
    assert detect_format(b"II*\x00\x08\x00\x00\x00", "scan.bin") is Format.IMAGE
    assert detect_format(b"MM\x00*\x00\x08\x00\x00", "scan.bin") is Format.IMAGE


def test_msg_wird_am_ole2_container_erkannt() -> None:
    assert (
        detect_format(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 8, "post.bin") is Format.MSG
    )


def test_eml_auch_ohne_erkennbare_kopfzeilen_an_der_endung() -> None:
    """Der Zweig, in dem allein die Endung entscheidet – bisher nie durchlaufen, weil die
    Testmail schon an den Kopfzeilen erkannt wurde."""
    assert detect_format(b"nur text, keine kopfzeilen", "nachricht.eml") is Format.EML


def test_textdatei_mit_einzelner_kopfzeile_ist_keine_mail() -> None:
    """Der Fehlalarm, der vorher auftrat: ein Protokoll mit einer Date:-Zeile wurde als
    E-Mail erkannt und wäre in der Extraktion still falsch verarbeitet worden."""
    protokoll = b"Aenderungsprotokoll Bauabschnitt 3\nDate: 2024-01-15\nAlle Gewerke abgenommen.\n"
    assert detect_format(protokoll, "protokoll.txt") is Format.UNKNOWN
    notiz = b"Notiz\nTo: Rechnungsabteilung\nBitte pruefen und ablegen.\n"
    assert detect_format(notiz, "notiz.txt") is Format.UNKNOWN


def test_html_mit_bom_wird_erkannt() -> None:
    """Ein UTF-8-BOM ist kein Whitespace für bytes.lstrip() und muss eigens entfernt werden."""
    bom = b"\xef\xbb\xbf"
    assert detect_format(bom + b"<html>x</html>", "ohne_endung") is Format.HTML


def test_mail_mit_bom_wird_erkannt() -> None:
    """Windows-Mailclients schreiben UTF-8 routinemäßig mit BOM. Eine BOM-präfigierte Mail
    ohne .eml-Endung darf deshalb nicht in UNKNOWN landen."""
    bom = b"\xef\xbb\xbf"
    roh = b"From: a@b.example\r\nTo: c@d.example\r\nSubject: Test\r\n\r\nHallo"
    assert detect_format(bom + roh, "ohne_endung") is Format.EML


def test_realistische_mail_varianten_werden_erkannt() -> None:
    """Echte Mails beginnen oft nicht mit einem der klassischen Felder aus einer festen
    Liste – Gmail-/Exchange-Exporte, Spamfilter-Header und mbox-Archive sehen anders aus."""
    delivered_to = b"Delivered-To: a@b.example\r\nFrom: x@y.example\r\nTo: a@b.example\r\n\r\ntext"
    assert detect_format(delivered_to, "ohne_endung") is Format.EML

    x_mailer = b"X-Mailer: Outlook\r\nFrom: x@y.example\r\nTo: a@b.example\r\n\r\ntext"
    assert detect_format(x_mailer, "ohne_endung") is Format.EML

    auth_results = (
        b"Authentication-Results: mx.example; spf=pass\r\n"
        b"From: x@y.example\r\nTo: a@b.example\r\n\r\ntext"
    )
    assert detect_format(auth_results, "ohne_endung") is Format.EML

    dkim = b"DKIM-Signature: v=1; a=rsa-sha256\r\nFrom: x@y.example\r\nTo: a@b.example\r\n\r\ntext"
    assert detect_format(dkim, "ohne_endung") is Format.EML

    mbox = (
        b"From user@example.com Mon Jan 1 00:00:00 2024\r\n"
        b"From: x@y.example\r\nTo: a@b.example\r\n\r\ntext"
    )
    assert detect_format(mbox, "ohne_endung") is Format.EML


def test_konfigdatei_im_yaml_stil_ist_keine_mail() -> None:
    """„name: …“ als erste Zeile ist kopfzeilenförmig im Sinn von FIELD_LINE – ohne die
    Prüfung auf bekannte Felder wäre jede Konfigdatei eine E-Mail."""
    yaml_artig = b"name: test\nversion: 2\nauthor: jemand\ndescription: irgendwas\n"
    assert detect_format(yaml_artig, "config.txt") is Format.UNKNOWN


def test_mehrere_gleiche_kopfzeilen_zaehlen_nur_einmal() -> None:
    """Eine Nachricht sammelt auf ihrem Weg mehrere Received:-Zeilen. Zählten die einzeln,
    wäre jedes Zustellprotokoll eine E-Mail."""
    nur_received = (
        b"Received: from mx1.example\nReceived: from mx2.example\n"
        b"Received: from mx3.example\n\ntext"
    )
    assert detect_format(nur_received, "log.txt") is Format.UNKNOWN


def test_einzelnes_kopffeld_reicht_trotz_passender_erster_zeile_nicht() -> None:
    """Die erste Zeile ist eine Kopfzeile, aber es kommt nur ein einziges Feld vor – das
    isoliert MIN_MAIL_HEADERS von der Prüfung der ersten Zeile."""
    kurznotiz = b"To: Rechnungsabteilung\nBitte pruefen und ablegen.\nDanke.\n"
    assert detect_format(kurznotiz, "kurznotiz.txt") is Format.UNKNOWN


def test_zwei_kopffelder_reichen_ohne_kopfzeile_am_anfang_nicht() -> None:
    """Zwei Kopffelder kommen vor, aber erst ab der zweiten Zeile – das isoliert die
    Prüfung der ersten Zeile von MIN_MAIL_HEADERS."""
    protokoll = b"Protokoll Nr. 42\nDate: 2024-01-15\nTo: Verteiler\nBesprechung abgeschlossen.\n"
    assert detect_format(protokoll, "protokoll.txt") is Format.UNKNOWN
