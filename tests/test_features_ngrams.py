"""Zeichen-n-Gramme loesen Komposita. Und: fit sieht nur die Trainingstexte."""

import numpy as np
import pytest

from doccls.features import FeatureConfig
from doccls.features.ngrams import NgramBlock, build_ngram_block

TRAININGSTEXTE = [
    "Rechnung ueber Wartungsarbeiten, Rechnungsbetrag 1.190,00 EUR, Zahlungsziel 14 Tage",
    "Gutschrift zur Rechnung RE-2026-0815, Erstattungsbetrag 240,00 EUR wegen Maengelruege",
    "Mietvertrag ueber Gewerberaeume, Kuendigungsfrist drei Monate zum Quartalsende",
    "Allgemeine Geschaeftsbedingungen, Paragraph 1 Geltungsbereich, Paragraph 2 Lieferung",
    "Protokoll der Sitzung, TOP 1 Begruessung, Anwesend waren die Mitglieder des Beirats",
    "Statusbericht Quartal zwei, Ampel gruen, Fortschritt planmaessig, keine Eskalation",
]


def block(*, ngram_min: int = 3, ngram_max: int = 5) -> NgramBlock:
    konfiguration = FeatureConfig(
        svd_components=4,
        ngram_max_features=2000,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )
    return build_ngram_block(konfiguration)


def test_transform_liefert_die_zugesicherte_breite() -> None:
    b = block()
    b.fit(TRAININGSTEXTE)
    assert b.transform(TRAININGSTEXTE).shape == (len(TRAININGSTEXTE), b.dimension)
    assert b.dimension == 4


def test_kompositum_liegt_naeher_am_grundwort_als_an_fremdem_wort() -> None:
    """Der Kern von Konzept Anhang A: 'Rechnungsbetrag' teilt Zeichenfolgen mit
    'Rechnung', aber kein Wort. Ein Wort-n-Gramm-Modell saehe hier gar nichts.

    Geprueft wird auf der TF-IDF-Ebene vor der SVD, weil die SVD bei sechs Texten zu
    wenig zu tun hat, um das Verhaeltnis stabil zu erhalten.
    """
    b = block()
    b.fit(TRAININGSTEXTE)
    roh = b.tfidf_only(["Rechnung", "Rechnungsbetrag", "Kuendigungsfrist"])
    normiert = roh / np.linalg.norm(roh, axis=1, keepdims=True)
    verwandt = float(normiert[0] @ normiert[1])
    fremd = float(normiert[0] @ normiert[2])
    assert verwandt > fremd, (
        f"Rechnung/Rechnungsbetrag {verwandt:.3f} liegt nicht ueber "
        f"Rechnung/Kuendigungsfrist {fremd:.3f} – die Zeichen-n-Gramme wirken nicht"
    )


def test_tippfehler_bleibt_nah_am_original() -> None:
    """Die zweite Begruendung aus Konzept § 6.2: OCR-Robustheit. 'Rechnunq' ist fuer ein
    Wortmodell ein unbekanntes Wort, fuer ein Zeichenmodell fast dasselbe."""
    b = block()
    b.fit(TRAININGSTEXTE)
    roh = b.tfidf_only(["Rechnung", "Rechnunq", "Mietvertrag"])
    normiert = roh / np.linalg.norm(roh, axis=1, keepdims=True)
    assert float(normiert[0] @ normiert[1]) > float(normiert[0] @ normiert[2])


def test_transform_ohne_fit_wirft() -> None:
    """Ein nicht angepasster Block darf keine Nullen liefern – das traegt sich lautlos
    durch bis in die Metriken."""
    with pytest.raises(RuntimeError, match="fit"):
        block().transform(TRAININGSTEXTE)


def test_fit_ist_reproduzierbar() -> None:
    """TruncatedSVD ist randomisiert. Ohne festen Seed waeren zwei Laeufe unvergleichbar."""
    a, b = block(), block()
    a.fit(TRAININGSTEXTE)
    b.fit(TRAININGSTEXTE)
    assert np.allclose(a.transform(TRAININGSTEXTE), b.transform(TRAININGSTEXTE))


def test_unbekannter_text_ergibt_keinen_nullvektor() -> None:
    """Ein Text aus lauter ungesehenen Zeichenfolgen wuerde zu Recht nahe null liegen –
    ein deutscher Satz mit bekannten Silben aber nicht. Faellt er trotzdem auf null,
    stimmt die Anwendung des angepassten Vokabulars nicht."""
    b = block()
    b.fit(TRAININGSTEXTE)
    vektor = b.transform(["Schlussrechnung fuer Wartungsleistungen im Quartal"])[0]
    assert np.linalg.norm(vektor) > 0.0


LANGE_WOERTER = [
    "Rechnungsbestaetigungsschreiben Verwaltungsangestelltenausbildung",
    "Kuendigungsfristverlaengerungsantrag Geschaeftsbedingungsaenderung",
]
"""Ausschliesslich lange Woerter: kein kurzes Token, das char_wb am Wortrand kappen
wuerde. Nur so entspricht die Menge der Vokabularlaengen exakt dem konfigurierten
n-Gramm-Bereich, ohne durch Randeffekte kurzer Woerter verwischt zu werden."""


def test_ngram_bereich_kommt_aus_der_konfiguration_und_ist_nicht_fest_verdrahtet() -> None:
    """(3, 5) ist zufaellig die Voreinstellung von ``FeatureConfig`` – kein anderer Test
    dieser Datei weicht davon ab. Ein fest verdrahtetes ``ngram_range=(3, 5)`` im
    Vectorizer wuerde deshalb von keinem der Tests oben bemerkt. Hier wird der Bereich
    explizit auf einen abweichenden Wert gesetzt und ueber die tatsaechlichen
    Vokabularlaengen geprueft."""
    schmal = block(ngram_min=3, ngram_max=5)
    schmal.fit(LANGE_WOERTER)
    breit = block(ngram_min=8, ngram_max=10)
    breit.fit(LANGE_WOERTER)
    assert {len(eintrag) for eintrag in schmal.vokabular()} == {3, 4, 5}
    assert {len(eintrag) for eintrag in breit.vokabular()} == {8, 9, 10}
