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


def block(
    *,
    ngram_min: int = 3,
    ngram_max: int = 5,
    svd_components: int = 4,
    ngram_max_features: int = 2000,
) -> NgramBlock:
    konfiguration = FeatureConfig(
        svd_components=svd_components,
        ngram_max_features=ngram_max_features,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )
    return build_ngram_block(konfiguration)


@pytest.mark.parametrize("komponenten", [3, 4])
def test_transform_liefert_die_zugesicherte_breite(komponenten: int) -> None:
    """Zwei Werte, weil alle uebrigen Tests svd_components=4 benutzen: Ein fest
    verdrahtetes ``n_components=4`` oder ``dimension = 4`` bliebe sonst unbemerkt."""
    b = block(svd_components=komponenten)
    b.fit(TRAININGSTEXTE)
    assert b.dimension == komponenten
    assert b.transform(TRAININGSTEXTE).shape == (len(TRAININGSTEXTE), komponenten)


def test_mehr_komponenten_als_texte_wirft_statt_schmaler_zu_liefern() -> None:
    """TruncatedSVD liefert hoechstens so viele Komponenten, wie es Texte gibt – ohne
    Warnung. Mit der Produktivkonfiguration (256) ergab das auf den 140 Trainingstexten
    140 Spalten bei zugesicherten 256."""
    b = block(svd_components=len(TRAININGSTEXTE) + 1)
    with pytest.raises(ValueError, match="svd_components"):
        b.fit(TRAININGSTEXTE)


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


@pytest.mark.parametrize("zugriff", ["transform", "tfidf_only", "vokabular"])
def test_zugriff_ohne_fit_wirft(zugriff: str) -> None:
    """Ein nicht angepasster Block darf keine Nullen liefern – das traegt sich lautlos
    durch bis in die Metriken. Gilt fuer alle drei Zugriffe, nicht nur fuer transform."""
    methode = getattr(block(), zugriff)
    argumente = [] if zugriff == "vokabular" else [TRAININGSTEXTE]
    with pytest.raises(RuntimeError, match="fit"):
        methode(*argumente)


def silbentexte(anzahl: int) -> list[str]:
    """Genug Texte, dass die SVD mehr Rang hat als Komponenten verlangt werden."""
    zufall = np.random.default_rng(0)
    silben = ["rech", "nung", "be", "trag", "miet", "ver", "kuen", "di", "gung", "frist"]
    silben += ["pro", "to", "koll", "sta", "tus", "richt", "gut", "schrift", "lie", "fer"]
    return [" ".join("".join(zufall.choice(silben, 3)) for _ in range(12)) for _ in range(anzahl)]


def test_fit_ist_reproduzierbar() -> None:
    """TruncatedSVD ist randomisiert – aber nur dort, wo der Seed Angriffsflaeche hat.

    Auf den sechs TRAININGSTEXTEN mit vier Komponenten ist die randomisierte SVD bereits
    exakt; ein fehlender Seed bliebe dort gruen. Gemessen: Bei 20 Texten und 3 Komponenten
    weichen zwei Seeds um rund 0,07 ab, bei 140 echten Trainingstexten und 64 Komponenten
    in 12 Komponenten um bis zu 0,01."""
    texte = silbentexte(20)
    a, b = block(svd_components=3), block(svd_components=3)
    a.fit(texte)
    b.fit(texte)
    assert np.allclose(a.transform(texte), b.transform(texte))


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
    schmal = block(ngram_min=3, ngram_max=5, svd_components=1)
    schmal.fit(LANGE_WOERTER)
    breit = block(ngram_min=8, ngram_max=10, svd_components=1)
    breit.fit(LANGE_WOERTER)
    assert {len(eintrag) for eintrag in schmal.vokabular()} == {3, 4, 5}
    assert {len(eintrag) for eintrag in breit.vokabular()} == {8, 9, 10}


def test_vokabular_wird_auf_ngram_max_features_begrenzt() -> None:
    """Alle anderen Tests benutzen 2000, und sechs Texte erreichen diese Grenze nie – der
    Parameter waere sonst ungebunden."""
    b = block(ngram_max_features=25)
    b.fit(TRAININGSTEXTE)
    assert len(b.vokabular()) == 25


def test_gross_und_kleinschreibung_ergeben_denselben_vektor() -> None:
    """Versalien in Betreffzeilen und Kopfzeilen ("RECHNUNG") duerfen kein eigenes
    Vokabular bilden."""
    b = block()
    b.fit(TRAININGSTEXTE)
    roh = b.tfidf_only(["RECHNUNG ZAHLUNGSZIEL", "Rechnung Zahlungsziel"])
    assert np.linalg.norm(roh[0]) > 0.0
    assert np.allclose(roh[0], roh[1])


def test_ngramme_ueberspannen_keine_wortgrenze() -> None:
    """``char_wb`` statt ``char``: Leerzeichen stehen nur am Rand eines n-Gramms, nie
    zwischen zwei Woertern. Ein Wechsel auf ``char`` gehoert nach features.yaml und in die
    feature_version (Plan Aufgabe 7), nicht still in den Code."""
    b = block()
    b.fit(TRAININGSTEXTE)
    assert all(" " not in eintrag.strip() for eintrag in b.vokabular())
