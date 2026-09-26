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

    # svd_components muss unter der Textzahl bleiben, sonst wirft fit() – siehe
    # test_mehr_komponenten_als_texte_wird_gemeldet. Geprueft wird hier ohnehin das
    # Vokabular, nicht die SVD.
    def schmaler_block(*, ngram_min: int, ngram_max: int) -> NgramBlock:
        return build_ngram_block(
            FeatureConfig(
                svd_components=1,
                ngram_max_features=2000,
                ngram_min=ngram_min,
                ngram_max=ngram_max,
            )
        )

    schmal = schmaler_block(ngram_min=3, ngram_max=5)
    schmal.fit(LANGE_WOERTER)
    breit = schmaler_block(ngram_min=8, ngram_max=10)
    breit.fit(LANGE_WOERTER)
    assert {len(eintrag) for eintrag in schmal.vokabular()} == {3, 4, 5}
    assert {len(eintrag) for eintrag in breit.vokabular()} == {8, 9, 10}


def test_mehr_komponenten_als_texte_wird_gemeldet() -> None:
    """``TruncatedSVD`` degradiert stillschweigend, wenn mehr Komponenten verlangt werden,
    als die Daten hergeben.

    Gemessen am echten Bestand: 140 Trainingstexte mit ``svd_components: 256`` ergaben
    eine Matrix mit 140 Spalten – ohne Fehler, ohne Warnung, waehrend ``dimension``
    weiter 256 behauptete. In der Merkmalsmatrix haette das die Bloecke falsch
    geschnitten, und keine Zahl haette verdaechtig ausgesehen. Deshalb wirft ``fit``.
    """
    b = build_ngram_block(FeatureConfig(svd_components=50, ngram_max_features=2000))
    with pytest.raises(ValueError, match="svd_components"):
        b.fit(TRAININGSTEXTE)


def test_transform_liefert_genau_so_viele_spalten_wie_dimension_behauptet() -> None:
    """Die Zusicherung hinter dem Waechter oben: ``dimension`` darf nicht luegen.

    Ohne diese Pruefung koennte ``transform`` weniger Spalten liefern als angekuendigt,
    und erst die Blockaufteilung in Aufgabe 9 wuerde daran zerbrechen – oder schlimmer,
    nicht zerbrechen und falsch schneiden.
    """
    b = block()
    b.fit(TRAININGSTEXTE)
    assert b.transform(TRAININGSTEXTE).shape[1] == b.dimension


def test_vokabulargroesse_folgt_der_konfiguration() -> None:
    """``ngram_max_features`` begrenzt das Vokabular. Ohne diesen Test ist der Parameter
    ungebunden: Eine Umsetzung, die ihn ignoriert, ueberlebt alle anderen Pruefungen, weil
    der Beispielkorpus mit 937 n-Grammen ohnehin unter der Grenze von 2000 bleibt.

    Dann stuende in ``features.yaml`` ein Wert ohne Wirkung, waehrend die
    ``feature_version`` sich bei seiner Aenderung sehr wohl aendert.
    """
    eng = build_ngram_block(FeatureConfig(svd_components=4, ngram_max_features=20))
    eng.fit(TRAININGSTEXTE)
    weit = block()
    weit.fit(TRAININGSTEXTE)
    assert len(eng.vokabular()) <= 20
    assert len(weit.vokabular()) > 20, (
        f"Der Beispielkorpus liefert nur {len(weit.vokabular())} n-Gramme – zu wenig, "
        "um die Begrenzung ueberhaupt zu pruefen"
    )
