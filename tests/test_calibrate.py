"""Tests für Aufgabe 11: Kalibrierung per Temperature Scaling (Konzept § 7.3, § 9.3).

Die Tests arbeiten auf **synthetischen** Logits, nicht auf dem echten Bestand. Gemessen:
Auf diesem Korpus erreichen alle drei Modellarten Macro-F1 1,000 auf dem Gold-Set, ihre
Vorhersagen sind praktisch entartet, und die daraus bestimmte Temperatur ist ohne Aussage.
Eine Zusicherung gegen eine solche Zahl prüfte den Korpus, nicht die Kalibrierung. Hier
wird stattdessen eine Überheblichkeit bekannter Stärke erzeugt – der Sollwert von ``T``
steht dadurch vorab fest und ist nicht aus dem Ergebnis abgelesen.

``scipy.special.softmax`` dient als unabhängige Vergleichsgröße: Die Tests prüfen
``apply_temperature`` gegen eine fremde Umsetzung, nicht gegen sich selbst.
"""

import numpy as np
import numpy.typing as npt
import pytest
from scipy.special import softmax as _scipy_softmax

from doccls.calibrate import (
    T_OBERGRENZE,
    T_UNTERGRENZE,
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)
from doccls.splits import calibration_documents, training_documents

KLASSEN = ["a", "b", "c", "d"]


def _softmax(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Zeilenweiser Softmax über eine fremde Umsetzung – bewusst nicht die eigene."""
    ergebnis: npt.NDArray[np.float64] = _scipy_softmax(logits, axis=1)
    return ergebnis


def _vorhersagen(
    seed: int, schaerfe: float, n: int = 400
) -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
    """Logits mit bekannter Fehleichung.

    Erzeugt latente, *ehrliche* Logits, zieht die wahren Klassen aus deren Softmax – die
    latenten Logits sind damit per Konstruktion perfekt geeicht – und gibt sie mit
    ``schaerfe`` multipliziert zurück. Das ist exakt die Umkehrung von Temperature
    Scaling: Die gesuchte Temperatur ist ``schaerfe``, unabhängig davon, was die
    Umsetzung liefert. ``schaerfe > 1`` heißt überheblich, ``< 1`` unterheblich.
    """
    rng = np.random.default_rng(seed)
    latente_logits = rng.normal(loc=0.0, scale=1.2, size=(n, len(KLASSEN)))
    wahrscheinlichkeiten = _softmax(latente_logits)
    y = [
        KLASSEN[int(rng.choice(len(KLASSEN), p=wahrscheinlichkeiten[zeile]))] for zeile in range(n)
    ]
    return latente_logits * schaerfe, y, KLASSEN


def _ueberhebliche_vorhersagen(
    seed: int,
) -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
    """Der Normalfall aus Konzept § 7.3: Das Modell ist um den Faktor 3 zu sicher."""
    return _vorhersagen(seed, schaerfe=3.0)


def _gut_geeichte_vorhersagen(
    seed: int,
) -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
    """Gegenprobe: bereits ehrliche Logits, die gesuchte Temperatur ist 1,0."""
    return _vorhersagen(seed, schaerfe=1.0)


def _unterhebliche_vorhersagen(
    seed: int,
) -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
    """Der seltene Gegenfall: ein zu zaghaftes Modell, gesuchte Temperatur 0,35."""
    return _vorhersagen(seed, schaerfe=0.35)


def _perfekt_trennbare_vorhersagen() -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
    """Der entartete Fall, in dem die Kalibriermenge kein ``T`` hergibt.

    Jede Zeile trifft ihre wahre Klasse mit einem festen, maessigen Abstand von 0,5 –
    perfekt trennbar. Die NLL faellt dann monoton, je kleiner ``T`` wird (gemessen:
    ``T=0,5 → 0,743668``, ``0,1 → 0,020012``, ``0,05 → 0,000136``, ``0,005 → -0,000000``),
    das Minimum liegt also ausserhalb des Suchbereichs. Das ist derselbe Verlauf, den der
    Pruefer auf den echten centroid-Scores gemessen hat (``T = 0,050007``).
    """
    abstand = 0.5
    block = np.eye(len(KLASSEN), dtype=np.float64) * abstand
    return np.tile(block, (5, 1)), KLASSEN * 5, KLASSEN


def _masslos_ueberhebliche_vorhersagen(
    seed: int,
) -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
    """Gegenstueck am oberen Rand: die gesuchte Temperatur 50 liegt jenseits von 20."""
    return _vorhersagen(seed, schaerfe=50.0)


def test_temperatur_aendert_keine_einzige_entscheidung() -> None:
    """Konzept § 7.3: Die Rangfolge bleibt gleich. Kalibrierung repariert die Zahl, nicht
    die Entscheidung.

    Der urspruengliche Plan behauptete an dieser Stelle, ein *Verschieben* statt
    Skalierens (``logits - T``) breche diese Zusicherung. Das ist falsch und wurde
    nachgemessen: ``softmax(x - c) == softmax(x)`` fuer skalares ``c``, Abweichung 1e-17.
    Ein solcher Fehler faellt nicht hier auf, sondern bei
    ``test_kalibrierung_senkt_den_eichfehler_deutlich`` – die Wahrscheinlichkeiten
    blieben schlicht unveraendert ueberheblich.

    Zaehne hat dieser Test gegen den Vorzeichenfehler (``-werte / T``): Dann wird die
    sicherste Klasse zur unsichersten, ohne dass irgendeine Zahl unplausibel aussaehe.
    """
    logits, y, klassen = _ueberhebliche_vorhersagen(seed=3)
    for T in (0.5, 1.0, 2.0, 5.0):
        vorher = logits.argmax(axis=1)
        nachher = apply_temperature(logits, T).argmax(axis=1)
        assert np.array_equal(vorher, nachher), f"T={T} hat Entscheidungen veraendert"


def test_temperatur_eins_ergibt_den_reinen_softmax() -> None:
    """Bindet, dass ``apply_temperature`` Wahrscheinlichkeiten liefert und nicht bloss
    skalierte Logits weiterreicht: Bei T=1 muss exakt der Softmax herauskommen. Ohne
    diese Zusicherung bestuende eine Umsetzung, die ``logits / T`` unnormiert
    zurueckgibt, den Rangfolgetest darueber muehelos."""
    logits, y, klassen = _ueberhebliche_vorhersagen(seed=3)
    assert np.allclose(apply_temperature(logits, 1.0), _softmax(logits), atol=1e-6)


def test_ueberhebliches_modell_bekommt_temperatur_ueber_eins() -> None:
    """Der Normalfall aus Konzept § 7.3. Ein T <= 1 hiesse, das Modell sei unterheblich –
    dann stimmt etwas mit der Anpassung nicht."""
    logits, y, klassen = _ueberhebliche_vorhersagen(seed=3)
    assert fit_temperature(logits, y, klassen) > 1.0


def test_kalibrierung_senkt_den_eichfehler_deutlich() -> None:
    """Die eigentliche Wirkung, gemessen statt behauptet. Ohne diese Pruefung bestuende
    eine Umsetzung, die immer T=1 liefert, alle anderen Tests."""
    logits, y, klassen = _ueberhebliche_vorhersagen(seed=3)
    vorher = expected_calibration_error(_softmax(logits), y, klassen)
    T = fit_temperature(logits, y, klassen)
    nachher = expected_calibration_error(apply_temperature(logits, T), y, klassen)
    assert nachher < vorher * 0.6, (
        f"ECE nur von {vorher:.4f} auf {nachher:.4f} – die Kalibrierung wirkt kaum"
    )


def test_bereits_geeichtes_modell_bleibt_nahe_bei_eins() -> None:
    """Gegenprobe: Auf gut geeichten Wahrscheinlichkeiten darf die Kalibrierung nichts
    kaputtmachen. Ein Verfahren, das immer nach oben skaliert, faellt hier auf."""
    logits, y, klassen = _gut_geeichte_vorhersagen(seed=3)
    assert 0.7 < fit_temperature(logits, y, klassen) < 1.4


def test_unterhebliches_modell_bekommt_temperatur_unter_eins() -> None:
    """Bindet die *untere* Suchgrenze an eine echte Verletzung.

    Der Test darueber kann das nicht: Verengt man die Grenzen auf ``(1.0, 20.0)``,
    liefert die Suche fuer ein leicht unterhebliches Modell knapp 1,0 – und 1,0 liegt
    mitten in seinem erlaubten Fenster ``0,7 < T < 1,4``. Die Schranke waere also
    ungeprueft. Hier ist das wahre T = 0,35: eine zu hoch gesetzte Untergrenze faellt auf.
    """
    logits, y, klassen = _unterhebliche_vorhersagen(seed=3)
    assert fit_temperature(logits, y, klassen) < 0.9


def test_ece_ist_null_bei_perfekter_eichung() -> None:
    """Bindet die Formel: Sagt das Modell durchgehend 1,0 und trifft immer, ist der
    Eichfehler null.

    Die urspruengliche Planvorgabe behauptete hier, eine falsch normierte Gewichtung
    falle auf. Das stimmt nicht und wurde nachgemessen: Alle Luecken sind 0, und jede
    Gewichtung eines Nullvektors ergibt wieder 0 – keine Gewichtungsmutation faellt hier
    auf. Gebunden ist die Gewichtung allein in
    ``test_ece_gewichtet_nach_korbbesetzung_und_zaehlt_leere_koerbe_nicht``. Was dieser
    Test bindet, ist der eine Pol: Eine Umsetzung mit konstanter Rueckgabe ungleich 0
    faellt durch (das Gegenstueck dazu ist
    ``test_ece_ist_gross_bei_voller_ueberheblichkeit``).
    """
    proba = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(proba, ["a", "a", "b"], ["a", "b"]) == 0.0


def test_ece_ist_gross_bei_voller_ueberheblichkeit() -> None:
    """Gegenstueck: Volle Sicherheit, durchgehend falsch – der Fehler muss nahe 1 liegen.
    Ohne beide Pole koennte eine Umsetzung, die immer 0 liefert, den Test oben bestehen."""
    proba = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert expected_calibration_error(proba, ["b", "b"], ["a", "b"]) > 0.9


def test_ece_gewichtet_nach_korbbesetzung_und_zaehlt_leere_koerbe_nicht() -> None:
    """Der Fall mit **ungleich** besetzten Koerben – ohne ihn bliebe offen, ob ueberhaupt
    gewichtet wird.

    Acht Vorhersagen mit Konfidenz 0,95, alle richtig (Korb 9, Luecke 0,05); zwei mit
    Konfidenz 0,55, beide falsch (Korb 5, Luecke 0,55). Die acht uebrigen Koerbe sind leer.
    Richtig gewichtet: 0,8 * 0,05 + 0,2 * 0,55 = 0,15. Ein ungewichteter Mittelwert ueber
    die besetzten Koerbe ergaebe 0,30, einer ueber alle zehn Koerbe 0,06 – beide
    Mutationen faellt dieser Test.
    """
    proba = np.array([[0.95, 0.05]] * 8 + [[0.55, 0.45]] * 2)
    y = ["a"] * 8 + ["b"] * 2
    assert expected_calibration_error(proba, y, ["a", "b"]) == pytest.approx(0.15)


def test_ece_haengt_an_der_uebergebenen_korbzahl() -> None:
    """Bindet ``bins``: Ohne diesen Test ist der Parameter ungeprueft, und eine Umsetzung
    mit fest verdrahteten 10 Koerben besteht die ganze Suite.

    Dieselben zehn Vorhersagen, zwei Korbzahlen, zwei nachgerechnete Werte. Bei
    ``bins=1`` liegt alles in einem Korb: mittlere Konfidenz
    ``(8*0,95 + 2*0,55)/10 = 0,87``, Trefferquote ``0,8``, Luecke ``0,07``. Bei
    ``bins=10`` trennen sich die beiden Gruppen und die Luecken heben sich nicht mehr
    gegenseitig auf: ``0,8*0,05 + 0,2*0,55 = 0,15``. Die Korbzahl ist damit nicht
    kosmetisch – sie entscheidet, ob eine zu hohe und eine zu niedrige Konfidenz
    gegeneinander aufgerechnet werden.
    """
    proba = np.array([[0.95, 0.05]] * 8 + [[0.55, 0.45]] * 2)
    y = ["a"] * 8 + ["b"] * 2
    assert expected_calibration_error(proba, y, ["a", "b"], bins=1) == pytest.approx(0.07)
    assert expected_calibration_error(proba, y, ["a", "b"], bins=10) == pytest.approx(0.15)


def test_temperatur_null_oder_negativ_wird_abgelehnt() -> None:
    """Waechter mit eigenem Test: Eine negative Temperatur dreht die Rangfolge um und
    verwandelt die sicherste Klasse in die unsicherste – der einzige Weg, auf dem
    ``apply_temperature`` eine Entscheidung veraendern kann. Eine Null teilt durch Null.
    Beides muss auffliegen, statt still ein Ergebnis zu liefern."""
    logits, y, klassen = _ueberhebliche_vorhersagen(seed=3)
    for T in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="Temperatur"):
            apply_temperature(logits, T)


def test_fit_temperature_lehnt_randtreffer_an_der_untergrenze_ab() -> None:
    """Waechter mit eigenem Test: Ein ``T`` vom Rand des Suchbereichs ist keine Eichung.

    Auf perfekt trennbaren Scores faellt die NLL monoton bis zur Untergrenze; die Suche
    gibt dann brav ``0,0500...`` zurueck, eine Zahl, die aussieht wie ein Ergebnis.
    Genau das ist auf dem echten Bestand passiert (centroid: ``T = 0,050007``). Ohne
    diesen Waechter koennte Aufgabe 16 einen Randtreffer nicht von einer echten
    Anpassung unterscheiden.
    """
    logits, y, klassen = _perfekt_trennbare_vorhersagen()
    with pytest.raises(ValueError, match="Untergrenze"):
        fit_temperature(logits, y, klassen)


def test_fit_temperature_lehnt_randtreffer_an_der_obergrenze_ab() -> None:
    """Dieselbe Zusicherung am anderen Ende – sonst bliebe die halbe Bedingung ungeprueft.

    Ein Modell mit der wahren Temperatur 50 liegt jenseits von ``T_OBERGRENZE = 20``.
    Die Suche laeuft in die obere Schranke (gemessen: ``19,999994``), und auch dieser
    Wert ist keine Eichung, sondern die Aussage, dass der Suchbereich nicht passt.
    """
    logits, y, klassen = _masslos_ueberhebliche_vorhersagen(seed=3)
    with pytest.raises(ValueError, match="Obergrenze"):
        fit_temperature(logits, y, klassen)


def test_fit_temperature_laesst_gewoehnliche_anpassungen_durch() -> None:
    """Gegenprobe zum Waechter: Er darf nur den Rand treffen, nicht das Feld.

    Alle drei Regime aus dieser Suite – ueberheblich (wahres T = 3), gut geeicht (1) und
    unterheblich (0,35) – muessen durchgehen. Die unterhebliche Anpassung landet bei
    ``0,4017`` und damit rund beim Achtfachen der Untergrenze: Ein zu grosszuegig
    gefasster Rand (etwa ``T <= T_UNTERGRENZE * 10``) wuerde sie faelschlich ablehnen
    und faellt hier auf.
    """
    for vorhersagen in (
        _ueberhebliche_vorhersagen(seed=3),
        _gut_geeichte_vorhersagen(seed=3),
        _unterhebliche_vorhersagen(seed=3),
    ):
        logits, y, klassen = vorhersagen
        T = fit_temperature(logits, y, klassen)
        assert T_UNTERGRENZE < T < T_OBERGRENZE


def test_ece_lehnt_nicht_normierte_zeilen_ab() -> None:
    """Waechter mit eigenem Test: Wer versehentlich Logits statt Wahrscheinlichkeiten
    uebergibt, bekaeme sonst eine Zahl, die wie ein Eichfehler aussieht."""
    logits, y, klassen = _ueberhebliche_vorhersagen(seed=3)
    with pytest.raises(ValueError, match="Zeilensumme"):
        expected_calibration_error(logits, y, klassen)


def test_ece_bindet_die_schwelle_der_zeilensummenpruefung() -> None:
    """Bindet den **Abstand** der Schwelle zur echten Verteilung, nicht bloss ihr Dasein.

    Der Test darueber schiebt rohe Logits hinein, deren Zeilensummen weit danebenliegen.
    Gemessen: Setzt man ``atol`` auf 0,5, 1, 2, 5 oder 10, bleibt die Suite gruen – die
    Schwelle duerfte praktisch alles sein. Hier steht sie zwischen zwei nahe beieinander
    liegenden Faellen: Eine Zeilensumme von 1,02 (0,51 + 0,51) ist ein echter
    Normierungsfehler und muss auffallen; eine von 1,00005 ist Gleitkommarauschen, wie es
    der ``float32``-Rueckgabetyp von ``apply_temperature`` erzeugt, und muss durchgehen.
    Eine gelockerte Schwelle macht die erste Haelfte rot, eine verschaerfte die zweite.
    """
    with pytest.raises(ValueError, match="Zeilensumme"):
        expected_calibration_error(np.array([[0.51, 0.51]]), ["a"], ["a", "b"])
    expected_calibration_error(np.array([[0.5, 0.50005]]), ["b"], ["a", "b"])


def test_ece_lehnt_wahrscheinlichkeiten_ausserhalb_null_bis_eins_ab() -> None:
    """Waechter mit eigenem Test: Werte ausserhalb ``[0, 1]`` werden abgelehnt, nicht
    gekappt.

    Gemessen am vorherigen Stand: ``proba=[[1.5, -0.5]]`` mit ``y=["a"]`` ergab ``0.0``.
    Die Zeilensumme ist exakt 1, der Waechter darueber greift also nicht, und ein
    ``np.clip`` verbuchte die Konfidenz 1,5 stillschweigend als 1,0 – gemeldet wurde
    perfekte Eichung fuer eine Eingabe, die gar keine Wahrscheinlichkeiten enthaelt.
    """
    with pytest.raises(ValueError, match="zwischen 0 und 1"):
        expected_calibration_error(np.array([[1.5, -0.5]]), ["a"], ["a", "b"])


def test_ece_lehnt_unpassende_korbzahl_ab() -> None:
    """Waechter mit eigenem Test: Bei ``bins=0`` liefe die Schleife ueber nichts und
    gaebe stillschweigend 0,0 zurueck – ein perfekt geeichtes Modell auf Knopfdruck."""
    logits, y, klassen = _ueberhebliche_vorhersagen(seed=3)
    proba = apply_temperature(logits, 1.0)
    for korbzahl in (0, -3):
        with pytest.raises(ValueError, match="Koerbe"):
            expected_calibration_error(proba, y, klassen, bins=korbzahl)


def test_unpassende_spaltenzahl_wird_abgelehnt() -> None:
    """Dass ``fit_temperature`` und der Eichfehler die Formpruefung *anwenden* – die Regel
    selbst steht in ``tests/test_zahlen.py``.

    Laufen Spalten und ``classes`` auseinander, zeigte jede Konfidenz auf die falsche
    Klasse (dieselbe lautlose Falle wie in ``classify._klassenreihenfolge_pruefen``) und
    alle Zahlen blieben trotzdem plausibel.
    """
    logits, y, klassen = _ueberhebliche_vorhersagen(seed=3)
    with pytest.raises(ValueError, match="Spalte"):
        fit_temperature(logits, y, [*klassen, "e"])
    with pytest.raises(ValueError, match="Spalte"):
        expected_calibration_error(apply_temperature(logits, 1.0), y, [*klassen, "e"])


def test_unpassende_zeilenzahl_wird_abgelehnt() -> None:
    """Dass ``fit_temperature`` die Zeilenpruefung anwendet (Regel: ``test_zahlen.py``).
    Weniger Wahrheiten als Vorhersagen wuerde sonst über ``zip`` still abgeschnitten – die
    Kalibrierung liefe auf einer Teilmenge."""
    logits, y, klassen = _ueberhebliche_vorhersagen(seed=3)
    with pytest.raises(ValueError, match="Zeile"):
        fit_temperature(logits, y[:-1], klassen)


def test_unbekannte_klasse_wird_abgelehnt() -> None:
    """Dass ``fit_temperature`` den Klassenindex anwendet (Regel: ``test_zahlen.py``).
    Eine Wahrheit, die in ``classes`` fehlt, hat keine Spalte – ohne Pruefung liefe sie auf
    einen falschen Index oder stillschweigend mit."""
    logits, y, klassen = _ueberhebliche_vorhersagen(seed=3)
    y_verfaelscht = [*y[:-1], "z"]
    with pytest.raises(ValueError, match="unbekannte"):
        fit_temperature(logits, y_verfaelscht, klassen)


def test_kalibriermenge_darf_nicht_die_trainingsmenge_sein() -> None:
    """Konzept § 7.3 verlangt eine SEPARATE Kalibriermenge. Auf den Trainingslogits
    angepasst, ergaebe Temperature Scaling ein T nahe 1 und taeuschte gute Eichung vor –
    dieser Test haelt fest, dass die beiden Mengen ueberhaupt verschieden sind."""
    assert set(training_documents()["document_id"]).isdisjoint(
        set(calibration_documents()["document_id"])
    )
