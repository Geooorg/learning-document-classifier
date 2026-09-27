"""Tests für Aufgabe 15: die Bilder und die Ablage (Konzept § 9.3, § 11).

Die Tests arbeiten auf **synthetischen** Daten. Auf dem echten Bestand erreichen alle drei
Modellarten Macro-F1 1,000; jede Konfidenz läge dort im obersten Korb, das Reliability
Diagram bestünde aus einem einzigen Punkt, und eine Zusicherung dagegen prüfte den Korpus
statt der Umsetzung.

**Ein Bildtest, der nur prüft, dass eine Datei entsteht, prüft nichts** – eine leere
Leinwand bestünde ihn. Deshalb binden die Tests hier drei Dinge, und keines davon ist die
Existenz einer Datei:

* die **Daten hinter dem Bild** (``reliability_bins``) gegen von Hand gesetzte Pole –
  perfekt geeicht, überheblich, nur hohe Konfidenz;
* die **Figur** des Reliability Diagrams über ihre Linien: wie viele, welche Punkte sie
  tragen, wie sie beschriftet sind. Geprüft wird am ``Figure``-Objekt, nicht an den Pixeln
  der PNG;
* die **Geometrie** der Konfusionsmatrix über die Bildpunkte. ``confusion_heatmap`` gibt
  keine Figur zurück, also wird die einzige Eigenschaft geprüft, die eine transponierte
  Matrix verrät: wohin sich ein erhöhter Zählwert im Bild bewegt, wenn er die Zeile oder
  die Spalte wechselt.

**Die Korbeinteilung ist dieselbe wie beim Eichfehler, und das steht hier nicht als
Behauptung.** ``reliability_bins`` *ist* die Rechnung, aus der
``expected_calibration_error`` seine Zahl bildet (siehe ``doccls.calibrate``);
``test_koerbe_und_eichfehler_benutzen_dieselbe_einteilung`` misst das zusätzlich über drei
Korbzahlen nach. Liefen Bild und Zahl auseinander, zeigte das Diagramm etwas anderes als
der ECE darunter.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from matplotlib.figure import Figure

from doccls.calibrate import apply_temperature, expected_calibration_error, fit_temperature
from doccls.evaluate import WAHR_SPALTE, Metrics, confusion, evaluate
from doccls.reports import (
    ReliabilityBin,
    confusion_heatmap,
    reliability_bins,
    reliability_diagram,
    write_metrics,
)

#: Bewusst **nicht** alphabetisch: Keine Zuordnung darf daran hängen, dass ``sorted``
#: zufällig dieselbe Reihenfolge ergibt.
KLASSEN_ZWEI = ("RECHNUNG", "GUTSCHRIFT")
KLASSEN_DREI = ("VERTRAG", "AGB", "RECHNUNG")


def _aus_konfidenz(
    konfidenz: npt.NDArray[np.float64],
    richtig: npt.NDArray[np.bool_],
    rng: np.random.Generator,
    klassen: Sequence[str] = KLASSEN_ZWEI,
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Zwei Klassen, vorgegebene Konfidenz und vorgegebene Korrektheit je Zeile.

    **Die vorhergesagte Klasse wird gewürfelt, nicht auf Spalte 0 gelegt.** Genau diese
    Falle hat in dieser Phase schon zugeschlagen: Lag die beste Klasse in jedem Testfall
    auf Index 0, überlebte die Mutation ``werte.max(axis=1) → werte[:, 0]`` die ganze
    Suite.
    """
    assert konfidenz.min() > 0.5, "sonst ist die vorgegebene Klasse nicht das argmax"
    vorhergesagt = rng.integers(0, 2, konfidenz.size)
    proba = np.empty((konfidenz.size, 2), dtype=np.float64)
    proba[np.arange(konfidenz.size), vorhergesagt] = konfidenz
    proba[np.arange(konfidenz.size), 1 - vorhergesagt] = 1.0 - konfidenz
    wahrheit = np.where(richtig, vorhergesagt, 1 - vorhergesagt)
    return proba, [klassen[index] for index in wahrheit]


def _perfekt_geeicht(
    seed: int,
) -> tuple[npt.NDArray[np.float64], list[str], tuple[str, ...]]:
    """Wo das Modell 0,8 sagt, trifft es in 0,8 der Fälle – die Diagonale, mit Rauschen.

    Die Korrektheit wird mit der Konfidenz als Wahrscheinlichkeit gewürfelt. Gemessen bei
    Seed 19 und 4000 Zeilen: der größte Abstand |Konfidenz − Trefferquote| über die fünf
    besetzten Körbe liegt bei 0,031.
    """
    rng = np.random.default_rng(seed)
    konfidenz = rng.uniform(0.5, 1.0, 4000)
    richtig = rng.random(konfidenz.size) < konfidenz
    proba, y = _aus_konfidenz(konfidenz, richtig, rng)
    return proba, y, KLASSEN_ZWEI


def _ueberheblich(seed: int) -> tuple[npt.NDArray[np.float64], list[str], tuple[str, ...]]:
    """Der Gegenpol: Das Modell sagt 0,8 und trifft in 0,55 der Fälle.

    Die Trefferwahrscheinlichkeit liegt fest 0,25 unter der Konfidenz. Die Körbe müssen
    damit **unter** der Diagonale liegen; ohne diesen Pol bestünde eine Umsetzung, die
    immer die Diagonale zurückgibt.
    """
    rng = np.random.default_rng(seed)
    konfidenz = rng.uniform(0.55, 1.0, 4000)
    richtig = rng.random(konfidenz.size) < konfidenz - 0.25
    proba, y = _aus_konfidenz(konfidenz, richtig, rng)
    return proba, y, KLASSEN_ZWEI


def _nur_hohe_konfidenz(
    seed: int,
) -> tuple[npt.NDArray[np.float64], list[str], tuple[str, ...]]:
    """Alle Konfidenzen über 0,7 – die sieben unteren Körbe bleiben leer.

    Das ist der Normalfall dieses Projekts, nicht der Ausnahmefall: Ein Klassifikator mit
    kalibrierter Konfidenz sitzt fast ausschließlich oben.
    """
    rng = np.random.default_rng(seed)
    konfidenz = rng.uniform(0.7, 1.0, 600)
    richtig = rng.random(konfidenz.size) < 0.8
    proba, y = _aus_konfidenz(konfidenz, richtig, rng)
    return proba, y, KLASSEN_ZWEI


def _vorher_nachher(
    seed: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], list[str], tuple[str, ...]]:
    """Überhebliche Logits und dieselben Logits nach Temperature Scaling.

    Gebaut über die echten Funktionen (``fit_temperature``/``apply_temperature``), nicht
    über zwei von Hand gesetzte Kurven – sonst prüfte das Bild eine Lage, die im Lauf gar
    nicht vorkommt. Gemessen bei Seed 19: ``T = 1,919``, ECE 0,1144 vor und 0,0441 nach
    der Kalibrierung.
    """
    rng = np.random.default_rng(seed)
    ziel = rng.integers(0, len(KLASSEN_DREI), 600)
    logits = rng.normal(0.0, 1.0, (ziel.size, len(KLASSEN_DREI)))
    logits[np.arange(ziel.size), ziel] += 1.4
    logits *= 3.0
    y = [KLASSEN_DREI[index] for index in ziel]
    vorher = apply_temperature(logits, 1.0)
    nachher = apply_temperature(logits, fit_temperature(logits, y, KLASSEN_DREI))
    return vorher, nachher, y, KLASSEN_DREI


# --------------------------------------------------------------------------------------
# Die Daten hinter dem Bild
# --------------------------------------------------------------------------------------


def test_reliability_daten_folgen_der_diagonale_bei_perfekter_eichung() -> None:
    """Bindet die Daten hinter dem Bild. Ein Test, der nur prueft, dass eine PNG-Datei
    entsteht, wuerde auch eine leere Leinwand durchgehen lassen."""
    proba, y, klassen = _perfekt_geeicht(seed=19)
    koerbe = reliability_bins(proba, y, klassen, bins=10)
    assert [korb for korb in koerbe if korb.anzahl >= 10], "sonst prueft die Schleife nichts"
    for korb in koerbe:
        if korb.anzahl >= 10:
            assert abs(korb.mittlere_konfidenz - korb.trefferquote) < 0.1


def test_reliability_daten_zeigen_ueberheblichkeit() -> None:
    """Gegenprobe: Bei einem ueberheblichen Modell muessen die Koerbe UNTER der Diagonale
    liegen - mittlere Konfidenz hoeher als Trefferquote. Ohne beide Pole bestuende eine
    Umsetzung, die immer die Diagonale zurueckgibt."""
    proba, y, klassen = _ueberheblich(seed=19)
    koerbe = [k for k in reliability_bins(proba, y, klassen, bins=10) if k.anzahl >= 10]
    assert koerbe, "Keine ausreichend besetzten Koerbe - der Test prueft sonst nichts"
    assert sum(k.mittlere_konfidenz - k.trefferquote for k in koerbe) > 0


def test_leere_koerbe_werden_nicht_als_nullpunkte_gezeichnet() -> None:
    """Ein leerer Korb, als (0,0) gezeichnet, zieht die Kurve nach unten und laesst ein
    gut geeichtes Modell schlecht aussehen."""
    proba, y, klassen = _nur_hohe_konfidenz(seed=19)
    koerbe = reliability_bins(proba, y, klassen, bins=10)
    assert all(k.anzahl > 0 for k in koerbe)
    # Ohne diese zweite Zusicherung waere die erste erfuellt, weil es gar keine leeren
    # Koerbe zu unterdruecken gibt - genau die Sorte Test, die nichts aussagt.
    assert len(koerbe) == 3, "erwartet sind nur die Koerbe 0,7-0,8, 0,8-0,9 und 0,9-1,0"
    assert min(k.untergrenze for k in koerbe) == pytest.approx(0.7)


def test_konfidenz_ist_das_zeilenmaximum_und_nicht_die_erste_spalte() -> None:
    """Die Konfidenz einer Zeile ist ihr Maximum. Hier trägt Spalte 0 nie das Maximum;
    eine Umsetzung, die ``werte[:, 0]`` liest, landet in den untersten Körben."""
    proba = np.array([[0.02, 0.95, 0.03], [0.05, 0.03, 0.92], [0.01, 0.10, 0.89]])
    koerbe = reliability_bins(proba, ["AGB", "RECHNUNG", "AGB"], KLASSEN_DREI, bins=10)
    assert [korb.untergrenze for korb in koerbe] == [pytest.approx(0.8), pytest.approx(0.9)]
    assert all(korb.mittlere_konfidenz > 0.5 for korb in koerbe)


def test_trefferquote_haengt_am_klassennamen_nicht_an_der_spaltennummer() -> None:
    """Dieselben Daten, Spalten und ``classes`` gemeinsam vertauscht – die Körbe müssen
    identisch bleiben. Der lautloseste Fehler dieser Phase ist die Zahl aus der einen und
    der Name aus einer anders sortierten Liste."""
    proba, y, klassen = _ueberheblich(seed=19)
    urspruenglich = reliability_bins(proba, y, klassen, bins=10)
    getauscht = reliability_bins(proba[:, ::-1], y, klassen[::-1], bins=10)
    assert getauscht == urspruenglich
    # Und zur Gegenprobe: Wird nur die Matrix vertauscht, muss sich etwas ändern.
    nur_matrix = reliability_bins(proba[:, ::-1], y, klassen, bins=10)
    assert nur_matrix != urspruenglich


# --------------------------------------------------------------------------------------
# Die Korbeinteilung – dieselbe wie beim Eichfehler, und abhängig von ``bins``
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("korbzahl", [4, 10, 17])
def test_koerbe_und_eichfehler_benutzen_dieselbe_einteilung(korbzahl: int) -> None:
    """Der ECE ist die nach Besetzung gewichtete Summe der Korbabstände. Zeigte das Bild
    eine andere Einteilung als die Zahl darunter, liefe genau diese Gleichung auseinander.

    Drei Korbzahlen, weil eine Umsetzung, die ``bins`` wegwirft und fest mit 10 rechnet,
    bei ``bins=10`` grün bliebe.
    """
    proba, y, klassen = _ueberheblich(seed=19)
    koerbe = reliability_bins(proba, y, klassen, bins=korbzahl)
    gesamt = sum(korb.anzahl for korb in koerbe)
    aus_koerben = sum(
        korb.anzahl / gesamt * abs(korb.mittlere_konfidenz - korb.trefferquote) for korb in koerbe
    )
    assert aus_koerben == pytest.approx(
        expected_calibration_error(proba, y, klassen, bins=korbzahl), abs=1e-12
    )


def test_jede_zeile_faellt_in_genau_einen_korb() -> None:
    """Die Besetzungen summieren sich auf die Zahl der Dokumente – keine Zeile doppelt,
    keine verloren. Zwei Korbzahlen, damit die Summe nicht an einer festen hängt."""
    proba, y, klassen = _ueberheblich(seed=19)
    for korbzahl in (3, 10):
        koerbe = reliability_bins(proba, y, klassen, bins=korbzahl)
        assert sum(korb.anzahl for korb in koerbe) == proba.shape[0]


def test_korbgrenzen_folgen_der_korbzahl() -> None:
    """Die Grenzen sind ``i/bins`` bis ``(i+1)/bins``, und die mittlere Konfidenz liegt
    zwischen ihnen. Mit **zwei** Korbzahlen: Eine Umsetzung, die ``bins`` wegwirft und
    immer zehn Körbe bildet, liefert bei ``bins=4`` Grenzen der Breite 0,1."""
    proba, y, klassen = _ueberheblich(seed=19)
    breiten = {}
    for korbzahl in (4, 10):
        koerbe = reliability_bins(proba, y, klassen, bins=korbzahl)
        assert koerbe
        for korb in koerbe:
            assert korb.obergrenze - korb.untergrenze == pytest.approx(1.0 / korbzahl)
            assert korb.untergrenze <= korb.mittlere_konfidenz <= korb.obergrenze
        breiten[korbzahl] = koerbe[0].obergrenze - koerbe[0].untergrenze
    assert breiten[4] > breiten[10], "sonst hängt die Breite nicht an der Korbzahl"


@pytest.mark.parametrize("korbzahl", [5, 10])
def test_der_oberste_korb_ist_rechts_geschlossen(korbzahl: int) -> None:
    """Eine Konfidenz von exakt 1,0 gehört in den obersten Korb und nicht in einen
    zusätzlichen elften – sonst stünde im Bild ein Punkt bei Konfidenz 1,0 mit einem
    einzigen Dokument neben dem Korb, in den er gehört."""
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [0.97, 0.03]])
    koerbe = reliability_bins(proba, ["RECHNUNG", "GUTSCHRIFT", "RECHNUNG"], KLASSEN_ZWEI, korbzahl)
    assert len(koerbe) == 1
    assert koerbe[0].obergrenze == pytest.approx(1.0)
    assert koerbe[0].anzahl == 3
    assert koerbe[0].trefferquote == pytest.approx(1.0)


def test_koerbe_ohne_dokumente_werden_gemeldet() -> None:
    """Über die leere Menge gibt es keine Eichung. Ohne diesen Wächter käme eine Meldung
    aus NumPy (``zero-size array to reduction``) oder eine Division durch null."""
    with pytest.raises(ValueError, match="Keine Dokumente"):
        reliability_bins(np.zeros((0, 2)), [], KLASSEN_ZWEI)


@pytest.mark.parametrize("korbzahl", [0, -1])
def test_weniger_als_ein_korb_wird_gemeldet(korbzahl: int) -> None:
    """Die Prüfung sitzt jetzt in ``reliability_bins`` und muss von dort aus weiter
    greifen – ``expected_calibration_error`` verlässt sich darauf."""
    proba, y, klassen = _nur_hohe_konfidenz(seed=19)
    with pytest.raises(ValueError, match="Koerbe"):
        reliability_bins(proba, y, klassen, bins=korbzahl)


# --------------------------------------------------------------------------------------
# Das Reliability Diagram
# --------------------------------------------------------------------------------------


def test_diagramm_enthaelt_beide_kurven(tmp_path: Path) -> None:
    """Konzept § 11 verlangt vorher UND nachher in einem Bild - nebeneinander sieht man
    den Unterschied nicht. Geprueft wird ueber die Zahl der Linien in der Figur, nicht
    ueber die Pixel."""
    vorher, nachher, y, klassen = _vorher_nachher(seed=19)
    pfad, figur = reliability_diagram(vorher, nachher, y, klassen, tmp_path / "r.png")
    assert pfad.exists() and pfad.stat().st_size > 0
    beschriftungen = [str(linie.get_label()) for linie in figur.axes[0].get_lines()]
    assert any("vor" in b.lower() for b in beschriftungen)
    assert any("nach" in b.lower() for b in beschriftungen)
    assert any("ideal" in b.lower() or "diagonale" in b.lower() for b in beschriftungen)
    assert len(beschriftungen) == 3, "genau zwei Kurven und die Diagonale"


def test_diagramm_zeichnet_die_gerechneten_koerbe(tmp_path: Path) -> None:
    """Die Punkte der beiden Kurven sind die Körbe aus ``reliability_bins`` – nicht
    irgendeine Kurve, die zufällig ähnlich aussieht. Ohne diese Bindung bestünde ein
    Diagramm, das beide Male dieselben Daten zeichnet, alle Beschriftungstests."""
    vorher, nachher, y, klassen = _vorher_nachher(seed=19)
    _, figur = reliability_diagram(vorher, nachher, y, klassen, tmp_path / "r.png")
    linien = {str(linie.get_label()): linie.get_xydata() for linie in figur.axes[0].get_lines()}
    for beschriftung, proba in (("vor", vorher), ("nach", nachher)):
        name = next(b for b in linien if beschriftung in b.lower())
        erwartet = [
            (korb.mittlere_konfidenz, korb.trefferquote)
            for korb in reliability_bins(proba, y, klassen, bins=10)
        ]
        assert np.asarray(linien[name]) == pytest.approx(np.asarray(erwartet))


def test_diagramm_zeichnet_zwei_verschiedene_kurven(tmp_path: Path) -> None:
    """Gegenprobe zum vorigen Test: Vorher und nachher dürfen nicht dieselben Punkte
    tragen, sonst zeigte das Bild die Wirkung der Kalibrierung nicht."""
    vorher, nachher, y, klassen = _vorher_nachher(seed=19)
    _, figur = reliability_diagram(vorher, nachher, y, klassen, tmp_path / "r.png")
    linien = {str(linie.get_label()): linie.get_xydata() for linie in figur.axes[0].get_lines()}
    vorher_punkte = next(v for k, v in linien.items() if "vor" in k.lower())
    nachher_punkte = next(v for k, v in linien.items() if "nach" in k.lower())
    assert not np.array_equal(vorher_punkte, nachher_punkte)


def test_die_diagonale_geht_von_null_nach_eins(tmp_path: Path) -> None:
    """Das Ideal ist die Winkelhalbierende über den ganzen Bereich. Eine Diagonale, die
    nur über die beobachteten Konfidenzen liefe, wäre kein Ideal, sondern eine dritte
    Messkurve."""
    vorher, nachher, y, klassen = _vorher_nachher(seed=19)
    _, figur = reliability_diagram(vorher, nachher, y, klassen, tmp_path / "r.png")
    diagonale = next(
        linie
        for linie in figur.axes[0].get_lines()
        if "ideal" in str(linie.get_label()).lower()
        or "diagonale" in str(linie.get_label()).lower()
    )
    ecken = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert np.asarray(diagonale.get_xydata()) == pytest.approx(ecken)


def test_die_beschriftung_nennt_den_eichfehler_der_kurve(tmp_path: Path) -> None:
    """Die Zahl steht im Bild, und zwar die, die aus denselben Körben kommt. Eine
    Beschriftung mit dem ECE der jeweils anderen Kurve fällt hier auf."""
    vorher, nachher, y, klassen = _vorher_nachher(seed=19)
    _, figur = reliability_diagram(vorher, nachher, y, klassen, tmp_path / "r.png")
    linien = {str(linie.get_label()) for linie in figur.axes[0].get_lines()}
    for stichwort, proba in (("vor", vorher), ("nach", nachher)):
        name = next(b for b in linien if stichwort in b.lower())
        erwartet = f"{expected_calibration_error(proba, y, klassen, bins=10):.3f}".replace(".", ",")
        assert erwartet in name
    assert expected_calibration_error(nachher, y, klassen) < expected_calibration_error(
        vorher, y, klassen
    ), "sonst tragen beide Beschriftungen zufaellig dieselbe Zahl"


def test_die_korbzahl_wirkt_auf_das_diagramm(tmp_path: Path) -> None:
    """``bins`` ist ein Parameter des Bildes, kein Schmuck: Mit vier Körben stehen weniger
    Punkte auf der Kurve als mit zehn."""
    vorher, nachher, y, klassen = _vorher_nachher(seed=19)

    def punkte(korbzahl: int, welche: str) -> int:
        _, figur = reliability_diagram(
            vorher, nachher, y, klassen, tmp_path / f"r{korbzahl}.png", bins=korbzahl
        )
        linie = next(
            li for li in figur.axes[0].get_lines() if str(li.get_label()).startswith(welche)
        )
        return len(np.asarray(linie.get_xydata()))

    # BEIDE Kurven, nicht nur die vordere. Gemessen, bevor das hier stand: Ein fest
    # verdrahtetes bins=10 fuer die Kurve "nach" liess alle 58 Tests gruen -- die beiden
    # Kurven im selben Bild trugen dann verschiedene Einteilungen und nannten in ihren
    # Beschriftungen zwei nicht vergleichbare Eichfehler.
    for welche in ("vor", "nach"):
        assert punkte(4, welche) < punkte(10, welche), welche


def test_diagramm_legt_das_verzeichnis_an(tmp_path: Path) -> None:
    """Der Lauf schreibt nach ``data/reports/<model_version>/`` – ein Verzeichnis, das es
    beim ersten Lauf noch nicht gibt."""
    vorher, nachher, y, klassen = _vorher_nachher(seed=19)
    ziel = tmp_path / "reports" / "logreg-1" / "reliability.png"
    pfad, _ = reliability_diagram(vorher, nachher, y, klassen, ziel)
    assert pfad == ziel and ziel.exists() and ziel.stat().st_size > 0


# --------------------------------------------------------------------------------------
# Die Konfusionsmatrix als Bild
# --------------------------------------------------------------------------------------

#: Grundmatrix der Bildtests: überall 3, unten rechts eine 9. Die 9 steht fest, damit
#: ``vmin``/``vmax`` über alle Spielarten gleich bleiben – sonst unterschieden sich zwei
#: Bilder schon durch ihre Farbskala und nicht durch die erhöhte Zelle.
_GRUND = [[3, 3, 3], [3, 3, 3], [3, 3, 9]]


def _matrix(zaehlung: Sequence[Sequence[int]]) -> pl.DataFrame:
    """Eine Konfusionsmatrix in der Form, die ``evaluate.confusion`` liefert."""
    tabelle: dict[str, list[str] | list[int]] = {WAHR_SPALTE: list(KLASSEN_DREI)}
    for spalte, name in enumerate(KLASSEN_DREI):
        tabelle[name] = [int(zeile[spalte]) for zeile in zaehlung]
    return pl.DataFrame(tabelle)


def _mit_erhoehter_zelle(zeile: int, spalte: int) -> pl.DataFrame:
    zaehlung = [list(z) for z in _GRUND]
    zaehlung[zeile][spalte] = 9
    return _matrix(zaehlung)


def _pfad(ergebnis: tuple[Path, Figure]) -> Path:
    """Nur der Pfad aus ``(Pfad, Figur)`` – die Figur braucht der Bildpunktvergleich nicht."""
    return ergebnis[0]


def _bild(pfad: Path) -> npt.NDArray[np.float64]:
    gelesen: npt.NDArray[np.float64] = np.asarray(matplotlib.image.imread(pfad), dtype=np.float64)
    return gelesen


def _schwerpunkt_des_unterschieds(
    links: npt.NDArray[np.float64], rechts: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """Wo im Bild sich zwei Darstellungen unterscheiden – als gewichteter Schwerpunkt.

    Das ist der einzige Weg, die Ausrichtung der Matrix an einer Funktion zu prüfen, die
    nur einen ``Path`` zurückgibt: Eine transponierte Umsetzung schiebt den Unterschied
    von der Spalte in die Zeile, und das ist messbar.
    """
    unterschied = np.abs(links - rechts).sum(axis=2)
    gewicht = float(unterschied.sum())
    assert gewicht > 0.0, "die beiden Bilder sind identisch - der Test prueft nichts"
    zeilen, spalten = np.mgrid[0 : unterschied.shape[0], 0 : unterschied.shape[1]]
    return (
        float((unterschied * spalten).sum() / gewicht),
        float((unterschied * zeilen).sum() / gewicht),
    )


def test_konfusionsbild_legt_die_wahrheit_in_die_zeile(tmp_path: Path) -> None:
    """Zeile = Wahrheit, Spalte = Vorhersage. Eine transponierte Darstellung vertauscht
    „Rechnung als Gutschrift gelesen" mit „Gutschrift als Rechnung gelesen", und alle
    Randsummen blieben plausibel.

    Gemessen wird über die Bildpunkte: Wandert der erhöhte Zählwert eine Spalte nach
    rechts, muss sich der Unterschied im Bild nach rechts verschieben und **nicht** nach
    unten.
    """
    grund = _bild(_pfad(confusion_heatmap(_matrix(_GRUND), tmp_path / "g.png")))
    oben_links = _bild(_pfad(confusion_heatmap(_mit_erhoehter_zelle(0, 0), tmp_path / "a.png")))
    oben_rechts = _bild(_pfad(confusion_heatmap(_mit_erhoehter_zelle(0, 1), tmp_path / "b.png")))
    unten_links = _bild(_pfad(confusion_heatmap(_mit_erhoehter_zelle(1, 0), tmp_path / "c.png")))

    x_a, y_a = _schwerpunkt_des_unterschieds(grund, oben_links)
    x_b, y_b = _schwerpunkt_des_unterschieds(grund, oben_rechts)
    x_c, y_c = _schwerpunkt_des_unterschieds(grund, unten_links)

    assert x_b - x_a > 50.0, "eine Spalte weiter muss im Bild weiter rechts liegen"
    assert abs(y_b - y_a) < 15.0, "dieselbe Zeile muss auf derselben Hoehe liegen"
    assert y_c - y_a > 50.0, "eine Zeile weiter muss im Bild weiter unten liegen"
    assert abs(x_c - x_a) < 15.0, "dieselbe Spalte muss an derselben Stelle liegen"


def test_konfusionsbild_beschriftet_die_achsen_in_der_richtigen_richtung(
    tmp_path: Path,
) -> None:
    """Die Geometrie der Zellen ist gebunden, die **Beschriftung** war es nicht.

    Gemessen, bevor dieser Test da war: x- und y-Beschriftung vertauscht liess die ganze
    Testreihe gruen. Alle Zahlen und alle Klassennamen bleiben dabei richtig – und
    trotzdem liest jeder Betrachter jede Zelle neben der Diagonale verkehrt herum. Genau
    dieser Unterschied ist laut Moduldoc „der ganze Inhalt".
    """
    _, figur = confusion_heatmap(_matrix(_GRUND), tmp_path / "achsen.png")
    achse = figur.axes[0]
    assert achse.get_xlabel() == "Vorhergesagt"
    assert achse.get_ylabel() == "Wahr"


def test_konfusionsbild_faengt_die_farbskala_bei_null_an(tmp_path: Path) -> None:
    """Der Docstring sagt zu, dass zwei Bilder desselben Laufs vergleichbar sind. Das
    haelt nur, wenn die Skala fest bei 0 beginnt – sonst wird dieselbe Zellbesetzung in
    einem schwach besetzten Bild dunkel und in einem stark besetzten hell. Ohne diesen
    Test war ``vmin=0.0`` ungebunden: Sein Entfernen liess alle Tests gruen, weil alle
    verwendeten Matrizen ohnehin dasselbe Minimum haben.
    """
    _, figur = confusion_heatmap(_matrix(_GRUND), tmp_path / "skala.png")
    bild = figur.axes[0].images[0]
    assert bild.get_clim()[0] == 0.0


def test_beide_bilder_ueberschreiben_eine_vorhandene_datei(tmp_path: Path) -> None:
    """Ein Lauf schreibt nach ``data/reports/<model_version>/``, und ein zweiter Lauf
    derselben Version muss die Bilder des ersten ersetzen. Bliebe ein altes Bild stehen,
    waere es in Aufgabe 16 vom aktuellen nicht zu unterscheiden – und genau das ist der
    Fehler, den niemand bemerkt. Geprueft wird ueber den Inhalt, nicht ueber die
    Aenderungszeit: Die Aufloesung der Zeitstempel reicht hier nicht.
    """
    vorher, nachher, y, klassen = _vorher_nachher(seed=19)
    ziel_diagramm = tmp_path / "reliability.png"
    reliability_diagram(vorher, nachher, y, klassen, ziel_diagramm, bins=4)
    vier_koerbe = ziel_diagramm.read_bytes()
    reliability_diagram(vorher, nachher, y, klassen, ziel_diagramm, bins=10)
    assert ziel_diagramm.read_bytes() != vier_koerbe

    ziel_matrix = tmp_path / "confusion.png"
    confusion_heatmap(_matrix(_GRUND), ziel_matrix)
    grund = ziel_matrix.read_bytes()
    confusion_heatmap(_mit_erhoehter_zelle(0, 1), ziel_matrix)
    assert ziel_matrix.read_bytes() != grund

    ziel_zahlen = tmp_path / "metrics.json"
    write_metrics(_kennzahlen(), ziel_zahlen)
    ziel_zahlen.write_text("{}", encoding="utf-8")
    write_metrics(_kennzahlen(), ziel_zahlen)
    assert ziel_zahlen.read_text(encoding="utf-8") != "{}"


def test_konfusionsbild_ist_reproduzierbar(tmp_path: Path) -> None:
    """Dieselbe Matrix ergibt dieselben Bildpunkte. Ohne das wäre der Vergleich zweier
    Läufe in ``docs/auswertungen.md`` nicht zu führen."""
    erst = _bild(_pfad(confusion_heatmap(_matrix(_GRUND), tmp_path / "erst.png")))
    zweit = _bild(_pfad(confusion_heatmap(_matrix(_GRUND), tmp_path / "zweit.png")))
    assert np.array_equal(erst, zweit)


def test_konfusionsbild_nimmt_die_matrix_aus_confusion(tmp_path: Path) -> None:
    """Die Tabelle aus ``evaluate.confusion`` geht ohne Umbau hinein – sonst stünde
    zwischen Rechnung und Bild eine zweite, ungeprüfte Umformung."""
    wahrheit = ["VERTRAG", "AGB", "RECHNUNG", "RECHNUNG"]
    vorhersage = ["VERTRAG", "RECHNUNG", "RECHNUNG", "AGB"]
    pfad, _ = confusion_heatmap(confusion(wahrheit, vorhersage, KLASSEN_DREI), tmp_path / "k.png")
    assert pfad.exists() and pfad.stat().st_size > 0


def test_konfusionsbild_legt_das_verzeichnis_an(tmp_path: Path) -> None:
    ziel = tmp_path / "reports" / "logreg-1" / "confusion.png"
    assert confusion_heatmap(_matrix(_GRUND), ziel)[0] == ziel
    assert ziel.exists()


def test_konfusionsbild_ohne_wahrheitsspalte_wird_gemeldet(tmp_path: Path) -> None:
    """Ohne die Zeilenbeschriftung ist nicht bekannt, welche Klasse welche Zeile ist –
    gezeichnet würde eine Matrix mit vertauschbaren Zeilen."""
    ohne = _matrix(_GRUND).drop(WAHR_SPALTE)
    with pytest.raises(ValueError, match=WAHR_SPALTE):
        confusion_heatmap(ohne, tmp_path / "k.png")


def test_konfusionsbild_mit_anderer_spaltenreihenfolge_wird_gemeldet(tmp_path: Path) -> None:
    """Zeilen und Spalten müssen dieselbe Klassenreihenfolge tragen. Eine andere
    Spaltenreihenfolge ergäbe ein Bild, dessen Diagonale nicht die Treffer sind – und
    das sähe völlig plausibel aus."""
    vertauscht = _matrix(_GRUND).select(
        WAHR_SPALTE, KLASSEN_DREI[1], KLASSEN_DREI[0], KLASSEN_DREI[2]
    )
    with pytest.raises(ValueError, match="Reihenfolge"):
        confusion_heatmap(vertauscht, tmp_path / "k.png")


def test_konfusionsbild_ohne_dokumente_wird_gemeldet(tmp_path: Path) -> None:
    """Eine Matrix aus lauter Nullen hat keine Farbskala – ``vmin`` und ``vmax`` fielen
    zusammen, und jede Zelle bekäme dieselbe Farbe."""
    with pytest.raises(ValueError, match="kein Dokument"):
        confusion_heatmap(_matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), tmp_path / "k.png")


# --------------------------------------------------------------------------------------
# Die Ablage der Kennzahlen
# --------------------------------------------------------------------------------------


def _kennzahlen(alle_richtig: bool = False) -> Metrics:
    """Kennzahlen aus einem echten ``evaluate``-Lauf, nicht von Hand zusammengesetzt."""
    proba, y, klassen = _ueberheblich(seed=19)
    if alle_richtig:
        proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        y = [klassen[0], klassen[1], klassen[0], klassen[1]]
    return evaluate(proba, y, klassen)


def test_write_metrics_ist_umkehrbar(tmp_path: Path) -> None:
    """Der Bericht ist nur so viel wert, wie er wieder einlesbar ist. Geprüft wird über
    alle zehn Felder, nicht über die Dateigröße."""
    kennzahlen = _kennzahlen()
    pfad = write_metrics(kennzahlen, tmp_path / "metrics.json")
    zurueck = Metrics.model_validate_json(pfad.read_text(encoding="utf-8"))
    assert zurueck == kennzahlen
    assert set(json.loads(pfad.read_text(encoding="utf-8"))) == set(Metrics.model_fields), (
        "kein Feld darf beim Schreiben verlorengehen"
    )


def test_write_metrics_haelt_die_klassenwerte_einzeln_fest(tmp_path: Path) -> None:
    """``per_class_f1`` ist der Teil, der verschwindet, wenn jemand nur die Skalare
    schreibt – und genau er sagt, *welche* Klasse durchfällt."""
    kennzahlen = _kennzahlen()
    pfad = write_metrics(kennzahlen, tmp_path / "metrics.json")
    geschrieben = json.loads(pfad.read_text(encoding="utf-8"))
    assert geschrieben["per_class_f1"] == kennzahlen.per_class_f1
    assert len(geschrieben["per_class_f1"]) == 2


def test_write_metrics_haelt_ein_fehlendes_auroc_als_null_fest(tmp_path: Path) -> None:
    """``auroc_confidence`` ist ``None``, wenn alle Vorhersagen richtig sind – auf dem
    echten Gold-Set der Normalfall. Als 0,5 oder als fehlender Schlüssel geschrieben,
    ginge diese Aussage in jeden Bericht als Messung ein."""
    kennzahlen = _kennzahlen(alle_richtig=True)
    assert kennzahlen.auroc_confidence is None
    pfad = write_metrics(kennzahlen, tmp_path / "metrics.json")
    geschrieben = json.loads(pfad.read_text(encoding="utf-8"))
    assert "auroc_confidence" in geschrieben
    assert geschrieben["auroc_confidence"] is None
    assert Metrics.model_validate_json(pfad.read_text(encoding="utf-8")) == kennzahlen


def test_write_metrics_legt_das_verzeichnis_an(tmp_path: Path) -> None:
    ziel = tmp_path / "reports" / "logreg-1" / "metrics.json"
    assert write_metrics(_kennzahlen(), ziel) == ziel
    assert ziel.exists()


# --------------------------------------------------------------------------------------
# Das Backend
# --------------------------------------------------------------------------------------


def test_das_backend_oeffnet_kein_fenster() -> None:
    """Ohne ausdrückliche Backend-Wahl versucht matplotlib ein Fenster zu öffnen, und der
    Testlauf hängt auf einem Rechner ohne Anzeige – und zwar erst dort, nicht hier."""
    assert matplotlib.get_backend().lower() == "agg"


def test_keine_figur_bleibt_in_der_pyplot_verwaltung_haengen(tmp_path: Path) -> None:
    """Figuren, die niemand schließt, sammeln sich an: matplotlib warnt ab zwanzig und
    hält bis dahin jede im Speicher. ``reports`` baut deshalb über ``Figure`` statt über
    ``pyplot`` – dann gibt es keine globale Verwaltung, in der etwas hängenbleiben kann."""
    vorher, nachher, y, klassen = _vorher_nachher(seed=19)
    offen_zuvor = set(plt.get_fignums())
    for lauf in range(25):
        reliability_diagram(vorher, nachher, y, klassen, tmp_path / f"r{lauf}.png")
        confusion_heatmap(_matrix(_GRUND), tmp_path / f"k{lauf}.png")
    assert set(plt.get_fignums()) == offen_zuvor


def test_der_korbtyp_traegt_die_fuenf_felder() -> None:
    """Die Signatur aus dem Aufgabentext, an einem Wert festgemacht."""
    korb = ReliabilityBin(
        untergrenze=0.9,
        obergrenze=1.0,
        mittlere_konfidenz=0.95,
        trefferquote=0.8,
        anzahl=40,
    )
    assert (korb.untergrenze, korb.obergrenze, korb.anzahl) == (0.9, 1.0, 40)
    assert korb.mittlere_konfidenz - korb.trefferquote == pytest.approx(0.15)
