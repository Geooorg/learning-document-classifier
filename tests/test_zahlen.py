"""Tests für die gemeinsamen Formprüfungen aus ``doccls.zahlen``.

Die Regeln selbst stehen hier, einmal und direkt an den Funktionen geprüft. Dass
``calibrate``, ``decide`` und ``evaluate`` sie auch **anwenden**, ist eine andere
Zusicherung und bleibt in den Tests der jeweiligen Module: Eine Prüfung, die nur hier
stünde, überlebte jedes versehentlich gelöschte ``pruefe_form(...)`` an einer
Aufrufstelle.
"""

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from doccls.zahlen import als_float64, als_maske, als_reihe, klassenindex, pruefe_form


def test_als_float64_hebt_einfache_genauigkeit_an() -> None:
    """``float32`` hinein, ``float64`` heraus – sonst hinge jede Optimierung an der
    Rundung der Eingabe."""
    werte = als_float64(np.array([[0.25, 0.75]], dtype=np.float32))
    assert werte.dtype == np.float64
    assert werte.tolist() == [[0.25, 0.75]]


def test_als_float64_weist_alles_ab_was_keine_matrix_ist() -> None:
    """Waechter mit eigenem Test: Eine Reihe statt einer Matrix haette keine Spalten, und
    jede spaetere Zuordnung Spalte-zu-Klasse liefe ins Leere."""
    with pytest.raises(ValueError, match="1 Dimensionen"):
        als_float64(np.array([0.25, 0.75]))
    with pytest.raises(ValueError, match="3 Dimensionen"):
        als_float64(np.zeros((2, 2, 2)))


def test_als_reihe_weist_mehrdimensionale_und_leere_eingaben_ab() -> None:
    """Waechter mit eigenem Test: Aus nichts laesst sich keine Schwelle ablesen, und eine
    Matrix ergaebe je Dokument mehr als einen Wert."""
    with pytest.raises(ValueError, match="konfidenz muss eindimensional"):
        als_reihe(np.zeros((2, 2)), "konfidenz")
    with pytest.raises(ValueError, match="konfidenz ist leer"):
        als_reihe(np.array([]), "konfidenz")


def test_als_reihe_weist_nicht_endliche_werte_ab() -> None:
    """``NaN`` sortiert in NumPy ans Ende und waere in einer sortierten Kurve unsichtbar."""
    for kaputt in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="nicht endlich"):
            als_reihe(np.array([0.9, kaputt, 0.7]), "konfidenz")


def test_als_maske_verlangt_den_typ_statt_ihn_umzudeuten() -> None:
    """Ein Gleitkommafeld mit 0,5 darin liesse sich mitteln und ergaebe eine „Praezision",
    die keine ist. Das ``cast`` ist der Kern des Tests: Es sagt aus, dass hier bewusst
    etwas uebergeben wird, was die Annotation ausschliesst – mypy sieht nur annotierte
    Aufrufstellen, der Waechter steht fuer den ungeprueften Rand."""
    with pytest.raises(ValueError, match="boolesches Feld"):
        als_maske(cast(npt.NDArray[np.bool_], np.array([1.0, 0.5, 0.0])), "richtig")
    with pytest.raises(ValueError, match="richtig muss eindimensional"):
        als_maske(np.zeros((2, 2), dtype=np.bool_), "richtig")
    # Gegenprobe: ein boolesches Feld kommt unveraendert durch.
    assert als_maske(np.array([True, False]), "richtig").tolist() == [True, False]


def test_pruefe_form_bindet_beide_richtungen() -> None:
    """Waechter mit eigenem Test: Zeilen gegen Wahrheiten und Spalten gegen Klassen sind
    zwei verschiedene Fehler mit zwei verschiedenen Folgen."""
    matrix = np.zeros((3, 2), dtype=np.float64)
    pruefe_form(matrix, ["a", "b", "a"], ["a", "b"])
    with pytest.raises(ValueError, match="3 Zeilen stehen 2 bekannten Klassen"):
        pruefe_form(matrix, ["a", "b"], ["a", "b"])
    with pytest.raises(ValueError, match="2 Spalten fuer 3 Klassen"):
        pruefe_form(matrix, ["a", "b", "a"], ["a", "b", "c"])


def test_klassenindex_folgt_der_reihenfolge_von_classes() -> None:
    """Nicht alphabetisch: Die Zuordnung Name-zu-Spalte darf nicht daran haengen, dass
    ``sorted(classes)`` zufaellig dasselbe ergibt."""
    index = klassenindex(["RECHNUNG", "AGB", "VERTRAG"], ["VERTRAG", "AGB", "RECHNUNG"])
    assert index.tolist() == [2, 1, 0]


def test_klassenindex_weist_eine_wahrheit_ohne_spalte_ab() -> None:
    """Waechter mit eigenem Test: Ohne ihn liefe die Klasse auf einen falschen Index oder
    stillschweigend mit."""
    with pytest.raises(ValueError, match="unbekannte Klassen \\['SONSTIGES'\\]"):
        klassenindex(["RECHNUNG", "SONSTIGES"], ["RECHNUNG", "GUTSCHRIFT"])
