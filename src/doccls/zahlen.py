"""Die Formprüfungen, die ``calibrate``, ``decide`` und ``evaluate`` gemeinsam brauchen.

**Warum dieses Modul existiert.** Die Prüfungen standen dreimal im Baum: ``Zahlenfeld`` war
in ``calibrate`` und ``decide`` zeichengleich doppelt definiert, ``decide`` führte mit
``_als_reihe`` / ``_als_maske`` / ``Zahlenreihe`` dieselbe Familie ein drittes Mal, und
``evaluate`` holte sich die modulprivaten ``_als_float64``, ``_pruefe_form`` und
``_klassenindex`` quer aus ``calibrate`` – ein Muster, das sonst kein Modul im Baum
verwendet. Zwei Kopien derselben Regel laufen irgendwann auseinander, und die zweite ist
dann die ungepflegte. Hier stehen sie einmal und öffentlich.

**Kein Vorrat, sondern eine Zusammenführung.** Jede Funktion hier hat heute mindestens
einen echten Aufrufer; eine Dublette ist dabei verschwunden. Was niemand ruft, gehört
nicht hierher.

Die Begründungen der einzelnen Prüfungen stehen bei ihnen – sie sind der Grund, warum es
sie gibt, und wandern mit ihnen mit.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Zahlenreihe = npt.NDArray[np.float32] | npt.NDArray[np.float64]
"""Ein Wert je Dokument – einfache oder doppelte Genauigkeit."""

Zahlenfeld = npt.NDArray[np.float32] | npt.NDArray[np.float64]
"""Eine Zeile je Dokument, eine Spalte je Klasse – einfache oder doppelte Genauigkeit."""


def als_float64(matrix: Zahlenfeld) -> npt.NDArray[np.float64]:
    """Zweidimensional und in doppelter Genauigkeit – die Optimierung rechnet nicht auf
    ``float32``, sonst hinge das Ergebnis an der Rundung der Eingabe."""
    werte = np.asarray(matrix, dtype=np.float64)
    if werte.ndim != 2:
        raise ValueError(
            f"Erwartet wird eine Matrix (Zeile je Dokument, Spalte je Klasse), "
            f"bekommen hat die Funktion {werte.ndim} Dimensionen."
        )
    return werte


def als_reihe(werte: Zahlenreihe, name: str) -> npt.NDArray[np.float64]:
    """Eindimensional, nicht leer, endlich – in doppelter Genauigkeit.

    ``NaN`` sortiert in NumPy ans Ende und wäre in einer nach Konfidenz sortierten Kurve
    unsichtbar: Die Schwelle verschöbe sich lautlos.
    """
    reihe = np.asarray(werte, dtype=np.float64)
    if reihe.ndim != 1:
        raise ValueError(
            f"{name} muss eindimensional sein (ein Wert je Dokument), hat aber "
            f"{reihe.ndim} Dimensionen."
        )
    if reihe.size == 0:
        raise ValueError(f"{name} ist leer - aus nichts laesst sich keine Schwelle ablesen.")
    if not np.isfinite(reihe).all():
        raise ValueError(
            f"{name} enthaelt Werte, die nicht endlich sind (NaN oder inf). Sie sortieren "
            "sich stillschweigend ans Ende und verschoeben die Schwelle."
        )
    return reihe


def als_maske(werte: npt.NDArray[np.bool_], name: str) -> npt.NDArray[np.bool_]:
    """Richtig oder falsch – nichts dazwischen.

    Ein Gleitkommafeld mit 0,5 darin ließe sich rechnen und ergäbe eine „Präzision", die
    keine ist. Deshalb wird der Typ verlangt statt umgedeutet.
    """
    maske = np.asarray(werte)
    if maske.dtype != np.bool_:
        raise ValueError(
            f"{name} muss ein boolesches Feld sein (richtig oder falsch je Dokument), hat "
            f"aber den Typ {maske.dtype}. Ein Zahlenfeld ergaebe eine Praezision, die "
            "keine ist."
        )
    if maske.ndim != 1:
        raise ValueError(
            f"{name} muss eindimensional sein (ein Wert je Dokument), hat aber "
            f"{maske.ndim} Dimensionen."
        )
    return maske


def pruefe_form(
    matrix: npt.NDArray[np.float64],
    y_true: Sequence[str] | npt.NDArray[np.str_],
    classes: Sequence[str],
) -> None:
    """Eine Zeile je Wahrheit, eine Spalte je Klasse.

    Ohne diese Prüfung liefe ein Auseinanderlaufen von Spalten und ``classes`` lautlos
    durch: Jede Konfidenz zeigte auf die falsche Klasse, und alle Zahlen blieben
    trotzdem plausibel – dieselbe Falle, gegen die
    ``classify._klassenreihenfolge_pruefen`` steht. Eine zu kurze ``y_true`` würde ohne
    sie stillschweigend abgeschnitten.
    """
    if matrix.shape[0] != len(y_true):
        raise ValueError(
            f"{matrix.shape[0]} Zeilen stehen {len(y_true)} bekannten Klassen gegenueber - "
            "die Zuordnung waere sonst stillschweigend abgeschnitten."
        )
    if matrix.shape[1] != len(classes):
        raise ValueError(
            f"{matrix.shape[1]} Spalten fuer {len(classes)} Klassen - jede Konfidenz "
            "zeigte sonst auf die falsche Klasse."
        )


def klassenindex(
    y_true: Sequence[str] | npt.NDArray[np.str_], classes: Sequence[str]
) -> npt.NDArray[np.intp]:
    """Die wahren Klassen als Spaltenindizes, in der Reihenfolge von ``classes``."""
    spalte = {klasse: index for index, klasse in enumerate(classes)}
    unbekannt = sorted({str(wert) for wert in y_true} - spalte.keys())
    if unbekannt:
        raise ValueError(
            f"Die Wahrheit enthaelt unbekannte Klassen {unbekannt!r}, die in classes "
            f"({list(classes)!r}) keine Spalte haben."
        )
    return np.array([spalte[str(wert)] for wert in y_true], dtype=np.intp)
