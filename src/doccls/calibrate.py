"""Kalibrierung: Temperature Scaling und Eichfehler (Konzept § 7.3, § 9.3).

Der Schritt, der meistens fehlt. Eine logistische Regression auf hochdimensionalen
Merkmalen ist systematisch überheblich: Sie sagt 0,97 und trifft in 0,84 der Fälle.
Temperature Scaling lernt **einen** Parameter ``T`` auf einer **separaten**
Kalibriermenge (``splits.calibration_documents``) durch Minimierung der Negative
Log-Likelihood und teilt jedes Logit durch ihn.

**Die tragende Eigenschaft:** ``T`` verändert keine einzige Entscheidung. Eine Division
durch eine positive Zahl ist streng monoton, die Rangfolge der Klassen bleibt Zeile für
Zeile dieselbe. Kalibrierung kann die Genauigkeit also weder verbessern noch
verschlechtern – sie repariert ausschließlich die Zahl. Wer statt zu skalieren
*verschöbe*, bräche nichts an den Entscheidungen (eine Verschiebung um einen Skalar ist
im Softmax spurlos), aber alles an der Wirkung: Die Wahrscheinlichkeiten blieben
unverändert überheblich. Deshalb prüft die Testsuite beides getrennt – die Rangfolge und
die gemessene Senkung des Eichfehlers.

**Numerik.** Die NLL wird über ``scipy.special.log_softmax`` auf ``logits / T``
gerechnet, nicht über ``log(softmax(...))``: Letzteres läuft bei großen Logits und
kleinem ``T`` in einen Unterlauf und liefert ``-inf``, wo ein endlicher, sehr negativer
Wert stünde – die Suche bekäme dann ein Plateau statt eines Gefälles zu sehen.

**Bewusst nicht gebaut** (Konzept § 7.3 nennt sie als Alternativen): Vector Scaling und
isotone Regression. Der Bedarf ist eine Messfrage – erst wenn Aufgabe 16 den ECE über
dem Ziel von 0,05 findet, lohnt die zweite Umsetzung. Isotone Regression braucht laut
Konzept ohnehin rund 1000 Kalibrierbeispiele; vorhanden sind 70.

**Kein Gold-Schutz in diesem Modul.** Die Funktionen hier sehen Zahlen, keine Dokumente –
sie haben keinen ``split``, den sie prüfen könnten. Der Schutz aus Konzept § 9.2 sitzt
eine Ebene höher, in ``splits.calibration_documents`` (``assert_no_gold``), und ist dort
geprüft.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

#: Suchbereich für ``T``. Die Untergrenze ist die praktische Grenze der Aussagekraft:
#: Unter 0,05 unterscheidet sich das Ergebnis nicht mehr von einer harten Entscheidung.
#: Die Untergrenze liegt ausdrücklich **unter** 1,0 – ein Modell kann auch zu zaghaft
#: sein, und eine Suche, die erst bei 1,0 beginnt, meldete dann fälschlich "gut geeicht".
T_UNTERGRENZE = 0.05
T_OBERGRENZE = 20.0

Zahlenfeld = npt.NDArray[np.float32] | npt.NDArray[np.float64]


def _als_float64(matrix: Zahlenfeld) -> npt.NDArray[np.float64]:
    """Zweidimensional und in doppelter Genauigkeit – die Optimierung rechnet nicht auf
    ``float32``, sonst hinge das Ergebnis an der Rundung der Eingabe."""
    werte = np.asarray(matrix, dtype=np.float64)
    if werte.ndim != 2:
        raise ValueError(
            f"Erwartet wird eine Matrix (Zeile je Dokument, Spalte je Klasse), "
            f"bekommen hat die Funktion {werte.ndim} Dimensionen."
        )
    return werte


def _pruefe_form(
    matrix: npt.NDArray[np.float64],
    y_true: Sequence[str] | npt.NDArray[np.str_],
    classes: Sequence[str],
) -> None:
    """Eine Zeile je Wahrheit, eine Spalte je Klasse.

    Ohne diese Prüfung liefe ein Auseinanderlaufen von Spalten und ``classes`` lautlos
    durch: Jede Konfidenz zeigte auf die falsche Klasse, und alle Zahlen blieben
    trotzdem plausibel – dieselbe Falle, gegen die ``classify._klassen_pruefen`` steht.
    Eine zu kurze ``y_true`` würde ohne sie stillschweigend abgeschnitten.
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


def _klassenindex(
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


def fit_temperature(
    logits: Zahlenfeld,
    y_true: Sequence[str] | npt.NDArray[np.str_],
    classes: Sequence[str],
) -> float:
    """Die Temperatur ``T`` auf einer Kalibriermenge anpassen (Konzept § 7.3).

    Minimiert die mittlere Negative Log-Likelihood von ``logits / T`` über ``T`` im
    Bereich ``(T_UNTERGRENZE, T_OBERGRENZE)``. Genau **ein** Parameter, angepasst auf
    einer Menge, die das Modell nie im Training gesehen hat – auf den Trainingslogits
    angepasst ergäbe das Verfahren ein ``T`` nahe 1 und täuschte gute Eichung vor.

    ``logits`` sind Rohwerte je Klasse (``TrainedModel.decision_scores``), keine
    Wahrscheinlichkeiten.
    """
    werte = _als_float64(logits)
    _pruefe_form(werte, y_true, classes)
    ziel = _klassenindex(y_true, classes)
    zeilen = np.arange(werte.shape[0])

    def negative_log_likelihood(temperatur: float) -> float:
        log_wahrscheinlichkeit = log_softmax(werte / temperatur, axis=1)
        return float(-log_wahrscheinlichkeit[zeilen, ziel].mean())

    ergebnis = minimize_scalar(
        negative_log_likelihood,
        bounds=(T_UNTERGRENZE, T_OBERGRENZE),
        method="bounded",
    )
    return float(ergebnis.x)


def apply_temperature(logits: Zahlenfeld, temperature: float) -> npt.NDArray[np.float32]:
    """``logits / T`` in Wahrscheinlichkeiten umrechnen – geeichte Konfidenz je Klasse.

    Die Rangfolge je Zeile bleibt dabei unverändert (Moduldoc). Eine Temperatur ``<= 0``
    wird abgelehnt statt still gerechnet: Sie ist der einzige Weg, auf dem diese
    Funktion eine Entscheidung verändern könnte, denn ein negatives ``T`` kehrt die
    Rangfolge um und macht die sicherste Klasse zur unsichersten.
    """
    if not temperature > 0.0:
        raise ValueError(
            f"Temperatur {temperature!r} ist nicht groesser als 0. Ein negatives T kehrt "
            "die Rangfolge der Klassen um, ein T von 0 teilt durch null."
        )
    werte = _als_float64(logits)
    ergebnis: npt.NDArray[np.float32] = np.asarray(
        softmax(werte / temperature, axis=1), dtype=np.float32
    )
    return ergebnis


def expected_calibration_error(
    proba: Zahlenfeld,
    y_true: Sequence[str] | npt.NDArray[np.str_],
    classes: Sequence[str],
    bins: int = 10,
) -> float:
    """Erwarteter Eichfehler (Konzept § 9.3).

    Die Vorhersagen werden nach ihrer Konfidenz (der höchsten Wahrscheinlichkeit der
    Zeile) in ``bins`` gleich breite Körbe über ``[0, 1]`` einsortiert. Je Korb der
    Abstand ``|mittlere Konfidenz − Trefferquote|``, gewichtet nach Korbbesetzung: Ein
    Korb mit 200 Dokumenten wiegt hundertmal so schwer wie einer mit zweien. Leere Körbe
    zählen nicht mit – sonst hinge der Wert an der Korbzahl statt an der Eichung, und
    ein Modell sähe allein dadurch besser aus, dass man ``bins`` erhöht.

    0,0 heißt: Wo das Modell 0,8 sagt, trifft es in 0,8 der Fälle. Nahe 1,0 heißt: volle
    Sicherheit, durchgehend falsch.
    """
    if bins < 1:
        raise ValueError(
            f"{bins} Koerbe sind keine Einteilung - die Summe liefe ueber nichts und "
            "gaebe stillschweigend 0,0 zurueck."
        )
    werte = _als_float64(proba)
    _pruefe_form(werte, y_true, classes)
    zeilensummen = werte.sum(axis=1)
    if not np.allclose(zeilensummen, 1.0, atol=1e-4):
        raise ValueError(
            "Die Zeilensumme muss 1 sein - erwartet werden Wahrscheinlichkeiten, keine "
            f"Logits (gefunden: Summen von {zeilensummen.min():.4f} bis "
            f"{zeilensummen.max():.4f})."
        )
    ziel = _klassenindex(y_true, classes)

    konfidenz = np.clip(werte.max(axis=1), 0.0, 1.0)
    treffer = (werte.argmax(axis=1) == ziel).astype(np.float64)
    # Der oberste Korb ist rechts geschlossen, damit eine Konfidenz von exakt 1,0 nicht
    # in einen elften Korb faellt.
    korb = np.minimum((konfidenz * bins).astype(np.intp), bins - 1)

    fehler = 0.0
    for index in range(bins):
        maske = korb == index
        besetzung = int(maske.sum())
        if besetzung == 0:
            continue
        luecke = abs(float(konfidenz[maske].mean()) - float(treffer[maske].mean()))
        fehler += besetzung / len(konfidenz) * luecke
    return fehler
