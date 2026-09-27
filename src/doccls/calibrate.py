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
Wert stünde. Mehr als das leistet ``log_softmax`` nicht. Ein Plateau verhindert es
**nicht** – nachgemessen auf trennbaren Logits ergibt die NLL ``-0,000e+00`` bei
``T = 0,05``, ``0,1`` und ``0,15`` und erst ab ``0,2`` wieder einen von null
unterscheidbaren Wert (``2,09e-16``). Wo die Kalibriermenge kein ``T`` hergibt, bleibt
die Zielfunktion flach, ganz gleich wie stabil sie gerechnet ist; abgefangen wird das
nicht hier, sondern vom Randwächter in :func:`fit_temperature`.

**Bewusst nicht gebaut** (Konzept § 7.3 nennt sie als Alternativen): Vector Scaling und
isotone Regression. Der Bedarf ist eine Messfrage – erst wenn Aufgabe 16 den ECE über
dem Ziel von 0,05 findet, lohnt die zweite Umsetzung. Isotone Regression braucht laut
Konzept ohnehin rund 1000 Kalibrierbeispiele; vorhanden sind 70.

**Die Körbe der Eichung stehen hier, nicht bei den Bildern.** :func:`reliability_bins` ist
die Rechnung, aus der :func:`expected_calibration_error` seine Zahl bildet; das Reliability
Diagram in ``doccls.reports`` zeichnet dieselben Körbe, weil es dieselbe Funktion ruft.
Zwei gleich gemeinte Einteilungen liefen irgendwann auseinander, und dann zeigte das Bild
etwas anderes als der ECE darunter.

**Kein Gold-Schutz in diesem Modul.** Die Funktionen hier sehen Zahlen, keine Dokumente –
sie haben keinen ``split``, den sie prüfen könnten. Der Schutz aus Konzept § 9.2 sitzt
eine Ebene höher, in ``splits.calibration_documents`` (``assert_no_gold``), und ist dort
geprüft.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from doccls.zahlen import Zahlenfeld, als_float64, klassenindex, pruefe_form

#: Suchbereich für ``T``. Die Untergrenze ist die praktische Grenze der Aussagekraft:
#: Unter 0,05 unterscheidet sich das Ergebnis nicht mehr von einer harten Entscheidung.
#: Die Untergrenze liegt ausdrücklich **unter** 1,0 – ein Modell kann auch zu zaghaft
#: sein, und eine Suche, die erst bei 1,0 beginnt, meldete dann fälschlich "gut geeicht".
T_UNTERGRENZE = 0.05
T_OBERGRENZE = 20.0

#: Wie nah an einer Schranke ein Suchergebnis als Randtreffer gilt – relativ, damit
#: dieselbe Zahl für beide Schranken taugt. Die beschränkte Suche läuft nie exakt auf die
#: Schranke, sondern bis auf ihre eigene Toleranz heran (gemessen: 0,050007 statt 0,05),
#: ein Vergleich auf Gleichheit ginge also immer ins Leere.
RAND_TOLERANZ = 0.01


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

    **Ein Randtreffer wird geworfen, nicht zurückgegeben.** Liegt das gefundene ``T`` auf
    einer der beiden Schranken, hat die Suche kein Minimum gefunden, sondern ist gegen den
    Rand ihres Bereichs gelaufen: Das Minimum liegt außerhalb. Auf dem echten Bestand ist
    das aufgetreten (centroid: ``T = 0,050007`` bei einer NLL, die bis unter die Schranke
    monoton fällt). Eine Temperatur vom Rand ist keine Eichung, sondern die Aussage, dass
    diese Kalibriermenge keine hergibt – weil die Scores perfekt trennbar oder entartet
    sind. Diese Aussage muss nach oben durchschlagen. Als ``float`` zurückgegeben wäre sie
    von einer echten Anpassung nicht mehr zu unterscheiden: Sie ist eine Zahl im erlaubten
    Bereich und jeder Aufrufer – ``apply_temperature``, die Auswertung in Aufgabe 16 –
    rechnete stillschweigend mit ihr weiter. Deshalb ``ValueError`` und keine Warnung; eine
    Warnung kann überhört werden, ein Rückgabewert kann nicht befragt werden.
    """
    werte = als_float64(logits)
    pruefe_form(werte, y_true, classes)
    ziel = klassenindex(y_true, classes)
    zeilen = np.arange(werte.shape[0])

    def negative_log_likelihood(temperatur: float) -> float:
        log_wahrscheinlichkeit = log_softmax(werte / temperatur, axis=1)
        return float(-log_wahrscheinlichkeit[zeilen, ziel].mean())

    ergebnis = minimize_scalar(
        negative_log_likelihood,
        bounds=(T_UNTERGRENZE, T_OBERGRENZE),
        method="bounded",
    )
    temperatur = float(ergebnis.x)
    if temperatur <= T_UNTERGRENZE * (1.0 + RAND_TOLERANZ):
        raise ValueError(
            f"Die Suche ist in die Untergrenze {T_UNTERGRENZE} gelaufen (gefunden: "
            f"{temperatur:.6f}). Das Minimum der NLL liegt darunter, also ausserhalb des "
            "Suchbereichs: Die Scores sind perfekt trennbar oder entartet, und diese "
            "Kalibriermenge bestimmt kein T. Ein Randwert waere keine Eichung."
        )
    if temperatur >= T_OBERGRENZE * (1.0 - RAND_TOLERANZ):
        raise ValueError(
            f"Die Suche ist in die Obergrenze {T_OBERGRENZE} gelaufen (gefunden: "
            f"{temperatur:.6f}). Das Minimum der NLL liegt darueber, also ausserhalb des "
            "Suchbereichs: Die Scores sind entartet oder masslos ueberheblich, und diese "
            "Kalibriermenge bestimmt kein T. Ein Randwert waere keine Eichung."
        )
    return temperatur


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
    werte = als_float64(logits)
    ergebnis: npt.NDArray[np.float32] = np.asarray(
        softmax(werte / temperature, axis=1), dtype=np.float32
    )
    return ergebnis


@dataclass(frozen=True)
class ReliabilityBin:
    """Ein **besetzter** Korb der Eichung: die Zeile eines Reliability Diagrams.

    ``untergrenze`` und ``obergrenze`` sind die Ränder des Korbs, ``mittlere_konfidenz``
    und ``trefferquote`` sein Punkt im Diagramm, ``anzahl`` sein Gewicht im Eichfehler.
    Ein gut geeichtes Modell legt ``mittlere_konfidenz ≈ trefferquote``, also die
    Diagonale; liegt der Punkt darunter, ist das Modell überheblich.
    """

    untergrenze: float
    obergrenze: float
    mittlere_konfidenz: float
    trefferquote: float
    anzahl: int


def reliability_bins(
    proba: Zahlenfeld,
    y_true: Sequence[str] | npt.NDArray[np.str_],
    classes: Sequence[str],
    bins: int = 10,
) -> list[ReliabilityBin]:
    """Die Körbe hinter dem Eichfehler – und hinter dem Reliability Diagram.

    Die Vorhersagen werden nach ihrer Konfidenz (der höchsten Wahrscheinlichkeit der
    Zeile) in ``bins`` gleich breite Körbe über ``[0, 1]`` einsortiert. Der oberste Korb
    ist rechts geschlossen, damit eine Konfidenz von exakt 1,0 nicht in einen zusätzlichen
    elften fällt.

    **Zurückgegeben werden nur besetzte Körbe**, aufsteigend nach Konfidenz. Ein leerer
    Korb hat keine mittlere Konfidenz und keine Trefferquote; als ``(0, 0)`` in die Kurve
    gezeichnet zöge er sie nach unten und ließe ein gut geeichtes Modell schlecht
    aussehen, und im Eichfehler mitgezählt hinge der Wert an der Korbzahl statt an der
    Eichung – ein Modell sähe allein dadurch besser aus, dass man ``bins`` erhöht.

    **Warum diese Funktion hier steht und nicht in ``reports``.** Das Bild und die Zahl
    darunter müssen dieselbe Einteilung benutzen; täten sie es nicht, zeigte das Diagramm
    etwas anderes als der ECE daneben. Sichergestellt wird das nicht durch eine zweite,
    gleich gemeinte Rechnung, sondern dadurch, dass es nur **eine** gibt:
    :func:`expected_calibration_error` bildet seine Zahl aus genau diesen Körben, und
    ``doccls.reports`` reicht die Funktion unter dem Namen der Aufgabe 15 weiter, statt sie
    nachzubauen. ``doccls.reports`` könnte sie auch nicht hierher zurückgeben – es hängt
    über ``doccls.evaluate`` an diesem Modul.

    Werte außerhalb ``[0, 1]`` werden **abgelehnt, nicht gekappt**. Ein ``np.clip`` stand
    hier und verbuchte eine Konfidenz von 1,5 stillschweigend als 1,0: ``[[1.5, -0.5]]``
    hat Zeilensumme 1, kam am Wächter darüber vorbei und ergab den Eichfehler 0,0 –
    perfekte Eichung, gemeldet für eine Eingabe, die keine Wahrscheinlichkeiten enthielt.
    Kappen macht eine falsche Eingabe rechenbar, statt sie zu melden.
    """
    if bins < 1:
        raise ValueError(
            f"{bins} Koerbe sind keine Einteilung - die Summe liefe ueber nichts und "
            "gaebe stillschweigend 0,0 zurueck."
        )
    werte = als_float64(proba)
    pruefe_form(werte, y_true, classes)
    if werte.shape[0] == 0:
        raise ValueError(
            "Keine Dokumente uebergeben - ueber die leere Menge ist keine Eichung "
            "messbar, und jeder zurueckgegebene Korb waere erfunden."
        )
    zeilensummen = werte.sum(axis=1)
    if not np.allclose(zeilensummen, 1.0, atol=1e-4):
        raise ValueError(
            "Die Zeilensumme muss 1 sein - erwartet werden Wahrscheinlichkeiten, keine "
            f"Logits (gefunden: Summen von {zeilensummen.min():.4f} bis "
            f"{zeilensummen.max():.4f})."
        )
    if werte.min() < 0.0 or werte.max() > 1.0:
        raise ValueError(
            "Jeder Wert muss zwischen 0 und 1 liegen - erwartet werden "
            f"Wahrscheinlichkeiten (gefunden: {werte.min():.4f} bis {werte.max():.4f})."
        )
    ziel = klassenindex(y_true, classes)

    konfidenz = werte.max(axis=1)
    treffer = (werte.argmax(axis=1) == ziel).astype(np.float64)
    korb = np.minimum((konfidenz * bins).astype(np.intp), bins - 1)

    koerbe: list[ReliabilityBin] = []
    for index in range(bins):
        maske = korb == index
        besetzung = int(maske.sum())
        if besetzung == 0:
            continue
        koerbe.append(
            ReliabilityBin(
                untergrenze=index / bins,
                obergrenze=(index + 1) / bins,
                mittlere_konfidenz=float(konfidenz[maske].mean()),
                trefferquote=float(treffer[maske].mean()),
                anzahl=besetzung,
            )
        )
    return koerbe


def expected_calibration_error(
    proba: Zahlenfeld,
    y_true: Sequence[str] | npt.NDArray[np.str_],
    classes: Sequence[str],
    bins: int = 10,
) -> float:
    """Erwarteter Eichfehler (Konzept § 9.3).

    Je Korb aus :func:`reliability_bins` der Abstand
    ``|mittlere Konfidenz − Trefferquote|``, gewichtet nach Korbbesetzung: Ein Korb mit
    200 Dokumenten wiegt hundertmal so schwer wie einer mit zweien. Die Einteilung, die
    Randfälle und die Eingabeprüfungen stehen dort und werden von hier aus nicht
    wiederholt – das Reliability Diagram zeichnet dieselben Körbe.

    0,0 heißt: Wo das Modell 0,8 sagt, trifft es in 0,8 der Fälle. Nahe 1,0 heißt: volle
    Sicherheit, durchgehend falsch.
    """
    koerbe = reliability_bins(proba, y_true, classes, bins)
    gesamt = sum(korb.anzahl for korb in koerbe)
    fehler = 0.0
    for korb in koerbe:
        luecke = abs(korb.mittlere_konfidenz - korb.trefferquote)
        fehler += korb.anzahl / gesamt * luecke
    return fehler
