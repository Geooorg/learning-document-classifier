"""Kennzahlen: was das Verfahren taugt (Konzept § 9.3).

Die Hauptzahl ist **Macro-F1**, nicht Accuracy. Accuracy belohnt die häufigste Klasse:
Ein Modell, das die seltene Klasse nie trifft, steht bei 0,95 und ist trotzdem unbrauchbar.
Macro-F1 mittelt über die Klassen statt über die Dokumente und bestraft genau das.

Die zweite tragende Zahl ist **Coverage@P98** – der Anteil der Dokumente, die bei
mindestens 98 % Präzision automatisch entschieden werden können. Sie koppelt Konfidenz und
Korrektheit und lässt sich, anders als die mittlere Konfidenz, nicht durch
Temperaturspielerei schönen: ``T = 0,3`` hebt jede Konfidenz auf 0,99 (Konzept § 9.1),
verschiebt aber die Schwelle mit, weil die Rangfolge der Dokumente dieselbe bleibt.
Dasselbe gilt für ``auroc_confidence`` und ``aurc`` – alle drei sind Rangmaße. Der ``ece``
ist es ausdrücklich **nicht**; er misst die Zahlenwerte und muss sich unter Temperatur
bewegen. Beides ist in ``tests/test_evaluate.py`` gemessen, nicht geglaubt.

**Die Konfusionsmatrix trägt hier mehr als jede Einzelzahl** (:func:`confusion`).
Rechnung↔Gutschrift und Vertrag↔AGB sind völlig verschiedene Probleme, und in welche
Richtung verwechselt wird, steht in keiner gemittelten Kennzahl.

**Kein Gold-Schutz in diesem Modul.** Wie in ``calibrate`` sehen die Funktionen hier Zahlen
und keine Dokumente – sie haben keinen ``split``, den sie prüfen könnten. Ausgewertet wird
gerade auf dem Gold-Set; der Schutz aus Konzept § 9.2 sitzt in ``splits`` und betrifft die
Funktionen, die *trainieren* oder *kalibrieren*.

**Wiederverwendet statt nachgebaut.** Der Eichfehler kommt aus
:func:`doccls.calibrate.expected_calibration_error`, die Risiko-Abdeckungs-Kurve aus
:func:`doccls.decide.risk_coverage`, und die Formprüfungen aus ``calibrate``. Die
Prüfungen dort und hier sind dieselbe Regel; zwei Kopien liefen irgendwann auseinander,
und die zweite wäre die ungepflegte. Deshalb werden die modulprivaten Helfer von
``calibrate`` innerhalb des Pakets importiert, statt sie abzuschreiben.
"""

import math
from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, Field, model_validator
from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix, f1_score

from doccls.calibrate import (
    Zahlenfeld,
    _als_float64,
    _klassenindex,
    _pruefe_form,
    expected_calibration_error,
)
from doccls.decide import risk_coverage

WAHRSCHEINLICHKEITS_UNTERGRENZE = 1e-12
"""Woran ``nll`` eine Wahrscheinlichkeit von 0 abfängt.

``−log 0`` ist ``inf`` und fräße sich durch jede Mittelung; eine einzige Zeile mit
``p̂ = 0`` für die wahre Klasse machte die Kennzahl für den ganzen Lauf unbrauchbar. Nach
der Kalibrierung ist eine exakte 0 selten, aber nicht unmöglich – ``float32``-Softmax
liefert sie bei großen Logit-Abständen. Gekappt wird **nur hier**, wo der Wert sonst
unendlich würde, nicht bei den übrigen Kennzahlen: Dort wäre Kappen das Verbergen einer
falschen Eingabe (siehe ``calibrate.expected_calibration_error``).
"""

NLL_OBERGRENZE = -math.log(WAHRSCHEINLICHKEITS_UNTERGRENZE)
"""Die größte NLL, die aus der Kappung folgen kann – rund 27,63."""

KENNZAHL_TOLERANZ = 1e-9
"""Spielraum der Kreuzprüfungen in :class:`Metrics`.

Gerechnet wird durchgehend in ``float64``; der Mittelwert über höchstens eine Handvoll
Klassen hat einen Fehler um ``1e-16``. ``1e-9`` lässt das durch und fängt trotzdem jede
Verwechslung, die eine Rolle spielte: Der kleinste Unterschied, um den sich Macro-F1 und
das Mittel seiner Klassenwerte inhaltlich unterscheiden können, liegt bei ``1 / n``.
"""

MINDESTZIEHUNGEN = 40
"""Wie viele Bootstrap-Ziehungen es mindestens braucht.

Das 2,5- und das 97,5-Perzentil verlangen, dass im Mittel wenigstens ein Wert außerhalb
liegt: ``rounds · 0,025 >= 1``, also 40. Darunter sitzen beide Grenzen auf dem kleinsten
und größten gezogenen Wert, und das Intervall misst die Zahl der Ziehungen statt die
Streuung der Kennzahl – dieselbe Überlegung wie in ``decide._mindestzahl_fuer_perzentil``.
"""

UNTERES_PERZENTIL = 2.5
OBERES_PERZENTIL = 97.5

WAHR_SPALTE = "true_class"
"""Name der Zeilenbeschriftung in :func:`confusion`. Eine Klasse dieses Namens wird
abgewiesen, weil ihre Spalte die Beschriftung überschriebe."""


class Metrics(BaseModel):
    """Die Kennzahlen eines Auswertungslaufs (Konzept § 9.3).

    Wie :class:`doccls.models.Prediction` hält das Modell nicht nur Felder, sondern prüft
    die Beziehungen zwischen ihnen. Der Grund ist derselbe: Die Zahlen stammen aus *einer*
    Auswertung und können einander nicht widersprechen. Tun sie es doch, kommen sie aus
    verschiedenen Quellen.

    * ``macro_f1`` ist das ungewichtete Mittel von ``per_class_f1``. Diese Prüfung bindet
      die Hauptzahl an ihre Bestandteile: Ein versehentlich gewichtetes Mittel (``average=
      "weighted"``) oder eine an ``accuracy`` verrutschte Zuweisung fällt hier auf.
    * ``accuracy · n`` ist ganzzahlig, denn Accuracy ist ein Anteil von ``n`` Dokumenten.
    * ``coverage_at_precision · n`` ist ganzzahlig – die Abdeckung ist der Anteil der
      Dokumente oberhalb einer Schwelle, und Schwellen liegen auf beobachteten Konfidenzen.

    **Zu den einzelnen Feldern, und was sie ausdrücklich nicht sind:**

    ``brier`` ist der mittlere quadratische Fehler **über alle Zellen** der
    One-Hot-Wahrheit, ``mean((p̂ − y)²)`` über Dokumente *und* Klassen, und liegt damit in
    ``[0, 1]``. Die in der Literatur ebenso gebräuchliche Form summiert über die Klassen
    und mittelt nur über die Dokumente; sie liegt in ``[0, 2]`` und ist bei fester
    Klassenzahl genau das ``k``-fache dieser hier. Jede Reihenfolge zweier Modelle ist
    deshalb dieselbe, der Zahlenwert aber nicht – wer Werte über Läufe mit verschiedener
    Klassenzahl vergleicht, muss das wissen. Der Aufgabentext verlangt den *mittleren*
    quadratischen Fehler, und dabei bleibt es.

    ``nll`` ist die mittlere negative Log-Likelihood der wahren Klasse, natürlicher
    Logarithmus, mit der Kappung aus :data:`WAHRSCHEINLICHKEITS_UNTERGRENZE`.

    ``coverage_at_precision`` ist **keine** Schwelle, sondern ein Anteil – deshalb gibt es
    hier, anders als bei ``decide.tau_for_precision``, keine Mindestbelegung und keine
    Ausnahme. Eine Abdeckung von 0,003 sagt selbst, dass sie auf drei von tausend
    Dokumenten beruht; ein τ von 0,997 sähe dagegen aus wie eine Schwelle und verschwiege
    es. Hält keine Schwelle das Ziel, ist die Abdeckung 0,0: ein gültiges, schlechtes
    Ergebnis.

    ``auroc_confidence`` ist ``None``, wenn **alle** Vorhersagen richtig oder **alle**
    falsch sind. Die AUROC ist die Wahrscheinlichkeit, dass eine zufällig gezogene richtige
    Vorhersage höher konfident ist als eine zufällig gezogene falsche; gibt es keine der
    beiden Sorten, ist die Menge dieser Paare leer und die Zahl nicht bestimmt. 0,5
    einzusetzen hieße „keine Trennschärfe gemessen", und das ist eine andere Aussage als
    „nicht messbar" – auf dem echten Gold-Set (Macro-F1 1,000 auf allen drei Modellen) wäre
    genau dieser Fall der Normalfall, und eine 0,5 ginge ungeprüft in jeden Bericht.
    ``None`` kann der Aufrufer befragen, eine Zahl nicht.
    """

    model_config = {"frozen": True}

    macro_f1: float = Field(ge=0.0, le=1.0)
    """Ungewichtetes Mittel der Klassen-F1 – die Hauptzahl (Konzept § 9.3)."""

    per_class_f1: dict[str, float] = Field(min_length=1)
    """F1 je Klasse, in der Reihenfolge von ``classes``. Steht neben ``macro_f1``, weil das
    Mittel verschweigt, *welche* Klasse durchfällt."""

    accuracy: float = Field(ge=0.0, le=1.0)
    brier: float = Field(ge=0.0, le=1.0)
    ece: float = Field(ge=0.0, le=1.0)
    nll: float = Field(ge=0.0, le=NLL_OBERGRENZE)
    coverage_at_precision: float = Field(ge=0.0, le=1.0)
    aurc: float = Field(ge=0.0, le=1.0)
    """Fläche unter der Risiko-Abdeckungs-Kurve, Trapezregel. **Kleiner ist besser.**"""

    auroc_confidence: float | None = Field(ge=0.0, le=1.0)
    n: int = Field(ge=1)

    @model_validator(mode="after")
    def _pruefe_zusammenhaenge(self) -> Self:
        """Die drei Kreuzprüfungen aus dem Klassendoc."""
        mittel = sum(self.per_class_f1.values()) / len(self.per_class_f1)
        if abs(self.macro_f1 - mittel) > KENNZAHL_TOLERANZ:
            raise ValueError(
                f"macro_f1={self.macro_f1} ist nicht das ungewichtete Mittel der "
                f"Klassenwerte ({mittel}). Macro-F1 mittelt ueber die Klassen, nicht ueber "
                "die Dokumente - hier stammen die beiden Zahlen aus verschiedenen "
                "Rechnungen."
            )
        for name, wert in (
            ("accuracy", self.accuracy),
            ("coverage_at_precision", self.coverage_at_precision),
        ):
            anzahl = wert * self.n
            if abs(anzahl - round(anzahl)) > KENNZAHL_TOLERANZ * self.n:
                raise ValueError(
                    f"{name}={wert} ist bei n={self.n} Dokumenten kein Anteil von ihnen: "
                    f"{name} * n = {anzahl} ist nicht ganzzahlig. Ein Anteil von n "
                    "Dokumenten ist immer ein Vielfaches von 1/n."
                )
        return self


def _klassen_pruefen(classes: Sequence[str]) -> list[str]:
    """Mindestens zwei Klassen, keine doppelt.

    Doppelte Namen kämen sonst lautlos durch: ``_klassenindex`` baut eine Abbildung
    ``Name → Spalte`` und behielte von zwei gleichen Namen nur den letzten. Jede Wahrheit
    dieser Klasse zeigte dann auf die falsche Spalte, und ``per_class_f1`` hätte einen
    Eintrag weniger als ``classes`` – das Mittel liefe über eine andere Zahl von Klassen.
    """
    namen = [str(klasse) for klasse in classes]
    if len(namen) < 2:
        raise ValueError(
            f"classes nennt {len(namen)} Klasse(n). Eine Klassifikation ueber weniger als "
            "zwei Klassen ist keine - jede Kennzahl waere per Bauart 1,0."
        )
    mehrfach = sorted({name for name in namen if namen.count(name) > 1})
    if mehrfach:
        raise ValueError(
            f"classes enthaelt {mehrfach!r} mehrfach. Zwei Spalten mit demselben Namen "
            "sind nicht unterscheidbar; die Wahrheit zeigte auf die falsche."
        )
    return namen


def evaluate(
    proba: Zahlenfeld,
    y_true: Sequence[str] | npt.NDArray[np.str_],
    classes: Sequence[str],
    target_precision: float = 0.98,
) -> Metrics:
    """Alle Kennzahlen eines Laufs auf einmal (Konzept § 9.3).

    ``proba`` ist die **kalibrierte** Wahrscheinlichkeitsmatrix (eine Zeile je Dokument,
    eine Spalte je Klasse) in der Spaltenreihenfolge von ``classes`` – wie überall in
    dieser Phase ``TrainedModel.classes_`` und nichts anderes. Die Zeilensummen müssen 1
    sein; geprüft wird das von ``expected_calibration_error``, weil dort schon steht,
    warum Rohwerte hier nichts zu suchen haben.

    ``target_precision`` ist das Präzisionsziel für ``coverage_at_precision``; der Startwert
    0,98 stammt aus Konzept § 7.4.

    **Vorhergesagt wird ``argmax`` je Zeile**, und der Name kommt aus ``classes`` an
    derselben Stelle. Das ist der lautloseste Fehler dieser Phase – Zahl aus der einen,
    Name aus einer anders sortierten Liste –, deshalb steht die Zuordnung an genau einer
    Stelle.

    **Wann eine Klasse keine F1 hat.** ``F1 = 2·TP / (2·TP + FP + FN)`` ist genau dann
    undefiniert, wenn eine Klasse weder in der Wahrheit noch unter den Vorhersagen
    vorkommt: Dann ist auch der Nenner 0. Dieser Fall wird **geworfen**, denn er sagt
    nichts über das Modell, sondern über die Klassenliste – eine Klasse, die in dieser
    Auswertung nicht vorkommt, zöge das Mittel mit einer 0 nach unten, die keine Messung
    ist. Das ist der Fall, in dem ``sklearn`` stillschweigend ``zero_division`` anwendet.

    Kommt eine Klasse dagegen in der Wahrheit vor und wird nur nie vorhergesagt, ist ihre
    F1 **0,0 und kein Sonderfall**: Die Trefferquote ist 0, und ``2PR / (P + R)`` geht
    gegen 0, wie hoch die Präzision auch wäre. ``zero_division=0.0`` wird deshalb
    ausdrücklich gesetzt und nicht der Voreinstellung überlassen. Genau dieser Fall ist der
    Grund, warum Macro-F1 die Hauptzahl ist.
    """
    namen = _klassen_pruefen(classes)
    werte = _als_float64(proba)
    _pruefe_form(werte, y_true, namen)
    if werte.shape[0] == 0:
        raise ValueError(
            "Keine Dokumente uebergeben. Ueber die leere Menge ist keine Kennzahl "
            "gebildet, und jede zurueckgegebene Zahl waere erfunden."
        )
    if not 0.0 < target_precision < 1.0:
        raise ValueError(
            f"target_precision={target_precision!r} liegt nicht echt zwischen 0 und 1. Ein "
            "Ziel von 1,0 ist auf endlich vielen Dokumenten nicht ablesbar, ein Ziel von 0 "
            "ist keine Anforderung."
        )

    ziel = _klassenindex(y_true, namen)
    vorhergesagt = werte.argmax(axis=1)
    vorhanden = set(ziel.tolist()) | set(vorhergesagt.tolist())
    fehlend = [name for index, name in enumerate(namen) if index not in vorhanden]
    if fehlend:
        raise ValueError(
            f"Die Klassen {fehlend!r} kommen weder in der Wahrheit noch unter den "
            "Vorhersagen vor. Ihre F1 ist 2*TP / (2*TP + FP + FN) = 0/0 und damit "
            "undefiniert; als 0,0 gemittelt zoege sie das Macro-F1 nach unten, ohne dass "
            "etwas gemessen waere. Auszuwerten ist gegen die Klassen, die vorkommen."
        )

    ece = expected_calibration_error(werte, y_true, namen)

    wahrheit = [namen[index] for index in ziel]
    vorhersage = [namen[index] for index in vorhergesagt]
    # zero_division ausdruecklich gesetzt, siehe Docstring: 0,0 ist hier der Grenzwert
    # von 2PR / (P + R) bei R = 0, keine Konvention.
    klassenwerte = f1_score(wahrheit, vorhersage, labels=namen, average=None, zero_division=0.0)
    per_class_f1 = {name: float(wert) for name, wert in zip(namen, klassenwerte, strict=True)}

    richtig: npt.NDArray[np.bool_] = vorhergesagt == ziel
    konfidenz = werte.max(axis=1)
    zeilen = np.arange(werte.shape[0])

    one_hot = np.zeros_like(werte)
    one_hot[zeilen, ziel] = 1.0
    brier = float(np.mean((werte - one_hot) ** 2))
    gekappt = np.clip(werte[zeilen, ziel], WAHRSCHEINLICHKEITS_UNTERGRENZE, 1.0)
    nll = float(-np.log(gekappt).mean())

    abdeckung, praezision = risk_coverage(konfidenz, richtig)
    haelt = praezision >= target_precision
    coverage = float(abdeckung[haelt].max()) if bool(haelt.any()) else 0.0
    aurc = float(np.trapezoid(1.0 - praezision, abdeckung))

    return Metrics(
        macro_f1=float(np.mean(klassenwerte)),
        per_class_f1=per_class_f1,
        accuracy=float(richtig.mean()),
        brier=brier,
        ece=ece,
        nll=nll,
        coverage_at_precision=coverage,
        aurc=aurc,
        auroc_confidence=_auroc(konfidenz, richtig),
        n=int(werte.shape[0]),
    )


def _auroc(konfidenz: npt.NDArray[np.float64], richtig: npt.NDArray[np.bool_]) -> float | None:
    """AUROC der Konfidenz als Fehlerdetektor – ``None``, wenn sie nicht bestimmt ist.

    Gerechnet wird über die Rangsumme (Mann-Whitney-U) und nicht über
    ``sklearn.metrics.roc_auc_score``: Jenes wirft, wenn nur eine Sorte vorkommt, und
    dieser Fall hat hier ein gültiges Ergebnis – ``None``, Begründung in :class:`Metrics`.

    ``rankdata`` vergibt bei Gleichstand den Mittelrang. Das ist nicht kosmetisch: Ohne
    ihn entschiede die Sortierreihenfolge, welche von zwei gleich konfidenten Vorhersagen
    „höher" steht, und eine Auswertung mit lauter gleichen Konfidenzen käme je nach
    Zeilenfolge auf 0 oder 1 statt auf 0,5.
    """
    anzahl_richtig = int(richtig.sum())
    anzahl_falsch = int(richtig.size) - anzahl_richtig
    if anzahl_richtig == 0 or anzahl_falsch == 0:
        return None
    raenge = rankdata(konfidenz, method="average")
    rangsumme = float(raenge[richtig].sum())
    kleinstmoegliche = anzahl_richtig * (anzahl_richtig + 1) / 2.0
    return (rangsumme - kleinstmoegliche) / (anzahl_richtig * anzahl_falsch)


def confusion(
    y_true: Sequence[str] | npt.NDArray[np.str_],
    y_pred: Sequence[str] | npt.NDArray[np.str_],
    classes: Sequence[str],
) -> pl.DataFrame:
    """Die Konfusionsmatrix: **Zeile = Wahrheit, Spalte = Vorhersage**.

    Die Richtung ist der ganze Inhalt dieser Tabelle. „Rechnung als Gutschrift gelesen" und
    „Gutschrift als Rechnung gelesen" sind verschiedene Probleme mit verschiedenen Kosten;
    eine transponierte Matrix vertauscht sie lautlos, und alle Randsummen bleiben plausibel.
    Die Zeilensummen sind die wahren Belegungen der Klassen, die Spaltensummen die Zahl der
    jeweiligen Vorhersagen.

    Der DataFrame hat eine Spalte :data:`WAHR_SPALTE` mit den Klassennamen in der
    Reihenfolge von ``classes`` und danach eine ``Int64``-Spalte je Klasse, ebenfalls in
    dieser Reihenfolge.

    Ein Name, der weder in ``classes`` steht noch dort stehen kann, wird **abgewiesen, nicht
    weggelassen**: ``sklearn.metrics.confusion_matrix`` lässt unbekannte Labels
    stillschweigend aus, und die Matrix summierte sich dann auf weniger Dokumente als
    übergeben, ohne dass es irgendwo stünde.
    """
    namen = _klassen_pruefen(classes)
    if WAHR_SPALTE in namen:
        raise ValueError(
            f"Eine Klasse heisst {WAHR_SPALTE!r} wie die Spalte mit der Zeilenbeschriftung. "
            "Ihre Zaehlspalte ueberschriebe die Beschriftung, und die Zeilen waeren nicht "
            "mehr zuzuordnen."
        )
    wahrheit = [str(wert) for wert in y_true]
    vorhersage = [str(wert) for wert in y_pred]
    if len(wahrheit) != len(vorhersage):
        raise ValueError(
            f"{len(wahrheit)} Wahrheitswerte stehen {len(vorhersage)} Vorhersagen "
            "gegenueber - je Dokument wird genau eines von beiden gebraucht."
        )
    if not wahrheit:
        raise ValueError("Keine Dokumente uebergeben - eine leere Matrix zaehlt nichts.")
    unbekannt = sorted(set(wahrheit + vorhersage) - set(namen))
    if unbekannt:
        raise ValueError(
            f"{unbekannt!r} kommt in y_true oder y_pred vor, steht aber nicht in classes "
            f"({namen!r}). Diese Dokumente fielen stillschweigend aus der Matrix, und ihre "
            "Summe waere kleiner als die Zahl der Dokumente."
        )

    zaehlung = confusion_matrix(wahrheit, vorhersage, labels=namen)
    tabelle: dict[str, list[str] | list[int]] = {WAHR_SPALTE: namen}
    for spalte, name in enumerate(namen):
        tabelle[name] = [int(wert) for wert in zaehlung[:, spalte]]
    return pl.DataFrame(tabelle)


#: Kennzahlen, über die ein Bootstrap-Intervall gebildet werden kann: die Gleitkommafelder
#: von :class:`Metrics`. ``per_class_f1`` ist keine einzelne Zahl, ``n`` ist über alle
#: Ziehungen konstant und ergäbe ein Intervall der Breite 0, das wie eine Messung aussähe.
BOOTSTRAP_METRIKEN = (
    "macro_f1",
    "accuracy",
    "brier",
    "ece",
    "nll",
    "coverage_at_precision",
    "aurc",
    "auroc_confidence",
)


def bootstrap_ci(
    proba: Zahlenfeld,
    y_true: Sequence[str] | npt.NDArray[np.str_],
    classes: Sequence[str],
    metric: str,
    rounds: int = 1000,
    seed: int = 7,
) -> tuple[float, float]:
    """95-%-Bootstrap-Intervall einer Kennzahl – das 2,5- und das 97,5-Perzentil.

    ``rounds`` Ziehungen mit Zurücklegen über die **Zeilen** (Dokument samt Wahrheit
    bleiben zusammen), je Ziehung :func:`evaluate`, danach die beiden Perzentile über die
    ``rounds`` Werte. Der Seed ist fest, und gleicher Seed heißt gleiches Ergebnis
    (global-constraints: „Zufall ist immer geseedet").

    Konzept § 9.2 begründet die 50 Dokumente je Klasse damit, dass Unterschiede sonst im
    Rauschen verschwinden – das Intervall ist die Zahl, an der man das sieht.

    ``coverage_at_precision`` wird mit dem Startziel 0,98 gezogen; ein eigenes
    ``target_precision`` hat diese Schnittstelle nicht.

    **Eine entartete Ziehung wird geworfen, nicht übergangen.** Trifft eine Ziehung eine
    Klasse weder in der Wahrheit noch in den Vorhersagen, ist deren F1 undefiniert
    (:func:`evaluate`). Solche Ziehungen zu überspringen verschöbe das Intervall
    stillschweigend – es wäre dann auf eine andere Verteilung bedingt als die genannte.
    Geworfen wird stattdessen mit der Nummer der Ziehung, weil der Fall bedeutet, dass die
    Menge für einen Bootstrap über die Zeilen zu dünn besetzt ist.
    """
    if metric not in BOOTSTRAP_METRIKEN:
        raise ValueError(
            f"{metric!r} ist keine Kennzahl, ueber die ein Intervall gebildet werden kann. "
            f"Moeglich sind {list(BOOTSTRAP_METRIKEN)!r}."
        )
    if rounds < MINDESTZIEHUNGEN:
        raise ValueError(
            f"{rounds} Ziehungen reichen fuer das {UNTERES_PERZENTIL}.- und das "
            f"{OBERES_PERZENTIL}.-Perzentil nicht; noetig sind {MINDESTZIEHUNGEN}. "
            "Darunter sitzen beide Grenzen auf dem kleinsten und groessten gezogenen Wert, "
            "und das Intervall misst die Zahl der Ziehungen statt die Streuung."
        )
    werte = _als_float64(proba)
    namen = _klassen_pruefen(classes)
    _pruefe_form(werte, y_true, namen)
    wahrheit = np.asarray([str(wert) for wert in y_true], dtype=np.str_)
    if wahrheit.size == 0:
        raise ValueError("Keine Dokumente uebergeben - aus nichts wird nicht gezogen.")

    rng = np.random.default_rng(seed)
    gezogen = np.empty(rounds, dtype=np.float64)
    for runde in range(rounds):
        index = rng.integers(0, wahrheit.size, wahrheit.size)
        try:
            kennzahl = getattr(evaluate(werte[index], list(wahrheit[index]), namen), metric)
        except ValueError as fehler:
            raise ValueError(
                f"Ziehung {runde} von {rounds} ist entartet: {fehler} Bei {wahrheit.size} "
                "Dokumenten trifft eine Ziehung mit Zuruecklegen eine duenn besetzte Klasse "
                "nicht mehr; sie zu ueberspringen verschoebe das Intervall stillschweigend."
            ) from fehler
        if kennzahl is None:
            raise ValueError(
                f"Ziehung {runde} von {rounds} liefert fuer {metric!r} keinen Wert (alle "
                "Vorhersagen richtig oder alle falsch). Ein eingesetzter Ersatzwert ginge "
                "als Messung ins Intervall ein."
            )
        gezogen[runde] = float(kennzahl)

    unten = float(np.percentile(gezogen, UNTERES_PERZENTIL, method="linear"))
    oben = float(np.percentile(gezogen, OBERES_PERZENTIL, method="linear"))
    return unten, oben
