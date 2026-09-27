"""Bilder und Ablage einer Auswertung (Konzept § 9.3, § 11).

Konzept § 9.3 nennt das Reliability Diagram „ein Bild, das mehr erklärt als drei Zahlen",
und § 11 verlangt von Phase 2 „ein Reliability Diagram vor und nach der Kalibrierung" –
**die erste sichtbare Erkenntnis des Projekts**. Beide Kurven gehören deshalb in *ein*
Bild, mit der Diagonale als Ideal: Nebeneinander in zwei Bildern sieht man den Unterschied
nicht, weil das Auge zwei Achsenbereiche nicht übereinanderlegt.

**Die Körbe werden hier nicht gerechnet.** :func:`reliability_bins` und
:class:`ReliabilityBin` sind dieselben Namen wie in :mod:`doccls.calibrate` und
ausdrücklich kein Nachbau – dort steht die verbindliche Einteilung, und
``expected_calibration_error`` bildet seine Zahl aus genau diesen Körben. Eine zweite
Einteilung hier hätte genau einen Effekt: Das Bild zeigte irgendwann etwas anderes als der
Eichfehler darunter. Umgekehrt geht es nicht – dieses Modul hängt über
:mod:`doccls.evaluate` an :mod:`doccls.calibrate` –, deshalb steht die Rechnung dort und
die Darstellung hier.

**Kein Fenster.** Gezeichnet wird über die Figure-Schnittstelle und nicht über ``pyplot``.
Das hat zwei Gründe, und beide fallen sonst erst spät auf: ``pyplot`` hält jede Figur in
einer globalen Verwaltung fest, bis jemand sie schließt (ab zwanzig offenen warnt
matplotlib, der Speicher wächst schon vorher), und ``pyplot`` wählt beim Import ein
Backend, das auf einem Rechner ohne Anzeige hängen bleibt. Eine Figur, die hier entsteht,
gehört ihrem Aufrufer: Gibt eine Funktion sie nicht zurück, ist sie nach dem Speichern
unerreichbar und wird eingesammelt. ``matplotlib.use("Agg")`` steht trotzdem unten im
Modul – falls später doch jemand ``pyplot`` dazunimmt, ist die Wahl schon getroffen.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.figure import Figure

from doccls.calibrate import ReliabilityBin, eichfehler_aus_koerben, reliability_bins
from doccls.evaluate import WAHR_SPALTE, Metrics
from doccls.zahlen import Zahlenfeld

__all__ = [
    "ReliabilityBin",
    "confusion_heatmap",
    "reliability_bins",
    "reliability_diagram",
    "write_metrics",
]

matplotlib.use("Agg")
"""Ein Backend ohne Fenster, ausdrücklich gesetzt und nicht der Umgebung überlassen.

Ohne diese Zeile sucht matplotlib sich beim ersten ``pyplot``-Import ein interaktives
Backend; auf einem Rechner ohne Anzeige – dem Testläufer, einem Container – hängt der
Aufruf dann, und zwar dort und nicht hier.
"""

BILDPUNKTE_JE_ZOLL = 120
"""Auflösung der erzeugten PNG. Fest, damit zwei Läufe vergleichbare Bilder ergeben."""


def reliability_diagram(
    proba_vorher: Zahlenfeld,
    proba_nachher: Zahlenfeld,
    y_true: Sequence[str] | npt.NDArray[np.str_],
    classes: Sequence[str],
    path: Path,
    bins: int = 10,
) -> tuple[Path, Figure]:
    """Eichung vor und nach der Kalibrierung – beide Kurven in einem Bild.

    ``proba_vorher`` sind die Wahrscheinlichkeiten des rohen Modells, ``proba_nachher``
    dieselben Zeilen nach :func:`doccls.calibrate.apply_temperature`. Gezeichnet wird je
    Kurve ein Punkt pro besetztem Korb: mittlere Konfidenz gegen Trefferquote. Die
    Diagonale von ``(0, 0)`` nach ``(1, 1)`` ist das Ideal; was darunter liegt, ist
    überheblich, was darüber liegt, zu zaghaft.

    **Der Eichfehler steht in der Beschriftung**, und zwar der aus denselben Körben. Die
    Zahl gehört ins Bild und nicht daneben: Getrennt gepflegt gerieten Kurve und Zahl
    irgendwann auseinander.

    Zurückgegeben wird ``(Pfad, Figur)``. Die Figur ist mitgegeben, damit der Aufrufer –
    und vor allem der Test – ihren Inhalt prüfen kann, ohne die PNG zu zerlegen; ein Test,
    der nur prüft, dass eine Datei entsteht, ginge auch bei einer leeren Leinwand durch.
    Sie gehört danach dem Aufrufer und hängt in keiner globalen Verwaltung (Moduldoc).
    """
    vorher = reliability_bins(proba_vorher, y_true, classes, bins)
    nachher = reliability_bins(proba_nachher, y_true, classes, bins)

    figur = Figure(figsize=(6.5, 6.0), dpi=BILDPUNKTE_JE_ZOLL)
    achse = figur.add_subplot()
    achse.plot([0.0, 1.0], [0.0, 1.0], color="0.5", linestyle="--", label="Ideal (Diagonale)")
    for koerbe, beschriftung, farbe, form in (
        (vorher, "vor der Kalibrierung", "tab:red", "o"),
        (nachher, "nach der Kalibrierung", "tab:blue", "s"),
    ):
        achse.plot(
            [korb.mittlere_konfidenz for korb in koerbe],
            [korb.trefferquote for korb in koerbe],
            marker=form,
            color=farbe,
            label=f"{beschriftung} (ECE {_deutsch(eichfehler_aus_koerben(koerbe))})",
        )

    achse.set_xlim(0.0, 1.0)
    achse.set_ylim(0.0, 1.0)
    achse.set_aspect("equal")
    achse.set_xlabel("Mittlere Konfidenz im Korb")
    achse.set_ylabel("Trefferquote im Korb")
    achse.set_title(f"Reliability Diagram, {bins} Koerbe")
    achse.grid(visible=True, color="0.9")
    achse.legend(loc="upper left")
    figur.tight_layout()

    return _speichern(figur, path), figur


def confusion_heatmap(matrix: pl.DataFrame, path: Path) -> tuple[Path, Figure]:
    """Die Konfusionsmatrix als Bild: **Zeile = Wahrheit, Spalte = Vorhersage**.

    ``matrix`` ist die Tabelle aus :func:`doccls.evaluate.confusion` und wird unverändert
    übernommen – zwischen Rechnung und Bild steht keine zweite Umformung, die
    auseinanderlaufen könnte. Die Richtung ist der ganze Inhalt: „Rechnung als Gutschrift
    gelesen" und „Gutschrift als Rechnung gelesen" sind verschiedene Probleme mit
    verschiedenen Kosten, und eine transponierte Darstellung vertauscht sie lautlos,
    während alle Randsummen plausibel bleiben.

    Zurückgegeben wird ``(Pfad, Figur)`` wie beim Reliability Diagram. Die Figur wird
    gebraucht, weil die **Achsenbeschriftung** aus einer PNG nicht lesbar ist: Die
    Geometrie der Zellen lässt sich über die Bildpunkte prüfen, aber ob über der
    x-Achse „Vorhergesagt" steht und neben der y-Achse „Wahr", nicht. Vertauscht man
    die beiden Texte, bleiben alle Zahlen und alle Klassennamen richtig, und trotzdem
    liest jeder Betrachter jede Zelle neben der Diagonale verkehrt herum – gemessen:
    ohne diese Bindung überlebte genau diese Vertauschung die ganze Testreihe.

    Die Farbskala beginnt fest bei 0, damit zwei Bilder desselben Laufs vergleichbar
    sind und eine schwach besetzte Zelle nicht allein durch Normierung dunkel wird.
    """
    namen, zaehlung = _matrix_lesen(matrix)

    figur = Figure(figsize=(6.0, 5.0), dpi=BILDPUNKTE_JE_ZOLL)
    achse = figur.add_subplot()
    bild = achse.imshow(zaehlung, cmap="Blues", vmin=0.0, vmax=float(zaehlung.max()))
    figur.colorbar(bild, ax=achse, label="Dokumente")

    schwelle = float(zaehlung.max()) / 2.0
    for zeile in range(len(namen)):
        for spalte in range(len(namen)):
            wert = int(zaehlung[zeile, spalte])
            achse.text(
                spalte,
                zeile,
                str(wert),
                ha="center",
                va="center",
                color="white" if wert > schwelle else "black",
            )

    achse.set_xticks(range(len(namen)), namen, rotation=45, ha="right")
    achse.set_yticks(range(len(namen)), namen)
    achse.set_xlabel("Vorhergesagt")
    achse.set_ylabel("Wahr")
    achse.set_title("Konfusionsmatrix")
    figur.tight_layout()

    return _speichern(figur, path), figur


def write_metrics(metrics: Metrics, path: Path) -> Path:
    """Die Kennzahlen als JSON ablegen – lesbar, vollständig, wieder einlesbar.

    Geschrieben wird das ganze Modell, nicht eine Auswahl: ``per_class_f1`` ist der Teil,
    der beim Verkürzen zuerst verschwindet, und genau er sagt, *welche* Klasse durchfällt.
    Ein fehlendes ``auroc_confidence`` bleibt ``null`` und wird nicht durch 0,5 ersetzt –
    „nicht messbar" ist eine andere Aussage als „keine Trennschärfe gemessen"
    (:class:`doccls.evaluate.Metrics`).

    ``Metrics.model_validate_json`` liest die Datei zurück und prüft dabei erneut die
    Kreuzbeziehungen der Felder; eine von Hand veränderte Zahl fällt damit beim Einlesen
    auf und nicht erst im Bericht.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metrics.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _deutsch(wert: float) -> str:
    """Drei Nachkommastellen mit Komma – wie überall sonst in diesem Projekt."""
    return f"{wert:.3f}".replace(".", ",")


def _matrix_lesen(matrix: pl.DataFrame) -> tuple[list[str], npt.NDArray[np.float64]]:
    """Klassennamen und Zählwerte aus der Tabelle von :func:`doccls.evaluate.confusion`.

    Geprüft wird, dass Zeilen und Spalten **dieselbe** Klassenreihenfolge tragen. Täten
    sie es nicht, wäre die Diagonale des Bildes nicht die Menge der Treffer – und das
    fiele niemandem auf, weil eine Konfusionsmatrix mit vertauschten Spalten genauso
    aussieht wie eine mit schlechtem Modell.
    """
    if WAHR_SPALTE not in matrix.columns:
        raise ValueError(
            f"Der Tabelle fehlt die Spalte {WAHR_SPALTE!r} mit der Zeilenbeschriftung; "
            f"vorhanden sind {matrix.columns!r}. Ohne sie ist nicht bekannt, welche "
            "Klasse welche Zeile ist."
        )
    namen = [str(wert) for wert in matrix[WAHR_SPALTE]]
    uebrige = [spalte for spalte in matrix.columns if spalte != WAHR_SPALTE]
    if uebrige != namen:
        raise ValueError(
            f"Die Spalten {uebrige!r} tragen eine andere Reihenfolge als die Zeilen "
            f"{namen!r}. Die Diagonale des Bildes waere dann nicht die Menge der Treffer, "
            "und das saehe wie ein schlechtes Modell aus."
        )
    zaehlung = np.asarray(matrix.select(namen).to_numpy(), dtype=np.float64)
    if zaehlung.sum() <= 0.0:
        raise ValueError(
            "Die Matrix zaehlt kein Dokument. Eine Farbskala von 0 bis 0 gibt jeder Zelle "
            "dieselbe Farbe, und das Bild behauptete eine gleichmaessige Verteilung."
        )
    return namen, zaehlung


def _speichern(figur: Figure, path: Path) -> Path:
    """Die Figur unter ``path`` ablegen, Verzeichnis inbegriffen.

    Der Lauf schreibt nach ``data/reports/<model_version>/``; beim ersten Lauf einer
    Modellversion gibt es dieses Verzeichnis noch nicht.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figur.savefig(path)
    return path
