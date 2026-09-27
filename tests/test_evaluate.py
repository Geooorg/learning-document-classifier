"""Tests für Aufgabe 14: die Kennzahlen (Konzept § 9.3).

Die Tests arbeiten auf **synthetischen** Daten, nicht auf dem echten Bestand. Gemessen:
Auf diesem Korpus erreichen alle drei Modellarten Macro-F1 1,000 auf dem Gold-Set. Dort
ist jede Kennzahl entartet – ``auroc_confidence`` ist gar nicht definiert, ``aurc`` ist 0,
``coverage_at_precision`` ist 1 –, und eine Zusicherung dagegen prüfte den Korpus, nicht
das Verfahren.

Stattdessen stehen hier zwei von Hand nachgerechnete Beispiele und drei Generatoren mit
vorab bekannter Struktur:

* ``_beleg_und_gutschrift`` – fünf Zeilen, zwei Klassen, jede Kennzahl vorher
  ausgerechnet (Accuracy 0,8, Macro-F1 4/9, ECE 0,162, Brier 0,15502, NLL 0,457394,
  Coverage@P98 0,6, AURC 0,07, AUROC 0,75).
* ``_vertrag_agb_rechnung`` – dreizehn Zeilen, drei Klassen, **asymmetrisch**: Präzision,
  Trefferquote und F1 fallen je Klasse auseinander (Macro-F1 0,7222 gegen
  Macro-Präzision 0,7460 und Macro-Trefferquote 0,7333). Ein symmetrisches Beispiel, in
  dem die drei zusammenfielen, bände F1 nicht – es bände nur, dass irgendeine der drei
  gerechnet wird.
* ``_gemischte_vorhersagen`` für die Temperaturfragen, ``_gemischte_wahrscheinlichkeiten``
  für den Bootstrap, ``_konfidenz_trennt_gut`` für die Richtung der AURC.

**Abweichung vom Aufgabentext – und warum.** Der Plan begründet die Temperaturtests mit
„Temperatur ändert die Rangfolge nicht". Das gilt **exakt nur bei zwei Klassen**: Dort ist
die Konfidenz ``sigmoid(|z₁ − z₂| / T)`` und damit streng monoton im Logit-Abstand. Ab
drei Klassen ist ``max softmax`` keine monotone Funktion einer einzigen Größe mehr, und
die Rangfolge der Dokumente ändert sich sehr wohl. Nachgemessen auf drei Klassen, 600
Dokumenten, dem Seed 11 des Plans::

    Klassenabstand 1,6: |Coverage@P98(T=0,3) − Coverage@P98(T=1,0)| = 0,0567
                        |AUROC(T=0,5) − AUROC(T=2,0)|               = 5,9e-03
    Klassenabstand 2,2: |ΔCoverage| = 0,0167,  |ΔAUROC| = 3,3e-03
    Klassenabstand 2,8: |ΔCoverage| = 0,0167,  |ΔAUROC| = 5,3e-03

Die Zusicherungen des Plans (``< 0,02`` für die Coverage, ``< 1e-6`` für die AUROC) sind
damit bei drei Klassen **nachweislich falsch** – die erste fällt beim Seed des Plans
durch, die zweite um drei bis vier Größenordnungen. Deshalb baut ``_gemischte_vorhersagen``
**zwei** Klassen: Dort ist die Rangfolge exakt erhalten, beide Zusicherungen des Plans
gelten unverändert und werden auch exakt erfüllt (gemessen: ΔCoverage 0,0, ΔAUROC
≤ 2,2e-16). Was sie binden, bleibt dasselbe – eine Coverage, die über eine feste
Konfidenzschwelle statt über die Rangfolge gebildet wird, fällt hier durch.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from pydantic import ValidationError

from doccls.calibrate import apply_temperature
from doccls.evaluate import Metrics, bootstrap_ci, confusion, evaluate

KLASSEN_ZWEI = ("a", "b")
KLASSEN_HAEUFIG_SELTEN = ("haeufig", "selten")

#: Bewusst **nicht** alphabetisch: Die Zuordnung Spalte ↔ Klassenname darf nicht daran
#: hängen, dass ``sorted(classes)`` zufällig dasselbe ergibt.
KLASSEN_DREI = ("VERTRAG", "AGB", "RECHNUNG")
KLASSEN_BELEG = ("RECHNUNG", "GUTSCHRIFT")


def _proba(
    vorhersagen: Sequence[str], konfidenzen: Sequence[float], klassen: Sequence[str]
) -> npt.NDArray[np.float64]:
    """Eine Wahrscheinlichkeitsmatrix mit vorgegebener Vorhersage und Konfidenz je Zeile.

    Die Restmasse verteilt sich gleichmäßig auf die übrigen Klassen. Solange die Konfidenz
    über ``1 / k`` liegt, ist die vorgegebene Klasse das ``argmax`` – die Vorhersage steht
    damit fest und muss nicht aus dem Ergebnis abgelesen werden.
    """
    spalte = {klasse: index for index, klasse in enumerate(klassen)}
    matrix = np.empty((len(vorhersagen), len(klassen)), dtype=np.float64)
    for zeile, (name, konfidenz) in enumerate(zip(vorhersagen, konfidenzen, strict=True)):
        assert konfidenz > 1.0 / len(klassen), "sonst ist die Vorhersage nicht das argmax"
        matrix[zeile, :] = (1.0 - konfidenz) / (len(klassen) - 1)
        matrix[zeile, spalte[name]] = konfidenz
    return matrix


def _sichere_vorhersage(
    vorhersagen: Sequence[str], klassen: Sequence[str], konfidenz: float = 0.99
) -> npt.NDArray[np.float64]:
    """Dieselbe hohe Konfidenz für jede Zeile."""
    return _proba(vorhersagen, [konfidenz] * len(vorhersagen), klassen)


def _beleg_und_gutschrift() -> tuple[npt.NDArray[np.float64], list[str], tuple[str, ...]]:
    """Fünf Zeilen, zwei Klassen, alle Kennzahlen von Hand ausgerechnet.

    Vorhergesagt wird durchgehend ``RECHNUNG`` mit den Konfidenzen 0,99 / 0,95 / 0,90 /
    0,85 / 0,80; die vierte Zeile ist in Wahrheit eine ``GUTSCHRIFT``. Die
    Präfix-Präzisionen sind damit 1, 1, 1, 0,75, 0,8 – die Reihenfolge ist absichtlich
    nicht monoton, sonst fielen „größte" und „kleinste haltende Abdeckung" zusammen.
    """
    konfidenzen = [0.99, 0.95, 0.90, 0.85, 0.80]
    proba = _proba(["RECHNUNG"] * 5, konfidenzen, KLASSEN_BELEG)
    y = ["RECHNUNG", "RECHNUNG", "RECHNUNG", "GUTSCHRIFT", "RECHNUNG"]
    return proba, y, KLASSEN_BELEG


def _vertrag_agb_rechnung() -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
    """Dreizehn Zeilen, drei Klassen, Präzision und Trefferquote je Klasse verschieden.

    Von Hand: ``VERTRAG`` 3 richtig, keine Verwechslung (F1 1,0). ``AGB`` 4 richtig, 1 als
    ``RECHNUNG`` gelesen, 3 fälschlich als ``AGB`` gelesen (P 4/7, R 4/5, F1 2/3).
    ``RECHNUNG`` 2 richtig, 3 als ``AGB`` gelesen, 1 fälschlich als ``RECHNUNG`` gelesen
    (P 2/3, R 2/5, F1 1/2).
    """
    y = ["AGB"] * 5 + ["RECHNUNG"] * 5 + ["VERTRAG"] * 3
    vorhergesagt = ["AGB"] * 4 + ["RECHNUNG"] + ["RECHNUNG"] * 2 + ["AGB"] * 3 + ["VERTRAG"] * 3
    return _sichere_vorhersage(vorhergesagt, KLASSEN_DREI, 0.6), y, vorhergesagt


def _gemischte_vorhersagen(
    seed: int, n: int = 600, abstand: float = 1.8
) -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
    """Rohe Logits mit gemischter Korrektheit – **zwei** Klassen (siehe Moduldoc).

    Die wahre Klasse bekommt einen um ``abstand`` verschobenen Zuschlag. Gemessen bei
    ``abstand = 1,8``: Accuracy 0,847, Coverage@P98 0,518, ECE 0,054 bei ``T = 0,5``
    gegen 0,136 bei ``T = 2,0`` – der Eichfehler bewegt sich also deutlich, die Rangmaße
    nicht.
    """
    rng = np.random.default_rng(seed)
    klassen = ["ALPHA", "BETA"]
    wahr = rng.integers(0, len(klassen), n)
    logits: npt.NDArray[np.float64] = rng.normal(0.0, 1.0, (n, len(klassen)))
    logits[np.arange(n), wahr] += rng.normal(abstand, 1.0, n)
    return logits, [klassen[index] for index in wahr], klassen


def _gemischte_wahrscheinlichkeiten(
    seed: int, n: int = 400
) -> tuple[npt.NDArray[np.float32], list[str], list[str]]:
    """Kalibrierte Wahrscheinlichkeiten über drei **gleich besetzte** Klassen.

    Die Gleichbesetzung ist kein Zierrat: Der Bootstrap zieht über die Zeilen, und eine
    dünn besetzte Klasse fehlte in manchen Ziehungen ganz – ``evaluate`` wirft dann (zu
    Recht), und der Test wäre vom Seed abhängig. Bei 13 Zeilen je Klasse in den ersten 40
    liegt die Wahrscheinlichkeit dafür bei ``(27/40)^40 ≈ 1,5e-07`` je Ziehung.
    """
    rng = np.random.default_rng(seed)
    klassen = ["ALPHA", "BETA", "GAMMA"]
    wahr = rng.permutation(np.arange(n) % len(klassen))
    logits: npt.NDArray[np.float64] = rng.normal(0.0, 1.0, (n, len(klassen)))
    logits[np.arange(n), wahr] += rng.normal(1.6, 1.0, n)
    return apply_temperature(logits, 1.0), [klassen[index] for index in wahr], klassen


def _konfidenz_trennt_gut(
    seed: int, n: int = 500
) -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
    """Ein perfekt geeichtes Modell: ``richtig ~ Bernoulli(konfidenz)``.

    Die Konfidenz trennt richtig von falsch dadurch so gut, wie es überhaupt geht, ohne
    dass die Trennung in die Zahlen hineingelegt wäre – sie folgt aus der Eichung.
    """
    rng = np.random.default_rng(seed)
    klassen = ["ALPHA", "BETA", "GAMMA"]
    wahr = rng.integers(0, len(klassen), n)
    konfidenz = rng.uniform(0.4, 1.0, n)
    richtig = rng.random(n) < konfidenz
    versatz = rng.integers(1, len(klassen), n)
    vorhergesagt = np.where(richtig, wahr, (wahr + versatz) % len(klassen))
    proba = _proba([klassen[index] for index in vorhergesagt], list(konfidenz), klassen)
    return proba, [klassen[index] for index in wahr], klassen


def _konfidenz_zufaellig(gut: npt.NDArray[np.float64], seed: int) -> npt.NDArray[np.float64]:
    """Dieselben Vorhersagen, dieselbe Korrektheit – die Konfidenzen durchgemischt.

    Jede Zeile behält ihr ``argmax``; nur *welche* Zeile welche Konfidenz trägt, wird
    permutiert. Accuracy, Macro-F1 und Konfusionsmatrix bleiben dadurch unverändert, und
    allein die Kopplung von Konfidenz und Korrektheit ist zerstört. Genau die misst die
    AURC.
    """
    rng = np.random.default_rng(seed)
    konfidenz = rng.permutation(gut.max(axis=1))
    klassenzahl = gut.shape[1]
    neu = np.tile(((1.0 - konfidenz) / (klassenzahl - 1))[:, None], (1, klassenzahl))
    neu[np.arange(gut.shape[0]), gut.argmax(axis=1)] = konfidenz
    return neu


def _zelle(matrix: pl.DataFrame, wahr: str, vorhergesagt: str) -> int:
    """Die Zelle „in Wahrheit ``wahr``, gelesen als ``vorhergesagt``"."""
    zeile = matrix.filter(pl.col("true_class") == wahr)
    assert zeile.height == 1, f"{wahr!r} steht {zeile.height}-mal in der Matrix"
    return int(zeile[vorhergesagt][0])


def _gueltige_kennzahlen(**abweichungen: Any) -> dict[str, Any]:
    """Ein in sich stimmiger Satz Kennzahlen, aus dem einzelne Felder ersetzt werden."""
    felder: dict[str, Any] = {
        "macro_f1": 0.5,
        "per_class_f1": {"a": 0.4, "b": 0.6},
        "accuracy": 0.5,
        "brier": 0.25,
        "ece": 0.1,
        "nll": 0.7,
        "coverage_at_precision": 0.5,
        "aurc": 0.2,
        "auroc_confidence": 0.75,
        "n": 4,
    }
    felder.update(abweichungen)
    return felder


# ---------------------------------------------------------------- Macro-F1 und F1 je Klasse


def test_macro_f1_gewichtet_klassen_gleich() -> None:
    """Der Grund, warum nicht Accuracy die Hauptzahl ist (Konzept § 9.3). Ein Modell, das
    die seltene Klasse nie trifft, hat hohe Accuracy und schlechtes Macro-F1."""
    y = ["haeufig"] * 95 + ["selten"] * 5
    proba = _sichere_vorhersage(["haeufig"] * 100, KLASSEN_HAEUFIG_SELTEN)
    m = evaluate(proba, y, KLASSEN_HAEUFIG_SELTEN)
    assert m.accuracy == pytest.approx(0.95)
    assert m.macro_f1 < 0.55, "Macro-F1 muss das Uebergehen der seltenen Klasse bestrafen"
    # Von Hand: F1(haeufig) = 2*95 / (2*95 + 5 + 0) = 190/195; F1(selten) = 0, weil die
    # Trefferquote 0 ist - der Grenzwert von 2PR/(P+R), keine Konvention.
    assert m.per_class_f1 == pytest.approx({"haeufig": 190 / 195, "selten": 0.0})
    assert m.macro_f1 == pytest.approx(95 / 195)


def test_per_class_f1_trennt_praezision_von_trefferquote() -> None:
    """F1 ist weder Präzision noch Trefferquote, und das muss der Test sehen.

    Alle drei Werte liegen hier auseinander: Macro-F1 0,72222, Macro-Präzision 0,74603,
    Macro-Trefferquote 0,73333. Ein Beispiel, in dem sie zusammenfielen, bände nur, dass
    irgendeine der drei gerechnet wird.
    """
    proba, y, _ = _vertrag_agb_rechnung()
    m = evaluate(proba, y, KLASSEN_DREI)
    assert m.per_class_f1 == pytest.approx({"VERTRAG": 1.0, "AGB": 2 / 3, "RECHNUNG": 0.5})
    assert m.macro_f1 == pytest.approx(13 / 18)
    assert m.accuracy == pytest.approx(9 / 13)
    macro_praezision = (1.0 + 4 / 7 + 2 / 3) / 3
    macro_trefferquote = (1.0 + 4 / 5 + 2 / 5) / 3
    assert m.macro_f1 != pytest.approx(macro_praezision)
    assert m.macro_f1 != pytest.approx(macro_trefferquote)
    assert m.macro_f1 != pytest.approx(m.accuracy)


def test_kennzahlen_auf_von_hand_gerechnetem_beispiel() -> None:
    """Jede Kennzahl gegen einen vorher ausgerechneten Wert – nicht gegen eine Richtung.

    Fünf Zeilen, Konfidenzen 0,99 / 0,95 / 0,90 / 0,85 / 0,80, die vierte falsch. Die
    Rechnungen stehen im Docstring von ``_beleg_und_gutschrift`` und hier je Zeile.
    """
    proba, y, klassen = _beleg_und_gutschrift()
    m = evaluate(proba, y, klassen)

    assert m.n == 5
    assert m.accuracy == pytest.approx(0.8)
    # RECHNUNG: 2*4 / (2*4 + 1 + 0) = 8/9; GUTSCHRIFT: nie vorhergesagt -> 0
    assert m.per_class_f1 == pytest.approx({"RECHNUNG": 8 / 9, "GUTSCHRIFT": 0.0})
    assert m.macro_f1 == pytest.approx(4 / 9)
    # Mittel ueber alle 10 Zellen: (2*0.01^2 + 2*0.05^2 + 2*0.10^2 + 2*0.85^2 + 2*0.20^2)/10
    assert m.brier == pytest.approx(0.15502)
    # -(ln .99 + ln .95 + ln .90 + ln .15 + ln .80) / 5
    assert m.nll == pytest.approx(0.4573935, abs=1e-6)
    # Korb 9 (0,99/0,95/0,90): |0,946667 - 1| * 3/5; Korb 8 (0,85/0,80): |0,825 - 0,5| * 2/5
    assert m.ece == pytest.approx(0.6 * (1.0 - 0.94666667) + 0.4 * 0.325, abs=1e-6)
    # Praefix-Praezisionen 1, 1, 1, 0,75, 0,8 - das Ziel 0,98 haelt bis Abdeckung 3/5
    assert m.coverage_at_precision == pytest.approx(0.6)
    # Trapez ueber (0,2 | 0) (0,4 | 0) (0,6 | 0) (0,8 | 0,25) (1,0 | 0,2)
    assert m.aurc == pytest.approx(0.07)
    # Richtige Konfidenzen {0,99 0,95 0,90 0,80}, falsche {0,85}: drei von vier Paaren
    assert m.auroc_confidence == pytest.approx(0.75)


# ---------------------------------------------------------------------------- Coverage@P


def test_coverage_at_precision_folgt_dem_ziel() -> None:
    """Zweiter Zielwert, damit die Kennzahl nicht an der Vorgabe 0,98 festgetackert ist.

    Die Präfix-Präzisionen sind 1, 1, 1, 0,75, 0,8. Bei 0,98 hält die Abdeckung 3/5; bei
    0,75 hält die **volle** Abdeckung, denn die Gesamtpräzision 0,8 reicht. Die *kleinste*
    haltende Abdeckung wäre in beiden Fällen 1/5 – eine Verwechslung von „größte" und
    „kleinste" fällt hier also doppelt auf.
    """
    proba, y, klassen = _beleg_und_gutschrift()
    assert evaluate(proba, y, klassen, target_precision=0.98).coverage_at_precision == (
        pytest.approx(0.6)
    )
    assert evaluate(proba, y, klassen, target_precision=0.75).coverage_at_precision == (
        pytest.approx(1.0)
    )
    assert evaluate(proba, y, klassen, target_precision=0.9).coverage_at_precision == (
        pytest.approx(0.6)
    )


def test_coverage_ist_null_wenn_kein_ziel_haelt() -> None:
    """Kein Fehler, sondern ein gültiges schlechtes Ergebnis (Aufgabentext).

    Die konfidenteste Vorhersage ist hier falsch; die Präfix-Präzisionen sind 0, 0,5, 2/3
    und erreichen 0,98 nie.
    """
    proba = _proba(["RECHNUNG"] * 3, [0.99, 0.95, 0.90], KLASSEN_BELEG)
    m = evaluate(proba, ["GUTSCHRIFT", "RECHNUNG", "RECHNUNG"], KLASSEN_BELEG)
    assert m.coverage_at_precision == 0.0


def test_coverage_at_p98_laesst_sich_nicht_durch_temperatur_schoenen() -> None:
    """Konzept § 9.1 und § 9.3: Wer alle Konfidenzen hochskaliert, verschiebt nur die
    Schwelle mit. Das ist die Zusicherung, die Coverage@P98 zur ehrlichen Antwort auf
    'steigt die Konfidenz?' macht – und sie wird hier gemessen, nicht geglaubt."""
    logits, y, klassen = _gemischte_vorhersagen(seed=11)
    kalt = evaluate(apply_temperature(logits, 0.3), y, klassen)
    normal = evaluate(apply_temperature(logits, 1.0), y, klassen)
    assert abs(kalt.coverage_at_precision - normal.coverage_at_precision) < 0.02, (
        f"Coverage@P98 springt von {normal.coverage_at_precision:.3f} auf "
        f"{kalt.coverage_at_precision:.3f}, nur weil T=0,3 ist - dann misst die Kennzahl "
        "die Temperatur statt die Nutzbarkeit"
    )
    # Ohne diese beiden Zeilen liesse sich der Test mit einer festen 0,0 bestehen.
    assert 0.1 < normal.coverage_at_precision < 0.9
    # Und die Vorlage taugt nur, wenn T=0,3 die mittlere Konfidenz wirklich hochtreibt:
    # gemessen von 0,818 auf 0,939. Eine Kennzahl, die auf diese Zahl hoert statt auf die
    # Rangfolge, muss hier auseinanderlaufen.
    kalte_konfidenz = float(apply_temperature(logits, 0.3).max(axis=1).mean())
    normale_konfidenz = float(apply_temperature(logits, 1.0).max(axis=1).mean())
    assert kalte_konfidenz - normale_konfidenz > 0.1


# -------------------------------------------------------------------- AUROC, AURC, Eichung


def test_auroc_der_konfidenz_misst_trennschaerfe_unabhaengig_von_der_eichung() -> None:
    """Konzept § 9.3: AUROC der Konfidenz als Fehlerdetektor. Temperatur ändert die
    Rangfolge nicht, also darf die AUROC sich nicht bewegen – anders als der ECE."""
    logits, y, klassen = _gemischte_vorhersagen(seed=11)
    a = evaluate(apply_temperature(logits, 0.5), y, klassen)
    b = evaluate(apply_temperature(logits, 2.0), y, klassen)
    assert a.auroc_confidence is not None and b.auroc_confidence is not None
    assert abs(a.auroc_confidence - b.auroc_confidence) < 1e-6
    assert abs(a.ece - b.ece) > 0.01, "Der ECE muss sich sehr wohl bewegen"
    # Sonst bestaende der Test auch mit einer festen 0,5 oder 1,0.
    assert 0.6 < a.auroc_confidence < 0.95


def test_auroc_ist_none_wenn_es_nichts_zu_trennen_gibt() -> None:
    """Alle richtig oder alle falsch: Die Menge der Paare, über die die AUROC definiert
    ist, ist leer. 0,5 hiesse „keine Trennschaerfe", und das ist eine andere Aussage."""
    alles_richtig = _sichere_vorhersage(["a", "b"], KLASSEN_ZWEI)
    assert evaluate(alles_richtig, ["a", "b"], KLASSEN_ZWEI).auroc_confidence is None
    alles_falsch = _sichere_vorhersage(["a", "b"], KLASSEN_ZWEI)
    assert evaluate(alles_falsch, ["b", "a"], KLASSEN_ZWEI).auroc_confidence is None
    # Gegenprobe: eine falsche Vorhersage dazu, und die Zahl steht wieder.
    gemischt = _sichere_vorhersage(["a", "b", "a"], KLASSEN_ZWEI)
    assert evaluate(gemischt, ["a", "b", "b"], KLASSEN_ZWEI).auroc_confidence is not None


def test_auroc_zaehlt_gleichstaende_als_halb() -> None:
    """Lauter gleiche Konfidenzen: Die Konfidenz trennt nichts, die AUROC ist **exakt**
    0,5.

    Ohne Mittelränge entschiede die Zeilenreihenfolge, welche von zwei gleich konfidenten
    Vorhersagen „hoeher" steht – dieselbe Auswertung käme je nach Sortierung auf 0 oder 1.
    Hier stehen die beiden richtigen Zeilen vorn, eine ordinale Rangvergabe ergäbe also
    1,0.
    """
    proba = _sichere_vorhersage(["a"] * 4, KLASSEN_ZWEI)
    m = evaluate(proba, ["a", "a", "b", "b"], KLASSEN_ZWEI)
    assert m.auroc_confidence == 0.5


def test_aurc_faellt_wenn_die_konfidenz_besser_trennt() -> None:
    """AURC ist die Fläche unter der Risiko-Abdeckungs-Kurve – kleiner ist besser.
    Bindet die Richtung: Ein Vorzeichenfehler wäre sonst unsichtbar."""
    gut, y, klassen = _konfidenz_trennt_gut(seed=17)
    schlecht = _konfidenz_zufaellig(gut, seed=17)
    a = evaluate(gut, y, klassen)
    b = evaluate(schlecht, y, klassen)
    assert a.aurc < b.aurc
    # Die Mischung aendert nur die Kopplung, nicht die Vorhersagen: Wenn Accuracy und
    # Macro-F1 mitwanderten, verglichen wir zwei verschiedene Modelle statt zweier
    # Konfidenzverlaeufe.
    assert a.accuracy == pytest.approx(b.accuracy)
    assert a.macro_f1 == pytest.approx(b.macro_f1)
    assert b.aurc == pytest.approx(1.0 - a.accuracy, abs=0.03), (
        "Bei entkoppelter Konfidenz ist das Risiko auf jeder Abdeckung ungefaehr die "
        "Gesamtfehlerquote - so weit ist die Flaeche vorab bekannt"
    )


def test_brier_und_nll_bestrafen_sichere_irrtuemer() -> None:
    sicher_falsch = np.array([[0.99, 0.01]])
    unsicher_falsch = np.array([[0.55, 0.45]])
    a = evaluate(sicher_falsch, ["b"], KLASSEN_ZWEI)
    b = evaluate(unsicher_falsch, ["b"], KLASSEN_ZWEI)
    assert a.brier > b.brier and a.nll > b.nll
    # Von Hand, damit nicht nur die Richtung gebunden ist:
    assert a.brier == pytest.approx(0.99**2)
    assert b.brier == pytest.approx((0.55**2 + 0.55**2) / 2)
    assert a.nll == pytest.approx(-math.log(0.01))
    assert b.nll == pytest.approx(-math.log(0.45))


def test_nll_kappt_die_wahrscheinlichkeit_null() -> None:
    """``-log 0`` ist unendlich und fraesse sich durch jede Mittelung. Gekappt wird bei
    ``1e-12``, die NLL steht damit bei ``-ln(1e-12)`` und nicht bei ``inf``."""
    proba = np.array([[1.0, 0.0]])
    m = evaluate(proba, ["b"], KLASSEN_ZWEI)
    assert m.nll == pytest.approx(-math.log(1e-12))
    assert math.isfinite(m.nll)


# -------------------------------------------------------------------- Konfusionsmatrix


def test_konfusionsmatrix_zeigt_die_richtung_der_verwechslung() -> None:
    """Rechnung als Gutschrift zu lesen ist ein anderes Problem als umgekehrt. Eine
    symmetrische Matrix waere unbrauchbar."""
    m = confusion(["RECHNUNG"] * 5, ["GUTSCHRIFT"] * 5, ["RECHNUNG", "GUTSCHRIFT"])
    assert _zelle(m, "RECHNUNG", "GUTSCHRIFT") == 5
    assert _zelle(m, "GUTSCHRIFT", "RECHNUNG") == 0


def test_konfusionsmatrix_zaehlt_jedes_dokument_genau_einmal() -> None:
    """Die volle Matrix des asymmetrischen Beispiels, Zelle für Zelle.

    Die beiden Richtungen sind hier verschieden besetzt (1 gegen 3); eine Transposition
    fiele auf, und die Randsummen blieben trotzdem plausibel.
    """
    _, y, vorhergesagt = _vertrag_agb_rechnung()
    m = confusion(y, vorhergesagt, KLASSEN_DREI)
    assert m.columns == ["true_class", "VERTRAG", "AGB", "RECHNUNG"]
    assert _zelle(m, "AGB", "AGB") == 4
    assert _zelle(m, "AGB", "RECHNUNG") == 1
    assert _zelle(m, "RECHNUNG", "AGB") == 3
    assert _zelle(m, "RECHNUNG", "RECHNUNG") == 2
    assert _zelle(m, "VERTRAG", "VERTRAG") == 3
    assert _zelle(m, "VERTRAG", "AGB") == 0
    assert _zelle(m, "AGB", "VERTRAG") == 0
    assert _zelle(m, "RECHNUNG", "VERTRAG") == 0
    assert _zelle(m, "VERTRAG", "RECHNUNG") == 0
    summe = sum(int(m[name].sum()) for name in KLASSEN_DREI)
    assert summe == len(y)


def test_konfusionsmatrix_weist_unbekannte_klassen_ab() -> None:
    """``sklearn`` liesse sie stillschweigend weg, und die Matrix summierte sich auf
    weniger Dokumente als uebergeben."""
    with pytest.raises(ValueError, match="SONSTIGES"):
        confusion(["RECHNUNG", "SONSTIGES"], ["RECHNUNG", "RECHNUNG"], KLASSEN_BELEG)
    with pytest.raises(ValueError, match="SONSTIGES"):
        confusion(["RECHNUNG", "RECHNUNG"], ["RECHNUNG", "SONSTIGES"], KLASSEN_BELEG)


def test_konfusionsmatrix_weist_eine_klasse_namens_true_class_ab() -> None:
    """Ihre Zaehlspalte ueberschriebe die Zeilenbeschriftung."""
    with pytest.raises(ValueError, match="true_class"):
        confusion(["true_class"], ["true_class"], ["true_class", "RECHNUNG"])


def test_konfusionsmatrix_weist_ungleiche_laengen_ab() -> None:
    with pytest.raises(ValueError, match="3 Wahrheitswerte"):
        confusion(["RECHNUNG"] * 3, ["RECHNUNG"] * 2, KLASSEN_BELEG)


# ------------------------------------------------------------------------------ Bootstrap


def test_bootstrap_intervall_umschliesst_den_punktwert() -> None:
    proba, y, klassen = _gemischte_wahrscheinlichkeiten(seed=13)
    punkt = evaluate(proba, y, klassen).macro_f1
    unten, oben = bootstrap_ci(proba, y, klassen, metric="macro_f1", rounds=200)
    assert unten <= punkt <= oben
    assert unten < oben, "Ein Punktintervall waere keine Streuung"


def test_bootstrap_ist_bei_kleiner_menge_breiter() -> None:
    """Konzept § 9.2 begruendet die 50 Dokumente je Klasse damit, dass Unterschiede sonst
    im Rauschen verschwinden. Diese Pruefung bindet genau das: Weniger Daten, breiteres
    Intervall. Ein Bootstrap, der die Streuung nicht abbildet, faellt hier auf."""
    proba, y, klassen = _gemischte_wahrscheinlichkeiten(seed=13, n=400)
    weit = bootstrap_ci(proba[:40], y[:40], klassen, metric="macro_f1", rounds=200)
    eng = bootstrap_ci(proba, y, klassen, metric="macro_f1", rounds=200)
    assert (weit[1] - weit[0]) > (eng[1] - eng[0]) * 1.5


def test_bootstrap_ist_reproduzierbar() -> None:
    proba, y, klassen = _gemischte_wahrscheinlichkeiten(seed=13)
    assert bootstrap_ci(proba, y, klassen, "macro_f1", rounds=100, seed=7) == bootstrap_ci(
        proba, y, klassen, "macro_f1", rounds=100, seed=7
    )


def test_bootstrap_haengt_am_seed_und_an_der_zahl_der_ziehungen() -> None:
    """Die Gegenprobe zur Reproduzierbarkeit: Gleicher Seed heisst gleiches Ergebnis, aber
    eine Umsetzung, die den Seed gar nicht benutzt oder ``rounds`` ignoriert, bestuende den
    Reproduzierbarkeitstest ebenso."""
    proba, y, klassen = _gemischte_wahrscheinlichkeiten(seed=13)
    mit_sieben = bootstrap_ci(proba, y, klassen, "macro_f1", rounds=100, seed=7)
    mit_neun = bootstrap_ci(proba, y, klassen, "macro_f1", rounds=100, seed=9)
    assert mit_sieben != mit_neun
    mit_vierzig = bootstrap_ci(proba, y, klassen, "macro_f1", rounds=40, seed=7)
    assert mit_sieben != mit_vierzig


def test_bootstrap_liefert_ein_95_prozent_intervall() -> None:
    """Die Perzentile sind 2,5 und 97,5 – nicht 5/95 und nicht 0,5/99,5.

    Auf 400 Dokumenten mit einer Accuracy von genau 0,8 ist die Streuung der
    Bootstrap-Accuracy bekannt: ``se = sqrt(0,8 · 0,2 / 400) = 0,02``, ein
    95-%-Intervall ist also ``2 · 1,96 · se = 0,0784`` breit. Gemessen bei Seed 7 und
    1000 Ziehungen: 0,0725. Die Nachbarn liegen deutlich daneben und fallen durch die
    Toleranz von 12 % – 90 % ergäbe 0,0625, 99 % ergäbe 0,0925.
    """
    klassen = ("a", "b")
    y = ["a"] * 320 + ["b"] * 80
    proba = _proba(["a"] * 400, [0.9] * 400, klassen)
    assert evaluate(proba, y, klassen).accuracy == pytest.approx(0.8)
    unten, oben = bootstrap_ci(proba, y, klassen, "accuracy", rounds=1000)
    erwartet = 2 * 1.96 * math.sqrt(0.8 * 0.2 / 400)
    assert oben - unten == pytest.approx(erwartet, rel=0.12)


def test_bootstrap_weist_zu_wenige_ziehungen_ab() -> None:
    """Unter 40 Ziehungen sitzen beide Grenzen auf dem kleinsten und groessten Wert."""
    proba, y, klassen = _gemischte_wahrscheinlichkeiten(seed=13, n=60)
    with pytest.raises(ValueError, match="39 Ziehungen"):
        bootstrap_ci(proba, y, klassen, "macro_f1", rounds=39)
    bootstrap_ci(proba, y, klassen, "macro_f1", rounds=40)


def test_bootstrap_weist_eine_unbekannte_kennzahl_ab() -> None:
    proba, y, klassen = _gemischte_wahrscheinlichkeiten(seed=13, n=60)
    with pytest.raises(ValueError, match="per_class_f1"):
        bootstrap_ci(proba, y, klassen, "per_class_f1", rounds=40)
    with pytest.raises(ValueError, match="f1_macro"):
        bootstrap_ci(proba, y, klassen, "f1_macro", rounds=40)


def test_bootstrap_wirft_bei_einer_entarteten_ziehung() -> None:
    """Eine Klasse, die eine Ziehung gar nicht trifft, hat keine F1. Solche Ziehungen zu
    ueberspringen verschoebe das Intervall stillschweigend."""
    proba = _sichere_vorhersage(["RECHNUNG"] * 5 + ["GUTSCHRIFT"], KLASSEN_BELEG)
    y = ["RECHNUNG"] * 5 + ["GUTSCHRIFT"]
    # Auf der vollen Menge ist alles definiert - erst die Ziehung entartet.
    assert evaluate(proba, y, KLASSEN_BELEG).macro_f1 == pytest.approx(1.0)
    with pytest.raises(ValueError, match="entartet"):
        bootstrap_ci(proba, y, KLASSEN_BELEG, "macro_f1", rounds=40)


# ------------------------------------------------------------------------------- Wächter


def test_evaluate_weist_eine_klasse_ohne_wahrheit_und_ohne_vorhersage_ab() -> None:
    """``2*TP / (2*TP + FP + FN)`` ist dann 0/0. ``sklearn`` setzte hier stillschweigend
    eine 0 ein und zoege das Macro-F1 nach unten, ohne dass etwas gemessen waere."""
    klassen = ["RECHNUNG", "GUTSCHRIFT", "AGB"]
    ohne_agb = _sichere_vorhersage(["RECHNUNG", "RECHNUNG"], klassen)
    with pytest.raises(ValueError, match="AGB"):
        evaluate(ohne_agb, ["RECHNUNG", "GUTSCHRIFT"], klassen)
    # Nur vorhergesagt, nie wahr: definiert (F1 = 0), also kein Fehler.
    mit_agb = _sichere_vorhersage(["RECHNUNG", "AGB"], klassen)
    assert evaluate(mit_agb, ["RECHNUNG", "GUTSCHRIFT"], klassen).per_class_f1["AGB"] == 0.0


def test_evaluate_weist_doppelte_klassen_ab() -> None:
    """Zwei Spalten mit demselben Namen sind nicht unterscheidbar."""
    proba = _sichere_vorhersage(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="'a'"):
        evaluate(np.hstack([proba, proba]), ["a"], ["a", "b", "a", "b"])


def test_evaluate_weist_weniger_als_zwei_klassen_ab() -> None:
    with pytest.raises(ValueError, match="1 Klasse"):
        evaluate(np.array([[1.0]]), ["a"], ["a"])


def test_evaluate_weist_ein_ziel_ausserhalb_von_null_bis_eins_ab() -> None:
    proba, y, klassen = _beleg_und_gutschrift()
    with pytest.raises(ValueError, match="target_precision"):
        evaluate(proba, y, klassen, target_precision=1.0)
    with pytest.raises(ValueError, match="target_precision"):
        evaluate(proba, y, klassen, target_precision=0.0)


def test_evaluate_weist_die_leere_menge_ab() -> None:
    with pytest.raises(ValueError, match="Keine Dokumente"):
        evaluate(np.empty((0, 2)), [], KLASSEN_ZWEI)


# ------------------------------------------------------------- Kreuzprüfungen in Metrics


def test_metrics_weist_macro_f1_neben_den_klassenwerten_ab() -> None:
    """Die Prüfung, die ein gewichtetes Mittel oder eine verrutschte Zuweisung fängt."""
    Metrics(**_gueltige_kennzahlen())
    with pytest.raises(ValidationError, match="ungewichtete Mittel"):
        Metrics(**_gueltige_kennzahlen(macro_f1=0.9))


def test_metrics_weist_einen_anteil_ab_der_kein_vielfaches_von_1_durch_n_ist() -> None:
    """Accuracy und Abdeckung sind Anteile von ``n`` Dokumenten."""
    with pytest.raises(ValidationError, match="accuracy"):
        Metrics(**_gueltige_kennzahlen(accuracy=0.3, n=4))
    with pytest.raises(ValidationError, match="coverage_at_precision"):
        Metrics(**_gueltige_kennzahlen(coverage_at_precision=0.3, n=4))


def test_metrics_weist_kennzahlen_ausserhalb_ihres_bereichs_ab() -> None:
    for feld, wert in (
        ("macro_f1", 1.5),
        ("accuracy", -0.1),
        ("ece", 1.5),
        ("brier", 1.5),
        ("nll", -0.1),
        ("aurc", 1.5),
        ("auroc_confidence", 1.5),
        ("n", 0),
    ):
        with pytest.raises(ValidationError):
            Metrics(**_gueltige_kennzahlen(**{feld: wert}))
