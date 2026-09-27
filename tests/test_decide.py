"""Tests für Aufgabe 12: die beiden Schwellen τ und δ (Konzept § 7.4).

Die Tests arbeiten auf **synthetischen** Daten, nicht auf dem echten Bestand. Gemessen:
Auf diesem Korpus erreichen alle drei Modellarten Trefferquote 1,000 auf der
Kalibriermenge. Eine Risiko-Abdeckungs-Kurve ohne einen einzigen Fehler ist waagerecht,
jedes τ erfüllt jedes Präzisionsziel, und eine Zusicherung dagegen prüfte den Korpus,
nicht das Verfahren.

Stattdessen wird hier eine Fehlerstruktur mit **vorab bekannter** Lösung erzeugt:

* Für τ ein perfekt geeichtes Modell (``_konfidenz_mit_rauschen``). Aus
  ``richtig | konfidenz ~ Bernoulli(konfidenz)`` und ``konfidenz ~ U(0,5; 1,0)`` folgt
  analytisch: die Präzision oberhalb einer Schwelle ``t`` ist ``E[k | k >= t] =
  (t + 1) / 2``. Das gesuchte τ für 0,98 ist damit **0,96** – vorher ausgerechnet, nicht
  aus dem Ergebnis abgelesen. Ebenso ist die ganze Kurve bekannt: bei Abdeckung ``c``
  beträgt die Präzision ``1 − c / 4``.
* Für δ Normalverteilungen, deren 95. Perzentil mit ``μ + 1,6449 σ`` feststeht.
* Für die OOD-Abstände Punktmengen, deren Kosinusabstand sich von Hand ausrechnen lässt
  (0 bei gleicher Richtung, 1 bei rechtem Winkel, 2 bei Gegenrichtung).

**Abweichung vom Aufgabentext.** Der Plan sieht als k-Bindung einen Streuungsvergleich
vor (``std`` bei ``k=1`` größer als bei ``k=10``, Punktwolken aus derselben Verteilung).
Diese Behauptung ist auf dem hier gebauten Maß nachweislich **falsch**; siehe
``test_ood_nutzt_wirklich_k_nachbarn`` und den Bericht zur Aufgabe. An ihre Stelle tritt
eine Bindung an den *Wert*: bei bekannter Geometrie sind die Abstände für ``k=1``,
``k=2`` und ``k=10`` von Hand ausrechenbar und verschieden.
"""

import math
from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from doccls.decide import (
    decide_one,
    delta_for_percentile,
    ood_scores,
    risk_coverage,
    tau_for_precision,
)
from doccls.models import VERTEILUNGS_TOLERANZ, Decision, Prediction

#: Analytisch gesuchtes τ für das Ziel 0,98 auf ``_konfidenz_mit_rauschen``:
#: ``(t + 1) / 2 = 0,98`` ⇒ ``t = 0,96``.
TAU_SOLL = 0.96


def _konfidenz_mit_rauschen(
    seed: int, n: int = 20_000
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Ein perfekt geeichtes Modell – die Fehlerstruktur steht damit vorab fest.

    ``konfidenz ~ U(0,5; 1,0)``, ``richtig ~ Bernoulli(konfidenz)``. Die Präzision
    oberhalb einer Schwelle ``t`` ist dadurch ``E[k | k >= t] = (t + 1) / 2``, das
    gesuchte τ also ``TAU_SOLL``.

    ``n`` ist bewusst groß. Gemessen mit ``n = 1000`` schwankt das abgelesene τ über die
    Seeds zwischen 0,917 und 0,994, und bei Seed 2 hielten nur noch acht Dokumente das
    Ziel – die Schwelle hinge dann am Zufall statt an der Verteilung. Mit ``n = 20000``
    liegt sie über die Seeds 1 bis 12 zwischen 0,956 und 0,971, also stabil um 0,96.
    """
    rng = np.random.default_rng(seed)
    konfidenz: npt.NDArray[np.float64] = rng.uniform(0.5, 1.0, n)
    richtig: npt.NDArray[np.bool_] = rng.random(n) < konfidenz
    return konfidenz, richtig


def _wolke(
    mittelpunkt: Sequence[float], anzahl: int, seed: int, streuung: float = 0.15
) -> npt.NDArray[np.float64]:
    """Eine Gauß-Wolke um ``mittelpunkt`` – Punkte einer Richtung im Merkmalsraum."""
    rng = np.random.default_rng(seed)
    mitte = np.asarray(mittelpunkt, dtype=np.float64)
    return mitte + rng.normal(0.0, streuung, (anzahl, mitte.size))


# --------------------------------------------------------------------------- Kurve


def test_risk_coverage_trifft_die_bekannte_kurve() -> None:
    """Bindet die Kurve an die Verteilung, nicht nur an ihre Form.

    Für ``_konfidenz_mit_rauschen`` gilt analytisch ``Praezision(c) = 1 − c / 4``. Ein
    Test, der nur „fällt monoton" prüfte, bliebe grün, wenn die Präzision um einen festen
    Betrag danebenläge – genau der Fehler, der eine Schwelle mitten in die Verteilung
    setzt.
    """
    konfidenz, richtig = _konfidenz_mit_rauschen(seed=5)
    abdeckung, praezision = risk_coverage(konfidenz, richtig)
    for ziel_abdeckung in (0.2, 0.5, 0.8, 1.0):
        stelle = int(np.argmin(np.abs(abdeckung - ziel_abdeckung)))
        erwartet = 1.0 - abdeckung[stelle] / 4.0
        assert abs(praezision[stelle] - erwartet) < 0.01, (
            f"Bei Abdeckung {abdeckung[stelle]:.3f} steht {praezision[stelle]:.4f}, "
            f"analytisch erwartet ist {erwartet:.4f}"
        )


def test_risk_coverage_endet_bei_voller_abdeckung() -> None:
    """Die unterste Schwelle lässt alles durch; dort ist die Präzision die Trefferquote."""
    konfidenz, richtig = _konfidenz_mit_rauschen(seed=5)
    abdeckung, praezision = risk_coverage(konfidenz, richtig)
    assert abdeckung[-1] == pytest.approx(1.0)
    assert praezision[-1] == pytest.approx(float(richtig.mean()))


def test_risk_coverage_steigt_streng_in_der_abdeckung() -> None:
    """Je Schwelle genau ein Punkt, und keine Schwelle doppelt.

    Gleiche Konfidenzen dürfen nicht zu zwei Punkten führen: Eine Schwelle, die zwei
    Dokumente mit demselben Wert trennt, gibt es nicht.
    """
    konfidenz = np.array([0.9, 0.9, 0.8, 0.7, 0.7, 0.7])
    richtig = np.array([True, True, False, True, False, True])
    abdeckung, praezision = risk_coverage(konfidenz, richtig)
    assert list(abdeckung) == pytest.approx([2 / 6, 3 / 6, 1.0])
    assert list(praezision) == pytest.approx([1.0, 2 / 3, 4 / 6])


# --------------------------------------------------------------------------- tau


def test_tau_erreicht_das_praezisionsziel() -> None:
    """Die Zusicherung, auf der Coverage@P98 beruht: Oberhalb von tau muss die Praezision
    das Ziel halten. Haelt sie es nicht, ist jede Aussage ueber automatisch entschiedene
    Dokumente falsch."""
    konfidenz, richtig = _konfidenz_mit_rauschen(seed=5)
    tau = tau_for_precision(konfidenz, richtig, target=0.98)
    oberhalb = konfidenz >= tau
    assert oberhalb.sum() > 0, "Kein Dokument oberhalb von tau – die Schwelle ist unbrauchbar"
    assert richtig[oberhalb].mean() >= 0.98


def test_tau_ist_das_kleinste_das_das_ziel_haelt() -> None:
    """Ein zu hohes tau haelt das Ziel auch – und schickt unnoetig viele Dokumente in die
    Pruefliste. Ohne diesen Test bestuende eine Umsetzung, die schlicht 1,0 liefert."""
    konfidenz, richtig = _konfidenz_mit_rauschen(seed=5)
    tau = tau_for_precision(konfidenz, richtig, target=0.98)
    kleiner = [k for k in np.unique(konfidenz) if k < tau]
    for kandidat in kleiner[-5:]:
        oberhalb = konfidenz >= kandidat
        assert richtig[oberhalb].mean() < 0.98, (
            f"tau={tau:.4f} ist nicht minimal – {kandidat:.4f} haelt das Ziel auch"
        )


def test_tau_liegt_am_analytisch_bekannten_ort() -> None:
    """Der Test, der τ an die **Verteilung** bindet statt nur an sich selbst.

    Die beiden Tests darüber sind Ordnungsaussagen: Sie blieben grün, wenn τ um 0,03
    danebenläge, solange es nur irgendwo im oberen Teil der Wolke sitzt – genau die Lücke,
    durch die in Phase 1 eine Schranke mitten in der realen Verteilung gerutscht ist. Hier
    steht der Sollwert 0,96 **vor** dem Lauf fest (Moduldoc), und eine Abweichung von mehr
    als 0,01 ist ein Befund.
    """
    konfidenz, richtig = _konfidenz_mit_rauschen(seed=5)
    tau = tau_for_precision(konfidenz, richtig, target=0.98)
    assert abs(tau - TAU_SOLL) < 0.01, f"tau={tau:.4f} statt analytisch {TAU_SOLL}"


def test_unerreichbares_ziel_wird_gemeldet() -> None:
    """Bei einem durchweg schlechten Modell gibt es kein tau, das 98 Prozent haelt. Eine
    stillschweigend zurueckgegebene 1,0 sähe wie eine strenge Schwelle aus und waere eine
    Luege."""
    konfidenz = np.linspace(0.3, 0.9, 50)
    richtig = np.zeros(50, dtype=bool)
    with pytest.raises(ValueError, match="Praezision"):
        tau_for_precision(konfidenz, richtig, target=0.98)


def test_alle_richtig_ergibt_die_kleinste_beobachtete_konfidenz() -> None:
    """Der Fall, den der Aufgabentext offenlässt: kein einziger Fehler.

    Dann hält jede Schwelle jedes Ziel, und die kleinste ist die unterste beobachtete
    Konfidenz – volle Abdeckung. Das ist die richtige Antwort und keine Entartung: Die
    Funktion liest ab, was dasteht. Dass „keine Fehler beobachtet" nicht „keine Fehler zu
    erwarten" heißt, ist eine Frage der Stichprobengröße und wird von der Mindestbelegung
    getragen, nicht von einer Sonderregel hier.
    """
    konfidenz = np.linspace(0.5, 1.0, 100)
    richtig = np.ones(100, dtype=bool)
    assert tau_for_precision(konfidenz, richtig, target=0.98) == pytest.approx(0.5)


def test_mindestbelegung_wirft_bei_einer_schwelle_aus_zu_wenig_dokumenten() -> None:
    """Die Verletzungsrichtung der Mindestbelegung, an der exakten Zahl aufgehängt.

    Zwei Grenzfälle, die einen zu kleinen Wächter von beiden Seiten einfangen:

    * 200 Dokumente, die 48 sichersten richtig, der Rest falsch. Oberhalb der Schwelle mit
      48 Dokumenten steht Präzision 1,0 – das Ziel wäre „erreicht", abgelesen aus 48
      Dokumenten, in denen 0,98 gar nicht darstellbar ist. Bei 49 wären es 48/49 = 0,9796.
    * 49 Dokumente, alle richtig. Jede Schwelle hält 1,0, keine erreicht die Belegung 50.
      Dieser Fall unterscheidet einen Wächter mit 49 von einem mit 50 – der erste gäbe hier
      eine Schwelle zurück.
    """
    konfidenz = np.linspace(1.0, 0.0, 200)
    richtig = np.zeros(200, dtype=bool)
    richtig[:48] = True
    with pytest.raises(ValueError, match="Mindestbelegung"):
        tau_for_precision(konfidenz, richtig, target=0.98)

    with pytest.raises(ValueError, match="Mindestbelegung"):
        tau_for_precision(np.linspace(1.0, 0.0, 49), np.ones(49, dtype=bool), target=0.98)


def test_mindestbelegung_laesst_genau_fuenfzig_dokumente_durch() -> None:
    """Die Gegenrichtung, ein Dokument weiter – sonst wäre der Wächter nur eine Sperre.

    Jetzt sind die 49 sichersten richtig und das 50. falsch: 49/50 = 0,98 hält das Ziel
    bei genau der Belegung, die ``1 / (1 − 0,98) = 50`` verlangt. Zusammen mit dem Test
    darüber ist die abgeleitete Zahl 50 von beiden Seiten gebunden: ein Wächter mit 48 oder
    49 fällt oben auf, einer mit 51 hier.
    """
    konfidenz = np.linspace(1.0, 0.0, 200)
    richtig = np.zeros(200, dtype=bool)
    richtig[:49] = True
    tau = tau_for_precision(konfidenz, richtig, target=0.98)
    oberhalb = konfidenz >= tau
    assert int(oberhalb.sum()) == 50
    assert float(richtig[oberhalb].mean()) == pytest.approx(0.98)


def test_mindestbelegung_ist_aus_dem_ziel_abgeleitet_nicht_gesetzt() -> None:
    """Bindet die Mindestbelegung an ein **zweites** Ziel – sonst ist sie nur eine Zahl.

    Die beiden Tests darüber prüfen ausschließlich den Vorgabewert 0,98 und seine 50. Ein
    festes ``return 50`` in :func:`doccls.decide._mindestbelegung` bliebe damit unbemerkt,
    und die tragende Behauptung des Moduldocs – die Zahl sei *aus dem Ziel abgeleitet* –
    wäre unbelegt. Genau durch diese Lücke ist der Fehler in der Zwillingsfunktion
    ``_mindestzahl_fuer_perzentil`` (zu streng für 80 und 90) unentdeckt geblieben.

    Von Hand nachgerechnet für ``target = 0,9``: gesucht ist das kleinste ``n`` mit
    ``(n − 1) / n >= 0,9``, also ``n = 10`` (9/10 = 0,9 hält, 8/9 = 0,889 hält nicht). Alle
    drei Richtungen werden geprüft:

    * 200 Dokumente, die 8 sichersten richtig: oberhalb der Schwelle mit 8 Dokumenten steht
      Präzision 1,0, bei 9 sind es 8/9 = 0,889. Ein Wächter mit 8 gäbe hier eine Schwelle
      zurück.
    * 9 Dokumente, alle richtig: jede Schwelle hält 1,0, keine erreicht die Belegung 10 –
      das unterscheidet einen Wächter mit 9 von einem mit 10. Die Meldung muss die
      abgeleitete Zahl nennen, sonst bliebe ein festes 50 auch hier unsichtbar.
    * 200 Dokumente, die 9 sichersten richtig: 9/10 = 0,9 hält das Ziel bei genau der
      Belegung 10. Ein Wächter mit 11 – oder ein festes 50 – fällt hier auf.
    """
    konfidenz = np.linspace(1.0, 0.0, 200)
    acht_richtig = np.zeros(200, dtype=bool)
    acht_richtig[:8] = True
    with pytest.raises(ValueError, match="noetigen 10"):
        tau_for_precision(konfidenz, acht_richtig, target=0.9)

    with pytest.raises(ValueError, match="noetigen 10"):
        tau_for_precision(np.linspace(1.0, 0.0, 9), np.ones(9, dtype=bool), target=0.9)

    neun_richtig = np.zeros(200, dtype=bool)
    neun_richtig[:9] = True
    tau = tau_for_precision(konfidenz, neun_richtig, target=0.9)
    oberhalb = konfidenz >= tau
    assert int(oberhalb.sum()) == 10
    assert float(neun_richtig[oberhalb].mean()) == pytest.approx(0.9)


def test_mindestbelegung_laesst_den_normalfall_ungehindert_durch() -> None:
    """Der Normalfall darf vom Wächter nicht angefasst werden.

    Auf der gut besetzten Kurve steht das abgelesene τ weit oberhalb der Mindestbelegung –
    ein Wächter, der hier zuschlüge, machte die Funktion unbrauchbar.
    """
    konfidenz, richtig = _konfidenz_mit_rauschen(seed=5)
    tau = tau_for_precision(konfidenz, richtig, target=0.98)
    assert int((konfidenz >= tau).sum()) > 50


def test_ziel_ausserhalb_von_null_bis_eins_wird_abgelehnt() -> None:
    """Ein Ziel von 1,0 ist auf endlich vielen Dokumenten nicht ablesbar: Es verlangte
    unendlich viele Belege, und ``1 / (1 − 1)`` ist keine Zahl."""
    konfidenz, richtig = _konfidenz_mit_rauschen(seed=5, n=100)
    for ziel in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="Ziel"):
            tau_for_precision(konfidenz, richtig, target=ziel)


def test_ungleich_lange_eingaben_werden_abgelehnt() -> None:
    """Ohne diese Prüfung liefe ein Versatz zwischen Konfidenz und Wahrheit lautlos durch:
    Jede Präzision bezöge sich dann auf die Trefferliste eines anderen Dokuments."""
    with pytest.raises(ValueError, match="Dokument"):
        risk_coverage(np.array([0.9, 0.8, 0.7]), np.array([True, False]))


def test_wahrheitswerte_muessen_boolesch_sein() -> None:
    """Ein Zahlenfeld statt eines booleschen ließe sich mitteln und ergäbe eine
    „Präzision", die keine ist.

    Der Fall ist nicht theoretisch: ``y_pred == y_true`` liefert booleschen Typ, eine
    Gewichtung oder eine Teilwertung (0,5 für „halb richtig") liefert Gleitkomma. Beides
    liefe durch ``mean()`` hindurch, und τ stünde danach auf einer Größe, die niemand
    Präzision genannt hätte.

    Das ``cast`` ist der Kern des Tests, keine Umgehung: Es sagt genau aus, dass hier
    bewusst etwas übergeben wird, was die Annotation ausschließt. mypy sieht nur
    annotierte Aufrufstellen; ein Zahlenfeld kommt aus Polars, aus einer Gewichtung oder
    aus einem ungeprüften Rand – und der Wächter steht für diesen Weg, nicht für den
    statisch geprüften.
    """
    with pytest.raises(ValueError, match="boolesches Feld"):
        risk_coverage(
            np.array([0.9, 0.8, 0.7]),
            cast(npt.NDArray[np.bool_], np.array([1.0, 0.5, 0.0])),
        )


def test_nicht_endliche_konfidenzen_werden_abgelehnt() -> None:
    """``NaN`` sortiert sich ans Ende und zöge die Kurve stillschweigend schief."""
    with pytest.raises(ValueError, match="endlich"):
        risk_coverage(np.array([0.9, np.nan, 0.7]), np.array([True, False, True]))


# --------------------------------------------------------------------------- OOD


def test_ood_abstand_ist_gross_fuer_fremdes_dokument() -> None:
    """Der Zweck der OOD-Pruefung (Konzept § 7.4): Ein lineares Modell kann auf einem
    Dokument, das keiner Klasse aehnelt, trotzdem selbstbewusst sein."""
    trainings_vektoren = _wolke(mittelpunkt=[1.0, 0.0], anzahl=50, seed=2)
    nah = _wolke(mittelpunkt=[1.0, 0.0], anzahl=5, seed=3)
    fern = _wolke(mittelpunkt=[-1.0, 0.0], anzahl=5, seed=4)
    assert ood_scores(fern, trainings_vektoren).mean() > ood_scores(nah, trainings_vektoren).mean()


def test_ood_trifft_den_von_hand_gerechneten_kosinusabstand() -> None:
    """Bindet die Formel an Zahlen, nicht nur an eine Rangfolge.

    Bei einem Trainingsbestand, der ganz auf ``[1, 0]`` liegt, ist der Abstand von Hand
    bekannt: 0 in gleicher Richtung, 1 im rechten Winkel, 2 in Gegenrichtung. Eine
    Umsetzung, die etwa den euklidischen Abstand rechnet oder das Vorzeichen dreht, kommt
    hier nicht durch.
    """
    trainings_vektoren = np.tile([1.0, 0.0], (12, 1))
    probe = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [5.0, 0.0]])
    assert ood_scores(probe, trainings_vektoren) == pytest.approx([0.0, 1.0, 2.0, 0.0])


def test_ood_nutzt_wirklich_k_nachbarn() -> None:
    """Bindet ``k`` an den **Wert**, nicht an eine Streuung.

    Der Aufgabentext sieht hier einen Streuungsvergleich vor (``std`` bei ``k=1`` größer
    als bei ``k=10``, beide Wolken aus derselben Verteilung). Nachgemessen trifft das für
    dieses Maß nicht zu, und zwar systematisch: Der Mittelwert der ``k`` nächsten Punkte
    schrumpft zur lokalen Mitte hin, und *wie stark* er das tut, hängt davon ab, wo im
    Bestand der Prüfpunkt liegt – dieser Anteil schwankt stärker als das Zittern des einen
    nächsten Nachbarn, der immer der nächste ist. Gemessen über Wolken mit 2 bis 768
    Dimensionen, Streuungen von 0,03 bis 1,0, in Grüppchen geteilte und schwerschwänzige
    Bestände und je 30 Seeds liegt der Quotient ``std(k=1) / std(k=10)`` im Median bei
    0,46 bis 1,11 und ist nur in Randlagen über 1. Ein Test darauf hinge am Seed.

    Hier stattdessen eine Geometrie, in der beide Werte von Hand feststehen: ein
    Trainingspunkt auf ``[1, 0]``, neun auf ``[0, 1]``, geprüft wird ``[1, 0]``.
    ``k=1`` trifft den einen gleichgerichteten Punkt (Abstand 0), ``k=2`` mittelt ihn mit
    einem rechtwinkligen (``1 − 1/√2``), ``k=10`` mit allen neun (``1 − 1/√82``). Eine
    Umsetzung, die ``k`` ignoriert und immer alle Nachbarn mittelt, liefert dreimal
    denselben Wert und fällt auf.
    """
    trainings_vektoren = np.vstack([np.array([[1.0, 0.0]]), np.tile([0.0, 1.0], (9, 1))])
    probe = np.array([[1.0, 0.0]])
    assert float(ood_scores(probe, trainings_vektoren, k=1)[0]) == pytest.approx(0.0)
    assert float(ood_scores(probe, trainings_vektoren, k=2)[0]) == pytest.approx(
        1.0 - 1.0 / math.sqrt(2.0)
    )
    assert float(ood_scores(probe, trainings_vektoren, k=10)[0]) == pytest.approx(
        1.0 - 1.0 / math.sqrt(82.0)
    )


def test_ood_haengt_nicht_an_der_vektorlaenge() -> None:
    """Bindet die L2-Normierung – die Mutation, die sonst nirgends auffiele.

    Ohne Normierung entscheidet nicht der Winkel, sondern das Skalarprodukt, welche
    Nachbarn die nächsten sind, und der Mittelwert wird vom längsten Vektor beherrscht.
    Auf Embeddings ist das kein theoretischer Fall: Ein längerer Dokumentausschnitt
    erzeugt einen längeren Vektor bei gleicher Richtung. Gemessen verschieben zufällige
    Längenfaktoren zwischen 0,2 und 5 die Abstände ohne Normierung um bis zu 0,009 – mit
    Normierung um 1,1e-16.
    """
    trainings_vektoren = _wolke(mittelpunkt=[1.0, 0.0], anzahl=50, seed=2)
    probe = _wolke(mittelpunkt=[1.0, 0.0], anzahl=20, seed=9)
    rng = np.random.default_rng(11)
    gestreckt_training = trainings_vektoren * rng.uniform(0.2, 5.0, (len(trainings_vektoren), 1))
    gestreckte_probe = probe * rng.uniform(0.2, 5.0, (len(probe), 1))
    vorher = ood_scores(probe, trainings_vektoren)
    nachher = ood_scores(gestreckte_probe, gestreckt_training)
    assert float(np.abs(vorher - nachher).max()) < 1e-9


def test_ood_lehnt_mehr_nachbarn_ab_als_es_gibt() -> None:
    """Stillschweigend auf die vorhandene Zahl zu kürzen hieße, ein anderes ``k`` zu
    rechnen als das angefragte – und die Schwelle δ wäre dann auf einer anderen Größe
    abgelesen als die späteren Abstände."""
    with pytest.raises(ValueError, match="Nachbarn"):
        ood_scores(np.array([[1.0, 0.0]]), np.tile([1.0, 0.0], (5, 1)), k=10)


def test_ood_lehnt_richtungslose_vektoren_ab() -> None:
    """Ein Nullvektor hat keine Richtung; der Kosinus teilte durch null und ergäbe ``nan``,
    das sich durch jede spätere Perzentilbildung fräße."""
    with pytest.raises(ValueError, match="Nullvektor"):
        ood_scores(np.array([[0.0, 0.0]]), np.tile([1.0, 0.0], (12, 1)), k=10)


def test_ood_lehnt_ein_richtungsloses_mittel_ab() -> None:
    """Der seltenere Fall: Die ``k`` nächsten heben sich gegenseitig auf. Das Mittel ist
    dann der Nullvektor und hat keine Richtung – rechenbar, aber ohne Bedeutung."""
    entgegengesetzt = np.array([[1.0, 0.0], [-1.0, 0.0]])
    with pytest.raises(ValueError, match="Nullvektor"):
        ood_scores(np.array([[0.0, 1.0]]), entgegengesetzt, k=2)


# --------------------------------------------------------------------------- delta


def test_delta_liegt_auf_dem_perzentil() -> None:
    werte = np.linspace(0.0, 1.0, 101)
    assert abs(delta_for_percentile(werte, percentile=95.0) - 0.95) < 0.01


def test_delta_wird_nicht_geraten_sondern_gemessen() -> None:
    """Konzept § 7.4 und Anhang D: Kosinuswerte sind stark gestaucht (0,79 bis 0,98).
    Zwei verschieden verteilte Mengen muessen deutlich verschiedene delta ergeben – eine
    fest verdrahtete Konstante faellt hier auf."""
    eng = np.random.default_rng(1).normal(0.10, 0.01, 500)
    weit = np.random.default_rng(1).normal(0.40, 0.08, 500)
    assert abs(delta_for_percentile(weit) - delta_for_percentile(eng)) > 0.1


def test_delta_trifft_das_bekannte_perzentil_der_verteilung() -> None:
    """Der Test, der δ an die **Verteilung** bindet, nicht nur an einen Unterschied.

    Der Test darüber bliebe grün, wenn δ um 0,05 danebenläge – bei Kosinuswerten, die
    laut Anhang D zwischen 0,79 und 0,98 gedrängt liegen, ist das die halbe Spannweite.
    Das 95. Perzentil einer Normalverteilung steht vorab fest: ``μ + 1,6449 σ``.
    """
    mittel, streuung = 0.40, 0.08
    werte = np.random.default_rng(1).normal(mittel, streuung, 5000)
    erwartet = mittel + 1.6448536 * streuung
    assert abs(delta_for_percentile(werte, percentile=95.0) - erwartet) < 0.01


def test_delta_bindet_auch_ein_anderes_perzentil() -> None:
    """Bindet den Parameter selbst: Eine Umsetzung, die ``percentile`` ignoriert und immer
    95 rechnet, liefert hier denselben Wert wie oben."""
    mittel, streuung = 0.40, 0.08
    werte = np.random.default_rng(1).normal(mittel, streuung, 5000)
    erwartet = mittel + 1.2815516 * streuung
    assert abs(delta_for_percentile(werte, percentile=90.0) - erwartet) < 0.01


def test_delta_wirft_bei_zu_wenigen_werten() -> None:
    """Verletzungsrichtung der Mindestzahl, an der exakten Grenze aufgehängt.

    Über dem 95. Perzentil liegen 5 Prozent der Verteilung. Bei 19 Werten ist das im
    Mittel weniger als ein einziger beobachteter Wert: Die Schwelle wird dann aus dem
    obersten Punkt der Stichprobe hochgerechnet statt aus der Verteilung abgelesen.
    """
    with pytest.raises(ValueError, match="Mindestzahl"):
        delta_for_percentile(np.linspace(0.0, 1.0, 19), percentile=95.0)


def test_delta_laesst_genau_zwanzig_werte_durch() -> None:
    """Die Gegenrichtung, ein Wert weiter – ``100 / (100 − 95) = 20``. Die beiden Tests
    zusammen binden die abgeleitete Zahl von beiden Seiten."""
    assert delta_for_percentile(np.linspace(0.0, 1.0, 20), percentile=95.0) == pytest.approx(
        0.95, abs=0.01
    )


def test_mindestzahl_ist_aus_dem_perzentil_abgeleitet_nicht_gesetzt() -> None:
    """Bindet die Mindestzahl an **zwei weitere** Perzentile – der Test, der fehlte.

    Die beiden Tests darüber prüfen nur den Vorgabewert 95 und seine 20. Ein festes
    ``return 20`` in :func:`doccls.decide._mindestzahl_fuer_perzentil` bliebe unbemerkt,
    und genau durch diese Lücke ist ein Rechenfehler durchgerutscht: ``1 − percentile/100``
    ergibt für 90 in Gleitkomma 0,09999999999999998 statt 0,1, und weil beide
    Korrekturschleifen mit demselben verlorenen Wert rechnen, kam 11 heraus statt 10.
    Gemessen war die Funktion für 80 und 90 um eins zu streng, für 95, 99 und 99,9 richtig.

    Von Hand nachgerechnet – das kleinste ``n`` mit ``n * (1 − percentile/100) >= 1``:

    * 90. Perzentil: ``n = 10``. Neun Werte müssen abgelehnt werden, zehn durchgehen.
    * 80. Perzentil: ``n = 5``. Vier abgelehnt, fünf durch.

    Die Sollwerte des Perzentils stehen dabei ebenfalls vorab fest: auf
    ``linspace(0, 1, 10)`` liegt das 90. Perzentil auf 0,9, auf ``linspace(0, 1, 5)`` das
    80. auf 0,8 (lineare Interpolation, siehe
    :func:`test_delta_interpoliert_linear_zwischen_den_nachbarwerten`).
    """
    with pytest.raises(ValueError, match="noetig sind 10"):
        delta_for_percentile(np.linspace(0.0, 1.0, 9), percentile=90.0)
    assert delta_for_percentile(np.linspace(0.0, 1.0, 10), percentile=90.0) == pytest.approx(
        0.9, abs=0.01
    )

    with pytest.raises(ValueError, match="noetig sind 5"):
        delta_for_percentile(np.linspace(0.0, 1.0, 4), percentile=80.0)
    assert delta_for_percentile(np.linspace(0.0, 1.0, 5), percentile=80.0) == pytest.approx(
        0.8, abs=0.01
    )


def test_delta_interpoliert_linear_zwischen_den_nachbarwerten() -> None:
    """Bindet die Interpolationsart – bei 20 Werten sitzt δ zwischen zwei Beobachtungen.

    ``np.percentile`` kennt mehrere Verfahren, und die Voreinstellung ``linear`` ist eine
    Wahl, keine Naturkonstante. Auf ``linspace(0, 1, 20)`` und dem 95. Perzentil liegt der
    gesuchte Rang bei ``19 * 0,95 = 18,05``, also 5 Prozent des Wegs vom zweitobersten Wert
    (0,947368) zum obersten (1,0) – gemessen: ``linear`` 0,95, ``lower`` 0,947368,
    ``nearest`` 0,947368, ``midpoint`` 0,973684, ``higher`` 1,0.

    Die bestehenden δ-Tests laufen mit einer Toleranz von 0,01 und unterscheiden ``linear``
    nicht von ``lower`` – eine Mutation zu ``lower`` blieb grün. Hier steht der exakte Wert
    0,95, und jedes andere Verfahren fällt auf.
    """
    werte = np.linspace(0.0, 1.0, 20)
    assert delta_for_percentile(werte, percentile=95.0) == pytest.approx(0.95, abs=1e-12)


def test_delta_lehnt_unmoegliche_perzentile_ab() -> None:
    """0 und 100 sind Minimum und Maximum, keine Perzentile: Ein Schwellwert, der auf dem
    größten je gesehenen Wert sitzt, lehnt per Bauart nichts ab."""
    werte = np.linspace(0.0, 1.0, 101)
    for perzentil in (0.0, 100.0, -1.0, 101.0):
        with pytest.raises(ValueError, match="Perzentil"):
            delta_for_percentile(werte, percentile=perzentil)


# ------------------------------------------------------------------- Entscheidung

KLASSEN: tuple[str, ...] = ("RECHNUNG", "GUTSCHRIFT", "VERTRAG")
"""Drei trainierte Klassen in der Spaltenreihenfolge von ``TrainedModel.classes_``. Die
Restklasse ``SONSTIGES`` steht bewusst **nicht** darin: Sie wird nicht trainiert und hat
keine Spalte in ``predict_proba``."""


def _entscheide(
    proba: Sequence[float] | npt.NDArray[np.float64],
    *,
    classes: Sequence[str] = KLASSEN,
    ood_score: float = 0.0,
    tau: float = 0.9,
    delta: float = 0.5,
) -> Prediction:
    """``decide_one`` mit Herkunftsangaben, die kein Test hier variiert."""
    return decide_one(
        np.asarray(proba, dtype=np.float64),
        classes,
        ood_score=ood_score,
        tau=tau,
        delta=delta,
        document_id="a3f1",
        model_version="clf-test",
        feature_version="feat-test",
    )


def test_hohe_konfidenz_und_bekanntes_dokument_ergibt_auto() -> None:
    p = _entscheide([0.95, 0.03, 0.02], ood_score=0.1, tau=0.9, delta=0.5)
    assert p.decision is Decision.AUTO and p.class_key == "RECHNUNG"


def test_niedrige_konfidenz_ergibt_review() -> None:
    p = _entscheide([0.5, 0.3, 0.2], ood_score=0.1, tau=0.9, delta=0.5)
    assert p.decision is Decision.REVIEW and p.class_key == "RECHNUNG"


def test_ood_schlaegt_hohe_konfidenz() -> None:
    """Der eigentliche Zweck der OOD-Prüfung: Ein Dokument jenseits von delta wird
    SONSTIGES – auch bei 0,99 Konfidenz. Das lineare Modell kennt nur seine trainierten
    Klassen und verteilt die Masse auf sie."""
    p = _entscheide([0.99, 0.005, 0.005], ood_score=0.9, tau=0.9, delta=0.5)
    assert p.class_key == "SONSTIGES" and p.decision is Decision.REVIEW
    # Die drei Zahlen gehoeren zur Verteilung, nicht zur Entscheidung: Auch auf dem
    # OOD-Pfad beschreiben sie, was das Modell gesagt hat. Ohne diese Zeilen bindet kein
    # Test sie hier -- gemessen: Ein "konfidenz, margin, entropie = 1.0, 1.0, 0.0" im
    # OOD-Zweig liess alle 339 Tests gruen. Das ist nicht kosmetisch: Phase 3 waehlt die
    # Pruefliste nach der Margin aus, und SONSTIGES-Dokumente sind gerade die, die
    # dorthin gehen.
    assert p.confidence == pytest.approx(0.99)
    assert p.margin == pytest.approx(0.985)
    assert p.entropy == pytest.approx(0.06293300616044681)


def test_ood_gewinnt_wenn_beide_pruefungen_greifen() -> None:
    """Die **Reihenfolge** der beiden Prüfungen, und nur dieser Test bindet sie.

    Der Test darüber tut es nicht: Bei Konfidenz 0,99 und tau 0,9 greift die
    Konfidenzprüfung gar nicht, und eine Umsetzung, die sie zuerst stellt, käme über
    denselben Umweg zum selben Ergebnis. Erst wenn **beide** Prüfungen greifen, trennen
    sich die Ergebnisse: 0,5 liegt unter tau 0,9 *und* 0,9 über delta 0,5. Die Regel aus
    § 7.4 verlangt hier ``SONSTIGES``; wer die Konfidenz zuerst prüft, liefert
    ``RECHNUNG`` – beide Male mit ``REVIEW``, weshalb die Entscheidung allein nichts
    verrät.
    """
    p = _entscheide([0.5, 0.3, 0.2], ood_score=0.9, tau=0.9, delta=0.5)
    assert p.class_key == "SONSTIGES" and p.decision is Decision.REVIEW


def test_margin_trennt_zwei_kandidaten_von_breiter_unsicherheit() -> None:
    """Konzept § 7.2: 0,45/0,44/0,11 sitzt auf der Grenze zwischen zwei Klassen und ist
    beim Labeln wertvoller als eine breite Unsicherheit. Konfidenz allein sieht das nicht –
    genau deshalb wird die Margin mitgeschrieben."""
    eng = _entscheide([0.45, 0.44, 0.11])
    breit = _entscheide([0.40, 0.30, 0.30])
    assert eng.margin < breit.margin
    assert eng.entropy < breit.entropy


def test_margin_ist_der_abstand_zur_zweitbesten_klasse() -> None:
    """Bindet die Formel an den Wert, nicht nur an ihre Richtung.

    Bei 0,5/0,3/0,2 sind alle drei naheliegenden Verwechslungen unterscheidbar: die beste
    Wahrscheinlichkeit (0,5), die zweitbeste (0,3), der Abstand zur *drittbesten* (0,3) –
    und der richtige Abstand zur zweitbesten (0,2).
    """
    p = _entscheide([0.5, 0.3, 0.2])
    assert p.confidence == pytest.approx(0.5)
    assert p.margin == pytest.approx(0.2)


def test_entropie_trifft_die_formel() -> None:
    """Bindet die Formel, nicht nur ihre Richtung: Bei drei gleich wahrscheinlichen
    Klassen ist die Entropie ln(3). Mit ``log10`` stünden dort 0,477."""
    p = _entscheide([1 / 3, 1 / 3, 1 / 3])
    assert abs(p.entropy - math.log(3)) < 1e-6


def test_entropie_bleibt_endlich_bei_wahrscheinlichkeit_null() -> None:
    """``0 · log 0`` ist 0, nicht ``nan``.

    Nach der Kalibrierung ist eine glatte Null selten, aber nicht unmöglich. Ein naives
    ``p * np.log(p)`` ergäbe hier ``nan``, und das fräße sich durch jede spätere Mittelung
    – eine mittlere Entropie über alle Dokumente wäre dann ``nan``, ohne dass eine einzige
    Vorhersage erkennbar falsch wäre.
    """
    p = _entscheide([1.0, 0.0, 0.0], tau=0.5)
    assert p.entropy == pytest.approx(0.0)
    assert p.decision is Decision.AUTO


def test_gleichstand_faellt_auf_die_vorderste_klasse() -> None:
    """Bei exakt gleicher Wahrscheinlichkeit gewinnt die vordere Spalte – wie ``argmax``.

    Der zweite Fall bindet ``kind="stable"`` in der Sortierung, und er braucht seine 17
    Klassen: NumPys Vorgabeverfahren sortiert Felder unter 16 Einträgen mit einem
    Einfügeverfahren, das ohnehin stabil ist – gemessen über alle Muster aus {0; 0,25;
    0,5} bis Länge 11 gibt es keinen Unterschied, und mit den sechs trainierten Klassen
    dieses Projekts käme man nie an einen. Erst darüber greift Introsort und liefert bei
    Gleichstand die Klasse 8 statt der Klasse 1. Ohne ``stable`` hängt der Name also an
    der Feldlänge und an der inneren Ordnung von NumPy; beides ist keine Zusage, auf die
    sich eine Vorhersage stützen darf.
    """
    nah_beieinander = _entscheide([0.5, 0.5, 0.0], tau=0.4)
    assert nah_beieinander.class_key == "RECHNUNG"
    assert nah_beieinander.margin == pytest.approx(0.0)

    viele = tuple(f"KLASSE_{i:02d}" for i in range(17))
    gleichstand = _entscheide([0.0] + [1 / 16] * 16, classes=viele, tau=0.0)
    assert gleichstand.class_key == "KLASSE_01"
    assert gleichstand.confidence == pytest.approx(1 / 16)


@pytest.mark.parametrize(("tau", "erwartet"), [(0.5, Decision.AUTO), (1.0, Decision.REVIEW)])
def test_konfidenz_und_klasse_gehoeren_zusammen(tau: float, erwartet: Decision) -> None:
    """Der lautlose Fehler: argmax über die Wahrscheinlichkeiten, aber der Name aus einer
    anders sortierten Liste. Dann stimmt die Zahl, und der Name stimmt nicht.

    **Beide Zweige, nicht nur der eine.** Der Name wird an zwei Stellen gesetzt – einmal
    für ``AUTO``, einmal für ``REVIEW`` –, und dieselbe Schleife muss durch beide laufen.
    Gemessen, bevor dieser Test parametrisiert war: ``classes[bester]`` im REVIEW-Zweig
    durch ``classes[0]`` ersetzt liess die ganze Suite mit 339 Tests grün, weil jeder Test,
    der den REVIEW-Zweig überhaupt erreichte, eine Verteilung benutzte, deren beste Klasse
    ohnehin auf Index 0 sass. Das ``tau`` von 1,0 liegt über jeder Konfidenz und schickt
    dieselben Verteilungen durch den anderen Zweig.
    """
    for i, klasse in enumerate(KLASSEN):
        proba = np.full(len(KLASSEN), 0.01, dtype=np.float64)
        proba[i] = 1.0 - 0.01 * (len(KLASSEN) - 1)
        p = _entscheide(proba, ood_score=0.0, tau=tau, delta=0.9)
        assert p.class_key == klasse and p.confidence == pytest.approx(proba[i])
        assert p.decision is erwartet


def test_genau_auf_der_schwelle_gilt_als_auto() -> None:
    """Konzept § 7.4 schreibt „konfidenz < tau → REVIEW". Gleichheit ist also AUTO.

    Ohne diesen Test bliebe ein ``<=`` statt ``<`` unbemerkt, und Coverage@P98 wäre leicht
    verschoben – bei einer Kennzahl, die auf zwei Stellen berichtet wird.
    """
    p = _entscheide([0.9, 0.05, 0.05], ood_score=0.0, tau=0.9, delta=0.9)
    assert p.decision is Decision.AUTO


@pytest.mark.parametrize(
    ("tau", "erwartet"),
    [(0.9, Decision.REVIEW), (0.8, Decision.AUTO), (0.7, Decision.AUTO)],
)
def test_die_konfidenzschwelle_wirkt_bei_einem_zweiten_tau(tau: float, erwartet: Decision) -> None:
    """tau ist ein Parameter – eine Umsetzung, die ihn ignoriert und fest 0,9 vergleicht,
    bestünde jeden Test mit nur einem tau.

    Dieselbe Konfidenz 0,8 fällt bei tau 0,9 in die Prüfliste, ist bei 0,7 automatisch und
    bei tau 0,8 – der Gleichheit – ebenfalls automatisch. Damit ist auch das ``<`` an
    einem zweiten Wert gebunden, nicht nur am 0,9 des Tests darüber.
    """
    assert _entscheide([0.8, 0.1, 0.1], ood_score=0.0, tau=tau, delta=0.9).decision is erwartet


@pytest.mark.parametrize(
    ("ood_score", "delta", "ist_sonstiges"),
    [
        (0.5, 0.4, True),
        (0.5, 0.5, False),
        (0.5, 0.6, False),
        (1.2, 1.1, True),
        (1.2, 1.2, False),
    ],
)
def test_die_ood_schwelle_wirkt_bei_einem_zweiten_delta(
    ood_score: float, delta: float, ist_sonstiges: bool
) -> None:
    """delta ist ebenso ein Parameter, und die Regel sagt ``ood_score > δ`` – Gleichheit
    ist also **kein** SONSTIGES.

    Zwei Abstände (0,5 und 1,2) mit je einem delta darunter und einem gleichauf: Ein fest
    verglichenes 0,5 fiele beim zweiten Paar auf, ein ``>=`` statt ``>`` bei beiden
    Gleichheitsfällen. Der Wertebereich bis 2 ist kein Zierrat – ``1 − cos`` reicht so
    weit.
    """
    p = _entscheide([0.95, 0.03, 0.02], ood_score=ood_score, delta=delta)
    assert p.class_key == ("SONSTIGES" if ist_sonstiges else "RECHNUNG")


@pytest.mark.parametrize("anzahl", [4, 6, 7, 20])
def test_was_die_verteilungspruefung_durchlaesst_nimmt_die_vorhersage_an(
    anzahl: int,
) -> None:
    """Die beiden Toleranzen müssen zueinander passen, sonst weist ``Prediction`` ab, was
    ``decide_one`` gerade durchgelassen hat.

    ``_als_verteilung`` erlaubt eine Summe von ``1 ± VERTEILUNGS_TOLERANZ``. Eine um ``ε``
    verkleinerte Verteilung hat aber eine um bis zu ``ε · H`` kleinere Entropie, während
    die Schranke ``−ln(max p)`` nur um ``ε`` steigt. Der Fehlbetrag wächst also mit der
    Klassenzahl und übersteigt ein festes ``TOLERANZ`` ab vier Klassen. Gemessen, bevor
    die Toleranzen gekoppelt waren, bei sieben gleichverteilten Klassen und ``ε = 6e-7``:

        ValidationError: Entropie 1.9459095815090444 ist zu klein fuer die Konfidenz
        0.14285705714285715 ... >= 1.9459107490554932

    Die Gleichverteilung ist hier der schärfste Fall: Sie hat die größte Entropie bei
    gegebener Klassenzahl und damit den größten Fehlbetrag.
    """
    klassen = tuple(f"KLASSE_{i:02d}" for i in range(anzahl))
    knapp_darunter = 1.0 - 0.6 * VERTEILUNGS_TOLERANZ
    p = _entscheide(
        np.full(anzahl, knapp_darunter / anzahl),
        classes=klassen,
        ood_score=0.1,
        tau=0.5,
        delta=0.9,
    )
    assert p.entropy == pytest.approx(math.log(anzahl), abs=1e-5)


def test_herkunft_und_bezug_stehen_in_der_vorhersage() -> None:
    """Modell- und Merkmalsversion dürfen nicht vertauscht durchgereicht werden – sonst
    ordnet jede spätere Auswertung die Vorhersage dem falschen Lauf zu (Konzept § 1)."""
    p = decide_one(
        np.array([0.95, 0.03, 0.02]),
        KLASSEN,
        ood_score=0.1,
        tau=0.9,
        delta=0.5,
        document_id="a3f1",
        model_version="clf-2026-09-25-r07",
        feature_version="feat-0a1b2c3d",
    )
    assert p.document_id == "a3f1"
    assert p.model_version == "clf-2026-09-25-r07"
    assert p.feature_version == "feat-0a1b2c3d"
    assert p.ood_score == pytest.approx(0.1)


def test_nicht_endliche_schwellen_werden_als_solche_gemeldet() -> None:
    """``nan`` fällt zwar auch durch jeden Bereichsvergleich – aber mit der falschen
    Begründung.

    Ein ``ood_score`` von ``nan`` entsteht dort, wo eine Umgebung keine Richtung hat; die
    Meldung soll das sagen und nicht behaupten, die Zahl liege neben einem Intervall.
    Ohne die eigene Prüfung bliebe der Unterschied unsichtbar, weil beide Wege eine
    Ausnahme werfen.
    """
    with pytest.raises(ValueError, match="endliche Zahl"):
        _entscheide([0.9, 0.05, 0.05], ood_score=float("nan"))


def test_proba_und_klassen_muessen_gleich_lang_sein() -> None:
    """Ohne diese Prüfung trüge jede Wahrscheinlichkeit den Namen einer anderen Klasse –
    und bei einer Spalte zu wenig bliebe der Fehler bis in die Auswertung unsichtbar."""
    with pytest.raises(ValueError, match="classes nennt"):
        _entscheide([0.5, 0.5])


@pytest.mark.parametrize(
    "proba",
    [
        [0.5, 0.3, 0.1],  # summiert sich zu 0,9
        [0.5, 0.3, 0.3],  # summiert sich zu 1,1
        [1.5, -0.3, -0.2],  # Summe 1, aber keine Wahrscheinlichkeiten
    ],
)
def test_proba_muss_eine_verteilung_sein(proba: list[float]) -> None:
    """Konfidenz und Entropie auf etwas zu rechnen, das keine Verteilung ist, ergibt
    Zahlen, die aussehen wie Wahrscheinlichkeiten und keine sind.

    Der Fall ist nicht theoretisch: ``classify.decision_scores`` liefert ebenfalls ein
    Feld je Klasse in derselben Reihenfolge und passt an dieselbe Stelle.
    """
    with pytest.raises(ValueError, match="Wahrscheinlichkeitsverteilung"):
        _entscheide(proba)


def test_weniger_als_zwei_klassen_werden_abgelehnt() -> None:
    """Ohne zweitbeste Klasse gibt es keine Margin – und eine Wahl zwischen einer
    einzigen Möglichkeit ist keine."""
    with pytest.raises(ValueError, match="zweitbeste"):
        _entscheide([1.0], classes=("RECHNUNG",))


def test_restklasse_darf_nicht_unter_den_trainierten_klassen_stehen() -> None:
    """Stünde ``SONSTIGES`` in ``classes``, könnte sie als bester Vorschlag mit ``AUTO``
    herauskommen. § 7.4 sieht das nicht vor: Die Restklasse entsteht ausschließlich durch
    Ablehnung und geht immer in die Prüfliste."""
    with pytest.raises(ValueError, match="Restklasse"):
        _entscheide([0.9, 0.05, 0.05], classes=("RECHNUNG", "GUTSCHRIFT", "SONSTIGES"))


@pytest.mark.parametrize(
    ("tau", "delta", "ood_score", "name"),
    [
        (-0.1, 0.5, 0.0, "tau"),
        (1.5, 0.5, 0.0, "tau"),
        (0.9, -0.1, 0.0, "delta"),
        (0.9, 2.5, 0.0, "delta"),
        (0.9, 0.5, -0.1, "ood_score"),
        (0.9, 0.5, 2.5, "ood_score"),
    ],
)
def test_schwellen_ausserhalb_ihres_wertebereichs_werden_abgelehnt(
    tau: float, delta: float, ood_score: float, name: str
) -> None:
    """Ein tau über 1 schickte jedes Dokument in die Prüfliste, ein delta über 2 könnte
    per Bauart nie greifen – beides sähe nach einer strengen Einstellung aus und wäre
    keine. tau ist eine Konfidenz (``[0, 1]``), delta und ood_score sind Kosinusabstände
    (``[0, 2]``)."""
    with pytest.raises(ValueError, match=name):
        _entscheide([0.9, 0.05, 0.05], tau=tau, delta=delta, ood_score=ood_score)
