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
    delta_for_percentile,
    ood_scores,
    risk_coverage,
    tau_for_precision,
)

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


def test_delta_lehnt_unmoegliche_perzentile_ab() -> None:
    """0 und 100 sind Minimum und Maximum, keine Perzentile: Ein Schwellwert, der auf dem
    größten je gesehenen Wert sitzt, lehnt per Bauart nichts ab."""
    werte = np.linspace(0.0, 1.0, 101)
    for perzentil in (0.0, 100.0, -1.0, 101.0):
        with pytest.raises(ValueError, match="Perzentil"):
            delta_for_percentile(werte, percentile=perzentil)
