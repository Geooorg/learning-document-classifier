"""Die beiden Schwellen τ und δ – abgelesen, nicht geraten (Konzept § 7.4).

Hier entscheidet sich, ob ein Dokument automatisch durchläuft oder in die Prüfliste geht.
Zwei Schwellen tragen diese Entscheidung, und **keine von beiden wird geschätzt**:

* **τ** ist das kleinste Konfidenzniveau, oberhalb dessen die Präzision ein Ziel hält
  (Startwert 98 %). Es wird aus der Risiko-Abdeckungs-Kurve auf der Kalibriermenge
  abgelesen.
* **δ** ist ein Perzentil der beobachteten OOD-Abstände (Startwert das 95.). Bei δ ist das
  keine Stilfrage: Kosinuswerte in Satz-Embeddings sind stark gestaucht – inhaltlich sehr
  verschiedene Dokumentausschnitte liegen gemessen zwischen 0,79 und 0,98 (Konzept
  Anhang D). Ein geschätzter Schwellwert wäre in diesem Band reine Willkür.

**Wann eine Schwelle keine Schwelle ist.** Beide Funktionen werfen, statt eine Zahl
zurückzugeben, sobald die Menge zu dünn ist, um die Schwelle zu tragen – aus demselben
Grund, aus dem ``calibrate.fit_temperature`` einen Randtreffer wirft: Ein ``float`` im
erlaubten Bereich ist von einer echten Messung nicht mehr zu unterscheiden, und jeder
Aufrufer rechnete stillschweigend damit weiter.

* Bei τ heißt „zu dünn": weniger als ``1 / (1 − target)`` Dokumente oberhalb der Schwelle.
  Unterhalb dieser Zahl kann die beobachtete Präzision nur 1,0 oder kleiner als das Ziel
  sein – ein einziger Fehler reißt sie unter die Marke. Die Schwelle misst dann
  „null Fehler gesehen", nicht „Präzision ≥ target"; sie ist eine Eigenschaft der
  Stichprobengröße, nicht des Modells. Für 0,98 sind das 50 Dokumente.
* Bei δ heißt es: weniger als ``100 / (100 − percentile)`` Werte. Oberhalb des 95.
  Perzentils liegen 5 Prozent der Verteilung; bei weniger als 20 Werten ist das im Mittel
  weniger als ein einziger beobachteter Wert, und δ wird aus dem obersten Punkt der
  Stichprobe hochgerechnet statt aus der Verteilung abgelesen.

Beide Zahlen sind **aus dem jeweiligen Ziel abgeleitet**, nicht gesetzt: Ein festes 50
wäre für ein Ziel von 0,999 (dort: 1000) genauso falsch wie für 0,9 (dort: 10).

**Kein Gold-Schutz in diesem Modul.** Wie in ``calibrate`` sehen die Funktionen hier
Zahlen, keine Dokumente – sie haben keinen ``split``, den sie prüfen könnten. Der Schutz
aus Konzept § 9.2 sitzt eine Ebene höher, in ``splits.calibration_documents``
(``assert_no_gold``), und ist dort geprüft.

**Kein Zufall in diesem Modul.** Keine Funktion hier zieht, mischt oder initialisiert
zufällig; ein ``seed``-Parameter hätte nichts zu steuern. Bei gleicher Eingabe kommt
dasselbe heraus.
"""

import math

import numpy as np
import numpy.typing as npt

Zahlenreihe = npt.NDArray[np.float32] | npt.NDArray[np.float64]
Zahlenfeld = npt.NDArray[np.float32] | npt.NDArray[np.float64]


def _als_reihe(werte: Zahlenreihe, name: str) -> npt.NDArray[np.float64]:
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


def _als_maske(werte: npt.NDArray[np.bool_], name: str) -> npt.NDArray[np.bool_]:
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


def _kurve(
    confidences: Zahlenreihe, correct: npt.NDArray[np.bool_]
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    """Abdeckung, Präzision, Schwellwert und Belegung – je Schwelle eine Zeile.

    Ausgewertet wird nur an den Stellen, an denen die nächstkleinere Konfidenz echt
    kleiner ist: Eine Schwelle, die zwei Dokumente mit *demselben* Wert trennt, gibt es
    nicht, und ``konfidenz >= tau`` nähme beide oder keines.

    Die Reihenfolge ist aufsteigend in der Abdeckung, also **absteigend** im Schwellwert;
    der letzte Eintrag ist die unterste Schwelle mit voller Abdeckung.
    """
    konfidenz = _als_reihe(confidences, "confidences")
    richtig = _als_maske(correct, "correct")
    if konfidenz.size != richtig.size:
        raise ValueError(
            f"{konfidenz.size} Konfidenzen stehen {richtig.size} Wahrheitswerten "
            "gegenueber - je Dokument wird genau eines von beiden gebraucht, sonst "
            "bezieht sich jede Praezision auf ein anderes Dokument."
        )

    ordnung = np.argsort(-konfidenz, kind="stable")
    sortiert = konfidenz[ordnung]
    treffer = np.cumsum(richtig[ordnung].astype(np.float64))
    belegung = np.arange(1, konfidenz.size + 1, dtype=np.int64)

    gruppenende = np.flatnonzero(np.r_[np.diff(sortiert) < 0.0, True])
    abdeckung = belegung[gruppenende] / float(konfidenz.size)
    praezision = treffer[gruppenende] / belegung[gruppenende]
    return abdeckung, praezision, sortiert[gruppenende], belegung[gruppenende]


def risk_coverage(
    confidences: Zahlenreihe, correct: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Die Risiko-Abdeckungs-Kurve (Konzept § 7.4).

    Zu jeder möglichen Schwelle: welcher Anteil der Dokumente sie passiert (*Abdeckung*)
    und welcher Anteil davon richtig ist (*Präzision*). Aufsteigend in der Abdeckung, der
    letzte Punkt ist die volle Menge – dort ist die Präzision die Trefferquote insgesamt.

    Aus dieser Kurve liest :func:`tau_for_precision` die Schwelle ab; Aufgabe 14 bildet
    daraus die Fläche unter der Risiko-Abdeckungs-Kurve.
    """
    abdeckung, praezision, _, _ = _kurve(confidences, correct)
    return abdeckung, praezision


def _mindestbelegung(target: float) -> int:
    """Wie viele Dokumente oberhalb der Schwelle stehen müssen, damit sie eine ist.

    Das kleinste ``n`` mit ``(n − 1) / n >= target``: die Zahl, ab der ein einziger Fehler
    die Präzision nicht sofort unter das Ziel reißt. Darunter kann die beobachtete
    Präzision nur 1,0 oder kleiner als ``target`` sein, und die Schwelle misst nicht das
    Ziel, sondern „null Fehler gesehen" (Moduldoc).

    Die Suche um ``1 / (1 − target)`` herum ist kein Zierrat: ``1 / (1 − 0,98)`` ergibt in
    Gleitkomma 49,99999999999996, und die Zahl darf nicht an der Rundungsrichtung hängen.
    """
    grenze = int(math.ceil(1.0 / (1.0 - target)))
    while grenze > 1 and (grenze - 2) / (grenze - 1) >= target:
        grenze -= 1
    while (grenze - 1) / grenze < target:
        grenze += 1
    return grenze


def tau_for_precision(
    confidences: Zahlenreihe, correct: npt.NDArray[np.bool_], target: float = 0.98
) -> float:
    """Das kleinste τ, oberhalb dessen die Präzision ``target`` hält (Konzept § 7.4).

    Gelesen wird auf der Kalibriermenge, aus der Risiko-Abdeckungs-Kurve. **Das kleinste**
    ist der Punkt: Ein höheres τ hält das Ziel auch und schickt unnötig viele Dokumente in
    die Prüfliste; die Grenze ``1,0`` hält es immer und sagt nichts.

    Sind **alle** Vorhersagen richtig, ist die Antwort die kleinste beobachtete Konfidenz –
    volle Abdeckung. Das ist kein Sonderfall, sondern die richtige Lesung dessen, was
    dasteht; dass „kein Fehler beobachtet" nicht „kein Fehler zu erwarten" heißt, fängt die
    Mindestbelegung ab und nicht eine Ausnahme hier.

    Geworfen wird in zwei Lagen, beide mit ``ValueError``:

    * **Keine Schwelle erreicht das Ziel.** Die Meldung nennt die höchste überhaupt
      erreichte Präzision, damit sichtbar ist, wie weit es fehlt. Eine stillschweigend
      zurückgegebene ``1,0`` sähe wie eine strenge Schwelle aus und wäre eine Lüge.
    * **Das Ziel wird nur über zu wenigen Dokumenten gehalten** (siehe
      :func:`_mindestbelegung`). Eine Schwelle, die aus einer Handvoll Dokumente abgelesen
      ist, ist keine Schwelle – gemessen auf 1000 synthetischen Dokumenten hing sie bei
      einem Seed an acht Dokumenten und lag dadurch um 0,03 neben ihrem analytischen Ort.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(
            f"Das Ziel {target!r} liegt nicht echt zwischen 0 und 1. Ein Ziel von 1,0 ist "
            "auf endlich vielen Dokumenten nicht ablesbar - es verlangte unendlich viele "
            "Belege -, und ein Ziel von 0 ist keine Anforderung."
        )
    _, praezision, schwellen, belegung = _kurve(confidences, correct)
    mindestens = _mindestbelegung(target)

    haelt = praezision >= target
    tragfaehig = haelt & (belegung >= mindestens)
    if bool(tragfaehig.any()):
        # Aufsteigend in der Abdeckung heisst absteigend im Schwellwert: der letzte
        # tragfaehige Eintrag ist das kleinste tau.
        return float(schwellen[tragfaehig][-1])

    if bool(haelt.any()):
        beste = int(np.argmax(np.where(haelt, belegung, -1)))
        raise ValueError(
            f"Das Ziel {target} wird nur oberhalb von {schwellen[beste]:.6f} gehalten, und "
            f"dort stehen {belegung[beste]} Dokumente statt der noetigen {mindestens} "
            f"(Mindestbelegung). Bei weniger als {mindestens} Dokumenten kann die "
            f"beobachtete Praezision nur 1,0 oder kleiner als {target} sein - die Schwelle "
            "misst dann die Stichprobengroesse und nicht das Modell."
        )
    raise ValueError(
        f"Keine Schwelle erreicht das Ziel {target}. Die hoechste oberhalb einer Schwelle "
        f"erreichte Praezision ist {float(praezision.max()):.6f}. Ein zurueckgegebenes "
        "1,0 saehe wie eine strenge Schwelle aus und waere eine Luege."
    )


def _l2_normiert(matrix: Zahlenfeld, name: str) -> npt.NDArray[np.float64]:
    """Zeilenweise auf Länge 1 – die Voraussetzung dafür, dass nur der Winkel zählt.

    Ohne diesen Schritt entschiede in :func:`ood_scores` das Skalarprodukt statt des
    Kosinus, welche Nachbarn die nächsten sind, und der längste Vektor beherrschte ihr
    Mittel. Auf Embeddings ist das kein theoretischer Fall: Ein längerer Ausschnitt
    erzeugt einen längeren Vektor bei gleicher Richtung.
    """
    werte = np.asarray(matrix, dtype=np.float64)
    if werte.ndim != 2:
        raise ValueError(
            f"{name} muss eine Matrix sein (Zeile je Dokument, Spalte je Merkmal), hat "
            f"aber {werte.ndim} Dimensionen."
        )
    if werte.size == 0:
        raise ValueError(f"{name} ist leer.")
    if not np.isfinite(werte).all():
        raise ValueError(f"{name} enthaelt Werte, die nicht endlich sind (NaN oder inf).")
    laengen = np.linalg.norm(werte, axis=1, keepdims=True)
    if bool((laengen == 0.0).any()):
        raise ValueError(
            f"{name} enthaelt {int((laengen == 0.0).sum())} Nullvektor(en). Ein Nullvektor "
            "hat keine Richtung - der Kosinus teilte durch null und ergaebe nan, das sich "
            "durch jede spaetere Perzentilbildung fraesse."
        )
    normiert: npt.NDArray[np.float64] = werte / laengen
    return normiert


def ood_scores(X: Zahlenfeld, X_train: Zahlenfeld, k: int = 10) -> npt.NDArray[np.float64]:
    """Kosinusabstand zum Mittel der ``k`` nächsten Trainingsnachbarn (Konzept § 7.4).

    ``1 − cos(x, m)`` mit ``m`` als Mittel der ``k`` ähnlichsten Trainingsvektoren, beide
    Seiten vorher L2-normiert. Der Wertebereich ist ``[0, 2]``: 0 bei gleicher Richtung, 1
    im rechten Winkel, 2 in Gegenrichtung.

    **Zum Mittel der Nachbarn, nicht zum globalen Klassenmittelpunkt.** Ein Klassenmittel
    über mehrere Vorlagen liegt zwischen ihnen und damit unter Umständen bei keiner; die
    ``k`` nächsten Nachbarn beschreiben die lokale Umgebung, in der das Dokument
    tatsächlich liegt.

    Die Prüfung ist nötig, weil ein lineares Modell auf einem Dokument, das keiner Klasse
    ähnelt, trotzdem selbstbewusst sein kann: Es kennt nur seine sieben Klassen und
    verteilt die Masse auf sie. Der Abstand im Embedding-Raum ist davon unabhängig.

    ``k`` größer als der Trainingsbestand wird **abgelehnt, nicht gekürzt**: Stillschweigend
    weniger Nachbarn zu mitteln hieße, ein anderes Maß zu rechnen als das angefragte – und
    δ wäre dann auf einer anderen Größe abgelesen als die späteren Abstände.

    **Bei gleicher Ähnlichkeit entscheidet die Reihenfolge von** ``np.argpartition``, und
    das ist am Ergebnis ablesbar: ``ood_scores([[1, 1]], [[1, 0], [1, 0], [0, 1], [0, 1]],
    k=2)`` liefert 0,29289 – gewählt werden die beiden gleichgerichteten ``[1, 0]``, deren
    Mittel wieder ``[1, 0]`` ist. Bei „je einer" aus beiden Paaren wäre das Mittel
    ``[0,5, 0,5]`` und der Abstand 0,0. Alle vier Nachbarn haben hier denselben Kosinus
    zur Probe; welche zwei genommen werden, sagt das Maß nicht.

    Bewusst nicht per Test gebunden: Auf den float64-Embeddings, für die diese Funktion
    gebaut ist, sind exakte Gleichstände praktisch ausgeschlossen, und eine Zusicherung auf
    0,29289 bände die innere Ordnung von ``np.argpartition`` – eine Eigenschaft von NumPy,
    die keine Zusage ist und sich mit einer Version ändern darf. Wer künstliche Achsen- oder
    Wiederholungsvektoren hineingibt, muss damit rechnen, dass die Zahl von der Auswahl
    unter Gleichen abhängt.
    """
    probe = _l2_normiert(X, "X")
    bestand = _l2_normiert(X_train, "X_train")
    if probe.shape[1] != bestand.shape[1]:
        raise ValueError(
            f"X hat {probe.shape[1]} Merkmale, X_train hat {bestand.shape[1]}. Ein Kosinus "
            "zwischen verschieden langen Vektoren ist nicht definiert."
        )
    if k < 1 or k > bestand.shape[0]:
        raise ValueError(
            f"k={k} Nachbarn sind aus {bestand.shape[0]} Trainingsvektoren nicht zu holen "
            "(und weniger als einer ist kein Nachbar). Stillschweigend zu kuerzen ergaebe "
            "ein anderes Mass als das angefragte."
        )

    aehnlichkeit = probe @ bestand.T
    naechste = np.argpartition(-aehnlichkeit, k - 1, axis=1)[:, :k]
    mittel = bestand[naechste].mean(axis=1)
    laengen = np.linalg.norm(mittel, axis=1)
    if bool((laengen == 0.0).any()):
        raise ValueError(
            f"Bei {int((laengen == 0.0).sum())} Dokumenten heben sich die {k} naechsten "
            "Nachbarn gegenseitig auf; ihr Mittel ist der Nullvektor und hat keine "
            "Richtung. Ein Abstand dazu waere rechenbar, aber ohne Bedeutung."
        )
    abstand: npt.NDArray[np.float64] = 1.0 - np.einsum("ij,ij->i", probe, mittel) / laengen
    return abstand


def _mindestzahl_fuer_perzentil(percentile: float) -> int:
    """Wie viele Werte es braucht, damit ein Perzentil abgelesen und nicht hochgerechnet ist.

    Das kleinste ``n``, bei dem im Mittel mindestens ein beobachteter Wert oberhalb des
    Perzentils liegt: ``n * (1 − percentile / 100) >= 1``. Für das 95. Perzentil sind das
    20 Werte. Darunter sitzt δ auf dem obersten Punkt der Stichprobe, und der ist eine
    Eigenschaft ihrer Größe, nicht der Verteilung.

    **Gerechnet wird in Prozentpunkten, nicht mit dem Anteil** – das ist kein Zierrat,
    sondern die Behebung eines gemessenen Fehlers. Die lesbarere Form
    ``1 − percentile / 100`` ist verlustbehaftet: Für 90 ergibt sie 0,09999999999999998
    statt 0,1, und ``10 * 0,09999999999999998`` bleibt unter 1. Weil beide
    Korrekturschleifen mit demselben verlorenen Wert rechnen, können sie den Fehler nicht
    einfangen – die Funktion war dadurch für 80 (6 statt 5) und 90 (11 statt 10) um eins zu
    streng. ``100 − percentile`` bleibt dagegen exakt, wo ``percentile`` ganzzahlig ist,
    und die Schleifen sehen denselben Faktor wie der erste Schätzer. Gegen exakte
    Bruchrechnung (``fractions.Fraction``) geprüft über alle ganzen und halben Perzentile
    von 1 bis 99 sowie 99,9 / 99,99 / 99,999: kein Unterschied.

    Bei einem Perzentil, das selbst nicht darstellbar ist, bleibt die Zahl an der
    Gleitkommadarstellung hängen – 99,9 ist als ``float`` etwas *größer* als 99,9, deshalb
    kommt 1001 heraus und nicht 1000. Das ist die richtige Antwort auf die übergebene Zahl
    und kein Rundungsfehler mehr; der Zwilling :func:`_mindestbelegung` liefert an der
    entsprechenden Stelle (0,999) die runde 1000, weil er exakte Ganzzahlverhältnisse
    ``(n − 1) / n`` vergleicht statt ein aus einer Differenz gebildetes Produkt. Ein Wert
    Unterschied bei tausend trägt keine Aussage, deshalb bleibt es bei der einfacheren
    Form; wer exakt dezimal rechnen will, müsste hier ``decimal`` oder ``fractions``
    hineinziehen.
    """
    prozentpunkte_oberhalb = 100.0 - percentile
    anzahl = int(math.ceil(100.0 / prozentpunkte_oberhalb))
    while anzahl > 1 and (anzahl - 1) * prozentpunkte_oberhalb >= 100.0:
        anzahl -= 1
    while anzahl * prozentpunkte_oberhalb < 100.0:
        anzahl += 1
    return anzahl


def delta_for_percentile(scores: Zahlenreihe, percentile: float = 95.0) -> float:
    """δ als Perzentil der beobachteten OOD-Abstände (Konzept § 7.4, Anhang D).

    Gemessen, nicht geschätzt: Kosinuswerte liegen gestaucht zwischen 0,79 und 0,98, und
    eine gesetzte Zahl in diesem Band ist Willkür. ``scores`` sind die Abstände der
    gelabelten Dokumente der Kalibriermenge zu ihrer eigenen Umgebung
    (:func:`ood_scores`); das Perzentil sagt, welcher Anteil von ihnen als „bekannt" gelten
    soll – beim Startwert 95 werden fünf Prozent der eigenen Kalibrierdokumente als
    ``SONSTIGES`` gemeldet.

    0 und 100 sind Minimum und Maximum und keine Perzentile: Ein δ auf dem größten je
    gesehenen Wert lehnt per Bauart nichts ab, eines auf dem kleinsten alles.

    **Zwischen zwei Beobachtungen wird linear interpoliert** (``method="linear"``,
    ausdrücklich gesetzt statt der Voreinstellung überlassen). „Ablesen" heißt damit nicht
    immer „einen gemessenen Wert nehmen": Der gesuchte Rang ist ``(n − 1) * p / 100`` und
    trifft im Allgemeinen zwischen zwei sortierte Werte. Gerade an der Mindestzahl ist das
    sichtbar – bei genau 20 Werten und dem 95. Perzentil liegt der Rang bei 18,05, δ also
    5 Prozent des Wegs vom zweitobersten zum obersten Wert. Auf ``linspace(0, 1, 20)``
    ergibt das 0,95, während ``lower`` 0,947368 und ``higher`` 1,0 lieferte.

    Die Wahl ist bewusst: ``higher`` sprünge auf den größten beobachteten Wert und
    verlöre genau die Eigenschaft, um derentwillen die Mindestzahl existiert; ``lower``
    ließe δ auf einer Beobachtung sitzen und machte die Schwelle unstetig in der
    Stichprobengröße. Gebunden ist die Wahl in
    ``test_delta_interpoliert_linear_zwischen_den_nachbarwerten``.
    """
    if not 0.0 < percentile < 100.0:
        raise ValueError(
            f"Das Perzentil {percentile!r} liegt nicht echt zwischen 0 und 100. 100 waere "
            "das Maximum - ein delta darauf lehnt per Bauart nichts ab -, 0 das Minimum."
        )
    werte = _als_reihe(scores, "scores")
    mindestens = _mindestzahl_fuer_perzentil(percentile)
    if werte.size < mindestens:
        raise ValueError(
            f"{werte.size} Werte reichen fuer das {percentile}. Perzentil nicht; noetig "
            f"sind {mindestens} (Mindestzahl). Darunter liegt im Mittel kein einziger "
            "beobachteter Wert oberhalb des Perzentils, und delta waere aus dem obersten "
            "Punkt der Stichprobe hochgerechnet statt aus der Verteilung abgelesen."
        )
    return float(np.percentile(werte, percentile, method="linear"))
