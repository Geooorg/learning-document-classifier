"""Klassifikator und zwei Vergleichsarme (Konzept § 7.1).

Die multinomiale logistische Regression ist die Hauptumsetzung: von Natur aus
Wahrscheinlichkeiten, Sekunden Trainingszeit auf der CPU, lesbare Gewichte. ``coef_``
gegen die Merkmalsnamen zu halten (``feature_weights``) ist die wichtigste Diagnose
dieses Projekts – sie zeigt, ob das Modell auf "Zahlungsziel" achtet oder auf eine
Abkürzung wie "Seite 1 von 3".

Daneben laufen zwei Vergleichsarme mit:

* **Nearest-Centroid** – die ehrliche Untergrenze, gegen die sich alles beweisen muss.
  Er hat **keine** Wahrscheinlichkeiten: Ein Softmax über negative Kosinusabstände
  sieht aus wie eine Wahrscheinlichkeit und ist keine
  (``liefert_echte_wahrscheinlichkeiten=False``).
* **Lineare SVM** – oft minimal genauer, liefert aber ebenfalls keine echten
  Wahrscheinlichkeiten; ``decision_function`` ist kein kalibriertes Logit.

Alle drei hinter derselben ``TrainedModel``-Fassade, mit ``classes_`` aus
``sorted(set(y))`` – einmal festgelegt und überall dieselbe Reihenfolge. Der
lautloseste Fehler dieser Aufgabe wäre, ``predict_proba``-Spalten und ``classes_``
auseinanderlaufen zu lassen: Jede Konfidenz zeigte dann auf die falsche Klasse, und
alle Metriken blieben trotzdem plausibel. Deshalb prüft ``train_model`` die
Übereinstimmung von ``classes_`` mit dem, was der jeweilige sklearn-Schätzer selbst
für Klassen hält, und ``predict`` verlässt sich für logreg/svm auf den *eigenen*
``predict`` des Schätzers statt auf einen selbstgebauten Umweg über
``predict_proba`` – nur so kann ein Auseinanderlaufen der beiden Reihenfolgen
überhaupt auffallen.

**Skalierung.** Die vier Merkmalsblöcke (``emb_mean``, ``emb_head``, ``ngram``,
``structural``) unterscheiden sich um den Faktor 5000 in ihrer Größenordnung (auf dem
abgelegten Bestand: Standardabweichung 0,036 bei den Embeddings gegen 175 bei den 39
Strukturmerkmalen). ``solver="saga"`` (Konzept § 7.1) ist ein Gradientenverfahren und
konvergiert auf so unterschiedlich skalierten Merkmalen nicht – gemessen läuft es ohne
Skalierung in die Iterationsgrenze (5000) statt zu konvergieren. Für logreg und svm
liegt der ``StandardScaler`` deshalb *im* Modell, als erster Schritt einer
``sklearn.pipeline.Pipeline``: Er wird beim Training nur auf den Trainingsdaten
angepasst (``fit``) und bei jeder späteren Vorhersage nur angewendet (``transform``)
– ein außerhalb angepasster Skalierer hätte Mittelwert und Streuung des Gold-Sets
gesehen, das ist genau die Art von Leckage, vor der Konzept § 9.2 warnt. Auch der
Centroid bekommt einen (separat verwalteten) ``StandardScaler``, obwohl Kosinusabstände
gegenüber einer *gemeinsamen* Skalierung aller Merkmale unempfindlich sind: Ohne
Skalierung dominiert allein die Norm des 39-dimensionalen Strukturblocks (Faktor 5000
gegenüber den Embeddings) den Winkel jedes Dokumentvektors, unabhängig davon, was in den
anderen drei Blöcken steht – das ist *blockweise* Skalierung, nicht die globale, gegen
die Kosinusähnlichkeit robust ist.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

MODEL_KINDS: tuple[str, ...] = ("logreg", "centroid", "svm")


def _softmax(werte: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Zeilenweiser Softmax, numerisch stabil (Maximum je Zeile abziehen)."""
    verschoben = werte - werte.max(axis=1, keepdims=True)
    exponiert = np.exp(verschoben)
    ergebnis: npt.NDArray[np.float64] = exponiert / exponiert.sum(axis=1, keepdims=True)
    return ergebnis


def _kosinusaehnlichkeit(
    X: npt.NDArray[np.float32] | npt.NDArray[np.float64], zentren: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Kosinusähnlichkeit jeder Zeile aus ``X`` zu jeder Zeile aus ``zentren``.

    ``X`` kommt beim Centroid bereits skaliert an (siehe Moduldoc) – ``float64`` statt
    des rohen ``float32`` der Merkmalsmatrix.

    ``1e-12`` gegen die Nullvektor-Division, die ein Dokument ohne ein einziges von
    Null verschiedenes Merkmal sonst in ein ``NaN`` verwandeln würde.
    """
    x = X.astype(np.float64)
    x_normiert = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    z_normiert = zentren / (np.linalg.norm(zentren, axis=1, keepdims=True) + 1e-12)
    ergebnis: npt.NDArray[np.float64] = x_normiert @ z_normiert.T
    return ergebnis


def _binaeren_score_erweitern(score: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``decision_function`` liefert bei genau zwei Klassen eine Spalte (den Score der
    positiven, d.h. zweiten Klasse in ``classes_``) statt einer je Klasse. Für eine
    einheitliche Fassade wird die fehlende Spalte durch das Negative ergänzt – dieselbe
    Konvention, die ``predict_proba`` bei zwei Klassen ohnehin verwendet."""
    if score.ndim == 1:
        return np.column_stack([-score, score])
    return score


def _klassenreihenfolge_pruefen(schaetzer_klassen: Sequence[str], klassen: list[str]) -> None:
    """Wirft, wenn der sklearn-Schätzer eine andere Klassenreihenfolge führt als
    ``klassen`` (``sorted(set(y))``). sklearn sortiert ``classes_`` selbst immer
    aufsteigend, genau wie ``sorted``, weshalb diese Prüfung im Normalfall nie greift –
    sie ist der Notnagel für den Tag, an dem das nicht mehr stimmt, denn ohne sie würde
    jede Konfidenz stillschweigend auf die falsche Klasse zeigen (Konzept § 7.1)."""
    if list(schaetzer_klassen) != klassen:
        raise RuntimeError(
            f"sklearn fuehrt classes_ in der Reihenfolge {list(schaetzer_klassen)!r}, "
            f"sorted(set(y)) aber in {klassen!r}. Das darf nicht auseinanderlaufen, sonst "
            "zeigt jede Konfidenz auf die falsche Klasse."
        )


class TrainedModel:
    """Einheitliche Fassade über logreg, centroid und svm.

    ``classes_`` legt die Spaltenreihenfolge von ``predict_proba`` und
    ``decision_scores`` fest – eine einzige Liste, überall dieselbe (siehe Moduldoc).
    """

    def __init__(
        self,
        kind: str,
        classes_: list[str],
        liefert_echte_wahrscheinlichkeiten: bool,
        *,
        schaetzer: Pipeline | None = None,
        zentren: npt.NDArray[np.float64] | None = None,
        skalierer: StandardScaler | None = None,
        coef: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.kind = kind
        self.classes_ = classes_
        self.liefert_echte_wahrscheinlichkeiten = liefert_echte_wahrscheinlichkeiten
        self.coef_ = coef
        self._schaetzer = schaetzer
        self._zentren = zentren
        self._skalierer = skalierer

    def decision_scores(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Ein Rohwert je Klasse, in ``classes_``-Reihenfolge – höher heißt sicherer.

        Bei logreg/svm das (bei zwei Klassen ergänzte) ``decision_function`` von
        sklearn – die Pipeline skaliert dabei intern mit dem beim Training angepassten
        ``StandardScaler``, hier nur noch angewendet, nie neu angepasst. Beim Centroid
        die Kosinusähnlichkeit zu den (ebenfalls skalierten) Klassenmittelpunkten,
        nachdem ``X`` mit demselben, beim Training angepassten Skalierer transformiert
        wurde (Moduldoc: Skalierung)."""
        if self._zentren is not None:
            assert self._skalierer is not None
            X_skaliert = np.asarray(self._skalierer.transform(X), dtype=np.float64)
            return _kosinusaehnlichkeit(X_skaliert, self._zentren)
        assert self._schaetzer is not None
        roh = np.asarray(self._schaetzer.decision_function(X), dtype=np.float64)
        return _binaeren_score_erweitern(roh)

    def predict_proba(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Eine Verteilung je Zeile, in ``classes_``-Reihenfolge, Zeilensumme 1.

        Bei logreg eine echte Wahrscheinlichkeit (``liefert_echte_wahrscheinlichkeiten
        = True``); bei centroid und svm ein Softmax über ``decision_scores`` – das
        sieht aus wie eine Wahrscheinlichkeit, ist aber keine (Moduldoc)."""
        if self.kind == "logreg":
            assert self._schaetzer is not None
            return np.asarray(self._schaetzer.predict_proba(X), dtype=np.float64)
        return _softmax(self.decision_scores(X))

    def predict(self, X: npt.NDArray[np.float32]) -> list[str]:
        """Die wahrscheinlichste Klasse je Zeile.

        Bei logreg/svm ausdrücklich über den *eigenen* ``predict`` des sklearn-
        Schätzers, nicht über einen Umweg durch ``predict_proba``/``classes_`` – sonst
        könnte ein Auseinanderlaufen der beiden Reihenfolgen (Moduldoc) nie auffallen,
        weil beide Wege denselben Fehler machen würden."""
        if self._schaetzer is not None:
            return [str(klasse) for klasse in self._schaetzer.predict(X)]
        auswahl = self.predict_proba(X).argmax(axis=1)
        return [self.classes_[i] for i in auswahl]


def _als_klassen_x_merkmale(
    coef: npt.NDArray[np.float64], anzahl_klassen: int
) -> npt.NDArray[np.float64]:
    """``coef_`` eines binären Modells hat die Form ``(1, n_merkmale)`` – eine Zeile
    für die positive (zweite) Klasse, keine für die erste. Für eine Zuordnung je
    Klasse (Konzept § 7.1) wird die fehlende Zeile durchs Negative ergänzt, dieselbe
    Konvention wie bei ``decision_function``."""
    if coef.shape[0] == 1 and anzahl_klassen == 2:
        return np.vstack([-coef[0], coef[0]])
    return coef


def train_model(
    X: npt.NDArray[np.float32],
    y: Sequence[str] | npt.NDArray[np.str_],
    kind: str,
    seed: int = 7,
) -> TrainedModel:
    """Eines der drei Modelle aus ``MODEL_KINDS`` trainieren.

    ``classes_`` wird hier ein einziges Mal als ``sorted(set(y))`` festgelegt.
    """
    if kind not in MODEL_KINDS:
        raise ValueError(f"Unbekannte Modellart {kind!r} - erlaubt sind {MODEL_KINDS}.")

    y_arr = np.asarray([str(wert) for wert in y])
    klassen = sorted(set(y_arr.tolist()))

    if kind == "logreg":
        pipeline_lr = Pipeline(
            [
                ("skalierer", StandardScaler()),
                (
                    "schaetzer",
                    LogisticRegression(
                        solver="saga",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipeline_lr.fit(X, y_arr)
        _klassenreihenfolge_pruefen([str(k) for k in pipeline_lr.classes_], klassen)
        coef = _als_klassen_x_merkmale(
            np.asarray(pipeline_lr.named_steps["schaetzer"].coef_, dtype=np.float64),
            len(klassen),
        )
        return TrainedModel(
            kind=kind,
            classes_=klassen,
            liefert_echte_wahrscheinlichkeiten=True,
            schaetzer=pipeline_lr,
            coef=coef,
        )

    if kind == "svm":
        pipeline_svm = Pipeline(
            [
                ("skalierer", StandardScaler()),
                ("schaetzer", LinearSVC(class_weight="balanced", random_state=seed)),
            ]
        )
        pipeline_svm.fit(X, y_arr)
        _klassenreihenfolge_pruefen([str(k) for k in pipeline_svm.classes_], klassen)
        coef = _als_klassen_x_merkmale(
            np.asarray(pipeline_svm.named_steps["schaetzer"].coef_, dtype=np.float64),
            len(klassen),
        )
        return TrainedModel(
            kind=kind,
            classes_=klassen,
            liefert_echte_wahrscheinlichkeiten=False,
            schaetzer=pipeline_svm,
            coef=coef,
        )

    # centroid: Kosinusabstaende zu den Klassenmittelpunkten, keine Wahrscheinlichkeiten.
    # Skalierung separat verwaltet (keine Pipeline noetig, da kein sklearn-Schaetzer
    # beteiligt ist) - angepasst ausschliesslich auf X, das hier bereits die
    # Trainingsmenge ist.
    skalierer_centroid = StandardScaler()
    X_skaliert = np.asarray(skalierer_centroid.fit_transform(X), dtype=np.float64)
    zentren = np.vstack([X_skaliert[y_arr == klasse].mean(axis=0) for klasse in klassen]).astype(
        np.float64
    )
    return TrainedModel(
        kind=kind,
        classes_=klassen,
        liefert_echte_wahrscheinlichkeiten=False,
        zentren=zentren,
        skalierer=skalierer_centroid,
    )


def feature_weights(model: TrainedModel, names: list[str]) -> dict[str, dict[str, float]]:
    """``coef_`` je Klasse gegen ``names`` legen (Konzept § 7.1) – die wichtigste
    Diagnose des Projekts: zeigt, ob ein Modell auf "Zahlungsziel" achtet oder auf
    eine Abkürzung wie "Seite 1 von 3".

    Nur für Modelle mit linearen Gewichten (logreg, svm) – der Centroid hat keine
    ``coef_``, seine Klassifikation ist eine Distanz, keine gewichtete Summe.

    ``coef_`` wird in ``train_model`` aus der Pipeline gezogen (``named_steps
    ["schaetzer"].coef_``), bevor sie auf dem Modell abgelegt wird – ``feature_weights``
    selbst sieht die Pipeline nie und bleibt dadurch unverändert, obwohl logreg/svm
    jetzt hinter einem ``StandardScaler`` stecken. Die Gewichte stehen dadurch in
    skalierten Einheiten (pro Standardabweichung, nicht pro Rohwert) – bei vier Blöcken
    mit Faktor 5000 Größenunterschied (Moduldoc) ist das die einzige Lesart, in der
    Gewichte über Blockgrenzen hinweg überhaupt vergleichbar sind; ein Gewicht auf
    Rohwert-Skala wäre für die 39 Strukturmerkmale systematisch winzig, egal wie stark
    das Modell tatsächlich auf sie achtet.

    Wirft, wenn ``names`` nicht exakt zur Merkmalszahl passt: Eine stillschweigend
    abgeschnittene Zuordnung wäre schlimmer als gar keine.
    """
    koeffizienten = model.coef_
    if koeffizienten is None:
        raise ValueError(
            f"Modellart {model.kind!r} hat keine linearen Gewichte (coef_) - "
            "feature_weights ist nur fuer logreg und svm sinnvoll."
        )
    if len(names) != koeffizienten.shape[1]:
        raise ValueError(
            f"Merkmalsnamen: {len(names)} Namen fuer {koeffizienten.shape[1]} Merkmale - "
            "die Zuordnung waere sonst stillschweigend abgeschnitten."
        )
    return {
        klasse: dict(zip(names, reihe.tolist(), strict=True))
        for klasse, reihe in zip(model.classes_, koeffizienten, strict=True)
    }
