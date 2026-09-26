"""Tests für Aufgabe 10: Klassifikator und zwei Vergleichsarme (Konzept § 7.1)."""

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from doccls.classify import MODEL_KINDS, TrainedModel, feature_weights, train_model


def _zwei_klare_wolken(seed: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.str_]]:
    """Zwei weit getrennte Punktwolken – auch für den Centroid trennbar: Die Wolken
    liegen in entgegengesetzter Richtung vom Ursprung (Kosinusähnlichkeit -1), nicht
    nur an verschiedenen Orten."""
    rng = np.random.default_rng(seed)
    n = 30
    a = rng.normal(loc=-5.0, scale=0.5, size=(n, 4))
    b = rng.normal(loc=5.0, scale=0.5, size=(n, 4))
    X = np.vstack([a, b]).astype(np.float32)
    y = np.array(["a"] * n + ["b"] * n)
    return X, y


def _schiefe_verteilung(
    haeufig: int, selten: int, seed: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.str_]]:
    """Zwei überlappende Klassen, eine davon selten. Kalibriert (siehe
    Mutationsprobe im Bericht): Ohne ``class_weight="balanced"`` sagt logreg die
    seltene Klasse auf diesen Daten kein einziges Mal voraus, mit ihr mehrfach."""
    rng = np.random.default_rng(seed)
    X_haeufig = rng.normal(loc=0.0, scale=0.9, size=(haeufig, 4))
    X_selten = rng.normal(loc=0.7, scale=0.9, size=(selten, 4))
    X = np.vstack([X_haeufig, X_selten]).astype(np.float32)
    y = np.array(["haeufig"] * haeufig + ["selten"] * selten)
    return X, y


def _bloecke_mit_extremer_skala(
    seed: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.str_]]:
    """Ein 3-dimensionaler Signalblock (klar, aber nicht trivial trennend) neben einem
    5-dimensionalen, klassenblinden Block mit 500.000-facher Streuung – wie beim echten
    Bestand, wo der Strukturblock die Embeddings um den Faktor 5000 überragt
    (Moduldoc), nur deutlicher, damit der Effekt auf 80 Dokumenten zuverlässig sichtbar
    wird. Der riesige Block trägt kein Klassensignal (identische Verteilung für beide
    Klassen) – ein Modell, das ihn nicht klein rechnet, verliert trotzdem, weil ihm die
    Optimierung auf dieser Skala numerisch entgleist (gemessen: saga läuft ohne
    Skalierung in die Iterationsgrenze)."""
    rng = np.random.default_rng(seed)
    n = 40
    signal_a = rng.normal(loc=-1.0, scale=1.0, size=(n, 3))
    signal_b = rng.normal(loc=1.0, scale=1.0, size=(n, 3))
    signal = np.vstack([signal_a, signal_b])
    rauschen = rng.normal(loc=0.0, scale=500_000.0, size=(2 * n, 5))
    X = np.hstack([signal, rauschen]).astype(np.float32)
    y = np.array(["a"] * n + ["b"] * n)
    return X, y


def _skalierer_von(modell: TrainedModel) -> StandardScaler:
    """Weißer-Kasten-Zugriff auf den beim Training angepassten ``StandardScaler`` – bei
    logreg/svm steckt er als erster Schritt in der Pipeline
    (``named_steps["skalierer"]``), beim Centroid liegt er direkt auf der Fassade. Nur
    so lässt sich prüfen, woher seine Kennzahlen stammen; von außen ist ``TrainedModel``
    bewusst nur über predict/predict_proba/decision_scores ansprechbar."""
    if modell._zentren is not None:
        assert modell._skalierer is not None
        return modell._skalierer
    assert modell._schaetzer is not None
    skalierer = modell._schaetzer.named_steps["skalierer"]
    assert isinstance(skalierer, StandardScaler)
    return skalierer


def test_skalierung_macht_extrem_unterschiedliche_bloecke_lernbar() -> None:
    """Ohne Skalierung dominiert der riesig (aber klassenblinde) skalierte Block die
    Optimierung aller drei Modellarten so stark, dass sie kaum besser als Zufall
    abschneiden (gemessen ohne Skalierer auf demselben Datensatz: logreg 0,64, svm
    0,64, centroid 0,63). Mit dem in ``train_model`` eingebauten Skalierer (Moduldoc)
    muss jede der drei Modellarten die Trainingsmenge dennoch nahezu perfekt lernen –
    die Mutationsprobe dieses Tests ist, den Skalierer aus ``train_model`` zu entfernen
    (siehe Bericht)."""
    X, y = _bloecke_mit_extremer_skala(seed=1)
    for kind in MODEL_KINDS:
        modell = train_model(X, y, kind=kind)
        genauigkeit = accuracy_score(y, modell.predict(X))
        assert genauigkeit >= 0.9, (
            f"{kind}: nur {genauigkeit:.2f} Genauigkeit trotz Skalierung - "
            "wirkt der Skalierer noch?"
        )


def test_skalierer_sieht_die_vorhersagedaten_nicht() -> None:
    """Leckagefrage (Konzept § 9.2): Ein Skalierer, der bei predict()/predict_proba()/
    decision_scores() erneut angepasst statt nur angewendet würde, hätte Mittelwert und
    Streuung der Vorhersagedaten gesehen. Trainings- und Vorhersagedaten liegen hier auf
    komplett verschiedenen Verteilungen (Training um 0/±1 mit einem 500.000-fach
    gestreuten Block, Vorhersage eng gebündelt um 10.000) – nach mehreren Vorhersagen
    muss der im Modell abgelegte Skalierer noch exakt die Trainingskennzahlen tragen,
    nicht die der Vorhersagedaten."""
    X_train, y_train = _bloecke_mit_extremer_skala(seed=2)
    mittelwert_training = X_train.astype(np.float64).mean(axis=0)
    streuung_training = X_train.astype(np.float64).std(axis=0)

    rng = np.random.default_rng(99)
    X_vorhersage = rng.normal(loc=10_000.0, scale=0.01, size=(5, X_train.shape[1])).astype(
        np.float32
    )

    for kind in MODEL_KINDS:
        modell = train_model(X_train, y_train, kind=kind)

        skalierer_vor = _skalierer_von(modell)
        assert np.allclose(np.asarray(skalierer_vor.mean_), mittelwert_training)
        assert np.allclose(np.asarray(skalierer_vor.scale_), streuung_training)

        modell.predict(X_vorhersage)
        modell.predict_proba(X_vorhersage)
        modell.decision_scores(X_vorhersage)

        skalierer_danach = _skalierer_von(modell)
        assert np.allclose(np.asarray(skalierer_danach.mean_), mittelwert_training), (
            f"{kind}: Skalierer traegt nach der Vorhersage nicht mehr die "
            "Trainingsstatistik - wurde er auf den Vorhersagedaten neu angepasst?"
        )
        assert np.allclose(np.asarray(skalierer_danach.scale_), streuung_training)
        assert not np.allclose(
            np.asarray(skalierer_danach.mean_),
            X_vorhersage.astype(np.float64).mean(axis=0),
        )


def test_feature_weights_stimmt_trotz_pipeline() -> None:
    """Nach dem Einbau der Skalierungs-Pipeline (Moduldoc) liegt ``coef_`` nicht mehr
    direkt am sklearn-Schätzer, sondern hinter ``pipeline.named_steps["schaetzer"]`` –
    ``train_model`` zieht es dort heraus und legt es auf der Fassade ab (Docstring von
    ``feature_weights``). Dieser Test bindet, dass die Zuordnung Gewicht -> Merkmalsname
    (Konzept § 7.1) dabei nicht kaputtgegangen ist: richtige Anzahl Gewichte je Klasse,
    korrekt benannt, und weiterhin ein Wurf bei falscher Namenszahl – auf Daten mit
    genau der Art von Skalenunterschied, die die Pipeline erst nötig gemacht hat."""
    X, y = _bloecke_mit_extremer_skala(seed=3)
    namen = [f"m{i}" for i in range(X.shape[1])]
    for kind in ("logreg", "svm"):
        gewichte = feature_weights(train_model(X, y, kind=kind), namen)
        assert set(gewichte) == set(np.unique(y))
        for je_klasse in gewichte.values():
            assert len(je_klasse) == len(namen)
            assert set(je_klasse) == set(namen)
        with pytest.raises(ValueError, match="Merkmalsnamen"):
            feature_weights(train_model(X, y, kind=kind), namen[:-1])


def test_jedes_modell_lernt_eine_trennbare_aufgabe() -> None:
    """Untergrenze der Brauchbarkeit: Auf klar getrennten Punktwolken muss jedes der drei
    Modelle die Trainingsmenge nahezu perfekt treffen. Schafft es das nicht, ist die
    Umsetzung kaputt – und kein spaeterer Test auf echten Daten koennte das von einem
    duennen Korpus unterscheiden."""
    X, y = _zwei_klare_wolken(seed=1)
    for kind in MODEL_KINDS:
        modell = train_model(X, y, kind=kind)
        assert accuracy_score(y, modell.predict(X)) >= 0.95, f"{kind} lernt nicht"


def test_wahrscheinlichkeiten_summieren_sich_zu_eins() -> None:
    X, y = _zwei_klare_wolken(seed=1)
    for kind in MODEL_KINDS:
        p = train_model(X, y, kind=kind).predict_proba(X)
        assert np.allclose(p.sum(axis=1), 1.0), f"{kind}: Zeilensumme ist nicht 1"
        assert np.all(p >= 0.0)


def test_klassenreihenfolge_ist_ueberall_dieselbe() -> None:
    """Der lautloseste Fehler der ganzen Phase: predict_proba liefert Spalten in
    sklearn-Reihenfolge, classes_ aber in einer anderen. Dann zeigt jede Konfidenz auf die
    falsche Klasse, und alle Metriken bleiben plausibel."""
    X, y = _zwei_klare_wolken(seed=1)
    for kind in MODEL_KINDS:
        modell = train_model(X, y, kind=kind)
        p = modell.predict_proba(X)
        aus_proba = [modell.classes_[i] for i in p.argmax(axis=1)]
        assert aus_proba == modell.predict(X), f"{kind}: predict und predict_proba uneins"


def test_training_ist_reproduzierbar() -> None:
    X, y = _zwei_klare_wolken(seed=1)
    for kind in MODEL_KINDS:
        a = train_model(X, y, kind=kind, seed=7).predict_proba(X)
        b = train_model(X, y, kind=kind, seed=7).predict_proba(X)
        assert np.allclose(a, b), f"{kind} ist nicht reproduzierbar"


def test_seltene_klasse_wird_nicht_uebergangen() -> None:
    """class_weight='balanced' (Konzept § 7.1). Ohne das lernt das Modell bei schiefer
    Verteilung, die seltene Klasse schlicht nie vorherzusagen – bei guter Accuracy."""
    X, y = _schiefe_verteilung(haeufig=200, selten=10, seed=1)
    modell = train_model(X, y, kind="logreg")
    assert "selten" in set(modell.predict(X))


def test_gewichte_lassen_sich_den_namen_zuordnen() -> None:
    """Konzept § 7.1: coef_ gegen die Merkmalsnamen zeigt, ob das Modell auf
    'Zahlungsziel' oder auf 'Seite 1 von 3' achtet. Das ist die wichtigste Diagnose des
    Projekts – sie braucht eine Zuordnung, die nicht verrutschen kann."""
    X, y = _zwei_klare_wolken(seed=1)
    namen = [f"m{i}" for i in range(X.shape[1])]
    gewichte = feature_weights(train_model(X, y, kind="logreg"), namen)
    assert set(gewichte) == set(np.unique(y))
    for je_klasse in gewichte.values():
        assert set(je_klasse) == set(namen)


def test_gewichte_bei_falscher_namenszahl_werfen() -> None:
    """Eine stillschweigend abgeschnittene Zuordnung waere schlimmer als gar keine."""
    X, y = _zwei_klare_wolken(seed=1)
    with pytest.raises(ValueError, match="Merkmalsnamen"):
        feature_weights(train_model(X, y, kind="logreg"), ["zu", "wenige"])


def test_centroid_gibt_zu_dass_er_keine_wahrscheinlichkeiten_hat() -> None:
    """Ein Softmax ueber negative Abstaende sieht aus wie eine Wahrscheinlichkeit und ist
    keine. Wandert er ungekennzeichnet in eine Kalibriermessung, verdirbt er sie."""
    X, y = _zwei_klare_wolken(seed=1)
    assert train_model(X, y, kind="centroid").liefert_echte_wahrscheinlichkeiten is False
    assert train_model(X, y, kind="logreg").liefert_echte_wahrscheinlichkeiten is True


def test_svm_gibt_ebenfalls_zu_dass_er_keine_wahrscheinlichkeiten_hat() -> None:
    """Die SVM ist der zweite Vergleichsarm ohne echte Wahrscheinlichkeiten (Moduldoc):
    ``decision_function`` ist kein kalibriertes Logit. Ohne diesen Test waere die SVM die
    einzige der drei Modellarten, deren Flag nie geprueft wird."""
    X, y = _zwei_klare_wolken(seed=1)
    assert train_model(X, y, kind="svm").liefert_echte_wahrscheinlichkeiten is False


def test_unbekannte_modellart_wirft() -> None:
    X, y = _zwei_klare_wolken(seed=1)
    with pytest.raises(ValueError, match="Modellart"):
        train_model(X, y, kind="baum")


def test_feature_weights_beim_centroid_wirft() -> None:
    """Der Centroid hat keine linearen Gewichte – eine Zuordnung vorzutaeuschen waere
    ein weiterer Fall von 'sieht aus wie, ist aber keine' (siehe Moduldoc)."""
    X, y = _zwei_klare_wolken(seed=1)
    namen = [f"m{i}" for i in range(X.shape[1])]
    with pytest.raises(ValueError, match="centroid"):
        feature_weights(train_model(X, y, kind="centroid"), namen)
