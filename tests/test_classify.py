"""Tests für Aufgabe 10: Klassifikator und zwei Vergleichsarme (Konzept § 7.1)."""

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.metrics import accuracy_score

from doccls.classify import MODEL_KINDS, feature_weights, train_model


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
