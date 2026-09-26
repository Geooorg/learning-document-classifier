"""Die Merkmalsversion ist der Schutz davor, ein Modell mit fremden Merkmalen zu füttern."""

from pathlib import Path

import pytest

from doccls.features import (
    feature_version,
    load_feature_config,
    version_of_values,
)


def test_version_ist_stabil_ueber_laeufe() -> None:
    """Gleiche Konfiguration, gleiche Version – sonst wäre jeder Lauf inkompatibel zum
    vorherigen und die Prüfung liefe ins Leere."""
    konfiguration = load_feature_config()
    assert feature_version(konfiguration) == feature_version(konfiguration)


def test_jeder_parameter_veraendert_die_version() -> None:
    """Der eigentliche Zweck: Ändert sich irgendein Merkmalsparameter, muss die Version
    sich ändern. Ein Test, der nur EINEN Parameter prüft, übersieht genau den, der später
    still geändert wird – deshalb werden hier alle durchgegangen."""
    basis = load_feature_config()
    original = feature_version(basis)
    aenderungen: dict[str, object] = {
        "embedder": "bge-m3",
        "chunk_chars": 900,
        "head_chars": 500,
        "ngram_min": 2,
        "ngram_max": 6,
        "ngram_max_features": 20000,
        "svd_components": 128,
        "position_decay": 8.0,
    }
    for feld, wert in aenderungen.items():
        geaendert = basis.model_copy(update={feld: wert})
        assert feature_version(geaendert) != original, (
            f"Feld {feld} aendert die Merkmale, aber nicht die Version – ein Modell wuerde "
            "stillschweigend mit fremden Merkmalen rechnen"
        )


def test_version_haengt_nicht_an_der_reihenfolge_der_werte() -> None:
    """``sort_keys=True`` in ``version_of_values``. Ohne das aenderte ein blosses
    Umsortieren der Felddeklarationen in ``FeatureConfig`` die Version – und machte jeden
    gerechneten Merkmalsvektor ungueltig, obwohl sich an den Merkmalen nichts geaendert hat.

    Geprueft wird ueber ``version_of_values`` und nicht ueber ``FeatureConfig``: Pydantics
    ``model_dump()`` liefert immer die Deklarationsreihenfolge, egal in welcher Reihenfolge
    die Kwargs uebergeben wurden. Ein Test ueber das Modell kann diese Eigenschaft deshalb
    strukturell nicht zum Fehlschlagen bringen – er war tautologisch und wurde ersetzt.
    """
    werte = load_feature_config().model_dump()
    umgedreht = dict(reversed(list(werte.items())))
    assert list(umgedreht) != list(werte), "Die Probe dreht die Reihenfolge nicht wirklich um"
    assert version_of_values(umgedreht) == version_of_values(werte)


def test_unbekanntes_feld_wird_abgewiesen(tmp_path: Path) -> None:
    """Ein Tippfehler in features.yaml darf nicht stillschweigend die Vorgabe benutzen –
    dann trüge die Version eine Einstellung, die gar nicht wirkt."""
    datei = tmp_path / "features.yaml"
    datei.write_text("embedder: e5\nchunk_charss: 1200\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_charss"):
        load_feature_config(datei)
