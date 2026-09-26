"""Die Merkmalsversion ist der Schutz davor, ein Modell mit fremden Merkmalen zu füttern."""

from pathlib import Path

import pytest

from doccls.features import FeatureConfig, feature_version, load_feature_config


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


def test_version_haengt_nicht_an_der_reihenfolge() -> None:
    """Zwei inhaltsgleiche Konfigurationen müssen dieselbe Version ergeben, auch wenn die
    Felder in anderer Reihenfolge gesetzt wurden – sonst würde ein harmloses Umsortieren
    in der YAML alle Merkmale ungültig machen."""
    a = load_feature_config()
    b = FeatureConfig(**dict(reversed(list(a.model_dump().items()))))
    assert feature_version(a) == feature_version(b)


def test_unbekanntes_feld_wird_abgewiesen(tmp_path: Path) -> None:
    """Ein Tippfehler in features.yaml darf nicht stillschweigend die Vorgabe benutzen –
    dann trüge die Version eine Einstellung, die gar nicht wirkt."""
    datei = tmp_path / "features.yaml"
    datei.write_text("embedder: e5\nchunk_charss: 1200\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_charss"):
        load_feature_config(datei)
