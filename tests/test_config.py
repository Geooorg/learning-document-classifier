"""Die Konfiguration muss Pfade liefern, die unterhalb des Projektverzeichnisses liegen."""

from doccls import config


def test_pfade_liegen_im_projekt() -> None:
    for pfad in (config.RAW_DIR, config.PARQUET_DIR, config.GENERATED_DIR):
        assert config.PROJECT_ROOT in pfad.parents


def test_ocr_schwelle_ist_positiv() -> None:
    assert config.MIN_CHARS_PER_PAGE > 0
