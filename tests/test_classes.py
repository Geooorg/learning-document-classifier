"""Das Klassenschema ist Konfiguration. Fehler darin müssen beim Laden auffallen, nicht später."""

from pathlib import Path

import pytest

from doccls.classes import load_classes


def test_schema_laedt_und_enthaelt_die_erwarteten_klassen() -> None:
    schema = load_classes()
    assert schema.version == 1
    assert set(schema.keys()) == {
        "RECHNUNG",
        "GUTSCHRIFT",
        "VERTRAG",
        "AGB",
        "STATUSBERICHT",
        "PROTOKOLL",
        "SONSTIGES",
    }


def test_sonstiges_ist_die_restklasse_und_nicht_trainierbar() -> None:
    schema = load_classes()
    assert schema.get("SONSTIGES").residual is True
    assert "SONSTIGES" not in [c.key for c in schema.trainable()]
    assert len(schema.trainable()) == 6


def test_jede_klasse_hat_eine_abgrenzung() -> None:
    """Ohne `not` ist die Klasse für Labelnde nicht abgrenzbar – und κ sinkt (Konzept § 9.5)."""
    for klasse in load_classes().classes:
        assert klasse.not_.strip(), f"{klasse.key} ohne Abgrenzung"
        assert len(klasse.description.split()) >= 15, f"{klasse.key}: Beschreibung zu knapp"


def test_doppelter_schluessel_wird_abgelehnt(tmp_path: Path) -> None:
    pfad = tmp_path / "classes.yaml"
    pfad.write_text(
        "version: 1\n"
        "classes:\n"
        "  - {key: A, name: A, description: x, not: y}\n"
        "  - {key: A, name: B, description: x, not: y}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="doppelt"):
        load_classes(pfad)
