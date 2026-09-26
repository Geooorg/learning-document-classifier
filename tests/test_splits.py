"""Das Gold-Set ist eingefroren. Diese Zusicherung wird geprueft, nicht geglaubt."""

from pathlib import Path

import polars as pl
import pytest

from doccls.splits import (
    assert_no_gold,
    calibration_documents,
    gold_documents,
    labelled_documents,
    training_documents,
)


def test_jede_menge_ist_nicht_leer() -> None:
    """Eine leere Menge wuerde jeden folgenden Test stillschweigend bestehen lassen."""
    for name, rahmen in [
        ("train", training_documents()),
        ("calib", calibration_documents()),
        ("gold", gold_documents()),
    ]:
        assert rahmen.height > 0, f"Menge {name} ist leer"


def test_die_mengen_ueberschneiden_sich_nicht() -> None:
    ids = [
        set(f["document_id"])
        for f in (training_documents(), calibration_documents(), gold_documents())
    ]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])


def test_keine_vorlage_liegt_in_zwei_mengen() -> None:
    """Schaerfer als die Dokument-Pruefung darueber: Zehn Varianten einer Vorlage sind
    praktisch dasselbe Dokument. Laegen sie auf beiden Seiten, waere jede Metrik geschoent
    (Konzept § 9.2)."""
    vorlagen = [
        set(f["template_id"])
        for f in (training_documents(), calibration_documents(), gold_documents())
    ]
    assert vorlagen[0].isdisjoint(vorlagen[1])
    assert vorlagen[0].isdisjoint(vorlagen[2])
    assert vorlagen[1].isdisjoint(vorlagen[2])


def test_training_enthaelt_kein_gold() -> None:
    assert training_documents().filter(pl.col("split") == "gold").height == 0


def test_kalibrierung_enthaelt_kein_gold() -> None:
    assert calibration_documents().filter(pl.col("split") == "gold").height == 0


def test_assert_no_gold_wirft_bei_gold() -> None:
    """Der Waechter selbst. Ohne diesen Test waere nicht geprueft, ob er ueberhaupt greift."""
    verseucht = pl.DataFrame({"document_id": ["a", "b"], "split": ["train", "gold"]})
    with pytest.raises(ValueError, match="Gold"):
        assert_no_gold(verseucht, "Training")


def test_assert_no_gold_nennt_die_schuldigen_ids() -> None:
    """Eine Meldung ohne die betroffene ID zwingt zur Suche von Hand."""
    verseucht = pl.DataFrame({"document_id": ["harmlos", "SCHULDIG"], "split": ["train", "gold"]})
    with pytest.raises(ValueError, match="SCHULDIG"):
        assert_no_gold(verseucht, "Training")


def test_assert_no_gold_laesst_saubere_mengen_durch() -> None:
    sauber = pl.DataFrame({"document_id": ["a"], "split": ["train"]})
    assert_no_gold(sauber, "Training")


def test_jedes_eingelesene_dokument_traegt_eine_klasse() -> None:
    """Bindet die Menge, nicht nur die Form: Faellt ein join daneben, schrumpft diese
    Tabelle stillschweigend, und alle Metriken danach messen auf weniger Dokumenten."""
    from doccls.config import PARQUET_DIR
    from doccls.pipeline import read_table

    dokumente = read_table(PARQUET_DIR, "documents")
    assert labelled_documents().height == dokumente.height, (
        "Nicht jedes eingelesene Dokument hat eine Klasse – vermutlich fehlen "
        "Manifestzeilen (siehe Aufgabe 1)"
    )


def test_gold_hat_mindestens_50_dokumente_je_klasse() -> None:
    """Konzept § 9.2: Bei weniger sind Unterschiede von zwei Punkten F1 nicht von
    Rauschen zu unterscheiden. Diese Schranke haelt fest, wann das Gold-Set zu klein wird –
    etwa wenn spaeter ein Format aus der Messung genommen wird."""
    je_klasse = gold_documents().group_by("class_key").len()
    zu_klein = je_klasse.filter(pl.col("len") < 50)
    assert zu_klein.height == 0, f"Zu kleine Gold-Klassen: {zu_klein.to_dicts()}"


def test_dokument_ohne_manifestzeile_wird_gemeldet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Der Waechter in ``labelled_documents`` gegen den stillen Verlust.

    Auf dem echten Bestand kann kein Test ihn ausloesen: Seit die Mailanhaenge eine
    Manifestzeile haben, sind ``documents`` und ``manifest`` genau deckungsgleich, und
    ``how="inner"`` liefert dasselbe wie ``how="left"``. Die Mutation ist dort ein
    Nulleffekt – und damit waere der Waechter ungeprueft, obwohl er genau die Luecke
    abdeckt, die dieses Projekt schon einmal zehn Dokumente gekostet hat.

    Deshalb ein praeparierter Bestand: ein Dokument mehr in ``documents`` als im
    Manifest. Faellt der Waechter weg oder wird aus ``left`` ein ``inner``, verschwindet
    die Zeile stillschweigend statt zu melden.
    """
    parquet = tmp_path / "parquet"
    generated = tmp_path / "generated"
    parquet.mkdir()
    generated.mkdir()

    pl.DataFrame(
        {"document_id": ["d1", "d2"], "source_path": ["pdf/a.pdf", "pdf/OHNE-WAHRHEIT.pdf"]}
    ).write_parquet(parquet / "documents.parquet")
    pl.DataFrame(
        {
            "source_path": ["pdf/a.pdf"],
            "class_key": ["RECHNUNG"],
            "template_id": ["T1"],
            "variant": [0],
            "split": ["train"],
            "format": ["pdf"],
        }
    ).write_parquet(generated / "manifest.parquet")

    monkeypatch.setattr("doccls.splits.PARQUET_DIR", parquet)
    monkeypatch.setattr("doccls.splits.GENERATED_DIR", generated)

    with pytest.raises(ValueError, match="OHNE-WAHRHEIT"):
        labelled_documents()
