"""Der Gold-Set-Schnitt ist Grundlage jeder späteren Messung. Er wird geprüft, nicht geglaubt."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from doccls.classes import load_classes
from doccls.config import GENERATED_DIR, PARQUET_DIR, RAW_DIR
from doccls.generation.content import DocumentSpec, build_corpus
from doccls.generation.manifest import (
    Split,
    assign_document_names,
    assign_splits,
    compare_with_frozen,
    hat_mailanhang,
    load_frozen_splits,
    manifest_frame,
    report_template_drift,
    save_splits,
)
from doccls.models import ANHANG_TRENNER
from doccls.pipeline import read_table


def _pfade_wie_generator(korpus: list[DocumentSpec], namen: list[str], root: Path) -> list[Path]:
    """Baut dieselben Pfade, die ``scripts/generate_documents.py`` schreiben würde – ``<root>/
    <erstes Format>/<neutraler Name>.<eml oder erstes Format>`` –, ohne etwas auf die Platte
    zu schreiben. So lassen sich ``manifest_frame`` und ``anhang_namen`` mit realistischen
    Pfaden prüfen, ohne den vollen Generatorlauf (PDF/DOCX/XLSX-Rendern) zu wiederholen.
    """
    pfade = []
    for spec, name in zip(korpus, namen, strict=True):
        fmt = spec.formats[0]
        endformat = "eml" if hat_mailanhang(spec) else fmt
        pfade.append(root / fmt / f"{name}.{endformat}")
    return pfade


def test_neue_vorlage_verschiebt_bestehende_nicht() -> None:
    """Der Kern der Sache: Eine Vorlage, die heute im Gold-Set ist, muss morgen dort sein.
    Vor der Umstellung auf config/splits.yaml verschob eine einzige neue Vorlage mehrere von
    acht bestehenden über die Schnittgrenze. Die neue Vorlage selbst muss in die Trainingsmenge
    fallen – das ist der eigentliche Regelfall (die Klasse hat bereits genug Gold und Kalibrier)."""
    korpus = build_corpus()
    fest = assign_splits(korpus)
    neue_vorlage = replace(korpus[0], template_id="NEU-TESTVORLAGE", variant=0)
    erweitert = [*korpus, neue_vorlage]
    danach = assign_splits(erweitert, frozen=fest)
    assert {k: v for k, v in danach.items() if k in fest} == fest
    assert danach["NEU-TESTVORLAGE"] is Split.TRAIN


def test_schnitt_ist_deterministisch() -> None:
    assert assign_splits(build_corpus()) == assign_splits(build_corpus())


def test_splits_datei_rundlauf(tmp_path: Path) -> None:
    ziel = tmp_path / "splits.yaml"
    splits = {"AGB-allgemein": Split.GOLD, "AGB-einkauf": Split.TRAIN}
    save_splits(splits, path=ziel)
    assert load_frozen_splits(path=ziel) == splits


def test_fehlender_gold_platz_wird_gemeldet() -> None:
    """Verschwindet eine eingefrorene Gold-Vorlage, darf das Gold-Set nicht stillschweigend
    unter die Sollzahl fallen – dann wären alle Messungen davor und danach unvergleichbar."""
    korpus = build_corpus()
    fest = assign_splits(korpus)
    entfernte_vorlage = next(v for v, s in sorted(fest.items()) if s is Split.GOLD)
    klasse = next(spec.class_key for spec in korpus if spec.template_id == entfernte_vorlage)
    gekuerzt = [spec for spec in korpus if spec.template_id != entfernte_vorlage]
    with pytest.raises(ValueError, match=klasse):
        assign_splits(gekuerzt, frozen=fest)


def test_fehlender_calib_platz_wird_gemeldet() -> None:
    """Spiegelbild zum Gold-Test: die Prüfung behandelt Gold und Kalibrierung symmetrisch
    (``if gold_frei or calib_frei``) – ohne diesen Test bliebe eine Verengung auf nur
    ``gold_frei`` unentdeckt."""
    korpus = build_corpus()
    fest = assign_splits(korpus)
    entfernte_vorlage = next(v for v, s in sorted(fest.items()) if s is Split.CALIB)
    klasse = next(spec.class_key for spec in korpus if spec.template_id == entfernte_vorlage)
    gekuerzt = [spec for spec in korpus if spec.template_id != entfernte_vorlage]
    with pytest.raises(ValueError, match=klasse):
        assign_splits(gekuerzt, frozen=fest)


def test_compare_with_frozen_findet_beide_richtungen() -> None:
    """Die Erkennung, nicht die Ausgabe: eine Vorlage ohne Eintrag und ein Eintrag ohne
    Vorlage sind zwei verschiedene Widersprüche und müssen beide auffallen."""
    korpus = build_corpus()
    vorhanden = next(spec.template_id for spec in korpus)
    frozen = {v: Split.TRAIN for v in {s.template_id for s in korpus} if v != vorhanden}
    frozen["ALT-VORLAGE-GIBT-ES-NICHT-MEHR"] = Split.GOLD

    neu, verwaist = compare_with_frozen(korpus, frozen)

    assert neu == [vorhanden]
    assert verwaist == ["ALT-VORLAGE-GIBT-ES-NICHT-MEHR"]


def test_compare_with_frozen_meldet_nichts_wenn_alles_stimmt() -> None:
    korpus = build_corpus()
    frozen = {s.template_id: Split.TRAIN for s in korpus}
    assert compare_with_frozen(korpus, frozen) == ([], [])


def test_neue_vorlage_wird_gemeldet(capsys: pytest.CaptureFixture[str]) -> None:
    """Spiegelbild zu verwaist: auch eine neue, noch nicht eingetragene Vorlage muss
    gemeldet werden – nicht nur der umgekehrte Fall."""
    report_template_drift(["NEU-VORLAGE"], [])
    ausgabe = capsys.readouterr().out
    assert "NEU-VORLAGE" in ausgabe


def test_verwaister_eintrag_wird_gemeldet(capsys: pytest.CaptureFixture[str]) -> None:
    """Ein Eintrag in splits.yaml ohne zugehörige Vorlage ist ein Widerspruch, kein Detail."""
    report_template_drift([], ["ALT-VORLAGE"])
    ausgabe = capsys.readouterr().out
    assert "ALT-VORLAGE" in ausgabe


def test_save_splits_ueberlebt_einen_abbruch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Die Datei ist die einzige Quelle der Wahrheit für den Schnitt. Ein abgebrochener
    Schreibvorgang darf den bestehenden Stand nicht zerstören."""
    ziel = tmp_path / "splits.yaml"
    save_splits({"AGB-allgemein": Split.GOLD}, path=ziel)
    inhalt_vorher = ziel.read_text(encoding="utf-8")

    def abbruch(self: Path, *args: object, **kwargs: object) -> Path:
        raise OSError("Abbruch simuliert")

    monkeypatch.setattr(Path, "replace", abbruch)
    with pytest.raises(OSError):
        save_splits({"AGB-allgemein": Split.TRAIN}, path=ziel)

    assert ziel.read_text(encoding="utf-8") == inhalt_vorher


def test_jede_klasse_hat_mindestens_50_gold_dokumente() -> None:
    korpus = build_corpus()
    splits = assign_splits(korpus)
    gold = Counter(s.class_key for s in korpus if splits[s.template_id] is Split.GOLD)
    for klasse in load_classes().keys():
        assert gold[klasse] >= 50, f"{klasse}: nur {gold[klasse]} Gold-Dokumente"


def test_jede_klasse_kommt_in_jedem_split_vor() -> None:
    korpus = build_corpus()
    splits = assign_splits(korpus)
    for klasse in load_classes().keys():
        vorhanden = {splits[s.template_id] for s in korpus if s.class_key == klasse}
        assert vorhanden == set(Split), f"{klasse} fehlt in {set(Split) - vorhanden}"


def test_mailanhang_steht_mit_im_manifest() -> None:
    """Der Anhang entsteht erst beim Einlesen, nicht beim Erzeugen – ohne eigene Zeile
    verliert ihn jeder join von documents auf manifest stillschweigend."""
    korpus = build_corpus()
    splits = assign_splits(korpus)
    assert any(hat_mailanhang(s) for s in korpus), "Keine Vorlage traegt einen Mailanhang"
    namen = assign_document_names(korpus)
    pfade = _pfade_wie_generator(korpus, namen, Path("."))

    rahmen = manifest_frame(korpus, splits, pfade, Path("."))

    anhaenge = rahmen.filter(pl.col("source_path").str.contains(ANHANG_TRENNER, literal=True))
    assert anhaenge.height > 0, "Kein Anhang im Manifest – die Mailvorlagen tragen keinen"
    for zeile in anhaenge.iter_rows(named=True):
        eltern_pfad = zeile["source_path"].split(ANHANG_TRENNER)[0]
        eltern = rahmen.filter(pl.col("source_path") == eltern_pfad)
        assert eltern.height == 1, f"Elternmail {eltern_pfad} fehlt im Manifest"
        assert zeile["split"] == eltern["split"][0], (
            f"{zeile['source_path']} liegt im Split {zeile['split']}, die Elternmail "
            f"{eltern_pfad} aber in {eltern['split'][0]} – zwei Dokumente mit 0,95 "
            "Textaehnlichkeit auf beiden Seiten des Gold-Schnitts"
        )
        assert zeile["class_key"] == eltern["class_key"][0]
        assert zeile["format"] == "pdf", (
            f"{zeile['source_path']}: Format {zeile['format']!r}, erwartet 'pdf' – "
            "Mailanhaenge sind in diesem Korpus immer PDFs, unabhaengig vom Anhangnamen"
        )


def test_manifest_frame_stimmt_mit_eingelesenem_bestand_ueberein() -> None:
    """Bindet ``manifest_frame`` selbst an den eingelesenen Bestand.

    ``test_jedes_eingelesene_dokument_hat_eine_wahrheit`` liest die auf Platte liegende
    ``manifest.parquet`` – eine Mutation in ``manifest_frame`` oder ``anhang_namen`` sieht
    dieser Test also gar nicht, solange niemand neu erzeugt und neu eingelesen hat. Hier
    wird ``manifest_frame`` mit den echten Namen (``assign_document_names``) und den Pfaden,
    die der Generator erzeugen würde, direkt aufgerufen und sein Ergebnis gegen den
    tatsächlich eingelesenen Bestand (``data/parquet/documents.parquet``) gestellt.
    """
    korpus = build_corpus()
    splits = assign_splits(korpus, frozen=load_frozen_splits())
    namen = assign_document_names(korpus)
    pfade = _pfade_wie_generator(korpus, namen, RAW_DIR)

    rahmen = manifest_frame(korpus, splits, pfade, root=RAW_DIR)
    dokumente = read_table(PARQUET_DIR, "documents")

    berechnet = set(rahmen["source_path"])
    eingelesen = set(dokumente["source_path"])
    nur_berechnet = berechnet - eingelesen
    nur_eingelesen = eingelesen - berechnet
    assert not nur_berechnet and not nur_eingelesen, (
        f"{len(nur_berechnet)} von manifest_frame berechnete Pfade fehlen im eingelesenen "
        f"Bestand: {sorted(nur_berechnet)[:5]}; "
        f"{len(nur_eingelesen)} eingelesene Pfade fehlen im berechneten Manifest: "
        f"{sorted(nur_eingelesen)[:5]}"
    )


def test_jedes_eingelesene_dokument_hat_eine_wahrheit() -> None:
    """Die Gegenprobe am echten Bestand: documents und manifest müssen deckungsgleich sein.

    Beide Richtungen zählen: Ein eingelesenes Dokument ohne Manifestzeile bliebe unbemerkt,
    sobald eine neue Dokumentart hinzukommt, die erst beim Einlesen entsteht (etwa ein
    ZIP-Eintrag oder eine verschachtelte Mail). Eine Manifestzeile ohne eingelesenes
    Dokument – etwa weil ``anhang_namen`` einen anderen Namen liefert als tatsächlich
    eingelesen wurde – fiele bei einem reinen ``anti``-Join von documents auf manifest nie
    auf, weil der nur die erste Richtung prüft.
    """
    dokumente = read_table(PARQUET_DIR, "documents")
    manifest = pl.read_parquet(GENERATED_DIR / "manifest.parquet")
    pfade_dokumente = set(dokumente["source_path"])
    pfade_manifest = set(manifest["source_path"])
    ohne_wahrheit = pfade_dokumente - pfade_manifest
    ohne_dokument = pfade_manifest - pfade_dokumente
    assert not ohne_wahrheit and not ohne_dokument, (
        f"{len(ohne_wahrheit)} eingelesene Dokumente ohne Zeile im Manifest: "
        f"{sorted(ohne_wahrheit)[:5]}; "
        f"{len(ohne_dokument)} Manifestzeilen ohne eingelesenes Dokument: "
        f"{sorted(ohne_dokument)[:5]}"
    )
