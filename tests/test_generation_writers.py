"""Erzeugte Dateien müssen byteweise reproduzierbar sein, sonst wandert jede document_id."""

import hashlib
from pathlib import Path

import pytest

from doccls.generation.content import DocumentSpec, build_corpus
from doccls.generation.writers import write

FORMATE = ["pdf", "docx", "xlsx", "eml"]


def _spec(fmt: str) -> DocumentSpec:
    for spec in build_corpus():
        if fmt in spec.formats:
            return spec
    raise AssertionError(f"Keine Vorlage für {fmt}")


@pytest.mark.parametrize("fmt", FORMATE)
def test_zweimal_schreiben_ergibt_denselben_hash(fmt: str, tmp_path: Path) -> None:
    spec = _spec(fmt)
    hashes = set()
    for lauf in ("a", "b"):
        pfad = write(tmp_path / lauf / "dok", spec, fmt)
        hashes.add(hashlib.sha256(pfad.read_bytes()).hexdigest())
    assert len(hashes) == 1, f"{fmt} ist nicht reproduzierbar"


@pytest.mark.parametrize("fmt", FORMATE)
def test_datei_ist_nicht_leer_und_hat_die_richtige_endung(fmt: str, tmp_path: Path) -> None:
    pfad = write(tmp_path / "dok", _spec(fmt), fmt)
    assert pfad.suffix == f".{fmt}"
    assert pfad.stat().st_size > 500


def test_eml_mit_anhang_enthaelt_den_anhang(tmp_path: Path) -> None:
    pfad = write(
        tmp_path / "mail", _spec("eml"), "eml", attachment=("rechnung.pdf", b"%PDF-1.7\nInhalt")
    )
    roh = pfad.read_bytes()
    assert b"rechnung.pdf" in roh
    assert b"multipart" in roh.lower()


def test_unbekanntes_format_wird_abgelehnt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unbekanntes Format"):
        write(tmp_path / "x", _spec("pdf"), "rtf")


def test_horizontaler_ueberlauf_wirft_statt_still_abzuschneiden(tmp_path: Path) -> None:
    """insert_htmlbox meldet nur vertikalen Platzmangel. Eine lange, nicht umbrechbare
    Tabellenzelle verlöre sonst lautlos ihren Inhalt – im Prüfkorpus unsichtbar."""
    from dataclasses import replace

    from doccls.generation.content import Block

    spec = _spec("pdf")
    zu_lang = "X" * 2000
    kaputt = replace(
        spec,
        blocks=(*spec.blocks[:-1], Block(heading="Test", table=(("Spalte",), ((zu_lang,),)))),
    )
    with pytest.raises(ValueError, match="verliert Inhalt"):
        write(tmp_path / "dok", kaputt, "pdf")


def test_eml_mit_anhang_ist_ebenfalls_reproduzierbar(tmp_path: Path) -> None:
    """Die MIME-Grenze ist von Haus aus zufällig; nur mit Anhang wird sie überhaupt gesetzt."""
    spec = _spec("eml")
    hashes = set()
    for lauf in ("a", "b"):
        pfad = write(
            tmp_path / lauf / "mail",
            spec,
            "eml",
            attachment=("rechnung.pdf", b"%PDF-1.7\nInhalt"),
        )
        hashes.add(hashlib.sha256(pfad.read_bytes()).hexdigest())
    assert len(hashes) == 1
