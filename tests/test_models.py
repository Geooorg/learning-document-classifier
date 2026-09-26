"""IDs müssen aus dem Inhalt abgeleitet und damit über Läufe hinweg stabil sein.

Dazu die Wächter von ``Prediction``: Die vier Zahlen einer Vorhersage stammen aus *einer*
Verteilung und können einander widersprechen. Ein Widerspruch ist kein Schönheitsfehler –
er bedeutet, dass Zahl und Klasse aus verschiedenen Quellen stammen. Geprüft wird jede
Ungleichung in beide Richtungen und an **zwei** Konfidenzen, weil beide Schranken aus der
Konfidenz abgeleitet sind: Ein fest eingesetzter Zahlenwert bestünde sonst die Suite.
"""

import math
from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from doccls.classes import load_classes
from doccls.models import (
    RESIDUAL_CLASS_KEY,
    SCHEMAS,
    Decision,
    Document,
    Prediction,
    Segment,
    SegmentKind,
    derive_id,
    normalize_text,
    to_frame,
)


def beispiel_dokument(inhalt: bytes = b"Rechnung 4711") -> Document:
    return Document.create(
        source_path="rechnungen/a.pdf",
        media_type="application/pdf",
        content=inhalt,
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_derive_id_trennt_die_teile() -> None:
    """Ohne Trenner ergäben ("ab", "c") und ("a", "bc") dieselbe ID – und zwei
    verschiedene Dokumente dieselbe Identität."""
    assert derive_id("ab", "c") != derive_id("a", "bc")


def test_derive_id_mischt_bytes_und_text() -> None:
    assert derive_id("pfad.pdf", b"inhalt") == derive_id("pfad.pdf", b"inhalt")
    assert derive_id("pfad.pdf", b"inhalt") != derive_id("pfad.pdf", b"anderes")


def test_id_ist_stabil_und_32_zeichen_lang() -> None:
    a, b = beispiel_dokument(), beispiel_dokument()
    assert a.document_id == b.document_id
    assert len(a.document_id) == 32


def test_geaenderter_inhalt_ergibt_neue_version() -> None:
    assert beispiel_dokument(b"x").document_id != beispiel_dokument(b"y").document_id


def test_gleicher_inhalt_an_zwei_pfaden_sind_zwei_dokumente() -> None:
    """Sonst würde die zweite Fundstelle übersprungen und wäre nirgends verzeichnet."""
    a = Document.create(
        source_path="a/x.pdf",
        media_type="application/pdf",
        content=b"gleich",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    b = Document.create(
        source_path="b/x.pdf",
        media_type="application/pdf",
        content=b"gleich",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert a.document_id != b.document_id
    assert a.content_sha256 == b.content_sha256


def test_segment_id_haengt_am_dokument_und_am_index() -> None:
    doc = beispiel_dokument()
    erst = Segment.create(document=doc, index=0, kind=SegmentKind.SEITE, locator="S. 1", text="a")
    zweit = Segment.create(document=doc, index=1, kind=SegmentKind.SEITE, locator="S. 2", text="a")
    assert erst.segment_id != zweit.segment_id


def test_normalize_text_loest_silbentrennung_und_whitespace_auf() -> None:
    assert normalize_text("Rech-\nnung  über\n\n  100") == "Rechnung über 100"


def test_normalize_text_vereinheitlicht_unicode() -> None:
    assert normalize_text("ﬁnal") == "final"  # Ligatur -> NFKC


def test_to_frame_haelt_sich_an_das_schema() -> None:
    rahmen = to_frame([beispiel_dokument()], "documents")
    assert rahmen.schema == pl.Schema(SCHEMAS["documents"])
    assert rahmen.height == 1


def test_leere_tabelle_hat_trotzdem_das_schema() -> None:
    """Sonst schlägt der erste Lauf beim Schreiben einer leeren Parquet-Datei fehl."""
    assert to_frame([], "segments").schema == pl.Schema(SCHEMAS["segments"])


# --------------------------------------------------------------------- Prediction


def beispiel_vorhersage(**abweichungen: Any) -> Prediction:
    """Eine in sich stimmige Vorhersage; jeder Test verdreht genau ein Feld.

    Die Vorgabewerte gehören zur Verteilung ``0,95 / 0,03 / 0,02``: Konfidenz 0,95,
    Margin 0,92, Entropie 0,2321 – von Hand nachgerechnet, damit kein Test versehentlich
    gegen eine ohnehin verletzte Kombination prüft.
    """
    felder: dict[str, Any] = {
        "document_id": "a3f1",
        "class_key": "RECHNUNG",
        "confidence": 0.95,
        "margin": 0.92,
        "entropy": 0.2321,
        "ood_score": 0.1,
        "decision": Decision.AUTO,
        "model_version": "clf-2026-09-25-r07",
        "feature_version": "feat-0a1b2c3d",
    }
    return Prediction(**(felder | abweichungen))


def _fehlerorte(**abweichungen: Any) -> list[tuple[int | str, ...]]:
    """Wo Pydantic die Vorhersage abweist – Feldname oder ``()`` für das ganze Modell.

    Die Stelle gehört in die Zusicherung, nicht nur die Tatsache einer Ausnahme. Gemessen
    in der Mutationsprobe: Ohne die Wertebereiche an ``confidence``, ``margin`` und
    ``entropy`` wird jeder hier geprüfte Wert trotzdem abgewiesen – nur eben von einer
    *anderen* Ungleichung im Modellvalidator, und mit einer Meldung, die auf das falsche
    Feld zeigt (``margin > confidence`` bei einer negativen Konfidenz). Ein
    ``pytest.raises(ValidationError)`` allein bliebe deshalb grün, wenn die Feldschranken
    ersatzlos verschwänden.
    """
    with pytest.raises(ValidationError) as fehler:
        beispiel_vorhersage(**abweichungen)
    return [e["loc"] for e in fehler.value.errors()]


def test_vorhersage_haelt_ihre_felder() -> None:
    """Der Normalfall darf von keinem Wächter angefasst werden."""
    vorhersage = beispiel_vorhersage()
    assert vorhersage.class_key == "RECHNUNG"
    assert vorhersage.decision is Decision.AUTO
    assert vorhersage.model_version == "clf-2026-09-25-r07"


@pytest.mark.parametrize("konfidenz", [-0.1, 0.0, 1.0001, 1.5])
def test_vorhersage_lehnt_konfidenz_ausserhalb_ihres_bereichs_ab(konfidenz: float) -> None:
    """Eine Konfidenz ist ``max p̂`` und liegt damit in ``(0, 1]``.

    Die 0 gehört dazu: Sie gäbe es nur, wenn jede Klasse die Wahrscheinlichkeit 0 hätte –
    das ist keine Verteilung, und ``−ln(0)`` ist keine Zahl. Und ohne die obere Schranke
    liefe eine Summe statt eines Maximums (0,95 + 0,03 + 0,02 + … > 1) unbemerkt durch.
    """
    assert _fehlerorte(confidence=konfidenz, margin=0.0, entropy=5.0) == [("confidence",)]


@pytest.mark.parametrize(
    ("konfidenz", "margin", "entropie"),
    [
        (0.3, -0.1, 1.5),  # unter 0 – die Schranke 2c − 1 = −0,4 fienge das nicht
        (1.0, 1.5, 0.0),  # über 1
    ],
)
def test_vorhersage_lehnt_eine_margin_ausserhalb_ihres_bereichs_ab(
    konfidenz: float, margin: float, entropie: float
) -> None:
    """Eine Margin ist eine Differenz zweier Wahrscheinlichkeiten und liegt in ``[0, 1]``.

    Die untere Schranke ist nicht doppelt gemoppelt: Bei kleiner Konfidenz lässt
    ``margin >= 2c − 1`` negative Werte durch (bei 0,3 bis −0,4), und eine negative Margin
    höbe die Prüfliste in Phase 3 aus den Angeln – sie sortiert aufsteigend danach.
    """
    assert _fehlerorte(confidence=konfidenz, margin=margin, entropy=entropie) == [("margin",)]


def test_vorhersage_lehnt_eine_negative_entropie_ab() -> None:
    """``−Σ p ln p`` ist nie negativ. Die Konfidenz 1,0 ist hier der Punkt: Nur dort ist
    die untere Schranke ``−ln(c)`` gleich 0, und nur dort zeigt sich, ob das Feld selbst
    eine Untergrenze hat."""
    assert _fehlerorte(confidence=1.0, margin=1.0, entropy=-0.5) == [("entropy",)]


def test_vorhersage_lehnt_eine_margin_ueber_der_konfidenz_ab() -> None:
    """``p̂₍₁₎ − p̂₍₂₎ <= p̂₍₁₎``, weil ``p̂₍₂₎ >= 0``.

    Das ist die Mutation „Margin ist ``p̂₍₁₎`` statt ``p̂₍₁₎ − p̂₍₂₎``" in ihrer harmlos
    aussehenden Form: Sie erzeugt Gleichheit, nicht Überschreitung. Abgewiesen wird
    deshalb nur echtes Größersein – hier eine Margin, die aus einer anderen Verteilung
    stammt als die Konfidenz.
    """
    with pytest.raises(ValidationError, match="groesser als die Konfidenz"):
        beispiel_vorhersage(confidence=0.6, margin=0.7, entropy=0.9)
    # Gleichheit ist moeglich: die zweitbeste Klasse hat dann 0.
    assert beispiel_vorhersage(confidence=1.0, margin=1.0, entropy=0.0).margin == 1.0


@pytest.mark.parametrize(
    ("konfidenz", "margin", "erlaubt"),
    [
        # Schranke 2 * 0,8 − 1 = 0,6
        (0.8, 0.65, True),
        (0.8, 0.55, False),
        # Schranke 2 * 0,95 − 1 = 0,9 – dieselbe Margin 0,65 ist hier unmoeglich
        (0.95, 0.95, True),
        (0.95, 0.65, False),
    ],
)
def test_vorhersage_bindet_die_kleinste_moegliche_margin_an_die_konfidenz(
    konfidenz: float, margin: float, erlaubt: bool
) -> None:
    """``margin >= 2 · confidence − 1``, weil ``p̂₍₁₎ + p̂₍₂₎ <= 1``.

    Die Schranke ist aus der Konfidenz abgeleitet, deshalb steht hier **eine zweite**
    Konfidenz: Margin 0,65 ist bei Konfidenz 0,8 möglich und bei 0,95 unmöglich (die
    zweitbeste Klasse hätte 0,3, zusammen 1,25). Eine fest eingesetzte Zahl bestünde einen
    Test mit nur einer Konfidenz.
    """
    if erlaubt:
        vorhersage = beispiel_vorhersage(confidence=konfidenz, margin=margin, entropy=0.6)
        assert vorhersage.margin == margin
    else:
        with pytest.raises(ValidationError, match="zu klein fuer die Konfidenz"):
            beispiel_vorhersage(confidence=konfidenz, margin=margin, entropy=0.6)


@pytest.mark.parametrize(
    ("konfidenz", "margin", "entropie", "erlaubt"),
    [
        # Schranke −ln(0,5) = 0,693
        (0.5, 0.1, 0.8, True),
        (0.5, 0.1, 0.3, False),
        # Schranke −ln(0,9) = 0,105 – dieselbe Entropie 0,3 ist hier moeglich
        (0.9, 0.85, 0.3, True),
        (0.9, 0.85, 0.05, False),
    ],
)
def test_vorhersage_bindet_die_kleinste_moegliche_entropie_an_die_konfidenz(
    konfidenz: float, margin: float, entropie: float, erlaubt: bool
) -> None:
    """``entropy >= −ln(confidence)``, weil kein ``p̂ᵢ`` größer ist als ``max p̂``.

    Diese Ungleichung ersetzt die Prüfung gegen die Klassenzahl, die das Modell nicht
    kennt. Auch sie ist aus der Konfidenz abgeleitet und steht deshalb an zwei
    Konfidenzen: Entropie 0,3 ist bei Konfidenz 0,9 möglich und bei 0,5 unmöglich.
    """
    if erlaubt:
        assert (
            beispiel_vorhersage(confidence=konfidenz, margin=margin, entropy=entropie).entropy
            == entropie
        )
    else:
        with pytest.raises(ValidationError, match="Entropie"):
            beispiel_vorhersage(confidence=konfidenz, margin=margin, entropy=entropie)


@pytest.mark.parametrize("konfidenz", [0.5, 0.25])
def test_vorhersage_laesst_die_kleinstmoegliche_entropie_genau_zu(konfidenz: float) -> None:
    """Die Gegenrichtung: Genau auf der Schranke ist erlaubt.

    Sonst wäre der Wächter eine Sperre, die den erreichbaren Grenzfall mit abschneidet.
    Beide Konfidenzen sind erreichbar: Die Gleichverteilung über ``1/c`` Klassen (zwei
    bzw. vier) hat Konfidenz ``c``, Margin 0 und genau die Entropie ``−ln(c)``.
    """
    assert beispiel_vorhersage(
        confidence=konfidenz,
        margin=0.0,
        entropy=-math.log(konfidenz),
    ).entropy == pytest.approx(-math.log(konfidenz))


def test_vorhersage_lehnt_einen_ood_wert_ausserhalb_des_kosinusbereichs_ab() -> None:
    """``1 − cos`` liegt in ``[0, 2]``. Ein Wert daneben ist kein Kosinusabstand – etwa
    eine Ähnlichkeit statt eines Abstands (Vorzeichen verdreht)."""
    for abstand in (-0.01, 2.5):
        assert _fehlerorte(ood_score=abstand) == [("ood_score",)]


@pytest.mark.parametrize("feld", ["document_id", "class_key", "model_version", "feature_version"])
def test_vorhersage_verlangt_herkunft_und_bezug(feld: str) -> None:
    """Leere Zeichenketten sind hier keine Angabe, sondern eine fehlende.

    Ohne ``model_version`` und ``feature_version`` ist eine Vorhersage nicht
    nachvollziehbar (Konzept § 1, Leitplanke 2) und zwei Auswertungen sind nicht
    vergleichbar; ohne ``document_id`` gehört sie zu keinem Dokument.
    """
    assert _fehlerorte(**{feld: ""}) == [(feld,)]


def test_vorhersage_kennt_nur_auto_und_review() -> None:
    """``SONSTIGES`` ist eine Klasse, keine Entscheidung (Konzept § 7.4)."""
    assert [d.value for d in Decision] == ["AUTO", "REVIEW"]
    with pytest.raises(ValidationError):
        beispiel_vorhersage(decision="SONSTIGES")


def test_restklasse_stimmt_mit_dem_klassenschema_ueberein() -> None:
    """``RESIDUAL_CLASS_KEY`` und ``config/classes.yaml`` dürfen nicht auseinanderlaufen.

    ``decide_one`` vergibt den Schlüssel, ohne das Schema zu laden. Eine Umbenennung in
    der Konfiguration ohne diesen Test bliebe unbemerkt, und jede Vorhersage trüge danach
    eine Klasse, die es nicht gibt.
    """
    restklassen = [k.key for k in load_classes().classes if k.residual]
    assert restklassen == [RESIDUAL_CLASS_KEY]
