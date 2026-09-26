"""Die drei Merkmalsbloecke zu einer Matrix zusammenfassen – und die versionierte Ablage.

Die Fallgrube dieser Aufgabe ist eine Reihenfolge, keine Formel: Der n-Gramm-Block darf
nur auf den Trainingstexten angepasst werden. Wer ihn auf allen Dokumenten (auch dem
Gold-Set) anpasst, gibt dem Modell dessen Wortstatistik mit – die gemessenen Zahlen sehen
dann besser aus, als sie sind. Anders als in Aufgabe 7 liegen Training und Gold hier zum
ersten Mal getrennt vor, deshalb ist die Pruefung erst hier faellig.
"""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from doccls.config import PARQUET_DIR
from doccls.features import (
    BLOCK_ORDER,
    FeatureConfig,
    FeatureMatrix,
    build_matrix,
    read_features,
    write_features,
)
from doccls.features.ngrams import NgramBlock, build_ngram_block
from doccls.models import Document, Segment, SegmentKind, to_frame
from doccls.pipeline import read_table


class FakeEmbedder:
    """Wie in ``tests/test_features_embedding.py`` – ohne ein 1-GB-Modell zu laden.

    Der Test hier soll die Zusammensetzung pruefen, nicht das Modell.
    """

    name = "fake"
    dimension = 4

    def __init__(self) -> None:
        self.gesehen: list[str] = []

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        self.gesehen.extend(texts)
        return np.array([[float(len(t)), 1.0, 0.0, 0.0] for t in texts], dtype=np.float32)


class AufzeichnenderEmbedder:
    """Wie ``FakeEmbedder``, aber mit waehlbarem Namen – fuer den Praefix-Test (Nachtrag
    des Steuernden zu Aufgabe 5)."""

    dimension = 4

    def __init__(self, name: str) -> None:
        self.name = name
        self.gesehen: list[str] = []

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        self.gesehen.extend(texts)
        return np.zeros((len(texts), self.dimension), dtype=np.float32)


def _konfiguration(
    *,
    svd_components: int = 2,
    ngram_max_features: int = 500,
    chunk_chars: int = 200,
    head_chars: int = 100,
) -> FeatureConfig:
    return FeatureConfig(
        svd_components=svd_components,
        ngram_max_features=ngram_max_features,
        chunk_chars=chunk_chars,
        head_chars=head_chars,
    )


def _dokumente_und_segmente(anzahl: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Eine Handvoll echter Dokumente aus dem Bestand, samt ihren Segmenten."""
    dokumente = read_table(PARQUET_DIR, "documents").head(anzahl)
    ids = dokumente["document_id"].to_list()
    segmente = read_table(PARQUET_DIR, "segments").filter(pl.col("document_id").is_in(ids))
    return dokumente, segmente


def _kleine_matrix() -> FeatureMatrix:
    dokumente, segmente = _dokumente_und_segmente(6)
    trainings_ids = set(dokumente["document_id"].to_list()[:4])
    return build_matrix(dokumente, segmente, trainings_ids, _konfiguration(), FakeEmbedder())


def test_ngramme_werden_nur_auf_trainingstexten_angepasst() -> None:
    """Die klassische Leckage dieser Art von Pipeline – und der Grund, warum diese
    Pruefung hier steht und nicht in Aufgabe 7: Erst hier liegen Training und Gold
    getrennt vor.

    Ein Wort, das AUSSCHLIESSLICH in Gold-Texten vorkommt, darf im angepassten Vokabular
    nicht auftauchen. Taete es das, haette das Modell die Wortstatistik des Gold-Sets
    gesehen, und jede gemessene Zahl waere zu gut.
    """
    marker = "zzqxwvunbekannteszeichenmuster"
    trainingstexte = ["Rechnung ueber Wartung", "Vertrag ueber Miete"]
    goldtexte = [f"Protokoll der Sitzung {marker}"]

    block = build_ngram_block(FeatureConfig(svd_components=1, ngram_max_features=500))
    block.fit(trainingstexte)

    vokabular = " ".join(block.vokabular())
    assert marker[:6] not in vokabular, (
        "Eine Zeichenfolge, die nur im Gold-Set vorkommt, steht im angepassten Vokabular"
    )
    block.transform(goldtexte)  # darf nicht werfen


def _mit_zusatzdokument(
    dokumente: pl.DataFrame, segmente: pl.DataFrame, text: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ein zusaetzliches, frei erfundenes Dokument an einen echten Bestandsausschnitt
    anhaengen – fuer den direkten Nachweis, dass ein Dokument ausserhalb von
    ``training_document_ids`` den n-Gramm-Block nicht beeinflusst."""
    zusatz_dokument = Document.create(
        source_path="pdf/zusatz-fuer-leckagetest.pdf",
        media_type="application/pdf",
        content=text.encode("utf-8"),
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    zusatz_segment = Segment.create(
        document=zusatz_dokument, index=0, kind=SegmentKind.SEITE, locator="S. 1", text=text
    )
    neue_dokumente = pl.concat([dokumente, to_frame([zusatz_dokument], "documents")])
    neue_segmente = pl.concat([segmente, to_frame([zusatz_segment], "segments")])
    return neue_dokumente, neue_segmente


def test_build_matrix_ngramme_ignorieren_nicht_trainingsdokumente() -> None:
    """Der direkte Nachweis am Zusammenspiel, nicht nur am ``NgramBlock`` allein: Ein
    Dokument ausserhalb von ``training_document_ids`` darf die n-Gramm-Anpassung nicht
    beeinflussen. Sonst veraenderte ein spaeter hinzugefuegtes Gold-Dokument lautlos die
    Merkmale der Trainingsdokumente, obwohl ``fit`` laut Vertrag nur Trainingstexte sehen
    darf.

    Geprueft wird schwarzkastenartig ueber die Ausgabe: Ist der n-Gramm-Block wirklich nur
    auf den Trainingsdokumenten angepasst, aendert ein zusaetzliches, nicht trainierendes
    Dokument die Ngramm-Spalten der Trainingsdokumente nicht – TF-IDF und SVD werden ja mit
    denselben Texten in derselben Reihenfolge angepasst.
    """
    # Mindestens drei Dokumente auf beiden Seiten des Vergleichs: strip_boilerplate schaltet
    # seine Erkennung unterhalb von drei Texten komplett ab (normalize.MIN_PAGES_FOR_DETECTION).
    # Bliebe die Basis bei zwei Dokumenten, wechselte allein das Hinzufuegen des dritten,
    # nicht-trainierenden Dokuments den Modus (aus -> an) und veraenderte die bereinigten
    # Trainingstexte – ein Alarm, der nichts mit n-Gramm-Leckage zu tun haette.
    basis_dokumente, basis_segmente = _dokumente_und_segmente(5)
    basis_ids = set(basis_dokumente["document_id"].to_list())
    konfiguration = _konfiguration(svd_components=1)

    ohne_drittes = build_matrix(
        basis_dokumente, basis_segmente, basis_ids, konfiguration, FakeEmbedder()
    )

    marker_text = ("zzqxwvunbekannteszeichenmuster " * 200).strip()
    mit_dokumente, mit_segmente = _mit_zusatzdokument(basis_dokumente, basis_segmente, marker_text)
    mit_drittem = build_matrix(
        mit_dokumente, mit_segmente, basis_ids, konfiguration, FakeEmbedder()
    )

    assert ohne_drittes.block_slices == mit_drittem.block_slices
    index_ohne = {doc_id: i for i, doc_id in enumerate(ohne_drittes.document_ids)}
    index_mit = {doc_id: i for i, doc_id in enumerate(mit_drittem.document_ids)}
    ngram_bereich = ohne_drittes.block_slices["ngram"]
    for doc_id in basis_ids:
        a = ohne_drittes.X[index_ohne[doc_id], ngram_bereich]
        b = mit_drittem.X[index_mit[doc_id], ngram_bereich]
        assert np.allclose(a, b), (
            f"Ngramm-Merkmale von Dokument {doc_id} aendern sich durch ein drittes "
            "Dokument, das nicht in training_document_ids steht – der n-Gramm-Block "
            "wurde offenbar auf mehr als den Trainingstexten angepasst"
        )


def test_build_matrix_nutzt_praefix_fuer_nach_dem_embeddernamen() -> None:
    """Nachtrag zu Aufgabe 5: ``praefix_fuer(embedder)`` war bis hierher ungebunden – eine
    Mutation, die immer ``e5_prefix`` liefert, ueberlebte die gesamte Suite, weil es bis
    Aufgabe 8 keinen Aufrufer gab. ``build_matrix`` ist jetzt der Aufrufer: Ein E5-Embedder
    muss das Praefix vorgelegt bekommen, ein BGE-M3-Embedder nicht.
    """
    dokumente, segmente = _dokumente_und_segmente(2)
    ids = set(dokumente["document_id"].to_list())
    konfiguration = _konfiguration(svd_components=1)

    e5 = AufzeichnenderEmbedder(name="e5")
    build_matrix(dokumente, segmente, ids, konfiguration, e5)
    assert e5.gesehen, "Dem Embedder wurde gar nichts vorgelegt"
    assert all(text.startswith("passage: ") for text in e5.gesehen), (
        "E5 bekommt nicht bei jedem Text das Praefix vorgelegt"
    )

    bge = AufzeichnenderEmbedder(name="bge-m3")
    build_matrix(dokumente, segmente, ids, konfiguration, bge)
    assert bge.gesehen, "Dem Embedder wurde gar nichts vorgelegt"
    assert not any(text.startswith("passage: ") for text in bge.gesehen), (
        "BGE-M3 bekommt das E5-Praefix vorgelegt – das verschoebe jeden Vektor, ohne dass "
        "irgendetwas fehlschluege"
    )


def test_matrix_hat_die_summe_der_blockbreiten() -> None:
    """Bindet die Zusammensetzung: Faellt ein Block weg, aendert sich die Breite."""
    matrix = _kleine_matrix()
    erwartet = sum(s.stop - s.start for s in matrix.block_slices.values())
    assert matrix.X.shape[1] == erwartet
    assert set(matrix.block_slices) == set(BLOCK_ORDER)


def test_kein_block_ist_durchgehend_null() -> None:
    """Der stille Fehler dieser Aufgabe: Ein Block wird gebaut, aber nie befuellt, und
    die Matrix hat trotzdem die richtige Breite. Eine reine Formpruefung saehe nichts."""
    matrix = _kleine_matrix()
    for name, bereich in matrix.block_slices.items():
        teil = matrix.X[:, bereich]
        assert float(np.abs(teil).sum()) > 0.0, f"Block {name} ist durchgehend null"


def test_jede_zeile_traegt_inhalt() -> None:
    matrix = _kleine_matrix()
    leere = [i for i in range(matrix.X.shape[0]) if not np.any(matrix.X[i])]
    assert not leere, f"Zeilen ohne jeden Wert: {leere}"


def test_keine_nan_und_kein_unendlich() -> None:
    matrix = _kleine_matrix()
    assert np.all(np.isfinite(matrix.X)), "NaN oder inf vergiften jedes Training lautlos"


def test_rundlauf_ueber_parquet_erhaelt_die_werte(tmp_path: Path) -> None:
    """Bindet die Werte, nicht nur die Form: Eine Umsetzung, die beim Lesen Nullen
    liefert, bestuende jede Formpruefung."""
    matrix = _kleine_matrix()
    write_features(matrix, tmp_path)
    zurueck = read_features(tmp_path, matrix.feature_version)
    assert zurueck.document_ids == matrix.document_ids
    assert np.allclose(zurueck.X, matrix.X)
    assert zurueck.block_slices == matrix.block_slices


def test_lesen_mit_falscher_version_wirft(tmp_path: Path) -> None:
    """Konzept § 6.4: Ein Modell darf nur Merkmale derselben Version sehen. Ohne diese
    Pruefung rechnete es nach einer Merkmalsaenderung stillschweigend weiter."""
    matrix = _kleine_matrix()
    write_features(matrix, tmp_path)
    with pytest.raises(FileNotFoundError, match="andere-version"):
        read_features(tmp_path, "andere-version")


def test_leeres_training_wird_gemeldet() -> None:
    """Der Aufrufer muss ``training_document_ids`` selbst richtig befuellen – ohne
    Trainingsdokumente kann der n-Gramm-Block nicht angepasst werden. Eine stille
    Rueckgabe von Nullen waere schlimmer als ein klarer Fehler."""
    dokumente, segmente = _dokumente_und_segmente(3)
    with pytest.raises(ValueError, match="training_document_ids"):
        build_matrix(dokumente, segmente, set(), _konfiguration(), FakeEmbedder())


def test_vokabular_ohne_fit_wirft() -> None:
    """Gegenprobe zu ``vokabular()``: Vor ``fit`` gibt es kein Vokabular – dieselbe
    Absicherung wie bei ``transform``."""
    block: NgramBlock = build_ngram_block(_konfiguration())
    with pytest.raises(RuntimeError, match="fit"):
        block.vokabular()
