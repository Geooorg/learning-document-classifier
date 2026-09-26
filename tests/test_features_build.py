"""Die drei Merkmalsbloecke zu einer Matrix zusammenfassen – und die versionierte Ablage.

Die Fallgrube dieser Aufgabe ist eine Reihenfolge, keine Formel: Der n-Gramm-Block darf
nur auf den Trainingstexten angepasst werden. Wer ihn auf allen Dokumenten (auch dem
Gold-Set) anpasst, gibt dem Modell dessen Wortstatistik mit – die gemessenen Zahlen sehen
dann besser aus, als sie sind. Anders als in Aufgabe 7 liegen Training und Gold hier zum
ersten Mal getrennt vor, deshalb ist die Pruefung erst hier faellig.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

import doccls.features.ngrams as ngrams_module
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
from doccls.features.structural import STRUCTURAL_NAMES
from doccls.features.text import chunks, document_text, head_text, position_weights
from doccls.models import Segment, SegmentKind
from doccls.pipeline import read_table
from doccls.splits import gold_documents, training_documents


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
    """Eine Handvoll echter Dokumente aus dem Bestand, samt ihren Segmenten.

    Nur fuer Faelle, in denen die IDs NICHT als ``training_document_ids`` an
    ``build_matrix`` gehen (z. B. die leere Trainingsmenge unten): ``head(anzahl)`` liefert
    irgendwelche Dokumente in Ablagereihenfolge, und der Bestand ist zu 360 von 570
    Dokumenten Gold – die ersten paar Zeilen der Tabelle sind es fast immer. Fuer echte
    ``training_document_ids`` gibt es ``_trainingsdokumente`` weiter unten.
    """
    dokumente = read_table(PARQUET_DIR, "documents").head(anzahl)
    ids = dokumente["document_id"].to_list()
    segmente = read_table(PARQUET_DIR, "segments").filter(pl.col("document_id").is_in(ids))
    return dokumente, segmente


def _trainingsdokumente(anzahl: int) -> tuple[pl.DataFrame, pl.DataFrame, set[str]]:
    """``anzahl`` echte Dokumente aus dem tatsaechlichen Trainingssplit, komplett mit
    Zeilen und Segmenten – und genau ihre IDs als gueltige ``training_document_ids``.

    Befund F: ``build_matrix`` prueft inzwischen selbst, dass ``training_document_ids``
    goldfrei ist. Ein Test darf ihr deshalb keine beliebigen Dokument-IDs unterschieben –
    er wuerde sofort an der eigenen Zusicherung scheitern, die er gar nicht pruefen will.
    """
    trainings_ids = set(training_documents()["document_id"].to_list()[:anzahl])
    dokumente = read_table(PARQUET_DIR, "documents").filter(
        pl.col("document_id").is_in(trainings_ids)
    )
    segmente = read_table(PARQUET_DIR, "segments").filter(
        pl.col("document_id").is_in(trainings_ids)
    )
    return dokumente, segmente, trainings_ids


def _kleine_matrix() -> FeatureMatrix:
    dokumente, segmente, trainings_ids = _trainingsdokumente(6)
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


def _mit_echtem_zusatzdokument(
    dokumente: pl.DataFrame, segmente: pl.DataFrame, ausgeschlossen: set[str]
) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    """Ein echtes, nicht in ``ausgeschlossen`` enthaltenes Bestandsdokument anhaengen.

    Ein Gold-Dokument passt inhaltlich am besten: Genau das Durchsickern von Gold-Text in
    den n-Gramm-Fit ist das Szenario, gegen das dieser Test schuetzt. Ein zusaetzliches,
    200-mal wiederholtes Kunstwort (die fruehere Fassung) ist eine fast konstante Spalte
    (nach TruncatedSVD mit ``svd_components=1``) und liegt in der ``np.allclose``-Toleranz
    von drei der fuenf geprueften Dokumente – ein echtes Dokument streut ueber die
    TF-IDF-Gewichte hinweg deutlich mehr (gemessene groesste Abweichung: Faktor 7)."""
    zusatz_id = next(
        i for i in gold_documents()["document_id"].to_list() if i not in ausgeschlossen
    )
    alle_dokumente = read_table(PARQUET_DIR, "documents")
    alle_segmente = read_table(PARQUET_DIR, "segments")
    neue_dokumente = pl.concat(
        [dokumente, alle_dokumente.filter(pl.col("document_id") == zusatz_id)]
    )
    neue_segmente = pl.concat([segmente, alle_segmente.filter(pl.col("document_id") == zusatz_id)])
    return neue_dokumente, neue_segmente, zusatz_id


def test_build_matrix_ngramme_ignorieren_nicht_trainingsdokumente(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Der direkte Nachweis am Zusammenspiel, nicht nur am ``NgramBlock`` allein: Ein
    Dokument ausserhalb von ``training_document_ids`` darf die n-Gramm-Anpassung nicht
    beeinflussen. Sonst veraenderte ein spaeter hinzugefuegtes Gold-Dokument lautlos die
    Merkmale der Trainingsdokumente, obwohl ``fit`` laut Vertrag nur Trainingstexte sehen
    darf.

    Zwei Bindungen, nicht nur eine:

    1. **Das tatsaechlich von ``build_matrix`` angepasste Vokabular** – ``build_ngram_block``
       wird abgefangen (``monkeypatch``, dasselbe lokale-Import-Muster wie in
       ``build_matrix`` selbst), sodass der reale ``NgramBlock`` greifbar ist. Bleibt
       ``training_document_ids`` unveraendert, muss ``fit`` auf denselben Texten laufen und
       ``vokabular()`` muss exakt gleich sein – unabhaengig von jeder Zahlentoleranz. Das ist
       die robustere Bindung, die der Pruefer nach Befund B verlangt hat: Eine fast
       konstante Spalte kann in ``np.allclose`` zufaellig durchrutschen, ein unterschiedliches
       Vokabular nie.
    2. **Die Ngramm-Spalten selbst** (wie zuvor), jetzt mit einem echten Bestandsdokument
       statt einem 200-mal wiederholten Kunstwort als Zusatzdokument – siehe
       ``_mit_echtem_zusatzdokument``.
    """
    basis_dokumente, basis_segmente, basis_ids = _trainingsdokumente(5)
    konfiguration = _konfiguration(svd_components=1)

    gefangene_bloecke: list[NgramBlock] = []
    original_build_ngram_block = ngrams_module.build_ngram_block

    def spion(config: FeatureConfig) -> NgramBlock:
        block = original_build_ngram_block(config)
        gefangene_bloecke.append(block)
        return block

    monkeypatch.setattr(ngrams_module, "build_ngram_block", spion)

    ohne_drittes = build_matrix(
        basis_dokumente, basis_segmente, basis_ids, konfiguration, FakeEmbedder()
    )
    mit_dokumente, mit_segmente, _zusatz_id = _mit_echtem_zusatzdokument(
        basis_dokumente, basis_segmente, basis_ids
    )
    mit_drittem = build_matrix(
        mit_dokumente, mit_segmente, basis_ids, konfiguration, FakeEmbedder()
    )

    assert len(gefangene_bloecke) == 2, "build_ngram_block wurde nicht zweimal aufgerufen"
    assert gefangene_bloecke[0].vokabular() == gefangene_bloecke[1].vokabular(), (
        "Das von build_matrix tatsaechlich angepasste Vokabular aendert sich durch ein "
        "zusaetzliches Dokument ausserhalb von training_document_ids – fit hat mehr als "
        "die Trainingstexte gesehen"
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
    dokumente, segmente, ids = _trainingsdokumente(2)
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


def test_jede_zeile_ist_ihrem_dokument_zugeordnet() -> None:
    """Bindet Zeile ``i`` an eine von ``build_matrix`` voellig unabhaengig bekannte
    Wahrheit – nicht nur an Form und Nichtnullheit wie die beiden Tests oben.

    Vier Mutationen liefern dieselbe Breite und dieselbe Nichtnullheit und ordnen trotzdem
    jedem Dokument fremde Merkmale zu: rotierte Strukturzeilen, rotierte Embeddingzeilen,
    eine um eine Spalte verschobene Grenze zwischen ``ngram`` und ``structural``, und
    vertauschte ``emb_mean``/``emb_head``-Bloecke. Alle vier werden von dieser einen
    Zusicherung gefangen (siehe Mutationsprobe im Bericht):

    * ``segment_count`` im Strukturblock gegen die tatsaechliche, direkt aus der
      Segmenttabelle gezaehlte Segmentanzahl – faengt rotierte Strukturzeilen UND eine
      verschobene Blockgrenze (dahinter stuenden dann fremde Werte).
    * Spalte 0 von ``emb_mean``/``emb_head`` gegen die mit ``FakeEmbedder`` unabhaengig
      nachgerechnete Chunk- bzw. Kopflaenge (``FakeEmbedder`` legt dort die Textlaenge ab)
      – faengt rotierte Embeddingzeilen UND einen emb_mean/emb_head-Tausch, weil beide
      Groessen bei realen, unterschiedlich langen Bestandsdokumenten verschieden ausfallen.
    """
    dokumente, segmente, trainings_ids = _trainingsdokumente(6)
    konfiguration = _konfiguration()
    matrix = build_matrix(dokumente, segmente, trainings_ids, konfiguration, FakeEmbedder())

    segmente_je_dokument: dict[str, list[Segment]] = defaultdict(list)
    for zeile in segmente.sort("index").iter_rows(named=True):
        segmente_je_dokument[zeile["document_id"]].append(
            Segment(
                segment_id=zeile["segment_id"],
                document_id=zeile["document_id"],
                index=zeile["index"],
                kind=SegmentKind(zeile["kind"]),
                locator=zeile["locator"],
                heading=zeile["heading"],
                text=zeile["text"],
            )
        )

    segment_count_index = STRUCTURAL_NAMES.index("segment_count")
    assert len(matrix.document_ids) == len(set(matrix.document_ids)), (
        "Doppelte document_ids machen die Zeilenzuordnung unten wertlos"
    )
    for i, doc_id in enumerate(matrix.document_ids):
        eigene_segmente = segmente_je_dokument[doc_id]

        struktur_zeile = matrix.X[i, matrix.block_slices["structural"]]
        assert struktur_zeile[segment_count_index] == len(eigene_segmente), (
            f"Dokument {doc_id}: segment_count in Zeile {i} passt nicht zur echten, "
            "unabhaengig gezaehlten Segmentanzahl – Strukturzeilen rotiert oder die "
            "Blockgrenze zum n-Gramm-Block verschoben"
        )

        text = document_text(eigene_segmente)
        stuecke = chunks(text, konfiguration.chunk_chars)
        gewichte = position_weights(len(stuecke), konfiguration.position_decay)
        erwartetes_mittel = sum(
            g * len(stueck) for g, stueck in zip(gewichte, stuecke, strict=True)
        )
        erwarteter_kopf = len(head_text(text, konfiguration.head_chars))

        emb_mean_zeile = matrix.X[i, matrix.block_slices["emb_mean"]]
        emb_head_zeile = matrix.X[i, matrix.block_slices["emb_head"]]
        assert emb_mean_zeile[0] == pytest.approx(erwartetes_mittel, rel=1e-4), (
            f"Dokument {doc_id}: emb_mean[0] in Zeile {i} passt nicht zur unabhaengig "
            "berechneten gewichteten Chunklaenge – Embeddingzeilen rotiert oder "
            "emb_mean/emb_head vertauscht"
        )
        assert emb_head_zeile[0] == pytest.approx(erwarteter_kopf, rel=1e-4), (
            f"Dokument {doc_id}: emb_head[0] in Zeile {i} passt nicht zu "
            "len(head_text(text)) – Embeddingzeilen rotiert oder emb_mean/emb_head "
            "vertauscht"
        )


def test_keine_nan_und_kein_unendlich() -> None:
    matrix = _kleine_matrix()
    assert np.all(np.isfinite(matrix.X)), "NaN oder inf vergiften jedes Training lautlos"


def test_rundlauf_ueber_parquet_erhaelt_die_werte(tmp_path: Path) -> None:
    """Bindet die Werte, nicht nur die Form: Eine Umsetzung, die beim Lesen Nullen
    liefert, bestuende jede Formpruefung.

    Befund D: ``feature_version`` und ``embedder`` waren hier bis eben ungebunden – genau
    die beiden Felder, fuer die ``FeatureMatrix`` um ein fuenftes Feld erweitert wurde. Ein
    ``read_features``, das ``embedder=""`` zurueckgibt, bestand diesen Test vorher.
    """
    matrix = _kleine_matrix()
    write_features(matrix, tmp_path)
    zurueck = read_features(tmp_path, matrix.feature_version)
    assert zurueck.document_ids == matrix.document_ids
    assert np.allclose(zurueck.X, matrix.X)
    assert zurueck.block_slices == matrix.block_slices
    assert zurueck.feature_version == matrix.feature_version
    assert zurueck.embedder == matrix.embedder
    assert zurueck.embedder == "fake", (
        "embedder muss aus meta.json gelesen werden, nicht leer bleiben"
    )


def test_lesen_mit_falscher_version_wirft(tmp_path: Path) -> None:
    """Konzept § 6.4: Ein Modell darf nur Merkmale derselben Version sehen. Ohne diese
    Pruefung rechnete es nach einer Merkmalsaenderung stillschweigend weiter."""
    matrix = _kleine_matrix()
    write_features(matrix, tmp_path)
    with pytest.raises(FileNotFoundError, match="andere-version"):
        read_features(tmp_path, "andere-version")


def _meta_pfad(tmp_path: Path, matrix: FeatureMatrix) -> Path:
    return tmp_path / matrix.feature_version / "meta.json"


def test_lesen_mit_abweichender_feature_version_in_meta_wirft(tmp_path: Path) -> None:
    """Befund E: Massgeblich fuer den Pfad ist bisher allein der Verzeichnisname –
    ``meta.json`` selbst wurde gegen gar nichts geprueft. Eine von Hand oder durch einen
    halben Schreibvorgang veraenderte ``feature_version`` in ``meta.json`` lieferte
    stillschweigend eine Matrix zurueck, die vorgibt, eine andere Version zu sein, als sie
    tatsaechlich ist."""
    matrix = _kleine_matrix()
    write_features(matrix, tmp_path)
    pfad = _meta_pfad(tmp_path, matrix)
    meta = json.loads(pfad.read_text(encoding="utf-8"))
    meta["feature_version"] = "von-hand-veraendert"
    pfad.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_version"):
        read_features(tmp_path, matrix.feature_version)


def test_lesen_mit_falscher_breite_in_meta_wirft(tmp_path: Path) -> None:
    """Befund E: Passt ``meta.json``s ``width`` nicht zur tatsaechlichen Parquet-Breite,
    sind auch die Blockgrenzen falsch – jeder Zugriff auf ``block_slices`` schnitte dann
    die falschen Spalten, ohne dass irgendetwas auffiele."""
    matrix = _kleine_matrix()
    write_features(matrix, tmp_path)
    pfad = _meta_pfad(tmp_path, matrix)
    meta = json.loads(pfad.read_text(encoding="utf-8"))
    meta["width"] = meta["width"] + 1
    pfad.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="width"):
        read_features(tmp_path, matrix.feature_version)


def test_lesen_mit_inkonsistenten_blockgrenzen_wirft(tmp_path: Path) -> None:
    """Befund E, Gegenprobe: ``width`` allein reicht nicht – die Blockgrenzen selbst
    koennen unabhaengig davon verstellt sein und trotzdem zufaellig zur width passen, wenn
    nur eine einzelne Grenze verschoben und eine andere kompensierend mitverschoben wurde.
    Hier wird nur die letzte Grenze verschoben, sodass sie nicht mehr bis zum Ende der
    Matrix reicht."""
    matrix = _kleine_matrix()
    write_features(matrix, tmp_path)
    pfad = _meta_pfad(tmp_path, matrix)
    meta = json.loads(pfad.read_text(encoding="utf-8"))
    letzter_block = BLOCK_ORDER[-1]
    meta["block_slices"][letzter_block][1] -= 1
    pfad.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="Blockgrenzen"):
        read_features(tmp_path, matrix.feature_version)


def test_leeres_training_wird_gemeldet() -> None:
    """Der Aufrufer muss ``training_document_ids`` selbst richtig befuellen – ohne
    Trainingsdokumente kann der n-Gramm-Block nicht angepasst werden. Eine stille
    Rueckgabe von Nullen waere schlimmer als ein klarer Fehler."""
    dokumente, segmente = _dokumente_und_segmente(3)
    with pytest.raises(ValueError, match="training_document_ids"):
        build_matrix(dokumente, segmente, set(), _konfiguration(), FakeEmbedder())


def test_build_matrix_lehnt_gold_in_training_document_ids_ab() -> None:
    """Befund F: ``build_matrix`` passt den n-Gramm-Block an – das ist Training – und
    muss deshalb selbst pruefen, dass ``training_document_ids`` goldfrei ist, statt sich
    auf einen Kommentar beim Aufrufer (``scripts/build_features.py``) zu verlassen
    (Konzept § 9.2, global-constraints.md: „im Code, nicht per Konvention")."""
    dokumente, segmente, trainings_ids = _trainingsdokumente(3)
    gold_id = gold_documents()["document_id"].to_list()[0]
    alle_dokumente = read_table(PARQUET_DIR, "documents")
    alle_segmente = read_table(PARQUET_DIR, "segments")
    dokumente_mit_gold = pl.concat(
        [dokumente, alle_dokumente.filter(pl.col("document_id") == gold_id)]
    )
    segmente_mit_gold = pl.concat(
        [segmente, alle_segmente.filter(pl.col("document_id") == gold_id)]
    )
    trainings_ids_mit_gold = trainings_ids | {gold_id}

    with pytest.raises(ValueError, match="Gold"):
        build_matrix(
            dokumente_mit_gold,
            segmente_mit_gold,
            trainings_ids_mit_gold,
            _konfiguration(),
            FakeEmbedder(),
        )


def test_vokabular_ohne_fit_wirft() -> None:
    """Gegenprobe zu ``vokabular()``: Vor ``fit`` gibt es kein Vokabular – dieselbe
    Absicherung wie bei ``transform``."""
    block: NgramBlock = build_ngram_block(_konfiguration())
    with pytest.raises(RuntimeError, match="fit"):
        block.vokabular()
