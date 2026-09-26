"""Merkmalsbildung: Embedding, Zeichen-n-Gramme, Strukturmerkmale – und ihre Version.

Konzept § 6.4: Jeder Merkmalsvektor trägt eine ``feature_version``. Ein Modell darf nur
Merkmale derselben Version sehen, mit denen es trainiert wurde. Ohne diese Regel rechnet
ein Modell nach einer Merkmalsänderung stillschweigend weiter und liefert Unsinn mit hoher
Konfidenz statt einer Fehlermeldung.

Die Version wird **aus dem Inhalt der Konfiguration berechnet**, nicht von Hand gepflegt.
Eine Zahl, die jemand hochzählen muss, wird irgendwann vergessen; ein Hash nie.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field

from doccls.config import PROJECT_ROOT
from doccls.features.structural import STRUCTURAL_NAMES, structural_features
from doccls.features.text import document_text
from doccls.models import Document, Segment, SegmentKind
from doccls.normalize import strip_boilerplate

if TYPE_CHECKING:
    # Nur für die Typpruefung: ``embedding.py`` und ``ngrams.py`` importieren ihrerseits
    # ``FeatureConfig`` aus diesem Paket. Ein Import auf Modulebene liefe deshalb in einen
    # Zirkel; ``build_matrix`` holt sich die tatsaechlichen Namen deshalb lokal (siehe
    # dort). Diese Zeile existiert nur fuer mypy, nie zur Laufzeit.
    from doccls.features.embedding import Embedder

DEFAULT_FEATURES_PATH = PROJECT_ROOT / "config" / "features.yaml"


class FeatureConfig(BaseModel):
    """Alle Parameter, die den Merkmalsvektor bestimmen – und nur diese.

    ``extra="forbid"``: Ein Tippfehler in der YAML soll auffallen. Würde ein unbekanntes
    Feld stillschweigend verworfen, trüge die Version eine Einstellung, die gar nicht wirkt.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    embedder: Literal["e5", "bge-m3"] = "e5"
    chunk_chars: int = Field(default=1200, gt=0)
    head_chars: int = Field(default=1000, gt=0)
    position_decay: float = Field(default=4.0, gt=0)
    ngram_min: int = Field(default=3, gt=0)
    ngram_max: int = Field(default=5, gt=0)
    ngram_max_features: int = Field(default=50_000, gt=0)
    svd_components: int = Field(default=256, gt=0)


def load_feature_config(path: Path | None = None) -> FeatureConfig:
    """``config/features.yaml`` lesen und prüfen."""
    quelle = path or DEFAULT_FEATURES_PATH
    rohdaten = yaml.safe_load(quelle.read_text(encoding="utf-8")) or {}
    return FeatureConfig(**rohdaten)


def version_of_values(values: Mapping[str, object]) -> str:
    """Fingerabdruck einer Parameterabbildung.

    ``sort_keys=True`` ist der eigentliche Inhalt dieser Funktion: Die Version darf nicht
    davon abhängen, in welcher **Reihenfolge** die Parameter dastehen, sondern nur davon,
    welche Werte sie haben. Ohne das änderte ein bloßes Umsortieren der Felddeklarationen
    in ``FeatureConfig`` die Version – und machte damit jeden gerechneten Merkmalsvektor
    ungültig, obwohl sich an den Merkmalen nichts geändert hat.

    Getrennt von ``feature_version``, weil die Reihenfolgeunabhängigkeit sonst nicht
    prüfbar wäre: ``model_dump()`` liefert immer die Deklarationsreihenfolge, egal wie das
    Objekt gebaut wurde. Ein Test über ``FeatureConfig`` kann diese Eigenschaft deshalb
    nicht zum Fehlschlagen bringen – über diese Naht schon.
    """
    text = json.dumps(dict(values), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def feature_version(config: FeatureConfig) -> str:
    """Kurzer, stabiler Fingerabdruck der Merkmalsparameter (Konzept § 6.4)."""
    return version_of_values(config.model_dump())


BLOCK_ORDER: tuple[str, ...] = ("emb_mean", "emb_head", "ngram", "structural")
"""Reihenfolge der Bloecke in der zusammengesetzten Matrix. Bindend fuer
``block_slices`` in ``FeatureMatrix`` – aendert sich diese Reihenfolge, aendert sich auch
das Layout der Matrix, und ``feature_version`` muss dann eine andere sein (sie tut das
automatisch, weil ``BLOCK_ORDER`` selbst kein Konfigurationsfeld ist, aber jede Aenderung
hier gehoert zusammen mit einer neuen ``FeatureConfig``-Vergangenheit committet)."""


@dataclass
class FeatureMatrix:
    """Eine fertige Merkmalsmatrix. Zeile ``i`` gehoert zu ``document_ids[i]``.

    ``block_slices`` sagt, welche Spalten zu welchem Block gehoeren – unverzichtbar, um
    ein trainiertes Gewicht auf einen Block zurueckzufuehren (Konzept § 7.1: ``coef_``
    gegen die Merkmalsnamen halten). ``embedder`` ist der Name des verwendeten Embedders
    (``"e5"``, ``"bge-m3"`` – oder ``"fake"`` in Tests); er entscheidet mit, welches
    Praefix beim Bau vorgelegt wurde, ist aber selbst nicht Teil der Version, weil er
    bereits in ``config.embedder`` steckt und damit die ``feature_version`` mitbestimmt.
    """

    document_ids: list[str]
    X: npt.NDArray[np.float32]
    feature_version: str
    block_slices: dict[str, slice]
    embedder: str = ""


def _segmente_gruppieren(segments: pl.DataFrame) -> dict[str, list[Segment]]:
    """Segmente je ``document_id``, sortiert nach ``index`` (Konzept § 6.1: die
    Reihenfolge traegt Bedeutung, ``document_text`` baut darauf auf)."""
    ergebnis: dict[str, list[Segment]] = defaultdict(list)
    for zeile in segments.sort("index").iter_rows(named=True):
        ergebnis[zeile["document_id"]].append(
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
    return ergebnis


def _dokument_aus_zeile(zeile: dict[str, Any]) -> Document:
    """Ein ``Document`` aus einer Tabellenzeile rekonstruieren, ohne die Originaldatei
    erneut zu lesen (wie in ``tests/test_features_structural.py``)."""
    return Document(
        document_id=zeile["document_id"],
        source_path=zeile["source_path"],
        file_name=zeile["file_name"],
        media_type=zeile["media_type"],
        content_sha256=zeile["content_sha256"],
        size_bytes=zeile["size_bytes"],
        parent_document_id=zeile["parent_document_id"],
        ingested_at=zeile["ingested_at"],
        needs_ocr=zeile["needs_ocr"],
        ocr_applied=zeile["ocr_applied"],
    )


def build_matrix(
    documents: pl.DataFrame,
    segments: pl.DataFrame,
    training_document_ids: set[str],
    config: FeatureConfig,
    embedder: Embedder,
) -> FeatureMatrix:
    """Die drei Merkmalsbloecke zu einer Matrix zusammenfassen.

    Reihenfolge je Dokument (die Reihenfolge ist der kritische Teil dieser Funktion, nicht
    die Formel dahinter): Segmente nach ``index`` sortieren → ``document_text`` →
    ``strip_boilerplate`` über die so entstandenen Dokumenttexte → ``document_vector``
    (Mittelwert ‖ Kopf) → ``ngram_block.transform`` → ``structural_features`` →
    waagerecht aneinanderhaengen.

    Der n-Gramm-Block wird **ausschliesslich auf den Texten der Dokumente in
    ``training_document_ids`` angepasst** und danach auf alle Dokumente angewandt. Wer ihn
    auf allen Dokumenten (also auch dem Gold-Set) anpasst, gibt dem Modell dessen
    Wortstatistik mit – die gemessenen Zahlen sehen dann besser aus, als sie sind. Das ist
    der klassische stille Fehler dieser Art von Pipeline, und hier – wo Training und Gold
    getrennt vorliegen – ist er zum ersten Mal prüfbar.

    Das E5/BGE-M3-Praefix wird nicht hier entschieden: ``praefix_fuer(embedder)`` waehlt
    es (Aufgabe 5), diese Funktion benutzt nur, was dabei herauskommt.
    """
    from doccls.features.embedding import document_vector, praefix_fuer
    from doccls.features.ngrams import build_ngram_block

    document_ids: list[str] = documents["document_id"].to_list()
    segmente_je_dokument = _segmente_gruppieren(segments)
    zeilen_je_dokument: dict[str, dict[str, Any]] = {
        zeile["document_id"]: zeile for zeile in documents.iter_rows(named=True)
    }

    texte = [document_text(segmente_je_dokument.get(doc_id, [])) for doc_id in document_ids]
    texte = strip_boilerplate(texte)

    praefix = praefix_fuer(embedder)
    emb_bloecke = np.vstack(
        [document_vector(embedder, text, config, praefix) for text in texte]
    ).astype(np.float32)
    emb_dim = embedder.dimension

    trainingstexte = [
        text
        for doc_id, text in zip(document_ids, texte, strict=True)
        if doc_id in training_document_ids
    ]
    if not trainingstexte:
        raise ValueError(
            "training_document_ids trifft keines der uebergebenen Dokumente – der "
            "n-Gramm-Block kann nicht angepasst werden. Das Gold-Set darf hier nicht "
            "landen (siehe Moduldoc)."
        )
    ngram_block = build_ngram_block(config)
    ngram_block.fit(trainingstexte)
    ngram_werte = ngram_block.transform(texte)

    struktur_werte = np.vstack(
        [
            structural_features(
                _dokument_aus_zeile(zeilen_je_dokument[doc_id]),
                segmente_je_dokument.get(doc_id, []),
            )
            for doc_id in document_ids
        ]
    ).astype(np.float32)

    bloecke: dict[str, npt.NDArray[np.float32]] = {
        "emb_mean": emb_bloecke[:, :emb_dim],
        "emb_head": emb_bloecke[:, emb_dim:],
        "ngram": ngram_werte,
        "structural": struktur_werte,
    }
    X = np.hstack([bloecke[name] for name in BLOCK_ORDER]).astype(np.float32)

    grenzen = np.cumsum([0] + [bloecke[name].shape[1] for name in BLOCK_ORDER])
    block_slices = {
        name: slice(int(grenzen[i]), int(grenzen[i + 1])) for i, name in enumerate(BLOCK_ORDER)
    }

    return FeatureMatrix(
        document_ids=document_ids,
        X=X,
        feature_version=feature_version(config),
        block_slices=block_slices,
        embedder=embedder.name,
    )


def _feature_names(block_slices: Mapping[str, slice]) -> list[str]:
    """Ein Name je Spalte – echte Namen fuer den Strukturblock (Konzept § 7.1), sonst ein
    durchnummerierter Platzhalter, der wenigstens Block und Position verraet."""
    breite = max(bereich.stop for bereich in block_slices.values())
    namen: list[str] = [""] * breite
    for block in BLOCK_ORDER:
        bereich = block_slices[block]
        for i in range(bereich.stop - bereich.start):
            namen[bereich.start + i] = (
                STRUCTURAL_NAMES[i] if block == "structural" else f"{block}_{i}"
            )
    return namen


def write_features(matrix: FeatureMatrix, features_dir: Path) -> Path:
    """Matrix und Metadaten unter ``<features_dir>/<feature_version>/`` ablegen.

    Die ``feature_version`` steht im Verzeichnisnamen: Ein Versionswechsel ist damit ein
    anderer Pfad statt eines stillen Ueberschreibens (Konzept § 6.4).
    """
    ziel = features_dir / matrix.feature_version
    ziel.mkdir(parents=True, exist_ok=True)

    breite = matrix.X.shape[1]
    tabelle = pl.DataFrame(
        {"document_id": matrix.document_ids, "vector": matrix.X.tolist()},
        schema={"document_id": pl.String(), "vector": pl.Array(pl.Float32, breite)},
    )
    tabelle.write_parquet(ziel / "matrix.parquet")

    meta = {
        "feature_version": matrix.feature_version,
        "width": breite,
        "embedder": matrix.embedder,
        "block_slices": {name: [b.start, b.stop] for name, b in matrix.block_slices.items()},
        "feature_names": _feature_names(matrix.block_slices),
    }
    (ziel / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ziel


def read_features(features_dir: Path, feature_version: str) -> FeatureMatrix:
    """Das Gegenstueck zu ``write_features``.

    Wirft, wenn die verlangte Version nicht unter ``features_dir`` liegt: Ein Modell darf
    nie mit Merkmalen einer anderen Version weiterrechnen (Konzept § 6.4). Ohne diese
    Pruefung wuerde ein Versionswechsel stillschweigend die letzte bekannte Matrix liefern.
    """
    quelle = features_dir / feature_version
    meta_pfad = quelle / "meta.json"
    matrix_pfad = quelle / "matrix.parquet"
    if not meta_pfad.exists() or not matrix_pfad.exists():
        raise FileNotFoundError(
            f"Keine Merkmalsmatrix fuer feature_version={feature_version!r} unter "
            f"{quelle} – erst 'uv run python scripts/build_features.py' laufen lassen."
        )

    meta = json.loads(meta_pfad.read_text(encoding="utf-8"))
    tabelle = pl.read_parquet(matrix_pfad)
    block_slices = {
        name: slice(grenze[0], grenze[1]) for name, grenze in meta["block_slices"].items()
    }

    return FeatureMatrix(
        document_ids=tabelle["document_id"].to_list(),
        X=tabelle["vector"].to_numpy().astype(np.float32),
        feature_version=meta["feature_version"],
        block_slices=block_slices,
        embedder=meta["embedder"],
    )
