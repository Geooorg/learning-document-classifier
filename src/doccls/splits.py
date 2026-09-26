"""Trainings-, Kalibrier- und Gold-Menge – und der Schutz des Gold-Sets.

Konzept § 9.2: „Nie im Training. Nie in der Kalibriermenge. Nie von der Prüfliste
wählbar. Die Auswahlfunktion filtert Gold-Set-IDs hart heraus – **im Code, nicht per
Konvention.**"

Deshalb steht der Filter hier in einer Funktion, die wirft, und nicht in einem Kommentar.
Ein Kommentar wird in der dritten Umbauwelle überlesen; eine Ausnahme nicht.
"""

import polars as pl

from doccls.config import GENERATED_DIR, PARQUET_DIR
from doccls.pipeline import read_table

GOLD = "gold"
TRAIN = "train"
CALIB = "calib"


def labelled_documents() -> pl.DataFrame:
    """Jedes eingelesene Dokument mit seiner bekannten Klasse und seinem Split.

    ``how="inner"`` wäre hier gefährlich: Ein Dokument ohne Manifestzeile fiele
    stillschweigend heraus, und jede Metrik danach rechnete auf weniger Dokumenten, ohne
    dass irgendetwas auffiele. Deshalb ``how="left"`` und eine Prüfung.
    """
    dokumente = read_table(PARQUET_DIR, "documents").select(["document_id", "source_path"])
    manifest = pl.read_parquet(GENERATED_DIR / "manifest.parquet").select(
        ["source_path", "class_key", "template_id", "split"]
    )
    verbunden = dokumente.join(manifest, on="source_path", how="left")

    ohne_klasse = verbunden.filter(pl.col("class_key").is_null())
    if ohne_klasse.height:
        raise ValueError(
            f"{ohne_klasse.height} eingelesene Dokumente haben keine Zeile im Manifest: "
            f"{ohne_klasse['source_path'].to_list()[:5]}. Ohne bekannte Wahrheit sind sie "
            "weder trainierbar noch messbar."
        )
    return verbunden


def assert_no_gold(frame: pl.DataFrame, wofuer: str) -> None:
    """Wirft, sobald ein Gold-Dokument dort auftaucht, wo es nicht hingehört."""
    gold = frame.filter(pl.col("split") == GOLD)
    if gold.height:
        raise ValueError(
            f"{gold.height} Gold-Dokumente in der Menge fuer {wofuer}: "
            f"{gold['document_id'].to_list()[:5]}. Das Gold-Set ist eingefroren "
            "(Konzept § 9.2) – jede Zahl, die so entsteht, ist geschoent."
        )


def training_documents() -> pl.DataFrame:
    rahmen = labelled_documents().filter(pl.col("split") == TRAIN)
    assert_no_gold(rahmen, "Training")
    return rahmen


def calibration_documents() -> pl.DataFrame:
    rahmen = labelled_documents().filter(pl.col("split") == CALIB)
    assert_no_gold(rahmen, "Kalibrierung")
    return rahmen


def gold_documents() -> pl.DataFrame:
    return labelled_documents().filter(pl.col("split") == GOLD)
