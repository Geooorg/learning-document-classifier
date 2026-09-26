"""Segmente → Merkmalsmatrix. Der n-Gramm-Block wird nur auf Trainingstexten angepasst.

Aufruf: uv run python scripts/build_features.py
"""

import time

import numpy as np

from doccls.config import FEATURES_DIR, PARQUET_DIR
from doccls.features import build_matrix, load_feature_config, write_features
from doccls.features.embedding import make_embedder
from doccls.pipeline import read_table
from doccls.splits import training_documents


def main() -> None:
    start = time.monotonic()

    config = load_feature_config()
    embedder = make_embedder(config)

    documents = read_table(PARQUET_DIR, "documents")
    segments = read_table(PARQUET_DIR, "segments")
    # training_documents() prueft selbst, dass kein Gold-Dokument darunter ist (Konzept
    # § 9.2) – der n-Gramm-Block darf nur diese Texte zum Anpassen sehen.
    training_ids = set(training_documents()["document_id"].to_list())

    matrix = build_matrix(documents, segments, training_ids, config, embedder)
    ziel = write_features(matrix, FEATURES_DIR)

    leere_zeilen = int(np.sum(~np.any(matrix.X, axis=1)))
    dauer = time.monotonic() - start

    print(f"{len(matrix.document_ids)} Dokumente, Breite {matrix.X.shape[1]}")
    for name in matrix.block_slices:
        bereich = matrix.block_slices[name]
        print(f"  {name}: {bereich.stop - bereich.start}")
    print(f"feature_version={matrix.feature_version}  Embedder={embedder.name}")
    print(f"Zeilen ohne jeden Wert: {leere_zeilen}")
    print(f"Dauer: {dauer:.1f}s")
    print(f"geschrieben nach {ziel}")


if __name__ == "__main__":
    main()
