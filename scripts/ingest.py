"""Liest alle Dokumente aus data/raw/ ein und schreibt data/parquet/.

Aufruf: uv run python scripts/ingest.py [--raw data/raw] [--out data/parquet]

Der Lauf ist inkrementell: Bekannte Dokumentversionen werden übersprungen.
"""

import argparse
from pathlib import Path

from doccls.config import PARQUET_DIR, RAW_DIR
from doccls.pipeline import ingest, read_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=PARQUET_DIR)
    args = parser.parse_args()

    ergebnis = ingest(args.raw, args.out)
    print(
        f"neu: {ergebnis.documents}  übersprungen: {ergebnis.skipped}  "
        f"Segmente: {ergebnis.segments}  Anhänge: {ergebnis.attachments}"
    )
    if ergebnis.failed:
        print(f"nicht lesbar ({len(ergebnis.failed)}): {', '.join(ergebnis.failed[:10])}")

    dokumente = read_table(args.out, "documents")
    print(
        f"Bestand: {dokumente.height} Dokumente, "
        f"davon {dokumente['needs_ocr'].sum()} vermutlich gescannt"
    )


if __name__ == "__main__":
    main()
