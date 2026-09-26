"""Erzeugt die synthetischen Testdokumente unter data/raw/ und das Manifest dazu.

Aufruf: uv run python scripts/generate_documents.py [--out data/raw] [--seed 42]

Die Ordnerstruktur ist bewusst flach nach Format, **nicht** nach Klasse: Läge die Klasse im
Ordnernamen, wäre die Versuchung groß, sie beim Einlesen daraus abzuleiten – und damit wäre
der Klassifikator überflüssig (vgl. bauprojekt-ai-pipeline, DOC_TYPE_BY_FOLDER).
"""

import argparse
import shutil
from pathlib import Path

from doccls.config import GENERATED_DIR, RAW_DIR
from doccls.generation.content import build_corpus
from doccls.generation.manifest import (
    anhang_namen,
    assign_document_names,
    assign_splits,
    compare_with_frozen,
    hat_mailanhang,
    load_frozen_splits,
    manifest_frame,
    report_template_drift,
    save_splits,
)
from doccls.generation.writers import write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    korpus = build_corpus(seed=args.seed)
    fest = load_frozen_splits()
    splits = assign_splits(korpus, frozen=fest)

    # Der Schnitt ist eingefroren (config/splits.yaml). Nur wirklich neue Vorlagen werden
    # zugeordnet und angehängt – bestehende Einträge ändern sich nie (Konzept § 9.2). Der
    # umgekehrte Fall (Eintrag ohne Vorlage, "verwaist") wird gemeldet, nicht stillschweigend
    # mitgeschleppt oder automatisch gelöscht – das entscheidet ein Mensch. Erst schreiben,
    # dann melden: sonst hätte der Nutzer bei einem fehlgeschlagenen save_splits das
    # Gegenteil von dem gelesen, was tatsächlich passiert ist.
    neu, verwaist = compare_with_frozen(korpus, fest)
    if neu:
        save_splits(splits)
    report_template_drift(neu, verwaist)

    # Vorlagen, deren Name auf "-anhang" endet, werden Mails mit angehängter PDF. Der
    # Anhang wird beim Einlesen ein eigenständiges Dokument mit Elternbezug (Konzept § 5).
    # Vereinfachung: Mailkörper und Anhang tragen denselben Inhalt – geprüft wird der Weg,
    # nicht die Redaktion.
    #
    # Die Dateinamen selbst sind neutral (siehe assign_document_names) – weder der
    # Dateiname noch, für Mailanhänge, der Anhangname dürfen den Klassenschlüssel tragen.
    # Klasse und Vorlage stehen ausschließlich im Manifest. Der Anhangname wird deshalb
    # nicht hier zusammengesetzt, sondern von anhang_namen (manifest.py) erfragt – derselben
    # Funktion, die manifest_frame gleich danach für dieselbe Zeile befragt. Zwei Stellen,
    # die den Namen unabhängig bilden, liefen sonst irgendwann auseinander (siehe deren
    # Docstring).
    namen = assign_document_names(korpus)
    pfade: list[Path] = []
    zwischenablage = args.out.parent / "tmp-anhaenge"
    for spec, name in zip(korpus, namen, strict=True):
        fmt = spec.formats[0]
        basis = args.out / fmt / name
        if hat_mailanhang(spec):
            # ``write`` hängt nur die Endung an (``with_suffix``) – der Zielpfad steht also
            # schon vor dem Schreiben fest. Der Assert danach sichert genau diese Annahme:
            # anhang_namen bekommt wirklich den Pfad, den write gleich zurückgibt, nicht
            # einen geratenen.
            ziel = basis.with_suffix(".eml")
            anhang_name = anhang_namen(spec, ziel)[0]
            anhang = write(zwischenablage / name, spec, "pdf").read_bytes()
            geschrieben = write(basis, spec, "eml", attachment=(anhang_name, anhang))
            assert geschrieben == ziel, (
                f"write() hat {geschrieben} geschrieben, erwartet wurde {ziel} – "
                "anhang_namen bekam den falschen Pfad."
            )
            pfade.append(geschrieben)
        else:
            pfade.append(write(basis, spec, fmt))
    shutil.rmtree(zwischenablage, ignore_errors=True)

    manifest = manifest_frame(korpus, splits, pfade, root=args.out)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    ziel = GENERATED_DIR / "manifest.parquet"
    manifest.write_parquet(ziel)

    anhaenge = sum(1 for s in korpus if hat_mailanhang(s))
    print(f"{len(pfade)} Dateien geschrieben nach {args.out}, davon {anhaenge} Mails mit Anhang")
    print(manifest.group_by("class_key", "split").len().sort("class_key", "split"))
    print(f"Manifest: {ziel}")


if __name__ == "__main__":
    main()
