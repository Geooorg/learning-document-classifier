"""Aufteilung in Gold-, Trainings- und Kalibriermenge – und das Manifest dazu.

Der Schnitt läuft über ``template_id``, nie über einzelne Dokumente: Zehn Varianten
derselben Vorlage sind praktisch dasselbe Dokument. Lägen sie auf beiden Seiten, wäre jede
gemessene Metrik geschönt (Konzept § 9.2, Aufteilung nach Dokumentfamilie).

Der Schnitt wird nicht bei jedem Lauf neu gewürfelt: ``random.Random(...).shuffle()`` auf
einer Liste, deren Länge sich ändert (eine neue Vorlage kommt hinzu), permutiert alle
Positionen neu – eine Vorlage, die gestern im Gold-Set war, wäre heute Trainingsmaterial,
ohne dass irgendetwas fehlschlägt. Deshalb hält ``config/splits.yaml`` den einmal gewürfelten
Schnitt fest (``load_frozen_splits``/``save_splits``); ``assign_splits`` ordnet nur noch
Vorlagen zu, die dort noch nicht stehen.
"""

import random
from collections import defaultdict
from enum import StrEnum
from pathlib import Path

import polars as pl
import yaml

from doccls.config import PROJECT_ROOT
from doccls.generation.content import DocumentSpec

GOLD_TEMPLATES_PER_CLASS = 5
CALIB_TEMPLATES_PER_CLASS = 1
"""Der Rest geht in die Trainingsmenge."""

DEFAULT_SPLITS_PATH = PROJECT_ROOT / "config" / "splits.yaml"

SPLITS_HEADER = (
    "# Der Gold-Set-Schnitt, einmal gewürfelt und seither fest. Läuft über Vorlagen, nicht "
    "über\n"
    "# Dokumente: Zehn Varianten einer Vorlage sind praktisch dasselbe Dokument und dürfen "
    "nicht\n"
    "# auf beide Seiten fallen (Konzept § 9.2).\n"
    "#\n"
    "# Diese Datei wird nicht von Hand gepflegt und nicht neu gewürfelt. Eine neue Vorlage "
    "wird\n"
    "# von scripts/generate_documents.py angehängt; bestehende Einträge ändern sich nie. "
    "Wer sie\n"
    "# löscht, macht alle vorher gemessenen Zahlen unvergleichbar.\n"
)


class Split(StrEnum):
    GOLD = "gold"
    """Eingefroren. Nie Training, nie Kalibrierung, nie von der Prüfliste wählbar."""

    TRAIN = "train"
    CALIB = "calib"
    """Für Temperature Scaling und die Schwellen τ und δ (Konzept § 7.3, § 7.4)."""


MANIFEST_SCHEMA: dict[str, pl.DataType] = {
    "source_path": pl.String(),
    "class_key": pl.String(),
    "template_id": pl.String(),
    "variant": pl.Int64(),
    "split": pl.String(),
    "format": pl.String(),
}


def load_frozen_splits(path: Path | None = None) -> dict[str, Split]:
    """Den festgeschriebenen Schnitt lesen. Fehlt die Datei, ist er leer (Erstlauf)."""
    quelle = path or DEFAULT_SPLITS_PATH
    if not quelle.exists():
        return {}
    rohdaten = yaml.safe_load(quelle.read_text(encoding="utf-8")) or {}
    return {str(vorlage): Split(wert) for vorlage, wert in (rohdaten.get("splits") or {}).items()}


def save_splits(splits: dict[str, Split], path: Path | None = None) -> None:
    """Den Schnitt schreiben. Sortiert, damit das Diff lesbar bleibt.

    Schreibt zuerst in eine temporäre Datei im selben Verzeichnis und ersetzt die Zieldatei
    erst danach atomar (``Path.replace``). Die Datei ist die einzige Quelle der Wahrheit für
    den Schnitt; ein abgebrochener Schreibvorgang darf den bestehenden Stand nicht zerstören.
    """
    ziel = path or DEFAULT_SPLITS_PATH
    daten = {"version": 1, "splits": {k: str(v) for k, v in sorted(splits.items())}}
    ziel.parent.mkdir(parents=True, exist_ok=True)
    temp = ziel.with_name(ziel.name + ".tmp")
    temp.write_text(
        SPLITS_HEADER + yaml.safe_dump(daten, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temp.replace(ziel)


def assign_splits(
    corpus: list[DocumentSpec], seed: int = 7, frozen: dict[str, Split] | None = None
) -> dict[str, Split]:
    """Schnitt je Vorlage. Einträge aus ``frozen`` bleiben unverändert; nur unbekannte Vorlagen
    werden zugeordnet, und zwar so, dass je Klasse die Sollzahlen aufgefüllt werden
    (5 gold, 1 calib, Rest train)."""
    frozen = frozen or {}
    je_klasse: dict[str, list[str]] = defaultdict(list)
    for spec in corpus:
        if spec.template_id not in je_klasse[spec.class_key]:
            je_klasse[spec.class_key].append(spec.template_id)

    splits: dict[str, Split] = dict(frozen)
    for klasse, vorlagen in sorted(je_klasse.items()):
        if len(vorlagen) < GOLD_TEMPLATES_PER_CLASS + CALIB_TEMPLATES_PER_CLASS + 1:
            raise ValueError(
                f"Klasse {klasse} hat nur {len(vorlagen)} Vorlagen; für den Schnitt werden "
                f"mindestens {GOLD_TEMPLATES_PER_CLASS + CALIB_TEMPLATES_PER_CLASS + 1} gebraucht."
            )
        unbekannt = sorted(v for v in vorlagen if v not in frozen)
        random.Random(f"{seed}:{klasse}").shuffle(unbekannt)

        bekannt_gold = sum(1 for v in vorlagen if frozen.get(v) is Split.GOLD)
        bekannt_calib = sum(1 for v in vorlagen if frozen.get(v) is Split.CALIB)
        gold_frei = max(GOLD_TEMPLATES_PER_CLASS - bekannt_gold, 0)
        calib_frei = max(CALIB_TEMPLATES_PER_CLASS - bekannt_calib, 0)

        for vorlage in unbekannt:
            if gold_frei:
                splits[vorlage] = Split.GOLD
                gold_frei -= 1
            elif calib_frei:
                splits[vorlage] = Split.CALIB
                calib_frei -= 1
            else:
                splits[vorlage] = Split.TRAIN

        if gold_frei or calib_frei:
            raise ValueError(
                f"Klasse {klasse}: {gold_frei} Gold- und {calib_frei} Kalibrier-Plätze sind "
                "unbesetzt und es gibt keine unbekannten Vorlagen, die sie füllen könnten. "
                "Vermutlich wurde eine eingefrorene Vorlage aus TEMPLATES entfernt. Entweder "
                "eine neue Vorlage für diese Klasse ergänzen, oder den Eintrag bewusst aus "
                "config/splits.yaml löschen und in Kauf nehmen, dass Messungen davor und "
                "danach unvergleichbar werden."
            )
    return splits


def compare_with_frozen(
    corpus: list[DocumentSpec], frozen: dict[str, Split]
) -> tuple[list[str], list[str]]:
    """Vergleicht die Vorlagen im Korpus mit dem eingefrorenen Schnitt.

    Liefert ``(neu, verwaist)``: ``neu`` sind Vorlagen im Korpus ohne Eintrag in ``frozen``,
    ``verwaist`` sind Einträge in ``frozen`` ohne zugehörige Vorlage mehr im Korpus – ein
    Widerspruch, der weder automatisch aufgelöst noch verschwiegen werden darf.
    """
    vorlagen = {spec.template_id for spec in corpus}
    neu = sorted(vorlagen - set(frozen))
    verwaist = sorted(set(frozen) - vorlagen)
    return neu, verwaist


def report_template_drift(neu: list[str], verwaist: list[str]) -> None:
    """Meldet neue und verwaiste Vorlagen deutlich – meldet, ändert aber nichts.

    Ein Mensch entscheidet, ob eine neue Vorlage zugeordnet oder ein verwaister Eintrag aus
    ``config/splits.yaml`` gelöscht wird.
    """
    if neu:
        print(
            f"{len(neu)} neue Vorlage(n) zugeordnet und in config/splits.yaml "
            f"festgeschrieben: {', '.join(neu)}"
        )
        print("ACHTUNG: config/splits.yaml hat sich geändert und muss committet werden.")
    if verwaist:
        print(
            f"ACHTUNG: {len(verwaist)} Eintrag/Einträge in config/splits.yaml haben keine "
            f"zugehörige Vorlage mehr in TEMPLATES (verwaist, nicht automatisch entfernt): "
            f"{', '.join(verwaist)}"
        )


def manifest_frame(
    corpus: list[DocumentSpec], splits: dict[str, Split], paths: list[Path], root: Path
) -> pl.DataFrame:
    """Die bekannte Wahrheit als Tabelle: Pfad → Klasse, Vorlage, Split."""
    return pl.DataFrame(
        [
            {
                "source_path": str(pfad.relative_to(root)),
                "class_key": spec.class_key,
                "template_id": spec.template_id,
                "variant": spec.variant,
                "split": str(splits[spec.template_id]),
                "format": pfad.suffix.lstrip("."),
            }
            for spec, pfad in zip(corpus, paths, strict=True)
        ],
        schema=MANIFEST_SCHEMA,
    )
