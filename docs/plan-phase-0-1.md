# Phase 0 + 1: Testdaten und Ingestion — Implementierungsplan

> **Abgearbeitet. Dieser Plan ist ab hier ein historisches Dokument, keine Referenz.**
>
> Der Code weicht an mehreren Stellen bewusst ab, und zwar überall dort, wo die Umsetzung
> gezeigt hat, dass der Plan falsch lag. Wer wissen will, wie das System heute arbeitet,
> liest den Code und `docs/konzept.md`, nicht diesen Plan. Die offenen Punkte stehen in
> `docs/phase-1-offene-punkte.md`.
>
> Die wichtigsten Abweichungen, jede aus einem belegten Befund:
>
> | Plan | Code | Grund |
> |---|---|---|
> | `MIN_CHARS_PER_PAGE = 120` | `40` | 120 lag mitten in der Verteilung der echten Dokumente (113–157 Zeichen je Seite) und markierte 30 von 230 PDF mit Textebene fälschlich als OCR-bedürftig |
> | Gold-Schnitt bei jedem Lauf gewürfelt | `config/splits.yaml`, einmal gewürfelt und festgeschrieben | eine neue Vorlage verschob drei von acht bestehenden über die Schnittgrenze |
> | Dateinamen `RECHNUNG-standard-03.pdf` | `doc-0001.pdf` | aus dem Dateinamen allein waren 480 von 560 Klassen zu raten (Grundrate 14 %) |
> | feste Blockzahl je Seite | gestreute Seitengruppen (`_page_groups`) | aus (Format, Segmentzahl) allein waren 61 % zu raten |
> | `write_pdf` ohne Überlaufprüfung | Prüfung über das Ergebnis (`MIN_RENDERED_SHARE`, NFKC) | 2000 Zeichen in einer Tabellenzelle gingen still verloren, der Platzzähler blieb positiv |
> | `needs_ocr` über die Segmentzahl | über die tatsächliche Seitenzahl | ein Scan mit getipptem Deckblatt galt als vollwertiges Textdokument |
> | XLSX `data_only=True` allein | Rückfall auf die Formel | Formelzellen ohne Cache wurden still zu Leerzellen |
> | mehrere Tests, die nicht fehlschlagen konnten | gestrichen oder durch Schranken ersetzt | siehe unten |
>
> Vier Tests aus diesem Plan konnten strukturell nicht fehlschlagen und wurden gestrichen
> oder ersetzt. Das ist der häufigste Fehlertyp dieses Plans gewesen. Wer daraus einen
> Folgeplan schreibt: jeder Test braucht eine **Mutationsprobe** — die Implementierung
> gezielt falsch machen und nachsehen, ob der Test rot wird.

> **Für Agenten:** Diese Aufgaben werden einzeln abgearbeitet. Schritte sind als Checkboxen
> (`- [ ]`) geführt. Nach jeder Aufgabe wird geprüft und committet, bevor die nächste beginnt.

**Ziel:** Synthetische Dokumente mit bekannter Wahrheit erzeugen (Phase 0) und Dokumente
beliebigen Formats in nachvollziehbare, inkrementell fortschreibbare Parquet-Tabellen
überführen (Phase 1).

**Architektur:** Reine Funktionen für Erkennung, Extraktion und Normalisierung; ein einziges
Modul (`pipeline.py`) fasst sie zusammen und ist die einzige Stelle mit Dateisystemzugriff.
Dokumentidentität über den Inhalts-Hash, damit wiederholte Läufe überspringen statt zu
verdoppeln. E-Mail-Anhänge werden eigenständige Dokumente mit Elternbezug.

**Tech-Stack:** uv, Python 3.14, Pydantic, Polars/PyArrow/Parquet, PyMuPDF, python-docx,
openpyxl, stdlib `email`, pytest, ruff, mypy.

**Grundlage:** [konzept.md](konzept.md) — insbesondere § 4 (Datenmodell), § 5 (Ingestion),
§ 9.2 (Gold-Set) und § 11 Phase 0/1.

## Globale Randbedingungen

- **Paketverwaltung ausschließlich mit `uv`.** Kein `pip`, kein manuelles venv.
- **Python 3.14** (`.python-version`). Fehlt ein Wheel, wird das in `pyproject.toml`
  kommentiert und nicht stillschweigend umgangen.
- **Auf diesem Rechner ist nur Podman installiert.** Nie `docker` oder `docker compose`.
  In Phase 0/1 werden keine Dienste gebraucht.
- **Typannotationen überall.** `uv run ruff check .`, `uv run ruff format --check .` und
  `uv run mypy .` müssen durchlaufen, bevor committet wird.
- **Docstrings und Kommentare auf Deutsch**, Bezeichner auf Englisch — wie in
  `bauprojekt-ai-pipeline`.
- **Datenschemata explizit** festlegen (Polars-Schema, Pydantic-Modell), nie implizit ableiten.
- **`data/` gehört nicht ins Git.** Nur `.gitkeep`.
- **Originale werden nie verändert.** Die Pipeline liest sie genau einmal.
- **Determinismus ist Pflicht:** Zweimal erzeugte Testdaten müssen byteweise gleich sein,
  sonst ändert sich bei jedem Lauf jede `document_id`.

## Dateien im Überblick

| Datei | Verantwortung |
|---|---|
| `pyproject.toml`, `.python-version`, `.gitignore` | Projektgerüst |
| `config/classes.yaml` | Klassenschema — Konfiguration, kein Code |
| `src/doccls/config.py` | Pfade und Schwellwerte aus Umgebungsvariablen |
| `src/doccls/classes.py` | Klassenschema laden und prüfen |
| `src/doccls/models.py` | `Document`, `Segment`, Polars-Schemas, ID-Ableitung |
| `src/doccls/detect.py` | Format aus Magic Bytes |
| `src/doccls/normalize.py` | NFKC, Silbentrennung, Kopf-/Fußzeilen |
| `src/doccls/extraction/pdf.py` · `office.py` · `mail.py` · `__init__.py` | je Format ein reiner Extraktor |
| `src/doccls/pipeline.py` | Orchestrierung, Hashing, Parquet |
| `src/doccls/generation/content.py` · `writers.py` | Testdaten: Inhalte und Ausgabeformate |
| `scripts/generate_documents.py` · `scripts/ingest.py` | dünne CLIs |
| `tests/` | pytest, Fixtures je Format |

## Bewusst nicht in Phase 1

Das Konzept nennt in § 5 mehr Formate, als hier gebaut werden. Die Formaterkennung
(Aufgabe 7) kennt sie bereits und liefert für sie `Format.PPTX`, `HTML`, `CSV`, `MSG` oder
`IMAGE`; einen Extraktor bekommen sie erst später, und bis dahin landen sie in
`IngestResult.failed` statt stillschweigend als leeres Dokument.

| Zurückgestellt | Warum | Wann |
|---|---|---|
| PPTX, HTML, CSV | im Korpus nicht vorhanden; kein Nutzen ohne Testdaten | mit echten Dokumenten |
| MSG (Outlook) | braucht `extract-msg`; EML deckt den Mailfall inhaltlich ab | Phase 4 |
| DOC, RTF, ODT über Apache Tika | einziger Java-Baustein, braucht einen Container | Phase 4 |
| Bilder (PNG, JPG, TIFF) | dieselbe OCR-Kette wie Aufgabe 12 | nach Aufgabe 12 |

Erkennen und Ablehnen ist Absicht: Ein Format stillschweigend zu übergehen erzeugt
Dokumente ohne Text, und die fallen erst in Phase 2 als unerklärliche Fehlklassifikationen
auf.

---

# Phase 0 — Testdaten mit bekannter Wahrheit

## Aufgabe 1: Projektgerüst

**Dateien:**
- Anlegen: `pyproject.toml`, `.python-version`, `.gitignore`
- Anlegen: `src/doccls/__init__.py`, `src/doccls/config.py`
- Anlegen: `tests/test_config.py`
- Anlegen: `data/.gitkeep`

**Schnittstellen:**
- Liefert: `doccls.config.RAW_DIR`, `PARQUET_DIR`, `GENERATED_DIR` (`Path`),
  `MIN_CHARS_PER_PAGE` (`int`)

- [ ] **Schritt 1: Projekt anlegen**

```bash
cd /Users/georg/Projekte/privat/learning-document-classifier
echo "3.14" > .python-version
uv init --lib --name doccls --python 3.14 .
```

`uv init --lib` erzeugt das src-Layout. Falls `uv init` vorhandene Dateien beanstandet:
`pyproject.toml` von Hand anlegen (Inhalt in Schritt 2) und `src/doccls/__init__.py` als
leere Datei.

- [ ] **Schritt 2: `pyproject.toml` schreiben**

```toml
[project]
name = "doccls"
version = "0.1.0"
description = "Lernfähige semantische Dokumentklassifikation"
requires-python = ">=3.14"
dependencies = [
    "openpyxl>=3.1.5",
    "polars>=1.44.2",
    "puremagic>=1.28",
    "pyarrow>=25.0.1",
    "pydantic>=2.13.5",
    "pymupdf>=1.28.2",
    "python-docx>=1.2.0",
    "pyyaml>=6.0.2",
]

[dependency-groups]
dev = [
    "duckdb>=1.5.5",
    "mypy>=2.3.1",
    "pytest>=9.1.1",
    "ruff>=0.16.8",
    "types-openpyxl>=3.1.5.20260827",
    "types-pyyaml>=6.0.12",
]

[build-system]
requires = ["uv_build>=0.12,<0.13"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = "doccls"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
strict = true
```

- [ ] **Schritt 3: `.gitignore` schreiben**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.mypy_cache/
data/*
!data/.gitkeep
```

- [ ] **Schritt 4: Test schreiben, der noch fehlschlägt**

`tests/test_config.py`:

```python
"""Die Konfiguration muss Pfade liefern, die unterhalb des Projektverzeichnisses liegen."""

from doccls import config


def test_pfade_liegen_im_projekt() -> None:
    for pfad in (config.RAW_DIR, config.PARQUET_DIR, config.GENERATED_DIR):
        assert config.PROJECT_ROOT in pfad.parents


def test_ocr_schwelle_ist_positiv() -> None:
    assert config.MIN_CHARS_PER_PAGE > 0
```

- [ ] **Schritt 5: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_config.py -v
```

Erwartet: `ModuleNotFoundError` bzw. `AttributeError` — `config` existiert noch nicht.

- [ ] **Schritt 6: `src/doccls/config.py` schreiben**

```python
"""Pfade und Schwellwerte. Alles über Umgebungsvariablen überschreibbar, nichts hart kodiert."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("DOCCLS_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
"""Originale. Werden nur gelesen, nie verändert."""

PARQUET_DIR = DATA_DIR / "parquet"
GENERATED_DIR = DATA_DIR / "generated"

MIN_CHARS_PER_PAGE = int(os.environ.get("DOCCLS_MIN_CHARS_PER_PAGE", "120"))
"""Liefert eine Seite weniger Zeichen, gilt das Dokument als gescannt und braucht OCR."""
```

- [ ] **Schritt 7: Test laufen lassen, Erfolg bestätigen**

```bash
uv run pytest tests/test_config.py -v
```

Erwartet: 2 passed.

- [ ] **Schritt 8: Prüfen und committen**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest
mkdir -p data && touch data/.gitkeep
git add -A && git commit -m "Projektgerüst: uv, src-Layout, Konfiguration"
```

---

## Aufgabe 2: Klassenschema

**Dateien:**
- Anlegen: `config/classes.yaml`
- Anlegen: `src/doccls/classes.py`
- Anlegen: `tests/test_classes.py`

**Schnittstellen:**
- Nutzt: `doccls.config.PROJECT_ROOT`
- Liefert: `ClassDef` (Felder `key: str`, `name: str`, `description: str`, `not_: str`,
  `residual: bool`), `ClassSchema` (Felder `version: int`, `classes: list[ClassDef]`;
  Methoden `keys() -> list[str]`, `get(key) -> ClassDef`, `trainable() -> list[ClassDef]`),
  `load_classes(path: Path | None = None) -> ClassSchema`

- [ ] **Schritt 1: `config/classes.yaml` schreiben**

Die `description` ist die Definition für Labelnde *und* der Kaltstart-Prototyp (Konzept
§ 2, § 8.1). Sie muss beschreiben, woran man die Klasse erkennt, nicht was sie bedeutet.

```yaml
version: 1
classes:
  - key: RECHNUNG
    name: Rechnung
    description: >
      Zahlungsaufforderung eines Lieferanten. Enthält Rechnungsnummer, Rechnungsdatum,
      Positionen mit Einzel- und Gesamtbetrag, Umsatzsteuerausweis und ein Zahlungsziel.
      Der Gesamtbetrag ist positiv und wird vom Empfänger geschuldet.
    not: >
      Keine Gutschrift (negativer Betrag oder Bezug auf eine stornierte Rechnung),
      kein Lieferschein (keine Beträge), kein Angebot (Leistung noch nicht erbracht).

  - key: GUTSCHRIFT
    name: Gutschrift
    description: >
      Korrektur einer früheren Rechnung zugunsten des Empfängers. Nennt die Nummer der
      ursprünglichen Rechnung, einen negativen Betrag oder einen Erstattungsbetrag und
      einen Grund wie Storno, Mängelrüge oder Rücksendung.
    not: >
      Keine Rechnung (dort wird gefordert, nicht erstattet), keine Mahnung.

  - key: VERTRAG
    name: Vertrag
    description: >
      Individuelle Vereinbarung zwischen benannten Parteien. Nennt die Vertragsparteien
      namentlich, einen Vertragsgegenstand, Laufzeit oder Beginn, Vergütung sowie
      Kündigungsregelungen, und sieht Unterschriften beider Seiten vor.
    not: >
      Keine AGB (die gelten einseitig und ohne namentliche Gegenpartei), keine Rechnung.

  - key: AGB
    name: Allgemeine Geschäftsbedingungen
    description: >
      Vorformulierte Bedingungen, die ein Anbieter einseitig für eine Vielzahl von
      Verträgen stellt. Durchnummerierte Paragraphen zu Geltungsbereich, Vertragsschluss,
      Preisen, Haftung, Gewährleistung, Datenschutz und Gerichtsstand. Keine namentlich
      genannte Gegenpartei, keine Unterschriftenzeile.
    not: >
      Kein Vertrag (dort stehen konkrete Parteien und eine Unterschrift),
      keine Datenschutzerklärung als eigenständiges Dokument.

  - key: STATUSBERICHT
    name: Statusbericht
    description: >
      Periodischer Bericht über den Stand eines Vorhabens. Nennt einen Berichtszeitraum,
      eine Ampel- oder Fortschrittsbewertung, erreichte Meilensteine, offene Punkte,
      Risiken und den Ausblick auf die nächste Periode.
    not: >
      Kein Protokoll (dort gibt es Teilnehmer, Tagesordnungspunkte und Beschlüsse einer
      konkreten Sitzung).

  - key: PROTOKOLL
    name: Protokoll
    description: >
      Mitschrift einer konkreten Sitzung. Nennt Datum, Ort und Teilnehmerliste, gliedert
      sich in Tagesordnungspunkte und hält je Punkt Beschluss, Verantwortlichen und Termin
      fest.
    not: >
      Kein Statusbericht (dort gibt es keine Teilnehmer und keine Beschlüsse einer Sitzung).

  - key: SONSTIGES
    name: Sonstiges
    residual: true
    description: >
      Alles, was keiner der übrigen Klassen entspricht — etwa Anschreiben, Lieferscheine,
      Werbesendungen, Bedienungsanleitungen oder Bescheinigungen.
    not: >
      Diese Klasse wird nicht trainiert, sondern durch Ablehnung erreicht (Konzept § 7.4).
```

- [ ] **Schritt 2: Test schreiben**

`tests/test_classes.py`:

```python
"""Das Klassenschema ist Konfiguration. Fehler darin müssen beim Laden auffallen, nicht später."""

from pathlib import Path

import pytest

from doccls.classes import load_classes


def test_schema_laedt_und_enthaelt_die_erwarteten_klassen() -> None:
    schema = load_classes()
    assert schema.version == 1
    assert set(schema.keys()) == {
        "RECHNUNG", "GUTSCHRIFT", "VERTRAG", "AGB",
        "STATUSBERICHT", "PROTOKOLL", "SONSTIGES",
    }


def test_sonstiges_ist_die_restklasse_und_nicht_trainierbar() -> None:
    schema = load_classes()
    assert schema.get("SONSTIGES").residual is True
    assert "SONSTIGES" not in [c.key for c in schema.trainable()]
    assert len(schema.trainable()) == 6


def test_jede_klasse_hat_eine_abgrenzung() -> None:
    """Ohne `not` ist die Klasse für Labelnde nicht abgrenzbar – und κ sinkt (Konzept § 9.5)."""
    for klasse in load_classes().classes:
        assert klasse.not_.strip(), f"{klasse.key} ohne Abgrenzung"
        assert len(klasse.description.split()) >= 15, f"{klasse.key}: Beschreibung zu knapp"


def test_doppelter_schluessel_wird_abgelehnt(tmp_path: Path) -> None:
    pfad = tmp_path / "classes.yaml"
    pfad.write_text(
        "version: 1\n"
        "classes:\n"
        "  - {key: A, name: A, description: x, not: y}\n"
        "  - {key: A, name: B, description: x, not: y}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="doppelt"):
        load_classes(pfad)
```

- [ ] **Schritt 3: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_classes.py -v
```

Erwartet: `ModuleNotFoundError: No module named 'doccls.classes'`.

- [ ] **Schritt 4: `src/doccls/classes.py` schreiben**

```python
"""Klassenschema laden und prüfen.

Das Schema ist Konfiguration (`config/classes.yaml`), kein Code: Klassen ändern sich, ohne
dass etwas neu gebaut wird. Die `description` dient doppelt – als Definition für Labelnde
und als Kaltstart-Prototyp für die Einbettung (Konzept § 8.1).
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from doccls.config import PROJECT_ROOT

DEFAULT_PATH = PROJECT_ROOT / "config" / "classes.yaml"


class ClassDef(BaseModel):
    """Eine Dokumentklasse."""

    model_config = {"populate_by_name": True}

    key: str
    name: str
    description: str
    not_: str = Field(alias="not")
    """Wogegen die Klasse abzugrenzen ist. Ohne diese Angabe sind sich Labelnde uneins."""

    residual: bool = False
    """Restklasse: wird nicht trainiert, sondern durch Ablehnung erreicht (Konzept § 7.4)."""


class ClassSchema(BaseModel):
    """Alle Klassen mit Versionsstand. Die Version ordnet Auswertungen einem Schema zu."""

    version: int
    classes: list[ClassDef]

    def keys(self) -> list[str]:
        return [klasse.key for klasse in self.classes]

    def get(self, key: str) -> ClassDef:
        for klasse in self.classes:
            if klasse.key == key:
                return klasse
        raise KeyError(f"Unbekannte Klasse {key!r}. Bekannt: {', '.join(self.keys())}")

    def trainable(self) -> list[ClassDef]:
        """Klassen ohne die Restklasse – nur diese bekommen Trainingsbeispiele."""
        return [klasse for klasse in self.classes if not klasse.residual]


def load_classes(path: Path | None = None) -> ClassSchema:
    """Schema laden und prüfen. Doppelte Schlüssel sind ein Fehler, kein stiller Überschreiber."""
    quelle = path or DEFAULT_PATH
    schema = ClassSchema.model_validate(yaml.safe_load(quelle.read_text(encoding="utf-8")))
    schluessel = schema.keys()
    doppelt = {k for k in schluessel if schluessel.count(k) > 1}
    if doppelt:
        raise ValueError(f"Klassenschlüssel doppelt vergeben: {', '.join(sorted(doppelt))}")
    return schema
```

- [ ] **Schritt 5: Test laufen lassen, Erfolg bestätigen**

```bash
uv run pytest tests/test_classes.py -v
```

Erwartet: 4 passed.

- [ ] **Schritt 6: Prüfen und committen**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest
git add -A && git commit -m "Klassenschema als Konfiguration, mit Abgrenzungen"
```

---

## Aufgabe 3: Datenmodelle

**Dateien:**
- Anlegen: `src/doccls/models.py`
- Anlegen: `tests/test_models.py`

**Schnittstellen:**
- Liefert: `derive_id(*parts: bytes | str) -> str` (32 Hex-Zeichen),
  `normalize_text(text: str) -> str`, `SegmentKind` (StrEnum: `SEITE`, `ABSCHNITT`,
  `BLATT`, `ZEILE`, `MAIL_KOPF`, `MAIL_KOERPER`),
  `Document` (`document_id`, `source_path`, `file_name`, `media_type`, `content_sha256`,
  `size_bytes`, `parent_document_id: str | None`, `ingested_at: datetime`,
  `needs_ocr: bool`, `ocr_applied: bool`),
  `Segment` (`segment_id`, `document_id`, `index`, `kind`, `locator`, `heading: str | None`,
  `text`) mit `Segment.create(...)`,
  `SCHEMAS: dict[str, dict[str, pl.DataType]]`, `to_frame(rows, name) -> pl.DataFrame`

- [ ] **Schritt 1: Test schreiben**

`tests/test_models.py`:

```python
"""IDs müssen aus dem Inhalt abgeleitet und damit über Läufe hinweg stabil sein."""

from datetime import UTC, datetime

import polars as pl

from doccls.models import SCHEMAS, Document, Segment, SegmentKind, derive_id, normalize_text, to_frame


def beispiel_dokument(inhalt: bytes = b"Rechnung 4711") -> Document:
    return Document.create(source_path="rechnungen/a.pdf", media_type="application/pdf",
                           content=inhalt, ingested_at=datetime(2026, 1, 1, tzinfo=UTC))


def test_id_ist_stabil_und_32_zeichen_lang() -> None:
    a, b = beispiel_dokument(), beispiel_dokument()
    assert a.document_id == b.document_id
    assert len(a.document_id) == 32


def test_geaenderter_inhalt_ergibt_neue_version() -> None:
    assert beispiel_dokument(b"x").document_id != beispiel_dokument(b"y").document_id


def test_gleicher_inhalt_an_zwei_pfaden_sind_zwei_dokumente() -> None:
    """Sonst würde die zweite Fundstelle übersprungen und wäre nirgends verzeichnet."""
    a = Document.create(source_path="a/x.pdf", media_type="application/pdf",
                        content=b"gleich", ingested_at=datetime(2026, 1, 1, tzinfo=UTC))
    b = Document.create(source_path="b/x.pdf", media_type="application/pdf",
                        content=b"gleich", ingested_at=datetime(2026, 1, 1, tzinfo=UTC))
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
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_models.py -v
```

Erwartet: `ModuleNotFoundError: No module named 'doccls.models'`.

- [ ] **Schritt 3: `src/doccls/models.py` schreiben**

```python
"""Datenmodelle und die zugehörigen Polars-Schemas.

Zwei Ebenen:

* ``Document`` – eine Version einer Datei. Die Identität umfasst Pfad **und** Inhalts-Hash:
  Dieselbe Datei an zwei Stellen sind zwei Dokumente, sonst würde die zweite übersprungen
  und wäre nirgends verzeichnet. Geänderter Inhalt ergibt eine neue ID; die alte Version
  bleibt erhalten.
* ``Segment``  – eine Einheit im Dokument: PDF-Seite, Abschnitt, Tabellenblatt, Mailteil.
  Ergebnis der Extraktion. Wird gespeichert, damit Merkmale neu gebildet werden können,
  ohne die Originale erneut zu lesen.

IDs sind aus dem Inhalt abgeleitet und damit stabil: Ein erneuter Lauf erzeugt dieselben
IDs und überschreibt, statt zu verdoppeln.
"""

import hashlib
import re
import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import Self

import polars as pl
from pydantic import BaseModel

ID_LENGTH = 32
"""Hex-Zeichen abgeleiteter IDs. 32 Zeichen = 128 Bit, für Kollisionen ausreichend."""

HYPHENATION = re.compile(r"(\w)-\s*\n\s*(\w)")
"""Silbentrennung am Zeilenende: „Rech-\\nnung“. Ohne Auflösung zerfällt jedes zweite
Fachwort in zwei unbekannte Bruchstücke."""

WHITESPACE = re.compile(r"\s+")


def derive_id(*parts: bytes | str) -> str:
    """ID aus dem Inhalt ableiten. Teile werden mit ``\\0`` getrennt, damit „ab“+„c“ und
    „a“+„bc“ verschiedene IDs ergeben."""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8") if isinstance(part, str) else part)
        hasher.update(b"\0")
    return hasher.hexdigest()[:ID_LENGTH]


def normalize_text(text: str) -> str:
    """NFKC, Silbentrennung auflösen, Whitespace zusammenfassen.

    NFKC vereinheitlicht Ligaturen („ﬁ“ → „fi“) und Kompatibilitätszeichen, die je nach
    PDF-Erzeuger verschieden kodiert sind. Ohne diesen Schritt wären zwei gleich aussehende
    Dokumente verschiedene Zeichenfolgen.
    """
    text = unicodedata.normalize("NFKC", text)
    text = HYPHENATION.sub(r"\1\2", text)
    return WHITESPACE.sub(" ", text).strip()


class SegmentKind(StrEnum):
    """Was für eine Einheit ein Segment ist. Bestimmt, wie ``locator`` zu lesen ist."""

    SEITE = "seite"
    ABSCHNITT = "abschnitt"
    BLATT = "blatt"
    ZEILE = "zeile"
    MAIL_KOPF = "mail_kopf"
    MAIL_KOERPER = "mail_koerper"


class Document(BaseModel):
    """Eine Version einer Datei."""

    model_config = {"frozen": True}

    document_id: str
    source_path: str
    """Pfad relativ zu ``RAW_DIR``. Bei Anhängen: ``eltern.eml!anhang.pdf``."""

    file_name: str
    media_type: str
    content_sha256: str
    size_bytes: int
    parent_document_id: str | None = None
    """Gesetzt bei E-Mail-Anhängen. Eine Mail und ihre Rechnung sind zwei Dokumente."""

    ingested_at: datetime
    needs_ocr: bool = False
    """Zu wenig Text je Seite – das Dokument ist vermutlich gescannt."""

    ocr_applied: bool = False

    @classmethod
    def create(
        cls,
        *,
        source_path: str,
        media_type: str,
        content: bytes,
        ingested_at: datetime,
        parent_document_id: str | None = None,
    ) -> Self:
        return cls(
            document_id=derive_id(source_path, content),
            source_path=source_path,
            file_name=source_path.rsplit("/", 1)[-1],
            media_type=media_type,
            content_sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            parent_document_id=parent_document_id,
            ingested_at=ingested_at,
        )


class Segment(BaseModel):
    """Eine Einheit im Dokument, Ergebnis der Extraktion."""

    model_config = {"frozen": True}

    segment_id: str
    document_id: str
    index: int
    kind: SegmentKind
    locator: str
    """Menschenlesbare Fundstelle: „S. 3“, „Blatt Positionen“, „Anhang 2“."""

    heading: str | None
    text: str

    @classmethod
    def create(
        cls,
        *,
        document: Document,
        index: int,
        kind: SegmentKind,
        locator: str,
        text: str,
        heading: str | None = None,
    ) -> Self:
        return cls(
            segment_id=derive_id(document.document_id, str(index)),
            document_id=document.document_id,
            index=index,
            kind=kind,
            locator=locator,
            heading=heading,
            text=text,
        )


SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "documents": {
        "document_id": pl.String(),
        "source_path": pl.String(),
        "file_name": pl.String(),
        "media_type": pl.String(),
        "content_sha256": pl.String(),
        "size_bytes": pl.Int64(),
        "parent_document_id": pl.String(),
        "ingested_at": pl.Datetime(time_unit="us", time_zone="UTC"),
        "needs_ocr": pl.Boolean(),
        "ocr_applied": pl.Boolean(),
    },
    "segments": {
        "segment_id": pl.String(),
        "document_id": pl.String(),
        "index": pl.Int64(),
        "kind": pl.String(),
        "locator": pl.String(),
        "heading": pl.String(),
        "text": pl.String(),
    },
}
"""Schemas explizit, nicht abgeleitet: Eine leere Tabelle bekommt sonst andere Spaltentypen
als eine gefüllte, und der zweite Lauf scheitert am ersten."""


def to_frame(rows: list[Document] | list[Segment], name: str) -> pl.DataFrame:
    """Modelle in einen DataFrame mit festem Schema überführen – auch wenn ``rows`` leer ist."""
    return pl.DataFrame([row.model_dump() for row in rows], schema=SCHEMAS[name])
```

- [ ] **Schritt 4: Test laufen lassen, Erfolg bestätigen**

```bash
uv run pytest tests/test_models.py -v
```

Erwartet: 8 passed. `SegmentKind` ist ein `StrEnum` und damit für Polars eine Zeichenkette –
`model_dump()` braucht deshalb kein `mode="json"`, das die Zeitstempel zerstören würde.

- [ ] **Schritt 5: Prüfen und committen**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest
git add -A && git commit -m "Datenmodelle Document und Segment mit stabilen IDs"
```

---

## Aufgabe 4: Testdaten — Inhalte

**Dateien:**
- Anlegen: `src/doccls/generation/__init__.py`, `src/doccls/generation/content.py`
- Anlegen: `tests/test_generation_content.py`

**Schnittstellen:**
- Nutzt: `doccls.classes.load_classes`
- Liefert: `Block` (`heading: str | None`, `paragraphs: tuple[str, ...]`,
  `table: tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] | None`),
  `DocumentSpec` (`class_key: str`, `template_id: str`, `variant: int`, `title: str`,
  `blocks: tuple[Block, ...]`, `formats: tuple[str, ...]`),
  `build_corpus(seed: int = 42) -> list[DocumentSpec]`

  Alles unveränderlich (`frozen=True`, Tupel statt Listen): Ein Korpus, den ein Aufrufer
  versehentlich ändert, wäre nicht mehr reproduzierbar.

**Entwurfsregeln** (Konzept § 9.5, § 8.5):

1. **Das Format darf die Klasse nicht verraten.** Jede Klasse erscheint in mindestens zwei
   Formaten, und jedes Format trägt mindestens drei Klassen. Sonst lernt der Klassifikator
   „XLSX ⇒ Rechnung" und sieht auf dem Gold-Set glänzend aus.
2. **Gemeinsamer Briefkopf und gemeinsame Fußzeile über alle Klassen.** Sonst wird das
   Layout zur Abkürzung.
3. **Absichtliche Verwechslungsfälle**: Gutschriften in Rechnungsaufmachung, AGB als
   Vertragsanhang, Protokolle in Berichtsform.
4. **Variation innerhalb einer Vorlage** über Beträge, Namen, Daten — aber `template_id`
   bleibt gleich. Der Gold-Set-Schnitt läuft über `template_id`, nicht über Dokumente
   (Konzept § 9.2: Aufteilung nach Dokumentfamilie).

- [ ] **Schritt 1: Test schreiben**

`tests/test_generation_content.py`:

```python
"""Der Korpus ist der Maßstab für alles Weitere. Seine Eigenschaften werden geprüft, nicht angenommen."""

from collections import Counter

from doccls.classes import load_classes
from doccls.generation.content import build_corpus


def test_korpus_ist_deterministisch() -> None:
    assert build_corpus(seed=42) == build_corpus(seed=42)


def test_jede_klasse_hat_genug_vorlagen_und_dokumente() -> None:
    """Fünf Vorlagen je Klasse gehen ins Gold-Set und ergeben dort 50 Dokumente (Konzept § 9.2)."""
    korpus = build_corpus()
    je_klasse = Counter(spec.class_key for spec in korpus)
    for klasse in load_classes().keys():
        assert je_klasse[klasse] >= 80, f"{klasse}: nur {je_klasse[klasse]} Dokumente"
        vorlagen = {s.template_id for s in korpus if s.class_key == klasse}
        assert len(vorlagen) >= 8, f"{klasse}: nur {len(vorlagen)} Vorlagen"


def test_format_verraet_die_klasse_nicht() -> None:
    korpus = build_corpus()
    for klasse in load_classes().keys():
        formate = {f for s in korpus if s.class_key == klasse for f in s.formats}
        assert len(formate) >= 2, f"{klasse} nur in {formate}"
    for fmt in ("pdf", "docx", "xlsx", "eml"):
        klassen = {s.class_key for s in korpus if fmt in s.formats}
        assert len(klassen) >= 3, f"{fmt} trägt nur {klassen}"


def test_varianten_einer_vorlage_unterscheiden_sich_im_text() -> None:
    korpus = build_corpus()
    eine = [s for s in korpus if s.template_id == "RECHNUNG-standard"]
    texte = {" ".join(p for b in s.blocks for p in b.paragraphs) for s in eine}
    assert len(texte) == len(eine), "Varianten sind identisch – keine echte Variation"


def test_gutschriften_sind_als_verwechslungsfall_gebaut() -> None:
    """Rechnung ↔ Gutschrift ist der harte Fall (Konzept Anhang D, Kosinus 0,964)."""
    korpus = build_corpus()
    gutschriften = [s for s in korpus if s.class_key == "GUTSCHRIFT"]
    text = " ".join(p for s in gutschriften for b in s.blocks for p in b.paragraphs)
    assert "Rechnung" in text, "Gutschriften müssen ihre Ursprungsrechnung nennen"


def test_alle_klassen_teilen_briefkopf_und_fusszeile() -> None:
    """Sonst wird das Layout zur Abkürzung statt des Inhalts (Konzept § 8.5)."""
    fusszeilen = {s.blocks[-1].paragraphs[-1] for s in build_corpus()}
    assert len(fusszeilen) <= 3, f"Zu viele verschiedene Fußzeilen: {len(fusszeilen)}"
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_generation_content.py -v
```

Erwartet: `ModuleNotFoundError: No module named 'doccls.generation'`.

- [ ] **Schritt 3: `src/doccls/generation/content.py` schreiben**

Der Aufbau trennt **Vorlage** (was für ein Dokument) von **Variante** (welche Zahlen und
Namen). Die Vorlagen stehen als Daten in `TEMPLATES`, die Variation erzeugt ein
eigenständig geseedeter `Random` je Dokument — so bleibt eine Variante gleich, auch wenn
vorher eine Vorlage ergänzt wird.

```python
"""Inhalte der synthetischen Testdokumente – reine Funktionen, kein Dateisystem.

Aufbau: ``TEMPLATES`` beschreibt je Klasse mehrere Vorlagen; ``build_corpus`` erzeugt je
Vorlage zehn Varianten mit anderen Zahlen, Namen und Daten. Der Gold-Set-Schnitt läuft
später über ``template_id``, nicht über Dokumente – zwei Varianten derselben Vorlage sind
fast dasselbe Dokument und dürfen nicht auf beide Seiten des Schnitts fallen
(Konzept § 9.2).

Die Wahrheit, die hier konstruiert wird, ist in ``docs/testdaten.md`` beschrieben. Beides
muss gemeinsam gepflegt werden.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass

VARIANTS_PER_TEMPLATE = 10

FOOTER = (
    "Muster Handels GmbH · Industriestraße 17 · 40211 Düsseldorf · "
    "HRB 44821 Amtsgericht Düsseldorf · USt-IdNr. DE812345678"
)
"""Gemeinsame Fußzeile über alle Klassen. Wäre sie klassenspezifisch, würde das Modell sie
statt des Inhalts lernen (Konzept § 8.5, Abkürzungen)."""

FIRMEN = ["Nordwind Logistik GmbH", "Steinbach Elektro KG", "Vogel & Partner mbB",
          "Auric Systems AG", "Lindner Bau GmbH", "Kessler Datentechnik e.K."]
PERSONEN = ["A. Albers", "B. Vogt", "C. Yilmaz", "D. Brandt", "E. Kessler", "F. Radtke"]
STAEDTE = ["Düsseldorf", "Leipzig", "Bremen", "Augsburg", "Rostock", "Kassel"]


@dataclass(frozen=True)
class Block:
    """Ein Abschnitt: optionale Überschrift, Absätze, optional eine Tabelle."""

    heading: str | None = None
    paragraphs: tuple[str, ...] = ()
    table: tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] | None = None


@dataclass(frozen=True)
class DocumentSpec:
    """Ein zu erzeugendes Dokument, unabhängig vom Ausgabeformat."""

    class_key: str
    template_id: str
    variant: int
    title: str
    blocks: tuple[Block, ...]
    formats: tuple[str, ...]


@dataclass(frozen=True)
class Template:
    """Eine Dokumentvorlage. ``build`` bekommt einen geseedeten Zufall und liefert die Blöcke."""

    template_id: str
    class_key: str
    formats: tuple[str, ...]
    build: Callable[[random.Random], tuple[str, tuple[Block, ...]]]
    """Liefert (Titel, Blöcke). Der Titel entsteht erst beim Bauen, weil er die Nummer der
    Variante nennt – „Rechnung RE-2026-4711“."""


def _betrag(rnd: random.Random) -> tuple[float, float, float]:
    """Netto, Umsatzsteuer, Brutto – konsistent gerundet, damit sie zueinander passen."""
    netto = round(rnd.uniform(180, 24_000), 2)
    ust = round(netto * 0.19, 2)
    return netto, ust, round(netto + ust, 2)


def _eur(wert: float) -> str:
    return f"{wert:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".") + " EUR"


def _kopf(rnd: random.Random) -> Block:
    """Briefkopf – für alle Klassen gleich aufgebaut."""
    return Block(paragraphs=(
        "Muster Handels GmbH, Industriestraße 17, 40211 Düsseldorf",
        f"{rnd.choice(FIRMEN)}, {rnd.choice(STAEDTE)}",
    ))


def _fuss() -> Block:
    return Block(paragraphs=(FOOTER,))


# --------------------------------------------------------------------------- Vorlagen

def _rechnung(rnd: random.Random, *, storniert: bool = False) -> tuple[str, tuple[Block, ...]]:
    nummer = f"RE-2026-{rnd.randint(1000, 9999)}"
    netto, ust, brutto = _betrag(rnd)
    tag = rnd.randint(1, 28)
    zeilen = tuple(
        (str(i), leistung, str(rnd.randint(1, 20)), _eur(round(netto / 3, 2)))
        for i, leistung in enumerate(
            rnd.sample(["Wartung Anlagentechnik", "Materiallieferung Profilschienen",
                        "Montagestunden", "Transport und Verpackung",
                        "Softwarepflege Jahreslizenz"], 3), start=1)
    )
    hinweis = ("Diese Rechnung ersetzt die stornierte Rechnung RE-2026-"
               f"{rnd.randint(1000, 9999)}.") if storniert else ""
    return (f"Rechnung {nummer}", (
        _kopf(rnd),
        Block(heading=f"Rechnung {nummer}", paragraphs=tuple(p for p in (
            f"Rechnungsdatum: {tag:02d}.03.2026",
            f"Leistungszeitraum: 01.02.2026 bis {tag:02d}.02.2026",
            f"Kundennummer: K-{rnd.randint(10000, 99999)}",
            hinweis,
        ) if p)),
        Block(heading="Positionen",
              table=(("Pos.", "Leistung", "Menge", "Einzelpreis"), zeilen)),
        Block(heading="Zahlung", paragraphs=(
            f"Nettobetrag: {_eur(netto)}",
            f"zzgl. 19 % Umsatzsteuer: {_eur(ust)}",
            f"Rechnungsbetrag: {_eur(brutto)}",
            f"Zahlbar bis {tag:02d}.04.2026 ohne Abzug.",
            "Bankverbindung: IBAN DE02 1203 0000 0000 2020 51, BIC BYLADEM1001",
        )),
        _fuss(),
    ))


def _gutschrift(rnd: random.Random, *, grund: str) -> tuple[str, tuple[Block, ...]]:
    """Bewusst in Rechnungsaufmachung – der harte Verwechslungsfall (Konzept Anhang D)."""
    nummer = f"GU-2026-{rnd.randint(100, 999)}"
    ursprung = f"RE-2026-{rnd.randint(1000, 9999)}"
    netto, ust, brutto = _betrag(rnd)
    return (f"Gutschrift {nummer}", (
        _kopf(rnd),
        Block(heading=f"Gutschrift {nummer}", paragraphs=(
            f"Gutschriftdatum: {rnd.randint(1, 28):02d}.04.2026",
            f"Bezug: Rechnung {ursprung} vom {rnd.randint(1, 28):02d}.03.2026",
            f"Grund der Gutschrift: {grund}",
        )),
        Block(heading="Positionen",
              table=(("Pos.", "Bezug", "Menge", "Betrag"),
                     (("1", f"Teilstorno zu {ursprung}", "1", _eur(-netto)),))),
        Block(heading="Erstattung", paragraphs=(
            f"Nettobetrag: {_eur(-netto)}",
            f"zzgl. 19 % Umsatzsteuer: {_eur(-ust)}",
            f"Gutschriftbetrag: {_eur(-brutto)}",
            "Der Betrag wird auf das uns bekannte Konto erstattet. Eine Zahlung Ihrerseits "
            "ist nicht erforderlich.",
        )),
        _fuss(),
    ))


def _vertrag(rnd: random.Random, *, art: str, mit_agb_anhang: bool) -> tuple[str, tuple[Block, ...]]:
    partner, vertreter = rnd.choice(FIRMEN), rnd.choice(PERSONEN)
    bloecke = [
        _kopf(rnd),
        Block(heading=f"{art} zwischen Muster Handels GmbH und {partner}", paragraphs=(
            f"Die Muster Handels GmbH, Industriestraße 17, 40211 Düsseldorf, vertreten durch "
            f"{rnd.choice(PERSONEN)} – nachfolgend Auftragnehmer –",
            f"und {partner}, {rnd.choice(STAEDTE)}, vertreten durch {vertreter} "
            "– nachfolgend Auftraggeber – schließen folgenden Vertrag:",
        )),
        Block(heading="§ 1 Vertragsgegenstand", paragraphs=(
            f"Der Auftragnehmer erbringt die Leistungen gemäß Leistungsverzeichnis "
            f"vom {rnd.randint(1, 28):02d}.01.2026.",)),
        Block(heading="§ 2 Laufzeit", paragraphs=(
            f"Der Vertrag beginnt am 01.{rnd.randint(1, 9):02d}.2026 und läuft "
            f"{rnd.choice(['12', '24', '36'])} Monate.",)),
        Block(heading="§ 3 Vergütung", paragraphs=(
            f"Die Vergütung beträgt {_eur(round(rnd.uniform(1200, 9000), 2))} monatlich "
            "zuzüglich gesetzlicher Umsatzsteuer.",)),
        Block(heading="§ 4 Kündigung", paragraphs=(
            "Der Vertrag kann mit einer Frist von drei Monaten zum Quartalsende "
            "schriftlich gekündigt werden.",)),
    ]
    if mit_agb_anhang:
        bloecke.append(Block(heading="Anlage 1: Allgemeine Geschäftsbedingungen",
                             paragraphs=("Es gelten ergänzend die AGB des Auftragnehmers in "
                                         "der Fassung vom 01.01.2026.",)))
    bloecke.append(Block(heading="Unterschriften", paragraphs=(
        "Düsseldorf, den ______________    ______________________ (Auftragnehmer)",
        f"{rnd.choice(STAEDTE)}, den ______________    ______________________ (Auftraggeber)",
    )))
    bloecke.append(_fuss())
    return (f"{art} {partner}", tuple(bloecke))


def _agb(rnd: random.Random, *, bereich: str) -> tuple[str, tuple[Block, ...]]:
    fassung = f"Fassung vom 01.{rnd.randint(1, 9):02d}.2026"
    paragraphen = [
        ("§ 1 Geltungsbereich",
         "Diese Bedingungen gelten für alle Verträge zwischen dem Anbieter und Unternehmern "
         "im Sinne des § 14 BGB. Abweichende Bedingungen des Kunden werden nicht Vertrags"
         "bestandteil, auch wenn ihnen nicht ausdrücklich widersprochen wird."),
        ("§ 2 Vertragsschluss",
         "Angebote des Anbieters sind freibleibend. Ein Vertrag kommt erst mit schriftlicher "
         "Auftragsbestätigung zustande."),
        ("§ 3 Preise und Zahlung",
         "Alle Preise verstehen sich netto zuzüglich Umsatzsteuer. Rechnungen sind innerhalb "
         "von 30 Tagen ohne Abzug zur Zahlung fällig."),
        ("§ 4 Gewährleistung",
         "Die Gewährleistungsfrist beträgt zwölf Monate ab Gefahrübergang."),
        ("§ 5 Haftung",
         "Der Anbieter haftet unbeschränkt bei Vorsatz und grober Fahrlässigkeit. Im Übrigen "
         "ist die Haftung auf den vertragstypischen, vorhersehbaren Schaden begrenzt."),
        ("§ 6 Gerichtsstand",
         f"Gerichtsstand für alle Streitigkeiten ist {rnd.choice(STAEDTE)}. Es gilt "
         "ausschließlich deutsches Recht."),
    ]
    return (f"Allgemeine Geschäftsbedingungen {bereich}", (
        _kopf(rnd),
        Block(heading=f"Allgemeine Geschäftsbedingungen – {bereich}",
              paragraphs=(fassung,)),
        *[Block(heading=titel, paragraphs=(text,)) for titel, text in paragraphen],
        _fuss(),
    ))


def _statusbericht(rnd: random.Random, *, ampel: str) -> tuple[str, tuple[Block, ...]]:
    kw = rnd.randint(10, 40)
    return (f"Statusbericht KW {kw}", (
        _kopf(rnd),
        Block(heading=f"Statusbericht – Berichtszeitraum KW {kw - 1} bis KW {kw}/2026",
              paragraphs=(f"Gesamtstatus: {ampel}",
                          f"Fortschritt: {rnd.randint(15, 95)} % der geplanten Leistung.")),
        Block(heading="Erreichte Meilensteine",
              table=(("Meilenstein", "Geplant", "Erreicht"),
                     (("Anforderungsaufnahme", "KW 08", "KW 08"),
                      ("Teillieferung 1", f"KW {kw - 2}", f"KW {kw - 1}"),
                      ("Abnahmetest", f"KW {kw + 4}", "offen")))),
        Block(heading="Offene Punkte und Risiken", paragraphs=(
            f"Die Lieferung der Bauteile verzögert sich um {rnd.randint(1, 6)} Wochen.",
            "Gegenmaßnahme: Ersatzlieferant angefragt, Entscheidung in der kommenden Woche.",
        )),
        Block(heading="Ausblick", paragraphs=(
            f"In KW {kw + 1} beginnt die Integrationsphase.",)),
        _fuss(),
    ))


def _protokoll(rnd: random.Random, *, berichtsform: bool) -> tuple[str, tuple[Block, ...]]:
    """``berichtsform`` erzeugt Protokolle, die wie Statusberichte aussehen – Verwechslungsfall."""
    nummer, tag = rnd.randint(1, 30), rnd.randint(1, 28)
    tops = (("1", "Freigabe Lastenheft", "Freigegeben", rnd.choice(PERSONEN), f"{tag:02d}.05.2026"),
            ("2", "Budgetnachtrag", "Vertagt", rnd.choice(PERSONEN), f"{tag:02d}.06.2026"),
            ("3", "Termine Abnahme", "Beschlossen", rnd.choice(PERSONEN), f"{tag:02d}.07.2026"))
    kopf = Block(heading=f"Protokoll der {nummer}. Projektbesprechung", paragraphs=(
        f"Datum: {tag:02d}.04.2026    Ort: {rnd.choice(STAEDTE)}, Besprechungsraum 2",
        "Teilnehmer: " + ", ".join(rnd.sample(PERSONEN, 4)),
        "Protokollführung: " + rnd.choice(PERSONEN),
    ))
    inhalt = (Block(heading="Zusammenfassung des Sitzungsverlaufs", paragraphs=(
        "Der Stand der Arbeitspakete wurde besprochen; die Ampel steht auf gelb.",
        f"Der Nachtrag über {_eur(round(rnd.uniform(2000, 40000), 2))} wurde vertagt.",))
        if berichtsform else
        Block(heading="Tagesordnungspunkte",
              table=(("TOP", "Thema", "Beschluss", "Verantwortlich", "Termin"), tops)))
    return (f"Protokoll Besprechung Nr. {nummer}", (
        _kopf(rnd), kopf, inhalt,
        Block(heading="Nächste Sitzung", paragraphs=(
            f"Die nächste Besprechung findet am {tag:02d}.05.2026 statt.",
            "Einwände gegen dieses Protokoll sind binnen fünf Werktagen mitzuteilen.")),
        _fuss(),
    ))


def _sonstiges(rnd: random.Random, *, art: str) -> tuple[str, tuple[Block, ...]]:
    """Die Restklasse ist mit Absicht heterogen – sie hat kein gemeinsames Merkmal (Konzept § 2)."""
    texte = {
        "Anschreiben": ("Sehr geehrte Damen und Herren,",
                        "anbei erhalten Sie die angeforderten Unterlagen zur weiteren "
                        "Verwendung. Für Rückfragen stehen wir gern zur Verfügung.",
                        "Mit freundlichen Grüßen"),
        "Lieferschein": (f"Lieferschein-Nr. LS-2026-{rnd.randint(100, 999)}",
                         "Die nachstehend aufgeführten Waren wurden vollständig geliefert. "
                         "Beträge sind diesem Beleg nicht zu entnehmen."),
        "Werbebrief": ("Jetzt Frühbucherrabatt sichern!",
                       "Bis zum 30.06.2026 erhalten Sie auf unser gesamtes Zubehörsortiment "
                       "einen Nachlass. Sprechen Sie uns an."),
        "Bedienungsanleitung": ("Inbetriebnahme",
                                "Verbinden Sie das Gerät mit der Spannungsversorgung und "
                                "warten Sie, bis die Betriebsanzeige dauerhaft leuchtet."),
        "Bescheinigung": (f"Bescheinigung Nr. B-{rnd.randint(100, 999)}",
                          "Hiermit wird bescheinigt, dass die genannte Person im Zeitraum "
                          "01.01.2026 bis 31.03.2026 an der Schulung teilgenommen hat."),
    }
    return (art, (_kopf(rnd), Block(heading=art, paragraphs=texte[art]), _fuss()))


TEMPLATES: tuple[Template, ...] = (
    # RECHNUNG – in vier Formaten, damit das Format nichts verrät
    Template("RECHNUNG-standard", "RECHNUNG", ("pdf",), _rechnung),
    Template("RECHNUNG-storno-ersatz", "RECHNUNG", ("pdf",),
             lambda r: _rechnung(r, storniert=True)),
    Template("RECHNUNG-tabelle", "RECHNUNG", ("xlsx",), _rechnung),
    Template("RECHNUNG-mail", "RECHNUNG", ("eml",), _rechnung),
    Template("RECHNUNG-docx", "RECHNUNG", ("docx",), _rechnung),
    Template("RECHNUNG-mail-anhang", "RECHNUNG", ("eml",), _rechnung),
    Template("RECHNUNG-storno-xlsx", "RECHNUNG", ("xlsx",),
             lambda r: _rechnung(r, storniert=True)),
    Template("RECHNUNG-storno-docx", "RECHNUNG", ("docx",),
             lambda r: _rechnung(r, storniert=True)),

    Template("GUTSCHRIFT-storno", "GUTSCHRIFT", ("pdf",),
             lambda r: _gutschrift(r, grund="Storno der Lieferung")),
    Template("GUTSCHRIFT-maengel", "GUTSCHRIFT", ("pdf",),
             lambda r: _gutschrift(r, grund="Mängelrüge, Preisminderung")),
    Template("GUTSCHRIFT-ruecksendung", "GUTSCHRIFT", ("docx",),
             lambda r: _gutschrift(r, grund="Rücksendung der Ware")),
    Template("GUTSCHRIFT-tabelle", "GUTSCHRIFT", ("xlsx",),
             lambda r: _gutschrift(r, grund="Teilstorno")),
    Template("GUTSCHRIFT-mail", "GUTSCHRIFT", ("eml",),
             lambda r: _gutschrift(r, grund="Doppelberechnung")),
    Template("GUTSCHRIFT-skonto", "GUTSCHRIFT", ("pdf",),
             lambda r: _gutschrift(r, grund="Nachträglicher Skontoabzug")),
    Template("GUTSCHRIFT-bonus", "GUTSCHRIFT", ("docx",),
             lambda r: _gutschrift(r, grund="Jahresbonus")),
    Template("GUTSCHRIFT-fracht", "GUTSCHRIFT", ("xlsx",),
             lambda r: _gutschrift(r, grund="Zu viel berechnete Frachtkosten")),

    Template("VERTRAG-dienst", "VERTRAG", ("pdf",),
             lambda r: _vertrag(r, art="Dienstleistungsvertrag", mit_agb_anhang=False)),
    Template("VERTRAG-wartung", "VERTRAG", ("docx",),
             lambda r: _vertrag(r, art="Wartungsvertrag", mit_agb_anhang=False)),
    Template("VERTRAG-miete", "VERTRAG", ("pdf",),
             lambda r: _vertrag(r, art="Mietvertrag", mit_agb_anhang=False)),
    Template("VERTRAG-mit-agb", "VERTRAG", ("pdf",),
             lambda r: _vertrag(r, art="Liefervertrag", mit_agb_anhang=True)),
    Template("VERTRAG-rahmen", "VERTRAG", ("docx",),
             lambda r: _vertrag(r, art="Rahmenvertrag", mit_agb_anhang=True)),
    Template("VERTRAG-werk", "VERTRAG", ("pdf",),
             lambda r: _vertrag(r, art="Werkvertrag", mit_agb_anhang=False)),
    Template("VERTRAG-lizenz", "VERTRAG", ("docx",),
             lambda r: _vertrag(r, art="Lizenzvertrag", mit_agb_anhang=True)),
    Template("VERTRAG-mail", "VERTRAG", ("eml",),
             lambda r: _vertrag(r, art="Beratungsvertrag", mit_agb_anhang=False)),

    Template("AGB-allgemein", "AGB", ("pdf",), lambda r: _agb(r, bereich="Lieferung und Leistung")),
    Template("AGB-einkauf", "AGB", ("pdf",), lambda r: _agb(r, bereich="Einkauf")),
    Template("AGB-online", "AGB", ("docx",), lambda r: _agb(r, bereich="Onlineshop")),
    Template("AGB-vermietung", "AGB", ("pdf",), lambda r: _agb(r, bereich="Vermietung")),
    Template("AGB-service", "AGB", ("docx",), lambda r: _agb(r, bereich="Servicearbeiten")),
    Template("AGB-software", "AGB", ("pdf",), lambda r: _agb(r, bereich="Softwareüberlassung")),
    Template("AGB-transport", "AGB", ("docx",), lambda r: _agb(r, bereich="Transport")),
    Template("AGB-mail", "AGB", ("eml",), lambda r: _agb(r, bereich="Wartung")),

    Template("STATUS-gruen", "STATUSBERICHT", ("docx",),
             lambda r: _statusbericht(r, ampel="grün")),
    Template("STATUS-gelb", "STATUSBERICHT", ("docx",),
             lambda r: _statusbericht(r, ampel="gelb")),
    Template("STATUS-rot", "STATUSBERICHT", ("pdf",),
             lambda r: _statusbericht(r, ampel="rot")),
    Template("STATUS-pdf-gruen", "STATUSBERICHT", ("pdf",),
             lambda r: _statusbericht(r, ampel="grün")),
    Template("STATUS-xlsx", "STATUSBERICHT", ("xlsx",),
             lambda r: _statusbericht(r, ampel="gelb")),
    Template("STATUS-mail", "STATUSBERICHT", ("eml",),
             lambda r: _statusbericht(r, ampel="grün")),
    Template("STATUS-monat", "STATUSBERICHT", ("docx",),
             lambda r: _statusbericht(r, ampel="gelb")),
    Template("STATUS-quartal", "STATUSBERICHT", ("pdf",),
             lambda r: _statusbericht(r, ampel="rot")),

    Template("PROTOKOLL-top", "PROTOKOLL", ("docx",),
             lambda r: _protokoll(r, berichtsform=False)),
    Template("PROTOKOLL-top-pdf", "PROTOKOLL", ("pdf",),
             lambda r: _protokoll(r, berichtsform=False)),
    Template("PROTOKOLL-bericht", "PROTOKOLL", ("docx",),
             lambda r: _protokoll(r, berichtsform=True)),
    Template("PROTOKOLL-bericht-pdf", "PROTOKOLL", ("pdf",),
             lambda r: _protokoll(r, berichtsform=True)),
    Template("PROTOKOLL-jour-fixe", "PROTOKOLL", ("docx",),
             lambda r: _protokoll(r, berichtsform=False)),
    Template("PROTOKOLL-mail", "PROTOKOLL", ("eml",),
             lambda r: _protokoll(r, berichtsform=False)),
    Template("PROTOKOLL-xlsx", "PROTOKOLL", ("xlsx",),
             lambda r: _protokoll(r, berichtsform=False)),
    Template("PROTOKOLL-abnahme", "PROTOKOLL", ("pdf",),
             lambda r: _protokoll(r, berichtsform=False)),

    Template("SONST-anschreiben", "SONSTIGES", ("pdf",),
             lambda r: _sonstiges(r, art="Anschreiben")),
    Template("SONST-anschreiben-mail", "SONSTIGES", ("eml",),
             lambda r: _sonstiges(r, art="Anschreiben")),
    Template("SONST-lieferschein", "SONSTIGES", ("pdf",),
             lambda r: _sonstiges(r, art="Lieferschein")),
    Template("SONST-lieferschein-xlsx", "SONSTIGES", ("xlsx",),
             lambda r: _sonstiges(r, art="Lieferschein")),
    Template("SONST-werbung", "SONSTIGES", ("docx",),
             lambda r: _sonstiges(r, art="Werbebrief")),
    Template("SONST-anleitung", "SONSTIGES", ("pdf",),
             lambda r: _sonstiges(r, art="Bedienungsanleitung")),
    Template("SONST-bescheinigung", "SONSTIGES", ("docx",),
             lambda r: _sonstiges(r, art="Bescheinigung")),
    Template("SONST-bescheinigung-pdf", "SONSTIGES", ("pdf",),
             lambda r: _sonstiges(r, art="Bescheinigung")),
)


def build_corpus(seed: int = 42) -> list[DocumentSpec]:
    """Je Vorlage ``VARIANTS_PER_TEMPLATE`` Varianten.

    Der Zufall wird je (Vorlage, Variante) neu geseedet. Dadurch bleibt eine bestehende
    Variante gleich, auch wenn vorher eine Vorlage ergänzt wird – sonst änderten sich alle
    ``document_id`` und das Gold-Set wäre wertlos.
    """
    specs: list[DocumentSpec] = []
    for template in TEMPLATES:
        for variant in range(VARIANTS_PER_TEMPLATE):
            rnd = random.Random(f"{seed}:{template.template_id}:{variant}")
            titel, bloecke = template.build(rnd)
            specs.append(DocumentSpec(
                class_key=template.class_key,
                template_id=template.template_id,
                variant=variant,
                title=titel,
                blocks=bloecke,
                formats=template.formats,
            ))
    return specs
```

- [ ] **Schritt 4: Test laufen lassen, Erfolg bestätigen**

```bash
uv run pytest tests/test_generation_content.py -v
```

Erwartet: 6 passed. Bei `test_format_verraet_die_klasse_nicht` scheitert es, wenn ein Format
weniger als drei Klassen trägt — dann Vorlagen umverteilen, **nicht** den Test lockern.

- [ ] **Schritt 5: Prüfen und committen**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest
git add -A && git commit -m "Testdaten-Inhalte: 56 Vorlagen über 7 Klassen und 4 Formate"
```

---

## Aufgabe 5: Testdaten — Ausgabeformate

**Dateien:**
- Anlegen: `src/doccls/generation/writers.py`
- Anlegen: `tests/test_generation_writers.py`

**Schnittstellen:**
- Nutzt: `doccls.generation.content.DocumentSpec`, `Block`
- Liefert: `write_pdf(path, spec)`, `write_docx(path, spec)`, `write_xlsx(path, spec)`,
  `write_eml(path, spec, attachment: tuple[str, bytes] | None = None)`,
  `write(path_without_suffix: Path, spec: DocumentSpec, fmt: str,
  attachment: tuple[str, bytes] | None = None) -> Path`

**Der Punkt, an dem es schiefgeht:** DOCX und XLSX sind ZIP-Archive mit Zeitstempeln, PDFs
und E-Mails tragen Erzeugungsdaten und Zufallsgrenzen. Ohne Fixierung ändert sich der
SHA-256 bei jedem Lauf — und damit jede `document_id`. Das Referenzprojekt löst das mit
`normalize_zip`; dieselbe Lösung wird hier übernommen.

- [ ] **Schritt 1: Test schreiben**

`tests/test_generation_writers.py`:

```python
"""Erzeugte Dateien müssen byteweise reproduzierbar sein, sonst wandert jede document_id."""

import hashlib
from pathlib import Path

import pytest

from doccls.generation.content import build_corpus
from doccls.generation.writers import write

FORMATE = ["pdf", "docx", "xlsx", "eml"]


def _spec(fmt: str):
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
    pfad = write(tmp_path / "mail", _spec("eml"), "eml",
                 attachment=("rechnung.pdf", b"%PDF-1.7\nInhalt"))
    roh = pfad.read_bytes()
    assert b"rechnung.pdf" in roh
    assert b"multipart" in roh.lower()


def test_unbekanntes_format_wird_abgelehnt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unbekanntes Format"):
        write(tmp_path / "x", _spec("pdf"), "rtf")
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_generation_writers.py -v
```

Erwartet: `ImportError: cannot import name 'write'`.

- [ ] **Schritt 3: `src/doccls/generation/writers.py` schreiben**

```python
"""Ein ``DocumentSpec`` in PDF, DOCX, XLSX oder EML ausschreiben.

Alles Zeitabhängige wird fixiert: ZIP-Zeitstempel (DOCX, XLSX), PDF-Erzeugungsdatum,
Mail-Datum und MIME-Grenze. Sonst hätte dieselbe Datei bei jedem Lauf einen anderen
SHA-256 – und damit eine andere ``document_id``.
"""

import re
import zipfile
from datetime import UTC, datetime
from email.message import EmailMessage
from email.policy import SMTP
from pathlib import Path

import docx
import openpyxl
import pymupdf
from docx.document import Document as DocxDocument

from doccls.generation.content import Block, DocumentSpec

FIXED_TIMESTAMP = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
BOUNDARY = "===============doccls-fixed-boundary=="

PDF_CSS = """
* { font-family: sans-serif; font-size: 10pt; }
h1 { font-size: 15pt; margin-bottom: 4pt; }
h2 { font-size: 12pt; margin-top: 10pt; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #999999; padding: 3pt; text-align: left; }
th { background-color: #e6e6e6; }
"""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def block_html(block: Block) -> str:
    """Einen Block als HTML – die Zwischenform für die PDF-Ausgabe."""
    teile: list[str] = []
    if block.heading:
        teile.append(f"<h2>{_escape(block.heading)}</h2>")
    teile += [f"<p>{_escape(absatz)}</p>" for absatz in block.paragraphs]
    if block.table:
        kopf, zeilen = block.table
        kopf_html = "".join(f"<th>{_escape(z)}</th>" for z in kopf)
        leib = "".join(
            "<tr>" + "".join(f"<td>{_escape(z)}</td>" for z in zeile) + "</tr>"
            for zeile in zeilen
        )
        teile.append(f"<table><tr>{kopf_html}</tr>{leib}</table>")
    return "".join(teile)


def block_text(block: Block) -> str:
    """Einen Block als Klartext – für die Mail-Ausgabe."""
    zeilen: list[str] = []
    if block.heading:
        zeilen += [block.heading, "-" * len(block.heading)]
    zeilen += list(block.paragraphs)
    if block.table:
        kopf, tabelle = block.table
        zeilen.append(" | ".join(kopf))
        zeilen += [" | ".join(zeile) for zeile in tabelle]
    return "\n".join(zeilen)


def normalize_zip(path: Path) -> None:
    """ZIP-Zeitstempel und das Änderungsdatum in ``docProps/core.xml`` fixieren.

    Ohne das ergibt derselbe Inhalt bei jedem Lauf einen anderen SHA-256. Übernommen aus
    ``bauprojekt-ai-pipeline``.
    """
    with zipfile.ZipFile(path) as quelle:
        eintraege = [(info.filename, quelle.read(info)) for info in quelle.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as ziel:
        for name, daten in eintraege:
            if name == "docProps/core.xml":
                daten = re.sub(rb"(<dcterms:(?:modified|created)[^>]*>)[^<]*",
                               rb"\g<1>2026-03-01T12:00:00Z", daten)
            info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            ziel.writestr(info, daten)


def write_pdf(path: Path, spec: DocumentSpec) -> None:
    """Ein Block je Seite, damit Seitenzahlen als Fundstelle stabil bleiben."""
    doc = pymupdf.open()
    seiten = ["<h1>" + _escape(spec.title) + "</h1>" + block_html(spec.blocks[0])]
    seiten += [block_html(block) for block in spec.blocks[1:]]
    for nummer, html in enumerate(seiten, start=1):
        seite = doc.new_page(width=595, height=842)  # A4
        rest, _ = seite.insert_htmlbox(seite.rect + (56, 56, -56, -56), html,
                                       css=PDF_CSS, scale_low=1)
        if rest < 0:
            raise ValueError(f"{path.name}: Seite {nummer} passt nicht auf eine Seite")
    datum = FIXED_TIMESTAMP.strftime("D:%Y%m%d%H%M%S")
    doc.set_metadata({"title": spec.title, "author": "Synthetische Testdaten",
                      "creationDate": datum, "modDate": datum})
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path, garbage=4, deflate=True, no_new_id=True)
    doc.close()


def write_docx(path: Path, spec: DocumentSpec) -> None:
    dokument: DocxDocument = docx.Document()
    dokument.add_heading(spec.title, level=1)
    for block in spec.blocks:
        if block.heading:
            dokument.add_heading(block.heading, level=2)
        for absatz in block.paragraphs:
            dokument.add_paragraph(absatz)
        if block.table:
            kopf, zeilen = block.table
            tabelle = dokument.add_table(rows=1, cols=len(kopf))
            tabelle.style = "Table Grid"
            for zelle, text in zip(tabelle.rows[0].cells, kopf, strict=True):
                zelle.text = text
            for zeile in zeilen:
                for zelle, text in zip(tabelle.add_row().cells, zeile, strict=True):
                    zelle.text = text
    eigenschaften = dokument.core_properties
    eigenschaften.title = spec.title
    eigenschaften.author = "Synthetische Testdaten"
    eigenschaften.created = eigenschaften.modified = FIXED_TIMESTAMP.replace(tzinfo=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    dokument.save(str(path))
    normalize_zip(path)


def write_xlsx(path: Path, spec: DocumentSpec) -> None:
    """Ein Blatt „Dokument“ mit Fließtext, je Tabelle ein eigenes Blatt."""
    mappe = openpyxl.Workbook()
    blatt = mappe.active
    blatt.title = "Dokument"
    blatt.append([spec.title])
    for block in spec.blocks:
        if block.heading:
            blatt.append([block.heading])
        for absatz in block.paragraphs:
            blatt.append([absatz])
    for nummer, block in enumerate((b for b in spec.blocks if b.table), start=1):
        assert block.table is not None
        kopf, zeilen = block.table
        tabellenblatt = mappe.create_sheet(f"Positionen {nummer}"[:31])
        tabellenblatt.append(list(kopf))
        for zeile in zeilen:
            tabellenblatt.append(list(zeile))
    mappe.properties.title = spec.title
    mappe.properties.creator = "Synthetische Testdaten"
    mappe.properties.created = mappe.properties.modified = FIXED_TIMESTAMP.replace(tzinfo=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    mappe.save(str(path))
    normalize_zip(path)


def write_eml(path: Path, spec: DocumentSpec,
              attachment: tuple[str, bytes] | None = None) -> None:
    """Mail mit Klartextkörper, optional mit Anhang.

    Der Anhang wird später ein **eigenständiges Dokument** mit Elternbezug (Konzept § 5).
    Die MIME-Grenze wird fixiert, weil sie sonst zufällig ist.
    """
    nachricht = EmailMessage(policy=SMTP)
    nachricht["Subject"] = spec.title
    nachricht["From"] = "buchhaltung@muster-handels.example"
    nachricht["To"] = "eingang@kunde.example"
    nachricht["Date"] = "Sun, 01 Mar 2026 12:00:00 +0000"
    nachricht["Message-ID"] = f"<{spec.template_id}.{spec.variant}@muster-handels.example>"
    nachricht.set_content("\n\n".join(block_text(block) for block in spec.blocks))
    if attachment is not None:
        name, daten = attachment
        nachricht.add_attachment(daten, maintype="application", subtype="octet-stream",
                                 filename=name)
        nachricht.set_boundary(BOUNDARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(nachricht.as_bytes())


def write(path_without_suffix: Path, spec: DocumentSpec, fmt: str,
          attachment: tuple[str, bytes] | None = None) -> Path:
    """Ausgabeformat wählen und schreiben. Liefert den geschriebenen Pfad."""
    pfad = path_without_suffix.with_suffix(f".{fmt}")
    match fmt:
        case "pdf":
            write_pdf(pfad, spec)
        case "docx":
            write_docx(pfad, spec)
        case "xlsx":
            write_xlsx(pfad, spec)
        case "eml":
            write_eml(pfad, spec, attachment)
        case _:
            raise ValueError(f"Unbekanntes Format: {fmt!r}")
    return pfad
```

- [ ] **Schritt 4: Test laufen lassen, Erfolg bestätigen**

```bash
uv run pytest tests/test_generation_writers.py -v
```

Erwartet: 10 passed.

Schlägt die Reproduzierbarkeit bei einem Format fehl, ist eine Zeitquelle übersehen worden.
Zum Eingrenzen die beiden Dateien auspacken und vergleichen:

```bash
uv run python -c "
import zipfile, sys
for p in sys.argv[1:]:
    print(p, sorted((i.filename, i.date_time) for i in zipfile.ZipFile(p).infolist())[:3])
" /tmp/a/dok.docx /tmp/b/dok.docx
```

- [ ] **Schritt 5: Prüfen und committen**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest
git add -A && git commit -m "Testdaten-Ausgabe in PDF, DOCX, XLSX, EML – byteweise reproduzierbar"
```

---

## Aufgabe 6: Generator-CLI, Manifest und Gold-Set-Schnitt

**Dateien:**
- Anlegen: `scripts/generate_documents.py`
- Anlegen: `src/doccls/generation/manifest.py`
- Anlegen: `tests/test_generation_manifest.py`
- Anlegen: `docs/testdaten.md`

**Schnittstellen:**
- Nutzt: `build_corpus`, `write`, `doccls.config.RAW_DIR`, `GENERATED_DIR`
- Liefert: `Split` (StrEnum: `GOLD`, `TRAIN`, `CALIB`),
  `assign_splits(corpus: list[DocumentSpec], seed: int = 7) -> dict[str, Split]`
  (Schlüssel: `template_id`),
  `manifest_frame(corpus: list[DocumentSpec], splits: dict[str, Split],
  paths: list[Path], root: Path) -> pl.DataFrame`

**Der entscheidende Punkt:** Der Schnitt läuft über `template_id`, nie über einzelne
Dokumente. Zehn Varianten derselben Vorlage sind praktisch dasselbe Dokument; lägen sie auf
beiden Seiten, wäre das Gold-Set geschönt (Konzept § 9.2).

Aufteilung je Klasse: 5 Vorlagen → `GOLD` (50 Dokumente), 2 → `TRAIN`, 1 → `CALIB`.

- [ ] **Schritt 1: Test schreiben**

`tests/test_generation_manifest.py`:

```python
"""Der Gold-Set-Schnitt ist die Grundlage jeder späteren Messung. Er wird geprüft, nicht geglaubt."""

from collections import Counter

from doccls.classes import load_classes
from doccls.generation.content import build_corpus
from doccls.generation.manifest import Split, assign_splits


def test_schnitt_ist_deterministisch() -> None:
    assert assign_splits(build_corpus()) == assign_splits(build_corpus())


def test_keine_vorlage_liegt_auf_zwei_seiten() -> None:
    """Der Kern der Sache: Der Schnitt trennt Vorlagen, nicht Dokumente (Konzept § 9.2)."""
    korpus = build_corpus()
    splits = assign_splits(korpus)
    je_vorlage: dict[str, set[Split]] = {}
    for spec in korpus:
        je_vorlage.setdefault(spec.template_id, set()).add(splits[spec.template_id])
    assert all(len(s) == 1 for s in je_vorlage.values())


def test_jede_klasse_hat_mindestens_50_gold_dokumente() -> None:
    korpus = build_corpus()
    splits = assign_splits(korpus)
    gold = Counter(s.class_key for s in korpus if splits[s.template_id] is Split.GOLD)
    for klasse in load_classes().keys():
        assert gold[klasse] >= 50, f"{klasse}: nur {gold[klasse]} Gold-Dokumente"


def test_jede_klasse_kommt_in_jedem_split_vor() -> None:
    korpus = build_corpus()
    splits = assign_splits(korpus)
    for klasse in load_classes().keys():
        vorhanden = {splits[s.template_id] for s in korpus if s.class_key == klasse}
        assert vorhanden == set(Split), f"{klasse} fehlt in {set(Split) - vorhanden}"
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_generation_manifest.py -v
```

Erwartet: `ModuleNotFoundError: No module named 'doccls.generation.manifest'`.

- [ ] **Schritt 3: `src/doccls/generation/manifest.py` schreiben**

```python
"""Aufteilung in Gold-, Trainings- und Kalibriermenge – und das Manifest dazu.

Der Schnitt läuft über ``template_id``, nie über einzelne Dokumente: Zehn Varianten
derselben Vorlage sind praktisch dasselbe Dokument. Lägen sie auf beiden Seiten, wäre jede
gemessene Metrik geschönt (Konzept § 9.2, Aufteilung nach Dokumentfamilie).
"""

import random
from collections import defaultdict
from enum import StrEnum
from pathlib import Path

import polars as pl

from doccls.generation.content import DocumentSpec

GOLD_TEMPLATES_PER_CLASS = 5
CALIB_TEMPLATES_PER_CLASS = 1
"""Der Rest geht in die Trainingsmenge."""


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


def assign_splits(corpus: list[DocumentSpec], seed: int = 7) -> dict[str, Split]:
    """Vorlagen je Klasse auf die drei Mengen verteilen. Schlüssel ist ``template_id``."""
    je_klasse: dict[str, list[str]] = defaultdict(list)
    for spec in corpus:
        if spec.template_id not in je_klasse[spec.class_key]:
            je_klasse[spec.class_key].append(spec.template_id)

    splits: dict[str, Split] = {}
    for klasse, vorlagen in sorted(je_klasse.items()):
        gemischt = sorted(vorlagen)
        random.Random(f"{seed}:{klasse}").shuffle(gemischt)
        if len(gemischt) < GOLD_TEMPLATES_PER_CLASS + CALIB_TEMPLATES_PER_CLASS + 1:
            raise ValueError(
                f"Klasse {klasse} hat nur {len(gemischt)} Vorlagen; für den Schnitt werden "
                f"mindestens {GOLD_TEMPLATES_PER_CLASS + CALIB_TEMPLATES_PER_CLASS + 1} gebraucht."
            )
        for position, vorlage in enumerate(gemischt):
            if position < GOLD_TEMPLATES_PER_CLASS:
                splits[vorlage] = Split.GOLD
            elif position < GOLD_TEMPLATES_PER_CLASS + CALIB_TEMPLATES_PER_CLASS:
                splits[vorlage] = Split.CALIB
            else:
                splits[vorlage] = Split.TRAIN
    return splits


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
```

- [ ] **Schritt 4: Test laufen lassen, Erfolg bestätigen**

```bash
uv run pytest tests/test_generation_manifest.py -v
```

Erwartet: 4 passed.

- [ ] **Schritt 5: `scripts/generate_documents.py` schreiben**

```python
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
from doccls.generation.manifest import assign_splits, manifest_frame
from doccls.generation.writers import write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    korpus = build_corpus(seed=args.seed)
    splits = assign_splits(korpus)

    # Vorlagen, deren Name auf "-anhang" endet, werden Mails mit angehängter PDF. Der
    # Anhang wird beim Einlesen ein eigenständiges Dokument mit Elternbezug (Konzept § 5).
    # Vereinfachung: Mailkörper und Anhang tragen denselben Inhalt – geprüft wird der Weg,
    # nicht die Redaktion.
    pfade: list[Path] = []
    zwischenablage = args.out.parent / "tmp-anhaenge"
    for spec in korpus:
        fmt = spec.formats[0]
        name = f"{spec.template_id}-{spec.variant:02d}"
        basis = args.out / fmt / name
        if fmt == "eml" and spec.template_id.endswith("-anhang"):
            anhang = write(zwischenablage / name, spec, "pdf").read_bytes()
            pfade.append(write(basis, spec, "eml", attachment=(f"{name}.pdf", anhang)))
        else:
            pfade.append(write(basis, spec, fmt))
    shutil.rmtree(zwischenablage, ignore_errors=True)

    manifest = manifest_frame(korpus, splits, pfade, root=args.out)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    ziel = GENERATED_DIR / "manifest.parquet"
    manifest.write_parquet(ziel)

    anhaenge = sum(1 for s in korpus if s.template_id.endswith("-anhang"))
    print(f"{len(pfade)} Dateien geschrieben nach {args.out}, davon {anhaenge} Mails mit Anhang")
    print(manifest.group_by("class_key", "split").len().sort("class_key", "split"))
    print(f"Manifest: {ziel}")


if __name__ == "__main__":
    main()
```

- [ ] **Schritt 6: Generator ausführen und das Ergebnis ansehen**

```bash
uv run python scripts/generate_documents.py
```

Erwartet: `560 Dateien geschrieben nach .../data/raw, davon 10 Mails mit Anhang`, danach eine
Tabelle mit je Klasse 50 `gold`, 20 `train`, 10 `calib`.

Die 10 angehängten PDFs sind keine eigenen Dateien unter `data/raw/` — sie stecken in den
Mails und werden erst beim Einlesen zu eigenen Dokumenten (Aufgabe 11).

- [ ] **Schritt 7: Reproduzierbarkeit im Ganzen prüfen**

```bash
find data/raw -type f | sort | xargs shasum -a 256 | shasum -a 256
uv run python scripts/generate_documents.py
find data/raw -type f | sort | xargs shasum -a 256 | shasum -a 256
```

Erwartet: beide Male derselbe Hash. Weicht er ab, ist in Aufgabe 5 eine Zeitquelle
übersehen worden — dort beheben, nicht hier umgehen.

- [ ] **Schritt 8: `docs/testdaten.md` schreiben**

Diese Datei ist der Maßstab für alle späteren Auswertungen und muss bei jeder Änderung am
Generator mitgepflegt werden.

```markdown
# Testdaten: die bekannte Wahrheit

Erzeugt von `scripts/generate_documents.py` (Seed 42). 56 Vorlagen, je 10 Varianten,
560 Dokumente über 7 Klassen und 4 Formate.

## Aufteilung

Der Schnitt läuft über `template_id`, nicht über Dokumente (Konzept § 9.2). Je Klasse:

| Split | Vorlagen | Dokumente | Verwendung |
|---|---|---|---|
| `gold` | 5 | 50 | eingefroren; nie Training, nie Kalibrierung, nie in der Prüfliste |
| `train` | 2 | 20 | Trainingsvorrat |
| `calib` | 1 | 10 | Temperature Scaling, Schwellen τ und δ |

## Absichtlich eingebaute Schwierigkeiten

| Nr. | Sachverhalt | Wo | Wozu |
|---|---|---|---|
| 1 | Gutschriften in Rechnungsaufmachung, mit Bezug auf eine Rechnungsnummer | alle `GUTSCHRIFT-*` | der harte Fall aus Konzept Anhang D (Kosinus 0,964) |
| 2 | Rechnungen, die eine stornierte Rechnung ersetzen und das Wort „storniert“ tragen | `RECHNUNG-storno-*` | Gegenrichtung zu Nr. 1 |
| 3 | Verträge mit AGB-Anlage | `VERTRAG-mit-agb`, `-rahmen`, `-lizenz` | Vertrag ↔ AGB |
| 4 | Protokolle in Berichtsform, ohne TOP-Tabelle | `PROTOKOLL-bericht*` | Protokoll ↔ Statusbericht |
| 5 | Jede Klasse in mindestens zwei Formaten; jedes Format trägt mindestens drei Klassen | gesamter Korpus | verhindert die Abkürzung „Format ⇒ Klasse“ |
| 6 | Gemeinsamer Briefkopf und identische Fußzeile über alle Klassen | gesamter Korpus | verhindert die Abkürzung „Layout ⇒ Klasse“ |
| 7 | `SONSTIGES` bewusst heterogen (Anschreiben, Lieferschein, Werbung, Anleitung, Bescheinigung) | `SONST-*` | die Restklasse hat kein gemeinsames Merkmal (Konzept § 2) |

## Was noch fehlt

- Keine gescannten Dokumente — OCR ist Aufgabe 12 und braucht `ocrmypdf`.
- Keine englischen Dokumente. Anhang D zeigt, dass sie eigene Beispiele bräuchten.
- Keine echten Dokumente. Ein Modell, das hier glänzt und auf Realität einbricht, hat den
  Generator gelernt (Konzept § 9.2).
```

- [ ] **Schritt 9: Prüfen und committen**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest
git add -A && git commit -m "Generator-CLI, Manifest und Gold-Set-Schnitt über Vorlagen"
```

**Phase 0 ist damit erreicht:** 560 Dokumente mit bekannter Wahrheit, reproduzierbar,
mit dokumentierten Schwierigkeiten und einem Gold-Set von 350 Dokumenten.

---

# Phase 1 — Ingestion und Extraktion

## Aufgabe 7: Formaterkennung

**Dateien:**
- Anlegen: `src/doccls/detect.py`
- Anlegen: `tests/test_detect.py`

**Schnittstellen:**
- Liefert: `Format` (StrEnum: `PDF`, `DOCX`, `XLSX`, `PPTX`, `EML`, `MSG`, `HTML`, `CSV`,
  `IMAGE`, `UNKNOWN`), `detect_format(data: bytes, file_name: str) -> Format`,
  `MEDIA_TYPES: dict[Format, str]`

**Warum nicht die Endung:** Sie ist eine Behauptung. In echten Beständen ist die `.xls`,
die HTML enthält, und die `.pdf`, die ein TIFF ist, der Normalfall. Die Endung entscheidet
nur, wo die Magic Bytes nichts hergeben (EML, CSV).

- [ ] **Schritt 1: Test schreiben**

`tests/test_detect.py`:

```python
"""Die Dateiendung ist eine Behauptung. Erkannt wird am Inhalt."""

from pathlib import Path

import pytest

from doccls.detect import Format, detect_format

ECHTE_DATEIEN = {
    Format.PDF: "pdf", Format.DOCX: "docx", Format.XLSX: "xlsx", Format.EML: "eml",
}


@pytest.mark.parametrize("fmt,ordner", ECHTE_DATEIEN.items())
def test_erzeugte_dateien_werden_erkannt(fmt: Format, ordner: str) -> None:
    from doccls.config import RAW_DIR

    dateien = sorted((RAW_DIR / ordner).glob(f"*.{ordner}"))
    assert dateien, f"Keine Testdaten in {RAW_DIR / ordner} – erst den Generator laufen lassen"
    datei = dateien[0]
    assert detect_format(datei.read_bytes(), datei.name) is fmt


def test_falsche_endung_schlaegt_nicht_durch(tmp_path: Path) -> None:
    """Eine als .xlsx benannte PDF ist eine PDF."""
    assert detect_format(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", "quartal.xlsx") is Format.PDF


def test_ooxml_wird_am_zip_inhalt_unterschieden() -> None:
    """DOCX, XLSX und PPTX haben dieselben Magic Bytes (PK). Nur der Inhalt trennt sie."""
    import io
    import zipfile

    def zip_mit(eintrag: str) -> bytes:
        puffer = io.BytesIO()
        with zipfile.ZipFile(puffer, "w") as z:
            z.writestr("[Content_Types].xml", "<Types/>")
            z.writestr(eintrag, "x")
        return puffer.getvalue()

    assert detect_format(zip_mit("word/document.xml"), "a.bin") is Format.DOCX
    assert detect_format(zip_mit("xl/workbook.xml"), "a.bin") is Format.XLSX
    assert detect_format(zip_mit("ppt/presentation.xml"), "a.bin") is Format.PPTX


def test_eml_wird_an_den_kopfzeilen_erkannt() -> None:
    roh = b"From: a@b.example\r\nTo: c@d.example\r\nSubject: Test\r\n\r\nHallo"
    assert detect_format(roh, "nachricht.eml") is Format.EML
    assert detect_format(roh, "ohne_endung") is Format.EML


def test_unbekanntes_bleibt_unbekannt() -> None:
    assert detect_format(b"\x00\x01\x02\x03 irgendwas", "x.dat") is Format.UNKNOWN
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_detect.py -v
```

Erwartet: `ModuleNotFoundError: No module named 'doccls.detect'`.

- [ ] **Schritt 3: `src/doccls/detect.py` schreiben**

```python
"""Format am Inhalt erkennen, nicht an der Endung.

Die Endung ist eine Behauptung des Erzeugers. In echten Beständen sind die ``.xls``, die
HTML enthält, und die ``.pdf``, die ein Scan ist, der Normalfall. Die Endung entscheidet
nur dort, wo es keine Magic Bytes gibt – bei E-Mails und CSV.
"""

import io
import re
import zipfile
from enum import StrEnum


class Format(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    EML = "eml"
    MSG = "msg"
    HTML = "html"
    CSV = "csv"
    IMAGE = "image"
    UNKNOWN = "unknown"


MEDIA_TYPES: dict[Format, str] = {
    Format.PDF: "application/pdf",
    Format.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    Format.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    Format.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    Format.EML: "message/rfc822",
    Format.MSG: "application/vnd.ms-outlook",
    Format.HTML: "text/html",
    Format.CSV: "text/csv",
    Format.IMAGE: "image/*",
    Format.UNKNOWN: "application/octet-stream",
}

OOXML_MARKER: tuple[tuple[str, Format], ...] = (
    ("word/document.xml", Format.DOCX),
    ("xl/workbook.xml", Format.XLSX),
    ("ppt/presentation.xml", Format.PPTX),
)
"""DOCX, XLSX und PPTX sind alle ZIP-Archive. Nur ihr Inhalt unterscheidet sie."""

IMAGE_MAGIC: tuple[bytes, ...] = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"II*\x00", b"MM\x00*")
OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

MAIL_HEADER = re.compile(
    rb"^(From|To|Subject|Date|Message-ID|Received|Return-Path|MIME-Version):", re.IGNORECASE
)
"""E-Mails haben keine Magic Bytes. Erkannt wird an einer Kopfzeile am Anfang."""


def _ooxml_kind(data: bytes) -> Format:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archiv:
            namen = set(archiv.namelist())
    except zipfile.BadZipFile:
        return Format.UNKNOWN
    for marker, format_ in OOXML_MARKER:
        if marker in namen:
            return format_
    return Format.UNKNOWN


def _sieht_aus_wie_mail(data: bytes) -> bool:
    kopf = data[:2048].lstrip()
    return any(MAIL_HEADER.match(zeile) for zeile in kopf.splitlines()[:12])


def detect_format(data: bytes, file_name: str) -> Format:
    """Format bestimmen. Magic Bytes zuerst, die Endung nur als letzte Auskunft."""
    endung = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if data.startswith(b"%PDF-"):
        return Format.PDF
    if data.startswith(b"PK\x03\x04"):
        art = _ooxml_kind(data)
        if art is not Format.UNKNOWN:
            return art
    if data.startswith(OLE2_MAGIC):
        return Format.MSG
    if any(data.startswith(magic) for magic in IMAGE_MAGIC):
        return Format.IMAGE
    if data[:1024].lstrip()[:15].lower().startswith((b"<!doctype html", b"<html")):
        return Format.HTML
    if _sieht_aus_wie_mail(data):
        return Format.EML

    # Ohne Magic Bytes entscheidet die Endung.
    match endung:
        case "eml":
            return Format.EML
        case "csv" | "tsv":
            return Format.CSV
        case "htm" | "html":
            return Format.HTML
        case _:
            return Format.UNKNOWN
```

- [ ] **Schritt 4: Test laufen lassen, Erfolg bestätigen**

```bash
uv run python scripts/generate_documents.py   # falls data/raw noch leer ist
uv run pytest tests/test_detect.py -v
```

Erwartet: 8 passed.

- [ ] **Schritt 5: Prüfen und committen**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest
git add -A && git commit -m "Formaterkennung an Magic Bytes statt an der Dateiendung"
```

---

## Aufgabe 8: Extraktion PDF, DOCX, XLSX

**Dateien:**
- Anlegen: `src/doccls/extraction/__init__.py`, `src/doccls/extraction/pdf.py`,
  `src/doccls/extraction/office.py`
- Anlegen: `tests/test_extraction_office.py`

**Schnittstellen:**
- Nutzt: `Document`, `Segment`, `SegmentKind`, `normalize_text`, `Format`
- Liefert: `extract_pdf(document, data) -> list[Segment]`,
  `extract_docx(document, data) -> list[Segment]`,
  `extract_xlsx(document, data) -> list[Segment]`

**Reine Funktionen:** Sie bekommen Bytes und greifen nie auf das Dateisystem zu. Dadurch
sind sie ohne Dateien auf der Platte prüfbar, und der Dateizugriff bleibt in `pipeline.py`
an einer Stelle — die spätere Umstellung auf Objektspeicher berührt nur dieses eine Modul.

- [ ] **Schritt 1: Test schreiben**

`tests/test_extraction_office.py`:

```python
"""Extraktion arbeitet auf Bytes, nicht auf Pfaden – prüfbar ohne Dateien auf der Platte."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from doccls.config import RAW_DIR
from doccls.extraction import extract
from doccls.models import Document, SegmentKind


def dokument(pfad: Path, media_type: str) -> tuple[Document, bytes]:
    daten = pfad.read_bytes()
    return Document.create(source_path=pfad.name, media_type=media_type, content=daten,
                           ingested_at=datetime(2026, 1, 1, tzinfo=UTC)), daten


def erste(ordner: str) -> Path:
    dateien = sorted((RAW_DIR / ordner).glob(f"*.{ordner}"))
    assert dateien, f"Keine Testdaten in {RAW_DIR / ordner}"
    return dateien[0]


def test_pdf_ergibt_ein_segment_je_seite() -> None:
    doc, daten = dokument(erste("pdf"), "application/pdf")
    segmente = extract(doc, daten)
    assert segmente
    assert all(s.kind is SegmentKind.SEITE for s in segmente)
    assert [s.index for s in segmente] == list(range(len(segmente)))
    assert segmente[0].locator == "S. 1"


def test_docx_liefert_abschnitte_mit_ueberschriften() -> None:
    doc, daten = dokument(erste("docx"), "application/vnd.openxmlformats-officedocument."
                                          "wordprocessingml.document")
    segmente = extract(doc, daten)
    assert any(s.heading for s in segmente)
    assert all(s.kind is SegmentKind.ABSCHNITT for s in segmente)


def test_xlsx_liefert_ein_segment_je_blatt() -> None:
    doc, daten = dokument(erste("xlsx"), "application/vnd.openxmlformats-officedocument."
                                          "spreadsheetml.sheet")
    segmente = extract(doc, daten)
    assert segmente
    assert all(s.kind is SegmentKind.BLATT for s in segmente)


def test_tabellenzeile_bleibt_beisammen() -> None:
    """Zerfiele eine Zeile in Zellen, verlöre man den Bezug zwischen Position und Betrag."""
    doc, daten = dokument(erste("xlsx"), "application/vnd.openxmlformats-officedocument."
                                          "spreadsheetml.sheet")
    text = " ".join(s.text for s in extract(doc, daten))
    assert "EUR" in text


def test_text_ist_normalisiert() -> None:
    doc, daten = dokument(erste("pdf"), "application/pdf")
    for segment in extract(doc, daten):
        assert "  " not in segment.text
        assert "\n" not in segment.text


def test_segment_ids_sind_eindeutig() -> None:
    doc, daten = dokument(erste("pdf"), "application/pdf")
    segmente = extract(doc, daten)
    assert len({s.segment_id for s in segmente}) == len(segmente)


def test_unbekanntes_format_wird_abgelehnt() -> None:
    doc, _ = dokument(erste("pdf"), "application/pdf")
    with pytest.raises(ValueError, match="Nicht unterstützt"):
        extract(doc, b"\x00\x01\x02 irgendwas")
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_extraction_office.py -v
```

Erwartet: `ModuleNotFoundError: No module named 'doccls.extraction'`.

- [ ] **Schritt 3: `src/doccls/extraction/pdf.py` schreiben**

```python
"""PDF: eine Seite wird ein Segment.

Die Seite bleibt die Einheit, weil die Seitenzahl die einzige Fundstelle ist, die ein
Mensch im Original wiederfindet. Die Überschrift wird an der Schriftgröße erkannt – der
größte Text auf der Seite, sofern er deutlich größer ist als der Fließtext.
"""

import pymupdf

from doccls.models import Document, Segment, SegmentKind, normalize_text

MAX_HEADING_LENGTH = 80
"""Längerer Text ist Fließtext, keine Überschrift."""

MIN_HEADING_SIZE_RATIO = 1.15
"""So viel größer als der Fließtext muss eine Zeile sein, um als Überschrift zu gelten."""


def heading_by_font_size(page: pymupdf.Page) -> str | None:
    """Die größte Textzeile der Seite, wenn sie deutlich größer ist als der Rest."""
    zeilen: list[tuple[float, str]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            groessen = [span["size"] for span in line["spans"]]
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text and groessen:
                zeilen.append((max(groessen), text))
    if not zeilen:
        return None
    groesse, text = max(zeilen, key=lambda eintrag: eintrag[0])
    fliesstext = sorted(g for g, _ in zeilen)[len(zeilen) // 2]
    if groesse < fliesstext * MIN_HEADING_SIZE_RATIO or len(text) > MAX_HEADING_LENGTH:
        return None
    return normalize_text(text)


def extract_pdf(document: Document, data: bytes) -> list[Segment]:
    segmente: list[Segment] = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for seitennummer, seite in enumerate(pdf, start=1):
            text = normalize_text(seite.get_text("text"))
            if not text:
                continue  # Leere oder rein grafische Seiten übergehen; needs_ocr fängt sie.
            segmente.append(Segment.create(
                document=document, index=len(segmente), kind=SegmentKind.SEITE,
                locator=f"S. {seitennummer}", heading=heading_by_font_size(seite), text=text,
            ))
    return segmente
```

- [ ] **Schritt 4: `src/doccls/extraction/office.py` schreiben**

```python
"""DOCX und XLSX.

DOCX: Ein Segment umfasst eine Überschrift und alles bis zur nächsten. Das hält
zusammenhängende Gedanken beieinander und gibt dem Segment eine sprechende Fundstelle.

XLSX: Ein Blatt wird ein Segment. Eine Tabellenzeile wird als ``Spalte: Wert``
geschrieben – zerfiele sie in einzelne Zellen, verlöre man den Bezug zwischen Position,
Menge und Betrag, und genau dieser Bezug unterscheidet eine Rechnung von einer Liste.
"""

import io

import docx
import openpyxl

from doccls.models import Document, Segment, SegmentKind, normalize_text

MIN_HEADER_CELLS = 2
"""Ab so vielen gefüllten Zellen gilt die erste Zeile eines Blattes als Kopfzeile."""


def extract_docx(document: Document, data: bytes) -> list[Segment]:
    quelle = docx.Document(io.BytesIO(data))
    segmente: list[Segment] = []
    ueberschrift: str | None = None
    absaetze: list[str] = []

    def abschliessen() -> None:
        text = normalize_text(" ".join(absaetze))
        if text:
            segmente.append(Segment.create(
                document=document, index=len(segmente), kind=SegmentKind.ABSCHNITT,
                locator=ueberschrift or f"Abschnitt {len(segmente) + 1}",
                heading=ueberschrift, text=text,
            ))

    for absatz in quelle.paragraphs:
        text = absatz.text.strip()
        if not text:
            continue
        if absatz.style.name.startswith("Heading"):
            abschliessen()
            ueberschrift, absaetze = normalize_text(text), []
        else:
            absaetze.append(text)
    abschliessen()

    for nummer, tabelle in enumerate(quelle.tables, start=1):
        zeilen = [" | ".join(zelle.text.strip() for zelle in zeile.cells)
                  for zeile in tabelle.rows]
        text = normalize_text(" ".join(zeilen))
        if text:
            segmente.append(Segment.create(
                document=document, index=len(segmente), kind=SegmentKind.ABSCHNITT,
                locator=f"Tabelle {nummer}", heading=None, text=text,
            ))
    return segmente


def extract_xlsx(document: Document, data: bytes) -> list[Segment]:
    mappe = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    segmente: list[Segment] = []
    for blatt in mappe.worksheets:
        zeilen = [[("" if zelle is None else str(zelle)).strip() for zelle in zeile]
                  for zeile in blatt.iter_rows(values_only=True)]
        zeilen = [zeile for zeile in zeilen if any(zeile)]
        if not zeilen:
            continue
        kopf = zeilen[0] if sum(bool(z) for z in zeilen[0]) >= MIN_HEADER_CELLS else None
        rest = zeilen[1:] if kopf else zeilen
        if kopf:
            formatiert = [
                ", ".join(f"{name}: {wert}" for name, wert in zip(kopf, zeile, strict=False) if wert)
                for zeile in rest
            ]
            inhalt = " | ".join([" | ".join(kopf), *formatiert])
        else:
            inhalt = " | ".join(" ".join(z for z in zeile if z) for zeile in rest)
        text = normalize_text(inhalt)
        if text:
            segmente.append(Segment.create(
                document=document, index=len(segmente), kind=SegmentKind.BLATT,
                locator=f"Blatt {blatt.title}", heading=blatt.title, text=text,
            ))
    mappe.close()
    return segmente
```

- [ ] **Schritt 5: `src/doccls/extraction/__init__.py` schreiben**

```python
"""Extraktion: Bytes in Segmente. Neue Formate kommen in ``extract`` als weiterer Fall dazu."""

from doccls.detect import Format, detect_format
from doccls.extraction.office import extract_docx, extract_xlsx
from doccls.extraction.pdf import extract_pdf
from doccls.models import Document, Segment

__all__ = ["extract", "extract_docx", "extract_pdf", "extract_xlsx"]


def extract(document: Document, data: bytes) -> list[Segment]:
    """Segmente aus dem Dateiinhalt. Das Format wird am Inhalt erkannt, nicht an der Endung."""
    match detect_format(data, document.file_name):
        case Format.PDF:
            return extract_pdf(document, data)
        case Format.DOCX:
            return extract_docx(document, data)
        case Format.XLSX:
            return extract_xlsx(document, data)
        case unbekannt:
            raise ValueError(
                f"Nicht unterstütztes Format {unbekannt} für {document.file_name!r}"
            )
```

- [ ] **Schritt 6: Test laufen lassen, Erfolg bestätigen**

```bash
uv run pytest tests/test_extraction_office.py -v
```

Erwartet: 7 passed.

- [ ] **Schritt 7: Prüfen und committen**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest
git add -A && git commit -m "Extraktion für PDF, DOCX und XLSX als reine Funktionen"
```

---

## Aufgabe 9: Extraktion E-Mail mit Anhängen

**Dateien:**
- Anlegen: `src/doccls/extraction/mail.py`
- Ändern: `src/doccls/extraction/__init__.py` (EML-Fall ergänzen)
- Anlegen: `tests/test_extraction_mail.py`

**Schnittstellen:**
- Liefert: `Attachment` (`file_name: str`, `content: bytes`),
  `extract_eml(document, data) -> tuple[list[Segment], list[Attachment]]`
- Ändert: `extract(document, data)` behandelt EML, indem es nur die Segmente zurückgibt;
  Anhänge holt sich `pipeline.py` über `extract_eml` direkt.

**Der Kern:** Eine Mail mit angehängter Rechnung sind **zwei Dokumente mit verschiedenen
Klassen** (Konzept § 5). Die Mail wird nie durch ihren Anhang klassifiziert und umgekehrt.
Absenderdomain und Betreff werden als Text mitgeführt, damit sie später Strukturmerkmale
werden können.

- [ ] **Schritt 1: Test schreiben**

`tests/test_extraction_mail.py`:

```python
"""Eine Mail ist ein Baum: Körper und jeder Anhang sind eigene Dokumente (Konzept § 5)."""

from datetime import UTC, datetime
from email.message import EmailMessage
from email.policy import SMTP

from doccls.extraction.mail import extract_eml
from doccls.models import Document, SegmentKind


def mail_bytes(*, mit_anhang: bool) -> bytes:
    nachricht = EmailMessage(policy=SMTP)
    nachricht["Subject"] = "Ihre Rechnung RE-2026-4711"
    nachricht["From"] = "buchhaltung@lieferant.example"
    nachricht["To"] = "eingang@kunde.example"
    nachricht["Date"] = "Sun, 01 Mar 2026 12:00:00 +0000"
    nachricht.set_content("Guten Tag,\n\nanbei die Rechnung.\n\nMit freundlichen Grüßen")
    if mit_anhang:
        nachricht.add_attachment(b"%PDF-1.7\nRechnungsinhalt", maintype="application",
                                 subtype="pdf", filename="RE-2026-4711.pdf")
    return nachricht.as_bytes()


def dokument(daten: bytes) -> Document:
    return Document.create(source_path="post/a.eml", media_type="message/rfc822",
                           content=daten, ingested_at=datetime(2026, 1, 1, tzinfo=UTC))


def test_kopf_und_koerper_werden_getrennte_segmente() -> None:
    daten = mail_bytes(mit_anhang=False)
    segmente, anhaenge = extract_eml(dokument(daten), daten)
    arten = [s.kind for s in segmente]
    assert SegmentKind.MAIL_KOPF in arten
    assert SegmentKind.MAIL_KOERPER in arten
    assert anhaenge == []


def test_kopf_enthaelt_betreff_und_absender() -> None:
    """Beide werden später Strukturmerkmale: Mails von rechnung@… tragen oft Rechnungen."""
    daten = mail_bytes(mit_anhang=False)
    segmente, _ = extract_eml(dokument(daten), daten)
    kopf = next(s for s in segmente if s.kind is SegmentKind.MAIL_KOPF)
    assert "RE-2026-4711" in kopf.text
    assert "lieferant.example" in kopf.text


def test_anhang_wird_herausgereicht_und_nicht_in_den_mailtext_gemischt() -> None:
    daten = mail_bytes(mit_anhang=True)
    segmente, anhaenge = extract_eml(dokument(daten), daten)
    assert [a.file_name for a in anhaenge] == ["RE-2026-4711.pdf"]
    assert anhaenge[0].content.startswith(b"%PDF")
    assert "Rechnungsinhalt" not in " ".join(s.text for s in segmente)


def test_mail_ohne_körper_liefert_wenigstens_den_kopf() -> None:
    nachricht = EmailMessage(policy=SMTP)
    nachricht["Subject"] = "Ohne Text"
    nachricht["From"] = "a@b.example"
    nachricht["To"] = "c@d.example"
    daten = nachricht.as_bytes()
    segmente, _ = extract_eml(dokument(daten), daten)
    assert any(s.kind is SegmentKind.MAIL_KOPF for s in segmente)
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_extraction_mail.py -v
```

Erwartet: `ModuleNotFoundError: No module named 'doccls.extraction.mail'`.

- [ ] **Schritt 3: `src/doccls/extraction/mail.py` schreiben**

```python
"""E-Mail: Kopf und Körper werden Segmente, jeder Anhang ein eigenes Dokument.

Eine Mail mit angehängter Rechnung besteht aus zwei Dokumenten mit verschiedenen Klassen
(Konzept § 5). Die Mail wird nie durch ihren Anhang klassifiziert und umgekehrt. Der
Anhang wird deshalb hier **nicht** in den Mailtext gemischt, sondern herausgereicht;
``pipeline.py`` macht daraus ein Dokument mit ``parent_document_id``.

Betreff und Absenderdomain stehen im Kopfsegment, damit sie später Strukturmerkmale werden
können – Mails von ``rechnung@…`` tragen überdurchschnittlich oft Rechnungen.
"""

from dataclasses import dataclass
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser

from doccls.models import Document, Segment, SegmentKind, normalize_text

HEADER_FIELDS = ("From", "To", "Cc", "Subject", "Date")


@dataclass(frozen=True)
class Attachment:
    """Ein Anhang, bereit als eigenständiges Dokument eingelesen zu werden."""

    file_name: str
    content: bytes


def extract_eml(document: Document, data: bytes) -> tuple[list[Segment], list[Attachment]]:
    nachricht = BytesParser(policy=policy.default).parsebytes(data)
    assert isinstance(nachricht, EmailMessage)

    segmente: list[Segment] = []
    kopf = normalize_text(
        " ".join(f"{feld}: {nachricht[feld]}" for feld in HEADER_FIELDS if nachricht[feld])
    )
    if kopf:
        segmente.append(Segment.create(
            document=document, index=0, kind=SegmentKind.MAIL_KOPF, locator="Kopf",
            heading=normalize_text(str(nachricht["Subject"] or "")) or None, text=kopf,
        ))

    koerper = nachricht.get_body(preferencelist=("plain", "html"))
    if koerper is not None:
        text = normalize_text(koerper.get_content())
        if text:
            segmente.append(Segment.create(
                document=document, index=len(segmente), kind=SegmentKind.MAIL_KOERPER,
                locator="Körper", heading=None, text=text,
            ))

    anhaenge = [
        Attachment(file_name=teil.get_filename() or "anhang.bin",
                   content=teil.get_payload(decode=True) or b"")
        for teil in nachricht.iter_attachments()
        if isinstance(teil, EmailMessage)
    ]
    return segmente, [a for a in anhaenge if a.content]
```

- [ ] **Schritt 4: `src/doccls/extraction/__init__.py` um den EML-Fall ergänzen**

Die beiden geänderten Stellen — Import und ein weiterer `case`:

```python
from doccls.extraction.mail import Attachment, extract_eml

__all__ = ["Attachment", "extract", "extract_docx", "extract_eml", "extract_pdf", "extract_xlsx"]
```

und im `match` vor `case unbekannt:`:

```python
        case Format.EML:
            segmente, _ = extract_eml(document, data)
            return segmente
```

- [ ] **Schritt 5: Test laufen lassen, Erfolg bestätigen**

```bash
uv run pytest tests/test_extraction_mail.py tests/test_extraction_office.py -v
```

Erwartet: 11 passed.

- [ ] **Schritt 6: Prüfen und committen**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest
git add -A && git commit -m "E-Mail-Extraktion: Anhänge werden eigenständige Dokumente"
```

---

## Aufgabe 10: Kopf- und Fußzeilen entfernen

**Dateien:**
- Anlegen: `src/doccls/normalize.py`
- Anlegen: `tests/test_normalize.py`

**Schnittstellen:**
- Liefert: `strip_boilerplate(texts: list[str], min_share: float = 0.6) -> list[str]`

**Warum das eine eigene Aufgabe ist:** In echten Beständen wiederholt sich die Fußzeile der
Buchhaltungssoftware auf jeder Seite. Sie ist je Absender verschieden — bliebe sie stehen,
lernte das Modell den Absender statt den Inhalt (Konzept § 5, § 8.5 Abkürzungen).

Der synthetische Korpus setzt die Fußzeile bewusst **nur einmal** ans Dokumentende und
über alle Klassen identisch. Diese Funktion greift dort also kaum — sie wird gebaut und
geprüft, weil sie für echte Dokumente gebraucht wird, nicht weil der Generator sie fordert.
Das ist ein Unterschied, den man kennen muss: Auf synthetischen Daten lässt sich ihre
Wirkung nicht messen.

- [ ] **Schritt 1: Test schreiben**

`tests/test_normalize.py`:

```python
"""Wiederkehrende Zeilen sind Layout, nicht Inhalt."""

from doccls.normalize import strip_boilerplate


def test_zeile_auf_allen_seiten_wird_entfernt() -> None:
    seiten = [f"Seite {i} Inhalt hier. Muster GmbH · HRB 44821" for i in range(1, 6)]
    bereinigt = strip_boilerplate(seiten)
    assert all("HRB 44821" not in seite for seite in bereinigt)
    assert all("Inhalt hier" in seite for seite in bereinigt)


def test_satz_auf_der_minderheit_der_seiten_bleibt_stehen() -> None:
    """Zwei von fünf Seiten liegen unter der Schwelle von 0,6 – das ist Inhalt, kein Layout."""
    wiederholt = "Dieser Hinweis steht nur auf zwei Seiten"
    seiten = [f"{wiederholt}. Eigener Inhalt Nummer {i} an dieser Stelle" if i < 2
              else f"Eigener Inhalt Nummer {i} an dieser Stelle" for i in range(5)]
    bereinigt = strip_boilerplate(seiten)
    assert wiederholt in " ".join(bereinigt)


def test_einzelne_seite_wird_nicht_leergeraeumt() -> None:
    """Bei einer Seite ist jede Zeile auf 100 % der Seiten – ohne Schutz bliebe nichts übrig."""
    assert strip_boilerplate(["Nur eine Seite mit Text."]) == ["Nur eine Seite mit Text."]


def test_leere_eingabe_bleibt_leer() -> None:
    assert strip_boilerplate([]) == []
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_normalize.py -v
```

Erwartet: `ModuleNotFoundError: No module named 'doccls.normalize'`.

- [ ] **Schritt 3: `src/doccls/normalize.py` schreiben**

```python
"""Wiederkehrende Kopf- und Fußzeilen entfernen.

Eine Zeile, die auf der Mehrzahl der Seiten eines Dokuments fast gleich auftaucht, ist
Layout und kein Inhalt. Bliebe sie stehen, wäre die Fußzeile der Buchhaltungssoftware das
häufigste Muster im ganzen Korpus – und das Modell lernte den Absender statt des Inhalts
(Konzept § 5, § 8.5).

Verglichen wird satzweise, weil die Extraktion Zeilenumbrüche bereits aufgelöst hat.
"""

from collections import Counter

MIN_PAGES_FOR_DETECTION = 3
"""Unter drei Seiten ist jede Zeile „häufig“. Dann wird nichts entfernt."""

MIN_SENTENCE_LENGTH = 12
"""Kürzere Bruchstücke sind zu unspezifisch, um sie als Layout zu verwerfen."""


def _saetze(text: str) -> list[str]:
    return [teil.strip() for teil in text.split(". ") if teil.strip()]


def strip_boilerplate(texts: list[str], min_share: float = 0.6) -> list[str]:
    """Sätze entfernen, die auf mindestens ``min_share`` der Seiten vorkommen."""
    if len(texts) < MIN_PAGES_FOR_DETECTION:
        return list(texts)

    haeufigkeit: Counter[str] = Counter()
    for text in texts:
        haeufigkeit.update({satz for satz in _saetze(text) if len(satz) >= MIN_SENTENCE_LENGTH})

    schwelle = min_share * len(texts)
    layout = {satz for satz, anzahl in haeufigkeit.items() if anzahl >= schwelle}
    if not layout:
        return list(texts)

    return [
        ". ".join(satz for satz in _saetze(text) if satz not in layout).strip()
        for text in texts
    ]
```

- [ ] **Schritt 4: Test laufen lassen, Erfolg bestätigen**

```bash
uv run pytest tests/test_normalize.py -v
```

Erwartet: 4 passed.

- [ ] **Schritt 5: Prüfen und committen**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest
git add -A && git commit -m "Wiederkehrende Kopf- und Fußzeilen als Layout entfernen"
```

---

## Aufgabe 11: Pipeline und CLI

**Dateien:**
- Anlegen: `src/doccls/pipeline.py`
- Anlegen: `scripts/ingest.py`
- Anlegen: `tests/test_pipeline.py`

**Schnittstellen:**
- Nutzt: alles Bisherige
- Liefert: `IngestResult` (`documents: int`, `skipped: int`, `segments: int`,
  `attachments: int`, `failed: list[str]`),
  `ingest(raw_dir: Path, parquet_dir: Path) -> IngestResult`,
  `read_table(parquet_dir: Path, name: str) -> pl.DataFrame`

**Verantwortung:** Dies ist die **einzige** Stelle mit Dateisystemzugriff. Wird später auf
MinIO umgestellt, ändert sich nur dieses Modul.

- [ ] **Schritt 1: Test schreiben**

`tests/test_pipeline.py`:

```python
"""Die Pipeline muss inkrementell sein und Anhänge als eigene Dokumente führen."""

from pathlib import Path

import polars as pl
import pytest

from doccls.generation.content import build_corpus
from doccls.generation.writers import write
from doccls.pipeline import ingest, read_table


@pytest.fixture
def kleiner_bestand(tmp_path: Path) -> Path:
    """Vier Dokumente, je eines pro Format, plus eine Mail mit PDF-Anhang."""
    roh = tmp_path / "raw"
    korpus = build_corpus()
    for fmt in ("pdf", "docx", "xlsx"):
        spec = next(s for s in korpus if fmt in s.formats)
        write(roh / fmt / "dok", spec, fmt)
    mail_spec = next(s for s in korpus if "eml" in s.formats)
    pdf_spec = next(s for s in korpus if "pdf" in s.formats)
    anhang = write(tmp_path / "tmp" / "anhang", pdf_spec, "pdf").read_bytes()
    write(roh / "eml" / "mit-anhang", mail_spec, "eml",
          attachment=("rechnung.pdf", anhang))
    return roh


def test_alle_dokumente_werden_eingelesen(kleiner_bestand: Path, tmp_path: Path) -> None:
    ergebnis = ingest(kleiner_bestand, tmp_path / "parquet")
    assert ergebnis.documents == 5  # 4 Dateien + 1 Anhang
    assert ergebnis.attachments == 1
    assert ergebnis.segments > 0
    assert ergebnis.failed == []


def test_zweiter_lauf_ueberspringt_alles(kleiner_bestand: Path, tmp_path: Path) -> None:
    """Inkrementell über den Inhalts-Hash – sonst verdoppelt jeder Lauf den Bestand."""
    ziel = tmp_path / "parquet"
    ingest(kleiner_bestand, ziel)
    zweiter = ingest(kleiner_bestand, ziel)
    assert zweiter.documents == 0
    assert zweiter.skipped == 4, (
        "Nur die vier Dateien werden erneut angefasst. Die übersprungene Mail wird gar nicht "
        "erst geöffnet, deshalb wird ihr Anhang nicht noch einmal besucht – genau so soll es sein."
    )
    assert read_table(ziel, "documents").height == 5


def test_geaenderte_datei_ergibt_eine_zusaetzliche_version(
    kleiner_bestand: Path, tmp_path: Path
) -> None:
    ziel = tmp_path / "parquet"
    ingest(kleiner_bestand, ziel)
    datei = next((kleiner_bestand / "pdf").glob("*.pdf"))
    spec = next(s for s in build_corpus() if "pdf" in s.formats and s.variant == 3)
    write(datei.with_suffix(""), spec, "pdf")
    ingest(kleiner_bestand, ziel)
    dokumente = read_table(ziel, "documents")
    assert dokumente.height == 6
    assert dokumente.filter(pl.col("source_path") == f"pdf/{datei.stem}.pdf").height == 2


def test_anhang_traegt_den_elternbezug(kleiner_bestand: Path, tmp_path: Path) -> None:
    """Eine Mail und ihre Rechnung sind zwei Dokumente (Konzept § 5)."""
    ingest(kleiner_bestand, tmp_path / "parquet")
    dokumente = read_table(tmp_path / "parquet", "documents")
    kinder = dokumente.filter(pl.col("parent_document_id").is_not_null())
    assert kinder.height == 1
    assert kinder["source_path"][0].endswith("!rechnung.pdf")
    eltern = kinder["parent_document_id"][0]
    assert eltern in set(dokumente["document_id"])


def test_segmente_verweisen_nur_auf_bekannte_dokumente(
    kleiner_bestand: Path, tmp_path: Path
) -> None:
    ingest(kleiner_bestand, tmp_path / "parquet")
    dokumente = read_table(tmp_path / "parquet", "documents")
    segmente = read_table(tmp_path / "parquet", "segments")
    assert set(segmente["document_id"]) <= set(dokumente["document_id"])


def test_unlesbare_datei_stoppt_den_lauf_nicht(kleiner_bestand: Path, tmp_path: Path) -> None:
    (kleiner_bestand / "kaputt.dat").write_bytes(b"\x00\x01\x02 kein bekanntes Format")
    ergebnis = ingest(kleiner_bestand, tmp_path / "parquet")
    assert ergebnis.failed == ["kaputt.dat"]
    assert ergebnis.documents >= 5


def test_gescanntes_pdf_wird_als_ocr_bedürftig_markiert(tmp_path: Path) -> None:
    """Ein PDF mit einer leeren Seite hat zu wenig Text – needs_ocr, aber kein Fehler."""
    import pymupdf

    roh = tmp_path / "raw"
    roh.mkdir(parents=True)
    leer = pymupdf.open()
    leer.new_page(width=595, height=842)
    leer.save(roh / "scan.pdf", no_new_id=True)
    leer.close()

    ingest(roh, tmp_path / "parquet")
    dokumente = read_table(tmp_path / "parquet", "documents")
    assert dokumente.filter(pl.col("file_name") == "scan.pdf")["needs_ocr"][0] is True
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_pipeline.py -v
```

Erwartet: `ModuleNotFoundError: No module named 'doccls.pipeline'`.

- [ ] **Schritt 3: `src/doccls/pipeline.py` schreiben**

```python
"""Orchestrierung: Originale → Parquet.

Hier – und nur hier – wird auf das Dateisystem zugegriffen. Erkennung, Extraktion und
Normalisierung bleiben reine Funktionen. Für den späteren Betrieb mit Objektspeicher ist
damit genau dieses Modul die Stelle, die umgestellt werden muss.

Inkrementell über die ``document_id`` (Pfad + Inhalts-Hash): Ein bereits bekanntes Dokument
wird übersprungen. Geänderter Inhalt ergibt eine neue ID und damit eine zusätzliche
Version; die alte bleibt erhalten und bleibt bewertbar.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from doccls.config import MIN_CHARS_PER_PAGE
from doccls.detect import MEDIA_TYPES, Format, detect_format
from doccls.extraction import extract
from doccls.extraction.mail import extract_eml
from doccls.models import SCHEMAS, Document, Segment, to_frame
from doccls.normalize import strip_boilerplate

TABLES = ("documents", "segments")


@dataclass
class IngestResult:
    """Was ein Durchlauf bewirkt hat."""

    documents: int = 0
    skipped: int = 0
    segments: int = 0
    attachments: int = 0
    failed: list[str] = field(default_factory=list)
    """Dateien, deren Format nicht unterstützt wird oder die nicht lesbar waren."""


def read_table(parquet_dir: Path, name: str) -> pl.DataFrame:
    """Tabelle lesen; noch nicht vorhanden heißt leer, nicht Fehler."""
    pfad = parquet_dir / f"{name}.parquet"
    return pl.read_parquet(pfad) if pfad.exists() else pl.DataFrame(schema=SCHEMAS[name])


def _write_table(parquet_dir: Path, name: str, neu: pl.DataFrame) -> None:
    """Anhängen statt ersetzen. Bestehende Zeilen bleiben, damit alte Versionen erhalten sind."""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    zusammen = pl.concat([read_table(parquet_dir, name), neu], how="vertical")
    zusammen.write_parquet(parquet_dir / f"{name}.parquet")


def _needs_ocr(segments: list[Segment], format_: Format) -> bool:
    """Zu wenig Text je Seite spricht für einen Scan. Die Entscheidung wird vermerkt,
    damit später auswertbar ist, ob OCR-Dokumente systematisch schlechter abschneiden."""
    if format_ is not Format.PDF:
        return False
    if not segments:
        return True
    return sum(len(s.text) for s in segments) / len(segments) < MIN_CHARS_PER_PAGE


def _process(
    source_path: str,
    data: bytes,
    now: datetime,
    parent_document_id: str | None,
    known: set[str],
    result: IngestResult,
) -> tuple[list[Document], list[Segment]]:
    """Ein Dokument samt seiner Anhänge verarbeiten. Rekursiv, weil Mails Mails enthalten können."""
    format_ = detect_format(data, source_path.rsplit("/", 1)[-1])
    dokument = Document.create(
        source_path=source_path, media_type=MEDIA_TYPES[format_], content=data,
        ingested_at=now, parent_document_id=parent_document_id,
    )
    if dokument.document_id in known:
        result.skipped += 1
        return [], []

    try:
        if format_ is Format.EML:
            segmente, anhaenge = extract_eml(dokument, data)
        else:
            segmente, anhaenge = extract(dokument, data), []
    except Exception:
        # Eine kaputte oder unbekannte Datei darf den Lauf nicht beenden. Sie wird vermerkt
        # und taucht in der Zusammenfassung auf; ein stiller Abbruch wäre schlimmer.
        result.failed.append(source_path)
        return [], []

    bereinigt = strip_boilerplate([s.text for s in segmente])
    segmente = [s.model_copy(update={"text": text})
                for s, text in zip(segmente, bereinigt, strict=True) if text]

    dokument = dokument.model_copy(update={"needs_ocr": _needs_ocr(segmente, format_)})
    known.add(dokument.document_id)
    result.documents += 1
    result.segments += len(segmente)

    dokumente, alle_segmente = [dokument], list(segmente)
    for anhang in anhaenge:
        result.attachments += 1
        kind_dokumente, kind_segmente = _process(
            f"{source_path}!{anhang.file_name}", anhang.content, now,
            dokument.document_id, known, result,
        )
        dokumente += kind_dokumente
        alle_segmente += kind_segmente
    return dokumente, alle_segmente


def ingest(raw_dir: Path, parquet_dir: Path) -> IngestResult:
    """Alle Dateien unterhalb ``raw_dir`` einlesen und als Parquet ablegen."""
    ergebnis = IngestResult()
    bekannt = set(read_table(parquet_dir, "documents")["document_id"])
    jetzt = datetime.now(UTC)

    dokumente: list[Document] = []
    segmente: list[Segment] = []
    for datei in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        neue_dokumente, neue_segmente = _process(
            str(datei.relative_to(raw_dir)), datei.read_bytes(), jetzt, None, bekannt, ergebnis
        )
        dokumente += neue_dokumente
        segmente += neue_segmente

    _write_table(parquet_dir, "documents", to_frame(dokumente, "documents"))
    _write_table(parquet_dir, "segments", to_frame(segmente, "segments"))
    return ergebnis
```

- [ ] **Schritt 4: Test laufen lassen, Erfolg bestätigen**

```bash
uv run pytest tests/test_pipeline.py -v
```

Erwartet: 7 passed.

- [ ] **Schritt 5: `scripts/ingest.py` schreiben**

```python
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
    print(f"neu: {ergebnis.documents}  übersprungen: {ergebnis.skipped}  "
          f"Segmente: {ergebnis.segments}  Anhänge: {ergebnis.attachments}")
    if ergebnis.failed:
        print(f"nicht lesbar ({len(ergebnis.failed)}): {', '.join(ergebnis.failed[:10])}")

    dokumente = read_table(args.out, "documents")
    print(f"Bestand: {dokumente.height} Dokumente, "
          f"davon {dokumente['needs_ocr'].sum()} vermutlich gescannt")


if __name__ == "__main__":
    main()
```

- [ ] **Schritt 6: Gesamtlauf über die echten Testdaten**

```bash
uv run python scripts/ingest.py
```

Erwartet: `neu: 570  übersprungen: 0  …  Anhänge: 10` — 560 Dateien plus die 10 Anhänge aus
den Mails. Segmentzahl im vierstelligen Bereich, keine unlesbaren Dateien.

Zweiter Lauf zur Kontrolle:

```bash
uv run python scripts/ingest.py
```

Erwartet: `neu: 0  übersprungen: 560`. Die 10 Anhänge erscheinen hier **nicht**: Eine
übersprungene Mail wird nicht geöffnet, also wird ihr Anhang nicht noch einmal besucht.

- [ ] **Schritt 7: Ergebnis mit DuckDB ansehen**

```bash
uv run python -c "
import duckdb
con = duckdb.connect()
print(con.sql('''
  SELECT media_type, count(*) AS dokumente, sum(needs_ocr::INT) AS ocr
  FROM read_parquet(\"data/parquet/documents.parquet\")
  GROUP BY 1 ORDER BY 2 DESC
'''))
print(con.sql('''
  SELECT kind, count(*) AS segmente, round(avg(length(text))) AS zeichen_schnitt
  FROM read_parquet(\"data/parquet/segments.parquet\")
  GROUP BY 1 ORDER BY 2 DESC
'''))
"
```

Erwartet: vier Medientypen, `needs_ocr` überall 0, Segmentarten `seite`, `abschnitt`,
`blatt`, `mail_kopf`, `mail_koerper`.

- [ ] **Schritt 8: Prüfen und committen**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest
git add -A && git commit -m "Ingestion-Pipeline: inkrementell, mit Anhängen als eigene Dokumente"
```

**Phase 1 ist damit erreicht:** 560 Dokumente aller vier Formate eingelesen, Segmente mit
nachvollziehbarer Herkunft, wiederholte Läufe überspringen.

---

## Aufgabe 12: OCR-Rückfall (optional, braucht eine Installation)

**Dateien:**
- Anlegen: `src/doccls/extraction/ocr.py`
- Ändern: `src/doccls/pipeline.py` (im `_process` nach `_needs_ocr`)
- Anlegen: `tests/test_extraction_ocr.py`
- Ändern: `pyproject.toml` (optionale Abhängigkeit)

**Warum getrennt:** `ocrmypdf` und `tesseract` sind auf diesem Rechner nicht installiert
und sind Systempakete, keine Python-Abhängigkeiten. Die Aufgabe bleibt deshalb hinter einem
`pytest.mark.skipif` und blockiert nichts.

- [ ] **Schritt 1: Systempakete installieren**

```bash
brew install ocrmypdf tesseract-lang
```

`tesseract-lang` bringt das deutsche Sprachmodell mit. Prüfen:

```bash
tesseract --list-langs | grep -E "^(deu|eng)$"
```

Erwartet: beide Zeilen.

- [ ] **Schritt 2: Test schreiben**

`tests/test_extraction_ocr.py`:

```python
"""OCR greift nur, wenn die digitale Extraktion zu wenig hergibt."""

import shutil
from pathlib import Path

import pymupdf
import pytest

from doccls.extraction.ocr import ocr_pdf

pytestmark = pytest.mark.skipif(
    shutil.which("ocrmypdf") is None, reason="ocrmypdf nicht installiert (brew install ocrmypdf)"
)


def gerastertes_pdf(tmp_path: Path) -> bytes:
    """Ein PDF mit Text, als Bild gerastert – so entsteht ein Scan ohne Textebene."""
    quelle = pymupdf.open()
    seite = quelle.new_page(width=595, height=842)
    seite.insert_htmlbox(seite.rect + (56, 56, -56, -56),
                         "<h1>Rechnung RE-2026-4711</h1><p>Rechnungsbetrag 1.234,00 EUR</p>")
    bild = quelle[0].get_pixmap(dpi=200)
    quelle.close()

    ziel = pymupdf.open()
    seite = ziel.new_page(width=595, height=842)
    seite.insert_image(seite.rect, pixmap=bild)
    daten: bytes = ziel.tobytes()
    ziel.close()
    return daten


def test_gerastertes_pdf_hat_vorher_keinen_text(tmp_path: Path) -> None:
    with pymupdf.open(stream=gerastertes_pdf(tmp_path), filetype="pdf") as pdf:
        assert pdf[0].get_text("text").strip() == ""


def test_ocr_macht_den_text_wieder_lesbar(tmp_path: Path) -> None:
    ergebnis = ocr_pdf(gerastertes_pdf(tmp_path))
    with pymupdf.open(stream=ergebnis, filetype="pdf") as pdf:
        text = pdf[0].get_text("text")
    assert "Rechnung" in text
    assert "1.234" in text or "1234" in text
```

- [ ] **Schritt 3: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_extraction_ocr.py -v
```

Erwartet: `ModuleNotFoundError: No module named 'doccls.extraction.ocr'` (oder `skipped`,
falls `ocrmypdf` fehlt — dann Schritt 1 nachholen).

- [ ] **Schritt 4: `src/doccls/extraction/ocr.py` schreiben**

```python
"""OCR-Rückfall für gescannte PDFs.

Aufgerufen wird das nur, wenn ``pipeline._needs_ocr`` zutrifft – also wenn die digitale
Extraktion zu wenig Zeichen je Seite geliefert hat. ``ocrmypdf`` ist ein Systemprogramm
(``brew install ocrmypdf tesseract-lang``), keine Python-Abhängigkeit.

``ocr_applied`` wird am Dokument vermerkt, damit später auswertbar ist, ob
OCR-Dokumente systematisch schlechter klassifiziert werden (Konzept § 5).
"""

import subprocess
import tempfile
from pathlib import Path

OCR_LANGUAGES = "deu+eng"
OCR_TIMEOUT_SECONDS = 300


class OcrUnavailableError(RuntimeError):
    """``ocrmypdf`` ist nicht installiert."""


def ocr_pdf(data: bytes, languages: str = OCR_LANGUAGES) -> bytes:
    """PDF durch OCR schicken und das Ergebnis mit Textebene zurückgeben."""
    with tempfile.TemporaryDirectory() as verzeichnis:
        quelle = Path(verzeichnis) / "ein.pdf"
        ziel = Path(verzeichnis) / "aus.pdf"
        quelle.write_bytes(data)
        try:
            subprocess.run(
                ["ocrmypdf", "--language", languages, "--force-ocr", "--quiet",
                 str(quelle), str(ziel)],
                check=True, capture_output=True, timeout=OCR_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as fehler:
            raise OcrUnavailableError(
                "ocrmypdf nicht gefunden. Installation: brew install ocrmypdf tesseract-lang"
            ) from fehler
        return ziel.read_bytes()
```

- [ ] **Schritt 5: Test laufen lassen, Erfolg bestätigen**

```bash
uv run pytest tests/test_extraction_ocr.py -v
```

Erwartet: 2 passed.

- [ ] **Schritt 6: Pipeline anbinden**

In `src/doccls/pipeline.py` den Import ergänzen:

```python
from doccls.extraction.ocr import OcrUnavailableError, ocr_pdf
```

und in `_process` die Zeile

```python
    dokument = dokument.model_copy(update={"needs_ocr": _needs_ocr(segmente, format_)})
```

ersetzen durch:

```python
    braucht_ocr = _needs_ocr(segmente, format_)
    ocr_angewandt = False
    if braucht_ocr and use_ocr:
        try:
            segmente = extract(dokument, ocr_pdf(data))
            ocr_angewandt = True
        except (OcrUnavailableError, OSError, ValueError):
            pass  # OCR ist ein Rückfall. Misslingt sie, bleibt needs_ocr stehen.
    dokument = dokument.model_copy(
        update={"needs_ocr": braucht_ocr, "ocr_applied": ocr_angewandt}
    )
```

Dazu ändern sich drei Signaturen. In `_process`:

```python
def _process(
    source_path: str,
    data: bytes,
    now: datetime,
    parent_document_id: str | None,
    known: set[str],
    result: IngestResult,
    use_ocr: bool = False,
) -> tuple[list[Document], list[Segment]]:
```

und im rekursiven Aufruf für Anhänge das Argument durchreichen:

```python
        kind_dokumente, kind_segmente = _process(
            f"{source_path}!{anhang.file_name}", anhang.content, now,
            dokument.document_id, known, result, use_ocr,
        )
```

In `ingest`:

```python
def ingest(raw_dir: Path, parquet_dir: Path, use_ocr: bool = False) -> IngestResult:
```

und dort im Aufruf:

```python
        neue_dokumente, neue_segmente = _process(
            str(datei.relative_to(raw_dir)), datei.read_bytes(), jetzt, None, bekannt,
            ergebnis, use_ocr,
        )
```

In `scripts/ingest.py` das Argument ergänzen:

```python
    parser.add_argument("--ocr", action="store_true",
                        help="Gescannte PDFs durch OCR schicken (braucht ocrmypdf)")
    ...
    ergebnis = ingest(args.raw, args.out, use_ocr=args.ocr)
```

**Wichtig:** Die Segmente werden nach der OCR neu extrahiert, aber die `document_id` bleibt
die des Originals — OCR erzeugt keine neue Dokumentversion, sie repariert nur die
Extraktion. Ein Lauf mit `--ocr` nach einem Lauf ohne `--ocr` überspringt die Dokumente
deshalb als bekannt; um OCR nachzuholen, müssen die betroffenen Zeilen entfernt oder
`data/parquet/` neu aufgebaut werden. Das ist eine bekannte Grobheit von Phase 1.

- [ ] **Schritt 7: Test für den Pipeline-Anschluss ergänzen**

An `tests/test_pipeline.py` anhängen:

```python
def test_ocr_bleibt_aus_wenn_nicht_verlangt(tmp_path: Path) -> None:
    """Vorgabe ist aus: OCR kostet Sekunden je Seite und ist ein Rückfall, kein Normalweg."""
    import pymupdf

    roh = tmp_path / "raw"
    roh.mkdir(parents=True)
    leer = pymupdf.open()
    leer.new_page(width=595, height=842)
    leer.save(roh / "scan.pdf", no_new_id=True)
    leer.close()

    ingest(roh, tmp_path / "parquet")
    dokumente = read_table(tmp_path / "parquet", "documents")
    assert dokumente["needs_ocr"][0] is True
    assert dokumente["ocr_applied"][0] is False
```

- [ ] **Schritt 8: Prüfen und committen**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest
git add -A && git commit -m "OCR-Rückfall für gescannte PDFs, standardmäßig aus"
```

---

## Abschluss Phase 0 + 1

- [ ] **Vollständiger Durchlauf von Null**

```bash
rm -rf data/raw data/parquet data/generated
uv run python scripts/generate_documents.py
uv run python scripts/ingest.py
uv run pytest -v
uv run ruff check . && uv run mypy .
```

Erwartet: 560 erzeugte Dateien, 570 eingelesene Dokumente (10 davon Mail-Anhänge), alle
Tests grün.

- [ ] **`docs/testdaten.md` gegen das Ergebnis prüfen** — stimmen die Zahlen in der Tabelle
  noch mit dem, was der Generator ausgibt? Wenn nicht: Dokument nachziehen, nicht den
  Generator anpassen.

- [ ] **README schreiben** mit den drei Befehlen oben und der Angabe, dass `data/` nicht
  versioniert wird.

- [ ] **Committen**

```bash
git add -A && git commit -m "README und Abschluss Phase 0 und 1"
```

## Was Phase 2 als Nächstes braucht

Aus diesem Stand ergibt sich der Einstieg in Phase 2 unmittelbar:

- `documents.parquet` + `segments.parquet` als Eingang für `features.py`
- `manifest.parquet` liefert über `source_path` die bekannte Wahrheit und den Split
- Der Join läuft über `source_path`, nicht über `document_id` — das Manifest kennt die IDs
  nicht, weil es vor der Ingestion entsteht. Anhänge (`…!rechnung.pdf`) stehen nicht im
  Manifest und bleiben in Phase 2 zunächst ungelabelt.
