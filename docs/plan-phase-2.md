# Phase 2: Merkmale, Klassifikator, Konfidenz — Implementierungsplan

> **Für Agenten:** Diese Aufgaben werden einzeln abgearbeitet. Schritte sind als Checkboxen
> (`- [ ]`) geführt. Nach jeder Aufgabe wird geprüft und committet, bevor die nächste beginnt.

**Ziel:** Aus den Segmenten der Phase-1-Pipeline einen Merkmalsvektor je Dokument bauen,
darauf einen lernfähigen Klassifikator trainieren, dessen Konfidenz kalibrieren und das
Ergebnis gegen das eingefrorene Gold-Set messen — mit Zahlen, die man glauben kann.

**Architektur:** Drei Merkmalsblöcke (Embedding, Zeichen-n-Gramme, Strukturmerkmale) werden
aneinandergehängt und mit einer `feature_version` versehen. Darauf läuft eine multinomiale
logistische Regression, deren rohe Softmax-Ausgabe auf einer **separaten** Kalibriermenge
per Temperature Scaling geeicht wird. Zwei Schwellen — τ für die Konfidenz, δ für die
OOD-Erkennung — werden aus Daten abgelesen, nicht geraten. Alles läuft lokal auf der CPU
bzw. über MPS.

**Tech Stack:** Python 3.14, uv, Polars/Parquet, scikit-learn, scipy, numpy, matplotlib,
sentence-transformers (E5), Ollama (BGE-M3), pytest, ruff, mypy.

**Spezifikation:** [konzept.md](konzept.md) §6 (Merkmale), §7 (Klassifikation und
Konfidenz), §9.3 (Kennzahlen). Der Phasenplan steht in §11.

**Vorbedingung:** Phase 1 ist abgeschlossen und zusammengeführt. `data/parquet/` enthält
`documents` (570 Zeilen) und `segments`; `data/generated/manifest.parquet` enthält die
bekannte Wahrheit. Fehlt der Bestand: `uv run python scripts/generate_documents.py`, dann
`uv run python scripts/ingest.py`.

---

## Global Constraints

Diese Vorgaben gelten für **jede** Aufgabe. Sie wiederholen sich nicht in den Aufgabentexten.

- **Paketverwaltung ausschließlich `uv`.** Kein `pip`, kein manuell angelegtes venv.
- **Container ausschließlich Podman.** Auf diesem Rechner ist kein Docker installiert;
  `docker compose` existiert nicht. In Phase 2 werden ohnehin keine Dienste gebraucht.
- **Python 3.14.** Geprüft: `torch>=2.14`, `scikit-learn>=1.9`, `scipy>=1.18`,
  `matplotlib>=3.11` und `sentence-transformers>=6.1` lösen darauf auf. Die Warnung im
  Konzept §10, man müsse auf 3.13 ausweichen, ist überholt.
- **Alles auf Deutsch:** Docstrings, Kommentare, Fehlermeldungen, Testnamen, lokale
  Variablen und Commit-Botschaften. Öffentliche Funktions- und Klassennamen bleiben
  englisch, wie im bestehenden Code (`extract_pdf`, `load_classes`, `ingest`).
- **Typannotationen überall.** `uv run mypy .` läuft im Strict-Modus und muss sauber sein.
- **Keine Unterdrückung von Prüfbefunden.** Kein `# noqa`, kein `# type: ignore`. Wer eine
  Meldung nicht anders wegbekommt, behebt die Ursache oder meldet sie. In diesem Projekt
  war jede vorgeschlagene Unterdrückung bisher ein Hinweis auf einen echten Mangel.
- **Abschlussprüfung je Aufgabe:** `uv run ruff format --check .`, `uv run ruff check .`,
  `uv run mypy .`, `uv run pytest -q` — alle vier sauber, sonst ist die Aufgabe nicht fertig.
- **Das Gold-Set wird im Code geschützt, nicht per Konvention.** Keine Funktion, die
  trainiert oder kalibriert, darf Dokumente mit `split == "gold"` sehen. Das wird in der
  jeweiligen Funktion geprüft und mit einer Ausnahme quittiert — nicht per Kommentar
  zugesichert. (Konzept §9.2)
- **Zufall ist immer geseedet.** Jede Funktion mit Zufallsanteil nimmt einen `seed` entgegen
  und gibt bei gleichem Seed dasselbe Ergebnis. Zweimal laufen lassen und vergleichen ist
  Teil des Tests.
- **Keine Abstraktion auf Vorrat.** Module späterer Phasen (`sampling.py`, `labels.py`,
  `registry.py`) werden in Phase 2 **nicht** angelegt.

### Testvorgaben — hier steht, woran Phase 1 wiederholt gescheitert ist

In Phase 1 stammte fast jeder gefundene Mangel aus dem Plan, nicht aus den Umsetzungen, und
fast alle waren derselbe Typ. Diese Vorgaben sind die Antwort darauf und sind **nicht
verhandelbar**:

- **Mutationsprobe für jeden Test.** Nach dem Grünwerden: die fertige Implementierung
  gezielt falsch machen, die Suite laufen lassen, festhalten welcher Test rot wird, die
  Änderung zurücknehmen. Ein Test, der bei seiner Mutation grün bleibt, prüft nichts —
  reparieren oder streichen. „Der Test war rot, weil die Funktion noch nicht existierte"
  ist **kein** Nachweis.
- **Committen, bevor mutiert wird.** `git checkout -- <datei>` setzt sonst die eigene, noch
  nicht committete Arbeit mit zurück.
- **Nicht in einer Kopie des Baums mutieren.** Über den editierbaren Install wird das Paket
  aus dem echten Arbeitsbaum importiert; die Mutation in der Kopie wirkt nicht, und die
  Probe misst nichts.
- **Tests binden die Inhaltsmenge, nicht die Anzahl.** `assert len(vektoren) == 570` ist
  wertlos, wenn die Vektoren Nullen sind. In Phase 1 überlebten fünf Mutationen mit stillem
  Inhaltsverlust eine ganze Testsuite, weil jeder Test nur zählte.
- **Eine Schwelle braucht einen Test, der ihren Abstand zur echten Verteilung bindet.** Ein
  Test je Richtung reicht nicht: Die OCR-Schwelle lag mitten in der Verteilung der echten
  Dokumente, und beide Richtungstests blieben grün.
- **Tests, die strukturell nicht fehlschlagen können, werden gestrichen, nicht repariert.**
  Vier davon steckten im Phase-1-Plan.

### Bekannte Grenzen des Korpus — hier nicht beheben, aber beim Deuten der Zahlen wissen

- **Die Trainingsmenge ist dünn.** Der Schnitt gibt je Klasse 5 Vorlagen ans Gold-Set, 1 an
  die Kalibrierung und nur **2 ans Training**. Das Modell lernt aus effektiv zwei Beispielen
  je Klasse (die 10 Varianten einer Vorlage sind fast dasselbe Dokument) und wird gegen fünf
  ungesehene Vorlagen geprüft. Aufgabe 15 baut die Diagnose ein, die „Korpus zu dünn" von
  „Verfahren zu schwach" unterscheidet.
- **Mailkörper und Anhang sind fast dasselbe Dokument** (gemessene Ähnlichkeit 0,95). Der
  Generator schreibt denselben Rechnungstext in beide, obwohl Konzept §5 verlangt, dass sie
  verschiedene Klassen tragen. Jede Messung, die beide enthält, sieht dadurch etwas zu gut
  aus. Bewusst offen gelassen, siehe [phase-1-offene-punkte.md](phase-1-offene-punkte.md).
- **`strip_boilerplate` entfernt auf diesem Bestand null Zeichen.** Die Funktion ist gebaut
  und geprüft, greift hier aber nicht, weil der Generator die Fußzeile nur einmal ans
  Dokumentende setzt. Sie bleibt in der Kette, ihre Wirkung ist auf diesem Korpus null.

---

## Dateistruktur

Angelegt werden:

| Datei | Verantwortung |
|---|---|
| `src/doccls/features/__init__.py` | `FeatureSet`, `build_features`, `feature_version` — setzt die Blöcke zusammen |
| `src/doccls/features/text.py` | Segmente → ein Dokumenttext, Chunks, Kopftext |
| `src/doccls/features/embedding.py` | `Embedder`-Protokoll, `E5Embedder`, `BgeM3Embedder` |
| `src/doccls/features/ngrams.py` | TF-IDF über Zeichen-n-Gramme + `TruncatedSVD` |
| `src/doccls/features/structural.py` | rund 40 erklärbare Zahlen je Dokument |
| `src/doccls/splits.py` | Trainings-, Kalibrier- und Gold-Menge laden; Gold-Schutz |
| `src/doccls/classify.py` | Training und Vorhersage, drei Modelle |
| `src/doccls/calibrate.py` | Temperature Scaling, Schwelle τ |
| `src/doccls/decide.py` | OOD-Abstand δ, Entscheidung AUTO/REVIEW/SONSTIGES |
| `src/doccls/evaluate.py` | Kennzahlen und Bootstrap |
| `src/doccls/reports.py` | Reliability Diagram, Konfusionsmatrix als PNG |
| `config/features.yaml` | Merkmalsparameter, aus denen die `feature_version` entsteht |
| `scripts/build_features.py` | CLI: Segmente → Merkmalsmatrix in Parquet |
| `scripts/train.py` | CLI: Training, Kalibrierung, Schwellen, Artefakt ablegen |
| `scripts/evaluate.py` | CLI: Messung gegen das Gold-Set, Bericht schreiben |

Geändert werden: `pyproject.toml` (Abhängigkeiten), `src/doccls/config.py` (Modellnamen und
Pfade), `src/doccls/generation/manifest.py` und `scripts/generate_documents.py` (Aufgabe 1).

Das Konzept §10 skizziert ein einzelnes `features.py`. Hier wird daraus ein Paket, weil vier
klar getrennte Verantwortungen darin liegen und eine Datei mit Embedding, TF-IDF, SVD und
40 Strukturmerkmalen zu groß würde, um sie am Stück zu überblicken.

---

## Aufgabe 1: Anhänge bekommen eine Wahrheit im Manifest

**Dateien:**
- Ändern: `src/doccls/generation/manifest.py`
- Ändern: `scripts/generate_documents.py`
- Ändern: `docs/testdaten.md`
- Test: `tests/test_generation_manifest.py`

**Schnittstellen:**
- Nutzt: `Split`, `manifest_frame`, `MANIFEST_SCHEMA` aus `doccls.generation.manifest`
- Liefert: `manifest_frame` schreibt zusätzlich eine Zeile je Mailanhang

**Warum das zuerst kommt:** `data/parquet/documents` hat 570 Zeilen, `manifest.parquet` nur
560. Die zehn fehlenden sind die Mailanhänge — sie entstehen erst beim Einlesen, der
Generator kennt sie nicht. Jeder `join` von `documents` auf `manifest` verliert sie
stillschweigend. Ab Aufgabe 8 hängt jede Messung an diesem `join`; der Fehler wäre dann
nicht mehr sichtbar, sondern nur noch wirksam.

Der Anhang bekommt **dieselbe Klasse und denselben Split wie seine Elternvorlage**. Der
Split ist der wichtigere Teil: Fielen Mail und Anhang auf verschiedene Seiten des
Gold-Schnitts, stünden zwei Dokumente mit 0,95 Textähnlichkeit gleichzeitig im Training und
im Gold-Set — genau die Leckage, die Konzept §9.2 mit dem Schnitt über Dokumentfamilien
verhindern will.

- [ ] **Schritt 1: Test schreiben**

An `tests/test_generation_manifest.py` anhängen:

```python
def test_mailanhang_steht_mit_im_manifest() -> None:
    """Der Anhang entsteht erst beim Einlesen, nicht beim Erzeugen – ohne eigene Zeile
    verliert ihn jeder join von documents auf manifest stillschweigend."""
    korpus = build_corpus()
    splits = assign_splits(korpus)
    mail_spec = next(s for s in korpus if "eml" in s.formats and s.hat_anhang)
    pfade = [Path(f"{s.template_id}-{s.variant:02d}.{s.formats[0]}") for s in korpus]

    rahmen = manifest_frame(korpus, splits, pfade, Path("."))

    anhaenge = rahmen.filter(pl.col("source_path").str.contains("!"))
    assert anhaenge.height > 0, "Kein Anhang im Manifest – die Mailvorlagen tragen keinen"
    for zeile in anhaenge.iter_rows(named=True):
        eltern_pfad = zeile["source_path"].split("!")[0]
        eltern = rahmen.filter(pl.col("source_path") == eltern_pfad)
        assert eltern.height == 1, f"Elternmail {eltern_pfad} fehlt im Manifest"
        assert zeile["split"] == eltern["split"][0], (
            f"{zeile['source_path']} liegt im Split {zeile['split']}, die Elternmail "
            f"{eltern_pfad} aber in {eltern['split'][0]} – zwei Dokumente mit 0,95 "
            "Textaehnlichkeit auf beiden Seiten des Gold-Schnitts"
        )
        assert zeile["class_key"] == eltern["class_key"][0]


def test_jedes_eingelesene_dokument_hat_eine_wahrheit() -> None:
    """Die Gegenprobe am echten Bestand: documents und manifest müssen deckungsgleich sein.

    Ohne diesen Test bliebe die Luecke unbemerkt, sobald eine neue Dokumentart hinzukommt,
    die erst beim Einlesen entsteht (etwa ein ZIP-Eintrag oder eine verschachtelte Mail).
    """
    dokumente = read_table(PARQUET_DIR, "documents")
    manifest = pl.read_parquet(GENERATED_DIR / "manifest.parquet")
    ohne_wahrheit = dokumente.join(manifest.select("source_path"), on="source_path", how="anti")
    assert ohne_wahrheit.height == 0, (
        f"{ohne_wahrheit.height} eingelesene Dokumente haben keine Zeile im Manifest: "
        f"{ohne_wahrheit['source_path'].to_list()[:5]}"
    )
```

Zusätzliche Importe am Kopf der Testdatei:

```python
import polars as pl

from doccls.config import GENERATED_DIR, PARQUET_DIR
from doccls.generation.manifest import manifest_frame
from doccls.pipeline import read_table
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_generation_manifest.py -v -k "manifest or wahrheit"
```

Erwartet: beide neuen Tests FAIL. `test_mailanhang_steht_mit_im_manifest` scheitert an
`anhaenge.height > 0`, `test_jedes_eingelesene_dokument_hat_eine_wahrheit` an den zehn
Dokumenten ohne Wahrheit.

Falls `DocumentSpec` kein Feld `hat_anhang` hat: in `src/doccls/generation/content.py`
nachsehen, wie die Mailvorlagen mit Anhang gekennzeichnet sind, und den Test auf den dort
vorhandenen Namen anpassen. **Nicht** ein Feld erfinden.

- [ ] **Schritt 3: `manifest_frame` erweitern**

In `src/doccls/generation/manifest.py`, in `manifest_frame`, nach den Zeilen für die
geschriebenen Dateien eine Zeile je Anhang ergänzen. Der Pfad folgt derselben Schreibweise,
die `pipeline.py` beim Einlesen erzeugt: `<mailpfad>!<anhangname>`.

```python
def manifest_frame(
    corpus: list[DocumentSpec], splits: dict[str, Split], paths: list[Path], root: Path
) -> pl.DataFrame:
    """Die bekannte Wahrheit als Tabelle: Pfad → Klasse, Vorlage, Split.

    Mailanhänge bekommen eine eigene Zeile mit Klasse und Split der Elternmail. Sie
    entstehen erst beim Einlesen (``pipeline.py`` macht aus jedem Anhang ein eigenes
    Dokument), der Generator schreibt sie nie als Datei. Ohne eigene Zeile verliert jeder
    ``join`` von ``documents`` auf das Manifest sie stillschweigend, und beim Split wären
    Mail und Anhang – zwei Dokumente mit 0,95 Textähnlichkeit – auf beiden Seiten des
    Gold-Schnitts möglich (Konzept § 9.2).
    """
    zeilen: list[dict[str, object]] = []
    for spec, pfad in zip(corpus, paths, strict=True):
        relativ = str(pfad.relative_to(root))
        zeilen.append(
            {
                "source_path": relativ,
                "class_key": spec.class_key,
                "template_id": spec.template_id,
                "variant": spec.variant,
                "split": str(splits[spec.template_id]),
                "format": pfad.suffix.lstrip("."),
            }
        )
        for anhang_name in anhang_namen(spec, pfad):
            zeilen.append(
                {
                    "source_path": f"{relativ}!{anhang_name}",
                    "class_key": spec.class_key,
                    "template_id": spec.template_id,
                    "variant": spec.variant,
                    "split": str(splits[spec.template_id]),
                    "format": Path(anhang_name).suffix.lstrip("."),
                }
            )
    return pl.DataFrame(zeilen, schema=MANIFEST_SCHEMA)
```

Dazu die Hilfsfunktion, die die Anhangsnamen aus derselben Quelle nimmt wie der Schreiber —
**nicht** aus einer zweiten, danebenlaufenden Liste:

```python
def anhang_namen(spec: DocumentSpec, pfad: Path) -> list[str]:
    """Namen der Anhänge dieser Vorlage, in derselben Schreibweise wie ``writers.write``.

    Eine zweite, unabhängig gepflegte Liste würde früher oder später auseinanderlaufen –
    deshalb wird hier dieselbe Regel angewandt, die den Anhang beim Schreiben benennt.
    """
```

**Der Rumpf hängt davon ab, wie `writers.write` den Anhang benennt.** Vor dem Schreiben
in `src/doccls/generation/writers.py` nachsehen, wie der Anhang in die `.eml` gelegt wird,
und genau diesen Namen hier herleiten. Ist der Name dort aus einer Variablen gebildet, diese
Bildung in eine gemeinsame Funktion ziehen und von beiden Stellen aufrufen. Zwei Stellen,
die denselben Namen unabhängig erzeugen, sind ein Fehler in Wartestellung.

- [ ] **Schritt 4: Tests laufen lassen**

```bash
uv run pytest tests/test_generation_manifest.py -v
```

Erwartet: alle PASS, einschließlich der bestehenden Tests zum eingefrorenen Schnitt.

- [ ] **Schritt 5: Bestand neu erzeugen und einlesen**

```bash
uv run python scripts/generate_documents.py
rm -rf data/parquet
uv run python scripts/ingest.py
uv run pytest -q
```

Erwartet: `config/splits.yaml` meldet **keinen** Drift (die `template_id` ändern sich
nicht), `ingest` meldet 570 Dokumente und 0 ohne Segmente, die Suite ist grün.

- [ ] **Schritt 6: `docs/testdaten.md` nachziehen**

Den Abschnitt über das Manifest um zwei Sätze ergänzen: dass Mailanhänge eine eigene Zeile
tragen, dass sie Klasse und Split der Elternmail erben, und warum (Textähnlichkeit 0,95,
Leckage über den Gold-Schnitt).

- [ ] **Schritt 7: Mutationsprobe**

Mindestens drei Mutationen, jede einzeln, mit `git checkout --` zurücknehmen:

1. Anhangszeilen gar nicht erzeugen → `test_mailanhang_steht_mit_im_manifest` muss rot werden.
2. Dem Anhang einen anderen Split geben als der Elternmail (etwa fest `Split.TRAIN`) →
   die Split-Zusicherung muss rot werden.
3. Dem Anhang eine andere Klasse geben → die Klassen-Zusicherung muss rot werden.

- [ ] **Schritt 8: Committen**

```bash
git add src/doccls/generation/manifest.py scripts/generate_documents.py \
        tests/test_generation_manifest.py docs/testdaten.md
git commit -m "Mailanhaenge bekommen eine Zeile im Manifest"
```

---

## Aufgabe 2: Abhängigkeiten, Merkmalsparameter und Versionierung

**Dateien:**
- Ändern: `pyproject.toml`
- Ändern: `src/doccls/config.py`
- Anlegen: `config/features.yaml`
- Anlegen: `src/doccls/features/__init__.py`
- Anlegen: `tests/test_features_version.py`

**Schnittstellen:**
- Liefert: `FeatureConfig` (Pydantic), `load_feature_config(path=None) -> FeatureConfig`,
  `feature_version(config: FeatureConfig) -> str`

**Worum es geht:** Konzept §6.4 verlangt, dass jeder Merkmalsvektor eine `feature_version`
trägt und ein Modell nur Merkmale derselben Version sehen darf, mit denen es trainiert
wurde. Ohne diese Regel entstehen Fehler, die man nicht findet: Ein Modell, das auf
256-dimensionalen n-Gramm-Merkmalen trainiert wurde und auf 128-dimensionalen vorhersagt,
liefert Unsinn mit hoher Konfidenz statt einer Fehlermeldung.

Die Version wird **berechnet, nicht gepflegt** — aus dem Inhalt der Konfiguration. Eine von
Hand hochgezählte Version wird irgendwann vergessen; ein Hash über die Parameter nie.

- [ ] **Schritt 1: Abhängigkeiten aufnehmen**

```bash
uv add "scikit-learn>=1.9.1" "scipy>=1.18.1" "numpy>=2.0" "matplotlib>=3.11.2"
uv add "sentence-transformers>=6.1.0" "torch>=2.14.0"
```

Kein HTTP-Client als Abhängigkeit: Der Ollama-Aufruf in Aufgabe 5 geht über
`urllib.request` aus der Standardbibliothek. Eine Bibliothek für einen einzigen
POST-Aufruf mit JSON-Rumpf wäre nicht zu rechtfertigen.

Geprüft: alle lösen auf Python 3.14 auf. `torch` bringt auf Apple Silicon MPS mit und lädt
mehrere Gigabyte — das ist erwartet und einmalig.

- [ ] **Schritt 2: Test schreiben**

`tests/test_features_version.py`:

```python
"""Die Merkmalsversion ist der Schutz davor, ein Modell mit fremden Merkmalen zu füttern."""

from pathlib import Path

import pytest

from doccls.features import FeatureConfig, feature_version, load_feature_config


def test_version_ist_stabil_ueber_laeufe() -> None:
    """Gleiche Konfiguration, gleiche Version – sonst wäre jeder Lauf inkompatibel zum
    vorherigen und die Prüfung liefe ins Leere."""
    konfiguration = load_feature_config()
    assert feature_version(konfiguration) == feature_version(konfiguration)


def test_jeder_parameter_veraendert_die_version() -> None:
    """Der eigentliche Zweck: Ändert sich irgendein Merkmalsparameter, muss die Version
    sich ändern. Ein Test, der nur EINEN Parameter prüft, übersieht genau den, der später
    still geändert wird – deshalb werden hier alle durchgegangen."""
    basis = load_feature_config()
    original = feature_version(basis)
    aenderungen: dict[str, object] = {
        "embedder": "bge-m3",
        "chunk_chars": 900,
        "head_chars": 500,
        "ngram_min": 2,
        "ngram_max": 6,
        "ngram_max_features": 20000,
        "svd_components": 128,
        "position_decay": 8.0,
    }
    for feld, wert in aenderungen.items():
        geaendert = basis.model_copy(update={feld: wert})
        assert feature_version(geaendert) != original, (
            f"Feld {feld} aendert die Merkmale, aber nicht die Version – ein Modell wuerde "
            "stillschweigend mit fremden Merkmalen rechnen"
        )


def test_version_haengt_nicht_an_der_reihenfolge() -> None:
    """Zwei inhaltsgleiche Konfigurationen müssen dieselbe Version ergeben, auch wenn die
    Felder in anderer Reihenfolge gesetzt wurden – sonst würde ein harmloses Umsortieren
    in der YAML alle Merkmale ungültig machen."""
    a = load_feature_config()
    b = FeatureConfig(**dict(reversed(list(a.model_dump().items()))))
    assert feature_version(a) == feature_version(b)


def test_unbekanntes_feld_wird_abgewiesen(tmp_path: Path) -> None:
    """Ein Tippfehler in features.yaml darf nicht stillschweigend die Vorgabe benutzen –
    dann trüge die Version eine Einstellung, die gar nicht wirkt."""
    datei = tmp_path / "features.yaml"
    datei.write_text("embedder: e5\nchunk_charss: 1200\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_charss"):
        load_feature_config(datei)
```

- [ ] **Schritt 3: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_features_version.py -v
```

Erwartet: FAIL, `ModuleNotFoundError: No module named 'doccls.features'`.

- [ ] **Schritt 4: `config/features.yaml` anlegen**

```yaml
# Parameter der Merkmalsbildung. Aus dem Inhalt dieser Datei wird die feature_version
# berechnet (src/doccls/features/__init__.py). Wer hier etwas ändert, macht alle bisher
# gerechneten Merkmalsvektoren ungültig – das ist gewollt und der Sinn der Sache.

embedder: e5              # e5 | bge-m3
chunk_chars: 1200         # Konzept § 6.1: Text in ~1200-Zeichen-Chunks
head_chars: 1000          # eigener Block aus den ersten 1000 Zeichen
position_decay: 4.0       # Gewicht w_i = 1 / (1 + i/position_decay)
ngram_min: 3              # Konzept § 6.2: Zeichen-n-Gramme 3 bis 5
ngram_max: 5
ngram_max_features: 50000
svd_components: 256       # danach TruncatedSVD auf 256 Dimensionen
```

- [ ] **Schritt 5: `src/doccls/features/__init__.py` anlegen**

```python
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
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from doccls.config import PROJECT_ROOT

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


def feature_version(config: FeatureConfig) -> str:
    """Kurzer, stabiler Fingerabdruck der Merkmalsparameter.

    ``sort_keys=True``: Die Version darf nicht davon abhängen, in welcher Reihenfolge die
    Felder in der YAML stehen – sonst machte ein harmloses Umsortieren alle gerechneten
    Vektoren ungültig.
    """
    text = json.dumps(config.model_dump(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
```

- [ ] **Schritt 6: `src/doccls/config.py` ergänzen**

Ans Ende anhängen:

```python
FEATURES_DIR = DATA_DIR / "features"
"""Merkmalsmatrizen je feature_version."""

MODELS_DIR = DATA_DIR / "models"
"""Modellartefakte je Lauf."""

REPORTS_DIR = DATA_DIR / "reports"
"""Diagramme und Kennzahlen je Lauf."""

E5_MODEL_NAME = os.environ.get("DOCCLS_E5_MODEL", "intfloat/multilingual-e5-base")
"""768 Dimensionen, 512 Token Kontext. Verlangt das Präfix ``passage: `` (Konzept § 6.1)."""

OLLAMA_URL = os.environ.get("DOCCLS_OLLAMA_URL", "http://localhost:11434")
BGE_M3_MODEL_NAME = os.environ.get("DOCCLS_BGE_M3_MODEL", "bge-m3:567m-fp16")
"""1024 Dimensionen, 8192 Token Kontext, keine Präfixe. Läuft über einen Ollama-Dienst."""
```

- [ ] **Schritt 7: Tests laufen lassen**

```bash
uv run pytest tests/test_features_version.py -v
uv run ruff format --check . && uv run ruff check . && uv run mypy . && uv run pytest -q
```

Erwartet: alle PASS.

- [ ] **Schritt 8: Mutationsprobe**

1. `sort_keys=True` aus `feature_version` entfernen → `test_version_haengt_nicht_an_der_reihenfolge` rot.
2. Ein Feld aus `FeatureConfig` beim Hashen auslassen (etwa `config.model_dump(exclude={"svd_components"})`)
   → `test_jeder_parameter_veraendert_die_version` rot, und zwar mit dem Namen des Feldes in
   der Meldung.
3. `extra="forbid"` auf `extra="ignore"` → `test_unbekanntes_feld_wird_abgewiesen` rot.

- [ ] **Schritt 9: Committen**

```bash
git add pyproject.toml uv.lock src/doccls/config.py config/features.yaml \
        src/doccls/features/__init__.py tests/test_features_version.py
git commit -m "Merkmalsparameter und berechnete Merkmalsversion"
```

---

## Aufgabe 3: Dokumenttext, Chunks und Kopftext

**Dateien:**
- Anlegen: `src/doccls/features/text.py`
- Anlegen: `tests/test_features_text.py`

**Schnittstellen:**
- Nutzt: `Segment` aus `doccls.models`, `strip_boilerplate` aus `doccls.normalize`
- Liefert:
  - `document_text(segments: list[Segment]) -> str`
  - `chunks(text: str, chunk_chars: int) -> list[str]`
  - `position_weights(count: int, decay: float) -> list[float]`
  - `head_text(text: str, head_chars: int) -> str`

**Worum es geht:** Zwischen den Segmenten aus Phase 1 und dem Embedder fehlt ein Schritt.
Konzept §6.1 verlangt: Text in ~1200-Zeichen-Chunks, jeder eingebettet, dann
positionsgewichteter Mittelwert mit `w_i = 1 / (1 + i/4)`, normiert — frühe Chunks zählen
mehr, weil der Dokumentkopf die Klasseninformation fast immer trägt. Zusätzlich die ersten
1000 Zeichen als eigener Block, damit der Mittelwert bei langen Verträgen nicht genau die
Stelle verwässert, auf die es ankommt.

Diese Funktionen sind rein: Text rein, Text raus. Kein Modell, kein Dateizugriff.

- [ ] **Schritt 1: Test schreiben**

`tests/test_features_text.py`:

```python
"""Zwischen Segment und Embedder: Text zusammensetzen, in Chunks schneiden, gewichten."""

import math

from doccls.features.text import chunks, document_text, head_text, position_weights
from doccls.models import Segment, SegmentKind


def segment(index: int, text: str) -> Segment:
    return Segment(
        segment_id=f"s{index}",
        document_id="d1",
        index=index,
        kind=SegmentKind.ABSCHNITT,
        locator=f"Abschnitt {index}",
        heading=None,
        text=text,
    )


def test_dokumenttext_folgt_der_segmentreihenfolge() -> None:
    """Die Reihenfolge trägt Bedeutung: Der Kopf eines Dokuments steht vorn, und die
    Positionsgewichtung baut darauf auf. Eine nach document_id sortierte oder gar
    unsortierte Zusammenfuehrung wuerde das zunichtemachen."""
    segmente = [segment(2, "drittens"), segment(0, "erstens"), segment(1, "zweitens")]
    assert document_text(segmente) == "erstens zweitens drittens"


def test_dokumenttext_verliert_kein_zeichen() -> None:
    """Bindet die Inhaltsmenge, nicht die Segmentzahl: Eine Umsetzung, die nur das erste
    Segment nimmt oder jedes auf 50 Zeichen kuerzt, muss hier auffallen."""
    texte = [f"Segment {i} mit reichlich eigenem Inhalt an dieser Stelle" for i in range(8)]
    segmente = [segment(i, t) for i, t in enumerate(texte)]
    ergebnis = document_text(segmente)
    for t in texte:
        assert t in ergebnis, f"{t!r} fehlt im Dokumenttext"
    assert len(ergebnis) >= sum(len(t) for t in texte)


def test_chunks_verlieren_keinen_text() -> None:
    """Der haeufigste stille Fehler beim Chunken: Der letzte, unvollstaendige Chunk faellt
    weg, oder die Ueberlappung frisst Zeichen."""
    text = "".join(f"{i:04d}-" for i in range(1000))  # 5000 Zeichen
    teile = chunks(text, chunk_chars=1200)
    assert "".join(teile) == text, "Zusammengefuegt muessen die Chunks den Text ergeben"
    assert len(teile) == math.ceil(len(text) / 1200)


def test_chunks_bei_leerem_text() -> None:
    assert chunks("", chunk_chars=1200) == []


def test_kurzer_text_ergibt_genau_einen_chunk() -> None:
    assert chunks("kurz", chunk_chars=1200) == ["kurz"]


def test_positionsgewichte_fallen_und_summieren_sich_zu_eins() -> None:
    """Konzept § 6.1: w_i = 1/(1 + i/4), normiert. Frueh zaehlt mehr."""
    gewichte = position_weights(5, decay=4.0)
    assert math.isclose(sum(gewichte), 1.0)
    assert all(a > b for a, b in zip(gewichte, gewichte[1:], strict=True)), (
        "Die Gewichte muessen streng fallen – sonst zaehlt der Dokumentkopf nicht mehr"
    )


def test_positionsgewichte_treffen_die_formel() -> None:
    """Bindet die Formel selbst, nicht nur ihre Form: Eine beliebige andere fallende Folge
    (etwa 1/(i+1)) wuerde den Test oben bestehen."""
    roh = [1.0 / (1.0 + i / 4.0) for i in range(5)]
    erwartet = [w / sum(roh) for w in roh]
    for ist, soll in zip(position_weights(5, decay=4.0), erwartet, strict=True):
        assert math.isclose(ist, soll)


def test_ein_einziger_chunk_bekommt_gewicht_eins() -> None:
    assert position_weights(1, decay=4.0) == [1.0]


def test_kopftext_schneidet_bei_der_vorgabe() -> None:
    text = "x" * 5000
    assert len(head_text(text, head_chars=1000)) == 1000


def test_kopftext_kuerzer_als_vorgabe_bleibt_ganz() -> None:
    assert head_text("kurz", head_chars=1000) == "kurz"
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_features_text.py -v
```

Erwartet: FAIL, `ModuleNotFoundError: No module named 'doccls.features.text'`.

- [ ] **Schritt 3: Implementierung schreiben**

`src/doccls/features/text.py`:

```python
"""Vom Segment zum Eingabetext des Embedders.

Konzept § 6.1: Der Text wird in ~1200-Zeichen-Chunks geschnitten, jeder Chunk eingebettet,
und die Chunk-Vektoren werden positionsgewichtet gemittelt – frühe Chunks zählen mehr, weil
der Dokumentkopf die Klasseninformation fast immer trägt (Betreff, Briefkopf, Überschrift).
Zusätzlich gehen die ersten 1000 Zeichen als eigener Block ein, damit der Mittelwert bei
langen Verträgen nicht genau die Stelle verwässert, auf die es ankommt.

Alle Funktionen hier sind rein: Text rein, Text raus. Kein Modell, kein Dateizugriff.
"""

from doccls.models import Segment


def document_text(segments: list[Segment]) -> str:
    """Segmenttexte in Segmentreihenfolge zu einem Dokumenttext.

    Sortiert wird nach ``index``, nicht nach der Reihenfolge in der Liste: Die Liste kommt
    aus einer Tabellenabfrage und trägt keine zugesicherte Reihenfolge. Die Reihenfolge
    trägt hier aber Bedeutung – die Positionsgewichtung baut darauf auf.
    """
    return " ".join(segment.text for segment in sorted(segments, key=lambda s: s.index))


def chunks(text: str, chunk_chars: int) -> list[str]:
    """Text in gleich lange Stücke schneiden, ohne Überlappung und ohne Rest zu verlieren.

    Keine Überlappung: Sie würde frühe Zeichen doppelt zählen und damit die
    Positionsgewichtung verzerren, die genau diese Gewichtung explizit regelt.
    """
    return [text[start : start + chunk_chars] for start in range(0, len(text), chunk_chars)]


def position_weights(count: int, decay: float) -> list[float]:
    """Gewichte ``w_i = 1 / (1 + i/decay)``, normiert auf Summe 1 (Konzept § 6.1)."""
    roh = [1.0 / (1.0 + i / decay) for i in range(count)]
    gesamt = sum(roh)
    return [w / gesamt for w in roh]


def head_text(text: str, head_chars: int) -> str:
    """Die ersten ``head_chars`` Zeichen als eigener Merkmalsblock."""
    return text[:head_chars]
```

- [ ] **Schritt 4: Tests laufen lassen**

```bash
uv run pytest tests/test_features_text.py -v
```

Erwartet: alle PASS.

- [ ] **Schritt 5: Mutationsprobe**

Mindestens vier Mutationen:

1. `sorted(...)` in `document_text` weglassen → `test_dokumenttext_folgt_der_segmentreihenfolge` rot.
2. In `chunks` den letzten Rest verwerfen (`range(0, len(text) - chunk_chars, chunk_chars)`)
   → `test_chunks_verlieren_keinen_text` rot.
3. In `position_weights` die Normierung weglassen → das Summenkriterium rot.
4. In `position_weights` die Formel auf `1/(i+1)` ändern → `test_positionsgewichte_treffen_die_formel`
   rot, `test_positionsgewichte_fallen_und_summieren_sich_zu_eins` bleibt grün. **Genau dafür
   gibt es zwei Tests.**

- [ ] **Schritt 6: Committen**

```bash
git add src/doccls/features/text.py tests/test_features_text.py
git commit -m "Dokumenttext, Chunks und Positionsgewichte"
```

---

## Aufgabe 4: Embedder-Schnittstelle und E5

**Dateien:**
- Anlegen: `src/doccls/features/embedding.py`
- Anlegen: `tests/test_features_embedding.py`

**Schnittstellen:**
- Nutzt: `E5_MODEL_NAME` aus `doccls.config`
- Liefert:
  - `Embedder` (Protocol) mit `name: str`, `dimension: int`,
    `embed(texts: list[str]) -> npt.NDArray[np.float32]`
  - `E5Embedder(model_name: str = E5_MODEL_NAME)` — implementiert `Embedder`
  - `e5_prefix(text: str) -> str`
  - `document_vector(embedder, text, config) -> npt.NDArray[np.float32]` — Mittelwert- und
    Kopfblock aneinandergehängt, Länge `2 * embedder.dimension`

**Worum es geht:** E5 verlangt das Präfix `passage: ` vor jedem einzubettenden Text
(Konzept §6.1). Wird es beim Training gesetzt und bei der Vorhersage vergessen, driften die
Vektoren auseinander, und niemand merkt es — die Zahlen sehen plausibel aus, sind aber
falsch. Das Präfix gehört deshalb in den Embedder, nicht an die Aufrufstelle.

Die Trennung in Protokoll und Umsetzung ist kein Vorrat: Aufgabe 5 liefert die zweite
Umsetzung, und Aufgabe 15 misst beide gegeneinander. Ohne Protokoll müsste die Messung
zwei Codepfade doppelt führen.

- [ ] **Schritt 1: Test schreiben**

`tests/test_features_embedding.py`:

```python
"""E5 verlangt ein Präfix. Fehlt es an einer Stelle, driften die Vektoren – lautlos."""

import numpy as np
import numpy.typing as npt

from doccls.features import FeatureConfig
from doccls.features.embedding import E5Embedder, document_vector, e5_prefix


class FakeEmbedder:
    """Zeichnet auf, was ihm vorgelegt wird, und liefert bestimmbare Vektoren.

    Damit lässt sich das Zusammenspiel prüfen, ohne ein Modell von 1 GB zu laden – und
    vor allem lässt sich prüfen, WELCHER Text beim Modell ankommt.
    """

    name = "fake"
    dimension = 4

    def __init__(self) -> None:
        self.gesehen: list[str] = []

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        self.gesehen.extend(texts)
        return np.array([[float(len(t)), 1.0, 0.0, 0.0] for t in texts], dtype=np.float32)


def test_praefix_wird_gesetzt() -> None:
    assert e5_prefix("Rechnung") == "passage: Rechnung"


def test_dokumentvektor_legt_jedem_chunk_das_praefix_vor() -> None:
    """Der eigentliche Fehler, den es zu verhindern gilt: ein Chunk ohne Präfix.

    Geprüft wird nicht, DASS die Funktion läuft, sondern welcher Text beim Modell ankommt.
    """
    embedder = FakeEmbedder()
    konfiguration = FeatureConfig(chunk_chars=10, head_chars=10)
    document_vector(embedder, "A" * 35, konfiguration, praefix=e5_prefix)
    assert embedder.gesehen, "Dem Modell wurde gar nichts vorgelegt"
    for text in embedder.gesehen:
        assert text.startswith("passage: "), f"Chunk ohne Praefix: {text[:20]!r}"


def test_dokumentvektor_hat_mittelwert_und_kopfblock() -> None:
    """Konzept § 6.1: zwei Blöcke, aneinandergehängt – nicht einer."""
    embedder = FakeEmbedder()
    konfiguration = FeatureConfig(chunk_chars=10, head_chars=10)
    vektor = document_vector(embedder, "A" * 35, konfiguration, praefix=e5_prefix)
    assert vektor.shape == (2 * embedder.dimension,)


def test_dokumentvektor_gewichtet_frueh_staerker() -> None:
    """Bindet die Gewichtung selbst: Bei ungewichtetem Mittel käme ein anderer Wert heraus.

    Der FakeEmbedder bildet die Chunklänge auf die erste Dimension ab. Chunks: 10, 10, 10,
    5 Zeichen plus je 9 Zeichen Präfix. Ungewichtet wäre der Mittelwert 25,25; gewichtet
    mit w_i = 1/(1+i/4) liegt er höher, weil die langen Chunks vorn stehen.
    """
    embedder = FakeEmbedder()
    konfiguration = FeatureConfig(chunk_chars=10, head_chars=10, position_decay=4.0)
    vektor = document_vector(embedder, "A" * 35, konfiguration, praefix=e5_prefix)
    ungewichtet = (19.0 + 19.0 + 19.0 + 14.0) / 4.0
    assert vektor[0] > ungewichtet, (
        f"Mittelwert {vektor[0]:.2f} liegt nicht ueber dem ungewichteten {ungewichtet:.2f} "
        "– die Positionsgewichtung wirkt nicht"
    )


def test_leerer_text_ergibt_nullvektor_statt_absturz() -> None:
    embedder = FakeEmbedder()
    konfiguration = FeatureConfig(chunk_chars=10, head_chars=10)
    vektor = document_vector(embedder, "", konfiguration, praefix=e5_prefix)
    assert vektor.shape == (2 * embedder.dimension,)
    assert not np.any(np.isnan(vektor)), "NaN im Vektor – das vergiftet jedes Training"


def test_e5_trennt_aehnliche_von_unaehnlichen_texten() -> None:
    """Der einzige Test, der das echte Modell lädt – und der einzige, der beweist, dass
    die Einbettung überhaupt etwas bedeutet.

    Beim ersten Lauf lädt sentence-transformers rund 1 GB nach. Eine Rechnung muss einer
    anderen Rechnung näher sein als einem Mietvertrag. Waere das nicht so, waere jede
    spaetere Zahl wertlos, und kein anderer Test wuerde es bemerken.
    """
    embedder = E5Embedder()
    rechnung_a = "Rechnung RE-2026-4711 ueber 1.190,00 EUR, zahlbar bis 30.04.2026."
    rechnung_b = "Rechnung RE-2026-0815 ueber 2.380,00 EUR, Zahlungsziel 14 Tage netto."
    vertrag = "Mietvertrag ueber Gewerberaeume, Kuendigungsfrist drei Monate zum Quartal."

    vektoren = embedder.embed([e5_prefix(t) for t in (rechnung_a, rechnung_b, vertrag)])
    normiert = vektoren / np.linalg.norm(vektoren, axis=1, keepdims=True)
    aehnlich = float(normiert[0] @ normiert[1])
    unaehnlich = float(normiert[0] @ normiert[2])

    assert aehnlich > unaehnlich, (
        f"Rechnung/Rechnung {aehnlich:.3f} liegt nicht ueber Rechnung/Vertrag "
        f"{unaehnlich:.3f} – das Embedding traegt keine Bedeutung"
    )
    assert aehnlich - unaehnlich > 0.02, (
        f"Abstand nur {aehnlich - unaehnlich:.4f}. Kosinuswerte in Satz-Embeddings sind "
        "stark gestaucht (Konzept Anhang D), aber so eng traegt das Signal nicht."
    )


def test_e5_liefert_die_zugesicherte_dimension() -> None:
    embedder = E5Embedder()
    assert embedder.dimension == 768
    assert embedder.embed(["passage: test"]).shape == (1, 768)
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_features_embedding.py -v
```

Erwartet: FAIL, `ModuleNotFoundError: No module named 'doccls.features.embedding'`.

- [ ] **Schritt 3: Implementierung schreiben**

`src/doccls/features/embedding.py`:

```python
"""Das semantische Merkmal: ein Vektor je Dokument.

Zwei Umsetzungen hinter einem Protokoll, weil Konzept Anhang B die Wahl zwischen E5 und
BGE-M3 ausdrücklich **gemessen** und nicht entschieden haben will. Aufgabe 15 misst beide
gegeneinander; ohne gemeinsame Schnittstelle liefe die Messung über zwei Codepfade.

E5 verlangt das Präfix ``passage: `` vor jedem Text (Konzept § 6.1). Das Präfix sitzt
deshalb hier und nicht an der Aufrufstelle: Wird es beim Training gesetzt und bei der
Vorhersage vergessen, driften die Vektoren auseinander – und die Zahlen sehen weiter
plausibel aus.
"""

from collections.abc import Callable
from typing import Protocol

import numpy as np
import numpy.typing as npt

from doccls.config import E5_MODEL_NAME
from doccls.features import FeatureConfig
from doccls.features.text import chunks, head_text, position_weights

Vektoren = npt.NDArray[np.float32]


class Embedder(Protocol):
    """Was ein Embedder können muss – mehr braucht die Merkmalsbildung nicht."""

    name: str
    dimension: int

    def embed(self, texts: list[str]) -> Vektoren:
        """Eine Zeile je Text, ``dimension`` Spalten."""
        ...


def e5_prefix(text: str) -> str:
    """E5 unterscheidet Anfrage und Inhalt. Hier ist durchgehend alles Inhalt."""
    return f"passage: {text}"


def ohne_praefix(text: str) -> str:
    """BGE-M3 kennt keine Präfixe (Konzept Anhang B)."""
    return text


def document_vector(
    embedder: Embedder,
    text: str,
    config: FeatureConfig,
    praefix: Callable[[str], str],
) -> Vektoren:
    """Zwei Blöcke aneinandergehängt: gewichteter Mittelwert über alle Chunks, dann Kopf.

    Der zweite Block ist kein Luxus: Bei einem langen Vertrag verwässert der Mittelwert
    genau die Stelle, an der die Klasse steht (Konzept § 6.1).
    """
    stuecke = chunks(text, config.chunk_chars)
    if not stuecke:
        return np.zeros(2 * embedder.dimension, dtype=np.float32)

    chunk_vektoren = embedder.embed([praefix(s) for s in stuecke])
    gewichte = np.array(position_weights(len(stuecke), config.position_decay), dtype=np.float32)
    mittel = gewichte @ chunk_vektoren

    kopf = embedder.embed([praefix(head_text(text, config.head_chars))])[0]
    return np.concatenate([mittel, kopf]).astype(np.float32)


class E5Embedder:
    """``intfloat/multilingual-e5-base`` über sentence-transformers, lokal.

    Das Modell wird beim ersten Zugriff geladen, nicht im Konstruktor: Ein Import dieses
    Moduls soll nicht 1 GB nachladen, nur weil irgendwo eine Typannotation gebraucht wird.
    """

    name = "e5"
    dimension = 768

    def __init__(self, model_name: str = E5_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: object | None = None

    def _geladen(self) -> "SentenceTransformer":
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self._model_name, device=_geraet())
        assert isinstance(self._model, SentenceTransformer)
        return self._model

    def embed(self, texts: list[str]) -> Vektoren:
        roh = self._geladen().encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(roh, dtype=np.float32)


def _geraet() -> str:
    """Auf Apple Silicon über MPS, sonst CPU. CUDA gibt es auf diesem Rechner nicht."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"
```

**Hinweis zur Typannotation von `_geladen`:** Der Vorwärtsverweis `"SentenceTransformer"`
braucht einen Import unter `if TYPE_CHECKING:` am Kopf der Datei. Das `assert isinstance`
darin ist eine Notlösung und **nicht erwünscht** — nacktes `assert` verschwindet unter
`python -O`. Sauber: `self._model` direkt als `SentenceTransformer | None` annotieren
(unter `TYPE_CHECKING` importiert) und das `assert` weglassen. Falls mypy dann meckert,
melden statt unterdrücken.

- [ ] **Schritt 4: Tests laufen lassen**

```bash
uv run pytest tests/test_features_embedding.py -v
```

Erwartet: alle PASS. Der erste Lauf dauert einige Minuten (Modelldownload).

- [ ] **Schritt 5: Mutationsprobe**

Mindestens fünf Mutationen:

1. `praefix` in `document_vector` ignorieren (roh `embedder.embed(stuecke)`) →
   `test_dokumentvektor_legt_jedem_chunk_das_praefix_vor` rot.
2. Den Kopfblock weglassen (nur `mittel` zurückgeben) → der Formtest rot.
3. Gewichte durch gleiche Gewichte ersetzen → `test_dokumentvektor_gewichtet_frueh_staerker` rot.
4. `if not stuecke` entfernen → `test_leerer_text_ergibt_nullvektor_statt_absturz` rot.
5. Im E5-Embedder alle Vektoren durch Nullen ersetzen →
   `test_e5_trennt_aehnliche_von_unaehnlichen_texten` rot. **Diese Mutation ist die
   wichtigste:** Sie prüft, ob überhaupt irgendein Test bemerkt, dass das Embedding nichts
   bedeutet.

- [ ] **Schritt 6: Committen**

```bash
git add src/doccls/features/embedding.py tests/test_features_embedding.py
git commit -m "Embedder-Protokoll und E5 mit Praefix im Embedder"
```

---

## Aufgabe 5: BGE-M3 über Ollama als zweite Umsetzung

**Dateien:**
- Ändern: `src/doccls/features/embedding.py`
- Ändern: `tests/test_features_embedding.py`

**Schnittstellen:**
- Liefert: `BgeM3Embedder(url: str = OLLAMA_URL, model: str = BGE_M3_MODEL_NAME)` —
  implementiert `Embedder`, `dimension = 1024`
- Liefert: `make_embedder(config: FeatureConfig) -> Embedder`

**Worum es geht:** Konzept Anhang B stellt E5 und BGE-M3 gegenüber und schließt, die Wahl
solle gemessen werden. BGE-M3 liegt auf diesem Rechner bereits in Ollama
(`bge-m3:567m-fp16`, 1,2 GB) und bringt 1024 Dimensionen und 8192 Token Kontext mit — die
meisten Dokumente passen ohne Chunking hinein. Dafür hängt die Merkmalsbildung an einem
laufenden Dienst.

Der Aufruf geht über `urllib.request` aus der Standardbibliothek. Eine HTTP-Bibliothek für
einen POST mit JSON-Rumpf wäre nicht zu rechtfertigen.

- [ ] **Schritt 1: Test schreiben**

An `tests/test_features_embedding.py` anhängen:

```python
def ollama_laeuft() -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


def test_make_embedder_folgt_der_konfiguration() -> None:
    """Die Wahl steht in features.yaml und geht in die feature_version ein – sie darf
    nirgends im Code fest verdrahtet sein."""
    assert make_embedder(FeatureConfig(embedder="e5")).name == "e5"
    assert make_embedder(FeatureConfig(embedder="bge-m3")).name == "bge-m3"


@pytest.mark.skipif(not ollama_laeuft(), reason="Ollama laeuft nicht")
def test_bge_m3_trennt_aehnliche_von_unaehnlichen_texten() -> None:
    """Dieselbe Prüfung wie bei E5, damit der Vergleich in Aufgabe 15 auf gleichem Grund
    steht: Wenn eine der beiden Umsetzungen nichts bedeutet, muss es hier auffallen."""
    embedder = BgeM3Embedder()
    rechnung_a = "Rechnung RE-2026-4711 ueber 1.190,00 EUR, zahlbar bis 30.04.2026."
    rechnung_b = "Rechnung RE-2026-0815 ueber 2.380,00 EUR, Zahlungsziel 14 Tage netto."
    vertrag = "Mietvertrag ueber Gewerberaeume, Kuendigungsfrist drei Monate zum Quartal."

    vektoren = embedder.embed([rechnung_a, rechnung_b, vertrag])
    assert vektoren.shape == (3, 1024)
    normiert = vektoren / np.linalg.norm(vektoren, axis=1, keepdims=True)
    aehnlich = float(normiert[0] @ normiert[1])
    unaehnlich = float(normiert[0] @ normiert[2])
    assert aehnlich > unaehnlich, (
        f"Rechnung/Rechnung {aehnlich:.3f} liegt nicht ueber Rechnung/Vertrag {unaehnlich:.3f}"
    )


@pytest.mark.skipif(not ollama_laeuft(), reason="Ollama laeuft nicht")
def test_bge_m3_bekommt_kein_e5_praefix() -> None:
    """BGE-M3 kennt keine Präfixe. Stünde 'passage: ' davor, waere es schlicht Text im
    Dokument und verschoebe jeden Vektor – ohne dass irgendetwas fehlschluege."""
    embedder = BgeM3Embedder()
    mit = embedder.embed(["passage: Rechnung ueber 100 EUR"])[0]
    ohne = embedder.embed(["Rechnung ueber 100 EUR"])[0]
    assert not np.allclose(mit, ohne), "Das Praefix ist folgenlos – dann stimmt der Aufruf nicht"


def test_ollama_nicht_erreichbar_sagt_es_deutlich() -> None:
    """Ein Verbindungsfehler darf nicht als leerer Vektor durchgehen – sonst trainierte
    das Modell auf Nullen und niemand saehe es."""
    embedder = BgeM3Embedder(url="http://localhost:1")
    with pytest.raises(RuntimeError, match="Ollama"):
        embedder.embed(["test"])
```

Zusätzliche Importe am Kopf der Testdatei:

```python
import pytest

from doccls.config import OLLAMA_URL
from doccls.features.embedding import BgeM3Embedder, make_embedder
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_features_embedding.py -v -k "bge or make_embedder or ollama"
```

Erwartet: FAIL, `ImportError: cannot import name 'BgeM3Embedder'`.

- [ ] **Schritt 3: Implementierung schreiben**

An `src/doccls/features/embedding.py` anhängen:

```python
class BgeM3Embedder:
    """``bge-m3`` über einen lokal laufenden Ollama-Dienst.

    1024 Dimensionen, 8192 Token Kontext, **keine Präfixe** (Konzept Anhang B). Der
    größere Kontext heißt: Die meisten Dokumente passen in einen einzigen Chunk. Das ist
    kein Freifahrtschein, die Chunk-Logik zu umgehen – sie bleibt dieselbe, damit die
    beiden Umsetzungen in Aufgabe 15 unter gleichen Bedingungen verglichen werden.
    """

    name = "bge-m3"
    dimension = 1024

    def __init__(self, url: str = OLLAMA_URL, model: str = BGE_M3_MODEL_NAME) -> None:
        self._url = url.rstrip("/")
        self._model = model

    def embed(self, texts: list[str]) -> Vektoren:
        rumpf = json.dumps({"model": self._model, "input": texts}).encode("utf-8")
        anfrage = urllib.request.Request(
            f"{self._url}/api/embed",
            data=rumpf,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(anfrage, timeout=120) as antwort:
                daten = json.loads(antwort.read())
        except (urllib.error.URLError, OSError) as fehler:
            raise RuntimeError(
                f"Ollama unter {self._url} nicht erreichbar: {fehler}. "
                "Ein Verbindungsfehler darf nicht als leerer Vektor durchgehen – das "
                "Modell wuerde sonst auf Nullen trainieren."
            ) from fehler
        return np.asarray(daten["embeddings"], dtype=np.float32)


def make_embedder(config: FeatureConfig) -> Embedder:
    """Den in ``features.yaml`` gewählten Embedder bauen.

    Die Wahl geht über ``feature_version`` in die Merkmale ein; sie darf deshalb nirgends
    im Code fest verdrahtet sein.
    """
    match config.embedder:
        case "e5":
            return E5Embedder()
        case "bge-m3":
            return BgeM3Embedder()


def praefix_fuer(embedder: Embedder) -> Callable[[str], str]:
    """Das Präfix gehört zum Modell, nicht zur Aufrufstelle."""
    return e5_prefix if embedder.name == "e5" else ohne_praefix
```

Zusätzliche Importe am Kopf des Moduls:

```python
import json
import urllib.error
import urllib.request

from doccls.config import BGE_M3_MODEL_NAME, E5_MODEL_NAME, OLLAMA_URL
```

- [ ] **Schritt 4: Tests laufen lassen**

```bash
ollama list
uv run pytest tests/test_features_embedding.py -v
```

Erwartet: alle PASS. Laufen die BGE-Tests als „skipped" durch, obwohl Ollama läuft, ist
`ollama_laeuft()` falsch — das dann klären, nicht hinnehmen.

- [ ] **Schritt 5: Mutationsprobe**

1. `make_embedder` immer `E5Embedder` liefern lassen → `test_make_embedder_folgt_der_konfiguration` rot.
2. Im Fehlerpfad `np.zeros(...)` statt `raise` → `test_ollama_nicht_erreichbar_sagt_es_deutlich` rot.
3. `praefix_fuer` immer `e5_prefix` liefern lassen → prüfen, **ob überhaupt ein Test rot
   wird.** Wird keiner rot, fehlt ein Test auf Ebene von `build_features` (Aufgabe 8); das
   dann dort nachtragen und hier im Bericht vermerken.

- [ ] **Schritt 6: Committen**

```bash
git add src/doccls/features/embedding.py tests/test_features_embedding.py
git commit -m "BGE-M3 ueber Ollama als zweite Embedder-Umsetzung"
```

---

## Aufgabe 6: Mengen laden — und das Gold-Set im Code schützen

**Dateien:**
- Anlegen: `src/doccls/splits.py`
- Anlegen: `tests/test_splits.py`

**Schnittstellen:**
- Nutzt: `read_table` aus `doccls.pipeline`, `PARQUET_DIR`/`GENERATED_DIR` aus `doccls.config`
- Liefert:
  - `labelled_documents() -> pl.DataFrame` — Spalten `document_id`, `source_path`,
    `class_key`, `template_id`, `split`
  - `training_documents()`, `calibration_documents()`, `gold_documents()` — je ein Teil davon
  - `assert_no_gold(frame: pl.DataFrame, wofuer: str) -> None`

**Worum es geht:** Konzept §9.2 sagt: „Nie im Training. Nie in der Kalibriermenge. Nie von
der Prüfliste wählbar. Die Auswahlfunktion filtert Gold-Set-IDs hart heraus — **im Code,
nicht per Konvention**."

Das ist der Satz, auf dem die Glaubwürdigkeit jeder späteren Zahl ruht. Ein Kommentar
„hier bitte kein Gold" wird in der dritten Umbauwelle überlesen. Eine Funktion, die wirft,
nicht.

- [ ] **Schritt 1: Test schreiben**

`tests/test_splits.py`:

```python
"""Das Gold-Set ist eingefroren. Diese Zusicherung wird geprueft, nicht geglaubt."""

import polars as pl
import pytest

from doccls.splits import (
    assert_no_gold,
    calibration_documents,
    gold_documents,
    labelled_documents,
    training_documents,
)


def test_jede_menge_ist_nicht_leer() -> None:
    """Eine leere Menge wuerde jeden folgenden Test stillschweigend bestehen lassen."""
    for name, rahmen in [
        ("train", training_documents()),
        ("calib", calibration_documents()),
        ("gold", gold_documents()),
    ]:
        assert rahmen.height > 0, f"Menge {name} ist leer"


def test_die_mengen_ueberschneiden_sich_nicht() -> None:
    ids = [set(f["document_id"]) for f in
           (training_documents(), calibration_documents(), gold_documents())]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])


def test_keine_vorlage_liegt_in_zwei_mengen() -> None:
    """Schaerfer als die Dokument-Pruefung darueber: Zehn Varianten einer Vorlage sind
    praktisch dasselbe Dokument. Laegen sie auf beiden Seiten, waere jede Metrik geschoent
    (Konzept § 9.2)."""
    vorlagen = [set(f["template_id"]) for f in
                (training_documents(), calibration_documents(), gold_documents())]
    assert vorlagen[0].isdisjoint(vorlagen[1])
    assert vorlagen[0].isdisjoint(vorlagen[2])
    assert vorlagen[1].isdisjoint(vorlagen[2])


def test_training_enthaelt_kein_gold() -> None:
    assert training_documents().filter(pl.col("split") == "gold").height == 0


def test_kalibrierung_enthaelt_kein_gold() -> None:
    assert calibration_documents().filter(pl.col("split") == "gold").height == 0


def test_assert_no_gold_wirft_bei_gold() -> None:
    """Der Waechter selbst. Ohne diesen Test waere nicht geprueft, ob er ueberhaupt greift."""
    verseucht = pl.DataFrame({"document_id": ["a", "b"], "split": ["train", "gold"]})
    with pytest.raises(ValueError, match="Gold"):
        assert_no_gold(verseucht, "Training")


def test_assert_no_gold_nennt_die_schuldigen_ids() -> None:
    """Eine Meldung ohne die betroffene ID zwingt zur Suche von Hand."""
    verseucht = pl.DataFrame({"document_id": ["harmlos", "SCHULDIG"], "split": ["train", "gold"]})
    with pytest.raises(ValueError, match="SCHULDIG"):
        assert_no_gold(verseucht, "Training")


def test_assert_no_gold_laesst_saubere_mengen_durch() -> None:
    sauber = pl.DataFrame({"document_id": ["a"], "split": ["train"]})
    assert_no_gold(sauber, "Training")


def test_jedes_eingelesene_dokument_traegt_eine_klasse() -> None:
    """Bindet die Menge, nicht nur die Form: Faellt ein join daneben, schrumpft diese
    Tabelle stillschweigend, und alle Metriken danach messen auf weniger Dokumenten."""
    from doccls.config import PARQUET_DIR
    from doccls.pipeline import read_table

    dokumente = read_table(PARQUET_DIR, "documents")
    assert labelled_documents().height == dokumente.height, (
        "Nicht jedes eingelesene Dokument hat eine Klasse – vermutlich fehlen "
        "Manifestzeilen (siehe Aufgabe 1)"
    )


def test_gold_hat_mindestens_50_dokumente_je_klasse() -> None:
    """Konzept § 9.2: Bei weniger sind Unterschiede von zwei Punkten F1 nicht von
    Rauschen zu unterscheiden. Diese Schranke haelt fest, wann das Gold-Set zu klein wird –
    etwa wenn spaeter ein Format aus der Messung genommen wird."""
    je_klasse = gold_documents().group_by("class_key").len()
    zu_klein = je_klasse.filter(pl.col("len") < 50)
    assert zu_klein.height == 0, f"Zu kleine Gold-Klassen: {zu_klein.to_dicts()}"
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_splits.py -v
```

Erwartet: FAIL, `ModuleNotFoundError: No module named 'doccls.splits'`.

- [ ] **Schritt 3: Implementierung schreiben**

`src/doccls/splits.py`:

```python
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
```

- [ ] **Schritt 4: Tests laufen lassen**

```bash
uv run pytest tests/test_splits.py -v
```

Erwartet: alle PASS.

- [ ] **Schritt 5: Mutationsprobe**

1. In `assert_no_gold` den `raise` weglassen → drei Tests rot.
2. Die IDs aus der Meldung nehmen → `test_assert_no_gold_nennt_die_schuldigen_ids` rot.
3. In `training_documents` auf `!= CALIB` filtern statt auf `== TRAIN` (Gold käme mit
   herein) → `test_training_enthaelt_kein_gold` rot.
4. In `labelled_documents` `how="inner"` statt `how="left"` und die Prüfung weglassen →
   `test_jedes_eingelesene_dokument_traegt_eine_klasse` rot. **Diese ist die wichtigste:**
   Sie ist genau der stille Verlust, den Aufgabe 1 behoben hat.
5. Die Gold-Schranke 50 auf 10 senken → prüfen, ob der Test überhaupt etwas bindet
   (er darf nicht schon bei 10 grün sein, ohne dass die Zahl stimmt).

- [ ] **Schritt 6: Committen**

```bash
git add src/doccls/splits.py tests/test_splits.py
git commit -m "Mengen laden und das Gold-Set im Code schuetzen"
```

---

## Aufgabe 7: Zeichen-n-Gramme und SVD

**Dateien:**
- Anlegen: `src/doccls/features/ngrams.py`
- Anlegen: `tests/test_features_ngrams.py`

**Schnittstellen:**
- Nutzt: `FeatureConfig`
- Liefert:
  - `NgramBlock` — mit `fit(texts: list[str]) -> None`,
    `transform(texts: list[str]) -> npt.NDArray[np.float32]`, `dimension: int`,
    `tfidf_only(texts) -> NDArray` und `vokabular() -> list[str]` (beide nur für
    Tests und Fehlersuche, beide prüfen vorher auf `fit`)
  - `build_ngram_block(config: FeatureConfig) -> NgramBlock`

**Worum es geht:** Konzept §6.2 und Anhang A. TF-IDF über **Zeichen**-n-Gramme (3–5) löst
deutsche Komposita, ohne sie zu zerlegen: „Rechnung" und „Rechnungsbetrag" teilen sich
Zeichenfolgen, aber kein Wort. Vokabular auf 50 000 begrenzt, danach `TruncatedSVD` auf 256
Dimensionen, damit der Block die dichten Embeddings nicht erschlägt.

**Die Fallgrube dieser Aufgabe** ist keine Formel, sondern eine Reihenfolge: `fit` darf
**ausschließlich** auf den Trainingstexten laufen. Wer TF-IDF und SVD auf allen Dokumenten
anpasst und danach aufteilt, hat dem Modell die Wortstatistik des Gold-Sets mitgegeben.
Die Zahlen sehen dann besser aus, als sie sind, und kein einzelner Test bemerkt es.

- [ ] **Schritt 1: Test schreiben**

`tests/test_features_ngrams.py`:

```python
"""Zeichen-n-Gramme loesen Komposita. Und: fit sieht nur die Trainingstexte."""

import numpy as np
import pytest

from doccls.features import FeatureConfig
from doccls.features.ngrams import build_ngram_block

TRAININGSTEXTE = [
    "Rechnung ueber Wartungsarbeiten, Rechnungsbetrag 1.190,00 EUR, Zahlungsziel 14 Tage",
    "Gutschrift zur Rechnung RE-2026-0815, Erstattungsbetrag 240,00 EUR wegen Maengelruege",
    "Mietvertrag ueber Gewerberaeume, Kuendigungsfrist drei Monate zum Quartalsende",
    "Allgemeine Geschaeftsbedingungen, Paragraph 1 Geltungsbereich, Paragraph 2 Lieferung",
    "Protokoll der Sitzung, TOP 1 Begruessung, Anwesend waren die Mitglieder des Beirats",
    "Statusbericht Quartal zwei, Ampel gruen, Fortschritt planmaessig, keine Eskalation",
]


def block(**ueberschreibungen: object) -> object:
    konfiguration = FeatureConfig(svd_components=4, ngram_max_features=2000, **ueberschreibungen)
    return build_ngram_block(konfiguration)


def test_transform_liefert_die_zugesicherte_breite() -> None:
    b = block()
    b.fit(TRAININGSTEXTE)
    assert b.transform(TRAININGSTEXTE).shape == (len(TRAININGSTEXTE), b.dimension)
    assert b.dimension == 4


def test_kompositum_liegt_naeher_am_grundwort_als_an_fremdem_wort() -> None:
    """Der Kern von Konzept Anhang A: 'Rechnungsbetrag' teilt Zeichenfolgen mit
    'Rechnung', aber kein Wort. Ein Wort-n-Gramm-Modell saehe hier gar nichts.

    Geprueft wird auf der TF-IDF-Ebene vor der SVD, weil die SVD bei sechs Texten zu
    wenig zu tun hat, um das Verhaeltnis stabil zu erhalten.
    """
    b = block()
    b.fit(TRAININGSTEXTE)
    roh = b.tfidf_only(["Rechnung", "Rechnungsbetrag", "Kuendigungsfrist"])
    normiert = roh / np.linalg.norm(roh, axis=1, keepdims=True)
    verwandt = float(normiert[0] @ normiert[1])
    fremd = float(normiert[0] @ normiert[2])
    assert verwandt > fremd, (
        f"Rechnung/Rechnungsbetrag {verwandt:.3f} liegt nicht ueber "
        f"Rechnung/Kuendigungsfrist {fremd:.3f} – die Zeichen-n-Gramme wirken nicht"
    )


def test_tippfehler_bleibt_nah_am_original() -> None:
    """Die zweite Begruendung aus Konzept § 6.2: OCR-Robustheit. 'Rechnunq' ist fuer ein
    Wortmodell ein unbekanntes Wort, fuer ein Zeichenmodell fast dasselbe."""
    b = block()
    b.fit(TRAININGSTEXTE)
    roh = b.tfidf_only(["Rechnung", "Rechnunq", "Mietvertrag"])
    normiert = roh / np.linalg.norm(roh, axis=1, keepdims=True)
    assert float(normiert[0] @ normiert[1]) > float(normiert[0] @ normiert[2])


def test_transform_ohne_fit_wirft() -> None:
    """Ein nicht angepasster Block darf keine Nullen liefern – das traegt sich lautlos
    durch bis in die Metriken."""
    with pytest.raises(RuntimeError, match="fit"):
        block().transform(TRAININGSTEXTE)


def test_fit_ist_reproduzierbar() -> None:
    """TruncatedSVD ist randomisiert. Ohne festen Seed waeren zwei Laeufe unvergleichbar."""
    a, b = block(), block()
    a.fit(TRAININGSTEXTE)
    b.fit(TRAININGSTEXTE)
    assert np.allclose(a.transform(TRAININGSTEXTE), b.transform(TRAININGSTEXTE))


def test_unbekannter_text_ergibt_keinen_nullvektor() -> None:
    """Ein Text aus lauter ungesehenen Zeichenfolgen wuerde zu Recht nahe null liegen –
    ein deutscher Satz mit bekannten Silben aber nicht. Faellt er trotzdem auf null,
    stimmt die Anwendung des angepassten Vokabulars nicht."""
    b = block()
    b.fit(TRAININGSTEXTE)
    vektor = b.transform(["Schlussrechnung fuer Wartungsleistungen im Quartal"])[0]
    assert np.linalg.norm(vektor) > 0.0
```

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_features_ngrams.py -v
```

Erwartet: FAIL, `ModuleNotFoundError: No module named 'doccls.features.ngrams'`.

- [ ] **Schritt 3: Implementierung schreiben**

`src/doccls/features/ngrams.py`:

```python
"""Der Wortlaut-Block: TF-IDF über Zeichen-n-Gramme, danach SVD.

Konzept § 6.2 und Anhang A. Zeichen-n-Gramme statt Wort-n-Gramme aus zwei deutschen
Gründen: Komposita („Rechnung" und „Rechnungsbetrag" teilen Zeichenfolgen, aber kein Wort)
und OCR-Robustheit („Rechnunq" ist für ein Zeichenmodell fast dasselbe wie „Rechnung").

**Die Reihenfolge ist der kritische Teil dieses Moduls.** ``fit`` darf ausschließlich
Trainingstexte sehen. Wer TF-IDF und SVD auf allen Dokumenten anpasst und erst danach
aufteilt, hat dem Modell die Wortstatistik des Gold-Sets mitgegeben – die Zahlen sehen dann
besser aus, als sie sind. Das ist keine Vorsichtsmaßnahme, sondern der häufigste stille
Fehler dieser Art von Pipeline.
"""

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from doccls.features import FeatureConfig

SVD_SEED = 7
"""Fest, damit zwei Läufe vergleichbar sind. ``TruncatedSVD`` ist randomisiert."""


class NgramBlock:
    """TF-IDF über Zeichen-n-Gramme, auf ``svd_components`` Dimensionen verdichtet."""

    def __init__(self, config: FeatureConfig) -> None:
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(config.ngram_min, config.ngram_max),
            max_features=config.ngram_max_features,
            lowercase=True,
        )
        self._svd = TruncatedSVD(n_components=config.svd_components, random_state=SVD_SEED)
        self.dimension = config.svd_components
        self._angepasst = False

    def fit(self, texts: list[str]) -> None:
        """Nur mit Trainingstexten aufrufen – siehe Moduldoc."""
        matrix = self._vectorizer.fit_transform(texts)
        self._svd.fit(matrix)
        self._angepasst = True

    def _pruefe_angepasst(self) -> None:
        if not self._angepasst:
            raise RuntimeError(
                "NgramBlock wurde nicht angepasst – erst fit(trainingstexte) aufrufen. "
                "Ein nicht angepasster Block wuerde Nullen liefern, und die traegen sich "
                "lautlos bis in die Metriken durch."
            )

    def transform(self, texts: list[str]) -> npt.NDArray[np.float32]:
        self._pruefe_angepasst()
        matrix = self._vectorizer.transform(texts)
        return np.asarray(self._svd.transform(matrix), dtype=np.float32)

    def tfidf_only(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Die TF-IDF-Matrix vor der SVD – nur für Tests und zur Fehlersuche.

        Bei wenigen Texten hat die SVD zu wenig zu tun, um Verhältnisse zwischen
        Textpaaren stabil zu erhalten; für die Prüfung der n-Gramm-Eigenschaft ist die
        rohe Matrix deshalb die ehrlichere Ebene.
        """
        self._pruefe_angepasst()
        return np.asarray(self._vectorizer.transform(texts).todense(), dtype=np.float32)


def build_ngram_block(config: FeatureConfig) -> NgramBlock:
    return NgramBlock(config)
```

**Zu `analyzer="char_wb"`:** Das begrenzt die n-Gramme auf Wortgrenzen und vermeidet
n-Gramme, die über Leerzeichen hinweg zwei Wörter verkleben. Falls sich in Aufgabe 16 zeigt,
dass `analyzer="char"` besser misst, ist das eine Änderung an `features.yaml` wert — dann
aber als Parameter, der in die `feature_version` eingeht, nicht als stille Codeänderung.

- [ ] **Schritt 4: Tests laufen lassen**

```bash
uv run pytest tests/test_features_ngrams.py -v
```

Erwartet: alle PASS.

- [ ] **Schritt 5: Mutationsprobe**

1. `analyzer="char_wb"` auf `analyzer="word"` → `test_kompositum_liegt_naeher_am_grundwort_als_an_fremdem_wort`
   und `test_tippfehler_bleibt_nah_am_original` rot. **Das ist die Kernmutation** — sie
   prüft, ob überhaupt ein Test die Begründung des ganzen Blocks bindet.
2. `random_state=SVD_SEED` entfernen → `test_fit_ist_reproduzierbar` rot.
3. Den `raise` in `_pruefe_angepasst` durch `return` ersetzen → `test_transform_ohne_fit_wirft` rot.
4. In `transform` `fit_transform` statt `transform` aufrufen (die klassische Leckage) →
   prüfen, **ob irgendein Test rot wird.** Wird keiner rot, ist das ein Befund: Die
   Leckageprüfung gehört auf die Ebene von Aufgabe 9, wo Training und Gold getrennt
   vorliegen. Dort dann nachtragen und hier im Bericht vermerken.

- [ ] **Schritt 6: Committen**

```bash
git add src/doccls/features/ngrams.py tests/test_features_ngrams.py
git commit -m "Zeichen-n-Gramme mit SVD als Wortlaut-Block"
```

---

## Aufgabe 8: Strukturmerkmale

**Dateien:**
- Anlegen: `src/doccls/features/structural.py`
- Anlegen: `tests/test_features_structural.py`

**Schnittstellen:**
- Nutzt: `Document`, `Segment` aus `doccls.models`
- Liefert:
  - `STRUCTURAL_NAMES: tuple[str, ...]` — die Merkmalsnamen in fester Reihenfolge
  - `structural_features(document: Document, segments: list[Segment]) -> npt.NDArray[np.float32]`

**Worum es geht:** Konzept §6.3 — rund 40 billige, erklärbare Zahlen. Sie sind **Merkmale,
keine Regeln**: Niemand schreibt `if IBAN then Rechnung`, das Modell entscheidet über das
Gewicht. Ihr Wert liegt darin, dass sie sehen, was ein gemitteltes Embedding übersieht —
ein Vorzeichen, eine IBAN, eine Unterschriftenzeile.

**Was aus §6.3 bewusst nicht umgesetzt wird, und warum:**

- **Dateinamen-Tokens.** In Phase 1 wurden die Dateinamen neutralisiert (`doc-0001.pdf`),
  weil sich aus ihnen 480 von 560 Klassen raten ließen. Ein Merkmal aus diesen Namen trüge
  jetzt exakt null Information. Es wird nicht gebaut — ein Merkmal, das nichts misst,
  verwässert nur die Gewichte.
- **Absenderdomain und Betreff-Tokens der Elternmail.** Brauchen einen Rückgriff auf das
  Elterndokument, den die Merkmalsbildung bisher nicht hat. `is_attachment` bildet den
  strukturellen Teil davon ab. Die Vererbung wird auf Phase 3 vertagt — dort, wo die
  Mailvorlagen ohnehin überarbeitet werden müssen (Körper und Anhang sind derzeit zu 0,95
  identisch). **Nicht vergessen:** in `docs/phase-1-offene-punkte.md` vermerken.

- [ ] **Schritt 1: Test schreiben**

`tests/test_features_structural.py`:

```python
"""Rund 40 erklaerbare Zahlen – und die Pruefung, dass keine davon die Klasse verraet."""

from collections import Counter, defaultdict
from datetime import UTC, datetime

import numpy as np
import polars as pl

from doccls.config import PARQUET_DIR
from doccls.features.structural import STRUCTURAL_NAMES, structural_features
from doccls.models import Document, Segment, SegmentKind
from doccls.pipeline import read_table
from doccls.splits import labelled_documents


def dokument(**ueberschreibungen: object) -> Document:
    vorgabe: dict[str, object] = {
        "source_path": "pdf/doc-0001.pdf",
        "media_type": "application/pdf",
        "content": b"x",
        "ingested_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    return Document.create(**{**vorgabe, **ueberschreibungen})


def segment(index: int, text: str, kind: SegmentKind = SegmentKind.ABSCHNITT) -> Segment:
    return Segment(
        segment_id=f"s{index}", document_id="d1", index=index, kind=kind,
        locator=f"L{index}", heading=None, text=text,
    )


def merkmal(name: str, dokument_: Document, segmente: list[Segment]) -> float:
    return float(structural_features(dokument_, segmente)[STRUCTURAL_NAMES.index(name)])


def test_laenge_stimmt_mit_den_namen_ueberein() -> None:
    """Ein Vektor, der nicht zu seinen Namen passt, macht jede Gewichtsauswertung
    (Konzept § 7.1: coef_ gegen die Merkmalsnamen) zu Unsinn – und zwar lautlos."""
    vektor = structural_features(dokument(), [segment(0, "Text")])
    assert vektor.shape == (len(STRUCTURAL_NAMES),)


def test_namen_sind_eindeutig() -> None:
    doppelt = [n for n, anzahl in Counter(STRUCTURAL_NAMES).items() if anzahl > 1]
    assert not doppelt, f"Doppelte Merkmalsnamen: {doppelt}"


def test_es_sind_rund_vierzig_merkmale() -> None:
    """Konzept § 6.3 nennt 'rund 40'. Die Schranke faengt ein versehentliches Abschneiden
    des Blocks – etwa wenn eine Gruppe beim Umbau herausfaellt."""
    assert 30 <= len(STRUCTURAL_NAMES) <= 50, f"{len(STRUCTURAL_NAMES)} Merkmale"


def test_betraege_werden_gezaehlt() -> None:
    text = "Positionen: 1.190,00 EUR und 240,50 EUR sowie 15,00 €"
    assert merkmal("amount_count", dokument(), [segment(0, text)]) == 3.0


def test_zahl_ohne_waehrung_ist_kein_betrag() -> None:
    """Bindet das Muster, nicht nur seine Anwesenheit: Ohne Waehrungszeichen ist 1.190,00
    eine beliebige Zahl. Ein zu weites Muster zaehlte jede Dezimalzahl mit."""
    assert merkmal("amount_count", dokument(), [segment(0, "Menge 1.190,00 Stueck")]) == 0.0


def test_iban_wird_erkannt() -> None:
    text = "Bankverbindung DE89370400440532013000 bei der Musterbank"
    assert merkmal("iban_count", dokument(), [segment(0, text)]) == 1.0


def test_ustid_und_steuernummer_sind_verschiedene_merkmale() -> None:
    text = "USt-IdNr. DE123456789, Steuernummer 151/815/08154"
    assert merkmal("ustid_count", dokument(), [segment(0, text)]) == 1.0
    assert merkmal("steuernummer_count", dokument(), [segment(0, text)]) == 1.0


def test_datumsdichte_bezieht_sich_auf_die_textlaenge() -> None:
    """Eine reine Anzahl waere nur ein Laengenmass. Die Dichte trennt ein Protokoll voller
    Termine von einem langen Vertrag mit ebenso vielen Datumsangaben."""
    kurz = [segment(0, "Termin 01.02.2026")]
    lang = [segment(0, "Termin 01.02.2026 " + "Fuelltext ohne Datum. " * 50)]
    assert merkmal("date_density", dokument(), kurz) > merkmal("date_density", dokument(), lang)


def test_format_merkmale_schliessen_sich_aus() -> None:
    vektor = structural_features(dokument(media_type="application/pdf"), [segment(0, "x")])
    formate = [vektor[STRUCTURAL_NAMES.index(n)] for n in ("is_pdf", "is_docx", "is_xlsx", "is_eml")]
    assert sum(formate) == 1.0, f"Genau ein Formatmerkmal muss gesetzt sein, nicht {formate}"


def test_anhang_wird_als_anhang_erkannt() -> None:
    kind = dokument(source_path="eml/doc-0051.eml!doc-0051.pdf", parent_document_id="eltern")
    assert merkmal("is_attachment", kind, [segment(0, "x")]) == 1.0
    assert merkmal("is_attachment", dokument(), [segment(0, "x")]) == 0.0


def test_leeres_dokument_ergibt_keine_nan() -> None:
    """Division durch die Textlaenge ist die haeufigste NaN-Quelle. Ein einziges NaN
    vergiftet das gesamte Training, ohne dass eine Fehlermeldung faellt."""
    vektor = structural_features(dokument(), [])
    assert not np.any(np.isnan(vektor)), f"NaN in: {
        [n for n, w in zip(STRUCTURAL_NAMES, vektor, strict=True) if np.isnan(w)]
    }"
    assert not np.any(np.isinf(vektor))


def test_kein_merkmal_ist_auf_dem_ganzen_bestand_konstant() -> None:
    """Ein Merkmal mit nur einem Wert traegt null Information und verwaessert die
    Gewichte. Bindet die Merkmale an den echten Bestand, nicht an Beispieltexte."""
    matrix, _ = _bestandsmatrix()
    konstant = [
        name for i, name in enumerate(STRUCTURAL_NAMES)
        if float(np.std(matrix[:, i])) == 0.0
    ]
    assert not konstant, f"Merkmale ohne jede Streuung: {konstant}"


def test_kein_einzelnes_strukturmerkmal_verraet_die_klasse() -> None:
    """Die Lehre aus Phase 1, hier angewandt: Ein einzelnes Merkmal, aus dem sich die
    Klasse fast perfekt ablesen laesst, ist eine Abkuerzung und kein Merkmal.

    Gemessen wird der bestmoegliche Entscheidungsstumpf je Merkmal: Merkmalswerte in zehn
    Koerbe, je Korb die haeufigste Klasse. Grundrate ist 14 Prozent (haeufigste Klasse).
    Die Schranke von 60 Prozent laesst starken, echten Merkmalen Raum – eine IBAN ist ein
    legitim starker Hinweis auf eine Rechnung – faengt aber ein Merkmal, das die Klasse
    praktisch eins zu eins abbildet.
    """
    matrix, klassen = _bestandsmatrix(mit_klassen=True)
    for i, name in enumerate(STRUCTURAL_NAMES):
        koerbe = np.digitize(matrix[:, i], np.quantile(matrix[:, i], np.linspace(0.1, 0.9, 9)))
        je_korb: dict[int, Counter[str]] = defaultdict(Counter)
        for korb, klasse in zip(koerbe, klassen, strict=True):
            je_korb[int(korb)][klasse] += 1
        treffer = sum(z.most_common(1)[0][1] for z in je_korb.values()) / len(klassen)
        assert treffer <= 0.60, (
            f"Merkmal {name} allein trifft {treffer:.1%} der Klassen (Grundrate 14 %) – "
            "das ist eine Abkuerzung, kein Merkmal"
        )
```

Dazu die Hilfsfunktion, die die Merkmalsmatrix über den echten Bestand baut (im Testmodul,
unter den Importen). Sie läuft über die Parquet-Tabellen und nicht über ausgedachte Sätze:
Merkmale, die nur an Beispieltexten geprüft werden, verhalten sich auf echten Dokumenten
anders.

```python
def _bestandsmatrix(mit_klassen: bool = False) -> tuple[np.ndarray, list[str]]:
    """Strukturmerkmale ueber alle eingelesenen Dokumente, dazu die Klassen.

    Document wird aus der Tabellenzeile rekonstruiert, nicht aus der Datei neu gelesen –
    der Test prueft die Merkmale, nicht die Ingestion.
    """
    dokumente = read_table(PARQUET_DIR, "documents")
    segmente = read_table(PARQUET_DIR, "segments").sort("index")
    je_dokument: dict[str, list[Segment]] = defaultdict(list)
    for zeile in segmente.iter_rows(named=True):
        je_dokument[zeile["document_id"]].append(
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

    klasse_von = dict(
        labelled_documents().select(["document_id", "class_key"]).iter_rows()
    )

    zeilen: list[np.ndarray] = []
    klassen: list[str] = []
    for zeile in dokumente.iter_rows(named=True):
        dokument_ = Document(
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
        zeilen.append(structural_features(dokument_, je_dokument[zeile["document_id"]]))
        klassen.append(klasse_von[zeile["document_id"]])

    matrix = np.vstack(zeilen)
    return (matrix, klassen) if mit_klassen else (matrix, [])
```

Die Spaltennamen stammen aus dem echten Bestand und sind geprüft:
`documents` hat `document_id, source_path, file_name, media_type, content_sha256,
size_bytes, parent_document_id, ingested_at, needs_ocr, ocr_applied`, `segments` hat
`segment_id, document_id, index, kind, locator, heading, text`. Weicht `Document` in
seinen Pflichtfeldern davon ab, ist das ein Befund — melden, nicht umgehen.

**Beachte:** Die Rückgabe ist immer ein Paar; `_bestandsmatrix()` ohne Argument liefert
eine leere Klassenliste. Die beiden Aufrufstellen in den Tests oben entpacken
entsprechend (`matrix, _ = _bestandsmatrix()` bzw. `matrix, klassen =
_bestandsmatrix(mit_klassen=True)`).

- [ ] **Schritt 2: Test laufen lassen, Fehlschlag bestätigen**

```bash
uv run pytest tests/test_features_structural.py -v
```

Erwartet: FAIL, `ModuleNotFoundError`.

- [ ] **Schritt 3: Implementierung schreiben**

`src/doccls/features/structural.py`:

```python
"""Rund 40 billige, erklärbare Zahlen je Dokument (Konzept § 6.3).

Sie sind **Merkmale, keine Regeln**: Niemand schreibt ``if IBAN then Rechnung``, das
Modell entscheidet über das Gewicht. Ihr Wert liegt darin, zu sehen, was ein gemitteltes
Embedding übersieht – ein Vorzeichen, eine IBAN, eine Unterschriftenzeile.

Die Reihenfolge in ``STRUCTURAL_NAMES`` ist die Reihenfolge im Vektor. Sie darf sich nur
zusammen mit der ``feature_version`` ändern: ``coef_`` gegen die Merkmalsnamen zu halten
(Konzept § 7.1) ist die wichtigste Diagnose dieses Projekts, und sie wird stillschweigend
falsch, wenn Namen und Spalten auseinanderlaufen.

Nicht umgesetzt aus § 6.3, mit Begründung:

* **Dateinamen-Tokens** – die Dateinamen wurden in Phase 1 neutralisiert (``doc-0001.pdf``),
  weil sich aus ihnen 480 von 560 Klassen raten ließen. Ein Merkmal daraus trüge jetzt null
  Information.
* **Absenderdomain und Betreff-Tokens der Elternmail** – brauchen einen Rückgriff aufs
  Elterndokument, den die Merkmalsbildung nicht hat. Vertagt auf Phase 3, zusammen mit der
  Überarbeitung der Mailvorlagen.
"""

import re
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from doccls.models import Document, Segment, SegmentKind

BETRAG = re.compile(r"\d{1,3}(?:\.\d{3})*,\d{2}\s*(?:€|EUR)\b")
PROZENT = re.compile(r"\d{1,3}(?:,\d+)?\s*%")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
USTID = re.compile(r"\bDE\s?\d{9}\b")
STEUERNUMMER = re.compile(r"\b\d{2,3}/\d{3}/\d{4,5}\b")
BELEGNUMMER = re.compile(r"\b[A-Z]{2,4}-\d{4}-\d{3,6}\b")
DATUM = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b")
PARAGRAF = re.compile(r"§\s*\d+")
TOP = re.compile(r"\bTOP\s*\d+", re.IGNORECASE)
UNTERSCHRIFT = re.compile(r"_{5,}|\bUnterschrift\b", re.IGNORECASE)
NEGATIVBETRAG = re.compile(r"-\s*\d{1,3}(?:\.\d{3})*,\d{2}\s*(?:€|EUR)\b")

SCHLUESSELWOERTER = (
    "zahlungsziel", "gutschrift", "vertragspartner", "kuendigungsfrist", "anwesend",
    "rechnungsnummer", "erstattung", "geltungsbereich", "protokoll", "ampel",
    "fortschritt", "leistungsbeschreibung",
)
"""Startliste. Konzept § 6.3 will sie **aus den Labels gelernt** haben – nach jeder
Trainingsrunde die Terme mit höchster punktweiser Transinformation je Klasse ausgeben und
ergänzen. Das Ausgeben kommt in Phase 3; hier steht der Anfang, bewusst als Liste und
nicht als Regel."""
```

Danach die Merkmalsfunktionen und `STRUCTURAL_NAMES`. **Bauart, die der Umsetzer einhält:**
eine Abbildung `name -> Funktion`, aus der sowohl `STRUCTURAL_NAMES` als auch der Vektor
entsteht. Zwei unabhängig gepflegte Listen — eine für Namen, eine für Werte — laufen
garantiert irgendwann auseinander, und der Fehler ist dann unsichtbar: Die Gewichte stehen
unter den falschen Namen, und die wichtigste Diagnose des Projekts zeigt Unsinn.

```python
def _text(segments: list[Segment]) -> str:
    return " ".join(s.text for s in segments)


def _je_tausend(treffer: int, laenge: int) -> float:
    """Dichte statt Anzahl: Eine reine Anzahl waere nur ein verkapptes Laengenmass."""
    return 1000.0 * treffer / laenge if laenge else 0.0


MERKMALE: dict[str, Callable[[Document, list[Segment], str], float]] = {
    # Umfang
    "segment_count": lambda d, s, t: float(len(s)),
    "chars_total": lambda d, s, t: float(len(t)),
    "chars_per_segment": lambda d, s, t: float(len(t)) / len(s) if s else 0.0,
    "distinct_headings": lambda d, s, t: float(len({x.heading for x in s if x.heading})),
    # ... weitere Gruppen analog, siehe unten
}

STRUCTURAL_NAMES: tuple[str, ...] = tuple(MERKMALE)


def structural_features(document: Document, segments: list[Segment]) -> npt.NDArray[np.float32]:
    """Ein Wert je Eintrag in ``MERKMALE``, in genau dieser Reihenfolge."""
    text = _text(segments)
    return np.array(
        [funktion(document, segments, text) for funktion in MERKMALE.values()],
        dtype=np.float32,
    )
```

**Die vollständige Merkmalsliste, die der Umsetzer ausschreibt** (Gruppen aus Konzept §6.3):

| Gruppe | Merkmale |
|---|---|
| Umfang (4) | `segment_count`, `chars_total`, `chars_per_segment`, `distinct_headings` |
| Zahlen (6) | `digit_ratio`, `amount_count`, `amount_density`, `percent_count`, `has_negative_amount`, `max_amount_log` |
| Muster (8) | `iban_count`, `ustid_count`, `steuernummer_count`, `belegnummer_count`, `date_count`, `date_density`, `paragraph_count`, `top_count` |
| Layout (5) | `table_segment_share`, `sheet_segment_share`, `short_segment_ratio`, `avg_segment_len`, `has_signature_line` |
| Schlüsselwörter (12) | je ein `kw_<wort>` aus `SCHLUESSELWOERTER`, normierte Häufigkeit je 1000 Zeichen |
| Herkunft (5) | `is_pdf`, `is_docx`, `is_xlsx`, `is_eml`, `is_attachment` |
| Verarbeitung (1) | `ocr_applied` |

Zusammen 41. Regeln, die dabei gelten:

- **Jedes Häufigkeitsmerkmal wird auf die Textlänge bezogen**, außer wo die reine Anzahl
  gemeint ist (`amount_count`, `iban_count`). Sonst misst das Merkmal nur die Länge.
- **Keine Division ohne Nullprüfung.** Ein einziges NaN vergiftet das gesamte Training,
  ohne dass eine Fehlermeldung fällt.
- `max_amount_log` ist `log1p(größter gefundener Betrag)` — roh gingen Beträge über vier
  Größenordnungen und erschlügen jedes andere Merkmal.
- `table_segment_share` und `sheet_segment_share` über `SegmentKind`, nicht über den
  `locator`-Text.

- [ ] **Schritt 4: Tests laufen lassen**

```bash
uv run pytest tests/test_features_structural.py -v
```

Erwartet: alle PASS. Schlägt `test_kein_einzelnes_strukturmerkmal_verraet_die_klasse` fehl,
ist das **ein Befund, kein Testfehler**: Das genannte Merkmal melden, nicht die Schranke
anheben.

- [ ] **Schritt 5: Mutationsprobe**

Mindestens sechs Mutationen:

1. `STRUCTURAL_NAMES` fest verdrahten statt aus `MERKMALE` ableiten, dann ein Merkmal
   ergänzen → `test_laenge_stimmt_mit_den_namen_ueberein` rot.
2. Die Währung aus `BETRAG` entfernen → `test_zahl_ohne_waehrung_ist_kein_betrag` rot.
3. `date_density` durch `date_count` ersetzen → `test_datumsdichte_bezieht_sich_auf_die_textlaenge` rot.
4. Die Nullprüfung bei einer Division entfernen → `test_leeres_dokument_ergibt_keine_nan` rot.
5. Alle Merkmale auf `0.0` setzen → `test_kein_merkmal_ist_auf_dem_ganzen_bestand_konstant` rot.
6. Ein künstliches Merkmal einbauen, das die Klasse abbildet (etwa `1.0` für Rechnungen) →
   `test_kein_einzelnes_strukturmerkmal_verraet_die_klasse` rot. **Diese ist die
   wichtigste:** Sie prüft, ob der Abkürzungswächter überhaupt greift.

- [ ] **Schritt 6: Committen**

```bash
git add src/doccls/features/structural.py tests/test_features_structural.py
git commit -m "Strukturmerkmale: 41 erklaerbare Zahlen je Dokument"
```

---

## Aufgabe 9: Merkmalsmatrix zusammensetzen und ablegen

**Dateien:**
- Ändern: `src/doccls/features/__init__.py`
- Anlegen: `scripts/build_features.py`
- Anlegen: `tests/test_features_build.py`

**Schnittstellen:**
- Nutzt: `document_text`, `document_vector`, `make_embedder`, `praefix_fuer`,
  `build_ngram_block`, `structural_features`, `strip_boilerplate`
- Liefert:
  - `BLOCK_ORDER: tuple[str, ...]` = `("emb_mean", "emb_head", "ngram", "structural")`
  - `build_matrix(...) -> FeatureMatrix` (Dataclass mit `document_ids: list[str]`,
    `X: npt.NDArray[np.float32]`, `feature_version: str`, `block_slices: dict[str, slice]`)
  - `write_features(matrix, features_dir) -> Path`
  - `read_features(features_dir, feature_version) -> FeatureMatrix`

**Worum es geht:** Hier laufen die drei Blöcke zusammen. Und hier sitzt die Leckage, die in
Aufgabe 7 angekündigt wurde: Der n-Gramm-Block wird **nur auf den Trainingstexten
angepasst** und danach auf alle angewandt. Wer ihn auf allen Dokumenten anpasst, gibt dem
Modell die Wortstatistik des Gold-Sets mit. Auf dieser Ebene — wo Training und Gold
getrennt vorliegen — ist das prüfbar; in Aufgabe 7 war es das nicht.

- [ ] **Schritt 1: Test schreiben**

`tests/test_features_build.py` — die tragenden Prüfungen:

```python
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

    block = build_ngram_block(FeatureConfig(svd_components=2, ngram_max_features=500))
    block.fit(trainingstexte)

    vokabular = " ".join(block.vokabular())
    assert marker[:6] not in vokabular, (
        "Eine Zeichenfolge, die nur im Gold-Set vorkommt, steht im angepassten Vokabular"
    )
    block.transform(goldtexte)  # darf nicht werfen


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
```

`_kleine_matrix()` baut die Matrix über eine Handvoll echter Dokumente aus dem Bestand mit
dem `FakeEmbedder` aus `tests/test_features_embedding.py` — nicht mit E5. Der Test soll die
Zusammensetzung prüfen, nicht das Modell; ein 1-GB-Ladevorgang je Testlauf wäre Verschwendung.
Dazu muss `NgramBlock` eine Methode `vokabular() -> list[str]` bekommen
(`self._vectorizer.get_feature_names_out().tolist()`, mit der Prüfung auf `fit`).

- [ ] **Schritt 2–4: Fehlschlag bestätigen, umsetzen, grün laufen lassen**

Die Kette in `build_matrix`, in genau dieser Reihenfolge:

```
Segmente je Dokument (nach index sortiert)
  → document_text
  → strip_boilerplate über die Segmenttexte
  → document_vector (emb_mean ‖ emb_head)
  → ngram_block.transform
  → structural_features
  → waagerecht aneinanderhängen
```

`strip_boilerplate` bleibt in der Kette, obwohl sie auf diesem Bestand null Zeichen
entfernt (siehe globale Vorgaben). Sie herauszunehmen hieße, sie bei echten Dokumenten zu
vergessen.

`write_features` legt zwei Dateien unter `<features_dir>/<feature_version>/` ab:
`matrix.parquet` (Spalte `document_id` plus eine `pl.Array(pl.Float32, breite)`-Spalte
`vector`) und `meta.json` (Blockgrenzen, Merkmalsnamen, Embeddername, Breite). Die
`feature_version` steht im Verzeichnisnamen — dann ist ein Versionswechsel ein anderer
Pfad und kein stilles Überschreiben.

- [ ] **Schritt 5: `scripts/build_features.py`**

Dünne CLI, ruft nur das Paket auf:

```python
"""Segmente → Merkmalsmatrix. Der n-Gramm-Block wird nur auf Trainingstexten angepasst."""
```

Ausgabe am Ende, damit ein Lauf ohne Nachsehen beurteilbar ist: Zahl der Dokumente, Breite
je Block, `feature_version`, Embeddername, Laufzeit — und **die Zahl der Zeilen ohne jeden
Wert**, analog zum `empty`-Feld, das sich in Phase 1 bewährt hat.

- [ ] **Schritt 6: Über den echten Bestand laufen lassen**

```bash
uv run python scripts/build_features.py
```

Erwartet: 570 Dokumente, Breite 768+768+256+41 = 1833 bei E5, null leere Zeilen.

- [ ] **Schritt 7: Mutationsprobe**

1. `fit` auf allen Texten statt nur den Trainingstexten →
   `test_ngramme_werden_nur_auf_trainingstexten_angepasst` rot. **Kernmutation.**
2. Einen Block durch Nullen ersetzen → `test_kein_block_ist_durchgehend_null` rot.
3. Beim Lesen die Werte verwerfen und Nullen liefern → `test_rundlauf_ueber_parquet_erhaelt_die_werte` rot.
4. Die Versionsprüfung beim Lesen weglassen → `test_lesen_mit_falscher_version_wirft` rot.
5. `strip_boilerplate` aus der Kette nehmen → prüfen, **ob ein Test rot wird.** Wird keiner
   rot, ist das erwartbar (die Funktion entfernt auf diesem Bestand null Zeichen) — dann im
   Bericht festhalten, dass die Kette an dieser Stelle ungeprüft ist, statt einen Test zu
   erfinden, der nichts misst.

- [ ] **Schritt 8: Committen**

```bash
git add src/doccls/features/__init__.py src/doccls/features/ngrams.py \
        scripts/build_features.py tests/test_features_build.py
git commit -m "Merkmalsmatrix zusammensetzen, versioniert ablegen"
```

---

## Aufgabe 10: Klassifikator und zwei Vergleichsarme

**Dateien:**
- Anlegen: `src/doccls/classify.py`
- Anlegen: `tests/test_classify.py`

**Schnittstellen:**
- Liefert:
  - `MODEL_KINDS: tuple[str, ...]` = `("logreg", "centroid", "svm")`
  - `train_model(X, y, kind, seed=7) -> TrainedModel`
  - `TrainedModel` mit `classes_: list[str]`, `decision_scores(X) -> NDArray`,
    `predict_proba(X) -> NDArray`, `predict(X) -> list[str]`, `kind: str`,
    `liefert_echte_wahrscheinlichkeiten: bool`
  - `feature_weights(model, names) -> dict[str, dict[str, float]]`

**Worum es geht:** Konzept §7.1. Die multinomiale logistische Regression ist die
Hauptumsetzung, weil sie von Natur aus Wahrscheinlichkeiten liefert, in Sekunden auf der
CPU trainiert und lesbare Gewichte hat. Daneben laufen zwei Vergleichsarme mit: der
Nearest-Centroid als **ehrliche Untergrenze**, gegen die sich alles beweisen muss, und eine
lineare SVM, die oft minimal genauer ist, aber keine Wahrscheinlichkeiten liefert.

**Beim Centroid ist die Ehrlichkeit der springende Punkt.** Er hat keine
Wahrscheinlichkeiten. Ein Softmax über negative Abstände sieht aus wie eine
Wahrscheinlichkeit und ist keine. Das gehört in den Docstring und in den Bericht — sonst
wandert eine Pseudo-Wahrscheinlichkeit in eine Kalibriermessung und verdirbt sie.

- [ ] **Schritt 1: Test schreiben**

Die tragenden Prüfungen in `tests/test_classify.py`:

```python
def test_jedes_modell_lernt_eine_trennbare_aufgabe() -> None:
    """Untergrenze der Brauchbarkeit: Auf klar getrennten Punktwolken muss jedes der drei
    Modelle die Trainingsmenge nahezu perfekt treffen. Schafft es das nicht, ist die
    Umsetzung kaputt – und kein spaeterer Test auf echten Daten koennte das von einem
    duennen Korpus unterscheiden."""
    X, y = _zwei_klare_wolken(seed=1)
    for kind in MODEL_KINDS:
        modell = train_model(X, y, kind=kind)
        assert accuracy_score(y, modell.predict(X)) >= 0.95, f"{kind} lernt nicht"


def test_wahrscheinlichkeiten_summieren_sich_zu_eins() -> None:
    X, y = _zwei_klare_wolken(seed=1)
    for kind in MODEL_KINDS:
        p = train_model(X, y, kind=kind).predict_proba(X)
        assert np.allclose(p.sum(axis=1), 1.0), f"{kind}: Zeilensumme ist nicht 1"
        assert np.all(p >= 0.0)


def test_klassenreihenfolge_ist_ueberall_dieselbe() -> None:
    """Der lautloseste Fehler der ganzen Phase: predict_proba liefert Spalten in
    sklearn-Reihenfolge, classes_ aber in einer anderen. Dann zeigt jede Konfidenz auf die
    falsche Klasse, und alle Metriken bleiben plausibel."""
    X, y = _zwei_klare_wolken(seed=1)
    for kind in MODEL_KINDS:
        modell = train_model(X, y, kind=kind)
        p = modell.predict_proba(X)
        aus_proba = [modell.classes_[i] for i in p.argmax(axis=1)]
        assert aus_proba == modell.predict(X), f"{kind}: predict und predict_proba uneins"


def test_training_ist_reproduzierbar() -> None:
    X, y = _zwei_klare_wolken(seed=1)
    for kind in MODEL_KINDS:
        a = train_model(X, y, kind=kind, seed=7).predict_proba(X)
        b = train_model(X, y, kind=kind, seed=7).predict_proba(X)
        assert np.allclose(a, b), f"{kind} ist nicht reproduzierbar"


def test_seltene_klasse_wird_nicht_uebergangen() -> None:
    """class_weight='balanced' (Konzept § 7.1). Ohne das lernt das Modell bei schiefer
    Verteilung, die seltene Klasse schlicht nie vorherzusagen – bei guter Accuracy."""
    X, y = _schiefe_verteilung(haeufig=200, selten=10, seed=1)
    modell = train_model(X, y, kind="logreg")
    assert "selten" in set(modell.predict(X))


def test_gewichte_lassen_sich_den_namen_zuordnen() -> None:
    """Konzept § 7.1: coef_ gegen die Merkmalsnamen zeigt, ob das Modell auf
    'Zahlungsziel' oder auf 'Seite 1 von 3' achtet. Das ist die wichtigste Diagnose des
    Projekts – sie braucht eine Zuordnung, die nicht verrutschen kann."""
    X, y = _zwei_klare_wolken(seed=1)
    namen = [f"m{i}" for i in range(X.shape[1])]
    gewichte = feature_weights(train_model(X, y, kind="logreg"), namen)
    assert set(gewichte) == set(np.unique(y))
    for je_klasse in gewichte.values():
        assert set(je_klasse) == set(namen)


def test_gewichte_bei_falscher_namenszahl_werfen() -> None:
    """Eine stillschweigend abgeschnittene Zuordnung waere schlimmer als gar keine."""
    X, y = _zwei_klare_wolken(seed=1)
    with pytest.raises(ValueError, match="Merkmalsnamen"):
        feature_weights(train_model(X, y, kind="logreg"), ["zu", "wenige"])


def test_centroid_gibt_zu_dass_er_keine_wahrscheinlichkeiten_hat() -> None:
    """Ein Softmax ueber negative Abstaende sieht aus wie eine Wahrscheinlichkeit und ist
    keine. Wandert er ungekennzeichnet in eine Kalibriermessung, verdirbt er sie."""
    X, y = _zwei_klare_wolken(seed=1)
    assert train_model(X, y, kind="centroid").liefert_echte_wahrscheinlichkeiten is False
    assert train_model(X, y, kind="logreg").liefert_echte_wahrscheinlichkeiten is True
```

- [ ] **Schritt 2–4: Umsetzen**

`LogisticRegression(solver="saga", class_weight="balanced", max_iter=5000, random_state=seed)`.
Der Centroid rechnet Kosinusabstände zu den Klassenmittelpunkten; die SVM ist ein
`LinearSVC`. Alle drei hinter derselben `TrainedModel`-Fassade, mit `classes_` aus
`sorted(set(y))` — **einmal festgelegt und überall dieselbe Reihenfolge.**

- [ ] **Schritt 5: Mutationsprobe**

1. `classes_` beim `predict_proba` umsortieren → `test_klassenreihenfolge_ist_ueberall_dieselbe` rot.
2. `class_weight="balanced"` entfernen → `test_seltene_klasse_wird_nicht_uebergangen` rot.
3. `random_state` entfernen → `test_training_ist_reproduzierbar` rot.
4. Die Längenprüfung in `feature_weights` weglassen → der Namenszahl-Test rot.
5. `liefert_echte_wahrscheinlichkeiten` beim Centroid auf `True` → der letzte Test rot.

- [ ] **Schritt 6: Committen**

```bash
git add src/doccls/classify.py tests/test_classify.py
git commit -m "Logistische Regression plus Centroid- und SVM-Vergleichsarm"
```

---

## Aufgabe 11: Kalibrierung — Temperature Scaling

**Dateien:**
- Anlegen: `src/doccls/calibrate.py`
- Anlegen: `tests/test_calibrate.py`

**Schnittstellen:**
- Liefert:
  - `fit_temperature(logits, y_true, classes) -> float`
  - `apply_temperature(logits, temperature) -> npt.NDArray[np.float32]`
  - `expected_calibration_error(proba, y_true, classes, bins=10) -> float`

**Worum es geht:** Konzept §7.3 — der Schritt, der meistens fehlt. Eine logistische
Regression auf hochdimensionalen Merkmalen ist systematisch überheblich: Sie sagt 0,97 und
trifft in 0,84 der Fälle. Temperature Scaling lernt **einen** Parameter `T` auf einer
**separaten** Kalibriermenge durch Minimierung der Negative Log-Likelihood.

**Zwei Stellen, an denen dieser Plan bewusst hinter §7.3 zurückbleibt:**

* Das Konzept nennt **Vector Scaling** und **isotone Regression** als austauschbare
  Alternativen, „falls Temperature Scaling nicht reicht". Sie werden hier nicht gebaut.
  Der Bedarf ist eine Messfrage, keine Planfrage — Aufgabe 16 misst den ECE, und erst wenn
  er das Ziel von 0,05 reißt, lohnt die zweite Umsetzung. Isotone Regression braucht laut
  Konzept ohnehin rund 1000 Kalibrierbeispiele; vorhanden sind 70.
* Das Konzept schlägt vor, die Kalibriermenge bei kleiner Datenlage per **stratifizierter
  Kreuzvalidierung** zu gewinnen, „damit kein Beispiel verloren geht". Hier gibt es
  stattdessen einen eigenen, über Vorlagen geschnittenen `calib`-Split. Der ist sauberer:
  Kreuzvalidierung über Dokumente würde Varianten derselben Vorlage auf beide Seiten legen
  und die Eichung schönen — dieselbe Leckage, die Konzept §9.2 für das Gold-Set verbietet.
  Der Preis sind 70 statt 140 Kalibrierdokumente. Für **einen** Parameter reicht das.

Die entscheidende Eigenschaft, die auch getestet wird: **`T` verändert keine einzige
Entscheidung.** Die Rangfolge der Klassen bleibt gleich. Kalibrierung kann die Genauigkeit
also weder verbessern noch verschlechtern — sie repariert ausschließlich die Zahl. Wer das
prüft, merkt sofort, wenn jemand versehentlich die Logits verschiebt statt sie zu skalieren.

- [ ] **Schritt 1: Test schreiben**

```python
def test_temperatur_aendert_keine_einzige_entscheidung() -> None:
    """Konzept § 7.3: Die Rangfolge bleibt gleich. Kalibrierung repariert die Zahl, nicht
    die Entscheidung. Wer statt zu skalieren verschiebt (logits - T), bricht das – und
    keine Metrik der Genauigkeit wuerde es zeigen, weil sie sich mitverschoebe."""
    logits, y, klassen = _ueberhebliche_vorhersagen(seed=3)
    for T in (0.5, 1.0, 2.0, 5.0):
        vorher = logits.argmax(axis=1)
        nachher = apply_temperature(logits, T).argmax(axis=1)
        assert np.array_equal(vorher, nachher), f"T={T} hat Entscheidungen veraendert"


def test_ueberhebliches_modell_bekommt_temperatur_ueber_eins() -> None:
    """Der Normalfall aus Konzept § 7.3. Ein T <= 1 hiesse, das Modell sei unterheblich –
    dann stimmt etwas mit der Anpassung nicht."""
    logits, y, klassen = _ueberhebliche_vorhersagen(seed=3)
    assert fit_temperature(logits, y, klassen) > 1.0


def test_kalibrierung_senkt_den_eichfehler_deutlich() -> None:
    """Die eigentliche Wirkung, gemessen statt behauptet. Ohne diese Pruefung bestuende
    eine Umsetzung, die immer T=1 liefert, alle anderen Tests."""
    logits, y, klassen = _ueberhebliche_vorhersagen(seed=3)
    vorher = expected_calibration_error(_softmax(logits), y, klassen)
    T = fit_temperature(logits, y, klassen)
    nachher = expected_calibration_error(apply_temperature(logits, T), y, klassen)
    assert nachher < vorher * 0.6, (
        f"ECE nur von {vorher:.4f} auf {nachher:.4f} – die Kalibrierung wirkt kaum"
    )


def test_bereits_geeichtes_modell_bleibt_nahe_bei_eins() -> None:
    """Gegenprobe: Auf gut geeichten Wahrscheinlichkeiten darf die Kalibrierung nichts
    kaputtmachen. Ein Verfahren, das immer nach oben skaliert, faellt hier auf."""
    logits, y, klassen = _gut_geeichte_vorhersagen(seed=3)
    assert 0.7 < fit_temperature(logits, y, klassen) < 1.4


def test_ece_ist_null_bei_perfekter_eichung() -> None:
    """Bindet die Formel: Sagt das Modell durchgehend 1,0 und trifft immer, ist der
    Eichfehler null. Eine falsch normierte Gewichtung faellt hier auf."""
    proba = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(proba, ["a", "a", "b"], ["a", "b"]) == 0.0


def test_ece_ist_gross_bei_voller_ueberheblichkeit() -> None:
    """Gegenstueck: Volle Sicherheit, durchgehend falsch – der Fehler muss nahe 1 liegen.
    Ohne beide Pole koennte eine Umsetzung, die immer 0 liefert, den Test oben bestehen."""
    proba = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert expected_calibration_error(proba, ["b", "b"], ["a", "b"]) > 0.9


def test_kalibriermenge_darf_nicht_die_trainingsmenge_sein() -> None:
    """Konzept § 7.3 verlangt eine SEPARATE Kalibriermenge. Auf den Trainingslogits
    angepasst, ergaebe Temperature Scaling ein T nahe 1 und taeuschte gute Eichung vor –
    dieser Test haelt fest, dass die beiden Mengen ueberhaupt verschieden sind."""
    assert set(training_documents()["document_id"]).isdisjoint(
        set(calibration_documents()["document_id"])
    )
```

- [ ] **Schritt 2–4: Umsetzen**

`fit_temperature` minimiert die NLL über `scipy.optimize.minimize_scalar` im Bereich
`(0.05, 20.0)` mit `method="bounded"`. Die NLL wird über `logits / T` und den
log-softmax gerechnet, **nicht** über `log(softmax(...))` — letzteres läuft bei großen
Logits in numerische Unterläufe. `scipy.special.log_softmax` nimmt das ab.

`expected_calibration_error`: Vorhersagen nach Konfidenz in `bins` gleich breite Körbe,
je Korb `|mittlere Konfidenz − Trefferquote|`, gewichtet nach Korbbesetzung (Konzept §9.3).
Leere Körbe zählen nicht mit — sonst hinge der Wert an der Korbzahl statt an der Eichung.

- [ ] **Schritt 5: Mutationsprobe**

1. `logits / T` durch `logits - T` ersetzen → `test_temperatur_aendert_keine_einzige_entscheidung` rot.
2. `fit_temperature` fest `1.0` liefern lassen → der ECE-Senkungstest rot.
3. Die Gewichtung nach Korbbesetzung weglassen → prüfen, welcher ECE-Test rot wird; bleibt
   keiner rot, fehlt ein Test mit ungleich besetzten Körben — dann nachtragen.
4. Leere Körbe mitzählen → derselbe Prüfpunkt.
5. Die Suchgrenzen auf `(1.0, 20.0)` verengen → `test_bereits_geeichtes_modell_bleibt_nahe_bei_eins`
   muss rot werden, sobald das wahre `T` unter 1 liegt. Wird es nicht rot, ist die
   Gegenprobe zu lasch gebaut.

- [ ] **Schritt 6: Committen**

```bash
git add src/doccls/calibrate.py tests/test_calibrate.py
git commit -m "Temperature Scaling und Eichfehler"
```

---

## Aufgabe 12: Die beiden Schwellen τ und δ — abgelesen, nicht geraten

**Dateien:**
- Anlegen: `src/doccls/decide.py`
- Anlegen: `tests/test_decide.py`

**Schnittstellen:**
- Liefert:
  - `risk_coverage(confidences, correct) -> tuple[NDArray, NDArray]` — Abdeckung, Präzision
  - `tau_for_precision(confidences, correct, target=0.98) -> float`
  - `ood_scores(X, X_train, k=10) -> NDArray` — Kosinusabstand zum Mittel der k nächsten
  - `delta_for_percentile(scores, percentile=95.0) -> float`

**Worum es geht:** Konzept §7.4. Beide Schwellen werden **auf der Kalibriermenge aus den
Daten abgelesen**, nicht geschätzt. Bei δ ist das kein Stilfrage: Kosinuswerte in
Satz-Embeddings sind stark gestaucht — inhaltlich sehr verschiedene Dokumentausschnitte
liegen gemessen zwischen 0,79 und 0,98 (Konzept Anhang D). Ein geschätzter Schwellwert wäre
in diesem Band reine Willkür.

- [ ] **Schritt 1: Test schreiben**

```python
def test_tau_erreicht_das_praezisionsziel() -> None:
    """Die Zusicherung, auf der Coverage@P98 beruht: Oberhalb von tau muss die Praezision
    das Ziel halten. Haelt sie es nicht, ist jede Aussage ueber automatisch entschiedene
    Dokumente falsch."""
    konfidenz, richtig = _konfidenz_mit_rauschen(seed=5)
    tau = tau_for_precision(konfidenz, richtig, target=0.98)
    oberhalb = konfidenz >= tau
    assert oberhalb.sum() > 0, "Kein Dokument oberhalb von tau – die Schwelle ist unbrauchbar"
    assert richtig[oberhalb].mean() >= 0.98


def test_tau_ist_das_kleinste_das_das_ziel_haelt() -> None:
    """Ein zu hohes tau haelt das Ziel auch – und schickt unnoetig viele Dokumente in die
    Pruefliste. Ohne diesen Test bestuende eine Umsetzung, die schlicht 1,0 liefert."""
    konfidenz, richtig = _konfidenz_mit_rauschen(seed=5)
    tau = tau_for_precision(konfidenz, richtig, target=0.98)
    kleiner = [k for k in np.unique(konfidenz) if k < tau]
    for kandidat in kleiner[-5:]:
        oberhalb = konfidenz >= kandidat
        assert richtig[oberhalb].mean() < 0.98, (
            f"tau={tau:.4f} ist nicht minimal – {kandidat:.4f} haelt das Ziel auch"
        )


def test_unerreichbares_ziel_wird_gemeldet() -> None:
    """Bei einem durchweg schlechten Modell gibt es kein tau, das 98 Prozent haelt. Eine
    stillschweigend zurueckgegebene 1,0 sähe wie eine strenge Schwelle aus und waere eine
    Luege."""
    konfidenz = np.linspace(0.3, 0.9, 50)
    richtig = np.zeros(50, dtype=bool)
    with pytest.raises(ValueError, match="Praezision"):
        tau_for_precision(konfidenz, richtig, target=0.98)


def test_ood_abstand_ist_gross_fuer_fremdes_dokument() -> None:
    """Der Zweck der OOD-Pruefung (Konzept § 7.4): Ein lineares Modell kann auf einem
    Dokument, das keiner Klasse aehnelt, trotzdem selbstbewusst sein."""
    trainings_vektoren = _wolke(mittelpunkt=[1.0, 0.0], anzahl=50, seed=2)
    nah = _wolke(mittelpunkt=[1.0, 0.0], anzahl=5, seed=3)
    fern = _wolke(mittelpunkt=[-1.0, 0.0], anzahl=5, seed=4)
    assert ood_scores(fern, trainings_vektoren).mean() > ood_scores(nah, trainings_vektoren).mean()


def test_ood_nutzt_wirklich_k_nachbarn() -> None:
    """Bindet k: Mit k=1 haengt der Abstand an einem einzigen Punkt und schwankt stark,
    mit k=10 ist er stabil. Eine Umsetzung, die k ignoriert, faellt hier auf."""
    trainings_vektoren = _wolke(mittelpunkt=[1.0, 0.0], anzahl=50, seed=2)
    probe = _wolke(mittelpunkt=[1.0, 0.0], anzahl=20, seed=9)
    assert float(np.std(ood_scores(probe, trainings_vektoren, k=1))) > float(
        np.std(ood_scores(probe, trainings_vektoren, k=10))
    )


def test_delta_liegt_auf_dem_perzentil() -> None:
    werte = np.linspace(0.0, 1.0, 101)
    assert abs(delta_for_percentile(werte, percentile=95.0) - 0.95) < 0.01


def test_delta_wird_nicht_geraten_sondern_gemessen() -> None:
    """Konzept § 7.4 und Anhang D: Kosinuswerte sind stark gestaucht (0,79 bis 0,98).
    Zwei verschieden verteilte Mengen muessen deutlich verschiedene delta ergeben – eine
    fest verdrahtete Konstante faellt hier auf."""
    eng = np.random.default_rng(1).normal(0.10, 0.01, 500)
    weit = np.random.default_rng(1).normal(0.40, 0.08, 500)
    assert abs(delta_for_percentile(weit) - delta_for_percentile(eng)) > 0.1
```

- [ ] **Schritt 2–4: Umsetzen**

`tau_for_precision`: über die absteigend sortierten Konfidenzen laufen, an jeder Schwelle
die Präzision oberhalb rechnen, das **kleinste** τ nehmen, das das Ziel hält. Hält keines,
werfen — mit der erreichten Höchstpräzision in der Meldung, damit man sieht, wie weit es
fehlt.

`ood_scores`: Kosinusabstand von `x` zum **Mittel der k=10 nächsten Trainingsnachbarn**
(Konzept §7.4), nicht zum globalen Klassenmittelpunkt. Beide Seiten vorher L2-normieren.

- [ ] **Schritt 5: Mutationsprobe**

1. In `tau_for_precision` das größte statt des kleinsten haltenden τ nehmen → Minimalitätstest rot.
2. Bei unerreichbarem Ziel `1.0` zurückgeben statt zu werfen → der Meldungstest rot.
3. `k` in `ood_scores` ignorieren (immer alle Nachbarn) → `test_ood_nutzt_wirklich_k_nachbarn` rot.
4. Die L2-Normierung weglassen → prüfen, welcher Test rot wird; bleibt keiner rot, fehlt
   ein Test mit unterschiedlich langen Vektoren — nachtragen.
5. `delta_for_percentile` eine Konstante liefern lassen → der Messtest rot.

- [ ] **Schritt 6: Committen**

```bash
git add src/doccls/decide.py tests/test_decide.py
git commit -m "Schwellen tau und delta aus den Daten ablesen"
```

---

## Aufgabe 13: Die Entscheidung — AUTO, REVIEW oder SONSTIGES

**Dateien:**
- Ändern: `src/doccls/models.py` (dort entsteht `Prediction`)
- Ändern: `src/doccls/decide.py`
- Ändern: `tests/test_decide.py`

**Schnittstellen:**
- Liefert:
  - `Decision` (StrEnum): `AUTO`, `REVIEW`
  - `Prediction` (Pydantic) — **in `src/doccls/models.py`**, nicht in `decide.py`:
    Konzept § 10 führt `Prediction` dort neben `Document` und `Segment` auf, und die
    Polars-Schemas liegen schon in derselben Datei. Felder: `document_id`,
    `class_key`, `confidence`, `margin`, `entropy`, `ood_score`, `decision`,
    `model_version`, `feature_version`
  - `decide_one(proba, classes, ood_score, tau, delta, document_id, ...) -> Prediction`

**Worum es geht:** Konzept §7.4, die Regel wörtlich:

```
wenn ood_score > δ                    →  SONSTIGES, decision = REVIEW
sonst wenn konfidenz < τ              →  bester Vorschlag, decision = REVIEW
sonst                                 →  bester Vorschlag, decision = AUTO
```

Die Reihenfolge ist Teil der Regel: Die OOD-Prüfung kommt **zuerst**. Ein Dokument, das
keiner Klasse ähnelt, kann trotzdem hohe Konfidenz haben — das lineare Modell kennt nur
seine sieben Klassen und verteilt die Masse auf sie.

Dazu die drei Ableitungen aus §7.2, weil sie Verschiedenes messen: Konfidenz
(`max p̂`), Margin (`p̂₍₁₎ − p̂₍₂₎`) und Entropie (`−Σ p̂ log p̂`). Die Margin ist das
Auswahlkriterium für die Prüfliste in Phase 3 und wird hier schon mitgeschrieben.

- [ ] **Schritt 1: Test schreiben**

```python
def test_hohe_konfidenz_und_bekanntes_dokument_ergibt_auto() -> None:
    p = decide_one(np.array([0.95, 0.03, 0.02]), KLASSEN, ood_score=0.1, tau=0.9, delta=0.5, ...)
    assert p.decision is Decision.AUTO and p.class_key == "RECHNUNG"


def test_niedrige_konfidenz_ergibt_review() -> None:
    p = decide_one(np.array([0.5, 0.3, 0.2]), KLASSEN, ood_score=0.1, tau=0.9, delta=0.5, ...)
    assert p.decision is Decision.REVIEW and p.class_key == "RECHNUNG"


def test_ood_schlaegt_hohe_konfidenz() -> None:
    """Die Reihenfolge der Regel, und ihr eigentlicher Zweck: Ein Dokument jenseits von
    delta wird SONSTIGES – auch bei 0,99 Konfidenz. Wuerde die Konfidenz zuerst geprueft,
    liefe genau der Fall durch, den die OOD-Pruefung abfangen soll."""
    p = decide_one(np.array([0.99, 0.005, 0.005]), KLASSEN, ood_score=0.9, tau=0.9, delta=0.5, ...)
    assert p.class_key == "SONSTIGES" and p.decision is Decision.REVIEW


def test_margin_trennt_zwei_kandidaten_von_breiter_unsicherheit() -> None:
    """Konzept § 7.2: 0,45/0,44/0,11 sitzt auf der Grenze zwischen zwei Klassen und ist
    beim Labeln wertvoller als 0,40/0,20/0,20/0,20. Konfidenz allein sieht das nicht –
    genau deshalb wird die Margin mitgeschrieben."""
    eng = decide_one(np.array([0.45, 0.44, 0.11]), KLASSEN, ...)
    breit = decide_one(np.array([0.40, 0.30, 0.30]), KLASSEN, ...)
    assert eng.margin < breit.margin
    assert eng.entropy < breit.entropy


def test_entropie_trifft_die_formel() -> None:
    """Bindet die Formel, nicht nur ihre Richtung: Bei drei gleich wahrscheinlichen
    Klassen ist die Entropie ln(3)."""
    p = decide_one(np.array([1 / 3, 1 / 3, 1 / 3]), KLASSEN, ...)
    assert abs(p.entropy - math.log(3)) < 1e-6


def test_konfidenz_und_klasse_gehoeren_zusammen() -> None:
    """Der lautlose Fehler: argmax ueber die Wahrscheinlichkeiten, aber der Name aus einer
    anders sortierten Liste. Dann stimmt die Zahl und der Name nicht."""
    for i, klasse in enumerate(KLASSEN):
        proba = np.full(len(KLASSEN), 0.01)
        proba[i] = 1.0 - 0.01 * (len(KLASSEN) - 1)
        p = decide_one(proba, KLASSEN, ood_score=0.0, tau=0.5, delta=0.9, ...)
        assert p.class_key == klasse and p.confidence == pytest.approx(proba[i])


def test_genau_auf_der_schwelle_gilt_als_auto() -> None:
    """Konzept § 7.4 schreibt 'konfidenz < tau → REVIEW'. Gleichheit ist also AUTO.
    Ohne diesen Test bliebe ein <= statt < unbemerkt, und Coverage@P98 waere leicht
    verschoben – bei einer Kennzahl, die auf zwei Stellen berichtet wird."""
    p = decide_one(np.array([0.9, 0.05, 0.05]), KLASSEN, ood_score=0.0, tau=0.9, delta=0.9, ...)
    assert p.decision is Decision.AUTO
```

- [ ] **Schritt 2–6:** umsetzen, grün laufen lassen, mutieren (mindestens: Reihenfolge der
beiden Bedingungen tauschen; `<` gegen `<=`; Margin als `p₍₁₎` statt `p₍₁₎ − p₍₂₎`;
Entropie mit `log10` statt `ln`; Klassenname aus fester Position statt `argmax`),
committen.

```bash
git add src/doccls/decide.py tests/test_decide.py
git commit -m "Entscheidung AUTO, REVIEW oder SONSTIGES mit OOD zuerst"
```

---

## Aufgabe 14: Kennzahlen

**Dateien:**
- Anlegen: `src/doccls/evaluate.py`
- Anlegen: `tests/test_evaluate.py`

**Schnittstellen:**
- Liefert:
  - `Metrics` (Pydantic): `macro_f1`, `per_class_f1: dict[str, float]`, `accuracy`, `ece`,
    `brier`, `nll`, `coverage_at_precision`, `aurc`, `auroc_confidence`, `n`
  - `evaluate(proba, y_true, classes, target_precision=0.98) -> Metrics`
  - `confusion(y_true, y_pred, classes) -> pl.DataFrame`
  - `bootstrap_ci(proba, y_true, classes, metric, rounds=1000, seed=7) -> tuple[float, float]`

**Worum es geht:** Konzept §9.3. Die Hauptzahl ist **Macro-F1**, nicht Accuracy — die
belohnt nur die häufigste Klasse. Die Konfusionsmatrix ist bei dieser Aufgabe wertvoller
als jede Einzelzahl: Rechnung↔Gutschrift und Vertrag↔AGB sind völlig verschiedene Probleme.

Und die Kennzahl, auf die es ankommt: **Coverage@P98** — der Anteil der Dokumente, die bei
mindestens 98 % Präzision automatisch entschieden werden können. Sie koppelt Konfidenz und
Korrektheit und lässt sich nicht durch Temperaturspielerei manipulieren. Konzept §9.1 zeigt
warum: Die mittlere Konfidenz allein ist trivial auf 0,99 zu bringen, indem man `T` auf 0,3
setzt — ohne dass eine einzige Vorhersage besser wird.

- [ ] **Schritt 1: Test schreiben**

Die tragenden Prüfungen:

```python
def test_macro_f1_gewichtet_klassen_gleich() -> None:
    """Der Grund, warum nicht Accuracy die Hauptzahl ist (Konzept § 9.3). Ein Modell, das
    die seltene Klasse nie trifft, hat hohe Accuracy und schlechtes Macro-F1."""
    y = ["haeufig"] * 95 + ["selten"] * 5
    proba = _sichere_vorhersage(["haeufig"] * 100, KLASSEN_ZWEI)
    m = evaluate(proba, y, KLASSEN_ZWEI)
    assert m.accuracy == pytest.approx(0.95)
    assert m.macro_f1 < 0.55, "Macro-F1 muss das Uebergehen der seltenen Klasse bestrafen"


def test_coverage_at_p98_laesst_sich_nicht_durch_temperatur_schoenen() -> None:
    """Konzept § 9.1 und § 9.3: Wer alle Konfidenzen hochskaliert, verschiebt nur die
    Schwelle mit. Das ist die Zusicherung, die Coverage@P98 zur ehrlichen Antwort auf
    'steigt die Konfidenz?' macht – und sie wird hier gemessen, nicht geglaubt."""
    logits, y, klassen = _gemischte_vorhersagen(seed=11)
    kalt = evaluate(apply_temperature(logits, 0.3), y, klassen).coverage_at_precision
    normal = evaluate(apply_temperature(logits, 1.0), y, klassen).coverage_at_precision
    assert abs(kalt - normal) < 0.02, (
        f"Coverage@P98 springt von {normal:.3f} auf {kalt:.3f}, nur weil T=0,3 ist – "
        "dann misst die Kennzahl die Temperatur statt die Nutzbarkeit"
    )


def test_auroc_der_konfidenz_misst_trennschaerfe_unabhaengig_von_der_eichung() -> None:
    """Konzept § 9.3: AUROC der Konfidenz als Fehlerdetektor. Temperatur aendert die
    Rangfolge nicht, also darf die AUROC sich nicht bewegen – anders als der ECE."""
    logits, y, klassen = _gemischte_vorhersagen(seed=11)
    a = evaluate(apply_temperature(logits, 0.5), y, klassen)
    b = evaluate(apply_temperature(logits, 2.0), y, klassen)
    assert abs(a.auroc_confidence - b.auroc_confidence) < 1e-6
    assert abs(a.ece - b.ece) > 0.01, "Der ECE muss sich sehr wohl bewegen"


def test_brier_und_nll_bestrafen_sichere_irrtuemer() -> None:
    sicher_falsch = np.array([[0.99, 0.01]])
    unsicher_falsch = np.array([[0.55, 0.45]])
    a = evaluate(sicher_falsch, ["b"], KLASSEN_ZWEI)
    b = evaluate(unsicher_falsch, ["b"], KLASSEN_ZWEI)
    assert a.brier > b.brier and a.nll > b.nll


def test_konfusionsmatrix_zeigt_die_richtung_der_verwechslung() -> None:
    """Rechnung als Gutschrift zu lesen ist ein anderes Problem als umgekehrt. Eine
    symmetrische Matrix waere unbrauchbar."""
    m = confusion(["RECHNUNG"] * 5, ["GUTSCHRIFT"] * 5, ["RECHNUNG", "GUTSCHRIFT"])
    assert _zelle(m, "RECHNUNG", "GUTSCHRIFT") == 5
    assert _zelle(m, "GUTSCHRIFT", "RECHNUNG") == 0


def test_bootstrap_intervall_umschliesst_den_punktwert() -> None:
    proba, y, klassen = _gemischte_wahrscheinlichkeiten(seed=13)
    punkt = evaluate(proba, y, klassen).macro_f1
    unten, oben = bootstrap_ci(proba, y, klassen, metric="macro_f1", rounds=200)
    assert unten <= punkt <= oben


def test_bootstrap_ist_bei_kleiner_menge_breiter() -> None:
    """Konzept § 9.2 begruendet die 50 Dokumente je Klasse damit, dass Unterschiede sonst
    im Rauschen verschwinden. Diese Pruefung bindet genau das: Weniger Daten, breiteres
    Intervall. Ein Bootstrap, der die Streuung nicht abbildet, faellt hier auf."""
    proba, y, klassen = _gemischte_wahrscheinlichkeiten(seed=13, n=400)
    weit = bootstrap_ci(proba[:40], y[:40], klassen, metric="macro_f1", rounds=200)
    eng = bootstrap_ci(proba, y, klassen, metric="macro_f1", rounds=200)
    assert (weit[1] - weit[0]) > (eng[1] - eng[0]) * 1.5


def test_bootstrap_ist_reproduzierbar() -> None:
    proba, y, klassen = _gemischte_wahrscheinlichkeiten(seed=13)
    assert bootstrap_ci(proba, y, klassen, "macro_f1", rounds=100, seed=7) == bootstrap_ci(
        proba, y, klassen, "macro_f1", rounds=100, seed=7
    )


def test_aurc_faellt_wenn_die_konfidenz_besser_trennt() -> None:
    """AURC ist die Flaeche unter der Risiko-Abdeckungs-Kurve – kleiner ist besser.
    Bindet die Richtung: Ein Vorzeichenfehler waere sonst unsichtbar."""
    gut, y, klassen = _konfidenz_trennt_gut(seed=17)
    schlecht = _konfidenz_zufaellig(gut, seed=17)
    assert evaluate(gut, y, klassen).aurc < evaluate(schlecht, y, klassen).aurc
```

- [ ] **Schritt 2–4: Umsetzen**

`macro_f1`, `per_class_f1`, `accuracy` über `sklearn.metrics`. `ece` aus
`doccls.calibrate` wiederverwenden — **nicht** ein zweites Mal schreiben. `brier` als
mittlerer quadratischer Fehler über die One-Hot-Wahrheit, `nll` über `log_softmax`-sichere
Rechnung mit `np.clip(proba, 1e-12, 1.0)`.

`coverage_at_precision`: Konfidenzen absteigend sortieren, an jeder Schwelle Abdeckung und
Präzision rechnen, die **größte** Abdeckung nehmen, bei der die Präzision das Ziel hält;
hält keine, ist die Abdeckung 0,0 (kein Fehler — das ist ein gültiges, schlechtes Ergebnis).

`aurc`: Risiko (= 1 − Präzision) über die Abdeckung integriert, Trapezregel.

`bootstrap_ci`: `rounds` Ziehungen mit Zurücklegen über die Zeilen, je Ziehung die Metrik,
danach das 2,5- und 97,5-Perzentil. Fester Seed.

- [ ] **Schritt 5: Mutationsprobe**

Mindestens sechs. Zwingend darunter:

1. `macro_f1` durch `accuracy` ersetzen → der erste Test rot.
2. Das Vorzeichen in `aurc` drehen → der AURC-Richtungstest rot.
3. Beim Bootstrap ohne Zurücklegen ziehen → `test_bootstrap_ist_bei_kleiner_menge_breiter` rot.
4. In `coverage_at_precision` die kleinste statt der größten haltenden Abdeckung nehmen →
   prüfen, welcher Test rot wird; bleibt keiner rot, fehlt ein Test — nachtragen.
5. Die Konfusionsmatrix transponieren → der Richtungstest rot.
6. Alle Metriken auf `0.0` setzen → **mindestens fünf** Tests müssen rot werden. Werden es
   weniger, sind die übrigen Metriken ungeprüft.

- [ ] **Schritt 6: Committen**

```bash
git add src/doccls/evaluate.py tests/test_evaluate.py
git commit -m "Kennzahlen: Macro-F1, Eichung, Coverage@P98, AURC, Bootstrap"
```

---

## Aufgabe 15: Reliability Diagram und Konfusionsmatrix als Bild

**Dateien:**
- Anlegen: `src/doccls/reports.py`
- Anlegen: `tests/test_reports.py`

**Schnittstellen:**
- Liefert:
  - `reliability_bins(proba, y_true, classes, bins=10) -> list[ReliabilityBin]`
  - `ReliabilityBin` (Dataclass): `untergrenze`, `obergrenze`, `mittlere_konfidenz`,
    `trefferquote`, `anzahl`
  - `reliability_diagram(proba_vorher, proba_nachher, y_true, classes, path)
    `-> tuple[Path, Figure]` — die Figur wird mitgegeben, damit der Test ihren Inhalt
    prüfen kann, ohne die PNG zu zerlegen
  - `confusion_heatmap(matrix, path) -> Path`
  - `write_metrics(metrics, path) -> Path`

**Worum es geht:** Konzept §9.3 nennt das Reliability Diagram „ein Bild, das mehr erklärt
als drei Zahlen". Und Phase 2 soll laut §11 „ein Reliability Diagram vor und nach der
Kalibrierung" liefern — **die erste sichtbare Erkenntnis des Projekts**. Beide Kurven
gehören deshalb in **ein** Bild, mit der Diagonale als Ideal; nebeneinander in zwei Bildern
sieht man den Unterschied nicht.

Diagramme zu testen ist heikel — man kann leicht prüfen, dass eine Datei entsteht, und
nichts über ihren Inhalt wissen. Die Tests hier binden deshalb die **Daten hinter dem
Bild**, nicht das Bild:

- [ ] **Schritt 1: Test schreiben**

```python
def test_reliability_daten_folgen_der_diagonale_bei_perfekter_eichung() -> None:
    """Bindet die Daten hinter dem Bild. Ein Test, der nur prueft, dass eine PNG-Datei
    entsteht, wuerde auch eine leere Leinwand durchgehen lassen."""
    proba, y, klassen = _perfekt_geeicht(seed=19)
    koerbe = reliability_bins(proba, y, klassen, bins=10)
    for korb in koerbe:
        if korb.anzahl >= 10:
            assert abs(korb.mittlere_konfidenz - korb.trefferquote) < 0.1


def test_reliability_daten_zeigen_ueberheblichkeit() -> None:
    """Gegenprobe: Bei einem ueberheblichen Modell muessen die Koerbe UNTER der Diagonale
    liegen – mittlere Konfidenz hoeher als Trefferquote. Ohne beide Pole bestuende eine
    Umsetzung, die immer die Diagonale zurueckgibt."""
    proba, y, klassen = _ueberheblich(seed=19)
    koerbe = [k for k in reliability_bins(proba, y, klassen, bins=10) if k.anzahl >= 10]
    assert koerbe, "Keine ausreichend besetzten Koerbe – der Test prueft sonst nichts"
    assert sum(k.mittlere_konfidenz - k.trefferquote for k in koerbe) > 0


def test_leere_koerbe_werden_nicht_als_nullpunkte_gezeichnet() -> None:
    """Ein leerer Korb, als (0,0) gezeichnet, zieht die Kurve nach unten und laesst ein
    gut geeichtes Modell schlecht aussehen."""
    proba, y, klassen = _nur_hohe_konfidenz(seed=19)
    assert all(k.anzahl > 0 for k in reliability_bins(proba, y, klassen, bins=10))


def test_diagramm_enthaelt_beide_kurven(tmp_path: Path) -> None:
    """Konzept § 11 verlangt vorher UND nachher in einem Bild – nebeneinander sieht man
    den Unterschied nicht. Geprueft wird ueber die Zahl der Linien in der Figur, nicht
    ueber die Pixel."""
    vorher, nachher, y, klassen = _vorher_nachher(seed=19)
    pfad, figur = reliability_diagram(vorher, nachher, y, klassen, tmp_path / "r.png")
    assert pfad.exists() and pfad.stat().st_size > 0
    beschriftungen = [linie.get_label() for linie in figur.axes[0].get_lines()]
    assert any("vor" in b.lower() for b in beschriftungen)
    assert any("nach" in b.lower() for b in beschriftungen)
    assert any("ideal" in b.lower() or "diagonale" in b.lower() for b in beschriftungen)
```

- [ ] **Schritt 2–4: Umsetzen**

`matplotlib` mit `matplotlib.use("Agg")` **vor** dem Pyplot-Import — ohne Backend-Wahl
versucht die Bibliothek ein Fenster zu öffnen und der Testlauf hängt.

`reliability_bins` liefert eine Liste aus `ReliabilityBin(untergrenze, obergrenze,
mittlere_konfidenz, trefferquote, anzahl)` — nur besetzte Körbe. `reliability_diagram`
gibt `(Path, Figure)` zurück, damit der Test die Figur prüfen kann, ohne die PNG zu
zerlegen.

- [ ] **Schritt 5: Mutationsprobe**

1. Leere Körbe als `(0, 0)` mitführen → `test_leere_koerbe_werden_nicht_als_nullpunkte_gezeichnet` rot.
2. `trefferquote` und `mittlere_konfidenz` vertauschen → `test_reliability_daten_zeigen_ueberheblichkeit` rot.
3. Nur die Kurve „nachher" zeichnen → der Kurventest rot.
4. Die Diagonale weglassen → der Kurventest rot.

- [ ] **Schritt 6: Committen**

```bash
git add src/doccls/reports.py tests/test_reports.py
git commit -m "Reliability Diagram vor und nach der Kalibrierung"
```

---

## Aufgabe 16: Die erste Messung — und die Diagnose, ob man ihr glauben darf

**Dateien:**
- Anlegen: `scripts/train.py`
- Anlegen: `scripts/evaluate.py`
- Anlegen: `tests/test_train_evaluate.py`
- Anlegen: `docs/auswertungen.md`

**Schnittstellen:**
- Nutzt: alles aus den Aufgaben 6 bis 15
- Liefert: `train_and_calibrate(...) -> TrainingRun` (Artefakt mit Modell, `T`, τ, δ,
  `feature_version`, `model_version`), `diagnose_thin_corpus(...) -> Diagnose`

**Worum es geht:** Hier läuft alles zusammen, und hier entsteht das Ergebnis, das Phase 2
laut Konzept §11 liefern soll: die erste Messung gegen das Gold-Set mit Bootstrap-Intervall,
die Nearest-Centroid-Untergrenze als Vergleich, und das Reliability Diagram vor und nach
der Kalibrierung.

**Die vierte Diagnose.** Konzept §9.5 nennt drei Ursachen für eine flache Lernkurve:
widersprüchliche Labels, zu schwache Merkmale, tatsächlich überlappende Klassen. Auf diesem
Korpus fehlt die wahrscheinlichste: **die Trainingsmenge ist zu dünn.** Das Modell sieht je
Klasse zwei Vorlagen und wird gegen fünf ungesehene geprüft. Ohne eine Diagnose, die das
trennt, wäre eine schlechte erste Zahl nicht zu deuten:

| Training | Gold | Deutung |
|---|---|---|
| hoch | hoch | Alles gut |
| **hoch** | **niedrig** | **Korpus zu dünn — das Modell hat die zwei Vorlagen auswendig gelernt** |
| niedrig | niedrig | Merkmale oder Verfahren zu schwach |
| niedrig | hoch | Fehler in der Auswertung — diese Zelle darf nicht vorkommen |

- [ ] **Schritt 1: Test schreiben**

```python
def test_gold_wird_im_ganzen_lauf_nie_beruehrt() -> None:
    """Die wichtigste Zusicherung der ganzen Phase, an der Stelle geprueft, an der alles
    zusammenlaeuft. Die Einzelpruefung in Aufgabe 6 sichert die Funktionen; dieser Test
    sichert die VERKETTUNG – dort ist in der Praxis das Leck."""
    lauf = train_and_calibrate(kind="logreg")
    gold = set(gold_documents()["document_id"])
    assert gold.isdisjoint(lauf.trainings_ids)
    assert gold.isdisjoint(lauf.kalibrier_ids)


def test_diagnose_erkennt_duennen_korpus() -> None:
    """Die vierte Diagnose, die Konzept § 9.5 fehlt. Kuenstlich erzeugt: perfekt auf dem
    Training, schlecht auf Gold."""
    d = diagnose_thin_corpus(train_macro_f1=0.99, gold_macro_f1=0.41)
    assert d.befund == "korpus_zu_duenn"


def test_diagnose_erkennt_schwaches_verfahren() -> None:
    d = diagnose_thin_corpus(train_macro_f1=0.44, gold_macro_f1=0.41)
    assert d.befund == "verfahren_zu_schwach"


def test_diagnose_meldet_unmoegliche_lage() -> None:
    """Gold deutlich besser als Training ist kein gutes Zeichen, sondern ein Hinweis auf
    einen Fehler in der Auswertung – etwa vertauschte Mengen."""
    with pytest.raises(ValueError, match="Auswertung"):
        diagnose_thin_corpus(train_macro_f1=0.40, gold_macro_f1=0.85)


def test_lauf_ist_reproduzierbar() -> None:
    a = train_and_calibrate(kind="logreg", seed=7)
    b = train_and_calibrate(kind="logreg", seed=7)
    assert a.temperature == pytest.approx(b.temperature)
    assert a.tau == pytest.approx(b.tau)
    assert a.delta == pytest.approx(b.delta)


def test_artefakt_traegt_beide_versionen() -> None:
    """Konzept § 6.4 und § 7.4: feature_version und die Schwellen gehoeren zur
    Modellversion. Ein Artefakt ohne sie laesst sich spaeter nicht mehr zuordnen."""
    lauf = train_and_calibrate(kind="logreg")
    assert lauf.feature_version and lauf.model_version
    assert lauf.tau is not None and lauf.delta is not None


def test_vorhersage_mit_fremder_merkmalsversion_wirft() -> None:
    """Der Fehler, den man sonst nicht findet (Konzept § 6.4): ein Modell, das auf anderen
    Merkmalen trainiert wurde, rechnet stillschweigend weiter und liefert Unsinn mit
    hoher Konfidenz."""
    lauf = train_and_calibrate(kind="logreg")
    with pytest.raises(ValueError, match="feature_version"):
        lauf.predict(_matrix_mit_version("fremde-version"))
```

- [ ] **Schritt 2–4: Umsetzen**

`scripts/train.py` mit `--model {logreg,centroid,svm}` (Konzept §7.1: „als Vergleichsarm in
derselben Bewertung mitlaufen lassen") und `--embedder {e5,bge-m3}`. Reihenfolge im Lauf:

```
Merkmale lesen (feature_version prüfen)
  → Training auf split=train
  → Logits auf split=calib  → T anpassen
  → Konfidenzen auf split=calib → τ aus der Risiko-Abdeckungs-Kurve
  → OOD-Abstände auf split=calib → δ als 95. Perzentil
  → Artefakt nach data/models/<model_version>/ schreiben
```

`scripts/evaluate.py` misst gegen das Gold-Set, schreibt Kennzahlen als JSON, die
Konfusionsmatrix als CSV und PNG, das Reliability Diagram vor/nach — alles nach
`data/reports/<model_version>/`.

- [ ] **Schritt 5: Die Messung durchführen**

```bash
uv run python scripts/build_features.py                       # embedder: e5
uv run python scripts/train.py --model logreg
uv run python scripts/train.py --model centroid
uv run python scripts/evaluate.py --model logreg --compare centroid
```

Danach `config/features.yaml` auf `embedder: bge-m3` stellen und dasselbe wiederholen.
Weil die `feature_version` sich dadurch ändert, liegen beide Merkmalssätze nebeneinander
und überschreiben sich nicht.

- [ ] **Schritt 6: `docs/auswertungen.md` schreiben**

Das ist das eigentliche Ergebnis der Phase, nicht der Code. Hinein gehören:

- Die Tabelle E5 gegen BGE-M3: Macro-F1 mit Bootstrap-Intervall, ECE, Coverage@P98 —
  **damit ist Konzept Anhang B beantwortet: gemessen statt entschieden.**
- Die Nearest-Centroid-Untergrenze daneben. Liegt die logistische Regression nicht
  deutlich darüber, ist das ein Befund und keine Fußnote.
- Das Reliability Diagram vor und nach der Kalibrierung, mit dem ECE beider Stände.
- Die Konfusionsmatrix und die zwei, drei größten Verwechslungen im Klartext.
- Die vierte Diagnose mit ihrem Befund, und was daraus folgt.
- Die zwanzig größten Gewichte je Klasse gegen die Merkmalsnamen (Konzept §7.1). **Hier
  wird sichtbar, ob das Modell auf Inhalt oder auf eine Abkürzung achtet.** Steht ein
  Strukturmerkmal wie `is_xlsx` ganz oben, ist das ein Befund.

- [ ] **Schritt 7: Mutationsprobe**

1. Im Lauf die Kalibriermenge durch die Trainingsmenge ersetzen → `test_gold_wird_im_ganzen_lauf_nie_beruehrt`
   bleibt grün (kein Gold!), aber es muss **ein anderer** Test rot werden. Wird keiner rot,
   fehlt eine Prüfung, dass Training und Kalibrierung verschieden sind — nachtragen.
2. Die Gold-Menge ins Training geben → der Gold-Test rot.
3. Die Versionsprüfung bei `predict` weglassen → der Versionstest rot.
4. Die Diagnoseschwellen so setzen, dass immer „verfahren_zu_schwach" herauskommt → der
   Diagnosetest rot.

- [ ] **Schritt 8: Committen**

```bash
git add scripts/train.py scripts/evaluate.py tests/test_train_evaluate.py docs/auswertungen.md
git commit -m "Erste Messung gegen das Gold-Set, mit Diagnose der Korpusdichte"
```

---

## Was am Ende dieser Phase dasteht

- Eine versionierte Merkmalsmatrix über alle 570 Dokumente, aus drei Blöcken.
- Ein Klassifikator mit kalibrierter Konfidenz und zwei Schwellen, die aus Daten abgelesen
  sind statt geraten.
- Eine Messung gegen das eingefrorene Gold-Set mit Bootstrap-Intervall, daneben die
  ehrliche Untergrenze.
- Die Antwort auf Konzept Anhang B: E5 oder BGE-M3, gemessen.
- Ein Reliability Diagram vor und nach der Kalibrierung.
- Und eine begründete Aussage darüber, **ob man dieser ersten Zahl glauben darf** — oder
  ob zuerst der Korpus wachsen muss.

Was Phase 2 ausdrücklich **nicht** liefert: die Lernschleife. Prüfliste, Auswahlstrategie,
Neutraining und Promotion-Gate sind Phase 3. Phase 2 stellt nur fest, wo man steht.
