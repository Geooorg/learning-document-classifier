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

Der Schnitt selbst steht in `config/splits.yaml` und ist versioniert. Er wird nicht bei
jedem Lauf neu gewürfelt: `random.Random(...).shuffle()` auf einer Liste, deren Länge sich
ändert, permutiert alle Positionen neu — eine einzige neue Vorlage würde sonst mehrere
bestehende über die Schnittgrenze verschieben, ohne dass irgendetwas fehlschlägt, und alle
vorher gemessenen Zahlen wären unvergleichbar. Kommt eine neue Vorlage hinzu, hängt
`scripts/generate_documents.py` sie an `config/splits.yaml` an; bestehende Einträge bleiben
unverändert. Die Datei wird nicht von Hand gepflegt.

Zwei Sicherungen dazu: `assign_splits` bricht mit einem `ValueError` ab, wenn Gold- oder
Kalibrierplätze einer Klasse unbesetzt bleiben und keine unbekannte Vorlage sie füllen
kann — typischerweise, weil eine eingefrorene Vorlage aus `TEMPLATES` entfernt wurde. Ein
stiller Fehlstand würde das Gold-Set unbemerkt unter die geforderten 50 Dokumente je Klasse
drücken. Und jeder Generatorlauf vergleicht `config/splits.yaml` mit dem tatsächlichen
Korpus und meldet sowohl neue als auch verwaiste Vorlagen (Einträge ohne zugehörige Vorlage
mehr). Verwaiste Einträge werden gemeldet, nicht automatisch gelöscht — ein Mensch
entscheidet.

## Dateinamen

Dateien heißen `doc-0001.pdf`, `doc-0002.docx`, … — fortlaufend über den gesamten Korpus in
der festen Reihenfolge von `TEMPLATES` (`assign_document_names`), unabhängig von Klasse,
Vorlage oder Format. Dasselbe gilt für den Dateinamen eines Mailanhangs innerhalb einer
`.eml`. Der Name trägt bewusst keine Information: Klasse, `template_id`, `variant` und
`split` stehen ausschließlich im Manifest (`data/generated/manifest.parquet`,
`config/splits.yaml`). Ein Dateiname wie das frühere `AGB-allgemein-00.pdf` wäre auf diesem
Korpus perfekt klassentrennscharf und in der Realität wertlos (Konzept § 6.3 nennt
„Dateinamen-Tokens" als Herkunftsmerkmal — eines, das ein Modell lernen würde, wenn der Name
es hergäbe). `test_dateiname_verraet_die_klasse_nicht` in `tests/test_generation_content.py`
deckelt die Trefferquote des bestmöglichen Dateiname-Raters.

Die `template_id`-Muster in der folgenden Tabelle (`GUTSCHRIFT-*` u. Ä.) beziehen sich auf
die gleichnamige Manifest-Spalte, nicht mehr auf den Dateinamen.

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
| 8 | Neutrale, fortlaufende Dateinamen ohne Klassen- oder Vorlagenbezug (`doc-0001.pdf`) | gesamter Korpus (`assign_document_names`) | verhindert die Abkürzung „Dateiname ⇒ Klasse“ |
| 9 | PDF-Seiten, DOCX-Abschnitte und XLSX-Tabellenblätter werden je Variante zufällig gruppiert, statt starr an der Blockzahl der Vorlage zu hängen | gesamter Korpus (`_page_groups`) | verhindert die Abkürzung „Segmentzahl ⇒ Klasse“ |

## Was noch fehlt

- Keine gescannten Dokumente — OCR ist Aufgabe 12 und braucht `ocrmypdf`.
- Keine englischen Dokumente. Anhang D zeigt, dass sie eigene Beispiele bräuchten.
- Keine echten Dokumente. Ein Modell, das hier glänzt und auf Realität einbricht, hat den
  Generator gelernt (Konzept § 9.2).
