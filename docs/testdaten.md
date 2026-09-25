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
