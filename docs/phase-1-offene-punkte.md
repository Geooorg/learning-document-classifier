# Offene Punkte nach Phase 1

Stand: Abschluss des Zweigs `phase-0-1-ingestion`, 130 Tests.

Diese Liste stammt aus der Schlussdurchsicht über den ganzen Zweig und aus den Prüfungen
je Aufgabe. Alles, was hier steht, ist **belegt** — jeder Punkt nennt die Eingabe und die
beobachtete Ausgabe. Was behoben wurde, steht nicht hier, sondern im Code.

Die Reihenfolge ist nach Dringlichkeit für Phase 2 sortiert, nicht nach Schwere.

## Vor dem Beginn von Phase 2

### 1. Anhänge haben keine Wahrheit im Manifest

`documents` enthält 570 Zeilen, `manifest.parquet` 560 — die zehn Mailanhänge fehlen.
`docs/testdaten.md` nennt das Manifest „die bekannte Wahrheit". Wer später
`documents ⋈ manifest` bildet, verliert zehn Dokumente stillschweigend.

Dazu kommt: Mailkörper und zugehöriger Anhang haben eine Textähnlichkeit von 0,82. Erbt
der Anhang die Klasse des Elterndokuments, geht dasselbe Dokumentenpaar zweimal in
dieselbe Messung ein — genau das, was der Schnitt nach Dokumentfamilie (Konzept § 9.2)
verhindern soll.

**Zu tun:** Anhänge mit eigener Zeile ins Manifest (`class_key`, `template_id`, **derselbe
Split wie die Elternvorlage**). Und entscheiden, ob Mailkörper und Anhang weiterhin
denselben Inhalt tragen sollen; `scripts/generate_documents.py` nennt das eine bewusste
Vereinfachung.

### 2. `strip_boilerplate` greift auf dem Bestand nicht und verfehlt die zwei häufigsten Realfälle

Gemessen über alle 560 Dateien: 450.245 Zeichen vorher, 450.245 Zeichen nachher, null
Dokumente mit Entfernung. Das ist erwartbar — der Generator setzt die Fußzeile nur einmal
ans Dokumentende, nicht je Seite.

Wichtiger sind die zwei Fälle, die die Funktion prinzipiell nicht trifft:

- **Fußzeile mit Seitenzahl.** `"… Inhalt der Seite 1. Seite 1 von 5."` wird nicht
  entfernt, weil verglichen wird auf exakte Zeichenkettengleichheit. Konzept § 5 verlangt
  „fast gleich". „Seite 1 von 3" ist wörtlich das Beispiel, das das Konzept als klassische
  Abkürzung nennt.
- **Kopfzeile ohne Satzzeichen.** Die Extraktion hat die Zeilenumbrüche bereits zu
  Leerzeichen zusammengefasst (`models.normalize_text`). Eine Kopfzeile ohne Punkt klebt
  danach am ersten Fließtextsatz und ist auf jeder Seite verschieden.

**Zu tun:** auf Zeilen statt Sätze arbeiten — dafür müsste die Extraktion die Seitenzeilen
mitliefern —, Ziffernfolgen vor dem Vergleich maskieren, und mindestens einen Test mit
seitenzahlbehafteter Fußzeile.

### 3. Verschachtelte Mails werden still verworfen

`message/rfc822`-Anhänge liefern bei `get_payload(decode=True)` `None` und fallen durch
den Filter in `extraction/mail.py`. Eine weitergeleitete Mail samt Inhalt verschwindet
spurlos. Konzept § 5 fordert ausdrücklich „rekursiv (Mail in Mail in ZIP)", und
`pipeline.py` trägt den Kommentar „Rekursiv, weil Mails Mails enthalten können" — für
diesen Fall kann die Rekursion nie auslösen.

Nachgewiesene Behebung: bei `get_content_type() == "message/rfc822"` stattdessen
`teil.get_payload(0).as_bytes()` verwenden. Der Korpus enthält nur
`application/octet-stream`-Anhänge, deshalb sieht es kein Test.

## Vor dem ersten Einsatz auf echten Beständen

### 4. Die Mail-Erkennung an Inhalt hat drei belegte Lücken

Bewusste Entscheidung: nicht beheben, solange Mails als `.eml` vorliegen und über den
Endungs-Rückfall erkannt werden. Die Inhaltsheuristik ist nur die Zusatzspur. Belegt sind:

- `MBOX_LINE` (`^From \S+`) trifft „From the Desk of the CEO"; ein Rundschreiben mit
  `Date:` und `To:` im Rumpf wird als Mail gelesen.
- Ein roher HTTP-Antwortkopf ohne Statuszeile wird über `Content-Type:` + `Date:` zur Mail.
- Umgekehrt fällt eine Mail durch, deren erste zwei bekannte Felder hinter mehr als
  16 Zeilen Zustellkette liegen — die Kippstelle liegt exakt bei 15 `X-`-Kopfzeilen.

Tragfähige Regel, wenn echte Bestände es verlangen: den zusammenhängenden Kopfblock bis
zur Leerzeile auswerten statt eines Zeilenfensters, und mindestens ein mailspezifisches
Pflichtfeld verlangen (`From`/`To`/`Subject`/`Message-ID`).

### 5. Reine HTML-Mails landen als Rohmarkup im Segment

`extraction/mail.py` bevorzugt bei `multipart/alternative` die Textvariante und begründet
das damit, dass Markup „zusätzliches Rauschen" wäre — speichert es im Rückfall aber genau
so: `'<html><body><h1>Rechnung …</h1></body></html>'`. In echten Beständen sind
HTML-only-Mails die Mehrheit. Tag-Entfernung wie in `generation/writers.plain_text` genügt.

### 6. XLSX-Formeln werden sichtbar gemacht, aber nicht ausgewertet

Eine Formelzelle ohne zwischengespeicherten Wert liefert jetzt `=B2*C2` statt zu
verschwinden. Für die spätere Merkmalsbildung ist das noch keine Zahl — der Betrag ist
aber genau das Merkmal, das Rechnung von Gutschrift trennt. Echte Formelauswertung ist
eine eigene Aufgabe.

### 7. DOCX: Reihenfolge und Kopf-/Fußzeilen

Tabellen werden nicht in Dokumentreihenfolge eingereiht, und `section.header`/`footer`
werden gar nicht gelesen. Folge: dasselbe logische Dokument hat je Format eine andere
Textmenge — die Fußzeile ist im PDF ein eigenes Segment, im DOCX unsichtbar. Das ist
klassenunabhängig und damit keine Abkürzung, verzerrt aber jedes längenbasierte Merkmal.

## Kleinigkeiten

- **Suite ohne Bestand rot.** Ohne erzeugten Korpus scheitern 24 von 130 Tests, und `data/`
  ist nicht versioniert. Auf einem frischen Klon ist die Suite rot. Die Meldungen weisen
  auf den Generator hin, aber es fehlt eine Fixture, die ihn bei leerem `RAW_DIR` selbst
  anwirft — oder ein dokumentierter Vorlaufschritt.
- **`ingested_at` ohne Zeitzonenzwang.** Ein naives `datetime` wird angenommen und still
  als UTC gelesen. Ein `AwareDatetime`-Typ oder ein Validator genügt.
- **`SegmentKind.ZEILE` wird nirgends erzeugt.** Konzept § 5 nennt für XLSX „Blatt, dann
  Zeilenblock". Entweder den Enumwert entfernen oder die Vertagung vermerken.
- **`puremagic`** ist in `pyproject.toml` deklariert und wird nirgends importiert;
  `detect.py` erkennt von Hand. Entfernen oder die Abweichung begründen.
- **`data/parquet` ist inkrementell.** Nach einer Änderung an einer Schwelle oder an der
  Extraktion korrigiert kein weiterer Lauf die schon geschriebenen Zeilen — sie werden
  übersprungen. `data/parquet` muss dann von Hand gelöscht werden.
- **`segment_id`** hängt an `(document_id, index)`, nicht am Text. Für die idempotente
  Ablage richtig; wird erst relevant, wenn Merkmalsvektoren darauf zwischengespeichert
  werden. Konzept § 6.4 löst das über `feature_version`.

## Was diese Liste über das Vorgehen sagt

Fast jeder Mangel, der in diesem Zweig gefunden wurde, stammt aus dem Plan, nicht aus
einer Umsetzung — und fast alle waren derselbe Typ: **ein Test, der die Anzahl bindet,
aber nicht die Inhaltsmenge.** Fünf Mutationen mit stillem Inhaltsverlust überlebten eine
komplette Testsuite; eine XLSX-Extraktion ließ ein Rechnungsblatt auf acht Zeichen
schrumpfen; eine OCR-Schwelle lag mitten in der Verteilung der echten Dokumente.

Das einzige Mittel, das das zuverlässig gefunden hat, war die **Mutationsprobe**: die
fertige Implementierung gezielt falsch machen und nachsehen, welcher Test rot wird. Ein
Test, der bei seiner Mutation grün bleibt, prüft nichts. Für Phase 2 gilt dasselbe, und
dort wiegt es schwerer: Ein Klassifikator, der auf einer Abkürzung lernt, meldet hohe
Konfidenz für die falsche Antwort.
