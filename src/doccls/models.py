"""Datenmodelle und die zugehörigen Polars-Schemas.

Drei Ebenen:

* ``Document`` – eine Version einer Datei. Die Identität umfasst Pfad **und** Inhalts-Hash:
  Dieselbe Datei an zwei Stellen sind zwei Dokumente, sonst würde die zweite übersprungen
  und wäre nirgends verzeichnet. Geänderter Inhalt ergibt eine neue ID; die alte Version
  bleibt erhalten.
* ``Segment``  – eine Einheit im Dokument: PDF-Seite, Abschnitt, Tabellenblatt, Mailteil.
  Ergebnis der Extraktion. Wird gespeichert, damit Merkmale neu gebildet werden können,
  ohne die Originale erneut zu lesen.
* ``Prediction`` – das Ergebnis der Klassifikation für ein Dokument, samt den Zahlen, aus
  denen die Entscheidung entstanden ist, und der Herkunft (Modell- und Merkmalsversion).

IDs sind aus dem Inhalt abgeleitet und damit stabil: Ein erneuter Lauf erzeugt dieselben
IDs und überschreibt, statt zu verdoppeln.
"""

import hashlib
import math
import re
import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import Self

import polars as pl
from pydantic import BaseModel, Field, model_validator

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


ANHANG_TRENNER = "!"
"""Trennt im ``source_path`` eines Mailanhangs den Pfad der Elternmail vom Anhangnamen
(``eml/doc-0001.eml!doc-0001.pdf``). Sowohl ``pipeline.py`` (bildet den Pfad beim
tatsächlichen Einlesen) als auch ``generation/manifest.py`` (bildet ihn vorweg für die
Wahrheit) brauchen exakt dieselbe Regel – zwei unabhängige Kopien liefen irgendwann
auseinander, und der Fehler wäre unsichtbar: eine Manifestzeile, die kein ``join`` mehr
trifft."""


def anhang_pfad(eltern_pfad: str, anhang_name: str) -> str:
    """Den ``source_path`` eines Mailanhangs aus dem Pfad der Elternmail und dem Anhangnamen
    bilden – die einzige Stelle, die das Trennzeichen kennt."""
    return f"{eltern_pfad}{ANHANG_TRENNER}{anhang_name}"


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


class Decision(StrEnum):
    """Was mit einer Vorhersage geschieht (Konzept § 7.4).

    Zwei Werte, nicht drei: ``SONSTIGES`` ist eine **Klasse**, keine Entscheidung. Ein
    Dokument jenseits von δ bekommt ``class_key = SONSTIGES`` *und* ``REVIEW`` – die
    Restklasse wird nicht trainiert (``classes.yaml``: ``residual: true``), sie entsteht
    ausschließlich durch Ablehnung und ist damit nie ein automatisch entschiedener Fall.
    """

    AUTO = "AUTO"
    REVIEW = "REVIEW"


RESIDUAL_CLASS_KEY = "SONSTIGES"
"""Schlüssel der Restklasse aus ``config/classes.yaml`` (``residual: true``).

Als Konstante hier, weil ``decide.decide_one`` sie vergibt, ohne das Klassenschema zu
laden: Die Funktion bekommt nur die trainierten Klassen, und die Restklasse ist per
Definition nicht darunter. Dass Konstante und Konfiguration nicht auseinanderlaufen,
bindet ``test_restklasse_stimmt_mit_dem_klassenschema_ueberein`` – zwei Stellen mit
derselben Zeichenkette sind sonst genau die Falle, die beim übernächsten Umbau zuschlägt.
"""

VERTEILUNGS_TOLERANZ = 1e-6
"""Wie weit sich ``proba`` von der Summe 1 entfernen darf (``decide._als_verteilung``).

Nicht enger: Eine Softmax-Ausgabe in ``float32`` summiert sich nur auf etwa ``1e-7`` genau
zu 1, und eine strengere Schranke wiese gültige Eingaben ab. Nicht weiter: Ab ``1e-6``
ginge die Prüfung an dem vorbei, wogegen sie steht – an Rohwerten (``decision_scores``)
statt Wahrscheinlichkeiten, an einer Verteilung über andere Klassen als ``classes``, an
einer Spalte, die beim Umsortieren verloren ging.

**Die Konstante steht hier und nicht in** ``decide``, **obwohl sie dort geprüft wird.**
:class:`Prediction` muss sie kennen (siehe :func:`_spielraum`), und ``decide`` hängt
ohnehin von diesem Modul ab – umgekehrt ginge es nicht.
"""

TOLERANZ = 1e-6
"""Fester Anteil am Spielraum für die Ungleichungen in :class:`Prediction`.

Nicht enger, weil eine Verteilung aus ``float32``-Wahrscheinlichkeiten (``Zahlenreihe``
lässt beides zu) sich nur auf etwa ``1e-7`` genau zu 1 summiert; die daraus abgeleiteten
Schranken erben diesen Fehler. Nicht weiter, weil die Prüfungen sonst nichts mehr fangen:
Die kleinste hier abgewiesene Verletzung (Konfidenz 0,95 mit Margin 0,01) liegt um 0,89
daneben.

Der feste Anteil allein reicht nicht – siehe :func:`_spielraum`.
"""


def _spielraum(entropie: float) -> float:
    """Der Spielraum der Ungleichungen, gekoppelt an :data:`VERTEILUNGS_TOLERANZ`.

    Eine Verteilung darf sich um ``ε ≤ VERTEILUNGS_TOLERANZ`` von der Summe 1 entfernen.
    Eine um ``ε`` verkleinerte Verteilung hat aber eine um bis zu ``ε · H`` kleinere
    Entropie, während die Schranke ``−ln(max p)`` nur um ``ε`` steigt. Der Fehlbetrag
    wächst also mit der Entropie und damit mit der Klassenzahl – ein **fester** Spielraum
    wies deshalb gültige Eingaben ab, gemessen ab vier Klassen (sieben gleichverteilte
    Klassen, Summe ``1 − 6·10⁻⁷``):

        ValidationError: Entropie 1.9459095815090444 ist zu klein fuer die Konfidenz
        0.14285705714285715 … >= 1.9459107490554932

    Deshalb ist der Spielraum an die Entropie gebunden. Die ``+ 1`` fängt den konstanten
    Anteil ``ε`` der Margin-Ungleichung mit ab, damit beide Prüfungen dieselbe Formel
    benutzen. Auf echte Verletzungen wirkt sich das nicht aus: Der Zuschlag liegt bei
    sieben Klassen unter ``3 · 10⁻⁶``, die kleinste abgewiesene Verletzung 0,89 daneben.
    """
    return TOLERANZ + VERTEILUNGS_TOLERANZ * (entropie + 1.0)


class Prediction(BaseModel):
    """Die Vorhersage für ein Dokument – Klasse, die drei Zahlen dahinter, Entscheidung,
    Herkunft (Konzept § 1, § 7.2, § 7.4).

    Die drei Zahlen messen Verschiedenes und stehen deshalb alle drei da: ``confidence``
    (``max p̂``) trägt die Schwelle, ``margin`` (``p̂₍₁₎ − p̂₍₂₎``) unterscheidet „unsicher
    zwischen zweien" von „unsicher zwischen allen" und steuert ab Phase 3 die Prüfliste,
    ``entropy`` (``−Σ p̂ log p̂``) erkennt breite Unsicherheit.

    **Warum das Modell rechnet, statt nur Felder zu halten.** Die vier Zahlen stammen aus
    *einer* Verteilung und können einander widersprechen. Ein Widerspruch ist hier kein
    Schönheitsfehler: Er bedeutet, dass Zahl und Klasse aus verschiedenen Quellen stammen –
    der lautloseste Fehler dieser Phase. Geprüft wird alles, was ohne Kenntnis der
    Klassenzahl entscheidbar ist:

    * ``0 < confidence <= 1``. Eine 0 gäbe es nur, wenn alle Klassen die
      Wahrscheinlichkeit 0 hätten; das ist keine Verteilung.
    * ``margin <= confidence``, denn ``p̂₍₁₎ − p̂₍₂₎ <= p̂₍₁₎``. Gleichheit heißt: die
      zweitbeste Klasse hat 0.
    * ``margin >= 2 · confidence − 1``, denn ``p̂₍₁₎ + p̂₍₂₎ <= 1``. Das ist die Schranke,
      die „Konfidenz 0,95 bei Margin 0,01" abweist: Dahinter stünden zwei Klassen mit
      zusammen 1,89.
    * ``entropy >= −ln(confidence)``, denn jedes ``p̂ᵢ <= max p̂``. Diese Ungleichung
      ersetzt die Prüfung gegen die Klassenzahl, die das Modell nicht kennt: Eine Entropie
      in ``log₁₀`` statt ``ln`` fällt bei drei gleich wahrscheinlichen Klassen sofort auf
      (0,477 statt 1,099), und eine Entropie 0 neben einer Konfidenz von 0,5 ebenso.

    Eine **obere** Entropieschranke gibt es hier bewusst nicht: Sie wäre ``ln n`` und
    hinge an der Klassenzahl, die kein Feld trägt. Sie zu erzwingen hieße, ein Feld
    ``n_classes`` aufzunehmen, das keine Auswertung braucht. ``decide_one`` kennt ``n``
    und hält sie durch Konstruktion ein.
    """

    model_config = {"frozen": True, "protected_namespaces": ()}
    """``protected_namespaces`` leer: Pydantic hält sich das Präfix ``model_`` für eigene
    Attribute frei und warnt sonst bei ``model_version``. Das Feld heißt im Konzept (§ 1,
    § 10) so, und ein anderer Name ginge in jede gespeicherte Zeile."""

    document_id: str = Field(min_length=1)
    class_key: str = Field(min_length=1)
    confidence: float = Field(gt=0.0, le=1.0)
    """``max_c p̂(c|x)`` – die berichtete Zahl, Grundlage der Schwelle τ."""

    margin: float = Field(ge=0.0, le=1.0)
    """``p̂₍₁₎ − p̂₍₂₎`` – das Auswahlkriterium der Prüfliste (Konzept § 7.2)."""

    entropy: float = Field(ge=0.0)
    """``−Σ p̂ log p̂``, natürlicher Logarithmus."""

    ood_score: float = Field(ge=0.0, le=2.0)
    """Kosinusabstand zum Mittel der k nächsten Trainingsnachbarn. Der Wertebereich
    ``[0, 2]`` ist der von ``1 − cos``: 0 gleiche Richtung, 1 rechter Winkel, 2
    Gegenrichtung."""

    decision: Decision
    model_version: str = Field(min_length=1)
    """Herkunft: ohne Modell- und Merkmalsversion ist eine Vorhersage nicht
    nachvollziehbar (Konzept § 1, Leitplanke 2) und zwei Auswertungen sind nicht
    vergleichbar."""

    feature_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _pruefe_unmoegliche_kombinationen(self) -> Self:
        """Die drei Ungleichungen aus dem Klassendoc – jede benennt, was sie ausschließt."""
        if self.margin > self.confidence:
            raise ValueError(
                f"Margin {self.margin} ist groesser als die Konfidenz {self.confidence}. "
                "Die Margin ist der Abstand der besten zur zweitbesten Klasse und kann "
                "die beste nicht uebersteigen - hier stammen die beiden Zahlen aus "
                "verschiedenen Verteilungen."
            )
        kleinste_margin = 2.0 * self.confidence - 1.0
        if self.margin < kleinste_margin - _spielraum(self.entropy):
            raise ValueError(
                f"Margin {self.margin} ist zu klein fuer die Konfidenz {self.confidence}: "
                f"Die zweitbeste Klasse haette dann {self.confidence - self.margin}, und "
                f"beide zusammen mehr als 1. Moeglich sind erst "
                f"{kleinste_margin} und mehr."
            )
        kleinste_entropie = -math.log(self.confidence)
        if self.entropy < kleinste_entropie - _spielraum(self.entropy):
            raise ValueError(
                f"Entropie {self.entropy} ist zu klein fuer die Konfidenz "
                f"{self.confidence}. Weil keine Klasse wahrscheinlicher ist als die "
                f"beste, gilt immer -Summe p ln p >= -ln(max p) = {kleinste_entropie}. "
                "Ein kleinerer Wert kommt aus einer anderen Verteilung oder aus einem "
                "anderen Logarithmus."
            )
        return self


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
