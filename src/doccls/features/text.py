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
