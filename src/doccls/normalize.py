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
        ". ".join(satz for satz in _saetze(text) if satz not in layout).strip() for text in texts
    ]
