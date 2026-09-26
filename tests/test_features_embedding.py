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
