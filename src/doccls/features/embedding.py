"""Das semantische Merkmal: ein Vektor je Dokument.

Zwei Umsetzungen hinter einem Protokoll, weil Konzept Anhang B die Wahl zwischen E5 und
BGE-M3 ausdrücklich **gemessen** und nicht entschieden haben will. Aufgabe 15 misst beide
gegeneinander; ohne gemeinsame Schnittstelle liefe die Messung über zwei Codepfade.

E5 verlangt das Präfix ``passage: `` vor jedem Text (Konzept § 6.1). Das Präfix sitzt
deshalb hier und nicht an der Aufrufstelle: Wird es beim Training gesetzt und bei der
Vorhersage vergessen, driften die Vektoren auseinander – und die Zahlen sehen weiter
plausibel aus.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from doccls.config import BGE_M3_MODEL_NAME, E5_MODEL_NAME, OLLAMA_URL
from doccls.features import FeatureConfig
from doccls.features.text import chunks, head_text, position_weights

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

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
        self._model: SentenceTransformer | None = None

    def _geladen(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self._model_name, device=_geraet())
        return self._model

    def embed(self, texts: list[str]) -> Vektoren:
        roh = self._geladen().encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(roh, dtype=np.float32)


def _geraet() -> str:
    """Auf Apple Silicon über MPS, sonst CPU. CUDA gibt es auf diesem Rechner nicht."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


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
