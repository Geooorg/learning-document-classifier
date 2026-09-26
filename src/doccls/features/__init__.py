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
