"""Klassenschema laden und prüfen.

Das Schema ist Konfiguration (`config/classes.yaml`), kein Code: Klassen ändern sich, ohne
dass etwas neu gebaut wird. Die `description` dient doppelt – als Definition für Labelnde
und als Kaltstart-Prototyp für die Einbettung (Konzept § 8.1).
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from doccls.config import PROJECT_ROOT

DEFAULT_PATH = PROJECT_ROOT / "config" / "classes.yaml"


class ClassDef(BaseModel):
    """Eine Dokumentklasse."""

    model_config = {"populate_by_name": True}

    key: str
    name: str
    description: str
    not_: str = Field(alias="not")
    """Wogegen die Klasse abzugrenzen ist. Ohne diese Angabe sind sich Labelnde uneins."""

    residual: bool = False
    """Restklasse: wird nicht trainiert, sondern durch Ablehnung erreicht (Konzept § 7.4)."""


class ClassSchema(BaseModel):
    """Alle Klassen mit Versionsstand. Die Version ordnet Auswertungen einem Schema zu."""

    version: int
    classes: list[ClassDef]

    def keys(self) -> list[str]:
        return [klasse.key for klasse in self.classes]

    def get(self, key: str) -> ClassDef:
        for klasse in self.classes:
            if klasse.key == key:
                return klasse
        raise KeyError(f"Unbekannte Klasse {key!r}. Bekannt: {', '.join(self.keys())}")

    def trainable(self) -> list[ClassDef]:
        """Klassen ohne die Restklasse – nur diese bekommen Trainingsbeispiele."""
        return [klasse for klasse in self.classes if not klasse.residual]


def load_classes(path: Path | None = None) -> ClassSchema:
    """Schema laden und prüfen. Doppelte Schlüssel sind ein Fehler, kein stiller Überschreiber."""
    quelle = path or DEFAULT_PATH
    schema = ClassSchema.model_validate(yaml.safe_load(quelle.read_text(encoding="utf-8")))
    schluessel = schema.keys()
    doppelt = {k for k in schluessel if schluessel.count(k) > 1}
    if doppelt:
        raise ValueError(f"Klassenschlüssel doppelt vergeben: {', '.join(sorted(doppelt))}")
    return schema
