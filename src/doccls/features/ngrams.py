"""Der Wortlaut-Block: TF-IDF über Zeichen-n-Gramme, danach SVD.

Konzept § 6.2 und Anhang A. Zeichen-n-Gramme statt Wort-n-Gramme aus zwei deutschen
Gründen: Komposita („Rechnung" und „Rechnungsbetrag" teilen Zeichenfolgen, aber kein Wort)
und OCR-Robustheit („Rechnunq" ist für ein Zeichenmodell fast dasselbe wie „Rechnung").

**Die Reihenfolge ist der kritische Teil dieses Moduls.** ``fit`` darf ausschließlich
Trainingstexte sehen. Wer TF-IDF und SVD auf allen Dokumenten anpasst und erst danach
aufteilt, hat dem Modell die Wortstatistik des Gold-Sets mitgegeben – die Zahlen sehen dann
besser aus, als sie sind. Das ist keine Vorsichtsmaßnahme, sondern der häufigste stille
Fehler dieser Art von Pipeline.
"""

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from doccls.features import FeatureConfig

SVD_SEED = 7
"""Fest, damit zwei Läufe vergleichbar sind. ``TruncatedSVD`` ist randomisiert."""


class NgramBlock:
    """TF-IDF über Zeichen-n-Gramme, auf ``svd_components`` Dimensionen verdichtet."""

    def __init__(self, config: FeatureConfig) -> None:
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(config.ngram_min, config.ngram_max),
            max_features=config.ngram_max_features,
            lowercase=True,
        )
        self._svd = TruncatedSVD(n_components=config.svd_components, random_state=SVD_SEED)
        self.dimension = config.svd_components
        self._angepasst = False

    def fit(self, texts: list[str]) -> None:
        """Nur mit Trainingstexten aufrufen – siehe Moduldoc."""
        matrix = self._vectorizer.fit_transform(texts)
        self._svd.fit(matrix)
        # TruncatedSVD kappt die Komponentenzahl stillschweigend auf die Zahl der Texte:
        # Auf den 140 Trainingstexten lieferte svd_components=256 nur 140 Spalten, während
        # ``dimension`` weiter 256 zusicherte. Die Breite hinge dann von der Größe der
        # Trainingsmenge ab, die in keine feature_version eingeht.
        breite = self._svd.components_.shape[0]
        if breite != self.dimension:
            raise ValueError(
                f"svd_components={self.dimension} ist nicht erreichbar: Die SVD liefert auf "
                f"{len(texts)} Texten nur {breite} Komponenten. svd_components in "
                "config/features.yaml muss unter der Zahl der Trainingstexte liegen."
            )
        self._angepasst = True

    def _pruefe_angepasst(self) -> None:
        if not self._angepasst:
            raise RuntimeError(
                "NgramBlock wurde nicht angepasst – erst fit(trainingstexte) aufrufen. "
                "Ein nicht angepasster Block wuerde Nullen liefern, und die traegen sich "
                "lautlos bis in die Metriken durch."
            )

    def transform(self, texts: list[str]) -> npt.NDArray[np.float32]:
        self._pruefe_angepasst()
        matrix = self._vectorizer.transform(texts)
        return np.asarray(self._svd.transform(matrix), dtype=np.float32)

    def tfidf_only(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Die TF-IDF-Matrix vor der SVD – nur für Tests und zur Fehlersuche.

        Bei wenigen Texten hat die SVD zu wenig zu tun, um Verhältnisse zwischen
        Textpaaren stabil zu erhalten; für die Prüfung der n-Gramm-Eigenschaft ist die
        rohe Matrix deshalb die ehrlichere Ebene.
        """
        self._pruefe_angepasst()
        return np.asarray(self._vectorizer.transform(texts).todense(), dtype=np.float32)

    def vokabular(self) -> list[str]:
        """Das angepasste n-Gramm-Vokabular – nur für Tests und zur Fehlersuche.

        Wird in Aufgabe 9 für den Leckagetest gebraucht: ein Vokabular, das
        Zeichenfolgen enthält, die nur im Gold-Set vorkommen, wäre der Beweis, dass
        ``fit`` mehr als die Trainingstexte gesehen hat.
        """
        self._pruefe_angepasst()
        return list(self._vectorizer.get_feature_names_out())


def build_ngram_block(config: FeatureConfig) -> NgramBlock:
    return NgramBlock(config)
