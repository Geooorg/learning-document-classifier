"""Der Korpus ist der Maßstab für alles Weitere. Seine Eigenschaften werden geprüft, nicht \
angenommen."""

from collections import Counter

from doccls.classes import load_classes
from doccls.generation.content import build_corpus


def test_korpus_ist_deterministisch() -> None:
    assert build_corpus(seed=42) == build_corpus(seed=42)


def test_jede_klasse_hat_genug_vorlagen_und_dokumente() -> None:
    """Fünf Vorlagen je Klasse gehen ins Gold-Set und ergeben dort 50 Dokumente (Konzept § 9.2)."""
    korpus = build_corpus()
    je_klasse = Counter(spec.class_key for spec in korpus)
    for klasse in load_classes().keys():
        assert je_klasse[klasse] >= 80, f"{klasse}: nur {je_klasse[klasse]} Dokumente"
        vorlagen = {s.template_id for s in korpus if s.class_key == klasse}
        assert len(vorlagen) >= 8, f"{klasse}: nur {len(vorlagen)} Vorlagen"


def test_format_verraet_die_klasse_nicht() -> None:
    korpus = build_corpus()
    for klasse in load_classes().keys():
        formate = {f for s in korpus if s.class_key == klasse for f in s.formats}
        assert len(formate) >= 2, f"{klasse} nur in {formate}"
    for fmt in ("pdf", "docx", "xlsx", "eml"):
        klassen = {s.class_key for s in korpus if fmt in s.formats}
        assert len(klassen) >= 3, f"{fmt} trägt nur {klassen}"


def test_jede_vorlage_liefert_zehn_unterscheidbare_varianten() -> None:
    """Alle 56 Vorlagen, nicht nur die, an der es zufällig auffällt. Exakte Dubletten
    blähen die scheinbare Abdeckung des Gold-Sets auf (Konzept § 9.2)."""
    je_vorlage: dict[str, list[str]] = {}
    for spec in build_corpus():
        text = " ".join(p for b in spec.blocks for p in b.paragraphs)
        je_vorlage.setdefault(spec.template_id, []).append(text)
    dubletten = {
        tid: f"{len(set(texte))}/{len(texte)} verschieden"
        for tid, texte in je_vorlage.items()
        if len(set(texte)) < len(texte)
    }
    assert not dubletten, f"Vorlagen mit wortgleichen Varianten: {dubletten}"


def test_gutschriften_sind_als_verwechslungsfall_gebaut() -> None:
    """Rechnung ↔ Gutschrift ist der harte Fall (Konzept Anhang D, Kosinus 0,964)."""
    korpus = build_corpus()
    gutschriften = [s for s in korpus if s.class_key == "GUTSCHRIFT"]
    text = " ".join(p for s in gutschriften for b in s.blocks for p in b.paragraphs)
    assert "Rechnung" in text, "Gutschriften müssen ihre Ursprungsrechnung nennen"


def test_alle_klassen_teilen_dieselbe_fusszeile() -> None:
    """Genau eine Fußzeile über alle Klassen – sonst wird das Layout zur Abkürzung
    statt des Inhalts (Konzept § 8.5). `<= 3` würde einen Fehler in `_fuss()` durchlassen."""
    fusszeilen = {s.blocks[-1].paragraphs[-1] for s in build_corpus()}
    assert len(fusszeilen) == 1, f"Erwartet eine gemeinsame Fußzeile, gefunden: {len(fusszeilen)}"
