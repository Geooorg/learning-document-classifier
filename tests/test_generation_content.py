"""Der Korpus ist der Maßstab für alles Weitere. Seine Eigenschaften werden geprüft, nicht \
angenommen."""

import re
from collections import Counter
from collections.abc import Hashable
from datetime import UTC, datetime
from pathlib import Path

from doccls.classes import load_classes
from doccls.extraction import extract
from doccls.extraction.mail import extract_eml
from doccls.generation.content import build_corpus
from doccls.generation.manifest import assign_document_names
from doccls.generation.writers import write
from doccls.models import Document
from doccls.normalize import strip_boilerplate


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


def _bestmoeglicher_rater(merkmale: list[Hashable], klassen: list[str]) -> float:
    """Trefferquote des bestmöglichen Raters, der je Merkmalswert stur die unter den
    Trainingsdaten häufigste Klasse vorhersagt. Kein denkbarer Rater, der nur dieses eine
    Merkmal kennt, kann darüber hinauskommen – die Zahl ist also eine echte obere Schranke
    für das, was das Merkmal an Klasseninformation trägt."""
    je_wert: dict[Hashable, Counter[str]] = {}
    for wert, klasse in zip(merkmale, klassen, strict=True):
        je_wert.setdefault(wert, Counter())[klasse] += 1
    richtig = sum(zaehler.most_common(1)[0][1] for zaehler in je_wert.values())
    return richtig / len(klassen)


_WORT = re.compile(r"[A-Za-z]+")


def _dateiname_tokens(name: str) -> tuple[str, ...]:
    """Alphabetische Tokens eines Dateinamens ohne Endung. Ziffern (die laufende Nummer)
    werden verworfen: Sie sind pro Dokument einzigartig und wären als „Token“ nur eine
    Erinnerung an das eine Dokument, kein wiederverwendbares Muster – ein Rater, der sie
    zulässt, würde sich selbst betrügen."""
    return tuple(_WORT.findall(Path(name).stem))


def test_dateiname_verraet_die_klasse_nicht() -> None:
    """Deckelt die Trefferquote des bestmöglichen Raters, der aus den alphabetischen Tokens
    des Dateinamens auf die Klasse schließt (häufigste Klasse je Tokenmuster).

    Vor der Behebung (Klassenschlüssel im Dateinamen, z. B. ``AGB-allgemein-00.pdf``) trifft
    dieser Rater 480 von 560 Dokumenten (85,7 %) – bei einer Grundrate von 14 % (häufigste
    Klasse, kein Merkmal). Die Schranke hier liegt bei 20 %: knapp über der Grundrate, aber
    weit unter dem historischen Wert. Ein neutraler Name wie ``doc-0001.pdf`` enthält außer
    dem konstanten Token „doc“ nichts, das mit der Klasse korreliert – der bestmögliche
    Rater darf hier kaum besser sein als blindes Raten. Wer diese Schranke anhebt, lässt
    wieder Klasseninformation in den Dateinamen zurück.
    """
    korpus = build_corpus()
    namen = assign_document_names(korpus)
    tokens = [_dateiname_tokens(name) for name in namen]
    klassen = [spec.class_key for spec in korpus]
    trefferquote = _bestmoeglicher_rater(list(tokens), klassen)
    assert trefferquote <= 0.20, (
        f"Dateiname-Rater trifft {trefferquote:.1%} der Dokumente – der Dateiname verrät die Klasse"
    )


def test_struktur_verraet_die_klasse_nicht(tmp_path: Path) -> None:
    """Deckelt die Trefferquote des bestmöglichen Raters, der aus (Format, Segmentzahl) auf
    die Klasse schließt (häufigste Klasse je Merkmalspaar) – auf echt geschriebenen und
    wieder eingelesenen Dokumenten, wie ``ingest.py`` sie auch sähe.

    Vor der Behebung (``write_pdf`` legte einen Block je Seite an, die Blockzahl steht je
    Vorlage fest) trifft dieser Rater 61 % der 560 Dokumente. Der Boden ist 20 %: Format
    allein trifft schon so viel, weil XLSX weder VERTRAG noch AGB trägt – kein Rater, der
    nur (Format, Segmentzahl) kennt, kommt darunter. Die Schranke hier liegt bei 30 %:
    spürbar über dem Boden, aber weit unter dem historischen Wert, weil ``_page_groups``
    PDF-Seiten, DOCX-Abschnitte und XLSX-Tabellenblätter je Variante aus dem geseedeten
    Zufall der Vorlage zufällig gruppiert, statt sie starr an der Blockzahl hängen zu
    lassen. Wer diese Schranke anhebt, lässt die Segmentzahl wieder zur Abkürzung werden.

    Drei Zahlen, damit die Schranke nachvollziehbar bleibt: Boden 19,6 % (Format allein),
    heute gemessen 29,8 %, historisch 61 %. Die Schranke liegt bei 38 % und nicht dicht
    über dem Messwert – ein Test mit zwei Zehntel Prozentpunkten Luft reißt beim nächsten
    harmlosen Eingriff an einer Vorlage und sagt dann nichts über die Abkürzung aus. Bei
    38 % werden beide vorgeführten Rückfälle noch gefangen: Streuung ganz aus ergibt
    62,5 %, Streuung nur in DOCX und XLSX aus ergibt 43,2 %.
    """
    korpus = build_corpus()
    merkmale: list[Hashable] = []
    for index, spec in enumerate(korpus):
        fmt = spec.formats[0]
        pfad = write(tmp_path / f"{index:04d}", spec, fmt)
        daten = pfad.read_bytes()
        dokument = Document.create(
            source_path=str(pfad),
            media_type="application/octet-stream",
            content=daten,
            ingested_at=datetime.now(UTC),
        )
        if fmt == "eml":
            segmente, _ = extract_eml(dokument, daten)
        else:
            segmente = extract(dokument, daten)
        texte = strip_boilerplate([segment.text for segment in segmente])
        merkmale.append((fmt, sum(1 for text in texte if text)))
    klassen = [spec.class_key for spec in korpus]
    trefferquote = _bestmoeglicher_rater(merkmale, klassen)
    assert trefferquote <= 0.38, (
        f"Struktur-Rater trifft {trefferquote:.1%} der Dokumente – (Format, Segmentzahl) "
        "verrät die Klasse"
    )


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
