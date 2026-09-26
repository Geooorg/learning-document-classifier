"""Zwischen Segment und Embedder: Text zusammensetzen, in Chunks schneiden, gewichten."""

import math

from doccls.features.text import (
    chunks,
    document_text,
    document_text_ohne_layout,
    head_text,
    position_weights,
)
from doccls.models import Segment, SegmentKind
from doccls.normalize import strip_boilerplate


def segment(index: int, text: str) -> Segment:
    return Segment(
        segment_id=f"s{index}",
        document_id="d1",
        index=index,
        kind=SegmentKind.ABSCHNITT,
        locator=f"Abschnitt {index}",
        heading=None,
        text=text,
    )


def test_dokumenttext_folgt_der_segmentreihenfolge() -> None:
    """Die Reihenfolge trägt Bedeutung: Der Kopf eines Dokuments steht vorn, und die
    Positionsgewichtung baut darauf auf. Eine nach document_id sortierte oder gar
    unsortierte Zusammenfuehrung wuerde das zunichtemachen."""
    segmente = [segment(2, "drittens"), segment(0, "erstens"), segment(1, "zweitens")]
    assert document_text(segmente) == "erstens zweitens drittens"


def test_dokumenttext_verliert_kein_zeichen() -> None:
    """Bindet die Inhaltsmenge, nicht die Segmentzahl: Eine Umsetzung, die nur das erste
    Segment nimmt oder jedes auf 50 Zeichen kuerzt, muss hier auffallen."""
    texte = [f"Segment {i} mit reichlich eigenem Inhalt an dieser Stelle" for i in range(8)]
    segmente = [segment(i, t) for i, t in enumerate(texte)]
    ergebnis = document_text(segmente)
    for t in texte:
        assert t in ergebnis, f"{t!r} fehlt im Dokumenttext"
    assert len(ergebnis) >= sum(len(t) for t in texte)


def test_chunks_verlieren_keinen_text() -> None:
    """Der haeufigste stille Fehler beim Chunken: Der letzte, unvollstaendige Chunk faellt
    weg, oder die Ueberlappung frisst Zeichen."""
    text = "".join(f"{i:04d}-" for i in range(1000))  # 5000 Zeichen
    teile = chunks(text, chunk_chars=1200)
    assert "".join(teile) == text, "Zusammengefuegt muessen die Chunks den Text ergeben"
    assert len(teile) == math.ceil(len(text) / 1200)


def test_chunks_bei_leerem_text() -> None:
    assert chunks("", chunk_chars=1200) == []


def test_kurzer_text_ergibt_genau_einen_chunk() -> None:
    assert chunks("kurz", chunk_chars=1200) == ["kurz"]


def test_positionsgewichte_fallen_und_summieren_sich_zu_eins() -> None:
    """Konzept § 6.1: w_i = 1/(1 + i/4), normiert. Frueh zaehlt mehr."""
    gewichte = position_weights(5, decay=4.0)
    assert math.isclose(sum(gewichte), 1.0)
    assert all(a > b for a, b in zip(gewichte, gewichte[1:], strict=False)), (
        "Die Gewichte muessen streng fallen – sonst zaehlt der Dokumentkopf nicht mehr"
    )


def test_positionsgewichte_treffen_die_formel() -> None:
    """Bindet die Formel selbst, nicht nur ihre Form: Eine beliebige andere fallende Folge
    (etwa 1/(i+1)) wuerde den Test oben bestehen."""
    roh = [1.0 / (1.0 + i / 4.0) for i in range(5)]
    erwartet = [w / sum(roh) for w in roh]
    for ist, soll in zip(position_weights(5, decay=4.0), erwartet, strict=True):
        assert math.isclose(ist, soll)


def test_ein_einziger_chunk_bekommt_gewicht_eins() -> None:
    assert position_weights(1, decay=4.0) == [1.0]


def test_kopftext_schneidet_bei_der_vorgabe() -> None:
    text = "x" * 5000
    assert len(head_text(text, head_chars=1000)) == 1000


def test_kopftext_kuerzer_als_vorgabe_bleibt_ganz() -> None:
    assert head_text("kurz", head_chars=1000) == "kurz"


def test_decay_wirkt_sich_aus() -> None:
    """Bindet den Parameter selbst, nicht nur die Formel bei einem festen Wert.

    Alle anderen Gewichtungstests rufen mit ``decay=4.0`` auf – ein fest verdrahtetes 4.0
    in der Implementierung ueberlebt sie deshalb alle. Dann stuende in ``features.yaml``
    ein ``position_decay``, das nichts bewirkt, waehrend die ``feature_version`` sich bei
    seiner Aenderung sehr wohl aendert: Alle Merkmalsvektoren wuerden neu gerechnet und
    kaemen identisch wieder heraus, ohne dass irgendetwas auffiele.

    Kleines Decay heisst schneller Abfall, also mehr Gewicht auf dem ersten Chunk.
    """
    steil = position_weights(5, decay=1.0)
    flach = position_weights(5, decay=16.0)
    assert steil[0] > flach[0], (
        f"decay wirkt nicht: erstes Gewicht {steil[0]:.4f} (decay=1) gegen "
        f"{flach[0]:.4f} (decay=16)"
    )


def test_fusszeile_auf_allen_seiten_eines_dokuments_verschwindet() -> None:
    """Wofuer ``strip_boilerplate`` gebaut ist: eine Zeile, die auf jeder Seite DIESES
    Dokuments steht, ist Layout und kein Inhalt."""
    seiten = [
        segment(i, f"Inhalt der Seite {i} steht hier ausfuehrlich. Muster GmbH HRB 44821")
        for i in range(5)
    ]
    ergebnis = document_text_ohne_layout(seiten)
    assert "HRB 44821" not in ergebnis
    assert all(f"Inhalt der Seite {i}" in ergebnis for i in range(5))


def test_satz_der_in_vielen_dokumenten_vorkommt_bleibt_stehen() -> None:
    """Der Unterschied, auf den es ankommt – und der Grund, warum diese Funktion die
    Segmente EINES Dokuments bekommt und nicht die Texte vieler.

    Waere ``strip_boilerplate`` korpusweit ueber alle Dokumenttexte angewandt (so war es
    zuerst verdrahtet), fiele ein Satz weg, der in der Mehrzahl der Dokumente je EINMAL
    vorkommt. Das ist aber kein Layout, sondern kann genau der Inhalt sein, der eine
    Klasse auszeichnet – etwa eine Zahlungsklausel, die in fast jeder Rechnung steht.

    Hier steht der Satz in jedem Dokument genau einmal. Je Dokument betrachtet ist er
    damit kein wiederkehrendes Layout und muss bleiben.
    """
    gemeinsam = "Zahlbar innerhalb von vierzehn Tagen ohne Abzug"

    def seiten_von(i: int) -> list[Segment]:
        # Der gemeinsame Satz muss eine eigene, punktabgegrenzte Einheit sein – auch nach
        # dem Zusammenfuegen der Segmente. Ohne den Punkt verschmilzt er mit dem folgenden
        # Text zu einer je Dokument verschiedenen Einheit, und der Test unterschiede die
        # beiden Verdrahtungen nicht mehr. Genau so war er zuerst gebaut.
        return [
            segment(0, f"Rechnung Nummer {i} an die Musterfirma AG. {gemeinsam}."),
            segment(1, f"Position eins mit Betrag {i}00,00 EUR und erlaeuternder Text."),
            segment(2, f"Position zwei mit Betrag {i}50,00 EUR und erlaeuternder Text."),
        ]

    for i in range(10):
        assert gemeinsam in document_text_ohne_layout(seiten_von(i)), (
            "Ein Satz, der je Dokument nur einmal vorkommt, wurde entfernt – "
            "strip_boilerplate laeuft ueber den Korpus statt ueber die Seiten"
        )

    # Gegenprobe: Korpusweit angewandt verschwaende derselbe Satz. Ohne diese Zeile
    # koennte der Test oben gruen bleiben, obwohl er die Verdrahtungen gar nicht trennt.
    korpusweit = strip_boilerplate([document_text(seiten_von(i)) for i in range(10)])
    assert all(gemeinsam not in text for text in korpusweit), (
        "Die Gegenprobe greift nicht: Der Satz ueberlebt auch die korpusweite Anwendung. "
        "Dann prueft der Test oben nicht, was sein Name behauptet."
    )


def test_dokumenttext_ohne_layout_verliert_sonst_nichts() -> None:
    """Bindet die Inhaltsmenge: Ohne wiederkehrende Zeile darf gar nichts verschwinden."""
    seiten = [segment(i, f"Ganz eigener Inhalt auf Seite {i} ohne Wiederholung") for i in range(4)]
    ergebnis = document_text_ohne_layout(seiten)
    for i in range(4):
        assert f"Ganz eigener Inhalt auf Seite {i}" in ergebnis
