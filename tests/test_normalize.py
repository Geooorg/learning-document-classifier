"""Wiederkehrende Zeilen sind Layout, nicht Inhalt."""

from doccls.normalize import strip_boilerplate


def test_zeile_auf_allen_seiten_wird_entfernt() -> None:
    seiten = [f"Seite {i} Inhalt hier. Muster GmbH · HRB 44821" for i in range(1, 6)]
    bereinigt = strip_boilerplate(seiten)
    assert all("HRB 44821" not in seite for seite in bereinigt)
    assert all("Inhalt hier" in seite for seite in bereinigt)


def test_satz_auf_der_minderheit_der_seiten_bleibt_stehen() -> None:
    """Zwei von fünf Seiten liegen unter der Schwelle von 0,6 – das ist Inhalt, kein Layout."""
    wiederholt = "Dieser Hinweis steht nur auf zwei Seiten"
    seiten = [
        f"{wiederholt}. Eigener Inhalt Nummer {i} an dieser Stelle"
        if i < 2
        else f"Eigener Inhalt Nummer {i} an dieser Stelle"
        for i in range(5)
    ]
    bereinigt = strip_boilerplate(seiten)
    assert wiederholt in " ".join(bereinigt)


def test_einzelne_seite_wird_nicht_leergeraeumt() -> None:
    """Bei einer Seite ist jede Zeile auf 100 % der Seiten – ohne Schutz bliebe nichts übrig."""
    assert strip_boilerplate(["Nur eine Seite mit Text."]) == ["Nur eine Seite mit Text."]


def test_leere_eingabe_bleibt_leer() -> None:
    assert strip_boilerplate([]) == []


def test_nur_die_wiederkehrende_zeile_verschwindet() -> None:
    """Bindet die Inhalts*menge*, nicht nur die Anwesenheit einer Zeichenkette.

    Ohne diese Prüfung überlebt eine Umsetzung, die von jeder Seite nur den ersten Satz
    zurückgibt: „HRB 44821" wäre weg und „Inhalt hier" wäre da — beide anderen Tests
    blieben grün, obwohl der halbe Text fehlt.
    """
    seiten = [
        f"Seite {i} Inhalt hier. Zweiter Satz auf Seite {i}. Dritter Satz auf Seite {i}. "
        "Muster GmbH · HRB 44821"
        for i in range(1, 6)
    ]
    bereinigt = strip_boilerplate(seiten)
    for vorher, nachher in zip(seiten, bereinigt, strict=True):
        ohne_fusszeile = vorher.replace("Muster GmbH · HRB 44821", "")
        verloren = len(ohne_fusszeile.strip()) - len(nachher.strip())
        assert verloren <= 2, (
            f"Entfernt wurde mehr als die Fußzeile: {verloren} Zeichen zu viel.\n"
            f"vorher:  {vorher!r}\nnachher: {nachher!r}"
        )
        assert "Dritter Satz" in nachher
