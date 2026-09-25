"""E-Mail: Kopf und Körper werden Segmente, jeder Anhang ein eigenes Dokument.

Eine Mail mit angehängter Rechnung besteht aus zwei Dokumenten mit verschiedenen Klassen
(Konzept § 5). Die Mail wird nie durch ihren Anhang klassifiziert und umgekehrt. Der
Anhang wird deshalb hier **nicht** in den Mailtext gemischt, sondern herausgereicht;
``pipeline.py`` macht daraus ein Dokument mit ``parent_document_id``.

Betreff und Absenderdomain stehen im Kopfsegment, damit sie später Strukturmerkmale werden
können – Mails von ``rechnung@…`` tragen überdurchschnittlich oft Rechnungen.

Entscheidungen zu Randfällen, die der Plan offenlässt:

* ``email.policy.default`` (statt ``compat32``) dekodiert MIME-kodierte Kopfzeilen
  (``=?utf-8?B?…?=``) selbständig beim Zugriff als String – ohne diese Policy blieben
  Umlaute in Betreff und Namen kodiert im Kopfsegment stehen.
* Bei ``multipart/alternative`` liefert ``get_body`` bevorzugt die Text-Variante, sonst
  HTML. Reiner Text ist das treuere Abbild des Inhalts; HTML-Markup wäre zusätzliches
  Rauschen für die spätere Merkmalsbildung.
* Eine Mail ganz ohne Rumpf (kein ``set_content``) liefert bei ``get_body`` die Nachricht
  selbst mit leerem Inhalt zurück – ``normalize_text`` macht daraus ``""``, das
  Körper-Segment entfällt dann einfach; das Kopf-Segment bleibt immer.
* Anhänge ohne eigenen Dateinamen (kein ``Content-Disposition``-Parameter ``filename``)
  bekommen einen fortlaufend nummerierten Ersatznamen (``anhang-1.bin``, ``anhang-2.bin``,
  …) nach ihrer Position in der Mail. Ein fester Name wie ``anhang.bin`` für alle
  namenlosen Anhänge einer Mail würde sie in ``source_path`` ununterscheidbar machen.
* Ein Anhang, dessen Inhalt sich nicht dekodieren lässt oder leer ist, wird verworfen –
  ein Dokument ohne Bytes ließe sich ohnehin nicht klassifizieren.
"""

from dataclasses import dataclass
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser

from doccls.models import Document, Segment, SegmentKind, normalize_text

HEADER_FIELDS = ("From", "To", "Cc", "Subject", "Date")
"""Kopfzeilen, die ins Kopfsegment übernommen werden – die für Absender/Betreff-Merkmale
relevanten, nicht der volle, oft lange Zustellweg (``Received:`` u. Ä.)."""


@dataclass(frozen=True)
class Attachment:
    """Ein Anhang, bereit als eigenständiges Dokument eingelesen zu werden."""

    file_name: str
    content: bytes


def extract_eml(document: Document, data: bytes) -> tuple[list[Segment], list[Attachment]]:
    """Kopf und Körper als Segmente, Anhänge getrennt herausgereicht.

    Kein Extraktionsschritt darf einen Anhang in den Mailtext mischen – siehe Moduldoc.
    """
    # ``policy.default`` legt den Nachrichtentyp bereits statisch auf ``EmailMessage``
    # fest (mypy leitet das aus der Policy her) – eine Typzusicherung ist hier nicht
    # nötig, im Unterschied zu ``teil.iter_attachments()`` unten, wo die Basisklasse
    # ``Message`` zurückkommt.
    nachricht = BytesParser(policy=policy.default).parsebytes(data)

    segmente: list[Segment] = []
    kopf = normalize_text(
        " ".join(f"{feld}: {nachricht[feld]}" for feld in HEADER_FIELDS if nachricht[feld])
    )
    if kopf:
        segmente.append(
            Segment.create(
                document=document,
                index=0,
                kind=SegmentKind.MAIL_KOPF,
                locator="Kopf",
                heading=normalize_text(str(nachricht["Subject"] or "")) or None,
                text=kopf,
            )
        )

    koerper = nachricht.get_body(preferencelist=("plain", "html"))
    if koerper is not None:
        text = normalize_text(koerper.get_content())
        if text:
            segmente.append(
                Segment.create(
                    document=document,
                    index=len(segmente),
                    kind=SegmentKind.MAIL_KOERPER,
                    locator="Körper",
                    heading=None,
                    text=text,
                )
            )

    anhaenge: list[Attachment] = []
    for nummer, teil in enumerate(nachricht.iter_attachments(), start=1):
        if not isinstance(teil, EmailMessage):
            continue
        inhalt = teil.get_payload(decode=True)
        if not isinstance(inhalt, bytes) or not inhalt:
            continue  # Nicht dekodierbar oder leer – kein Dokument ohne Bytes.
        dateiname = teil.get_filename() or f"anhang-{nummer}.bin"
        anhaenge.append(Attachment(file_name=dateiname, content=inhalt))

    return segmente, anhaenge
