"""
Build the fictional Outlook emails in tests/emails/ that the import tests use, and their answer key
in tests/emails/manifest.json.

    python tools/make_email_fixtures.py

Every person and company is fictional (the same customers as the beta inbox, so the attached
drawings match the senders). The attachments are files from data/rfq_beta/files/: their text is in
the committed OCR cache, so importing these emails needs no live OCR. The output is the same on
every run (fixed dates, Message-IDs, MIME boundaries, and zip timestamps).

The .msg files are written by tools/msgwriter.py (there is no Outlook here); the .eml files by the
standard email package. Together they cover what real exports contain: plain text, HTML only with a
signature logo, 8-bit strings in an old code page, an Exchange sender, a body kept only as
compressed RTF, an email attached to an email, a forward-as-attachment bundle, and a zip of an
Outlook folder with the junk a Mac adds to it.

Entries in manifest.json that this tool did not make (scrubbed real emails, source "scrubbed") are
kept as they are.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import struct
import sys
import zipfile
import zlib
from email import policy
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import docgen  # noqa: E402
import msgwriter  # noqa: E402

OUT = ROOT / "tests" / "emails"
FILES = ROOT / "data" / "rfq_beta" / "files"
SHOP = "quotes@mesaridgeprecision.com"


def beta_file(rel: str) -> bytes:
    return (FILES / rel).read_bytes()


def when(day: int, hour: int, minute: int, offset_hours: int) -> dt.datetime:
    return dt.datetime(2026, 9, day, hour, minute, tzinfo=dt.timezone(dt.timedelta(hours=offset_hours)))


def tiny_png(width: int = 96, height: int = 28) -> bytes:
    """A small two-tone logo, the kind of signature image every Outlook email carries (stdlib only)."""
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            dark = (x // 8 + y // 7) % 2 == 0 and 4 < y < height - 4
            row += bytes((24, 64, 120) if dark else (230, 236, 244))
        rows.append(bytes(row))
    raw = zlib.compress(b"".join(rows), 9)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


def resume_pdf() -> bytes:
    """A one-page resume with a text layer: an attachment that is not in the OCR cache."""
    page = docgen.Page(*docgen.LETTER)
    y = 72.0
    for size, bold, text in [
        (18, True, "Jordan Vance"),
        (10, False, "CNC Machinist | Mesa, AZ | jvance.machinist@example.net"),
        (12, True, "Experience"),
        (10, False, "Setup machinist, 5 years: Haas VF-2 and VF-4, Okuma LB3000 lathes."),
        (10, False, "First article inspection, GD&T, CMM programming (Calypso)."),
        (12, True, "Certifications"),
        (10, False, "NIMS Machining Level I. OSHA 10."),
    ]:
        page.text(72, y, text, size, bold)
        y += size + 12
    return docgen.to_pdf([page], title="Resume", author="Jordan Vance")


# --------------------------------------------------------------------------- #
# The emails
# --------------------------------------------------------------------------- #
def email_spec(**kw: Any) -> Dict[str, Any]:
    spec = {"to": [SHOP], "cc": [], "attachments": [], "embedded": [], "html": None}
    spec.update(kw)
    return spec


PUMP_SHAFT = email_spec(
    subject="RFQ: pump shaft HPV-3310 Rev C, 150 and 300 pcs",
    from_name="Greg Lindqvist", from_email="glindqvist@halvorsenpv.com",
    date=when(21, 7, 42, -5), message_id="hpv-3310-rfq-0921@halvorsenpv.com",
    body=("Morning,\n\nPlease quote the attached pump shaft, HPV-3310 Rev C, at 150 and 300 pcs. 303 stainless, "
          "ground bearing journals, passivate per ASTM A967.\n\nWe need the quote back within two weeks.\n\n"
          "Greg Lindqvist\nPurchasing Manager, Halvorsen Pump & Valve\n"),
    attachments=[{"name": "HPV-3310_shaft_RevC.pdf", "content_type": "application/pdf",
                  "data": beta_file("E03/HPV-3310_shaft_RevC.pdf")}],
)

FIXTURES: List[Dict[str, Any]] = [
    {
        "file": "01_rfq_pump_shaft_plain.eml", "format": "eml", "email": PUMP_SHAFT,
        "note": "Plain text from new Outlook, one drawing with a text layer.",
        "expect": [{"lane": "turning", "is_rfq": True, "part_numbers": ["HPV-3310"]}],
    },
    {
        "file": "02_rfq_heatsink_html_only.eml", "format": "eml",
        "note": "HTML only with a signature logo (left out), a faxed drawing read with OCR, and a STEP model.",
        "email": email_spec(
            subject="RFQ: heat sink plate FR-2290, price breaks please",
            from_name="Lena Fischer", from_email="lfischer@ferroviarobotics.com",
            date=when(22, 9, 5, 1), message_id="fr-2290-0922@ferroviarobotics.com",
            body=None,
            html=("<html><body><p>Hello,</p><p>Attached is a scan of our heat sink plate <b>FR-2290</b> and its STEP model: "
                  "6063 aluminum, pocketed fins, clear anodize. Could you send pricing and lead time for these breaks?</p>"
                  "<table border=\"1\"><tr><th>Qty</th><th>Need by</th></tr><tr><td>25</td><td>November</td></tr>"
                  "<tr><td>100</td><td>January</td></tr></table>"
                  "<p>Thanks,<br>Lena Fischer<br>Hardware Engineer, Ferrovia Robotics</p>"
                  "<p><img src=\"cid:logo-ferrovia\" alt=\"Ferrovia Robotics\"></p></body></html>"),
            attachments=[
                {"name": "image001.png", "content_type": "image/png", "data": tiny_png(), "inline": True,
                 "content_id": "logo-ferrovia"},
                {"name": "FR-2290_heatsink.pdf", "content_type": "application/pdf", "data": beta_file("E07/FR-2290_heatsink.pdf")},
                {"name": "FR-2290_heatsink.step", "content_type": "application/step", "data": beta_file("E07/FR-2290_heatsink.step")},
            ]),
        "expect": [{"lane": "milling_3axis", "is_rfq": True, "part_numbers": ["FR-2290"],
                    "attachments": ["FR-2290_heatsink.pdf", "FR-2290_heatsink.step"]}],
    },
    {
        "file": "03_rfq_cover_plate_cui_fax.msg", "format": "msg", "msg": {"unicode": True, "body": "text"},
        "note": "Classic Outlook .msg. The CUI marking is only on the faxed drawing, so it takes OCR to route it.",
        "email": email_spec(
            subject="Quote request: 316L cover plate AGI-3052, 12 pcs",
            from_name="Ada Lindgren", from_email="alindgren@ashgroveinst.com",
            date=when(22, 14, 30, -7), message_id="agi-3052-0922@ashgroveinst.com",
            body=("Hi there,\n\nI'm looking for pricing on a small batch of cover plates for a sensor enclosure. "
                  "Drawing AGI-3052 Rev A is attached (faxed over from our other site).\n\n316L stainless, "
                  "passivate per ASTM A967. Need 12 pcs now, possibly 40 more in the spring.\n\nThanks!\n"
                  "Ada Lindgren\nMechanical Engineer\nAshgrove Instruments\n"),
            attachments=[{"name": "AGI-3052_RevA.pdf", "content_type": "application/pdf", "data": beta_file("E52/AGI-3052_RevA.pdf")}]),
        "expect": [{"lane": "itar", "is_rfq": True, "part_numbers": ["AGI-3052"]}],
    },
    {
        "file": "04_po_followup_exchange_ansi.msg", "format": "msg",
        "msg": {"unicode": False, "body": "text", "exchange_sender": True},
        "note": "Older .msg with 8-bit (cp1252) strings and an Exchange sender address.",
        "email": email_spec(
            subject="PO 45890: please confirm the ship date for QA-41127",
            from_name="Rachel Kim", from_email="rkim@quillonaero.com",
            date=when(23, 8, 10, -7), message_id="po-45890-0923@quillonaero.com",
            body=("Hi,\n\nWe’ve released PO 45890 for 60 pcs of the QA-41127 Rev B sensor bracket at the quoted price. "
                  "Can you confirm the ship date and send the order acknowledgment?\n\nThanks,\nRachel Kim\n"
                  "Senior Buyer, Quillon Aerospace\n")),
        "expect": [{"lane": "orders", "is_rfq": False}],
    },
    {
        "file": "05_vendor_promo_latin1.eml", "format": "eml", "charset": "iso-8859-1",
        "note": "A vendor newsletter in Latin-1, quoted-printable.",
        "email": email_spec(
            subject="Fall tooling sale: 20% off carbide end mills",
            from_name="Kyle Brennan", from_email="kbrennan@apexcuttingsupply.com",
            date=when(23, 6, 0, -6), message_id="promo-fall-0923@apexcuttingsupply.com",
            body=("Hello from Apex Cutting Supply!\n\nOur fall sale is on: 20% off solid carbide end mills and "
                  "15% off indexable inserts through October. Stop by our booth at the Fabtech expo, or reply "
                  "to set up a demo this week.\n\nSeñor Brennan says gracias, and à bientôt to our "
                  "Québec customers.\n\nTo unsubscribe, reply STOP.\n")),
        "expect": [{"lane": "filtered", "is_rfq": False}],
    },
    {
        "file": "06_forward_bundle_three_emails.eml", "format": "eml",
        "note": "How new Outlook and Outlook on the web export several emails: forwarded as attachments. The bundle "
                "itself is not imported, only the three emails in it.",
        "email": email_spec(
            subject="FW: three emails for the routing test",
            from_name="Tom Becker", from_email="tbecker@mesaridgeprecision.com",
            date=when(24, 7, 5, -7), message_id="bundle-0924@mesaridgeprecision.com",
            body="Forwarding three emails from the weekend.\n",
            embedded=[
                email_spec(subject="RFQ: brass contact pin NPE-0062, 2,500 pcs",
                           from_name="Hannah Ortiz", from_email="hortiz@northpeakenergy.com",
                           date=when(20, 10, 12, -6), message_id="npe-0062-0920@northpeakenergy.com",
                           body=("Hello,\n\nPlease quote a first release of 2,500 pcs of the attached brass contact pin "
                                 "NPE-0062, with an annual volume around 10,000 pcs. C360 brass, gold flash plated.\n\n"
                                 "Hannah Ortiz\nSourcing Specialist, NorthPeak Energy\n"),
                           attachments=[{"name": "NPE-0062_pin.pdf", "content_type": "application/pdf",
                                         "data": beta_file("E19/NPE-0062_pin.pdf")}]),
                email_spec(subject="PO 77120 status?",
                           from_name="Nora Castellanos", from_email="ncastellanos@kestrelfluid.com",
                           date=when(20, 13, 40, -5), message_id="po-77120-0920@kestrelfluid.com",
                           body=("Hi,\n\nCould you give me a status update on PO 77120 for the KF-3408 manifolds? "
                                 "Our line needs them by the 30th. Tracking number when they ship, please.\n\n"
                                 "Thank you,\nNora Castellanos\nBuyer, Kestrel Fluid Controls\n")),
                email_spec(subject="Experienced CNC machinist looking for work",
                           from_name="Jordan Vance", from_email="jvance.machinist@example.net",
                           date=when(21, 18, 2, -7), message_id="resume-0921@example.net",
                           body=("Hello,\n\nI'm a setup machinist with five years on Haas mills and Okuma lathes. "
                                 "My resume is attached. Are you hiring?\n\nJordan Vance\n"),
                           attachments=[{"name": "Jordan_Vance_resume.pdf", "content_type": "application/pdf",
                                         "data": resume_pdf()}]),
            ]),
        "expect": [
            {"lane": "turning", "is_rfq": True, "part_numbers": ["NPE-0062"]},
            {"lane": "orders", "is_rfq": False},
            {"lane": "filtered", "is_rfq": False},
        ],
    },
    {
        "file": "07_rfq_pivot_pin_rtf_only.msg", "format": "msg", "msg": {"unicode": True, "body": "rtf"},
        "note": "A .msg whose body exists only as compressed RTF, as older Outlook versions saved it.",
        "email": email_spec(
            subject="RFQ: pivot pin BWM-3105 Rev A, 200 pcs",
            from_name="Duncan Mireles", from_email="dmireles@brightwatermed.com",
            date=when(24, 9, 20, -5), message_id="bwm-3105-0924@brightwatermed.com",
            body=("Good morning,\n\nPlease quote 200 pcs of the Swiss-turned pivot pin BWM-3105 Rev A, 17-4 PH "
                  "stainless, condition H1025, passivated. Drawing attached.\n\nResponses are due October 15.\n\n"
                  "Best regards,\nDuncan Mireles\nProcurement Specialist, Brightwater Medical\n"),
            attachments=[{"name": "BWM-3105_RevA.pdf", "content_type": "application/pdf", "data": beta_file("E62/BWM-3105_RevA.pdf")}]),
        "expect": [{"lane": "turning", "is_rfq": True, "part_numbers": ["BWM-3105"]}],
    },
    {
        "file": "08_quantity_change_with_original.msg", "format": "msg", "msg": {"unicode": True, "body": "all"},
        "note": "A .msg that carries the customer's earlier email as an attached Outlook item: both are imported.",
        "email": email_spec(
            subject="RE: RFQ end plate KF-3412 Rev A, quantity change",
            from_name="Nora Castellanos", from_email="ncastellanos@kestrelfluid.com",
            date=when(24, 11, 0, -5), message_id="kf-3412-change-0924@kestrelfluid.com",
            body=("Hi,\n\nPlease change the quantity on the end plate KF-3412 Rev A to 75 pcs. My original request is "
                  "attached, and the drawing again for convenience.\n\nThank you,\nNora Castellanos\n"
                  "Buyer, Kestrel Fluid Controls\n"),
            html=("<html><body><p>Hi,</p><p>Please change the quantity on the end plate KF-3412 Rev A to <b>75 pcs</b>. "
                  "My original request is attached, and the drawing again for convenience.</p>"
                  "<p>Thank you,<br>Nora Castellanos<br>Buyer, Kestrel Fluid Controls</p></body></html>"),
            attachments=[{"name": "KF-3412_RevA.pdf", "content_type": "application/pdf", "data": beta_file("E32/KF-3412_RevA.pdf")}],
            embedded=[email_spec(subject="RFQ: end plate KF-3412 Rev A",
                                 from_name="Nora Castellanos", from_email="ncastellanos@kestrelfluid.com",
                                 date=when(17, 15, 45, -5), message_id="kf-3412-rfq-0917@kestrelfluid.com",
                                 body=("Hi Mesa Ridge team,\n\nPlease quote 50 pcs of the end plate KF-3412 Rev A, "
                                       "6061-T6511, clear anodize. Drawing to follow.\n\nThank you,\nNora Castellanos\n"))]),
        "expect": [
            {"lane": "milling_3axis", "is_rfq": True, "part_numbers": ["KF-3412"]},
            {"lane": "milling_3axis", "is_rfq": True, "part_numbers": ["KF-3412"]},
        ],
    },
    {
        "file": "09_capability_question.eml", "format": "eml",
        "note": "A vague question with no part named: someone has to look at it.",
        "email": email_spec(
            subject="Do you do sheet metal enclosures?",
            from_name="Brandon Lee", from_email="blee@lumenfieldsolar.com",
            date=when(24, 16, 25, -7), message_id="capability-0924@lumenfieldsolar.com",
            body=("Hey there,\n\nWe're looking for someone who can make a few sheet metal enclosures and maybe some "
                  "machined brackets down the road. Is that something you do? What would you need from us?\n\n"
                  "Brandon Lee\nLumenfield Solar\n")),
        "expect": [{"lane": "review", "is_rfq": False}],
    },
    {
        "file": "10_rfq_mirror_mount_screenshot.eml", "format": "eml",
        "note": "A screenshot of a drawing instead of the PDF, read with OCR.",
        "email": email_spec(
            subject="RFQ: fold mirror mount LO-1186 Rev A, qty 8",
            from_name="Felix Durand", from_email="fdurand@lucerneopto.com",
            date=when(25, 8, 50, 2), message_id="lo-1186-0925@lucerneopto.com",
            body=("Hello Mesa Ridge,\n\nCan you quote 8 pcs of the fold mirror mount LO-1186 Rev A? The PDF is stuck in "
                  "our PDM release queue, so here is a screenshot of the drawing. The mirror seat is a compound-angle "
                  "face, so I'd expect a 5-axis setup.\n\nThanks,\nFelix Durand\nLucerne Optics\n"),
            attachments=[{"name": "LO-1186_RevA_screenshot.png", "content_type": "image/png",
                          "data": beta_file("E71/LO-1186_RevA_screenshot.png")}]),
        "expect": [{"lane": "milling_5axis", "is_rfq": True, "part_numbers": ["LO-1186"]}],
    },
]

ZIP_MEMBERS = [
    {"name": "Inbox/RE quote WS-4471 Rev B.eml", "format": "eml",
     "email": email_spec(
         subject="RE: RFQ WS-26-0388, output shaft WS-4471 Rev B",
         from_name="Trent Haverford", from_email="thaverford@wexcombesys.com",
         date=when(25, 10, 5, -6), message_id="ws-4471-re-0925@wexcombesys.com",
         body=("Hello,\n\nFollowing up on the output shaft WS-4471 Rev B. This part is ITAR controlled: only U.S. "
               "persons may see the drawing, so please confirm your registration before you quote 40 pcs.\n\n"
               "Best regards,\nTrent Haverford\nBuyer | Wexcombe Systems\n"),
         attachments=[{"name": "WS-4471_RevB.pdf", "content_type": "application/pdf", "data": beta_file("E72/WS-4471_RevB.pdf")}]),
     "expect": {"lane": "itar", "is_rfq": True, "part_numbers": ["WS-4471"]}},
    {"name": "Inbox/RFQ lens cell TO-5520.msg", "format": "msg", "msg": {"unicode": True, "body": "html"},
     "email": email_spec(
         subject="Request for quote: lens cell TO-5520, 40 pcs",
         from_name="Owen Marsh", from_email="omarsh@tessaractoptics.com",
         date=when(25, 11, 30, -8), message_id="to-5520-0925@tessaractoptics.com",
         body=None,
         html=("<html><body><p>Hi,</p><p>Please quote 40 pcs of the attached lens cell TO-5520, 6061-T6 with black "
               "anodize. The drawing is marked CUI and the program flows down DFARS 252.204-7012.</p>"
               "<p>Owen Marsh<br>Supply Chain, Tessaract Optics</p></body></html>"),
         attachments=[{"name": "TO-5520_lens_cell_CUI.pdf", "content_type": "application/pdf",
                       "data": beta_file("E05/TO-5520_lens_cell_CUI.pdf")}]),
     "expect": {"lane": "itar", "is_rfq": True, "part_numbers": ["TO-5520"]}},
    {"name": "Inbox/copy of pump shaft RFQ.eml", "format": "eml", "email": PUMP_SHAFT,
     "expect": {"lane": "turning", "is_rfq": True, "part_numbers": ["HPV-3310"]}},
]
ZIP_JUNK = {
    "__MACOSX/Inbox/._RFQ lens cell TO-5520.msg": b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        ",
    "Inbox/notes.txt": b"Exported from Outlook for the routing test.\n",
}


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def build_eml(spec: Dict[str, Any], charset: Optional[str] = None) -> EmailMessage:
    msg = EmailMessage(policy=policy.SMTP)
    msg["From"] = f"{spec['from_name']} <{spec['from_email']}>"
    msg["To"] = ", ".join(spec["to"])
    if spec["cc"]:
        msg["Cc"] = ", ".join(spec["cc"])
    msg["Subject"] = spec["subject"]
    msg["Date"] = format_datetime(spec["date"])
    msg["Message-ID"] = f"<{spec['message_id']}>"
    msg["X-Mailer"] = "Microsoft Outlook 16.0"
    inline = [a for a in spec["attachments"] if a.get("inline")]
    files = [a for a in spec["attachments"] if not a.get("inline")]
    if spec["body"] is None:
        msg.set_content(spec["html"], subtype="html", charset="utf-8")
        html_part = msg
    else:
        if charset:
            msg.set_content(spec["body"], charset=charset, cte="quoted-printable")
        else:
            msg.set_content(spec["body"])
        html_part = None
        if spec["html"]:
            msg.add_alternative(spec["html"], subtype="html")
            html_part = msg.get_payload()[1]
    for att in inline:
        maintype, subtype = att["content_type"].split("/", 1)
        (html_part or msg).add_related(att["data"], maintype=maintype, subtype=subtype, cid=f"<{att['content_id']}>",
                                       filename=att["name"], disposition="inline")
    for att in files:
        maintype, subtype = att["content_type"].split("/", 1)
        msg.add_attachment(att["data"], maintype=maintype, subtype=subtype, filename=att["name"])
    for inner in spec["embedded"]:
        part = build_eml(inner)
        msg.add_attachment(part, filename=f"{inner['subject'][:60]}.eml")
    return msg


def eml_bytes(spec: Dict[str, Any], charset: Optional[str] = None) -> bytes:
    msg = build_eml(spec, charset)
    # Fixed MIME boundaries: the generated files are the same on every run.
    for n, part in enumerate(p for p in msg.walk() if p.get_content_maintype() == "multipart"):
        part.set_boundary(f"=_mesa_fixture_{n}")
    return msg.as_bytes(policy=policy.SMTP)


def msg_dict(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "subject": spec["subject"], "from_name": spec["from_name"], "from_email": spec["from_email"],
        "to": spec["to"], "cc": spec["cc"], "date": spec["date"].isoformat(), "message_id": spec["message_id"],
        "body": spec["body"] or "", "html": spec["html"],
        "attachments": [dict(a) for a in spec["attachments"]],
        "embedded": [msg_dict(e) for e in spec["embedded"]],
    }


def msg_bytes(spec: Dict[str, Any], options: Dict[str, Any]) -> bytes:
    return msgwriter.build_msg(msg_dict(spec), unicode=options.get("unicode", True), body=options.get("body", "text"),
                               exchange_sender=options.get("exchange_sender", False))


def file_bytes(item: Dict[str, Any]) -> bytes:
    if item["format"] == "msg":
        return msg_bytes(item["email"], item.get("msg") or {})
    return eml_bytes(item["email"], item.get("charset"))


def expected(spec: Dict[str, Any], expect: Dict[str, Any]) -> Dict[str, Any]:
    names = expect.get("attachments") or [a["name"] for a in spec["attachments"] if not a.get("inline")]
    out = {"subject": spec["subject"], "from_email": spec["from_email"], "attachments": names,
           "lane": expect["lane"], "is_rfq": expect["is_rfq"]}
    if expect.get("part_numbers"):
        out["part_numbers"] = expect["part_numbers"]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    kept: List[Dict[str, Any]] = []
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        kept = [f for f in old.get("files", []) if f.get("source") != "generated"]
    entries: List[Dict[str, Any]] = []
    for fx in FIXTURES:
        (OUT / fx["file"]).write_bytes(file_bytes(fx))
        spec = fx["email"]
        if spec["embedded"] and len(fx["expect"]) == len(spec["embedded"]):
            emails = [expected(inner, e) for inner, e in zip(spec["embedded"], fx["expect"])]  # a bundle
        elif spec["embedded"]:
            emails = [expected(spec, fx["expect"][0])] + [expected(inner, e) for inner, e in
                                                          zip(spec["embedded"], fx["expect"][1:])]
        else:
            emails = [expected(spec, fx["expect"][0])]
        entries.append({"file": fx["file"], "source": "generated", "format": fx["format"], "note": fx["note"],
                        "emails": emails})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        members = [(m["name"], file_bytes(m)) for m in ZIP_MEMBERS] + list(ZIP_JUNK.items())
        for name, data in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 25, 12, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    (OUT / "11_outlook_folder_export.zip").write_bytes(buf.getvalue())
    entries.append({
        "file": "11_outlook_folder_export.zip", "source": "generated", "format": "zip",
        "note": "A zipped Outlook folder: a .msg, two .eml files (one a copy of 01, so a second import reports it "
                "as already in the inbox), the junk a Mac adds, and a text file that is left out.",
        "emails": [expected(m["email"], m["expect"]) for m in sorted(ZIP_MEMBERS, key=lambda m: m["name"])],
        "skipped": ["Inbox/notes.txt"],
    })
    manifest = {
        "about": ("Answer key for the emails in this folder. tests/test_import.py parses every file listed here and "
                  "checks the emails in it, then imports them into a running server. Generated fixtures come from "
                  "tools/make_email_fixtures.py; scrubbed real emails come from tools/scrub_email.py (review them "
                  "before committing). lane is where the email belongs; is_rfq says whether it asks for a quote."),
        "files": entries + kept,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    total = sum(len(f["emails"]) for f in manifest["files"])
    print(f"{len(entries)} generated files, {total} emails in the manifest -> {OUT}")


if __name__ == "__main__":
    main()
