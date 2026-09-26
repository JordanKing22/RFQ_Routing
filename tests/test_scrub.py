"""
Tests for tools/scrub_email.py, the privacy scrub that turns real emails into fictional test
fixtures.

    python -m unittest tests.test_scrub -v

Every "real" person, company, street, and phone number below is invented for these tests (phone
numbers use 555 exchanges and the UK and Australian ranges set aside for fiction). The emails are
built to look like real Outlook exports: signature blocks with titles, office and cell phones,
addresses, disclaimers, quoted reply chains with other people, HTML-only bodies, .msg files from
tools/msgwriter.py, and zips. Each end-to-end test scrubs them with the command line and checks that
none of the invented identifying strings survive anywhere in the output: the raw bytes, the
decoded headers (encoded words included), the decoded body, what mailfile.load reads back, and the
manifest.
"""

from __future__ import annotations

import io
import json
import re
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import mailfile  # noqa: E402
from tools import scrub_email as scrub  # noqa: E402
from tools.msgwriter import build_msg  # noqa: E402


# ---------------------------------------------------------------------------------------------
# Invented "real" emails

def make_eml(headers: Sequence[tuple], text: Optional[str] = None, html: Optional[str] = None,
             attachments: Sequence[tuple] = ()) -> bytes:
    msg = EmailMessage()
    for k, v in headers:
        msg[k] = v
    if text is not None:
        msg.set_content(text)
        if html:
            msg.add_alternative(html, subtype="html")
    elif html is not None:
        msg.set_content(html, subtype="html")
    for name, ctype, data in attachments:
        maintype, _, subtype = ctype.partition("/")
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)
    return msg.as_bytes()


# A: classic Outlook, plain text, a reply chain with the shop's own people in it.
A_BODY = """\
Hi Tobiah,

Following up on the manifold block below. Ignatius Whitlock in our Grand Rapids plant will own the
PO. Marisol's out next week, so please reach Ignatius at 269-555-7314 or cell 269.555.7315.

Please quote P/N BFS-30418 Rev D, qty 40/80/150, 6061-T6, clear anodize per MIL-A-8625 Type II.
Ship to: Brackwater Fluid Systems, 2750 Ottleby Industrial Pkwy, Suite 340, Grand Rapids, MI 49512

Best regards,

Marisol Quintero-Vance
Purchasing Manager | Brackwater Fluid Systems, LLC
2750 Ottleby Industrial Pkwy, Suite 340
Grand Rapids, MI 49512
Office: (616) 555-3390 ext. 117
Cell: 616.555.2284 | Fax (616) 555-3391
mquintero@brackwaterfluid.com | https://www.brackwaterfluid.com/rfq?src=sig&rep=mquintero
<mailto:mquintero@brackwaterfluid.com>

This message and any attachments are confidential and intended only for the named recipient.
Brackwater Fluid Systems, LLC | 2750 Ottleby Industrial Pkwy | Grand Rapids, Michigan 49512

________________________________
From: Tobiah Runcorn <truncorn@ferncastmachine.com>
Sent: Monday, September 21, 2026 4:02 PM
To: Quintero-Vance, Marisol <mquintero@brackwaterfluid.com>
Cc: Whitlock, Ignatius <iwhitlock@brackwaterfluid.com>; Oriel Mbatha <ombatha@ferncastmachine.com>
Subject: RE: RFQ BW-2291

Marisol,

Thanks, we will look at it this week. Oriel Mbatha will run the numbers.
Call me direct at +1 (231) 555-6620 x4.

Tobiah Runcorn
Estimating Lead, Ferncast Machine Co.
T: 231-555-6600
"""

A_HEADERS = [
    ("From", '"Quintero-Vance, Marisol" <mquintero@brackwaterfluid.com>'),
    ("To", "RFQ Desk <rfq@ferncastmachine.com>"),
    ("Cc", "Ignatius Whitlock <iwhitlock@brackwaterfluid.com>"),
    ("Subject", "RE: RFQ BW-2291: Brackwater manifold block, P/N BFS-30418 Rev D"),
    ("Date", "Tue, 22 Sep 2026 09:14:00 -0400"),
    ("Message-ID", "<BN8PR11MB3764A1B2C3@BN8PR11MB3764.namprd11.prod.outlook.com>"),
    ("In-Reply-To", "<CAF7x@ferncastmachine.com>"),
    ("Thread-Topic", "RFQ BW-2291 Brackwater manifold"),
    ("Received", "from mail.brackwaterfluid.com (mail.brackwaterfluid.com [10.20.30.40])"),
    ("X-Originating-IP", "[10.20.30.40]"),
]
A_ATTACHMENTS = [
    ("BFS-30418_RevD.pdf", "application/pdf", b"%PDF-1.4\n% drawing\n"),
    ("Brackwater RFQ BW-2291 - Marisol.xlsx", "application/vnd.ms-excel", b"PK\x03\x04sheet"),
]
A_SECRETS = ["Marisol", "Quintero", "Vance", "Ignatius", "Whitlock", "Tobiah", "Runcorn", "Oriel",
             "Mbatha", "Brackwater", "brackwaterfluid", "Ferncast", "ferncastmachine", "Ottleby",
             "Grand Rapids", "49512", "Suite 340", "BN8PR11MB3764", "10.20.30.40", "mquintero",
             "iwhitlock", "truncorn", "ombatha", "rep=", "ext. 117"]
A_DIGITS = ["5553390", "5552284", "5553391", "5557314", "5557315", "5556620", "5556600"]


def email_a() -> bytes:
    return make_eml(A_HEADERS, text=A_BODY, attachments=A_ATTACHMENTS)


# B: new Outlook / Outlook on the web, HTML only, signature in a table, a safe-links URL.
B_HTML = """\
<html><head><meta charset="utf-8"><style>p.MsoNormal{margin:0}</style></head><body>
<p>Good morning Tobiah,</p>
<p>Attached is our RFQ for the valve bonnet, part QVW-1187-02, 316 stainless, qty 12 and 30.
Please send pricing to me and cc <a href="mailto:afairleigh@quellmoorvalve.com">Anatole Fairleigh</a>.</p>
<p>The model is also on our portal:
<a href="https://nam12.safelinks.protection.outlook.com/?url=https%3A%2F%2Fportal.quellmoorvalve.com%2Frfq%2F7781&amp;data=05%7C02%7Cpashworth-lund%40quellmoorvalve.com">https://portal.quellmoorvalve.com/rfq/7781</a></p>
<p>Thank you,</p>
<table><tr><td><b>Priscilla</b></td><td><b>Ashworth-Lund</b></td></tr>
<tr><td colspan="2">Sr. Buyer, Quellmoor Valve Works Inc.<br>918 Brindlecombe Ave NW<br>Canton, OH 44708<br>
Direct: +1 330-555-4471 | Mobile: 330 555 4472<br>
<a href="mailto:pashworth-lund@quellmoorvalve.com">pashworth-lund@quellmoorvalve.com</a> |
<a href="http://www.quellmoorvalve.com/">www.quellmoorvalve.com</a></td></tr></table>
<p style="font-size:8pt;color:gray">CONFIDENTIALITY NOTICE: This e-mail is the property of Quellmoor
Valve Works Inc. and may contain confidential information.</p>
</body></html>
"""
B_HEADERS = [
    ("From", "Priscilla Ashworth-Lund <pashworth-lund@quellmoorvalve.com>"),
    ("To", '"Ferncast Quotes" <quotes@ferncastmachine.com>'),
    ("Subject", "Quellmoor RFQ 7781: valve bonnet QVW-1187-02"),
    ("Date", "Wed, 23 Sep 2026 07:41:12 -0500"),
    ("Message-ID", "<PH0PR19MB5E0@PH0PR19MB5E0.namprd19.prod.outlook.com>"),
]
B_SECRETS = ["Priscilla", "Ashworth", "Lund", "Quellmoor", "quellmoorvalve", "Anatole", "Fairleigh",
             "afairleigh", "pashworth", "Brindlecombe", "Canton", "44708", "PH0PR19MB5E0", "Tobiah",
             "Ferncast", "ferncastmachine", "rfq/7781"]
B_DIGITS = ["5554471", "5554472"]


def email_b() -> bytes:
    return make_eml(B_HEADERS, html=B_HTML)


# C: classic Outlook .msg, an Exchange sender, international phones, a signature split over lines.
C_BODY = """\
Hello Ferncast team,

Please quote the gripper finger HX-44120 Rev A (drawing and STEP attached), 17-4PH H1150,
qty 8 prototypes then 200/yr. Thessaly Oduya will send the updated model Friday.

Our UK office handles export paperwork: +44 20 7946 0321. For urgent questions call our
engineer in Australia on +61 491 570 158.

Regards,
Evander
Szczepanski
Director of Supply Chain
Cordwainer Robotics Inc. | 61 Pellingham Rd | Worcester, MA 01605
Tel +1 508 555 7720 | evander.szczepanski@cordwainerrobotics.com
"""
C_SECRETS = ["Evander", "Szczepanski", "Thessaly", "Oduya", "Cordwainer", "cordwainerrobotics",
             "Pellingham", "Worcester", "01605", "toduya", "0a1b2c3d4e", "Ferncast"]
C_DIGITS = ["79460321", "491570158", "5557720"]


def email_c_dict() -> Dict:
    return {
        "subject": "RFQ: Cordwainer gripper finger HX-44120 Rev A",
        "from_name": "Szczepanski, Evander",
        "from_email": "evander.szczepanski@cordwainerrobotics.com",
        "to": ["Ferncast Quotes <quotes@ferncastmachine.com>"],
        "cc": ["Thessaly Oduya <toduya@cordwainerrobotics.com>"],
        "date": "2026-09-24T08:15:00-07:00",
        "message_id": "0a1b2c3d4e@cordwainerrobotics.com",
        "body": C_BODY,
        "attachments": [
            {"name": "Cordwainer_HX-44120_RevA.pdf", "content_type": "application/pdf",
             "data": b"%PDF-1.4\n% finger\n", "inline": False},
            {"name": "HX-44120_RevA.step", "content_type": "application/step",
             "data": b"ISO-10303-21;\n", "inline": False},
        ],
    }


def email_c(**kw) -> bytes:
    return build_msg(email_c_dict(), **kw)


# E: Mac Mail reply from a hobbyist on a public mail service, quoted with ">".
E_BODY = """\
Dear Mr. Runcorn,

I restore vintage motorcycles and need 6 custom axle spacers turned from 303 stainless,
sketch attached. Daniel Throckmorton at Westbury Cycle Works LLC said Ferncast did good work for him.

You can reach me at (412) 555-8830 or wendy.achterkirk1987@gmail.com. I'm at
88 Larkspur Ln, Apt 4B, Sewickley, PA 15143.

Thanks!
Wendy

On Sep 21, 2026, at 4:02 PM, Tobiah Runcorn <truncorn@ferncastmachine.com> wrote:

> Hi Wendeline,
> Happy to look at it. Send a sketch with dimensions.
> Tobiah
"""
E_HEADERS = [
    ("From", "Wendeline Achterkirk <wendy.achterkirk1987@gmail.com>"),
    ("To", "Tobiah Runcorn <truncorn@ferncastmachine.com>"),
    ("Subject", "Re: Axle spacers for a 1972 restoration"),
    ("Date", "Fri, 25 Sep 2026 19:02:44 -0400"),
    ("Message-ID", "<6C1D2E3F-AAAA-4BBB-8CCC-1234567890AB@gmail.com>"),
]
E_SECRETS = ["Wendeline", "Wendy", "Achterkirk", "Daniel", "Throckmorton", "Westbury", "Larkspur",
             "Sewickley", "15143", "Tobiah", "Runcorn", "Ferncast", "ferncastmachine", "6C1D2E3F",
             "Apt 4B"]
E_DIGITS = ["5558830"]


def email_e() -> bytes:
    return make_eml(E_HEADERS, text=E_BODY, attachments=[("sketch.jpg", "image/jpeg", b"\xff\xd8\xff" + b"0" * 40000)])


# ---------------------------------------------------------------------------------------------
# Helpers

class ScrubTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.inbox = self.dir / "private_emails"
        self.inbox.mkdir()
        self.out = self.dir / "emails"
        self.map = self.inbox / ".scrub_map.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def put(self, name: str, data: bytes) -> Path:
        path = self.inbox / name
        path.write_bytes(data)
        return path

    def scrub(self, *args: str, out: Optional[Path] = None, map_file: Optional[Path] = None,
              expect: int = 0) -> str:
        buf, err = io.StringIO(), io.StringIO()
        argv = list(args) + ["--out", str(out or self.out), "--map-file", str(map_file or self.map)]
        with redirect_stderr(err):
            code = scrub.run(argv, stdout=buf)
        self.last_stderr = err.getvalue()
        self.assertEqual(code, expect, buf.getvalue() + err.getvalue())
        return buf.getvalue()

    def outputs(self, out: Optional[Path] = None) -> List[Path]:
        return sorted((out or self.out).glob("*.eml"))

    def one_output(self, out: Optional[Path] = None) -> bytes:
        files = self.outputs(out)
        self.assertEqual(len(files), 1, files)
        return files[0].read_bytes()

    def views(self, data: bytes) -> str:
        """Every way the file can be read: raw bytes, parsed headers (encoded words decoded),
        the decoded body, and what mailfile.load returns."""
        parts = [data.decode("utf-8", "replace")]
        msg = BytesParser(policy=policy.default).parsebytes(data)
        for k, v in msg.items():
            parts.append(f"{k}: {v}")
        for part in msg.walk():
            if part.get_content_maintype() == "text":
                parts.append(part.get_content())
            if part.get_filename():
                parts.append(part.get_filename())
        res = mailfile.load("check.eml", data)
        for em in res["emails"]:
            parts += [em.get("subject") or "", em.get("from_name") or "", em.get("from_email") or "",
                      em.get("body") or "", em.get("message_id") or ""]
            parts += list(em.get("to") or []) + list(em.get("cc") or [])
            parts += [a.get("name") or "" for a in em.get("attachments") or []]
        return "\n".join(parts)

    def assert_clean(self, data: bytes, secrets: Sequence[str], digits: Sequence[str] = (),
                     extra: str = "") -> None:
        text = self.views(data) + "\n" + extra
        low = text.lower()
        for s in secrets:
            i = low.find(s.lower())
            self.assertEqual(i, -1, f"{s!r} survived: ...{text[max(0, i - 60):i + 60]!r}...")
        for line in text.splitlines():
            only = re.sub(r"\D", "", line)
            for d in digits:
                self.assertNotIn(d, only, f"digits {d} survived in {line!r}")

    def manifest(self, out: Optional[Path] = None) -> Dict:
        return json.loads(((out or self.out) / "manifest.json").read_text(encoding="utf-8"))


def fresh(**kw) -> "scrub.Scrubber":
    return scrub.Scrubber(scrub.ScrubMap(None), **kw)


def scrub_one_text(text: str, headers: Optional[Dict] = None, **kw):
    """Scrub a body the way the tool does: collect first, then replace."""
    s = fresh(**kw)
    em = {"body": text, "subject": "", "to": [], "cc": [], "attachments": []}
    em.update(headers or {})
    s.collect(em)
    s.finish_collect()
    return s.scrub_text(text), s


# ---------------------------------------------------------------------------------------------
# End to end: no invented identifying string survives

class NoLeakTests(ScrubTestCase):
    def test_outlook_text_reply_chain(self):
        self.put("RFQ BW-2291.eml", email_a())
        report = self.scrub(str(self.inbox / "RFQ BW-2291.eml"), "--lane", "milling_3axis")
        data = self.one_output()
        self.assert_clean(data, A_SECRETS, A_DIGITS, extra=json.dumps(self.manifest()))
        text = self.views(data)
        # What the email is about survives.
        for keep in ("BFS-30418", "6061-T6", "MIL-A-8625", "qty 40/80/150", "Purchasing Manager",
                     "Fluid Systems, LLC", "Machine Co.", "RFQ BW-2291"):
            self.assertIn(keep, text)
        # Fakes: fictional 555-01xx phones and extensions, a fake street and suite.
        self.assertRegex(text, r"Office: \(\d{3}\) 555-01\d\d ext\. \d{3}")
        self.assertRegex(text, r"Cell: \d{3}\.555\.01\d\d")
        self.assertRegex(text, r"\d+ \w+ Industrial Pkwy, Suite \d{3}|\d+ \w+ Pkwy, Suite \d{3}")
        self.assertIn("Marisol", report)          # the report shows real -> fake for review
        self.assertIn("Check these:", report)

    def test_html_only_new_outlook(self):
        self.put("bonnet.eml", email_b())
        self.scrub(str(self.inbox / "bonnet.eml"), "--lane", "turning")
        data = self.one_output()
        self.assert_clean(data, B_SECRETS, B_DIGITS, extra=json.dumps(self.manifest()))
        text = self.views(data)
        self.assertIn("QVW-1187-02", text)
        self.assertIn("Valve Works Inc.", text)
        self.assertNotIn("safelinks", text)       # the wrapper URL came out of the HTML

    def test_msg_exchange_sender_all_body_kinds(self):
        for body in ("text", "html", "rtf"):
            with self.subTest(body=body):
                em = email_c_dict()
                if body != "text":
                    em["html"] = "<html><body>" + "".join(
                        f"<p>{line}</p>" if line else "<br>" for line in C_BODY.split("\n")) + "</body></html>"
                data = build_msg(em, body=body, exchange_sender=True)
                name = f"gripper_{body}.msg"
                self.put(name, data)
                out = self.dir / f"out_{body}"
                self.scrub(str(self.inbox / name), "--lane", "milling_5axis", out=out)
                result = self.one_output(out)
                self.assert_clean(result, C_SECRETS, C_DIGITS, extra=json.dumps(self.manifest(out)))
                text = self.views(result)
                self.assertIn("HX-44120", text)
                self.assertIn("+44 20 7946 0", text)        # country code kept, number fake
                self.assertNotIn("7946 0321", text)

    def test_mac_reply_public_mail_split_signature(self):
        self.put("spacers.eml", email_e())
        self.scrub(str(self.inbox / "spacers.eml"), "--lane", "turning")
        data = self.one_output()
        self.assert_clean(data, E_SECRETS, E_DIGITS, extra=json.dumps(self.manifest()))
        back = mailfile.load("x.eml", data)["emails"][0]
        # A public mail service stays public; the mailbox name is replaced.
        self.assertTrue(back["from_email"].endswith("@gmail.com"), back["from_email"])
        self.assertNotIn("wendy", back["from_email"])
        self.assertIn("wrote:", back["body"])
        self.assertIn(" Works LLC", back["body"])

    def test_zip_of_eml_and_msg(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("Inbox/RFQ BW-2291.eml", email_a())
            zf.writestr("Inbox/gripper.msg", email_c())
            zf.writestr("Inbox/notes.txt", "not an email")
            zf.writestr("__MACOSX/Inbox/._gripper.msg", b"junk")
        self.put("export.zip", buf.getvalue())
        report = self.scrub(str(self.inbox / "export.zip"), "--lane", "review", "--rfq")
        files = self.outputs()
        self.assertEqual(len(files), 2, report)
        for f in files:
            self.assert_clean(f.read_bytes(), A_SECRETS + C_SECRETS, A_DIGITS + C_DIGITS)
        self.assertIn("notes.txt", report)          # reported as skipped
        self.assertNotIn("._gripper", report)
        entries = self.manifest()["files"]
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(e["emails"][0]["lane"] == "review" and e["emails"][0]["is_rfq"] for e in entries))

    def test_folder_input_scrubs_everything_together(self):
        # A name seen only in one email's header is still replaced in another email's body.
        self.put("a.eml", email_a())
        self.put("e.eml", email_e())
        self.put("c.msg", email_c())
        self.scrub(str(self.inbox), "--lane", "review", "--not-rfq")
        files = self.outputs()
        self.assertEqual(len(files), 3)
        for f in files:
            self.assert_clean(f.read_bytes(), A_SECRETS + E_SECRETS + C_SECRETS, A_DIGITS + E_DIGITS + C_DIGITS)
        self.assertFalse((self.inbox / ".scrub_map.json.tmp").exists())
        self.assertTrue(self.map.exists())

    def test_wildcard_input(self):
        self.put("a.eml", email_a())
        self.put("e.eml", email_e())
        self.put("notes.txt", b"not an email")
        self.scrub(str(self.inbox / "*.eml"), "--lane", "review", "--rfq")
        self.assertEqual(len(self.outputs()), 2)

    def test_forward_bundle_writes_the_attached_emails(self):
        outer = EmailMessage()
        outer["From"] = "Marisol Quintero-Vance <mquintero@brackwaterfluid.com>"
        outer["To"] = "rfq@ferncastmachine.com"
        outer["Subject"] = "FW: two RFQs"
        outer["Date"] = "Thu, 24 Sep 2026 10:00:00 -0400"
        outer.set_content("See attached, Marisol")
        for raw in (email_a(), email_e()):
            outer.add_attachment(BytesParser(policy=policy.default).parsebytes(raw))
        self.put("bundle.eml", outer.as_bytes())
        report = self.scrub(str(self.inbox / "bundle.eml"), "--lane", "review", "--rfq")
        files = self.outputs()
        self.assertEqual(len(files), 2, report)
        self.assertIn("forward bundle", report)
        for f in files:
            self.assert_clean(f.read_bytes(), A_SECRETS + E_SECRETS, A_DIGITS + E_DIGITS)


# ---------------------------------------------------------------------------------------------
# The output file and the manifest

class OutputTests(ScrubTestCase):
    def test_round_trip_through_mailfile(self):
        self.put("a.eml", email_a())
        report = self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis")
        data = self.one_output()
        msg = BytesParser(policy=policy.default).parsebytes(data)
        self.assertEqual(msg.get_content_type(), "text/plain")
        self.assertEqual(msg.get_content_charset(), "utf-8")
        self.assertIn(msg["Content-Transfer-Encoding"], ("8bit", "quoted-printable"))
        back = mailfile.load("x.eml", data)["emails"][0]
        self.assertEqual(back["subject"], str(msg["Subject"]))
        self.assertEqual(back["from_email"], msg["From"].addresses[0].addr_spec)
        self.assertEqual(back["body"], msg.get_content().strip("\n"))
        entry = self.manifest()["files"][0]["emails"][0]
        self.assertEqual(entry["subject"], back["subject"])
        self.assertEqual(entry["from_email"], back["from_email"])
        self.assertNotIn("reading the file back gave a different", report)
        # Headers that carry names or addresses are not copied.
        for h in ("Received", "Thread-Topic", "In-Reply-To", "X-Originating-IP"):
            self.assertIsNone(msg[h], h)
        self.assertTrue(str(msg["Message-ID"]).startswith("<scrub."))

    def test_long_lines_use_quoted_printable_and_still_round_trip(self):
        body = "Hi Tobiah,\n\n" + ("Please quote this long paragraph. " * 60) + "\n\nThanks,\nMarisol\n"
        self.put("long.eml", make_eml(A_HEADERS[:5], text=body))
        self.scrub(str(self.inbox / "long.eml"), "--lane", "milling_3axis")
        data = self.one_output()
        msg = BytesParser(policy=policy.default).parsebytes(data)
        self.assertEqual(msg["Content-Transfer-Encoding"], "quoted-printable")
        back = mailfile.load("x.eml", data)["emails"][0]
        self.assertEqual(back["body"], msg.get_content().strip("\n"))
        self.assert_clean(data, ["Tobiah", "Marisol"])

    def test_non_ascii_names_are_encoded_and_scrubbed(self):
        headers = [("From", "José Ibáñez-Grolier <jose.ibanez-grolier@vendramoor.com>"),
                   ("To", "rfq@ferncastmachine.com"), ("Subject", "Cotización para Vendramoor S.A."),
                   ("Date", "Tue, 22 Sep 2026 09:14:00 -0600")]
        body = "Hola,\n\nAdjunto el plano.\n\nSaludos,\nJosé Ibáñez-Grolier\nVendramoor S.A.\n"
        self.put("es.eml", make_eml(headers, text=body))
        self.scrub(str(self.inbox / "es.eml"), "--lane", "review", "--rfq")
        data = self.one_output()
        self.assert_clean(data, ["José", "Ibáñez", "Grolier", "Vendramoor", "ibanez", "jose."])
        back = mailfile.load("x.eml", data)["emails"][0]
        self.assertIn("Cotización", back["subject"])

    def test_attachments_dropped_and_listed_with_scrubbed_names(self):
        self.put("a.eml", email_a())
        self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis")
        data = self.one_output()
        msg = BytesParser(policy=policy.default).parsebytes(data)
        self.assertFalse(msg.is_multipart())
        listed = [n.strip() for n in str(msg["X-Scrubbed-Attachments"]).split(";")]
        self.assertEqual(len(listed), 2, listed)
        self.assertIn("BFS-30418_RevD.pdf", listed)
        self.assertTrue(any(n.endswith(".xlsx") and "RFQ BW-2291" in n for n in listed), listed)
        for n in listed:
            self.assertNotIn("Brackwater", n)
            self.assertNotIn("Marisol", n)
        entry = self.manifest()["files"][0]["emails"][0]
        self.assertEqual(sorted(entry["attachments"]), sorted(listed))
        back = mailfile.load("x.eml", data)["emails"][0]
        names = [a["name"] for a in back["attachments"]]
        if not names:
            self.skipTest("mailfile.load does not read X-Scrubbed-Attachments yet")
        self.assertEqual(sorted(names), sorted(listed))
        self.assertTrue(all(a["data"] is None and a["size"] == 0 for a in back["attachments"]))

    def test_keep_attachments_keeps_them_with_scrubbed_names_and_warns(self):
        self.put("c.msg", email_c())
        report = self.scrub(str(self.inbox / "c.msg"), "--lane", "milling_5axis", "--keep-attachments")
        data = self.one_output()
        msg = BytesParser(policy=policy.default).parsebytes(data)
        names = [p.get_filename() for p in msg.iter_attachments()]
        self.assertEqual(len(names), 2, names)
        self.assertTrue(all("Cordwainer" not in n and "HX-44120" in n for n in names), names)
        self.assertIsNone(msg["X-Scrubbed-Attachments"])
        self.assertIn("UNSCRUBBED", report)
        self.assertIn("WARNING", self.last_stderr)
        self.assert_clean(data, C_SECRETS, C_DIGITS)
        # Rerunning gives the same bytes (fixed MIME boundary).
        self.scrub(str(self.inbox / "c.msg"), "--lane", "milling_5axis", "--keep-attachments")
        self.assertEqual(self.one_output(), data)

    def test_manifest_entry_and_idempotent_updates(self):
        self.out.mkdir()
        generated = {"file": "rfq_bracket.eml", "source": "generated",
                     "emails": [{"subject": "RFQ", "from_email": "a@example.com",
                                 "attachments": ["QA-41127_RevB.pdf"], "lane": "milling_3axis", "is_rfq": True}]}
        (self.out / "manifest.json").write_text(json.dumps({"about": "fixtures", "files": [generated]}, indent=2),
                                                encoding="utf-8")
        self.put("a.eml", email_a())
        self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis")
        first = (self.out / "manifest.json").read_bytes()
        data = self.manifest()
        self.assertEqual(data["about"], "fixtures")
        self.assertEqual(data["files"][0], generated)
        self.assertEqual(len(data["files"]), 2)
        entry = data["files"][1]
        self.assertEqual(set(entry), {"file", "source", "reviewed", "emails"})
        self.assertEqual(entry["source"], "scrubbed")
        self.assertIs(entry["reviewed"], False)
        self.assertEqual(set(entry["emails"][0]), {"subject", "from_email", "attachments", "lane", "is_rfq"})
        self.assertEqual(entry["emails"][0]["lane"], "milling_3axis")
        self.assertIs(entry["emails"][0]["is_rfq"], True)
        self.assertTrue((self.out / entry["file"]).exists())
        eml_before = (self.out / entry["file"]).read_bytes()
        report = self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis")
        self.assertEqual((self.out / "manifest.json").read_bytes(), first)
        self.assertEqual((self.out / entry["file"]).read_bytes(), eml_before)
        self.assertIn("unchanged", report)
        # A new lane updates the same entry instead of adding one.
        self.scrub(str(self.inbox / "a.eml"), "--lane", "review", "--rfq")
        data = self.manifest()
        self.assertEqual(len(data["files"]), 2)
        self.assertEqual(data["files"][1]["emails"][0]["lane"], "review")
        self.assertEqual(len(self.outputs()), 1)

    def test_changed_subject_replaces_the_old_file(self):
        self.put("a.eml", email_a())
        self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis")
        old = self.outputs()[0].name
        self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis", "--part-numbers")
        files = self.outputs()
        self.assertEqual(len(files), 1)
        self.assertNotEqual(files[0].name, old)
        self.assertEqual([f["file"] for f in self.manifest()["files"]], [files[0].name])

    def test_review_flag_survives_an_unchanged_rerun_only(self):
        self.put("a.eml", email_a())
        self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis")
        data = self.manifest()
        data["files"][0]["reviewed"] = True
        (self.out / "manifest.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        report = self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis")
        self.assertIn("still reviewed", report)
        self.assertIs(self.manifest()["files"][0]["reviewed"], True)
        # A different lane is a different answer: review it again.
        self.scrub(str(self.inbox / "a.eml"), "--lane", "review", "--rfq")
        self.assertIs(self.manifest()["files"][0]["reviewed"], False)

    def test_existing_manifest_format_is_kept(self):
        real = ROOT / "tests" / "emails" / "manifest.json"
        if not real.exists():
            self.skipTest("tests/emails/manifest.json is not there yet")
        self.out.mkdir()
        original = real.read_text(encoding="utf-8")
        (self.out / "manifest.json").write_text(original, encoding="utf-8")
        before = json.loads(original)
        self.put("a.eml", email_a())
        self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis")
        text = (self.out / "manifest.json").read_text(encoding="utf-8")
        after = json.loads(text)
        self.assertEqual(after["about"], before["about"])
        self.assertEqual(after["files"][:len(before["files"])], before["files"])
        self.assertEqual(len(after["files"]), len(before["files"]) + 1)
        indent = scrub.manifest_indent(real)
        self.assertEqual(text, json.dumps(after, indent=indent, ensure_ascii=False) + "\n")
        # The generated part of the file is byte for byte what it was.
        head = original[:original.rstrip().rindex("]")].rstrip()      # up to the last entry's "}"
        self.assertTrue(text.startswith(head), "generated entries were reformatted")

    def test_scrubbed_files_pass_the_import_fixture_checks(self):
        # The same per-file checks tests/test_import.py makes on every file in tests/emails.
        self.put("a.eml", email_a())
        self.put("b.eml", email_b())
        self.put("c.msg", email_c(body="rtf", exchange_sender=True))
        self.put("e.eml", email_e())
        self.scrub(str(self.inbox), "--lane", "review", "--rfq")
        on_disk = {p.name for p in self.out.iterdir() if p.name != "manifest.json"}
        entries = self.manifest()["files"]
        self.assertEqual(on_disk, {e["file"] for e in entries})
        for entry in entries:
            with self.subTest(file=entry["file"]):
                res = mailfile.load(entry["file"], (self.out / entry["file"]).read_bytes())
                emails = [e for e in res["emails"] if not e.get("container_only")]
                self.assertEqual(len(emails), len(entry["emails"]))
                for got, want in zip(emails, entry["emails"]):
                    self.assertEqual(got["subject"], want["subject"])
                    self.assertEqual(got["from_email"], want["from_email"].lower())
                    self.assertEqual([a["name"] for a in got["attachments"]], want["attachments"])
                    self.assertTrue(got["body"].strip())
                    self.assertTrue(got["date"])
                    for text in (got["subject"], got["body"]):
                        self.assertNotIn(chr(0x2014), text)
                        self.assertNotIn(chr(0x2013), text)

    def test_dashes_become_hyphens_and_a_missing_date_gets_a_fixed_one(self):
        em = chr(0x2014)
        en = chr(0x2013)
        headers = [("From", "Marisol Quintero-Vance <mquintero@brackwaterfluid.com>"),
                   ("To", "rfq@ferncastmachine.com"), ("Subject", f"RFQ {em} manifold block, qty 10{en}20")]
        body = f"Hi Tobiah {em} please quote 10{en}20 pcs.\n\nThanks,\nMarisol\n"
        self.put("dash.eml", make_eml(headers, text=body, attachments=[(f"Block {en} RevA.pdf", "application/pdf", b"%PDF")]))
        report = self.scrub(str(self.inbox / "dash.eml"), "--lane", "milling_3axis")
        data = self.one_output()
        text = self.views(data)
        self.assertNotIn(em, text)
        self.assertNotIn(en, text)
        self.assertIn("RFQ - manifold block, qty 10-20", text)
        self.assertIn("Block - RevA.pdf", text)
        self.assertIn("dash(es) turned into plain hyphens", report)
        back = mailfile.load("x.eml", data)["emails"][0]
        self.assertTrue(back["date"].startswith("2026-"))
        self.assertIn("has no date", report)
        self.scrub(str(self.inbox / "dash.eml"), "--lane", "milling_3axis")
        self.assertEqual(self.one_output(), data)

    def test_dry_run_writes_nothing(self):
        self.put("a.eml", email_a())
        report = self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis", "--dry-run",
                            "--map", "Ferncast=Hollowby")
        self.assertIn("DRY RUN", report)
        self.assertIn("would write", report)
        self.assertIn("Marisol", report)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.map.exists())
        self.assertEqual(sorted(p.name for p in self.inbox.iterdir()), ["a.eml"])

    def test_review_lane_needs_a_flag_and_lanes_are_checked(self):
        self.put("a.eml", email_a())
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                scrub.run([str(self.inbox / "a.eml"), "--lane", "review", "--out", str(self.out),
                           "--map-file", str(self.map)], stdout=io.StringIO())
            with self.assertRaises(SystemExit):
                scrub.run([str(self.inbox / "a.eml"), "--lane", "welding", "--out", str(self.out),
                           "--map-file", str(self.map)], stdout=io.StringIO())
            with self.assertRaises(SystemExit):
                scrub.run([str(self.inbox / "a.eml"), "--lane", "orders", "--map", "no-equals-sign",
                           "--out", str(self.out), "--map-file", str(self.map)], stdout=io.StringIO())
        self.assertFalse(self.out.exists())

    def test_default_rfq_flag_follows_the_lane(self):
        self.put("a.eml", email_a())
        self.scrub(str(self.inbox / "a.eml"), "--lane", "orders")
        self.assertIs(self.manifest()["files"][0]["emails"][0]["is_rfq"], False)
        self.scrub(str(self.inbox / "a.eml"), "--lane", "itar")
        self.assertIs(self.manifest()["files"][0]["emails"][0]["is_rfq"], True)

    def test_unreadable_input_is_reported(self):
        self.put("junk.eml", b"\x00\x01\x02 not mail")
        report = self.scrub(str(self.inbox / "junk.eml"), str(self.inbox / "missing.msg"), "--lane", "review",
                            "--rfq", expect=1)
        self.assertIn("missing.msg", report)
        self.assertFalse(self.outputs())


# ---------------------------------------------------------------------------------------------
# Consistency across runs, part numbers, --map

class ConsistencyTests(ScrubTestCase):
    def test_same_fakes_across_runs_and_emails(self):
        self.put("a.eml", email_a())
        self.put("e.eml", email_e())
        out1, out2 = self.dir / "o1", self.dir / "o2"
        self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis", out=out1)
        smap = json.loads(self.map.read_text(encoding="utf-8"))
        fake_first, fake_last = smap["first"]["tobiah"], smap["last"]["runcorn"]
        self.assertIn(f"{fake_first} {fake_last}", self.views(self.one_output(out1)))
        # A later run, a different email, the same person: the same fake.
        self.scrub(str(self.inbox / "e.eml"), "--lane", "turning", out=out2)
        text = self.views(self.one_output(out2))
        self.assertIn(f"{fake_first} {fake_last}", text)
        self.assertIn(f"Mr. {fake_last}", text)
        # The shop's domain gets the same fake domain in both.
        dom = smap["domain"]["ferncastmachine.com"]
        self.assertIn("@" + dom, text)
        # Rerunning the first email gives identical bytes.
        before = self.one_output(out1)
        self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis", out=out1)
        self.assertEqual(self.one_output(out1), before)

    def test_fakes_are_deterministic_without_the_map(self):
        self.put("a.eml", email_a())
        o1, o2 = self.dir / "o1", self.dir / "o2"
        self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis", out=o1, map_file=self.dir / "m1.json")
        self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis", out=o2, map_file=self.dir / "m2.json")
        self.assertEqual(self.one_output(o1), self.one_output(o2))

    def test_part_numbers_kept_and_listed_by_default(self):
        self.put("a.eml", email_a())
        report = self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis")
        self.assertIn("Part numbers KEPT", report)
        self.assertRegex(report, r"BFS-30418\s+\(x\d+, labeled\)")
        text = self.views(self.one_output())
        self.assertIn("P/N BFS-30418 Rev D", text)
        self.assertIn("BFS-30418_RevD.pdf", text)
        self.assertNotIn("6061-T6  (x", report)        # specs are not part numbers
        self.assertNotIn("MIL-A-8625  (x", report)
        self.assertNotIn("BW-2291  (x", report)        # an RFQ number is not a part number

    def test_part_numbers_replaced_consistently(self):
        self.put("a.eml", email_a())
        report = self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis", "--part-numbers")
        self.assertIn("Part numbers replaced", report)
        data = self.one_output()
        self.assert_clean(data, A_SECRETS + ["BFS-30418", "30418"], A_DIGITS)
        smap = json.loads(self.map.read_text(encoding="utf-8"))
        fake = smap["part"]["BFS-30418"]
        self.assertRegex(fake, r"^[A-Z]{3}-\d{5}$")
        back = mailfile.load("x.eml", data)["emails"][0]
        self.assertIn(fake, back["subject"])
        self.assertIn(f"P/N {fake} Rev D", back["body"])
        msg = BytesParser(policy=policy.default).parsebytes(data)
        self.assertIn(f"{fake}_RevD.pdf", str(msg["X-Scrubbed-Attachments"]))
        self.assertIn("6061-T6", back["body"])
        self.assertIn("MIL-A-8625", back["body"])

    def test_map_pairs_apply_everywhere_and_are_remembered(self):
        self.put("a.eml", email_a())
        self.put("e.eml", email_e())
        o1, o2 = self.dir / "o1", self.dir / "o2"
        self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis", out=o1)
        # A fresh map for the --map run, so the pairs are not fighting earlier fakes.
        self.map.unlink()
        self.scrub(str(self.inbox / "a.eml"), "--lane", "milling_3axis",
                   "--map", "ferncastmachine.com=mesaridgeprecision.com",
                   "--map", "Ferncast Machine=Mesa Ridge Precision",
                   "--map", "Tobiah Runcorn=Tom Becker",
                   "--map", "Grand Rapids plant=north plant", out=o2)
        text = self.views(self.one_output(o2))
        self.assertIn("rfq@mesaridgeprecision.com", text)
        self.assertIn("Tom Becker", text)
        self.assertIn("Hi Tom,", text)
        self.assertIn("Mesa Ridge Precision Co.", text)
        self.assertIn("our north plant", text)
        self.assertIn("tbecker@mesaridgeprecision.com", text)
        self.assert_clean(self.one_output(o2), A_SECRETS, A_DIGITS)
        # Remembered: a later run without --map uses the same pairs.
        o3 = self.dir / "o3"
        self.scrub(str(self.inbox / "e.eml"), "--lane", "turning", out=o3)
        text = self.views(self.one_output(o3))
        self.assertIn("Tom Becker <tbecker@mesaridgeprecision.com>", text)
        self.assertIn("Dear Mr. Becker", text)


# ---------------------------------------------------------------------------------------------
# What each kind of replacement catches

class ReplacementTests(unittest.TestCase):
    def test_phone_formats(self):
        samples = [
            "(602) 555-2231", "602-555-2231", "602.555.2231", "602 555 2231", "+1 602 555 2231",
            "1-602-555-2231", "+1 (602) 555-2231 ext. 45", "602-555-2231 x1234", "Tel: 6025552231",
            "Fax: 555-2231", "M: 602.555.2231", "p. 602.555.2231", "Office 602 555 2231 Ext 9",
            "+44 20 7946 0321", "+49 (0)30 5550 2231", "+52 81 5555 2231", "Phone: 0044 20 7946 0321",
            "Direct line: +61 491 570 158", "Cell (602)555-2231",
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                out, s = scrub_one_text(f"Call me at {sample} today.")
                digits = re.sub(r"\D", "", sample)
                self.assertNotIn(digits[-7:], re.sub(r"\D", "", out), out)
                self.assertTrue(out.startswith("Call me at ") and out.endswith(" today."), out)
                if sample.startswith(("(602", "602", "+1", "1-", "Tel", "M:", "p.", "Office", "Cell")):
                    self.assertIn("55501", re.sub(r"\D", "", out), out)
                if "ext" in sample.lower() or "x1234" in sample:
                    self.assertRegex(out, r"(?i)(ext\.?|x) ?\d+")

    def test_phone_format_is_kept(self):
        out, _ = scrub_one_text("O: (480) 555-2200 x214 | M: 602.555.8812 | +1 623-555-9087")
        self.assertRegex(out, r"^O: \(\d{3}\) 555-01\d\d x\d{3} \| M: \d{3}\.555\.01\d\d \| \+1 \d{3}-555-01\d\d$")

    def test_things_that_are_not_phones(self):
        text = ("Dates 2026-09-24 and 09/24/2026, qty 25/50/100, RFQ 26-0412, ZIP 85043-1234, "
                "P/N 602-555-2231, bore .1880 +.0005, 6061-T6, 8-32 UNC-2B, $1,250.00, PO 4500123456")
        out, _ = scrub_one_text(text)
        self.assertEqual(out, text)

    def test_street_addresses_and_postal_lines(self):
        cases = [
            "1500 W Industrial Park Dr, Bldg 3", "PO Box 4471", "P.O. Box 12", "22 N 5th St",
            "4410 E Warner Rd Ste 120", "100 State Route 9", "Tempe, AZ 85284-1234",
            "Toronto, ON M5V 2T6", "Phoenix, Arizona 85043", "7 Quarrendon Court", "31 Hollin Lane NE",
            "Suite 400", "12 Kestermoor Blvd., Suite 210",
        ]
        for case in cases:
            with self.subTest(case=case):
                out, _ = scrub_one_text(f"Ship to:\n{case}\nThanks")
                line = out.splitlines()[1]
                self.assertNotEqual(line, case)
                for token in re.findall(r"[A-Za-z]{4,}|\d{2,}", case):
                    if token.lower() in ("suite", "blvd", "court", "lane", "route", "state", "industrial",
                                         "park", "arizona", "warner", "bldg") or len(token) < 3:
                        continue
                    self.assertNotIn(token, line, line)

    def test_city_state_line_alone_and_known_city_in_prose(self):
        out, _ = scrub_one_text("Visit our Sewickley shop.\n\n88 Larkspur Ln\nSewickley, PA 15143\nMesa, AZ\n")
        self.assertNotIn("Sewickley", out)
        self.assertNotIn("15143", out)
        self.assertNotIn("Mesa, AZ", out)

    def test_company_names_with_suffixes(self):
        cases = {
            "Oxenholme Tool & Die, Inc.": "Oxenholme", "Pillsworth Corp.": "Pillsworth",
            "Velloway Gear Co.": "Velloway", "Dunstanton Machining LLC": "Dunstanton",
            "Grisedale Industries GmbH": "Grisedale", "ACME-TORVALD MFG CORPORATION": "TORVALD",
            "Wexholt Limited": "Wexholt",
        }
        for name, word in cases.items():
            with self.subTest(name=name):
                out, s = scrub_one_text(f"Please quote for {name}. {word} needs it soon.")
                self.assertNotIn(word.lower(), out.lower(), out)
                self.assertTrue(out.startswith("Please quote for "), out)

    def test_generic_trade_words_are_kept(self):
        out, _ = scrub_one_text("Regards,\nJo Ellen\nBuyer\nHarbrook Precision Aerospace, Inc.\n")
        self.assertIn("Precision Aerospace, Inc.", out)
        self.assertNotIn("Harbrook", out)

    def test_a_company_of_only_generic_words_gets_a_whole_fake(self):
        out, _ = scrub_one_text("From all of us at Precision Machine Products Company.")
        self.assertNotIn("Precision Machine Products", out)
        self.assertTrue(out.endswith(" Company."), out)

    def test_company_matching_the_sender_domain(self):
        text = "Thanks,\nArlen\nHarwick Brothers Precision\nharwickbros.com"
        out, _ = scrub_one_text(text, {"from_name": "Arlen Voskuijlen", "from_email": "arlen@harwickbrothers.com"})
        self.assertNotIn("Harwick", out)
        self.assertNotIn("harwick", out)
        self.assertNotIn("Arlen", out)

    def test_name_forms(self):
        headers = {"from_name": "Marisol Quintero-Vance", "from_email": "mquintero@brackwaterfluid.com"}
        text = ("MARISOL QUINTERO-VANCE\nQuintero-Vance, Marisol\nM. Quintero-Vance\nMarisol Q.\n"
                "Marisol's drawing\nQuintero-Vance's team\nMarisol\nQuintero-Vance\nmarisol quintero-vance\n"
                "Ms. Quintero-Vance")
        out, _ = scrub_one_text(text, headers)
        self.assertNotRegex(out.lower(), r"marisol|quintero|vance")
        lines = out.splitlines()
        self.assertTrue(lines[0].isupper())
        self.assertRegex(lines[1], r"^[A-Z][a-z]+, [A-Z][a-z]+$")
        self.assertRegex(lines[2], r"^[A-Z]\. [A-Z][a-z]+$")
        self.assertRegex(lines[4], r"^[A-Z][a-z]+'s drawing$")
        self.assertTrue(lines[8].islower())

    def test_name_split_across_lines_in_a_signature_without_headers(self):
        text = "Please see attached.\n\nThanks,\nGwynneth\nPardoe\nQuality Engineer\nSt. Brevard Instruments\n"
        out, _ = scrub_one_text(text, {"from_email": "quality@stbrevardinst.com"})
        self.assertNotIn("Gwynneth", out)
        self.assertNotIn("Pardoe", out)
        self.assertIn("Quality Engineer", out)

    def test_signature_table_cells(self):
        text = "Thank you,\n\nOsric | Vandermolen\n\nProgram Manager, Halsworth Dynamics Inc.\n"
        out, _ = scrub_one_text(text)
        self.assertNotIn("Osric", out)
        self.assertNotIn("Vandermolen", out)
        self.assertNotIn("Halsworth", out)

    def test_greetings_honorifics_and_common_first_names(self):
        text = ("Hi Dorrit and Ambrosine,\n\nDr. Quennell asked me to send this. Please loop in "
                "Kevin Stroudley too.\n\nCheers,\nPenhaligon")
        out, _ = scrub_one_text(text)
        for word in ("Dorrit", "Ambrosine", "Quennell", "Kevin", "Stroudley", "Penhaligon"):
            self.assertNotIn(word, out)
        self.assertTrue(out.startswith("Hi "))

    def test_quoted_header_blocks(self):
        text = ("> From: Doe, Radomir [mailto:rdoe@kelsharrow.com]\n> Sent: Monday\n"
                "> To: Ilsabet Frame; Corvin Ashpole <cashpole@kelsharrow.com>\n> Subject: RE: quote\n>\n"
                "> Radomir, Ilsabet, and Corvin: see below.\n")
        out, _ = scrub_one_text(text)
        for word in ("Radomir", "Ilsabet", "Corvin", "Ashpole", "rdoe", "kelsharrow"):
            self.assertNotIn(word, out)
        self.assertIn("[mailto:", out)

    def test_ambiguous_names_only_where_they_are_names(self):
        headers = {"from_name": "Mark Price", "from_email": "mprice@ottermere.com"}
        text = ("Hi Mark,\n\nPlease mark the parts per the drawing. The unit price should include anodize.\n"
                "Mark will send the PO. Talk to Price about freight. Mark Price approved it.\n\nThanks,\nMark")
        out, _ = scrub_one_text(text, headers)
        self.assertIn("mark the parts", out)
        self.assertIn("unit price", out)
        self.assertNotIn("Hi Mark", out)
        self.assertNotIn("Mark Price", out)
        self.assertNotIn("Mark will send", out)
        self.assertFalse(out.rstrip().endswith("Mark"), out)

    def test_labeled_company_lines(self):
        text = "Company: Ravelstoke Instruments\nShip To: Ravelstoke Instruments, 12 Oakmere St\nRavelstoke needs it."
        out, _ = scrub_one_text(text)
        self.assertNotIn("Ravelstoke", out)
        self.assertNotIn("Oakmere", out)
        self.assertIn("Instruments", out)

    def test_curly_apostrophes_in_names(self):
        headers = {"from_name": "Siobhan O'Farrelly", "from_email": "sofarrelly@glenvarra.com"}
        text = "Thanks,\nSiobhan O\u2019Farrelly\nO\u2019Farrelly\u2019s team"
        out, _ = scrub_one_text(text, headers)
        self.assertNotIn("Farrelly", out)
        self.assertNotIn("Siobhan", out)

    def test_international_signatures(self):
        cases = [
            ({"from_name": "Rhys Tennant-Ogilvy", "from_email": "rto@vexmoor.co.uk",
              "body": "Cheers,\nRhys\nVexmoor Engineering Ltd\n+44 (0)1457 555 123\n"
                      "https://www.linkedin.com/in/rhys-tennant-ogilvy-12345"},
             ["Rhys", "Tennant", "Ogilvy", "Vexmoor", "1457 555 123", "rhys-tennant"],
             ["Cheers,", "Engineering Ltd", "+44 (0)", "linkedin.com/s/"]),
            ({"from_name": "Ana-Lucia Ferreira dos Santos", "from_email": "alsantos@orvalla.com.br",
              "body": "Obrigada,\nAna-Lucia Ferreira dos Santos\nOrvalla Usinagem Ltda.\nSantos will call."},
             ["Ana-Lucia", "Ferreira", "Santos", "Orvalla", "alsantos"],
             ["Obrigada,", " dos ", "Usinagem Ltda.", ".com.br"]),
        ]
        for em, secrets, kept in cases:
            with self.subTest(sender=em["from_name"]):
                em.update(subject="", to=[], cc=[])
                s = fresh()
                s.collect(em)
                s.finish_collect()
                res = scrub.scrub_one(s, em)
                text = res["eml"].decode("utf-8")
                for secret in secrets:
                    self.assertNotIn(secret.lower(), text.lower())
                for keep in kept:
                    self.assertIn(keep, text)
                self.assertEqual(res["checks"], [])

    def test_names_in_the_subject_and_sign_off_words(self):
        em = {"subject": "Quote request from Dmitri Zolotarev", "body": "Cheers,\nsee attached",
              "from_name": "", "from_email": "rfq@kelsharrow.com", "to": [], "cc": []}
        s = fresh()
        s.collect(em)
        s.finish_collect()
        res = scrub.scrub_one(s, em)
        self.assertNotIn("Dmitri", res["subject"])
        self.assertNotIn("Zolotarev", res["subject"])
        self.assertTrue(res["body"].startswith("Cheers,"), res["body"])

    def test_emails_urls_and_domains(self):
        headers = {"from_name": "Jana Kolvenbach", "from_email": "jana.kolvenbach@ostrander-hydraulic.com"}
        text = ("Write to jana.kolvenbach@ostrander-hydraulic.com or quotes@ostrander-hydraulic.com.\n"
                "Link: <mailto:jana.kolvenbach@ostrander-hydraulic.com?subject=Jana%20Kolvenbach%20RFQ>\n"
                "Portal https://portal.ostrander-hydraulic.com/rfq/jana-kolvenbach?id=77\n"
                "Share https://ostrander-my.sharepoint.com/personal/jana_kolvenbach/Documents/rfq.zip\n"
                "Wrapped https://urldefense.com/v3/__https://ostrander-hydraulic.com/__;!!jana$\n"
                "Site: ostrander-hydraulic.com, www.ostrander-hydraulic.com\n"
                "Personal: jkolvenbach77@gmail.com\n")
        out, s = scrub_one_text(text, headers)
        for word in ("jana", "kolvenbach", "ostrander", "77@"):
            self.assertNotIn(word, out.lower())
        self.assertIn("quotes@", out)                       # a role mailbox keeps its name
        self.assertIn("sharepoint.com/s/", out)             # public service kept, path replaced
        self.assertIn("@gmail.com", out)
        self.assertIn("<mailto:", out)
        fake = s.fake_email("jana.kolvenbach@ostrander-hydraulic.com")
        first, last = fake.split("@")[0].split(".")
        self.assertEqual(first, s.map.get("first", "jana").lower())
        self.assertEqual(last, s.map.get("last", "kolvenbach").lower())

    def test_things_that_are_not_domains(self):
        text = "Dwg.No. 1234, see Note.3, file.pdf and model.step, e.g. this, U.S. made, rev.B"
        out, _ = scrub_one_text(text)
        self.assertEqual(out, text)
        out, _ = scrub_one_text("Site: KELSHARROW-TOOL.COM and kelsharrow-tool.co")
        self.assertNotIn("kelsharrow", out.lower())

    def test_email_address_pattern_is_mimicked(self):
        s = fresh()
        s.add_person("Jana", "Kolvenbach")
        ff, fl = s.fake_token("Jana", "first").lower(), s.fake_token("Kolvenbach", "last").lower()
        for real, want in (("jana.kolvenbach", f"{ff}.{fl}"), ("jkolvenbach", f"{ff[0]}{fl}"),
                           ("janak", f"{ff}{fl[0]}"), ("kolvenbachj", f"{fl}{ff[0]}"),
                           ("jana_kolvenbach2", None)):
            got = scrub.mimic_local(real, "Jana", "Kolvenbach", ff, fl)
            if want:
                self.assertEqual(got, want)
            else:
                self.assertRegex(got, rf"^{ff}_{fl}\d$")

    def test_check_list_flags_what_it_could_not_place(self):
        out, s = scrub_one_text("The Cassian Throckmorton account is open. Also ask Brenda.\nCall 602 5552 231 if needed.")
        checks = scrub.check_these(s, [("body", out)])
        joined = "\n".join(checks)
        self.assertIn("Cassian Throckmorton", joined)
        self.assertIn("Brenda", joined)
        # Fakes are never flagged.
        s2 = fresh()
        s2.add_person("Radomir", "Doe")
        fake = s2.scrub_text("Radomir Doe")
        self.assertEqual(scrub.check_these(s2, [("body", fake)]), [])

    def test_check_list_flags_a_known_name_left_behind(self):
        s = fresh()
        s.add_person("Radomir", "Keswold")
        checks = scrub.check_these(s, [("body", "radomirkeswold was here")])
        self.assertEqual(checks, [])
        checks = scrub.check_these(s, [("body", "RADOMIR wrote")])
        self.assertTrue(any("radomir" in c.lower() for c in checks), checks)


# ---------------------------------------------------------------------------------------------
# Repo rules

class RepoRuleTests(unittest.TestCase):
    def test_fictional_lists_stay_clear_of_the_demo(self):
        files = [ROOT / "data" / "sample_emails.json", ROOT / "data" / "rfq_beta" / "emails.json",
                 ROOT / "shop_config.json", ROOT / "tools" / "make_email_fixtures.py"]
        text = "\n".join(p.read_text(encoding="utf-8") for p in files if p.exists())
        low = text.lower()
        people = set()
        for m in re.finditer(r'"(?:from_name|owner|account_manager)": "([^"]+)"', text):
            people.update(m.group(1).lower().split())
        words = set(re.findall(r"[a-z]{3,}", low))
        for name in scrub.FAKE_FIRST + scrub.FAKE_LAST:
            self.assertNotIn(name.lower(), people, name)
            self.assertNotIn(name.lower(), words, name)
        for stem in scrub.COMPANY_STEMS:
            self.assertNotIn(stem.lower(), low, stem)
        for street in scrub.FAKE_STREETS:
            self.assertNotIn(street.lower(), words, street)
        for city, _ in scrub.FAKE_CITIES:
            self.assertNotIn(city.lower(), low, city)
        self.assertEqual(len(set(scrub.FAKE_FIRST)), len(scrub.FAKE_FIRST))
        self.assertEqual(len(set(scrub.FAKE_LAST)), len(scrub.FAKE_LAST))
        self.assertEqual(len(set(scrub.COMPANY_STEMS)), len(scrub.COMPANY_STEMS))
        # A fake is never an ordinary word, or "check these" could not tell it from text.
        for name in scrub.FAKE_FIRST + scrub.FAKE_LAST + scrub.COMPANY_STEMS:
            self.assertNotIn(name.lower(), scrub.AMBIGUOUS, name)

    def test_private_emails_is_git_ignored(self):
        lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("private_emails/", [line.strip() for line in lines])
        self.assertEqual(scrub.DEFAULT_MAP.parent.name, "private_emails")

    def test_no_em_or_en_dashes(self):
        for path in (ROOT / "tools" / "scrub_email.py", Path(__file__)):
            text = path.read_text(encoding="utf-8")
            for dash in (chr(0x2013), chr(0x2014)):
                self.assertNotIn(dash, text, path)

    def test_standard_library_only(self):
        source = (ROOT / "tools" / "scrub_email.py").read_text(encoding="utf-8")
        imports = set(re.findall(r"^(?:from|import) ([\w.]+)", source, re.M))
        allowed = {"__future__", "argparse", "glob", "hashlib", "json", "os", "re", "sys", "unicodedata",
                   "collections", "datetime", "email", "email.headerregistry", "email.message",
                   "email.policy", "pathlib", "typing", "mailfile"}
        self.assertLessEqual(imports, allowed, imports - allowed)


if __name__ == "__main__":
    unittest.main()
