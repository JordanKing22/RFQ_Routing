"""
Tests for mailfile.py (reading uploaded .eml, .msg, and .zip files) and tools/msgwriter.py (writing
.msg fixtures): round trips through the writer for every option, .eml and .zip cases a mail export
really produces, the LZFu test vectors from MS-OXRTFCP, and hostile files (truncated, looped,
oversized, out of range, zip bombs, path traversal, deep nesting) that must become a warning or a
skipped entry, never an exception.

    python -m unittest tests.test_mailfile -v

Standard library only. Every person, company, and address here is made up.
"""

from __future__ import annotations

import base64
import email.message
import email.policy
import io
import random
import struct
import sys
import time
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import mailfile  # noqa: E402
import msgwriter  # noqa: E402
from mailfile import Limits  # noqa: E402

KEYS = {"source", "format", "message_id", "subject", "from_name", "from_email", "to", "cc", "date", "body",
        "body_format", "attachments", "container_only", "warnings"}
ATTACHMENT_KEYS = {"name", "content_type", "data", "size", "inline"}

PDF = b"%PDF-1.7\n" + bytes(range(256)) * 24 + b"\n%%EOF\n"          # 6 KB: a regular CFB stream
LOGO = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1500                            # a small signature logo
SCREENSHOT = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 200             # 51 KB: a pasted screenshot


def rfq(**changes):
    """A fictional RFQ as a msgwriter email dict."""
    em = {
        "subject": "RFQ 4471 - bracket, 6061-T6",
        "from_name": "Avery Quill",
        "from_email": "Avery.Quill@example.com",
        "to": ["Jordan Vale <quotes@example.net>", "sales@example.net"],
        "cc": ["Kit Marsh <kit.marsh@example.org>"],
        "date": "2026-09-24T08:15:00-07:00",
        "message_id": "rfq-4471@example.com",
        "body": "Hello Jordan,\nPlease quote 50 and 250 pcs of QA-41127 rev B.\nThanks,\nAvery",
        "attachments": [{"name": "QA-41127_RevB.pdf", "content_type": "application/pdf", "data": PDF}],
    }
    em.update(changes)
    return em


def inner_rfq(n):
    return rfq(subject=f"RFQ {n} - spacer", from_name="Rowan Pike", from_email=f"rowan{n}@example.org",
               to=["quotes@example.net"], cc=[], date="2026-09-20T10:00:00-05:00", message_id=f"inner-{n}@example.org",
               body=f"Please quote spacer SP-{n:03d}, qty 100.",
               attachments=[{"name": f"SP-{n:03d}.pdf", "content_type": "application/pdf", "data": b"%PDF-1.4 spacer"}])


def load_one(filename, data, limits=mailfile.LIMITS):
    result = mailfile.load(filename, data, limits)
    assert_shape(result)
    return result


def assert_shape(result):
    """The contract's shape, whatever went in."""
    assert set(result) == {"emails", "skipped"}, result.keys()
    for em in result["emails"]:
        assert set(em) == KEYS, set(em) ^ KEYS
        assert em["format"] in ("eml", "msg")
        assert em["body_format"] in ("text", "html", "rtf", "none")
        assert isinstance(em["subject"], str) and isinstance(em["body"], str)
        assert isinstance(em["to"], list) and isinstance(em["cc"], list)
        assert isinstance(em["warnings"], list) and isinstance(em["container_only"], bool)
        assert em["from_email"] == em["from_email"].lower()
        for a in em["attachments"]:
            assert set(a) == ATTACHMENT_KEYS, a.keys()
            assert a["data"] is None or isinstance(a["data"], bytes)
            assert a["name"] and len(a["name"]) <= 120 and "/" not in a["name"] and "\\" not in a["name"]
        for text in [em["subject"], em["body"], em["from_name"], *em["to"], *em["cc"]]:
            text.encode("utf-8")  # no lone surrogates
    for s in result["skipped"]:
        assert set(s) == {"source", "reason"} and s["reason"], s


def eml_bytes(msg):
    return msg.as_bytes(policy=email.policy.SMTP)


def simple_eml(subject="RFQ 5102 - hinge plate", body="Please quote 20 pcs of HP-5102.", sender="Avery Quill <avery.quill@example.com>",
               **headers):
    msg = email.message.EmailMessage()
    msg["From"] = sender
    msg["To"] = "Jordan Vale <quotes@example.net>"
    msg["Subject"] = subject
    msg["Date"] = "Thu, 24 Sep 2026 08:15:00 -0700"
    msg["Message-ID"] = "<rfq-5102@example.com>"
    for k, v in headers.items():
        msg[k.replace("_", "-")] = v
    msg.set_content(body)
    return msg


def zip_of(entries, compression=zipfile.ZIP_DEFLATED):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression) as zf:
        for name, data in entries:
            if name.endswith("/"):
                zf.writestr(zipfile.ZipInfo(name), b"")
            else:
                zf.writestr(zipfile.ZipInfo(name), data, compress_type=compression)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# CFB helpers for the hostile .msg tests (the writer lays sectors out in order)
# --------------------------------------------------------------------------- #
def dir_entries(data):
    """[(index, name, type, left, right, child, start, size, offset)] of a file msgwriter wrote."""
    first_dir = struct.unpack_from("<I", data, 48)[0]
    out = []
    off = 512 + first_dir * 512
    i = 0
    while off + 128 <= len(data):
        raw = data[off:off + 128]
        name_len = struct.unpack_from("<H", raw, 64)[0]
        kind = raw[66]
        if kind == 0 and name_len == 0:
            break
        name = raw[:max(0, name_len - 2)].decode("utf-16-le")
        left, right, child = struct.unpack_from("<III", raw, 68)
        start, size = struct.unpack_from("<IQ", raw, 116)
        out.append((i, name, kind, left, right, child, start, size, off))
        off += 128
        i += 1
    return out


def entry_named(data, name, min_size=0):
    for e in dir_entries(data):
        if e[1] == name and e[7] >= min_size:
            return e
    raise KeyError(name)


def patched(data, offset, fmt, value):
    out = bytearray(data)
    struct.pack_into(fmt, out, offset, value)
    return bytes(out)


def fat_offset(sector):
    return 512 + 4 * sector  # the writer puts the FAT in the first sectors


# --------------------------------------------------------------------------- #
class SniffTests(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(mailfile.sniff(msgwriter.build_msg(rfq())), "msg")
        self.assertEqual(mailfile.sniff(zip_of([("a.eml", b"x")])), "zip")
        self.assertEqual(mailfile.sniff(zip_of([])), "zip")  # an empty zip is only an end record
        self.assertEqual(mailfile.sniff(eml_bytes(simple_eml())), "eml")
        self.assertEqual(mailfile.sniff(b"\xef\xbb\xbf\r\nReceived: from mx.example.net\r\nSubject: x\r\n\r\nhi"), "eml")
        self.assertEqual(mailfile.sniff(b"From quotes@example.net Thu Sep 24 08:15:00 2026\nFrom: a@example.com\n\nhi"), "eml")

    def test_not_email(self):
        for data in (b"", b"%PDF-1.4\n", b"hello world", b"Note: just a text file\n\nbody", b"\x00" * 100, None, "text"):
            self.assertIsNone(mailfile.sniff(data), data)


# --------------------------------------------------------------------------- #
class LzfuTests(unittest.TestCase):
    # MS-OXRTFCP 3.1.1 and 3.1.2: the specification's own compressed examples.
    VECTOR_1 = bytes.fromhex("2d0000002b0000004c5a4675f1c5c7a703000a007263706731323542320af32068656c090020627705b06c647d0a800fa0")
    RAW_1 = b"{\\rtf1\\ansi\\ansicpg1252\\pard hello world}\r\n"
    VECTOR_2 = bytes.fromhex("1a0000001c0000004c5a4675e2d44b51410004205758595a0d6e7d010eb0")
    RAW_2 = b"{\\rtf1 WXYZWXYZWXYZWXYZWXYZ}"

    def test_spec_vectors_decompress(self):
        for vector, raw in ((self.VECTOR_1, self.RAW_1), (self.VECTOR_2, self.RAW_2)):
            out, warnings = mailfile.decompress_rtf(vector)
            self.assertEqual(out, raw)
            self.assertEqual(warnings, [])  # the CRC matched

    def test_spec_vectors_compress_byte_for_byte(self):
        self.assertEqual(msgwriter.compress_rtf(self.RAW_1), self.VECTOR_1)
        self.assertEqual(msgwriter.compress_rtf(self.RAW_2), self.VECTOR_2)  # the overlapping run

    def test_prebuffer(self):
        self.assertEqual(len(mailfile._RTF_PREBUF), 207)
        self.assertEqual(mailfile._RTF_PREBUF, msgwriter.RTF_PREBUF)

    def test_round_trips(self):
        rng = random.Random(4471)
        samples = [b"", b"a", b"ab" * 3000, bytes(range(256)) * 40, msgwriter.RTF_PREBUF * 5]
        for _ in range(25):
            alphabet = rng.choice([b"ab", b"{\\rtf1 par pard }", bytes(range(256))])
            samples.append(bytes(rng.choice(alphabet) for _ in range(rng.choice([1, 17, 18, 4095, 4096, 4097, 9000]))))
        for raw in samples:
            packed = msgwriter.compress_rtf(raw)
            out, warnings = mailfile.decompress_rtf(packed)
            self.assertEqual(out, raw)
            self.assertEqual(warnings, [])

    def test_back_references_shrink_rtf(self):
        rtf = msgwriter.text_to_rtf("\n".join(f"Line {i}: bracket BR-{i:04d}, 6061-T6, qty {i * 5}" for i in range(300)))
        packed = msgwriter.compress_rtf(rtf)
        self.assertLess(len(packed), len(rtf) // 3)
        self.assertEqual(struct.unpack_from("<I", packed, 8)[0], 0x75465A4C)  # "LZFu"

    def test_crc_mismatch_is_only_a_warning(self):
        bad = self.VECTOR_1[:12] + b"\0\0\0\0" + self.VECTOR_1[16:]
        out, warnings = mailfile.decompress_rtf(bad)
        self.assertEqual(out, self.RAW_1)
        self.assertTrue(any("checksum" in w for w in warnings), warnings)

    def test_damaged_streams(self):
        out, warnings = mailfile.decompress_rtf(self.VECTOR_1[:25])
        self.assertTrue(out.startswith(b"{\\rtf1\\ansi\\"))
        self.assertTrue(warnings)
        self.assertEqual(mailfile.decompress_rtf(b"short")[0], None)
        unknown = self.VECTOR_1[:8] + b"ABCD" + self.VECTOR_1[12:]
        self.assertEqual(mailfile.decompress_rtf(unknown)[0], None)
        stored = struct.pack("<IIII", 12 + 5, 5, 0x414C454D, 0) + b"{\\rtf"
        self.assertEqual(mailfile.decompress_rtf(stored)[0], b"{\\rtf")
        huge = self.VECTOR_1[:4] + struct.pack("<I", 0xFFFFFFFF) + self.VECTOR_1[8:]
        self.assertEqual(mailfile.decompress_rtf(huge)[0], self.RAW_1)  # no allocation from the header
        rng = random.Random(9)
        for _ in range(200):
            junk = bytearray(self.VECTOR_1 + msgwriter.compress_rtf(b"{\\rtf1 " + b"xyz " * 400 + b"}"))
            for _ in range(rng.randint(1, 8)):
                junk[rng.randrange(16, len(junk))] = rng.randrange(256)
            mailfile.decompress_rtf(bytes(junk))  # never raises

    def test_output_cap(self):
        packed = msgwriter.compress_rtf(b"A" * 100_000)
        out, warnings = mailfile.decompress_rtf(packed, max_bytes=1000)
        self.assertLessEqual(len(out), 1000)
        self.assertTrue(warnings)


# --------------------------------------------------------------------------- #
class RtfTextTests(unittest.TestCase):
    def test_plain_rtf(self):
        rtf = (b"{\\rtf1\\ansi\\ansicpg1252\\deff0{\\fonttbl{\\f0\\fswiss Arial;}}{\\colortbl;\\red0\\green0\\blue0;}"
               b"{\\*\\generator Riched20 10.0;}{\\info{\\author Kit Marsh}}\\pard\\plain Qty\\tab 50\\par "
               b"Unit price \\'80 12\\par Caf\\u233?\\par A\\emdash B\\par {\\field{\\*\\fldinst HYPERLINK \"x\"}{\\fldrslt link}}}")
        text, was_html = mailfile.rtf_to_text(rtf)
        self.assertFalse(was_html)
        self.assertEqual(text, "Qty\t50\nUnit price \u20ac 12\nCaf\u00e9\nA-B\nlink")
        self.assertNotIn("Kit Marsh", text)
        self.assertNotIn("Riched", text)

    def test_code_pages(self):
        cyrillic = b"{\\rtf1\\ansi\\ansicpg1251 \\'cf\\'f0\\'e8\\'e2\\'e5\\'f2}"
        self.assertEqual(mailfile.rtf_to_text(cyrillic)[0], "\u041f\u0440\u0438\u0432\u0435\u0442")
        # The code page chosen by the font (\fcharset134, GBK), not by \ansicpg.
        chinese = (b"{\\rtf1\\ansi\\ansicpg1252{\\fonttbl{\\f0\\fnil\\fcharset134 \\'cb\\'ce\\'cc\\'e5;}{\\f1 Arial;}}"
                   b"\\f1 Part \\f0\\'d6\\'d0\\'ce\\'c4\\f1  ok}")
        self.assertEqual(mailfile.rtf_to_text(chinese)[0], "Part \u4e2d\u6587 ok")
        emoji = b"{\\rtf1 smile \\u-10179?\\u-8704?!}"
        self.assertEqual(mailfile.rtf_to_text(emoji)[0], "smile \U0001F600!")

    def test_binary_and_hostile_rtf(self):
        self.assertEqual(mailfile.rtf_to_text(b"{\\rtf1 A\\bin3 x}yB}")[0], "AB")
        self.assertEqual(mailfile.rtf_to_text(b"{\\rtf1 A\\bin999999 xyz")[0], "A")
        for junk in (b"", b"\\", b"{{{{{{", b"}}}}}", b"{\\rtf1 \\u99999999999?", b"{" * 5000 + b"deep" + b"}" * 5000,
                     b"{\\rtf1 \\'zz \\'4", bytes(range(256)) * 10):
            mailfile.rtf_to_text(junk)  # never raises

    def test_html_deencapsulation(self):
        rtf = (b"{\\rtf1\\ansi\\ansicpg1252\\fromhtml1 \\deff0{\\fonttbl{\\f0\\fswiss Arial;}}"
               b"{\\*\\htmltag19 <html>}{\\*\\htmltag34 <head><style>p {margin:0}</style></head>}{\\*\\htmltag50 <body>}"
               b"{\\*\\htmltag64 <p>}Ship to \\htmlrtf {\\b\\htmlrtf0 dock 4\\htmlrtf }\\htmlrtf0 "
               b"{\\*\\htmltag84 &amp;}\\htmlrtf &\\htmlrtf0  call{\\*\\htmltag72 </p>}\\htmlrtf \\par\\htmlrtf0 "
               b"{\\*\\mhtmltag0 <img src=\"cid:x\">}{\\*\\htmltag64 <p>}Rev \\'c9 ok{\\*\\htmltag72 </p>}"
               b"{\\*\\htmltag58 </body>}{\\*\\htmltag27 </html>}}")
        text, was_html = mailfile.rtf_to_text(rtf)
        self.assertTrue(was_html)
        self.assertEqual(text, "Ship to dock 4 & call\n\nRev \u00c9 ok")

    def test_writer_encapsulation_round_trip(self):
        page = ("<html><head><style>p.MsoNormal{margin:0}</style></head><body><p class=MsoNormal>Hello &amp; welcome,</p>"
                "<table><tr><td>Part</td><td>Qty</td></tr><tr><td>QA-7 {rev} \\B</td><td>50</td></tr></table>"
                "<p>Stra\u00dfe 12, \u00a950 \u4e2d</p></body></html>")
        text, was_html = mailfile.rtf_to_text(msgwriter.html_to_rtf(page))
        self.assertTrue(was_html)
        self.assertEqual(text, mailfile.html_to_text(page))
        self.assertIn("QA-7 {rev} \\B | 50", text)


# --------------------------------------------------------------------------- #
class HtmlTextTests(unittest.TestCase):
    def test_readable_text(self):
        page = ("<html><head><title>t</title><style>p {color:red}</style><script>alert(1)</script></head><body>"
                "<p>Please quote the parts below.</p><table><tr><th>Part</th><th>Qty</th></tr>"
                "<tr><td>QA-1</td><td>50</td></tr></table><ul><li>Anodize black</li><li>Deburr</li></ul>"
                "<div>Line one<br>Line&nbsp;two &lt;3</div><!-- a comment --></body></html>")
        text = mailfile.html_to_text(page)
        self.assertEqual(text, "Please quote the parts below.\n\nPart | Qty\nQA-1 | 50\n\n- Anodize black\n- Deburr\n\n"
                               "Line one\nLine two <3")

    def test_outlook_paragraphs_are_lines(self):
        page = ("<html><body><p class=MsoNormal>Hi Jordan,<o:p></o:p></p><p class=MsoNormal><o:p>&nbsp;</o:p></p>"
                "<p class=MsoNormal>Qty 50<o:p></o:p></p><!--[if gte mso 9]><xml><o:x/></xml><![endif]--></body></html>")
        self.assertEqual(mailfile.html_to_text(page), "Hi Jordan,\n\nQty 50")

    def test_unclosed_head_and_junk(self):
        self.assertEqual(mailfile.html_to_text("<html><head><title>x<body><p>Still here</p>"), "Still here")
        for junk in ("", "<", "<<<>>>", "<a " * 5000, "&#99999999;", "<![CDATA[x]]>", "<p" + "x" * 10000):
            mailfile.html_to_text(junk)  # never raises


# --------------------------------------------------------------------------- #
class MsgRoundTripTests(unittest.TestCase):
    def check_basics(self, em, body_format="text"):
        self.assertEqual(em["format"], "msg")
        self.assertEqual(em["subject"], "RFQ 4471 - bracket, 6061-T6")
        self.assertEqual(em["from_name"], "Avery Quill")
        self.assertEqual(em["from_email"], "avery.quill@example.com")
        self.assertEqual(em["to"], ["Jordan Vale <quotes@example.net>", "sales@example.net"])
        self.assertEqual(em["cc"], ["Kit Marsh <kit.marsh@example.org>"])
        self.assertEqual(em["date"], "2026-09-24T15:15:00+00:00")  # PR_CLIENT_SUBMIT_TIME, in UTC
        self.assertEqual(em["message_id"], "rfq-4471@example.com")
        self.assertEqual(em["body_format"], body_format)
        self.assertIn("Please quote 50 and 250 pcs of QA-41127 rev B.", em["body"])
        self.assertEqual([(a["name"], a["content_type"], a["data"], a["size"]) for a in em["attachments"]],
                         [("QA-41127_RevB.pdf", "application/pdf", PDF, len(PDF))])
        self.assertFalse(em["container_only"])

    def test_every_option(self):
        for unicode in (True, False):
            for body in ("text", "html", "rtf", "all"):
                for exchange in (False, True):
                    with self.subTest(unicode=unicode, body=body, exchange=exchange):
                        data = msgwriter.build_msg(rfq(), unicode=unicode, body=body, exchange_sender=exchange)
                        result = load_one("RFQ 4471.msg", data)
                        self.assertEqual(result["skipped"], [])
                        self.assertEqual(len(result["emails"]), 1)
                        em = result["emails"][0]
                        self.check_basics(em, {"text": "text", "all": "text", "html": "html", "rtf": "rtf"}[body])
                        self.assertEqual(em["source"], "RFQ 4471.msg")
                        self.assertEqual(em["warnings"], [])

    def test_bodies(self):
        page = "<html><body><p class=MsoNormal>Hello &amp; welcome,</p><p class=MsoNormal>Qty <b>50</b></p></body></html>"
        for body, fmt in (("html", "html"), ("rtf", "rtf")):
            em = load_one("x.msg", msgwriter.build_msg(rfq(html=page), body=body))["emails"][0]
            self.assertEqual((em["body"], em["body_format"]), ("Hello & welcome,\nQty 50", fmt))
        em = load_one("x.msg", msgwriter.build_msg(rfq(body="Line one\nLine two"), body="rtf"))["emails"][0]
        self.assertEqual((em["body"], em["body_format"]), ("Line one\nLine two", "rtf"))
        em = load_one("x.msg", msgwriter.build_msg(rfq(body="Plain wins", html=page), body="all"))["emails"][0]
        self.assertEqual((em["body"], em["body_format"]), ("Plain wins", "text"))
        em = load_one("x.msg", msgwriter.build_msg(rfq(body=""), body="text"))["emails"][0]
        self.assertEqual((em["body"], em["body_format"]), ("", "none"))

    def test_strings_unicode_and_cp1252(self):
        text = "Stra\u00dfe 12, Caf\u00e9, \u20ac 40, 5\u00b0 chamfer"
        em = load_one("x.msg", msgwriter.build_msg(rfq(subject=text, body=text), unicode=False))["emails"][0]
        self.assertEqual((em["subject"], em["body"]), (text, text))
        wide = "\u4e2d\u6587 \u0420\u0443\u0441 " + text
        em = load_one("x.msg", msgwriter.build_msg(rfq(subject=wide, body=wide, from_name="Zo\u00eb \u00c5berg")))["emails"][0]
        self.assertEqual((em["subject"], em["body"], em["from_name"]), (wide, wide, "Zo\u00eb \u00c5berg"))

    def test_exchange_sender_uses_smtp_property(self):
        data = msgwriter.build_msg(rfq(), exchange_sender=True)
        self.assertIn("/O=EXCHANGELABS/".encode("utf-16-le"), data)
        em = load_one("x.msg", data)["emails"][0]
        self.assertEqual(em["from_email"], "avery.quill@example.com")

    def test_exchange_sender_from_transport_headers(self):
        tree = msgwriter.message_tree(rfq(transport_headers="From: Avery Quill <avery.quill@example.com>\r\n"
                                                          "Date: Thu, 24 Sep 2026 09:40:00 -0400\r\n"),
                                      exchange_sender=True)
        del tree["__substg1.0_5D01001F"]  # no PR_SENDER_SMTP_ADDRESS left
        em = load_one("x.msg", msgwriter.write_cfb(tree))["emails"][0]
        self.assertEqual(em["from_email"], "avery.quill@example.com")
        self.assertEqual(em["date"], "2026-09-24T09:40:00-04:00")  # the transport headers' Date wins

    def test_exchange_sender_with_nothing_else(self):
        tree = msgwriter.message_tree(rfq(), exchange_sender=True)
        del tree["__substg1.0_5D01001F"]
        em = load_one("x.msg", msgwriter.write_cfb(tree))["emails"][0]
        self.assertEqual(em["from_email"], "")
        self.assertEqual(em["from_name"], "Avery Quill")
        self.assertIn("no sender address", em["warnings"])

    def test_delivery_time_when_never_submitted(self):
        tree = msgwriter.message_tree(rfq(date="2026-09-24T08:15:00-07:00"))
        props = bytearray(tree["__properties_version1.0"])
        for off in range(32, len(props), 16):
            if struct.unpack_from("<I", props, off)[0] == 0x00390040:   # PR_CLIENT_SUBMIT_TIME
                struct.pack_into("<I", props, off, 0x00380040)           # renamed to something unused
        tree["__properties_version1.0"] = bytes(props)
        em = load_one("x.msg", msgwriter.write_cfb(tree))["emails"][0]
        self.assertEqual(em["date"], "2026-09-24T15:15:00+00:00")

    def test_signature_logo_left_out_screenshot_kept(self):
        page = '<html><body><p>See below</p><img src="cid:shot@01"><img src="cid:logo@01"></body></html>'
        atts = [{"name": "image001.png", "content_type": "image/png", "data": LOGO, "inline": True, "content_id": "logo@01"},
                {"name": "image002.png", "content_type": "image/png", "data": SCREENSHOT, "inline": True, "content_id": "shot@01"},
                {"name": "QA-41127_RevB.pdf", "content_type": "application/pdf", "data": PDF}]
        em = load_one("x.msg", msgwriter.build_msg(rfq(html=page, attachments=atts), body="all"))["emails"][0]
        self.assertEqual([(a["name"], a["inline"]) for a in em["attachments"]],
                         [("image002.png", True), ("QA-41127_RevB.pdf", False)])
        self.assertIn("1 signature image left out", em["warnings"])

    def test_embedded_messages(self):
        outer = rfq(subject="FW: two RFQs", body="Forwarding two RFQs.", attachments=[], embedded=[inner_rfq(1), inner_rfq(2)])
        for body in ("text", "rtf"):
            result = load_one("FW two.msg", msgwriter.build_msg(outer, body=body))
            self.assertEqual([e["source"] for e in result["emails"]],
                             ["FW two.msg", "FW two.msg > RFQ 1 - spacer.msg", "FW two.msg > RFQ 2 - spacer.msg"])
            carrier, first, second = result["emails"]
            self.assertTrue(carrier["container_only"])
            self.assertEqual(carrier["attachments"], [])
            self.assertEqual((first["subject"], first["from_email"], first["date"]),
                             ("RFQ 1 - spacer", "rowan1@example.org", "2026-09-20T15:00:00+00:00"))
            self.assertEqual([a["name"] for a in second["attachments"]], ["SP-002.pdf"])
            self.assertIn("SP-002", second["body"])

    def test_embedded_depth_limit(self):
        em = inner_rfq(5)
        for n in (4, 3, 2, 1):
            em = inner_rfq(n) | {"attachments": [], "embedded": [em]}
        result = load_one("chain.msg", msgwriter.build_msg(em))
        self.assertEqual(len(result["emails"]), 4)  # depth 0 to 3
        self.assertEqual(len(result["skipped"]), 1)
        self.assertIn("nested too deeply", result["skipped"][0]["reason"])
        self.assertEqual(result["skipped"][0]["source"].count(" > "), 4)

    def test_attached_msg_and_eml_files_are_opened(self):
        inner_msg = msgwriter.build_msg(inner_rfq(7))
        inner_eml = eml_bytes(simple_eml(subject="RFQ 8 - collar"))
        atts = [{"name": "RFQ 7.msg", "content_type": "application/vnd.ms-outlook", "data": inner_msg},
                {"name": "RFQ 8.eml", "content_type": "message/rfc822", "data": inner_eml}]
        result = load_one("carrier.msg", msgwriter.build_msg(rfq(attachments=atts)))
        self.assertEqual([(e["source"], e["format"]) for e in result["emails"]],
                         [("carrier.msg", "msg"), ("carrier.msg > RFQ 7.msg", "msg"), ("carrier.msg > RFQ 8.eml", "eml")])
        self.assertTrue(result["emails"][0]["container_only"])

    def test_attachment_limits(self):
        many = [{"name": f"sheet {i}.pdf", "content_type": "application/pdf", "data": b"%PDF" + bytes([i])} for i in range(40)]
        em = load_one("x.msg", msgwriter.build_msg(rfq(attachments=many)))["emails"][0]
        self.assertEqual(len(em["attachments"]), 25)
        self.assertTrue(any(w.startswith("15 more attachments left out (the limit is 25)") for w in em["warnings"]), em["warnings"])
        small = Limits(max_attachment_bytes=1000, max_body_chars=20)
        em = load_one("x.msg", msgwriter.build_msg(rfq()), small)["emails"][0]
        self.assertEqual((em["attachments"][0]["data"], em["attachments"][0]["size"]), (None, len(PDF)))
        self.assertEqual(len(em["body"]), 20)
        self.assertTrue(any("larger than" in w for w in em["warnings"]))
        self.assertTrue(any("body cut" in w for w in em["warnings"]))

    def test_large_file_uses_difat_sectors(self):
        big = bytes(range(251)) * 30_000  # 7.5 MB: more than 109 FAT sectors
        data = msgwriter.build_msg(rfq(attachments=[{"name": "scan.tif", "content_type": "image/tiff", "data": big}]))
        self.assertGreater(struct.unpack_from("<I", data, 72)[0], 0)  # DIFAT sectors in the header
        em = load_one("big.msg", data, Limits(max_file_bytes=50 << 20))["emails"][0]
        self.assertEqual(em["attachments"][0]["data"], big)
        self.assertEqual(em["warnings"], [])

    def test_version_4_sectors(self):
        data = msgwriter.write_cfb(msgwriter.message_tree(rfq(), body="rtf"), sector_shift=12)
        self.assertEqual(struct.unpack_from("<H", data, 30)[0], 12)
        self.check_basics(load_one("v4.msg", data)["emails"][0], "rtf")

    def test_recipient_listed_twice_and_display_fallback(self):
        em = load_one("x.msg", msgwriter.build_msg(rfq(to=["quotes@example.net", "Quotes <QUOTES@example.net>"])))["emails"][0]
        self.assertEqual(em["to"], ["quotes@example.net"])
        tree = msgwriter.message_tree(rfq())
        for key in [k for k in tree if k.startswith("__recip_version1.0_")]:
            del tree[key]
        em = load_one("x.msg", msgwriter.write_cfb(tree))["emails"][0]
        self.assertEqual(em["to"], ["Jordan Vale", "sales@example.net"])  # PR_DISPLAY_TO names
        self.assertEqual(em["cc"], ["Kit Marsh"])

    def test_not_an_email_compound_file(self):
        result = load_one("report.msg", msgwriter.write_cfb({"WordDocument": b"x" * 100, "\x05SummaryInformation": b"y"}))
        self.assertEqual(result["emails"], [])
        self.assertIn("not an Outlook email", result["skipped"][0]["reason"])


class WriterStructureTests(unittest.TestCase):
    def test_header_and_nameid(self):
        data = msgwriter.build_msg(rfq())
        self.assertEqual(data[:8], mailfile.CFB_SIGNATURE)
        minor, major, order, shift, mini = struct.unpack_from("<HHHHH", data, 24)
        self.assertEqual((minor, major, order, shift, mini), (0x3E, 3, 0xFFFE, 9, 6))
        self.assertEqual(struct.unpack_from("<I", data, 56)[0], 4096)
        self.assertEqual(len(data) % 512, 0)
        names = {e[1]: e for e in dir_entries(data)}
        self.assertEqual(names["__nameid_version1.0"][2], 1)
        for stream in ("__substg1.0_00020102", "__substg1.0_00030102", "__substg1.0_00040102"):
            self.assertEqual(names[stream][7], 0)

    def test_property_stream_headers(self):
        outer = rfq(embedded=[inner_rfq(1)])
        tree = msgwriter.message_tree(outer)
        self.assertEqual((len(tree["__properties_version1.0"]) - 32) % 16, 0)
        embedded = tree["__attach_version1.0_#00000001"]["__substg1.0_3701000D"]
        self.assertEqual((len(embedded["__properties_version1.0"]) - 24) % 16, 0)
        self.assertNotIn("__nameid_version1.0", embedded)
        self.assertEqual((len(tree["__recip_version1.0_#00000000"]["__properties_version1.0"]) - 8) % 16, 0)

    def test_directory_trees_are_red_black(self):
        for n in list(range(1, 40)) + [63, 64, 65, 127, 128, 200]:
            tree = {f"s{i:03d}": bytes([i % 256]) * (i % 7) for i in range(n)}
            data = msgwriter.write_cfb(tree)
            entries = {e[0]: e for e in dir_entries(data)}
            colors = {}
            first_dir = struct.unpack_from("<I", data, 48)[0]
            for i in entries:
                colors[i] = data[512 + first_dir * 512 + i * 128 + 67]  # 0 red, 1 black
            root = entries[0][5]
            self.assertEqual(colors[root], 1)
            order = []

            def walk(i, blacks):
                if i == 0xFFFFFFFF:
                    return [blacks]
                e = entries[i]
                if colors[i] == 0:
                    for kid in (e[3], e[4]):
                        self.assertTrue(kid == 0xFFFFFFFF or colors[kid] == 1, f"red-red at n={n}")
                left = walk(e[3], blacks + colors[i])
                order.append(e[1])
                return left + walk(e[4], blacks + colors[i])

            heights = walk(root, 0)
            self.assertEqual(len(set(heights)), 1, f"black heights differ at n={n}")
            self.assertEqual(order, sorted(order, key=lambda s: (len(s), s.upper())))
            self.assertEqual(len(order), n)
            for i in range(n):
                self.assertEqual(mailfile._CFB(data).read(mailfile._CFB(data).children(0)[f"S{i:03d}"]),
                                 bytes([i % 256]) * (i % 7))


# --------------------------------------------------------------------------- #
class HostileMsgTests(unittest.TestCase):
    def setUp(self):
        self.data = msgwriter.build_msg(rfq(embedded=[inner_rfq(1)]), body="all")

    def test_truncated_everywhere(self):
        for cut in (0, 7, 8, 100, 511, 512, 513, 1024, 2048, len(self.data) // 3, len(self.data) // 2, len(self.data) - 600,
                    len(self.data) - 1):
            with self.subTest(cut=cut):
                load_one("cut.msg", self.data[:cut])

    def test_signature_then_garbage(self):
        rng = random.Random(1)
        garbage = mailfile.CFB_SIGNATURE + bytes(rng.randrange(256) for _ in range(5000))
        result = load_one("junk.msg", garbage)
        self.assertEqual(result["emails"], [])
        self.assertIn("not a readable Outlook .msg file", result["skipped"][0]["reason"])

    def test_bad_header_fields(self):
        for offset, fmt, value in ((30, "<H", 7), (28, "<H", 0x1234), (48, "<I", 0x7FFFFFFF), (48, "<I", 0xFFFFFFFE),
                                   (44, "<I", 0xFFFFFFFF), (72, "<I", 0xFFFFFFFF), (76, "<I", 0xFFFFFF00), (60, "<I", 12345)):
            with self.subTest(offset=offset, value=value):
                load_one("hdr.msg", patched(self.data, offset, fmt, value))
        root = dir_entries(self.data)[0]
        result = load_one("root.msg", patched(self.data, root[8] + 66, "<B", 1))
        self.assertIn("no root entry", result["skipped"][0]["reason"])

    def test_cyclic_fat_in_a_stream(self):
        e = entry_named(self.data, "__substg1.0_37010102", min_size=4096)
        count = (e[7] + 511) // 512
        looped = patched(self.data, fat_offset(e[6] + count - 1), "<I", e[6])
        em = load_one("loop.msg", looped)["emails"][0]
        self.assertEqual(em["subject"], "RFQ 4471 - bracket, 6061-T6")
        pdf = [a for a in em["attachments"] if a["name"] == "QA-41127_RevB.pdf"][0]
        self.assertIsNone(pdf["data"])
        self.assertIn("parts of this Outlook file are damaged and were not read", em["warnings"])

    def test_cyclic_directory_chain(self):
        first_dir = struct.unpack_from("<I", self.data, 48)[0]
        dir_sectors = (len(dir_entries(self.data)) * 128 + 511) // 512
        looped = patched(self.data, fat_offset(first_dir + dir_sectors - 1), "<I", first_dir)
        result = load_one("dirloop.msg", looped)
        self.assertEqual(result["emails"], [])
        self.assertIn("loops", result["skipped"][0]["reason"])

    def test_cyclic_directory_tree(self):
        entries = dir_entries(self.data)
        root_child = entries[0][5]
        # A sibling that points back at itself, and a storage whose child is the root.
        variants = [patched(self.data, entries[root_child][8] + 68, "<I", root_child),
                    patched(self.data, entries[root_child][8] + 72, "<I", root_child)]
        storage = entry_named(self.data, "__attach_version1.0_#00000000")
        variants.append(patched(self.data, storage[8] + 76, "<I", 0))
        variants.append(patched(self.data, storage[8] + 76, "<I", 99999))
        for data in variants:
            result = load_one("tree.msg", data)
            self.assertTrue(result["emails"] or result["skipped"])
            if result["emails"]:
                self.assertIn("parts of this Outlook file are damaged and were not read", result["emails"][0]["warnings"])

    def test_cyclic_mini_fat(self):
        e = entry_named(self.data, "__substg1.0_0037001F")  # the subject, in the mini stream
        minifat = struct.unpack_from("<I", self.data, 60)[0]
        looped = patched(self.data, 512 + minifat * 512 + 4 * e[6], "<I", e[6])
        em = load_one("miniloop.msg", looped)["emails"][0]
        self.assertIn("parts of this Outlook file are damaged and were not read", em["warnings"])

    def test_huge_declared_sizes(self):
        e = entry_named(self.data, "__substg1.0_37010102", min_size=4096)
        for size in (0xFFFFFFFF, len(self.data) + 1, 0x7FFFFFFFFFFFFFFF):
            em = load_one("size.msg", patched(self.data, e[8] + 120, "<Q", size))["emails"][0]
            self.assertIn("parts of this Outlook file are damaged and were not read", em["warnings"])
        small = entry_named(self.data, "__substg1.0_0037001F")
        em = load_one("size.msg", patched(self.data, small[8] + 120, "<Q", 4000))["emails"][0]  # claims more than its chain
        self.assertIn("parts of this Outlook file are damaged and were not read", em["warnings"])

    def test_sector_numbers_out_of_range(self):
        e = entry_named(self.data, "__substg1.0_37010102", min_size=4096)
        for start in (0x00FFFFFF, 0xFFFFFFFA, 0xFFFFFFFD, len(self.data)):
            em = load_one("sector.msg", patched(self.data, e[8] + 116, "<I", start))["emails"][0]
            self.assertIn("parts of this Outlook file are damaged and were not read", em["warnings"])
        em = load_one("fat.msg", patched(self.data, fat_offset(e[6]), "<I", 0x00FFFFF0))["emails"][0]
        self.assertIn("parts of this Outlook file are damaged and were not read", em["warnings"])

    def test_difat_loop(self):
        e = entry_named(self.data, "__substg1.0_37010102", min_size=4096)
        sector = e[6]
        data = patched(self.data, 512 + (sector + 1) * 512 - 4 - 512, "<I", sector)  # its last word points to itself
        data = patched(data, 68, "<I", sector)
        data = patched(data, 72, "<I", 1)
        result = load_one("difat.msg", data)
        self.assertEqual(result["emails"][0]["subject"], "RFQ 4471 - bracket, 6061-T6")

    def test_shared_sectors_cannot_multiply_reads(self):
        cfb = mailfile._CFB(self.data)
        idx = entry_named(self.data, "__substg1.0_37010102", min_size=4096)[0]
        cfb.budget = 1000
        self.assertIsNone(cfb.read(idx))
        self.assertTrue(cfb.damaged)

    def test_thousands_of_attachments(self):
        many = [{"name": f"p{i}.txt", "content_type": "text/plain", "data": b"x"} for i in range(1500)]
        start = time.time()
        em = load_one("many.msg", msgwriter.build_msg(rfq(attachments=many)))["emails"][0]
        self.assertLess(time.time() - start, 20)
        self.assertEqual(len(em["attachments"]), 25)


# --------------------------------------------------------------------------- #
class EmlTests(unittest.TestCase):
    def test_plain(self):
        result = load_one("RFQ 5102.eml", eml_bytes(simple_eml()))
        self.assertEqual(result["skipped"], [])
        em = result["emails"][0]
        self.assertEqual((em["source"], em["format"], em["subject"], em["from_name"], em["from_email"]),
                         ("RFQ 5102.eml", "eml", "RFQ 5102 - hinge plate", "Avery Quill", "avery.quill@example.com"))
        self.assertEqual(em["to"], ["Jordan Vale <quotes@example.net>"])
        self.assertEqual(em["date"], "2026-09-24T08:15:00-07:00")
        self.assertEqual(em["message_id"], "rfq-5102@example.com")
        self.assertEqual((em["body"], em["body_format"]), ("Please quote 20 pcs of HP-5102.", "text"))
        self.assertEqual((em["attachments"], em["container_only"], em["warnings"]), ([], False, []))

    def test_html_only(self):
        msg = simple_eml()
        msg.set_content("<html><head><style>p {x:y}</style></head><body><p>Please quote</p><table><tr><td>HP-5102</td>"
                        "<td>20</td></tr></table><script>alert(1)</script></body></html>", subtype="html")
        em = load_one("h.eml", eml_bytes(msg))["emails"][0]
        self.assertEqual((em["body"], em["body_format"]), ("Please quote\n\nHP-5102 | 20", "html"))

    def test_multipart_with_attachments(self):
        msg = simple_eml(body="Drawing and model attached.")
        msg.add_alternative("<p>Drawing and model <b>attached</b>.</p>", subtype="html")
        msg.add_attachment(PDF, "application", "pdf", filename="HP-5102_RevA.pdf")
        msg.add_attachment(b"ISO-10303-21;\nHEADER;\n", "application", "step", filename="HP-5102.step")
        em = load_one("m.eml", eml_bytes(msg))["emails"][0]
        self.assertEqual((em["body"], em["body_format"]), ("Drawing and model attached.", "text"))
        self.assertEqual([(a["name"], a["content_type"], a["size"], a["inline"]) for a in em["attachments"]],
                         [("HP-5102_RevA.pdf", "application/pdf", len(PDF), False),
                          ("HP-5102.step", "application/step", 22, False)])
        self.assertEqual(em["attachments"][0]["data"], PDF)

    def test_rfc2047_and_raw_8bit_headers(self):
        raw = (b"From: =?utf-8?Q?Zo=C3=AB_=C3=85berg?= <zoe@example.org>\r\n"
               b"To: =?utf-8?Q?M=C3=BCller=2C_Kai?= <Kai@Example.org>, \"Vale, Jordan\" <quotes@example.net>\r\n"
               b"Subject: =?UTF-8?B?QW5mcmFnZTogRnLDpHN0ZWlsZQ==?= =?iso-8859-1?Q?_f=FCr_Halter?=\r\n"
               b"Date: Thu, 24 Sep 2026 08:15:00 +0200\r\n\r\nBody\r\n")
        em = load_one("h.eml", raw)["emails"][0]
        self.assertEqual(em["subject"], "Anfrage: Fr\u00e4steile f\u00fcr Halter")
        self.assertEqual((em["from_name"], em["from_email"]), ("Zo\u00eb \u00c5berg", "zoe@example.org"))
        self.assertEqual(em["to"], ["M\u00fcller, Kai <kai@example.org>", "Vale, Jordan <quotes@example.net>"])
        raw8 = b"From: a@example.com\r\nSubject: Pi\xc3\xa8ces usin\xc3\xa9es / Pi\xe8ce\r\n\r\nx\r\n"
        self.assertEqual(load_one("h.eml", raw8)["emails"][0]["subject"], "Pi\u00e8ces usin\u00e9es / Pi\u00e8ce")

    def test_transfer_encodings_and_charsets(self):
        head = b"From: a@example.com\r\nSubject: x\r\nMIME-Version: 1.0\r\n"
        qp = head + (b"Content-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: quoted-printable\r\n\r\n"
                     b"Qty =3D 50 pcs, finish: anodize=\r\nd black, Stra=C3=9Fe\r\n")
        self.assertEqual(load_one("q.eml", qp)["emails"][0]["body"], "Qty = 50 pcs, finish: anodized black, Stra\u00dfe")
        b64 = head + (b"Content-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: base64\r\n\r\n"
                      + base64.encodebytes("Menge: 500 St\u00fcck".encode("utf-8")))
        self.assertEqual(load_one("b.eml", b64)["emails"][0]["body"], "Menge: 500 St\u00fcck")
        latin = head + b"Content-Type: text/plain; charset=iso-8859-1\r\n\r\nStra\xdfe 12, \x93quoted\x94\r\n"
        self.assertEqual(load_one("l.eml", latin)["emails"][0]["body"], "Stra\u00dfe 12, \u201cquoted\u201d")
        bogus = head + b"Content-Type: text/plain; charset=x-bogus-8\r\n\r\nCaf\xe9 ol\xe9\r\n"
        em = load_one("x.eml", bogus)["emails"][0]
        self.assertEqual(em["body"], "Caf\u00e9 ol\u00e9")
        self.assertTrue(any("unknown character set (x-bogus-8)" in w for w in em["warnings"]), em["warnings"])
        subject = b"From: a@example.com\r\nSubject: =?x-bogus?Q?Caf=E9?=\r\n\r\nx\r\n"
        em = load_one("s.eml", subject)["emails"][0]
        self.assertEqual(em["subject"], "Caf\u00e9")
        self.assertTrue(any("subject used an unknown character set" in w for w in em["warnings"]))
        wrong = head + b"Content-Type: text/plain; charset=us-ascii\r\n\r\nCaf\xc3\xa9 and na\xefve\r\n"
        self.assertEqual(load_one("w.eml", wrong)["emails"][0]["body"], "Caf\u00e9 and na\u00efve")

    def test_forwarded_bundle(self):
        outer = simple_eml(subject="FW: three RFQs", body="Three RFQs attached.")
        for n in (1, 2, 3):
            inner = simple_eml(subject=f"RFQ {n} - spacer", body=f"Please quote SP-{n:03d}.",
                               sender=f"Rowan Pike <rowan{n}@example.org>")
            outer.add_attachment(inner, filename=f"RFQ {n}.eml")
        result = load_one("bundle.eml", eml_bytes(outer))
        self.assertEqual([e["source"] for e in result["emails"]],
                         ["bundle.eml", "bundle.eml > RFQ 1.eml", "bundle.eml > RFQ 2.eml", "bundle.eml > RFQ 3.eml"])
        self.assertTrue(result["emails"][0]["container_only"])
        self.assertEqual(result["emails"][0]["attachments"], [])
        self.assertEqual([e["from_email"] for e in result["emails"][1:]],
                         ["rowan1@example.org", "rowan2@example.org", "rowan3@example.org"])
        self.assertEqual(result["emails"][2]["body"], "Please quote SP-002.")

    def test_base64_encoded_attached_email_and_attached_msg(self):
        inner = eml_bytes(simple_eml(subject="RFQ 11 - pin"))
        raw = (b"From: a@example.com\r\nSubject: FW\r\nMIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=B\r\n\r\n"
               b"--B\r\nContent-Type: text/plain\r\n\r\nsee attached\r\n"
               b"--B\r\nContent-Type: message/rfc822\r\nContent-Transfer-Encoding: base64\r\n"
               b"Content-Disposition: attachment; filename=\"RFQ 11.eml\"\r\n\r\n" + base64.encodebytes(inner) +
               b"--B\r\nContent-Type: application/vnd.ms-outlook; name=\"RFQ 12.msg\"\r\nContent-Transfer-Encoding: base64\r\n"
               b"Content-Disposition: attachment; filename=\"RFQ 12.msg\"\r\n\r\n" +
               base64.encodebytes(msgwriter.build_msg(inner_rfq(12))) + b"--B--\r\n")
        result = load_one("fw.eml", raw)
        self.assertEqual([(e["source"], e["format"], e["subject"]) for e in result["emails"]],
                         [("fw.eml", "eml", "FW"), ("fw.eml > RFQ 11.eml", "eml", "RFQ 11 - pin"),
                          ("fw.eml > RFQ 12.msg", "msg", "RFQ 12 - spacer")])

    def test_winmail_dat(self):
        msg = simple_eml()
        msg.add_attachment(b"\x78\x9f\x3e\x22" + b"\0" * 100, "application", "ms-tnef", filename="winmail.dat")
        em = load_one("t.eml", eml_bytes(msg))["emails"][0]
        self.assertIn("winmail.dat (Outlook rich text) is not read", em["warnings"])
        self.assertEqual([a["name"] for a in em["attachments"]], ["winmail.dat"])

    def test_inline_logo_vs_pasted_screenshot(self):
        msg = simple_eml(body="See the screenshot.")
        msg.add_alternative('<p>See the screenshot.</p><img src="cid:shot1"><p>--</p><img src="cid:logo1">', subtype="html")
        html_part = msg.get_payload()[1]
        html_part.add_related(SCREENSHOT, "image", "png", cid="<shot1>", filename="image001.png")
        html_part.add_related(LOGO, "image", "png", cid="<logo1>", filename="image002.png")
        msg.add_attachment(LOGO, "image", "png", filename="tiny-but-attached.png")
        em = load_one("s.eml", eml_bytes(msg))["emails"][0]
        self.assertEqual([(a["name"], a["inline"]) for a in em["attachments"]],
                         [("image001.png", True), ("tiny-but-attached.png", False)])
        self.assertIn("1 signature image left out", em["warnings"])

    def test_missing_date_and_from(self):
        raw = b"To: quotes@example.net\r\nSubject: no sender here\r\n\r\nPlease quote.\r\n"
        em = load_one("n.eml", raw)["emails"][0]
        self.assertEqual((em["from_name"], em["from_email"], em["date"], em["message_id"]), ("", "", None, None))
        self.assertIn("no sender address", em["warnings"])
        self.assertIn("no date", em["warnings"])
        bad = b"From: a@example.com\r\nDate: someday soon\r\nSubject: x\r\n\r\nx\r\n"
        self.assertIn("the date could not be read", load_one("d.eml", bad)["emails"][0]["warnings"])
        utc = b"From: a@example.com\r\nDate: Thu, 24 Sep 2026 08:15:00 -0000\r\nSubject: x\r\n\r\nx\r\n"
        self.assertEqual(load_one("u.eml", utc)["emails"][0]["date"], "2026-09-24T08:15:00+00:00")

    def test_scrubbed_attachments_header(self):
        msg = simple_eml(X_Scrubbed_Attachments="QA-1_RevA.pdf; model 7.step ;  ")
        msg.add_attachment(PDF, "application", "pdf", filename="kept.pdf")
        em = load_one("s.eml", eml_bytes(msg))["emails"][0]
        self.assertEqual([(a["name"], a["content_type"], a["data"], a["size"]) for a in em["attachments"]],
                         [("kept.pdf", "application/pdf", PDF, len(PDF)), ("QA-1_RevA.pdf", "", None, 0),
                          ("model 7.step", "", None, 0)])
        self.assertFalse(em["container_only"])

    def test_names_are_sanitized(self):
        msg = simple_eml()
        for name in ("../../etc/passwd", "C:\\Users\\kit\\drawing.pdf", "evil\u202efdp.exe", "a\x01b\x7f.txt",
                     "x" * 300 + ".pdf", "..", "trailing dots..."):
            msg.add_attachment(b"data", "application", "octet-stream", filename=name)
        msg.add_attachment(PDF, "application", "pdf")
        names = [a["name"] for a in load_one("n.eml", eml_bytes(msg))["emails"][0]["attachments"]]
        self.assertEqual(names[:4], ["passwd", "drawing.pdf", "evilfdp.exe", "ab.txt"])
        self.assertEqual((len(names[4]), names[4][-4:]), (120, ".pdf"))
        self.assertEqual(names[5:], ["attachment", "trailing dots", "attachment.pdf"])

    def test_deep_nesting(self):
        msg = simple_eml(subject="level 6")
        for level in (5, 4, 3, 2, 1, 0):
            outer = simple_eml(subject=f"level {level}")
            outer.add_attachment(msg, filename=f"level {level + 1}.eml")
            msg = outer
        result = load_one("deep.eml", eml_bytes(msg))
        self.assertEqual([e["subject"] for e in result["emails"]], ["level 0", "level 1", "level 2", "level 3"])
        self.assertEqual(len(result["skipped"]), 1)
        self.assertIn("nested too deeply", result["skipped"][0]["reason"])
        absurd = b"From: a@example.com\r\nSubject: x\r\n" + b"".join(
            b"Content-Type: multipart/mixed; boundary=b%d\r\n\r\n--b%d\r\n" % (i, i) for i in range(3000))
        load_one("absurd.eml", absurd)

    def test_email_limit(self):
        outer = simple_eml(subject="FW: many")
        for n in range(5):
            outer.add_attachment(simple_eml(subject=f"RFQ {n}"), filename=f"RFQ {n}.eml")
        result = load_one("many.eml", eml_bytes(outer), Limits(max_emails=3))
        self.assertEqual(len(result["emails"]), 3)
        self.assertEqual(len(result["skipped"]), 1)
        self.assertIn("more than 3 emails", result["skipped"][0]["reason"])

    def test_thousands_of_attachments(self):
        parts = [b"--B\r\nContent-Type: text/plain\r\n\r\nPlease quote the lot.\r\n"]
        for i in range(3000):
            parts.append(b"--B\r\nContent-Type: application/octet-stream\r\nContent-Disposition: attachment; "
                         b"filename=\"part %d.bin\"\r\nContent-Transfer-Encoding: base64\r\n\r\nAAEC\r\n" % i)
        raw = (b"From: a@example.com\r\nSubject: lots\r\nMIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=B\r\n\r\n"
               + b"".join(parts) + b"--B--\r\n")
        start = time.time()
        em = load_one("lots.eml", raw)["emails"][0]
        self.assertLess(time.time() - start, 30)
        self.assertEqual(len(em["attachments"]), 25)
        self.assertEqual(em["attachments"][0]["data"], b"\x00\x01\x02")
        self.assertTrue(any(w.startswith("2975 more attachments left out") for w in em["warnings"]), em["warnings"][-1])

    def test_attachment_and_body_limits(self):
        msg = simple_eml(body="x" * 1000)
        msg.add_attachment(PDF, "application", "pdf", filename="big.pdf")
        em = load_one("l.eml", eml_bytes(msg), Limits(max_attachment_bytes=1000, max_body_chars=100))["emails"][0]
        self.assertEqual(len(em["body"]), 100)
        self.assertEqual((em["attachments"][0]["data"], em["attachments"][0]["size"]), (None, len(PDF)))

    def test_signed_email(self):
        raw = (b"From: a@example.com\r\nSubject: signed RFQ\r\nMIME-Version: 1.0\r\n"
               b"Content-Type: multipart/signed; protocol=\"application/pkcs7-signature\"; micalg=sha-256; boundary=S\r\n\r\n"
               b"--S\r\nContent-Type: multipart/mixed; boundary=M\r\n\r\n--M\r\nContent-Type: text/plain\r\n\r\nQuote please.\r\n"
               b"--M\r\nContent-Type: application/pdf; name=\"D-1.pdf\"\r\nContent-Disposition: attachment; filename=\"D-1.pdf\"\r\n"
               b"Content-Transfer-Encoding: base64\r\n\r\n" + base64.encodebytes(PDF) + b"--M--\r\n"
               b"--S\r\nContent-Type: application/pkcs7-signature; name=smime.p7s\r\nContent-Transfer-Encoding: base64\r\n\r\n"
               b"MIIB\r\n--S--\r\n")
        em = load_one("signed.eml", raw)["emails"][0]
        self.assertEqual((em["body"], [a["name"] for a in em["attachments"]]), ("Quote please.", ["D-1.pdf"]))
        self.assertIn("a signed email; the signature was not checked", em["warnings"])


# --------------------------------------------------------------------------- #
class ZipTests(unittest.TestCase):
    def email(self, n):
        return eml_bytes(simple_eml(subject=f"RFQ {n} - flange"))

    def test_folders_and_junk(self):
        data = zip_of([("Inbox/", b""), ("Inbox/RFQ 1.eml", self.email(1)), ("Inbox/Sub/RFQ 2.msg", msgwriter.build_msg(rfq())),
                       ("__MACOSX/Inbox/._RFQ 1.eml", b"\0\5\x16\7"), ("Inbox/._RFQ 1.eml", b"\0\5\x16\7"), (".DS_Store", b"\0"),
                       ("notes.txt", b"call Kit"), ("maildir/cur/1727180100.M1P2", self.email(3)), ("readme", b"just text")])
        result = load_one("orders.zip", data)
        self.assertEqual([e["source"] for e in result["emails"]],
                         ["orders.zip/Inbox/RFQ 1.eml", "orders.zip/Inbox/Sub/RFQ 2.msg", "orders.zip/maildir/cur/1727180100.M1P2"])
        self.assertEqual(result["skipped"], [{"source": "orders.zip/notes.txt", "reason": "not an email file"},
                                             {"source": "orders.zip/readme", "reason": "not an email file"}])

    def test_password_protected_entry(self):
        data = bytearray(zip_of([("secret.eml", self.email(1)), ("open.eml", self.email(2))], zipfile.ZIP_STORED))
        local = data.find(b"PK\x03\x04")
        central = data.find(b"PK\x01\x02")
        struct.pack_into("<H", data, local + 6, 1)
        struct.pack_into("<H", data, central + 8, 1)
        result = load_one("p.zip", bytes(data))
        self.assertEqual([e["source"] for e in result["emails"]], ["p.zip/open.eml"])
        self.assertEqual(result["skipped"], [{"source": "p.zip/secret.eml", "reason": "password protected"}])

    def test_zip_inside_zip_and_other_methods(self):
        inner = zip_of([("RFQ 1.eml", self.email(1))])
        result = load_one("outer.zip", zip_of([("inner.zip", inner)]))
        self.assertEqual(result["emails"], [])
        self.assertEqual(result["skipped"], [{"source": "outer.zip/inner.zip", "reason": "a zip inside a zip is not opened"}])
        try:
            bz = zip_of([("RFQ 1.eml", self.email(1))], zipfile.ZIP_BZIP2)
        except RuntimeError:
            self.skipTest("no bz2 module")
        result = load_one("bz.zip", bz)
        self.assertIn("compressed with a method", result["skipped"][0]["reason"])

    def test_bomb_and_total_limit(self):
        bomb = b"From: a@example.com\r\nSubject: bomb\r\n\r\n" + b"A" * 5_000_000
        limits = Limits(max_file_bytes=1_000_000, max_zip_total_bytes=2_000_000)
        data = zip_of([("bomb.eml", bomb), ("ok.eml", self.email(1))])
        self.assertLess(len(data), 50_000)
        result = load_one("bomb.zip", data, limits)
        self.assertEqual([e["source"] for e in result["emails"]], ["bomb.zip/ok.eml"])
        self.assertEqual(result["skipped"], [{"source": "bomb.zip/bomb.eml", "reason": "larger than 0.953674 MB"}])
        chunk = b"From: a@example.com\r\nSubject: part\r\n\r\n" + b"B" * 800_000
        data = zip_of([(f"part {i}.eml", chunk) for i in range(4)])
        result = load_one("total.zip", data, limits)
        self.assertEqual(len(result["emails"]), 2)
        self.assertIn("holds more than", result["skipped"][0]["reason"])
        self.assertEqual(result["skipped"][0]["source"], "total.zip/part 2.eml")

    def test_lying_headers(self):
        data = bytearray(zip_of([("RFQ 1.eml", self.email(1) + b"x" * 200_000)]))
        central = data.find(b"PK\x01\x02")
        small = bytearray(data)
        struct.pack_into("<I", small, central + 24, 100)          # claims 100 bytes uncompressed
        result = load_one("liar.zip", bytes(small))
        self.assertEqual(result["emails"], [])
        self.assertIn("could not be unzipped", result["skipped"][0]["reason"])
        big = bytearray(data)
        struct.pack_into("<I", big, central + 24, 0xFFFFFFF0)     # claims 4 GB: the real size is what counts
        result = load_one("liar.zip", bytes(big), Limits(max_zip_total_bytes=1_000_000))
        self.assertEqual(len(result["emails"]), 1)

    def test_traversal_names(self):
        data = zip_of([("../../evil.eml", self.email(1)), ("/abs/root.eml", self.email(2)),
                       ("C:\\Users\\kit\\win.eml", self.email(3)), ("a/./b/../z.eml", self.email(4)),
                       ("bad\x01name.eml", self.email(5))])
        result = load_one("t.zip", data)
        self.assertEqual(sorted(e["source"] for e in result["emails"]),
                         ["t.zip/Users/kit/win.eml", "t.zip/a/b/z.eml", "t.zip/abs/root.eml", "t.zip/badname.eml", "t.zip/evil.eml"])

    def test_entry_limit_and_empty(self):
        data = zip_of([(f"RFQ {i}.eml", self.email(i)) for i in range(8)])
        result = load_one("e.zip", data, Limits(max_zip_entries=5))
        self.assertEqual(len(result["emails"]), 5)
        self.assertEqual(result["skipped"], [{"source": "e.zip", "reason": "the zip has 8 entries; only the first 5 were read"}])
        self.assertEqual(load_one("empty.zip", zip_of([])), {"emails": [], "skipped": [{"source": "empty.zip", "reason": "no emails found"}]})
        many = zip_of([(f"x{i}", b"") for i in range(25_000)], zipfile.ZIP_STORED)
        result = load_one("many.zip", many, Limits(max_zip_entries=1000))
        self.assertIn("far more than", result["skipped"][0]["reason"])

    def test_damaged_zips(self):
        data = bytearray(zip_of([("RFQ 1.eml", self.email(1))], zipfile.ZIP_STORED))
        data[60] ^= 0xFF  # inside the stored email: the CRC no longer matches
        result = load_one("crc.zip", bytes(data))
        self.assertIn("could not be unzipped", result["skipped"][0]["reason"])
        result = load_one("junk.zip", b"PK\x03\x04" + b"\x00" * 200)
        self.assertIn("not a readable zip file", result["skipped"][0]["reason"])

    def test_nesting_through_a_zip(self):
        msg = simple_eml(subject="level 4")
        for level in (3, 2, 1):
            outer = simple_eml(subject=f"level {level}")
            outer.add_attachment(msg, filename=f"level {level + 1}.eml")
            msg = outer
        result = load_one("z.zip", zip_of([("chain.eml", eml_bytes(msg))]))
        self.assertEqual([e["subject"] for e in result["emails"]], ["level 1", "level 2", "level 3"])
        self.assertIn("nested too deeply", result["skipped"][0]["reason"])


# --------------------------------------------------------------------------- #
class LoadContractTests(unittest.TestCase):
    def test_inputs_that_are_not_bytes(self):
        text = "From: a@example.com\nSubject: typed in\n\nhello"
        self.assertEqual(load_one("t.eml", text)["emails"][0]["subject"], "typed in")
        self.assertEqual(load_one("t.eml", bytearray(text.encode()))["emails"][0]["body"], "hello")
        self.assertEqual(load_one("t.eml", memoryview(text.encode()))["emails"][0]["body"], "hello")
        self.assertEqual(load_one("t.eml", None)["skipped"], [{"source": "t.eml", "reason": "empty file"}])

    def test_sources_and_reasons(self):
        self.assertEqual(load_one("C:\\Users\\kit\\Desktop\\RFQ 1.eml", eml_bytes(simple_eml()))["emails"][0]["source"], "RFQ 1.eml")
        self.assertEqual(load_one("", eml_bytes(simple_eml()))["emails"][0]["source"], "upload")
        cases = {"drawing.pdf": (b"%PDF-1.4\n", "not an email file (.eml, .msg, or .zip)"),
                 "x.msg": (b"From nowhere", "not an Outlook .msg file (no Compound File signature)"),
                 "x.eml": (b"%PDF-1.4\n", "does not look like an email (no mail headers at the top)"),
                 "x.zip": (b"hello", "not a zip file"), "y.eml": (b"", "empty file")}
        for name, (data, reason) in cases.items():
            self.assertEqual(load_one(name, data)["skipped"], [{"source": name, "reason": reason}])
        result = load_one("big.eml", eml_bytes(simple_eml()), Limits(max_file_bytes=100))
        self.assertIn("larger than", result["skipped"][0]["reason"])

    def test_limits_are_frozen(self):
        with self.assertRaises(Exception):
            mailfile.LIMITS.max_emails = 1  # type: ignore[misc]
        self.assertEqual(mailfile.LIMITS.max_file_bytes, 40 * 1024 * 1024)


class FuzzTests(unittest.TestCase):
    """Random damage to real-looking files: load() must return its shape and never raise."""

    def mutate(self, seed, data, count):
        rng = random.Random(seed)
        out = []
        for _ in range(count):
            b = bytearray(data)
            action = rng.random()
            if action < 0.6:
                for _ in range(rng.randint(1, 12)):
                    b[rng.randrange(len(b))] = rng.randrange(256)
            elif action < 0.8:
                b = b[:rng.randrange(len(b))]
            else:
                pos = rng.randrange(len(b))
                b[pos:pos] = bytes(rng.randrange(256) for _ in range(rng.randint(1, 64)))
            out.append(bytes(b))
        return out

    def test_fuzz(self):
        msg = msgwriter.build_msg(rfq(embedded=[inner_rfq(1)], html="<p>Hi &amp; bye</p>"), body="rtf")
        fw = simple_eml(subject="FW")
        fw.add_attachment(simple_eml(subject="RFQ"), filename="RFQ.eml")
        fw.add_attachment(PDF, "application", "pdf", filename="d.pdf")
        eml = eml_bytes(fw)
        zipped = zip_of([("a/RFQ.eml", eml), ("b/RFQ.msg", msg)])
        start = time.time()
        for name, data in (("f.msg", msg), ("f.eml", eml), ("f.zip", zipped)):
            for i, damaged in enumerate(self.mutate(len(name) * 7919, data, 150)):
                with self.subTest(file=name, case=i):
                    load_one(name, damaged)
        self.assertLess(time.time() - start, 120)


class RepoRulesTests(unittest.TestCase):
    def test_no_long_dashes_in_these_files(self):
        for path in (ROOT / "mailfile.py", ROOT / "tools" / "msgwriter.py", Path(__file__)):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("\u2014", text, path.name)
            self.assertNotIn("\u2013", text, path.name)


if __name__ == "__main__":
    unittest.main()
