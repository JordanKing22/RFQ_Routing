"""
Tests for rfq_details.py: the RFQ details extractor and the consolidated file. Standard library only.

    python -m unittest tests.test_rfq_details -v
    python tests/test_rfq_details.py

The attachment text comes from data/rfq_beta/ocr_cache.json, so no OCR runs when the committed cache
is there. Without it the scans are read live with tesseract (about half a minute). Without tesseract
either, the tests that need text from a scan are skipped and the rest still run. The answer key is
tests/rfq_beta_fields_truth.json; tests/rfq_beta_truth.json (what the generator printed on every
file) says which files are scans.
"""

from __future__ import annotations

import csv
import difflib
import io
import json
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rfq_details  # noqa: E402

INBOX = json.loads((ROOT / "data" / "rfq_beta" / "emails.json").read_text(encoding="utf-8"))["emails"]
EMAILS = {e["id"]: e for e in INBOX}
SHOP = json.loads((ROOT / "shop_config.json").read_text(encoding="utf-8"))
TRUTH = json.loads((ROOT / "tests" / "rfq_beta_fields_truth.json").read_text(encoding="utf-8"))
FILES = json.loads((ROOT / "tests" / "rfq_beta_truth.json").read_text(encoding="utf-8"))["files"]
TODAY = rfq_details.SAMPLE_INBOX_DATE
SCANS = {Path(rel).name for rel, f in FILES.items() if not f["copyable"]}
TEXT_PDFS = {Path(rel).name for rel, f in FILES.items() if f["copyable"] and rel.endswith(".pdf")}
CSV_OUT = ROOT / "data" / "rfq_beta" / "rfq_details.csv"
JSON_OUT = ROOT / "data" / "rfq_beta" / "rfq_details.json"

_STATE: dict = {}
_READING = rfq_details.REGION_READING


def beta() -> dict:
    """Texts, records, and the graded answer key for the beta inbox, computed once."""
    if not _STATE:
        texts = rfq_details.load_texts(INBOX, rfq_details.BETA_CACHE, allow_ocr=True)
        by_name = {name: res for per in texts.values() for name, res in per.items()}
        records = rfq_details.extract_all(INBOX, texts, SHOP, today=TODAY)
        _STATE.update(texts=texts, records=records, by_id={r["email_id"]: r for r in records},
                      graded=rfq_details.grade(records, TRUTH)["rows"],
                      scans_read=all(by_name.get(n, {}).get("method") == "ocr" for n in SCANS),
                      text_read=all(by_name.get(n, {}).get("method") == "text-layer" for n in TEXT_PDFS))
    return _STATE


def needs_scans(test):
    def run(self, *a, **kw):
        if not beta()["scans_read"]:
            self.skipTest("no OCR cache and no tesseract: the scans cannot be read")
        return test(self, *a, **kw)
    run.__name__, run.__doc__ = test.__name__, test.__doc__
    return run


def ocr_read(eid: str, name: str, text: str) -> bool:
    """Whether OCR got these words off the file at all (up to O/0 and I/1 slips). A value OCR never
    read is an OCR miss, not an extraction bug, so the tests below skip it and say so."""
    got = beta()["texts"].get(eid, {}).get(name) or {}
    words = {rfq_details.fold(w) for w in re.split(r"[\s,]+", got.get("text") or "")}

    def seen(w: str) -> bool:  # '6061-16511' still counts as a reading of '6061-T6511'
        return w in words or any(difflib.SequenceMatcher(None, w, x).ratio() >= 0.75 for x in words)
    return all(seen(rfq_details.fold(w)) for w in re.split(r"[\s,]+", text) if len(rfq_details.fold(w)) >= 2)


def value(field: dict):
    return (field or {}).get("value")


def squash(text) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


# --------------------------------------------------------------------------- #
# Parts that need no attachment text
# --------------------------------------------------------------------------- #
class HeuristicTests(unittest.TestCase):
    def test_is_rfq_without_a_decision(self):
        """E10 (PO status), E13 (vendor promotion), and E17 (capability question) are not RFQs."""
        not_rfq = {e["id"] for e in INBOX if not rfq_details.is_rfq(e, None)}
        self.assertEqual(not_rfq, {"E10", "E13", "E17"})
        self.assertEqual(not_rfq, set(TRUTH["not_rfq"]))
        self.assertEqual({e["id"] for e in INBOX} - not_rfq, set(TRUTH["rfqs"]))

    def test_is_rfq_on_the_main_sample_inbox(self):
        """The same heuristic on data/sample_emails.json: orders and vendor mail are not RFQs,
        every email the router sends to an estimating lane is."""
        sample = json.loads((ROOT / "data" / "sample_emails.json").read_text(encoding="utf-8"))["emails"]
        for e in sample:
            lane = (e.get("expected") or {}).get("lane")
            if lane in ("orders", "filtered"):
                with self.subTest(email=e["id"], lane=lane):
                    self.assertFalse(rfq_details.is_rfq(e, None))
            elif lane in ("milling_3axis", "milling_5axis", "turning"):
                with self.subTest(email=e["id"], lane=lane):
                    self.assertTrue(rfq_details.is_rfq(e, None))

    def test_main_sample_inbox_part_lines(self):
        """On data/sample_emails.json, with each drawing and form rendered as text the way the server
        does, every part line is a part some file or the email really quotes, and each drawing's
        title block reaches its line."""
        import attachments as att_mod
        sample = json.loads((ROOT / "data" / "sample_emails.json").read_text(encoding="utf-8"))["emails"]
        for e in sample:
            if not rfq_details.is_rfq(e, None):
                continue
            specs = [att_mod.normalize(x) for x in e.get("attachments") or []]
            texts = {a["name"]: {"method": "step-header" if a["kind"] == "model" else "text-layer",
                                 "text": att_mod.spec_text(a)} for a in specs if a.get("kind") in att_mod.SPEC_KINDS}
            rec = rfq_details.extract(e, texts, SHOP, today=TODAY)
            named = {rfq_details.fold(a["part_number"]) for a in specs if a.get("part_number")}
            named |= {rfq_details.fold(ln["part_number"]) for a in specs for ln in a.get("lines") or []
                      if ln.get("part_number")}
            with self.subTest(email=e["id"]):
                for ln in rec["lines"]:
                    pn = value(ln["part_number"])
                    if pn and named:
                        self.assertTrue(any(rfq_details.fold(pn).startswith(k) for k in named), pn)
                for a in specs:
                    if a.get("kind") != "drawing" or not a.get("part_number"):
                        continue
                    ln = next((ln for ln in rec["lines"] if value(ln["part_number"]) and rfq_details.fold(
                        value(ln["part_number"])).startswith(rfq_details.fold(a["part_number"]))), None)
                    self.assertIsNotNone(ln, a["part_number"])
                    if a.get("material") and "RFQ" not in (ln["material"]["source"] or ""):
                        self.assertEqual(rfq_details.norm(value(ln["material"])), rfq_details.norm(a["material"]))

    def test_e17_decision_is_explained_in_the_answer_key(self):
        # E17 asks what something would cost, so calling it an RFQ with everything missing is
        # defensible; the key and the heuristic both say no, and the key says why.
        self.assertIn("E17", TRUTH.get("not_rfq_why", {}))
        self.assertIn("capability", TRUTH["not_rfq_why"]["E17"].lower())

    def test_a_jev_decision_wins(self):
        self.assertTrue(rfq_details.is_rfq(EMAILS["E17"], {"is_rfq": True}))
        self.assertFalse(rfq_details.is_rfq(EMAILS["E01"], {"is_rfq": False}))

    def test_e17_as_an_rfq_lists_what_is_missing(self):
        """When Jev does call E17 an RFQ, the record says a quote cannot be prepared from it."""
        rec = rfq_details.extract(EMAILS["E17"], {}, SHOP, {"is_rfq": True, "lane": "review"}, today=TODAY)
        self.assertTrue(rec["is_rfq"])
        self.assertEqual(len(rec["lines"]), 1)
        self.assertIsNone(value(rec["lines"][0]["part_number"]))
        self.assertTrue({"quantity", "drawing", "material"} <= set(rec["missing"]), rec["missing"])

    def test_relative_dates(self):
        cases = {"Quote due by October 3": "2026-10-03", "within two weeks": "2026-10-09",
                 "Quote due in 10 business days": "2026-10-09", "Target quote date is next Friday": "2026-10-02",
                 "Could you get back to me by Oct 12?": "2026-10-12", "Can you quote today": "2026-09-25",
                 "respond by 10/10": "2026-10-10"}
        for text, want in cases.items():
            with self.subTest(text=text):
                got = rfq_details.parse_date(text, TODAY)
                self.assertIsNotNone(got)
                self.assertEqual(got[0].isoformat(), want)

    def test_dates_are_never_part_numbers(self):
        """The E22 photo once gave a part line 'ZOZS-06-17': the title block date 2025-06-17."""
        for text in ("2025-06-17", "ZOZS-06-17", "Z0Z5-O6-17", "DATE 2026-09-02", "2026-10"):
            with self.subTest(text=text):
                self.assertEqual(rfq_details._pn_candidates(text, True), [])
        self.assertEqual(rfq_details._pn_candidates("C1-1O442", True), ["CI-10442"])
        self.assertEqual(rfq_details._pn_candidates("8WM-3106", True), ["BWM-3106"])
        self.assertEqual(rfq_details._pn_candidates("PER MIL-A-8625 TYPE II", False), [])

    def test_drawing_part_number_needs_a_part_number_context(self):
        notes = ("NOTES:\n1. MATCH DRILL WITH QX-5512 BRACKET AT ASSEMBLY.\n2. BREAK SHARP EDGES.\n"
                 "REVISIONS\nA INITIAL RELEASE 2025-06-17 TW\n")
        doc = rfq_details.Doc("d.pdf", "pdf", {"method": "text-layer", "text": notes})
        self.assertIsNone(rfq_details.parse_drawing(doc)["part_number"], "a number in a note is not the part")
        doc = rfq_details.Doc("d.pdf", "pdf", {"method": "text-layer", "text": notes + "DWG NO.\nQX-5520\nREV\nA\n"})
        self.assertEqual(rfq_details.parse_drawing(doc)["part_number"], "QX-5520")
        doc = rfq_details.Doc("d.pdf", "pdf", {"method": "text-layer", "text": notes})
        self.assertEqual(rfq_details.parse_drawing(doc, ["QX-5512"])["part_number"], "QX-5512",
                         "a number the email names counts")

    def test_a_wrapped_title_block_value_is_kept_whole(self):
        def line(text, x, y, h=10):
            return {"text": text, "conf": 95.0, "page": 1, "bbox": [x, y, x + 8 * len(text), y + h]}
        lines = [line("TITLE", 960, 770), line("MOUNT, FOLD MIRROR", 960, 786),
                 line("MATERIAL", 960, 813), line("ALUMINUM 6061-T651 PER AMS 4027", 960, 828),
                 line("FINISH", 960, 853), line("BLACK ANODIZE PER MIL-A-8625 TYPE II CLASS 2. MASK", 960, 866),
                 line("PADS, BORES, AND DATUM A", 960, 879), line("SIZE", 960, 900), line("DWG NO.", 1005, 900),
                 line("LO-1186", 1005, 915), line("DRAWN", 960, 940)]
        doc = rfq_details.Doc("s.png", "png", {"method": "ocr", "confidence": 90.0, "lines": lines,
                                               "text": "\n".join(ln["text"] for ln in lines)})
        got = rfq_details.parse_drawing(doc)
        self.assertEqual(got["finish"], "BLACK ANODIZE PER MIL-A-8625 TYPE II CLASS 2. MASK PADS, BORES, AND DATUM A")
        self.assertEqual(got["material"], "ALUMINUM 6061-T651 PER AMS 4027")
        self.assertEqual(got["part_number"], "LO-1186")
        text = "TITLE\nPIN, PIVOT\nMATERIAL\nSTAINLESS STEEL 17-4 PH PER ASTM A564, CONDITION\nH1025\nFINISH\nNONE\nSIZE\n"
        doc = rfq_details.Doc("d.pdf", "pdf", {"method": "text-layer", "text": text})
        self.assertEqual(rfq_details.parse_drawing(doc)["material"],
                         "STAINLESS STEEL 17-4 PH PER ASTM A564, CONDITION H1025")

    def test_a_photo_is_typed_by_what_is_printed_on_it(self):
        drawing = ("REVISIONS\nTOP VIEW\nFRONT VIEW\nUNLESS OTHERWISE SPECIFIED:\nTOLERANCES:\nTITLE\n"
                   "BLOCK, KNIFE HOLDER\nMATERIAL\nSTEEL\nFINISH\nBLACK OXIDE\nDWG NO.\nOPM-22817\nDRAWN\nSCALE\n")
        doc = rfq_details.Doc("IMG_2231.jpg", "jpg", {"method": "ocr", "text": drawing, "confidence": 70.0})
        self.assertEqual(rfq_details.classify(doc), ("drawing", "photo"))
        doc = rfq_details.Doc("IMG_2232.jpg", "jpg", {"method": "ocr", "text": "CRACKED BOSS\nSEE ARROW", "confidence": 70.0})
        self.assertEqual(rfq_details.classify(doc), ("photo", "photo"))

    def test_csv_cells_cannot_run_as_formulas(self):
        rec = {"email_id": "X1", "customer": "=HYPERLINK(\"http://x\")", "contact": "+1 555", "subject": "s",
               "lines": [{"line": 1, "part_number": {"value": "@SUM(A1)"}, "description": {"value": "-cmd"},
                          "quantities": {"value": [5]}, "finish": {"value": "-5 C"}}]}
        rows = list(csv.reader(io.StringIO(rfq_details.to_csv([rec]), newline="")))
        cells = dict(zip(rows[0], rows[1]))
        self.assertEqual(cells["Customer"], "'=HYPERLINK(\"http://x\")")
        self.assertEqual(cells["Contact"], "'+1 555")
        self.assertEqual(cells["Part number"], "'@SUM(A1)")
        self.assertEqual(cells["Description"], "'-cmd")
        self.assertEqual(cells["Finish"], "-5 C")


# --------------------------------------------------------------------------- #
# Odd input, and emails written in other words than the beta inbox's
# --------------------------------------------------------------------------- #
def _ocr_line(text, x, y, h=12, conf=95.0, x1=None):
    """One OCR line as ocr.file_text reports it; about 9 px a character unless x1 says."""
    return {"text": text, "conf": conf, "page": 1, "bbox": [x, y, x1 or x + 9 * len(text), y + h]}


def _ocr(lines, conf=90.0):
    return {"method": "ocr", "confidence": conf, "lines": lines, "text": "\n".join(ln["text"] for ln in lines)}


class RobustnessTests(unittest.TestCase):
    """The server calls extract on whatever arrives; it must not crash, hang, or invent."""

    def test_odd_input_shapes_never_crash(self):
        base = EMAILS["E01"]
        bad_lines = {"method": "ocr", "confidence": "high", "text": "x",
                     "lines": ["junk", None, 5, {"text": None}, {"text": "X", "bbox": [1, 2, 3]},
                               {"text": "Y", "bbox": "abc", "conf": "high", "page": "two"}]}
        cases = [
            (dict(base, attachments=None), None, SHOP),
            (dict(base, attachments=["QA-41127_RevB.pdf", 7, None, {"kind": "file"}]), {}, SHOP),
            (dict(base, attachments=[{"name": 5, "media": 7}]), {5: {"text": "x"}}, SHOP),
            (base, {a["name"]: bad_lines for a in base["attachments"]}, SHOP),
            ({}, {}, SHOP),
            ({"id": 3, "subject": 12, "body": 5, "from_email": None, "from_name": 8}, "nope", None),
            (base, {}, {"customers": [{"domain": None}, "x", {"name": "N"}]}),
        ]
        for email, texts, shop in cases:
            with self.subTest(email=str(email)[:60]):
                rec = rfq_details.extract(email, texts, shop, today=TODAY)
                self.assertTrue(rec["lines"])
                rfq_details.to_csv([rec])
                json.loads(rfq_details.to_json([rec]))

    def test_unreadable_attachments_are_noted_not_guessed(self):
        base = EMAILS["E01"]
        rec = rfq_details.extract(base, {}, SHOP, today=TODAY)
        self.assertIn("No text could be read from QA-41127_RevB.pdf", rec["check"])
        self.assertEqual(value(rec["lines"][0]["part_number"]), "QA-41127")
        self.assertEqual(rec["lines"][0]["part_number"]["source"], "email body")
        garbage = {"QA-41127_RevB.pdf": _ocr([_ocr_line("~~ ||| ;;;", 0, 0, conf=5.0)], conf=12.0)}
        rec = rfq_details.extract(base, garbage, SHOP, today=TODAY)
        self.assertTrue(any("not recognizably a drawing" in c for c in rec["check"]), rec["check"])
        for field in ("description", "material", "finish"):  # nothing taken from the garbage
            self.assertNotIn("QA-41127_RevB.pdf", rec["lines"][0][field]["source"] or "")

    def test_pathological_text_stays_fast(self):
        import time
        base = EMAILS["E01"]
        bodies = ["Please quote " + "1" * 50_000 + " pcs", "Quantities: " + ",".join(["1"] * 20_000),
                  base["body"] + "\nFiller about the program, 6061-T6, 25 pcs, QA-41127. " * 5_000,
                  "size 1 x " + "1 x " * 20_000, "Finish: " + "anodize per " * 10_000]
        for body in bodies:
            with self.subTest(body=body[:40]):
                t = time.time()
                rec = rfq_details.extract(dict(base, body=body), {}, SHOP, today=TODAY)
                self.assertLess(time.time() - t, 5.0)
                for ln in rec["lines"]:
                    self.assertLessEqual(len(value(ln["quantities"]) or []), rfq_details.MAX_BREAKS)

    def test_odd_unicode_is_read_as_plain_text(self):
        body = ("Please quote \uff31\uff21\u2013\uff14\uff11\uff11\u200b\uff12\uff17 Rev\u00a0B, "
                "qty \uff12\uff15\u2009pcs.\u202e Quote due by October 3.\ufeff")
        rec = rfq_details.extract(dict(EMAILS["E01"], subject="RFQ", body=body, attachments=[]), {}, SHOP,
                                  today=TODAY)
        self.assertEqual(value(rec["lines"][0]["part_number"]), "QA-41127")
        self.assertEqual(value(rec["lines"][0]["rev"]), "B")
        self.assertEqual(value(rec["lines"][0]["quantities"]), [25])
        self.assertEqual(value(rec["respond_by"]), "2026-10-03")

    def test_with_and_without_a_jev_decision(self):
        base = EMAILS["E01"]
        decision = {"is_rfq": True, "lane": "milling_3axis", "lane_name": "3-Axis Milling", "owner": "Priya Nair",
                    "priority": "high", "due": {"date": "2026-10-01"}}
        rec = rfq_details.extract(base, {}, SHOP, decision, today=TODAY)
        self.assertEqual(rec["routing"], {"lane": "3-Axis Milling", "lane_id": "milling_3axis",
                                          "estimator": "Priya Nair", "priority": "high", "quote_by": "2026-10-01"})
        self.assertIsNone(rfq_details.extract(base, {}, SHOP, today=TODAY)["routing"])
        self.assertFalse(rfq_details.extract(base, {}, SHOP, {"is_rfq": False}, today=TODAY)["is_rfq"])
        # a decision without a usable is_rfq answer leaves the call to the heuristic
        for dec in ({}, {"is_rfq": None}, {"lane": "review"}):
            with self.subTest(decision=dec):
                self.assertTrue(rfq_details.is_rfq(base, dec))
                self.assertFalse(rfq_details.is_rfq(EMAILS["E13"], dec))
        records = rfq_details.extract_all(INBOX[:8], {}, SHOP, {"E01": {"is_rfq": False}, "E10": {"is_rfq": True}},
                                          today=TODAY)
        self.assertEqual([r["email_id"] for r in records][:2], ["E03", "E04"])
        self.assertIn("E10", [r["email_id"] for r in records])

    def test_every_value_has_a_source_even_header_fields(self):
        rec = rfq_details.extract(EMAILS["E09"], {}, SHOP, today=TODAY)
        self.assertEqual(rec["contact_source"], "email header")
        self.assertEqual(rec["contact_email_source"], "email header")
        self.assertEqual((rec["request"], rec["request_source"]), ("quote revision", "email body"))
        rec = rfq_details.extract(EMAILS["E26"], {}, SHOP, today=TODAY)
        self.assertEqual(rec["lines"][0]["rev"], {"value": "B", "source": "email body"})
        self.assertEqual(rec["lines"][0]["part_number"]["source"], "subject")


class FreshEmailTests(unittest.TestCase):
    """Emails written for these tests in phrasings the beta inbox does not use, so the rules are
    checked for being general rather than fitted to 22 emails."""

    def rec(self, subject, body, **kw):
        email = dict({"id": "T1", "from_name": "Pat Buyer", "from_email": "pat@examplemotion.com",
                      "subject": subject, "body": body, "attachments": []}, **kw)
        return rfq_details.extract(email, {}, SHOP, today=TODAY)

    def test_numeric_part_number_after_a_label(self):
        r = self.rec("Price request", "Can I get a price on 150 pieces of part number 7731-004? Material is 304 SS, "
                     "no finish. Need it quoted by Wednesday.\n\nThanks\nPat")
        ln = r["lines"][0]
        self.assertEqual(value(ln["part_number"]), "7731-004")
        self.assertEqual(value(ln["quantities"]), [150])
        self.assertEqual(value(ln["finish"]), "no finish")
        self.assertIsNone(value(ln["description"]), "'pieces' is a unit, not a description")
        self.assertEqual(value(r["respond_by"]), "2026-09-30")

    def test_one_line_per_part_with_its_own_quantity(self):
        r = self.rec("Request for Quotation RFQ# 55821",
                     "Please quote the items below.\n\nP/N 400-1187-02 Rev B, handle, Ti-6Al-4V ELI, qty 25/50\n"
                     "PL-1003 knob, Delrin, 80 pcs\n\nQuotes are due 10/7/2026. Certs of conformance are required.")
        self.assertEqual(value(r["rfq_number"]), "55821")
        self.assertEqual(value(r["respond_by"]), "2026-10-07")
        got = [(value(ln["part_number"]), value(ln["rev"]), value(ln["description"]), value(ln["material"]),
                value(ln["quantities"])) for ln in r["lines"]]
        self.assertEqual(got, [("400-1187-02", "B", "handle", "Ti-6Al-4V ELI", [25, 50]),
                               ("PL-1003", None, "knob", "Delrin", [80])])

    def test_quantity_and_date_phrasings(self):
        cases = [
            ("Please provide pricing for the AH-220 housing in quantities of 50, 100 and 250.", [50, 100, 250]),
            ("Can you quote qty 10 and 25 of QB-88 spacer in brass?", [10, 25]),
            ("Need a quote on 1ea of the attached bracket, A2 tool steel.", [1]),
            ("Please quote the manifold in 2 pcs for prototype and 50 pcs for production.", [2, 50]),
            ("Please quote 200 fittings per the attached drawing.", [200]),
        ]
        for body, want in cases:
            with self.subTest(body=body):
                self.assertEqual(value(self.rec("RFQ", body)["lines"][0]["quantities"]), want)
        for phrase, want in (("Quote by end of next week please.", "2026-10-02"),
                             ("We need your quote by the end of the month.", "2026-09-30"),
                             ("Due Oct 16.", "2026-10-16"), ("Please quote by COB Thursday.", "2026-10-01")):
            with self.subTest(phrase=phrase):
                self.assertEqual(value(self.rec("RFQ", "Please quote AB-100. " + phrase)["respond_by"]), want)
        r = self.rec("Budgetary pricing", "Budgetary pricing on gripper fingers, 6061 aluminum, qty 4 sets "
                     "initially, 500 sets a year in production.")
        self.assertEqual(value(r["lines"][0]["annual_usage"]), 500)
        self.assertEqual(value(r["lines"][0]["material"]), "6061 aluminum")

    def test_rfq_or_not_on_new_wordings(self):
        cases = [
            (True, "New part", "What would you charge for 50 of these? Drawing AB-1234 attached."),
            (True, "Looking for a quote", "Looking for a quote on 25 of the attached housing HS-100."),
            (True, "Re: Quote Q-26-0550", "Could you requote SF-3301 at 500 pcs instead of 250?"),
            (False, "Can you make these?", "Can you do this kind of work? What would something like this cost?"),
            (False, "Quote accepted", "We accept your quote Q-26-0550. PO 88200 to follow tomorrow."),
            (False, "Re: Quote Q-26-0550", "Thanks for the quote, we'll review internally and get back to you."),
            (False, "PO 88200 attached", "Please find attached PO 88200 for 100 pcs of HD-201 per your quote Q-26-0550."),
            (False, "Shipment notification", "Your order PO 88123 shipped today via UPS, tracking 1Z999."),
            (False, "Supplier survey", "Please complete the attached supplier quality survey by Oct 15."),
        ]
        for want, subject, body in cases:
            with self.subTest(subject=subject, body=body[:40]):
                email = {"subject": subject, "body": body, "from_name": "Pat Buyer", "from_email": "pat@example.com"}
                self.assertEqual(rfq_details.is_rfq(email, None), want)

    def test_a_number_after_rfq_that_the_body_calls_the_drawing(self):
        r = self.rec("RFQ NA-7710: fitting", "Please quote 200 fittings per the attached drawing NA-7710 rev D.")
        self.assertIsNone(value(r["rfq_number"]))
        self.assertEqual((value(r["lines"][0]["part_number"]), value(r["lines"][0]["rev"])), ("NA-7710", "D"))
        r = self.rec("RFQ SCD-7781: gimbal yoke", "Please quote 12 yokes. The package is on our portal.")
        self.assertEqual(value(r["rfq_number"]), "SCD-7781")


class OcrTextTests(unittest.TestCase):
    """Rules for OCR text, on small made-up readings."""

    def test_quantity_lists(self):
        cases = {"25 / 75 / 150": ([25, 75, 150], True), "250 / 500 / 1,000": ([250, 500, 1000], True),
                 "25, 50, 100": ([25, 50, 100], True), "1,000, 2,500": ([1000, 2500], True),
                 "1,O00": ([1000], True), "SO / 150 / 300": ([50, 150, 300], True),
                 "250 / 500 /": ([250, 500], False), "10k": ([10000], True)}
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(rfq_details.parse_quantities(text), want)

    def test_look_alike_repair_keeps_real_callouts(self):
        fix = rfq_details.ocr_fix_spec
        self.assertEqual(fix("ALUM1NUM 6O61-T6 PER AMS-QQ-A-25O/11"), "ALUMINUM 6061-T6 PER AMS-QQ-A-250/11")
        self.assertEqual(fix("STAINLESS STEEL 3I6L, TYPE Ill CLASS l"), "STAINLESS STEEL 316L, TYPE III CLASS 1")
        for keep in ("316L", "H1025", "1ST ARTICLE", "FR4", "C36000", "AS9102", "A2 TOOL STEEL", "4X"):
            with self.subTest(keep=keep):
                self.assertEqual(fix(keep), keep)
        self.assertEqual(rfq_details.ocr_fix_words("MATER1AL CERTS TRACEABLE TO HEAT OR L0T"),
                         "MATERIAL CERTS TRACEABLE TO HEAT OR LOT")

    def test_requirement_numbers_and_crumbs_are_dropped(self):
        for raw, want in (("1.\u00b0 ISO 13485:2016 CERTIFIED SUPPLIER REQUIRED", "ISO 13485:2016 CERTIFIED SUPPLIER REQUIRED"),
                          ("I. MATERIAL CERTS AND C OFC REQUIRED", "MATERIAL CERTS AND C OF C REQUIRED"),
                          ("l) FIRST ARTICLE ON FIRST LOT \u00b0", "FIRST ARTICLE ON FIRST LOT")):
            with self.subTest(raw=raw):
                self.assertEqual(rfq_details._req_text(raw), want)

    def test_standards_and_cited_specs_never_become_part_lines(self):
        self.assertEqual(rfq_details._pn_candidates("INTERPRET DRAWING PER ASME Y14.S-2018.", True), [])
        piece = rfq_details._row_piece("CLEAN, DOUBLE BAG, AND LABEL PER BWM-QS-0412. NO STERILIZATION", True)
        self.assertIsNone(piece["pn"])

    def test_a_form_whose_header_and_dates_ocr_ran_together(self):
        lines = [_ocr_line("REQUEST FOR QUOTATION", 1600, 100), _ocr_line("RFQ NO.", 1360, 280),
                 _ocr_line("DATE", 1710, 285), _ocr_line("RESPOND BY", 2060, 290),
                 _ocr_line("WS-26-0388", 1360, 325), _ocr_line("2026-09-23 2026-10-12", 1715, 330, x1=2245),
                 _ocr_line("ITEM PART NUMBER", 220, 1000), _ocr_line("REV", 690, 1005),
                 _ocr_line("DESCRIPTION", 800, 1010), _ocr_line("MATERIAL / FINISH QUANTITIES UNIT PRICE LEAD TIME", 1280, 1015, x1=2346),
                 _ocr_line("1", 245, 1066), _ocr_line("WS-4471 B", 335, 1066), _ocr_line("SHAFT, OUTPUT", 800, 1070),
                 _ocr_line("17-4 PH COND H1150 /", 1282, 1072), _ocr_line("25 / 100 / 250", 1715, 1080, conf=0.0),
                 _ocr_line("PASSIVATE PER AMS 2700", 1282, 1100), _ocr_line("QUOTE REQUIREMENTS", 180, 1300),
                 _ocr_line("1. MATERIAL CERTS AND C OF C REQUIRED WITH EACH SHIPMENT.", 180, 1340),
                 _ocr_line("TERMS: NET 45, FOB ORIGIN", 180, 1400)]
        doc = rfq_details.Doc("RFQ.pdf", "pdf", _ocr(lines))
        self.assertEqual(rfq_details.classify(doc)[0], "RFQ form")
        form = rfq_details.parse_form(doc)
        self.assertEqual(form["respond_by"], "2026-10-12", "the date under RESPOND BY, not the one beside it")
        self.assertEqual(len(form["rows"]), 1)
        row = form["rows"][0]
        self.assertEqual((row["part_number"], row["rev"], row["description"], row["quantities"]),
                         ("WS-4471", "B", "SHAFT, OUTPUT", [25, 100, 250]))
        self.assertEqual(form["requirements"], ["MATERIAL CERTS AND C OF C REQUIRED WITH EACH SHIPMENT"])

    def test_a_value_that_stops_at_a_label_is_trimmed_and_noted(self):
        """'... F2026. MARKERS:' whose second line is missing: the dangling label is dropped and the
        record says the callout may be incomplete (the UI once showed '... F2026. MARKERS')."""
        text = ("UNLESS OTHERWISE SPECIFIED\nTOLERANCES\nTHIRD ANGLE PROJECTION\nTITLE\nCAGE, INTERBODY\nMATERIAL\n"
                "PEEK, IMPLANT GRADE, PER ASTM F2026. MARKERS:\nFINISH\nNONE\nSIZE\nA\nDWG NO.\nAB-3140\nREV\nA\n")
        email = {"id": "T2", "subject": "RFQ AB-3140", "body": "Please quote 10 pcs of AB-3140.", "from_name": "Pat",
                 "from_email": "pat@example.com", "attachments": [{"name": "AB-3140.pdf", "media": "pdf"}]}
        rec = rfq_details.extract(email, {"AB-3140.pdf": {"method": "text-layer", "text": text}}, SHOP, today=TODAY)
        self.assertEqual(value(rec["lines"][0]["material"]), "PEEK, IMPLANT GRADE, PER ASTM F2026")
        self.assertTrue(any("stops at 'MARKERS:'" in c for c in rec["check"]), rec["check"])

    def test_border_text_run_into_a_title_block_value(self):
        lines = [_ocr_line("TITLE", 1590, 1290), _ocr_line("MANIFOLD BLOCK, VALVE", 1590, 1320),
                 _ocr_line("MATERIAL", 1590, 1365), _ocr_line("DO NOT SCALE DRAWING STAINLESS STEEL 316L PER ASTM A240", 1290, 1390),
                 _ocr_line("FINISH", 1590, 1430), _ocr_line("PASSIVATE PER ASTM A967", 1590, 1455),
                 _ocr_line("SIZE", 1590, 1500), _ocr_line("DWG NO.", 1660, 1500), _ocr_line("REV", 2010, 1505),
                 _ocr_line("A", 1600, 1530), _ocr_line("AB-2045", 1660, 1530), _ocr_line("D", 2045, 1532),
                 _ocr_line("DRAWN", 1590, 1600), _ocr_line("THIRD ANGLE PROJECTION", 1330, 1610)]
        got = rfq_details.parse_drawing(rfq_details.Doc("d.pdf", "pdf", _ocr(lines)))
        self.assertEqual((got["part_number"], got["rev"], got["material"]),
                         ("AB-2045", "D", "STAINLESS STEEL 316L PER ASTM A240"))


# --------------------------------------------------------------------------- #
# The beta inbox, end to end
# --------------------------------------------------------------------------- #
class BetaInboxTests(unittest.TestCase):
    def rows(self, kind=None, field=None):
        return [r for r in beta()["graded"] if r["kind"] != "extra" and (kind is None or r["kind"] == kind)
                and (field is None or r["field"] == field)]

    def assertAccuracy(self, rows, floor, what):
        self.assertTrue(rows, f"no {what} fields graded")
        wrong = [f"{r['email']} {r['field']}: got {r['got']!r}, want {r['want']!r}" for r in rows if not r["ok"]]
        ok = len(rows) - len(wrong)
        self.assertGreaterEqual(ok / len(rows), floor, f"{what}: {ok}/{len(rows)}\n" + "\n".join(wrong))

    def test_every_rfq_has_a_record(self):
        self.assertEqual(set(beta()["by_id"]), set(TRUTH["rfqs"]))

    def test_email_fields(self):
        self.assertAccuracy(self.rows("email"), 0.97, "email")

    def test_text_layer_fields(self):
        if not beta()["text_read"]:
            self.skipTest("no OCR cache and no pypdf: text layers cannot be read")
        self.assertAccuracy(self.rows("text layer"), 0.97, "text layer")
        self.assertAccuracy(self.rows("STEP header"), 1.0, "STEP header")

    @needs_scans
    def test_ocr_fields(self):
        self.assertAccuracy(self.rows("OCR"), 0.90, "OCR")
        self.assertAccuracy(self.rows(), 0.95, "all")

    def test_no_extra_items(self):
        """Nothing beyond the answer key: no stray part line anywhere, no stray requirement from an
        email or a text layer (OCR can garble a form item beyond matching, e.g. words run together)."""
        by_id = beta()["by_id"]
        extra = []
        for r in beta()["graded"]:
            if r["kind"] != "extra":
                continue
            if r["field"] == "requirements (extra)":
                src = next((x["source"] for x in by_id[r["email"]]["requirements"] if x["value"] == r["got"]), "")
                ocr_form = any(f["type"] == "RFQ form" and f["text_from"].startswith("OCR")
                               for f in by_id[r["email"]]["files"])
                if "(OCR" in src or ocr_form:
                    continue  # a garbled form item, or an email line that repeats one
            extra.append(f"{r['email']} {r['field']}: {r['got']!r}")
        self.assertEqual(extra, [])

    def test_export_control(self):
        want = {"E04": "ITAR", "E05": "CUI", "E52": "CUI", "E61": "ITAR", "E72": "ITAR"}
        for eid, rec in beta()["by_id"].items():
            with self.subTest(email=eid):
                if eid in ("E52", "E61", "E72") and not beta()["scans_read"]:
                    continue  # the marking is printed only on a scan
                self.assertEqual(value(rec["export_control"]), want.get(eid))
        for eid in ("E52", "E61", "E72"):
            if beta()["scans_read"]:
                src = beta()["by_id"][eid]["export_control"]["source"]
                self.assertIn("(OCR", src, f"{eid}: the marking is only on the scanned file")

    @needs_scans
    def test_quantities_from_scanned_rfq_forms(self):
        want = {"E32": [[25, 75, 150], [50, 150, 300]],
                "E56": [[50, 150, 300], [75, 200, 400], [50, 150, 300]],
                "E62": [[250, 500, 1000], [250, 500, 1000]],
                "E72": [[25, 100, 250]]}
        forms = {"E32": "RFQ-26-0931.pdf", "E56": "RFQ-26-0317.pdf", "E62": "RFQ-26-0318.pdf",
                 "E72": "WS-RFQ-26-0388.pdf"}
        for eid, qtys in want.items():
            with self.subTest(email=eid):
                text = (beta()["texts"][eid][forms[eid]].get("text") or "").replace("O", "0")
                need = Counter(q for row in qtys for q in row)
                unread = [q for q, n in need.items()
                          if len(re.findall(r"(?<![\d,])" + f"{q:,}".replace(",", ",?") + r"(?![\d,])", text)) < n]
                if unread:
                    self.skipTest(f"OCR did not read {unread} on {forms[eid]}")
                lines = beta()["by_id"][eid]["lines"]
                self.assertEqual([value(ln["quantities"]) for ln in lines], qtys)
                for ln in lines:
                    self.assertRegex(ln["quantities"]["source"], r"RFQ-26-\d{4}\.pdf(?:, line table)? \(OCR \d+%\)$")
                self.assertFalse([c for c in beta()["by_id"][eid]["check"] if "incomplete" in c])

    def test_no_bogus_part_lines(self):
        for eid, t in TRUTH["rfqs"].items():
            with self.subTest(email=eid):
                rec = beta()["by_id"][eid]
                if not beta()["scans_read"] and any(f["name"] in SCANS for f in rec["files"]):
                    continue
                want = [squash(value(ln["part_number"])) for ln in t["lines"]]
                got = [squash(value(ln["part_number"])) for ln in rec["lines"]]
                self.assertEqual(len(got), len(want), f"lines {got} vs {want}")
                self.assertEqual(sorted(g.translate(str.maketrans("OI", "01")) for g in got),
                                 sorted(w.translate(str.maketrans("OI", "01")) for w in want))
                for ln in rec["lines"]:
                    pn = value(ln["part_number"])
                    if pn:
                        self.assertFalse(rfq_details._looks_like_date(pn), pn)
                        self.assertRegex(pn, r"^[A-Z][A-Z0-9]{0,4}-\d")

    @needs_scans
    def test_photographed_drawing(self):
        """E22's drawing is a phone photo: it is a drawing, and its title block fills the line."""
        rec = beta()["by_id"]["E22"]
        self.assertEqual([(f["type"], f["capture"]) for f in rec["files"]], [("drawing", "photo")])
        self.assertNotIn("drawing", rec["missing"])
        self.assertEqual(len(rec["lines"]), 1)
        ln = rec["lines"][0]
        self.assertEqual(value(ln["part_number"]), "OPM-22817")
        self.assertEqual(value(ln["rev"]), "B")
        self.assertEqual(value(ln["description"]), "BLOCK, KNIFE HOLDER")
        self.assertIn("photo.jpg", ln["description"]["source"])
        self.assertIn("AISI 4140", value(ln["material"]))
        self.assertIn("MIL-DTL-13924", value(ln["finish"]))

    @needs_scans
    def test_wrapped_and_hard_to_read_title_block_values(self):
        by_id = beta()["by_id"]
        cases = [("E71", "LO-1186_RevA_screenshot.png", "finish",
                  "BLACK ANODIZE PER MIL-A-8625 TYPE II CLASS 2. MASK PADS, BORES, AND DATUM A"),
                 ("E09", "HPV-2045_manifold_RevD.pdf", "finish", "HARD ANODIZE PER MIL-A-8625 TYPE III CLASS 1"),
                 ("E09", "HPV-2045_manifold_RevD.pdf", "material", "ALUMINUM 6061-T6511 PER ASTM B221"),
                 ("E01", "QA-41127_RevB.pdf", "material", "ALUMINUM 6061-T6 PER AMS-QQ-A-250/11"),
                 ("E61", "CI-10442_RevC.pdf", "part_number", "CI-10442"),
                 ("E61", "CI-10442_RevC.pdf", "finish", "ELECTROLESS NICKEL PER AMS 2404 CLASS 1, .0003-.0005 THK")]
        for eid, name, field, want in cases:
            with self.subTest(email=eid, field=field):
                if not ocr_read(eid, name, want):
                    self.skipTest(f"OCR did not read {want!r} on {name}")
                got = by_id[eid]["lines"][0][field]
                self.assertEqual(squash(value(got)), squash(want))
                self.assertIn(name, got["source"])

    def test_wrapped_text_layer_values(self):
        if not beta()["text_read"]:
            self.skipTest("text layers cannot be read")
        by_id = beta()["by_id"]
        mats = {value(ln["material"]) for ln in by_id["E56"]["lines"]}
        self.assertEqual(mats, {"PEEK, IMPLANT GRADE, PER ASTM F2026. MARKERS: TANTALUM PER ASTM F560"})
        self.assertEqual(value(by_id["E62"]["lines"][0]["material"]),
                         "STAINLESS STEEL 17-4 PH PER ASTM A564, CONDITION H1025")
        self.assertEqual(value(by_id["E72"]["lines"][0]["finish"]),
                         "PASSIVATE PER AMS 2700 METHOD 1 AFTER ALL MACHINING")
        self.assertEqual(value(by_id["E32"]["lines"][0]["finish"]),
                         "HARD ANODIZE PER MIL-A-8625 TYPE III CLASS 1, .002 THK. MASK PORT THREADS.")

    def test_requirement_wording(self):
        reqs = lambda eid: [x["value"] for x in beta()["by_id"][eid]["requirements"]]  # noqa: E731
        self.assertEqual(reqs("E20"), ["12-month blanket pricing, released monthly"])
        self.assertIn("precision clean for high vacuum and double bag", reqs("E26"))
        self.assertEqual(reqs("E61"), ["Please break out setup and first article charges",
                                       "include material certs and a C of C with shipment"])
        self.assertEqual(reqs("E01"), ["Inspection: standard, FAI on the first lot"])

    @needs_scans
    def test_form_requirements_one_per_item(self):
        for eid in ("E32", "E56", "E62", "E72"):
            with self.subTest(email=eid):
                got = [x["value"] for x in beta()["by_id"][eid]["requirements"] if "(OCR" in x["source"]]
                want = [x["value"] for x in TRUTH["rfqs"][eid]["requirements"] if "email" not in x["where"]]
                self.assertEqual(len(got), len(want), got)
                for g in got:
                    self.assertNotRegex(g, r"^\W*\d|[^\w.)]$", "no item numbers or OCR crumbs at the ends")

    def test_email_only_rfqs(self):
        by_id = beta()["by_id"]
        self.assertEqual(value(by_id["E06"]["lines"][0]["size"]), '1.25" OD x 0.50" ID x 0.75" long')
        self.assertEqual(by_id["E06"]["missing"], ["drawing"])
        self.assertEqual(value(by_id["E04"]["lines"][0]["quantities"]), [12])
        self.assertEqual(value(by_id["E04"]["respond_by"]), "2026-10-09")
        self.assertEqual(value(by_id["E26"]["lines"][0]["quantities"]), [25, 60])
        self.assertTrue(any("behind a link" in c for c in by_id["E26"]["check"]))

    def test_per_part_email_values_stay_on_their_part(self):
        if not beta()["text_read"]:
            self.skipTest("text layers cannot be read")
        # "passivated" in E62 is said of BWM-3105 only; BWM-3106 has no finish
        self.assertFalse([c for c in beta()["by_id"]["E62"]["check"] if "finish differs" in c])

    def test_every_value_names_its_source(self):
        for rec in beta()["records"]:
            for key in ("rfq_number", "quote_ref", "respond_by", "export_control"):
                if value(rec[key]) not in (None, "", []):
                    self.assertTrue(rec[key]["source"], f"{rec['email_id']} {key}")
            for req in rec["requirements"]:
                self.assertTrue(req["source"])
            for ln in rec["lines"]:
                for field, v in ln.items():
                    if isinstance(v, dict) and v.get("value") not in (None, "", []):
                        self.assertTrue(v["source"], f"{rec['email_id']} line {ln['line']} {field}")

    def test_csv_round_trip(self):
        records = beta()["records"]
        text = rfq_details.to_csv(records)
        self.assertFalse(text.startswith("\ufeff"), "the file itself is plain UTF-8; the server adds the BOM")
        self.assertTrue(text.endswith("\r\n"))
        self.assertNotIn("\x00", text)
        rows = list(csv.reader(io.StringIO(text, newline="")))
        self.assertEqual(rows[0], rfq_details.CSV_COLUMNS)
        self.assertEqual(len(rows) - 1, sum(len(r["lines"]) for r in records))
        for row in rows[1:]:
            self.assertEqual(len(row), len(rows[0]))
            for cell in row:
                self.assertLess(len(cell), 32767, "Excel's cell limit")
                self.assertNotIn("\n", cell, "one spreadsheet row per part line")
                self.assertFalse(cell[:1] in ("=", "+", "@", "\t", "\r") or re.match(r"-[^\d]", cell), cell)
        cells = [dict(zip(rows[0], row)) for row in rows[1:]]
        self.assertEqual([c["Email"] for c in cells], [r["email_id"] for r in records for _ in r["lines"]])
        served = rfq_details.to_csv(records).encode("utf-8-sig")  # what /api/rfq_details.csv sends
        self.assertEqual(list(csv.reader(io.StringIO(served.decode("utf-8-sig"), newline=""))), rows)

    def test_json_round_trip(self):
        records = beta()["records"]
        back = json.loads(rfq_details.to_json(records))
        self.assertEqual(back["records"], json.loads(json.dumps(records)))

    def test_committed_files_match_a_fresh_run(self):
        """data/rfq_beta/rfq_details.csv and .json (written by python rfq_details.py) are current."""
        if not CSV_OUT.exists() or not JSON_OUT.exists():
            self.skipTest("run python rfq_details.py to write the consolidated file")
        if not beta()["scans_read"]:
            self.skipTest("the scans cannot be read here")
        raw = CSV_OUT.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), "the committed file is plain UTF-8")
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
        self.assertEqual(rows[0], rfq_details.CSV_COLUMNS)
        fresh = list(csv.reader(io.StringIO(rfq_details.to_csv(beta()["records"]), newline="")))
        key = lambda rs: [(r[0], r[11], r[12], r[18]) for r in rs[1:]]  # noqa: E731 - email, line, P/N, quantities
        self.assertEqual(key(rows), key(fresh), "regenerate with: python rfq_details.py")
        committed = json.loads(JSON_OUT.read_text(encoding="utf-8"))["records"]
        self.assertEqual([r["email_id"] for r in committed], [r["email_id"] for r in beta()["records"]])

    def test_no_dashes_in_the_output(self):
        text = rfq_details.to_csv(beta()["records"]) + rfq_details.to_json(beta()["records"])
        self.assertNotIn("\u2014", text)
        self.assertNotIn("\u2013", text)



# --------------------------------------------------------------------------- #
# Regions: where on the page a value was printed (layout.py's boxes, tagged on the OCR lines)
# --------------------------------------------------------------------------- #
def _regioned(rows, regions):
    """An OCR result whose text rows hold one or more cells, each cell (text, x, y, region); the
    lines carry the region tag ocr.py gives them, and the result the detector's regions."""
    lines, text = [], []
    for row in rows:
        parts = []
        for cell, x, y, region in row:
            ln = _ocr_line(cell, x, y)
            if region:
                ln["region"] = region
            lines.append(ln)
            parts.append(cell)
        text.append("   ".join(parts))
    return {"method": "ocr", "confidence": 90.0, "lines": lines, "text": "\n".join(text),
            "regions": [{"page": 1, "label": label, "conf": 0.9, "box": box} for label, box in regions]}


def _strip_regions(result):
    out = json.loads(json.dumps(result))
    out.pop("regions", None)
    for ln in out.get("lines") or []:
        ln.pop("region", None)
    return out


# A drawing whose DWG NO. cell OCR broke ('BWM 3105 ©', no dash) and whose REV letter it lost, with a
# proprietary notice that ran into the title block's date cell on one text row: the rogue row reads
# like a revision table row ('OR USE IT ... 2026-09-02'), as on the copier-scanned held-out pages.
_TB = ("title_block", [1280, 1200, 2120, 1650])
_ROGUE = [("OR USE IT FOR ANY PURPOSE OTHER THAN QUOTING WITHOUT WRITTEN PERMISSION.", 100, 1610, "proprietary_notice"),
          ("M. HALE", 1590, 1610, "title_block"), ("2026-09-02", 1750, 1610, "title_block")]
_BLOCK = [[("TITLE", 1590, 1290, "title_block")], [("PIN, PIVOT, JAW", 1590, 1320, "title_block")],
          [("MATERIAL", 1590, 1365, "title_block")], [("STAINLESS STEEL 17-4 PH PER ASTM A564", 1590, 1390, "title_block")],
          [("FINISH", 1590, 1430, "title_block")], [("PASSIVATE PER ASTM A967", 1590, 1455, "title_block")],
          [("SIZE", 1590, 1500, "title_block"), ("DWG NO.", 1660, 1500, "title_block")],
          [("A", 1600, 1530, "title_block"), ("BWM 3105 \u00a9", 1660, 1530, "title_block")],
          [("DRAWN", 1590, 1580, "title_block"), ("DATE", 1750, 1580, "title_block")]]


class RegionTests(unittest.TestCase):
    def tearDown(self):
        rfq_details.REGION_READING = _READING

    def test_revision_rows_come_from_the_revision_block(self):
        rows = [[("REVISIONS", 1500, 80, "revision_block")],
                [("A", 1420, 150, "revision_block"), ("INITIAL RELEASE", 1480, 150, "revision_block"),
                 ("2026-08-01", 1900, 150, "revision_block"), ("MH", 2050, 150, "revision_block")]] + _BLOCK + [_ROGUE]
        res = _regioned(rows, [("revision_block", [1400, 60, 2120, 200]), _TB,
                               ("proprietary_notice", [90, 1600, 1100, 1630])])
        rfq_details.REGION_READING = True
        got = rfq_details.parse_drawing(rfq_details.Doc("d.pdf", "pdf", res), ["BWM-3105"])
        self.assertEqual((got["part_number"], got["rev"]), ("BWM-3105", "A"))
        self.assertEqual(got["regions"]["rev"], "revision_block")
        # without regions (an old cache entry, a host without the detector) the rogue row wins, as before
        old = rfq_details.parse_drawing(rfq_details.Doc("d.pdf", "pdf", _strip_regions(res)), ["BWM-3105"])
        self.assertEqual(old["rev"], "OR")
        rfq_details.REGION_READING = False
        self.assertEqual(rfq_details.parse_drawing(rfq_details.Doc("d.pdf", "pdf", res), ["BWM-3105"])["rev"], "OR")

    def test_rev_cell_inside_the_title_block_needs_no_readable_drawing_number(self):
        block = [list(r) for r in _BLOCK]
        block[6].append(("REV", 2010, 1505, "title_block"))
        block[7].append(("A", 2045, 1532, "title_block"))
        res = _regioned(block + [_ROGUE], [_TB, ("proprietary_notice", [90, 1600, 1100, 1630])])
        rfq_details.REGION_READING = True
        got = rfq_details.parse_drawing(rfq_details.Doc("d.pdf", "pdf", res), ["BWM-3105"])
        self.assertEqual((got["part_number"], got["rev"], got["regions"]["rev"]), ("BWM-3105", "A", "title_block"))
        self.assertEqual(rfq_details.parse_drawing(rfq_details.Doc("d.pdf", "pdf", _strip_regions(res)),
                                                   ["BWM-3105"])["rev"], "OR")

    def test_sources_name_the_region_and_keep_the_ocr_confidence_last(self):
        res = _regioned(_BLOCK + [[("WARNING - THIS DOCUMENT CONTAINS TECHNICAL DATA WHOSE EXPORT IS RESTRICTED BY "
                                    "THE ARMS EXPORT CONTROL ACT (ITAR)", 100, 60, "export_legend")]],
                        [_TB, ("export_legend", [90, 50, 1300, 80])])
        email = {"id": "T3", "subject": "RFQ BWM-3105", "body": "Please quote 10 pcs of BWM-3105.", "from_name": "Pat",
                 "from_email": "pat@example.com", "attachments": [{"name": "BWM-3105.pdf", "media": "pdf"}]}
        rec = rfq_details.extract(email, {"BWM-3105.pdf": res}, SHOP, today=TODAY)
        ln = rec["lines"][0]
        self.assertEqual(ln["material"]["source"], "BWM-3105.pdf, title block (OCR 90%)")
        self.assertEqual(rec["export_control"]["value"], "ITAR")
        self.assertEqual(rec["export_control"]["source"], "BWM-3105.pdf, export legend (OCR 90%)")
        self.assertEqual(rec["files"][0]["regions"], ["export legend", "title block"])
        plain = rfq_details.extract(email, {"BWM-3105.pdf": _strip_regions(res)}, SHOP, today=TODAY)
        self.assertEqual(plain["lines"][0]["material"]["source"], "BWM-3105.pdf (OCR 90%)")
        self.assertEqual(plain["export_control"]["source"], "BWM-3105.pdf (OCR 90%)")
        self.assertNotIn("regions", plain["files"][0])

    def test_a_region_tag_the_extractor_does_not_know_is_ignored(self):
        res = _regioned(_BLOCK, [_TB])
        res["lines"][0]["region"] = "<script>"
        res["regions"].append({"label": 7, "box": "x"})
        doc = rfq_details.Doc("d.pdf", "pdf", res)
        self.assertIsNone(doc.lines[0].region)
        self.assertEqual(doc.regions, ["title_block"])

    @needs_scans
    def test_beta_sources_name_the_regions(self):
        texts = beta()["texts"]
        if "regions" not in texts["E61"]["CI-10442_RevC.pdf"]:
            self.skipTest("the OCR cache was built without the region detector")
        by_id = beta()["by_id"]
        self.assertEqual(by_id["E61"]["export_control"]["source"], "CI-10442_RevC.pdf, export legend (OCR 88%)"
                         if "88%" in by_id["E61"]["export_control"]["source"] else by_id["E61"]["export_control"]["source"])
        self.assertRegex(by_id["E61"]["export_control"]["source"], r"^CI-10442_RevC\.pdf, export legend \(OCR \d+%\)$")
        self.assertRegex(by_id["E52"]["export_control"]["source"], r"AGI-3052_RevA\.pdf, export legend \(OCR \d+%\)")
        self.assertRegex(by_id["E72"]["export_control"]["source"], r"WS-RFQ-26-0388\.pdf, export legend \(OCR \d+%\)")
        e61 = by_id["E61"]["lines"][0]
        for field in ("part_number", "material", "finish", "description"):
            self.assertRegex(e61[field]["source"], r"^CI-10442_RevC\.pdf, title block \(OCR \d+%\)$", field)
        self.assertRegex(by_id["E32"]["respond_by"]["source"], r"^RFQ-26-0931\.pdf, form header \(OCR \d+%\)$")
        form_reqs = [r["source"] for r in by_id["E32"]["requirements"] if "RFQ-26-0931" in r["source"]]
        self.assertTrue(form_reqs)
        for src in form_reqs:
            self.assertRegex(src, r"^RFQ-26-0931\.pdf, requirements \(OCR \d+%\)$")
        # every OCR source still ends in "(OCR NN%)", the part the UI's OCR chip reads (index.html
        # ocrInfo), and without it the file and region remain (the "found in" text)
        chip = re.compile(r"(?:^|[(,]\s*)OCR(?:\s*(\d{1,3}(?:\.\d+)?)\s*%)?\s*(?:\)|$)")
        for rec in beta()["records"]:
            values = [rec[k] for k in ("rfq_number", "respond_by", "export_control")] + rec["requirements"] + \
                [v for ln in rec["lines"] for v in ln.values() if isinstance(v, dict)]
            for v in values:
                for part in re.split(r"\s*;\s*", v.get("source") or ""):
                    if "(OCR" in part:
                        self.assertTrue(chip.search(part).group(1), part)
                        left = re.sub(r"\s*\((?:OCR[^)]*|text layer|STEP header)\)\s*$", "", part)
                        self.assertNotIn("(", left, part)

    @needs_scans
    def test_values_do_not_depend_on_the_regions(self):
        """Without the regions (a cache built where the detector could not run) every value on the
        beta inbox is the same; only the sources lose their region."""
        texts = {eid: {n: _strip_regions(r) for n, r in per.items()} for eid, per in beta()["texts"].items()}
        plain = rfq_details.extract_all(INBOX, texts, SHOP, today=TODAY)

        def values(recs):
            out = json.loads(json.dumps(recs))
            for rec in out:
                rec.pop("check", None)
                for f in rec["files"]:
                    f.pop("regions", None)
            return re.sub(r'"source": "([^"]*?), [a-z ]+ \(OCR', r'"source": "\1 (OCR', json.dumps(out))
        self.assertEqual(values(plain), values(beta()["records"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
