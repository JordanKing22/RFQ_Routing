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
                    self.assertRegex(ln["quantities"]["source"], r"RFQ-26-\d{4}\.pdf \(OCR \d+%\)")
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
        self.assertFalse(text.startswith("﻿"), "the file itself is plain UTF-8; the server adds the BOM")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
