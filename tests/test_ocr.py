"""
Tests for ocr.py: PDFs with a text layer skip OCR, every uncopyable beta file is OCR'd and
gives the fields a buyer needs, the cache round-trips, and broken input comes back as an
"error" string instead of an exception.

    python -m unittest tests.test_ocr -v

The OCR tests skip when tesseract, pdftoppm, or Pillow is missing. Expected values come from
tests/rfq_beta_truth.json; tests may read it, ocr.py reads it only in --evaluate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ocr  # noqa: E402

DATA = ROOT / "data"
FILES = DATA / "rfq_beta" / "files"
TRUTH = json.loads((ROOT / "tests" / "rfq_beta_truth.json").read_text(encoding="utf-8"))["files"]
TOOLS = ocr.available()
HAVE_OCR = bool(TOOLS["ocr"] and TOOLS["pdftoppm"] and TOOLS["pillow"])
NO_OCR = "needs tesseract, pdftoppm, and Pillow"
HAVE_PDF_TEXT = importlib.util.find_spec("pypdf") is not None or shutil.which("pdftotext") is not None

UNCOPYABLE = sorted(rel for rel, t in TRUTH.items() if not t.get("copyable"))
TEXT_PDFS = sorted(rel for rel, t in TRUTH.items() if t.get("copyable") and rel.endswith(".pdf"))
STEPS = sorted(rel for rel, t in TRUTH.items() if rel.endswith(".step"))
# The source type the pipeline should find in each kind of file, from the pixels alone.
EXPECTED_TYPE = {"scan": "scan", "copier": "lowres", "fax": "bilevel", "photo": "photo", "screen": "screen"}


def rel_of(fragment: str) -> str:
    hits = [rel for rel in TRUTH if fragment in rel]
    assert len(hits) == 1, (fragment, hits)
    return hits[0]


def media_of(rel: str) -> str:
    return {"pdf": "pdf", "jpg": "jpg", "png": "png", "step": "step"}[rel.rsplit(".", 1)[1]]


def read(rel: str) -> bytes:
    return (DATA / rel).read_bytes()


_results: dict = {}
_results_lock = threading.Lock()


def ocr_result(rel: str) -> dict:
    """OCR each beta file once per test run (the tuned "best" pipeline, no cache), with no file
    name, so nothing can come from the name."""
    with _results_lock:
        if rel not in _results:
            _results[rel] = ocr.file_text(read(rel), media_of(rel), "", effort="best", timeout=900)
        return _results[rel]


def found(res: dict, value: str) -> bool:
    """The evaluation's rule: the value is in the text allowing only whitespace and case
    differences (a wrapped table cell counts as one run of text)."""
    rx = ocr._field_regex(value)
    text = (res.get("text") or "").upper()
    return bool(rx.search(text)) or any(rx.search(c.upper()) for c in ocr._cell_chains(res.get("lines") or []))


class AvailableTest(unittest.TestCase):
    def test_keys(self):
        info = ocr.available()
        for key in ("tesseract", "tesseract_version", "pdftoppm", "pillow", "ocr"):
            self.assertIn(key, info)
        self.assertIsInstance(info["ocr"], bool)
        if info["tesseract"]:
            self.assertRegex(info["tesseract_version"], r"^\d+\.\d+")


@unittest.skipUnless(HAVE_PDF_TEXT, "needs pypdf or pdftotext")
class TextLayerTest(unittest.TestCase):
    def test_text_pdfs_skip_ocr(self):
        self.assertEqual(len(TEXT_PDFS), 11)
        with mock.patch.object(ocr, "_ocr", side_effect=AssertionError("OCR ran on a text PDF")):
            for rel in TEXT_PDFS:
                with self.subTest(rel=rel):
                    res = ocr.file_text(read(rel), "pdf", Path(rel).name)
                    self.assertEqual(res["method"], "text-layer", res.get("error"))
                    self.assertIsNone(res["error"])
                    self.assertIsNone(res["confidence"])
                    spec = TRUTH[rel]["spec"]
                    key = spec.get("rfq_number") or spec.get("part_number")
                    self.assertIn(key, res["text"])

    def test_scan_has_no_text_layer(self):
        # Scanner software writes its name into some scans; that is not a text layer.
        layer = ocr._pdf_text_layer(read(rel_of("E61/CI-10442_RevC.pdf")))
        self.assertFalse(ocr._has_text_layer(layer))


class StepTest(unittest.TestCase):
    def test_step_headers(self):
        self.assertEqual(len(STEPS), 6)
        for rel in STEPS:
            with self.subTest(rel=rel):
                res = ocr.file_text(read(rel), "step", Path(rel).name)
                self.assertEqual(res["method"], "step-header", res.get("error"))
                self.assertIn(f"Part number: {TRUTH[rel]['spec']['part_number']}", res["text"])

    def test_not_a_step_file(self):
        res = ocr.file_text(b"solid cube\nfacet normal 0 0 1\n", "step")
        self.assertEqual(res["method"], "none")
        self.assertTrue(res["error"])


@unittest.skipUnless(HAVE_OCR, NO_OCR)
class BetaOcrTest(unittest.TestCase):
    def test_every_uncopyable_file_is_ocrd(self):
        self.assertEqual(len(UNCOPYABLE), 13)
        for rel in UNCOPYABLE:
            with self.subTest(rel=rel):
                res = ocr_result(rel)
                self.assertEqual(res["method"], "ocr", res.get("error"))
                self.assertIsNone(res["error"])
                self.assertEqual(res["pages"], 1)
                self.assertGreater(len(res["text"]), 400)
                self.assertGreaterEqual(res["confidence"], 60)
                self.assertTrue(res["lines"])
                for line in res["lines"][:5]:
                    self.assertEqual(len(line["bbox"]), 4)
                    self.assertEqual(line["page"], 1)
                page = res["settings"]["pages"][0]
                self.assertEqual(page["source"], EXPECTED_TYPE[TRUTH[rel]["render"]])
                self.assertIn("psm", page["recipe"])

    def test_type_comes_from_the_pixels(self):
        # The screenshot under a scan's name, and a scan under a photo's name, read the same.
        shot = rel_of("E71/")
        res = ocr.file_text(read(shot), "png", "CI-10442_RevC_scan.pdf", effort="best", timeout=900)
        self.assertEqual(res["settings"]["pages"][0]["source"], "screen")
        self.assertEqual(res["text"], ocr_result(shot)["text"])
        from PIL import Image
        import io
        self.assertEqual(ocr.classify(Image.open(io.BytesIO(read(rel_of("E22/")))))["type"], "photo")

    def assertFields(self, fragment: str, values: list) -> None:
        res = ocr_result(rel_of(fragment))
        for value in values:
            with self.subTest(file=fragment, value=value):
                self.assertTrue(found(res, value), f"{value!r} not in the OCR text of {fragment}:\n{res['text']}")

    def assertAllKeyFields(self, fragment: str, known_misses: tuple = ()) -> None:
        rel = rel_of(fragment)
        missed = ocr.score(ocr_result(rel), TRUTH[rel])["missed"]
        self.assertEqual(sorted(set(missed) - set(known_misses)), [], f"{fragment}: {missed}")

    def test_e61_itar_drawing(self):
        self.assertFields("E61/CI-10442_RevC.pdf", ["CI-10442", "ALUMINUM 6061-T6511 PER ASTM B221",
                                          "ELECTROLESS NICKEL PER AMS 2404 CLASS 1, .0003-.0005 THK",
                                          "INTERNATIONAL TRAFFIC IN ARMS REGULATIONS", "ITAR"])
        # Includes the rev letter C alone in the title block's REV cell, which tesseract reads
        # as "Cc" and repair turns back into "C".
        self.assertAllKeyFields("E61/CI-10442_RevC.pdf")

    def test_e72_itar_rfq_form(self):
        self.assertFields("E72/WS-RFQ", ["WS-26-0388", "2026-10-12", "WS-4471", "25 / 100 / 250",
                                        "17-4 PH COND H1150 PER AMS 5643", "PASSIVATE PER AMS 2700 METHOD 1",
                                        "INTERNATIONAL TRAFFIC IN ARMS REGULATIONS"])
        self.assertAllKeyFields("E72/WS-RFQ")

    def test_e52_cui_fax(self):
        self.assertFields("E52/AGI-3052", ["AGI-3052", "STAINLESS STEEL 316L PER ASTM A240, ANNEALED",
                                          "PASSIVATE PER ASTM A967 NITRIC 2", "CUI"])
        self.assertAllKeyFields("E52/AGI-3052")

    def test_e62_copier_form(self):
        # The first table row sits under a dark header band; on the copier scan the band must be
        # turned light before flatten evens out the page, or its item number and rev A are lost.
        self.assertFields("E62/RFQ-26-0318", ["RFQ-26-0318", "2026-10-15", "BWM-3105", "BWM-3106",
                                             "250 / 500 / 1,000", "PASSIVATE PER ASTM A967"])
        self.assertAllKeyFields("E62/RFQ-26-0318")

    def test_e32_form_quantities(self):
        self.assertFields("E32/RFQ-26-0931", ["RFQ-26-0931", "2026-10-09", "KF-3408", "KF-3412",
                                             "25 / 75 / 150", "50 / 150 / 300"])

    def test_e56_form_quantities(self):
        self.assertFields("E56/RFQ-26-0317", ["RFQ-26-0317", "2026-10-15", "BWM-3140-08", "BWM-3140-10",
                                             "BWM-3140-12", "50 / 150 / 300", "75 / 200 / 400"])

    def test_photo_part_number(self):
        self.assertFields("E22/", ["OPM-22817"])

    def test_screenshot_part_number(self):
        self.assertFields("E71/", ["LO-1186"])

    def test_fast_effort_is_one_pass(self):
        rel = rel_of("E71/")
        res = ocr.file_text(read(rel), "png", effort="fast", timeout=900)
        self.assertEqual(res["method"], "ocr", res.get("error"))
        self.assertEqual(len(res["settings"]["pages"][0]["recipe"]["psm"]), 1)
        self.assertTrue(found(res, "LO-1186"))


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ocr-cache-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_round_trip(self):
        path = Path(self.tmp) / "cache.json"
        result = {"method": "ocr", "text": "CI-10442 REV C", "confidence": 88.5, "pages": 1,
                  "lines": [{"text": "CI-10442", "conf": 90.0, "page": 1, "bbox": [1, 2, 3, 4]}],
                  "settings": {"pages": [{"source": "scan"}]}, "seconds": 2.0, "error": None,
                  "sha256": "x", "cached": False}
        cache = ocr.OcrCache(path)
        cache.put("ab" * 32, result)
        cache.save()
        self.assertEqual([p.name for p in Path(self.tmp).iterdir()], ["cache.json"])  # no temp file left
        again = ocr.OcrCache(path)
        got = again.get("ab" * 32)
        want = {k: v for k, v in result.items() if k not in ("sha256", "cached")}
        self.assertEqual(got, want)
        self.assertIsNone(again.get("cd" * 32))
        self.assertIn("tesseract_version", json.loads(path.read_text(encoding="utf-8")))
        got["text"] = "changed"  # callers get copies, never the stored entry
        self.assertEqual(again.get("ab" * 32)["text"], "CI-10442 REV C")

    def test_file_text_uses_the_cache(self):
        data = b"%PDF-1.4 a scan the cache already knows"
        sha = hashlib.sha256(data).hexdigest()
        cache = ocr.OcrCache(None)
        cache.put(sha, {"method": "ocr", "text": "FROM THE CACHE", "confidence": 90.0, "pages": 1, "lines": [],
                        "settings": {}, "seconds": 1.0, "error": None})
        with mock.patch.object(ocr, "_ocr", side_effect=AssertionError("OCR ran")), \
                mock.patch.object(ocr, "_pdf_text_layer", side_effect=AssertionError("pypdf ran")):
            res = ocr.file_text(data, "pdf", cache=cache)
        self.assertEqual(res["text"], "FROM THE CACHE")
        self.assertTrue(res["cached"])
        self.assertEqual(res["sha256"], sha)

    def test_broken_cache_file(self):
        path = Path(self.tmp) / "cache.json"
        for junk in ("{not json", "[1, 2, 3]", '{"entries": [1]}', ""):
            path.write_text(junk, encoding="utf-8")
            self.assertEqual(ocr.OcrCache(path).entries, {})

    @unittest.skipUnless(ocr.DEFAULT_CACHE.exists(), "the committed cache is not built")
    def test_committed_cache_covers_every_beta_file(self):
        cache = ocr.OcrCache(ocr.DEFAULT_CACHE)
        self.assertTrue(cache.meta.get("tesseract_version"))
        self.assertTrue(cache.meta.get("settings"))
        files = sorted(p for p in FILES.rglob("*") if p.is_file())
        self.assertEqual(len(files), 30)
        with mock.patch.object(ocr, "_ocr", side_effect=AssertionError("OCR ran")), \
                mock.patch.object(ocr, "_pdf_text_layer", side_effect=AssertionError("pypdf ran")):
            for p in files:
                rel = p.relative_to(DATA).as_posix()
                with self.subTest(rel=rel):
                    res = ocr.file_text(p.read_bytes(), media_of(rel), p.name, cache=cache)
                    self.assertTrue(res.get("cached"), res.get("error"))
                    want = ("step-header" if rel.endswith(".step") else
                            "text-layer" if TRUTH[rel].get("copyable") else "ocr")
                    self.assertEqual(res["method"], want)
                    self.assertTrue(res["text"].strip())
                    self.assertIsNone(res["error"])
                    self.assertEqual(res["file"], rel)
                    if want == "ocr":
                        self.assertTrue(res["settings"]["pages"][0]["recipe"])


class BrokenInputTest(unittest.TestCase):
    """Every failure is an "error" string in a normal result; nothing raises."""

    def assertError(self, res: dict) -> None:
        self.assertIsInstance(res, dict)
        self.assertEqual(res["method"], "none")
        self.assertIsInstance(res["error"], str)
        self.assertTrue(res["error"])
        self.assertEqual(res["text"], "")

    def test_zero_bytes(self):
        for media in ("pdf", "png", "jpg", "step"):
            with self.subTest(media=media):
                self.assertError(ocr.file_text(b"", media))
        self.assertError(ocr.file_text(None, "pdf"))  # type: ignore[arg-type]

    def test_garbage(self):
        junk = bytes((i * 131 + 7) % 256 for i in range(5000))
        for media in ("pdf", "png", "jpg", "step", "docx", ""):
            with self.subTest(media=media):
                self.assertError(ocr.file_text(junk, media, timeout=60))
        self.assertError(ocr.file_text(b"%PDF-1.4\n" + junk, "pdf", timeout=60))

    def test_truncated(self):
        for frag in ("E61/CI-10442_RevC.pdf", "E07/FR-2290_heatsink.pdf", "E22/", "E71/"):
            rel = rel_of(frag)
            data = read(rel)
            with self.subTest(rel=rel):
                self.assertError(ocr.file_text(data[: len(data) // 2], media_of(rel), timeout=60))

    def test_picture_claiming_a_huge_size(self):
        import struct
        import zlib
        ihdr = struct.pack(">IIBBBBB", 100_000, 100_000, 8, 0, 0, 0, 0)
        png = (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr
               + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr)) + b"\x00" * 64)
        self.assertError(ocr.file_text(png, "png", timeout=60))

    @unittest.skipUnless(HAVE_OCR, NO_OCR)
    def test_timeout(self):
        res = ocr.file_text(read(rel_of("E61/CI-10442_RevC.pdf")), "pdf", timeout=0.3)
        self.assertError(res)
        self.assertRegex(res["error"], r"time|too long")

    def test_tesseract_missing(self):
        with mock.patch.object(ocr, "available", return_value={"tesseract": False, "tesseract_version": None,
                                                               "pdftoppm": False, "pillow": True, "ocr": False}):
            res = ocr.file_text(read(rel_of("E71/")), "png")
        self.assertError(res)
        self.assertIn("tesseract", res["error"])

    def test_ocr_turned_off(self):
        res = ocr.file_text(read(rel_of("E71/")), "png", allow_ocr=False)
        self.assertError(res)

    def test_unexpected_exception_inside(self):
        with mock.patch.object(ocr, "_file_text", side_effect=RuntimeError("boom")):
            res = ocr.file_text(b"data", "pdf")
        self.assertError(res)
        self.assertIn("boom", res["error"])

    def test_huge_picture_without_pillow(self):
        # Without Pillow the picture goes to tesseract as it is, so its claimed size is read from
        # the header first.
        import struct
        import zlib
        ihdr = struct.pack(">IIBBBBB", 100_000, 100_000, 8, 0, 0, 0, 0)
        png = (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr
               + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr)) + b"\x00" * 64)
        self.assertEqual(ocr._picture_size(png), (100_000, 100_000))
        on = {"tesseract": True, "tesseract_version": "5.3.0", "pdftoppm": True, "pillow": False, "ocr": True}
        with mock.patch.object(ocr, "Image", None), mock.patch.object(ocr, "available", return_value=on), \
                mock.patch.object(ocr, "_tesseract_words", side_effect=AssertionError("tesseract ran")):
            res = ocr.file_text(png, "png", timeout=30)
        self.assertError(res)
        self.assertIn("too large", res["error"])

    def test_busy_ocr_slot(self):
        # RFQ_OCR_WORKERS: a file that waits longer than its time limit for a free slot gives up
        # with an error instead of queueing forever.
        on = {"tesseract": True, "tesseract_version": "5.3.0", "pdftoppm": True, "pillow": True, "ocr": True}
        slots = threading.BoundedSemaphore(1)
        slots.acquire()
        with mock.patch.object(ocr, "_ocr_slots", slots), mock.patch.object(ocr, "available", return_value=on), \
                mock.patch.object(ocr, "_ocr", side_effect=AssertionError("OCR ran without a slot")):
            started = time.monotonic()
            res = ocr.file_text(read(rel_of("E71/")), "png", timeout=0.5)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertError(res)
        self.assertIn("busy", res["error"])


def _tiny_pdf(pages: list) -> bytes:
    """A PDF of pages (width_pt, height_pt, jpeg_or_None, (w_px, h_px), text_or_None, picture_box_or_None)."""
    objects: list = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    catalog, root = add(b""), add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    kids = []
    for pw, ph, jpeg, size, text, box in pages:
        res, content = [], b""
        if jpeg:
            im = add(f"<< /Type /XObject /Subtype /Image /Width {size[0]} /Height {size[1]} /ColorSpace /DeviceGray "
                     f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(jpeg)} >>\nstream\n".encode()
                     + jpeg + b"\nendstream")
            x, y, w, h = box or (0, 0, pw, ph)
            content += f"q {w} 0 0 {h} {x} {y} cm /Im0 Do Q\n".encode()
            res.append(f"/XObject << /Im0 {im} 0 R >>")
        if text:
            content += f"BT /F1 7 Tf 20 10 Td ({text}) Tj ET\n".encode()
            res.append(f"/Font << /F1 {font} 0 R >>")
        st = add(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
        kids.append(add(f"<< /Type /Page /Parent {root} 0 R /MediaBox [0 0 {pw} {ph}] "
                        f"/Resources << {' '.join(res)} >> /Contents {st} 0 R >>".encode()))
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {root} 0 R >>".encode()
    objects[root - 1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _e61_jpeg() -> bytes:
    """The JPEG inside the E61 office scan, as the scanner stored it."""
    from pypdf import PdfReader
    page = PdfReader(str(DATA / rel_of("E61/CI-10442_RevC.pdf"))).pages[0]
    xobjects = page["/Resources"]["/XObject"]
    return xobjects[list(xobjects)[0]].get_object().get_data()


HAVE_POPPLER_INFO = all(shutil.which(t) for t in ("pdftotext", "pdfimages", "pdfinfo"))
STAMP = "Scanned by ScanDesk Pro 4.2 on 2026-09-24 at 15:00 UTC, operator station 7, page 1 of 1"


class PdfShapesTest(unittest.TestCase):
    """PDFs that are not what they first look like: a scan with a typed stamp over it, a page
    200 inches square, a password."""

    @unittest.skipUnless(HAVE_POPPLER_INFO and importlib.util.find_spec("pypdf"), "needs poppler-utils and pypdf")
    def test_scan_with_a_typed_stamp_needs_ocr(self):
        jpeg = _e61_jpeg()
        stamped = _tiny_pdf([(792, 612, jpeg, (3300, 2550), STAMP, None)])
        layer = ocr._pdf_text_layer(stamped)
        self.assertTrue(ocr._has_text_layer(layer))  # the stamp alone passes the plain letter count
        self.assertEqual(ocr._scanned_pages(stamped, layer, time.monotonic() + 60), [1])
        # The same text next to a small picture (a logo) is a typed page.
        logo = _tiny_pdf([(792, 612, jpeg, (3300, 2550), STAMP, (20, 500, 110, 85))])
        self.assertEqual(ocr._scanned_pages(logo, ocr._pdf_text_layer(logo), time.monotonic() + 60), [])

    @unittest.skipUnless(HAVE_OCR and HAVE_POPPLER_INFO and importlib.util.find_spec("pypdf"), NO_OCR)
    def test_stamped_scan_is_read_with_ocr(self):
        stamped = _tiny_pdf([(792, 612, _e61_jpeg(), (3300, 2550), STAMP, None)])
        res = ocr.file_text(stamped, "pdf", effort="fast", timeout=900)
        self.assertEqual(res["method"], "ocr", res.get("error"))
        self.assertTrue(found(res, "CI-10442"), res["text"])

    def test_render_resolution_is_capped(self):
        self.assertEqual(ocr._raster_dpi(300, (11.0, 8.5)), 300)
        self.assertEqual(ocr._raster_dpi(300, None), 300)
        self.assertLess(ocr._raster_dpi(300, (200.0, 200.0)), 31)  # the largest page a PDF allows
        dpi = ocr._raster_dpi(1200, (11.0, 8.5))  # a 1200 dpi scan: 135 million pixels in full
        self.assertLessEqual(11.0 * 8.5 * dpi * dpi, ocr.MAX_OCR_PIXELS)

    @unittest.skipUnless(HAVE_OCR and shutil.which("pdfinfo"), NO_OCR)
    def test_huge_page_is_rendered_small(self):
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("L", (330, 255), 255).save(buf, "JPEG")
        huge = _tiny_pdf([(14400, 14400, buf.getvalue(), (330, 255), None, None)])
        with mock.patch.object(ocr, "_rasterize", wraps=ocr._rasterize) as raster:
            res = ocr.file_text(huge, "pdf", effort="fast", timeout=120)
        self.assertIn(res["method"], ("ocr", "none"))
        self.assertLess(raster.call_args[0][2], 31)  # rendered at under 31 dpi, not 300

    @unittest.skipUnless(HAVE_OCR and importlib.util.find_spec("pypdf"), NO_OCR)
    def test_password_protected_pdf(self):
        from pypdf import PdfReader, PdfWriter
        import io
        writer = PdfWriter(clone_from=PdfReader(str(DATA / TEXT_PDFS[0])))
        writer.encrypt(user_password="secret", owner_password="owner", algorithm="RC4-128")
        buf = io.BytesIO()
        writer.write(buf)
        res = ocr.file_text(buf.getvalue(), "pdf", timeout=60)
        self.assertEqual(res["method"], "none")
        self.assertIn("password", res["error"])

    @unittest.skipUnless(HAVE_OCR and importlib.util.find_spec("pypdf"), NO_OCR)
    def test_without_pdftoppm_pages_are_still_classified(self):
        # Without poppler the pictures come out of the PDF through pypdf; a fax must still get
        # the fax recipe and a copier scan the copier recipe, not the office scan one.
        tools = dict(ocr._tools(), pdftoppm=None, pdfimages=None, pdftotext=None)
        with mock.patch.object(ocr, "_tools", return_value=tools), \
                mock.patch.object(ocr.shutil, "which", return_value=None):
            for frag, kind, value in (("E52/AGI-3052", "bilevel", "AGI-3052"),
                                      ("E09/HPV-2045_manifold_RevD.pdf", "lowres", "HPV-2045")):
                with self.subTest(file=frag):
                    res = ocr.file_text(read(rel_of(frag)), "pdf", effort="fast", timeout=900)
                    self.assertEqual(res["method"], "ocr", res.get("error"))
                    self.assertEqual(res["settings"]["pages"][0]["source"], kind)
                    self.assertTrue(found(res, value))


@unittest.skipUnless(TOOLS["pillow"], "needs Pillow")
class PictureModesTest(unittest.TestCase):
    def test_sixteen_bit_gray_is_not_white(self):
        # Pillow's own convert("L") clips 16-bit values at 255, which would make the page white.
        from PIL import Image
        ramp = Image.linear_gradient("L").resize((64, 64)).convert("I").point(lambda v: v * 257)
        for img in (ramp, ramp.convert("I;16"), ramp.convert("F")):
            with self.subTest(mode=img.mode):
                gray = ocr._gray(img)
                self.assertEqual(gray.mode, "L")
                lo, hi = gray.getextrema()
                self.assertLess(lo, 10)
                self.assertGreater(hi, 245)

    def test_transparent_picture_reads_as_white_paper(self):
        from PIL import Image
        clear = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        self.assertEqual(ocr._gray(clear).getextrema(), (255, 255))
        self.assertFalse(ocr.classify(clear)["color"])

    def test_picture_sizes_from_headers(self):
        from PIL import Image
        import io
        for fmt, mode in (("PNG", "L"), ("JPEG", "RGB"), ("JPEG", "CMYK"), ("PPM", "L")):
            buf = io.BytesIO()
            Image.new(mode, (123, 45)).save(buf, fmt)
            with self.subTest(fmt=fmt, mode=mode):
                self.assertEqual(ocr._picture_size(buf.getvalue()), (123, 45))
        self.assertIsNone(ocr._picture_size(b"not a picture"))


TSV_HEAD = b"level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"


class PiecesTest(unittest.TestCase):
    def test_settings_text_round_trip(self):
        for kind, recipe in list(ocr.RECIPES.items()) + list(ocr.FAST_RECIPES.items()):
            with self.subTest(kind=kind):
                text = ocr.settings_text(recipe)
                self.assertEqual(ocr.settings_text(ocr.parse_settings(text)), text)

    def test_every_recipe_setting_is_known(self):
        known = {"raster_dpi", "target_dpi", "scale", "color", "median", "flatten", "invert", "autocontrast",
                 "unsharp", "deskew", "page", "psm", "merge", "threshold", "tess_dpi", "pis", "nodict", "repair",
                 "orient", "min_conf", "tessdata", "resample", "raw"}
        for recipes in (ocr.RECIPES, ocr.FAST_RECIPES):
            self.assertEqual(set(recipes), set(ocr.SOURCE_TYPES))
            for kind, recipe in recipes.items():
                self.assertLessEqual(set(recipe), known, kind)
                for psm in recipe["psm"]:
                    self.assertIn(psm, (3, 4, 6, 11, 12))
                self.assertIn(recipe.get("threshold"), (None, 0, 1, 2))

    def test_parameters_exist_in_tesseract_530(self):
        # Debian bookworm (the Render image) ships tesseract 5.3.0. Build the command line for
        # every recipe, and for one with every optional switch on, and check each option and
        # -c variable against what a 5.3.0 build accepts: its --help-extra options, its
        # --help-psm modes (0 to 13), and these names from its --print-parameters output.
        in_530 = {"thresholding_method", "preserve_interword_spaces", "load_system_dawg", "load_freq_dawg",
                  "tessedit_create_tsv"}
        options = {"-l", "--oem", "--psm", "--dpi", "--tessdata-dir"}
        everything = {"psm": [3, 11], "threshold": 2, "pis": True, "nodict": True, "tessdata": "/models"}
        recipes = list(ocr.RECIPES.values()) + list(ocr.FAST_RECIPES.values()) + [everything]
        fake = dict(ocr._tools(), tesseract="tesseract", tesseract_version="5.3.0", tesseract_params=in_530)
        seen = []
        with mock.patch.object(ocr, "_tools", return_value=fake), \
                mock.patch.object(ocr, "_run", side_effect=lambda cmd, *a, **k: seen.append(cmd) or TSV_HEAD):
            for recipe in recipes:
                for psm in recipe["psm"]:
                    ocr._tesseract_words("page.tif", psm, 300.0, recipe, 1e18)
        self.assertTrue(any("thresholding_method=2" in cmd for cmd in seen))  # the switches got through
        for cmd in seen:
            self.assertEqual(cmd[1:3], ["page.tif", "stdout"])
            self.assertEqual(cmd[-2:], ["-c", "tessedit_create_tsv=1"])
            args = cmd[3:]
            for i, arg in enumerate(args):
                if arg.startswith("-") and arg != "-c":
                    self.assertIn(arg, options)
                if arg == "-c":
                    self.assertIn(args[i + 1].split("=")[0], in_530)
                if arg == "--psm":
                    self.assertIn(int(args[i + 1]), range(14))
                if arg == "--oem":
                    self.assertEqual(args[i + 1], "1")  # LSTM only: the model Debian ships has no legacy data
        for name in in_530:
            self.assertLessEqual(ocr._PARAM_SINCE[name], (5, 3, 0), name)

    def test_plain_text_instead_of_tsv_is_an_error(self):
        # A tesseract that cannot make its word table (for example a config it cannot find)
        # prints plain text; that must be an error, not a page with no words.
        fake = dict(ocr._tools(), tesseract="tesseract", tesseract_version="5.3.0", tesseract_params=set())
        with mock.patch.object(ocr, "_tools", return_value=fake), \
                mock.patch.object(ocr, "_run", return_value=b"DWG NO. CI-10442\n"):
            with self.assertRaises(ocr.OcrError):
                ocr._tesseract_words("page.tif", 3, 300.0, {"psm": [3]}, 1e18)

    def test_unknown_parameters_are_left_out(self):
        fake = dict(ocr._tools(), tesseract_params={"preserve_interword_spaces"}, tesseract_version="4.1.1")
        with mock.patch.object(ocr, "_tools", return_value=fake):
            self.assertFalse(ocr._supports("thresholding_method"))
            self.assertTrue(ocr._supports("preserve_interword_spaces"))
        fake = dict(fake, tesseract_params=set())
        with mock.patch.object(ocr, "_tools", return_value=fake):
            self.assertFalse(ocr._supports("thresholding_method"))  # 4.1 predates it
            self.assertTrue(ocr._supports("load_system_dawg"))

    def test_merge_passes(self):
        first = [{"text": "Cc", "conf": 40.0, "box": [100, 10, 120, 30], "line": (3, 1, 1, 1)}]
        second = [{"text": "C", "conf": 95.0, "box": [101, 10, 119, 30], "line": (11, 1, 1, 1)},
                  {"text": "B", "conf": 90.0, "box": [300, 10, 312, 30], "line": (11, 2, 1, 1)}]
        filled = ocr._merge_passes([first, second], "fill")
        self.assertEqual([w["text"] for w in filled], ["Cc", "B"])
        swapped = ocr._merge_passes([first, second], "conf")
        self.assertEqual([w["text"] for w in swapped], ["C", "B"])
        self.assertEqual(first[0]["text"], "Cc")  # the passes themselves are not changed

    def test_repair_case(self):
        words = [{"text": t, "conf": 90.0, "box": [0, 0, 1, 1], "line": (3, 1, 1, 1)}
                 for t in ("DWG", "NO.", "Cl-10442", "TYPE", "Ill")]
        ocr._repair_case(words)
        self.assertEqual([w["text"] for w in words], ["DWG", "NO.", "CI-10442", "TYPE", "III"])
        # A rev letter alone in its cell, read as the capital and its lowercase twin; a real
        # two-letter word or a mixed-case line is left alone.
        words = [{"text": t, "conf": 57.0, "box": [0, 0, 1, 1], "line": (3, 2, 1, i)}
                 for i, t in enumerate(("Cc", "Oo", "Ok", "Cl", "cC"))]
        ocr._repair_case(words)
        self.assertEqual([w["text"] for w in words], ["C", "O", "Ok", "Cl", "cC"])


if __name__ == "__main__":
    unittest.main()
