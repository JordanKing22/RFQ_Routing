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
# The region detector (layout.py): numpy, onnxruntime, Pillow, and models/rfq_layout.onnx.
try:
    import layout  # noqa: E402

    HAVE_LAYOUT = bool(layout.available().get("ready"))
except Exception:  # noqa: BLE001
    layout = None  # type: ignore[assignment]
    HAVE_LAYOUT = False
NO_LAYOUT = "needs the region detector (numpy, onnxruntime, Pillow, models/rfq_layout.onnx)"

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
        # Well formed after the header, it is refused for its size, and says so.
        def chunk(kind: bytes, body: bytes) -> bytes:
            return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00" * 1000))
               + chunk(b"IEND", b""))
        on = {"tesseract": True, "tesseract_version": "5.3.0", "pdftoppm": True, "pillow": TOOLS["pillow"], "ocr": True}
        with mock.patch.object(ocr, "available", return_value=on), \
                mock.patch.object(ocr, "_tesseract_words", side_effect=AssertionError("tesseract ran")):
            res = ocr.file_text(png, "png", timeout=60)
        self.assertError(res)
        self.assertIn("too large", res["error"])

    def test_pdftotext_keeps_to_the_time_limit(self):
        # The text layer check runs before OCR; pdftotext must not outlast the file's limit.
        import subprocess
        seen: list = []
        real_run = subprocess.run

        def fake_run(cmd, *args, **kw):
            if cmd and cmd[0] == "fake-pdftotext":
                seen.append(kw.get("timeout"))
                return subprocess.CompletedProcess(cmd, 0, b"", b"")
            return real_run(cmd, *args, **kw)
        tools = dict(ocr._tools(), pdftotext="fake-pdftotext")
        with mock.patch.object(ocr, "_tools", return_value=tools), \
                mock.patch.object(ocr.subprocess, "run", side_effect=fake_run), \
                mock.patch.dict(sys.modules, {"attachments": None}):
            ocr._pdf_text_layer(b"%PDF-1.4\n", time.monotonic() + 2)
            ocr._pdf_text_layer(b"%PDF-1.4\n", time.monotonic() - 1)  # already out of time
        self.assertEqual(len(seen), 1)
        self.assertLessEqual(seen[0], 2.0)

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

    def test_transparent_gray_matches_compositing_on_white(self):
        # Blended in gray to save memory; it must still be the gray of the picture over white.
        from PIL import Image, ImageChops
        import random
        rnd = random.Random(7)
        img = Image.new("RGBA", (32, 32))
        img.putdata([tuple(rnd.randrange(256) for _ in range(4)) for _ in range(32 * 32)])
        ref = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img).convert("L")
        self.assertLessEqual(ImageChops.difference(ocr._gray(img), ref).getextrema()[1], 2)
        pal = Image.new("P", (4, 4), 0)
        pal.putpalette([0, 0, 0, 10, 10, 10] + [0] * 762)
        pal.info["transparency"] = 0  # palette entry 0 is see-through
        pal.putpixel((0, 0), 1)
        gray = ocr._gray(pal)
        self.assertEqual((gray.getpixel((1, 1)), gray.getpixel((0, 0))), (255, 10))

    def test_big_pictures_stay_within_the_memory_budget(self):
        # A picture that would pass MAX_INPUT_BYTES once decoded is refused (a PNG) or, for a
        # JPEG, decoded at a half or a quarter of its size by the JPEG decoder itself.
        from PIL import Image
        import io
        png, jpg = io.BytesIO(), io.BytesIO()
        Image.new("RGBA", (400, 300), (0, 0, 0, 255)).save(png, "PNG")
        Image.new("RGB", (400, 300), (255, 255, 255)).save(jpg, "JPEG")
        with mock.patch.object(ocr, "MAX_INPUT_BYTES", 200 * 150 * 4):
            with self.assertRaisesRegex(ocr.OcrError, "too large"):
                ocr._open_image(png.getvalue())
            self.assertEqual(ocr._open_image(jpg.getvalue()).size, (200, 150))
        self.assertEqual(ocr._open_image(jpg.getvalue()).size, (400, 300))
        # CMYK goes through a whole RGB copy on its way to gray, so it counts double
        cmyk = io.BytesIO()
        Image.new("CMYK", (400, 300), (0, 0, 0, 0)).save(cmyk, "JPEG")
        with mock.patch.object(ocr, "MAX_INPUT_BYTES", 400 * 300 * 4):
            self.assertEqual(ocr._open_image(jpg.getvalue()).size, (400, 300))
            self.assertEqual(ocr._open_image(cmyk.getvalue()).size, (200, 150))
        # Pillow keeps RGB at 4 bytes a pixel. The budget is for a server that has been used (200
        # to 250 MB before the upload): a 48 megapixel phone photo is over it, so a JPEG is read at
        # half size (4000 x 3000, which fits) and a PNG is refused; so is a 79 megapixel RGBA PNG
        # (1.2 GB of Python memory before the budget) and an 11 x 17 in 1-bit drawing at 600 dpi.
        self.assertGreater(ocr._decoded_bytes("RGB", (8000, 6000)), ocr.MAX_INPUT_BYTES)
        self.assertLessEqual(ocr._decoded_bytes("RGB", (4000, 3000)), ocr.MAX_INPUT_BYTES)
        self.assertGreater(ocr._decoded_bytes("RGBA", (8900, 8900)), ocr.MAX_INPUT_BYTES)
        self.assertGreater(10200 * 6600, ocr.MAX_INPUT_PIXELS)
        self.assertGreaterEqual(ocr.MAX_INPUT_PIXELS, ocr.MAX_OCR_PIXELS)  # a rendered PDF page always fits

    def test_sideways_photo_is_turned_by_its_exif(self):
        from PIL import Image
        import io
        img = Image.new("RGB", (40, 20), "white")
        exif = img.getexif()
        exif[0x0112] = 6  # stored on its side: turn 90 degrees to view
        turned, plain = io.BytesIO(), io.BytesIO()
        img.save(turned, "JPEG", exif=exif.tobytes())
        img.save(plain, "JPEG")
        self.assertEqual(ocr._open_image(turned.getvalue()).size, (20, 40))
        self.assertEqual(ocr._open_image(plain.getvalue()).size, (40, 20))

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
        # Debian bookworm's tesseract 5.3.0 is the oldest build the recipes must run on (Render's
        # python:3.12-slim is now Debian 13 with 5.5.0). Build the command line for
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



class RegionsTest(unittest.TestCase):
    """The YOLO regions paired with the OCR lines (docs/ocr_settings.md, "Regions")."""

    def test_region_at_picks_the_smallest_region_around_the_center(self):
        regions = [{"page": 1, "label": "title_block", "conf": 0.9, "box": [100, 100, 500, 400]},
                   {"page": 1, "label": "export_legend", "conf": 0.9, "box": [120, 120, 300, 200]},
                   {"page": 2, "label": "notes", "conf": 0.9, "box": [0, 0, 1000, 1000]}]
        self.assertEqual(ocr.region_at(regions, 1, [130, 130, 200, 150]), "export_legend")
        self.assertEqual(ocr.region_at(regions, 1, [350, 300, 450, 320]), "title_block")
        # a line that sticks out of a box but has its center inside is in it; one centered outside is not
        self.assertEqual(ocr.region_at(regions, 1, [60, 380, 400, 390]), "title_block")
        self.assertIsNone(ocr.region_at(regions, 1, [480, 390, 700, 400]))
        self.assertIsNone(ocr.region_at(regions, 1, [600, 600, 700, 620]))
        self.assertEqual(ocr.region_at(regions, 2, [600, 600, 700, 620]), "notes")
        self.assertIsNone(ocr.region_at(regions, 1, None))
        self.assertIsNone(ocr.region_at([], 1, [0, 0, 1, 1]))

    def test_regions_are_saved_in_the_line_boxes_pixels(self):
        # The detector ran on the cleaned picture (here upscaled 2x); the line boxes are in the
        # source picture's pixels, and so must the region boxes be.
        out = {"lines": [{"text": "CI-10442", "conf": 90.0, "page": 1, "bbox": [110, 110, 150, 120]},
                         {"text": "NOTES:", "conf": 90.0, "page": 1, "bbox": [10, 10, 40, 20]}], "settings": {}}
        found = [([{"label": "title_block", "conf": 0.91234, "box": [200.0, 200.0, 400.0, 300.0]}], 0.1)]
        ocr._attach_regions(out, found, [2.0])
        self.assertEqual(out["regions"], [{"page": 1, "label": "title_block", "conf": 0.912, "box": [100, 100, 200, 150]}])
        self.assertEqual(out["lines"][0]["region"], "title_block")
        self.assertNotIn("region", out["lines"][1])
        self.assertEqual(out["settings"]["layout"]["conf"], ocr.REGION_CONF)
        # the detector ran and found nothing: an empty list, which is not the same as "not run"
        out = {"lines": [], "settings": {}}
        ocr._attach_regions(out, [([], 0.1)], [1.0])
        self.assertEqual(out["regions"], [])

    def test_without_the_detector_the_result_is_what_it_was(self):
        lines = [{"text": "CI-10442", "conf": 90.0, "page": 1, "bbox": [110, 110, 150, 120]}]
        out = {"lines": json.loads(json.dumps(lines)), "settings": {"pages": []}}
        ocr._attach_regions(out, [None, None], [1.0, 1.0])
        self.assertEqual(out, {"lines": lines, "settings": {"pages": []}})
        with mock.patch.object(ocr, "OCR_REGIONS", False):
            self.assertIsNone(ocr._layout())
        with mock.patch.object(ocr, "Image", None):
            self.assertIsNone(ocr._layout())

    @unittest.skipUnless(layout is not None, "layout.py cannot be imported")
    def test_a_missing_model_file_means_no_regions(self):
        with mock.patch.object(layout, "MODEL_PATH", "/nonexistent/rfq_layout.onnx"):
            layout.reset()
            try:
                self.assertIsNone(ocr._layout())
            finally:
                layout.reset()
        self.assertEqual(ocr._layout() is not None, HAVE_LAYOUT)

    def test_a_broken_layout_module_means_no_regions(self):
        broken = mock.MagicMock()
        broken.available.side_effect = RuntimeError("damaged")
        with mock.patch.dict(sys.modules, {"layout": broken}):
            self.assertIsNone(ocr._layout())

    @unittest.skipUnless(HAVE_OCR, NO_OCR)
    def test_graceful_fallback_reads_the_same_text(self):
        # The fax is the quickest page. With the detector off (as on a host without the model or
        # onnxruntime) the result has no region keys at all; with it on, the words are the same.
        data = read(rel_of("E07/FR-2290_heatsink.pdf"))
        with mock.patch.object(ocr, "OCR_REGIONS", False):
            off = ocr.file_text(data, "pdf", "", effort="fast", timeout=900)
        self.assertEqual(off["method"], "ocr", off.get("error"))
        self.assertNotIn("regions", off)
        self.assertNotIn("layout", off["settings"])
        self.assertFalse([ln for ln in off["lines"] if "region" in ln])
        if not HAVE_LAYOUT:
            return
        on = ocr.file_text(data, "pdf", "", effort="fast", timeout=900)
        self.assertEqual(on["text"], off["text"])
        self.assertEqual([{k: v for k, v in ln.items() if k != "region"} for ln in on["lines"]], off["lines"])
        self.assertIn("title_block", {r["label"] for r in on["regions"]})
        self.assertTrue(any(ln.get("region") == "title_block" for ln in on["lines"]))

    @unittest.skipUnless(HAVE_OCR and HAVE_LAYOUT, NO_OCR + " and " + NO_LAYOUT)
    def test_regions_on_the_itar_scan(self):
        res = ocr_result(rel_of("E61/CI-10442_RevC.pdf"))
        labels = {r["label"] for r in res["regions"]}
        self.assertLessEqual({"title_block", "revision_block", "notes", "export_legend"}, labels)
        self.assertTrue(any(ln.get("region") == "title_block" for ln in res["lines"] if "10442" in ln["text"]))
        legend = " ".join(ln["text"] for ln in res["lines"] if ln.get("region") == "export_legend").upper()
        self.assertIn("ITAR", legend)
        self.assertEqual(res["settings"]["layout"]["model"], Path(layout.MODEL_PATH).name)
        w, h = 3300, 2550  # the scan's own pixels (300 dpi letter landscape), like the line boxes
        for r in res["regions"]:
            x0, y0, x1, y1 = r["box"]
            self.assertTrue(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, r)

    def test_old_cache_entries_without_regions_still_load(self):
        data = b"%PDF-1.4 an old cached scan"
        sha = hashlib.sha256(data).hexdigest()
        old = {"method": "ocr", "text": "CI-10442 REV C", "confidence": 88.0, "pages": 1,
               "lines": [{"text": "CI-10442", "conf": 90.0, "page": 1, "bbox": [1, 2, 3, 4]}],
               "settings": {"pages": [{"source": "scan"}]}, "seconds": 2.0, "error": None}
        path = Path(tempfile.mkdtemp(prefix="ocr-cache-test-")) / "cache.json"
        self.addCleanup(shutil.rmtree, path.parent, True)
        path.write_text(json.dumps({"tesseract_version": "5.3.4", "entries": {sha: old}}), encoding="utf-8")
        cache = ocr.OcrCache(path)
        with mock.patch.object(ocr, "_ocr", side_effect=AssertionError("OCR ran")):
            res = ocr.file_text(data, "pdf", cache=cache)
        self.assertEqual({k: v for k, v in res.items() if k not in ("sha256", "cached")}, old)
        sys.path.insert(0, str(ROOT))
        import rfq_details
        doc = rfq_details.Doc("CI-10442.pdf", "pdf", res)
        self.assertIsNone(doc.regions)
        self.assertEqual(doc.src("title_block"), "CI-10442.pdf (OCR 88%)")

    @unittest.skipUnless(TOOLS["pillow"], "needs Pillow")
    def test_region_reading_maps_words_back_to_the_page(self):
        # The "regions" recipe setting (measured, used by no recipe): a detected region is cut out
        # with a margin, scaled to its target dpi, read alone, and its word boxes are put back
        # into the page picture's pixels, where they merge with the page's own words.
        from PIL import Image
        g = Image.new("L", (1000, 800), 255)
        regions = [{"label": "title_block", "conf": 0.9, "box": [600.0, 500.0, 900.0, 700.0]},
                   {"label": "notes", "conf": 0.9, "box": [0.0, 0.0, 100.0, 100.0]}]
        recipe = {"psm": [3], "tess_dpi": True, "regions": ocr.parse_regions("title_block:6:600")}
        seen = []

        def fake(path, psm, dpi, rec, deadline, stats=None):
            with Image.open(path) as im:
                seen.append((psm, dpi, im.size))
            return [{"text": "CI-10442", "conf": 95.0, "box": [100, 60, 300, 100], "line": (psm, 1, 1, 1)}]
        tmp = tempfile.mkdtemp(prefix="ocr-region-test-")
        self.addCleanup(shutil.rmtree, tmp, True)
        with mock.patch.object(ocr, "_tesseract_words", side_effect=fake):
            passes = ocr._region_words(g, 300.0, regions, recipe, tmp, time.monotonic() + 60, "p1")
        pad = 9  # 0.03 inch at 300 dpi
        self.assertEqual(seen, [(6, 600.0, ((300 + 2 * pad) * 2, (200 + 2 * pad) * 2))])  # notes not asked for
        self.assertEqual(len(passes), 1)
        word = passes[0][0]
        self.assertEqual(word["box"], [591 + 50, 491 + 30, 591 + 150, 491 + 50])
        self.assertEqual(word["line"][0], "title_block0:6")
        page = [{"text": "Cl-1O442", "conf": 40.0, "box": [640, 520, 740, 540], "line": (3, 1, 1, 1)},
                {"text": "NOTES:", "conf": 95.0, "box": [10, 10, 60, 30], "line": (3, 1, 1, 2)}]
        merged = ocr._merge_region([dict(w) for w in page], passes, "conf")
        self.assertEqual([w["text"] for w in merged], ["CI-10442", "NOTES:"])
        replaced = ocr._merge_region([dict(w) for w in page], passes, "replace")
        self.assertEqual(sorted(w["text"] for w in replaced), ["CI-10442", "NOTES:"])
        self.assertEqual(ocr.parse_regions("title_block:6:600,line_table:4+6"),
                         {"title_block": {"psm": [6], "target_dpi": 600.0}, "line_table": {"psm": [4, 6]}})
        self.assertEqual(ocr.settings_text(recipe), "psm=3,regions=title_block:6:600,tess_dpi=1")

    @unittest.skipUnless(ocr.DEFAULT_CACHE.exists(), "the committed cache is not built")
    def test_committed_cache_has_regions_for_the_uncopyable_files(self):
        cache = ocr.OcrCache(ocr.DEFAULT_CACHE)
        self.assertEqual((cache.meta.get("regions") or {}).get("model"), "rfq_layout.onnx",
                         "rebuild the cache where the detector runs: python ocr.py --build-cache")
        by_file = {e.get("file"): e for e in cache.entries.values()}
        legends = {rel_of("E61/CI-10442_RevC.pdf"), rel_of("E52/AGI-3052_RevA.pdf"), rel_of("E72/WS-RFQ")}
        for rel in UNCOPYABLE:
            with self.subTest(rel=rel):
                e = by_file[rel]
                labels = {r["label"] for r in e["regions"]}
                kind = TRUTH[rel]["spec"]["kind"]
                want = {"title_block", "revision_block", "notes"} if kind == "drawing" else \
                    {"form_header", "line_table", "requirements"}
                self.assertLessEqual(want, labels)
                self.assertEqual("export_legend" in labels, rel in legends)
                self.assertEqual(e["settings"]["layout"]["model"], "rfq_layout.onnx")
                tagged = [ln for ln in e["lines"] if ln.get("region")]
                self.assertGreater(len(tagged), 10)
                for ln in e["lines"]:
                    self.assertEqual(ln.get("region"), ocr.region_at(e["regions"], ln["page"], ln["bbox"]))
        for rel, e in by_file.items():
            if e["method"] != "ocr":
                self.assertNotIn("regions", e, rel)


if __name__ == "__main__":
    unittest.main()
