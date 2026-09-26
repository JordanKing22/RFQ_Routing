"""
Tests for layout.py, the YOLO region detector: the model loads, it finds the title block on every
beta drawing, the export legends on the three restricted files, the line table on every RFQ form,
boxes stay inside the page, broken input gives [] instead of an exception, and one page takes well
under a few seconds on one thread.

    python -m unittest tests.test_layout -v

Skips when numpy, onnxruntime, or Pillow is missing (the demo runs without the detector). The PDF
tests also need pdftoppm. Which files are drawings and forms comes from tests/rfq_beta_truth.json;
tests may read it, layout.py never does.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.pop("RFQ_LAYOUT_THREADS", None)  # the tests check the default of one thread
os.environ.pop("RFQ_LAYOUT_MODEL", None)

import layout  # noqa: E402

HAVE_RUNTIME = all(importlib.util.find_spec(m) is not None for m in ("numpy", "onnxruntime", "PIL"))
NO_RUNTIME = "needs numpy, onnxruntime, and Pillow"
HAVE_PDFTOPPM = shutil.which("pdftoppm") is not None

DATA = ROOT / "data"
TRUTH = json.loads((ROOT / "tests" / "rfq_beta_truth.json").read_text(encoding="utf-8"))["files"]
DRAWINGS = sorted(rel for rel, t in TRUTH.items() if t["spec"]["kind"] == "drawing")
FORMS = {
    "E32": "rfq_beta/files/E32/RFQ-26-0931.pdf",      # office scan
    "E56": "rfq_beta/files/E56/RFQ-26-0317.pdf",      # office scan
    "E62": "rfq_beta/files/E62/RFQ-26-0318.pdf",      # copier scan
    "E72": "rfq_beta/files/E72/WS-RFQ-26-0388.pdf",   # office scan, ITAR
}
LEGENDS = {
    "E61": "rfq_beta/files/E61/CI-10442_RevC.pdf",    # ITAR drawing, office scan
    "E52": "rfq_beta/files/E52/AGI-3052_RevA.pdf",    # CUI drawing, fax
    "E72": "rfq_beta/files/E72/WS-RFQ-26-0388.pdf",   # ITAR RFQ form, office scan
}


def _page(rel: str):
    """(picture, detections) for page 1 of a beta file, the way the server sees it: PDFs through
    pdftoppm at 150 dpi, photos and screenshots as their own bytes. Cached: one inference per file."""
    if rel not in _CACHE:
        data = (DATA / rel).read_bytes()
        if rel.endswith(".pdf"):
            img = layout.render_pdf_page(data, 1, 150)
            dets = layout.detect(img) if img is not None else []
        else:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            img.load()
            dets = layout.detect(data)  # bytes path: JPEG draft decoding, PNG
        _CACHE[rel] = (img, dets)
    return _CACHE[rel]


_CACHE: dict = {}


def _labels(dets):
    return [d["label"] for d in dets]


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


@unittest.skipUnless(HAVE_RUNTIME, NO_RUNTIME)
class ModelTests(unittest.TestCase):
    def test_model_loads_with_the_expected_shape(self):
        info = layout.available()
        self.assertTrue(info["ready"], info["error"])
        self.assertTrue(info["loaded"])
        self.assertEqual(info["classes"], layout.CLASSES)
        sess = layout._session()
        self.assertEqual(list(sess.get_inputs()[0].shape), [1, 3, 640, 640])
        self.assertEqual(sess.get_outputs()[0].shape[1], 4 + len(layout.CLASSES))

    def test_model_file_is_small_enough_for_the_repository(self):
        size = os.path.getsize(layout.MODEL_PATH)
        self.assertLess(size, 12_500_000, f"{size / 1e6:.1f} MB")

    def test_classes_keep_their_names_and_order(self):
        self.assertEqual(layout.CLASSES, ["title_block", "revision_block", "notes", "export_legend",
                                          "proprietary_notice", "form_header", "line_table", "requirements"])

    def test_session_runs_on_one_thread_by_default(self):
        self.assertEqual(layout._threads(), 1)
        opts = layout._session().get_session_options()
        self.assertEqual(opts.intra_op_num_threads, 1)
        self.assertEqual(opts.inter_op_num_threads, 1)


@unittest.skipUnless(HAVE_RUNTIME and HAVE_PDFTOPPM, NO_RUNTIME + ", and pdftoppm")
class BetaPageTests(unittest.TestCase):
    def test_title_block_on_every_beta_drawing(self):
        self.assertEqual(len(DRAWINGS), 20)
        missing = []
        for rel in DRAWINGS:
            img, dets = _page(rel)
            blocks = [d for d in dets if d["label"] == "title_block"]
            if not blocks:
                missing.append(rel)
                continue
            # the title block sits in the lower right corner of every one of these sheets, including
            # the phone photo and the screenshot
            cx, cy = _center(blocks[0]["box"])
            self.assertGreater(cx, img.size[0] * 0.5, rel)
            self.assertGreater(cy, img.size[1] * 0.5, rel)
        self.assertEqual(missing, [])

    def test_revision_block_and_notes_on_every_beta_drawing(self):
        for rel in DRAWINGS:
            labels = _labels(_page(rel)[1])
            self.assertIn("revision_block", labels, rel)
            self.assertIn("notes", labels, rel)

    def test_export_legends_on_the_restricted_files(self):
        for email, rel in LEGENDS.items():
            img, dets = _page(rel)
            legends = [d for d in dets if d["label"] == "export_legend"]
            self.assertTrue(legends, f"{email} {rel}: {_labels(dets)}")
        # the ITAR warning box runs along the top of the E61 sheet and under the header on the E72 form
        for email in ("E61", "E72"):
            img, dets = _page(LEGENDS[email])
            top = min(_center(d["box"])[1] for d in dets if d["label"] == "export_legend")
            self.assertLess(top, img.size[1] * 0.3, email)

    def test_no_export_legend_on_unmarked_uncopyable_drawings(self):
        for rel in DRAWINGS:
            spec = TRUTH[rel]["spec"]
            if TRUTH[rel]["copyable"] or spec.get("legend") in ("itar", "ear", "cui"):
                continue
            self.assertNotIn("export_legend", _labels(_page(rel)[1]), rel)

    def test_line_table_and_header_on_every_rfq_form(self):
        for email, rel in FORMS.items():
            labels = _labels(_page(rel)[1])
            self.assertIn("line_table", labels, f"{email} {labels}")
            self.assertIn("form_header", labels, f"{email} {labels}")

    def test_forms_have_no_title_block(self):
        for rel in FORMS.values():
            self.assertNotIn("title_block", _labels(_page(rel)[1]), rel)

    def test_boxes_are_inside_the_picture_and_well_formed(self):
        for rel in sorted(set(DRAWINGS) | set(FORMS.values())):
            img, dets = _page(rel)
            w, h = img.size
            for d in dets:
                self.assertEqual(set(d), {"label", "conf", "box"}, rel)
                self.assertIn(d["label"], layout.CLASSES)
                self.assertGreaterEqual(d["conf"], 0.35)
                self.assertLessEqual(d["conf"], 1.0)
                x0, y0, x1, y1 = d["box"]
                self.assertTrue(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, (rel, d, img.size))
            confs = [d["conf"] for d in dets]
            self.assertEqual(confs, sorted(confs, reverse=True), rel)

    def test_detect_pdf_page_matches_detect_on_the_rendered_page(self):
        rel = LEGENDS["E61"]
        data = (DATA / rel).read_bytes()
        img, dets = _page(rel)
        self.assertEqual(img.size, (1650, 1275))  # letter landscape at 150 dpi
        self.assertEqual(layout.detect_pdf_page(data), dets)
        self.assertEqual(layout.detect_pdf_page(data, page=2), [])  # the scan has one page

    def test_bytes_and_picture_give_the_same_regions(self):
        rel = "rfq_beta/files/E71/LO-1186_RevA_screenshot.png"
        img, dets = _page(rel)
        again = layout.detect(img)
        self.assertEqual(_labels(again), _labels(dets))


@unittest.skipUnless(HAVE_RUNTIME, NO_RUNTIME)
class RobustnessTests(unittest.TestCase):
    def test_garbage_returns_an_empty_list(self):
        from PIL import Image
        tiny = io.BytesIO()
        Image.new("RGB", (1, 1), "white").save(tiny, "PNG")
        png_header_only = tiny.getvalue()[:40]
        # a PNG header that claims 40000 x 40000 pixels: refused before anything is decoded
        bomb = bytearray(tiny.getvalue())
        bomb[16:24] = (40000).to_bytes(4, "big") + (40000).to_bytes(4, "big")
        for bad in (b"", b"not an image at all", os.urandom(5000), b"%PDF-1.4 broken", tiny.getvalue(),
                    png_header_only, bytes(bomb), None, 12345, "path/that/does/not/exist.png", [1, 2, 3]):
            self.assertEqual(layout.detect(bad), [], repr(bad)[:40])
        for bad in (b"", b"%PDF-1.4 broken", b"GIF89a", None):
            self.assertEqual(layout.detect_pdf_page(bad), [], repr(bad)[:40])

    def test_blank_page_has_no_regions(self):
        from PIL import Image
        self.assertEqual(layout.detect(Image.new("RGB", (1275, 1650), "white")), [])

    def test_odd_modes_are_accepted(self):
        from PIL import Image
        img, _ = _page("rfq_beta/files/E71/LO-1186_RevA_screenshot.png")
        for mode in ("L", "1", "RGBA", "P"):
            dets = layout.detect(img.convert(mode))
            self.assertIn("title_block", _labels(dets), mode)

    def test_phone_exif_orientation_is_turned_upright(self):
        # A phone stores a portrait shot sideways and sets EXIF orientation 6; viewers turn it.
        from PIL import Image
        data = (DATA / "rfq_beta/files/E22/OPM-22817_RevB_photo.jpg").read_bytes()
        upright = layout.detect(data)
        side = Image.open(io.BytesIO(data)).transpose(Image.Transpose.ROTATE_90)
        exif = Image.Exif()
        exif[0x0112] = 6
        buf = io.BytesIO()
        side.save(buf, "JPEG", quality=90, exif=exif.tobytes())
        turned = layout.detect(buf.getvalue())
        self.assertEqual(_labels(turned), _labels(upright))
        for a, b in zip(turned, upright):
            self.assertLess(max(abs(x - y) for x, y in zip(a["box"], b["box"])), 8, (a, b))

    def test_missing_model_means_not_ready_and_no_regions(self):
        with mock.patch.object(layout, "MODEL_PATH", "/nonexistent/rfq_layout.onnx"):
            layout.reset()
            try:
                info = layout.available()
                self.assertFalse(info["ready"])
                self.assertIn("not found", info["error"])
                self.assertEqual(layout.detect((DATA / "rfq_beta/files/E71/LO-1186_RevA_screenshot.png").read_bytes()), [])
            finally:
                layout.reset()
        self.assertTrue(layout.available()["ready"])


@unittest.skipUnless(HAVE_RUNTIME, NO_RUNTIME)
class SpeedTests(unittest.TestCase):
    def test_one_page_on_one_thread_is_fast(self):
        # Measured about 0.2 s per page on one thread of this machine (docs/layout_model.md). The
        # bound is loose because the tests share the CPU with other work; it catches a model or a
        # thread setting that is an order of magnitude off, not small regressions.
        data = (DATA / "rfq_beta/files/E22/OPM-22817_RevB_photo.jpg").read_bytes()
        layout.detect(data)  # first run allocates
        times = []
        for _ in range(3):
            t = time.perf_counter()
            dets = layout.detect(data)
            times.append(time.perf_counter() - t)
        self.assertIn("title_block", _labels(dets))
        self.assertLess(statistics.median(times), 2.0, times)


@unittest.skipUnless(HAVE_RUNTIME and HAVE_PDFTOPPM, NO_RUNTIME + ", and pdftoppm")
class CliTests(unittest.TestCase):
    def test_cli_prints_regions_and_saves_a_picture(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.png"
            proc = subprocess.run([sys.executable, str(ROOT / "layout.py"), str(DATA / LEGENDS["E61"]),
                                   "--save", str(out)], capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("title_block", proc.stdout)
            self.assertIn("export_legend", proc.stdout)
            self.assertTrue(out.is_file() and out.stat().st_size > 10_000)
            proc = subprocess.run([sys.executable, str(ROOT / "layout.py"), str(DATA / LEGENDS["E61"]), "--json"],
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(proc.stdout)
            self.assertEqual(report["size"], [1650, 1275])
            self.assertIn("title_block", [d["label"] for d in report["detections"]])


if __name__ == "__main__":
    unittest.main()
