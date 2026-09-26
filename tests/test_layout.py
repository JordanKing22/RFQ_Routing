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
        # The re-encoded JPEG shifts confidences a little, so two regions with close scores can
        # swap places in the confidence order: compare region by region instead.
        self.assertEqual(sorted(_labels(turned)), sorted(_labels(upright)))
        self.assertEqual(len(set(_labels(upright))), len(upright))  # one region per class on this sheet
        by_label = {d["label"]: d for d in upright}
        for a in turned:
            b = by_label[a["label"]]
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
        # Measured 0.1 to 0.15 s per page on one thread of this machine (docs/layout_model.md). The
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


# --------------------------------------------------------------------------- #
# Hostile and unusual input, missing pieces, and parity with Ultralytics' pre- and post-processing
# --------------------------------------------------------------------------- #
SHOT = "rfq_beta/files/E71/LO-1186_RevA_screenshot.png"
PHOTO = "rfq_beta/files/E22/OPM-22817_RevB_photo.jpg"


def _encode(img, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def _png_header(w: int, h: int, depth: int = 8, color_type: int = 2) -> bytes:
    """A PNG that claims w x h pixels and has no pixel data at all."""
    import struct
    import zlib

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, depth, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


def _jpeg_claiming(data: bytes, w: int, h: int) -> bytes:
    """A real JPEG whose frame header is edited to claim w x h pixels."""
    b, i = bytearray(data), 2
    while i < len(b):
        marker, length = b[i + 1], int.from_bytes(b[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            b[i + 5:i + 7] = h.to_bytes(2, "big")
            b[i + 7:i + 9] = w.to_bytes(2, "big")
            return bytes(b)
        i += 2 + length
    raise ValueError("no frame header")


def _minimal_pdf(pages: int, width: float = 612, height: float = 792) -> bytes:
    """A small, valid PDF (correct xref offsets) with blank pages of the given size in points."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{' '.join(f'{3 + i} 0 R' for i in range(pages))}] /Count {pages} >>".encode()]
    objs += [f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] >>".encode()] * pages
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


# Peak memory of one detect() call in a fresh process, after the model is loaded and has run once,
# so the number is what the call itself adds.
MEMORY_CHILD = r"""
import json, resource, sys
sys.path.insert(0, sys.argv[1])
import layout
assert layout.available()["ready"]
layout.detect(open(sys.argv[2], "rb").read())
out = []
for path in sys.argv[3:]:
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    dets = layout.detect(open(path, "rb").read())
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out.append({"grew_mb": round((after - before) / 1024, 1), "labels": sorted(d["label"] for d in dets),
                "boxes": [[d["label"]] + d["box"] for d in dets]})
print(json.dumps({"peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1), "runs": out}))
"""


def _memory_runs(paths):
    proc = subprocess.run([sys.executable, "-c", MEMORY_CHILD, str(ROOT), str(DATA / SHOT)] + [str(p) for p in paths],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


@unittest.skipUnless(HAVE_RUNTIME, NO_RUNTIME)
class HostileInputTests(unittest.TestCase):
    def test_truncated_and_header_only_pictures(self):
        photo = (DATA / PHOTO).read_bytes()
        shot = (DATA / SHOT).read_bytes()
        for bad in (photo[: len(photo) // 2], photo[:600], shot[: len(shot) // 3], shot[:100]):
            self.assertEqual(layout.detect(bad), [])

    def test_huge_headers_are_refused_without_allocating(self):
        photo = (DATA / PHOTO).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            cases = {"png_40000": _png_header(40000, 40000), "png_rgba16_7700": _png_header(7700, 7700, 16, 6),
                     "png_1x60M": _png_header(1, 60_000_000), "jpeg_65000": _jpeg_claiming(photo, 65000, 65000)}
            paths = []
            for name, data in cases.items():
                paths.append(Path(tmp) / name)
                paths[-1].write_bytes(data)
            report = _memory_runs(paths)
        for name, run in zip(cases, report["runs"]):
            self.assertEqual(run["labels"], [], name)
            self.assertLess(run["grew_mb"], 40, name)

    def test_big_opaque_rgba_and_16_bit_pictures_shrink_without_full_size_copies(self):
        # Pillow reduces RGBA through a full-size premultiplied copy (a 36 MP picture: 144 MB more), and a
        # 16-bit scan used to be stretched to 8 bits at full size first. Both now shrink before any copy.
        import struct
        import zlib

        def png(w, h, depth, ctype, pixel):
            def chunk(kind, body):
                return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
            row = b"\x00" + pixel * w
            comp = zlib.compressobj(9)
            data = b"".join(comp.compress(row) for _ in range(h)) + comp.flush()
            ihdr = struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 0)
            return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", data) + chunk(b"IEND", b"")
        with tempfile.TemporaryDirectory() as tmp:
            rgba = Path(tmp) / "rgba.png"
            rgba.write_bytes(png(6000, 6000, 8, 6, b"\xf0\xf0\xf0\xff"))  # 36 MP, opaque, 144 MB decoded
            report = _memory_runs([rgba])
        self.assertEqual(report["runs"][0]["labels"], [])
        self.assertLess(report["runs"][0]["grew_mb"], 230, report)  # 275 to 290 MB before
        # the per-channel path gives what reducing the flattened picture gives
        from PIL import Image
        img = Image.open(DATA / SHOT).convert("RGBA")
        img = img.resize((img.size[0] * 2, img.size[1] * 2), Image.Resampling.NEAREST)
        self.assertEqual(layout._reduced(img, 3).tobytes(), img.convert("RGB").reduce(3).tobytes())
        # and a big 16-bit scan finds what its 8-bit twin finds, with boxes in its own pixels
        small = Image.open(DATA / SHOT).convert("L")
        big16 = small.resize((small.size[0] * 4, small.size[1] * 4), Image.Resampling.NEAREST).convert("I")
        big16 = big16.point(lambda v: v * 256).convert("I;16")
        want = {d["label"]: d["box"] for d in layout.detect(small)}
        got = layout.detect(_encode(big16, "PNG", compress_level=1))
        self.assertEqual(sorted(d["label"] for d in got), sorted(want))
        for d in got:
            for a, b in zip(d["box"], want[d["label"]]):
                self.assertLess(abs(a - 4 * b), 0.02 * big16.size[0], (d, want[d["label"]]))

    def test_relative_model_path_survives_a_change_of_directory(self):
        # The session loads lazily; a relative RFQ_LAYOUT_MODEL must still point at the same file afterwards.
        code = ("import os, sys; sys.path.insert(0, os.getcwd()); import layout; os.chdir('/'); "
                "print(layout.available()['ready'], layout.MODEL_PATH)")
        env = dict(os.environ, RFQ_LAYOUT_MODEL=os.path.join("models", "rfq_layout.onnx"))
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True,
                              timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        ready, path = proc.stdout.split()
        self.assertEqual(ready, "True", proc.stdout)
        self.assertEqual(path, str(ROOT / "models" / "rfq_layout.onnx"))

    def test_big_picture_is_shrunk_before_the_color_conversion(self):
        # A 5760 x 4680 greyscale PNG (27 MP): converting it to RGB at full size would add about
        # 110 MB; box-averaging it down first keeps the call near the size of the decoded picture.
        from PIL import Image
        small = Image.open(DATA / SHOT).convert("L")
        big = small.resize((small.size[0] * 4, small.size[1] * 4), Image.Resampling.NEAREST)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.png"
            big.save(path, compress_level=1)
            report = _memory_runs([path])
        run = report["runs"][0]
        self.assertLess(run["grew_mb"], 80, run)
        # and the boxes come back in the big picture's pixels: four times the small page's boxes
        want = {d["label"]: d["box"] for d in layout.detect(small)}
        self.assertEqual(sorted(want), run["labels"])
        for label, *box in run["boxes"]:
            for got, ref in zip(box, want[label]):
                self.assertLess(abs(got - 4 * ref), 0.02 * big.size[0], (label, box, want[label]))

    def test_every_pillow_mode_and_common_format_reads(self):
        from PIL import Image
        img = Image.open(DATA / SHOT)
        img.load()
        i16 = img.convert("I").point(lambda v: v * 256).convert("I;16")  # a 16-bit scan: 0 to 65535
        pictures = {"L": img.convert("L"), "1": img.convert("1"), "RGBA": img.convert("RGBA"),
                    "P": img.convert("P"), "LA": img.convert("LA"), "CMYK": img.convert("CMYK"),
                    "I;16": i16, "F": img.convert("F"), "YCbCr": img.convert("YCbCr")}
        for mode, pic in pictures.items():
            self.assertIn("title_block", _labels(layout.detect(pic)), mode)
        files = {"16-bit PNG": _encode(i16, "PNG"), "CMYK JPEG": _encode(img.convert("CMYK"), "JPEG", quality=90),
                 "1-bit PNG": _encode(img.convert("1"), "PNG"), "grey JPEG": _encode(img.convert("L"), "JPEG"),
                 "RGBA PNG": _encode(img.convert("RGBA"), "PNG"), "TIFF": _encode(img, "TIFF"),
                 "WEBP": _encode(img, "WEBP"), "BMP": _encode(img, "BMP"), "GIF": _encode(img.convert("P"), "GIF")}
        for name, data in files.items():
            self.assertIn("title_block", _labels(layout.detect(data)), name)
        # a 16-bit scan reads like its 8-bit twin (a clipping conversion would turn it white)
        grey = sorted(_labels(layout.detect(img.convert("L"))))
        self.assertEqual(sorted(_labels(layout.detect(i16))), grey)
        self.assertEqual(sorted(_labels(layout.detect(files["16-bit PNG"]))), grey)

    def test_only_plain_raster_formats_are_opened(self):
        # Pillow would pass an EPS file to Ghostscript; detect() never offers bytes to that plugin.
        from PIL import Image, UnidentifiedImageError
        eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 100 100\nshowpage\n%%EOF\n"
        with self.assertRaises(UnidentifiedImageError):
            Image.open(io.BytesIO(eps), formats=layout.IMAGE_FORMATS)
        self.assertNotIn("EPS", layout.IMAGE_FORMATS)
        self.assertEqual(layout.detect(eps), [])

    def test_damaged_model_file_means_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "rfq_layout.onnx"
            bad.write_bytes(b"\x08\x07" + os.urandom(4000))
            with mock.patch.object(layout, "MODEL_PATH", str(bad)):
                layout.reset()
                try:
                    info = layout.available()
                    self.assertFalse(info["ready"])
                    self.assertTrue(info["error"])
                    self.assertEqual(layout.detect((DATA / SHOT).read_bytes()), [])
                finally:
                    layout.reset()
        self.assertTrue(layout.available()["ready"])

    def test_missing_onnxruntime_means_not_ready(self):
        code = ("import json, sys; sys.modules['onnxruntime'] = None; sys.path.insert(0, sys.argv[1]); import layout; "
                "info = layout.available(); data = open(sys.argv[2], 'rb').read(); "
                "print(json.dumps({'ready': info['ready'], 'error': info['error'], 'dets': layout.detect(data)}))")
        proc = subprocess.run([sys.executable, "-c", code, str(ROOT), str(DATA / SHOT)], capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.count("\n"), 1, "layout.py must not print")
        out = json.loads(proc.stdout)
        self.assertFalse(out["ready"])
        self.assertIn("onnxruntime", out["error"])
        self.assertEqual(out["dets"], [])


@unittest.skipUnless(HAVE_RUNTIME, NO_RUNTIME)
class PdfTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_PDFTOPPM, "needs pdftoppm")
    def test_pdf_without_pages_or_past_the_last_page(self):
        self.assertIsNone(layout.render_pdf_page(_minimal_pdf(0)))
        self.assertEqual(layout.detect_pdf_page(_minimal_pdf(0)), [])
        self.assertIsNone(layout.render_pdf_page(_minimal_pdf(1), page=3))
        blank = layout.render_pdf_page(_minimal_pdf(1), dpi=100)
        self.assertEqual(blank.size, (850, 1100))
        self.assertEqual(layout.detect(blank), [])

    @unittest.skipUnless(HAVE_PDFTOPPM and shutil.which("pdfinfo"), "needs pdftoppm and pdfinfo")
    def test_huge_pdf_page_renders_at_a_lower_dpi(self):
        # A 200 x 200 inch MediaBox at 150 dpi would be a 30000 x 30000 bitmap (2.7 GB).
        img = layout.render_pdf_page(_minimal_pdf(1, 14400, 14400), dpi=150)
        self.assertIsNotNone(img)
        self.assertLessEqual(img.size[0] * img.size[1], layout.MAX_PDF_PIXELS * 1.01)
        # pdftoppm's peak memory, measured from a small fresh Python process. Measured from this one,
        # RUSAGE_CHILDREN would also count this whole test process: a child started with vfork shares
        # the parent's pages until it runs pdftoppm, and by now the parent holds the model and pages.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "huge.pdf"
            src.write_bytes(_minimal_pdf(1, 14400, 14400))
            code = ("import json, resource, sys; sys.path.insert(0, sys.argv[1]); import layout; "
                    "img = layout.render_pdf_page(open(sys.argv[2], 'rb').read(), dpi=150); "
                    "print(json.dumps({'size': list(img.size) if img else None, "
                    "'child_mb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024}))")
            proc = subprocess.run([sys.executable, "-c", code, str(ROOT), str(src)], capture_output=True,
                                  text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(report["size"], list(img.size))
        self.assertLess(report["child_mb"], 400, report)

    @unittest.skipUnless(HAVE_PDFTOPPM, "needs pdftoppm")
    def test_huge_pdf_page_without_pdfinfo_is_cropped(self):
        # Without pdfinfo the page size is unknown, so pdftoppm is told to render at most a
        # 5000 x 5000 pixel area; uncropped, this page is a 30000 x 30000 bitmap.
        real = shutil.which
        with mock.patch.object(layout.shutil, "which", side_effect=lambda n: None if n == "pdfinfo" else real(n)):
            img = layout.render_pdf_page(_minimal_pdf(1, 14400, 14400), dpi=150)
            self.assertEqual(img.size, (5000, 5000))
            # ordinary pages are not affected by the crop area
            page = layout.render_pdf_page((DATA / LEGENDS["E61"]).read_bytes())
            self.assertEqual(page.size, (1650, 1275))
            self.assertIn("title_block", _labels(layout.detect(page)))

    def test_missing_pdftoppm(self):
        data = (DATA / LEGENDS["E61"]).read_bytes()
        with mock.patch.object(layout.shutil, "which", return_value=None):
            self.assertFalse(layout.available()["pdftoppm"])
            self.assertIsNone(layout.render_pdf_page(data))
            self.assertEqual(layout.detect_pdf_page(data), [])
            self.assertIn("title_block", _labels(layout.detect((DATA / SHOT).read_bytes())))  # pictures still work


class _FakeSession:
    """Stands in for onnxruntime: returns a fixed head output and remembers the input tensor."""

    def __init__(self, head):
        self.head, self.seen = head, None

    def get_inputs(self):
        import types
        return [types.SimpleNamespace(name="images", shape=[1, 3, 640, 640])]

    def run(self, _names, feeds):
        self.seen = feeds["images"]
        return [self.head]


def _head(rows):
    """A YOLO11 head output [1, 4 + classes, N] from (cx, cy, w, h, class, score) rows in 640 space."""
    import numpy as np
    out = np.zeros((1, 4 + len(layout.CLASSES), max(1, len(rows))), np.float32)
    for j, (cx, cy, w, h, c, s) in enumerate(rows):
        out[0, :4, j] = (cx, cy, w, h)
        out[0, 4 + c, j] = s
    return out


def _ultralytics_letterbox(w: int, h: int):
    """Ultralytics 8.4 LetterBox(new_shape=640, auto=False, scaleup=True, center=True), written out:
    the scale, the resized size, and the top-left padding (verified against the real class for these
    sizes when the model was exported; see docs/layout_model.md)."""
    r = min(640 / h, 640 / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (640 - nw) / 2, (640 - nh) / 2
    return r, nw, nh, int(round(dw - 0.1)), int(round(dh - 0.1))


SIZES = {"letter portrait 150 dpi": (1275, 1650), "server portrait": (1391, 1800), "server landscape": (1800, 1391),
         "4:3 photo": (1800, 1350), "12 MP phone": (4032, 3024), "screenshot": (1440, 1170),
         "1366 x 768 screen": (1366, 768), "odd": (641, 479), "small, scaled up": (100, 80), "strip": (5000, 300)}


@unittest.skipUnless(HAVE_RUNTIME, NO_RUNTIME)
class ParityTests(unittest.TestCase):
    def test_letterbox_matches_ultralytics(self):
        import numpy as np
        from PIL import Image
        for name, (w, h) in SIZES.items():
            x, r, left, top = layout._letterbox(Image.new("RGB", (w, h), "white"))
            r2, nw, nh, left2, top2 = _ultralytics_letterbox(w, h)
            self.assertEqual((left, top), (left2, top2), name)
            self.assertAlmostEqual(r, r2, 9, name)
            self.assertEqual(x.shape, (1, 3, 640, 640))
            content = np.argwhere(np.abs(x[0, 0] - 114 / 255) > 1e-6)
            self.assertEqual(tuple(content.min(0)), (top, left), name)
            self.assertEqual(tuple(content.max(0) + 1), (top + nh, left + nw), name)

    def test_boxes_map_back_to_the_callers_pixels(self):
        from PIL import Image
        for name, (w, h) in SIZES.items():
            if min(w, h) < 200:
                continue
            want = [0.2 * w, 0.3 * h, 0.6 * w, 0.5 * h]
            r, nw, nh, left, top = _ultralytics_letterbox(w, h)
            x0, y0, x1, y1 = (want[0] * r + left, want[1] * r + top, want[2] * r + left, want[3] * r + top)
            fake = _FakeSession(_head([((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 2, 0.9)]))
            picture = Image.new("RGB", (w, h), "white")
            inputs = {"PIL image": picture, "JPEG bytes": _encode(picture, "JPEG")}
            with mock.patch.dict(layout._state, {"session": fake, "tried": True, "error": None}):
                for kind, image in inputs.items():
                    dets = layout.detect(image)
                    self.assertEqual(len(dets), 1, (name, kind))
                    self.assertEqual(dets[0]["label"], "notes")
                    # JPEG draft decoding reads a 12 MP photo at half size, so allow a pixel of rounding
                    tol = 0.6 if kind == "PIL image" else 2.5 * w / 1280
                    for got, ref in zip(dets[0]["box"], want):
                        self.assertLess(abs(got - ref), tol, (name, kind, dets[0]["box"], want))

    def test_boxes_are_clipped_to_the_picture(self):
        from PIL import Image
        fake = _FakeSession(_head([(5, 320, 60, 100, 0, 0.9)]))  # sticks out past the left edge
        with mock.patch.dict(layout._state, {"session": fake, "tried": True, "error": None}):
            dets = layout.detect(Image.new("RGB", (1800, 1391), "white"))
        self.assertEqual(dets[0]["box"][0], 0.0)

    def test_nms_is_class_wise_greedy_and_strict_on_the_threshold(self):
        rows = [(100, 100, 80, 40, 0, 0.90),   # title block
                (100, 100, 80, 40, 3, 0.80),   # the same box as an export legend: kept, other class
                (104, 100, 80, 40, 0, 0.70),   # overlaps the first title block at IoU 0.9: suppressed
                (160, 100, 80, 40, 0, 0.60),   # IoU 0.14 with the first: kept
                (400, 400, 50, 50, 6, 0.35),   # exactly at the threshold: dropped, as Ultralytics does
                (400, 500, 50, 50, 6, 0.36)]
        found = layout._decode(_head(rows), 0.35, 0.5)
        self.assertEqual([(d["label"], round(d["conf"], 2)) for d in found],
                         [("title_block", 0.9), ("export_legend", 0.8), ("title_block", 0.6), ("line_table", 0.36)])

    def test_model_file_carries_the_class_names_in_order(self):
        import ast
        meta = layout._session().get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"])
        self.assertEqual([names[i] for i in sorted(names)], layout.CLASSES)
        self.assertEqual(ast.literal_eval(meta["imgsz"]), [640, 640])


def _raster_font_available() -> bool:
    try:
        import docgen
        docgen._font(False, 12)
        return True
    except Exception:  # no Pillow or no TrueType font: the dataset builder cannot run here either
        return False


@unittest.skipUnless(importlib.util.find_spec("PIL") is not None and _raster_font_available(),
                     "needs Pillow and a TrueType font (Liberation Sans or DejaVu Sans)")
class LabelTests(unittest.TestCase):
    """The training boxes come from page operations (tools/make_layout_dataset.py). A text line's box
    must hold the ink docgen.to_image really draws, which uses a whole number of pixels per font size."""

    def test_text_boxes_hold_the_rendered_ink(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import docgen
        import make_layout_dataset as mld
        from PIL import ImageOps
        line = "MATERIAL AND FINISH PER DRAWING FR-1904 REV C. MATERIAL CERTS AND C OF C REQUIRED WITH SHIPMENT."
        # (size, dpi, supersample, anchor, x): 7 pt at 150 dpi is a 15 px font (2.9% wide), 8 pt at 110 dpi
        # with 2x supersampling a 24 px font (1.8% narrow)
        for size, dpi, ss, anchor, x in ((7, 150, 1, "start", 20), (7, 180, 1, "start", 20), (8, 110, 2, "start", 20),
                                         (6, 150, 1, "end", 590), (7.5, 200, 1, "middle", 306), (9, 300, 1, "start", 20)):
            page = docgen.Page(612, 120)
            page.text(x, 60, line, size, False, anchor)
            ink = ImageOps.invert(docgen.to_image(page, dpi, supersample=ss).convert("L")).point(
                lambda v: 255 if v > 60 else 0).getbbox()
            box = [v * dpi / 72.0 for v in mld.text_box(page.ops[-1][1], dpi / 72.0 * ss)]
            slack = 0.02 * (ink[2] - ink[0]) + 2
            case = (size, dpi, ss, anchor, ink, [round(v, 1) for v in box])
            self.assertLessEqual(box[0], ink[0] + 1, case)
            self.assertGreaterEqual(box[2], ink[2] - 1, case)
            self.assertGreater(box[0], ink[0] - slack, case)
            self.assertLess(box[2], ink[2] + slack, case)


if __name__ == "__main__":
    unittest.main()
