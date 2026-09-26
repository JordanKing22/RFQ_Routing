"""
Text from any attachment, including the ones whose text cannot be copied: scanned PDFs, faxes,
a phone photo of a print, a screenshot of a PDF viewer. Standard library plus optional Pillow.
It runs two programs: tesseract (OCR) and pdftoppm (PDF pages to pictures).
On Debian or Ubuntu: apt install tesseract-ocr poppler-utils.

    file_text(data, media, name)  -> the document text and how it was read
    OcrCache                      -> saved results keyed by the SHA-256 of the file bytes
    available()                   -> which of the programs are installed

PDFs with a real text layer are read with pypdf (attachments.extract_pdf_text), so only scans
are OCR'd. STEP files give their header and PRODUCT entities. Every failure comes back as an
"error" string instead of an exception, because this runs inside the demo server.

The OCR settings were chosen by measurement against tests/rfq_beta_truth.json. What was tried,
the scores, and the time per page are in docs/ocr_settings.md.

    python ocr.py FILE [FILE ...]            print method, confidence, seconds, and the text
    python ocr.py --build-cache              OCR the beta files into data/rfq_beta/ocr_cache.json
    python ocr.py --evaluate [--settings X]  score settings against tests/rfq_beta_truth.json
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE / "data" / "rfq_beta" / "ocr_cache.json"
BETA_FILES = HERE / "data" / "rfq_beta" / "files"
TRUTH_FILE = HERE / "tests" / "rfq_beta_truth.json"

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageStat
except ImportError:  # Pillow is optional: without it OCR still runs, with no clean-up steps
    Image = ImageChops = ImageDraw = ImageFilter = ImageOps = ImageStat = None  # type: ignore[assignment]

# One OCR job at a time by default: Render's free plan has about 0.1 CPU, and two tesseract
# processes there only make both of them slower. Raise it on a real machine.
OCR_WORKERS = max(1, int(os.environ.get("RFQ_OCR_WORKERS", "1") or 1))
_ocr_slots = threading.BoundedSemaphore(OCR_WORKERS)
# Seconds for one whole file. A letter page takes a few seconds on one desktop core and about
# ten times that on 0.1 CPU, so the default leaves room for a slow host.
OCR_TIMEOUT = float(os.environ.get("RFQ_OCR_TIMEOUT", "300") or 300)
OCR_MAX_PAGES = max(1, int(os.environ.get("RFQ_OCR_MAX_PAGES", "6") or 6))
# Optional folder with another eng.traineddata (for example tessdata_best); see docs/ocr_settings.md.
OCR_TESSDATA = os.environ.get("RFQ_OCR_TESSDATA", "").strip() or None
# The biggest picture handed to tesseract. A letter page at 400 dpi is 15 million pixels.
MAX_OCR_PIXELS = 36_000_000
# Refuse to decode pictures bigger than this at all (a small PNG can claim a huge size). An
# 11 x 17 in drawing at 600 dpi (67 million pixels, 1-bit) got a used server killed on 512 MB.
MAX_INPUT_PIXELS = 40_000_000
# And bigger than this once decoded (Pillow keeps RGB, RGBA, and CMYK at 4 bytes a pixel). The
# clean-up adds about half that again, and Tesseract runs beside it. The budget is for a server
# that has been used, not an idle one: with the viewer's detector loaded and its page caches
# full the server sits at 200 to 250 MB, and a 48 megapixel phone photo read whole (192 MB
# decoded) then took it to 490 to 520 MB, 690 with Tesseract. A JPEG over this or
# MAX_INPUT_PIXELS is decoded at a half, a quarter, or an eighth of its size instead, which the
# JPEG decoder does for free: that photo is read at 4000 x 3000 (still above the photo recipe's
# 300 dpi on a letter page) and peaks at 265 MB, 362 with Tesseract. A PNG over 25 megapixels in
# color is refused (docs/ocr_settings.md, "Very large pages").
MAX_INPUT_BYTES = 100_000_000
# A scan finer than this is rendered down to it: on 600 dpi office scans of the held-out
# drawings, 300 dpi found 19 of 20 key fields against 17 at 600 or 400 dpi, for less than half
# the time (docs/ocr_settings.md, "Review").
MAX_RASTER_DPI = 300.0
# A PDF whose text layer has fewer letters and digits than this per page is treated as a scan.
# Scanner software sometimes adds a line like "Scanned by ScanDesk"; that is not the document.
TEXT_LAYER_MIN_CHARS = 40
# A longer stamp still marks a scan when one picture covers most of the page: a page with
# fewer letters and digits than this and a picture over this share of it is OCR'd. The typed
# drawings and forms of the beta set have 714 to 1189 letters and digits a page.
SCAN_STAMP_MAX_CHARS = 200
SCAN_PICTURE_COVER = 0.6
# Long side of the page, in inches, used to estimate the resolution of a photo or screenshot
# (letter and A4 landscape drawings and forms are 11 to 11.7 in).
PAGE_LONG_SIDE_IN = 11.0
# Regions: when layout.py's YOLO model can run here (numpy, onnxruntime, Pillow, and
# models/rfq_layout.onnx), every OCR'd page also goes through the detector, on the very picture
# tesseract reads, so its boxes and the line boxes share pixels. Each line is tagged with the
# region its center falls in (title block, export legend, line table, ...) and the regions are
# saved with the result, so rfq_details can say where a value was printed. It costs about 0.1
# CPU s a page (about 1 s on a 0.1 CPU host) and changes no text: docs/ocr_settings.md,
# "Regions". RFQ_OCR_REGIONS=0 turns it off; without the model it is off, and results are
# exactly what they were before regions.
OCR_REGIONS = os.environ.get("RFQ_OCR_REGIONS", "1").strip().lower() not in ("0", "false", "no", "off")
# layout.detect's own default: 48 of 48 regions on the 13 uncopyable files and no false box,
# found on the OCR raster as well as on the viewer's page image (docs/ocr_settings.md, "Regions").
REGION_CONF = 0.35

_tool_info: Dict[str, Any] = {}
_tool_lock = threading.Lock()


class OcrError(Exception):
    """A failure that becomes the result's "error" string."""


# --------------------------------------------------------------------------- #
# What is installed
# --------------------------------------------------------------------------- #
def _tools() -> Dict[str, Any]:
    with _tool_lock:
        if _tool_info:
            return _tool_info
        # RFQ_TESSERACT points at another build (the settings were checked on 5.3.0 and 5.5.0
        # builds as well as the 5.3.4 installed here; the Docker image, python:3.12-slim on
        # Debian trixie, installs 5.5.0).
        tess = os.environ.get("RFQ_TESSERACT", "").strip() or shutil.which("tesseract")
        version = None
        params: set = set()
        if tess:
            try:
                proc = subprocess.run([tess, "--version"], capture_output=True, timeout=20)
                out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
                m = re.search(r"tesseract\s+v?(\d+\.\d+(?:\.\d+)?)", out)
                version = m.group(1) if m else None
                if not version:
                    tess = None
                else:
                    proc = subprocess.run([tess, "--print-parameters"], capture_output=True, timeout=20)
                    for line in proc.stdout.decode("utf-8", "replace").splitlines()[1:]:
                        name = line.split("\t", 1)[0].strip()
                        if name:
                            params.add(name)
            except (OSError, subprocess.SubprocessError):
                tess = None
        _tool_info.update(tesseract=tess, tesseract_version=version, tesseract_params=params,
                          pdftoppm=shutil.which("pdftoppm"), pdfimages=shutil.which("pdfimages"),
                          pdftotext=shutil.which("pdftotext"))
        return _tool_info


def _version_tuple(version: Optional[str]) -> Tuple[int, ...]:
    try:
        return tuple(int(p) for p in (version or "0").split("."))
    except ValueError:
        return (0,)


# The oldest Tesseract that has each -c variable this module sets. All of them are in 5.3.0
# (Debian bookworm) and 5.5.0 (Debian trixie, which python:3.12-slim, the Render image, is
# built on); the check keeps an older or stripped build from failing the whole page over one
# unknown variable.
_PARAM_SINCE = {"thresholding_method": (5, 0), "preserve_interword_spaces": (3, 4),
                "load_system_dawg": (3, 0), "load_freq_dawg": (3, 0), "tessedit_create_tsv": (3, 5)}


def _supports(param: str) -> bool:
    t = _tools()
    if t["tesseract_params"]:
        return param in t["tesseract_params"]
    return _version_tuple(t["tesseract_version"]) >= _PARAM_SINCE.get(param, (99,))


def available() -> Dict[str, Any]:
    t = _tools()
    return {"tesseract": bool(t["tesseract"]), "tesseract_version": t["tesseract_version"],
            "pdftoppm": bool(t["pdftoppm"]), "pillow": Image is not None, "ocr": bool(t["tesseract"])}


def _layout() -> Any:
    """layout.py when its region detector can run here, else None. Asked only when a page is
    about to be OCR'd, so a text-layer PDF never loads numpy and onnxruntime; a missing or broken
    layout.py only means no regions."""
    if not OCR_REGIONS or Image is None:
        return None
    try:
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import layout
        return layout if layout.available().get("ready") else None
    except Exception:  # noqa: BLE001 - no regions is never a reason to fail a page
        return None


def _env() -> Dict[str, str]:
    env = dict(os.environ)
    # Tesseract's OpenMP threads fight each other on a fraction of a CPU; one thread is as fast
    # per page there, and here the evaluation runs pages side by side instead.
    env["OMP_THREAD_LIMIT"] = "1"
    return env


def _run(cmd: List[str], deadline: float, what: str, data: Optional[bytes] = None) -> bytes:
    left = deadline - time.monotonic()
    if left <= 0:
        raise OcrError(f"{what} ran out of time")
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, timeout=left, env=_env())
    except subprocess.TimeoutExpired:
        # The limit is per file (RFQ_OCR_TIMEOUT, or the caller's timeout), shared by every step.
        raise OcrError(f"{what} took too long: the time limit for reading this file ran out")
    except OSError as exc:
        raise OcrError(f"could not start {what} ({exc})")
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no message"]
        raise OcrError(f"{what} failed ({tail[0][:160]})")
    return proc.stdout


# --------------------------------------------------------------------------- #
# The settings
# --------------------------------------------------------------------------- #
# A recipe is a dict of these keys (missing keys are off):
#   raster_dpi    pdftoppm resolution; "native" renders a scanned page at its own resolution
#   target_dpi    upscale (Lanczos) a low-resolution source until the text is this dense;
#                 never downscales
#   scale         a fixed resize factor instead of target_dpi (0 means "use target_dpi")
#   color         keep the colors of a photo or screenshot instead of converting to gray
#   median        median filter size (3 removes the 1-pixel salt noise of a fax line)
#   flatten       remove uneven light: subtract the blurred paper level
#   invert        turn dark bands with light text (form table headers) into dark on light
#   autocontrast  stretch the gray levels
#   unsharp       unsharp mask after resizing
#   deskew        find the skew with a projection profile and rotate it out
#   page          find the sheet of paper in a photo or screenshot and warp it flat
#   psm           tesseract page segmentation modes; with two, the passes are merged
#   merge         how a second pass joins the first: "fill" adds its words where the first
#                 pass found nothing; "conf" also swaps in a word read with clearly higher
#                 confidence at the same place
#   threshold     tesseract thresholding_method (0 Otsu, 1 adaptive Otsu, 2 Sauvola)
#   tess_dpi      tell tesseract the resolution of the picture it gets
#   pis           -c preserve_interword_spaces=1
#   nodict        -c load_system_dawg=0 -c load_freq_dawg=0; with --oem 1 this changes
#                 nothing (see _tesseract_words), kept so the search log stays readable
#   repair        uppercase an l that sits in an all-caps line (Helvetica I and l look alike)
#   orient        when the first reading looks sideways or upside down, ask tesseract's
#                 orientation detection and read the turned page (on unless set to False)
#   min_conf      drop words tesseract is less sure of than this (drawing line work read
#                 as letters), except words with a digit and lone capitals (_keep_word)
#   regions       a second reading of each detected region named here, cut out and read on
#                 its own: {"title_block": {"psm": [6], "target_dpi": 600}} (psm, and scale or
#                 target_dpi for the crop); needs the region detector (layout.py)
#   region_merge  how that reading joins the page's: "conf" or "fill" as for two passes, or
#                 "replace" (inside the region the reading with the higher mean confidence wins)
#                 No recipe uses regions: every crop reading tried cost 12 to 29 % more time for
#                 no field the extractor gets right (docs/ocr_settings.md, "Regions").
# The per-source recipes below are the measured winners: docs/ocr_settings.md has every setting
# tried, the scores, and the time per page. Change a value there and here together, and rerun
# python ocr.py --evaluate. BASELINE is plain tesseract on a 300 dpi rendering, for comparison.
BASELINE: Dict[str, Any] = {"raster_dpi": 300, "psm": [3], "raw": True}

RECIPES: Dict[str, Dict[str, Any]] = {
    # Office scans (300 dpi gray): read at the scan's own resolution. invert turns the dark
    # header bands of the RFQ form tables light; repair (Helvetica's I read as l, a lone capital
    # read as "Cc") found three more fields; min_conf drops the junk words read from line work.
    # The page layout pass (psm 3) comes first, and a sparse text pass (psm 11) adds the title
    # block and callout words it missed: the same fields, but word recall up on every page, for
    # 1.8 times the time.
    "scan": {"raster_dpi": "native", "invert": True, "psm": [3, 11], "merge": "conf", "repair": True,
             "min_conf": 60, "tess_dpi": True},
    # Copier scans (200 dpi gray, darker at one edge): even out the light, then upsample to
    # 400 dpi; psm 4 (one column of lines) keeps the form's table rows together. invert (run
    # before flatten, which would otherwise erase the band) reads the first row of the E62
    # form, the last field the search left on the real files; it finds no band on drawings.
    "lowres": {"raster_dpi": "native", "flatten": True, "invert": True, "target_dpi": 400, "psm": [4],
               "repair": True, "tess_dpi": True},
    # Faxes (200 dpi, 1 bit): no resize (upsampling a 1-bit page lost fields), and two passes:
    # sparse text (psm 11) finds the title block cells, the page layout pass (psm 3) fills in,
    # and a word read with clearly higher confidence at the same place replaces the first.
    "bilevel": {"raster_dpi": "native", "psm": [11, 3], "merge": "conf", "repair": True, "tess_dpi": True},
    # Phone photo: find the sheet and warp it flat at 300 dpi, then the same two passes.
    "photo": {"page": True, "target_dpi": 300, "psm": [11, 3], "merge": "conf", "repair": True,
              "min_conf": 60, "tess_dpi": True},
    # Viewer screenshot (about 120 dpi, anti-aliased): upsample to 300 dpi, sparse text. Cropping
    # to the page (page) read cleaner but lost fields on the held-out screenshots.
    "screen": {"target_dpi": 300, "psm": [11], "repair": True, "min_conf": 60, "tess_dpi": True},
}
# Live uploads on a slow host: one tesseract pass per page, the best single pass measured for
# each type. Where the best recipe is already one pass it is used as it is.
FAST_RECIPES: Dict[str, Dict[str, Any]] = {
    "scan": {"raster_dpi": "native", "invert": True, "psm": [3], "repair": True, "min_conf": 60, "tess_dpi": True},
    "lowres": dict(RECIPES["lowres"]),
    "bilevel": {"raster_dpi": "native", "psm": [3], "repair": True, "tess_dpi": True},
    "photo": {"page": True, "target_dpi": 300, "psm": [3], "repair": True, "min_conf": 60, "tess_dpi": True},
    "screen": dict(RECIPES["screen"]),
}
SOURCE_TYPES = tuple(RECIPES)


# --------------------------------------------------------------------------- #
# Pictures
# --------------------------------------------------------------------------- #
def _decoded_bytes(mode: str, size: Tuple[int, int]) -> int:
    """Memory Pillow uses for a picture: 1 byte a pixel for 1-bit, gray, and palette pictures,
    2 for 16-bit gray, 4 for everything else (RGB is stored padded to 4)."""
    per = 1 if mode in ("1", "L", "P") else 2 if mode.startswith("I;16") else 4
    return size[0] * size[1] * per


def _open_image(data: bytes) -> "Image.Image":
    if Image is None:
        raise OcrError("Pillow is not installed")
    try:
        img = Image.open(io.BytesIO(data))
        w, h = img.size

        # A CMYK picture (print workflows save JPEGs that way) goes through a whole RGB copy on its
        # way to gray: an 8900 x 8900 one read at half size still took a used server to 510 MB
        # with Tesseract, so it counts double.
        per = 2 if img.mode == "CMYK" else 1

        def too_big(size: Tuple[int, int]) -> bool:
            return size[0] * size[1] > MAX_INPUT_PIXELS or _decoded_bytes(img.mode, size) * per > MAX_INPUT_BYTES
        if img.format == "JPEG" and too_big(img.size):
            k = 2
            while k < 8 and too_big((w // k, h // k)):
                k *= 2
            img.draft(img.mode, (max(1, w // k), max(1, h // k)))
        if too_big(img.size):
            raise OcrError(f"the picture is too large to read ({w} x {h} pixels)")
        img.load()
    except OcrError:
        raise
    except Exception as exc:  # noqa: BLE001 - Pillow raises many types for broken files
        if type(exc).__name__ == "DecompressionBombError":  # Pillow's own size guard
            raise OcrError("the picture is too large to read")
        raise OcrError(f"the picture could not be read ({type(exc).__name__})")
    try:
        # A phone stores a sideways photo as is and says in EXIF how to turn it. Only then:
        # exif_transpose returns a full copy even when there is nothing to turn.
        if img.getexif().get(0x0112, 1) not in (None, 1):
            img = ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001 - a broken EXIF block only means no turning
        pass
    return img


# Picture modes that carry color. Everything else is gray: 8-bit, 1-bit, or 16-bit and float.
_COLOR_MODES = ("RGB", "RGBA", "RGBX", "CMYK", "YCbCr", "LAB", "HSV", "P", "PA")


def _picture_size(data: bytes) -> Optional[Tuple[int, int]]:
    """(width, height) from a PNG, JPEG, or PGM/PBM header, without Pillow."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 <= len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker == 0xFF or marker == 0x01 or 0xD0 <= marker <= 0xD8:
                i += 1 if marker == 0xFF else 2
                continue
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
            i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
        return None
    m = re.match(rb"P[45]\s+(?:#[^\n]*\s+)*(\d+)\s+(\d+)", data[:200])
    return (int(m.group(1)), int(m.group(2))) if m else None


def _gray(img: "Image.Image") -> "Image.Image":
    if img.mode == "L":
        return img
    if img.mode in ("RGBA", "LA", "P", "PA"):
        # Transparent screenshots read as white paper. Blended in gray (the gray of the colors
        # over white, weighted by alpha): three one-byte pictures instead of three RGBA ones.
        if img.mode in ("P", "PA"):
            img = img.convert("RGBA")  # a palette may carry its transparency in the palette
        alpha = img.getchannel("A")
        return Image.composite(img.convert("L"), Image.new("L", img.size, 255), alpha)
    elif img.mode in ("I", "F") or img.mode.startswith("I;16"):
        # 16-bit gray (some scanners save PNGs that way) and float pictures: Pillow's
        # convert("L") clips them at 255 instead of scaling, which turns the whole page white.
        # Stretch the values actually used onto 0 to 255 first.
        f = img.convert("F")
        lo, hi = f.getextrema()
        if hi > 255 or lo < 0:
            k = 255.0 / max(hi - lo, 1e-6)
            f = f.point(lambda v: (v - lo) * k)
        img = f
    return img.convert("L")


def _otsu(hist: List[int]) -> int:
    total = sum(hist)
    if not total:
        return 128
    sum_all = sum(i * h for i, h in enumerate(hist))
    w_b = sum_b = 0.0
    best, best_t = -1.0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, best_t = between, t
    return best_t


def _probe(img: "Image.Image", pixels: int = 2_000_000) -> "Image.Image":
    """A small copy for statistics (color, bit depth), picked pixel by pixel so no new gray
    levels appear. Converting a 48 megapixel photo whole for them cost 400 MB."""
    k = int((img.size[0] * img.size[1] / float(pixels)) ** 0.5)
    if k < 2:
        return img
    return img.resize((max(1, img.size[0] // k), max(1, img.size[1] // k)), Image.NEAREST)


def _is_bilevel(img: "Image.Image") -> bool:
    if img.mode == "1":
        return True
    hist = _gray(img if img.mode == "L" else _probe(img)).histogram()
    total = sum(hist) or 1
    return (sum(hist[:8]) + sum(hist[248:])) / total > 0.995


def estimate_skew(img: "Image.Image", max_deg: float = 3.0) -> float:
    """The rotation (degrees, Pillow's sign) that makes text rows and frame lines horizontal.
    A projection profile: rotate a small ink mask and keep the angle whose row sums are the
    most uneven. Drawings have long border lines, which makes the peak sharp."""
    g = _gray(img)
    f = max(1, int(round(max(g.size) / 1100)))
    small = g.reduce(f) if f > 1 else g
    t = _otsu(small.histogram())
    ink = small.point(lambda v: 255 if v < t else 0).convert("F")

    def score(angle: float) -> float:
        r = ink.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        col = r.resize((1, r.size[1]), Image.BOX)
        # Pillow 12 renamed getdata (it goes away in Pillow 14); take whichever this one has.
        rows = list(col.get_flattened_data() if hasattr(col, "get_flattened_data") else col.getdata())
        return sum(v * v for v in rows)

    best = max((a / 2.0 for a in range(int(-max_deg * 2), int(max_deg * 2) + 1)), key=score)
    fine = [best + d / 10.0 for d in range(-5, 6)]
    best = max(fine, key=score)
    fine = [best + d / 50.0 for d in range(-4, 5)]
    return round(max(fine, key=score), 2)


def _solve(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    m = [row[:] + [v] for row, v in zip(a, b)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise OcrError("the page corners could not be solved")
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                k = m[r][col] / m[col][col]
                m[r] = [x - k * y for x, y in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


def _perspective(src: List[Tuple[float, float]], dst: List[Tuple[float, float]]) -> List[float]:
    """Coefficients for Image.transform(PERSPECTIVE): output point dst[i] samples input src[i]."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    return _solve(rows, rhs)


def _fit_line(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    """Fit q = a*p + b to (p, q) points, dropping outliers. Returns (a, b, inlier share)."""
    pts = list(points)
    if len(pts) < 8:
        return None
    keep = pts
    a = b = 0.0
    for _ in range(4):
        n = len(keep)
        sp = sum(p for p, _ in keep)
        sq = sum(q for _, q in keep)
        spp = sum(p * p for p, _ in keep)
        spq = sum(p * q for p, q in keep)
        den = n * spp - sp * sp
        if abs(den) < 1e-9:
            return None
        a = (n * spq - sp * sq) / den
        b = (sq - a * sp) / n
        res = sorted(abs(q - (a * p + b)) for p, q in pts)
        tol = max(2.0, 2.5 * res[len(res) // 2])
        keep = [(p, q) for p, q in pts if abs(q - (a * p + b)) <= tol]
        if len(keep) < 8:
            return None
    return a, b, len(keep) / len(pts)


def find_page(img: "Image.Image") -> Optional[List[Tuple[float, float]]]:
    """Corners (top-left, top-right, bottom-right, bottom-left, in pixels) of a sheet of paper
    lying on a darker background: a phone photo on a desk, a PDF viewer window. None when the
    page fills the picture, which is the case for every scanner and fax."""
    g = _gray(img)
    f = max(1.0, max(g.size) / 800.0)
    small = g.resize((max(1, int(g.size[0] / f)), max(1, int(g.size[1] / f))), Image.BOX)
    small = small.filter(ImageFilter.GaussianBlur(1.2))
    w, h = small.size
    px = small.load()
    d = 3

    def first_rise(values: List[int]) -> Optional[int]:
        # The page edge is the first strong dark-to-bright step seen from outside the page.
        # Text and frame lines inside the page are bright-to-dark first, and the desk's light
        # falloff is too gradual to count as a step.
        span = int(len(values) * 0.42)
        steps = [values[i + d] - values[i] for i in range(max(0, span - d))]
        if not steps:
            return None
        top = max(steps)
        if top < 30:
            return None
        need = max(30, 0.45 * top)
        for i, s in enumerate(steps):
            if s >= need:
                j = max(range(i, min(i + 2 * d, len(steps))), key=lambda k: steps[k])
                return j + d // 2
        return None

    left, right, top, bottom = [], [], [], []
    for y in range(int(h * 0.1), int(h * 0.9), 2):
        row = [px[x, y] for x in range(w)]
        a = first_rise(row)
        if a is not None:
            left.append((y, a))
        b = first_rise(row[::-1])
        if b is not None:
            right.append((y, w - 1 - b))
    for x in range(int(w * 0.1), int(w * 0.9), 2):
        col = [px[x, y] for y in range(h)]
        a = first_rise(col)
        if a is not None:
            top.append((x, a))
        b = first_rise(col[::-1])
        if b is not None:
            bottom.append((x, h - 1 - b))
    fits = [_fit_line(p) for p in (left, right, top, bottom)]
    if any(fit is None or fit[2] < 0.5 for fit in fits):
        return None
    (la, lb, _), (ra, rb, _), (ta, tb, _), (ba, bb, _) = fits  # type: ignore[misc]

    def cross(va: float, vb: float, ha: float, hb: float) -> Tuple[float, float]:
        # x = va*y + vb (a vertical edge) meets y = ha*x + hb (a horizontal edge)
        y = (ha * vb + hb) / (1 - ha * va)
        return va * y + vb, y

    corners = [cross(la, lb, ta, tb), cross(ra, rb, ta, tb), cross(ra, rb, ba, bb), cross(la, lb, ba, bb)]
    if any(not (-0.02 * w <= x <= 1.02 * w and -0.02 * h <= y <= 1.02 * h) for x, y in corners):
        return None
    area = 0.5 * abs(sum(corners[i][0] * corners[(i + 1) % 4][1] - corners[(i + 1) % 4][0] * corners[i][1]
                         for i in range(4)))
    if not 0.2 * w * h <= area <= 0.94 * w * h:
        return None
    mask = Image.new("L", small.size, 0)
    ImageDraw.Draw(mask).polygon(corners, fill=255)
    inside = ImageStat.Stat(small, mask).mean[0]
    outside = ImageStat.Stat(small, ImageOps.invert(mask)).mean[0]
    if outside > 0.8 * inside:  # the "page" must be clearly brighter than what is around it
        return None
    return [(x * f, y * f) for x, y in corners]


def _warp_page(img: "Image.Image", corners: List[Tuple[float, float]], scale: float) -> "Image.Image":
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners
    dist = lambda ax, ay, bx, by: ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5  # noqa: E731
    w = (dist(x0, y0, x1, y1) + dist(x3, y3, x2, y2)) / 2 * scale
    h = (dist(x0, y0, x3, y3) + dist(x1, y1, x2, y2)) / 2 * scale
    w, h = max(1, int(round(w))), max(1, int(round(h)))
    coeffs = _perspective(corners, [(0, 0), (w, 0), (w, h), (0, h)])
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor="white")


def _invert_bands(g: "Image.Image", dpi: float) -> Tuple["Image.Image", int]:
    """Turn dark bands with light text (the header row of an RFQ form table) into dark text on
    light paper. Tesseract takes such a band for a picture, and then also drops the small
    cells just under it: the item number and the one-letter rev of the first row. Works on a
    grid of about 1 mm cells and follows the band column by column, so a skewed scan is fine.
    Returns the picture and how many bands were inverted."""
    cell = max(4, int(round(dpi / 25.0)))
    w, h = g.size
    gw, gh = max(1, w // cell), max(1, h // cell)
    lum = g if g.mode == "L" else g.convert("L")
    small = lum.resize((gw, gh), Image.BOX)
    hist = small.histogram()
    total, acc, paper = sum(hist), 0, 255
    for v in range(255, -1, -1):  # the paper level: the brightest 30 % of the page
        acc += hist[v]
        if acc >= 0.3 * total:
            paper = v
            break
    px = small.load()
    # Cells with white letters in them are lighter than the band itself, so the band is
    # traced at a softer level and must then be mostly truly dark.
    soft = [[px[x, y] < paper * 0.62 for x in range(gw)] for y in range(gh)]
    seen = [[False] * gw for _ in range(gh)]
    mask = Image.new("L", (gw, gh), 0)
    draw = ImageDraw.Draw(mask)
    turned = 0
    for y0 in range(gh):
        for x0 in range(gw):
            if not soft[y0][x0] or seen[y0][x0]:
                continue
            stack, cols = [(x0, y0)], {}
            seen[y0][x0] = True
            edge = False
            while stack:
                x, y = stack.pop()
                lo, hi = cols.get(x, (y, y))
                cols[x] = (min(lo, y), max(hi, y))
                edge = edge or x == 0 or y == 0 or x == gw - 1 or y == gh - 1
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < gw and 0 <= ny < gh and soft[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            # A band is at least an inch long, a few mm to about 15 mm tall, and does not touch
            # the edge of the picture (a copier's edge shadow, a viewer's dark surround).
            if edge or len(cols) * cell < dpi:
                continue
            heights = sorted(hi - lo + 1 for lo, hi in cols.values())
            if not 3 <= heights[len(heights) // 2] <= max(3, dpi * 0.6 / cell):
                continue
            spans = [(x, lo, hi) for x, (lo, hi) in cols.items()]
            dark = sum(1 for x, lo, hi in spans for y in range(lo, hi + 1) if px[x, y] < paper * 0.45)
            area = sum(hi - lo + 1 for _, lo, hi in spans)
            if dark < 0.5 * area:
                continue
            # Only a band with light marks in it (text) is worth turning over.
            band = Image.new("L", (gw, gh), 0)
            bd = ImageDraw.Draw(band)
            for x, lo, hi in spans:
                bd.line([(x, lo), (x, hi)], fill=255)
            full = band.resize((w, h), Image.NEAREST)
            rh = lum.histogram(mask=full)
            light = sum(rh[int(paper * 0.7):]) / max(1, sum(rh))
            if not 0.02 <= light <= 0.45:
                continue
            for x, lo, hi in spans:
                draw.line([(x, lo), (x, hi)], fill=255)
            turned += 1
    if turned:
        full = mask.resize((w, h), Image.NEAREST)
        g = Image.composite(ImageOps.invert(g), g, full)
    return g, turned


def _flatten(g: "Image.Image") -> "Image.Image":
    """Even out the light: estimate the paper level (text removed by a max filter, then
    blurred) and lift every pixel by how much darker its paper is than white."""
    f = 8
    small = g.reduce(f)
    paper = small.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(3))
    paper = paper.resize(g.size, Image.BILINEAR)
    return ImageChops.add(g, ImageOps.invert(paper))


# --------------------------------------------------------------------------- #
# Classifying a page
# --------------------------------------------------------------------------- #
def classify(img: "Image.Image", ppi: Optional[float] = None, bits: Optional[int] = None) -> Dict[str, Any]:
    """What kind of source a page picture is, from the picture alone: its bit depth, its
    resolution (from the PDF, or estimated from the page size), and whether it shows a sheet of
    paper on a background. Never looks at file names."""
    info: Dict[str, Any] = {"size": list(img.size), "mode": img.mode}
    color = img.mode in _COLOR_MODES
    if color:
        sat = ImageStat.Stat(_probe(img).convert("RGB").convert("HSV").getchannel("S")).mean[0]
        color = sat > 12
    info["color"] = bool(color)
    corners = None
    if ppi is None:
        corners = find_page(img)
    if corners:
        info["page_corners"] = [[round(x), round(y)] for x, y in corners]
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners
        long_side = max(abs(x1 - x0), abs(x2 - x3), abs(y3 - y0), abs(y2 - y1))
        est = long_side / PAGE_LONG_SIDE_IN
        tilt = max(abs(y1 - y0), abs(y2 - y3), abs(x3 - x0), abs(x2 - x1)) / max(1.0, long_side)
        info["dpi"] = round(est)
        # A screenshot shows the page square to the screen; a photo never quite does.
        info["type"] = "screen" if tilt < 0.004 else "photo"
        return info
    if ppi:
        info["dpi"] = round(ppi)
    else:
        info["dpi"] = round(max(img.size) / PAGE_LONG_SIDE_IN)
    if bits == 1 or _is_bilevel(img):
        info["type"] = "bilevel"
    elif info["dpi"] >= 250:
        info["type"] = "scan"
    else:
        info["type"] = "lowres"
    return info


def prepare(img: "Image.Image", info: Dict[str, Any], recipe: Dict[str, Any]) -> Tuple["Image.Image", float, Dict[str, Any]]:
    """Clean a page picture for tesseract. Returns (picture, scale from the source, steps)."""
    steps: Dict[str, Any] = {}
    dpi = float(info.get("dpi") or 300)
    scale = recipe.get("scale")
    if scale:
        scale = float(scale)
    elif recipe.get("target_dpi"):
        scale = max(1.0, float(recipe["target_dpi"]) / max(dpi, 30.0))
    else:
        scale = 1.0
    if img.size[0] * img.size[1] * scale * scale > MAX_OCR_PIXELS:
        scale = max(0.25, (MAX_OCR_PIXELS / float(img.size[0] * img.size[1])) ** 0.5)
    if recipe.get("color") and img.mode in _COLOR_MODES:
        # Tesseract makes its own gray picture from the colors (it weighs the channels its
        # own way); the filters below then work on each channel.
        g = img.convert("RGB")
        steps["color"] = True
    else:
        g = _gray(img)
    if recipe.get("median"):
        # Before any resize: the fax's salt noise is one pixel at the source resolution.
        g = g.filter(ImageFilter.MedianFilter(int(recipe["median"])))
        steps["median"] = int(recipe["median"])
    corners = info.get("page_corners") if recipe.get("page") else None
    if corners:
        g = _warp_page(g, [tuple(c) for c in corners], scale)
        steps["page"] = "warped" if info.get("type") == "photo" else "cropped"
    elif abs(scale - 1.0) > 0.01:
        resample = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC,
                    "box": Image.BOX}.get(str(recipe.get("resample") or ""), Image.LANCZOS)
        g = g.resize((max(1, int(round(g.size[0] * scale))), max(1, int(round(g.size[1] * scale)))), resample)
    if abs(scale - 1.0) > 0.01:
        steps["scale"] = round(scale, 3)
    if recipe.get("invert"):
        # Before flatten: flatten takes a dark band for shadowed paper and lifts it to light
        # gray, and then there is no band left to find.
        g, steps["invert"] = _invert_bands(g, dpi * scale)
    if recipe.get("flatten"):
        g = _flatten(g)
        steps["flatten"] = True
    if recipe.get("autocontrast"):
        g = ImageOps.autocontrast(g, cutoff=1)
        steps["autocontrast"] = True
    if recipe.get("unsharp"):
        g = g.filter(ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=3))
        steps["unsharp"] = True
    if recipe.get("deskew"):
        angle = estimate_skew(g)
        if abs(angle) >= 0.1:
            g = g.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
        steps["deskew"] = angle
    return g, scale, steps


# --------------------------------------------------------------------------- #
# Tesseract
# --------------------------------------------------------------------------- #
_DASHES = {cp: "-" for cp in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212)}


def _tesseract_words(path: str, psm: int, dpi: Optional[float], recipe: Dict[str, Any],
                     deadline: float, stats: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    tess = _tools()["tesseract"]
    if not tess:
        raise OcrError("tesseract is not installed")
    cmd = [tess, path, "stdout", "-l", "eng", "--oem", "1", "--psm", str(int(psm))]
    tessdata = recipe.get("tessdata") or OCR_TESSDATA
    if tessdata:
        cmd += ["--tessdata-dir", str(tessdata)]
    if dpi:
        cmd += ["--dpi", str(int(round(dpi)))]
    if recipe.get("threshold") is not None and _supports("thresholding_method"):
        cmd += ["-c", f"thresholding_method={int(recipe['threshold'])}"]
    if recipe.get("pis") and _supports("preserve_interword_spaces"):
        cmd += ["-c", "preserve_interword_spaces=1"]
    if recipe.get("nodict") and _supports("load_system_dawg") and _supports("load_freq_dawg"):
        # Meant to turn off the English word lists. With --oem 1 it does nothing: the LSTM
        # recognizer loads its own word lists with default settings (LSTMRecognizer::
        # LoadDictionary, Tesseract 5.3 and 5.5), so the output is byte for byte the same. A
        # model with lstm-word-dawg removed really reads without it (docs/ocr_settings.md).
        cmd += ["-c", "load_system_dawg=0", "-c", "load_freq_dawg=0"]
    # TSV output (one row per word, with its box and confidence). Set as a variable rather than
    # with the "tsv" config name: that config file lives in the system tessdata folder, and with
    # --tessdata-dir pointing elsewhere tesseract cannot find it and prints plain text instead.
    cmd += ["-c", "tessedit_create_tsv=1"]
    out = _run(cmd, deadline, "tesseract").decode("utf-8", "replace")
    if not out.startswith("level\t"):
        raise OcrError("tesseract did not write its word table (TSV)")
    words = []
    for line in out.splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 12 or f[0] != "5":
            continue
        # Drawings and forms print plain hyphens ("CI-10442", "2026-10-09"); tesseract reads
        # some of them, and bits of line work, as the long dash characters.
        text = f[11].strip().translate(_DASHES)
        try:
            conf = float(f[10])
        except ValueError:
            continue
        if not text or conf < 0:
            continue
        x, y, w, h = int(f[6]), int(f[7]), int(f[8]), int(f[9])
        words.append({"text": text, "conf": conf, "box": [x, y, x + w, y + h],
                      "line": (psm, int(f[2]), int(f[3]), int(f[4]))})
    if stats is not None:
        # How the raw reading looks, before any filtering: a sideways page gives tall word
        # boxes, an upside-down one gives low confidence.
        long_words = [w for w in words if len(w["text"]) >= 3]
        stats["conf"] = sum(w["conf"] for w in words) / len(words) if words else 0.0
        stats["tall"] = (sum(1 for w in long_words if w["box"][3] - w["box"][1] > w["box"][2] - w["box"][0])
                         / len(long_words)) if long_words else 0.0
        stats["words"] = len(words)
    if recipe.get("repair"):
        _repair_case(words)
    floor = float(recipe.get("min_conf") or 0)
    if floor > 0:
        words = [w for w in words if _keep_word(w, floor)]
    return words


def _keep_word(w: Dict[str, Any], floor: float) -> bool:
    """Hatching, center lines, and the shaded iso view come back as low-confidence "words"
    like "ius", "Ww", or "~<". Tesseract is also unsure of things a buyer needs, so those are
    always kept: anything with a digit (quantities like "25/ 75/150" can score 0) and a lone
    capital or two (a rev letter in its own table cell scores about 20)."""
    if w["conf"] >= floor:
        return True
    text = w["text"]
    if any(c.isdigit() for c in text):
        return True
    return len(text) <= 2 and text.isalpha() and text.isupper()


# Letters whose lowercase is a smaller copy of the capital. A big lone capital, like the rev
# letter in a title block's REV cell, can come back as both: "Cc", "Oo".
_TWIN_CASE = {c + c.lower(): c for c in "CKOPSUVWXZ"}


def _repair_case(words: List[Dict[str, Any]]) -> None:
    """In Helvetica an uppercase I and a lowercase l are the same glyph, so tesseract writes
    "Cl-10442" and "TYPE Ill". On a line that is otherwise uppercase, a word whose only
    lowercase letters are l is really uppercase: make it so. And a word that is one capital
    read twice ("Cc", never a real word) is that capital, on any line: the rev letter sits
    alone in its cell, so its line has no other letters to judge by."""
    lines: Dict[Tuple, List[Dict[str, Any]]] = {}
    for w in words:
        if w["text"] in _TWIN_CASE:
            w["text"] = _TWIN_CASE[w["text"]]
        lines.setdefault(w["line"], []).append(w)
    for group in lines.values():
        letters = "".join(w["text"] for w in group)
        upper = sum(c.isupper() for c in letters)
        lower = sum(c.islower() and c != "l" for c in letters)
        if upper < 2 or lower > 0.1 * upper:
            continue
        for w in group:
            low = {c for c in w["text"] if c.islower()}
            if low == {"l"}:
                w["text"] = w["text"].replace("l", "I")


def _overlap(a: List[float], b: List[float]) -> float:
    """Intersection over the smaller box."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1
    return ix * iy / small


def _iou(a: List[float], b: List[float]) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / (union or 1)


def _merge_passes(passes: List[List[Dict[str, Any]]], mode: str = "fill") -> List[Dict[str, Any]]:
    """Words of the first pass, plus words from later passes where the first found nothing.
    Drawings mix paragraphs (notes) with sparse boxed text (title block, callouts), and each
    page segmentation mode misses some of one or the other. With mode "conf", a later word
    that covers the same box as a first-pass word and is read with clearly higher confidence
    replaces its text (the first pass keeps its line structure)."""
    merged = [dict(w) for w in passes[0]]
    for extra in passes[1:]:
        for w in extra:
            hits = [m for m in merged if _overlap(w["box"], m["box"]) >= 0.3]
            if not hits:
                merged.append(w)
            elif mode == "conf" and len(hits) == 1 and _iou(w["box"], hits[0]["box"]) >= 0.6 \
                    and w["conf"] >= hits[0]["conf"] + 15 and w["text"] != hits[0]["text"]:
                hits[0]["text"], hits[0]["conf"] = w["text"], w["conf"]
    return merged


def _group_lines(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Words grouped into tesseract's lines, in reading order. Lines found only by a later pass
    go after the first-pass line just above them, so the text keeps its local order."""
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    order: List[Tuple] = []
    for w in words:
        if w["line"] not in groups:
            groups[w["line"]] = []
            order.append(w["line"])
        groups[w["line"]].append(w)
    if not order:
        return []
    first_psm = order[0][0]
    primary = [groups[k] for k in order if k[0] == first_psm]
    extras = [groups[k] for k in order if k[0] != first_psm]

    def box(ws: List[Dict[str, Any]]) -> List[float]:
        return [min(w["box"][0] for w in ws), min(w["box"][1] for w in ws),
                max(w["box"][2] for w in ws), max(w["box"][3] for w in ws)]

    out = [list(g) for g in primary]
    for g in sorted(extras, key=lambda ws: (box(ws)[1], box(ws)[0])):
        b = box(g)
        best, best_gap = None, None
        for i, p in enumerate(out):
            pb = box(p)
            if pb[1] <= b[1] and min(pb[2], b[2]) - max(pb[0], b[0]) > -0.5 * (b[3] - b[1]) * 4:
                gap = b[1] - pb[3]
                if best_gap is None or gap < best_gap:
                    best, best_gap = i, gap
        if best is None:
            out.insert(0, g)
        else:
            out.insert(best + 1, g)
    for g in out:
        g.sort(key=lambda w: w["box"][0])
    return out


def _segments(line: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split one text line at wide gaps: table columns and title block cells become separate
    entries, each with its own box, so a reader can pair a label with the value under it."""
    heights = sorted(w["box"][3] - w["box"][1] for w in line if not _RULE.match(w["text"]))
    unit = max(4, heights[len(heights) // 2]) if heights else 4
    segs: List[List[Dict[str, Any]]] = [[]]
    prev = None
    for w in line:
        if _RULE.match(w["text"]):
            # A table rule read as "|" or "[": a column border, not text.
            if segs[-1]:
                segs.append([])
            continue
        if prev is not None and segs[-1] and w["box"][0] - prev["box"][2] > 1.6 * unit:
            segs.append([])
        segs[-1].append(w)
        prev = w
    return [seg for seg in segs if seg]


_RULE = re.compile(r"^[|\[\]{}!]+$")


def _assemble(pages: List[List[Dict[str, Any]]], scales: List[float]) -> Dict[str, Any]:
    text_lines: List[str] = []
    lines: List[Dict[str, Any]] = []
    confs: List[float] = []
    for pno, (words, scale) in enumerate(zip(pages, scales), start=1):
        if pno > 1 and text_lines:
            text_lines.append("")
        for group in _group_lines(words):
            parts = []
            for seg in _segments(group):
                t = " ".join(w["text"] for w in seg)
                c = sum(w["conf"] for w in seg) / len(seg)
                bb = [min(w["box"][0] for w in seg), min(w["box"][1] for w in seg),
                      max(w["box"][2] for w in seg), max(w["box"][3] for w in seg)]
                lines.append({"text": t, "conf": round(c, 1), "page": pno,
                              "bbox": [int(round(v / scale)) for v in bb]})
                parts.append(t)
                confs.extend(w["conf"] for w in seg)
            text_lines.append("   ".join(parts))
    return {"text": "\n".join(text_lines).strip("\n"), "lines": lines,
            "confidence": round(sum(confs) / len(confs), 1) if confs else None}


def _read_words(path: str, dpi: float, recipe: Dict[str, Any], deadline: float,
                stats: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    passes = [_tesseract_words(path, psm, dpi if recipe.get("tess_dpi") else None, recipe, deadline,
                               stats if i == 0 else None)
              for i, psm in enumerate(recipe.get("psm") or [3])]
    return _merge_passes(passes, str(recipe.get("merge") or "fill"))


def _sure_words(words: List[Dict[str, Any]]) -> int:
    return sum(1 for w in words if w["conf"] >= 60 and len(w["text"]) >= 2)


def _maybe_turned(stats: Dict[str, Any]) -> bool:
    """Does the first reading look like a sideways or upside-down page? Upright pages of the
    beta set read at a mean confidence of 67 to 93 with almost no tall word boxes."""
    return stats.get("tall", 0) > 0.4 or stats.get("conf", 100) < 55 or stats.get("words", 0) < 15


def _orientation(path: str, recipe: Dict[str, Any], deadline: float) -> Tuple[int, float]:
    """(degrees to turn the picture clockwise, confidence) from tesseract's orientation and
    script detection (--psm 0, which needs osd.traineddata; Debian's tesseract-ocr has it)."""
    tess = _tools()["tesseract"]
    cmd = [tess, path, "stdout", "--psm", "0"]
    tessdata = recipe.get("tessdata") or OCR_TESSDATA
    if tessdata:
        cmd += ["--tessdata-dir", str(tessdata)]
    out = _run(cmd, deadline, "tesseract orientation").decode("utf-8", "replace")
    rot = re.search(r"Rotate:\s*(\d+)", out)
    conf = re.search(r"Orientation confidence:\s*([\d.]+)", out)
    return (int(rot.group(1)) if rot else 0), (float(conf.group(1)) if conf else 0.0)


def _detect_regions(pic: "Image.Image") -> Optional[Tuple[List[Dict[str, Any]], float]]:
    """(regions in the picture's pixels, seconds) from layout.py, or None when the detector
    cannot run here."""
    lay = _layout()
    if lay is None:
        return None
    t0 = time.monotonic()
    return lay.detect(pic, conf=REGION_CONF), time.monotonic() - t0


def _region_words(g: "Image.Image", dpi: float, regions: List[Dict[str, Any]], recipe: Dict[str, Any],
                  tmp: str, deadline: float, tag: str) -> List[List[Dict[str, Any]]]:
    """A second reading of each region the recipe names under "regions", alone: the crop (with a
    small margin) is optionally scaled up and read with its own page segmentation modes, and the
    word boxes are put back into the page picture's pixels."""
    passes: List[List[Dict[str, Any]]] = []
    specs = recipe.get("regions") or {}
    pad = max(4, int(round(dpi * 0.03)))
    for k, r in enumerate(regions):
        spec = specs.get(r["label"])
        if not spec:
            continue
        x0, y0, x1, y1 = r["box"]
        box = (max(0, int(x0) - pad), max(0, int(y0) - pad),
               min(g.size[0], int(x1 + 0.999) + pad), min(g.size[1], int(y1 + 0.999) + pad))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        crop = g.crop(box)
        f = float(spec.get("scale") or 1.0)
        if spec.get("target_dpi"):
            f = max(1.0, float(spec["target_dpi"]) / max(dpi, 30.0))
        if abs(f - 1.0) > 0.01:
            crop = crop.resize((max(1, int(round(crop.size[0] * f))), max(1, int(round(crop.size[1] * f)))),
                               Image.LANCZOS)
        path = os.path.join(tmp, f"{tag}-{r['label']}{k}.tif")
        crop.save(path, "TIFF", dpi=(dpi * f, dpi * f))
        sub = dict(recipe, min_conf=spec.get("min_conf", recipe.get("min_conf")))
        for psm in spec.get("psm") or [6]:
            words = _tesseract_words(path, psm, dpi * f if recipe.get("tess_dpi") else None, sub, deadline)
            for w in words:
                w["box"] = [box[0] + w["box"][0] / f, box[1] + w["box"][1] / f,
                            box[0] + w["box"][2] / f, box[1] + w["box"][3] / f]
                w["line"] = (f"{r['label']}{k}:{psm}",) + tuple(w["line"][1:])
            passes.append(words)
    return passes


def _merge_region(words: List[Dict[str, Any]], extra: List[List[Dict[str, Any]]], mode: str) -> List[Dict[str, Any]]:
    """Join the region readings to the page reading. "conf" and "fill" work as between two page
    passes (_merge_passes). "replace": inside the region, the reading with the higher mean
    confidence wins as a whole."""
    if mode != "replace":
        return _merge_passes([words] + extra, mode)
    out = list(words)
    for ws in extra:
        if not ws:
            continue
        bx = [min(w["box"][0] for w in ws), min(w["box"][1] for w in ws),
              max(w["box"][2] for w in ws), max(w["box"][3] for w in ws)]

        def inside(w: Dict[str, Any]) -> bool:
            cx, cy = (w["box"][0] + w["box"][2]) / 2, (w["box"][1] + w["box"][3]) / 2
            return bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3]
        old = [w for w in out if inside(w)]
        mean = lambda ws_: sum(w["conf"] for w in ws_) / len(ws_) if ws_ else 0.0  # noqa: E731
        if mean(ws) > mean(old):
            out = [w for w in out if not inside(w)] + ws
    return out


def _ocr_picture(prepared: Tuple["Image.Image", float, Dict[str, Any]], info: Dict[str, Any],
                 recipe: Dict[str, Any], tmp: str, deadline: float,
                 tag: str) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any], Any]:
    """Read one page picture already cleaned by prepare: (picture, scale, steps). Returns the
    words, the scale, the steps, and (regions, seconds) from the detector or None."""
    g, scale, steps = prepared
    path = os.path.join(tmp, f"{tag}.tif")
    dpi = float(info.get("dpi") or 300) * scale
    g.save(path, "TIFF", dpi=(dpi, dpi))  # uncompressed: fastest to write and to read
    stats: Dict[str, Any] = {}
    words = _read_words(path, dpi, recipe, deadline, stats)
    if recipe.get("orient", True) and _maybe_turned(stats):
        # A drawing scanned sideways, a photo taken upside down: turn it and read it again,
        # and keep whichever reading has more words tesseract is sure of.
        try:
            rot, conf = _orientation(path, recipe, deadline)
        except OcrError:
            rot, conf = 0, 0.0
        if rot in (90, 180, 270) and conf >= 1.5:
            turned = g.rotate(-rot, expand=True)
            path2 = os.path.join(tmp, f"{tag}-r.tif")
            turned.save(path2, "TIFF", dpi=(dpi, dpi))
            stats2: Dict[str, Any] = {}
            again = _read_words(path2, dpi, recipe, deadline, stats2)
            # Tesseract reads a page on its side by itself, but the words come back in its
            # own frame and out of order; an upright reading of about as many words wins.
            upright = stats["tall"] > 0.4 and stats2.get("tall", 1) < 0.2
            if _sure_words(again) > (0.7 if upright else 1.0) * _sure_words(words):
                words = again
                steps["rotated"] = rot
                g = turned
    # The regions come from the same picture as the words (the turned one, when it was turned),
    # so a line and the region around it are in the same pixels.
    # Skipped when the file's time is up: a page without regions is still a page read.
    found = _detect_regions(g) if time.monotonic() < deadline else None
    if found and found[0] and recipe.get("regions"):
        extra = _region_words(g, dpi, found[0], recipe, tmp, deadline, tag)
        if extra:
            words = _merge_region(words, extra, str(recipe.get("region_merge") or "conf"))
    return words, scale, steps, found


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def _pdf_images(pdf_path: str, deadline: float) -> Dict[int, Dict[str, Any]]:
    """The biggest picture on each page, from pdfimages -list: size, bits, resolution."""
    tool = _tools()["pdfimages"]
    if not tool:
        return {}
    try:
        out = _run([tool, "-list", pdf_path], deadline, "pdfimages").decode("utf-8", "replace")
    except OcrError:
        return {}
    best: Dict[int, Dict[str, Any]] = {}
    for line in out.splitlines()[2:]:
        f = line.split()
        if len(f) < 14 or not f[0].isdigit() or f[2] != "image":
            continue
        try:
            xppi, yppi = float(f[12]), float(f[13])
            im = {"width": int(f[3]), "height": int(f[4]), "color": f[5], "bits": int(f[7]),
                  "ppi": min(xppi, yppi),
                  # the area the picture covers on the page, in square inches
                  "sq_in": (int(f[3]) / xppi) * (int(f[4]) / yppi) if xppi > 0 and yppi > 0 else 0.0}
        except ValueError:
            continue
        page = int(f[0])
        if page not in best or im["width"] * im["height"] > best[page]["width"] * best[page]["height"]:
            best[page] = im
    return best


def _pdf_page_count(pdf_path: str, deadline: float) -> Optional[int]:
    tool = shutil.which("pdfinfo")
    if not tool:
        return None
    try:
        out = _run([tool, pdf_path], deadline, "pdfinfo").decode("utf-8", "replace")
    except OcrError:
        return None
    m = re.search(r"^Pages:\s+(\d+)", out, re.MULTILINE)
    return int(m.group(1)) if m else None


def _pdf_page_sizes(pdf_path: str, pages: List[int], deadline: float) -> Dict[int, Tuple[float, float]]:
    """Page number -> (width, height) in inches, from pdfinfo (poppler-utils, like pdftoppm)."""
    tool = shutil.which("pdfinfo")
    if not tool or not pages:
        return {}
    try:
        out = _run([tool, "-f", str(min(pages)), "-l", str(max(pages)), pdf_path], deadline,
                   "pdfinfo").decode("utf-8", "replace")
    except OcrError:
        return {}
    sizes = {}
    for m in re.finditer(r"^Page\s+(\d+)\s+size:\s+([\d.]+)\s+x\s+([\d.]+)\s+pts", out, re.MULTILINE):
        w, h = float(m.group(2)) / 72.0, float(m.group(3)) / 72.0
        if w > 0 and h > 0:
            sizes[int(m.group(1))] = (w, h)
    return sizes


def _raster_dpi(dpi: float, size: Optional[Tuple[float, float]]) -> float:
    """The resolution to render a page at: what the recipe asks for, but never more pixels than
    MAX_OCR_PIXELS. A PDF page can be up to 200 inches square; at 300 dpi that is 3.6 billion
    pixels, and a 1200 dpi letter scan is 135 million, which pdftoppm would render in full
    (gigabytes of memory on a 512 MB host) before the picture is scaled down."""
    if size:
        cap = (MAX_OCR_PIXELS / (size[0] * size[1])) ** 0.5
        if cap < dpi:
            # A whole number: pdftoppm takes an integer, and rounding up would pass the cap.
            dpi = float(int(cap))
    return max(10.0, dpi)


def _rasterize(pdf_path: str, page: int, dpi: float, deadline: float) -> bytes:
    tool = _tools()["pdftoppm"]
    if not tool:
        raise OcrError("pdftoppm is not installed")
    # PGM, not PNG: pdftoppm's PNG writer spends about 13 s compressing one noisy 300 dpi
    # page (0.2 s for PGM), which would be two minutes on a 0.1 CPU host. PGM carries no
    # resolution, so tesseract is always told the dpi.
    out = _run([tool, "-r", str(int(round(dpi))), "-f", str(page), "-l", str(page), "-gray", pdf_path],
               deadline, "pdftoppm")
    if not out:
        raise OcrError("pdftoppm returned no picture for the page")
    return out


def _pdf_pages_fallback(data: bytes, pages: List[int], deadline: float) -> List[Dict[str, Any]]:
    """Without pdftoppm: the biggest picture on each page, pulled out with pypdf in a child
    process (the same care attachments.py takes with hostile PDFs). Each entry is {page,
    image (bytes), page_in (width, height in inches), bits}, so the page can still be
    classified by its resolution and bit depth like a rendered one."""
    try:
        proc = subprocess.run([sys.executable, str(HERE / "ocr.py"), "--pdf-images", ",".join(map(str, pages))],
                              input=data, capture_output=True, timeout=max(1.0, deadline - time.monotonic()),
                              cwd=str(HERE))
    except subprocess.TimeoutExpired:
        raise OcrError("pulling the pictures out of the PDF took too long")
    except OSError as exc:
        raise OcrError(f"could not start the PDF reader ({exc})")
    try:
        result = json.loads(proc.stdout.decode("utf-8") or "{}")
    except ValueError:
        result = {}
    if result.get("error"):
        raise OcrError(result["error"])
    import base64
    return [dict(p, image=base64.b64decode(p["image"])) for p in result.get("pages") or []]


def _pdf_images_main(pages_arg: str = "") -> int:
    """Child process: PDF bytes on stdin, JSON {pages: [{page, image (base64 JPEG or PNG),
    page_in, bits}], error} on stdout."""
    try:
        import resource  # Unix only: cap memory so a PDF bomb cannot take the host down
        limit = 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:  # noqa: BLE001
        pass
    import base64
    out: Dict[str, Any] = {"pages": [], "error": None}
    try:
        import logging
        logging.disable(logging.CRITICAL)
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(sys.stdin.buffer.read()))
        if reader.is_encrypted and not reader.decrypt(""):
            raise OcrError("the PDF is protected by a password, so it cannot be opened")
        wanted = [int(p) for p in pages_arg.split(",") if p.strip().isdigit()]
        wanted = wanted or list(range(1, min(len(reader.pages), OCR_MAX_PAGES) + 1))

        def biggest(res: Any, depth: int, best: List[Any]) -> None:
            # Walk the page's pictures (and those inside forms) by their declared size, without
            # decoding any of them.
            xobjects = res.get_object().get("/XObject") if res is not None else None
            if xobjects is None:
                return
            xobjects = xobjects.get_object()
            for name in xobjects:
                obj = xobjects[name].get_object()
                if obj.get("/Subtype") == "/Image":
                    area = int(obj.get("/Width", 0) or 0) * int(obj.get("/Height", 0) or 0)
                    if not best or area > best[0]:
                        best[:] = [area, name, obj]
                elif obj.get("/Subtype") == "/Form" and depth < 3:
                    biggest(obj.get("/Resources"), depth + 1, best)

        for pno in wanted[:OCR_MAX_PAGES]:
            if not 1 <= pno <= len(reader.pages):
                continue
            page = reader.pages[pno - 1]
            best: List[Any] = []
            biggest(page.get("/Resources"), 0, best)
            if not best:
                continue
            _, name, obj = best
            filters = obj.get("/Filter")
            filters = [filters] if isinstance(filters, str) else list(filters or [])
            if filters == ["/DCTDecode"]:
                raw = obj.get_data()  # the JPEG itself, not decoded and saved again
            else:
                pics = [im for im in page.images if im.name.rsplit("/", 1)[-1].split(".")[0] == name.lstrip("/")]
                if not pics:
                    continue
                raw = pics[0].data
            box = page.mediabox
            out["pages"].append({"page": pno, "image": base64.b64encode(raw).decode("ascii"),
                                 "page_in": [float(box.width) / 72.0, float(box.height) / 72.0],
                                 "bits": int(obj.get("/BitsPerComponent", 8) or 8)})
    except OcrError as exc:
        out["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"the PDF pictures could not be read ({type(exc).__name__})"
    sys.stdout.write(json.dumps(out))
    return 0


def _pdf_text_layer(data: bytes, deadline: Optional[float] = None) -> Dict[str, Any]:
    """{text, pages, per_page, error}. pypdf (through attachments.py, in a child process) gives the
    text; pdftotext, when installed, also says which pages have none, for mixed PDFs. deadline
    (time.monotonic) bounds pdftotext; attachments.py has its own limit for pypdf
    (RFQ_PDF_TEXT_TIMEOUT)."""
    result: Dict[str, Any] = {"text": "", "pages": None, "per_page": None, "error": None}
    try:
        sys.path.insert(0, str(HERE)) if str(HERE) not in sys.path else None
        import attachments
        got = attachments.extract_pdf_text(data)
        result.update(text=got.get("text") or "", pages=got.get("pages"), error=got.get("error"))
    except Exception as exc:  # noqa: BLE001 - attachments.py missing or broken: try pdftotext
        result["error"] = f"pypdf text reader unavailable ({type(exc).__name__})"
    tool = _tools()["pdftotext"]
    if tool:
        try:
            left = 30.0 if deadline is None else min(30.0, deadline - time.monotonic())
            if left <= 0:
                raise subprocess.TimeoutExpired(tool, 0)
            out = subprocess.run([tool, "-q", "-", "-"], input=data, capture_output=True, timeout=left).stdout
            pages = out.decode("utf-8", "replace").split("\f")
            if pages and not pages[-1].strip():
                pages = pages[:-1]
            if pages:
                result["per_page"] = [len(re.findall(r"[A-Za-z0-9]", p)) for p in pages]
                if not result["text"].strip():
                    result["text"] = "\n".join(pages)
                    result["pages"] = result["pages"] or len(pages)
                    result["reader"] = "pdftotext"
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def _has_text_layer(layer: Dict[str, Any]) -> bool:
    chars = len(re.findall(r"[A-Za-z0-9]", layer.get("text") or ""))
    pages = max(1, int(layer.get("pages") or 1))
    return chars >= TEXT_LAYER_MIN_CHARS * pages


def _scanned_pages(data: bytes, layer: Dict[str, Any], deadline: float) -> List[int]:
    """Pages of a PDF with a text layer that still need OCR: pages with no real text, and pages
    that are one big picture whose only text is a line the scanning software typed over it
    ("Scanned by ... on ... page 1 of 1", a fax header, a Bates number). Such a line can pass
    TEXT_LAYER_MIN_CHARS, and then the whole scan would be taken for a typed page."""
    per_page = layer.get("per_page") or []
    pages = [i + 1 for i, n in enumerate(per_page) if n < TEXT_LAYER_MIN_CHARS]
    thin = [i + 1 for i, n in enumerate(per_page) if TEXT_LAYER_MIN_CHARS <= n < SCAN_STAMP_MAX_CHARS]
    if thin and _tools()["pdfimages"]:
        with tempfile.TemporaryDirectory(prefix="rfq-ocr-") as tmp:
            pdf_path = os.path.join(tmp, "in.pdf")
            with open(pdf_path, "wb") as fh:
                fh.write(data)
            images = _pdf_images(pdf_path, deadline)
            sizes = _pdf_page_sizes(pdf_path, thin, deadline) if images else {}
        for p in thin:
            im, size = images.get(p), sizes.get(p)
            if im and size and im["sq_in"] >= SCAN_PICTURE_COVER * size[0] * size[1]:
                pages.append(p)
    return sorted(pages)


def _page_items(data: bytes, media: str, recipes: Dict[str, Dict[str, Any]], tmp: str, deadline: float,
                only_pages: Optional[List[int]] = None) -> List[Tuple[Any, Dict[str, Any], Dict[str, Any]]]:
    """(picture, info, recipe) for every page to OCR. The picture is a Pillow image, or raw
    bytes when Pillow is missing or the recipe says to hand tesseract the file as it is."""
    items = []
    if media == "pdf":
        pdf_path = os.path.join(tmp, "in.pdf")
        with open(pdf_path, "wb") as fh:
            fh.write(data)
        images = _pdf_images(pdf_path, deadline)
        count = _pdf_page_count(pdf_path, deadline) or (max(images) if images else 1)
        pages = [p for p in range(1, count + 1) if only_pages is None or p in only_pages][:OCR_MAX_PAGES]
        if not _tools()["pdftoppm"]:
            # Without poppler the page count may be unknown; the child then reads the first pages.
            for pic in _pdf_pages_fallback(data, (only_pages or [])[:OCR_MAX_PAGES], deadline):
                w_in, h_in = pic["page_in"]
                if Image is None:
                    size = _picture_size(pic["image"])
                    if size and size[0] * size[1] > MAX_INPUT_PIXELS:
                        raise OcrError(f"the picture is too large to read ({size[0]} x {size[1]} pixels)")
                    dpi = max(size) / max(w_in, h_in, 0.1) if size else 300.0
                    items.append((pic["image"], {"dpi": dpi, "type": "scan"}, recipes["scan"]))
                    continue
                img = _open_image(pic["image"])
                # The picture's resolution on the page (a scan fills the page it was put on).
                ppi = max(img.size) / max(w_in, h_in, 0.1)
                info = classify(img, ppi=ppi if ppi >= 50 else 300.0, bits=pic["bits"])
                items.append((img, info, recipes[info["type"]]))
            if not items:
                raise OcrError("the PDF has no scanned pictures to read, and pdftoppm (poppler-utils), "
                               "which could render its pages, is not installed")
            return items
        sizes = _pdf_page_sizes(pdf_path, pages, deadline)
        for p in pages:
            im = images.get(p)
            if im and im["ppi"] >= 50:
                pre = "bilevel" if im["bits"] == 1 else ("scan" if im["ppi"] >= 250 else "lowres")
                native = im["ppi"]
            else:
                pre, native = "scan", 300.0
            recipe = recipes[pre]
            if recipe.get("raster_dpi", "native") == "native":
                dpi = min(native, MAX_RASTER_DPI)
            else:
                dpi = float(recipe["raster_dpi"])
            dpi = _raster_dpi(dpi, sizes.get(p))
            raw = _rasterize(pdf_path, p, dpi, deadline)
            if Image is None or recipe.get("raw"):
                items.append((raw, {"dpi": dpi, "type": pre, "source_dpi": round(native)}, recipe))
                continue
            img = _open_image(raw)
            info = classify(img, ppi=dpi, bits=im["bits"] if im else None)
            info["source_dpi"] = round(native)
            if im and im["bits"] != 1 and native < 250:
                info["type"] = "lowres"
            items.append((img, info, recipes.get(info["type"], recipe)))
        return items
    any_recipe = next(iter(recipes.values()))
    if Image is None or any_recipe.get("raw"):
        # Without Pillow the picture goes to tesseract as it is, so check its claimed size here:
        # a small PNG can claim billions of pixels.
        size = _picture_size(data)
        if size and size[0] * size[1] > MAX_INPUT_PIXELS:
            raise OcrError(f"the picture is too large to read ({size[0]} x {size[1]} pixels)")
        return [(data, {"type": "image"}, any_recipe)]
    img = _open_image(data)
    info = classify(img)
    return [(img, info, recipes[info["type"]])]


def _ocr(data: bytes, media: str, recipes: Dict[str, Dict[str, Any]], deadline: float,
         only_pages: Optional[List[int]] = None) -> Dict[str, Any]:
    """OCR every page. Raises OcrError."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rfq-ocr-") as tmp:
        items = _page_items(data, media, recipes, tmp, deadline, only_pages)
        if not items:
            raise OcrError("the file has no pages to read")
        pages, scales, used, found = [], [], [], []
        for i in range(1, len(items) + 1):
            pic, info, recipe = items[i - 1]
            items[i - 1] = (None, info, recipe)  # so the source picture can be freed below
            if isinstance(pic, (bytes, bytearray)):
                ext = {b"P5": ".pgm", b"P4": ".pbm"}.get(bytes(pic[:2]), ".jpg")
                ext = ".png" if bytes(pic[:4]) == b"\x89PNG" else ext
                path = os.path.join(tmp, f"raw{i}{ext}")
                with open(path, "wb") as fh:
                    fh.write(pic)
                dpi = info.get("dpi") if (recipe.get("tess_dpi") or ext == ".pgm") else None
                words = _merge_passes([_tesseract_words(path, psm, dpi, recipe, deadline)
                                       for psm in recipe.get("psm") or [3]], str(recipe.get("merge") or "fill"))
                scale, steps, regions = 1.0, {}, None
            else:
                prepared = prepare(pic, info, recipe)
                # The cleaned picture is all tesseract needs: let the source go before it runs
                # (a 48 megapixel photo is 192 MB that would otherwise sit beside tesseract).
                del pic
                words, scale, steps, regions = _ocr_picture(prepared, info, recipe, tmp, deadline, f"p{i}")
                del prepared
            pages.append(words)
            scales.append(scale)
            found.append(regions)
            used.append({"page": i, "source": info.get("type"), "dpi": info.get("source_dpi") or info.get("dpi"),
                         "recipe": recipe_record(recipe), "steps": steps})
    out = _assemble(pages, scales)
    out["pages"] = len(items)
    out["settings"] = {"engine": f"tesseract {_tools()['tesseract_version']}", "oem": 1, "lang": "eng",
                       "model": str(recipes.get("scan", {}).get("tessdata") or OCR_TESSDATA or "installed"),
                       "pages": used}
    _attach_regions(out, found, scales)
    out["seconds"] = round(time.monotonic() - started, 2)
    return out


def region_at(regions: List[Dict[str, Any]], page: int, bbox: List[float]) -> Optional[str]:
    """The label of the region the center of bbox falls in, on that page; the smallest one when
    regions overlap (the most specific). None outside every region."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    inside = [r for r in regions if r.get("page", 1) == page and r["box"][0] <= cx <= r["box"][2]
              and r["box"][1] <= cy <= r["box"][3]]
    if not inside:
        return None
    return min(inside, key=lambda r: (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]))["label"]


def _attach_regions(out: Dict[str, Any], found: List[Any], scales: List[float]) -> None:
    """Save the detector's regions with the result, in the same pixels as the line boxes (the
    page picture at the source resolution), and tag each line with the region it sits in. Adds
    nothing when the detector did not run, so a result without it is what it always was."""
    if all(f is None for f in found):
        return
    regions: List[Dict[str, Any]] = []
    seconds = 0.0
    for pno, (f, scale) in enumerate(zip(found, scales), start=1):
        if not f:
            continue
        seconds += f[1]
        for d in f[0]:
            regions.append({"page": pno, "label": d["label"], "conf": round(float(d["conf"]), 3),
                            "box": [int(round(v / scale)) for v in d["box"]]})
    for ln in out["lines"]:
        label = region_at(regions, ln["page"], ln["bbox"])
        if label:
            ln["region"] = label
    out["regions"] = regions
    lay = _layout()
    out["settings"]["layout"] = {"model": Path(lay.MODEL_PATH).name if lay else None, "conf": REGION_CONF,
                                 "seconds": round(seconds, 2)}


def recipe_record(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """The settings of a recipe as they are saved with a result: every switch that is on, and
    the model by folder name instead of a path on this machine."""
    out = {k: v for k, v in sorted(recipe.items()) if v not in (None, False, 0, "") and k != "tessdata"}
    out["psm"] = list(recipe.get("psm") or [3])
    if recipe.get("tessdata"):
        out["model"] = Path(str(recipe["tessdata"])).name
    return out


def _result(method: str, **kw: Any) -> Dict[str, Any]:
    base = {"method": method, "text": "", "confidence": None, "pages": None, "lines": [], "settings": {},
            "seconds": 0.0, "error": None}
    base.update(kw)
    return base


def recipes_for(effort: str) -> Dict[str, Dict[str, Any]]:
    return FAST_RECIPES if effort == "fast" else RECIPES


def file_text(data: bytes, media: str, name: str = "", *, cache: "Optional[OcrCache]" = None,
              allow_ocr: bool = True, effort: str = "best", timeout: Optional[float] = None) -> Dict[str, Any]:
    """The text of one attachment, and how it was read. Never raises.

    method "text-layer": a PDF with real text, read with pypdf.
    method "ocr":        a scan, fax, photo, or screenshot, read with tesseract. "lines" holds one
                         entry per run of words on a line (a wide gap, like a table column or a
                         title block cell border, starts a new entry), with its box in pixels of
                         the page picture at the source resolution.
    method "step-header": the header and PRODUCT entities of a STEP model.
    method "none":       nothing could be read; "error" says why.
    effort "best" runs the measured multi-step recipe for the source type; "fast" runs one pass
    (for slow hosts and live uploads). timeout bounds the whole file (default RFQ_OCR_TIMEOUT).
    """
    started = time.monotonic()
    try:
        return _file_text(data, media, name, cache=cache, allow_ocr=allow_ocr, effort=effort,
                          timeout=OCR_TIMEOUT if timeout is None else float(timeout), started=started)
    except Exception as exc:  # noqa: BLE001 - never take the server down over one file
        return _result("none", error=f"reading the file failed ({type(exc).__name__}: {str(exc)[:120]})",
                       seconds=round(time.monotonic() - started, 2))


def _file_text(data: bytes, media: str, name: str, *, cache: "Optional[OcrCache]", allow_ocr: bool,
               effort: str, timeout: float, started: float) -> Dict[str, Any]:
    if not isinstance(data, (bytes, bytearray)) or not data:
        return _result("none", error="the file is empty")
    data = bytes(data)
    media = (media or "").lower().lstrip(".")
    media = {"jpeg": "jpg", "stp": "step"}.get(media, media)
    sha = hashlib.sha256(data).hexdigest()
    if cache is not None:
        hit = cache.get(sha)
        if hit is not None:
            hit = dict(hit, sha256=sha, cached=True)
            return hit
    if media == "step":
        return dict(step_text(data), sha256=sha)
    if media not in ("pdf", "png", "jpg"):
        return _result("none", error=f"no text reader for {media or 'this'} files", sha256=sha)
    layer: Dict[str, Any] = {}
    ocr_pages: Optional[List[int]] = None
    if media == "pdf":
        if not data.lstrip()[:1024].count(b"%PDF-"):
            return _result("none", error="the file is not a PDF", sha256=sha)
        layer = _pdf_text_layer(data, started + timeout)
        if _has_text_layer(layer):
            per_page = layer.get("per_page") or []
            scanned = _scanned_pages(data, layer, started + timeout) if allow_ocr and available()["ocr"] else []
            text = (layer.get("text") or "").strip()
            if not scanned:
                return _result("text-layer", text=text, pages=layer.get("pages"), sha256=sha,
                               settings={"reader": layer.get("reader") or "pypdf"},
                               seconds=round(time.monotonic() - started, 2))
            if len(scanned) < len(per_page):
                ocr_pages = scanned  # a mixed PDF: typed pages plus scanned ones
            else:
                # Every page is a scan and the text layer only a stamp typed over it, which
                # OCR reads off the page anyway.
                layer = {"pages": layer.get("pages")}
    if not allow_ocr:
        return _result("none", pages=layer.get("pages"), sha256=sha,
                       error="the file has no text layer and OCR is turned off")
    if not available()["ocr"]:
        return _result("none", pages=layer.get("pages"), sha256=sha,
                       error="the file has no text layer and tesseract is not installed")
    deadline = started + timeout
    wait = max(0.0, deadline - time.monotonic())
    if not _ocr_slots.acquire(timeout=wait):
        return _result("none", sha256=sha, error="the server is busy reading other scans; try again shortly",
                       seconds=round(time.monotonic() - started, 2))
    try:
        out = _ocr(data, media, recipes_for(effort), deadline, ocr_pages)
    except OcrError as exc:
        error = str(exc)
        if media == "pdf" and "password" in error.lower():
            error = "the PDF is protected by a password, so it cannot be opened"
        return _result("none", pages=layer.get("pages"), sha256=sha, error=error,
                       seconds=round(time.monotonic() - started, 2))
    finally:
        _ocr_slots.release()
    if ocr_pages:
        out["text"] = ((layer.get("text") or "").strip() + "\n\n" + out["text"]).strip()
        out["settings"]["text_layer_pages"] = [i + 1 for i in range(len(layer.get("per_page") or []))
                                               if i + 1 not in ocr_pages]
        out["pages"] = layer.get("pages") or out["pages"]
    out["settings"]["effort"] = effort
    res = _result("ocr", **out)
    res["seconds"] = round(time.monotonic() - started, 2)
    res["sha256"] = sha
    if not res["text"].strip():
        res["error"] = "OCR found no text"
    return res


# --------------------------------------------------------------------------- #
# STEP models
# --------------------------------------------------------------------------- #
def _step_str(s: str) -> str:
    s = s.replace("''", "'")

    def u(m: "re.Match[str]") -> str:
        hexs = m.group(1)
        try:
            return "".join(chr(int(hexs[i:i + 4], 16)) for i in range(0, len(hexs), 4))
        except ValueError:
            return ""
    return re.sub(r"\\X2\\([0-9A-Fa-f]+)\\X0\\", u, s)


def step_text(data: bytes) -> Dict[str, Any]:
    """Text from a STEP file's header and PRODUCT entities: part number, title, revision, units.
    Only the first 2 MB are read; the geometry after that says nothing a person would type."""
    started = time.monotonic()
    head = data[:2_000_000].decode("latin-1", "replace")
    if "ISO-10303-21" not in head[:200]:
        return _result("none", error="the file is not a STEP (ISO 10303-21) model")
    lines: List[str] = []
    quoted = r"'((?:[^']|'')*)'"
    m = re.search(r"FILE_DESCRIPTION\s*\(\s*\((.*?)\)\s*,", head, re.S)
    if m:
        desc = [_step_str(x) for x in re.findall(quoted, m.group(1)) if x.strip()]
        if desc:
            lines.append("Description: " + "; ".join(desc))
    m = re.search(r"FILE_NAME\s*\((.*?)\)\s*;", head, re.S)
    if m:
        parts = re.findall(quoted, m.group(1))
        if parts:
            lines.insert(0, f"STEP file: {_step_str(parts[0])}")
        if len(parts) >= 6 and parts[5].strip():
            lines.append(f"Originating system: {_step_str(parts[5])}")
    m = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*" + quoted, head)
    if m:
        lines.append(f"Schema: {_step_str(m.group(1)).split('{')[0].strip()}")
    for pn, title, desc in re.findall(r"=\s*PRODUCT\s*\(\s*" + quoted + r"\s*,\s*" + quoted + r"\s*,\s*" + quoted, head)[:20]:
        lines.append(f"Part number: {_step_str(pn)}")
        if title.strip():
            lines.append(f"Title: {_step_str(title)}")
        if desc.strip():
            lines.append(f"Product description: {_step_str(desc)}")
    for rev in re.findall(r"PRODUCT_DEFINITION_FORMATION(?:_WITH_SPECIFIED_SOURCE)?\s*\(\s*" + quoted, head)[:20]:
        if rev.strip():
            lines.append(f"Revision: {_step_str(rev)}")
    units = None
    m = re.search(r"GLOBAL_UNIT_ASSIGNED_CONTEXT\s*\(\s*\(([^)]*)\)", head)
    refs = re.findall(r"#(\d+)", m.group(1)) if m else []
    for ref in refs:
        e = re.search(r"#" + ref + r"\s*=\s*(.*?);", head, re.S)
        if not e or "LENGTH_UNIT" not in e.group(1):
            continue
        body = e.group(1)
        c = re.search(r"CONVERSION_BASED_UNIT\s*\(\s*" + quoted, body)
        if c:
            units = _step_str(c.group(1)).lower()
        elif ".MILLI." in body and ".METRE." in body:
            units = "mm"
        elif ".CENTI." in body and ".METRE." in body:
            units = "cm"
        elif ".METRE." in body:
            units = "m"
        break
    if units:
        lines.append(f"Units: {units}")
    if len(lines) <= 1:
        return _result("none", error="the STEP header names no product", seconds=round(time.monotonic() - started, 3))
    return _result("step-header", text="\n".join(lines), pages=None, settings={"reader": "STEP header"},
                   seconds=round(time.monotonic() - started, 3))


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class OcrCache:
    """Saved results keyed by the SHA-256 of the file bytes, so the committed beta files never
    need pypdf or tesseract on a slow host: text-layer, STEP, and OCR results alike. The file
    also records the tesseract version and the settings the results were made with; an entry
    is used as long as the file hash matches (rebuild with python ocr.py --build-cache)."""

    def __init__(self, path: "str | Path | None"):
        self.path = Path(path) if path else None
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.meta: Dict[str, Any] = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                entries = raw.get("entries") if isinstance(raw, dict) else None
                if isinstance(entries, dict):
                    self.entries = {k: v for k, v in entries.items() if isinstance(v, dict)}
                    self.meta = {k: v for k, v in raw.items() if k not in ("entries", "about")}
            except (OSError, ValueError):
                self.entries = {}  # a broken cache is only a slower start, never a crash

    def get(self, sha256: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            hit = self.entries.get(sha256)
            return json.loads(json.dumps(hit)) if hit is not None else None

    def put(self, sha256: str, result: Dict[str, Any]) -> None:
        keep = {k: v for k, v in result.items() if k not in ("sha256", "cached")}
        keep.setdefault("settings", {})
        with self._lock:
            self.entries[sha256] = json.loads(json.dumps(keep))

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            head = {"about": "Text of the beta attachment files keyed by the SHA-256 of the file bytes: PDF "
                             "text layers, STEP headers, and OCR results. Built by python ocr.py --build-cache; "
                             "the OCR settings are explained in docs/ocr_settings.md.",
                    "tesseract_version": _tools()["tesseract_version"], **self.meta}
            head["tesseract_version"] = self.meta.get("tesseract_version") or _tools()["tesseract_version"]
            # One entry per line: the file stays small and a rebuild shows up as a readable diff.
            body = ",\n".join(f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False, separators=(',', ':'))}"
                               for k, v in sorted(self.entries.items()))
            text = json.dumps(head, indent=1, ensure_ascii=False)[:-2] + ',\n "entries": {\n' + body + "\n }\n}\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".ocr_cache.", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            # mkstemp makes the file private (0600); the cache is a normal data file that the
            # server may read as another user, so give it the usual permissions.
            os.chmod(tmp, 0o644)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


# --------------------------------------------------------------------------- #
# Evaluation (reads tests/rfq_beta_truth.json; product code above never does)
# --------------------------------------------------------------------------- #
LEGEND_PHRASES = {"itar": ["ITAR", "INTERNATIONAL TRAFFIC IN ARMS REGULATIONS"],
                  "ear": ["EAR", "EXPORT ADMINISTRATION REGULATIONS"],
                  "cui": ["CUI", "CONTROLLED TECHNICAL INFORMATION"]}


def _eval_tokens(text: str) -> Counter:
    out: Counter = Counter()
    for tok in text.upper().split():
        tok = re.sub(r"^[^A-Z0-9]+|[^A-Z0-9]+$", "", tok)
        if tok:
            out[tok] += 1
    return out


def key_fields(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The values an RFQ reader must get from a file: part numbers, revs, materials, finishes,
    quantity lists, RFQ number, respond-by date, and export legend phrases. Each distinct value
    once; a finish of NONE is left out because it says nothing."""
    fields: List[Dict[str, Any]] = []
    seen = set()

    def add(kind: str, value: Any, part: Optional[str] = None) -> None:
        value = re.sub(r"\s+", " ", str(value or "")).strip()
        if value and (kind, value, part) not in seen and value.upper() != "NONE":
            seen.add((kind, value, part))
            fields.append({"field": kind, "value": value, **({"part": part} if part else {})})

    rows = spec.get("lines") if spec.get("kind") == "rfq_form" else [spec]
    add("rfq_number", spec.get("rfq_number"))
    add("respond_by", spec.get("respond_by"))
    for row in rows or []:
        pn = row.get("part_number")
        add("part_number", pn)
        if row.get("rev"):
            add("rev", row["rev"], pn)
        add("material", row.get("material"))
        add("finish", row.get("finish"))
        if row.get("quantities"):
            add("quantities", " / ".join(f"{int(q):,}" for q in row["quantities"]))
    for phrase in LEGEND_PHRASES.get(spec.get("legend") or "", []):
        add("legend", phrase)
    return fields


def _field_regex(value: str) -> "re.Pattern[str]":
    chars = [re.escape(c) for c in value.upper() if not c.isspace()]
    return re.compile(r"(?<![A-Z0-9])" + r"\s*".join(chars) + r"(?![A-Z0-9])")


def _cell_chains(lines: List[Dict[str, Any]]) -> List[str]:
    """Text of wrapped table cells: a line entry followed by the entries left-aligned under it.
    An RFQ form prints "AL 6061-T6511 / HARD" over "ANODIZE TYPE III" over "CLASS 1" in one cell,
    and a row-by-row reading puts other columns between them. Only whitespace is added."""
    chains = []
    for i, a in enumerate(lines):
        text, cur = a["text"], a
        for _ in range(4):
            h = max(4, cur["bbox"][3] - cur["bbox"][1])
            nxt = [b for b in lines if b["page"] == cur["page"] and b is not cur
                   and 0 < b["bbox"][1] - cur["bbox"][1] < 2.4 * h and abs(b["bbox"][0] - cur["bbox"][0]) < 1.5 * h]
            if not nxt:
                break
            cur = min(nxt, key=lambda b: b["bbox"][1])
            text += " " + cur["text"]
            chains.append(text)
    return chains


def _rev_found(pn: str, rev: str, text: str, lines: List[Dict[str, Any]]) -> bool:
    """The rev letter next to its part number: "KF-3408 B" in a table row, "REV B" after it,
    or the title block cell to the right of the drawing number on the same row."""
    pat = _field_regex(pn).pattern + r"\s*(?:REV\.?\s*:?\s*)?" + r"\s*".join(re.escape(c) for c in rev.upper()) + r"(?![A-Z0-9])"
    if re.search(pat, text.upper()):
        return True
    rx = _field_regex(pn)
    for a in lines:
        if not rx.search(a["text"].upper()):
            continue
        ay0, ay1 = a["bbox"][1], a["bbox"][3]
        right = [b for b in lines if b["page"] == a["page"] and b["bbox"][0] >= a["bbox"][2] - 2
                 and ay0 - 0.3 * (ay1 - ay0) <= (b["bbox"][1] + b["bbox"][3]) / 2 <= ay1 + 0.3 * (ay1 - ay0)]
        if right:
            b = min(right, key=lambda b: b["bbox"][0])
            first = (b["text"].upper().split() or [""])[0].strip(".:,;|")
            if first == rev.upper():
                return True
    return False


def score(result: Dict[str, Any], truth: Dict[str, Any]) -> Dict[str, Any]:
    """Word recall and precision (multiset, uppercase, end punctuation stripped) and key-field
    recall (whitespace and case differences only) of one OCR result against its truth entry."""
    text = result.get("text") or ""
    lines = result.get("lines") or []
    want, got = _eval_tokens(" ".join(truth.get("words") or [])), _eval_tokens(text)
    hit = sum((want & got).values())
    upper = text.upper()
    chains = [c.upper() for c in _cell_chains(lines)]
    found, missed = [], []
    for f in key_fields(truth["spec"]):
        if f["field"] == "rev":
            ok = _rev_found(f["part"], f["value"], upper, lines)
        else:
            rx = _field_regex(f["value"])
            ok = bool(rx.search(upper)) or any(rx.search(c) for c in chains)
        (found if ok else missed).append(f)
    pages = result.get("pages") or 1
    return {"word_recall": hit / max(1, sum(want.values())), "word_precision": hit / max(1, sum(got.values())),
            "keys": len(found), "key_total": len(found) + len(missed),
            "key_recall": len(found) / max(1, len(found) + len(missed)),
            "missed": [f"{f['field']}={f['value']}" for f in missed],
            "seconds": result.get("seconds") or 0.0, "sec_per_page": (result.get("seconds") or 0.0) / pages,
            "confidence": result.get("confidence")}


def truth_ceiling() -> List[Tuple[str, List[str]]]:
    """Key fields the harness cannot find even in the truth words (should be empty)."""
    truth = json.loads(TRUTH_FILE.read_text(encoding="utf-8"))["files"]
    bad = []
    for rel, t in truth.items():
        if t.get("copyable") or not t.get("words"):
            continue
        text = " ".join(t["words"])
        missed = []
        for f in key_fields(t["spec"]):
            if f["field"] == "rev":
                ok = _rev_found(f["part"], f["value"], text.upper(), [])
            else:
                ok = bool(_field_regex(f["value"]).search(text.upper()))
            if not ok:
                missed.append(f"{f['field']}={f['value']}")
        if missed:
            bad.append((rel, missed))
    return bad


def parse_settings(spec: str) -> Dict[str, Any]:
    """"psm=3+11,threshold=2,target_dpi=300,median=3" -> a recipe dict."""
    out: Dict[str, Any] = {}
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if key == "psm":
            out["psm"] = [int(v) for v in val.split("+")]
        elif key in ("scale", "target_dpi", "min_conf"):
            out[key] = float(val)
        elif key == "raster_dpi":
            out[key] = val if val == "native" else float(val)
        elif key in ("median", "threshold"):
            out[key] = int(val)
        elif key in ("tessdata", "resample", "merge"):
            out[key] = val
        else:
            out[key] = val.lower() not in ("0", "false", "no", "off", "")
    return out


def parse_regions(spec: str) -> Dict[str, Dict[str, Any]]:
    """"title_block:6:600,line_table:4" -> {"title_block": {"psm": [6], "target_dpi": 600.0},
    "line_table": {"psm": [4]}}: the recipe's "regions" setting from the command line."""
    out: Dict[str, Dict[str, Any]] = {}
    for part in (spec or "").split(","):
        bits = [b.strip() for b in part.split(":")]
        if not bits[0]:
            continue
        entry: Dict[str, Any] = {"psm": [int(p) for p in (bits[1] if len(bits) > 1 and bits[1] else "6").split("+")]}
        if len(bits) > 2 and bits[2]:
            entry["target_dpi"] = float(bits[2])
        out[bits[0]] = entry
    return out


def settings_text(recipe: Dict[str, Any]) -> str:
    """The inverse of parse_settings, for tables and logs."""
    parts = []
    for k, v in sorted(recipe.items()):
        if v in (None, False, "") or (k != "threshold" and v == 0):
            continue
        if k == "psm":
            v = "+".join(str(p) for p in v)
        elif k == "tessdata":
            v = Path(str(v)).name
        elif k == "regions" and isinstance(v, dict):
            v = ";".join(f"{lb}:{'+'.join(str(p) for p in e.get('psm') or [6])}"
                         + (f":{int(float(e['target_dpi']))}" if e.get("target_dpi") else "") for lb, e in v.items())
        elif v is True:
            v = 1
        elif isinstance(v, float) and v.is_integer():
            v = int(v)
        parts.append(f"{k}={v}")
    return ",".join(parts)


# How the truth file names each kind of uncopyable file, and the source type the pipeline
# detects for it from the picture alone (classify). Only the harness uses this table.
RENDER_TYPES = {"scan": "scan", "copier": "lowres", "fax": "bilevel", "photo": "photo", "screen": "screen"}


def evaluate_data(rel: str, data: bytes, media: str, truth: Dict[str, Any], recipes: Dict[str, Dict[str, Any]],
                  timeout: float = 900.0) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        res = _ocr(data, media, recipes, started + timeout)
        res["seconds"] = round(time.monotonic() - started, 2)
    except OcrError as exc:
        res = {"text": "", "lines": [], "pages": 1, "seconds": round(time.monotonic() - started, 2), "error": str(exc)}
    out = score(res, truth)
    detected = [p.get("source") for p in (res.get("settings") or {}).get("pages") or []]
    out.update(file=rel, render=truth["render"], detected=detected, settings=res.get("settings"),
               pages=res.get("pages") or 1, error=res.get("error"))
    return out


def evaluate_one(rel: str, recipes: Dict[str, Dict[str, Any]], timeout: float = 900.0,
                 items: Optional[Dict[str, Tuple[bytes, str, Dict[str, Any]]]] = None) -> Dict[str, Any]:
    if items and rel in items:
        data, media, truth = items[rel]
    else:
        truth = json.loads(TRUTH_FILE.read_text(encoding="utf-8"))["files"][rel]
        path = HERE / "data" / rel
        data = path.read_bytes()
        media = {".pdf": "pdf", ".jpg": "jpg", ".png": "png"}[path.suffix.lower()]
    return evaluate_data(rel, data, media, truth, recipes, timeout)


def heldout_items(modes: Tuple[str, ...] = tuple(RENDER_TYPES), part: str = "all"
                  ) -> Dict[str, Tuple[bytes, str, Dict[str, Any]]]:
    """Degraded copies of the 11 digital beta PDFs, made with the beta generator's own effects
    (tools/make_rfq_beta.py: office scan, copier, fax, phone photo, viewer screenshot). There
    is one real photo and one real screenshot, too few to choose settings from, so the search
    also scores the "tune" part (6 drawings per kind); the "check" part (the other 5) is never
    seen by the search and tells whether the choice carries over."""
    import random as _random
    for p in (str(HERE), str(HERE / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import make_rfq_beta as mk  # the generator itself; only the harness imports it
    truth = json.loads(TRUTH_FILE.read_text(encoding="utf-8"))["files"]
    items: Dict[str, Tuple[bytes, str, Dict[str, Any]]] = {}
    digital = sorted(rel for rel, t in truth.items() if t.get("copyable") and t.get("words") and rel.endswith(".pdf"))
    if part != "all":
        digital = digital[0::2] if part == "tune" else digital[1::2]
    for rel in digital:
        t = truth[rel]
        email, spec = rel.split("/")[2], t["spec"]
        stem = spec["name"].rsplit(".", 1)[0]
        rnd = _random.Random(rel)
        for mode in modes:
            cfg: Dict[str, Any] = {"mode": mode}
            if mode in ("scan", "copier"):
                cfg["skew"] = round(rnd.uniform(0.3, 1.4) * rnd.choice([-1, 1]), 2)
            elif mode == "fax":
                cfg["skew"] = round(rnd.uniform(-0.5, 0.5), 2)
            elif mode == "photo":
                cfg["rename"] = f"{stem}_photo.jpg"
            elif mode == "screen":
                cfg["rename"] = f"{stem}_screenshot.png"
            key = (email, spec["name"])
            saved = mk.RENDER.get(key)
            mk.RENDER[key] = cfg
            try:
                name, data, tr = mk.render_file(email, spec)
            finally:
                if saved is None:
                    mk.RENDER.pop(key, None)
                else:
                    mk.RENDER[key] = saved
            media = {"jpg": "jpg", "png": "png"}.get(name.rsplit(".", 1)[-1], "pdf")
            items[f"heldout/{mode}/{email}/{name}"] = (data, media, tr)
    return items


def beta_uncopyable(render: Optional[str] = None) -> List[str]:
    truth = json.loads(TRUTH_FILE.read_text(encoding="utf-8"))["files"]
    return [rel for rel, t in truth.items() if not t.get("copyable") and render in (None, t.get("render"))]


def _cpu_seconds() -> Optional[float]:
    """CPU time of this process and of its finished children (tesseract, pdftoppm). CPU time,
    not the clock, is what predicts a host with a fraction of a CPU, and it does not change
    when other programs share the machine."""
    try:
        import resource
        ch = resource.getrusage(resource.RUSAGE_CHILDREN)
        return time.process_time() + ch.ru_utime + ch.ru_stime
    except Exception:  # noqa: BLE001 - not on Unix: fall back to the clock only
        return None


def evaluate(named: Dict[str, Dict[str, Dict[str, Any]]], files: Optional[List[str]] = None,
             workers: int = 1, items: Optional[Dict[str, Tuple[bytes, str, Dict[str, Any]]]] = None
             ) -> Dict[str, List[Dict[str, Any]]]:
    """Score each named set of recipes on the uncopyable beta files. One set at a time, so the
    CPU seconds of a set can be told apart; with one worker also the CPU seconds of each file."""
    from concurrent.futures import ThreadPoolExecutor
    files = files or beta_uncopyable()
    table: Dict[str, List[Dict[str, Any]]] = {}
    for name, recipes in named.items():
        before = _cpu_seconds()
        if workers <= 1:
            rows = []
            for rel in files:
                t0 = _cpu_seconds()
                row = evaluate_one(rel, recipes, items=items)
                t1 = _cpu_seconds()
                if t0 is not None and t1 is not None:
                    row["cpu_per_page"] = round((t1 - t0) / max(1, row.get("pages") or 1), 2)
                rows.append(row)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(lambda rel: evaluate_one(rel, recipes, items=items), files))
        after = _cpu_seconds()
        pages = sum(r.get("pages") or 1 for r in rows)
        for r in rows:
            if before is not None and after is not None:
                r["set_cpu_per_page"] = round((after - before) / max(1, pages), 2)
        table[name] = rows
    return table


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Totals over files: key fields found, mean word recall, precision and F1, mean s/page
    (clock) and CPU seconds per page."""
    n = max(1, len(rows))
    rec = sum(r["word_recall"] for r in rows) / n
    prec = sum(r["word_precision"] for r in rows) / n
    f1 = sum(2 * r["word_recall"] * r["word_precision"] / max(1e-9, r["word_recall"] + r["word_precision"])
             for r in rows) / n
    if rows and all("cpu_per_page" in r for r in rows):
        cpu = sum(r["cpu_per_page"] for r in rows) / n
    else:
        cpu = rows[0].get("set_cpu_per_page") if rows else None
    return {"keys": sum(r["keys"] for r in rows), "key_total": sum(r["key_total"] for r in rows),
            "word_recall": round(rec, 4), "word_precision": round(prec, 4), "word_f1": round(f1, 4),
            "sec_per_page": round(sum(r["sec_per_page"] for r in rows) / n, 2),
            "cpu_per_page": round(cpu, 2) if cpu is not None else None, "files": len(rows),
            "errors": sum(1 for r in rows if r.get("error"))}


def _print_table(table: Dict[str, List[Dict[str, Any]]]) -> None:
    def cpu(v: Any) -> str:
        return f"{v:6.2f}" if isinstance(v, (int, float)) else f"{'-':>6s}"

    for name, rows in table.items():
        print(f"\n== {name}")
        print(f"{'file':38s} {'type':7s} {'keys':>7s} {'recall':>7s} {'prec':>6s} {'conf':>5s} {'s/page':>6s} "
              f"{'cpu/pg':>6s}  missed")
        for r in rows:
            print(f"{r['file'].replace('rfq_beta/files/', ''):38s} {r['render']:7s} {r['keys']:>3d}/{r['key_total']:<3d} "
                  f"{r['word_recall']:7.3f} {r['word_precision']:6.3f} {(r['confidence'] or 0):5.1f} "
                  f"{r['sec_per_page']:6.2f} {cpu(r.get('cpu_per_page'))}  {'; '.join(r['missed'])[:100]}"
                  f"{'  ERROR ' + r['error'] if r.get('error') else ''}")
        for render in RENDER_TYPES:
            part = [r for r in rows if r["render"] == render]
            if part and len(part) < len(rows):
                s = summarize(part)
                print(f"{'  ' + render:38s} {'':7s} {s['keys']:>3d}/{s['key_total']:<3d} {s['word_recall']:7.3f} "
                      f"{s['word_precision']:6.3f} {'':5s} {s['sec_per_page']:6.2f} "
                      f"{cpu(s['cpu_per_page'] if all('cpu_per_page' in r for r in part) else None)}")
        s = summarize(rows)
        print(f"{'ALL':38s} {'':7s} {s['keys']:>3d}/{s['key_total']:<3d} {s['word_recall']:7.3f} "
              f"{s['word_precision']:6.3f} {'':5s} {s['sec_per_page']:6.2f} {cpu(s['cpu_per_page'])}")


# --------------------------------------------------------------------------- #
# The settings search (python ocr.py --evaluate --search)
# --------------------------------------------------------------------------- #
# Values tried for each source type, one setting at a time from the current best (coordinate
# descent, up to three rounds). None and False mean "off"; a two-pass psm is tried with both
# merges. The start point is the pipeline as it was before the search (START below).
# A first round on the 13 real files alone also tried --psm 4+11, 6+11 and 3+12 (never better
# than 3+11, and slower) and nodict and pis, which changed no word (see docs/ocr_settings.md);
# they are left out here to keep the search inside an hour on a shared 4-core machine.
_COMMON_SEARCH: Dict[str, List[Any]] = {
    "invert": [False, True],
    "psm": [[3], [4], [6], [11], [12], [3, 11], [11, 3]],
    "threshold": [None, 1, 2],
    "repair": [False, True],
    "min_conf": [None, 40, 60],
    "tess_dpi": [True, False],
    "deskew": [False, True],
    "autocontrast": [False, True],
    "unsharp": [False, True],
}
SEARCH: Dict[str, Dict[str, List[Any]]] = {
    "scan": {"raster_dpi": ["native", 400], **_COMMON_SEARCH, "median": [None, 3]},
    "lowres": {"target_dpi": [None, 300, 400], "flatten": [False, True], **_COMMON_SEARCH, "median": [None, 3]},
    "bilevel": {"target_dpi": [None, 300, 400], **_COMMON_SEARCH, "median": [None, 3]},
    "photo": {"page": [False, True], "target_dpi": [None, 300, 400], "color": [False, True],
              "flatten": [False, True], **_COMMON_SEARCH},
    "screen": {"page": [False, True], "target_dpi": [None, 240, 300, 400], "color": [False, True], **_COMMON_SEARCH},
}
START: Dict[str, Dict[str, Any]] = {
    "scan": {"raster_dpi": "native", "psm": [3], "tess_dpi": True},
    "lowres": {"raster_dpi": "native", "target_dpi": 300, "psm": [3], "tess_dpi": True},
    "bilevel": {"raster_dpi": "native", "target_dpi": 300, "psm": [3], "tess_dpi": True},
    "photo": {"page": True, "target_dpi": 300, "psm": [3], "tess_dpi": True},
    "screen": {"page": True, "target_dpi": 300, "psm": [3], "tess_dpi": True},
}


def _better(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Is summary a better choice than summary b? Key fields first; then word F1, where a
    slower setting has to earn its time and a faster one may give up a hair of F1."""
    if a["errors"] != b["errors"]:
        return a["errors"] < b["errors"]
    if a["keys"] != b["keys"]:
        return a["keys"] > b["keys"]
    gain = a["word_f1"] - b["word_f1"]
    cost = "cpu_per_page" if a.get("cpu_per_page") and b.get("cpu_per_page") else "sec_per_page"
    ratio = a[cost] / max(0.01, b[cost])
    if gain > 0.01:
        return True
    if gain > 0.002 and ratio < 1.5:
        return True
    return gain > -0.003 and ratio < 0.8


def search(render: str, workers: int = 1, rounds: int = 3, log: Optional[Path] = None,
           start: Optional[Dict[str, Any]] = None, heldout: bool = True) -> Dict[str, Any]:
    """Coordinate descent over SEARCH for one kind of file, scored on the real beta files of
    that kind plus (heldout=True) the "tune" part of the held-out pages (see heldout_items).
    Every run is appended to log (JSON lines) and read back on a rerun, so an interrupted
    search picks up where it stopped."""
    kind = RENDER_TYPES[render]
    real = beta_uncopyable(render)
    items = heldout_items((render,), "tune") if heldout else {}
    files = real + list(items)
    memo: Dict[str, Dict[str, Any]] = {}
    if log and log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("render") == render and bool(row.get("heldout")) == heldout:
                memo[row["settings"]] = row
    tried: List[Dict[str, Any]] = []

    def run(recipe: Dict[str, Any], change: str) -> Dict[str, Any]:
        key = settings_text(recipe)
        if key not in memo:
            rows = evaluate({key: {t: recipe for t in SOURCE_TYPES}}, files, workers, items)[key]
            real_rows = [r for r in rows if not r["file"].startswith("heldout/")]
            ho_rows = [r for r in rows if r["file"].startswith("heldout/")]
            memo[key] = {"render": render, "settings": key, "heldout": heldout, "summary": summarize(rows),
                         "real": summarize(real_rows), "held_out": summarize(ho_rows) if ho_rows else None,
                         "files": [{k: r[k] for k in ("file", "keys", "key_total", "missed", "word_recall",
                                                       "word_precision", "sec_per_page", "detected")} for r in rows]}
            if log:
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(memo[key]) + "\n")
        row = dict(memo[key], change=change)
        tried.append(row)
        return row

    current = dict(start or START[kind])
    best = run(current, "start")
    for _ in range(rounds):
        moved = False
        for dim, values in SEARCH[kind].items():
            if dim == "merge" and len(current.get("psm") or [3]) < 2:
                continue  # merging needs two passes
            for value in values:
                if value == current.get(dim) or (value in (None, False) and not current.get(dim)):
                    continue
                cand = dict(current)
                if value in (None, False):
                    cand.pop(dim, None)
                else:
                    cand[dim] = value
                change = f"{dim}={settings_text({dim: value}).partition('=')[2] or 'off'}"
                cands = [(cand, change)]
                if dim == "psm" and len(value) > 1:
                    # A second pass is only worth its time with the right merge, so each
                    # two-pass mode is tried with both.
                    cands = [(dict(cand, merge="fill"), change + " (fill)"), (dict(cand, merge="conf"), change + " (conf)")]
                for cand, change in cands:
                    row = run(cand, change)
                    if _better(row["summary"], best["summary"]):
                        current, best, moved = cand, row, True
                        row["kept"] = True
        if not moved:
            break
    return {"render": render, "type": kind, "files": real, "heldout": len(items), "best": current,
            "best_row": best, "tried": tried}


def _print_search(res: Dict[str, Any]) -> None:
    print(f"\n### {res['render']} ({res['type']}): {len(res['files'])} real file(s)"
          + (f" + {res['heldout']} held-out" if res.get("heldout") else "") + "\n")
    print("| change | key fields, real | key fields, held-out | word recall | word precision | CPU s/page | kept |")
    print("|---|---|---|---|---|---|---|")
    seen = set()
    for row in res["tried"]:
        if row["settings"] in seen and not row.get("kept"):
            continue  # a later round meets settings an earlier one already scored
        seen.add(row["settings"])
        s, r, h = row["summary"], row.get("real") or row["summary"], row.get("held_out")
        held = f"{h['keys']}/{h['key_total']}" if h else "-"
        print(f"| {row['change']} | {r['keys']}/{r['key_total']} | {held} | {s['word_recall']:.3f} | "
              f"{s['word_precision']:.3f} | {s['cpu_per_page'] or s['sec_per_page']:.2f} | "
              f"{'yes' if row.get('kept') else ''} |")
    s = res["best_row"]["summary"]
    print(f"\nbest for {res['render']}: `{settings_text(res['best'])}`  keys {s['keys']}/{s['key_total']}, "
          f"F1 {s['word_f1']:.3f}, {s['cpu_per_page'] or s['sec_per_page']:.2f} CPU s/page")


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _media_of(path: Path) -> str:
    return {".pdf": "pdf", ".png": "png", ".jpg": "jpg", ".jpeg": "jpg", ".step": "step", ".stp": "step"}.get(
        path.suffix.lower(), "")


def build_cache(path: Path = DEFAULT_CACHE, effort: str = "best", files: Optional[List[Path]] = None) -> int:
    """Read every beta file (all of them: text layers and STEP headers too, so a restarted server
    needs neither pypdf nor tesseract for them) and save the results keyed by SHA-256."""
    cache = OcrCache(path)
    cache.entries = {}
    t = _tools()
    lay = _layout()
    cache.meta = {"tesseract_version": t["tesseract_version"], "effort": effort,
                  "settings": {kind: recipe_record(r) for kind, r in recipes_for(effort).items()},
                  # the region detector the OCR results were tagged with (None: no regions in them)
                  "regions": {"model": Path(lay.MODEL_PATH).name, "model_bytes": os.path.getsize(lay.MODEL_PATH),
                              "conf": REGION_CONF} if lay else None,
                  "made_by": "python ocr.py --build-cache"}
    files = files if files is not None else sorted(p for p in BETA_FILES.rglob("*") if p.is_file())
    failed = 0
    for p in files:
        res = file_text(p.read_bytes(), _media_of(p), p.name, effort=effort, timeout=max(OCR_TIMEOUT, 900))
        try:
            rel = p.resolve().relative_to(HERE / "data").as_posix()
        except ValueError:
            rel = p.name
        if res["method"] == "none" or res.get("error"):
            failed += 1  # never cache a failure: the next reader should try again
        else:
            cache.put(res["sha256"], dict(res, file=rel))
        conf = f"conf {res['confidence']:5.1f}" if res.get("confidence") is not None else " " * 10
        print(f"{res['method']:12s} {res['seconds']:6.1f} s  {conf}  {rel}"
              + (f"  ERROR {res['error']}" if res.get("error") else ""))
    cache.save()
    counts = Counter(e["method"] for e in cache.entries.values())
    print(f"{len(cache.entries)} results saved to {path} ({', '.join(f'{n} {m}' for m, n in sorted(counts.items()))})"
          + (f"; {failed} file(s) failed" if failed else ""))
    return 1 if failed else 0


def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--settings", action="append", default=[],
                    help="for --evaluate: baseline | before | best | fast | key=value,... (repeatable)")
    ap.add_argument("--only", default="", help="for --evaluate: comma-separated substrings of file paths")
    ap.add_argument("--type", default="", help="for --evaluate: comma-separated kinds (scan,copier,fax,photo,screen)")
    ap.add_argument("--search", action="store_true", help="for --evaluate: search the settings per kind of file")
    ap.add_argument("--rounds", type=int, default=3, help="for --search: passes over the settings")
    ap.add_argument("--log", default=None, help="for --search: JSON-lines log; a rerun resumes from it")
    ap.add_argument("--real-only", action="store_true",
                    help="for --search: score on the real beta files only, without the held-out pages")
    ap.add_argument("--heldout", nargs="?", const="check", choices=("check", "tune", "all"),
                    help="for --evaluate: score degraded copies of the digital beta PDFs instead "
                         "(check: the 5 per kind the search never saw; tune; all)")
    ap.add_argument("--model", default=None,
                    help="for --evaluate: also score best and fast with the eng.traineddata in this folder")
    ap.add_argument("--regions", default="",
                    help="for --evaluate: add a second reading of these detected regions to every set, e.g. "
                         "title_block:6:600,line_table:6:450 (label:psm:target dpi)")
    ap.add_argument("--region-merge", default="conf", choices=("conf", "fill", "replace"),
                    help="for --evaluate --regions: how the region reading joins the page's")
    ap.add_argument("--workers", type=int, default=1, help="for --evaluate: files OCR'd side by side")
    ap.add_argument("--effort", default="best", choices=("best", "fast"))
    ap.add_argument("--cache", default=None, help="cache file for FILE mode (default: none)")
    ap.add_argument("--json", default=None, help="for --evaluate: also write the rows here")
    args = ap.parse_args(argv)
    if args.build_cache:
        return build_cache(effort=args.effort)
    if args.evaluate:
        if not available()["ocr"]:
            print("tesseract is not installed")
            return 1
        bad = truth_ceiling()
        for rel, missed in bad:
            print(f"harness cannot find in the truth words of {rel}: {missed}")
        print(f"tesseract {_tools()['tesseract_version']} ({_tools()['tesseract']}), OMP_THREAD_LIMIT=1, "
              f"{args.workers} file(s) at a time")
        renders = [r for r in args.type.split(",") if r] or list(RENDER_TYPES)
        if args.search:
            results = []
            for render in renders:
                res = search(render, args.workers, args.rounds, Path(args.log) if args.log else None,
                             heldout=not args.real_only)
                _print_search(res)
                results.append(res)
            print("\nchosen per kind:")
            for res in results:
                print(f"  {res['type']:8s} {settings_text(res['best'])}")
            if args.json:
                Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
            return 0
        named: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for s in args.settings or ["baseline", "before", "best", "fast"]:
            if s == "baseline":
                named[s] = {t: dict(BASELINE) for t in SOURCE_TYPES}
            elif s == "before":
                named[s] = {t: dict(START[t]) for t in SOURCE_TYPES}
            elif s in ("best", "fast"):
                named[s] = recipes_for(s)
            else:
                named[s] = {t: parse_settings(s) for t in SOURCE_TYPES}
        if args.regions:
            spec = parse_regions(args.regions)
            named = {f"{n}+regions": {t: dict(r, regions=spec, region_merge=args.region_merge) for t, r in rs.items()}
                     for n, rs in named.items()}
        if args.model:
            for s in ("best", "fast"):
                named[f"{s}+{Path(args.model).name}"] = {t: dict(r, tessdata=args.model)
                                                         for t, r in recipes_for(s).items()}
        items = heldout_items(tuple(renders), args.heldout) if args.heldout else None
        files = list(items) if items else [f for r in renders for f in beta_uncopyable(r)]
        if args.only:
            files = [f for f in files if any(o in f for o in args.only.split(","))]
        table = evaluate(named, files, args.workers, items)
        _print_table(table)
        if args.json:
            Path(args.json).write_text(json.dumps(table, indent=1), encoding="utf-8")
        return 0
    if not args.files:
        ap.print_help()
        return 2
    cache = OcrCache(args.cache) if args.cache else None
    for name in args.files:
        p = Path(name)
        res = file_text(p.read_bytes(), _media_of(p), p.name, cache=cache, effort=args.effort)
        conf = f"{res['confidence']:.1f}" if res.get("confidence") is not None else "-"
        print(f"== {p.name}: {res['method']}, confidence {conf}, {res['seconds']:.1f} s, pages {res.get('pages')}"
              + (f", error: {res['error']}" if res.get("error") else ""))
        if res.get("settings"):
            print(f"   settings: {json.dumps(res['settings'])}")
        print(res["text"])
    return 0


if __name__ == "__main__":
    if "--pdf-images" in sys.argv:
        i = sys.argv.index("--pdf-images")
        sys.exit(_pdf_images_main(sys.argv[i + 1] if len(sys.argv) > i + 1 else ""))
    sys.exit(main(sys.argv[1:]))
