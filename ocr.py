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
# Refuse to decode pictures bigger than this at all (a small PNG can claim a huge size).
MAX_INPUT_PIXELS = 80_000_000
# A PDF whose text layer has fewer letters and digits than this per page is treated as a scan.
# Scanner software sometimes adds a line like "Scanned by ScanDesk"; that is not the document.
TEXT_LAYER_MIN_CHARS = 40
# Long side of the page, in inches, used to estimate the resolution of a photo or screenshot
# (letter and A4 landscape drawings and forms are 11 to 11.7 in).
PAGE_LONG_SIDE_IN = 11.0

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
        tess = shutil.which("tesseract")
        version = None
        if tess:
            try:
                proc = subprocess.run([tess, "--version"], capture_output=True, timeout=20)
                out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
                m = re.search(r"tesseract\s+v?(\d+\.\d+(?:\.\d+)?)", out)
                version = m.group(1) if m else None
                if not version:
                    tess = None
            except (OSError, subprocess.SubprocessError):
                tess = None
        _tool_info.update(tesseract=tess, tesseract_version=version, pdftoppm=shutil.which("pdftoppm"),
                          pdfimages=shutil.which("pdfimages"), pdftotext=shutil.which("pdftotext"))
        return _tool_info


def available() -> Dict[str, Any]:
    t = _tools()
    return {"tesseract": bool(t["tesseract"]), "tesseract_version": t["tesseract_version"],
            "pdftoppm": bool(t["pdftoppm"]), "pillow": Image is not None, "ocr": bool(t["tesseract"])}


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
        raise OcrError(f"{what} took too long (the limit is {OCR_TIMEOUT:g} s per file)")
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
#   scale         resize factor before OCR (Lanczos); 0 means "reach target_dpi"
#   target_dpi    with scale 0: upscale low-resolution sources until the text is this dense
#   median        median filter size (3 removes the 1-pixel salt noise of a fax line)
#   flatten       remove uneven light: subtract the blurred paper level
#   autocontrast  stretch the gray levels
#   unsharp       unsharp mask after resizing
#   deskew        find the skew with a projection profile and rotate it out
#   page          find the sheet of paper in a photo or screenshot and warp it flat
#   psm           tesseract page segmentation modes; with two, the passes are merged
#   threshold     tesseract thresholding_method (0 Otsu, 1 adaptive Otsu, 2 Sauvola)
#   tess_dpi      tell tesseract the resolution of the picture it gets
#   pis           -c preserve_interword_spaces=1
# The per-source recipes below are the measured winners (docs/ocr_settings.md). Change a value
# there and here together, and rerun python ocr.py --evaluate.
BASELINE: Dict[str, Any] = {"raster_dpi": 300, "psm": [3], "raw": True}

RECIPES: Dict[str, Dict[str, Any]] = {
    "scan": {"raster_dpi": "native", "psm": [3], "tess_dpi": True},
    "lowres": {"raster_dpi": "native", "scale": 0, "target_dpi": 300, "psm": [3], "tess_dpi": True},
    "bilevel": {"raster_dpi": "native", "scale": 0, "target_dpi": 300, "psm": [3], "tess_dpi": True},
    "photo": {"page": True, "scale": 0, "target_dpi": 300, "psm": [3], "tess_dpi": True},
    "screen": {"page": True, "scale": 0, "target_dpi": 300, "psm": [3], "tess_dpi": True},
}
FAST_RECIPES: Dict[str, Dict[str, Any]] = {k: dict(v, psm=[3]) for k, v in RECIPES.items()}
SOURCE_TYPES = tuple(RECIPES)


# --------------------------------------------------------------------------- #
# Pictures
# --------------------------------------------------------------------------- #
def _open_image(data: bytes) -> "Image.Image":
    if Image is None:
        raise OcrError("Pillow is not installed")
    try:
        img = Image.open(io.BytesIO(data))
        if img.size[0] * img.size[1] > MAX_INPUT_PIXELS:
            raise OcrError(f"the picture is too large to read ({img.size[0]} x {img.size[1]} pixels)")
        img.load()
    except OcrError:
        raise
    except Exception as exc:  # noqa: BLE001 - Pillow raises many types for broken files
        raise OcrError(f"the picture could not be read ({type(exc).__name__})")
    return img


def _gray(img: "Image.Image") -> "Image.Image":
    if img.mode == "L":
        return img
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)  # transparent screenshots read as white paper
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


def _is_bilevel(img: "Image.Image") -> bool:
    if img.mode == "1":
        return True
    hist = _gray(img).histogram()
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
        rows = list(r.resize((1, r.size[1]), Image.BOX).getdata())
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
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor=255)


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
    color = img.mode not in ("1", "L", "LA", "I", "I;16")
    if color:
        sat = ImageStat.Stat(img.convert("RGB").convert("HSV").split()[1]).mean[0]
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
    scale = float(recipe.get("scale", 1.0) if recipe.get("scale", 1.0) is not None else 1.0)
    if scale == 0:
        scale = max(1.0, float(recipe.get("target_dpi") or 300) / max(dpi, 30.0))
    if img.size[0] * img.size[1] * scale * scale > MAX_OCR_PIXELS:
        scale = max(0.25, (MAX_OCR_PIXELS / float(img.size[0] * img.size[1])) ** 0.5)
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
            g = g.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
        steps["deskew"] = angle
    return g, scale, steps


# --------------------------------------------------------------------------- #
# Tesseract
# --------------------------------------------------------------------------- #
def _tesseract_words(path: str, psm: int, dpi: Optional[float], recipe: Dict[str, Any],
                     deadline: float) -> List[Dict[str, Any]]:
    tess = _tools()["tesseract"]
    if not tess:
        raise OcrError("tesseract is not installed")
    cmd = [tess, path, "stdout", "-l", "eng", "--oem", "1", "--psm", str(int(psm))]
    tessdata = recipe.get("tessdata") or OCR_TESSDATA
    if tessdata:
        cmd += ["--tessdata-dir", str(tessdata)]
    if dpi:
        cmd += ["--dpi", str(int(round(dpi)))]
    if recipe.get("threshold") is not None:
        cmd += ["-c", f"thresholding_method={int(recipe['threshold'])}"]
    if recipe.get("pis"):
        cmd += ["-c", "preserve_interword_spaces=1"]
    if recipe.get("nodict"):
        # Part numbers and specs are not dictionary words; without the word lists the model
        # reads the characters it sees instead of the nearest English word.
        cmd += ["-c", "load_system_dawg=0", "-c", "load_freq_dawg=0"]
    cmd.append("tsv")
    out = _run(cmd, deadline, "tesseract").decode("utf-8", "replace")
    words = []
    for line in out.splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 12 or f[0] != "5":
            continue
        text = f[11].strip()
        try:
            conf = float(f[10])
        except ValueError:
            continue
        if not text or conf < 0:
            continue
        x, y, w, h = int(f[6]), int(f[7]), int(f[8]), int(f[9])
        words.append({"text": text, "conf": conf, "box": [x, y, x + w, y + h],
                      "line": (psm, int(f[2]), int(f[3]), int(f[4]))})
    if recipe.get("repair"):
        _repair_case(words)
    return words


def _repair_case(words: List[Dict[str, Any]]) -> None:
    """In Helvetica an uppercase I and a lowercase l are the same glyph, so tesseract writes
    "Cl-10442" and "TYPE Ill". On a line that is otherwise uppercase, a word whose only
    lowercase letters are l is really uppercase: make it so."""
    lines: Dict[Tuple, List[Dict[str, Any]]] = {}
    for w in words:
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


def _merge_passes(passes: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Words of the first pass, plus words from later passes where the first found nothing.
    Drawings mix paragraphs (notes) with sparse boxed text (title block, callouts), and each
    page segmentation mode misses some of one or the other."""
    merged = list(passes[0])
    for extra in passes[1:]:
        for w in extra:
            if all(_overlap(w["box"], m["box"]) < 0.3 for m in merged):
                merged.append(w)
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


def _ocr_picture(img: "Image.Image", info: Dict[str, Any], recipe: Dict[str, Any], tmp: str,
                 deadline: float, tag: str) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    g, scale, steps = prepare(img, info, recipe)
    path = os.path.join(tmp, f"{tag}.tif")
    dpi = float(info.get("dpi") or 300) * scale
    g.save(path, "TIFF", dpi=(dpi, dpi))  # uncompressed: fastest to write and to read
    passes = [_tesseract_words(path, psm, dpi if recipe.get("tess_dpi") else None, recipe, deadline)
              for psm in (recipe.get("psm") or [3])]
    return _merge_passes(passes), scale, steps


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
            im = {"width": int(f[3]), "height": int(f[4]), "color": f[5], "bits": int(f[7]),
                  "ppi": min(float(f[12]), float(f[13]))}
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


def _pdf_pages_fallback(data: bytes, deadline: float) -> List[bytes]:
    """Without pdftoppm: the page pictures themselves, pulled out with pypdf in a child process
    (the same care attachments.py takes with hostile PDFs)."""
    try:
        proc = subprocess.run([sys.executable, str(HERE / "ocr.py"), "--pdf-images"], input=data,
                              capture_output=True, timeout=max(1.0, deadline - time.monotonic()), cwd=str(HERE))
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
    return [base64.b64decode(s) for s in result.get("images") or []]


def _pdf_images_main() -> int:
    """Child process: PDF bytes on stdin, JSON {images: [base64 PNG or JPEG], error} on stdout."""
    try:
        import resource  # Unix only: cap memory so a PDF bomb cannot take the host down
        limit = 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:  # noqa: BLE001
        pass
    import base64
    out: Dict[str, Any] = {"images": [], "error": None}
    try:
        import logging
        logging.disable(logging.CRITICAL)
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(sys.stdin.buffer.read()))
        for page in list(reader.pages)[:OCR_MAX_PAGES]:
            pics = sorted(page.images, key=lambda im: len(im.data), reverse=True)
            if pics:
                out["images"].append(base64.b64encode(pics[0].data).decode("ascii"))
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"the PDF pictures could not be read ({type(exc).__name__})"
    sys.stdout.write(json.dumps(out))
    return 0


def _pdf_text_layer(data: bytes) -> Dict[str, Any]:
    """{text, pages, per_page, error}. pypdf (through attachments.py, in a child process) gives the
    text; pdftotext, when installed, also says which pages have none, for mixed PDFs."""
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
            out = subprocess.run([tool, "-q", "-", "-"], input=data, capture_output=True, timeout=30).stdout
            pages = out.decode("utf-8", "replace").split("\f")
            if pages and not pages[-1].strip():
                pages = pages[:-1]
            if pages:
                result["per_page"] = [len(re.findall(r"[A-Za-z0-9]", p)) for p in pages]
                if not result["text"].strip():
                    result["text"] = "\n".join(pages)
                    result["pages"] = result["pages"] or len(pages)
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def _has_text_layer(layer: Dict[str, Any]) -> bool:
    chars = len(re.findall(r"[A-Za-z0-9]", layer.get("text") or ""))
    pages = max(1, int(layer.get("pages") or 1))
    return chars >= TEXT_LAYER_MIN_CHARS * pages


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
            pics = _pdf_pages_fallback(data, deadline)
            return [(_open_image(b) if Image is not None else b, {"dpi": 300, "type": "scan"},
                     recipes["scan"]) for b in pics]
        for p in pages:
            im = images.get(p)
            if im and im["ppi"] >= 50:
                pre = "bilevel" if im["bits"] == 1 else ("scan" if im["ppi"] >= 250 else "lowres")
                native = im["ppi"]
            else:
                pre, native = "scan", 300.0
            recipe = recipes[pre]
            dpi = native if recipe.get("raster_dpi", "native") == "native" else float(recipe["raster_dpi"])
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
        pages, scales, used = [], [], []
        for i, (pic, info, recipe) in enumerate(items, start=1):
            if isinstance(pic, (bytes, bytearray)):
                ext = {b"P5": ".pgm", b"P4": ".pbm"}.get(bytes(pic[:2]), ".jpg")
                ext = ".png" if bytes(pic[:4]) == b"\x89PNG" else ext
                path = os.path.join(tmp, f"raw{i}{ext}")
                with open(path, "wb") as fh:
                    fh.write(pic)
                dpi = info.get("dpi") if (recipe.get("tess_dpi") or ext == ".pgm") else None
                words = _merge_passes([_tesseract_words(path, psm, dpi, recipe, deadline)
                                       for psm in recipe.get("psm") or [3]])
                scale, steps = 1.0, {}
            else:
                words, scale, steps = _ocr_picture(pic, info, recipe, tmp, deadline, f"p{i}")
            pages.append(words)
            scales.append(scale)
            used.append({"page": i, "source": info.get("type"), "dpi": info.get("source_dpi") or info.get("dpi"),
                         "psm": list(recipe.get("psm") or [3]),
                         "threshold": recipe.get("threshold"), **steps})
    out = _assemble(pages, scales)
    out["pages"] = len(items)
    out["settings"] = {"engine": f"tesseract {_tools()['tesseract_version']}", "oem": 1, "lang": "eng",
                       "model": str(recipes.get("scan", {}).get("tessdata") or OCR_TESSDATA or "installed"),
                       "pages": used}
    out["seconds"] = round(time.monotonic() - started, 2)
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
        layer = _pdf_text_layer(data)
        if _has_text_layer(layer):
            per_page = layer.get("per_page") or []
            blank = [i + 1 for i, n in enumerate(per_page) if n < TEXT_LAYER_MIN_CHARS]
            text = (layer.get("text") or "").strip()
            if not blank or not allow_ocr or not available()["ocr"]:
                return _result("text-layer", text=text, pages=layer.get("pages"), sha256=sha,
                               seconds=round(time.monotonic() - started, 2))
            ocr_pages = blank  # a mixed PDF: typed pages plus scanned ones
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
        return _result("none", pages=layer.get("pages"), sha256=sha, error=str(exc),
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
    return _result("step-header", text="\n".join(lines), pages=None, seconds=round(time.monotonic() - started, 3))


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class OcrCache:
    """Saved results keyed by the SHA-256 of the file bytes, so the committed beta files never
    need OCR on a slow host. Each entry keeps the tesseract version and settings it was made
    with; an entry is used as long as the file hash matches (rebuild with --build-cache)."""

    def __init__(self, path: "str | Path | None"):
        self.path = Path(path) if path else None
        self.entries: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                entries = raw.get("entries") if isinstance(raw, dict) else None
                if isinstance(entries, dict):
                    self.entries = {k: v for k, v in entries.items() if isinstance(v, dict)}
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
            payload = {"about": "OCR results for files whose text cannot be copied, keyed by the SHA-256 of "
                                "the file bytes. Built by python ocr.py --build-cache; see docs/ocr_settings.md.",
                       "tesseract_version": _tools()["tesseract_version"],
                       "entries": dict(sorted(self.entries.items()))}
            text = json.dumps(payload, indent=1, ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".ocr_cache.", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
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
    """"psm=3+11,threshold=2,scale=1.5,median=3" -> a recipe dict (applied to every source type)."""
    out: Dict[str, Any] = {}
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if key == "psm":
            out["psm"] = [int(v) for v in val.split("+")]
        elif key in ("scale", "target_dpi"):
            out[key] = float(val)
        elif key == "raster_dpi":
            out[key] = val if val == "native" else float(val)
        elif key in ("median", "threshold"):
            out[key] = int(val)
        elif key in ("tessdata", "resample"):
            out[key] = val
        else:
            out[key] = val.lower() not in ("0", "false", "no", "off", "")
    return out


def evaluate_one(rel: str, recipes: Dict[str, Dict[str, Any]], timeout: float = 900.0) -> Dict[str, Any]:
    truth = json.loads(TRUTH_FILE.read_text(encoding="utf-8"))["files"][rel]
    path = HERE / "data" / rel
    data = path.read_bytes()
    media = {".pdf": "pdf", ".jpg": "jpg", ".png": "png"}[path.suffix.lower()]
    started = time.monotonic()
    try:
        res = _ocr(data, media, recipes, started + timeout)
        res["seconds"] = round(time.monotonic() - started, 2)
    except OcrError as exc:
        res = {"text": "", "lines": [], "pages": 1, "seconds": round(time.monotonic() - started, 2), "error": str(exc)}
    out = score(res, truth)
    out.update(file=rel, render=truth["render"], settings=res.get("settings"), error=res.get("error"))
    return out


def beta_uncopyable() -> List[str]:
    truth = json.loads(TRUTH_FILE.read_text(encoding="utf-8"))["files"]
    return [rel for rel, t in truth.items() if not t.get("copyable")]


def evaluate(named: Dict[str, Dict[str, Dict[str, Any]]], files: Optional[List[str]] = None,
             workers: int = 1) -> Dict[str, List[Dict[str, Any]]]:
    """Score each named set of recipes on the uncopyable beta files."""
    from concurrent.futures import ThreadPoolExecutor
    files = files or beta_uncopyable()
    jobs = [(name, rel) for name in named for rel in files]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda job: (job[0], evaluate_one(job[1], named[job[0]])), jobs))
    table: Dict[str, List[Dict[str, Any]]] = {name: [] for name in named}
    for name, row in results:
        table[name].append(row)
    return table


def _print_table(table: Dict[str, List[Dict[str, Any]]]) -> None:
    for name, rows in table.items():
        print(f"\n== {name}")
        print(f"{'file':44s} {'type':7s} {'keys':>7s} {'recall':>7s} {'prec':>6s} {'conf':>5s} {'s/page':>7s}  missed")
        for r in rows:
            print(f"{r['file'].replace('rfq_beta/files/', ''):44s} {r['render']:7s} {r['keys']:>3d}/{r['key_total']:<3d} "
                  f"{r['word_recall']:7.3f} {r['word_precision']:6.3f} {(r['confidence'] or 0):5.1f} "
                  f"{r['sec_per_page']:7.2f}  {'; '.join(r['missed'])[:120]}{'  ERROR ' + r['error'] if r.get('error') else ''}")
        keys = sum(r["keys"] for r in rows)
        total = sum(r["key_total"] for r in rows)
        print(f"{'ALL':44s} {'':7s} {keys:>3d}/{total:<3d} {sum(r['word_recall'] for r in rows) / len(rows):7.3f} "
              f"{sum(r['word_precision'] for r in rows) / len(rows):6.3f} {'':5s} "
              f"{sum(r['sec_per_page'] for r in rows) / len(rows):7.2f}")


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _media_of(path: Path) -> str:
    return {".pdf": "pdf", ".png": "png", ".jpg": "jpg", ".jpeg": "jpg", ".step": "step", ".stp": "step"}.get(
        path.suffix.lower(), "")


def build_cache(path: Path = DEFAULT_CACHE, effort: str = "best") -> int:
    cache = OcrCache(path)
    cache.entries = {}
    files = sorted(p for p in BETA_FILES.rglob("*") if p.is_file())
    for p in files:
        res = file_text(p.read_bytes(), _media_of(p), p.name, effort=effort, timeout=max(OCR_TIMEOUT, 900))
        rel = p.relative_to(HERE).as_posix()
        if res["method"] == "ocr" and not res.get("error"):
            cache.put(res["sha256"], dict(res, file=rel))
        print(f"{res['method']:12s} {res['seconds']:6.1f} s  conf {res['confidence'] or 0:5.1f}  {rel}"
              + (f"  ERROR {res['error']}" if res.get("error") else ""))
    cache.save()
    print(f"{len(cache.entries)} OCR results saved to {path}")
    return 0


def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--settings", action="append", default=[],
                    help="baseline | fast | best | key=value,... (repeatable), for --evaluate")
    ap.add_argument("--only", default="", help="for --evaluate: comma-separated substrings of file paths")
    ap.add_argument("--workers", type=int, default=1, help="for --evaluate: files OCR'd side by side")
    ap.add_argument("--effort", default="best", choices=("best", "fast"))
    ap.add_argument("--cache", default=None, help="cache file for FILE mode (default: none)")
    ap.add_argument("--json", default=None, help="for --evaluate: also write the rows here")
    args = ap.parse_args(argv)
    if args.build_cache:
        return build_cache(effort=args.effort)
    if args.evaluate:
        bad = truth_ceiling()
        for rel, missed in bad:
            print(f"harness cannot find in the truth words of {rel}: {missed}")
        named: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for s in args.settings or ["baseline", "fast", "best"]:
            if s == "baseline":
                named[s] = {t: dict(BASELINE) for t in SOURCE_TYPES}
            elif s in ("best", "fast"):
                named[s] = recipes_for(s)
            else:
                named[s] = {t: parse_settings(s) for t in SOURCE_TYPES}
        files = beta_uncopyable()
        if args.only:
            files = [f for f in files if any(o in f for o in args.only.split(","))]
        table = evaluate(named, files, args.workers)
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
        sys.exit(_pdf_images_main())
    sys.exit(main(sys.argv[1:]))
