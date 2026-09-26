"""
Build the training data for the page layout detector (layout.py, models/rfq_layout.onnx).

The detector finds the regions of a page that matter to the RFQ extractor, so Tesseract can read
each one with settings suited to it and the extractor knows where every piece of text came from:

    0 title_block         drawing title block: company, TITLE, MATERIAL, FINISH, DWG NO., REV,
                          SCALE, SHEET cells and the tolerance block beside them
    1 revision_block      the REVISIONS table
    2 notes               NOTES: and its numbered lines
    3 export_legend       ITAR or EAR warning box, CUI banners, the CUI designation block
    4 proprietary_notice  the PROPRIETARY AND CONFIDENTIAL lines
    5 form_header         RFQ or PO header: company, document title, number/date/respond-by cells
    6 line_table          the line-item table on RFQ forms and POs
    7 requirements        numbered quote requirements (RFQ forms), notes and quality clauses (POs)

Nobody draws the boxes by hand. The sample files are laid out as drawing operations on a
docgen.Page (drawings.drawing_pages, attachments.pages_for), so the boxes are derived from those
operations, anchored on the printed labels (the rectangle that holds "MATERIAL", the table under
"REVISIONS", the box around "WARNING:") rather than on fixed coordinates. That way the labels
follow the renderer when its layout is tweaked, and the renderers themselves are not changed.

Each synthetic page is a random spec (vocabulary from data/sample_emails.json, recombined) rendered
through the same scan, copier, fax, phone photo and screenshot effects that made the beta inbox
(tools/make_rfq_beta.py), with the boxes carried through the same geometry (skew rotation, photo
perspective, screenshot offset). The 30 beta files are never rendered into training: they are the
honest test set ("beta"), built from the real files on disk with boxes derived from their specs.

    python tools/make_layout_dataset.py --out ../yolo_work/layout_data_v2 --train 1200 --val 200 --workers 3
                                                  (the shipped model's data: about 4.5 minutes on 3 cores)
    python tools/make_layout_dataset.py --out ../yolo_work/layout_data_v2 --beta-only
    python tools/make_layout_dataset.py --preview ../yolo_work/preview --count 12

Needs Pillow and the pdftoppm program (poppler) for the beta set. Writes YOLO format:
    <out>/images/{train,val,beta}/*.jpg   <out>/labels/{train,val,beta}/*.txt   <out>/data.yaml
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import attachments  # noqa: E402
import docgen  # noqa: E402
import drawings  # noqa: E402
import make_rfq_beta as beta  # noqa: E402  (the scan, copier, fax, photo and screenshot effects)

from PIL import Image, ImageChops, ImageDraw, ImageFilter  # noqa: E402

CLASSES = ["title_block", "revision_block", "notes", "export_legend", "proprietary_notice",
           "form_header", "line_table", "requirements"]
CID = {name: i for i, name in enumerate(CLASSES)}

Box = Tuple[float, float, float, float]
Label = Tuple[int, Box]

# Saved images are at most this many pixels on the long side. Training runs at 640, and the
# runtime letterboxes every page to 640 too, so more resolution than this only costs disk and time.
MAX_SIDE = 1280


# --------------------------------------------------------------------------- #
# Labels from the page operations
# --------------------------------------------------------------------------- #
def raster_size(size: float, k: Optional[float]) -> float:
    """The font size, in points, that text of this size really has in a picture drawn at k pixels per
    point. docgen.to_image asks Pillow for a whole number of pixels (round(size * k)), so a 7 pt line
    drawn at 150 dpi is 15 px tall and 2.9% wider than 7 pt Helvetica; at 72 to 220 dpi the error runs
    from -4% to +5%, which is 10 to 25 points at the end of a long note. k None means exact metrics
    (vector PDFs rendered by pdftoppm place every glyph at its Helvetica width)."""
    if not k:
        return size
    return max(1, int(round(size * k))) / k


def text_box(o: Dict[str, Any], k: Optional[float] = None) -> Box:
    """The ink box of a text op in page points: measured width, Helvetica cap height and descender,
    at the size the text really has in a picture drawn at k pixels per point (see raster_size)."""
    size = raster_size(o["size"], k)
    w = docgen.text_width(o["text"], size, o["bold"])
    x = o["x"] - (w if o["anchor"] == "end" else w / 2 if o["anchor"] == "middle" else 0.0)
    return (x, o["y"] - 0.76 * size, x + w, o["y"] + 0.22 * size)


def union(boxes: Iterable[Box]) -> Optional[Box]:
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def pad(b: Box, d: float) -> Box:
    return (b[0] - d, b[1] - d, b[2] + d, b[3] + d)


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inside(inner: Box, outer: Box, tol: float = 0.6) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


class PageIndex:
    """The text and visible rectangles of one page, with the lookups the labelers need."""

    def __init__(self, page: docgen.Page, k: Optional[float] = None):
        self.page = page
        self.k = k  # pixels per point of the picture the labels are for (text_box)
        self.ops = page.ops
        self.page_area = page.width * page.height
        self.texts: List[Tuple[int, Dict[str, Any], Box]] = []
        self.rects: List[Box] = []
        for i, (kind, o) in enumerate(page.ops):
            if kind == "text":
                self.texts.append((i, o, text_box(o, k)))
            elif kind == "rect":
                stroked = bool(o.get("stroke")) and o.get("lw", 0.6) > 0
                filled = bool(o.get("fill")) and tuple(o["fill"]) != (1.0, 1.0, 1.0) and tuple(o["fill"]) != (1, 1, 1)
                if stroked or filled:  # white patches behind dimension text are not boxes anyone sees
                    x0, x1 = sorted((o["x"], o["x"] + o["w"]))
                    y0, y1 = sorted((o["y"], o["y"] + o["h"]))
                    self.rects.append((x0, y0, x1, y1))

    def named(self, text: str) -> List[Tuple[int, Dict[str, Any], Box]]:
        return [t for t in self.texts if t[1]["text"] == text]

    def starting(self, prefix: str) -> List[Tuple[int, Dict[str, Any], Box]]:
        return [t for t in self.texts if t[1]["text"].startswith(prefix)]

    def enclosing_rect(self, box: Box, max_frac: float = 0.45) -> Optional[Box]:
        """The smallest visible rectangle that holds the box (page borders are too big to count)."""
        best = None
        for r in self.rects:
            if inside(box, r) and area(r) < max_frac * self.page_area:
                if best is None or area(r) < area(best):
                    best = r
        return best

    def table_below(self, head: Box, max_row: float = 140.0) -> Box:
        """A table grows down from its heading cell: every rectangle inside the heading's columns that
        starts where the table so far ends."""
        x0, x1, top = head[0] - 1.5, head[2] + 1.5, head[1] - 0.5
        cand = [r for r in self.rects if r[0] >= x0 and r[2] <= x1 and r[1] >= top and r[3] - r[1] < max_row]
        bottom = head[3]
        grown = True
        while grown:
            grown = False
            for r in cand:
                if r[1] <= bottom + 1.0 and r[3] > bottom + 0.5:
                    bottom = r[3]
                    grown = True
        return union([head] + [r for r in cand if r[3] <= bottom + 0.5]) or head

    def run_after(self, idx: int, x_left: float, x_span: float = 30.0, gap: float = 16.0,
                  stop: Sequence[str] = (), start_y: Optional[float] = None) -> Tuple[List[Box], int]:
        """Text lines that follow op idx in drawing order and stay in the same left column with
        normal line spacing: the numbered lines under NOTES: or QUOTE REQUIREMENTS."""
        boxes: List[Box] = []
        prev = start_y if start_y is not None else self.ops[idx][1]["y"]
        color = None
        last = idx
        for j in range(idx + 1, len(self.ops)):
            kind, o = self.ops[j]
            if kind != "text":
                continue
            b = text_box(o, self.k)
            if not (x_left - 1.5 <= b[0] <= x_left + x_span) or not (prev - 0.5 <= o["y"] <= prev + gap):
                break
            if any(o["text"].startswith(s) for s in stop):
                break
            if color is None:
                color = o["color"]
            elif tuple(o["color"]) != tuple(color):
                break
            boxes.append(b)
            prev = o["y"]
            last = j
        return boxes, last

    def paragraph(self, idx: int) -> List[Box]:
        """A wrapped paragraph: op idx and the lines after it at the same x, size and color."""
        o0 = self.ops[idx][1]
        boxes = [text_box(o0, self.k)]
        prev = o0["y"]
        for j in range(idx + 1, len(self.ops)):
            kind, o = self.ops[j]
            if kind != "text":
                continue
            if (abs(o["x"] - o0["x"]) > 1.0 or abs(o["size"] - o0["size"]) > 0.3
                    or tuple(o["color"]) != tuple(o0["color"]) or not (prev < o["y"] <= prev + 1.8 * o0["size"])):
                break
            boxes.append(text_box(o, self.k))
            prev = o["y"]
        return boxes

    def only_footer_after(self, idx: int) -> bool:
        """True when nothing but the grey page footer follows op idx (a list that ran off the page)."""
        for j in range(idx + 1, len(self.ops)):
            kind, o = self.ops[j]
            if kind == "text" and tuple(o["color"]) == tuple(docgen.BLACK):
                return False
        return True


TB_LABELS = ("TITLE", "MATERIAL", "FINISH", "DWG NO.", "SCALE", "SHEET")


def label_title_block(ix: PageIndex) -> List[Box]:
    cells = []
    for name in TB_LABELS:
        for _, _, b in ix.named(name):
            r = ix.enclosing_rect(b)
            if r:
                cells.append(r)
    if len(cells) < 3:
        return []
    core = union(cells)
    # The tolerance block sits beside the cells, level with them. A drawing note can also start with
    # "UNLESS OTHERWISE SPECIFIED:" (cast or welded parts), and that one must not stretch the box
    # across the notes, so only text whose own rectangle touches the cells and shares their rows counts.
    for _, _, b in ix.starting("UNLESS OTHERWISE SPECIFIED"):
        r = ix.enclosing_rect(b) or b
        level = r[1] < core[3] - 2 and r[3] > core[1] + 2
        beside = r[2] >= core[0] - 3 and r[0] <= core[2] + 3
        if level and beside:
            cells.append(r)
    box = union(cells)
    outer = ix.enclosing_rect(box, 0.3)  # the heavy frame around all the cells, when it exists
    return [union([box, outer]) if outer else box]


def label_revision_block(ix: PageIndex) -> List[Box]:
    out = []
    for _, _, b in ix.named("REVISIONS"):
        head = ix.enclosing_rect(b, 0.1)
        if head:
            out.append(ix.table_below(head))
    return out


def label_notes(ix: PageIndex) -> List[Box]:
    out = []
    for i, o, b in ix.named("NOTES:"):
        lines, _ = ix.run_after(i, b[0])
        out.append(union([b] + lines))
    return out


def label_export_legends(ix: PageIndex) -> List[Box]:
    out = []
    for i, o, b in ix.texts:
        t = o["text"]
        if t.startswith("WARNING:") or t.startswith("CUI. CONTROLLED BY"):
            para = union(ix.paragraph(i))
            frame = ix.enclosing_rect(para or b, 0.3)
            out.append(union([para, frame]))
        elif t == "CUI" and o["bold"] and o["size"] >= 9:  # the CUI banner at the top and bottom
            out.append(pad(b, 3.0))
    return [b for b in out if b]


def label_proprietary(ix: PageIndex) -> List[Box]:
    return [union(ix.paragraph(i)) for i, _, _ in ix.starting("PROPRIETARY AND CONFIDENTIAL")]


HEADER_TITLES = ("REQUEST FOR QUOTATION", "PURCHASE ORDER")
HEADER_CELLS = ("RFQ NO.", "DATE", "RESPOND BY", "PO NUMBER", "REVISION")


def label_form_header(ix: PageIndex) -> List[Box]:
    out = []
    for _, o, b in ix.texts:
        if o["text"] not in HEADER_TITLES or o["size"] < 11 or not o["bold"]:
            continue
        ty = o["y"]
        cells = []
        for name in HEADER_CELLS:
            for _, lo, lb in ix.named(name):
                if ty - 12 <= lo["y"] <= ty + 45:
                    r = ix.enclosing_rect(lb, 0.05)
                    if r:
                        cells.append(r)
        bottom = max([c[3] for c in cells] + [ty + 16])
        texts = [tb for _, to, tb in ix.texts if ty - 20 <= to["y"] <= bottom and to["text"] != "CUI"]
        out.append(union([b] + cells + texts))
    return out


def label_line_tables(ix: PageIndex) -> List[Box]:
    out = []
    for _, _, b in ix.named("PART NUMBER"):
        head = ix.enclosing_rect(b, 0.2)
        if head and head[2] - head[0] > 200:  # the dark heading strip across the table
            out.append(ix.table_below(head))
    return out


REQ_HEADINGS = ("QUOTE REQUIREMENTS", "PURCHASE ORDER NOTES AND QUALITY CLAUSES")


def label_requirements(ix: PageIndex, prev: Optional[PageIndex] = None) -> List[Box]:
    out = []
    for i, o, b in ix.texts:
        if o["text"] in REQ_HEADINGS:
            lines, _ = ix.run_after(i, b[0], stop=("TERMS:",))
            out.append(union([b] + lines))
    # a list that ran off the previous page continues at the top of this one, without its heading
    if prev is not None:
        for i, o, b in prev.texts:
            if o["text"] not in REQ_HEADINGS:
                continue
            lines, last = prev.run_after(i, b[0], stop=("TERMS:",))
            if not lines or not prev.only_footer_after(last):
                continue
            first = next(((j, to, tb) for j, to, tb in ix.texts if to["text"] != "CUI"), None)
            if first and abs(first[2][0] - b[0]) < 20 and tuple(first[1]["color"]) == tuple(docgen.BLACK) \
                    and not first[1]["text"].startswith("TERMS:") and first[1]["y"] < 90:
                more, _ = ix.run_after(first[0], b[0], stop=("TERMS:",), start_y=first[1]["y"])
                out.append(union([first[2]] + more))
    return out


def page_labels(page: docgen.Page, kind: str, prev: Optional[docgen.Page] = None,
                k: Optional[float] = None) -> List[Label]:
    """Class boxes (page points, origin top left) for one page of a spec of this kind, as drawn at k
    pixels per point (None: exact font metrics, for vector PDFs)."""
    ix = PageIndex(page, k)
    out: List[Label] = []

    def add(name: str, boxes: Iterable[Optional[Box]], margin: float) -> None:
        for b in boxes:
            if b and area(b) > 4:
                out.append((CID[name], pad(b, margin)))

    if kind == "drawing":
        add("title_block", label_title_block(ix), 1.0)
        add("revision_block", label_revision_block(ix), 1.0)
        add("notes", label_notes(ix), 2.0)
    elif kind in ("rfq_form", "po"):
        pix = PageIndex(prev, k) if prev is not None else None
        add("form_header", label_form_header(ix), 2.0)
        add("line_table", label_line_tables(ix), 1.0)
        add("requirements", label_requirements(ix, pix), 2.0)
    if kind in ("drawing", "rfq_form", "po"):
        add("export_legend", label_export_legends(ix), 1.0)
        add("proprietary_notice", label_proprietary(ix), 2.0)
    # "document" pages (resumes, brochures, letters) are negatives: no boxes at all
    return out


def spec_page_labels(spec: Dict[str, Any], k: Optional[float] = None) -> Tuple[List[docgen.Page], List[List[Label]]]:
    pages = attachments.pages_for(spec)
    labels = [page_labels(p, spec["kind"], pages[i - 1] if i else None, k) for i, p in enumerate(pages)]
    return pages, labels


# --------------------------------------------------------------------------- #
# Geometry: page points -> pixels of the final picture
# --------------------------------------------------------------------------- #
Point = Callable[[float, float], Tuple[float, float]]


def scale_map(k: float) -> Point:
    return lambda x, y: (x * k, y * k)


def then(first: Point, second: Point) -> Point:
    return lambda x, y: second(*first(x, y))


def rotate_map(angle_deg: float, w: int, h: int) -> Point:
    """Where Image.rotate(angle, expand=False) moves a pixel: counterclockwise about the center."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    cx, cy = w / 2.0, h / 2.0

    def f(x: float, y: float) -> Tuple[float, float]:
        dx, dy = x - cx, y - cy
        return cx + dx * c + dy * s, cy - dx * s + dy * c
    return f


def perspective_map(src: List[Tuple[float, float]], dst: List[Tuple[float, float]]) -> Point:
    """Forward projective map taking src[i] to dst[i] (the inverse of what Image.transform uses)."""
    a, b, c, d, e, f, g, h = beta._perspective_coeffs(dst, src)

    def m(x: float, y: float) -> Tuple[float, float]:
        den = g * x + h * y + 1.0
        return (a * x + b * y + c) / den, (d * x + e * y + f) / den
    return m


def map_box(fn: Point, b: Box, size: Tuple[int, int]) -> Optional[Box]:
    pts = [fn(b[0], b[1]), fn(b[2], b[1]), fn(b[2], b[3]), fn(b[0], b[3])]
    x0 = max(0.0, min(p[0] for p in pts))
    y0 = max(0.0, min(p[1] for p in pts))
    x1 = min(float(size[0]), max(p[0] for p in pts))
    y1 = min(float(size[1]), max(p[1] for p in pts))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1, y1)


def map_labels(fn: Point, labels: List[Label], size: Tuple[int, int]) -> List[Label]:
    out = []
    for cls, b in labels:
        m = map_box(fn, b, size)
        if m:
            out.append((cls, m))
    return out


def beta_photo_corners(name: str, W: int = 2400, H: int = 1800) -> List[Tuple[float, float]]:
    """The page corners make_rfq_beta.phone_photo picks for a file name (same seeded draws)."""
    rnd = beta.rng_for(name)
    j = lambda s: rnd.uniform(-s, s)  # noqa: E731
    return [(170 + j(30), 150 + j(30)), (W - 140 + j(30), 210 + j(30)),
            (W - 230 + j(30), H - 120 + j(25)), (120 + j(30), H - 170 + j(25))]


# --------------------------------------------------------------------------- #
# Rendering through the beta effects
# --------------------------------------------------------------------------- #
MODES = [("digital", 0.25), ("scan", 0.25), ("copier", 0.13), ("fax", 0.14), ("photo", 0.12), ("screen", 0.11)]


def photo_any(page_img: Image.Image, rnd: random.Random) -> Tuple[Image.Image, List[Tuple[float, float]]]:
    """make_rfq_beta.phone_photo for any page orientation and framing: the same desk, keystone,
    uneven light, blur and noise, with the corners drawn at random. Returns the picture and the
    corners, so the boxes can follow."""
    page = page_img.convert("RGB")
    r, g, b = page.split()
    warm = rnd.uniform(0.84, 0.97)
    page = Image.merge("RGB", (r.point(lambda v: int(v * 0.99)), g.point(lambda v: int(v * (warm + 1) / 2)),
                               b.point(lambda v: int(v * warm))))
    pw, ph = page.size
    portrait = ph > pw
    long_side = rnd.choice([1600, 2000, 2400])
    W, H = (int(long_side * 0.75), long_side) if portrait else (long_side, int(long_side * 0.75))
    tone = rnd.randint(60, 130)
    desk = Image.new("RGB", (W, H), (tone, int(tone * 0.86), int(tone * 0.72)))
    grad = Image.linear_gradient("L").resize((W, H)).rotate(rnd.uniform(0, 360), expand=False, fillcolor=128)
    desk = Image.composite(Image.new("RGB", (W, H), (tone + 28, int((tone + 28) * 0.86), int((tone + 28) * 0.7))),
                           desk, grad)
    mx, my = W * rnd.uniform(0.03, 0.09), H * rnd.uniform(0.03, 0.09)
    jx, jy = W * 0.025, H * 0.025
    j = lambda s: rnd.uniform(-s, s)  # noqa: E731
    dst = [(mx + j(jx), my + j(jy)), (W - mx + j(jx), my + j(jy)),
           (W - mx + j(jx), H - my + j(jy)), (mx + j(jx), H - my + j(jy))]
    src = [(0, 0), (pw, 0), (pw, ph), (0, ph)]
    coeffs = beta._perspective_coeffs(src, dst)
    warped = page.transform((W, H), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)
    mask = Image.new("L", (pw, ph), 255).transform((W, H), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)
    photo = Image.composite(warped, desk, mask)
    light = Image.radial_gradient("L").resize((W, H))
    strength = rnd.uniform(0.25, 0.5)
    light = light.point(lambda v: int(255 - v * strength))
    shade = Image.linear_gradient("L").rotate(rnd.uniform(-80, 80), expand=False, fillcolor=128).resize((W, H))
    shade = shade.point(lambda v: int(255 - v * 0.22))
    lighting = Image.merge("RGB", (ImageChops.multiply(light, shade),) * 3)
    photo = ImageChops.multiply(photo, lighting)
    photo = photo.point(lambda v: min(255, int(v * 1.18)))
    photo = photo.filter(ImageFilter.GaussianBlur(rnd.uniform(0.5, 1.3)))
    noise = Image.effect_noise((W, H), rnd.uniform(3, 8)).convert("RGB")
    return ImageChops.add(photo, noise, scale=1.0, offset=-128), dst


def render(page: docgen.Page, mode: str, rnd: random.Random, name: str,
           title: str) -> Tuple[Image.Image, Point, float]:
    """One page through one of the beta effects. Returns the picture, the point map, and the pixels
    per point docgen.to_image drew it at (dpi / 72 times the supersampling), which text_box needs."""
    if mode == "digital":
        dpi = rnd.choice([72, 96, 110, 120, 150, 200])
        img = docgen.to_image(page, dpi, supersample=2)
        if rnd.random() < 0.5:
            img = img.convert("L")
        return img, scale_map(dpi / 72.0), dpi / 72.0 * 2
    if mode in ("scan", "copier"):
        dpi = rnd.choice([200, 240, 300]) if mode == "scan" else rnd.choice([150, 200])
        skew = rnd.uniform(-1.6, 1.6) if mode == "scan" else rnd.uniform(-2.2, 2.2)
        ss = 1 if dpi >= 240 else 2
        base = docgen.to_image(page, dpi, supersample=ss)
        img = beta.office_scan(base, {"mode": mode, "skew": skew}, name, dpi)
        img = Image.open(io.BytesIO(beta.jpeg_bytes(img, rnd.randint(50, 80), dpi)))
        return img, then(scale_map(dpi / 72.0), rotate_map(skew, *img.size)), dpi / 72.0 * ss
    if mode == "fax":
        dpi = rnd.choice([150, 200, 200])
        skew = rnd.uniform(-1.2, 1.2)
        img = beta.fax(docgen.to_image(page, dpi, supersample=1), {"mode": "fax", "skew": skew}, name).convert("L")
        return img, then(scale_map(dpi / 72.0), rotate_map(skew, *img.size)), dpi / 72.0
    if mode == "photo":
        dpi = rnd.choice([150, 180, 220])
        base = docgen.to_image(page, dpi, supersample=1)
        if page.width > page.height and rnd.random() < 0.35:
            img = beta.phone_photo(base, name)  # exactly the beta photo pipeline
            dst = beta_photo_corners(name)
        else:
            img, dst = photo_any(base, rnd)
        pw, ph = base.size
        return img, then(scale_map(dpi / 72.0), perspective_map([(0, 0), (pw, 0), (pw, ph), (0, ph)], dst)), dpi / 72.0
    if mode == "screen":
        dpi = rnd.choice([96, 110, 120, 144])
        img = beta.screenshot(docgen.to_image(page, dpi, supersample=3), title)
        return img, then(scale_map(dpi / 72.0), lambda x, y: (x + 60, y + 108)), dpi / 72.0 * 3
    raise ValueError(mode)


def shrink(img: Image.Image, labels: List[Label], max_side: int = MAX_SIDE) -> Tuple[Image.Image, List[Label]]:
    w, h = img.size
    k = min(1.0, max_side / float(max(w, h)))
    if k >= 1.0:
        return img, labels
    img = img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
    return img, [(c, (b[0] * k, b[1] * k, b[2] * k, b[3] * k)) for c, b in labels]


def yolo_lines(labels: List[Label], size: Tuple[int, int]) -> str:
    w, h = size
    rows = []
    for cls, (x0, y0, x1, y1) in labels:
        rows.append(f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return "\n".join(rows) + ("\n" if rows else "")


def draw_labels(img: Image.Image, labels: List[Label], width: int = 3) -> Image.Image:
    colors = [(220, 40, 40), (40, 150, 40), (40, 90, 220), (230, 130, 0), (150, 60, 200),
              (0, 170, 170), (200, 0, 120), (120, 120, 0)]
    out = img.convert("RGB")
    d = ImageDraw.Draw(out)
    font = docgen._font(True, max(12, out.size[0] // 70))
    for cls, (x0, y0, x1, y1) in labels:
        c = colors[cls % len(colors)]
        d.rectangle([x0, y0, x1, y1], outline=c, width=width)
        d.text((x0 + 3, max(0, y0 - font.size - 2)), CLASSES[cls], fill=c, font=font)
    return out


# --------------------------------------------------------------------------- #
# Random specs from the sample vocabulary
# --------------------------------------------------------------------------- #
class Vocab:
    def __init__(self, emails: List[Dict[str, Any]], held_out_pns: set):
        atts = [a for e in emails for a in (e.get("attachments") or []) if isinstance(a, dict)]
        self.drawings = [a for a in atts if a.get("kind") == "drawing"]
        self.forms = [a for a in atts if a.get("kind") == "rfq_form"]
        self.pos = [a for a in atts if a.get("kind") == "po"]
        self.docs = [a for a in atts if a.get("kind") == "document"]
        companies = {a.get("company") for a in atts if a.get("company")}
        self.companies = sorted(companies)
        self.addresses = sorted({a["company_address"] for a in self.forms + self.pos if a.get("company_address")})
        self.titles = sorted({a["title"] for a in self.drawings}
                             | {ln.get("description") for f in self.forms + self.pos for ln in f.get("lines") or []
                                if ln.get("description")})
        self.materials = sorted({a["material"] for a in self.drawings if a.get("material")}
                                | {ln["material"] for f in self.forms for ln in f.get("lines") or [] if ln.get("material")})
        self.finishes = sorted({a["finish"] for a in self.drawings if a.get("finish")}
                               | {ln["finish"] for f in self.forms for ln in f.get("lines") or [] if ln.get("finish")})
        self.notes = sorted({n for a in self.drawings for n in a.get("notes") or []})
        self.callouts = sorted({c for a in self.drawings for c in a.get("callouts") or []})
        self.tolerances = sorted({a["tolerances"] for a in self.drawings if a.get("tolerances")})
        self.people = sorted({a["drawn_by"] for a in self.drawings if a.get("drawn_by")})
        self.rev_texts = sorted({r["description"] for a in self.drawings for r in a.get("revisions") or []
                                 if isinstance(r, dict) and r.get("description")})
        self.requirements = sorted({r for f in self.forms for r in f.get("requirements") or []})
        self.po_notes = sorted({n for p in self.pos for n in p.get("notes") or []})
        self.terms = sorted({f["terms"] for f in self.forms + self.pos if f.get("terms")})
        self.ship_via = sorted({p["ship_via"] for p in self.pos if p.get("ship_via")})
        self.buyers = sorted({f["buyer"] for f in self.forms + self.pos if f.get("buyer")})
        self.prefixes = sorted({a["part_number"].rsplit("-", 1)[0] for a in self.drawings
                                if "-" in a.get("part_number", "")})
        self.line_notes = sorted({ln["notes"] for f in self.forms for ln in f.get("lines") or [] if ln.get("notes")})
        self.held_out = held_out_pns
        self.sections = [s for d in self.docs for s in d.get("sections") or [] if isinstance(s, dict)]
        self.doc_titles = [(d.get("doc_type"), d.get("title"), d.get("subtitle")) for d in self.docs]


SUFFIXES = ["Aerospace", "Instruments", "Robotics", "Fluid Controls", "Defense Systems", "Medical", "Optics",
            "Motion", "Energy", "Automation", "Industries", "Precision Products", "Kiosks", "Marine"]
WORDS = ["BRACKET", "HOUSING", "COVER", "PLATE", "SHAFT", "SPACER", "MANIFOLD", "FITTING", "MOUNT", "ADAPTER",
         "FLANGE", "BLOCK", "RETAINER", "CLAMP", "BUSHING", "NOZZLE", "ENCLOSURE", "BASE", "ARM", "LEVER"]
QUALIFIERS = ["SENSOR", "MOTOR", "ACTUATOR", "PUMP", "VALVE", "BEARING", "OPTICAL", "CAMERA", "LASER", "WRIST",
              "PILOT", "OUTPUT", "INPUT", "UPPER", "LOWER", "LEFT HAND", "RIGHT HAND", "FRONT", "REAR", "COOLING"]


def _pick(rnd: random.Random, seq: Sequence[Any], k: int) -> List[Any]:
    seq = list(seq)
    return rnd.sample(seq, min(k, len(seq)))


def _date(rnd: random.Random, year0: int = 2024, year1: int = 2026) -> str:
    return f"{rnd.randint(year0, year1)}-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}"


class SpecMaker:
    """Random but realistic specs. Every part number is new, and a spec is never one of the beta specs."""

    def __init__(self, vocab: Vocab, rnd: random.Random):
        self.v = vocab
        self.rnd = rnd

    def company(self) -> str:
        r = self.rnd
        roll = r.random()
        if roll < 0.55:
            return r.choice(self.v.companies)
        first = r.choice(self.v.companies).split()[0]
        if roll < 0.9:
            return f"{first} {r.choice(SUFFIXES)}"
        return f"{first} {r.choice(SUFFIXES)} and {r.choice(SUFFIXES)} Group"  # long enough to shrink

    def title(self) -> str:
        r = self.rnd
        roll = r.random()
        if roll < 0.5:
            return r.choice(self.v.titles).upper()
        if roll < 0.85:
            return f"{r.choice(WORDS)}, {r.choice(QUALIFIERS)}"
        return f"{r.choice(WORDS)}, {r.choice(QUALIFIERS)} {r.choice(WORDS)}, {r.choice(QUALIFIERS)} ASSEMBLY"

    def part_number(self, company: str) -> str:
        r = self.rnd
        while True:
            if r.random() < 0.6:
                prefix = r.choice(self.v.prefixes)
            else:
                prefix = "".join(w[0] for w in company.split() if w[:1].isalpha())[:3].upper() or "PN"
            pn = f"{prefix}-{r.randint(10, 99999):0{r.choice([3, 4, 5])}d}"
            if r.random() < 0.1:
                pn += f"-{r.randint(1, 99):02d}"
            if pn not in self.v.held_out:
                return pn

    def size(self, shape: str, units: str) -> List[float]:
        r = self.rnd
        k = 25.4 if units == "mm" else 1.0

        def q(v: float) -> float:
            return round(v * k, 1 if units == "mm" else 3)
        if shape in drawings.ROUND3:
            od = r.uniform(0.3, 6)
            return [q(r.uniform(0.1, 4)), q(od), q(od * r.uniform(0.3, 0.8))]
        if shape in drawings.ROUND2:
            return [q(r.uniform(0.3, 20)), q(r.uniform(0.1, 5))]
        L = r.uniform(0.5, 30)
        return [q(L), q(L * r.uniform(0.2, 1.0)), q(L * r.uniform(0.05, 0.7))]

    def drawing(self) -> Dict[str, Any]:
        r = self.rnd
        company = self.company()
        units = "mm" if r.random() < 0.3 else "in"
        shape = r.choice(sorted(drawings.PRISMATIC | drawings.COMPLEX | drawings.ROUND | drawings.OTHER))
        n_rev = r.randint(1, 4)
        revs = [{"rev": chr(ord("A") + i), "description": r.choice(self.v.rev_texts),
                 "date": _date(r)} for i in range(n_rev)]
        if r.random() < 0.15:
            revs = [dict(x, rev=str(i + 1)) for i, x in enumerate(revs)]
        legend = r.choices([None, "proprietary", "itar", "ear", "cui"], [0.3, 0.25, 0.17, 0.12, 0.16])[0]
        spec: Dict[str, Any] = {
            "name": "x.pdf", "kind": "drawing", "company": company, "title": self.title(),
            "part_number": self.part_number(company), "rev": revs[-1]["rev"],
            "material": r.choice(self.v.materials),
            "finish": r.choice(self.v.finishes) if r.random() < 0.85 else None,
            "shape": shape, "size": self.size(shape, units), "units": units,
            "scale": r.choice(["1:1", "2:1", "1:2", "1:4", "4:1", "1:3", "1:8", "10:1", "1:5"]),
            "notes": _pick(r, self.v.notes, r.randint(3, 8)),
            "callouts": _pick(r, self.v.callouts, r.choice([0, 0, 1, 2, 3, 4])),
            "drawn_by": r.choice(self.v.people) if r.random() < 0.9 else None,
            "date": revs[-1]["date"], "revisions": revs,
        }
        if r.random() < 0.75:
            spec["tolerances"] = r.choice(self.v.tolerances)
        if r.random() < 0.1:
            spec["sheet"] = r.choice(["1 OF 2", "1 OF 3", "2 OF 2"])
        if legend:
            spec["legend"] = legend
        if legend == "ear":
            spec["eccn"] = r.choice(["9E610", "9A610", "3A001", "6A003", "0A606", "5A002", "9E610.a"])
        return spec

    def _form_common(self, kind: str) -> Dict[str, Any]:
        r = self.rnd
        company = self.company()
        buyer = r.choice(self.v.buyers)
        domain = "".join(ch for ch in company.split()[0].lower() if ch.isalpha()) + r.choice([".com", "corp.com", "inc.com"])
        spec = {"name": "x.pdf", "kind": kind, "company": company,
                "company_address": r.choice(self.v.addresses) if r.random() < 0.92 else "",
                "date": _date(r, 2026, 2026), "buyer": buyer,
                "buyer_email": f"{buyer.split()[0][0].lower()}{buyer.split()[-1].lower()}@{domain}"}
        return spec

    def rfq_form(self) -> Dict[str, Any]:
        r = self.rnd
        spec = self._form_common("rfq_form")
        yy = r.choice(["26", "2026"])
        spec["rfq_number"] = r.choice([f"RFQ-{yy}-{r.randint(1, 9999):04d}",
                                       f"{self.part_number(spec['company']).split('-')[0]}-RFQ-{r.randint(100, 9999)}",
                                       f"RFQ{r.randint(10000, 99999)}"])
        spec["respond_by"] = _date(r, 2026, 2026)
        lines = []
        for _ in range(r.randint(1, 5)):
            line = {"part_number": self.part_number(spec["company"]), "rev": r.choice("ABCDEF-"),
                    "description": self.title(), "material": r.choice(self.v.materials),
                    "finish": r.choice(self.v.finishes) if r.random() < 0.8 else "",
                    "quantities": sorted(r.sample([1, 2, 5, 10, 20, 25, 40, 50, 75, 100, 150, 200, 250, 300, 500,
                                                   750, 1000, 2500, 5000], r.randint(1, 4)))}
            if r.random() < 0.15 and self.v.line_notes:
                line["notes"] = r.choice(self.v.line_notes)
            lines.append(line)
        spec["lines"] = lines
        spec["requirements"] = _pick(r, self.v.requirements, r.randint(0, 6))
        if r.random() < 0.85:
            spec["terms"] = r.choice(self.v.terms)
        legend = r.choices([None, "proprietary", "itar", "ear", "cui"], [0.45, 0.15, 0.17, 0.11, 0.12])[0]
        if legend:
            spec["legend"] = legend
        if legend == "ear":
            spec["eccn"] = r.choice(["9E610", "9A610", "3A001", "6A003"])
        return spec

    def po(self) -> Dict[str, Any]:
        r = self.rnd
        spec = self._form_common("po")
        spec["po_number"] = r.choice([str(r.randint(1000, 99999)), f"45000{r.randint(10000, 99999)}",
                                      f"PO-26-{r.randint(1, 9999):04d}", f"{r.randint(1, 99)}-{r.randint(100, 9999)}"])
        spec["terms"] = r.choice(self.v.terms + ["NET 30", "NET 45", "NET 60"])
        spec["ship_via"] = r.choice(self.v.ship_via)
        if r.random() < 0.85:
            spec["quote_ref"] = f"Q-26-{r.randint(1000, 1999)}"
        if r.random() < 0.15:
            spec["po_rev"] = str(r.randint(1, 3))
        spec["lines"] = [{"part_number": self.part_number(spec["company"]), "rev": r.choice("ABCDE-"),
                          "description": self.title(), "qty": r.choice([5, 10, 25, 50, 100, 250, 500, 1200]),
                          "unit_price": round(r.uniform(2, 900), 2), "due": _date(r, 2026, 2027)}
                         for _ in range(r.randint(1, 5))]
        spec["notes"] = _pick(r, self.v.po_notes + self.v.requirements, r.randint(1, 6))
        legend = r.choices([None, "proprietary", "itar", "cui"], [0.6, 0.15, 0.15, 0.1])[0]
        if legend:
            spec["legend"] = legend
        return spec

    def document(self) -> Dict[str, Any]:
        r = self.rnd
        doc_type, title, subtitle = r.choice(self.v.doc_titles)
        doc_type = r.choice([doc_type, doc_type, "letter", "brochure", "resume", "invoice", "report", "spec",
                             "newsletter", "cert", "packing_slip"])
        sections = _pick(r, self.v.sections, r.randint(2, 6))
        spec = {"name": "x.pdf", "kind": "document", "doc_type": doc_type, "title": title, "subtitle": subtitle,
                "sections": sections}
        if r.random() < 0.8:
            spec["company"] = self.company()
        return spec


KIND_MIX = [("drawing", 0.55), ("rfq_form", 0.20), ("po", 0.12), ("document", 0.13)]


def beta_specs() -> Dict[str, Dict[str, Any]]:
    """The 30 beta files: spec, render mode and dpi (tests/rfq_beta_truth.json, the evaluation record)."""
    truth = json.loads((ROOT / "tests" / "rfq_beta_truth.json").read_text(encoding="utf-8"))
    return truth["files"]


def held_out(files: Dict[str, Dict[str, Any]]) -> Tuple[set, set]:
    pns, pairs = set(), set()
    for entry in files.values():
        s = entry["spec"]
        for pn in [s.get("part_number")] + [ln.get("part_number") for ln in s.get("lines") or []]:
            if pn:
                pns.add(pn)
        pairs.add((s.get("company"), s.get("title"), s.get("rfq_number"), s.get("kind")))
    return pns, pairs


# --------------------------------------------------------------------------- #
# One synthetic sample (runs in a worker process)
# --------------------------------------------------------------------------- #
_VOCAB: Optional[Vocab] = None
_PAIRS: set = set()


def _init_worker() -> None:
    global _VOCAB, _PAIRS
    emails = json.loads((ROOT / "data" / "sample_emails.json").read_text(encoding="utf-8"))["emails"]
    pns, _PAIRS = held_out(beta_specs())
    _VOCAB = Vocab(emails, pns)


def make_sample(args: Tuple[str, int, int, str]) -> Dict[str, Any]:
    split, index, seed, out_dir = args
    if _VOCAB is None:
        _init_worker()
    rnd = random.Random(seed)
    maker = SpecMaker(_VOCAB, rnd)
    kind = rnd.choices([k for k, _ in KIND_MIX], [w for _, w in KIND_MIX])[0]
    while True:
        spec = getattr(maker, kind)()
        if (spec.get("company"), spec.get("title"), spec.get("rfq_number"), spec["kind"]) not in _PAIRS:
            break
    pages = attachments.pages_for(spec)
    pi = 0 if len(pages) == 1 or rnd.random() < 0.6 else rnd.randrange(len(pages))
    mode = rnd.choices([m for m, _ in MODES], [w for _, w in MODES])[0]
    name = f"{split}_{index:05d}"
    title = docgen.clean(spec.get("part_number") or spec.get("rfq_number") or spec.get("po_number") or name)
    img, fn, k = render(pages[pi], mode, rnd, name, title)
    labels = page_labels(pages[pi], spec["kind"], pages[pi - 1] if pi else None, k)
    boxes = map_labels(fn, labels, img.size)
    img, boxes = shrink(img, boxes)
    img_path = Path(out_dir) / "images" / split / f"{name}.jpg"
    img.convert("RGB" if img.mode not in ("L", "RGB") else img.mode).save(img_path, "JPEG", quality=90)
    (Path(out_dir) / "labels" / split / f"{name}.txt").write_text(yolo_lines(boxes, img.size))
    return {"name": name, "kind": spec["kind"], "legend": spec.get("legend"), "mode": mode, "page": pi,
            "pages": len(pages), "boxes": [CLASSES[c] for c, _ in boxes], "size": list(img.size)}


# --------------------------------------------------------------------------- #
# The honest test set: the real beta files
# --------------------------------------------------------------------------- #
BETA_DPI = 150  # how the runtime rasterizes PDF pages (layout.detect_pdf_page default)
# tools/make_rfq_beta.py draws scans, faxes and the photo with docgen.to_image's default supersampling
# (2) and the screenshot at 3; text PDFs are vector files whose glyphs sit at exact Helvetica widths.
BETA_SUPERSAMPLE = {"scan": 2, "copier": 2, "fax": 2, "photo": 2, "screen": 3}


def beta_raster_scale(entry: Dict[str, Any]) -> Optional[float]:
    """Pixels per point the beta file's text was drawn at, or None for a vector PDF."""
    ss = BETA_SUPERSAMPLE.get(entry["render"])
    return entry["dpi"] / 72.0 * ss if ss and entry.get("dpi") else None


def pdf_pages(path: Path, dpi: int) -> List[Image.Image]:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(path), os.path.join(tmp, "p")],
                       check=True, capture_output=True, timeout=120)
        return [Image.open(os.path.join(tmp, f)).convert("RGB") for f in sorted(os.listdir(tmp))]


def beta_samples() -> List[Tuple[str, Image.Image, List[Label], Dict[str, Any]]]:
    """(name, picture, boxes, info) for every page of the 23 beta files that have pages."""
    out = []
    for rel, entry in sorted(beta_specs().items()):
        spec = entry["spec"]
        if spec["kind"] == "model":
            continue
        path = ROOT / "data" / rel
        pages, labels = spec_page_labels(spec, beta_raster_scale(entry))
        mode = entry["render"]
        email = rel.split("/")[-2]
        cfg = beta.RENDER.get((email, spec["name"]), {})
        if path.suffix.lower() == ".pdf":
            imgs = pdf_pages(path, BETA_DPI)
            for i, (img, page) in enumerate(zip(imgs, pages)):
                fn = scale_map(BETA_DPI / 72.0)
                if mode in ("scan", "copier", "fax"):
                    fn = then(fn, rotate_map(cfg.get("skew", 0.0), *img.size))
                boxes = map_labels(fn, labels[i], img.size)
                out.append((f"{email}_{Path(rel).stem}_p{i + 1}", img, boxes, {"file": rel, "mode": mode, "page": i + 1}))
        elif mode == "photo":
            img = Image.open(path).convert("RGB")
            dpi = entry.get("dpi") or 220
            pw, ph = round(pages[0].width * dpi / 72.0), round(pages[0].height * dpi / 72.0)
            dst = beta_photo_corners(path.name, *img.size)
            fn = then(scale_map(dpi / 72.0), perspective_map([(0, 0), (pw, 0), (pw, ph), (0, ph)], dst))
            out.append((f"{email}_{path.stem}", img, map_labels(fn, labels[0], img.size),
                        {"file": rel, "mode": mode, "page": 1}))
        elif mode == "screen":
            img = Image.open(path).convert("RGB")
            dpi = entry.get("dpi") or 120
            fn = then(scale_map(dpi / 72.0), lambda x, y: (x + 60, y + 108))
            out.append((f"{email}_{path.stem}", img, map_labels(fn, labels[0], img.size),
                        {"file": rel, "mode": mode, "page": 1}))
    return out


def write_beta(out: Path) -> List[Dict[str, Any]]:
    (out / "images" / "beta").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "beta").mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, img, boxes, info in beta_samples():
        img2, boxes2 = shrink(img, boxes)
        img2.save(out / "images" / "beta" / f"{name}.jpg", "JPEG", quality=92)
        (out / "labels" / "beta" / f"{name}.txt").write_text(yolo_lines(boxes2, img2.size))
        manifest.append(dict(info, name=name, boxes=[CLASSES[c] for c, _ in boxes2], size=list(img2.size)))
    return manifest


# --------------------------------------------------------------------------- #
def preview(out: Path, count: int, seed: int) -> None:
    """Pictures with the derived boxes drawn on them, to check the labels by eye."""
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    for split in ("images/prev", "labels/prev"):
        (tmp / split).mkdir(parents=True, exist_ok=True)
    for i in range(count):
        info = make_sample(("prev", i, seed * 100003 + i, str(tmp)))
        img = Image.open(tmp / "images" / "prev" / f"{info['name']}.jpg")
        labels = read_yolo(tmp / "labels" / "prev" / f"{info['name']}.txt", img.size)
        draw_labels(img, labels).save(out / f"{info['name']}_{info['kind']}_{info['mode']}_{info.get('legend')}.jpg",
                                      quality=85)
    for name, img, boxes, info in beta_samples():
        draw_labels(img, boxes, 4).save(out / f"beta_{name}.jpg", quality=85)
    shutil.rmtree(tmp, ignore_errors=True)


def read_yolo(path: Path, size: Tuple[int, int]) -> List[Label]:
    w, h = size
    out = []
    for line in path.read_text().split("\n"):
        if line.strip():
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            out.append((int(c), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="dataset folder (outside the repository)")
    ap.add_argument("--train", type=int, default=900)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2609)
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--beta-only", action="store_true", help="rebuild only the beta test set")
    ap.add_argument("--preview", type=Path, help="write pictures with boxes drawn on them instead")
    ap.add_argument("--count", type=int, default=12)
    args = ap.parse_args()
    if args.preview:
        preview(args.preview, args.count, args.seed)
        return 0
    if not args.out:
        ap.error("--out is required")
    out = args.out.resolve()
    if ROOT in out.parents or out == ROOT:
        ap.error("keep datasets out of the repository")
    t0 = time.time()
    manifest: Dict[str, Any] = {"classes": CLASSES, "seed": args.seed}
    if not args.beta_only:
        for split in ("train", "val"):
            shutil.rmtree(out / "images" / split, ignore_errors=True)
            shutil.rmtree(out / "labels" / split, ignore_errors=True)
            (out / "images" / split).mkdir(parents=True)
            (out / "labels" / split).mkdir(parents=True)
        jobs = [("train", i, args.seed * 1000003 + i, str(out)) for i in range(args.train)]
        jobs += [("val", i, args.seed * 1000003 + 500000 + i, str(out)) for i in range(args.val)]
        with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
            results = list(pool.map(make_sample, jobs, chunksize=4))
        manifest["synthetic"] = results
        print(f"{len(results)} synthetic images in {time.time() - t0:.0f} s")
    shutil.rmtree(out / "images" / "beta", ignore_errors=True)
    shutil.rmtree(out / "labels" / "beta", ignore_errors=True)
    manifest["beta"] = write_beta(out)
    old = out / "manifest.json"
    if args.beta_only and old.exists():
        prev = json.loads(old.read_text())
        prev["beta"] = manifest["beta"]
        manifest = prev
    old.write_text(json.dumps(manifest, indent=1))
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/beta\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    print(f"beta: {len(manifest['beta'])} pages; done in {time.time() - t0:.0f} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
