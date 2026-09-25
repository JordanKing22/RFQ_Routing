"""
Build the RFQ details beta inbox: 22 emails from data/sample_emails.json and exactly 30 real
attachment files, 13 of them uncopyable (image-only scans, a fax, a phone photo, a screenshot)
so the text has to come from OCR.

    pip install pillow            (needed only to run this tool)
    python tools/make_rfq_beta.py

Writes:
    data/rfq_beta/emails.json          the beta inbox (attachments are real files)
    data/rfq_beta/files/<email>/...    the 30 files
    tests/rfq_beta_truth.json          what is really printed on each file (for OCR and
                                       extraction tests), taken from the specs

Everything is seeded, so a rerun produces the same files. Rerun after changing the drawing
renderer, then rebuild the OCR cache (python ocr.py --build-cache).
"""

from __future__ import annotations

import io
import json
import math
import random
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import attachments  # noqa: E402
import docgen  # noqa: E402
import drawings  # noqa: E402

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFilter
except ImportError:  # pragma: no cover - a dev tool, say what is missing
    raise SystemExit("This tool needs Pillow: pip install pillow")

OUT_DIR = ROOT / "data" / "rfq_beta"
FILES_DIR = OUT_DIR / "files"
TRUTH_FILE = ROOT / "tests" / "rfq_beta_truth.json"

# Emails in the beta inbox, in inbox order. 16 RFQs with files (30 files), 3 RFQs with no files,
# and 3 emails that are not RFQs, so the extractor has something to skip.
EMAIL_IDS = ["E01", "E03", "E04", "E05", "E06", "E07", "E09", "E10", "E13", "E16", "E17", "E19",
             "E20", "E22", "E26", "E32", "E52", "E56", "E61", "E62", "E71", "E72"]

# How each uncopyable file was "sent". Everything not listed is the normal digital file.
#   scan     office scanner, 300 dpi grayscale JPEG, slight skew
#   copier   older copier, 200 dpi grayscale, darker, edge shadow, more skew
#   fax      fax-quality black and white, 200 dpi, 1 bit
#   photo    phone photo of the printed page (JPG), perspective and uneven light
#   screen   screenshot of a PDF viewer window (PNG), about 120 dpi
RENDER = {
    ("E01", "QA-41127_RevB.pdf"): {"mode": "scan", "skew": 0.6},
    ("E07", "FR-2290_heatsink.pdf"): {"mode": "fax", "skew": 0.3},
    ("E09", "HPV-2045_manifold_RevD.pdf"): {"mode": "copier", "skew": -1.4},
    ("E16", "AW-310_frame_assy.pdf"): {"mode": "scan", "skew": -0.8},
    ("E20", "FR-3102.pdf"): {"mode": "scan", "skew": 0.4},
    ("E22", "OPM-22817_RevB.pdf"): {"mode": "photo", "rename": "OPM-22817_RevB_photo.jpg"},
    ("E32", "RFQ-26-0931.pdf"): {"mode": "scan", "skew": 0.9},
    ("E52", "AGI-3052_RevA.pdf"): {"mode": "fax", "skew": -0.5},
    ("E56", "RFQ-26-0317.pdf"): {"mode": "scan", "skew": -0.5},
    ("E61", "CI-10442_RevC.pdf"): {"mode": "scan", "skew": 1.1},
    ("E62", "RFQ-26-0318.pdf"): {"mode": "copier", "skew": 1.2},
    ("E71", "LO-1186_RevA.pdf"): {"mode": "screen", "rename": "LO-1186_RevA_screenshot.png"},
    ("E72", "WS-RFQ-26-0388.pdf"): {"mode": "scan", "skew": -1.0},
}

# Where a sender would mention how the file was sent. Plain replacements in the beta copy only.
BODY_EDITS = {
    "E07": [("Attached is our heat sink plate FR-2290:",
             "Attached is a scan of our heat sink plate FR-2290 (sorry, it came off the old copier):")],
    "E16": [("Drawings attached.", "Scanned drawing attached.")],
    "E22": [("Drawing OPM-22817 Rev B attached.",
             "I'm out at the plant with no scanner, so I took a photo of the print, OPM-22817 Rev B.")],
    "E71": [("Can you quote the attached fold mirror mount, LO-1186 Rev A?",
             "Can you quote the fold mirror mount LO-1186 Rev A? The PDF is still stuck in our PDM release "
             "queue, so I attached a screenshot of the drawing.")],
}

SCANNER_NAMES = {"scan": "ScanDesk 4.2", "copier": "OfficeCopy 2500 Scan", "fax": "FaxLine Scan to PDF"}


def rng_for(name: str) -> random.Random:
    return random.Random(zlib.crc32(name.encode("utf-8")))


# --------------------------------------------------------------------------- #
# Scan effects (Pillow only)
# --------------------------------------------------------------------------- #
def add_noise(img: "Image.Image", sigma: float, seed: int) -> "Image.Image":
    random.seed(seed)  # effect_noise has no seed argument; keep runs repeatable
    noise = Image.effect_noise(img.size, sigma)
    return ImageChops.add(img, noise, scale=1.0, offset=-128)


def levels(img: "Image.Image", ink: int, paper: int) -> "Image.Image":
    return img.point(lambda v: int(ink + (paper - ink) * v / 255))


def speckle(img: "Image.Image", rnd: random.Random, count: int, shade: int) -> None:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(count):
        x, y, r = rnd.randrange(w), rnd.randrange(h), rnd.choice([1, 1, 1, 2, 2, 3])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=shade)


def edge_shadow(img: "Image.Image", width: int, darkest: int) -> "Image.Image":
    w, h = img.size
    grad = Image.new("L", (w, 1), 255)
    for x in range(min(width, w)):
        grad.putpixel((x, 0), int(darkest + (255 - darkest) * (x / width) ** 0.5))
    return ImageChops.multiply(img, grad.resize((w, h)))


def office_scan(page_img: "Image.Image", cfg: Dict[str, Any], name: str, dpi: int) -> "Image.Image":
    rnd = rng_for(name)
    img = page_img.convert("L")
    img = img.filter(ImageFilter.GaussianBlur(0.45 if dpi >= 300 else 0.35))
    img = levels(img, ink=38 if cfg["mode"] == "scan" else 22, paper=241 if cfg["mode"] == "scan" else 226)
    if cfg["mode"] == "copier":
        img = edge_shadow(img, int(img.size[0] * 0.05), 120)
    img = add_noise(img, 7 if cfg["mode"] == "scan" else 11, rnd.randrange(1 << 30))
    speckle(img, rnd, 60 if cfg["mode"] == "scan" else 180, 70)
    img = img.rotate(cfg.get("skew", 0.0), resample=Image.BICUBIC, expand=False,
                     fillcolor=241 if cfg["mode"] == "scan" else 226)
    return img


def fax(page_img: "Image.Image", cfg: Dict[str, Any], name: str) -> "Image.Image":
    rnd = rng_for(name)
    img = page_img.convert("L")
    img = img.rotate(cfg.get("skew", 0.0), resample=Image.BICUBIC, expand=False, fillcolor=255)
    img = add_noise(img, 16, rnd.randrange(1 << 30))
    img = img.filter(ImageFilter.GaussianBlur(0.3))
    img = img.point(lambda v: 255 if v > 150 else 0).convert("1", dither=Image.Dither.NONE)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(260):  # salt noise from the phone line
        x, y = rnd.randrange(w), rnd.randrange(h)
        draw.point((x, y), fill=0)
    return img


def _perspective_coeffs(src: List[Tuple[float, float]], dst: List[Tuple[float, float]]) -> List[float]:
    """Coefficients for Image.transform(PERSPECTIVE): map each output point dst[i] to input src[i]."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    n = 8  # Gaussian elimination with partial pivoting (no numpy)
    m = [row + [b] for row, b in zip(rows, rhs)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


def phone_photo(page_img: "Image.Image", name: str) -> "Image.Image":
    rnd = rng_for(name)
    page = page_img.convert("RGB")
    r, g, b = page.split()
    page = Image.merge("RGB", (r.point(lambda v: int(v * 0.99)), g.point(lambda v: int(v * 0.96)),
                               b.point(lambda v: int(v * 0.88))))
    W, H = 2400, 1800
    pw, ph = page.size
    # desk behind the page
    desk = Image.new("RGB", (W, H), (98, 84, 70))
    grad = Image.linear_gradient("L").resize((W, H)).rotate(35, expand=False, fillcolor=128)
    desk = Image.composite(Image.new("RGB", (W, H), (126, 108, 88)), desk, grad)
    # where the page corners land in the photo (a little keystone and rotation)
    j = lambda s: rnd.uniform(-s, s)  # noqa: E731
    dst = [(170 + j(30), 150 + j(30)), (W - 140 + j(30), 210 + j(30)),
           (W - 230 + j(30), H - 120 + j(25)), (120 + j(30), H - 170 + j(25))]
    src = [(0, 0), (pw, 0), (pw, ph), (0, ph)]
    coeffs = _perspective_coeffs(src, dst)
    warped = page.transform((W, H), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)
    mask = Image.new("L", (pw, ph), 255).transform((W, H), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)
    photo = Image.composite(warped, desk, mask)
    # uneven light: brighter top left, darker bottom right, soft vignette
    light = Image.radial_gradient("L").resize((W, H))
    light = light.point(lambda v: int(255 - v * 0.42))
    shade = Image.linear_gradient("L").rotate(-40, expand=False, fillcolor=128).resize((W, H))
    shade = shade.point(lambda v: int(255 - v * 0.22))
    lighting = ImageChops.multiply(light, shade)
    photo = ImageChops.multiply(photo, Image.merge("RGB", (lighting, lighting, lighting)))
    photo = photo.point(lambda v: min(255, int(v * 1.18)))
    photo = photo.filter(ImageFilter.GaussianBlur(0.9))
    noise = Image.effect_noise((W, H), 5).convert("RGB")
    return ImageChops.add(photo, noise, scale=1.0, offset=-128)


def screenshot(page_img: "Image.Image", title: str) -> "Image.Image":
    page = page_img.convert("RGB")
    pw, ph = page.size
    W, H = pw + 120, ph + 150
    shot = Image.new("RGB", (W, H), (82, 86, 89))
    draw = ImageDraw.Draw(shot)
    draw.rectangle([0, 0, W, 38], fill=(233, 234, 237))
    draw.rectangle([0, 38, W, 78], fill=(50, 54, 57))
    font = docgen._font(False, 15)
    draw.text((16, 26), f"{title}  -  PDM Viewer", fill=(40, 40, 40), font=font, anchor="ls")
    draw.text((16, 64), "File   View   Markup   Help", fill=(232, 234, 237), font=font, anchor="ls")
    draw.text((W - 20, 64), "100%   Page 1 / 1", fill=(232, 234, 237), font=font, anchor="rs")
    x, y = 60, 108
    draw.rectangle([x + 6, y + 6, x + pw + 6, y + ph + 6], fill=(40, 42, 44))
    shot.paste(page, (x, y))
    return shot


def jpeg_bytes(img: "Image.Image", quality: int, dpi: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, dpi=(dpi, dpi), optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
def render_file(email_id: str, spec: Dict[str, Any]) -> Tuple[str, bytes, Dict[str, Any]]:
    """(file name, bytes, truth entry) for one attachment."""
    cfg = RENDER.get((email_id, spec["name"]))
    kind = spec["kind"]
    truth: Dict[str, Any] = {"render": cfg["mode"] if cfg else "digital", "spec": spec,
                             "copyable": cfg is None}
    if kind == "model":
        return spec["name"], drawings.step_file(spec).encode("utf-8"), truth
    pages = attachments.pages_for(spec)
    truth["words"] = docgen.page_words(pages)
    if cfg is None:
        data, _ = attachments.spec_file(spec)
        return spec["name"], data, truth
    mode, name = cfg["mode"], cfg.get("rename", spec["name"])
    title = docgen.clean(spec.get("part_number") or spec.get("rfq_number") or spec.get("title") or name)
    if mode in ("scan", "copier"):
        dpi = 300 if mode == "scan" else 200
        imgs = [office_scan(docgen.to_image(p, dpi), cfg, f"{name}#{i}", dpi) for i, p in enumerate(pages)]
        data = docgen.image_pdf([(jpeg_bytes(im, 68 if mode == "scan" else 58, dpi), im.size[0], im.size[1],
                                  "jpeg-gray", dpi) for im in imgs], producer=SCANNER_NAMES[mode])
    elif mode == "fax":
        dpi = 200
        imgs = [fax(docgen.to_image(p, dpi), cfg, f"{name}#{i}") for i, p in enumerate(pages)]
        data = docgen.image_pdf([(im.tobytes(), im.size[0], im.size[1], "bilevel", dpi) for im in imgs],
                                producer=SCANNER_NAMES[mode])
    elif mode == "photo":
        data = jpeg_bytes(phone_photo(docgen.to_image(pages[0], 220), name), 84, 72)
    elif mode == "screen":
        buf = io.BytesIO()
        screenshot(docgen.to_image(pages[0], 120, supersample=3), title).save(buf, "PNG", optimize=True)
        data = buf.getvalue()
    else:
        raise ValueError(mode)
    truth["dpi"] = {"scan": 300, "copier": 200, "fax": 200, "photo": 220, "screen": 120}[mode]
    return name, data, truth


def main() -> int:
    data = json.loads((ROOT / "data" / "sample_emails.json").read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in data["emails"]}
    missing = [(e, n) for (e, n) in RENDER if e not in EMAIL_IDS
               or n not in [a["name"] for a in by_id[e]["attachments"]]]
    if missing:
        raise SystemExit(f"RENDER names attachments that are not in the beta inbox: {missing}")
    if FILES_DIR.exists():
        for old in sorted(FILES_DIR.rglob("*"), reverse=True):
            old.unlink() if old.is_file() else old.rmdir()
    emails, truth_files = [], {}
    for eid in EMAIL_IDS:
        src = by_id[eid]
        body = src["body"]
        for old, new in BODY_EDITS.get(eid, []):
            if old not in body:
                raise SystemExit(f"{eid}: body edit target not found: {old!r}")
            body = body.replace(old, new)
        atts = []
        for spec in src.get("attachments") or []:
            name, blob, truth = render_file(eid, spec)
            path = FILES_DIR / eid / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            rel = path.relative_to(ROOT / "data").as_posix()
            media = {".pdf": "pdf", ".step": "step", ".stp": "step", ".jpg": "jpg", ".png": "png"}[path.suffix.lower()]
            atts.append({"name": name, "kind": "file", "path": rel, "media": media})
            truth_files[rel] = truth
        emails.append({"id": eid, "received": src["received"], "from_name": src["from_name"],
                       "from_email": src["from_email"], "subject": src["subject"], "body": body,
                       "attachments": atts, "expected": src["expected"]})
    count = sum(len(e["attachments"]) for e in emails)
    if count != 30:
        raise SystemExit(f"The beta inbox must have exactly 30 files, it has {count}.")
    inbox = {
        "about": "RFQ details beta inbox: 22 emails copied from sample_emails.json with 30 real attachment "
                 "files. 13 files are uncopyable (image-only scans, a fax, a phone photo, a screenshot), "
                 "so their text comes from OCR. Built by tools/make_rfq_beta.py. All companies and people "
                 "are fictional; 'expected' is the answer key.",
        "emails": emails,
        "paste_examples": data.get("paste_examples", []),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "emails.json").write_text(json.dumps(inbox, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    TRUTH_FILE.write_text(json.dumps({"about": "What is printed on each beta file, from the specs that made it.",
                                      "files": truth_files}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    sizes = sorted(((p.stat().st_size, p.relative_to(ROOT).as_posix()) for p in FILES_DIR.rglob("*") if p.is_file()),
                   reverse=True)
    print(f"{len(emails)} emails, {count} files ({sum(1 for t in truth_files.values() if not t['copyable'])} "
          f"uncopyable), {sum(s for s, _ in sizes) / 1e6:.1f} MB")
    for size, rel in sizes[:6]:
        print(f"  {size / 1e3:8.0f} KB  {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
