"""
Page layout detection: find the regions of an RFQ page that matter, so the extractor knows where
every piece of text was printed (reading each region with its own OCR settings was measured and
is off; see docs/layout_model.md).

A small YOLO detector (YOLO11n, fine-tuned on this demo's drawings, RFQ forms and POs rendered
through scan, copier, fax, photo and screenshot effects; see docs/layout_model.md) runs from
models/rfq_layout.onnx with onnxruntime. No torch and no ultralytics at runtime: numpy,
onnxruntime and Pillow only, plus the pdftoppm program for PDF pages.

    CLASSES                     the 8 region labels, in model order
    available() -> dict         what is installed and whether the model loads
    detect(image, conf=0.35, iou=0.5) -> [{"label", "conf", "box": [x0, y0, x1, y1]}]
                                image is a PIL image or PNG/JPEG bytes; boxes in its pixels
    detect_pdf_page(data, page=1, dpi=150) -> same, in pixels of the page rendered at dpi
    render_pdf_page(data, page=1, dpi=150) -> PIL image or None (what detect_pdf_page sees)
    crops(image, detections, pad=6) -> [(detection, PIL image)] regions ready for OCR

Nothing here raises on bad input: a broken file, a missing model or a missing package gives []
(available() says why), and nothing is printed. The onnxruntime session is created on first use
and runs on RFQ_LAYOUT_THREADS threads (default 1), so it behaves on a host with 0.1 CPU;
RFQ_LAYOUT_WORKERS (default 1) limits how many detections, decoding included, run at the same time.

    python layout.py FILE [--page N] [--dpi 150] [--conf 0.35] [--save out.png] [--json]
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("rfq.layout")

# onnxruntime (1.30) sends usage events (model loads, sessions, device) to Microsoft
# (mobile.events.data.microsoft.com) unless this is set before it is imported. No document content
# goes out, but this demo handles ITAR and CUI drawings and should call no one but Jev.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
# Absolute from the start: the session loads lazily, and a relative RFQ_LAYOUT_MODEL would otherwise be
# looked up from whatever the working directory is by then.
MODEL_PATH = os.path.abspath(os.environ.get("RFQ_LAYOUT_MODEL") or os.path.join(HERE, "models", "rfq_layout.onnx"))

CLASSES = ["title_block", "revision_block", "notes", "export_legend", "proprietary_notice",
           "form_header", "line_table", "requirements"]

IMGSZ = 640                 # the model's fixed input size (exported at 640 x 640)
PAD_VALUE = 114             # letterbox grey, the value the model was trained with
MAX_DET = 100
RESIZE = "bilinear"         # how a page is shrunk to 640: "bilinear" or "box" (area averaging)
MAX_PIXELS = 60_000_000     # refuse absurd pictures instead of decoding them (about 7750 x 7750)
IMAGE_FORMATS = ("PNG", "JPEG", "TIFF", "BMP", "GIF", "WEBP", "PPM")  # what detect() opens from bytes (JPEG covers MPO)
PDF_TIMEOUT = float(os.environ.get("RFQ_LAYOUT_PDF_TIMEOUT", "60"))
MAX_PDF_DPI = 400
MAX_PDF_PIXELS = 25_000_000  # a PDF page renders at a lower dpi rather than past this (5000 x 5000)


def _threads() -> int:
    try:
        return max(1, int(os.environ.get("RFQ_LAYOUT_THREADS", "1")))
    except ValueError:
        return 1


def _workers() -> int:
    try:
        return max(1, int(os.environ.get("RFQ_LAYOUT_WORKERS", "1")))
    except ValueError:
        return 1


_lock = threading.Lock()
_run_gate = threading.BoundedSemaphore(_workers())
_state: Dict[str, Any] = {"session": None, "error": None, "tried": False}


def reset() -> None:
    """Forget the session (tests, or after replacing the model file)."""
    with _lock:
        _state.update(session=None, error=None, tried=False)


def _session() -> Any:
    """The onnxruntime session, created once. Returns None (and records why) when it cannot load."""
    if _state["tried"]:
        return _state["session"]
    with _lock:
        if _state["tried"]:
            return _state["session"]
        try:
            import onnxruntime as ort
            if not os.path.isfile(MODEL_PATH):
                raise FileNotFoundError(f"model not found: {MODEL_PATH}")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = _threads()
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.log_severity_level = 3  # errors only; onnxruntime logs to stderr, never stdout
            sess = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
            shape = sess.get_inputs()[0].shape
            if list(shape[-2:]) != [IMGSZ, IMGSZ]:
                raise ValueError(f"model input is {shape}, expected [1, 3, {IMGSZ}, {IMGSZ}]")
            outs = sess.get_outputs()[0].shape
            if len(outs) != 3 or outs[1] != 4 + len(CLASSES):
                raise ValueError(f"model output is {outs}, expected [1, {4 + len(CLASSES)}, N]")
            _state["session"] = sess
        except Exception as exc:  # missing package, missing or damaged model
            _state["error"] = f"{type(exc).__name__}: {exc}"
            log.debug("layout model unavailable: %s", _state["error"])
        _state["tried"] = True
        return _state["session"]


def available() -> Dict[str, Any]:
    """What is installed and whether the model loads. "ready" is the one flag callers need: True when
    detect() can run (attachments.layout_ready reads it). Loading the model here is the only side effect."""
    info: Dict[str, Any] = {"ready": False, "model": None, "numpy": False, "onnxruntime": False,
                            "onnxruntime_version": None, "pillow": False,
                            "pdftoppm": shutil.which("pdftoppm") is not None, "model_path": MODEL_PATH,
                            "model_bytes": os.path.getsize(MODEL_PATH) if os.path.isfile(MODEL_PATH) else None,
                            "threads": _threads(), "loaded": False, "error": None, "classes": list(CLASSES)}
    try:
        import numpy  # noqa: F401
        info["numpy"] = True
    except ImportError:
        pass
    try:
        import onnxruntime
        info["onnxruntime"] = True
        info["onnxruntime_version"] = onnxruntime.__version__
    except ImportError:
        pass
    try:
        import PIL  # noqa: F401
        info["pillow"] = True
    except ImportError:
        pass
    if info["numpy"] and info["onnxruntime"] and info["pillow"]:
        info["loaded"] = _session() is not None
        info["error"] = _state["error"]
        info["ready"] = info["loaded"]
        info["model"] = os.path.basename(MODEL_PATH) if info["loaded"] else None
    else:
        missing = [k for k in ("numpy", "onnxruntime", "pillow") if not info[k]]
        info["error"] = "missing: " + ", ".join(missing)
    return info


# --------------------------------------------------------------------------- #
# Pictures in, boxes out
# --------------------------------------------------------------------------- #
def _plain_mode(img: Any) -> Any:
    """The picture in L, LA, RGB or RGBA, the modes Image.reduce and the RGB flattening handle.
    16-bit and 32-bit greyscale (some scanners write 16-bit PNG or TIFF) is stretched to 8 bits over
    its own range: a plain convert() clips everything above 255, which turns such a scan white."""
    mode = img.mode
    if mode in ("L", "LA", "RGB", "RGBA"):
        return img
    if mode == "1":
        return img.convert("L")
    if mode.startswith("I;16") or mode in ("I", "F"):
        if mode.startswith("I;16"):
            img = img.convert("I")
        lo, hi = img.getextrema()
        if hi > 255 or lo < 0 or (mode == "F" and hi <= 1.0):
            k = 255.0 / max(float(hi) - float(lo), 1e-6)
            img = img.point(lambda v: v * k + (-float(lo) * k))
        return img.convert("L")
    if mode in ("P", "PA"):
        return img.convert("RGBA" if mode == "PA" or "transparency" in img.info else "RGB")
    return img.convert("RGB")  # CMYK, YCbCr, LAB, HSV


def _reduced(img: Any, factor: int) -> Any:
    """The picture box-averaged down by a whole factor, making as few full-size copies as possible.
    Image.reduce cannot average 1-bit, palette or 16-bit pictures, so those change mode first: 1 bit
    and palettes to L, RGB or RGBA, and 16 bits to 32-bit I, which is reduced before it is stretched
    to 8 bits (stretching first held three full-size copies: 530 MB for a 49 MP 16-bit scan). Pillow
    reduces RGBA and LA through a full-size premultiplied copy; when the alpha is opaque, as it is in
    most scans and screenshots saved with alpha, the colour channels are reduced one at a time instead."""
    from PIL import Image
    if img.mode.startswith("I;16"):
        img = img.convert("I")
    elif img.mode in ("1", "P", "PA"):
        img = _plain_mode(img)
    if img.mode in ("RGBA", "LA") and img.getextrema()[-1] == (255, 255):
        bands = [img.getchannel(b).reduce(factor) for b in img.getbands()[:-1]]
        return bands[0] if len(bands) == 1 else Image.merge("RGB", bands)
    return img.reduce(factor)


def _open(image: Any, draft: bool = True) -> Tuple[Any, float, float]:
    """A PIL RGB image and the factors that take its pixels back to the caller's pixels.
    With draft (the detector's path) a big picture is made small early, since the detector only
    needs 640: JPEGs decode at 1/2, 1/4 or 1/8 scale (JPEG draft mode), and anything else 2560
    pixels or more across (four times 640) is box-averaged down by a whole factor before the color
    conversions, which would otherwise copy the full-size picture two or three times.
    Bytes from a phone can carry an EXIF orientation (the pixels are stored sideways and viewers
    turn them); those are turned upright first, because the model only knows upright pages, so
    boxes for such a file are in the upright picture's pixels, the way a viewer shows it."""
    from PIL import Image, ImageOps
    if isinstance(image, Image.Image):
        img = image
        w0, h0 = img.size
    elif isinstance(image, (bytes, bytearray, memoryview)):
        # Only plain raster formats: Pillow's EPS plugin, for one, would hand the bytes to Ghostscript.
        img = Image.open(io.BytesIO(bytes(image)), formats=IMAGE_FORMATS)
        w0, h0 = img.size
        if w0 < 8 or h0 < 8 or w0 * h0 > MAX_PIXELS:  # from the header, before anything is decoded
            raise ValueError(f"unusable picture size {w0} x {h0}")
        try:
            orientation = int(img.getexif().get(0x0112, 1))
        except Exception:  # a damaged EXIF block is not a reason to give up on the picture
            orientation = 1
        if draft and img.format in ("JPEG", "MPO"):
            img.draft("RGB", (IMGSZ * 2, IMGSZ * 2))  # never below 1280 on either side
        img.load()
        if orientation in (2, 3, 4, 5, 6, 7, 8):
            img = ImageOps.exif_transpose(img)
            if orientation >= 5:  # a quarter turn swaps width and height
                w0, h0 = h0, w0
    else:
        raise TypeError(f"expected a PIL image or PNG/JPEG bytes, got {type(image).__name__}")
    if img.size[0] < 8 or img.size[1] < 8 or img.size[0] * img.size[1] > MAX_PIXELS:
        raise ValueError(f"unusable picture size {img.size}")
    factor = max(img.size) // (IMGSZ * 2) if draft else 1
    if factor >= 2:
        img = _reduced(img, factor)
    img = _plain_mode(img)
    if img.mode in ("LA", "RGBA"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))  # transparent areas read as paper
        bg.paste(rgba, mask=rgba.getchannel("A"))
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img, w0 / img.size[0], h0 / img.size[1]


def _letterbox(img: Any) -> Tuple[Any, float, int, int]:
    """Scale to fit 640 x 640 keeping the aspect ratio, pad with grey, same arithmetic as Ultralytics
    (sizes, padding and rounding are identical; the resize filter is RESIZE, see there)."""
    import numpy as np
    from PIL import Image
    w, h = img.size
    r = min(IMGSZ / w, IMGSZ / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.Resampling.BOX if RESIZE == "box" else Image.Resampling.BILINEAR)
    left = int(round((IMGSZ - nw) / 2 - 0.1))
    top = int(round((IMGSZ - nh) / 2 - 0.1))
    canvas = Image.new("RGB", (IMGSZ, IMGSZ), (PAD_VALUE,) * 3)
    canvas.paste(img, (left, top))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), r, left, top


def _nms(boxes: Any, scores: Any, iou: float) -> List[int]:
    """Greedy non-maximum suppression, highest score first."""
    import numpy as np
    order = scores.argsort()[::-1]
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        iw = np.maximum(0.0, np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]))
        ih = np.maximum(0.0, np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]))
        inter = iw * ih
        overlap = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[overlap <= iou]
    return keep


def _decode(out: Any, conf: float, iou: float) -> List[Dict[str, Any]]:
    """YOLO11 head output [1, 4 + classes, anchors] (cx, cy, w, h in 640 pixels, class scores
    already through a sigmoid) to detections in 640-space, with class-wise NMS."""
    import numpy as np
    pred = np.asarray(out, dtype=np.float32)[0].T  # [anchors, 4 + classes]
    scores = pred[:, 4:]
    cls = scores.argmax(1)
    best = scores[np.arange(scores.shape[0]), cls]
    mask = best > conf  # strictly above, as Ultralytics filters
    if not mask.any():
        return []
    pred, cls, best = pred[mask], cls[mask], best[mask]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    found: List[Dict[str, Any]] = []
    for c in np.unique(cls):  # class-wise: a notes box never suppresses a legend box
        idx = np.nonzero(cls == c)[0]
        for k in _nms(boxes[idx], best[idx], iou):
            j = idx[k]
            found.append({"label": CLASSES[int(c)], "conf": float(best[j]), "box": [float(v) for v in boxes[j]]})
    found.sort(key=lambda d: -d["conf"])
    return found[:MAX_DET]


def detect(image: Any, conf: float = 0.35, iou: float = 0.5) -> List[Dict[str, Any]]:
    """Regions on one page picture. Boxes are [x0, y0, x1, y1] in the picture's own pixels, sorted by
    confidence. Returns [] when the input is unusable or the model cannot run."""
    try:
        sess = _session()
        if sess is None:
            return []
        # The gate covers decoding too: a big picture briefly holds its full decoded size (a 60 MP
        # PNG is about 240 MB), so RFQ_LAYOUT_WORKERS bounds memory as well as CPU.
        with _run_gate:
            img, fx, fy = _open(image)
            x, r, left, top = _letterbox(img)
            out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        W, H = img.size[0] * fx, img.size[1] * fy
        dets = []
        for d in _decode(out, float(conf), float(iou)):
            x0, y0, x1, y1 = d["box"]
            box = [min(max((x0 - left) / r * fx, 0.0), W), min(max((y0 - top) / r * fy, 0.0), H),
                   min(max((x1 - left) / r * fx, 0.0), W), min(max((y1 - top) / r * fy, 0.0), H)]
            if box[2] - box[0] >= 1 and box[3] - box[1] >= 1:
                dets.append({"label": d["label"], "conf": round(d["conf"], 4), "box": [round(v, 1) for v in box]})
        return dets
    except Exception as exc:  # never let a bad file take the caller down
        log.debug("layout detect failed: %s: %s", type(exc).__name__, exc)
        return []


def _pdf_page_points(path: str, page: int) -> Optional[Tuple[float, float]]:
    """Width and height of one page in points, from pdfinfo (poppler, next to pdftoppm), or None."""
    if shutil.which("pdfinfo") is None:
        return None
    proc = subprocess.run(["pdfinfo", "-f", str(page), "-l", str(page), path], capture_output=True,
                          timeout=PDF_TIMEOUT)
    m = re.search(rb"Page\s+\d+\s+size:\s+([0-9.]+)\s+x\s+([0-9.]+)", proc.stdout)
    return (float(m.group(1)), float(m.group(2))) if m else None


def render_pdf_page(data: bytes, page: int = 1, dpi: int = 150) -> Any:
    """One PDF page as a PIL image via pdftoppm, or None. A page so large that it would pass
    MAX_PDF_PIXELS at this dpi (a wall-sized sheet, or a hostile MediaBox) renders at a lower dpi,
    so neither pdftoppm nor Pillow holds more than about 100 MB for it. The page size comes from
    pdfinfo; without it, pdftoppm renders at most the top left 5000 x 5000 pixels of the page."""
    try:
        from PIL import Image
        if not data or b"%PDF" not in bytes(data[:1024]) or shutil.which("pdftoppm") is None:
            return None
        page = max(1, int(page))
        dpi = float(max(36, min(MAX_PDF_DPI, int(dpi))))
        with tempfile.TemporaryDirectory(prefix="rfq_layout_") as tmp:
            src = os.path.join(tmp, "in.pdf")
            with open(src, "wb") as fh:
                fh.write(bytes(data))
            size = _pdf_page_points(src, page)
            crop: List[str] = []
            if size:
                px = size[0] * size[1] * (dpi / 72.0) ** 2
                if px > MAX_PDF_PIXELS:
                    dpi = max(1.0, dpi * (MAX_PDF_PIXELS / px) ** 0.5)
            else:
                # No page size (pdfinfo missing or confused): cap the bitmap pdftoppm allocates instead.
                # It renders only this crop area, so a hostile MediaBox costs at most MAX_PDF_PIXELS.
                side = str(int(MAX_PDF_PIXELS ** 0.5))
                crop = ["-x", "0", "-y", "0", "-W", side, "-H", side]
            out = os.path.join(tmp, "page")
            subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", f"{dpi:.2f}"] + crop +
                           ["-png", "-singlefile", src, out], check=True, capture_output=True, timeout=PDF_TIMEOUT)
            with Image.open(out + ".png", formats=("PNG",)) as im:
                if im.size[0] * im.size[1] > MAX_PIXELS:  # no page size from pdfinfo and still huge
                    return None
                im.load()
                return im if im.mode == "RGB" else im.convert("RGB")
    except Exception as exc:
        log.debug("pdf page render failed: %s: %s", type(exc).__name__, exc)
        return None


def detect_pdf_page(data: bytes, page: int = 1, dpi: int = 150, conf: float = 0.35,
                    iou: float = 0.5) -> List[Dict[str, Any]]:
    """Regions on one PDF page, in pixels of that page rendered at dpi (the picture render_pdf_page
    returns, which is smaller for a page too large to render at that dpi)."""
    img = render_pdf_page(data, page, dpi)
    return detect(img, conf, iou) if img is not None else []


def crops(image: Any, detections: List[Dict[str, Any]], pad: int = 6) -> List[Tuple[Dict[str, Any], Any]]:
    """Each detected region cut out of the picture (with a small margin), ready for OCR."""
    try:
        img, _, _ = _open(image, draft=False)
        out = []
        for d in detections:
            x0, y0, x1, y1 = d["box"]
            box = (max(0, int(x0) - pad), max(0, int(y0) - pad),
                   min(img.size[0], int(x1 + 0.999) + pad), min(img.size[1], int(y1 + 0.999) + pad))
            if box[2] > box[0] and box[3] > box[1]:
                out.append((d, img.crop(box)))
        return out
    except Exception as exc:
        log.debug("layout crops failed: %s: %s", type(exc).__name__, exc)
        return []


def draw(image: Any, detections: List[Dict[str, Any]]) -> Any:
    """A copy of the picture with the boxes and labels drawn on it."""
    from PIL import ImageDraw, ImageFont
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    colors = [(220, 40, 40), (40, 150, 40), (40, 90, 220), (230, 130, 0), (150, 60, 200),
              (0, 170, 170), (200, 0, 120), (120, 120, 0)]
    width = max(2, img.size[0] // 500)
    try:
        font = ImageFont.load_default(size=max(12, img.size[0] // 80))
    except TypeError:  # Pillow older than 10.1
        font = ImageFont.load_default()
    for det in detections:
        c = colors[CLASSES.index(det["label"]) % len(colors)]
        x0, y0, x1, y1 = det["box"]
        d.rectangle([x0, y0, x1, y1], outline=c, width=width)
        d.text((x0 + 2, max(0, y0 - 2)), f"{det['label']} {det['conf']:.2f}", fill=c, font=font, anchor="lb")
    return img


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Find title blocks, legends, tables and notes on an RFQ page.")
    ap.add_argument("file", help="PDF, PNG or JPEG")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=150, help="PDF rasterizing resolution")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--save", help="write the picture with the boxes drawn on it")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args(argv)
    info = available()
    if not info["loaded"]:
        print(f"layout model not available: {info['error']}")
        return 2
    try:
        with open(args.file, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        print(f"cannot read {args.file}: {exc}")
        return 1
    t0 = time.perf_counter()
    if b"%PDF" in data[:1024]:
        img = render_pdf_page(data, args.page, args.dpi)
    else:
        img = None
        try:
            img = _open(data, draft=False)[0]  # upright and RGB, full size, as detect() reads bytes
        except Exception as exc:
            print(f"cannot read {args.file}: {exc}")
            return 1
    t1 = time.perf_counter()
    if img is None:
        print(f"cannot render {args.file}")
        return 1
    dets = detect(img, args.conf, args.iou)
    t2 = time.perf_counter()
    if args.json:
        print(json.dumps({"file": args.file, "size": list(img.size), "detections": dets,
                          "seconds": {"load": round(t1 - t0, 3), "detect": round(t2 - t1, 3)}}, indent=1))
    else:
        print(f"{args.file}: {img.size[0]} x {img.size[1]} px, load {t1 - t0:.2f} s, detect {t2 - t1:.2f} s "
              f"on {_threads()} thread(s)")
        for d in dets:
            print(f"  {d['label']:20s} {d['conf']:.2f}  [{', '.join(f'{v:.0f}' for v in d['box'])}]")
        if not dets:
            print("  (nothing found)")
    if args.save:
        draw(img, dets).save(args.save)
        if not args.json:
            print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
