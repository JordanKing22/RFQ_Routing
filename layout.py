"""
Page layout detection: find the regions of an RFQ page that matter, so OCR can read each one with
settings suited to it and the extractor knows where every piece of text came from.

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
RFQ_LAYOUT_WORKERS (default 1) limits how many detections run at the same time.

    python layout.py FILE [--page N] [--dpi 150] [--conf 0.35] [--save out.png] [--json]
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("rfq.layout")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("RFQ_LAYOUT_MODEL") or os.path.join(HERE, "models", "rfq_layout.onnx")

CLASSES = ["title_block", "revision_block", "notes", "export_legend", "proprietary_notice",
           "form_header", "line_table", "requirements"]

IMGSZ = 640                 # the model's fixed input size (exported at 640 x 640)
PAD_VALUE = 114             # letterbox grey, the value the model was trained with
MAX_DET = 100
MAX_PIXELS = 60_000_000     # refuse absurd pictures instead of decoding them (about 7750 x 7750)
PDF_TIMEOUT = float(os.environ.get("RFQ_LAYOUT_PDF_TIMEOUT", "60"))
MAX_PDF_DPI = 400


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
def _open(image: Any, draft: bool = True) -> Tuple[Any, float, float]:
    """A PIL RGB image and the factors that take its pixels back to the caller's pixels.
    Big JPEGs are decoded at a reduced scale (JPEG draft mode): the detector only needs 640.
    Bytes from a phone can carry an EXIF orientation (the pixels are stored sideways and viewers
    turn them); those are turned upright first, because the model only knows upright pages, so
    boxes for such a file are in the upright picture's pixels, the way a viewer shows it."""
    from PIL import Image, ImageOps
    if isinstance(image, Image.Image):
        img = image
        fx = fy = 1.0
    elif isinstance(image, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(bytes(image)))
        w0, h0 = img.size
        if w0 * h0 > MAX_PIXELS:
            raise ValueError(f"picture too large: {w0} x {h0}")
        try:
            orientation = int(img.getexif().get(0x0112, 1))
        except Exception:  # a damaged EXIF block is not a reason to give up on the picture
            orientation = 1
        if draft and img.format == "JPEG":
            img.draft("RGB", (IMGSZ * 2, IMGSZ * 2))  # decodes at 1/2, 1/4 or 1/8 scale, never below 1280
        img.load()
        if orientation in (2, 3, 4, 5, 6, 7, 8):
            img = ImageOps.exif_transpose(img)
            if orientation >= 5:  # a quarter turn swaps width and height
                w0, h0 = h0, w0
        fx, fy = w0 / img.size[0], h0 / img.size[1]
    else:
        raise TypeError(f"expected a PIL image or PNG/JPEG bytes, got {type(image).__name__}")
    if img.size[0] < 8 or img.size[1] < 8 or img.size[0] * img.size[1] > MAX_PIXELS:
        raise ValueError(f"unusable picture size {img.size}")
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA", "P", "PA"):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))  # transparent areas read as paper
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    return img, fx, fy


def _letterbox(img: Any) -> Tuple[Any, float, int, int]:
    """Scale to fit 640 x 640 keeping the aspect ratio, pad with grey, same arithmetic as Ultralytics.
    Pillow's bilinear resize filters when it shrinks, like the anti-aliased pages the model trained on."""
    import numpy as np
    from PIL import Image
    w, h = img.size
    r = min(IMGSZ / w, IMGSZ / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.BILINEAR)
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
    mask = best >= conf
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
        img, fx, fy = _open(image)
        x, r, left, top = _letterbox(img)
        with _run_gate:
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


def render_pdf_page(data: bytes, page: int = 1, dpi: int = 150) -> Any:
    """One PDF page as a PIL image via pdftoppm, or None."""
    try:
        from PIL import Image
        if not data or not bytes(data[:1024]).lstrip().startswith(b"%PDF") or shutil.which("pdftoppm") is None:
            return None
        page = max(1, int(page))
        dpi = max(36, min(MAX_PDF_DPI, int(dpi)))
        with tempfile.TemporaryDirectory(prefix="rfq_layout_") as tmp:
            src = os.path.join(tmp, "in.pdf")
            with open(src, "wb") as fh:
                fh.write(bytes(data))
            out = os.path.join(tmp, "page")
            subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-png", "-singlefile",
                            src, out], check=True, capture_output=True, timeout=PDF_TIMEOUT)
            with Image.open(out + ".png") as im:
                im.load()
                return im.convert("RGB")
    except Exception as exc:
        log.debug("pdf page render failed: %s: %s", type(exc).__name__, exc)
        return None


def detect_pdf_page(data: bytes, page: int = 1, dpi: int = 150, conf: float = 0.35,
                    iou: float = 0.5) -> List[Dict[str, Any]]:
    """Regions on one PDF page, in pixels of that page rendered at dpi (what render_pdf_page returns)."""
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
    if data.lstrip()[:4] == b"%PDF":
        img = render_pdf_page(data, args.page, args.dpi)
    else:
        img = None
        try:
            from PIL import Image, ImageOps
            img = Image.open(io.BytesIO(data))
            img.load()
            img = ImageOps.exif_transpose(img)  # upright, as detect() does with bytes
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
