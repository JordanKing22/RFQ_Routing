"""
Train, evaluate, and export the page layout detector (models/rfq_layout.onnx, used by layout.py).

It fine-tunes Ultralytics YOLO11n (COCO pretrained) on the synthetic pages from
tools/make_layout_dataset.py, on CPU, then exports ONNX for the runtime, which needs only numpy,
onnxruntime and Pillow. Needs torch and ultralytics (a separate virtualenv; the demo does not).

    python tools/make_layout_dataset.py --out ../yolo_work/layout_data
    python tools/train_layout.py time   --data ../yolo_work/layout_data/data.yaml   # one epoch, prints s/epoch
    python tools/train_layout.py train  --data ../yolo_work/layout_data/data.yaml --epochs 20 --budget-min 70
    python tools/train_layout.py eval   --data ../yolo_work/layout_data/data.yaml   # per-class mAP, val and beta
    python tools/train_layout.py export --data ../yolo_work/layout_data/data.yaml   # models/rfq_layout.onnx + parity

Runs, downloaded weights, and metrics go to --work (default ../yolo_work next to the repository),
never into the repository. Only the exported ONNX file lands in models/.

Why these settings:
    imgsz 640     the runtime letterboxes every page to 640, and CPU time grows with the square
    fliplr 0      mirrored text and mirrored title blocks never happen on a real page
    mosaic        pages glued at random scales, so the detector does not memorize where this
                  renderer puts the title block (another agent keeps polishing drawings.py)
    batch 16, workers 3, cache ram   fits 4 CPU cores and 15 GB; the dataset is about 1 GB decoded
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSES = ["title_block", "revision_block", "notes", "export_legend", "proprietary_notice",
           "form_header", "line_table", "requirements"]
MODEL_OUT = ROOT / "models" / "rfq_layout.onnx"
DEFAULT_WORK = ROOT.parent / "yolo_work"
RUN_NAME = "rfq_layout"


def _yolo():
    from ultralytics import YOLO  # imported late: --help works without torch
    return YOLO


def train_args(args: argparse.Namespace, epochs: int, name: str) -> Dict[str, Any]:
    return dict(
        data=str(Path(args.data).resolve()), imgsz=640, epochs=epochs, batch=args.batch, workers=args.workers,
        device="cpu", project=str(Path(args.work).resolve() / "runs"), name=name, exist_ok=True,
        fliplr=0.0, flipud=0.0, mosaic=1.0, close_mosaic=max(2, epochs // 6), degrees=0.0, translate=0.1,
        scale=0.5, mixup=0.0, hsv_h=0.01, hsv_s=0.4, hsv_v=0.4, cache="ram", patience=1000, seed=0,
        deterministic=False, plots=False, amp=False, verbose=True, val=True,
    )


def weights_path(args: argparse.Namespace) -> Path:
    if getattr(args, "weights", None):
        return Path(args.weights).resolve()
    return Path(args.work).resolve() / "runs" / RUN_NAME / "weights" / "best.pt"


def cmd_time(args: argparse.Namespace) -> int:
    """One full epoch (train + val) with the real settings, to pick the epoch count for the budget.
    The steady-state batch time is measured over the second half of the epoch (the first batches
    are slow while threads spin up), and a one-epoch run has mosaic switched off, so mosaic's extra
    loading time is added back as a factor measured on this machine (about 1.2)."""
    YOLO = _yolo()
    stamps: List[float] = []
    val: Dict[str, float] = {}
    t0 = time.time()
    model = YOLO(args.base)
    model.add_callback("on_train_batch_end", lambda tr: stamps.append(time.time()))
    model.add_callback("on_val_start", lambda v: val.setdefault("start", time.time()))
    model.add_callback("on_val_end", lambda v: val.setdefault("end", time.time()))
    model.train(**train_args(args, 1, "timing"))
    seconds = time.time() - t0
    half = stamps[len(stamps) // 2:]
    per_batch = (half[-1] - half[0]) / max(1, len(half) - 1) if len(half) > 1 else seconds
    val_s = val.get("end", t0) - val.get("start", t0)
    per_epoch = per_batch * len(stamps) * 1.2 + val_s
    epochs = max(1, int((args.budget_min * 60 - 120) // per_epoch))  # 2 minutes for setup and the final val
    out = {"seconds_one_epoch_wall": round(seconds, 1), "batches": len(stamps), "seconds_per_batch": round(per_batch, 2),
           "seconds_val": round(val_s, 1), "estimated_seconds_per_epoch": round(per_epoch, 1),
           "budget_min": args.budget_min, "suggested_epochs": epochs}
    (Path(args.work) / "timing.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train for the epochs picked from the timing run. The machine may be shared (OCR jobs run on
    the same cores), so the budget is also handed to Ultralytics (its time argument): it re-plans
    the epoch count and the learning-rate schedule after every epoch to finish inside the budget,
    instead of running long or being cut off before the final no-mosaic epochs."""
    YOLO = _yolo()
    epochs = args.epochs
    if not epochs:
        timing = json.loads((Path(args.work) / "timing.json").read_text())
        epochs = timing["suggested_epochs"]
    t0 = time.time()
    done = {"epochs": 0}
    model = YOLO(args.base)
    model.add_callback("on_fit_epoch_end", lambda tr: done.update(epochs=tr.epoch + 1))
    kw = train_args(args, epochs, RUN_NAME)
    if args.budget_min and not args.no_time_cap:
        kw["time"] = round(args.budget_min / 60.0, 3)
    model.train(**kw)
    minutes = (time.time() - t0) / 60
    info = {"planned_epochs": epochs, "epochs_run": done["epochs"], "train_minutes": round(minutes, 1),
            "time_cap_hours": kw.get("time"), "base": args.base, "batch": args.batch, "imgsz": 640}
    (Path(args.work) / "train_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    return 0


def _metrics(model: Any, data: str, split: str, workers: int) -> Dict[str, Any]:
    r = model.val(data=data, split=split, imgsz=640, batch=8, device="cpu", workers=workers, plots=False,
                  conf=0.001, iou=0.6, verbose=False)
    box = r.box
    per = {}
    for i, c in enumerate(box.ap_class_index):
        per[CLASSES[int(c)]] = {"mAP50": round(float(box.ap50[i]), 4), "mAP50-95": round(float(box.ap[i]), 4)}
    return {"mAP50": round(float(box.map50), 4), "mAP50-95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4), "per_class": per}


def _instances(data_yaml: Path, split: str) -> Dict[str, int]:
    labels = data_yaml.parent / "labels" / {"val": "val", "test": "beta", "train": "train"}[split]
    counts = {c: 0 for c in CLASSES}
    images = 0
    for f in labels.glob("*.txt"):
        images += 1
        for line in f.read_text().split("\n"):
            if line.strip():
                counts[CLASSES[int(line.split()[0])]] += 1
    counts["images"] = images
    return counts


def cmd_eval(args: argparse.Namespace) -> int:
    YOLO = _yolo()
    model = YOLO(str(weights_path(args)))
    data = str(Path(args.data).resolve())
    out = {}
    for split, label in (("val", "synthetic_val"), ("test", "beta")):
        out[label] = _metrics(model, data, split, args.workers)
        out[label]["instances"] = _instances(Path(data), split)
    (Path(args.work) / "metrics.json").write_text(json.dumps(out, indent=1))
    print_table(out)
    return 0


def print_table(out: Dict[str, Any]) -> None:
    print(f"{'class':20s} " + " ".join(f"{k:>24s}" for k in out))
    print(f"{'':20s} " + " ".join(f"{'mAP50  mAP50-95   n':>24s}" for _ in out))
    for c in CLASSES + ["all"]:
        cells = []
        for k, m in out.items():
            if c == "all":
                cells.append(f"{m['mAP50']:7.3f} {m['mAP50-95']:9.3f} {sum(m['instances'][x] for x in CLASSES):4d}")
            elif c in m["per_class"]:
                pc = m["per_class"][c]
                cells.append(f"{pc['mAP50']:7.3f} {pc['mAP50-95']:9.3f} {m['instances'][c]:4d}")
            else:
                cells.append(f"{'-':>7s} {'-':>9s} {m['instances'][c]:4d}")
        print(f"{c:20s} " + " ".join(f"{x:>24s}" for x in cells))


def cmd_export(args: argparse.Namespace) -> int:
    YOLO = _yolo()
    src = weights_path(args)
    model = YOLO(str(src))
    onnx_path = Path(model.export(format="onnx", imgsz=640, opset=args.opset, simplify=True, dynamic=False,
                                  half=False, nms=False, device="cpu"))
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx_path, MODEL_OUT)
    print(f"{MODEL_OUT} {MODEL_OUT.stat().st_size / 1e6:.2f} MB (opset {args.opset}, from {src})")
    report = parity(model, Path(args.data).resolve().parent / "images" / "beta")
    (Path(args.work) / "parity.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "images"}, indent=1))
    return 0 if report["ok"] else 1


def _match(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pair detections by class and IoU; report unmatched ones and the largest corner difference."""
    def iou(p: List[float], q: List[float]) -> float:
        ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
        iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
        inter = ix * iy
        u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
        return inter / u if u > 0 else 0.0
    used, diffs, unmatched = set(), [], []
    for d in a:
        best, bj = 0.0, None
        for j, e in enumerate(b):
            if j not in used and e["label"] == d["label"]:
                v = iou(d["box"], e["box"])
                if v > best:
                    best, bj = v, j
        if bj is None or best < 0.5:
            unmatched.append(d)
            continue
        used.add(bj)
        diffs.append(max(abs(x - y) for x, y in zip(d["box"], b[bj]["box"])))
    return {"max_px": max(diffs) if diffs else 0.0, "unmatched_a": unmatched,
            "unmatched_b": [e for j, e in enumerate(b) if j not in used]}


def parity(model: Any, images_dir: Path, conf: float = 0.35, iou: float = 0.5) -> Dict[str, Any]:
    """onnxruntime (layout.py) against ultralytics on the beta pages, two ways:
    exact: the same letterboxed tensor into both, which checks export, decoding and NMS;
    end to end: each side does its own resizing from the file (Pillow vs OpenCV), the way both run."""
    import numpy as np
    import torch
    from PIL import Image
    import layout
    layout.reset()
    rows, worst_exact, worst_e2e, ok = [], 0.0, 0.0, True
    for path in sorted(images_dir.glob("*.jpg")):
        img = Image.open(path).convert("RGB")
        x, r, left, top = layout._letterbox(img)
        # exact: shared tensor, boxes in 640-space
        res = model.predict(torch.from_numpy(x.copy()), conf=conf, iou=iou, imgsz=640, device="cpu", verbose=False)[0]
        ul = [{"label": CLASSES[int(c)], "box": [float(v) for v in b]}
              for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist())]
        ox = [{"label": d["label"], "box": d["box"]} for d in layout._decode(layout._session().run(None, {
            layout._session().get_inputs()[0].name: x})[0], conf, iou)]
        exact = _match(ul, ox)
        # end to end, boxes in image pixels
        res2 = model.predict(np.asarray(img)[:, :, ::-1].copy(), conf=conf, iou=iou, imgsz=640, device="cpu",
                             verbose=False)[0]
        ul2 = [{"label": CLASSES[int(c)], "box": [float(v) for v in b]}
               for b, c in zip(res2.boxes.xyxy.tolist(), res2.boxes.cls.tolist())]
        ox2 = layout.detect(img, conf=conf, iou=iou)
        e2e = _match(ul2, ox2)
        scale = max(img.size) / 640.0
        exact_bad = bool(exact["unmatched_a"] or exact["unmatched_b"]) or exact["max_px"] > 1.0
        ok = ok and not exact_bad
        worst_exact = max(worst_exact, exact["max_px"])
        worst_e2e = max(worst_e2e, e2e["max_px"] / scale)
        rows.append({"image": path.name, "n_ultralytics": len(ul2), "n_onnxruntime": len(ox2),
                     "exact_max_px_640": round(exact["max_px"], 3), "exact_unmatched": len(exact["unmatched_a"]) + len(exact["unmatched_b"]),
                     "e2e_max_px_640": round(e2e["max_px"] / scale, 2),
                     "e2e_unmatched": [d["label"] for d in e2e["unmatched_a"] + e2e["unmatched_b"]]})
    return {"ok": ok, "images": rows, "exact_worst_px_640": round(worst_exact, 3),
            "e2e_worst_px_640": round(worst_e2e, 2),
            "e2e_images_with_unmatched": sum(1 for r in rows if r["e2e_unmatched"]), "count": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["time", "train", "eval", "export"])
    ap.add_argument("--data", required=True, help="data.yaml written by make_layout_dataset.py")
    ap.add_argument("--work", default=str(DEFAULT_WORK), help="runs, weights and metrics (outside the repo)")
    ap.add_argument("--base", default="yolo11n.pt", help="pretrained weights (downloaded into --work)")
    ap.add_argument("--weights", help="trained weights for eval/export (default: the best.pt of the run)")
    ap.add_argument("--epochs", type=int, default=0, help="default: from the timing run and --budget-min")
    ap.add_argument("--budget-min", type=float, default=70.0, help="training budget in minutes")
    ap.add_argument("--no-time-cap", action="store_true", help="run exactly --epochs, ignoring the budget")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    work = Path(args.work).resolve()
    if ROOT in work.parents or work == ROOT:
        ap.error("keep runs out of the repository")
    work.mkdir(parents=True, exist_ok=True)
    args.data = str(Path(args.data).resolve())
    if args.weights:
        args.weights = str(Path(args.weights).resolve())
    os.chdir(work)  # ultralytics downloads the base weights into the working directory
    args.work = str(work)
    return {"time": cmd_time, "train": cmd_train, "eval": cmd_eval, "export": cmd_export}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
