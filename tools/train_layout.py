"""
Train, evaluate, and export the page layout detector (models/rfq_layout.onnx, used by layout.py).

It fine-tunes Ultralytics YOLO11n on the synthetic pages from tools/make_layout_dataset.py, on CPU,
then exports ONNX for the runtime, which needs only numpy, onnxruntime and Pillow. Training needs
torch and ultralytics (a separate virtualenv; the demo does not). See docs/layout_model.md.

    python tools/make_layout_dataset.py --out ../yolo_work/layout_data_v2 --train 1200 --val 200 --workers 3
    python tools/train_layout.py time   --data ../yolo_work/layout_data_v2/data.yaml   # one epoch, prints s/epoch
    python tools/train_layout.py train  --data ../yolo_work/layout_data_v2/data.yaml --base <best.pt> --epochs 14 --budget-min 45
    python tools/train_layout.py resume --data ../yolo_work/layout_data_v2/data.yaml   # after an interrupt
    python tools/train_layout.py eval   --data ../yolo_work/layout_data_v2/data.yaml   # per-class mAP, val and beta
    python tools/train_layout.py export --data ../yolo_work/layout_data_v2/data.yaml   # models/rfq_layout.onnx + parity
    python tools/train_layout.py bench  --data ../yolo_work/layout_data_v2/data.yaml   # runtime speed and memory

Runs, downloaded weights, and metrics go to --work (default ../yolo_work next to the repository),
never into the repository. Only the exported ONNX file lands in models/. "bench" needs only the
runtime packages (numpy, onnxruntime, Pillow), so it can run in the demo's own virtualenv.

Why these settings:
    imgsz 640     the runtime letterboxes every page to 640, and CPU time grows with the square
    fliplr 0      mirrored text and mirrored title blocks never happen on a real page
    mosaic        pages glued at random scales, so the detector does not memorize where this
                  renderer puts the title block; switched off for the last epochs (close_mosaic)
    threads 3     the machine has 4 cores shared with OCR jobs; Ultralytics loads data in the main
                  process on CPU (workers 0), so 3 torch threads is the whole footprint
    cache ram     the dataset decodes to about 1.5 GB, and decoding JPEGs every epoch is slow
    save_period 1 a checkpoint every epoch (weights/epochN.pt, plus last.pt), so an interrupt
                  costs at most one epoch: "resume" continues from last.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSES = ["title_block", "revision_block", "notes", "export_legend", "proprietary_notice",
           "form_header", "line_table", "requirements"]
MODEL_OUT = ROOT / "models" / "rfq_layout.onnx"
DEFAULT_WORK = ROOT.parent / "yolo_work"
RUN_NAME = "rfq_layout"  # default --name


def _yolo(threads: int):
    """Ultralytics' YOLO class, with torch held to `threads` threads. Ultralytics resets the torch
    thread count to cores - 1 when it selects the CPU device, so its constant is patched too."""
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    import torch
    import ultralytics.utils.torch_utils as tu
    from ultralytics import YOLO  # imported late: --help and bench work without torch
    torch.set_num_threads(threads)
    tu.NUM_THREADS = threads
    return YOLO


def train_args(args: argparse.Namespace, epochs: int, name: str) -> Dict[str, Any]:
    fine_tune = Path(args.base).name != "yolo11n.pt"
    return dict(
        data=str(Path(args.data).resolve()), imgsz=640, epochs=epochs, batch=args.batch, workers=args.workers,
        device="cpu", project=str(Path(args.work).resolve() / "runs"), name=name, exist_ok=True,
        fliplr=0.0, flipud=0.0, mosaic=1.0, close_mosaic=max(2, epochs // 6), degrees=0.0, translate=0.1,
        scale=0.5, mixup=0.0, hsv_h=0.01, hsv_s=0.4, hsv_v=0.4, cache="ram", patience=1000, seed=0,
        deterministic=False, plots=False, amp=False, verbose=True, val=True, save_period=1,
        # a model that already knows these pages needs a short warmup, a COCO model the usual three
        warmup_epochs=1.0 if fine_tune else 3.0,
    )


def weights_path(args: argparse.Namespace) -> Path:
    if getattr(args, "weights", None):
        return Path(args.weights).resolve()
    return Path(args.work).resolve() / "runs" / args.name / "weights" / "best.pt"


def cmd_time(args: argparse.Namespace) -> int:
    """One full epoch (train + val) with the real settings, to pick the epoch count for the budget.
    The steady-state batch time is measured over the second half of the epoch (the first batches
    are slow while threads spin up), and a one-epoch run has mosaic switched off, so mosaic's extra
    loading time is added back as a factor measured on this machine (about 1.2)."""
    YOLO = _yolo(args.threads)
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
    YOLO = _yolo(args.threads)
    epochs = args.epochs
    if not epochs:
        timing = json.loads((Path(args.work) / "timing.json").read_text())
        epochs = timing["suggested_epochs"]
    t0 = time.time()
    done = {"epochs": 0}
    model = YOLO(args.base)
    model.add_callback("on_fit_epoch_end", lambda tr: done.update(epochs=tr.epoch + 1))
    kw = train_args(args, epochs, args.name)
    if args.budget_min and not args.no_time_cap:
        kw["time"] = round(args.budget_min / 60.0, 3)
    model.train(**kw)
    minutes = (time.time() - t0) / 60
    info = {"planned_epochs": epochs, "epochs_run": done["epochs"], "train_minutes": round(minutes, 1),
            "time_cap_hours": kw.get("time"), "base": args.base, "batch": args.batch, "imgsz": 640,
            "threads": args.threads}
    (Path(args.work) / "train_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Continue an interrupted run from its last.pt (same settings, same schedule)."""
    YOLO = _yolo(args.threads)
    last = Path(args.work).resolve() / "runs" / args.name / "weights" / "last.pt"
    if not last.is_file():
        print(f"nothing to resume: {last} does not exist")
        return 1
    YOLO(str(last)).train(resume=True)
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


def _read_yolo(path: Path, w: int, h: int) -> List[Dict[str, Any]]:
    out = []
    for line in path.read_text().split("\n"):
        if line.strip():
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            out.append({"label": CLASSES[int(c)], "box": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]})
    return out


def _iou(p: List[float], q: List[float]) -> float:
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
    return inter / u if u > 0 else 0.0


def beta_recall(model: Any, data_yaml: Path, conf: float = 0.35, iou_match: float = 0.5) -> Dict[str, Any]:
    """What the runtime would find on the beta pages at its default threshold: every labeled region
    counts as found when a detection of the same class overlaps it with IoU >= 0.5. Reported for
    the 13 uncopyable files (scans, faxes, photo, screenshot) and the digital ones separately."""
    from PIL import Image
    root = data_yaml.parent
    manifest = json.loads((root / "manifest.json").read_text())
    groups: Dict[str, Dict[str, Any]] = {}
    per_class: Dict[str, List[int]] = {c: [0, 0] for c in CLASSES}
    misses, extras, files = [], [], {}
    for entry in manifest["beta"]:
        img_path = root / "images" / "beta" / f"{entry['name']}.jpg"
        w, h = Image.open(img_path).size
        gts = _read_yolo(root / "labels" / "beta" / f"{entry['name']}.txt", w, h)
        res = model.predict(str(img_path), conf=conf, iou=0.5, imgsz=640, device="cpu", verbose=False)[0]
        dets = [{"label": CLASSES[int(c)], "box": b, "conf": float(s)}
                for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())]
        used = set()
        found = 0
        for g in gts:
            best, bj = 0.0, None
            for j, d in enumerate(dets):
                if j not in used and d["label"] == g["label"]:
                    v = _iou(g["box"], d["box"])
                    if v > best:
                        best, bj = v, j
            ok = bj is not None and best >= iou_match
            per_class[g["label"]][1] += 1
            if ok:
                used.add(bj)
                found += 1
                per_class[g["label"]][0] += 1
            else:
                misses.append({"page": entry["name"], "label": g["label"], "best_iou": round(best, 2)})
        for j, d in enumerate(dets):
            if j not in used:
                extras.append({"page": entry["name"], "label": d["label"], "conf": round(d["conf"], 2)})
        group = "uncopyable" if entry["mode"] != "digital" else "digital"
        g = groups.setdefault(group, {"pages": 0, "regions": 0, "found": 0, "detections": 0, "extra": 0})
        g["pages"] += 1
        g["regions"] += len(gts)
        g["found"] += found
        g["detections"] += len(dets)
        g["extra"] += len(dets) - len(used)
        files.setdefault(entry["file"], {"mode": entry["mode"], "regions": 0, "found": 0})
        files[entry["file"]]["regions"] += len(gts)
        files[entry["file"]]["found"] += found
    for g in groups.values():
        g["recall"] = round(g["found"] / g["regions"], 4) if g["regions"] else None
        g["precision"] = round((g["detections"] - g["extra"]) / g["detections"], 4) if g["detections"] else None
    unc = {k: v for k, v in files.items() if v["mode"] != "digital"}
    return {"conf": conf, "iou_match": iou_match, "groups": groups,
            "uncopyable_files": len(unc), "uncopyable_files_all_found": sum(1 for v in unc.values() if v["found"] == v["regions"]),
            "per_class_recall": {c: (f"{a}/{n}" if n else "-") for c, (a, n) in per_class.items()},
            "misses": misses, "extra_detections": extras, "files": files}


def cmd_eval(args: argparse.Namespace) -> int:
    YOLO = _yolo(args.threads)
    model = YOLO(str(weights_path(args)))
    data = str(Path(args.data).resolve())
    out: Dict[str, Any] = {}
    for split, label in (("val", "synthetic_val"), ("test", "beta")):
        out[label] = _metrics(model, data, split, args.workers)
        out[label]["instances"] = _instances(Path(data), split)
    rec = beta_recall(model, Path(data))
    report = dict(out, beta_recall=rec, weights=str(weights_path(args)))
    name = args.metrics_name or "metrics.json"
    (Path(args.work) / name).write_text(json.dumps(report, indent=1))
    print_table(out)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("files", "extra_detections")}, indent=1))
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
    YOLO = _yolo(args.threads)
    src = weights_path(args)
    model = YOLO(str(src))
    onnx_path = Path(model.export(format="onnx", imgsz=640, opset=args.opset, simplify=True, dynamic=False,
                                  half=False, nms=False, device="cpu"))
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx_path, MODEL_OUT)
    print(f"{MODEL_OUT} {MODEL_OUT.stat().st_size / 1e6:.2f} MB (opset {args.opset}, from {src})")
    report = parity(model, Path(args.data).resolve().parent / "images" / "beta")
    report["onnx_bytes"] = MODEL_OUT.stat().st_size
    report["opset"] = args.opset
    report["weights"] = str(src)
    (Path(args.work) / "parity.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "images"}, indent=1))
    return 0 if report["ok"] else 1


def _match(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pair detections by class and IoU; report unmatched ones and the largest corner difference."""
    used, diffs, unmatched = set(), [], []
    for d in a:
        best, bj = 0.0, None
        for j, e in enumerate(b):
            if j not in used and e["label"] == d["label"]:
                v = _iou(d["box"], e["box"])
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
    end to end: each side does its own resizing from the file (Pillow vs OpenCV), the way both run.
    A detection close to the threshold can fall on either side of it end to end, so those are
    listed with their confidence rather than counted as failures."""
    import numpy as np
    import torch
    from PIL import Image
    os.environ["RFQ_LAYOUT_MODEL"] = str(MODEL_OUT)
    import layout
    layout.MODEL_PATH = str(MODEL_OUT)
    layout.reset()
    rows, worst_exact, worst_e2e, ok = [], 0.0, 0.0, True
    for path in sorted(images_dir.glob("*.jpg")):
        img = Image.open(path).convert("RGB")
        x, r, left, top = layout._letterbox(img)
        # exact: shared tensor, boxes in 640-space
        res = model.predict(torch.from_numpy(x.copy()), conf=conf, iou=iou, imgsz=640, device="cpu", verbose=False)[0]
        ul = [{"label": CLASSES[int(c)], "box": [float(v) for v in b], "conf": float(s)}
              for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())]
        sess = layout._session()
        ox = layout._decode(sess.run(None, {sess.get_inputs()[0].name: x})[0], conf, iou)
        exact = _match(ul, ox)
        conf_diff = 0.0
        for d in ul:
            cands = [e for e in ox if e["label"] == d["label"] and _iou(d["box"], e["box"]) > 0.9]
            if cands:
                conf_diff = max(conf_diff, min(abs(d["conf"] - e["conf"]) for e in cands))
        # end to end, boxes in image pixels
        res2 = model.predict(np.asarray(img)[:, :, ::-1].copy(), conf=conf, iou=iou, imgsz=640, device="cpu",
                             verbose=False)[0]
        ul2 = [{"label": CLASSES[int(c)], "box": [float(v) for v in b], "conf": float(s)}
               for b, c, s in zip(res2.boxes.xyxy.tolist(), res2.boxes.cls.tolist(), res2.boxes.conf.tolist())]
        ox2 = layout.detect(img, conf=conf, iou=iou)
        e2e = _match(ul2, ox2)
        scale = max(img.size) / 640.0
        exact_bad = bool(exact["unmatched_a"] or exact["unmatched_b"]) or exact["max_px"] > 1.0
        ok = ok and not exact_bad
        worst_exact = max(worst_exact, exact["max_px"])
        worst_e2e = max(worst_e2e, e2e["max_px"] / scale)
        rows.append({"image": path.name, "n_ultralytics": len(ul2), "n_onnxruntime": len(ox2),
                     "exact_max_px_640": round(exact["max_px"], 3), "exact_max_conf_diff": round(conf_diff, 4),
                     "exact_unmatched": len(exact["unmatched_a"]) + len(exact["unmatched_b"]),
                     "e2e_max_px_640": round(e2e["max_px"] / scale, 2),
                     "e2e_unmatched": [f"{d['label']} {d['conf']:.2f}" for d in e2e["unmatched_a"] + e2e["unmatched_b"]]})
    return {"ok": ok, "images": rows, "exact_worst_px_640": round(worst_exact, 3),
            "exact_worst_conf_diff": max((r["exact_max_conf_diff"] for r in rows), default=0.0),
            "e2e_worst_px_640": round(worst_e2e, 2),
            "e2e_images_with_unmatched": sum(1 for r in rows if r["e2e_unmatched"]), "count": len(rows)}


BENCH_CHILD = r"""
import json, os, resource, sys, time
sys.path.insert(0, sys.argv[1])
os.environ["RFQ_LAYOUT_THREADS"] = sys.argv[2]
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
import numpy, onnxruntime, PIL.Image
import layout
rss_imports = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t0 = time.perf_counter(); ok = layout.available()["ready"]; t_load = time.perf_counter() - t0
rss_loaded = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
pages = [open(p, "rb").read() for p in sys.argv[3:]]
layout.detect(pages[0])
times = []
for data in pages:
    t = time.perf_counter(); layout.detect(data); times.append(time.perf_counter() - t)
times.sort()
print(json.dumps({"ready": ok, "threads": int(sys.argv[2]), "pages": len(pages), "load_s": round(t_load, 3),
                  "detect_mean_s": round(sum(times) / len(times), 3), "detect_median_s": round(times[len(times) // 2], 3),
                  "detect_max_s": round(times[-1], 3),
                  "rss_mb": {"python": round(rss0 / 1024, 1), "after_imports": round(rss_imports / 1024, 1),
                             "after_model_load": round(rss_loaded / 1024, 1),
                             "peak": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}}))
"""


def cmd_bench(args: argparse.Namespace) -> int:
    """Runtime speed and memory of layout.detect on the beta pages (JPEG bytes, as the server passes
    them), in a fresh process per thread count so the memory numbers are the detector's alone."""
    pages = sorted(str(p) for p in (Path(args.data).resolve().parent / "images" / "beta").glob("*.jpg"))
    out = []
    for threads in args.bench_threads:
        proc = subprocess.run([sys.executable, "-c", BENCH_CHILD, str(ROOT), str(threads)] + pages,
                              capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
            return 1
        out.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        print(json.dumps(out[-1]))
    (Path(args.work) / "bench.json").write_text(json.dumps({"python": sys.version.split()[0], "runs": out}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["time", "train", "resume", "eval", "export", "bench"])
    ap.add_argument("--data", required=True, help="data.yaml written by make_layout_dataset.py")
    ap.add_argument("--work", default=str(DEFAULT_WORK), help="runs, weights and metrics (outside the repo)")
    ap.add_argument("--base", default="yolo11n.pt", help="starting weights: yolo11n.pt (COCO) or an earlier best.pt")
    ap.add_argument("--name", default=RUN_NAME, help="run folder under <work>/runs")
    ap.add_argument("--weights", help="trained weights for eval/export (default: the best.pt of the run)")
    ap.add_argument("--epochs", type=int, default=0, help="default: from the timing run and --budget-min")
    ap.add_argument("--budget-min", type=float, default=45.0, help="training budget in minutes")
    ap.add_argument("--no-time-cap", action="store_true", help="run exactly --epochs, ignoring the budget")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2, help="dataloader workers (Ultralytics uses 0 on CPU)")
    ap.add_argument("--threads", type=int, default=3, help="torch threads")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--metrics-name", help="file name for eval results in --work (default metrics.json)")
    ap.add_argument("--bench-threads", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()
    work = Path(args.work).resolve()
    if ROOT in work.parents or work == ROOT:
        ap.error("keep runs out of the repository")
    work.mkdir(parents=True, exist_ok=True)
    args.data = str(Path(args.data).resolve())
    if args.weights:
        args.weights = str(Path(args.weights).resolve())
    if args.base.endswith(".pt") and Path(args.base).exists():
        args.base = str(Path(args.base).resolve())
    os.chdir(work)  # ultralytics downloads the base weights into the working directory
    args.work = str(work)
    commands = {"time": cmd_time, "train": cmd_train, "resume": cmd_resume, "eval": cmd_eval,
                "export": cmd_export, "bench": cmd_bench}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
