# Page layout detector (YOLO)

`layout.py` finds the regions of an RFQ page that matter, so Tesseract can read each one with
settings that suit it and the extractor knows where every piece of text came from. The model is
`models/rfq_layout.onnx`, a YOLO11n detector fine-tuned on this demo's own drawings, RFQ forms, and
purchase orders. At runtime it needs numpy, onnxruntime, and Pillow (plus `pdftoppm` for PDF pages),
never torch or ultralytics.

Status: DRAFT written while training runs. The numbers below are for the provisional model now in
`models/` (the first run, 5 epochs on the old renderer's pages) measured on the NEW pages. They are
replaced when the fine-tuned model is exported.

## What it detects, and why

| id | class | what it covers | why the extractor wants it |
| --- | --- | --- | --- |
| 0 | `title_block` | the drawing title block: company, TITLE, MATERIAL, FINISH, SIZE, DWG NO., REV, DRAWN, DATE, SCALE, SHEET cells and the tolerance block beside them | part number, rev, material, finish, and description come from here and win over the email |
| 1 | `revision_block` | the REVISIONS table | the latest rev and what changed |
| 2 | `notes` | NOTES: and its numbered lines | finish, heat treat, marking, and inspection requirements |
| 3 | `export_legend` | an ITAR or EAR warning box, the CUI banners at the top and bottom, the CUI designation block | routes the email to the restricted lane, and the region says where the marking was printed |
| 4 | `proprietary_notice` | the PROPRIETARY AND CONFIDENTIAL lines | not export control; kept apart so it is never mistaken for one |
| 5 | `form_header` | RFQ or PO header: company, document title, RFQ NO. / PO NUMBER, DATE, RESPOND BY cells | RFQ number and respond-by date |
| 6 | `line_table` | the line-item table on RFQ forms and POs | one row per part: part number, rev, material, quantities |
| 7 | `requirements` | numbered quote requirements (RFQ forms), notes and quality clauses (POs) | certs, FAI, packaging, delivery |

Each region type reads best with a different Tesseract page segmentation mode: a title block is
sparse text in boxes (`--psm 11` or `--psm 6` per cell), notes and requirements are a single column
of lines (`--psm 4` or `--psm 6`), and a line table reads row by row. `layout.crops(image,
detections)` cuts each region out with a small margin, ready for `ocr.py`. The per-region settings
above are starting points; `docs/ocr_settings.md` holds the measured page-level settings.

## Runtime API

```python
import layout
layout.CLASSES                    # the 8 labels above, in model order
layout.available()                # {"ready": bool, "error": str | None, "model_bytes": ..., "threads": 1, ...}
layout.detect(image, conf=0.35, iou=0.5)
    # image: a PIL image or PNG/JPEG bytes. Returns [{"label", "conf", "box": [x0, y0, x1, y1]}]
    # in the image's own pixels, highest confidence first.
layout.detect_pdf_page(data, page=1, dpi=150)   # same, for one PDF page rasterized by pdftoppm
layout.render_pdf_page(data, page=1, dpi=150)   # the picture detect_pdf_page sees
layout.crops(image, detections, pad=6)          # [(detection, PIL image)] for OCR
```

```
python layout.py FILE [--page N] [--dpi 150] [--conf 0.35] [--save out.png] [--json]
```

How it runs: the picture is letterboxed to 640 x 640 (aspect kept, grey padding, the same
arithmetic as Ultralytics), one onnxruntime call returns 8400 candidate boxes with 8 class scores,
and numpy keeps the ones above `conf` and applies class-wise non-maximum suppression (a notes box
never suppresses a legend box). Big JPEGs are decoded at reduced scale (JPEG draft mode), since
the detector only needs 640 pixels. The onnxruntime session is created on first use with
`RFQ_LAYOUT_THREADS` threads (default 1) so it behaves on Render's 0.1 CPU; `RFQ_LAYOUT_WORKERS`
(default 1) limits how many detections run at once. Nothing raises: a broken file, a missing
model, or a missing package gives `[]`, and `available()` says why.

## How the data and labels are made

Nobody draws boxes by hand. The demo renders every page as drawing operations on a
`docgen.Page` (`drawings.drawing_pages`, `attachments.pages_for`), so
`tools/make_layout_dataset.py` derives each box from those operations, anchored on the printed
labels rather than fixed coordinates:

- `title_block`: the rectangles that hold TITLE, MATERIAL, FINISH, DWG NO., SCALE, SHEET, plus the
  tolerance block beside them (only the one level with the cells: a drawing note can also begin with
  "UNLESS OTHERWISE SPECIFIED:") and the heavy frame around them.
- `revision_block`: the REVISIONS heading cell and every table row that grows down from it.
- `notes`: NOTES: and the numbered lines that follow in the same column.
- `export_legend`: the WARNING: paragraph and its frame, the "CUI. CONTROLLED BY" paragraph and its
  frame, and each bold CUI banner.
- `proprietary_notice`: the PROPRIETARY AND CONFIDENTIAL paragraph.
- `form_header`: the REQUEST FOR QUOTATION or PURCHASE ORDER title, the number, date, and
  respond-by cells, and the company lines beside them.
- `line_table`: the dark PART NUMBER heading strip and the rows below it.
- `requirements`: QUOTE REQUIREMENTS or PURCHASE ORDER NOTES AND QUALITY CLAUSES and the numbered
  lines, stopping at TERMS:, continued at the top of the next page when the list runs over.

Because the boxes follow the renderer's own operations, they stay right when `drawings.py`
changes. They were checked three ways on the polished renderer: a count check (every drawing has
exactly one title block, revision block, and notes; the expected legends per marking), a geometric
check over 1,500 random specs (every note line, requirement, part number, header cell, and legend
line lies inside its box; boxes of different classes do not overlap), and by eye on 20 random pages
and all 24 beta pages with the boxes drawn. The geometric check found one real bug, the tolerance
block rule above, which is fixed.

Each synthetic page is a random but realistic spec (companies, titles, materials, finishes, notes,
requirements, and terms recombined from `data/sample_emails.json`, with new part numbers) rendered
through the same effects that made the beta inbox (`tools/make_rfq_beta.py`):

| mode | share | what it does |
| --- | --- | --- |
| digital | 25% | the clean page at 72 to 200 dpi, color or gray |
| scan | 25% | office scanner: 200 to 300 dpi gray, blur, speckle, skew up to 1.6 degrees, JPEG quality 50 to 80 |
| copier | 13% | 150 to 200 dpi, darker, edge shadow, skew up to 2.2 degrees |
| fax | 14% | 150 to 200 dpi, 1 bit, salt noise, skew up to 1.2 degrees |
| photo | 12% | a phone photo on a desk: keystone, uneven light, blur, noise (a third of landscape photos use the exact beta photo pipeline) |
| screen | 11% | a PDF viewer screenshot at 96 to 144 dpi |

The boxes go through the same geometry as the pixels: scaled by the dpi, rotated about the page
center with the skew, pushed through the photo's perspective map, shifted by the screenshot frame;
a rotated or warped box becomes the axis-aligned box around its four corners. Page kinds: 55%
drawings, 20% RFQ forms, 12% POs, 13% other documents (letters, brochures, resumes, invoices,
packing slips) with no boxes at all, so the detector learns to say nothing. Pictures are saved at
most 1280 pixels on the long side.

The honest test set is the 24 pages of the 30 real beta files (the 6 STEP files have no pages),
13 of them uncopyable. They are never rendered into training: their part numbers and
company/title pairs are excluded from the random specs, and their boxes are derived from their
specs and mapped onto the real files on disk (the scans' skew, the photo's corners).

Dataset used here: 1,200 training and 200 validation pages (seed 2609), built in about 4.5 minutes
on 3 cores.

## Training

YOLO11n (2.6 M parameters, 6.5 GFLOPs at 640), fine-tuned on CPU with Ultralytics 8.4.163 and
torch 2.14:

| setting | value | why |
| --- | --- | --- |
| start | the first run's best.pt (5 epochs from COCO `yolo11n.pt` on the old renderer's pages) | it already knows these page types; on the new pages it scored mAP50 0.98 before any fine-tuning |
| imgsz | 640 | the runtime letterboxes to 640; CPU time grows with the square |
| batch | 16 | |
| optimizer | AdamW, lr 0.000833 (Ultralytics' automatic choice for short runs), 1 warmup epoch | |
| augmentation | mosaic 1.0 (off for the last 2 epochs), translate 0.1, scale 0.5, HSV, no flips, no rotation | mirrored text never happens; skew is already in the data |
| threads | 3 torch threads, data loaded in the main process | 4 shared cores |
| budget | 45 minutes wall time (Ultralytics `time`), checkpoint every epoch | |

## Metrics

Provisional model (see status above). mAP from `model.val` at conf 0.001, IoU 0.6:

| class | synthetic val mAP50 | synthetic val mAP50-95 | beta mAP50 | beta mAP50-95 | beta n |
| --- | --- | --- | --- | --- | --- |
| title_block | 0.995 | 0.994 | 0.995 | 0.986 | 20 |
| revision_block | 0.995 | 0.907 | 0.995 | 0.828 | 20 |
| notes | 0.995 | 0.866 | 0.995 | 0.862 | 20 |
| export_legend | 0.961 | 0.717 | 0.995 | 0.705 | 8 |
| proprietary_notice | 0.869 | 0.316 | 0.875 | 0.266 | 8 |
| form_header | 0.995 | 0.889 | 0.995 | 0.920 | 4 |
| line_table | 0.995 | 0.960 | 0.995 | 0.970 | 4 |
| requirements | 0.995 | 0.822 | 0.995 | 0.937 | 4 |
| all | 0.975 | 0.809 | 0.980 | 0.809 | 88 |

Recall at the runtime default (conf 0.35, a region counts as found at IoU 0.5 or better with the
same class), through the server's own path (`attachments.page_image`, then `layout.detect` with
onnxruntime): uncopyable files 45 of 48 regions (0.94), digital files 39 of 40, no false detections.

## Speed and memory

One thread, onnxruntime 1.30, the 24 beta pages as JPEG bytes: about 0.15 s per page (measured
while the machine was busy with other work).

## Retrain

Training uses a separate virtualenv with torch and ultralytics; the demo never needs it. Datasets
and runs stay outside the repository.

```
python -m venv ../yolo-venv && ../yolo-venv/bin/pip install torch ultralytics onnx onnxslim onnxruntime
../yolo-venv/bin/python tools/make_layout_dataset.py --preview ../yolo_work/preview --count 20   # look first
../yolo-venv/bin/python tools/make_layout_dataset.py --out ../yolo_work/layout_data_v2 --train 1200 --val 200 --workers 3
../yolo-venv/bin/python tools/train_layout.py train  --data ../yolo_work/layout_data_v2/data.yaml --base <earlier best.pt or yolo11n.pt> --name rfq_layout_v2 --epochs 14 --budget-min 45
../yolo-venv/bin/python tools/train_layout.py resume --data ../yolo_work/layout_data_v2/data.yaml --name rfq_layout_v2   # after an interrupt
../yolo-venv/bin/python tools/train_layout.py eval   --data ../yolo_work/layout_data_v2/data.yaml --name rfq_layout_v2
../yolo-venv/bin/python tools/train_layout.py export --data ../yolo_work/layout_data_v2/data.yaml --name rfq_layout_v2 --opset 17
../venv/bin/python tools/train_layout.py bench --data ../yolo_work/layout_data_v2/data.yaml   # runtime only
../venv/bin/python -m unittest tests.test_layout -v
```

## Limitations

- It learned this renderer's layouts. Every training page came from `drawings.py` and
  `attachments.py`, with one title block design, one RFQ form, one PO. The scan, fax, photo, and
  screenshot effects make it robust to how a page was sent, not to what it looks like. Real
  customer drawings from SolidWorks, Inventor, NX, Creo, or AutoCAD templates, and real customer RFQ
  forms, put these regions in other places with other borders and fonts. Before trusting it on real
  mail, label a few hundred real pages (any YOLO labeling tool works) and fine-tune on them mixed
  with the synthetic pages.
- The small, faint `proprietary_notice` lines are the weakest class (lowest mAP50-95); the
  extractor should treat a missing proprietary box as "not detected", not as "not there".
- A box is axis-aligned. On a skewed scan or a keystoned photo it is the rectangle around the
  tilted region, so a crop can include a sliver of a neighbor.
- One page at a time. Multi-page PDFs are detected page by page (`detect_pdf_page(data, page=N)`).
- A region outside the 8 classes (a parts list, a flag note box, a views callout) is never boxed.

## Licensing, in plain words

- Ultralytics YOLO (the training code and the YOLO11 weights the model started from) is licensed
  under AGPL-3.0. Ultralytics says models trained with it fall under the same license unless you
  buy an Ultralytics Enterprise license. So treat `models/rfq_layout.onnx` as AGPL-3.0: if the demo
  is offered to others over a network (the Render deployment), AGPL expects the complete source of
  the service to be available to its users under AGPL. For a private demo this is usually fine; for
  a product, either buy the Enterprise license or retrain with a permissively licensed detector.
- onnxruntime, which runs the model, is MIT licensed. numpy is BSD and Pillow uses the permissive
  HPND license, so the runtime side adds no obligations beyond keeping their notices.
- YOLOX (Megvii, Apache-2.0) is a permissive alternative of the same size class (YOLOX-Nano or
  YOLOX-Tiny). The same dataset works after converting the labels to COCO JSON, it exports to ONNX,
  and `layout.py` would need only a different output decoder (YOLOX adds an objectness score and
  decodes grid offsets itself).
- This is a summary, not legal advice.
