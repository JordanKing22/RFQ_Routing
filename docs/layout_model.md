# Page layout detector (YOLO)

`layout.py` finds the parts of an RFQ page that matter, so Tesseract can read each one with
settings that suit it and the extractor knows where every piece of text came from. The viewer's
"Detected regions" button draws these boxes. The model is `models/rfq_layout.onnx`: a YOLO11n
detector fine-tuned on this demo's own drawings, RFQ forms, and purchase orders. At runtime it needs
only numpy, onnxruntime, and Pillow (plus `pdftoppm` for PDF pages). It never needs torch or
ultralytics.

## What it detects, and why

| id | class | what it covers | why the extractor wants it |
| --- | --- | --- | --- |
| 0 | `title_block` | the drawing title block: company, TITLE, MATERIAL, FINISH, SIZE, DWG NO., REV, DRAWN, DATE, SCALE, SHEET cells, and the tolerance block beside them | part number, rev, material, finish, and description come from here and win over the email |
| 1 | `revision_block` | the REVISIONS table | the latest rev and what changed |
| 2 | `notes` | NOTES: and its numbered lines | finish, heat treat, marking, and inspection requirements |
| 3 | `export_legend` | an ITAR or EAR warning box, the CUI banners at the top and bottom, the CUI designation block | routes the email to the restricted lane, and the box shows where the marking was printed |
| 4 | `proprietary_notice` | the PROPRIETARY AND CONFIDENTIAL lines | not export control; kept apart so it is never mistaken for one |
| 5 | `form_header` | RFQ or PO header: company, document title, RFQ NO. / PO NUMBER, DATE, RESPOND BY cells | RFQ number and respond-by date |
| 6 | `line_table` | the line-item table on RFQ forms and POs | one row per part: part number, rev, material, quantities |
| 7 | `requirements` | numbered quote requirements (RFQ forms), notes and quality clauses (POs) | certs, FAI, packaging, delivery |

The regions hold different kinds of text. A title block is sparse text in boxed cells. Notes and
requirements are one column of lines. A line table reads row by row. So each region should suit a
different Tesseract page segmentation mode (`--psm 11` or `--psm 6` for title block cells, `--psm 4`
or `--psm 6` for notes and requirements), and `layout.crops(image, detections)` cuts each region out
with a small margin, ready for `ocr.py`. These per-region settings have not been measured yet. Today
`ocr.py` reads whole pages with the settings in `docs/ocr_settings.md`, and only the viewer uses the
boxes (see "Not wired in yet" below).

## Runtime API

```python
import layout
layout.CLASSES                    # the 8 labels above, in model order
layout.available()                # {"ready": bool, "error": str | None, "model_bytes": ..., "threads": 1, ...}
layout.detect(image, conf=0.35, iou=0.5)
    # image: a PIL image or image bytes (PNG, JPEG, TIFF, ...). Returns
    # [{"label", "conf", "box": [x0, y0, x1, y1]}] in the image's own pixels, highest confidence first.
layout.detect_pdf_page(data, page=1, dpi=150)   # the same for one PDF page, rasterized by pdftoppm
layout.render_pdf_page(data, page=1, dpi=150)   # the picture detect_pdf_page sees
layout.crops(image, detections, pad=6)          # [(detection, PIL image)] for OCR
```

```
python layout.py FILE [--page N] [--dpi 150] [--conf 0.35] [--save out.png] [--json]
```

How a page is processed:

1. The picture is letterboxed to 640 x 640: scaled to fit with the aspect ratio kept, then padded
   with grey. The sizes, padding, and rounding are the same as Ultralytics uses.
2. One onnxruntime call returns 8400 candidate boxes, each with 8 class scores.
3. numpy keeps the candidates scoring above `conf` and runs non-maximum suppression separately for
   each class, so a notes box never suppresses a legend box.

Big pictures are made small early, because the detector only needs 640 pixels. JPEGs decode at a
reduced scale (JPEG draft mode), and other pictures 2560 pixels or more across are box-averaged
down by a whole factor. The
onnxruntime session is created on first use, on `RFQ_LAYOUT_THREADS` threads (default 1), so it
behaves on Render's 0.1 CPU. `RFQ_LAYOUT_WORKERS` (default 1) limits how many detections run at
once. Nothing raises: a broken file, a missing model, or a missing package gives `[]`, and
`available()` says why.

## How the data and labels are made

Nobody draws boxes by hand. The demo builds every page as drawing operations on a `docgen.Page`
(`drawings.drawing_pages`, `attachments.pages_for`), and `tools/make_layout_dataset.py` derives each
box from those operations. Each box is anchored on the printed labels, not on fixed coordinates:

- `title_block`: the rectangles holding TITLE, MATERIAL, FINISH, DWG NO., SCALE, and SHEET, plus
  the heavy frame around them and the tolerance block beside them. Only a tolerance block level
  with the cells counts, because a drawing note can also begin with "UNLESS OTHERWISE SPECIFIED:".
- `revision_block`: the REVISIONS heading cell and every table row that grows down from it.
- `notes`: NOTES: and the numbered lines that follow in the same column.
- `export_legend`: the WARNING: paragraph and its frame, the "CUI. CONTROLLED BY" paragraph and its
  frame, and each bold CUI banner.
- `proprietary_notice`: the PROPRIETARY AND CONFIDENTIAL paragraph.
- `form_header`: the REQUEST FOR QUOTATION or PURCHASE ORDER title, the number, date, and
  respond-by cells, and the company lines beside them.
- `line_table`: the dark PART NUMBER heading strip and the rows below it.
- `requirements`: QUOTE REQUIREMENTS or PURCHASE ORDER NOTES AND QUALITY CLAUSES and the numbered
  lines. The box stops at TERMS: and continues at the top of the next page when the list runs over.

A text line's box is its measured Helvetica width at the size the text really has in the picture.
`docgen.to_image` asks Pillow for a whole number of pixels per font size (a 7 pt line drawn at
150 dpi becomes a 15 px font, 2.9% wider than 7 pt), so the dataset builder applies the same
rounding for the resolution each page is drawn at. Before this was done, the ends of long note and
requirement lines were off by up to 4 to 5% of the line width, cut short on some pages ("SHIPMEN|T.")
and padded on others. The text PDFs among the beta files need no rounding, because pdftoppm places
each glyph at its exact width.

The boxes follow the renderer's own operations, so they stay right when `drawings.py` changes. They
were checked four ways on the polished renderer:

- A count check: every drawing has exactly one title block, revision block, and notes region, and
  each marking has the expected legends.
- A geometric check over 1,500 random specs: every note line, requirement, part number, header cell,
  and legend line lies inside its box, and boxes of different classes do not overlap. This check
  found one real bug, the tolerance-block rule above, which is fixed.
- By eye, with the boxes drawn: on 20 random pages and all 24 beta pages, then again in review on
  27 more pages (every document kind in every render mode, every legend type, and page 2 of
  multi-page forms), on all 24 beta pages, and on the line ends of six text-heavy pages. The review
  found the font rounding error above.
- A leak check: every synthetic spec was regenerated from its seed (all 1,400 match the manifest),
  and none shares a part number, an RFQ number, or a company and title pair with a beta file.

Each synthetic page is a random but realistic spec. Companies, titles, materials, finishes, notes,
requirements, and terms are recombined from `data/sample_emails.json`, with new part numbers. Each
page goes through the same effects that made the beta inbox (`tools/make_rfq_beta.py`):

| mode | share | what it does |
| --- | --- | --- |
| digital | 25% | the clean page at 72 to 200 dpi, color or gray |
| scan | 25% | office scanner: 200 to 300 dpi gray, blur, speckle, skew up to 1.6 degrees, JPEG quality 50 to 80 |
| copier | 13% | 150 to 200 dpi, darker, edge shadow, skew up to 2.2 degrees |
| fax | 14% | 150 to 200 dpi, 1 bit, salt noise, skew up to 1.2 degrees |
| photo | 12% | a phone photo on a desk: keystone, uneven light, blur, noise (a third of landscape photos use the exact beta photo pipeline) |
| screen | 11% | a PDF viewer screenshot at 96 to 144 dpi |

The boxes go through the same geometry as the pixels. They are scaled by the dpi, rotated about the
page center with the skew, pushed through the photo's perspective map, and shifted by the screenshot
frame. A rotated or warped box becomes the axis-aligned box around its four corners.

The page kinds are 55% drawings, 20% RFQ forms, 12% POs, and 13% other documents with no boxes at
all (letters, brochures, resumes, invoices, packing slips), so the detector learns to find nothing
on them. Pictures are saved at most 1280 pixels on the long side.

The honest test set is the 24 pages of the 30 real beta files; the 6 STEP files have no pages. 13
of the 24 pages are uncopyable. They are never rendered into training: their part numbers and
company/title pairs are excluded from the random specs. Their boxes are derived from their specs and
mapped onto the real files on disk, following the scans' skew and the photo's corners.

The shipped model used `layout_data_v2`: 1,200 training and 200 validation pages (seed 2609), built
in 4.5 minutes on 3 cores.

| split | pages | drawings | RFQ forms | POs | other (no boxes) | regions |
| --- | --- | --- | --- | --- | --- | --- |
| train | 1,200 | 656 | 236 | 149 | 159 | 3,990 |
| val | 200 | 113 | 39 | 24 | 24 | 672 |
| beta (test) | 24 | 20 | 4 | 0 | 0 | 88 |

## Training

The model is YOLO11n (2.6 M parameters, 6.5 GFLOPs at 640), trained on CPU with Ultralytics 8.4.163
and torch 2.14, in two runs:

1. From the COCO `yolo11n.pt` weights, 5 epochs on a first dataset (1,050 pages) made with the
   renderer as it was before `drawings.py` was polished.
2. A fine-tune of run 1's best weights on `layout_data_v2`, made with the polished renderer that
   also made the final beta files. This run's epoch 4 is the shipped model.

| setting | value | why |
| --- | --- | --- |
| start | run 1's best.pt | it already knew these page types; on the new beta pages it scored mAP50 0.980 before any fine-tuning |
| imgsz | 640 | the runtime letterboxes to 640, and CPU time grows with the square |
| batch | 16 | |
| optimizer | AdamW, lr 0.000833 (Ultralytics' automatic choice), 1 warmup epoch, linear decay | |
| augmentation | mosaic 1.0, switched off for the last 2 epochs; translate 0.1, scale 0.5, HSV; no flips, no rotation | mirrored text never happens, and the skew is already in the data |
| threads | 3 torch threads, data loaded in the main process, images cached in RAM | 4 cores shared with OCR jobs |
| budget | 45 minutes wall time (Ultralytics `time`), a checkpoint every epoch | |

The budget decided the length. After the first epoch, Ultralytics re-planned the 14 requested
epochs down to 4, because the machine was shared (load average 8 to 10 on 4 cores) and an epoch took
12 to 14 minutes. Epochs 3 and 4 ran without mosaic.

The first attempt at epoch 4 did not finish. The time cap stopped it after 19 of 75 batches, and
then the out-of-memory killer ended the process during its final validation. With images cached in
RAM, training peaks near 6.5 GB, and the machine's memory is shared with other jobs. Epoch 4 was
then run in full with `train_layout.py resume`, which continues from the epoch 3 checkpoint with the
same schedule. On a machine with less free memory, train with `--cache disk`.

| epoch | minutes | precision | recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| 1 | 11.6 | 0.953 | 0.950 | 0.974 | 0.817 |
| 2 | 13.0 | 0.975 | 0.956 | 0.986 | 0.870 |
| 3 | 14.4 | 0.976 | 0.977 | 0.985 | 0.878 |
| 4 (resumed) | 19.6 | 0.994 | 0.980 | 0.989 | 0.880 |

These are synthetic validation numbers. Both epoch 3 and epoch 4 were evaluated on the beta pages,
and epoch 4 is better there: mAP50-95 0.903 against 0.880, precision 0.984 against 0.962, and
recall 0.985 against 0.984. So the shipped model is epoch 4, which is also Ultralytics' best.pt for the run.

## Metrics

The shipped model (epoch 4) is compared with two others on the same data:

- epoch 3 of the same run;
- run 1's model, the provisional model trained on the old renderer's pages.

"Beta" means the 24 held-out pages of the real beta files. mAP comes from `model.val` at conf 0.001
and IoU 0.6.

| class | synthetic val mAP50 | synthetic val mAP50-95 | beta mAP50 | beta mAP50-95 | beta n | epoch 3, beta mAP50-95 | old model, beta mAP50 / mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| title_block | 0.995 | 0.995 | 0.995 | 0.995 | 20 | 0.995 | 0.995 / 0.986 |
| revision_block | 0.995 | 0.965 | 0.995 | 0.964 | 20 | 0.988 | 0.995 / 0.828 |
| notes | 0.995 | 0.900 | 0.995 | 0.936 | 20 | 0.952 | 0.995 / 0.862 |
| export_legend | 0.995 | 0.845 | 0.995 | 0.812 | 8 | 0.761 | 0.995 / 0.705 |
| proprietary_notice | 0.957 | 0.634 | 0.995 | 0.719 | 8 | 0.647 | 0.875 / 0.266 |
| form_header | 0.995 | 0.867 | 0.995 | 0.920 | 4 | 0.899 | 0.995 / 0.920 |
| line_table | 0.995 | 0.968 | 0.995 | 0.951 | 4 | 0.964 | 0.995 / 0.970 |
| requirements | 0.995 | 0.871 | 0.995 | 0.928 | 4 | 0.833 | 0.995 / 0.937 |
| all | 0.990 | 0.880 | 0.995 | 0.903 | 88 | 0.880 | 0.980 / 0.809 |

Across all classes, the shipped model's precision and recall are 0.994 and 0.980 on synthetic
validation, and 0.984 and 0.985 on beta. The old model scored mAP50 0.975 and mAP50-95 0.809 on
synthetic validation. There are only 4 beta forms and 8 beta legends, so one box moves those beta
numbers by a few hundredths.

The table below is what the viewer actually gets. It runs through the server's own path
(`attachments.page_image`: pdftoppm at 150 dpi, fit to 1800 pixels, JPEG) and then `layout.detect`
with onnxruntime, at the default conf 0.35. A labeled region counts as found when a detection of the
same class overlaps it with IoU 0.5 or better.

| pages | regions found | false detections | old model |
| --- | --- | --- | --- |
| 13 uncopyable files (7 scans, 2 copier, 2 fax, photo, screenshot) | 47 of 48 (0.98); every region found on 12 of the 13 files | 0 | 45 of 48 |
| 11 digital pages | 40 of 40 | 0 | 39 of 40 |

The per-class results:

- title_block 20/20, revision_block 20/20, notes 20/20.
- export_legend 8/8: the ITAR boxes on E61 and E72, and the CUI banners and designation blocks on
  E05 and E52.
- proprietary_notice 7/8.
- form_header 4/4, line_table 4/4, requirements 4/4.

The one miss is the faint one-line proprietary footer on the E62 copier-scanned RFQ form. The model
scores it 0.15 there (with IoU 0.82), below the 0.35 threshold, and the old model missed it too. The matched boxes
overlap the truth with a median IoU of 0.96 (lowest 0.81).

Export check, onnxruntime (`layout.py`) against Ultralytics on the 24 beta pages:

- Fed the same letterboxed tensor, both give identical detections: the same boxes to 0.001 pixel
  and the same confidences to 0.0001. This checks the export, the decoder, and the NMS.
- When each side resizes the file itself (Pillow in `layout.py`, OpenCV in Ultralytics), both find
  the same regions on every page, and box corners agree within 6.1 pixels at 640 (E52, the fax).

## Speed and memory

The model file is 10.6 MB (10,610,286 bytes): fp32, ONNX opset 17, simplified with onnxslim, input
`images` [1, 3, 640, 640], output [1, 12, 8400]. Timings are for onnxruntime 1.30 on a 2.1 GHz Xeon
with the machine otherwise quiet:

| what | one thread | two threads |
| --- | --- | --- |
| load the model (first use) | 0.05 s | 0.05 s |
| detect, the 24 beta pages as saved (at most 1280 px JPEG) | 0.091 s mean, 0.115 s max | 0.065 s mean, 0.083 s max |
| detect, the server's page images (1800 px JPEG), wall time | 0.095 s mean, 0.117 s max | |
| detect, the server's page images, CPU time | 0.094 s mean, 0.117 s max | |

While other jobs loaded the machine (load average 6 to 9), the same one-thread detection took 0.09
to 0.16 s per page.

Memory, as resident set size in a fresh process: 14 MB for Python, 47 MB after importing numpy,
onnxruntime, and Pillow, 82 MB after loading the model, and a 177 MB peak while detecting.

On Render's free plan (about 0.1 CPU), 0.1 CPU seconds per page means roughly 1 second per page.
The server remembers the regions of each page it has shown.

## Retrain

Training uses a separate virtualenv with torch and ultralytics. The demo never needs it. Run these
from the repository root. Datasets, runs, and metrics stay in `../yolo_work`, outside the
repository.

```
python -m venv ../yolo-venv && ../yolo-venv/bin/pip install torch ultralytics onnx onnxslim onnxruntime

# 1. look at a few pages with their boxes, then build the dataset (about 4.5 minutes on 3 cores)
../yolo-venv/bin/python tools/make_layout_dataset.py --preview ../yolo_work/preview --count 20
../yolo-venv/bin/python tools/make_layout_dataset.py --out ../yolo_work/layout_data_v2 --train 1200 --val 200 --workers 3

# 2. train: fine-tune earlier weights (as shipped), or start from COCO with --base yolo11n.pt
../yolo-venv/bin/python tools/train_layout.py train --data ../yolo_work/layout_data_v2/data.yaml --base ../yolo_work/base_old/best_old_renderer.pt --name rfq_layout_v2 --epochs 14 --budget-min 45
../yolo-venv/bin/python tools/train_layout.py resume --data ../yolo_work/layout_data_v2/data.yaml --name rfq_layout_v2   # after an interrupt

# 3. per-class mAP on synthetic val and beta, plus beta recall at the runtime threshold
../yolo-venv/bin/python tools/train_layout.py eval --data ../yolo_work/layout_data_v2/data.yaml --name rfq_layout_v2

# 4. export to models/rfq_layout.onnx and check onnxruntime against Ultralytics
../yolo-venv/bin/python tools/train_layout.py export --data ../yolo_work/layout_data_v2/data.yaml --name rfq_layout_v2 --opset 17

# 5. runtime only (the demo's virtualenv is enough): recall through the server's own path, speed, memory, tests
../venv/bin/python tools/train_layout.py runtime --data ../yolo_work/layout_data_v2/data.yaml --beta-only
../venv/bin/python tools/train_layout.py bench --data ../yolo_work/layout_data_v2/data.yaml
../venv/bin/python -m unittest tests.test_layout -v
```

Useful flags for `train`:

- `--threads` (default 3) and `--workers` (default 2): how much CPU training takes.
- `--cache ram|disk|none`: where decoded images are kept. `ram` peaks near 6.5 GB.
- `--no-time-cap`: run exactly `--epochs` and ignore `--budget-min`.

`tools/train_layout.py time` runs one epoch with the real settings and suggests an epoch count for
`--budget-min`, for when `--epochs` is left out.

## Limitations

- **It learned this renderer's layouts.** Every training page came from `drawings.py` and
  `attachments.py`, which have one title block design, one RFQ form, and one PO. The scan, fax,
  photo, and screenshot effects make the detector robust to how a page was sent, not to what the
  page looks like. Real customer drawings from SolidWorks, Inventor, NX, Creo, or AutoCAD templates,
  and real customer RFQ forms, put these regions in other places, with other borders and fonts.
  Before trusting the detector on real mail, label a few hundred real pages (any YOLO labeling tool
  works) and fine-tune on them mixed with the synthetic pages.
- **The beta set is small.** It has 88 regions, and only 4 RFQ forms and 8 legends. It shows the
  detector works on these files, not a precise error rate.
- **`proprietary_notice` is the weakest class.** Its small, faint lines have the lowest mAP50-95,
  and its box ends are loose. The extractor should read a missing proprietary box as "not
  detected", not as "not there".
- **Boxes are axis-aligned.** On a skewed scan or a keystoned photo, a box is the rectangle around
  the tilted region, so a crop can include a sliver of a neighbor.
- **One page at a time.** Multi-page PDFs are detected page by page with
  `detect_pdf_page(data, page=N)`. The viewer shows page 1.
- **Only the 8 classes are boxed.** A parts list, a flag-note box ("ALL WELDS PER AWS D1.1"), or a
  views callout is never boxed.

## Not wired in yet

The viewer's "Detected regions" overlay uses the detector (`attachments.page_regions`). Two things
remain:

- `ocr.py` does not yet OCR region by region with `layout.crops`.
- `rfq_details.py` does not yet record which region a value came from.

Both need the OCR line boxes, which are in the OCR raster's pixels, mapped into the page picture's
pixels and matched to the region that contains each line.

## Licensing, in plain words

- Ultralytics YOLO is licensed under AGPL-3.0. That covers the training code and the COCO-trained
  YOLO11 weights the model started from. Ultralytics' position is that models trained with it, and
  their exports, fall under AGPL-3.0 too unless you hold an Ultralytics Enterprise License, and the
  ONNX file's own metadata says "AGPL-3.0 License". So treat `models/rfq_layout.onnx` as AGPL-3.0.
- AGPL-3.0 has a network clause (section 13): people who use the software over a network, as they do
  on the Render deployment, must be offered its source code under AGPL-3.0. If the model counts as
  part of the service, that means the source of the whole service. The runtime contains no
  Ultralytics code (`layout.py` is this repository's own numpy decoder), but that does not remove
  Ultralytics' claim on the weights. For a private demo shown to a few people the practical risk is
  low. For a product, either buy the Enterprise License or retrain with a permissively licensed
  detector.
- The training pages are synthetic, drawn by this repository's own renderer, so no third-party
  images or labels are involved.
- onnxruntime, which runs the model, is MIT licensed. numpy is BSD-3-Clause (a few bundled parts use
  other permissive licenses such as MIT and Zlib), and Pillow uses the permissive MIT-CMU license
  (older releases listed it as HPND). The runtime side adds no obligations beyond keeping their
  notices.
- YOLOX (Megvii) is licensed under Apache-2.0 and is a permissive alternative in the same size
  class (YOLOX-Nano or YOLOX-Tiny). Switching would take four changes:
  - convert the same dataset's labels to COCO JSON and train with YOLOX's own code;
  - export the trained model to ONNX;
  - change the preprocessing in `layout.py`: YOLOX pads at the top left, not in the center, and
    takes pixel values from 0 to 255, not 0 to 1;
  - give `layout.py` a different output decoder: YOLOX's raw output is grid offsets plus an
    objectness score, which have to be turned into boxes and multiplied into the class scores before
    the same non-maximum suppression.
- This is a summary, not legal advice.
