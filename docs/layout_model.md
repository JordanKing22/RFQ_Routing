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
requirements are one column of lines. A line table reads row by row. So each region might suit a
different Tesseract page segmentation mode (`--psm 11` or `--psm 6` for title block cells, `--psm 4`
or `--psm 6` for notes and requirements), and `layout.crops(image, detections)` cuts each region out
with a small margin. That was measured and did not pay: `ocr.py` still reads whole pages with the
settings in `docs/ocr_settings.md`, and uses the boxes to say where each line was printed (see "How
the extractor uses it" below).

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
reduced scale (JPEG draft mode). Other pictures 2560 pixels or more across are box-averaged down by
a whole factor before any colour conversion. Pictures over 60 MP are refused from their header. A
PDF page that would pass 25 MP at the requested dpi renders at a lower dpi. The page size comes from
pdfinfo; without it, pdftoppm renders at most a 5000 x 5000 pixel area.

The onnxruntime session is created on first use, on `RFQ_LAYOUT_THREADS` threads (default 1), so
it behaves on Render's 0.1 CPU. `RFQ_LAYOUT_WORKERS` (default 1) limits how many detections run at
once. `RFQ_LAYOUT_MODEL` points at another ONNX file. Nothing raises: a broken file, a missing
model, or a missing package gives `[]`, and `available()` says why.

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
each glyph at its exact width. The first dataset built with the polished renderer, `layout_data_v2`,
still had this error, and so did the model trained on it (run 2 below). `layout_data_v3` is the same
dataset rebuilt with the fix: the same 1,400 specs, seeds, and page geometry, with corrected labels
in 977 of the 1,200 training files and 166 of the 200 validation files. The shipped model is
trained on `layout_data_v3`.

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
  found the font rounding error above. A last review looked at 30 more `layout_data_v3` pages (one
  of each document kind in each render mode, both second pages in the dataset, and a drawing with
  each legend type), cropped the tight spots at full size, and looked at all 24 beta pages again.
  It found no wrong label. All 30 pages were also rebuilt from their seeds, and their labels match
  the files on disk exactly.
- A leak check: every synthetic spec was regenerated from its seed (all 1,400 match the manifest),
  and none shares a part number, an RFQ or PO number, or a company and title pair (compared without
  regard to case) with a beta file. No synthetic drawing has both the company and most of the notes
  of a beta drawing. No training picture is a copy or a near copy of a beta page: the closest
  pairs by a 1,024-bit image hash are different documents. 96 synthetic pages do reuse a beta
  title under another company, because titles come from the shared vocabulary.

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

The shipped model used `layout_data_v3`: 1,200 training and 200 validation pages (seed 2609), built
in about 4.5 minutes on 3 cores. A rebuild gives the same specs, geometry, and labels, but not
byte-identical pictures: the scan, copier, fax, and photo effects use Pillow's noise generator,
which Python's seed does not control, so their speckle and grain differ from build to build.

| split | pages | drawings | RFQ forms | POs | other (no boxes) | regions |
| --- | --- | --- | --- | --- | --- | --- |
| train | 1,200 | 656 | 236 | 149 | 159 | 3,990 |
| val | 200 | 113 | 39 | 24 | 24 | 672 |
| beta (test) | 24 | 20 | 4 | 0 | 0 | 88 |

## Training

The model is YOLO11n (2.6 M parameters, 6.5 GFLOPs at 640), trained on CPU with Ultralytics 8.4.163
and torch 2.14, in three runs, each continuing from the one before:

1. From the COCO `yolo11n.pt` weights, 5 epochs on a first dataset (1,050 pages) made with the
   renderer as it was before `drawings.py` was polished.
2. A fine-tune of run 1's best weights on `layout_data_v2`, made with the polished renderer that
   also made the final beta files, but still with the font rounding error in its labels. Its epoch
   4 was the shipped model until the last review.
3. A fine-tune of run 2's epoch 4 on `layout_data_v3`, the same pages with corrected labels. Its
   epoch 6 is the shipped model.

| setting | value | why |
| --- | --- | --- |
| start | the previous run's best.pt | it already knew these page types; run 1's model scored mAP50 0.980 on the new beta pages before any fine-tuning |
| imgsz | 640 | the runtime letterboxes to 640, and CPU time grows with the square |
| batch | 16 | |
| optimizer | AdamW, lr 0.000833 (Ultralytics' automatic choice), 1 warmup epoch, linear decay | |
| augmentation | mosaic 1.0, switched off for the last 2 epochs; translate 0.1, scale 0.5, HSV; no flips, no rotation | mirrored text never happens, and the skew is already in the data |
| threads | 2 or 3 torch threads, data loaded in the main process, images cached in RAM | 4 cores shared with OCR jobs |
| budget | 45 minutes (run 2) or 40 minutes (run 3) of wall time (Ultralytics `time`), a checkpoint every epoch | |

The budget decided the length. Ultralytics re-plans the epoch count after every epoch so that the
run fits its `time` budget.

Run 2: 14 epochs were requested and re-planned down to 4, because the machine was shared (load
average 8 to 10 on 4 cores) and an epoch took 12 to 14 minutes. Epochs 3 and 4 ran without mosaic.
The first attempt at epoch 4 did not finish: the time cap stopped it after 19 of 75 batches, and
then the out-of-memory killer ended the process during its final validation. With images cached in
RAM, training peaks near 6.5 GB, and the machine's memory is shared with other jobs. Epoch 4 was
then run in full with `train_layout.py resume`, which continues from the epoch 3 checkpoint with the
same schedule. On a machine with less free memory, train with `--cache disk`.

| run 2 epoch | minutes | precision | recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| 1 | 11.6 | 0.953 | 0.950 | 0.974 | 0.817 |
| 2 | 13.0 | 0.975 | 0.956 | 0.986 | 0.870 |
| 3 | 14.4 | 0.976 | 0.977 | 0.985 | 0.878 |
| 4 (resumed) | 19.6 | 0.994 | 0.980 | 0.989 | 0.880 |

Run 3: 3 epochs were requested. On a quieter machine an epoch took 3 to 8 minutes, so Ultralytics
re-planned the run up to 8 epochs, with mosaic for the first 6. The process stopped after epoch 6
without its final step (its checkpoints were never stripped). The last review resumed it with
`train_layout.py resume`. The resumed run trained epoch 7 without mosaic, and then it stopped:
on a resume, Ultralytics re-plans from the time spent since the resume but counts every epoch done
before it, so it decided the budget was used up. Epoch 7 scored a little lower than epoch 6 on
synthetic validation (mAP50-95 0.927 against 0.932), so Ultralytics kept epoch 6 as best.pt. The
choice was made on synthetic validation only; epoch 7 was never scored on the beta pages.

| run 3 epoch | minutes | precision | recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| 1 | 8.3 | 0.952 | 0.968 | 0.974 | 0.846 |
| 2 | 5.4 | 0.995 | 0.987 | 0.991 | 0.889 |
| 3 | 4.7 | 0.986 | 0.991 | 0.992 | 0.887 |
| 4 | 4.3 | 0.994 | 0.985 | 0.993 | 0.911 |
| 5 | 3.2 | 0.996 | 0.981 | 0.989 | 0.916 |
| 6 (shipped) | 3.2 | 0.995 | 0.976 | 0.992 | 0.932 |
| 7 (resumed, no mosaic) | 8.1 | 0.996 | 0.996 | 0.994 | 0.927 |

These are synthetic validation numbers, run 2's on the `layout_data_v2` labels and run 3's on the
corrected `layout_data_v3` labels.

## Metrics

The shipped model (run 3, epoch 6) is compared with run 2's epoch 4, the model it replaced, on the
same data and the same corrected labels. "Beta" means the 24 held-out pages of the real beta files.
mAP comes from `model.val` at conf 0.001 and IoU 0.6.

| class | synthetic val mAP50 | synthetic val mAP50-95 | beta mAP50 | beta mAP50-95 | beta n | run 2, synthetic val mAP50-95 | run 2, beta mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| title_block | 0.995 | 0.995 | 0.995 | 0.995 | 20 | 0.995 | 0.995 |
| revision_block | 0.995 | 0.985 | 0.995 | 0.995 | 20 | 0.969 | 0.964 |
| notes | 0.995 | 0.964 | 0.995 | 0.984 | 20 | 0.907 | 0.941 |
| export_legend | 0.995 | 0.884 | 0.995 | 0.900 | 8 | 0.830 | 0.812 |
| proprietary_notice | 0.984 | 0.736 | 0.995 | 0.808 | 8 | 0.654 | 0.719 |
| form_header | 0.995 | 0.954 | 0.995 | 0.920 | 4 | 0.859 | 0.920 |
| line_table | 0.995 | 0.992 | 0.995 | 0.995 | 4 | 0.968 | 0.951 |
| requirements | 0.995 | 0.951 | 0.995 | 0.933 | 4 | 0.872 | 0.929 |
| all | 0.994 | 0.933 | 0.995 | 0.941 | 88 | 0.882 | 0.904 |

Across all classes, the shipped model's precision and recall are 0.983 and 0.988 on synthetic
validation, and 0.974 and 0.982 on beta. Run 2 scored mAP50 0.991 on synthetic validation and
0.995 on beta. mAP50 was already at its ceiling, so the gain from the corrected labels is in how
tightly the boxes fit: mAP50-95 went from 0.882 to 0.933 on synthetic validation and from 0.904 to
0.941 on beta. There are only 4 beta forms and 8 beta legends, so one box moves those beta numbers
by a few hundredths.

The table below is what the viewer actually gets. It runs through the server's own path
(`attachments.page_image`: pdftoppm at 150 dpi, fit to 1800 pixels, JPEG) and then `layout.detect`
with onnxruntime, at the default conf 0.35. A labeled region counts as found when a detection of the
same class overlaps it with IoU 0.5 or better.

| pages | regions found | false detections | run 2 |
| --- | --- | --- | --- |
| 13 uncopyable files (7 scans, 2 copier, 2 fax, photo, screenshot) | 48 of 48 | 0 | 47 of 48, 0 false |
| 11 digital pages | 40 of 40 | 0 | 40 of 40, 0 false |
| 200 synthetic validation pages (as saved) | 669 of 672 | 0 | 667 of 672, 1 false |

Every class is complete on the beta pages: title_block, revision_block, and notes 20/20;
export_legend 8/8 (the ITAR boxes on E61 and E72, and the CUI banners and designation blocks on E05
and E52); proprietary_notice 8/8; form_header, line_table, and requirements 4/4. The matched boxes
overlap the truth with a median IoU of 0.97. The loosest is the faint one-line proprietary footer on
the E62 copier-scanned RFQ form, at IoU 0.77. That footer was run 2's one miss (it scored 0.15
there).

The photo, the screenshot, and the faxes, looked at down to conf 0.05:

- E22, the phone photo: all 3 regions at IoU 0.97 to 0.99, and nothing on the desk around the page.
- E71, the screenshot: all 4 regions, and nothing on the viewer's title bar, menu, or grey margin.
  The proprietary notice is the loosest box there (IoU 0.86): it is two lines of 5.6 pt grey text
  drawn at about 120 dpi.
- E07 and E52, the faxes: all 9 regions at IoU 0.96 or better. On these four files and the two
  copier scans, the only candidate between 0.05 and the 0.35 threshold is a form_header at 0.05
  spread over the drawing views of E07.

The 3 synthetic validation misses are all proprietary notices (two on phone photos, one on a
digital RFQ form). Each one is found in the right place (IoU 0.80 to 0.82) but scores 0.24 to 0.31,
under the threshold: a single faint grey line at the foot of a page is the hardest thing on it.

Export check, onnxruntime (`layout.py`) against Ultralytics 8.4.163:

- Letterbox: `layout.py` computes the same scale, resized size, and padding as Ultralytics'
  `LetterBox` class for 3,020 picture sizes: portrait and landscape pages, 4:3 and 3:4 photos, phone
  pictures, screenshots from 1280 x 800 to 2560 x 1440, strips, and 3,000 random sizes. On 20 of them
  the padded area in the real tensors is identical pixel for pixel.
- Fed the same letterboxed tensor, both give identical detections on 84 pages (the 24 beta pages
  and 60 validation pages) at conf 0.35 and at 0.05: the same boxes to 0.0002 pixel and the same
  confidences. This checks the export, the decoder, and the class-wise NMS (Ultralytics offsets
  each class's boxes so that they never overlap; `layout.py` runs NMS class by class; both drop a
  box whose IoU with a kept box is above the threshold and keep one exactly at it).
- When each side resizes the file itself (Pillow in `layout.py`, OpenCV in Ultralytics), both find
  the same regions on 83 of the 84 pages. On the other one, a proprietary notice scores 0.40 in
  `layout.py` and 0.25 in Ultralytics. Box corners agree to 0.4 pixel at 640 in the median, and to
  7 pixels on the beta pages. The worst case is 21 pixels, at the end of one notes box (the two
  boxes still overlap at IoU 0.93). The difference is the resize filter: Pillow's bilinear filter
  averages over the whole shrink, OpenCV's samples 2 x 2 pixels.

## Speed and memory

The model file is 10.6 MB (10,610,252 bytes): fp32, ONNX opset 17, simplified with onnxslim, input
`images` [1, 3, 640, 640], output [1, 12, 8400]. It has the same architecture as run 2's model, so
it runs at the same speed. Timings are for onnxruntime 1.30 on a 2.1 GHz Xeon:

| what | one thread | two threads |
| --- | --- | --- |
| load the model (first use) | 0.05 s | 0.05 s |
| detect, the 24 beta pages as saved (at most 1280 px JPEG), machine quiet | 0.091 s mean, 0.115 s max | 0.065 s mean, 0.083 s max |
| the same, other jobs running (load average 3.5 to 4) | 0.095 s mean, 0.108 s max | 0.075 s mean, 0.109 s max |
| detect, the server's page images (1800 px JPEG), machine quiet, wall time | 0.095 s mean, 0.117 s max | |
| detect, the server's page images, machine quiet, CPU time | 0.094 s mean, 0.117 s max | |

While other jobs loaded the machine harder (load average 6 to 9), the same one-thread detection took
0.09 to 0.16 s per page.

Memory, as resident set size in a fresh process: 14 MB for Python, 47 MB after importing numpy,
onnxruntime, and Pillow, 81 MB after loading the model, and a 177 MB peak while detecting. The
server hands `detect` its page images, at most 1800 pixels on a side, and the peak above is for
pictures of that size. A big picture given to `detect` directly costs more while it is decoded. For
example, a 36 MP opaque RGBA PNG adds about 180 MB, and a 49 MP 16-bit greyscale scan adds about 290
MB. Pictures over 60 MP are refused from their header, before anything is decoded. The last review
cut these two from about 280 and 520 MB: such pictures are now shrunk before their colour
conversion.

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
../yolo-venv/bin/python tools/make_layout_dataset.py --out ../yolo_work/layout_data_v3 --train 1200 --val 200 --workers 3

# 2. train: fine-tune earlier weights (as shipped: run 2's epoch 4), or start from COCO with --base yolo11n.pt
../yolo-venv/bin/python tools/train_layout.py train --data ../yolo_work/layout_data_v3/data.yaml --base ../yolo_work/final/rfq_layout_v2_e4_best.pt --name rfq_layout_v3 --epochs 3 --budget-min 40
../yolo-venv/bin/python tools/train_layout.py resume --data ../yolo_work/layout_data_v3/data.yaml --name rfq_layout_v3   # after an interrupt

# 3. per-class mAP on synthetic val and beta, plus beta recall at the runtime threshold
../yolo-venv/bin/python tools/train_layout.py eval --data ../yolo_work/layout_data_v3/data.yaml --name rfq_layout_v3

# 4. export to models/rfq_layout.onnx and check onnxruntime against Ultralytics
../yolo-venv/bin/python tools/train_layout.py export --data ../yolo_work/layout_data_v3/data.yaml --name rfq_layout_v3 --opset 17

# 5. runtime only (the demo's virtualenv is enough): recall through the server's own path, speed, memory, tests
../venv/bin/python tools/train_layout.py runtime --data ../yolo_work/layout_data_v3/data.yaml --beta-only
../venv/bin/python tools/train_layout.py bench --data ../yolo_work/layout_data_v3/data.yaml
../venv/bin/python -m unittest tests.test_layout -v
```

Useful flags for `train`:

- `--threads` (default 3) and `--workers` (default 2): how much CPU training takes.
- `--cache ram|disk|none`: where decoded images are kept. `ram` peaks near 6.5 GB.
- `--no-time-cap`: run exactly `--epochs` and ignore `--budget-min`.

`tools/train_layout.py time` runs one epoch with the real settings and suggests an epoch count for
`--budget-min`, for when `--epochs` is left out.

A resumed run keeps the time budget, but Ultralytics counts the epochs done before the resume against
the time spent after it, so a resume usually ends after one epoch (see run 3). To finish a schedule
exactly, train again from the checkpoint with `--no-time-cap` and the number of epochs left.
`runtime` and `bench` stop with an error when the model does not load (set `RFQ_LAYOUT_MODEL` to try
another ONNX file), instead of scoring every page 0.

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
- **The beta photo is an easy photo for this model.** A third of the landscape training photos go
  through the exact pipeline that made E22 (same desk, lighting, and blur, with other corners), so
  its perfect score says little about real phone pictures.
- **`proprietary_notice` is the weakest class.** Its small, faint lines have the lowest mAP50-95,
  and its box ends are loose. The extractor should read a missing proprietary box as "not
  detected", not as "not there".
- **Boxes are axis-aligned.** On a skewed scan or a keystoned photo, a box is the rectangle around
  the tilted region, so a crop can include a sliver of a neighbor.
- **One page at a time.** Multi-page PDFs are detected page by page with
  `detect_pdf_page(data, page=N)`. The viewer shows page 1.
- **Second pages are rare in training.** Only 2 of the 1,400 synthetic pages are page 2 of a form
  (most forms fit on one page), and none of the beta pages is. A requirements list that runs over
  to the top of page 2 is labeled, but the detector has hardly seen one.
- **Only the 8 classes are boxed.** A parts list, a flag-note box ("ALL WELDS PER AWS D1.1"), or a
  views callout is never boxed.

## How the extractor uses it

The detector is paired with the OCR in `ocr.py` and read by `rfq_details.py`. The measurements,
before and after, are in `docs/ocr_settings.md`, "Regions".

- **Every OCR'd page is detected too.** After Tesseract reads a page, `ocr.py` runs
  `layout.detect` (conf 0.35) on the same picture Tesseract read: the cleaned page, after an
  upscale, after a phone photo's sheet is flattened, after a sideways page is turned. A region box
  and a line box therefore share pixels; both are saved in the page picture's pixels at the source
  resolution (for a phone photo, the flattened sheet's). On the 13 uncopyable files that picture
  gives the same regions as the viewer's page image: every labeled region found, no false box.
- **What is saved.** The `ocr.file_text` result gets `"regions": [{"page", "label", "conf",
  "box"}]`, every line gets the `"region"` its center falls in (the smallest box when two overlap),
  and `settings["layout"]` names the model. `data/rfq_beta/ocr_cache.json` holds them, and its
  header says which model made them; `python ocr.py --build-cache` has to run where the detector
  works for the committed cache to carry them.
- **Sources.** `rfq_details.py` finds the line each value was read from and names its region:
  `CI-10442_RevC.pdf, title block (OCR 88%)`, `RFQ-26-0931.pdf, line table (OCR 93%)`,
  `RFQ-26-0931.pdf, form header (OCR 93%)`, and for export control the legend box the marking was
  printed in, `CI-10442_RevC.pdf, export legend (OCR 88%)`, which the CSV's "Export control found
  in" column carries too. 85 of the 89 OCR values in the beta records name a region; the other 4
  are TERMS lines, which no class boxes. Each file in a record lists the regions found on it.
- **Reading rules.** Where the regions fixed fields without losing any (`REGION_READING` in
  `rfq_details.py`): revision table rows come only from the revision block (a proprietary notice
  that OCR ran into the title block's date cell once read as revision "OR"); the REV label and
  its letter inside the title block make the rev even when the drawing number beside them is
  unreadable; pieces of one title block value that OCR returned apart on one baseline are joined;
  and a REV cell read as "Ce" is C. Rules that changed nothing (reading title block, form header,
  and line table fields from their regions first, matching export legends loosely only inside the
  legend box) were left out, and so was one that lost a field: the `title_block` box also covers
  the tolerance block, so "inside the title block" does not make a number a drawing number
  ("Y14.5-2018" read as "YI4S-2018" became a part line).
- **Region crops read again: not used.** A second Tesseract reading of the title block (`--psm 6`
  or `--psm 11`, at 600 dpi or at its own resolution) and of the line table, merged by
  confidence, found 2 or 3 more key fields on 55 held-out pages but no field the extractor did not
  already get, for 12 to 29 % more CPU; the variant that replaced the page's words lost 9 real
  keys. The code stays, off (the recipe setting `regions`, or `python ocr.py --evaluate --regions
  title_block:6:600`), so the comparison can be rerun.
- **Without the detector nothing changes.** No numpy, onnxruntime, or model file, or
  `RFQ_OCR_REGIONS=0`: OCR results carry no region keys and are what they were before, older
  cache entries load as they are, sources read `CI-10442_RevC.pdf (OCR 88%)`, and the reading
  rules have nothing to act on.
- **Cost.** About 0.1 CPU second a page, only for pages that are OCR'd (text layers, STEP files,
  and the committed cache cost nothing), so about a second a page on Render's 0.1 CPU; the model
  session is the one the viewer overlay already loads.
- **The viewer overlay is separate.** `attachments.page_regions` still detects on the 150 dpi page
  image it shows, in that image's pixels. The regions in the OCR results are in the OCR raster's
  pixels and are not sent to the browser.

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
    takes pixel values from 0 to 255, not 0 to 1, in OpenCV's BGR channel order;
  - give `layout.py` a different output decoder: YOLOX's raw output is grid offsets plus an
    objectness score, which have to be turned into boxes and multiplied into the class scores before
    the same non-maximum suppression.
- This is a summary, not legal advice.
