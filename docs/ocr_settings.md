# OCR settings for uncopyable RFQ files

`ocr.py` reads the 13 beta files that have no text layer (office scans, copier scans, faxes, a
phone photo, a viewer screenshot) with Tesseract. This page says how the settings in `ocr.py`
(`RECIPES` for the committed files, `FAST_RECIPES` for live uploads) were chosen: what was
tried, the numbers, what won and why, what it costs in time, and where it stops working.

Rerun everything here with:

    python ocr.py --evaluate                          # baseline, before, best, fast on the 13 files
    python ocr.py --evaluate --heldout                # the 25 held-out pages the search never saw
    python ocr.py --evaluate --search --type scan     # the settings search for one kind of file
    python ocr.py --evaluate --settings best --model /path/to/tessdata_best

## The chosen settings

The pipeline decides the source type from the file itself, never from its name: a 1-bit page
is a fax (`bilevel`), a gray page of 250 dpi or more is an office scan (`scan`), a lower one is
a copier scan (`lowres`), and a picture showing a sheet of paper on a background is a phone
photo (`photo`), or a screenshot (`screen`) when the sheet is square to the picture. Every
recipe runs Tesseract with `--oem 1` (the LSTM engine; the model Debian ships has no legacy
data) and `-l eng`.

| type | how it is detected | `RECIPES` (effort "best", the committed cache) | `FAST_RECIPES` (effort "fast", live uploads) |
| --- | --- | --- | --- |
| `scan` | gray, 250 dpi or more | native resolution, `invert`, `--psm 3` then `--psm 11` merged (`conf`), `repair`, `min_conf` 60, `--dpi` | the same with `--psm 3` only |
| `lowres` | gray, under 250 dpi | native resolution, upscale to 400 dpi, `invert` then `flatten`, `--psm 4`, `repair`, `--dpi` | the same (already one pass) |
| `bilevel` | 1 bit per pixel | native resolution (no resize), `--psm 11` then `--psm 3` merged (`conf`), `repair`, `--dpi` | `--psm 3` only |
| `photo` | a sheet on a background, not square to the picture | find the sheet and warp it flat at 300 dpi, `--psm 11` then `--psm 3` merged (`conf`), `repair`, `min_conf` 60, `--dpi` | the same with `--psm 3` only |
| `screen` | a sheet square to the picture (a viewer) | upscale to 300 dpi (whole screenshot), `--psm 11`, `repair`, `min_conf` 60, `--dpi` | the same (already one pass) |

Why, from the tables further down (keys are real files plus the 6 "tune" drawings of that kind):

- **scan**: `invert` made the white-on-dark header rows of the RFQ forms readable (44 to 46 of
  49 real keys), `repair` fixed "Cl-10442" and "TYPE Ill" (48 of 49; with the twin-capital rule
  below, 49 of 49), `min_conf` 60 dropped
  junk from hatching and line work (word precision 0.886 to 0.950 with the same keys). The
  second pass (`--psm 11`, merged by confidence) found the same keys and raised word recall on
  every one of the 13 pages (F1 0.915 to 0.931) for 1.8 times the CPU. Rasterizing at 400 dpi
  instead of the scan's own 300 lost keys (43 of 49) and cost 40 % more; `deskew`,
  `autocontrast`, `unsharp`, `median`, and both other thresholding methods each lost keys.
- **copier** (`lowres`): the upscale (7 of 14 real keys without it, 11 at 300 dpi, 10 at 400 dpi
  but with more held-out keys) and `flatten`, which removes the edge shadow (13 of 14, and 22
  of 24 held-out), did most of the work; `--psm 4` read the form's table rows better than 3
  (F1 up, same keys), `repair` found one more held-out key. Sauvola (`threshold=2`), which won
  on the 2 real copier scans alone in round 1, matched the keys but read fewer words for 20 %
  more time; adaptive Otsu lost 6 held-out keys. `invert` was added in the review (below).
- **fax** (`bilevel`): upscaling a 1-bit page to 300 dpi, the starting point, lost keys (9 of 10
  real, 15 of 24 held-out) against no resize (10 of 10, 19 of 24), which is also faster. Two
  passes, sparse text first (`--psm 11`) and the page layout pass filling in, found 22 of 24
  held-out keys. The 3x3 median filter meant for the salt noise also eats the thin strokes of a
  1-bit 200 dpi page and halved the key fields (5 of 10); Tesseract reads through the salt
  without it.
- **photo**: finding the sheet and flattening its perspective is worth 6 held-out keys (15
  against 21 of 24); the two passes 11+3 read the title block cells that one pass misses (22 of
  24), `repair` and `min_conf` complete it (23 of 24, precision 0.686 to 0.880). Adaptive Otsu took
  13 CPU seconds a page and lost keys.
- **screen**: the upscale to 300 dpi matters most (no upscale: 17 of 24 held-out keys and F1
  0.72); `--psm 11` beat 3 and 4 on the real screenshot (4 of 4 against 3 of 4). Cropping to the
  page (`page`) gave cleaner text (F1 0.898 against 0.866) but lost 2 held-out keys, so the whole
  screenshot is read. The search's last step moved to `--psm 12` for +0.003 F1 at 22 % more
  time, which is within the noise, so `--psm 11` was kept (the extraction agent also found full
  field accuracy on the screenshot with it).

One change came after the search. The misses it left were mostly rev letters, and the real
one on the office scans, E61's rev C, had a simple cause: the big C alone in the title block's
REV cell comes back as "Cc" (the capital and its lowercase twin, at a confidence of about 57),
which then also falls under `min_conf`. `repair` now turns a word that is one capital read
twice ("Cc", "Oo", for the ten letters whose lowercase is a smaller capital) into that
capital. It found the E61 rev (79 to 80 of 81 real keys) and 2 more held-out keys (a rev C on
the scan and on the screenshot of FR-3101), and raised word F1 slightly on E01 and E16 (the
same doubling on other big lone capitals) with no file worse.

A second change came in the review (see "Review" at the end). The last field missed on the
real files was E62's rev A, in the first row of the copier form's table, the same miss `invert`
had fixed on the office scan forms. On the copier the search had tried `invert` and it did
nothing, because `flatten` ran first: it takes the dark header band for shadowed paper, lifts
it to light gray, and leaves no band to find. `invert` now runs before `flatten`, and the
copier recipe has it: E62 reads its first row (81 of 81 real keys), and the 11 held-out copier
drawings, which have no dark bands, read exactly as before. The before-and-after, held-out,
time, model, and version tables below are the final run with both changes; the search tables
were made before them.

Settings that never helped anywhere: `-c preserve_interword_spaces=1` and the word lists off
(`load_system_dawg=0`, `load_freq_dawg=0`) gave exactly the same words as the same settings
without them on all five kinds (the TSV output lists words one by one, so kept spaces do not
show, and the LSTM model hardly leans on the word lists); `autocontrast`, `unsharp`, and
`color` never gained a key and lost some on most kinds; `deskew` lost keys on the scans,
copier, and fax and changed nothing on the photo and screenshot, for 0.4 to 1.1 CPU seconds a page (the skews here
are 1.4 degrees at most, which Tesseract follows by itself); `--psm 6` (one uniform block)
breaks tables and title blocks apart; `--psm 12` lost keys everywhere but the screenshot.

## Before and after, per file

`python ocr.py --evaluate` on the 13 real uncopyable files, Tesseract 5.3.4, one file at a time.
"baseline" is plain Tesseract on a 300 dpi rendering (`--psm 3`, no clean-up); "before" is the
pipeline as it stood before the search (native resolution, 300 dpi upscale for low-resolution
sources, page finding for the photo and screenshot, `--psm 3`); "best" and "fast" are the
chosen `RECIPES` and `FAST_RECIPES`. CPU seconds are Tesseract, pdftoppm, and the Python
clean-up together, per page.

| file | kind | keys, baseline | keys, before | keys, best | keys, fast | word recall / precision, before | word recall / precision, best | CPU s/page, best | CPU s/page, fast | still missed (best) |
|---|---|---|---|---|---|---|---|---|---|---|
| E01/QA-41127_RevB.pdf | scan | 4/4 | 4/4 | 4/4 | 4/4 | 0.898 / 0.877 | 0.933 / 0.944 | 3.01 | 1.61 | - |
| E16/AW-310_frame_assy.pdf | scan | 4/4 | 4/4 | 4/4 | 4/4 | 0.864 / 0.941 | 0.876 / 0.987 | 3.07 | 1.72 | - |
| E20/FR-3102.pdf | scan | 4/4 | 4/4 | 4/4 | 4/4 | 0.874 / 0.702 | 0.913 / 0.879 | 2.63 | 1.46 | - |
| E32/RFQ-26-0931.pdf | scan | 10/11 | 10/11 | 11/11 | 11/11 | 0.930 / 0.958 | 0.982 / 0.988 | 2.34 | 1.46 | - |
| E56/RFQ-26-0317.pdf | scan | 10/11 | 10/11 | 11/11 | 11/11 | 0.937 / 0.973 | 0.948 / 0.968 | 2.61 | 1.46 | - |
| E61/CI-10442_RevC.pdf | scan | 4/6 | 4/6 | 6/6 | 6/6 | 0.909 / 0.742 | 0.937 / 0.878 | 3.80 | 2.09 | - |
| E72/WS-RFQ-26-0388.pdf | scan | 8/9 | 8/9 | 9/9 | 9/9 | 0.966 / 0.990 | 0.980 / 0.990 | 2.56 | 1.45 | - |
| E09/HPV-2045_manifold_RevD.pdf | copier | 3/4 | 3/4 | 4/4 | 4/4 | 0.924 / 0.885 | 0.934 / 0.724 | 2.66 | 2.68 | - |
| E62/RFQ-26-0318.pdf | copier | 9/10 | 8/10 | 10/10 | 10/10 | 0.819 / 0.921 | 0.924 / 0.908 | 1.89 | 1.88 | - |
| E07/FR-2290_heatsink.pdf | fax | 3/4 | 4/4 | 4/4 | 4/4 | 0.842 / 0.938 | 0.944 / 0.923 | 1.83 | 0.99 | - |
| E52/AGI-3052_RevA.pdf | fax | 6/6 | 5/6 | 6/6 | 6/6 | 0.862 / 0.918 | 0.886 / 0.845 | 2.13 | 1.13 | - |
| E22/OPM-22817_RevB_photo.jpg | photo | 2/4 | 3/4 | 4/4 | 3/4 | 0.872 / 0.661 | 0.909 / 0.796 | 3.41 | 2.06 | - |
| E71/LO-1186_RevA_screenshot.png | screen | 4/4 | 3/4 | 4/4 | 4/4 | 0.911 / 0.955 | 0.871 / 0.917 | 1.97 | 1.99 | - |

| kind | files | keys, baseline | keys, before | keys, best | keys, fast | F1, baseline | F1, before | F1, best | F1, fast | CPU s/page, baseline | CPU s/page, before | CPU s/page, best | CPU s/page, fast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scan | 7 | 44/49 | 44/49 | 49/49 | 49/49 | 0.894 | 0.894 | 0.942 | 0.933 | 1.67 | 1.62 | 2.86 | 1.61 |
| copier | 2 | 12/14 | 11/14 | 14/14 | 14/14 | 0.808 | 0.885 | 0.866 | 0.866 | 1.98 | 1.57 | 2.27 | 2.28 |
| fax | 2 | 9/10 | 9/10 | 10/10 | 10/10 | 0.887 | 0.888 | 0.899 | 0.871 | 1.69 | 1.41 | 1.98 | 1.06 |
| photo | 1 | 2/4 | 3/4 | 4/4 | 3/4 | 0.820 | 0.752 | 0.849 | 0.830 | 1.05 | 2.10 | 3.41 | 2.06 |
| screen | 1 | 4/4 | 3/4 | 4/4 | 4/4 | 0.787 | 0.932 | 0.893 | 0.893 | 0.84 | 2.05 | 1.97 | 1.99 |
| ALL | 13 | 71/81 | 70/81 | 81/81 | 80/81 | 0.866 | 0.884 | 0.913 | 0.902 | 1.61 | 1.65 | 2.61 | 1.69 |

Key fields found: 71 of 81 with plain Tesseract, 70 before the search, 81 with the chosen
settings, 80 with the fast ones (the fast photo recipe misses the E22 material line). What
changed, file by file (traced in the search logs):

- E32, E56 (RFQ forms) and E72 (ITAR RFQ form): the rev letter of each part line, in a table
  under a dark header band, is read once `invert` turns the band light. E32's finish "HARD
  ANODIZE TYPE III CLASS 1" then needs `repair` ("TYPE Ill").
- E61 (ITAR drawing): `repair` turns "Cl-10442" into "CI-10442", so the part number matches,
  and the rev letter C, read as "Cc" alone in the title block's REV cell, becomes "C".
- E09 (copier): the material line needs the 400 dpi upscale and the finish line, under the
  edge shadow, needs `flatten`. E62 (copier form): the finish row comes back at 400 dpi, and
  the first table row (item 1, BWM-3105 rev A) once `invert` turns the header band light before
  `flatten` runs. The form's small "RFQ NO. / DATE / RESPOND BY" labels then read as junk (the
  values under them still read), which is why E62's word recall is 0.924 against 0.942 without
  `invert`.
- E52 (CUI fax): "CONTROLLED TECHNICAL INFORMATION" is read once the 1-bit page is not upscaled.
- E22 (phone photo): the material line of the title block comes from the sparse text pass
  (11+3). The date "2025-06-17", which the restart notes had as "ZOZS-06-17", reads correctly.
- E71 (screenshot): an I in the finish note read as l (Helvetica draws them alike); `repair`
  fixes it.

Word recall rose on 12 of 13 files (all but the screenshot). Precision fell on the E09 copier
scan (0.885 to 0.724: `flatten` also brings up the drawing's hatching, which reads as junk), on
the E52 fax (0.918 to 0.845: the sparse text pass adds words from line work, and the fax recipe
keeps low-confidence words because dropping them lost a field), and a little on E07 and the
screenshot, where the viewer's toolbar is now read with the page.

## The held-out check

`python ocr.py --evaluate --heldout`: the 25 "check" pages (5 drawings the search never saw,
each through the five effects), scored with the final settings. These pages carry random noise
that changes from run to run (see "Review"); this table is the sample that command gives in a
fresh process, and three more samples are in the review.

| kind | pages | keys, before | keys, best | keys, fast | word F1, before | word F1, best | word F1, fast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scan | 5 | 16/20 | 19/20 | 19/20 | 0.856 | 0.921 | 0.890 |
| copier | 5 | 15/20 | 14/20 | 14/20 | 0.823 | 0.823 | 0.823 |
| fax | 5 | 13/20 | 16/20 | 15/20 | 0.871 | 0.864 | 0.854 |
| photo | 5 | 16/20 | 16/20 | 17/20 | 0.795 | 0.878 | 0.820 |
| screen | 5 | 16/20 | 18/20 | 18/20 | 0.864 | 0.880 | 0.880 |
| all | 25 | 76/100 | 83/100 | 83/100 | 0.842 | 0.873 | 0.853 |

The gain carries over for scans, faxes, and screenshots. The copier recipe is one field worse
than before on this sample (it gains the TO-5520 finish but loses the KF-3408 material and the
BWM-3106 part number), after being 5 better on the tune pages and 3 better on the real copier
scans; on three more noise samples it was 1 better, 2 better, and even, so on drawings the
copier recipe is about as good as before, and its gain is on the real copier scans. Most
fields still missed here are long notes that wrap onto a second line in the drawing's notes or
title block (".002 THK. MASK PORT THREADS.", "MARKERS: TANTALUM PER ASTM F560"): the
harness wants the whole value in one run of text or one chained table cell, so a reading that
has every word but breaks the line elsewhere still counts as a miss. The rest are rev letters
alone in a small title block cell (FR-3101 C on the copier page, BWM-3106 A on the copier page
and the screenshot) that were not read beside their drawing number.

## Time: here and on Render

Measured as CPU time with `OMP_THREAD_LIMIT=1` (Tesseract's threads only fight each other on a
fraction of a CPU), on this machine's shared cores while other jobs ran. CPU time barely moves
with the load: the search measured 2.83 s a page for the scan recipe, the final run 2.86 s.
Seconds per page, from the per-file table above:

| effort | mean CPU s/page here | range over the 13 files | expected on Render free (0.1 CPU) | all 13 files on Render |
| --- | --- | --- | --- | --- |
| best (`RECIPES`) | 2.61 | 1.83 (E07 fax) to 3.80 (E61 scan) | about 26 s a page, 18 to 38 s | about 6 minutes |
| fast (`FAST_RECIPES`) | 1.69 | 0.99 (E07 fax) to 2.68 (E09 copier) | about 17 s a page, 10 to 27 s | about 4 minutes |
| baseline (plain `--psm 3`) | 1.61 | 0.84 (screenshot) to 2.56 (E09 copier) | about 16 s a page | about 3.5 minutes |

On the free plan a container gets about a tenth of one CPU, so wall time is about ten times
the CPU time (more if the host's cores are slower than these). Two things keep that off the
demo's path:

- **The committed cache.** `data/rfq_beta/ocr_cache.json` holds the result for all 30 beta
  files, keyed by the SHA-256 of the file bytes: 13 OCR results, 11 PDF text layers, and 6 STEP
  headers, with the Tesseract version and the settings they were made with. `server.py` loads
  it through `OCR_SEED_FILE`, finds every beta file in it at start, and runs neither Tesseract
  nor pypdf for them. Render only runs OCR for a file that is not in the cache: a live upload,
  or a beta file whose bytes changed. Rebuild the cache after changing a file or a recipe:
  `python ocr.py --build-cache` (65 to 100 s here, depending on the load).
- **Uploads use `FAST_RECIPES`** (`RFQ_UPLOAD_OCR_EFFORT=fast` in `attachments.py`): one
  Tesseract pass per page, 80 of 81 real keys against 81 for best, about a third less time.
  `RFQ_OCR_WORKERS` (default 1) keeps a second upload waiting instead of halving the speed of
  both, and `RFQ_OCR_TIMEOUT` (default 300 s a file) leaves room for a slow page.

Memory: Tesseract peaks at about 90 MB on a 400 dpi letter page and at about 190 MB on the
largest picture it is ever given (36 million pixels, see "Very large pages" below), inside the
free plan's 512 MB.

## How it was measured

`python ocr.py --evaluate` is the only code that reads `tests/rfq_beta_truth.json`. For every
uncopyable file it scores the OCR text against the words printed on the file:

- **Key-field recall** (the number that matters): does the text contain each part number, rev,
  material, finish, quantity list, RFQ number, respond-by date, and export legend phrase of the
  file's spec, allowing only whitespace and case differences. A rev counts when it follows its
  part number ("KF-3408 B", "KF-3408 REV B") or is the next cell to the right of the drawing
  number in the title block. A table cell that wraps ("AL 6061-T6511 / HARD" over "ANODIZE TYPE
  III" over "CLASS 1") counts when its lines, joined with spaces, contain the value.
- **Word recall and precision**: multiset overlap of words after uppercasing and stripping
  punctuation at word ends. Precision falls when drawing line work is read as junk words.
- **CPU seconds per page**: tesseract, pdftoppm, and the Python clean-up, measured as CPU time
  (`getrusage`) with `OMP_THREAD_LIMIT=1`, so it does not change when other programs share the
  machine and it predicts a host with a fraction of a CPU. Wall-clock seconds are printed too.

Two sets of pages were scored:

- **The 13 real uncopyable beta files**: 7 office scans (300 dpi gray JPEG, skew up to 1.1
  degrees, speckle), 2 copier scans (200 dpi, darker, edge shadow, skew 1.2 to 1.4 degrees), 2
  faxes (200 dpi, 1 bit, salt noise), 1 phone photo, 1 viewer screenshot (about 120 dpi).
- **55 held-out pages**: the 11 digital beta PDFs (all drawings) passed through the same five
  effects by the generator itself (`tools/make_rfq_beta.py`, `python ocr.py --evaluate
  --heldout`), split by drawing into a "tune" half that the search scored and a "check" half
  it never saw (see "What was tried"). The generator's scan and fax noise comes from Pillow's
  `effect_noise`, which draws from C `rand()`, not from the seeded Python generator, so the
  held-out pages are the same only for the same command in a fresh process. A search compares
  every candidate on one set of pages, so its choices are fair, but each held-out count is one
  noise sample, good to about 3 keys per kind (see "Review").

Tesseract: 5.3.4 is installed here (Ubuntu 24.04). The Render image is Debian bookworm, which
ships 5.3.0, so 5.3.0 was built from the upstream tag (`cmake`, no OpenMP) and the chosen
settings were run on it too (`RFQ_TESSERACT=/path/to/tesseract python ocr.py --evaluate`).
Both use the same `eng.traineddata` (the Debian and Ubuntu package `tesseract-ocr-eng`
1:4.1.0, a 4 MB integer LSTM model).

## What was tried

Every setting below was tried on every kind of file, one at a time from the best so
far (coordinate descent: a setting is kept when it finds more key fields, or the same number
with a clearly better word F1 for its time; `_better` in `ocr.py`). The options:

| setting | values tried | what it does |
| --- | --- | --- |
| rasterization | the scan's own resolution (`native`, read from `pdfimages -list`), 300, 400 dpi | pdftoppm resolution for a scanned PDF page |
| upscale | none, to 240, 300, or 400 dpi (Lanczos) | for the 200 dpi copier and fax pages and the photo and screenshot |
| `color` | gray (default), keep colors | photo and screenshot only |
| `median` | off, 3x3 | median filter for fax salt noise, before any resize |
| `flatten` | off, on | remove uneven light (copier edge shadow, photo light falloff) |
| `invert` | off, on | turn dark table header bands (white text) dark on light |
| `autocontrast` | off, on | stretch the gray levels (1 % cutoff) |
| `unsharp` | off, on | unsharp mask (radius 1.5, 80 %) after resizing |
| `deskew` | off, on | projection-profile skew search within 3 degrees, then rotate |
| `page` | off, on | find the sheet in a photo (warp its perspective flat) or screenshot (crop the viewer) |
| `--psm` | 3, 4, 6, 11, 12, and the two-pass 3+11, 11+3 (4+11, 6+11, 3+12 in round 1) | page segmentation |
| `merge` | `fill`, `conf` | how a second pass joins the first: add words where the first found none; `conf` also swaps in a word read at the same place with 15 points more confidence |
| `-c thresholding_method` | default (0, Otsu), 1 (adaptive Otsu), 2 (Sauvola) | Tesseract's own binarization |
| `--dpi` | on, off | tell Tesseract the resolution of the picture it gets |
| `-c preserve_interword_spaces=1` | off, on (round 1) | |
| `-c load_system_dawg=0 -c load_freq_dawg=0` | off, on (round 1) | no English word lists |
| `repair` | off, on | an `l` in an all-caps line becomes `I` (Helvetica draws them alike: "Cl-10442", "TYPE Ill"); after the search, also a capital read twice ("Cc") becomes one |
| `min_conf` | off, 40, 60 | drop words Tesseract is less sure of, except words with a digit and lone capitals |
| model | installed `eng` (tessdata_fast based, 4 MB), `tessdata_best` `eng` (15 MB) | see below |

Rounds:

- **Rounds 1 and 2: the 13 real files only** (284 runs). `preserve_interword_spaces` and the
  word lists gave the same words as the same settings without them, so they were dropped from
  later rounds. The two-pass modes 4+11, 6+11, and 3+12 cost twice the time and never beat
  the best single pass or 3+11 / 11+3. These rounds picked Sauvola thresholding for the
  copier and adaptive Otsu at 400 dpi for the fax, each from 2 real files, and the page
  finder lost a field on the one real photo. That is too little to choose from: on degraded
  copies of the other drawings, adaptive Otsu lost 4 of 24 fax fields and the page finder
  gained 6 of 24 photo fields.
- **Held-out pages.** `python ocr.py --evaluate --heldout` runs the beta generator's own
  effects (`tools/make_rfq_beta.py`: office scan, copier, fax, phone photo, viewer screenshot)
  on the 11 digital beta drawings, 55 pages with known words. They are split by drawing: a
  "tune" half (6 drawings: E03, E19, E20/FR-3103, E32/KF-3412, E62/BWM-3105, E72) and a
  "check" half (5 drawings: E05, E20/FR-3101, E32/KF-3408, E56, E62/BWM-3106).
- **Round 4, the one used** (181 runs, after a first attempt on all 11 held-out drawings was
  stopped at 17 runs to keep the check half unseen): every candidate was scored on the real
  files of a kind plus the 6 tune pages of that kind, every key field counting the same. The
  check half was scored only once, with the final settings.

The fax and copier searches ran to the end (a full round with no change). The scan and photo
searches were stopped in their second round, after it had tried every page segmentation mode
(the settings left to re-test had all lost once already), and the screenshot search late in its
second round, to fit the time on a shared 4-core machine. The tables below are every run, in
the order the search made them; the search writes them as JSON lines (`--log`).

### Office scans (`scan`, 7 real files + 6 tune pages)

| change from the best so far | keys, real | keys, tune pages | word recall | word precision | CPU s/page | kept |
|---|---|---|---|---|---|---|
| start: psm=3,raster_dpi=native,tess_dpi=1 | 44/49 | 21/24 | 0.889 | 0.884 | 1.49 | yes |
| raster_dpi=400 | 43/49 | 20/24 | 0.890 | 0.875 | 2.01 |  |
| invert=1 | 46/49 | 21/24 | 0.894 | 0.885 | 1.50 | yes |
| psm=4 | 40/49 | 19/24 | 0.902 | 0.861 | 1.49 |  |
| psm=6 | 41/49 | 19/24 | 0.788 | 0.815 | 1.52 |  |
| psm=11 | 40/49 | 21/24 | 0.875 | 0.886 | 1.44 |  |
| psm=12 | 30/49 | 21/24 | 0.793 | 0.817 | 1.84 |  |
| merge=fill, psm=3+11 | 46/49 | 21/24 | 0.903 | 0.866 | 2.73 |  |
| merge=conf, psm=3+11 | 46/49 | 21/24 | 0.918 | 0.880 | 2.73 |  |
| merge=fill, psm=11+3 | 44/49 | 22/24 | 0.922 | 0.864 | 2.77 |  |
| merge=conf, psm=11+3 | 45/49 | 21/24 | 0.928 | 0.870 | 2.78 |  |
| threshold=1 | 39/49 | 20/24 | 0.806 | 0.891 | 2.20 |  |
| threshold=2 | 42/49 | 22/24 | 0.890 | 0.881 | 1.68 |  |
| repair=1 | 48/49 | 21/24 | 0.895 | 0.886 | 1.53 | yes |
| min_conf=40 | 48/49 | 21/24 | 0.888 | 0.937 | 1.52 | yes |
| min_conf=60 | 48/49 | 21/24 | 0.883 | 0.950 | 1.56 | yes |
| tess_dpi=off | 48/49 | 21/24 | 0.883 | 0.950 | 1.56 |  |
| deskew=1 | 45/49 | 22/24 | 0.883 | 0.942 | 2.25 |  |
| autocontrast=1 | 46/49 | 19/24 | 0.874 | 0.947 | 1.56 |  |
| unsharp=1 | 40/49 | 20/24 | 0.874 | 0.946 | 1.70 |  |
| median=3 | 47/49 | 19/24 | 0.894 | 0.939 | 2.44 |  |
| raster_dpi=400 | 43/49 | 20/24 | 0.874 | 0.954 | 2.13 |  |
| invert=off | 44/49 | 21/24 | 0.876 | 0.950 | 1.55 |  |
| psm=4 | 44/49 | 19/24 | 0.887 | 0.942 | 1.54 |  |
| psm=6 | 42/49 | 18/24 | 0.765 | 0.938 | 1.56 |  |
| psm=11 | 40/49 | 21/24 | 0.867 | 0.955 | 1.48 |  |
| psm=12 | 31/49 | 21/24 | 0.784 | 0.887 | 2.10 |  |
| merge=fill, psm=3+11 | 48/49 | 21/24 | 0.918 | 0.938 | 2.85 | yes |
| merge=conf | 48/49 | 21/24 | 0.922 | 0.942 | 2.83 | yes **chosen** |
| merge=fill, psm=11+3 | 45/49 | 22/24 | 0.921 | 0.932 | 2.84 |  |
| psm=11+3 | 46/49 | 21/24 | 0.924 | 0.935 | 2.78 |  |

The search's own pick: `invert=1,merge=conf,min_conf=60,psm=3+11,raster_dpi=native,repair=1,tess_dpi=1`.

### Copier scans (`lowres`, 2 real files + 6 tune pages)

| change from the best so far | keys, real | keys, tune pages | word recall | word precision | CPU s/page | kept |
|---|---|---|---|---|---|---|
| start: psm=3,raster_dpi=native,target_dpi=300,tess_dpi=1 | 11/14 | 18/24 | 0.862 | 0.783 | 1.54 | yes |
| target_dpi=off | 7/14 | 19/24 | 0.834 | 0.826 | 1.14 |  |
| target_dpi=400 | 10/14 | 20/24 | 0.879 | 0.777 | 2.00 | yes |
| flatten=1 | 13/14 | 22/24 | 0.880 | 0.797 | 2.21 | yes |
| invert=1 | 13/14 | 22/24 | 0.880 | 0.797 | 2.28 |  |
| psm=4 | 13/14 | 22/24 | 0.900 | 0.813 | 2.12 | yes |
| psm=6 | 12/14 | 21/24 | 0.755 | 0.661 | 2.81 |  |
| psm=11 | 12/14 | 15/24 | 0.857 | 0.830 | 2.05 |  |
| psm=12 | 12/14 | 14/24 | 0.847 | 0.821 | 2.35 |  |
| merge=fill, psm=3+11 | 13/14 | 22/24 | 0.890 | 0.769 | 3.89 |  |
| merge=conf, psm=3+11 | 13/14 | 22/24 | 0.908 | 0.784 | 3.86 |  |
| merge=fill, psm=11+3 | 13/14 | 18/24 | 0.906 | 0.764 | 3.92 |  |
| merge=conf, psm=11+3 | 13/14 | 18/24 | 0.909 | 0.765 | 3.98 |  |
| threshold=1 | 13/14 | 17/24 | 0.766 | 0.796 | 4.19 |  |
| threshold=2 | 13/14 | 22/24 | 0.897 | 0.782 | 2.57 |  |
| repair=1 | 13/14 | 23/24 | 0.900 | 0.814 | 2.13 | yes **chosen** |
| min_conf=40 | 13/14 | 22/24 | 0.889 | 0.881 | 2.14 |  |
| min_conf=60 | 13/14 | 22/24 | 0.880 | 0.908 | 2.12 |  |
| tess_dpi=off | 13/14 | 23/24 | 0.900 | 0.814 | 2.17 |  |
| deskew=1 | 13/14 | 22/24 | 0.898 | 0.787 | 3.26 |  |
| autocontrast=1 | 13/14 | 20/24 | 0.890 | 0.791 | 2.21 |  |
| unsharp=1 | 11/14 | 20/24 | 0.883 | 0.788 | 2.44 |  |
| median=3 | 11/14 | 17/24 | 0.833 | 0.744 | 2.80 |  |
| target_dpi=off | 13/14 | 18/24 | 0.855 | 0.783 | 1.25 |  |
| target_dpi=300 | 12/14 | 18/24 | 0.899 | 0.775 | 1.69 |  |
| flatten=off | 12/14 | 20/24 | 0.886 | 0.782 | 1.98 |  |
| invert=1 | 13/14 | 23/24 | 0.900 | 0.814 | 2.23 |  |
| psm=3 | 13/14 | 22/24 | 0.880 | 0.797 | 2.33 |  |
| psm=6 | 13/14 | 21/24 | 0.756 | 0.661 | 2.94 |  |
| psm=11 | 12/14 | 16/24 | 0.858 | 0.831 | 2.10 |  |
| psm=12 | 12/14 | 15/24 | 0.848 | 0.821 | 2.46 |  |
| merge=fill, psm=3+11 | 13/14 | 22/24 | 0.890 | 0.769 | 4.29 |  |
| merge=conf, psm=3+11 | 13/14 | 22/24 | 0.908 | 0.784 | 4.06 |  |
| merge=fill, psm=11+3 | 13/14 | 19/24 | 0.907 | 0.764 | 4.02 |  |
| merge=conf, psm=11+3 | 13/14 | 19/24 | 0.910 | 0.766 | 4.03 |  |
| threshold=1 | 13/14 | 17/24 | 0.766 | 0.796 | 4.30 |  |
| threshold=2 | 13/14 | 23/24 | 0.898 | 0.782 | 2.56 |  |

The search's own pick: `flatten=1,psm=4,raster_dpi=native,repair=1,target_dpi=400,tess_dpi=1`.

### Faxes (`bilevel`, 2 real files + 6 tune pages)

| change from the best so far | keys, real | keys, tune pages | word recall | word precision | CPU s/page | kept |
|---|---|---|---|---|---|---|
| start: psm=3,raster_dpi=native,target_dpi=300,tess_dpi=1 | 9/10 | 15/24 | 0.852 | 0.860 | 1.40 | yes |
| target_dpi=off | 10/10 | 19/24 | 0.857 | 0.882 | 1.00 | yes |
| target_dpi=400 | 10/10 | 15/24 | 0.864 | 0.889 | 1.75 |  |
| invert=1 | 10/10 | 19/24 | 0.857 | 0.878 | 1.02 |  |
| psm=4 | 10/10 | 16/24 | 0.875 | 0.854 | 1.02 |  |
| psm=6 | 10/10 | 20/24 | 0.741 | 0.785 | 1.07 | yes |
| psm=11 | 8/10 | 18/24 | 0.843 | 0.897 | 0.98 |  |
| psm=12 | 8/10 | 18/24 | 0.829 | 0.890 | 1.32 |  |
| merge=fill, psm=3+11 | 10/10 | 19/24 | 0.873 | 0.858 | 1.91 |  |
| merge=conf, psm=3+11 | 10/10 | 19/24 | 0.878 | 0.864 | 1.89 |  |
| merge=fill, psm=11+3 | 9/10 | 21/24 | 0.888 | 0.871 | 1.90 | yes |
| merge=conf | 9/10 | 22/24 | 0.891 | 0.876 | 1.91 | yes |
| threshold=1 | 9/10 | 18/24 | 0.891 | 0.852 | 1.96 |  |
| threshold=2 | 9/10 | 16/24 | 0.881 | 0.855 | 2.04 |  |
| repair=1 | 10/10 | 22/24 | 0.892 | 0.877 | 1.89 | yes **chosen** |
| min_conf=40 | 9/10 | 22/24 | 0.887 | 0.905 | 1.88 |  |
| min_conf=60 | 9/10 | 21/24 | 0.881 | 0.924 | 1.86 |  |
| tess_dpi=off | 10/10 | 22/24 | 0.892 | 0.877 | 1.89 |  |
| deskew=1 | 10/10 | 20/24 | 0.910 | 0.868 | 2.45 |  |
| autocontrast=1 | 10/10 | 22/24 | 0.892 | 0.877 | 1.93 |  |
| unsharp=1 | 10/10 | 22/24 | 0.898 | 0.847 | 2.02 |  |
| median=3 | 5/10 | 10/24 | 0.750 | 0.698 | 2.13 |  |
| target_dpi=300 | 8/10 | 18/24 | 0.896 | 0.860 | 2.64 |  |
| target_dpi=400 | 10/10 | 17/24 | 0.903 | 0.865 | 3.33 |  |
| invert=1 | 10/10 | 22/24 | 0.892 | 0.872 | 1.96 |  |
| psm=3 | 10/10 | 19/24 | 0.857 | 0.882 | 1.00 |  |
| psm=4 | 10/10 | 16/24 | 0.875 | 0.854 | 1.04 |  |
| psm=6 | 10/10 | 20/24 | 0.741 | 0.785 | 1.11 |  |
| psm=11 | 9/10 | 18/24 | 0.844 | 0.898 | 1.01 |  |
| psm=12 | 9/10 | 18/24 | 0.830 | 0.890 | 1.39 |  |
| merge=fill, psm=3+11 | 10/10 | 19/24 | 0.873 | 0.858 | 1.96 |  |
| psm=3+11 | 10/10 | 19/24 | 0.878 | 0.864 | 1.96 |  |
| threshold=1 | 10/10 | 19/24 | 0.892 | 0.854 | 2.05 |  |
| threshold=2 | 10/10 | 17/24 | 0.883 | 0.856 | 2.13 |  |

The search's own pick: `merge=conf,psm=11+3,raster_dpi=native,repair=1,tess_dpi=1`.

### Phone photo (`photo`, 1 real file + 6 tune pages)

| change from the best so far | keys, real | keys, tune pages | word recall | word precision | CPU s/page | kept |
|---|---|---|---|---|---|---|
| start: page=1,psm=3,target_dpi=300,tess_dpi=1 | 3/4 | 21/24 | 0.820 | 0.779 | 1.85 | yes |
| page=off | 4/4 | 15/24 | 0.694 | 0.890 | 1.60 |  |
| target_dpi=off | 4/4 | 18/24 | 0.764 | 0.781 | 1.40 |  |
| target_dpi=400 | 4/4 | 18/24 | 0.822 | 0.796 | 2.34 |  |
| color=1 | 3/4 | 20/24 | 0.805 | 0.771 | 2.56 |  |
| flatten=1 | 3/4 | 20/24 | 0.862 | 0.821 | 1.97 |  |
| invert=1 | 3/4 | 21/24 | 0.820 | 0.779 | 1.85 |  |
| psm=4 | 4/4 | 19/24 | 0.825 | 0.762 | 1.80 |  |
| psm=6 | 3/4 | 13/24 | 0.696 | 0.636 | 2.23 |  |
| psm=11 | 3/4 | 17/24 | 0.823 | 0.727 | 1.91 |  |
| psm=12 | 3/4 | 16/24 | 0.800 | 0.727 | 2.20 |  |
| merge=fill, psm=3+11 | 3/4 | 21/24 | 0.841 | 0.658 | 3.29 |  |
| merge=conf, psm=3+11 | 3/4 | 21/24 | 0.852 | 0.667 | 3.28 |  |
| merge=fill, psm=11+3 | 4/4 | 22/24 | 0.909 | 0.682 | 3.31 | yes |
| merge=conf | 4/4 | 22/24 | 0.913 | 0.685 | 3.35 | yes |
| threshold=1 | 4/4 | 17/24 | 0.880 | 0.282 | 13.13 |  |
| threshold=2 | 3/4 | 21/24 | 0.916 | 0.815 | 3.51 |  |
| repair=1 | 4/4 | 23/24 | 0.913 | 0.686 | 3.37 | yes |
| min_conf=40 | 4/4 | 23/24 | 0.908 | 0.854 | 3.41 | yes |
| min_conf=60 | 4/4 | 23/24 | 0.906 | 0.880 | 3.37 | yes **chosen** |
| tess_dpi=off | 4/4 | 23/24 | 0.906 | 0.880 | 3.40 |  |
| deskew=1 | 4/4 | 23/24 | 0.906 | 0.880 | 3.79 |  |
| autocontrast=1 | 3/4 | 23/24 | 0.876 | 0.893 | 3.36 |  |
| unsharp=1 | 3/4 | 20/24 | 0.902 | 0.879 | 3.65 |  |
| page=off | 3/4 | 17/24 | 0.838 | 0.918 | 3.23 |  |
| target_dpi=off | 2/4 | 21/24 | 0.870 | 0.889 | 2.46 |  |
| target_dpi=400 | 3/4 | 21/24 | 0.891 | 0.895 | 4.26 |  |
| color=1 | 3/4 | 22/24 | 0.895 | 0.893 | 4.43 |  |
| flatten=1 | 3/4 | 21/24 | 0.919 | 0.931 | 3.72 |  |
| invert=1 | 4/4 | 23/24 | 0.906 | 0.880 | 3.44 |  |
| psm=3 | 3/4 | 22/24 | 0.805 | 0.905 | 1.91 |  |
| psm=4 | 4/4 | 20/24 | 0.805 | 0.897 | 1.86 |  |
| psm=6 | 2/4 | 14/24 | 0.662 | 0.892 | 2.28 |  |
| psm=11 | 3/4 | 18/24 | 0.810 | 0.921 | 1.95 |  |
| psm=12 | 3/4 | 17/24 | 0.785 | 0.923 | 2.30 |  |
| merge=fill, psm=3+11 | 3/4 | 22/24 | 0.892 | 0.871 | 3.38 |  |
| psm=3+11 | 3/4 | 22/24 | 0.896 | 0.875 | 3.35 |  |

The search's own pick: `merge=conf,min_conf=60,page=1,psm=11+3,repair=1,target_dpi=300,tess_dpi=1`.

### Viewer screenshot (`screen`, 1 real file + 6 tune pages)

| change from the best so far | keys, real | keys, tune pages | word recall | word precision | CPU s/page | kept |
|---|---|---|---|---|---|---|
| start: page=1,psm=3,target_dpi=300,tess_dpi=1 | 3/4 | 21/24 | 0.871 | 0.856 | 1.90 | yes |
| page=off | 3/4 | 22/24 | 0.760 | 0.880 | 1.56 | yes |
| target_dpi=off | 3/4 | 16/24 | 0.593 | 0.797 | 0.71 |  |
| target_dpi=240 | 3/4 | 18/24 | 0.759 | 0.900 | 1.29 |  |
| target_dpi=400 | 3/4 | 20/24 | 0.741 | 0.887 | 2.04 |  |
| color=1 | 3/4 | 18/24 | 0.754 | 0.889 | 2.00 |  |
| invert=1 | 3/4 | 22/24 | 0.760 | 0.880 | 1.61 |  |
| psm=4 | 3/4 | 22/24 | 0.764 | 0.885 | 1.57 | yes |
| psm=6 | 2/4 | 20/24 | 0.745 | 0.790 | 1.66 |  |
| psm=11 | 3/4 | 22/24 | 0.835 | 0.848 | 1.67 | yes |
| psm=12 | 3/4 | 22/24 | 0.838 | 0.827 | 1.98 |  |
| merge=fill, psm=3+11 | 3/4 | 22/24 | 0.811 | 0.835 | 3.10 |  |
| merge=conf, psm=3+11 | 3/4 | 22/24 | 0.819 | 0.842 | 3.07 |  |
| merge=fill, psm=11+3 | 3/4 | 22/24 | 0.836 | 0.847 | 3.08 |  |
| merge=conf, psm=11+3 | 3/4 | 22/24 | 0.841 | 0.852 | 3.07 |  |
| threshold=1 | 3/4 | 19/24 | 0.800 | 0.842 | 1.60 |  |
| threshold=2 | 3/4 | 20/24 | 0.837 | 0.851 | 1.77 |  |
| repair=1 | 4/4 | 23/24 | 0.836 | 0.850 | 1.69 | yes |
| min_conf=40 | 4/4 | 23/24 | 0.836 | 0.882 | 1.67 | yes |
| min_conf=60 | 4/4 | 23/24 | 0.835 | 0.899 | 1.69 | yes **chosen** |
| tess_dpi=off | 4/4 | 23/24 | 0.835 | 0.899 | 1.69 |  |
| deskew=1 | 4/4 | 23/24 | 0.835 | 0.899 | 2.14 |  |
| autocontrast=1 | 4/4 | 21/24 | 0.835 | 0.902 | 1.71 |  |
| unsharp=1 | 4/4 | 21/24 | 0.826 | 0.905 | 1.84 |  |
| page=1 | 4/4 | 21/24 | 0.847 | 0.957 | 1.72 |  |
| target_dpi=off | 3/4 | 17/24 | 0.653 | 0.817 | 0.79 |  |
| target_dpi=240 | 4/4 | 20/24 | 0.829 | 0.903 | 1.38 |  |
| target_dpi=400 | 4/4 | 22/24 | 0.829 | 0.909 | 2.17 |  |
| color=1 | 4/4 | 20/24 | 0.838 | 0.909 | 2.13 |  |
| invert=1 | 4/4 | 23/24 | 0.835 | 0.899 | 1.74 |  |
| psm=3 | 3/4 | 23/24 | 0.757 | 0.914 | 1.64 |  |
| psm=4 | 3/4 | 23/24 | 0.763 | 0.918 | 1.60 |  |
| psm=6 | 2/4 | 21/24 | 0.716 | 0.882 | 1.74 |  |
| psm=12 | 4/4 | 23/24 | 0.837 | 0.904 | 2.07 | yes |
| merge=fill, psm=3+11 | 3/4 | 23/24 | 0.850 | 0.904 | 3.21 |  |
| merge=conf, psm=3+11 | 3/4 | 23/24 | 0.852 | 0.907 | 3.23 |  |
| merge=fill, psm=11+3 | 4/4 | 23/24 | 0.837 | 0.896 | 3.19 |  |
| merge=conf, psm=11+3 | 4/4 | 23/24 | 0.843 | 0.902 | 3.22 |  |
| threshold=1 | 3/4 | 20/24 | 0.800 | 0.902 | 2.03 |  |
| threshold=2 | 4/4 | 21/24 | 0.835 | 0.919 | 2.21 |  |
| repair=off | 3/4 | 22/24 | 0.836 | 0.903 | 2.06 |  |
| min_conf=off | 4/4 | 23/24 | 0.839 | 0.829 | 2.06 |  |

The search's own pick: `min_conf=60,psm=12,repair=1,target_dpi=300,tess_dpi=1`.

## tessdata_best: not worth fetching

The `eng.traineddata` that Debian installs (package `tesseract-ocr-eng` 1:4.1.0) is an integer
LSTM model of 4.1 MB. The float model from
https://github.com/tesseract-ocr/tessdata_best (`eng.traineddata`, 15.4 MB) was downloaded and
run with the same recipes (`python ocr.py --evaluate --settings best --model DIR`, or
`RFQ_OCR_TESSDATA=DIR` at run time; the folder also needs `osd.traineddata` for the orientation
check):

| model | recipes | key fields | word recall | word precision | word F1 | CPU s/page |
| --- | --- | --- | --- | --- | --- | --- |
| installed (4.1 MB) | best | 81/81 | 0.926 | 0.904 | 0.913 | 2.63 |
| tessdata_best (15.4 MB) | best | 77/81 | 0.931 | 0.904 | 0.916 | 5.52 |
| installed (4.1 MB) | fast | 80/81 | 0.898 | 0.909 | 0.902 | 1.67 |
| tessdata_best (15.4 MB) | fast | 78/81 | 0.905 | 0.910 | 0.906 | 3.37 |

The best model reads the same words (F1 within 0.004) at twice the CPU, and on these files it
loses fields the installed one gets: the E61 rev, the E72 quantity list "25 / 100 / 250", the
E09 finish, and the E22 material. The recipes were tuned on the installed model, so part of that gap may be
the tuning, but nothing here suggests a retuned float model would pay for doubling the time on
a 0.1 CPU host and adding 15 MB to the image. The Docker build should keep Debian's model.

## Tesseract 5.3.0 (the Render image)

The Render image (python:3.12-slim, Debian bookworm) gets tesseract 5.3.0 and leptonica 1.82.0
from apt. Both were built here from the upstream tags (`cmake`, release, no OpenMP), and every
option `ocr.py` can pass was checked against that build's `--help-extra` and
`--print-parameters` (5.3.4 adds only debug, graphics, and curl variables):

| option | used by | in 5.3.0 |
| --- | --- | --- |
| `-l eng --oem 1 --psm 3/4/11 --dpi N` | every recipe | yes (`--dpi` since 4.0) |
| `--psm 0` | orientation check, only for a page that looks turned | yes (needs `osd.traineddata`) |
| `-c tessedit_create_tsv=1` | every call: the word table with boxes and confidences | yes |
| `-c thresholding_method=1/2` | not chosen, search only | yes (added in 5.0) |
| `-c preserve_interword_spaces=1` | not chosen, search only | yes |
| `-c load_system_dawg=0 -c load_freq_dawg=0` | not chosen, search only | yes |
| `--tessdata-dir` | `RFQ_OCR_TESSDATA`, another model | yes |

`ocr.py` also guards at run time: it reads `tesseract --print-parameters` once and leaves out
any `-c` variable the installed build does not list (without that list, by version:
`thresholding_method` only from 5.0), so an older or stripped build reads the page with
defaults instead of failing. `tests/test_ocr.py` builds the command line for every recipe and
checks each option against the 5.3.0 list. Debian's `tesseract-ocr` package depends on
`tesseract-ocr-osd`; if `osd.traineddata` is missing anyway, the orientation check is skipped.

The same evaluation on the 5.3.0 build (`RFQ_TESSERACT=/path/to/5.3.0/tesseract python ocr.py
--evaluate --settings best --settings fast`), with the same Debian `eng.traineddata`:

| | keys, best | word F1, best | CPU s/page, best | keys, fast | CPU s/page, fast |
| --- | --- | --- | --- | --- | --- |
| tesseract 5.3.4 (here) | 81/81 | 0.913 | 2.61 | 80/81 | 1.69 |
| tesseract 5.3.0 (Render) | 80/81 | 0.911 | 2.40 | 79/81 | 1.57 |

Every file reads the same key fields on both versions except the screenshot, where 5.3.0 does
not read the rev letter "A" beside the drawing number (3 of 4). Word recall and precision
differ by 0.04 at most on any file. The committed cache was built with 5.3.4, so the demo on
Render shows the 5.3.4 results for the beta files; only live uploads run on 5.3.0.

## Practical limits

What the measurements above cover: letter-size pages at 120 to 300 dpi, skewed up to
1.4 degrees, printed in Helvetica, one page each. Beyond that:

- **Low resolution.** The copier scans at 200 dpi read well once upsampled to 400 dpi, and the
  screenshot at about 120 dpi works because the viewer drew the text anti-aliased. A 100 or
  150 dpi scan of paper loses the small title block text for good: upsampling smooths the
  strokes but cannot bring back a character that was never resolved. Tesseract's own
  guidance is 300 dpi; ask the customer for that.
- **Skew.** Tesseract follows the 0.3 to 1.4 degree skews here by itself, and the
  projection-profile `deskew` never helped. It searches within 3 degrees either way; pages fed
  more crooked than that were not tested.
- **Rotated pages.** A page on its side or upside down is caught by the orientation check (tall
  word boxes or low confidence trigger tesseract's `--psm 0`, then the turned page is read
  again). It costs one more tesseract run, only on such pages. A sideways note on an upright
  drawing is read in the page's main direction only.
- **Wrapped notes and lone rev letters.** The misses that are left are a few rev letters (none
  on the real files since the review's `invert` change; on the held-out pages rev letters in
  small title block cells that come back empty or as junk rather than as "Cc", which `repair`
  would fix; a single-character pass over empty title block cells could get them, not tried)
  and long notes that wrap onto a second line, whose words are all in the text: pairing them
  with their label is the extractor's job, and every run of words in `lines` carries its box on
  the page for that.
- **Dark table headers.** Handled by `invert` on office scans and copier scans (on the copier
  it has to run before `flatten`, which would otherwise lift the band to gray first). Faxes,
  photos, and screenshots do not use it: no fax, photo, or screenshot of a form was measured.
  A dark band that is not a clean rectangle of light text (a logo, a photo) is left alone.
- **Not tested here: handwriting, color stamps, highlighter.** The English model reads print,
  so hand-written quantities and red-line markups will come back as junk or be dropped by
  `min_conf`. Every page is turned to gray before OCR (`color` did not help on these files), so
  a colored stamp over the title block becomes a gray blotch over the text under it.
- **Very large pages.** A PDF page is rendered at a resolution that keeps it under 36 million
  pixels (`MAX_OCR_PIXELS`), read from its size in `pdfinfo`: a D-size drawing (36 x 24 in) at
  204 dpi, the largest page PDF allows (200 in square) at 30 dpi. Before the review pdftoppm
  rendered the page in full first: a 97 in page took 2.4 GB in pdftoppm and 1.6 GB in Python
  before failing, which would have killed a 512 MB host. A picture file above 36 million
  pixels is scaled down before OCR, and one claiming more than 80 million is refused (without
  Pillow, from its header). At most 6 pages of a PDF are read (`RFQ_OCR_MAX_PAGES`).
- **Scans finer than 300 dpi.** Rendered down to 300 dpi (`MAX_RASTER_DPI`): on 600 dpi office
  scans of the five check drawings, 300 dpi found 19 of 20 key fields, 400 dpi and the scan's
  own 600 dpi 17, and 600 dpi took 6.7 CPU seconds a page against 3.0. Only 600 dpi was measured.
- **A text layer that is only a stamp.** Some scanners type a line over the page ("Scanned by
  ... page 1 of 1"). A page is OCR'd when its text layer has fewer than 40 letters and digits,
  or fewer than 200 (`SCAN_STAMP_MAX_CHARS`) while one picture covers at least 60 % of it
  (`SCAN_PICTURE_COVER`); the typed beta drawings and forms have 714 to 1189 a page. A page
  with a real OCR text layer from the scanner (a "searchable PDF") keeps that layer, good or
  bad. Without poppler's `pdfimages` only the 40-letter rule applies.
- **Without poppler or Pillow.** Without `pdftoppm` (poppler-utils) the pictures come out of
  the PDF through pypdf and are classified by their resolution on the page and bit depth like
  a rendered page: 74 of 81 real keys, against 81, because the recipes were tuned on
  pdftoppm's rendering, which differs from Pillow's decoding of the same JPEG by about 6 gray
  levels a pixel: E61 and E72 lose 3 fields each and E62 one. A page drawn as vectors (text
  turned to outlines, no picture) cannot be read at all without pdftoppm. Without Pillow the
  page goes to Tesseract with no clean-up: 71 of 81.
- **Slow hosts.** One OCR job at a time (`RFQ_OCR_WORKERS`, default 1) and a limit of 300 s a
  file (`RFQ_OCR_TIMEOUT`): on 0.1 CPU a multi-page scan upload can run into it, and then the
  file shows an error instead of text.

## Review

A second pass over this page and `ocr.py` reran every measurement, tried settings the search
rejected or never tried, fed the pipeline broken and unusual files, and checked the cache, the
server, and the tests. It changed two settings (`invert` for copier scans, `MAX_RASTER_DPI`)
and fixed seven faults in how files are read. Of the 30 beta files only E62 reads differently:
its first table row is now read.

### The numbers reproduce

Before any change, `python ocr.py --evaluate` and `python ocr.py --evaluate --heldout` (each in
a fresh process, as at the top of this page) gave every key count and word F1 in the per-file,
per-kind, and held-out tables exactly, and CPU seconds within 3 % (best 2.64 against 2.61 s a
page, fast 1.70 against 1.68). The tables above are the final run with the review's changes.

### The held-out pages carry fresh noise

`tools/make_rfq_beta.py` seeds Python's `random` before `Image.effect_noise`, but Pillow draws
that noise from C `rand()`, so every run of the generator in a new order makes new noise: the
same held-out drawing, skew, and effect, different speckle and grain. The documented commands
give the same pages in a fresh process, which is why the tables reproduce; a search or a
`--type` filter makes its own. Three more noise samples of the 25 check pages, burning a
different number of `rand()` values before generating them:

| noise sample | keys, before | keys, best | keys, fast | best minus before: scan, copier, fax, photo, screen |
| --- | --- | --- | --- | --- |
| the documented run | 76/100 | 83/100 | 83/100 | +3, -1, +3, 0, +2 |
| 2 | 75/100 | 86/100 | 84/100 | +3, +1, +3, +2, +2 |
| 3 | 74/100 | 86/100 | 86/100 | +1, +2, +6, +1, +2 |
| 4 | 79/100 | 86/100 | 83/100 | +3, 0, +1, +1, +2 |

The chosen settings beat the old pipeline by 7 to 12 keys in 100 on every sample, and the fast
ones by 4 to 12. One kind moves by up to 3 keys between samples (best on faxes: 16, 17, 19,
17 of 20), so a one or two key difference within a kind, such as the copier row of the
held-out table, is noise. The screenshot effect draws no random noise, so its row never moves.

### Settings tried in the review

Each against the chosen recipe on the same pages: the real files of that kind and all 11
held-out drawings of that kind (one noise sample, made once and reused for both). Keys are
real / held-out; word F1 and CPU seconds a page are over all of those pages. The first seven
rows were scored before the copier `invert` change, so their chosen copier recipe has 13 of 14
real keys.

| setting | kind | keys, chosen | keys, tried | word F1, chosen / tried | CPU s/page, chosen / tried | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `-c tessedit_do_invert=0` (Tesseract stops re-reading unsure words inverted), never tried | all five | 80/81 real, 196/220 held-out | 80/81, 197/220 | 0.924 / 0.925 scan, 0.835 / 0.841 copier, 0.880 / 0.880 fax, 0.891 / 0.896 photo, 0.872 / 0.882 screen | 2 to 8 % less | not taken: one held-out key and a few percent of time, while word F1 on the two real copier scans fell from 0.875 to 0.848 |
| `--psm 4` then `--psm 11`, merged by confidence | copier | 13/14, 37/44 | 13/14, 37/44 | 0.835 / 0.833 | 2.27 / 3.98 | no: the same keys for 75 % more time |
| `--psm 4` then `--psm 3`, merged by confidence | copier | 13/14, 37/44 | 13/14, 38/44 | 0.835 / 0.832 | 2.27 / 4.05 | no: one held-out key for 78 % more time |
| `--psm 12` (the search's own last pick) | screen | 4/4, 42/44 | 4/4, 42/44 | 0.872 / 0.872 | 1.72 / 1.99 | no: confirms keeping `--psm 11` |
| `page` (crop the viewer to the sheet) | screen | 4/4, 42/44 | 4/4, 39/44 | 0.872 / 0.897 | 1.72 / 1.76 | no: cleaner text, 3 keys fewer, as the search found |
| upscale the native raster to 400 dpi (Lanczos; the search tried pdftoppm at 400) | scan | 49/49, 39/44 | 47/49, 40/44 | 0.924 / 0.920 | 2.80 / 3.69 | no: loses 2 real keys |
| `--psm 6` then `--psm 11` (psm 6 found the most tune keys alone) | fax | 10/10, 39/44 | 10/10, 37/44 | 0.880 / 0.809 | 1.97 / 2.05 | no |
| `invert`, run before `flatten` | copier | 13/14, 37/44 | 14/14, 37/44 | 0.843 / 0.842 | 2.19 / 2.23 | taken: E62's rev A, the last real miss; the held-out drawings read the same |
| render 600 dpi scans at 300 dpi (5 check drawings, the generator's office scan at 600 dpi) | scan | 17/20 at 600 dpi | 19/20 at 300, 17/20 at 400 | 0.938 / 0.922 | 6.73 / 2.98 | taken: `MAX_RASTER_DPI` 300 |

### Faults fixed

| input | before the review | now |
| --- | --- | --- |
| a scan with a 65-letter stamp typed over it ("Scanned by ScanDesk Pro 4.2 on ... page 1 of 1") | taken for a typed PDF: 87 characters of stamp, the drawing never read | OCR'd: a page with under 200 letters and digits and a picture over 60 % of it is a scan |
| a 16-bit gray PNG of E61 | "OCR found no text": Pillow's `convert("L")` clips 16-bit values to white | read like the 8-bit scan (1823 characters) |
| a PDF page 97 inches square | pdftoppm rendered it in full: 2.4 GB, then 1.6 GB in Python, then an error | rendered under 36 million pixels: 4 s, 194 MB |
| a 1200 dpi letter scan | "the picture is too large to read" | read at 300 dpi |
| a password-protected PDF | "pdftoppm failed (Command Line Error: Incorrect password)" | "the PDF is protected by a password, so it cannot be opened" |
| no pdftoppm | every page got the office scan recipe at an assumed 300 dpi, from a JPEG pypdf had decoded and saved again: 72 of 81 real keys | pages classified by resolution and bit depth, the JPEG as stored: see "Without poppler or Pillow" |
| no Pillow, a PNG claiming 100,000 x 100,000 pixels | handed to Tesseract ("Error during processing") | refused from its header |

Also: the timeout message quoted the 300 s default even when the caller set another limit;
it now says the file's time limit ran out.

### Checked and fine

- **Broken files.** PDFs cut at 50 %, 95 %, and 99.9 %, a JPEG cut at 90 %, garbage bytes, and
  pictures claiming 65,000 or 100,000 pixels a side all come back as `method` "none" with an
  error string, in under 0.4 s.
- **Odd but valid files.** PDFs encrypted with an empty password and copying forbidden (the
  usual "uncopyable" PDF): the typed one gives its text layer, the scanned one is OCR'd. A PNG
  with black text on a transparent background, an LA PNG, and a palette PNG with transparency
  read as on white paper, with the same text as the opaque screenshot. A CMYK JPEG reads. 1 x 1
  and 3 x 2000 pixel pictures give `method` "ocr" with the error "OCR found no text". A sideways
  scan is turned by the orientation check (7.5 s); a two-page scan gives both pages.
- **Time limits and the worker limit.** With limits of 0.5 to 3 s on a scan, a sideways scan,
  and the photo, `file_text` returned within 0.04 s of the limit every time, killing Tesseract.
  With `RFQ_OCR_WORKERS=1`, three files at once never ran more than one Tesseract, and a file
  with a 1.5 s limit waiting behind a 4.3 s one gave up at 1.5 s with "the server is busy".
- **No peeking.** With an audit hook recording every file opened and program started, the 13
  files were read with no name and again under the name of another kind of file ("E61 ITAR
  scan.pdf" for a fax): identical text and source type every time, `tests/rfq_beta_truth.json`
  never opened, and no program handed a path from `data/`. Only `--evaluate` reads the truth
  file.
- **The cache and a restart.** `data/rfq_beta/ocr_cache.json` has one entry per file, 30,
  keyed by the SHA-256 of the bytes, each naming its file. Rebuilt with the reviewed code, every
  entry was identical to the earlier build apart from the seconds, except E62 (its new text)
  and E09 (its settings now list `invert`, which found no band). `server.py` started in 0.3 s
  with an empty `PATH` (no tesseract, no poppler) and a fresh `RFQ_CACHE_DIR`, served the text
  of all 24 PDFs and pictures from the cache (`/api/att/E61/0/text` has CI-10442 and the ITAR
  legend), and wrote no OCR result to its own cache. `python rfq_details.py --check` gives
  504 of 504 fields with the old cache and with the new one.
- **Tests.** `tests/test_ocr.py` (46 tests, 12 of them new for the faults above and E62) and
  `tests/test_beta.py` (10) pass; with an empty `PATH`, 16 OCR tests skip and the rest pass.
