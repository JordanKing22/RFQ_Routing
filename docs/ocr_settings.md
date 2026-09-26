# OCR settings for uncopyable RFQ files

Status: DRAFT while the settings search runs. The tables are filled in from
`python ocr.py --evaluate --search` when it finishes.

`ocr.py` reads the 13 beta files that have no text layer (office scans, copier scans, faxes, a
phone photo, a viewer screenshot) with Tesseract. This page says how the settings in
`ocr.py` (`RECIPES` and `FAST_RECIPES`) were chosen: what was tried, the numbers, what won, and
what it costs in time.

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
  --heldout`). One real photo and one real screenshot are too few to choose from: on the real
  photo, finding the page and flattening its perspective lost a field (3/4 against 4/4), while
  on the 11 held-out photos it gained ten (38/44 against 28/44).

The search scored each candidate on the real files of a kind plus its 11 held-out pages, with
every key field counting the same.

Tesseract: 5.3.4 is installed here (Ubuntu 24.04). The Render image is Debian bookworm, which
ships 5.3.0, so 5.3.0 was built from the upstream tag (`cmake`, no OpenMP) and the chosen
settings were run on it too (`RFQ_TESSERACT=/path/to/tesseract python ocr.py --evaluate`).
Both use the same `eng.traineddata` (the Debian and Ubuntu package `tesseract-ocr-eng`
1:4.1.0, a 4 MB integer LSTM model).
