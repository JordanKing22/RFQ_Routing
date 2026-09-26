# RFQ Router demo (Jev)

A local demo that sorts a CNC shop's shared quoting inbox with **Jev**, TypeSafe AI's decision model, called through your Vercel AI Gateway key.

The sample inbox holds 100 fictional emails with realistic attachments: engineering drawings with title blocks, material, finish, notes, and ITAR or CUI legends, STEP 3D models, customer RFQ forms, and purchase orders. Open an email to see its files as thumbnail tiles, the way Gmail shows them. Click one to open it: PDFs in a viewer, images inline, and STEP files as a 3D model you can turn. Jev reads the email together with the text of its attachments, so an ITAR marking that only appears on the drawing still sends the email to the restricted queue.

Jev doesn't write text. It answers typed questions (pick one, yes/no, score) and gives a probability for each answer. That makes it a good fit for routing: Jev makes the judgment calls, and plain Python applies the shop's rules.

**This branch (`RFQ_Details_Beta`) adds the RFQ details beta.** The demo opens a smaller inbox: 22 of the sample emails, whose attachments are 30 real files. 13 of those files cannot be copied from (scans, faxes, a phone photo, a screenshot), so their text is read with Tesseract OCR. The details of every RFQ go into one consolidated file. See [RFQ details beta](#rfq-details-beta). To open the 100-email inbox instead, set `RFQ_EMAILS_FILE=data/sample_emails.json`.

## Run it

1. Double-click **`run_demo.bat`**.
   - First run only: it checks your key with one real call (`check_jev.py`) and saves that result.
   - Then it opens **http://127.0.0.1:8765** in your browser.
   - Nothing to install. It uses only Python's standard library. You need Python 3.10 or newer.
   - Optional: `pip install -r requirements.txt` adds pypdf, Pillow, numpy, and onnxruntime. They read the text of PDFs you upload in **Paste an RFQ**, make image thumbnails, prepare scans and photos for OCR, and run the region detector. OCR of uploaded scans and photos also needs Tesseract and poppler (see [Dependencies](#dependencies-and-running-without-them)). Without any of them the demo still runs, and the 30 beta files still show their text, because it comes from a committed cache.
2. Click **Route inbox**.
3. To stop, close the black window or press Ctrl+C in it.

Your key stays in `Jev_Test\.env` as `AI_GATEWAY_API_KEY=vck_...`. The browser never sees it. The local server talks to Jev, and it only listens on 127.0.0.1.

To check the key again at any time, double-click `check_jev.bat`.

On a Mac or Linux, run `./run_demo.sh` instead. It does the same steps. To check the key, run `python3 check_jev.py`.

## Run it in the cloud

The demo runs anywhere that has Python 3.10 or newer, or that runs a Docker image. Two things change when it leaves your laptop:

- **Give the host your key** as a secret or environment variable: `AI_GATEWAY_API_KEY=vck_...`.
- **A password is required.** Set `RFQ_DEMO_PASSWORD`. The browser asks for it once, and any user name works. The server won't listen on a public address without a password, because anyone who found the URL could spend your Jev key. Use the host's HTTPS address, because the browser sends the password with every request.

| Setting | What it does |
| --- | --- |
| `AI_GATEWAY_API_KEY` | Your Vercel AI Gateway key (`vck_...`). Or `TYPESAFE_API_KEY` for TypeSafe direct. |
| `RFQ_DEMO_PASSWORD` | The demo password. Required whenever other machines can connect. |
| `RFQ_DEMO_HOST` | The address to listen on. `0.0.0.0` in the Docker image, otherwise `127.0.0.1` (this computer only). Same as `--host`. |
| `PORT` or `RFQ_DEMO_PORT` | The port. Cloud hosts set `PORT` for you. The default is 8765. |

These can also go in `.env` next to `server.py`. The beta has a few more settings, listed under [Dependencies](#dependencies-and-running-without-them).

**Render, in one click**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/JordanKing22/RFQ_Routing)

1. Click the button and sign in to Render (a free account works).
2. When Render asks, paste your AI Gateway key into `AI_GATEWAY_API_KEY` and choose a password for `RFQ_DEMO_PASSWORD`.
3. Click **Deploy Blueprint**. The first build takes a few minutes.
4. Open the `https://....onrender.com` address on the service page.

`render.yaml` holds these settings. The free plan sleeps after 15 idle minutes and takes up to a minute to wake, so open the demo a minute before a meeting. The Starter plan stays awake. Both have 512 MB of memory, and a server that has been used sits at 200 to 250 MB, so big pictures are cut down to fit: a JPEG over 25 megapixels is read for OCR at half size or less (a 48 megapixel phone photo at 4000 x 3000), and a PNG over 25 megapixels in color (40 in gray) is not read. The viewer's page image has a similar limit, so such a picture gets no **Detected regions**. Page images render one at a time (`RFQ_PAGE_WORKERS`), and the image sets `MALLOC_ARENA_MAX=2`: 24 page requests at once used to leave the server at 560 MB for good, and now peak at 285 MB and leave 250.

**Docker**

```
docker build -t rfq-router .
docker run --rm -p 8765:8765 -e AI_GATEWAY_API_KEY=vck_... -e RFQ_DEMO_PASSWORD=pick-one rfq-router
```

Then open http://localhost:8765. To keep saved Jev results between runs, add `-v rfq-cache:/app/cache`. The image installs Tesseract and poppler with apt, and pypdf, Pillow, numpy, and onnxruntime from `requirements.txt`. Its base, `python:3.12-slim-trixie`, is pinned to Debian 13, which installs Tesseract 5.5.0 (measured with the same results as the 5.3.4 used here).

**A container host** (Google Cloud Run, Render, Railway, Fly.io, and others)

1. Point the host at this repository. It builds the `Dockerfile`.
2. Add `AI_GATEWAY_API_KEY` and `RFQ_DEMO_PASSWORD` as secrets or environment variables.
3. If the host asks for a health check path, use `/healthz`. It is the only address that needs no password.
4. Open the HTTPS address the host gives you.

**Keep saved results through restarts.** New Jev answers are saved in the container, and a restart, a redeploy, or a free host waking from sleep starts with an empty disk. So the demo also loads `data/saved_results.json` from the repository at startup, and every saved answer in it replays instantly:

1. Route the whole inbox once on the deployed demo (on the free tier this takes about a minute per email, see below).
2. Open **Settings** and click **Download saved results**.
3. Save the file as `data/saved_results.json` in the repository, then commit and push. Render redeploys on its own.

After that, **Route inbox** replays every email in a few seconds, even right after a restart. Repeat the three steps when you change the questions or add emails, because saved answers only match the exact email, attachments, and questions they were made with.

**A plain Linux server, without Docker**

Create `.env` next to `server.py` with `AI_GATEWAY_API_KEY=vck_...` and `RFQ_DEMO_PASSWORD=...`, then run `python3 server.py --host 0.0.0.0 --no-browser`. Put HTTPS in front of it (Caddy does this in a few lines). Or skip the public address: run `python3 server.py --no-browser` (it keeps the default 127.0.0.1), connect with `ssh -L 8765:127.0.0.1:8765 you@your-server`, and open http://127.0.0.1:8765 on your laptop. That way no password is needed.

## Cost and the free tier

- Jev on AI Gateway was free until September 25, 2026. It now costs $0.042 per million input tokens, and output tokens are free. Each email, with its attachment text, is roughly 2,000 input tokens, so a full 100-email run costs under a cent.
- The AI Gateway **free tier** only allows a handful of Jev calls every few minutes. The demo handles this for you: it shows a countdown, paces itself at about one call a minute, and keeps going. A full run on the free tier takes about 22 minutes for the beta inbox and about 100 minutes for the 100-email inbox. **Buying any amount of AI Gateway credits removes the limit.**
- Every Jev answer is saved to `cache\jev_results.json`, and `data/saved_results.json` (see above) seeds the cache at startup. After one full run, the demo replays instantly, even offline. **Do the first full run well before a meeting, then download and commit the saved results.**

## A 5-minute demo

1. **The problem.** Every RFQ lands in one inbox. Someone senior triages it each morning. ITAR mail gets forwarded around, and missing drawings get noticed days later.
2. **Click Route inbox.** The whole inbox gets sorted in seconds. Each email is a single Jev call with 8 questions about the email and its attachments, answered in a few hundred milliseconds.
3. **Open an email with attachments.** The files sit under the body as thumbnail tiles. Click the drawing to read its title block, material, finish, and notes, and click the STEP file to turn the part in 3D. The panel beside each file shows exactly what Jev read from it.
4. **Open the Corvane housing RFQ (E61).** The email reads like any commercial RFQ. The ITAR warning is printed only on the attached drawing, and Jev still sent it to the restricted queue. The trace names the file the warning came from. In the beta inbox that drawing is a scan with no text layer, so the warning was read with OCR. E72 (marking only on the RFQ form) and E52 (a CUI banner on the drawing) show the same catch.
5. **Open the Kestrel manifold RFQ (E32).** The email has no quantities. They are on the attached RFQ form, so nothing gets flagged as missing. Compare the heat sink RFQ (E07), where Jev caught the missing quantity and the estimator gets the RFQ with the missing-info request already drafted.
6. **Open Needs Human Review.** Jev says when it's unsure, and you can see the probabilities split. In Settings, slide the confidence bar up and more mail goes to a person. Slide it down and more is automated. The shop picks the risk level.
7. **Paste a live RFQ.** Use the prospect's own email and drop in their PDF drawing or a photo (up to 5 files, 10 MB each), or pick the example "ITAR notice hidden in the signature."
8. **Show the RFQ details.** Scroll an RFQ's drawer to **RFQ details**: every value says where it came from, and values read by OCR carry an **OCR** chip with Tesseract's confidence. Click **RFQ details** in the top bar to download every RFQ as one spreadsheet. In a scanned file's viewer, **Detected regions** shows the title block, notes, and legends the model found.
9. **Close:** "Jev makes the calls, your rules route them. Every decision has a probability and a paper trail, and there's no text for a model to make up."

Turn on **Show the answer key** in Settings to score Jev against the expected routes for the sample emails. One tricky case on purpose, in the 100-email inbox: E25 is a plating vendor that proudly says it is ITAR registered. It belongs in Filtered, because a vendor pitch is not controlled technical data.

## How it works

For each email:

1. **One Jev call.** It sends the state (sender, subject, body, and each attachment's file name and text) with 8 questions:

   | Question | Type |
   | --- | --- |
   | What kind of email is this? (new RFQ, quote revision, PO, order follow-up, vendor, other) | choice |
   | Which estimator should quote it? (3-axis, 5-axis, turning, mixed or unclear) | choice |
   | Do the email or its attachments say it is export-controlled (ITAR, EAR, CUI)? | yes/no |
   | How urgent is it? (0 to 3) | score |
   | Do the email or its attachments state a quantity? | yes/no |
   | Do the email or its attachments provide drawings or CAD? | yes/no |
   | Do the email or its attachments call for a finish or outside processing? | yes/no |
   | What production volume? | choice |

   **Attachment text.** In the 100-email inbox, sample files are built from specs in `data/sample_emails.json`, so Jev reads the same title block, notes, legends, and RFQ-form quantities you see in the viewer. In the beta inbox the attachments are real files, and their text comes from the PDF text layer, OCR, or the STEP header (see below). Uploaded PDFs are read with pypdf. Uploaded scans and images are read with OCR when Tesseract is installed; otherwise Jev sees only their file names. Each file gets up to 1,800 characters and all attachments together up to 6,000, so one long PDF cannot crowd out the email.

2. **Code decides the route** (`router.py`, `decide()`):
   - If ITAR / CUI likelihood is 50% or more, the email goes to the **restricted queue**. This check runs first.
   - If Jev's confidence on the email type is below the bar, the email goes to **Needs Human Review**. When Jev splits between two types that go to the same place (a PO or an order follow-up, a new RFQ or a quote revision), their combined probability counts. A split between vendor and "other" does not, because filtering archives an email with no reply.
   - Vendor pitches and job seekers are **filtered**. POs and order questions go to **Orders & CS**.
   - RFQs go to the **estimator** Jev picked. If the work is mixed or Jev isn't confident enough, the RFQ goes to review.
   - **Flags:** missing drawings (code checks attachment types, Jev checks for links and "drawing to follow"), missing quantity, rush, outside processing, and volume.
   - **Paper trail:** when an attachment carries export-control wording, the trace names the file, so a reviewer can see the marking was on the drawing and not in the email.
   - **Priority and quote-by date:** based on rush, customer tier (looked up in `shop_config.json`), volume, and `sla_business_days`.
   - **Reply draft:** filled in from a template by code.

## RFQ details beta

The beta pulls the details a quoting manager needs out of every RFQ and its files, and puts them in one consolidated file. Some of the files cannot be copied from, so their text is read with Tesseract, using settings chosen by measurement. A YOLO model finds the regions of each page (title block, notes, legends, form tables) and tells the extractor where each value was printed.

### The beta inbox

`data/rfq_beta/emails.json` holds 22 of the sample emails. 19 of them are RFQs: 16 with files, and 3 (E04, E06, E26) without. The attachments are 30 real files in `data/rfq_beta/files/`, 30 because that was the limit for this beta:

- 17 normal files: 11 typed drawing PDFs with a text layer, and 6 STEP models.
- 13 uncopyable files, with no text layer at all:
  - 7 office scans (300 dpi gray, a little skewed, speckled): E01, E16, E20 (FR-3102), E32 (RFQ form), E56 (RFQ form), E61 (ITAR drawing), E72 (ITAR RFQ form).
  - 2 copier scans (200 dpi, darker, a shadow along the edge): E09, E62 (RFQ form).
  - 2 faxes (200 dpi, black and white, salt noise): E07, E52 (CUI drawing).
  - A phone photo of a printed drawing on a desk: E22.
  - A screenshot of a PDF viewer, at about 120 dpi: E71.

`tools/make_rfq_beta.py` made them. `tests/rfq_beta_truth.json` lists every word printed on each file. Only the tests and the OCR evaluation read it, never the demo.

### How the text is read

`ocr.py` reads each attachment the way a mail system would:

- **Text layer.** A PDF with real text is read with pypdf, in a separate process with a time limit. A page with almost no text (under 40 letters and digits, or only a scanner's stamp over a full-page picture) counts as a scan.
- **OCR.** Scans, faxes, photos, and screenshots go to Tesseract (the LSTM engine, English). The pipeline decides what kind of source a page is from the file itself, never from its name: a 1-bit page is a fax, a gray page of 250 dpi or more is an office scan, a lower one is a copier scan, a sheet of paper on a background is a photo, and a sheet square to the picture is a screenshot. Each kind has its own settings:

  | Source | Settings |
  | --- | --- |
  | Office scan | its own resolution, dark table headers inverted, `--psm 3` then `--psm 11` merged by confidence, words under 60% confidence dropped |
  | Copier scan | upscaled to 400 dpi, headers inverted, the edge shadow flattened, `--psm 4` |
  | Fax | no resize, `--psm 11` then `--psm 3` merged by confidence |
  | Phone photo | the sheet found and its perspective flattened at 300 dpi, `--psm 11` then `--psm 3`, words under 60% dropped |
  | Screenshot | upscaled to 300 dpi, `--psm 11`, words under 60% dropped |

  All five fix letters Tesseract confuses in capitals ("Cl-10442" becomes "CI-10442", a lone "Cc" becomes "C"). On the 13 uncopyable files these settings find 81 of 81 key fields (part numbers, revs, materials, finishes, quantities, RFQ numbers, respond-by dates, export legends), against 71 with plain Tesseract. On 25 held-out pages the settings search never saw, they find 83 of 100, against 77 for the pipeline as it was before. They cost about 2.8 CPU seconds a page. `docs/ocr_settings.md` has every setting tried, the scores, the time, and the limits. The larger `tessdata_best` model was tried too: it lost 4 key fields and took twice the time, so the image keeps Debian's model.
- **STEP header.** A STEP model gives its part number, title, and units from its header (`stepfile.py`).

**The committed OCR cache.** `data/rfq_beta/ocr_cache.json` holds the result for all 30 files, keyed by the SHA-256 of the file bytes: 13 OCR results, 11 text layers, and 6 STEP headers, with the Tesseract version (5.3.4) and the settings that made them. The server loads it at start, so it runs no OCR and no pypdf for the beta files. Render needs it. The free plan has about a tenth of a CPU, where OCR takes about half a minute a page and all 13 files about 6 minutes, and every restart, redeploy, or wake from sleep starts with an empty disk. With the cache, every file shows its text at once, even on a host without Tesseract. It also keeps the demo the same everywhere: Render's Tesseract 5.5.0 finds the same key fields as 5.3.4, but not always the same words. A beta file whose bytes change is not in the cache, so it is read live and saved in `cache/ocr_cache.json`.

**Uploads** in Paste an RFQ are read with the fast settings (`RFQ_UPLOAD_OCR_EFFORT=fast`): one Tesseract pass a page, 80 of the 81 key fields, and about a third less time (1.8 CPU seconds a page). One OCR job runs at a time (`RFQ_OCR_WORKERS`), a file gets up to 300 seconds (`RFQ_OCR_TIMEOUT`), and at most 6 pages are read.

### The region detector (YOLO)

`layout.py` runs a small YOLO model, `models/rfq_layout.onnx` (YOLO11n, 10.6 MB). It finds 8 kinds of region on a page: title block, revision block, notes, export legend, proprietary notice, RFQ form header, line table, and requirements. It was fine-tuned on 1,400 synthetic pages drawn by the demo's own renderer and passed through the same scan, copier, fax, photo, and screenshot effects as the beta files. Nobody drew boxes by hand: each label comes from the drawing operations that made the page. The beta files were never used for training. On their 24 pages the model finds all 88 labeled regions with no false boxes (mAP50 0.995, mAP50-95 0.941). It takes about 0.1 CPU seconds a page on one thread, and at run time it needs only numpy, onnxruntime, and Pillow, not torch. It learned this renderer's layouts, so it needs real labeled pages before it can be trusted on customer drawings. `docs/layout_model.md` has the data, training, metrics, and limits.

**How it pairs with OCR.** Every page that is OCR'd also goes through the detector, on the same picture Tesseract read, so each line of OCR text knows the region it was printed in. The regions are saved with the OCR results and in the committed cache. What was kept, after measuring each step:

- **Where each value was printed.** The extractor names the region in a value's source: `CI-10442_RevC.pdf, title block (OCR 88%)`, or `CI-10442_RevC.pdf, export legend (OCR 88%)` for the ITAR marking. 85 of the 89 values read by OCR name their region. The other 4 are the TERMS lines of the RFQ forms, which no region class covers.
- **Four reading rules** in `rfq_details.py`: revision rows come only from the revision block; the REV cell inside the title block gives the rev, even when the drawing number beside it is unreadable; pieces of one title block value that OCR split on one line are joined; and a REV cell read as "Ce" is rev C. They change nothing on the real files (504 of 504 fields before and after). On 55 held-out pages they fix 5 of the 7 wrong fields (1,763 to 1,768 of 1,770), and no field that was right before is wrong after.

What was not kept: region rules that changed nothing (reading title block, form header, and line table fields from their regions first, matching export legends loosely only inside the legend box), and one rule that lost a field. Reading the title block and line table again as cropped regions with a second Tesseract pass is still in the code, switched off: it found no field the extractor did not already have, for 12 to 29% more CPU. The detector adds 0.10 to 0.13 CPU seconds to each OCR'd page, and nothing for text layers, STEP files, or anything in the cache. `RFQ_OCR_REGIONS=0` turns the pairing off.

### The consolidated file

`rfq_details.py` makes one record per RFQ and writes them all as one file, with one row per part line.

- **Download it** with **RFQ details** in the top bar, with **Download all (CSV)** in an email's RFQ details, or from the CSV and JSON links in Settings. The addresses are `/api/rfq_details.csv` (UTF-8 with a byte order mark, so Excel opens it correctly) and `/api/rfq_details.json` (the same records, every value with its source). `/api/rfq_details/E32` returns one record.
- **Columns:** Email, Received, Customer, Tier, Contact, Contact email, RFQ number, Quote ref, Request, Respond by, Delivery, Terms, Line, Part number, Rev, Description, Material, Finish, Size, Quantities, Annual usage, Export control, Export control found in, Requirements, Files, Missing info, Check, Lane, Estimator, Priority, Quote by. The last four fill in once Jev has routed the email. Files lists each attachment with its type and how its text was read (`RFQ-26-0931.pdf (RFQ form, scan, OCR 93%)`).
- **Sources.** Every value says where it came from: the subject, the email body, or a file and how it was read (`RFQ-26-0931.pdf, form header (OCR 93%)`). When sources disagree, the RFQ form wins for quantities and dates, the drawing title block wins for part number, rev, material, finish, and description, and the email counts when it is the only source. Each disagreement goes in Check, and so does an OCR value under 60% confidence that no other source confirms. Nothing is invented: a value that is not found stays empty.
- **In the drawer,** an RFQ's details have their own section, **RFQ details**. Hover a value to see its source, or click **Sources** to show them all. A value read by OCR has an **OCR** chip with Tesseract's confidence, in amber below 70%. For a part number, rev, description, material, or finish that is the confidence of the line it was read from, which can be far below the file's (E16's AW-310 was read at 42% on a page read at 92%); for other values it is the file's. The JSON carries a value's own confidence as `conf` next to its source.
- **Dates.** The sample emails carry a time but no date, so relative dates ("within two weeks", "today", "next Friday") resolve against the server's date, and the words they came from are kept. On September 26, 2026 the download gives E03 ("within two weeks") a respond-by date of 2026-10-10. The committed `data/rfq_beta/rfq_details.csv` was made with `python rfq_details.py`, which resolves against September 25, 2026, the day the inbox was built, and says 2026-10-09 (`--today` picks another date).
- **Which emails.** Until an email is routed, a keyword check decides whether it is an RFQ. Once Jev has routed it, Jev's answer decides: a new RFQ or a quote revision counts, even when Jev was unsure and sent it to review. So the download can hold an RFQ the committed file leaves out. With `tests/mock_jev.py`, E17 ("What would something like this cost?", no part named) counts as one: its row names no part, lists the quantity, drawing, and material as missing, and shows the review lane.
- **Accuracy.** `python rfq_details.py --check` grades all 19 RFQs against `tests/rfq_beta_fields_truth.json`: 504 of 504 fields.

### Detected regions

In the file viewer, a PDF or an image has a **Detected regions** button when the server can run the detector. It swaps the file for page 1 as a picture, with the detector's boxes drawn on it: one color per class, each labeled with its name and confidence, and a legend underneath. The server detects on the 150 dpi page image it shows (`/api/att/<email>/<n>/page.jpg` and `regions.json`, or `/api/upload/<id>/page.jpg` and `regions.json` for an upload), and remembers the answer for each page. Only page 1 is shown. These boxes are for looking at; the extractor uses the ones saved with the OCR results.

### Dependencies and running without them

The server is still standard library only. The beta uses these, and each one is optional:

| Install | What it adds |
| --- | --- |
| `apt install tesseract-ocr tesseract-ocr-eng` | OCR for files that are not in the committed cache: uploaded scans and photos, or a beta file that changed |
| `apt install poppler-utils` | `pdftoppm`, which turns PDF pages into pictures: PDF thumbnails, the page image for Detected regions, and the rendering the OCR settings were tuned on |
| `pip install -r requirements.txt` | pypdf (text layers of uploaded PDFs), Pillow (image thumbnails, clean-up before OCR), numpy and onnxruntime (the region detector) |

The Docker image installs all of them. On Windows or a Mac, install Tesseract 5 and poppler from your usual package source, or leave them out.

**Without them,** the committed cache still covers all 30 beta files. A server started with plain Python and no programs on its PATH showed the text of every beta file, and its `/api/rfq_details.csv` was byte for byte the same as a full install's. What you lose:

- **No Tesseract:** uploaded scans and photos count by their file names only. The start banner says "Tesseract not found; scans use the committed OCR cache only".
- **No poppler:** no PDF thumbnails (the tile shows an icon) and no page image for Detected regions on PDFs. OCR of a scanned upload falls back to the pictures pypdf pulls out of the PDF (74 of 81 key fields).
- **No pypdf:** an uploaded PDF counts by its file name only.
- **No Pillow:** no image thumbnails and no Detected regions. OCR runs without clean-up (71 of 81 key fields).
- **No numpy or onnxruntime, or no model file:** no Detected regions button, and new OCR results carry no regions.

Settings for the beta:

| Setting | What it does |
| --- | --- |
| `RFQ_EMAILS_FILE` | The inbox to open. The default is `data/rfq_beta/emails.json`. `data/sample_emails.json` is the 100-email inbox. |
| `RFQ_OCR_CACHE` | The committed OCR cache. The default is `data/rfq_beta/ocr_cache.json`. |
| `RFQ_UPLOAD_OCR_EFFORT` | `fast` (the default) or `best`: the OCR settings for uploads. |
| `RFQ_OCR_WORKERS` | How many OCR jobs run at once. The default is 1, because two on a tenth of a CPU only slow each other down. |
| `RFQ_OCR_TIMEOUT` | Seconds OCR may spend on one file. The default is 300. |
| `RFQ_OCR_REGIONS` | `0` turns off the region pairing in OCR. |
| `RFQ_LAYOUT_THREADS` | Threads for the region detector. The default is 1. |
| `RFQ_PAGE_WORKERS` | How many viewer page images are rendered at once, across all files. The default is 1, which keeps many viewers at once inside 512 MB. |

### Rebuild the OCR cache and the consolidated file

After changing a beta file, an OCR setting, or the model, rebuild both and commit them:

```
python ocr.py --build-cache
python rfq_details.py
python rfq_details.py --check
```

The first reads all 30 files into `data/rfq_beta/ocr_cache.json` (about 40 seconds here). The second writes `data/rfq_beta/rfq_details.csv` and `.json`, and the third grades them. Run `--build-cache` where Tesseract, poppler, Pillow, and the detector all work, so the entries carry their regions (a test checks that the committed cache has them). Other useful commands:

```
python ocr.py FILE                  # how one file is read: method, confidence, settings, text
python ocr.py --effort fast FILE    # the same with the upload settings
python ocr.py --evaluate            # score the settings on the 13 uncopyable files (about 2 minutes)
python layout.py FILE               # the regions the detector finds on page 1
```

`python tools/make_rfq_beta.py` rebuilds the beta inbox, its 30 files, and `tests/rfq_beta_truth.json` (it needs Pillow). The committed files were made before the last changes to the drawings, so a rerun changes them (the scan and fax noise is repeatable when the script runs as its own process). Rebuild the cache and rerun the checks after it.

### Retrain the region detector

Training needs torch and Ultralytics, in a separate virtualenv. The demo never does. From the repository root:

```
python -m venv ../yolo-venv && ../yolo-venv/bin/pip install torch ultralytics onnx onnxslim onnxruntime
../yolo-venv/bin/python tools/make_layout_dataset.py --out ../yolo_work/layout_data_v3 --train 1200 --val 200 --workers 3
../yolo-venv/bin/python tools/train_layout.py train --data ../yolo_work/layout_data_v3/data.yaml --base yolo11n.pt --name rfq_layout_v3 --epochs 3 --budget-min 40
../yolo-venv/bin/python tools/train_layout.py eval --data ../yolo_work/layout_data_v3/data.yaml --name rfq_layout_v3
../yolo-venv/bin/python tools/train_layout.py export --data ../yolo_work/layout_data_v3/data.yaml --name rfq_layout_v3
python tools/train_layout.py runtime --data ../yolo_work/layout_data_v3/data.yaml --beta-only
```

The dataset takes about 4.5 minutes on 3 cores. `--base yolo11n.pt` starts from Ultralytics' COCO weights (downloaded into `../yolo_work`); the shipped model was fine-tuned from earlier weights instead (pass that `best.pt` as `--base`). `--budget-min` is a wall-clock budget: Ultralytics re-plans the number of epochs after each one to fill it, so `--epochs 3` can become 8, or far more on a small dataset. Add `--no-time-cap` to run exactly `--epochs`. `export` writes `models/rfq_layout.onnx` and checks onnxruntime against Ultralytics, and `runtime` scores the new model through the server's own path (it runs with the demo's own Python, which needs only numpy, onnxruntime, and Pillow for it). Then rebuild the OCR cache, because the regions in it come from the model. With images cached in memory, training peaks near 6.5 GB; on a smaller machine add `--cache disk`. `docs/layout_model.md` has every step, flag, and timing.

**Network note.** In the Claude Code cloud sandbox, download.pytorch.org is blocked (the proxy answers 403), so PyTorch's own CPU-only wheels cannot be used there, and the `pip install` above takes torch from PyPI. On Linux that wheel brings NVIDIA's CUDA libraries with it: the training virtualenv came to about 6 GB (1.2 GB of torch 2.14.0 and 3.2 GB of CUDA packages). It trains on the CPU all the same.

### Licensing of the region detector

- Ultralytics YOLO is licensed under AGPL-3.0. Ultralytics' position is that models trained with it, and their exports, are covered too unless you hold an Ultralytics Enterprise License, and the ONNX file's own metadata says "AGPL-3.0 License". So treat `models/rfq_layout.onnx` as AGPL-3.0. The runtime contains no Ultralytics code (`layout.py` is this repository's own decoder), but that does not remove the claim on the weights.
- AGPL-3.0 has a network clause: people who use the software over a network, as they do on a public deployment such as the Render demo, must be offered its source code under AGPL-3.0. If the model counts as part of the service, that means the source of the whole service. For a private demo shown to a few people the practical risk is low.
- For a product or a public deployment, either buy an Ultralytics Enterprise License, or retrain with a permissively licensed detector. YOLOX (Apache-2.0) is in the same size class; `docs/layout_model.md` lists the four changes a switch takes.
- onnxruntime (MIT), numpy (BSD), and Pillow (MIT-CMU) add nothing beyond keeping their notices. The training pages are synthetic, drawn by this repository.
- onnxruntime sends usage events (model loads, sessions, the device) to Microsoft unless `ORT_DISABLE_TELEMETRY=1` is set before it loads. `layout.py` sets it, and so does the Dockerfile, so the demo calls no one but Jev. No document content was ever in those events.
- This is a summary, not legal advice.

## Files

| File | What it is |
| --- | --- |
| `run_demo.bat` | One-click start |
| `run_demo.sh` | The same start for macOS and Linux |
| `check_jev.bat` / `check_jev.py` | Key check: routes the first email of the inbox (E01) with one real call |
| `Dockerfile` | Container image for cloud hosts, with Tesseract and poppler (see Run it in the cloud) |
| `render.yaml` | One-click deploy settings for Render |
| `server.py` | Local web server and background Jev worker (rate limits, retries, cache, attachment, upload, RFQ details, and region endpoints) |
| `jev_client.py` | Standard-library Jev client (AI Gateway or TypeSafe direct) |
| `router.py` | The 8 questions and the routing policy |
| `attachments.py` | Sample files from specs, real files on disk, the text Jev reads from each file, page images and regions for the viewer, and in-memory uploads |
| `ocr.py` | Text from any file: the PDF text layer, Tesseract OCR with settings per kind of source, or the STEP header. Also the OCR cache and the evaluation harness |
| `rfq_details.py` | Pulls the details out of each RFQ and writes the consolidated file (CSV and JSON). `--check` grades it |
| `layout.py` | The YOLO region detector at run time (numpy and onnxruntime, no torch) |
| `stepfile.py` | Reads STEP files: header, part number, title, units, and the mesh for the 3D view |
| `drawings.py` | Engineering drawing sheets, 3D meshes, isometric views, and STEP files |
| `docgen.py` | A small page canvas that writes the same layout as a PDF and as an SVG thumbnail |
| `requirements.txt` | pypdf, Pillow, numpy, and onnxruntime (each optional when running locally) |
| `shop_config.json` | Lanes, owners, customer tiers, thresholds, SLAs |
| `models/rfq_layout.onnx` | The region detector: YOLO11n fine-tuned on synthetic pages, 10.6 MB, AGPL-3.0 (see Licensing) |
| `data/sample_emails.json` | 100 fictional emails with attachment specs, the answer key, and paste examples |
| `data/rfq_beta/` | The beta inbox: `emails.json`, the 30 files in `files/`, the committed `ocr_cache.json`, and `rfq_details.csv` and `.json` |
| `data/saved_results.json` | Saved Jev answers you commit (Settings, Download saved results), loaded at startup |
| `docs/` | `ocr_settings.md` (how the OCR settings were chosen, with every measurement) and `layout_model.md` (the detector's data, training, metrics, and licensing) |
| `static/index.html` | The dashboard |
| `tools/make_rfq_beta.py` | Builds the beta inbox, its 30 files, and `tests/rfq_beta_truth.json` |
| `tools/make_layout_dataset.py` / `tools/train_layout.py` | Build the detector's synthetic training pages; train, evaluate, export, and time the detector |
| `tests/` | The test suites (see Test it without a key), `mock_jev.py` (a stand-in Jev endpoint), and the answer keys for the beta |
| `cache/` | Saved Jev results and live OCR results (created on first run) |

## Customize

- **Another shop** (for example sheet metal): edit the `estimating` lanes in `shop_config.json`. Each lane's `jev_description` is the text Jev uses to pick it.
- **Different questions:** edit `build_questions()` in `router.py`. The cache is keyed on the exact question text, so an edit triggers fresh calls automatically.
- **Customers and tiers:** edit `customers` in `shop_config.json`. Senders are matched by email domain.
- **Sample emails and attachments:** edit `data/sample_emails.json`. Each attachment is a spec (`drawing`, `model`, `rfq_form`, `po`, or `document`) that the demo turns into the file, its thumbnail, and the text Jev reads. Drawings take a `shape`, a `size`, `notes`, and an optional `legend` (`itar`, `ear`, `cui`, or `proprietary`).
- **The beta inbox:** change `tools/make_rfq_beta.py` and rerun it, then rebuild the OCR cache (see [Rebuild](#rebuild-the-ocr-cache-and-the-consolidated-file)).
- **Thresholds:** use the Settings panel for live changes, or set defaults under `thresholds` in `shop_config.json`.
- **Direct TypeSafe key:** put `TYPESAFE_API_KEY=...` in `.env` and remove `AI_GATEWAY_API_KEY`. The client switches to `https://api.typesafe.ai` with model `jev-latest`.
- **Port:** `run_demo.bat --port 9000`

## Using Jev in your own code

Everything here is one HTTP call: `POST https://ai-gateway.vercel.sh/typesafe/v1/systemone` with `Authorization: Bearer <AI_GATEWAY_API_KEY>`. You can also use the official SDK:

```python
# pip install typesafe-sdk
import os
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(
    api_key=os.environ["AI_GATEWAY_API_KEY"],
    base_url="https://ai-gateway.vercel.sh/typesafe",
    model="typesafe-ai/jev",
)
result = client.system_one(
    state={"subject": "RFQ: bracket", "body": "Please quote 50 pcs, drawing attached."},
    questions={
        "is_rfq": Noul(instructions="Is the sender asking for a price quote?"),
        "machine": Choice(
            instructions="Which machine would make this part?",
            criteria={"mill": "Prismatic parts", "lathe": "Round parts"},
        ),
    },
)
print(result.answers["is_rfq"].noul, result.answers["machine"].choice)
```

## Troubleshooting

| You see | Do this |
| --- | --- |
| `401` or "key was rejected" | Check `AI_GATEWAY_API_KEY` in `Jev_Test\.env`, or make a new key in Vercel under AI Gateway, then API Keys |
| `402` or a billing message | Add AI Gateway credits in the Vercel dashboard |
| A countdown banner | That's the free-tier rate limit. Wait, or buy credits to remove it |
| "Could not reach ai-gateway.vercel.sh" | Check your internet connection, VPN, or firewall |
| The browser didn't open | Go to the address printed in the black window |
| You want a clean slate | Close the demo and delete the `cache` folder |
| The browser asks for a user name and password | The demo is password protected. Type any user name and the `RFQ_DEMO_PASSWORD` value |
| "Set RFQ_DEMO_PASSWORD before serving on ..." | Set it, or listen on 127.0.0.1 only |
| `forbidden host` | You reached a server that listens on 127.0.0.1 through a proxy or a forwarded port. Set `RFQ_DEMO_PASSWORD`, which switches the server to password checks |
| An uploaded file's text says "there is no text to read" or "no text could be read from it" | It is a scan or a photo and Tesseract is not installed, so Jev sees only its name. Install Tesseract and poppler (see [Dependencies](#dependencies-and-running-without-them)) |
| "This server cannot read PDF text (pypdf is not installed)" | Run `pip install -r requirements.txt` |
| An uploaded scan says "the server is busy reading other scans" | OCR runs one file at a time, and this one waited longer than its time limit. Try again, or raise `RFQ_OCR_WORKERS` on a bigger host |
| No **Detected regions** button | The server cannot run the detector. Run `pip install -r requirements.txt` (numpy, onnxruntime, Pillow) and restart |
| Detected regions says the page image could not be made | For PDFs, install poppler (`pdftoppm`) |
| The downloaded RFQ details differ from `data/rfq_beta/rfq_details.csv` | Relative dates ("within two weeks") resolve against the server's date, and the committed file uses September 25, 2026. After routing, Jev decides which emails are RFQs (see [The consolidated file](#the-consolidated-file)) |
| Replays are slow again after a cloud restart | Commit a fresh `data/saved_results.json` (Settings, Download saved results). Saved answers only match the exact emails, attachments, and questions they were made with |

## Test it without a key

`tests/mock_jev.py` stands in for the Jev endpoint with rough keyword rules, so you can rehearse the demo or run the tests without spending calls:

```
python tests/mock_jev.py --port 8799
JEV_BASE_URL=http://127.0.0.1:8799 AI_GATEWAY_API_KEY=mock python server.py
python -m unittest discover -s tests -v
```

The mock's answers are guesses, not Jev's, so do not judge accuracy with it.

The suite has 200 tests and takes about two and a half minutes:

| File | What it tests |
| --- | --- |
| `tests/test_demo.py` | The 100 sample emails and their files, the routing rules, and the server: the password and `/healthz`, routing and replay, uploads, free-tier pacing (it starts its own mock Jev and server) |
| `tests/test_beta.py` | The beta inbox (30 files, 13 uncopyable), markings found only on scans, the RFQ details download, the committed cache without Tesseract, an uploaded scan read with OCR, and uploaded scans and photos whose title block values reach the RFQ details |
| `tests/test_ocr.py` | The OCR pipeline: text layers skip OCR, every uncopyable file gives its key fields, broken files give an error instead of a crash |
| `tests/test_rfq_details.py` | The extractor and the consolidated file, against `tests/rfq_beta_fields_truth.json` |
| `tests/test_layout.py` | The region detector: the model loads, finds the right regions, and handles broken input |

The tests that need Tesseract, poppler, Pillow, numpy, or onnxruntime skip when those are missing: with plain Python and only poppler installed, the suite passes with 70 of the 200 skipped, and with neither poppler nor Tesseract with 72 skipped.

Jev reads text: the email and the text of its attachments (sample files from their specs, real files from their text layer or OCR, uploaded PDFs through pypdf). It never sees pixels, so a scan or a photo counts through the text OCR reads from it, or only by its file name when Tesseract is not installed. All companies, people, and emails in the sample and beta inboxes are fictional.
