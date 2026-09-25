# RFQ Router demo (Jev)

A local demo that sorts a CNC shop's shared quoting inbox with **Jev**, TypeSafe AI's decision model, called through your Vercel AI Gateway key.

The sample inbox holds 100 fictional emails with realistic attachments: engineering drawings with title blocks, material, finish, notes, and ITAR or CUI legends, STEP 3D models, customer RFQ forms, and purchase orders. Open an email to see its files as thumbnail tiles, the way Gmail shows them. Click one to open it: PDFs in a viewer, images inline, and STEP files as a 3D model you can turn. Jev reads the email together with the text of its attachments, so an ITAR marking that only appears on the drawing still sends the email to the restricted queue.

Jev doesn't write text. It answers typed questions (pick one, yes/no, score) and gives a probability for each answer. That makes it a good fit for routing: Jev makes the judgment calls, and plain Python applies the shop's rules.

## Run it

1. Double-click **`run_demo.bat`**.
   - First run only: it checks your key with one real call (`check_jev.py`) and saves that result.
   - Then it opens **http://127.0.0.1:8765** in your browser.
   - Nothing to install. It uses only Python's standard library. You need Python 3.10 or newer.
   - Optional: `pip install -r requirements.txt` adds pypdf, which lets Jev read the text of PDFs you upload in **Paste an RFQ**. Without it, Jev sees only the names of uploaded PDFs. The sample files work either way.
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

These can also go in `.env` next to `server.py`.

**Render, in one click**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/JordanKing22/RFQ_Routing)

1. Click the button and sign in to Render (a free account works).
2. When Render asks, paste your AI Gateway key into `AI_GATEWAY_API_KEY` and choose a password for `RFQ_DEMO_PASSWORD`.
3. Click **Deploy Blueprint**. The first build takes a few minutes.
4. Open the `https://....onrender.com` address on the service page.

`render.yaml` holds these settings. The free plan sleeps after 15 idle minutes and takes up to a minute to wake, so open the demo a minute before a meeting. The Starter plan stays awake.

**Docker**

```
docker build -t rfq-router .
docker run --rm -p 8765:8765 -e AI_GATEWAY_API_KEY=vck_... -e RFQ_DEMO_PASSWORD=pick-one rfq-router
```

Then open http://localhost:8765. To keep saved Jev results between runs, add `-v rfq-cache:/app/cache`. The image installs pypdf from `requirements.txt`.

**A container host** (Google Cloud Run, Render, Railway, Fly.io, and others)

1. Point the host at this repository. It builds the `Dockerfile`.
2. Add `AI_GATEWAY_API_KEY` and `RFQ_DEMO_PASSWORD` as secrets or environment variables.
3. If the host asks for a health check path, use `/healthz`. It is the only address that needs no password.
4. Open the HTTPS address the host gives you.

**Keep saved results through restarts.** New Jev answers are saved in the container, and a restart, a redeploy, or a free host waking from sleep starts with an empty disk. So the demo also loads `data/saved_results.json` from the repository at startup, and every saved answer in it replays instantly:

1. Route the whole inbox once on the deployed demo (on the free tier this takes about 100 minutes, see below).
2. Open **Settings** and click **Download saved results**.
3. Save the file as `data/saved_results.json` in the repository, then commit and push. Render redeploys on its own.

After that, **Route inbox** replays all 100 emails in a few seconds, even right after a restart. Repeat the three steps when you change the questions or add emails, because saved answers only match the exact email, attachments, and questions they were made with.

**A plain Linux server, without Docker**

Create `.env` next to `server.py` with `AI_GATEWAY_API_KEY=vck_...` and `RFQ_DEMO_PASSWORD=...`, then run `python3 server.py --host 0.0.0.0 --no-browser`. Put HTTPS in front of it (Caddy does this in a few lines). Or skip the public address: run `python3 server.py --no-browser` (it keeps the default 127.0.0.1), connect with `ssh -L 8765:127.0.0.1:8765 you@your-server`, and open http://127.0.0.1:8765 on your laptop. That way no password is needed.

## Cost and the free tier

- Jev on AI Gateway is free until September 25, 2026. After that it costs $0.042 per million input tokens, and output tokens are free. Each email, with its attachment text, is roughly 2,000 input tokens, so a full 100-email run costs under a cent.
- The AI Gateway **free tier** only allows a handful of Jev calls every few minutes. The demo handles this for you: it shows a countdown, paces itself at about one call a minute, and keeps going. A full 100-email run on the free tier takes about 100 minutes. **Buying any amount of AI Gateway credits removes the limit.**
- Every Jev answer is saved to `cache\jev_results.json`, and `data/saved_results.json` (see above) seeds the cache at startup. After one full run, the demo replays instantly, even offline. **Do the first full run well before a meeting, then download and commit the saved results.**

## A 5-minute demo

1. **The problem.** Every RFQ lands in one inbox. Someone senior triages it each morning. ITAR mail gets forwarded around, and missing drawings get noticed days later.
2. **Click Route inbox.** Twenty emails get sorted in seconds. Each one is a single Jev call with 8 questions, answered in a few hundred milliseconds.
3. **Open an ITAR card.** Export control is checked first, at a deliberately low bar. The email skips the shared queues, and the reply draft avoids technical details.
4. **Open the heat sink RFQ.** Jev caught the missing quantity. The estimator gets the RFQ with the missing-info request already drafted.
5. **Open Needs Human Review.** Jev says when it's unsure, and you can see the probabilities split. In Settings, slide the confidence bar up and more mail goes to a person. Slide it down and more is automated. The shop picks the risk level.
6. **Paste a live RFQ.** Use the prospect's own email, or pick the example "ITAR notice hidden in the signature."
7. **Close:** "Jev makes the calls, your rules route them. Every decision has a probability and a paper trail, and there's no text for a model to make up."

Turn on **Show the answer key** in Settings to score Jev against the expected routes for the 20 sample emails.

## How it works

For each email:

1. **One Jev call.** It sends the state (sender, subject, body, attachment file names) with 8 questions:

   | Question | Type |
   | --- | --- |
   | What kind of email is this? (new RFQ, quote revision, PO, order follow-up, vendor, other) | choice |
   | Which estimator should quote it? (3-axis, 5-axis, turning, mixed or unclear) | choice |
   | Is it export-controlled (ITAR / CUI)? | yes/no |
   | How urgent is it? (0 to 3) | score |
   | Is a quantity stated? | yes/no |
   | Are drawings or CAD provided? | yes/no |
   | Does it need finishing or outside processing? | yes/no |
   | What production volume? | choice |

2. **Code decides the route** (`router.py`, `decide()`):
   - If ITAR / CUI likelihood is 50% or more, the email goes to the **restricted queue**. This check runs first.
   - If Jev's confidence on the email type is below the bar, the email goes to **Needs Human Review**.
   - Vendor pitches and job seekers are **filtered**. POs and order questions go to **Orders & CS**.
   - RFQs go to the **estimator** Jev picked. If the work is mixed or Jev isn't confident enough, the RFQ goes to review.
   - **Flags:** missing drawings (code checks attachment types, Jev checks for links and "drawing to follow"), missing quantity, rush, outside processing, and volume.
   - **Priority and quote-by date:** based on rush, customer tier (looked up in `shop_config.json`), volume, and `sla_business_days`.
   - **Reply draft:** filled in from a template by code.

## Files

| File | What it is |
| --- | --- |
| `run_demo.bat` | One-click start |
| `run_demo.sh` | The same start for macOS and Linux |
| `check_jev.bat` / `check_jev.py` | Key check: routes sample email E01 with one real call |
| `Dockerfile` | Container image for cloud hosts (see Run it in the cloud) |
| `render.yaml` | One-click deploy settings for Render |
| `server.py` | Local web server and background Jev worker (rate limits, retries, cache) |
| `jev_client.py` | Standard-library Jev client (AI Gateway or TypeSafe direct) |
| `router.py` | The 8 questions and the routing policy |
| `shop_config.json` | Lanes, owners, customer tiers, thresholds, SLAs |
| `data/sample_emails.json` | 20 fictional emails, the answer key, and paste examples |
| `static/index.html` | The dashboard |
| `cache/` | Saved Jev results (created on first run) |

## Customize

- **Another shop** (for example sheet metal): edit the `estimating` lanes in `shop_config.json`. Each lane's `jev_description` is the text Jev uses to pick it.
- **Different questions:** edit `build_questions()` in `router.py`. The cache is keyed on the exact question text, so an edit triggers fresh calls automatically.
- **Customers and tiers:** edit `customers` in `shop_config.json`. Senders are matched by email domain.
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

Jev reads only the text of each email. Attachments are passed as file names, so Jev never sees what's inside a drawing. All companies, people, and emails in the sample inbox are fictional.
