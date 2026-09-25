"""
RFQ Router demo server. Standard library only.

    python server.py            start on http://127.0.0.1:8765 and open the browser
    python server.py --port 9000 --no-browser
    RFQ_DEMO_PASSWORD=... python server.py --host 0.0.0.0 --no-browser
                                serve other machines too (cloud); every request needs the password

The browser never sees your API key: the page talks to this local server, and
this server talks to Jev.

Saved Jev answers: cache/jev_results.json (written as results come in) on top of
data/saved_results.json (a read-only seed you commit, so replays stay instant after a cloud
host restarts). Settings > Download saved results gives you that seed file.

RFQ details beta: the inbox is data/rfq_beta/emails.json, whose attachments are 30 real files. Their
text comes from the PDF text layer or, for the uncopyable scans, faxes, photos, and screenshots,
from Tesseract OCR (ocr.py), with results cached in data/rfq_beta/ocr_cache.json. rfq_details.py
pulls the details out of every RFQ; /api/rfq_details.csv is the consolidated file.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import queue as queue_mod
import re
import secrets
import signal
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

import attachments as att_mod
import jev_client
import router

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
CONFIG_FILE = HERE / "shop_config.json"
DATA_DIR = HERE / "data"
# This branch opens the RFQ details beta inbox. RFQ_EMAILS_FILE=data/sample_emails.json brings back
# the 100-email inbox with generated attachments.
EMAILS_FILE = Path(os.environ.get("RFQ_EMAILS_FILE") or DATA_DIR / "rfq_beta" / "emails.json")
CACHE_FILE = Path(os.environ.get("RFQ_CACHE_DIR") or HERE / "cache") / "jev_results.json"
SEED_FILE = Path(os.environ.get("RFQ_SEED_FILE") or DATA_DIR / "saved_results.json")
OCR_SEED_FILE = Path(os.environ.get("RFQ_OCR_CACHE") or DATA_DIR / "rfq_beta" / "ocr_cache.json")
OCR_LIVE_FILE = CACHE_FILE.parent / "ocr_cache.json"

try:
    import rfq_details
except Exception:  # noqa: BLE001 - the demo still routes without the details extractor
    rfq_details = None  # type: ignore[assignment]

# Stop the queue on these: retrying cannot fix them.
FATAL_KINDS = {"auth", "billing", "validation", "network", "not_found", "config", "bad_response"}
# AI Gateway's free tier allows only a handful of Jev calls every few minutes (Vercel does not
# publish the number). After the first 429 the worker paces itself to one call per this many seconds.
FREE_TIER_PACE = float(os.environ.get("JEV_FREE_TIER_PACE_SECONDS", "61"))
RATE_LIMIT_BACKOFF = [60.0, 60.0, 90.0, 120.0, 180.0, 300.0]
try:
    RATE_LIMIT_BACKOFF = [float(os.environ["JEV_RATE_LIMIT_WAIT_SECONDS"])] * 6  # testing hook
except (KeyError, ValueError):
    pass


# --------------------------------------------------------------------------- #
# Logging. A background thread does the printing, so a console that blocks
# output (Windows QuickEdit selection) can never freeze the Jev worker.
# --------------------------------------------------------------------------- #
_log_queue: "queue_mod.Queue[str]" = queue_mod.Queue()


def _log_printer() -> None:
    while True:
        line = _log_queue.get()
        try:
            print(line, flush=True)
        except Exception:  # noqa: BLE001 - never let logging kill anything
            pass


threading.Thread(target=_log_printer, name="log-printer", daemon=True).start()


def log(message: str) -> None:
    _log_queue.put(f"[{time.strftime('%H:%M:%S')}] {message}")


def disable_quickedit() -> None:
    """Windows: clicking in the console window pauses output while QuickEdit is on. Turn it off."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            enable_extended_flags, enable_quick_edit = 0x0080, 0x0040
            kernel32.SetConsoleMode(handle, (mode.value | enable_extended_flags) & ~enable_quick_edit)
    except Exception:  # noqa: BLE001 - cosmetic, best effort
        pass


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------- #
# Result cache: every live Jev answer is saved, so a demo can be replayed
# instantly and offline later. Keyed by endpoint + model + exact state + questions.
# --------------------------------------------------------------------------- #
class ResultCache:
    """Live results in `path` (writable), on top of an optional read-only `seed_path`.

    The seed is data/saved_results.json: a file you download from Settings and commit, so a
    cloud host that restarts with an empty disk still replays every saved answer instantly.
    """

    def __init__(self, path: Path, seed_path: Optional[Path] = None):
        self.path = path
        self.lock = threading.Lock()
        self.records: Dict[str, Dict[str, Any]] = {}
        self.seed: Dict[str, Dict[str, Any]] = {}
        if seed_path is not None and seed_path.is_file():
            try:
                self.seed = self._read(seed_path)
            except (ValueError, OSError, AttributeError) as exc:
                log(f"Seed file {seed_path.name} unreadable ({exc}); ignoring it.")
        if path.is_file():
            try:
                self.records = self._read(path)
            except (ValueError, OSError, AttributeError) as exc:
                log(f"Cache file unreadable ({exc}); starting fresh.")

    @staticmethod
    def _read(path: Path) -> Dict[str, Dict[str, Any]]:
        records = json.loads(path.read_text(encoding="utf-8")).get("records", {})
        if not isinstance(records, dict):
            raise ValueError("'records' is not an object")
        return {k: v for k, v in records.items() if isinstance(v, dict) and isinstance(v.get("answers"), dict)}

    @staticmethod
    def key(state: Any, questions: Dict[str, Any], base_url: str, model: str) -> str:
        blob = json.dumps({"v": router.QUESTION_SET_VERSION, "endpoint": base_url.rstrip("/"),
                           "model": model, "state": state, "questions": questions},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.records.get(key) or self.seed.get(key)

    def put(self, key: str, email_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(result)
        record["key"] = key
        record["email_id"] = email_id
        record["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            self.records[key] = record
            payload = json.dumps({"question_set": router.QUESTION_SET_VERSION,
                                  "records": self.records}, indent=1)
            # Antivirus or OneDrive can lock the file for a moment on Windows: retry, and if
            # it still fails keep the answer in memory rather than losing it.
            for attempt in range(6):
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = self.path.with_suffix(".tmp")
                    tmp.write_text(payload, encoding="utf-8")
                    os.replace(tmp, self.path)
                    break
                except OSError as exc:
                    if attempt == 5:
                        log(f"Could not save the cache file ({exc}). The result is kept in memory.")
                    else:
                        time.sleep(0.25)
        return record

    def export(self) -> Dict[str, Any]:
        """Seed plus live results, in the seed file format."""
        with self.lock:
            merged = dict(self.seed)
            merged.update(self.records)
        return {"question_set": router.QUESTION_SET_VERSION,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "about": "Saved Jev answers for the RFQ Router demo. Commit this file as "
                         "data/saved_results.json and replays stay instant after a restart.",
                "records": dict(sorted(merged.items()))}

    @property
    def seed_count(self) -> int:
        return len(self.seed)

    def __len__(self) -> int:
        with self.lock:
            return len(set(self.records) | set(self.seed))


def cache_key_for(cfg: Optional[jev_client.JevConfig], state: Any, questions: Dict[str, Any]) -> str:
    base = cfg.base_url if cfg else jev_client.GATEWAY_BASE_URL
    model = cfg.model if cfg else jev_client.GATEWAY_MODEL
    return ResultCache.key(state, questions, base, model)


# --------------------------------------------------------------------------- #
# App state + background worker
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, jev: Optional[jev_client.JevConfig], cache_file: Path = CACHE_FILE,
                 seed_file: Optional[Path] = SEED_FILE):
        self.jev = jev
        self.shop = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        data = json.loads(EMAILS_FILE.read_text(encoding="utf-8"))
        self.paste_examples = data.get("paste_examples", [])
        for ex in self.paste_examples:
            ex["attachments"] = [att_mod.normalize(a) for a in ex.get("attachments") or []]
        self.emails: Dict[str, Dict[str, Any]] = {}
        self.order: List[str] = []
        for email in data["emails"]:
            email["attachments"] = [att_mod.normalize(a) for a in email.get("attachments") or []]
            self.emails[email["id"]] = email
            self.order.append(email["id"])
        self.questions = router.build_questions(self.shop)
        self.thresholds = dict(router.DEFAULT_THRESHOLDS)
        self.thresholds.update(self.shop.get("thresholds", {}))
        self.thresholds_version = 0
        self.cache = ResultCache(cache_file, seed_file)
        self.uploads = att_mod.UploadStore()
        self.items: Dict[str, Dict[str, Any]] = {eid: self._blank_item() for eid in self.order}
        self.queue: deque = deque()
        self.cond = threading.Condition(threading.RLock())
        # Bumped by Stop and Reset. The worker drops work that started under an older generation.
        self.generation = 0
        self.running_gen: Optional[int] = None  # generation of the item the worker is on
        self.version = 0
        # /api/state?since=N sends only what changed. Each item's signature is compared on every
        # snapshot, so nothing can be missed even if some code path forgets to bump the version.
        self.boot_id = secrets.token_hex(6)
        self._sig: Dict[str, Any] = {}
        self._changed_at: Dict[str, int] = {}
        self._added_at: Dict[str, int] = {eid: 0 for eid in self.order}
        self._sig_version = -1
        self.order_version = 0
        self._keys: Dict[str, str] = {}
        self._decisions: Dict[str, Any] = {}
        # Real attachment files: their text (text layer or OCR) must be ready before an email's
        # Jev state, and so its cache key, can be computed.
        self._prepared: set = set()
        self._prep_locks: Dict[str, threading.Lock] = {}
        self._email_changed_at: Dict[str, int] = {}
        self._details: Dict[str, Any] = {}
        self.ocr_cache = self._open_ocr_cache()
        self.live_count = 0
        self.last_call_at: Optional[float] = None
        self.pace_seconds: Optional[float] = None
        pace = os.environ.get("JEV_PACE_SECONDS", "").strip()
        if pace:
            try:
                self.pace_seconds = float(pace)
            except ValueError:
                pass
        self.worker = {"state": "idle", "message": "", "resume_at": None, "current": None,
                       "wait_reason": None, "last_error": None, "calls": 0, "cache_hits": 0,
                       "rate_limited": False}
        threading.Thread(target=self._worker_loop, name="jev-worker", daemon=True).start()
        specs = [a for e in self.emails.values() for a in e["attachments"]]
        specs += [a for ex in self.paste_examples for a in ex["attachments"]]
        threading.Thread(target=att_mod.warm, args=(specs,), name="file-warmup", daemon=True).start()
        # Emails whose files are all in the OCR cache are ready now (hashing and a lookup); the rest
        # are read in the background, and the worker reads any it reaches first.
        slow = []
        for eid in self.order:
            files = [a for a in self.emails[eid]["attachments"] if a.get("kind") == "file"]
            if not files:
                self._prepared.add(eid)
            elif all(self._cached(a) for a in files):
                self.prepare_email(eid)
            else:
                slow.append(eid)
        if slow:
            threading.Thread(target=lambda: [self.prepare_email(e) for e in slow], name="file-prep",
                             daemon=True).start()

    # ---- real attachment files ---------------------------------------------- #
    @staticmethod
    def _open_ocr_cache() -> Any:
        ocr_mod = att_mod._ocr()
        if not ocr_mod:
            return None
        try:
            seed = ocr_mod.OcrCache(OCR_SEED_FILE) if OCR_SEED_FILE.is_file() else None
            return att_mod.LayeredCache(seed, ocr_mod.OcrCache(OCR_LIVE_FILE))
        except Exception as exc:  # noqa: BLE001
            log(f"OCR cache unavailable ({exc}); scans will be read live.")
            return None

    def _cached(self, att: Dict[str, Any]) -> bool:
        if att.get("media") == "step" or self.ocr_cache is None:
            return att.get("media") == "step"
        path = att_mod.resolve_data_path(DATA_DIR, att.get("path", ""))
        if not path:
            return True  # missing file: preparing it is instant
        try:
            with open(path, "rb") as fh:
                return self.ocr_cache.get(hashlib.sha256(fh.read()).hexdigest()) is not None
        except OSError:
            return True

    def prepare_email(self, eid: str) -> None:
        """Read an email's real files (text layer, OCR, STEP header). Safe to call from any thread."""
        if eid in self._prepared:
            return
        lock = self._prep_locks.setdefault(eid, threading.Lock())
        with lock:
            if eid in self._prepared:
                return
            email = self.emails[eid]
            started = time.time()
            for att in email.get("attachments") or []:
                if att.get("kind") == "file" and not att.get("prepared"):
                    att_mod.prepare_file(att, DATA_DIR, self.ocr_cache)
            if self.ocr_cache is not None:
                self.ocr_cache.save()
            seconds = time.time() - started
            if seconds > 2:
                log(f"Read the attachments of {eid} in {seconds:.1f}s (OCR)")
            with self.cond:
                self._keys.pop(eid, None)
                self._prepared.add(eid)
                self._bump()
                self._email_changed_at[eid] = self.version

    @staticmethod
    def _blank_item() -> Dict[str, Any]:
        return {"status": "idle", "source": None, "record": None, "error": None, "routed_at": None}

    def _bump(self) -> None:
        self.version += 1

    def key_for(self, email: Dict[str, Any]) -> str:
        # Emails never change once added, so the key (a hash of the full Jev state) is computed once.
        eid = email["id"]
        key = self._keys.get(eid)
        if key is None:
            key = self._keys[eid] = cache_key_for(self.jev, router.jev_state(email), self.questions)
        return key

    # ---- public actions ---------------------------------------------------- #
    def run(self, ids: List[str], use_cache: bool, front: bool = False) -> Dict[str, int]:
        queued = cached = 0
        with self.cond:
            if self.worker["state"] == "stopped":
                self.worker.update(state="idle", message="", last_error=None)
            to_queue = []
            for eid in ids:
                item = self.items.get(eid)
                if item is None or item["status"] == "queued":
                    continue
                if item["status"] == "running":
                    # Stop was pressed while this call was in flight: the worker is about to drop
                    # it, so queue it again (the saved answer will be reused if it lands).
                    if self.worker.get("current") == eid and self.running_gen != self.generation:
                        item.update(status="queued", use_cache=use_cache)
                        to_queue.append(eid)
                    continue
                email = self.emails[eid]
                # Files not read yet (a scan waiting for OCR): the worker reads them first, then
                # still uses a saved answer if there is one.
                if use_cache and eid in self._prepared:
                    record = self.cache.get(self.key_for(email))
                    if record:
                        item.update(status="done", source="cache", record=record, error=None,
                                    routed_at=time.time())
                        cached += 1
                        continue
                if self.jev is None and use_cache and eid not in self._prepared:
                    pass  # it may still replay from the cache once its files are read
                elif self.jev is None:
                    item.update(status="error", error={
                        "kind": "config", "message": "No Jev API key found.",
                        "hint": "Put AI_GATEWAY_API_KEY=vck_... in Jev_Test/.env, then restart the demo."})
                    continue
                item.update(status="queued", error=None, source=None, use_cache=use_cache)
                to_queue.append(eid)
            if front:
                self.queue.extendleft(reversed(to_queue))  # single-email requests jump the line
            else:
                self.queue.extend(to_queue)
            queued = len(to_queue)
            self.worker["cache_hits"] += cached
            self._bump()
            self.cond.notify_all()
        return {"queued": queued, "cached": cached}

    def stop(self) -> None:
        with self.cond:
            self.generation += 1
            while self.queue:
                eid = self.queue.popleft()
                if self.items[eid]["status"] == "queued":
                    self.items[eid]["status"] = "idle"
            if self.worker["state"] == "waiting":
                self.worker.update(message="Stopping", resume_at=None, wait_reason=None)
            self._bump()
            self.cond.notify_all()

    def reset(self) -> None:
        with self.cond:
            self.stop()
            for eid, item in self.items.items():
                if item["status"] != "running":
                    self.items[eid] = self._blank_item()
            self.worker.update(message="", last_error=None)
            if self.worker["state"] == "stopped":
                self.worker["state"] = "idle"
            self._bump()

    def add_email(self, payload: Dict[str, Any]) -> str:
        subject = str(payload.get("subject", "")).strip()
        body = str(payload.get("body", "")).strip()
        if not subject and not body:
            raise ValueError("Add a subject or a body.")
        names = payload.get("attachments") or []
        if isinstance(names, str):
            names = [a.strip() for a in names.replace(";", ",").split(",") if a.strip()]
        attachments: List[Any] = []
        # Sample files that came with a paste example (the user may have removed some).
        example = payload.get("example")
        if isinstance(example, str) and example.strip().isdigit():
            example = int(example)
        if isinstance(example, int) and not isinstance(example, bool) and 0 <= example < len(self.paste_examples):
            keep = payload.get("example_files")
            for att in self.paste_examples[example]["attachments"]:
                if att.get("kind") in att_mod.SPEC_KINDS and (keep is None or att["name"] in keep):
                    attachments.append(dict(att))
        uploads = payload.get("uploads") or []
        if not isinstance(uploads, list) or len(uploads) > att_mod.MAX_UPLOADS_PER_EMAIL:
            raise ValueError(f"Attach up to {att_mod.MAX_UPLOADS_PER_EMAIL} files.")
        for uid in uploads:
            item = self.uploads.claim(str(uid))
            if item is None:
                raise ValueError("An uploaded file is no longer in memory. Remove it and add it again.")
            attachments.append(att_mod.upload_attachment(item))
        attachments += [str(a)[:120] for a in names if isinstance(a, (str, int, float))]
        attachments = [att_mod.normalize(a) for a in attachments][:20]
        with self.cond:
            self.live_count += 1
            eid = f"L{self.live_count:02d}"
            while eid in self.emails:
                self.live_count += 1
                eid = f"L{self.live_count:02d}"
            self.emails[eid] = {
                "id": eid,
                "received": time.strftime("%H:%M"),
                "from_name": str(payload.get("from_name", "")).strip(),
                "from_email": str(payload.get("from_email", "")).strip() or "unknown@example.com",
                "subject": subject,
                "body": body,
                "attachments": attachments,
                "live": True,
            }
            self.order.append(eid)
            self.items[eid] = self._blank_item()
            self._bump()
            self.order_version = self._added_at[eid] = self.version
        self.run([eid], use_cache=True, front=True)
        return eid

    def set_thresholds(self, updates: Dict[str, Any]) -> Dict[str, float]:
        with self.cond:
            for key, value in updates.items():
                if key not in router.DEFAULT_THRESHOLDS:
                    continue
                value = float(value)
                limit = 3.0 if key in ("rush_score", "soon_score") else 1.0
                self.thresholds[key] = max(0.0, min(limit, value))
            self.thresholds_version += 1
            self._bump()
            return dict(self.thresholds)

    def test_connection(self) -> Dict[str, Any]:
        if self.jev is None:
            return {"ok": False, "error": {"kind": "config", "message": "No Jev API key found.",
                                           "hint": "Put AI_GATEWAY_API_KEY=vck_... in Jev_Test/.env, then restart."}}
        try:
            result = jev_client.ping(self.jev)
        except jev_client.JevError as exc:
            if exc.kind == "rate_limit":
                self._learn_rate_limit()
            self.last_call_at = time.time()
            return {"ok": False, "error": exc.to_dict()}
        self.last_call_at = time.time()
        with self.cond:
            self.worker["calls"] += 1
            self._bump()
        log(f"[Jev] connection test OK: {result['latency_ms']:.0f} ms, model {result['model']}")
        return {"ok": True, "latency_ms": result["latency_ms"], "model": result["model"],
                "probability": result["answers"]["is_rfq"]["p"], "input_tokens": result["input_tokens"],
                "cost_usd": result["cost_usd"], "cost_is_estimate": result["cost_is_estimate"]}

    # ---- worker ------------------------------------------------------------ #
    def _learn_rate_limit(self) -> None:
        with self.cond:
            self.worker["rate_limited"] = True
            if not self.pace_seconds or self.pace_seconds < FREE_TIER_PACE:
                self.pace_seconds = FREE_TIER_PACE

    def _check(self, gen: int) -> None:
        if self.generation != gen:
            raise Cancelled()

    def _wait(self, seconds: float, reason: str, message: str, gen: int) -> None:
        seconds = max(0.0, seconds)
        with self.cond:
            self._check(gen)
            self.worker.update(state="waiting", wait_reason=reason, message=message,
                               resume_at=time.time() + seconds)
            self._bump()
        deadline = time.time() + seconds
        while time.time() < deadline:
            self._check(gen)
            time.sleep(0.1)
        with self.cond:
            self._check(gen)
            self.worker.update(state="running", wait_reason=None, resume_at=None)
            self._bump()

    def _pace_remaining(self) -> float:
        if self.pace_seconds and self.last_call_at:
            return max(0.0, self.last_call_at + self.pace_seconds - time.time())
        return 0.0

    def _call_jev(self, eid: str, email: Dict[str, Any], gen: int) -> Dict[str, Any]:
        state = router.jev_state(email)
        key = cache_key_for(self.jev, state, self.questions)
        rate_attempt = 0
        transient = 0
        skip_pace = False
        while True:
            self._check(gen)
            remaining = 0.0 if skip_pace else self._pace_remaining()
            skip_pace = False
            if remaining > 0.05:
                self._wait(remaining, "pace",
                           "Free-tier pacing: about one Jev call per minute. "
                           "Buying any AI Gateway credits removes this limit.", gen)
            with self.cond:
                self._check(gen)
                self.worker.update(state="running", message=f"Asking Jev about {eid}")
                self._bump()
            try:
                result = jev_client.system_one(self.jev, state, self.questions)
            except jev_client.JevError as exc:
                self.last_call_at = time.time()
                if exc.kind == "rate_limit":
                    self._learn_rate_limit()
                    base = exc.retry_after if exc.retry_after else \
                        RATE_LIMIT_BACKOFF[min(rate_attempt, len(RATE_LIMIT_BACKOFF) - 1)]
                    wait = max(base, self.pace_seconds or 0.0)  # one wait, the longer of the two
                    rate_attempt += 1
                    log(f"[Jev] {eid}: rate limited (HTTP 429). Waiting {wait:.0f}s, then retrying.")
                    self._wait(wait, "rate_limit",
                               "AI Gateway free-tier limit reached. The demo resumes automatically. "
                               "Buying any AI Gateway credits removes the limit.", gen)
                    skip_pace = True
                    continue
                if exc.kind in ("overloaded", "server", "timeout") and transient < 3:
                    transient += 1
                    log(f"[Jev] {eid}: {exc}. Retry {transient}/3.")
                    self._wait(2 ** transient, "retry", f"Jev had a hiccup ({exc.kind}). Retrying.", gen)
                    skip_pace = True
                    continue
                raise
            self.last_call_at = time.time()
            # Save even if Stop/Reset happened mid-call: the answer is valid and replays later.
            return self.cache.put(key, eid, result)

    def _worker_loop(self) -> None:
        while True:
            with self.cond:
                while not self.queue:
                    if self.worker["state"] in ("running", "waiting"):
                        self.worker.update(state="idle", current=None, resume_at=None,
                                           wait_reason=None, message="")
                        self._bump()
                    self.cond.wait()
                eid = self.queue.popleft()
                item = self.items.get(eid)
                if item is None or item["status"] != "queued":
                    continue
                email = self.emails[eid]
                needs_files = eid not in self._prepared
                if needs_files:
                    gen = self.generation
                    self.worker.update(state="running", current=eid, message=f"Reading the attachments of {eid}",
                                       resume_at=None, wait_reason=None)
                    self._bump()
            if needs_files:
                self.prepare_email(eid)  # OCR can take a while: never hold the lock for it
            with self.cond:
                if needs_files and (self.generation != gen or item["status"] != "queued"):
                    if item["status"] == "queued":
                        item["status"] = "idle"
                    self._bump()
                    continue
                if item.get("use_cache"):
                    record = self.cache.get(self.key_for(email))
                    if record:
                        item.update(status="done", source="cache", record=record, error=None,
                                    routed_at=time.time())
                        self._bump()
                        continue
                if self.jev is None:
                    item.update(status="error", error={
                        "kind": "config", "message": "No Jev API key found.",
                        "hint": "Put AI_GATEWAY_API_KEY=vck_... in Jev_Test/.env, then restart the demo."})
                    self._bump()
                    continue
                gen = self.generation
                self.running_gen = gen
                item["status"] = "running"
                self.worker.update(state="running", current=eid, message=f"Asking Jev about {eid}",
                                   resume_at=None, wait_reason=None)
                self._bump()
            try:
                record = self._call_jev(eid, email, gen)
            except Cancelled:
                with self.cond:
                    if item["status"] == "running":
                        item["status"] = "idle"
                    self.worker.update(state="idle", current=None, resume_at=None, wait_reason=None,
                                       message="Stopped.")
                    self._bump()
                continue
            except jev_client.JevError as exc:
                log(f"[Jev] {eid}: FAILED. {exc}")
                with self.cond:
                    item.update(status="error", error=exc.to_dict())
                    if exc.kind in FATAL_KINDS:
                        while self.queue:
                            other = self.queue.popleft()
                            if self.items[other]["status"] == "queued":
                                self.items[other]["status"] = "idle"
                        self.worker.update(state="stopped", current=None, resume_at=None,
                                           wait_reason=None, last_error=exc.to_dict(),
                                           message=str(exc))
                    self._bump()
                continue
            except Exception as exc:  # noqa: BLE001 - keep the worker alive
                log(f"[Jev] {eid}: unexpected error {exc!r}")
                with self.cond:
                    item.update(status="error", error={"kind": "internal", "message": repr(exc), "hint": ""})
                    self._bump()
                continue

            with self.cond:
                self.worker["calls"] += 1
                if self.generation != gen:
                    # Stopped or reset while the call was in flight: keep the board as the user left it.
                    if item["status"] == "running":
                        item["status"] = "idle"
                    self._bump()
                    log(f"[Jev] {eid}: answered after Stop/Reset; saved for replay.")
                    continue
                item.update(status="done", source="live", record=record, error=None, routed_at=time.time())
                self._bump()
                decision = self._decision(eid)
            cost = record.get("cost_usd") or 0.0
            lane_name = decision["lane_name"] if decision else "?"
            log(f"[Jev] {eid} -> {lane_name:<28} "
                f"{record['latency_ms']:>6.0f} ms  {record.get('input_tokens') or 0:>5} tokens  ${cost:.6f}")

    # ---- views ------------------------------------------------------------- #
    def _decision(self, eid: str) -> Optional[Dict[str, Any]]:
        """Routing decision for an item, memoized. It depends only on the saved answers, the
        thresholds, and today's date (for quote-by dates), so those make up the memo key."""
        item = self.items[eid]
        record = item.get("record")
        if not record:
            return None
        memo_key = (record.get("key"), record.get("saved_at"), self.thresholds_version,
                    time.strftime("%Y-%m-%d"))
        memo = self._decisions.get(eid)
        if memo and memo[0] == memo_key:
            return memo[1]
        decision = router.decide(self.emails[eid], record["answers"], self.shop, self.thresholds)
        self._decisions[eid] = (memo_key, decision)
        return decision

    def _signature(self, eid: str) -> Any:
        item = self.items[eid]
        record = item.get("record") or {}
        err = item.get("error")
        return (item["status"], item["source"], item["routed_at"],
                json.dumps(err, sort_keys=True) if err else None, record.get("key"), record.get("saved_at"),
                self.thresholds_version if record else None, time.strftime("%Y-%m-%d") if record else None)

    def _entry(self, eid: str) -> Dict[str, Any]:
        item = self.items[eid]
        entry: Dict[str, Any] = {"status": item["status"], "source": item["source"],
                                 "error": item["error"], "routed_at": item["routed_at"]}
        record = item.get("record")
        if record and item["status"] == "done":
            entry["decision"] = self._decision(eid)
            entry["jev"] = {k: record.get(k) for k in (
                "answers", "model", "latency_ms", "input_tokens", "output_tokens", "cost_usd",
                "cost_is_estimate", "list_price_usd", "saved_at", "generation_id", "request_id")}
        return entry

    def snapshot(self, since: Any = None, boot: Any = None) -> Dict[str, Any]:
        """The board. With `since` (a version this browser already has, from this same server run),
        only the items that changed after it are included, plus any newly added emails."""
        with self.cond:
            changed = []
            for eid in self.order:
                sig = self._signature(eid)
                if self._sig.get(eid) != sig:
                    self._sig[eid] = sig
                    changed.append(eid)
            if changed:
                if self.version == self._sig_version:  # a change nobody announced: make it visible
                    self.version += 1
                for eid in changed:
                    self._changed_at[eid] = self.version
            self._sig_version = self.version
            try:
                since_v = int(since) if since is not None and str(since) != "" else None
            except (TypeError, ValueError):
                since_v = None
            full = since_v is None or since_v > self.version or since_v < 0 or (boot and boot != self.boot_id)
            send = self.order if full else [e for e in self.order if self._changed_at.get(e, 0) > since_v]
            items = {eid: self._entry(eid) for eid in send}

            latencies: List[float] = []
            cost = list_price = 0.0
            routed = auto = review = matches = graded = 0
            cost_reported = False
            for eid in self.order:
                item = self.items[eid]
                record = item.get("record")
                if not (record and item["status"] == "done"):
                    continue
                decision = self._decision(eid)
                routed += 1
                if decision["lane"] == "review":
                    review += 1
                else:
                    auto += 1
                if decision["matches_expected"] is not None:
                    graded += 1
                    matches += 1 if decision["matches_expected"] else 0
                if record.get("latency_ms"):
                    latencies.append(float(record["latency_ms"]))
                cost += float(record.get("cost_usd") or 0.0)
                list_price += float(record.get("list_price_usd") or 0.0)
                cost_reported = cost_reported or not record.get("cost_is_estimate", True)
            worker = dict(self.worker)
            worker["queue_length"] = len(self.queue)
            worker["pace_seconds"] = self.pace_seconds
            worker["now"] = time.time()
            latencies.sort()
            state: Dict[str, Any] = {
                "version": self.version,
                "boot_id": self.boot_id,
                "full": bool(full),
                "items": items,
                "worker": worker,
                "thresholds": dict(self.thresholds),
                "stats": {
                    "total": len(self.order), "routed": routed, "auto": auto, "review": review,
                    "matches": matches, "graded": graded,
                    "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
                    "median_latency_ms": latencies[len(latencies) // 2] if latencies else None,
                    "cost_usd": cost, "list_price_usd": list_price, "cost_reported": cost_reported,
                    "cache_size": len(self.cache), "seed_size": self.cache.seed_count,
                },
            }
            if full or self.order_version > since_v:
                state["order"] = list(self.order)
            if not full:
                # New emails, and emails whose files were just read (labels, sizes, how the text was read).
                added = [eid for eid in self.order if self._added_at.get(eid, 0) > since_v
                         or self._email_changed_at.get(eid, 0) > since_v]
                if added:
                    state["emails"] = [self.public_email(self.emails[eid]) for eid in added]
            return state

    # ---- attachments ------------------------------------------------------- #
    def _attachment_list(self, eid: str) -> Optional[List[Dict[str, Any]]]:
        if re.fullmatch(r"X\d{1,2}", eid):
            idx = int(eid[1:]) - 1
            if 0 <= idx < len(self.paste_examples):
                return self.paste_examples[idx]["attachments"]
            return None
        with self.cond:
            email = self.emails.get(eid)
        return email["attachments"] if email else None

    def attachment(self, eid: str, idx: int) -> Optional[Dict[str, Any]]:
        atts = self._attachment_list(eid)
        if atts is None or not (0 <= idx < len(atts)):
            return None
        att = atts[idx]
        return att if isinstance(att, dict) and att.get("kind") == "file" else att_mod.normalize(att)

    def attachment_text(self, eid: str, idx: int) -> Optional[Dict[str, Any]]:
        """The text Jev reads for one attachment, exactly as it goes into the state."""
        atts = self._attachment_list(eid)
        if atts is None or not (0 <= idx < len(atts)):
            return None
        entries = att_mod.jev_entries(atts)
        full = att_mod.jev_text(att_mod.normalize(atts[idx]))
        sent = entries[idx]["content"]
        return {"name": entries[idx]["file"], "text": sent, "truncated": sent.endswith("[truncated]"),
                "full_chars": len(full)}

    def _describe(self, base_id: str, idx: int, att: Dict[str, Any]) -> Dict[str, Any]:
        att = att_mod.normalize(att)
        kind = att.get("kind")
        base = f"/api/att/{base_id}/{idx}" if kind in att_mod.SPEC_KINDS or kind in ("upload", "file") else None
        available = True
        if kind == "upload":
            available = self.uploads.get(att.get("upload_id") or "") is not None
        return att_mod.describe(att, base, available)

    def public_email(self, email: Dict[str, Any]) -> Dict[str, Any]:
        out = {k: v for k, v in email.items() if k != "attachments"}
        out["attachments"] = [self._describe(email["id"], i, a) for i, a in enumerate(email.get("attachments") or [])]
        return out

    def public_examples(self) -> List[Dict[str, Any]]:
        out = []
        for i, ex in enumerate(self.paste_examples, start=1):
            entry = {k: v for k, v in ex.items() if k != "attachments"}
            entry["attachments"] = [self._describe(f"X{i}", j, a) for j, a in enumerate(ex["attachments"])]
            out.append(entry)
        return out

    # ---- RFQ details --------------------------------------------------------- #
    def _texts(self, email: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """What each attachment says, in the shape rfq_details.extract expects (ocr.file_text results)."""
        texts: Dict[str, Dict[str, Any]] = {}
        for raw in email.get("attachments") or []:
            att = att_mod.normalize(raw)
            kind = att.get("kind")
            if kind == "file":
                texts[att["name"]] = att.get("_file_text") or {"method": att.get("text_method") or "none",
                                                               "text": att.get("text") or ""}
            elif kind == "upload":
                texts[att["name"]] = {"method": att.get("text_method") or ("text-layer" if att.get("text") else "none"),
                                      "text": att.get("text") or "", "confidence": att.get("text_conf")}
            elif kind in att_mod.SPEC_KINDS:
                texts[att["name"]] = {"method": "step-header" if kind == "model" else "text-layer",
                                      "text": att_mod.spec_text(att)}
        return texts

    def rfq_record(self, eid: str) -> Optional[Dict[str, Any]]:
        """The extracted details of one RFQ, or None when the email is not an RFQ."""
        if rfq_details is None or eid not in self.emails:
            return None
        self.prepare_email(eid)
        with self.cond:
            item = self.items[eid]
            decision = self._decision(eid) if item["status"] == "done" else None
            memo_key = (json.dumps(decision, sort_keys=True, default=str) if decision else None,
                        self._email_changed_at.get(eid))
            memo = self._details.get(eid)
            if memo and memo[0] == memo_key:
                return memo[1]
            email = self.emails[eid]
        try:
            record = rfq_details.extract(email, self._texts(email), self.shop, decision) \
                if rfq_details.is_rfq(email, decision) else None
        except Exception as exc:  # noqa: BLE001 - one odd email must not break the download
            log(f"RFQ details for {eid} failed: {exc!r}")
            record = None
        with self.cond:
            self._details[eid] = (memo_key, record)
        return record

    def rfq_records(self) -> List[Dict[str, Any]]:
        with self.cond:
            order = list(self.order)
        return [r for r in (self.rfq_record(eid) for eid in order) if r]

    def bootstrap(self) -> Dict[str, Any]:
        lanes = router.lane_index(self.shop)
        with self.cond:
            emails = [self.public_email(self.emails[eid]) for eid in self.order]
        questions = []
        for name, q in self.questions.items():
            questions.append({"name": name, "label": router.QUESTION_LABELS.get(name, name),
                              "type": q["type"], "instructions": q.get("instructions"),
                              "criteria": q.get("criteria")})
        jev_info = self.jev.public() if self.jev else {
            "configured": False,
            "searched": [str(path) for path in jev_client.env_search_paths()],
        }
        return {
            "brand": self.shop.get("brand", {}),
            "shop": self.shop["shop"],
            "lanes": list(lanes.values()),
            "emails": emails,
            "questions": questions,
            "email_type_labels": router.EMAIL_TYPE_LABELS,
            "volume_labels": router.VOLUME_LABELS,
            "question_set_version": router.QUESTION_SET_VERSION,
            "paste_examples": self.public_examples(),
            "customers": {c["domain"].lower(): c for c in self.shop.get("customers", [])},
            "jev": jev_info,
            "free_tier_pace_seconds": FREE_TIER_PACE,
            "uploads": {"max_files": att_mod.MAX_UPLOADS_PER_EMAIL, "max_bytes": att_mod.MAX_UPLOAD_BYTES,
                        "types": sorted(att_mod.UPLOAD_TYPES), "pdf_text": att_mod.pypdf_available(),
                        "ocr": att_mod.ocr_ready()},
            "features": {"rfq_details": rfq_details is not None, "ocr": att_mod.ocr_ready()},
            "state": self.snapshot(),
        }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    app: App = None  # type: ignore[assignment]
    password = ""  # RFQ_DEMO_PASSWORD. When set, every request needs it (HTTP Basic auth, any user name).
    server_version = "RFQRouterDemo/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet: the worker logs Jev calls
        return

    def _send(self, status: int, body: bytes, content_type: str,
              headers: Optional[Dict[str, str]] = None) -> None:
        headers = dict(headers or {})
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", headers.pop("Cache-Control", "no-store"))
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200, headers: Optional[Dict[str, str]] = None) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8", headers)

    @staticmethod
    def _disposition(kind: str, filename: str) -> str:
        ascii_name = re.sub(r'[^A-Za-z0-9 ._()+-]', "_", filename) or "file"
        return f"{kind}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"

    def _file(self, data: bytes, content_type: str, filename: str, download: bool, cacheable: bool) -> None:
        headers = {"Content-Disposition": self._disposition("attachment" if download else "inline", filename),
                   "X-Frame-Options": "SAMEORIGIN"}
        if cacheable:  # sample files never change while the server runs; URLs carry a content hash
            headers["Cache-Control"] = "private, max-age=3600"
        self._send(200, data, content_type, headers)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1", "")

    def _password_ok(self) -> bool:
        scheme, _, token = (self.headers.get("Authorization") or "").partition(" ")
        if scheme.lower() != "basic":
            return False
        try:
            supplied = base64.b64decode(token.strip(), validate=True).decode("utf-8").partition(":")[2]
        except ValueError:  # not base64, or not UTF-8
            return False
        return hmac.compare_digest(supplied.encode("utf-8"), self.password.encode("utf-8"))

    def _allowed(self) -> bool:
        """Gate every request. When it returns False the refusal has already been sent.

        With a password (cloud), HTTP Basic auth guards everything, so the Host header no longer
        matters. Without one, only requests addressed to this computer are answered, which also
        stops web pages from reaching the server through DNS rebinding.
        """
        if self.password:
            if self._password_ok():
                return True
            self._json({"error": "password required"}, 401,
                       {"WWW-Authenticate": 'Basic realm="RFQ Router demo", charset="UTF-8"'})
            return False
        if self._host_ok():
            return True
        self._json({"error": "forbidden host"}, 403)
        return False

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/healthz":  # for cloud health checks: no password needed, no data returned
            return self._json({"ok": True})
        if not self._allowed():
            return
        query = parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if path == "/api/bootstrap":
            return self._json(self.app.bootstrap())
        if path == "/api/state":
            return self._json(self.app.snapshot((query.get("since") or [None])[0], (query.get("boot") or [None])[0]))
        if path == "/api/saved_results":
            body = json.dumps(self.app.cache.export(), indent=1).encode("utf-8")
            return self._send(200, body, "application/json; charset=utf-8",
                              {"Content-Disposition": self._disposition("attachment", "saved_results.json")})
        if path in ("/api/rfq_details.csv", "/api/rfq_details.json"):
            if rfq_details is None:
                return self._json({"error": "rfq_details.py is not available on this server"}, 404)
            records = self.app.rfq_records()
            if path.endswith(".csv"):
                body = rfq_details.to_csv(records).encode("utf-8-sig")  # the BOM makes Excel read UTF-8
                return self._send(200, body, "text/csv; charset=utf-8",
                                  {"Content-Disposition": self._disposition("attachment", "rfq_details.csv")})
            return self._send(200, rfq_details.to_json(records).encode("utf-8"), "application/json; charset=utf-8",
                              {"Content-Disposition": self._disposition("inline", "rfq_details.json")})
        match = re.fullmatch(r"/api/rfq_details/([A-Z]\d{1,4})", path)
        if match:
            record = self.app.rfq_record(match.group(1))
            return self._json({"ok": True, "record": record} if record else {"ok": False, "record": None})
        match = re.fullmatch(r"/api/att/([A-Z]\d{1,4})/(\d{1,3})/(file|thumb\.svg|thumb\.jpg|mesh\.json|text)", path)
        if match:
            return self._attachment(match.group(1), int(match.group(2)), match.group(3), "download" in query)
        match = re.fullmatch(r"/api/upload/([0-9a-f]{16})/(file|text)", path)
        if match:
            return self._upload(match.group(1), match.group(2), "download" in query)
        return self._json({"error": "not found"}, 404)

    def _attachment(self, eid: str, idx: int, what: str, download: bool) -> None:
        app = self.app
        att = app.attachment(eid, idx)
        if att is None:
            return self._json({"error": "not found"}, 404)
        kind = att.get("kind")
        if what == "text":
            if kind == "file":
                app.prepare_email(eid) if eid in app.emails else None
            return self._json(app.attachment_text(eid, idx))
        if kind == "file":
            return self._real_file(eid, att, what, download)
        if kind == "upload":
            if what != "file":
                return self._json({"error": "not found"}, 404)
            return self._upload(att.get("upload_id") or "", "file", download)
        if kind not in att_mod.SPEC_KINDS:
            return self._json({"error": "This attachment is a file name only."}, 404)
        try:
            if what == "file":
                data, ctype = att_mod.spec_file(att)
                return self._file(data, ctype, att["name"], download, True)
            if what == "thumb.svg":
                return self._send(200, att_mod.spec_thumb(att).encode("utf-8"), "image/svg+xml",
                                  {"Cache-Control": "private, max-age=3600"})
            if what == "mesh.json" and kind == "model":
                return self._send(200, json.dumps(att_mod.spec_mesh(att)).encode("utf-8"),
                                  "application/json; charset=utf-8", {"Cache-Control": "private, max-age=3600"})
        except Exception as exc:  # noqa: BLE001 - a bad spec should not take the page down
            log(f"Could not build {eid}/{idx} {what}: {exc!r}")
            return self._json({"error": "could not build this file"}, 500)
        return self._json({"error": "not found"}, 404)

    def _real_file(self, eid: str, att: Dict[str, Any], what: str, download: bool) -> None:
        """A real attachment file on disk (the RFQ details beta inbox)."""
        path = att_mod.resolve_data_path(DATA_DIR, att.get("path", ""))
        if not path:
            return self._json({"error": "This file is missing on the server."}, 404)
        if what == "file":
            media = att.get("media") or "pdf"
            ctype = att_mod.MEDIA_TYPES.get(media, "application/octet-stream")
            with open(path, "rb") as fh:
                return self._file(fh.read(), ctype, att["name"], download or media == "step", True)
        if what in ("thumb.jpg", "thumb.svg"):
            if eid in self.app.emails:
                self.app.prepare_email(eid)
            thumb = att_mod.file_thumb(att, DATA_DIR)
            if not thumb:
                return self._json({"error": "no preview"}, 404)
            return self._send(200, thumb[0], thumb[1], {"Cache-Control": "private, max-age=3600"})
        if what == "mesh.json":
            if eid in self.app.emails:
                self.app.prepare_email(eid)
            if not att.get("_mesh"):
                return self._json({"error": "no mesh in this file"}, 404)
            return self._send(200, json.dumps(att["_mesh"]).encode("utf-8"), "application/json; charset=utf-8",
                              {"Cache-Control": "private, max-age=3600"})
        return self._json({"error": "not found"}, 404)

    def _upload(self, uid: str, what: str, download: bool) -> None:
        item = self.app.uploads.get(uid)
        if item is None:
            return self._json({"error": "This uploaded file is no longer in memory."}, 410)
        if what == "text":
            entry = att_mod.jev_entries([att_mod.upload_attachment(item)])[0]
            return self._json({"name": item["name"], "text": entry["content"],
                               "truncated": entry["content"].endswith("[truncated]"),
                               "full_chars": len(item["text"] or "")})
        return self._file(item["data"], att_mod.MEDIA_TYPES[item["media"]], item["name"], download, False)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def _static(self, name: str) -> None:
        target = (STATIC_DIR / name).resolve()
        if STATIC_DIR.resolve() not in target.parents or not target.is_file():
            return self._json({"error": "not found"}, 404)
        types = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png"}
        self._send(200, target.read_bytes(), types.get(target.suffix, "application/octet-stream"))

    def do_POST(self) -> None:  # noqa: N802
        if not self._allowed():
            return
        # JSON-only POSTs: a random web page cannot trigger calls without a CORS preflight.
        # Match the media type exactly: "text/plain; application/json" would skip the preflight.
        media_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if urlparse(self.path).path == "/api/uploads":
            return self._receive_upload(media_type)
        if media_type != "application/json":
            return self._json({"error": "JSON body required"}, 415)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 1_000_000:
                return self._json({"error": "body too large"}, 413)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError):
            return self._json({"error": "invalid JSON"}, 400)
        path = urlparse(self.path).path
        app = self.app
        if not isinstance(payload, dict):
            return self._json({"error": "JSON object required"}, 400)
        since, boot = payload.pop("since", None), payload.pop("boot", None)
        try:
            if path == "/api/run":
                ids = payload.get("ids") or list(app.order)
                ids = [i for i in ids if i in app.items]
                result = app.run(ids, bool(payload.get("use_cache", True)), bool(payload.get("front", False)))
                return self._json({"ok": True, **result, "state": app.snapshot(since, boot)})
            if path == "/api/stop":
                app.stop()
                return self._json({"ok": True, "state": app.snapshot(since, boot)})
            if path == "/api/reset":
                app.reset()
                return self._json({"ok": True, "state": app.snapshot(since, boot)})
            if path == "/api/emails":
                eid = app.add_email(payload)
                return self._json({"ok": True, "id": eid, "email": app.public_email(app.emails[eid]),
                                   "state": app.snapshot(since, boot)})
            if path == "/api/thresholds":
                app.set_thresholds(payload)
                return self._json({"ok": True, "state": app.snapshot(since, boot)})
            if path == "/api/test":
                return self._json(app.test_connection())
        except ValueError as exc:
            return self._json({"ok": False, "error": {"message": str(exc)}}, 400)
        return self._json({"error": "not found"}, 404)


    def _receive_upload(self, media_type: str) -> None:
        """Raw file body (not JSON, not a form): the non-simple Content-Type still forces a CORS
        preflight, so another site cannot post files here. The bytes decide the real type."""
        if media_type not in ("application/pdf", "image/png", "image/jpeg", "image/jpg", "application/octet-stream"):
            return self._json({"ok": False, "error": {"message": "Only PDF, PNG, and JPG files can be attached."}}, 415)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length <= 0:
            return self._json({"ok": False, "error": {"message": "The file is empty."}}, 400)
        if length > att_mod.MAX_UPLOAD_BYTES:
            self.close_connection = True
            return self._json({"ok": False, "error": {"message": "Files can be up to 10 MB each."}}, 413)
        try:
            data = self.rfile.read(length)
        except OSError:
            return self._json({"ok": False, "error": {"message": "The upload was interrupted."}}, 400)
        name = unquote(self.headers.get("X-File-Name") or "upload")
        try:
            item = self.app.uploads.add(name, data)
        except ValueError as exc:
            return self._json({"ok": False, "error": {"message": str(exc)}}, 400)
        log(f"Upload {item['name']} ({item['size']:,} bytes, {item['media']}"
            + (f", {len(item['text']):,} characters of text" if item["media"] == "pdf" else "") + ")")
        return self._json({"ok": True, "upload": att_mod.upload_public(item)})


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second copy of the demo silently share the port.
    allow_reuse_address = os.name != "nt"


def bind(host: str, port: int) -> ThreadingHTTPServer:
    last_error: Optional[OSError] = None
    for candidate in range(port, port + 15):
        try:
            return DemoHTTPServer((host, candidate), Handler)
        except OSError as exc:
            last_error = exc
    raise SystemExit(f"Could not open a port near {port}: {last_error}")


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def main() -> None:
    cfg = jev_client.resolve_config()  # loads .env first, so the settings below can live there too
    parser = argparse.ArgumentParser(description="RFQ Router demo (Jev)")
    # PORT is what cloud hosts (Cloud Run, Render, Railway, Heroku) tell the app to listen on.
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("RFQ_DEMO_PORT") or os.environ.get("PORT") or 8765))
    parser.add_argument("--host", default=os.environ.get("RFQ_DEMO_HOST") or "127.0.0.1",
                        help="0.0.0.0 lets other machines connect; it needs RFQ_DEMO_PASSWORD")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    password = os.environ.get("RFQ_DEMO_PASSWORD", "").strip()
    public = not is_loopback(args.host)
    if public and not password:
        raise SystemExit(f"  Set RFQ_DEMO_PASSWORD before serving on {args.host}. Without it, anyone who "
                         "can reach this server could use your Jev key. Or use --host 127.0.0.1.")

    disable_quickedit()
    app = App(cfg)
    Handler.app = app
    Handler.password = password
    server = bind(args.host, args.port)
    port = server.server_address[1]
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{port}/"

    lines = ["", "  RFQ Router demo  |  Jev by TypeSafe AI",
             "  -------------------------------------------------------------"]
    if cfg:
        lines.append(f"  Jev:    {cfg.provider}, model {cfg.model}")
        lines.append(f"  Key:    {cfg.key_env} = {cfg.masked_key()}")
        files = jev_client.env_files()
        if files:
            lines.append(f"  From:   {', '.join(str(p) for p in files)}")
    else:
        lines.append("  Jev:    NO API KEY FOUND. Cached results can still be replayed.")
        lines.append("          Set AI_GATEWAY_API_KEY=vck_... as an environment variable, or in .env")
        lines.append("          next to server.py or in the folder above it.")
    lines += [f"  Cache:  {len(app.cache)} saved Jev results ({app.cache.seed_count} from data/saved_results.json)",
              f"  Open:   {url}" + (f"  (listening on {args.host}:{port} for other machines)" if public else "")]
    ocr_info = att_mod._ocr().available() if att_mod._ocr() else {}
    lines.append("  OCR:    " + (f"Tesseract {ocr_info.get('tesseract_version') or ''}".strip() if ocr_info.get("ocr")
                                 else "Tesseract not found; scans use the committed OCR cache only"))
    lines.append(f"  Inbox:  {len(app.order)} emails from {EMAILS_FILE.relative_to(HERE) if EMAILS_FILE.is_relative_to(HERE) else EMAILS_FILE}")
    if not att_mod.pypdf_available():
        lines.append("  Note:   pypdf is not installed, so Jev sees only the names of uploaded PDFs.")
        lines.append("          Run: pip install -r requirements.txt")
    if password:
        lines.append("  Login:  any user name, password = RFQ_DEMO_PASSWORD")
    lines += ["  Stop:   close this window or press Ctrl+C", ""]
    print("\n".join(lines), flush=True)

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    # docker stop and cloud hosts send SIGTERM: shut down the same way as Ctrl+C.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n  Stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
