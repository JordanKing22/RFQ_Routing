"""
RFQ Router demo server. Standard library only.

    python server.py            start on http://127.0.0.1:8765 and open the browser
    python server.py --port 9000 --no-browser

The browser never sees your API key: the page talks to this local server, and
this server talks to Jev.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue as queue_mod
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import jev_client
import router

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
CONFIG_FILE = HERE / "shop_config.json"
EMAILS_FILE = HERE / "data" / "sample_emails.json"
CACHE_FILE = HERE / "cache" / "jev_results.json"

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
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.records: Dict[str, Dict[str, Any]] = {}
        if path.is_file():
            try:
                self.records = json.loads(path.read_text(encoding="utf-8")).get("records", {})
            except (ValueError, OSError) as exc:
                log(f"Cache file unreadable ({exc}); starting fresh.")

    @staticmethod
    def key(state: Any, questions: Dict[str, Any], base_url: str, model: str) -> str:
        blob = json.dumps({"v": router.QUESTION_SET_VERSION, "endpoint": base_url.rstrip("/"),
                           "model": model, "state": state, "questions": questions},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.records.get(key)

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

    def __len__(self) -> int:
        return len(self.records)


def cache_key_for(cfg: Optional[jev_client.JevConfig], state: Any, questions: Dict[str, Any]) -> str:
    base = cfg.base_url if cfg else jev_client.GATEWAY_BASE_URL
    model = cfg.model if cfg else jev_client.GATEWAY_MODEL
    return ResultCache.key(state, questions, base, model)


# --------------------------------------------------------------------------- #
# App state + background worker
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, jev: Optional[jev_client.JevConfig]):
        self.jev = jev
        self.shop = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        data = json.loads(EMAILS_FILE.read_text(encoding="utf-8"))
        self.paste_examples = data.get("paste_examples", [])
        self.emails: Dict[str, Dict[str, Any]] = {}
        self.order: List[str] = []
        for email in data["emails"]:
            self.emails[email["id"]] = email
            self.order.append(email["id"])
        self.questions = router.build_questions(self.shop)
        self.thresholds = dict(router.DEFAULT_THRESHOLDS)
        self.thresholds.update(self.shop.get("thresholds", {}))
        self.cache = ResultCache(CACHE_FILE)
        self.items: Dict[str, Dict[str, Any]] = {eid: self._blank_item() for eid in self.order}
        self.queue: deque = deque()
        self.cond = threading.Condition(threading.RLock())
        # Bumped by Stop and Reset. The worker drops work that started under an older generation.
        self.generation = 0
        self.running_gen: Optional[int] = None  # generation of the item the worker is on
        self.version = 0
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

    @staticmethod
    def _blank_item() -> Dict[str, Any]:
        return {"status": "idle", "source": None, "record": None, "error": None, "routed_at": None}

    def _bump(self) -> None:
        self.version += 1

    def key_for(self, email: Dict[str, Any]) -> str:
        return cache_key_for(self.jev, router.jev_state(email), self.questions)

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
                if use_cache:
                    record = self.cache.get(self.key_for(email))
                    if record:
                        item.update(status="done", source="cache", record=record, error=None,
                                    routed_at=time.time())
                        cached += 1
                        continue
                if self.jev is None:
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
        attachments = payload.get("attachments") or []
        if isinstance(attachments, str):
            attachments = [a.strip() for a in attachments.replace(";", ",").split(",") if a.strip()]
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
                "attachments": [str(a)[:120] for a in attachments][:20],
                "live": True,
            }
            self.order.append(eid)
            self.items[eid] = self._blank_item()
            self._bump()
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
                if item.get("use_cache"):
                    record = self.cache.get(self.key_for(email))
                    if record:
                        item.update(status="done", source="cache", record=record, error=None,
                                    routed_at=time.time())
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
        item = self.items[eid]
        record = item.get("record")
        if not record:
            return None
        return router.decide(self.emails[eid], record["answers"], self.shop, self.thresholds)

    def snapshot(self) -> Dict[str, Any]:
        with self.cond:
            items = {}
            latencies: List[float] = []
            cost = list_price = 0.0
            routed = auto = review = matches = graded = 0
            cost_reported = False
            for eid in self.order:
                item = self.items[eid]
                entry: Dict[str, Any] = {"status": item["status"], "source": item["source"],
                                         "error": item["error"], "routed_at": item["routed_at"]}
                record = item.get("record")
                if record and item["status"] == "done":
                    decision = self._decision(eid)
                    entry["decision"] = decision
                    entry["jev"] = {k: record.get(k) for k in (
                        "answers", "model", "latency_ms", "input_tokens", "output_tokens", "cost_usd",
                        "cost_is_estimate", "list_price_usd", "saved_at", "generation_id", "request_id")}
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
                items[eid] = entry
            worker = dict(self.worker)
            worker["queue_length"] = len(self.queue)
            worker["pace_seconds"] = self.pace_seconds
            worker["now"] = time.time()
            latencies.sort()
            return {
                "version": self.version,
                "order": list(self.order),
                "items": items,
                "worker": worker,
                "thresholds": dict(self.thresholds),
                "stats": {
                    "total": len(self.order), "routed": routed, "auto": auto, "review": review,
                    "matches": matches, "graded": graded,
                    "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
                    "median_latency_ms": latencies[len(latencies) // 2] if latencies else None,
                    "cost_usd": cost, "list_price_usd": list_price, "cost_reported": cost_reported,
                    "cache_size": len(self.cache),
                },
            }

    def bootstrap(self) -> Dict[str, Any]:
        lanes = router.lane_index(self.shop)
        with self.cond:
            emails = [self.emails[eid] for eid in self.order]
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
            "paste_examples": self.paste_examples,
            "customers": {c["domain"].lower(): c for c in self.shop.get("customers", [])},
            "jev": jev_info,
            "free_tier_pace_seconds": FREE_TIER_PACE,
            "state": self.snapshot(),
        }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    app: App = None  # type: ignore[assignment]
    server_version = "RFQRouterDemo/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet: the worker logs Jev calls
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1", "")

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._json({"error": "forbidden host"}, 403)
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if path == "/api/bootstrap":
            return self._json(self.app.bootstrap())
        if path == "/api/state":
            return self._json(self.app.snapshot())
        return self._json({"error": "not found"}, 404)

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
        if not self._host_ok():
            return self._json({"error": "forbidden host"}, 403)
        # JSON-only POSTs: a random web page cannot trigger calls without a CORS preflight.
        if "application/json" not in (self.headers.get("Content-Type") or ""):
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
        try:
            if path == "/api/run":
                ids = payload.get("ids") or list(app.order)
                ids = [i for i in ids if i in app.items]
                result = app.run(ids, bool(payload.get("use_cache", True)), bool(payload.get("front", False)))
                return self._json({"ok": True, **result, "state": app.snapshot()})
            if path == "/api/stop":
                app.stop()
                return self._json({"ok": True, "state": app.snapshot()})
            if path == "/api/reset":
                app.reset()
                return self._json({"ok": True, "state": app.snapshot()})
            if path == "/api/emails":
                eid = app.add_email(payload)
                return self._json({"ok": True, "id": eid, "email": app.emails[eid], "state": app.snapshot()})
            if path == "/api/thresholds":
                app.set_thresholds(payload)
                return self._json({"ok": True, "state": app.snapshot()})
            if path == "/api/test":
                return self._json(app.test_connection())
        except ValueError as exc:
            return self._json({"ok": False, "error": {"message": str(exc)}}, 400)
        return self._json({"error": "not found"}, 404)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="RFQ Router demo (Jev)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("RFQ_DEMO_PORT", "8765")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    disable_quickedit()
    cfg = jev_client.resolve_config()
    app = App(cfg)
    Handler.app = app
    server = bind(args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"

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
        lines.append(f"          Add AI_GATEWAY_API_KEY=vck_... to {HERE.parent / '.env'}")
    lines += [f"  Cache:  {len(app.cache)} saved Jev results in cache/jev_results.json",
              f"  Open:   {url}",
              "  Stop:   close this window or press Ctrl+C", ""]
    print("\n".join(lines), flush=True)

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n  Stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
