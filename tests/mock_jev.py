"""
A stand-in for Jev's /v1/systemone endpoint, for rehearsing and testing without a key.

    python tests/mock_jev.py --port 8799
    JEV_BASE_URL=http://127.0.0.1:8799 AI_GATEWAY_API_KEY=mock python server.py

It answers every question with keyword rules over the state, in the same JSON shape the real
API uses. It is not Jev: the answers are rough guesses, good enough to exercise the demo.

Options:
    --latency-ms 40      delay each answer
    --rate-limit-every 5 answer every 5th call with HTTP 429 (Retry-After: 2) to test pacing
    --reject-key         answer every call with HTTP 401

GET /debug/requests returns the states it received (newest last, up to 300).
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

LOCK = threading.Lock()
SEEN: List[Dict[str, Any]] = []
CALLS = {"n": 0}


def text_of(state: Any) -> str:
    return json.dumps(state, ensure_ascii=False).lower() if not isinstance(state, str) else state.lower()


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


def choice(options: List[str], winner: str, strength: float = 0.8) -> Dict[str, Any]:
    if winner not in options:
        winner = options[0]
    rest = [o for o in options if o != winner]
    probs = {winner: strength}
    for o in rest:
        probs[o] = round((1.0 - strength) / max(1, len(rest)), 4)
    k = len(options)
    confidence = max(0.0, min(1.0, (k * strength - 1.0) / (k - 1.0))) if k > 1 else 1.0
    return {"type": "choice", "choice": winner, "probabilities": probs, "confidence": round(confidence, 4)}


def answer(name: str, question: Dict[str, Any], state: Any) -> Dict[str, Any]:
    t = text_of(state)
    kind = question.get("type")
    body = t
    if isinstance(state, dict):
        body = " ".join(str(state.get(k, "")) for k in ("subject", "body")).lower()
    if kind == "noul":
        if name == "export_controlled":
            p = 0.92 if has(r"\bitar\b|\bcui\b|\bear\b|eccn|export|dfars|u\.s\. persons|arms export", t) else 0.04
            if has(r"we are itar registered|itar registered", t) and has(r"our services|we offer|special pricing|capabilit", t):
                p = 0.55
        elif name == "quantity_given":
            p = 0.88 if has(r"\b\d[\d,]*\s*(pcs|pieces|each|ea|units|parts)\b|qty|quantit|annual|/yr|per year|\d+\s*/\s*\d+", t) else 0.12
        elif name == "drawings_provided":
            p = 0.9 if has(r"engineering drawing|\.step|\.stp|drawing attached|attached drawing|dwg no|3d model", t) else 0.15
        elif name == "outside_processing":
            p = 0.85 if has(r"anodiz|plat(e|ed|ing)\b|passivat|heat treat|powder coat|chem film|black oxide|electroless|paint", t) else 0.1
        else:
            p = 0.9
        return {"type": "noul", "noul": p}
    if kind == "choice":
        options = list((question.get("criteria") or {}).keys()) or ["yes", "no"]
        if name == "email_type":
            if has(r"new job|resume|hiring|staffing|promo|% off|webinar|newsletter|expo|demo this week|our services|unsubscribe|lunch|parking|gift card|verify your account", body):
                w = "vendor_or_solicitation"
            elif has(r"\bpo\b|purchase order|please proceed|award", body) and not has(r"quote|rfq|pricing", body):
                w = "purchase_order"
            elif has(r"status|tracking|ship date|shipped|certs|certification|invoice|fai report|on track|rma|nonconform|change order", body):
                w = "order_followup"
            elif has(r"rev [a-z]\b.*(update|revise)|revised|revision|update(d)? quote", body):
                w = "quote_revision"
            elif has(r"quote|rfq|pricing|price|can you make", body):
                w = "new_rfq"
            else:
                w = "other"
            strength = 0.45 if has(r"can you make|what would .* cost|can you do this", body) else 0.82
            return choice(options, w, strength)
        if name == "process":
            if has(r"weld|fabricat|sheet metal|casting|foundry|3d print|injection mold|molded", t):
                w = "mixed_or_unclear"
            elif has(r"5-axis|five-axis|impeller|implant|contour|sculpt|compound|blade|vane|wing rib|structural fitting", t):
                w = "milling_5axis"
            elif has(r"shaft|pin\b|pins\b|bushing|spacer|lathe|turn(ed|ing)|swiss|thread|nozzle|gear blank|roller|stud", t):
                w = "turning"
            elif has(r"plate|bracket|block|housing|cover|manifold|heat sink|heatsink|fixture|enclosure", t):
                w = "milling_3axis"
            else:
                w = "mixed_or_unclear"
            return choice(options, w, 0.78)
        if name == "volume":
            nums = [int(n.replace(",", "")) for n in re.findall(r"\b(\d[\d,]{0,6})\s*(?:pcs|pieces|each|ea|/yr|per year|units)", t)]
            if not nums:
                w = "not_stated"
            elif max(nums) > 250:
                w = "production"
            elif max(nums) > 10:
                w = "low_volume"
            else:
                w = "prototype"
            return choice(options, w, 0.7)
        return choice(options, options[0], 0.7)
    if kind == "score":
        levels = len(question.get("criteria") or []) or 4
        if has(r"urgent|asap|line down|aog|rush|expedite|today|tomorrow", body):
            top = levels - 1
        elif has(r"this week|by friday|end of the week|few days", body):
            top = min(2, levels - 1)
        elif has(r"due|by \w+ \d|within|weeks", body):
            top = min(1, levels - 1)
        else:
            top = 0
        probs = {str(i): (0.76 if i == top else round(0.24 / (levels - 1), 4)) for i in range(levels)}
        score = sum(i * p for i, p in ((int(k), v) for k, v in probs.items()))
        return {"type": "score", "score": round(score, 4), "probabilities": probs, "confidence": 0.68}
    return {"type": kind or "unknown"}


class Handler(BaseHTTPRequestHandler):
    opts: argparse.Namespace = None  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, obj: Any, status: int = 200, headers: Dict[str, str] = None) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/debug/requests"):
            with LOCK:
                return self._json({"calls": CALLS["n"], "requests": SEEN[-300:]})
        return self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/v1/systemone"):
            return self._json({"error": {"message": "not found"}}, 404)
        if not (self.headers.get("Authorization") or "").startswith("Bearer "):
            return self._json({"error": {"message": "missing key"}}, 401)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json({"detail": [{"loc": ["body"], "msg": "invalid JSON"}]}, 422)
        with LOCK:
            CALLS["n"] += 1
            n = CALLS["n"]
            SEEN.append({"n": n, "state": payload.get("state"), "questions": sorted((payload.get("questions") or {}).keys())})
            del SEEN[:-300]
        if self.opts.reject_key:
            return self._json({"error": {"message": "Invalid API key"}}, 401)
        if self.opts.rate_limit_every and n % self.opts.rate_limit_every == 0:
            return self._json({"error": {"message": "Rate limit exceeded"}}, 429, {"Retry-After": "2"})
        if self.opts.latency_ms:
            time.sleep(self.opts.latency_ms / 1000.0)
        state = payload.get("state")
        questions = payload.get("questions") or {}
        answers = {name: answer(name, q, state) for name, q in questions.items()}
        tokens = max(1, len(json.dumps(payload)) // 4)
        return self._json({
            "model": payload.get("model") or "mock-jev",
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
            "provider_metadata": {"gateway": {"cost": "0", "generationId": f"gen_mock_{n:05d}"}},
        }, 200, {"x-typesafe-request-id": f"req_mock_{n:05d}"})


def main() -> None:
    ap = argparse.ArgumentParser(description="Mock Jev /v1/systemone")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--latency-ms", type=float, default=40.0)
    ap.add_argument("--rate-limit-every", type=int, default=0)
    ap.add_argument("--reject-key", action="store_true")
    Handler.opts = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", Handler.opts.port), Handler)
    print(f"Mock Jev listening on http://127.0.0.1:{Handler.opts.port}/v1/systemone", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
