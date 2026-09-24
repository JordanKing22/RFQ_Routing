"""
Minimal client for Jev, TypeSafe AI's System One decision model.

Standard library only, so there is nothing to pip install.

Two ways in, picked automatically from your .env:
  * Vercel AI Gateway  AI_GATEWAY_API_KEY -> https://ai-gateway.vercel.sh/typesafe/v1/systemone
                                             model "typesafe-ai/jev"
  * TypeSafe direct    TYPESAFE_API_KEY   -> https://api.typesafe.ai/v1/systemone
                                             model "jev-latest"

Optional overrides: JEV_BASE_URL, JEV_MODEL, JEV_TIMEOUT_SECONDS.

A request is a `state` (the thing being judged) plus named `questions`. Jev
answers every question in parallel and returns typed answers with calibrated
probabilities:
  noul   -> probability the statement is true
  choice -> picked option, probability per option, confidence
  score  -> expected level on an ordered rubric, probability per level, confidence
"""

from __future__ import annotations

import email.utils
import json
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent

GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"
GATEWAY_MODEL = "typesafe-ai/jev"
TYPESAFE_BASE_URL = "https://api.typesafe.ai"
TYPESAFE_MODEL = "jev-latest"
SYSTEM_ONE_PATH = "/v1/systemone"

# List price for Jev input tokens (output tokens are free). Used only when the
# provider does not report a cost for the call.
LIST_PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000

_loaded_env_files: List[Path] = []


# --------------------------------------------------------------------------- #
# .env handling
# --------------------------------------------------------------------------- #
def env_search_paths() -> List[Path]:
    """This folder, the parent folder (Jev_Test/.env), then ~/Jev_Test/.env as a fallback."""
    seen, paths = set(), []
    for path in (HERE / ".env", HERE.parent / ".env", Path.home() / "Jev_Test" / ".env"):
        try:
            key = str(path.resolve()).lower()
        except OSError:
            key = str(path).lower()
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return paths


def load_env(paths: Optional[List[Path]] = None) -> List[Path]:
    """Read KEY=VALUE lines into os.environ. Real environment variables win."""
    found = []
    for path in paths or env_search_paths():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[7:].lstrip()
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            if key and key not in os.environ:
                os.environ[key] = value
        found.append(path)
    _loaded_env_files[:] = found
    return found


def env_files() -> List[Path]:
    return list(_loaded_env_files)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class JevConfig:
    api_key: str
    base_url: str
    model: str
    provider: str
    key_env: str
    timeout: float = 20.0

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + SYSTEM_ONE_PATH

    @property
    def host(self) -> str:
        return self.base_url.split("://", 1)[-1].split("/", 1)[0]

    def masked_key(self) -> str:
        key = self.api_key
        return f"{key[:6]}...({len(key)} chars)" if len(key) > 8 else "(short key)"

    def public(self) -> Dict[str, Any]:
        """Safe-to-display summary. Never includes the key itself."""
        return {
            "configured": True,
            "provider": self.provider,
            "model": self.model,
            "host": self.host,
            "endpoint": self.endpoint,
            "key_env": self.key_env,
            "key_hint": self.masked_key(),
            "env_files": [str(p) for p in env_files()],
        }


def resolve_config() -> Optional[JevConfig]:
    """Build a config from environment variables and .env files, or None."""
    load_env()
    gateway_key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
    typesafe_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    base_override = os.environ.get("JEV_BASE_URL", "").strip()
    model_override = os.environ.get("JEV_MODEL", "").strip()
    try:
        timeout = float(os.environ.get("JEV_TIMEOUT_SECONDS", "20"))
    except ValueError:
        timeout = 20.0

    if gateway_key:
        cfg = JevConfig(gateway_key, GATEWAY_BASE_URL, GATEWAY_MODEL,
                        "Vercel AI Gateway", "AI_GATEWAY_API_KEY", timeout)
    elif typesafe_key:
        cfg = JevConfig(typesafe_key, TYPESAFE_BASE_URL, TYPESAFE_MODEL,
                        "TypeSafe API", "TYPESAFE_API_KEY", timeout)
    else:
        return None
    if base_override:
        cfg.base_url = base_override
        cfg.provider += " (custom URL)"
    if model_override:
        cfg.model = model_override
    return cfg


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class JevError(Exception):
    """A failed Jev call. `kind` drives how the caller reacts.

    kinds: auth, billing, rate_limit, validation, not_found, overloaded,
           server, network, timeout, bad_response
    """

    def __init__(self, message: str, kind: str, status: Optional[int] = None,
                 retry_after: Optional[float] = None, detail: Any = None):
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.retry_after = retry_after
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        return {"message": str(self), "kind": self.kind, "status": self.status,
                "retry_after": self.retry_after, "hint": hint_for(self)}


def hint_for(err: JevError) -> str:
    if err.kind == "auth":
        return ("The API key was rejected. Check AI_GATEWAY_API_KEY in Jev_Test/.env "
                "(it should start with vck_) or create a new key in the Vercel dashboard "
                "under AI Gateway > API Keys.")
    if err.kind == "billing":
        return ("AI Gateway needs credits for this call. Add credits in the Vercel dashboard "
                "under AI Gateway. Jev costs $0.042 per million input tokens, so $5 lasts a long time.")
    if err.kind == "rate_limit":
        return ("The AI Gateway free tier only allows a handful of Jev calls every few minutes. The demo waits "
                "and continues automatically. Buying any amount of AI Gateway credits removes the limit.")
    if err.kind == "validation":
        return "Jev rejected the request format. Share this message with whoever set up the demo."
    if err.kind in ("network", "timeout"):
        return ("Could not reach the Jev API. Check your internet connection, VPN, or firewall, "
                "then try again.")
    if err.kind in ("overloaded", "server"):
        return "The Jev service had a temporary problem. Try again in a minute."
    return ""


def _parse_retry_after(headers: Any) -> Optional[float]:
    if headers is None:
        return None
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            return max(0.0, float(ms) / 1000.0)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError, IndexError):
        return None


def _error_message(body: Any, fallback: str) -> str:
    """Pull a readable message out of the several error shapes in the wild."""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
        if body.get("message"):
            return str(body["message"])
        detail = body.get("detail")
        if isinstance(detail, list) and detail:
            first = detail[0]
            if isinstance(first, dict):
                loc = ".".join(str(p) for p in first.get("loc", []))
                return f"{loc}: {first.get('msg', 'invalid')}".strip(": ")
        if isinstance(detail, str):
            return detail
    if isinstance(body, str) and body.strip():
        return body.strip()[:300]
    return fallback


def _http_error(status: int, raw: bytes, headers: Any) -> JevError:
    try:
        body: Any = json.loads(raw.decode("utf-8", errors="replace")) if raw else None
    except ValueError:
        body = raw.decode("utf-8", errors="replace") if raw else None
    message = _error_message(body, f"HTTP {status}")
    if status in (401, 403):
        kind = "auth"
    elif status == 402:
        kind = "billing"
    elif status == 429:
        kind = "rate_limit"
    elif status in (400, 422):
        kind = "validation"
    elif status == 404:
        kind = "not_found"
    elif status == 529 or status == 503:
        kind = "overloaded"
    elif status >= 500:
        kind = "server"
    else:
        kind = "bad_response"
    # Some gateways return 403 for "model not available on your plan".
    if status == 403 and any(w in message.lower() for w in ("credit", "billing", "payment", "plan", "tier")):
        kind = "billing"
    return JevError(f"Jev API returned {status}: {message}", kind, status,
                    _parse_retry_after(headers), body)


# --------------------------------------------------------------------------- #
# Response normalization
# --------------------------------------------------------------------------- #
def _confidence_from(probs: Dict[str, float]) -> Optional[float]:
    """Fallback only: TypeSafe's documented statistic, (k * p_max - 1) / (k - 1)."""
    if not probs:
        return None
    k = len(probs)
    if k < 2:
        return 1.0
    p_max = max(probs.values())
    return max(0.0, min(1.0, (k * p_max - 1.0) / (k - 1.0)))


def normalize_answer(answer: Dict[str, Any], question: Dict[str, Any]) -> Dict[str, Any]:
    """One shape for the UI and the router, whichever API flavor answered."""
    kind = answer.get("type") or question.get("type")
    if kind in ("noul", "boolean"):
        p = answer.get("noul", answer.get("probability"))
        if p is None:
            raise JevError("Jev answer is missing its probability", "bad_response", detail=answer)
        return {"type": "noul", "p": float(p)}

    if kind == "choice":
        probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
        choice = answer.get("choice") or (max(probs, key=probs.get) if probs else None)
        if choice is None:
            raise JevError("Jev choice answer is empty", "bad_response", detail=answer)
        conf = answer.get("confidence")
        estimated = conf is None
        if estimated:
            conf = _confidence_from(probs)
        return {"type": "choice", "choice": str(choice),
                "p": probs.get(str(choice)), "confidence": None if conf is None else float(conf),
                "confidence_estimated": estimated, "probabilities": probs}

    if kind == "score":
        probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
        score = answer.get("score")
        if score is None and probs:
            score = sum(int(k) * v for k, v in probs.items())
        if score is None:
            raise JevError("Jev score answer is empty", "bad_response", detail=answer)
        conf = answer.get("confidence")
        estimated = conf is None
        if estimated:
            conf = _confidence_from(probs)
        legend = answer.get("legend") or {
            str(i): c for i, c in enumerate(question.get("criteria") or [])}
        return {"type": "score", "score": float(score),
                "confidence": None if conf is None else float(conf),
                "confidence_estimated": estimated, "probabilities": probs,
                "legend": {str(k): v for k, v in legend.items()}}

    return {"type": str(kind), "raw": answer}


def _first(d: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if isinstance(d, dict) and d.get(key) is not None:
            return d[key]
    return None


def normalize_response(data: Dict[str, Any], questions: Dict[str, Dict[str, Any]],
                       latency_ms: float, headers: Any) -> Dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        raise JevError("Jev response did not include answers", "bad_response", detail=data)
    answers = {}
    for name, question in questions.items():
        raw = data["answers"].get(name)
        if raw is None:
            raise JevError(f"Jev response is missing the answer to '{name}'", "bad_response", detail=data)
        answers[name] = normalize_answer(raw, question)

    usage = data.get("usage") or {}
    input_tokens = _first(usage, "input_tokens", "inputTokens")
    output_tokens = _first(usage, "output_tokens", "outputTokens")
    meta = _first(data, "provider_metadata", "providerMetadata") or {}
    gateway = meta.get("gateway") if isinstance(meta, dict) else None
    cost = None
    if isinstance(gateway, dict) and gateway.get("cost") is not None:
        try:
            cost = float(gateway["cost"])
        except (TypeError, ValueError):
            cost = None
    list_price = (input_tokens or 0) * LIST_PRICE_PER_INPUT_TOKEN
    request_id = None
    if headers is not None:
        request_id = headers.get("x-typesafe-request-id") or headers.get("x-vercel-id")
    return {
        "answers": answers,
        "model": data.get("model"),
        "latency_ms": round(latency_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost if cost is not None else list_price,
        "cost_is_estimate": cost is None,
        "list_price_usd": list_price,
        "generation_id": gateway.get("generationId") if isinstance(gateway, dict) else None,
        "request_id": request_id,
    }


# --------------------------------------------------------------------------- #
# The call
# --------------------------------------------------------------------------- #
_ssl_context: Optional[ssl.SSLContext] = None


def _context() -> ssl.SSLContext:
    global _ssl_context
    if _ssl_context is None:
        _ssl_context = ssl.create_default_context()
    return _ssl_context


def system_one(cfg: JevConfig, state: Any, questions: Dict[str, Dict[str, Any]],
               timeout: Optional[float] = None) -> Dict[str, Any]:
    """POST /v1/systemone once. Raises JevError on any failure (no retries here)."""
    payload = {"model": cfg.model, "state": state, "questions": questions}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        cfg.endpoint, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "rfq-router-demo/1.0 (python-urllib)",
        })
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout or cfg.timeout,
                                    context=_context()) as response:
            raw = response.read()
            headers = response.headers
    except urllib.error.HTTPError as exc:
        raw = b""
        try:
            raw = exc.read()
        except Exception:  # noqa: BLE001 - body is best effort
            pass
        raise _http_error(exc.code, raw, exc.headers) from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise JevError(f"Timed out talking to {cfg.host}", "timeout") from None
        raise JevError(f"Could not reach {cfg.host}: {reason}", "network") from None
    except (socket.timeout, TimeoutError):
        raise JevError(f"Timed out talking to {cfg.host}", "timeout") from None
    except (ConnectionError, OSError) as exc:
        raise JevError(f"Connection to {cfg.host} failed: {exc}", "network") from None

    latency_ms = (time.perf_counter() - started) * 1000.0
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise JevError("Jev returned something that is not JSON", "bad_response",
                       detail=raw[:300].decode("utf-8", errors="replace")) from None
    return normalize_response(data, questions, latency_ms, headers)


def ping(cfg: JevConfig) -> Dict[str, Any]:
    """Smallest useful call: one yes/no question about one sentence."""
    questions = {"is_rfq": {"type": "noul",
                            "instructions": "Is the sender asking for a price quote?"}}
    return system_one(cfg, "Please quote 25 pcs of the attached aluminum bracket.", questions)
