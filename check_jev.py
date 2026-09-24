"""
Check that your Jev API key works, using one real call.

    python check_jev.py          route sample email E01 with the full question set
    python check_jev.py --ping   one tiny yes/no question instead

The E01 result is saved to the demo cache, so this call is not wasted.
Exit code 0 = working, 1 = not working.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jev_client
import router
from server import CACHE_FILE, CONFIG_FILE, EMAILS_FILE, ResultCache, cache_key_for

HERE = Path(__file__).resolve().parent
OK_MARKER = HERE / "cache" / ".jev_ok"


def bar(p: float, width: int = 20) -> str:
    filled = int(round(max(0.0, min(1.0, p)) * width))
    return "#" * filled + "." * (width - filled)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the Jev API key")
    parser.add_argument("--ping", action="store_true", help="send one tiny question only")
    args = parser.parse_args()

    print()
    print("  Jev API check")
    print("  -------------------------------------------------------------")
    cfg = jev_client.resolve_config()
    if cfg is None:
        print("  [FAIL] No API key found.")
        print("         Looked in: " + ", ".join(str(p) for p in jev_client.env_search_paths()))
        print("         Add a line like:  AI_GATEWAY_API_KEY=vck_your_key_here")
        return 1
    print(f"  Provider: {cfg.provider}")
    print(f"  Endpoint: {cfg.endpoint}")
    print(f"  Model:    {cfg.model}")
    print(f"  Key:      {cfg.key_env} = {cfg.masked_key()}")
    print()

    try:
        if args.ping:
            result = jev_client.ping(cfg)
            p = result["answers"]["is_rfq"]["p"]
            print(f"  Q: Is 'Please quote 25 pcs of the attached aluminum bracket.' a quote request?")
            print(f"  A: yes, {p:.0%}  [{bar(p)}]")
        else:
            shop = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            email = json.loads(EMAILS_FILE.read_text(encoding="utf-8"))["emails"][0]
            questions = router.build_questions(shop)
            state = router.jev_state(email)
            print(f"  Routing sample email {email['id']}: \"{email['subject']}\"")
            print(f"  ({len(questions)} questions in one request)")
            result = jev_client.system_one(cfg, state, questions)
            cache = ResultCache(CACHE_FILE)
            cache.put(cache_key_for(cfg, state, questions), email["id"], result)
            decision = router.decide(email, result["answers"], shop,
                                     dict(router.DEFAULT_THRESHOLDS, **shop.get("thresholds", {})))
            print()
            for name, answer in result["answers"].items():
                label = router.QUESTION_LABELS.get(name, name)
                if answer["type"] == "noul":
                    shown = f"yes {answer['p']:.0%}"
                elif answer["type"] == "choice":
                    shown = f"{answer['choice']} ({(answer.get('p') or 0):.0%}, confidence {(answer.get('confidence') or 0):.0%})"
                else:
                    shown = f"{answer['score']:.2f} of 3 (confidence {(answer.get('confidence') or 0):.0%})"
                print(f"    {label:<40} {shown}")
            print()
            print(f"  Routed to: {decision['lane_name']} ({decision['owner']}), priority {decision['priority']}")
    except jev_client.JevError as exc:
        print(f"  [FAIL] {exc}")
        hint = jev_client.hint_for(exc)
        if hint:
            print(f"         {hint}")
        if exc.kind == "rate_limit":
            print("         Your key works, you are just rate limited right now.")
            OK_MARKER.parent.mkdir(parents=True, exist_ok=True)
            OK_MARKER.write_text("rate limited but key accepted\n", encoding="utf-8")
            return 0
        return 1

    cost = result.get("cost_usd") or 0.0
    note = "estimated at list price" if result.get("cost_is_estimate") else "reported by AI Gateway"
    print()
    print(f"  [OK] Jev answered in {result['latency_ms']:.0f} ms using model {result['model']}.")
    print(f"       {result.get('input_tokens') or '?'} input tokens, cost ${cost:.6f} ({note}).")
    OK_MARKER.parent.mkdir(parents=True, exist_ok=True)
    OK_MARKER.write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
