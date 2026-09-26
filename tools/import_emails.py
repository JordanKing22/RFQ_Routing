"""
Send many Outlook emails to a running demo in one go, route them, and score the lanes.

    python tools/import_emails.py URL PATH... [--password PW] [--route] [--check]

    python tools/import_emails.py http://127.0.0.1:8765 tests/emails --route --check
    python tools/import_emails.py https://your-demo.onrender.com ~/Downloads/rfqs --route

PATH is a .eml, .msg, or .zip file, or a folder (searched for those files, not recursively into
dot folders). Each file goes to POST /api/import, the same as dropping it on Import emails in the
page. --route asks Jev about the emails that were added and waits for the answers. --check compares
each email's lane with the answer key in tests/emails/manifest.json (matched by subject) and prints
the score. The password comes from --password or RFQ_DEMO_PASSWORD. Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "emails" / "manifest.json"
TYPES = (".eml", ".msg", ".zip")


def find_files(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in TYPES
                          and not any(part.startswith(".") for part in f.relative_to(p).parts))
        elif p.is_file():
            out.append(p)
        else:
            print(f"  not found: {p}")
    return out


class Demo:
    def __init__(self, url: str, password: str):
        self.url = url.rstrip("/")
        self.auth = "Basic " + base64.b64encode(f"import:{password}".encode()).decode()

    def request(self, path: str, data: Optional[bytes] = None, ctype: str = "application/json",
                headers: Optional[Dict[str, str]] = None, timeout: float = 600) -> Tuple[int, Any]:
        h = {"Authorization": self.auth, **(headers or {})}
        if data is not None:
            h["Content-Type"] = ctype
        req = urllib.request.Request(self.url + path, data=data, headers=h, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read() or b"{}")
            except ValueError:
                return exc.code, {}

    def post_json(self, path: str, body: Dict[str, Any]) -> Tuple[int, Any]:
        return self.request(path, json.dumps(body).encode(), "application/json")


def answer_key() -> Dict[str, Dict[str, Any]]:
    if not MANIFEST.is_file():
        return {}
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {e["subject"]: e for f in data.get("files", []) for e in f.get("emails", [])}


def main() -> int:
    ap = argparse.ArgumentParser(description="Import Outlook emails into a running demo.")
    ap.add_argument("url", help="the demo's address, for example http://127.0.0.1:8765")
    ap.add_argument("paths", nargs="+", help=".eml, .msg, or .zip files, or folders of them")
    ap.add_argument("--password", default=os.environ.get("RFQ_DEMO_PASSWORD", ""))
    ap.add_argument("--route", action="store_true", help="ask Jev about the added emails and wait for the answers")
    ap.add_argument("--check", action="store_true", help="score the lanes against tests/emails/manifest.json")
    args = ap.parse_args()

    demo = Demo(args.url, args.password)
    files = find_files(args.paths)
    if not files:
        print("No .eml, .msg, or .zip files found.")
        return 1
    added: List[Dict[str, Any]] = []
    for f in files:
        status, body = demo.request("/api/import", f.read_bytes(), "application/octet-stream",
                                    {"X-File-Name": urllib.parse.quote(f.name)})
        if status == 401:
            print("The demo asked for a password: pass --password or set RFQ_DEMO_PASSWORD.")
            return 1
        if status != 200 or not body.get("ok"):
            print(f"  {f.name}: failed ({status}) {(body.get('error') or {}).get('message', '')}")
            continue
        added += body["added"]
        print(f"  {f.name}: {len(body['added'])} added, {len(body['duplicates'])} already in the inbox, "
              f"{len(body['skipped'])} left out")
        for s in body["skipped"]:
            print(f"      left out {s['source']}: {s['reason']}")
    print(f"{len(added)} emails added from {len(files)} files.")
    if not (args.route or args.check) or not added:
        return 0

    ids = [a["id"] for a in added]
    demo.post_json("/api/run", {"ids": ids, "use_cache": True})
    print("Routing. On the AI Gateway free tier Jev answers about one email a minute.")
    items: Dict[str, Any] = {}
    last = -1
    while True:
        status, st = demo.request("/api/state")
        items = st.get("items", {})
        done = [i for i in ids if items.get(i, {}).get("status") in ("done", "error")]
        if len(done) != last:
            print(f"  {len(done)} of {len(ids)} answered")
            last = len(done)
        if len(done) == len(ids):
            break
        if st.get("worker", {}).get("state") == "stopped":
            print(f"  Jev stopped: {st['worker'].get('message')}")
            break
        time.sleep(3)

    key = answer_key() if args.check else {}
    right = scored = 0
    for a in added:
        item = items.get(a["id"], {})
        lane = (item.get("decision") or {}).get("lane") or item.get("status")
        want = key.get(a["subject"], {}).get("lane")
        mark = ""
        if want:
            scored += 1
            right += lane == want
            mark = "ok" if lane == want else f"expected {want}"
        print(f"  {a['id']:<5} {str(lane):<14} {mark:<22} {a['subject'][:70]}")
    if args.check:
        print(f"Lanes right: {right} of {scored} with an answer key"
              + (f" ({100 * right / scored:.0f}%)" if scored else "") + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
