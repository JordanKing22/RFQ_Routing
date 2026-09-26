"""
Tests for importing Outlook emails (.eml, .msg, and .zip files of them).

Every file in tests/emails/ must parse to exactly the emails its manifest entry lists, and a running
server must import them, read their attachments, route them, and pull out their RFQ details.
Adding an email to the suite is: put the file in tests/emails/ (tools/scrub_email.py does this for
real emails, after scrubbing them) and give it a manifest entry with its lane.

    python -m unittest tests.test_import -v
"""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import mailfile  # noqa: E402
from test_demo import Client, free_port, wait_http  # noqa: E402

EMAIL_DIR = HERE / "emails"
MANIFEST = json.loads((EMAIL_DIR / "manifest.json").read_text(encoding="utf-8"))
LANES = {"milling_3axis", "milling_5axis", "turning", "itar", "orders", "review", "filtered"}


def imported(entry):
    """The emails a file gives the importer: a forward bundle stands aside for the emails it carries."""
    result = mailfile.load(entry["file"], (EMAIL_DIR / entry["file"]).read_bytes())
    return [e for e in result["emails"] if not e.get("container_only")], result["skipped"]


class ManifestTests(unittest.TestCase):
    def test_manifest_lists_every_file(self):
        on_disk = {p.name for p in EMAIL_DIR.iterdir() if p.is_file() and p.name != "manifest.json"}
        listed = [f["file"] for f in MANIFEST["files"]]
        self.assertEqual(len(listed), len(set(listed)), "a file is listed twice")
        self.assertEqual(on_disk, set(listed))

    def test_every_file_parses_to_its_emails(self):
        for entry in MANIFEST["files"]:
            with self.subTest(file=entry["file"]):
                emails, skipped = imported(entry)
                self.assertEqual(len(emails), len(entry["emails"]), [e["subject"] for e in emails])
                for got, want in zip(emails, entry["emails"]):
                    self.assertEqual(got["subject"], want["subject"])
                    self.assertEqual(got["from_email"], want["from_email"].lower())
                    self.assertEqual([a["name"] for a in got["attachments"]], want["attachments"])
                    self.assertTrue(got["body"].strip(), "no text came out of the body")
                    self.assertTrue(got["date"], "no date")
                for name in entry.get("skipped", []):
                    self.assertTrue(any(name in s["source"] for s in skipped), (name, skipped))

    def test_answer_key_is_complete(self):
        for entry in MANIFEST["files"]:
            self.assertIn(entry.get("source"), ("generated", "scrubbed"), entry["file"])
            for want in entry["emails"]:
                with self.subTest(file=entry["file"], subject=want["subject"]):
                    self.assertIn(want["lane"], LANES)
                    self.assertIsInstance(want["is_rfq"], bool)
                    if want["lane"] in ("milling_3axis", "milling_5axis", "turning"):
                        self.assertTrue(want["is_rfq"], "an estimating lane means a quote request")

    def test_formats_cover_every_outlook(self):
        formats = {Path(f["file"]).suffix for f in MANIFEST["files"]}
        self.assertTrue({".eml", ".msg", ".zip"} <= formats, formats)

    def test_fixture_text_has_no_dashes(self):
        for entry in MANIFEST["files"]:
            emails, _ = imported(entry)
            for email in emails:
                for text in (email["subject"], email["body"]):
                    self.assertNotIn("\u2014", text, entry["file"])
                    self.assertNotIn("\u2013", text, entry["file"])


class ServerImportTests(unittest.TestCase):
    """The import end to end against the mock Jev endpoint."""

    procs: list = []

    @classmethod
    def start(cls, *cmd, env=None):
        cls.procs.append(subprocess.Popen(list(cmd), env=env, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL, cwd=str(ROOT)))

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        mock_port, cls.port = free_port(), free_port()
        cls.start(sys.executable, str(HERE / "mock_jev.py"), "--port", str(mock_port), "--latency-ms", "5")
        wait_http(f"http://127.0.0.1:{mock_port}/debug/requests")
        env = dict(os.environ, JEV_BASE_URL=f"http://127.0.0.1:{mock_port}", AI_GATEWAY_API_KEY="mock-key",
                   RFQ_DEMO_PASSWORD="pw", RFQ_CACHE_DIR=cls.tmp.name,
                   RFQ_SEED_FILE=os.path.join(cls.tmp.name, "no-seed.json"))
        env.pop("RFQ_EMAILS_FILE", None)
        cls.start(sys.executable, str(ROOT / "server.py"), "--no-browser", "--port", str(cls.port), env=env)
        wait_http(f"http://127.0.0.1:{cls.port}/healthz", timeout=60)
        cls.c = Client(cls.port)
        # Import every fixture once, in manifest order; the tests below look at the result.
        cls.results = []
        for entry in MANIFEST["files"]:
            status, body = cls.post(entry["file"], (EMAIL_DIR / entry["file"]).read_bytes())
            cls.results.append((entry, status, body))

    @classmethod
    def tearDownClass(cls):
        for proc in cls.procs:
            proc.terminate()
        for proc in cls.procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        cls.tmp.cleanup()

    @classmethod
    def post(cls, name, data, ctype="application/octet-stream"):
        return cls.c.json("/api/import", raw=data, ctype=ctype, headers={"X-File-Name": urllib.parse.quote(name)})

    def ids_by_subject(self):
        out = {}
        for _, _, body in self.results:
            for a in body.get("added", []):
                out.setdefault(a["subject"], a["id"])
        return out

    def wait_until(self, check, timeout=240, what="the server"):
        deadline = time.time() + timeout
        while time.time() < deadline:
            value = check()
            if value:
                return value
            time.sleep(0.3)
        self.fail(f"{what} did not finish in {timeout}s")

    def test_1_every_fixture_is_imported(self):
        seen = set()
        for entry, status, body in self.results:
            with self.subTest(file=entry["file"]):
                self.assertEqual(status, 200, body)
                self.assertTrue(body["ok"])
                want_new = [w for w in entry["emails"] if (w["subject"], w["from_email"]) not in seen]
                want_dup = [w for w in entry["emails"] if (w["subject"], w["from_email"]) in seen]
                self.assertEqual([a["subject"] for a in body["added"]], [w["subject"] for w in want_new])
                self.assertEqual([d["subject"] for d in body["duplicates"]], [w["subject"] for w in want_dup])
                for w in entry["emails"]:
                    seen.add((w["subject"], w["from_email"]))
                for name in entry.get("skipped", []):
                    self.assertTrue(any(name in s["source"] for s in body["skipped"]), body["skipped"])
                for a in body["added"]:
                    self.assertRegex(a["id"], r"^M\d{2,4}$")

    def test_2_a_second_import_finds_duplicates(self):
        entry = MANIFEST["files"][0]
        status, body = self.post(entry["file"], (EMAIL_DIR / entry["file"]).read_bytes())
        self.assertEqual(status, 200)
        self.assertEqual(body["added"], [])
        self.assertEqual(len(body["duplicates"]), len(entry["emails"]))

    def test_3_imported_emails_reach_the_inbox_unrouted(self):
        status, boot = self.c.json("/api/bootstrap")
        self.assertTrue(boot["features"]["import"])
        by_id = {e["id"]: e for e in boot["emails"]}
        ids = self.ids_by_subject()
        for entry, _, _ in self.results:
            for want in entry["emails"]:
                email = by_id[ids[want["subject"]]]
                self.assertTrue(email["imported"])
                self.assertTrue(email["date"])
                self.assertEqual([a["name"] for a in email["attachments"]], want["attachments"])
        items = boot["state"]["items"]
        self.assertTrue(all(items[i]["status"] == "idle" for i in ids.values()), "imports are not routed on their own")

    def test_4_attachments_are_read_in_the_background(self):
        ids = self.ids_by_subject()
        eid = ids["Quote request: 316L cover plate AGI-3052, 12 pcs"]

        def read():
            status, boot = self.c.json("/api/bootstrap")
            att = next(e for e in boot["emails"] if e["id"] == eid)["attachments"][0]
            return att if not att.get("reading") and att.get("text_from") else None

        att = self.wait_until(read, what="reading the faxed drawing")
        self.assertTrue(att["text_from"].startswith("OCR"), att)  # a fax: its text came from OCR (the cache)
        status, text = self.c.json(f"/api/att/{eid}/0/text")
        self.assertIn("AGI-3052", text["text"])
        for what in ("file", "thumb.jpg", "page.jpg"):
            self.assertEqual(self.c.call(att["url"].split("?")[0].replace("/file", f"/{what}"))[0], 200, what)
        status, _, data = self.c.call(att["url"])
        self.assertEqual(data, (ROOT / "data" / "rfq_beta" / "files" / "E52" / "AGI-3052_RevA.pdf").read_bytes())

    def test_5_step_models_open_in_3d(self):
        ids = self.ids_by_subject()
        eid = ids["RFQ: heat sink plate FR-2290, price breaks please"]
        status, boot = self.c.json("/api/bootstrap")
        model = next(e for e in boot["emails"] if e["id"] == eid)["attachments"][1]
        self.assertEqual(model["kind"], "model")
        status, mesh = self.c.json(model["mesh"].split("?")[0])
        self.assertEqual(status, 200)
        self.assertTrue(mesh["faces"])

    def test_6_routing_and_rfq_details(self):
        ids = self.ids_by_subject()
        status, body = self.c.json("/api/run", {"ids": list(ids.values()), "use_cache": True})
        self.assertEqual(status, 200)

        def routed():
            status, st = self.c.json("/api/state")
            items = st["items"]
            done = all(items.get(i, {}).get("status") == "done" for i in ids.values())
            return st if done and st["worker"]["state"] == "idle" else None

        st = self.wait_until(routed, what="routing the imported emails")
        # The mock's keyword rules catch export control, including a CUI marking seen only on a fax.
        for entry, _, _ in self.results:
            for want in entry["emails"]:
                decision = st["items"][ids[want["subject"]]]["decision"]
                self.assertTrue(decision and decision["lane"], want["subject"])
                if want["lane"] == "itar":
                    self.assertEqual(decision["lane"], "itar", want["subject"])
        for entry, _, _ in self.results:
            for want in entry["emails"]:
                if not want.get("part_numbers"):
                    continue
                with self.subTest(subject=want["subject"]):
                    status, rec = self.c.json(f"/api/rfq_details/{ids[want['subject']]}")
                    self.assertTrue(rec["ok"], rec)
                    found = [line["part_number"]["value"] for line in rec["record"]["lines"]]
                    for pn in want["part_numbers"]:
                        self.assertIn(pn, found)
        # "within two weeks" counts from the day the email was sent (Sep 21), not from today.
        status, rec = self.c.json(f"/api/rfq_details/{ids['RFQ: pump shaft HPV-3310 Rev C, 150 and 300 pcs']}")
        self.assertEqual(rec["record"]["respond_by"]["value"], "2026-10-05")

    def test_7_bad_requests(self):
        self.assertEqual(self.post("x.json", b"{}", ctype="application/json")[0], 415)
        self.assertEqual(self.post("x.eml", b"", )[0], 400)
        status, body = self.post("not-an-email.eml", b"\x00\x01garbage" * 50)
        self.assertEqual(status, 200)
        self.assertEqual(body["added"], [])
        self.assertTrue(body["skipped"])
        # A declared size over the limit is refused before the body is read.
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", "/api/import")
        conn.putheader("Authorization", self.c.auth)
        conn.putheader("Content-Type", "application/octet-stream")
        conn.putheader("Content-Length", str(mailfile.LIMITS.max_file_bytes + 1))
        conn.endheaders()
        self.assertEqual(conn.getresponse().status, 413)
        conn.close()
        status, _, _ = self.c.call("/api/import", raw=b"x", ctype="application/octet-stream", auth=False)
        self.assertEqual(status, 401)


if __name__ == "__main__":
    unittest.main(verbosity=2)
