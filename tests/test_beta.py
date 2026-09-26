"""
Tests for the RFQ details beta inbox and the server around it: the 30 real files, the uncopyable
ones, OCR through the server, the committed OCR cache, and the consolidated RFQ details download.

    python -m unittest discover -s tests -v

Module tests live next to this file: test_ocr.py (Tesseract pipeline), test_rfq_details.py
(extraction accuracy), test_layout.py (the YOLO region detector).
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import attachments  # noqa: E402
import stepfile  # noqa: E402
from test_demo import Client, free_port, wait_http  # noqa: E402

INBOX = json.loads((ROOT / "data" / "rfq_beta" / "emails.json").read_text(encoding="utf-8"))
OCR_CACHE = ROOT / "data" / "rfq_beta" / "ocr_cache.json"
HAVE_TESSERACT = shutil.which("tesseract") is not None
HAVE_PDFTOPPM = shutil.which("pdftoppm") is not None  # poppler renders the viewer's page images of PDFs
try:
    import pypdf  # noqa: F401
    HAVE_PYPDF = True
except Exception:  # noqa: BLE001
    HAVE_PYPDF = False


def beta_files():
    for email in INBOX["emails"]:
        for att in email["attachments"]:
            yield email["id"], att


def text_layer(path: Path) -> str:
    from pypdf import PdfReader
    return "".join((page.extract_text() or "") for page in PdfReader(str(path)).pages).strip()


class InboxTests(unittest.TestCase):
    def test_exactly_thirty_real_files(self):
        files = list(beta_files())
        self.assertEqual(len(files), 30)
        for eid, att in files:
            with self.subTest(eid=eid, name=att["name"]):
                self.assertEqual(att["kind"], "file")
                path = attachments.resolve_data_path(ROOT / "data", att["path"])
                self.assertIsNotNone(path, "the file must exist inside data/")
                self.assertIn(att["media"], ("pdf", "jpg", "png", "step"))
        on_disk = [p for p in (ROOT / "data" / "rfq_beta" / "files").rglob("*") if p.is_file()]
        self.assertEqual(len(on_disk), 30, "no stray files next to the 30")

    @unittest.skipUnless(HAVE_PYPDF, "pypdf not installed")
    def test_thirteen_files_are_uncopyable(self):
        uncopyable = []
        for eid, att in beta_files():
            path = ROOT / "data" / att["path"]
            if att["media"] in ("jpg", "png"):
                uncopyable.append(att["name"])
            elif att["media"] == "pdf" and not text_layer(path):
                uncopyable.append(att["name"])
        self.assertEqual(len(uncopyable), 13, uncopyable)
        # the export-control markings that exist only in attachments sit on uncopyable files
        for name in ("CI-10442_RevC.pdf", "WS-RFQ-26-0388.pdf", "AGI-3052_RevA.pdf"):
            self.assertIn(name, uncopyable)

    def test_paths_cannot_escape_the_data_folder(self):
        for rel in ("../server.py", "rfq_beta/../../server.py", "/etc/passwd", "rfq_beta/files/../../../x"):
            self.assertIsNone(attachments.resolve_data_path(ROOT / "data", rel))

    def test_answer_key_and_rfqs(self):
        lanes = [e["expected"]["lane"] for e in INBOX["emails"]]
        self.assertEqual(len(lanes), 22)
        self.assertEqual(lanes.count("itar"), 5)
        for eid in ("E10", "E13"):
            self.assertIn(next(e for e in INBOX["emails"] if e["id"] == eid)["expected"]["lane"], ("orders", "filtered"))

    def test_step_files_parse_to_meshes(self):
        for eid, att in beta_files():
            if att["media"] != "step":
                continue
            with self.subTest(name=att["name"]):
                info = stepfile.parse((ROOT / "data" / att["path"]).read_bytes())
                self.assertIsNone(info["error"])
                self.assertTrue(info["part_number"])
                self.assertIn(info["units"], ("in", "mm"))
                self.assertTrue(info["mesh"] and info["mesh"]["faces"])
        self.assertEqual(stepfile.parse(b"not a step file")["error"], "not a STEP file (no ISO-10303-21 header)")


class BetaServerTests(unittest.TestCase):
    """The beta server end to end against the mock Jev endpoint."""

    procs: list = []

    @classmethod
    def start(cls, *cmd: str, env=None) -> None:
        proc = subprocess.Popen(list(cmd), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ROOT))
        cls.procs.append(proc)

    @classmethod
    def start_server(cls, cache_dir: str, **env: str) -> Client:
        port = free_port()
        environ = dict(os.environ, JEV_BASE_URL=f"http://127.0.0.1:{cls.mock_port}", AI_GATEWAY_API_KEY="mock-key",
                       RFQ_DEMO_PASSWORD="pw", RFQ_CACHE_DIR=cache_dir,
                       RFQ_SEED_FILE=os.path.join(cache_dir, "no-seed.json"), **env)
        environ.pop("RFQ_EMAILS_FILE", None)  # the branch default: the beta inbox
        cls.start(sys.executable, str(ROOT / "server.py"), "--no-browser", "--port", str(port), env=environ)
        wait_http(f"http://127.0.0.1:{port}/healthz", timeout=60)
        return Client(port)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.mock_port = free_port()
        cls.start(sys.executable, str(HERE / "mock_jev.py"), "--port", str(cls.mock_port), "--latency-ms", "5")
        wait_http(f"http://127.0.0.1:{cls.mock_port}/debug/requests")
        cls.c = cls.start_server(os.path.join(cls.tmp.name, "cache1"))

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

    def route_all(self, client: Client, timeout: float = 600) -> dict:
        client.json("/api/run", {"use_cache": True})
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, st = client.json("/api/state")
            if st["stats"]["routed"] == st["stats"]["total"] and st["worker"]["state"] == "idle":
                return st
            time.sleep(0.5)
        self.fail("routing the beta inbox did not finish")

    def test_1_bootstrap_describes_real_files(self):
        # Without the committed OCR cache, files are read in the background after startup; asking
        # for a file's text reads its email's files right away.
        self.assertEqual(self.c.call("/api/att/E61/0/text")[0], 200)
        status, boot = self.c.json("/api/bootstrap")
        self.assertEqual(status, 200)
        self.assertEqual(len(boot["emails"]), 22)
        self.assertTrue(boot["features"]["rfq_details"])
        e61 = next(e for e in boot["emails"] if e["id"] == "E61")
        drawing, model = e61["attachments"]
        self.assertEqual((drawing["kind"], drawing["media"]), ("file", "pdf"))
        self.assertEqual(model["kind"], "model")  # a real STEP file opens in the 3D viewer
        for url in (drawing["url"], model["mesh"], model["thumb"]):
            self.assertEqual(self.c.call(url)[0], 200, url)
        self.assertEqual(self.c.call(drawing["page_url"])[0], 200 if HAVE_PDFTOPPM else 404)
        if HAVE_TESSERACT or OCR_CACHE.is_file():
            self.assertTrue(drawing["scanned"])
            self.assertTrue(drawing["text_from"].startswith("OCR"))
            self.assertEqual(drawing["legend"], "itar")
        status, regions = self.c.json(drawing["regions_url"])
        self.assertIn("available", regions)

    @unittest.skipUnless(HAVE_TESSERACT or OCR_CACHE.is_file(), "needs tesseract or the committed OCR cache")
    def test_2_markings_only_on_scans_reach_the_restricted_lane(self):
        st = self.route_all(self.c)
        for eid, marked in (("E61", "CI-10442_RevC.pdf"), ("E72", "WS-RFQ-26-0388.pdf"), ("E52", "AGI-3052_RevA.pdf")):
            decision = st["items"][eid]["decision"]
            self.assertEqual(decision["lane"], "itar", eid)
            self.assertIn(marked, decision["export_marked"], eid)
        status, text = self.c.json("/api/att/E61/0/text")
        self.assertIn("INTERNATIONAL TRAFFIC IN ARMS", " ".join(text["text"].split()).upper())

    def test_3_consolidated_rfq_details(self):
        status, headers, body = self.c.call("/api/rfq_details.csv")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertTrue(body.startswith(b"\xef\xbb\xbf"), "a BOM so Excel reads UTF-8")
        rows = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
        self.assertGreaterEqual(len(rows), 20)
        emails = {r[next(k for k in r if k.lower().startswith("email"))] for r in rows}
        self.assertNotIn("E10", emails)
        self.assertNotIn("E13", emails)
        status, one = self.c.json("/api/rfq_details/E32")
        self.assertTrue(one["ok"])
        quantities = [line["quantities"]["value"] for line in one["record"]["lines"]]
        if HAVE_TESSERACT or OCR_CACHE.is_file():
            self.assertIn([25, 75, 150], quantities)  # printed only on the scanned RFQ form
        self.assertFalse(self.c.json("/api/rfq_details/E10")[1]["ok"])

    @unittest.skipUnless(OCR_CACHE.is_file(), "needs the committed OCR cache")
    def test_4_committed_cache_works_without_tesseract(self):
        # Render could lose Tesseract, or a laptop may never have had it: the committed cache
        # still gives Jev the scans' text, so the export-control catch still works.
        client = self.start_server(os.path.join(self.tmp.name, "cache2"), PATH=os.path.join(self.tmp.name, "empty-bin"))
        status, boot = client.json("/api/bootstrap")
        self.assertFalse(boot["features"]["ocr"])
        e61 = next(e for e in boot["emails"] if e["id"] == "E61")
        self.assertTrue(e61["attachments"][0]["text_from"].startswith("OCR"))
        st = self.route_all(client)
        self.assertEqual(st["items"]["E61"]["decision"]["lane"], "itar")

    @unittest.skipUnless(HAVE_TESSERACT, "needs tesseract")
    def test_5_uploaded_scan_is_read_with_ocr(self):
        scan = (ROOT / "data" / "rfq_beta" / "files" / "E72" / "WS-RFQ-26-0388.pdf").read_bytes()
        status, up = self.c.json("/api/uploads", raw=scan, ctype="application/pdf",
                                 headers={"X-File-Name": "scanned%20rfq.pdf"})
        self.assertEqual(status, 200, up)
        self.assertTrue(up["upload"]["scanned"])
        self.assertGreater(up["upload"]["text_chars"], 200)
        status, sent = self.c.json("/api/emails", {"subject": "Quote please", "body": "See the attached RFQ.",
                                                   "from_email": "buyer@example.com", "uploads": [up["upload"]["id"]]})
        self.assertEqual(status, 200, sent)
        status, text = self.c.json(f"/api/att/{sent['id']}/0/text")
        self.assertIn("WS-4471", text["text"])

    @unittest.skipUnless(HAVE_TESSERACT, "needs tesseract")
    def test_6_uploaded_scan_and_photo_keep_their_title_blocks(self):
        # The extractor gets an upload's whole OCR result (lines with boxes, regions), as it does a
        # committed file's. From the text alone it lost the title blocks: the scan's part line
        # went missing and the photo's finish with it.
        ids = []
        for rel, ctype in (("E61/CI-10442_RevC.pdf", "application/pdf"), ("E22/OPM-22817_RevB_photo.jpg", "image/jpeg")):
            data = (ROOT / "data" / "rfq_beta" / "files" / rel).read_bytes()
            status, up = self.c.json("/api/uploads", raw=data, ctype=ctype, headers={"X-File-Name": rel.split("/")[1]})
            self.assertEqual(status, 200, up)
            self.assertTrue(up["upload"]["scanned"])
            self.assertNotIn("_file_text", up["upload"])  # the whole result stays on the server
            ids.append(up["upload"]["id"])
        status, sent = self.c.json("/api/emails", {"subject": "RFQ: two parts", "from_email": "buyer@example.com",
                                                   "body": "Please quote the two parts on the attached prints.\n\n"
                                                           "Quantities: 10 / 25 pcs", "uploads": ids})
        self.assertEqual(status, 200, sent)
        status, one = self.c.json(f"/api/rfq_details/{sent['id']}")
        self.assertTrue(one["ok"], one)
        lines = {ln["part_number"]["value"]: ln for ln in one["record"]["lines"]}
        self.assertEqual(set(lines), {"CI-10442", "OPM-22817"})
        for pn, material, finish in (("CI-10442", "6061-T6511", "ELECTROLESS NICKEL"), ("OPM-22817", "4140", "BLACK OXIDE")):
            self.assertIn(material, lines[pn]["material"]["value"] or "", pn)
            self.assertIn(finish, lines[pn]["finish"]["value"] or "", pn)


if __name__ == "__main__":
    unittest.main(verbosity=2)
