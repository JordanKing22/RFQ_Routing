"""
Tests for the RFQ Router demo. Standard library only (pypdf, when installed, adds PDF checks).

    python -m unittest discover -s tests -v
    python tests/test_demo.py

The server tests start tests/mock_jev.py and server.py on free ports, with a temporary cache
and password, so they never touch your real cache or spend Jev calls.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import attachments  # noqa: E402
import docgen  # noqa: E402
import drawings  # noqa: E402
import router  # noqa: E402

try:
    import pypdf  # noqa: F401
    HAVE_PYPDF = True
except Exception:  # noqa: BLE001 - a broken install counts as missing
    HAVE_PYPDF = False

DATA = json.loads((ROOT / "data" / "sample_emails.json").read_text(encoding="utf-8"))
SHOP = json.loads((ROOT / "shop_config.json").read_text(encoding="utf-8"))
EXPORT_WORDS = re.compile(r"\b(itar|ear|eccn|cui|export|controlled|dfars|nist|u\.?s\.? persons?|defen[cs]e|"
                          r"military|classified|restricted|arms)\b", re.IGNORECASE)


def all_specs():
    for email in DATA["emails"]:
        for att in email.get("attachments") or []:
            if isinstance(att, dict):
                yield email["id"], att
    for i, ex in enumerate(DATA.get("paste_examples") or [], start=1):
        for att in ex.get("attachments") or []:
            if isinstance(att, dict):
                yield f"X{i}", att


def make_png(w: int, h: int) -> bytes:
    raw = b"".join(b"\x00" + b"\x30\x60\x90" * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


# --------------------------------------------------------------------------- #
class DatasetTests(unittest.TestCase):
    def test_hundred_emails_with_the_answer_key_mix(self):
        emails = DATA["emails"]
        self.assertEqual(len(emails), 100)
        self.assertEqual([e["id"] for e in emails], [f"E{i:02d}" for i in range(1, 101)])
        lanes = Counter(e["expected"]["lane"] for e in emails[20:])
        self.assertEqual(dict(lanes), {"milling_3axis": 16, "milling_5axis": 10, "turning": 14, "itar": 8,
                                       "orders": 13, "review": 6, "filtered": 13})

    def test_three_itar_emails_are_marked_only_in_an_attachment(self):
        hidden = []
        for e in DATA["emails"]:
            if e["expected"]["lane"] != "itar":
                continue
            text = " ".join([e["subject"], e["body"], e["from_name"], e["from_email"]])
            marked = [a for a in e["attachments"] if isinstance(a, dict) and a.get("legend") in docgen.EXPORT_LEGENDS]
            if not EXPORT_WORDS.search(text) and marked:
                hidden.append(e["id"])
        self.assertEqual(len(hidden), 3, hidden)

    def test_export_legends_only_on_restricted_lane_emails(self):
        for e in DATA["emails"]:
            for a in e["attachments"]:
                if isinstance(a, dict) and a.get("legend") in docgen.EXPORT_LEGENDS:
                    self.assertEqual(e["expected"]["lane"], "itar", (e["id"], a["name"]))

    def test_sample_emails_have_attachment_specs(self):
        for e in DATA["emails"]:
            for a in e["attachments"]:
                self.assertIsInstance(a, dict, e["id"])
                self.assertIn(a.get("kind"), attachments.SPEC_KINDS, (e["id"], a))

    def test_no_em_or_en_dashes_anywhere(self):
        skip_dirs = {".git", "cache", "__pycache__"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or skip_dirs & set(path.relative_to(ROOT).parts):
                continue
            if path.suffix.lower() not in {".py", ".md", ".html", ".json", ".yaml", ".yml", ".txt", ".bat",
                                           ".sh", ".css", ".js", ""}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for bad in ("\u2014", "\u2013"):
                self.assertNotIn(bad, text, f"{path.relative_to(ROOT)} contains U+{ord(bad):04X}")

    def test_known_companies_are_not_real_ones(self):
        # A spot check: the inbox should only use invented companies and example.* personal mail.
        for e in DATA["emails"]:
            domain = e["from_email"].split("@")[-1].lower()
            self.assertNotIn(domain, {"gmail.com", "yahoo.com", "outlook.com", "boeing.com", "lockheedmartin.com",
                                      "raytheon.com", "ge.com", "honeywell.com"}, e["id"])


# --------------------------------------------------------------------------- #
class DocumentTests(unittest.TestCase):
    def test_every_sample_file_renders(self):
        for owner, att in all_specs():
            with self.subTest(owner=owner, name=att["name"]):
                data, ctype = attachments.spec_file(att)
                self.assertTrue(data)
                if att["kind"] == "model":
                    self.assertEqual(ctype, "model/step")
                    text = data.decode("utf-8")
                    self.assertTrue(text.startswith("ISO-10303-21;"))
                    self.assertTrue(text.rstrip().endswith("END-ISO-10303-21;"))
                    defined = set(re.findall(r"^#(\d+)=", text, re.MULTILINE))
                    used = set(re.findall(r"#(\d+)", text))
                    self.assertFalse(used - defined, "STEP references undefined entities")
                    mesh = attachments.spec_mesh(att)
                    n = len(mesh["vertices"])
                    self.assertTrue(n and all(0 <= i < n for face in mesh["faces"] for i in face))
                else:
                    self.assertEqual(ctype, "application/pdf")
                    self.assertTrue(data.startswith(b"%PDF-1.4"))
                    self.assertTrue(data.rstrip().endswith(b"%%EOF"))
                ET.fromstring(attachments.spec_thumb(att))  # the thumbnail is well-formed SVG

    def test_pdf_xref_offsets_point_at_objects(self):
        page = docgen.Page(*docgen.LETTER)
        page.text(40, 40, "Hello (world) \\ ±0.5° Ø12", 12, True)
        page.rect(20, 20, 100, 50, fill=(0.9, 0.9, 0.9))
        page.circle(200, 200, 30)
        pdf = docgen.to_pdf([page, page], title="Test")
        start = int(re.search(rb"startxref\n(\d+)", pdf).group(1))
        entries = re.findall(rb"(\d{10}) 00000 n ", pdf[start:])
        for i, off in enumerate(entries, start=1):
            self.assertTrue(pdf[int(off):].startswith(f"{i} 0 obj".encode()))

    @unittest.skipUnless(HAVE_PYPDF, "pypdf not installed")
    def test_pdf_text_extracts_title_block_and_legends(self):
        from pypdf import PdfReader
        for owner, att in all_specs():
            if att["kind"] == "model":
                continue
            with self.subTest(owner=owner, name=att["name"]):
                data, _ = attachments.spec_file(att)
                text = " ".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
                squeezed = re.sub(r"\s+", " ", text).upper()
                key = att.get("part_number") or att.get("rfq_number") or att.get("po_number") or att.get("title")
                self.assertIn(docgen.clean(str(key)).upper()[:12], squeezed)
                if att.get("legend") == "itar":
                    self.assertIn("INTERNATIONAL TRAFFIC IN ARMS", squeezed)
                if att.get("legend") == "cui":
                    self.assertIn("CUI", squeezed)

    def test_spec_text_carries_what_jev_needs(self):
        for owner, att in all_specs():
            with self.subTest(owner=owner, name=att["name"]):
                text = attachments.spec_text(att)
                self.assertTrue(text)
                if att["kind"] == "drawing":
                    self.assertIn("MATERIAL:", text)
                    self.assertIn("FINISH:", text)
                if att["kind"] == "rfq_form":
                    self.assertIn("QUANTITIES", text)
                if att.get("legend") in ("itar", "ear", "cui"):
                    self.assertTrue(router.EXPORT_MARKINGS.search(text))

    def test_every_shape_renders_in_both_units(self):
        shapes = sorted(drawings.PRISMATIC | drawings.COMPLEX | drawings.ROUND | drawings.OTHER)
        for shape in shapes:
            for units in ("in", "mm"):
                size = ([40, 30, 12] if units == "mm" else [4, 3, 1.2])
                if shape in drawings.ROUND2:
                    size = size[:2]
                elif shape in drawings.ROUND3:
                    size = [size[0], size[1], size[1] * 0.6]
                spec = {"name": f"{shape}.pdf", "kind": "drawing", "company": "Test Co", "title": shape.upper(),
                        "part_number": "T-1", "rev": "A", "material": "AL 6061-T6", "shape": shape, "size": size,
                        "units": units, "notes": ["NOTE ONE.", "NOTE TWO.", "NOTE THREE."], "legend": "cui"}
                with self.subTest(shape=shape, units=units):
                    started = time.perf_counter()
                    data, _ = attachments.spec_file(spec)
                    self.assertLess(time.perf_counter() - started, 1.0)
                    self.assertTrue(data.startswith(b"%PDF"))
                    model = dict(spec, name=f"{shape}.step", kind="model")
                    self.assertIn("FACETED_BREP", attachments.spec_file(model)[0].decode())


# --------------------------------------------------------------------------- #
class RouterTests(unittest.TestCase):
    def test_question_set_mentions_attachments(self):
        self.assertEqual(router.QUESTION_SET_VERSION, "rfq-cnc-v2")
        qs = router.build_questions(SHOP)
        for name in ("export_controlled", "outside_processing", "drawings_provided", "quantity_given"):
            self.assertIn("the email or its attachments", qs[name]["instructions"].lower().replace("do the ", "the "))

    def test_jev_state_includes_attachment_text(self):
        email = next(e for e in DATA["emails"] if e["id"] == "E01")
        state = router.jev_state(email)
        self.assertIsInstance(state["attachments"], list)
        first = state["attachments"][0]
        self.assertEqual(first["file"], email["attachments"][0]["name"])
        self.assertIn("MATERIAL", first["content"])
        self.assertEqual(router.jev_state({"subject": "x", "body": "y"})["attachments"], "none")

    def test_attachment_text_budget(self):
        big = {"name": "big.pdf", "kind": "upload", "media": "pdf", "text": "word " * 5000}
        entries = attachments.jev_entries([big, big, big, big, big])
        self.assertTrue(all(len(e["content"]) <= attachments.JEV_TEXT_PER_FILE + 20 for e in entries))
        self.assertLessEqual(sum(len(e["content"]) for e in entries), attachments.JEV_TEXT_TOTAL + 400)

    def test_cad_detection_uses_kinds(self):
        atts = [{"name": "drawing.pdf", "kind": "drawing"}, {"name": "PO_1.pdf", "kind": "po"},
                {"name": "RFQ-1.pdf", "kind": "rfq_form"}, {"name": "part.step", "kind": "model"},
                "sketch.pdf", "PO_4411.pdf", "RFQ_form.pdf", "part.stp",
                {"name": "photo.jpg", "kind": "upload", "media": "jpg"}]
        self.assertEqual(router.cad_attachments(atts), ["drawing.pdf", "part.step", "sketch.pdf", "part.stp"])

    def test_export_marking_is_traced_to_the_attachment(self):
        email = next(e for e in DATA["emails"] if e["expected"]["lane"] == "itar"
                     and not EXPORT_WORDS.search(e["subject"] + " " + e["body"]))
        answers = fake_answers(export=0.9)
        decision = router.decide(email, answers, SHOP)
        self.assertEqual(decision["lane"], "itar")
        self.assertTrue(decision["export_marked"])
        self.assertIn("Export-control wording found in", decision["trace"][0]["detail"])

    def test_same_lane_email_types_count_together(self):
        email = next(e for e in DATA["emails"] if e["id"] == "E10")

        def split(a, b):
            answers = fake_answers()
            probs = {t: 0.02 for t in router.EMAIL_TYPES}
            probs[a], probs[b] = 0.46, 0.46
            answers["email_type"] = {"type": "choice", "choice": a, "p": 0.46, "confidence": 0.352,
                                     "probabilities": probs}
            return router.decide(email, answers, SHOP)
        self.assertEqual(split("purchase_order", "order_followup")["lane"], "orders")
        self.assertEqual(split("new_rfq", "quote_revision")["lane"], "milling_3axis")
        # Filtering archives mail with no reply, so a vendor / other split still goes to a person.
        self.assertEqual(split("vendor_or_solicitation", "other")["lane"], "review")
        # So does a split between lanes.
        self.assertEqual(split("purchase_order", "new_rfq")["lane"], "review")


def fake_answers(export=0.02, email_type="new_rfq", process="milling_3axis"):
    def choice(options, winner):
        probs = {o: (0.8 if o == winner else 0.2 / (len(options) - 1)) for o in options}
        return {"type": "choice", "choice": winner, "p": 0.8, "confidence": 0.7, "probabilities": probs}
    return {
        "email_type": choice(list(router.EMAIL_TYPES), email_type),
        "process": choice(["milling_3axis", "milling_5axis", "turning", "mixed_or_unclear"], process),
        "export_controlled": {"type": "noul", "p": export},
        "urgency": {"type": "score", "score": 1.0, "confidence": 0.6, "probabilities": {"0": .1, "1": .8, "2": .05, "3": .05}},
        "quantity_given": {"type": "noul", "p": 0.9},
        "drawings_provided": {"type": "noul", "p": 0.9},
        "outside_processing": {"type": "noul", "p": 0.2},
        "volume": choice(["prototype", "low_volume", "production", "not_stated"], "low_volume"),
    }


# --------------------------------------------------------------------------- #
class UploadTests(unittest.TestCase):
    def test_sniff_and_sizes(self):
        png = make_png(33, 21)
        self.assertEqual(attachments.sniff(png), "png")
        self.assertEqual(attachments.image_size(png, "png"), (33, 21))
        jpg = (b"\xff\xd8\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
               + b"\xff\xc0" + struct.pack(">HBHH", 17, 8, 480, 640) + b"\x00" * 12)
        self.assertEqual(attachments.sniff(jpg), "jpg")
        self.assertEqual(attachments.image_size(jpg, "jpg"), (640, 480))
        self.assertEqual(attachments.sniff(b"%PDF-1.7\n..."), "pdf")
        self.assertIsNone(attachments.sniff(b"<html><script>alert(1)</script>"))

    def test_safe_filenames(self):
        self.assertEqual(attachments.safe_filename("../../etc/passwd", "png"), "passwd.png")
        self.assertEqual(attachments.safe_filename("C:\\Users\\x\\scan.PDF", "pdf"), "scan.PDF")
        self.assertEqual(attachments.safe_filename('bad"<name>.jpg', "jpg"), "badname.jpg")
        self.assertTrue(attachments.safe_filename("x" * 300 + ".pdf", "pdf").endswith(".pdf"))

    def test_store_limits(self):
        store = attachments.UploadStore(max_total=500)
        with self.assertRaises(ValueError):
            store.add("a.txt", b"hello")
        with self.assertRaises(ValueError):
            store.add("big.png", b"\x89PNG\r\n\x1a\n" + b"0" * attachments.MAX_UPLOAD_BYTES)
        first = store.add("a.png", make_png(8, 8))
        for _ in range(20):
            store.add("b.png", make_png(8, 8))
        self.assertIsNone(store.get(first["id"]))  # evicted past the memory budget
        self.assertLessEqual(store.total(), 500)

    def test_tidy_pdf_text(self):
        self.assertEqual(attachments.tidy_pdf_text("8\n8\nD D\nMATERIAL: 6061-T6\nA RELEASED TW\n12"),
                         "MATERIAL: 6061-T6\nA RELEASED TW")

    @unittest.skipUnless(HAVE_PYPDF, "pypdf not installed")
    def test_extract_pdf_text_in_a_child_process(self):
        spec = next(att for _, att in all_specs() if att["kind"] == "drawing")
        data, _ = attachments.spec_file(spec)
        result = attachments.extract_pdf_text(data)
        self.assertIsNone(result["error"])
        self.assertEqual(result["pages"], 1)
        self.assertIn(spec["part_number"], result["text"])
        broken = attachments.extract_pdf_text(b"%PDF-1.4\nthis is not a pdf")
        self.assertTrue(broken["error"])


# --------------------------------------------------------------------------- #
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Client:
    def __init__(self, port: int, password: str = "pw"):
        self.base = f"http://127.0.0.1:{port}"
        self.auth = "Basic " + base64.b64encode(f"any:{password}".encode()).decode()

    def call(self, path, body=None, raw=None, ctype="application/json", headers=None, auth=True):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        h = dict(headers or {})
        if auth:
            h["Authorization"] = self.auth
        if data is not None:
            h["Content-Type"] = ctype
        req = urllib.request.Request(self.base + path, data=data, method="POST" if data is not None else "GET", headers=h)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers, exc.read()

    def json(self, path, body=None, **kw):
        status, _, data = self.call(path, body, **kw)
        return status, json.loads(data or b"{}")


def wait_http(url: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except urllib.error.HTTPError:
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    raise RuntimeError(f"{url} did not come up")


class ServerTests(unittest.TestCase):
    """End to end against the mock Jev endpoint."""

    procs: list = []

    @classmethod
    def start_mock(cls, *args: str) -> int:
        port = free_port()
        proc = subprocess.Popen([sys.executable, str(ROOT / "tests" / "mock_jev.py"), "--port", str(port),
                                 "--latency-ms", "5", *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.procs.append(proc)
        wait_http(f"http://127.0.0.1:{port}/debug/requests")
        return port

    @classmethod
    def start_server(cls, mock_port: int, cache_dir: str, seed: str, **env: str) -> int:
        port = free_port()
        environ = dict(os.environ, JEV_BASE_URL=f"http://127.0.0.1:{mock_port}", AI_GATEWAY_API_KEY="mock-key",
                       RFQ_DEMO_PASSWORD="pw", RFQ_CACHE_DIR=cache_dir, RFQ_SEED_FILE=seed,
                       RFQ_EMAILS_FILE=str(ROOT / "data" / "sample_emails.json"), **env)
        environ.pop("TYPESAFE_API_KEY", None)
        proc = subprocess.Popen([sys.executable, str(ROOT / "server.py"), "--no-browser", "--port", str(port)],
                                env=environ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ROOT))
        cls.procs.append(proc)
        wait_http(f"http://127.0.0.1:{port}/healthz")
        return port

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.mock_port = cls.start_mock()
        cls.seed = os.path.join(cls.tmp.name, "seed.json")
        cls.port = cls.start_server(cls.mock_port, os.path.join(cls.tmp.name, "cache1"), cls.seed)
        cls.c = Client(cls.port)

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

    def mock_requests(self, port=None):
        with urllib.request.urlopen(f"http://127.0.0.1:{port or self.mock_port}/debug/requests") as resp:
            return json.loads(resp.read())

    def route_all(self, client: Client):
        status, boot = client.json("/api/bootstrap")
        self.assertEqual(status, 200)
        version, boot_id = boot["state"]["version"], boot["state"]["boot_id"]
        status, run = client.json("/api/run", {"use_cache": True, "since": version, "boot": boot_id})
        version = run["state"]["version"]
        deadline = time.time() + 60
        while time.time() < deadline:
            status, st = client.json(f"/api/state?since={version}&boot={boot_id}")
            self.assertFalse(st["full"])
            version = st["version"]
            if st["stats"]["routed"] == st["stats"]["total"] and st["worker"]["state"] == "idle":
                return run, st
            time.sleep(0.2)
        self.fail("routing did not finish")

    def test_1_password_and_healthz(self):
        status, _, _ = self.c.call("/healthz", auth=False)
        self.assertEqual(status, 200)
        status, _, _ = self.c.call("/api/bootstrap", auth=False)
        self.assertEqual(status, 401)
        status, _, _ = self.c.call("/api/att/E01/0/file", auth=False)
        self.assertEqual(status, 401)

    def test_2_route_everything_with_deltas_then_replay_after_restart(self):
        run, st = self.route_all(self.c)
        self.assertEqual(run["queued"], 100)
        self.assertEqual(st["stats"]["routed"], 100)
        # once idle, a delta is empty
        status, idle = self.c.json(f"/api/state?since={st['version']}&boot={st['boot_id']}")
        self.assertEqual(idle["items"], {})
        self.assertNotIn("order", idle)
        # Jev saw the attachment text
        seen = self.mock_requests()["requests"]
        states = [r["state"] for r in seen]
        with_text = [s for s in states if isinstance(s.get("attachments"), list)
                     and any("ENGINEERING DRAWING" in a["content"] for a in s["attachments"])]
        self.assertGreater(len(with_text), 50)
        # download saved results, use them as the seed of a fresh server with an empty cache
        status, headers, body = self.c.call("/api/saved_results")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        saved = json.loads(body)
        self.assertGreaterEqual(len(saved["records"]), 100)
        Path(self.seed).write_bytes(body)
        calls_before = self.mock_requests()["calls"]
        port2 = self.start_server(self.mock_port, os.path.join(self.tmp.name, "cache2"), self.seed)
        c2 = Client(port2)
        status, boot2 = c2.json("/api/bootstrap")
        self.assertEqual(boot2["state"]["stats"]["seed_size"], len(saved["records"]))
        status, run2 = c2.json("/api/run", {"use_cache": True})
        self.assertEqual((run2["queued"], run2["cached"]), (0, 100))
        self.assertEqual(run2["state"]["stats"]["routed"], 100)
        self.assertEqual(self.mock_requests()["calls"], calls_before)  # no new Jev calls

    def test_3_threshold_change_resends_decisions(self):
        status, st = self.c.json("/api/state")
        status, r = self.c.json("/api/thresholds", {"email_type_confidence": 0.95, "since": st["version"],
                                                     "boot": st["boot_id"]})
        routed = [i for i in r["state"]["items"].values() if i["status"] == "done"]
        self.assertEqual(len(routed), st["stats"]["routed"])
        self.c.json("/api/thresholds", {"email_type_confidence": 0.5})

    def test_4_attachment_endpoints(self):
        status, boot = self.c.json("/api/bootstrap")
        email = next(e for e in boot["emails"] if e["id"] == "E01")
        drawing, model = email["attachments"][0], email["attachments"][1]
        status, headers, body = self.c.call(drawing["url"])
        self.assertEqual((status, headers["Content-Type"]), (200, "application/pdf"))
        self.assertIn("inline", headers["Content-Disposition"])
        status, headers, body = self.c.call(drawing["url"] + "&download=1")
        self.assertIn("attachment", headers["Content-Disposition"])
        status, headers, body = self.c.call(drawing["thumb"])
        self.assertEqual(headers["Content-Type"], "image/svg+xml")
        ET.fromstring(body)
        status, text = self.c.json(drawing["text_url"])
        self.assertIn("MATERIAL", text["text"])
        status, mesh = self.c.json(model["mesh"])
        self.assertTrue(mesh["vertices"] and mesh["faces"])
        status, headers, body = self.c.call(model["url"])
        self.assertTrue(body.startswith(b"ISO-10303-21;"))
        self.assertEqual(self.c.call("/api/att/E01/9/file")[0], 404)
        self.assertEqual(self.c.call("/api/att/X1/0/thumb.svg")[0], 200)

    def test_5_uploads_reach_jev(self):
        spec = next(att for _, att in all_specs() if att["kind"] == "drawing" and att.get("legend") == "itar")
        pdf, _ = attachments.spec_file(spec)
        status, up = self.c.json("/api/uploads", raw=pdf, ctype="application/pdf",
                                 headers={"X-File-Name": "customer%20drawing.pdf"})
        self.assertEqual(status, 200, up)
        self.assertEqual(up["upload"]["name"], "customer drawing.pdf")
        status, img = self.c.json("/api/uploads", raw=make_png(40, 30), ctype="image/png",
                                  headers={"X-File-Name": "photo.png"})
        self.assertEqual((img["upload"]["width"], img["upload"]["height"]), (40, 30))
        self.assertEqual(self.c.json("/api/uploads", raw=b"<html></html>", ctype="application/pdf")[0], 400)
        self.assertEqual(self.c.json("/api/uploads", raw=b"hello", ctype="text/plain")[0], 415)
        self.assertEqual(self.c.json("/api/uploads", raw=b"x", ctype="application/json")[0], 415)
        self.assertEqual(self.c.json("/api/uploads", raw=b"", ctype="application/pdf")[0], 400)
        status, sent = self.c.json("/api/emails", {
            "from_name": "Pat Doe", "from_email": "pat@example.com", "subject": "Quote please",
            "body": "Can you quote 10 pcs of the attached part?", "example": 0, "example_files": [],
            "uploads": [up["upload"]["id"], img["upload"]["id"]], "attachments": "notes.docx"})
        self.assertEqual(status, 200, sent)
        kinds = [a["kind"] for a in sent["email"]["attachments"]]
        self.assertEqual(kinds, ["upload", "upload", "name_only"])
        eid = sent["id"]
        deadline = time.time() + 20
        while time.time() < deadline:
            status, st = self.c.json("/api/state")
            if st["items"][eid]["status"] == "done":
                break
            time.sleep(0.2)
        decision = st["items"][eid]["decision"]
        if HAVE_PYPDF:
            self.assertEqual(decision["lane"], "itar")
            self.assertEqual(decision["export_marked"], ["customer drawing.pdf"])
        status, headers, body = self.c.call(f"/api/att/{eid}/0/file")
        self.assertEqual((status, body), (200, pdf))
        status, headers, body = self.c.call(f"/api/att/{eid}/1/file")
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertEqual(self.c.call(f"/api/att/{eid}/2/file")[0], 404)
        status, err = self.c.json("/api/emails", {"subject": "x", "uploads": ["0" * 16]})
        self.assertEqual(status, 400)
        status, err = self.c.json("/api/emails", {"subject": "x", "uploads": ["a"] * 6})
        self.assertEqual(status, 400)

    def test_6_paste_example_files_come_along(self):
        status, boot = self.c.json("/api/bootstrap")
        example = boot["paste_examples"][0]
        names = [a["name"] for a in example["attachments"]]
        status, sent = self.c.json("/api/emails", {"subject": example["subject"], "body": example["body"],
                                                   "from_name": example["from_name"], "from_email": example["from_email"],
                                                   "example": 0, "example_files": names[:1]})
        self.assertEqual([a["name"] for a in sent["email"]["attachments"]], names[:1])
        self.assertEqual(sent["email"]["attachments"][0]["kind"], "drawing")

    def test_7_free_tier_pacing_after_a_429(self):
        mock = self.start_mock("--rate-limit-every", "2")
        port = self.start_server(mock, os.path.join(self.tmp.name, "cache3"), os.path.join(self.tmp.name, "none.json"),
                                 JEV_FREE_TIER_PACE_SECONDS="0.6", JEV_RATE_LIMIT_WAIT_SECONDS="0.4")
        c = Client(port)
        status, r = c.json("/api/run", {"ids": ["E01", "E02", "E03"], "use_cache": False})
        waited = False
        deadline = time.time() + 30
        while time.time() < deadline:
            status, st = c.json("/api/state")
            waited = waited or st["worker"]["state"] == "waiting"
            if all(st["items"][i]["status"] == "done" for i in ("E01", "E02", "E03")):
                break
            time.sleep(0.05)
        self.assertTrue(all(st["items"][i]["status"] == "done" for i in ("E01", "E02", "E03")))
        self.assertTrue(st["worker"]["rate_limited"])
        self.assertEqual(st["worker"]["pace_seconds"], 0.6)
        self.assertTrue(waited)


if __name__ == "__main__":
    unittest.main(verbosity=2)
