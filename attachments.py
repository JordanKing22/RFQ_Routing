"""
Email attachments for the demo: sample files built from specs, and real uploaded files.

Sample emails describe each attachment as a spec (see data/sample_emails.json):
    kind "drawing"   engineering drawing PDF with a title block    -> drawings.py lays it out
    kind "model"     STEP 3D model                                  -> drawings.py builds the mesh
    kind "rfq_form"  the customer's request-for-quotation form PDF
    kind "po"        a purchase order PDF
    kind "document"  anything else (resume, brochure, cert, invoice, letter)
From one spec this module makes three things that always agree:
    the file itself (PDF or STEP), the thumbnail tile (SVG), and the text Jev reads.

Uploaded files (Paste an RFQ) are kept in memory only. PDFs are read with pypdf, in a separate
process with a time limit, so a hostile or broken PDF cannot hang the server. Without pypdf the
demo still works; Jev then sees only the file name of an uploaded PDF.

Real files on disk (kind "file", the RFQ details beta inbox in data/rfq_beta) are read the way a
mail system would read them: the PDF text layer when there is one, Tesseract OCR (ocr.py) for scans,
faxes, photos, and screenshots, and the header of a STEP file. Scanned or photographed uploads get
OCR too when Tesseract is installed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import struct
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import docgen
import drawings
import stepfile
from docgen import BLACK, LETTER, Page, clean, legend_text, text_width, wrap

HERE = os.path.dirname(os.path.abspath(__file__))

SPEC_KINDS = ("drawing", "model", "rfq_form", "po", "document")
KIND_LABELS = {
    "drawing": "Drawing", "model": "3D model", "rfq_form": "RFQ form", "po": "Purchase order",
    "document": "Document", "upload": "Uploaded file", "name_only": "File name only", "file": "File",
}
DOC_TYPE_LABELS = {
    "resume": "Resume", "brochure": "Brochure", "cert": "Certificate", "invoice": "Invoice",
    "packing_slip": "Packing slip", "letter": "Letter", "spec": "Spec sheet", "report": "Report",
    "newsletter": "Newsletter",
}
SHOP_ADDRESS = ["Mesa Ridge Precision", "3120 W Foundry Way", "Phoenix, AZ 85043", "quotes@mesaridgeprecision.com"]

# How much attachment text Jev gets. The email body is capped at 8,000 characters in router.py;
# attachments share a separate budget so one long PDF cannot crowd out the email itself.
JEV_TEXT_PER_FILE = 1800
JEV_TEXT_TOTAL = 6000

GREY = (0.42, 0.44, 0.48)
LIGHT = (0.93, 0.93, 0.92)
RULE = (0.55, 0.56, 0.6)


# --------------------------------------------------------------------------- #
# Normalizing attachment entries
# --------------------------------------------------------------------------- #
def normalize(att: Any) -> Dict[str, Any]:
    """Every attachment becomes a dict. A bare string is a file name with no file behind it."""
    if isinstance(att, dict):
        out = dict(att)
        out["name"] = clean(out.get("name") or "attachment")[:120]
        out.setdefault("kind", "name_only")
        return out
    return {"name": clean(att)[:120], "kind": "name_only"}


def names(attachments: Any) -> List[str]:
    return [normalize(a)["name"] for a in (attachments or [])]


def media_of(att: Dict[str, Any]) -> Optional[str]:
    kind = att.get("kind")
    if kind in ("drawing", "rfq_form", "po", "document"):
        return "pdf"
    if kind == "model":
        return "step"
    if kind in ("upload", "file"):
        return att.get("media")
    return None


# --------------------------------------------------------------------------- #
# Text for Jev
# --------------------------------------------------------------------------- #
def _join(parts: List[str]) -> str:
    return "\n".join(p for p in parts if p and p.strip())


def _qty_list(q: Any) -> str:
    if isinstance(q, list) and q:
        return " / ".join(f"{int(x):,}" for x in q)
    return "NOT STATED"


def spec_text(att: Dict[str, Any]) -> str:
    """What a person would read on the sample file, as compact text. Jev gets this."""
    kind = att.get("kind")
    legend = legend_text(att.get("legend"), att.get("company", ""), att.get("eccn", ""),
                         att.get("drawn_by", "") or att.get("buyer", ""))
    if kind == "drawing":
        notes = " ".join(f"{i}. {clean(n)}" for i, n in enumerate(att.get("notes") or [], start=1))
        revs = "; ".join(f"{clean(r.get('rev'))} {clean(r.get('description'))} {clean(r.get('date'))}".strip()
                         for r in att.get("revisions") or [] if isinstance(r, dict))
        return _join([
            "ENGINEERING DRAWING (PDF)",
            f"TITLE BLOCK: {clean(att.get('company'))}. TITLE: {clean(att.get('title'))}. "
            f"DWG NO: {clean(att.get('part_number'))} REV {clean(att.get('rev'))}. "
            f"SCALE {clean(att.get('scale') or '1:1')}. SHEET {clean(att.get('sheet') or '1 OF 1')}."
            + (f" DRAWN: {clean(att.get('drawn_by'))} {clean(att.get('date'))}." if att.get("drawn_by") else ""),
            f"MATERIAL: {clean(att.get('material'))}",
            f"FINISH: {clean(att.get('finish') or 'NONE')}",
            f"UNLESS OTHERWISE SPECIFIED: DIMENSIONS IN {'MILLIMETERS' if att.get('units') == 'mm' else 'INCHES'}. "
            f"TOLERANCES: {clean(att.get('tolerances') or drawings.default_tolerances(att.get('units')))}",
            f"OVERALL SIZE: {drawings.size_phrase(att)}",
            (f"FEATURES: {'; '.join(clean(c) for c in att.get('callouts') or [])}" if att.get("callouts") else ""),
            f"NOTES: {notes}" if notes else "",
            f"REVISIONS: {revs}" if revs else "",
            legend,
        ])
    if kind == "model":
        return _join([
            f"3D CAD MODEL (STEP {clean(att.get('schema') or 'AP214')})",
            f"PART: {clean(att.get('part_number'))} REV {clean(att.get('rev') or '-')}"
            + (f", {clean(att.get('title'))}" if att.get("title") else ""),
            f"UNITS: {'MILLIMETER' if att.get('units') == 'mm' else 'INCH'}. BOUNDING BOX: {drawings.bbox_phrase(att)}.",
            (f"ORIGINATING SYSTEM: {clean(att.get('originating_system'))}." if att.get("originating_system") else ""),
            legend,
        ])
    if kind == "rfq_form":
        lines = []
        for i, line in enumerate(att.get("lines") or [], start=1):
            lines.append(
                f"LINE {i}: P/N {clean(line.get('part_number'))}"
                + (f" REV {clean(line.get('rev'))}" if line.get("rev") else "")
                + f", {clean(line.get('description'))}"
                + (f", MATERIAL {clean(line.get('material'))}" if line.get("material") else "")
                + (f", FINISH {clean(line.get('finish'))}" if line.get("finish") else "")
                + f", QUANTITIES {_qty_list(line.get('quantities'))}"
                + (f", {clean(line.get('notes'))}" if line.get("notes") else "") + ".")
        reqs = " ".join(f"{i}. {clean(r)}" for i, r in enumerate(att.get("requirements") or [], start=1))
        return _join([
            f"REQUEST FOR QUOTATION {clean(att.get('rfq_number'))} FROM {clean(att.get('company'))}",
            f"DATE: {clean(att.get('date'))}. RESPOND BY: {clean(att.get('respond_by'))}. "
            f"BUYER: {clean(att.get('buyer'))}" + (f" ({clean(att.get('buyer_email'))})" if att.get("buyer_email") else "") + ".",
            *lines,
            f"REQUIREMENTS: {reqs}" if reqs else "",
            f"TERMS: {clean(att.get('terms'))}" if att.get("terms") else "",
            legend,
        ])
    if kind == "po":
        lines, total = [], 0.0
        for i, line in enumerate(att.get("lines") or [], start=1):
            qty, price = _num(line.get("qty")), _num(line.get("unit_price"))
            total += qty * price
            lines.append(
                f"LINE {i}: P/N {clean(line.get('part_number'))}"
                + (f" REV {clean(line.get('rev'))}" if line.get("rev") else "")
                + f", {clean(line.get('description'))}, QTY {qty:,.0f} AT ${price:,.2f} EACH, "
                f"DUE {clean(line.get('due'))}.")
        notes = " ".join(f"{i}. {clean(n)}" for i, n in enumerate(att.get("notes") or [], start=1))
        return _join([
            f"PURCHASE ORDER {clean(att.get('po_number'))} FROM {clean(att.get('company'))}",
            f"DATE: {clean(att.get('date'))}. BUYER: {clean(att.get('buyer'))}."
            + (f" QUOTE REF: {clean(att.get('quote_ref'))}." if att.get("quote_ref") else "")
            + (f" TERMS: {clean(att.get('terms'))}." if att.get("terms") else "")
            + (f" SHIP VIA: {clean(att.get('ship_via'))}." if att.get("ship_via") else ""),
            *lines,
            f"ORDER TOTAL: ${total:,.2f}",
            f"NOTES: {notes}" if notes else "",
            legend,
        ])
    if kind == "document":
        parts = [f"{DOC_TYPE_LABELS.get(att.get('doc_type'), 'Document').upper()}: {clean(att.get('title'))}"]
        if att.get("subtitle"):
            parts.append(clean(att["subtitle"]))
        for sec in att.get("sections") or []:
            if isinstance(sec, dict):
                parts.append(f"{clean(sec.get('heading'))}: " + " ".join(clean(x) for x in sec.get("lines") or []))
        parts.append(legend)
        return _join(parts)
    return ""


def _num(v: Any) -> float:
    try:
        return float(str(v).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return 0.0


def jev_text(att: Dict[str, Any]) -> str:
    """The full text Jev would read from one attachment (before the size budget)."""
    kind = att.get("kind")
    if kind in SPEC_KINDS:
        return spec_text(att)
    if kind == "file":
        text = (att.get("text") or "").strip()
        if text:
            return text
        if att.get("text_error"):
            return f"(file; its text could not be read: {att['text_error']})"
        return "(no text was found in this file)"
    if kind == "upload":
        text = (att.get("text") or "").strip()
        if att.get("media") in ("png", "jpg"):
            if text:
                return text
            w, h = att.get("width"), att.get("height")
            dims = f", {w} x {h} px" if w and h else ""
            return f"(image file{dims}; no text could be read from it)"
        if text:
            return text
        if att.get("text_error"):
            return f"(PDF file; its text could not be read: {att['text_error']})"
        return "(PDF file with no text layer, for example a scan; there is no text to read)"
    return "(attached to the email; only the file name is available, not its contents)"


def _squeeze(text: str) -> str:
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def tidy_pdf_text(text: str) -> str:
    """Drop the 1 and 2 character fragments that drawing PDFs are full of (zone letters, stray
    digits, arrowheads), which only dilute what Jev reads."""
    lines = [line.strip() for line in (text or "").splitlines()]
    keep = [line for line in lines if len(line) > 2 and not all(len(tok) <= 1 for tok in line.split())]
    return _squeeze("\n".join(keep))


def jev_entries(attachments: Any) -> List[Dict[str, str]]:
    """The attachment part of Jev's state: file name plus the text on it, within the budget."""
    entries = []
    budget = JEV_TEXT_TOTAL
    for raw in attachments or []:
        att = normalize(raw)
        text = _squeeze(jev_text(att))
        limit = min(JEV_TEXT_PER_FILE, max(200, budget))
        if len(text) > limit:
            text = text[:limit].rstrip() + " [truncated]"
        budget -= len(text)
        entries.append({"file": att["name"], "content": text})
    return entries


# --------------------------------------------------------------------------- #
# Page layouts for forms, POs, and documents (drawings live in drawings.py)
# --------------------------------------------------------------------------- #
class Flow:
    """Top-to-bottom layout helper that starts a new page when one fills up."""

    def __init__(self, footer: str = "", legend: Optional[str] = None, company: str = "",
                 eccn: str = "", poc: str = ""):
        self.pages: List[Page] = []
        self.footer = footer
        self.legend = legend
        self.company = company
        self.eccn = eccn
        self.poc = poc
        self.x0, self.x1 = 42.0, LETTER[0] - 42.0
        self.bottom = LETTER[1] - 62.0
        self.new_page()

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    def new_page(self) -> Page:
        page = Page(*LETTER)
        self.pages.append(page)
        self.page = page
        self.y = 40.0
        if self.legend == "cui":
            page.text(LETTER[0] / 2, 24, "CUI", 11, True, "middle")
            page.text(LETTER[0] / 2, LETTER[1] - 14, "CUI", 11, True, "middle")
            self.y = 44.0
        return page

    def need(self, height: float) -> None:
        if self.y + height > self.bottom:
            self.new_page()

    def finish(self) -> List[Page]:
        total = len(self.pages)
        for i, page in enumerate(self.pages, start=1):
            y = LETTER[1] - 30
            page.line(self.x0, y - 10, self.x1, y - 10, 0.4, RULE)
            if self.footer:
                page.text_fit(self.x0, y, self.footer, self.width - 70, 6.5, color=GREY)
            page.text(self.x1, y, f"PAGE {i} OF {total}", 6.5, anchor="end", color=GREY)
        return self.pages


def _legend_box(flow: Flow, title: str) -> None:
    kind = flow.legend
    if kind not in docgen.EXPORT_LEGENDS:
        return
    text = legend_text(kind, flow.company, flow.eccn, flow.poc)
    lines = wrap(text, flow.width - 16, 7.2, True)
    height = 22 + len(lines) * 9
    flow.need(height + 6)
    page, x, y = flow.page, flow.x0, flow.y
    page.rect(x, y, flow.width, height, 1.4, BLACK, (1.0, 0.97, 0.9) if kind != "cui" else (0.96, 0.96, 0.96))
    page.text(x + 8, y + 13, title, 8.5, True)
    yy = y + 24
    for line in lines:
        page.text(x + 8, yy, line, 7.2, True)
        yy += 9
    flow.y = y + height + 10


def _proprietary_footer(flow: Flow) -> None:
    if flow.legend == "proprietary":
        flow.footer = legend_text("proprietary", flow.company)


def _header(flow: Flow, company: str, address: str, title: str, fields: List[Tuple[str, str]]) -> None:
    page, y = flow.page, flow.y
    page.text_fit(flow.x0, y + 16, company.upper(), 280, 15, True)
    if address:
        page.text_fit(flow.x0, y + 29, address, 280, 7.5, color=GREY)
    page.text(flow.x1, y + 16, title, 14, True, "end")
    cell_w = 84.0
    bx = flow.x1 - cell_w * len(fields)
    by = y + 24
    for i, (label, value) in enumerate(fields):
        cx = bx + i * cell_w
        page.rect(cx, by, cell_w, 26, 0.7)
        page.text(cx + 4, by + 8, label, 5.8, True, color=GREY)
        page.text_fit(cx + 4, by + 20, value, cell_w - 8, 8.5, True)
    flow.y = by + 36
    page.line(flow.x0, flow.y, flow.x1, flow.y, 1.2)
    flow.y += 12


def _address_blocks(flow: Flow, blocks: List[Tuple[str, List[str]]]) -> None:
    page, y = flow.page, flow.y
    col = flow.width / len(blocks)
    tallest = 0
    for i, (label, lines) in enumerate(blocks):
        x = flow.x0 + i * col
        page.text(x, y + 7, label, 6.2, True, color=GREY)
        yy = y + 19
        for j, line in enumerate(lines):
            page.text_fit(x, yy, line, col - 14, 8.2, j == 0)
            yy += 10.5
        tallest = max(tallest, yy - y)
    flow.y = y + tallest + 6


def _info_row(flow: Flow, cells: List[Tuple[str, str]]) -> None:
    cells = [c for c in cells if c[1]]
    if not cells:
        return
    page, y = flow.page, flow.y
    w = flow.width / len(cells)
    for i, (label, value) in enumerate(cells):
        x = flow.x0 + i * w
        page.rect(x, y, w, 26, 0.6)
        page.text(x + 4, y + 8, label, 5.8, True, color=GREY)
        page.text_fit(x + 4, y + 20, value, w - 8, 8, False)
    flow.y = y + 34


def _table(flow: Flow, columns: List[Tuple[str, float, str]], rows: List[List[str]], size: float = 7.8) -> None:
    """columns: (label, width fraction, align). Cells wrap; rows split across pages."""
    widths = [flow.width * frac for _, frac, _ in columns]

    def head() -> None:
        page, y = flow.page, flow.y
        page.rect(flow.x0, y, flow.width, 16, 0.7, BLACK, (0.16, 0.17, 0.2))
        x = flow.x0
        for (label, _, align), w in zip(columns, widths):
            tx = x + (w - 4 if align == "end" else w / 2 if align == "middle" else 4)
            page.text(tx, y + 11, label, 6.2, True, align, (1, 1, 1))
            x += w
        flow.y = y + 16

    flow.need(40)
    head()
    for row in rows:
        cells = [wrap(str(v), w - 8, size) or [""] for v, w in zip(row, widths)]
        height = max(len(c) for c in cells) * (size + 2.6) + 8
        if flow.y + height > flow.bottom:
            flow.new_page()
            head()
        page, y = flow.page, flow.y
        x = flow.x0
        for (label, _, align), w, lines in zip(columns, widths, cells):
            page.rect(x, y, w, height, 0.5)
            yy = y + 4 + size
            for line in lines:
                tx = x + (w - 4 if align == "end" else w / 2 if align == "middle" else 4)
                page.text(tx, yy, line, size, anchor=align)
                yy += size + 2.6
            x += w
        flow.y = y + height
    flow.y += 12


def _numbered(flow: Flow, title: str, items: List[str], size: float = 7.8) -> None:
    items = [clean(i) for i in items or [] if clean(i)]
    if not items:
        return
    flow.need(30)
    flow.page.text(flow.x0, flow.y + 8, title, 7, True)
    flow.y += 20
    for i, item in enumerate(items, start=1):
        lines = wrap(item, flow.width - 20, size)
        flow.need(len(lines) * (size + 3) + 2)
        flow.page.text(flow.x0, flow.y, f"{i}.", size, True)
        for line in lines:
            flow.page.text(flow.x0 + 16, flow.y, line, size)
            flow.y += size + 3
        flow.y += 1.5
    flow.y += 8


def layout_rfq_form(att: Dict[str, Any]) -> List[Page]:
    company = clean(att.get("company"))
    flow = Flow(footer=f"{company.upper()} REQUEST FOR QUOTATION. PLEASE QUOTE EACH LINE AND QUANTITY.",
                legend=att.get("legend"), company=company, eccn=att.get("eccn", ""), poc=att.get("buyer", ""))
    _proprietary_footer(flow)
    _header(flow, company, clean(att.get("company_address")), "REQUEST FOR QUOTATION",
            [("RFQ NO.", clean(att.get("rfq_number"))), ("DATE", clean(att.get("date"))),
             ("RESPOND BY", clean(att.get("respond_by")))])
    _legend_box(flow, "EXPORT-CONTROLLED TECHNICAL DATA" if att.get("legend") != "cui" else "CONTROLLED UNCLASSIFIED INFORMATION")
    buyer = [clean(att.get("buyer"))]
    if att.get("buyer_email"):
        buyer.append(clean(att["buyer_email"]))
    if att.get("buyer_phone"):
        buyer.append(clean(att["buyer_phone"]))
    _address_blocks(flow, [("TO SUPPLIER", SHOP_ADDRESS), ("FROM / BUYER", [company] + buyer)])
    rows = []
    for i, line in enumerate(att.get("lines") or [], start=1):
        matfin = " / ".join(x for x in (clean(line.get("material")), clean(line.get("finish"))) if x)
        desc = clean(line.get("description")) + (f". {clean(line.get('notes'))}" if line.get("notes") else "")
        rows.append([str(i), clean(line.get("part_number")), clean(line.get("rev") or "-"), desc, matfin,
                     _qty_list(line.get("quantities")), "", ""])
    _table(flow, [("ITEM", .06, "middle"), ("PART NUMBER", .15, "start"), ("REV", .06, "middle"),
                  ("DESCRIPTION", .22, "start"), ("MATERIAL / FINISH", .19, "start"),
                  ("QUANTITIES", .12, "middle"), ("UNIT PRICE", .10, "middle"), ("LEAD TIME", .10, "middle")], rows)
    _numbered(flow, "QUOTE REQUIREMENTS", att.get("requirements") or [])
    if att.get("terms"):
        flow.need(20)
        flow.page.text(flow.x0, flow.y, f"TERMS: {clean(att['terms'])}", 7.8, True)
        flow.y += 16
    flow.need(60)
    page, y = flow.page, flow.y + 8
    page.rect(flow.x0, y, flow.width, 46, 0.6, BLACK, (0.975, 0.975, 0.97))
    page.text(flow.x0 + 6, y + 11, "SUPPLIER RESPONSE", 6.5, True, color=GREY)
    for i, label in enumerate(("QUOTED BY", "QUOTE NO.", "DATE", "VALID FOR (DAYS)")):
        x = flow.x0 + 6 + i * (flow.width - 12) / 4
        page.line(x, y + 34, x + (flow.width - 12) / 4 - 14, y + 34, 0.5)
        page.text(x, y + 42, label, 5.8, color=GREY)
    flow.y = y + 56
    return flow.finish()


def layout_po(att: Dict[str, Any]) -> List[Page]:
    company = clean(att.get("company"))
    flow = Flow(footer=f"THIS ORDER IS SUBJECT TO {company.upper()} STANDARD TERMS AND CONDITIONS OF PURCHASE. "
                       f"ACKNOWLEDGE WITHIN 2 BUSINESS DAYS.",
                legend=att.get("legend"), company=company, eccn=att.get("eccn", ""), poc=att.get("buyer", ""))
    _header(flow, company, clean(att.get("company_address")), "PURCHASE ORDER",
            [("PO NUMBER", clean(att.get("po_number"))), ("DATE", clean(att.get("date"))),
             ("REVISION", clean(att.get("po_rev") or "0"))])
    _legend_box(flow, "EXPORT-CONTROLLED ITEMS" if att.get("legend") != "cui" else "CONTROLLED UNCLASSIFIED INFORMATION")
    ship_to = att.get("ship_to")
    ship_lines = [company] + ([clean(ship_to)] if ship_to else
                              [clean(att.get("company_address"))] if att.get("company_address") else [])
    _address_blocks(flow, [("VENDOR", SHOP_ADDRESS[:3]), ("SHIP TO", ship_lines)])
    _info_row(flow, [("BUYER", clean(att.get("buyer"))), ("BUYER EMAIL", clean(att.get("buyer_email"))),
                     ("TERMS", clean(att.get("terms"))), ("SHIP VIA", clean(att.get("ship_via"))),
                     ("QUOTE REF", clean(att.get("quote_ref")))])
    rows, total = [], 0.0
    for i, line in enumerate(att.get("lines") or [], start=1):
        qty, price = _num(line.get("qty")), _num(line.get("unit_price"))
        total += qty * price
        rows.append([str(i), clean(line.get("part_number")), clean(line.get("rev") or "-"),
                     clean(line.get("description")), f"{qty:,.0f}", f"${price:,.2f}", f"${qty * price:,.2f}",
                     clean(line.get("due"))])
    _table(flow, [("ITEM", .06, "middle"), ("PART NUMBER", .15, "start"), ("REV", .06, "middle"),
                  ("DESCRIPTION", .27, "start"), ("QTY", .09, "end"), ("UNIT PRICE", .12, "end"),
                  ("EXTENDED", .13, "end"), ("DUE", .12, "middle")], rows)
    flow.need(40)
    page, y = flow.page, flow.y - 6
    for label, value, bold in (("SUBTOTAL", f"${total:,.2f}", False), ("TAX", "EXEMPT, RESALE", False),
                               ("ORDER TOTAL", f"${total:,.2f}", True)):
        page.text(flow.x1 - 110, y + 8, label, 7.5, bold, "end")
        page.text(flow.x1 - 4, y + 8, value, 8.2 if bold else 7.8, bold, "end")
        y += 12
    flow.y = y + 10
    _numbered(flow, "PURCHASE ORDER NOTES AND QUALITY CLAUSES", att.get("notes") or [])
    if att.get("legend") == "proprietary":
        flow.need(30)
        flow.y = flow.page.paragraph(flow.x0, flow.y, legend_text("proprietary", company), flow.width, 6.8,
                                     color=GREY) + 6
    flow.need(40)
    page, y = flow.page, flow.y + 14
    page.line(flow.x0, y + 10, flow.x0 + 200, y + 10, 0.6)
    page.text(flow.x0, y + 20, f"AUTHORIZED BY: {clean(att.get('buyer')).upper()}, PURCHASING", 6.5, color=GREY)
    page.line(flow.x1 - 170, y + 10, flow.x1, y + 10, 0.6)
    page.text(flow.x1 - 170, y + 20, "VENDOR ACKNOWLEDGMENT AND DATE", 6.5, color=GREY)
    flow.y = y + 30
    return flow.finish()


DOC_ACCENT = {
    "brochure": (0.11, 0.36, 0.67), "newsletter": (0.13, 0.15, 0.2), "resume": (0.2, 0.22, 0.26),
    "cert": (0.1, 0.35, 0.25), "invoice": (0.2, 0.22, 0.26), "packing_slip": (0.2, 0.22, 0.26),
    "letter": (0.2, 0.22, 0.26), "spec": (0.35, 0.2, 0.1), "report": (0.45, 0.12, 0.12),
}


def layout_document(att: Dict[str, Any]) -> List[Page]:
    doc_type = att.get("doc_type") or "letter"
    company = clean(att.get("company"))
    accent = DOC_ACCENT.get(doc_type, (0.2, 0.22, 0.26))
    flow = Flow(footer=company.upper() if company else "", legend=att.get("legend"), company=company)
    _proprietary_footer(flow)
    page = flow.page
    title, subtitle = clean(att.get("title")), clean(att.get("subtitle"))
    if doc_type in ("brochure", "newsletter"):
        page.rect(0, 0, LETTER[0], 96, 0, None, accent)
        page.text_fit(flow.x0, 50, title, flow.width, 20, True, color=(1, 1, 1))
        if subtitle:
            page.text_fit(flow.x0, 70, subtitle, flow.width, 10.5, color=(0.86, 0.9, 0.97))
        if company:
            page.text_fit(flow.x0, 86, company.upper(), flow.width, 7, True, color=(0.8, 0.85, 0.95))
        flow.y = 122
    elif doc_type == "resume":
        page.text_fit(LETTER[0] / 2, 64, title, flow.width, 22, True, "middle")
        if subtitle:
            page.text_fit(LETTER[0] / 2, 82, subtitle, flow.width, 10, color=GREY, anchor="middle")
        page.line(flow.x0, 94, flow.x1, 94, 1.0, accent)
        flow.y = 116
    elif doc_type == "cert":
        page.rect(24, 24, LETTER[0] - 48, LETTER[1] - 48, 2.2, accent)
        page.rect(30, 30, LETTER[0] - 60, LETTER[1] - 60, 0.6, accent)
        page.text_fit(LETTER[0] / 2, 80, title.upper(), flow.width - 20, 17, True, "middle", accent)
        if subtitle:
            page.text_fit(LETTER[0] / 2, 98, subtitle, flow.width - 20, 9.5, color=GREY, anchor="middle")
        if company:
            page.text_fit(LETTER[0] / 2, 112, company.upper(), flow.width - 20, 7.5, True, "middle", GREY)
        flow.y = 136
        flow.x0, flow.x1 = 56.0, LETTER[0] - 56.0
    else:
        if company:
            page.text_fit(flow.x0, 54, company.upper(), 300, 13, True, color=accent)
        label = DOC_TYPE_LABELS.get(doc_type, "Document").upper()
        page.text(flow.x1, 54, label, 13, True, "end", accent)
        page.line(flow.x0, 64, flow.x1, 64, 1.2, accent)
        flow.y = 88
        page.text_fit(flow.x0, flow.y, title, flow.width, 13, True)
        flow.y += 15
        if subtitle:
            page.text_fit(flow.x0, flow.y, subtitle, flow.width, 9, color=GREY)
            flow.y += 14
        flow.y += 6
    _legend_box(flow, "EXPORT-CONTROLLED INFORMATION" if att.get("legend") != "cui" else "CONTROLLED UNCLASSIFIED INFORMATION")
    for sec in att.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        flow.need(34)
        page = flow.page
        page.text(flow.x0, flow.y + 9, clean(sec.get("heading")).upper(), 8, True, color=accent)
        page.line(flow.x0, flow.y + 13, flow.x1, flow.y + 13, 0.4, RULE)
        flow.y += 26
        for line in sec.get("lines") or []:
            text = clean(line)
            bullet = text.startswith("- ")
            if bullet:
                text = text[2:]
            lines = wrap(text, flow.width - (14 if bullet else 0), 8.6)
            for j, piece in enumerate(lines):
                flow.need(12)
                if bullet and j == 0:
                    flow.page.circle(flow.x0 + 3, flow.y - 2.8, 1.3, 0, None, BLACK)
                flow.page.text(flow.x0 + (14 if bullet else 0), flow.y, piece, 8.6)
                flow.y += 11.2
            flow.y += 2.5
        flow.y += 8
    return flow.finish()


def pages_for(att: Dict[str, Any]) -> List[Page]:
    kind = att.get("kind")
    if kind == "drawing":
        return drawings.drawing_pages(att)
    if kind == "rfq_form":
        return layout_rfq_form(att)
    if kind == "po":
        return layout_po(att)
    if kind == "document":
        return layout_document(att)
    raise ValueError(f"no pages for kind {kind}")


# --------------------------------------------------------------------------- #
# Files and thumbnails from specs (memoized: specs never change while running)
# --------------------------------------------------------------------------- #
_memo_lock = threading.Lock()
_memo: Dict[Tuple[str, str], Any] = {}


def _spec_key(att: Dict[str, Any]) -> str:
    return json.dumps(att, sort_keys=True, ensure_ascii=False)


def _memoized(att: Dict[str, Any], what: str, build) -> Any:
    key = (what, _spec_key(att))
    with _memo_lock:
        if key in _memo:
            return _memo[key]
    value = build()
    with _memo_lock:
        _memo[key] = value
    return value


def spec_file(att: Dict[str, Any]) -> Tuple[bytes, str]:
    """(file bytes, content type) for a sample attachment."""
    kind = att.get("kind")
    if kind == "model":
        return _memoized(att, "file", lambda: (drawings.step_file(att).encode("utf-8"), "model/step"))
    title = " ".join(x for x in (clean(att.get("part_number") or att.get("rfq_number") or att.get("po_number")),
                                 clean(att.get("title") or KIND_LABELS.get(kind, ""))) if x)
    return _memoized(att, "file", lambda: (docgen.to_pdf(pages_for(att), title=title,
                                                         author=clean(att.get("company") or att.get("author"))),
                                           "application/pdf"))


def spec_page_count(att: Dict[str, Any]) -> Optional[int]:
    if att.get("kind") == "model":
        return None
    return _memoized(att, "pages", lambda: len(pages_for(att)))


def spec_thumb(att: Dict[str, Any]) -> str:
    """SVG of the first page (or the shaded 3D view for a model)."""
    def build() -> str:
        if att.get("kind") == "model":
            return docgen.to_svg(drawings.model_thumb_page(att), background=(0.96, 0.97, 0.98))
        return docgen.to_svg(pages_for(att)[0])
    return _memoized(att, "thumb", build)


def spec_mesh(att: Dict[str, Any]) -> Dict[str, Any]:
    return _memoized(att, "mesh", lambda: drawings.mesh_for(att))


def warm(atts: List[Dict[str, Any]], pause: float = 0.005) -> None:
    """Build every sample file and thumbnail once in the background, so tiles open instantly."""
    for att in atts:
        try:
            if att.get("kind") in SPEC_KINDS:
                spec_file(att)
                spec_thumb(att)
        except Exception:  # noqa: BLE001 - a bad spec shows up when opened, not here
            pass
        time.sleep(pause)


# --------------------------------------------------------------------------- #
# Uploads (kept in memory)
# --------------------------------------------------------------------------- #
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOADS_PER_EMAIL = 5
MAX_UPLOAD_TOTAL = int(os.environ.get("RFQ_UPLOAD_MEMORY_MB", "100")) * 1024 * 1024
UNATTACHED_TTL = 3600.0
UPLOAD_TYPES = {"application/pdf": "pdf", "image/png": "png", "image/jpeg": "jpg"}
MEDIA_TYPES = {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg", "step": "model/step"}


def sniff(data: bytes) -> Optional[str]:
    """Trust the bytes, not the file name or the browser's type."""
    if data[:1024].lstrip(b"\x00\t\r\n ").startswith(b"%PDF-") or b"%PDF-" in data[:1024]:
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    return None


def image_size(data: bytes, media: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        if media == "png" and len(data) >= 24 and data[12:16] == b"IHDR":
            w, h = struct.unpack(">II", data[16:24])
            return int(w), int(h)
        if media == "jpg":
            i = 2
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                length = struct.unpack(">H", data[i + 2:i + 4])[0]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return int(w), int(h)
                i += 2 + length
    except (struct.error, IndexError):
        pass
    return None, None


def safe_filename(name: str, media: str) -> str:
    name = str(name or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f\"<>|:*?]", "", name)
    name = re.sub(r"\s+", " ", clean(name)).strip(" .") or "upload"
    ext = {"pdf": (".pdf",), "png": (".png",), "jpg": (".jpg", ".jpeg")}[media]
    if not name.lower().endswith(ext):
        name = name.rsplit(".", 1)[0] if "." in name[-6:] else name
        name = f"{name}{ext[0]}"
    if len(name) > 100:
        stem, dot, ext_part = name.rpartition(".")
        name = stem[:95 - len(ext_part)] + dot + ext_part
    return name


def pypdf_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("pypdf") is not None
    except Exception:  # noqa: BLE001
        return False


PDF_TEXT_TIMEOUT = float(os.environ.get("RFQ_PDF_TEXT_TIMEOUT", "25"))
# At most this many PDF readers at once: each one is a separate process, and small cloud
# instances (512 MB on Render's free plan) cannot hold many.
_pdf_slots = threading.BoundedSemaphore(max(1, int(os.environ.get("RFQ_PDF_WORKERS", "2"))))
PDF_TEXT_MAX_CHARS = 60000
PDF_TEXT_MAX_PAGES = 40


def extract_pdf_text(data: bytes) -> Dict[str, Any]:
    """Read a PDF's text with pypdf in a child process. Never raises."""
    if not pypdf_available():
        return {"text": "", "pages": None, "error": "pypdf is not installed on the server"}
    if not _pdf_slots.acquire(timeout=PDF_TEXT_TIMEOUT * 2):
        return {"text": "", "pages": None, "error": "the server is busy reading other PDFs"}
    try:
        proc = subprocess.run([sys.executable, os.path.join(HERE, "attachments.py"), "--extract-pdf"],
                              input=data, capture_output=True, timeout=PDF_TEXT_TIMEOUT, cwd=HERE)
    except subprocess.TimeoutExpired:
        return {"text": "", "pages": None, "error": "reading the PDF took too long"}
    except OSError as exc:
        return {"text": "", "pages": None, "error": f"could not start the PDF reader ({exc})"}
    finally:
        _pdf_slots.release()
    try:
        result = json.loads(proc.stdout.decode("utf-8") or "{}")
    except ValueError:
        result = {}
    if proc.returncode != 0 and not result:
        return {"text": "", "pages": None, "error": "the PDF reader stopped unexpectedly"}
    return {"text": str(result.get("text") or "")[:PDF_TEXT_MAX_CHARS], "pages": result.get("pages"),
            "error": result.get("error")}


def _extract_main() -> int:
    """Child process: PDF bytes on stdin, JSON {text, pages, error} on stdout."""
    try:
        import resource  # Unix only: cap memory so a PDF bomb cannot take the host down
        limit = 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:  # noqa: BLE001
        pass
    data = sys.stdin.buffer.read()
    out: Dict[str, Any] = {"text": "", "pages": None, "error": None}
    try:
        import logging
        logging.disable(logging.CRITICAL)
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001
                out["error"] = "the PDF is password protected"
        pages = reader.pages
        out["pages"] = len(pages)
        chunks, size = [], 0
        for i, page in enumerate(pages):
            if i >= PDF_TEXT_MAX_PAGES or size >= PDF_TEXT_MAX_CHARS:
                break
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - one bad page should not lose the rest
                text = ""
            chunks.append(text)
            size += len(text)
        out["text"] = "\n".join(chunks)[:PDF_TEXT_MAX_CHARS]
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"the PDF could not be read ({type(exc).__name__})"
    sys.stdout.write(json.dumps(out))
    return 0


class UploadStore:
    """Uploaded files, in memory only. Oldest files are dropped past the memory budget."""

    def __init__(self, max_total: int = MAX_UPLOAD_TOTAL):
        self.lock = threading.Lock()
        self.items: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.max_total = max_total

    def total(self) -> int:
        return sum(item["size"] for item in self.items.values())

    def add(self, filename: str, data: bytes) -> Dict[str, Any]:
        if not data:
            raise ValueError("The file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("Files can be up to 10 MB each.")
        media = sniff(data)
        if media is None:
            raise ValueError("Only PDF, PNG, and JPG files can be attached.")
        item: Dict[str, Any] = {
            "id": secrets.token_hex(8), "name": safe_filename(filename, media), "media": media,
            "size": len(data), "data": data, "created": time.time(), "attached": False,
            "pages": None, "width": None, "height": None, "text": "", "text_error": None,
        }
        if media == "pdf":
            result = extract_pdf_text(data)
            item.update(text=tidy_pdf_text(result["text"]), pages=result["pages"], text_error=result["error"])
        else:
            item["width"], item["height"] = image_size(data, media)
        # A scan or a photo has no text layer: read it with Tesseract when the server has it.
        if not item["text"] and ocr_ready():
            result = _ocr().file_text(data, media, item["name"], effort=UPLOAD_OCR_EFFORT)
            if result.get("method") == "ocr" and (result.get("text") or "").strip():
                item.update(text=tidy_pdf_text(result["text"])[:PDF_TEXT_MAX_CHARS], text_error=None,
                            text_method="ocr", text_conf=result.get("confidence"),
                            pages=result.get("pages") or item["pages"])
            elif result.get("error") and media != "pdf":
                item["text_error"] = result["error"]
        with self.lock:
            self.items[item["id"]] = item
            self._evict()
        return item

    def _evict(self) -> None:
        now = time.time()
        for uid in [u for u, it in self.items.items() if not it["attached"] and now - it["created"] > UNATTACHED_TTL]:
            del self.items[uid]
        while self.items and self.total() > self.max_total:
            self.items.popitem(last=False)

    def get(self, uid: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.items.get(uid)

    def claim(self, uid: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            item = self.items.get(uid)
            if item:
                item["attached"] = True
            return item


def upload_attachment(item: Dict[str, Any]) -> Dict[str, Any]:
    """The attachment entry stored on the email. The text stays even if the bytes are dropped later."""
    return {"name": item["name"], "kind": "upload", "media": item["media"], "upload_id": item["id"],
            "size": item["size"], "pages": item["pages"], "width": item["width"], "height": item["height"],
            "text": item["text"][:PDF_TEXT_MAX_CHARS], "text_error": item["text_error"],
            "text_method": item.get("text_method"), "text_conf": item.get("text_conf")}


def upload_public(item: Dict[str, Any]) -> Dict[str, Any]:
    """What the browser gets right after an upload, for the preview tile."""
    att = upload_attachment(item)
    return dict(describe(att, f"/api/upload/{item['id']}", available=True), id=item["id"])


# --------------------------------------------------------------------------- #
# Real files on disk, OCR, and thumbnails
# --------------------------------------------------------------------------- #
UPLOAD_OCR_EFFORT = os.environ.get("RFQ_UPLOAD_OCR_EFFORT", "fast")
_ocr_mod: Any = None


def _ocr() -> Any:
    """ocr.py, imported lazily so the demo still runs where it cannot be imported."""
    global _ocr_mod
    if _ocr_mod is None:
        try:
            import ocr as mod
            _ocr_mod = mod
        except Exception:  # noqa: BLE001
            _ocr_mod = False
    return _ocr_mod or None


def ocr_ready() -> bool:
    mod = _ocr()
    if not mod:
        return False
    try:
        return bool(mod.available().get("ocr"))
    except Exception:  # noqa: BLE001
        return False


class LayeredCache:
    """OCR results: the committed cache (read only) on top of a writable one in the cache folder."""

    def __init__(self, seed: Any, live: Any):
        self.seed, self.live = seed, live

    def get(self, sha256: str) -> Optional[Dict[str, Any]]:
        return (self.live.get(sha256) if self.live else None) or (self.seed.get(sha256) if self.seed else None)

    def put(self, sha256: str, result: Dict[str, Any]) -> None:
        if self.live:
            self.live.put(sha256, result)

    def save(self) -> None:
        if self.live:
            try:
                self.live.save()
            except Exception:  # noqa: BLE001 - a read-only disk only loses the speedup
                pass


_MARKINGS = [("itar", re.compile(r"INTERNATIONAL\s+TRAFFIC\s+IN\s+ARMS|\bITAR\b|ARMS\s+EXPORT\s+CONTROL", re.I)),
             ("ear", re.compile(r"EXPORT\s+ADMINISTRATION\s+REGULATIONS|\bECCN\b", re.I)),
             ("cui", re.compile(r"\bCUI\b|CONTROLLED\s+UNCLASSIFIED", re.I))]


def detect_legend(text: str) -> Optional[str]:
    """The export-control marking printed on a file, read from its text (the tile's lock badge)."""
    for key, rx in _MARKINGS:
        if rx.search(text or ""):
            return key
    return None


def classify(text: str, media: Optional[str]) -> str:
    """What a file is, from what is printed on it (not its name)."""
    t = (text or "").upper()
    if media == "step":
        return "3D model"
    if "REQUEST FOR QUOTATION" in t or re.search(r"\bRFQ\s*NO", t):
        return "RFQ form"
    if "PURCHASE ORDER" in t and re.search(r"PO\s*NUMBER|ORDER\s+TOTAL", t):
        return "Purchase order"
    if re.search(r"DWG\.?\s*NO|UNLESS\s+OTHERWISE\s+SPECIFIED|THIRD\s+ANGLE|REVISIONS", t):
        return "Drawing"
    return {"jpg": "Photo", "png": "Image", "pdf": "Document"}.get(media or "", "File")


def resolve_data_path(data_dir: Any, rel: str) -> Optional[str]:
    """A beta file path, kept inside data/. Returns None for anything that escapes it."""
    base = os.path.realpath(str(data_dir))
    target = os.path.realpath(os.path.join(base, rel or ""))
    return target if target.startswith(base + os.sep) and os.path.isfile(target) else None


def prepare_file(att: Dict[str, Any], data_dir: Any, cache: Any = None) -> Dict[str, Any]:
    """Read one real attachment: its bytes, its text (text layer, OCR, or STEP header), and what it
    is. Updates the attachment dict in place and returns it. Never raises."""
    path = resolve_data_path(data_dir, att.get("path", ""))
    if not path:
        att.update(prepared=True, available=False, text="", text_error="the file is missing")
        return att
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        att.update(prepared=True, available=False, text="", text_error=f"could not read the file ({exc})")
        return att
    media = att.get("media") or sniff(data) or ("step" if data[:12] == b"ISO-10303-21" else None)
    att.update(media=media, size=len(data), sha256=hashlib.sha256(data).hexdigest(), available=True)
    if media in ("png", "jpg"):
        att["width"], att["height"] = image_size(data, media)
    if media == "step":
        info = stepfile.parse(data)
        att["step"] = {k: info.get(k) for k in ("part_number", "title", "units", "schema", "system")}
        att["step"]["bbox"] = stepfile.bbox_phrase(info.get("mesh"), info.get("units"))
        att["_mesh"] = info.get("mesh")
    result: Optional[Dict[str, Any]] = None
    mod = _ocr()
    if mod and hasattr(mod, "file_text"):
        try:
            result = mod.file_text(data, media or "", att.get("name", ""), cache=cache)
        except Exception as exc:  # noqa: BLE001 - fall back to the readers below
            result = {"method": "none", "text": "", "confidence": None, "pages": None, "error": repr(exc)}
    if not result or (not (result.get("text") or "").strip() and media in ("pdf", "step")):
        # Without ocr.py: the PDF text layer through pypdf, and the STEP header. Scans stay unread.
        if media == "pdf":
            r = extract_pdf_text(data)
            result = {"method": "text-layer" if r["text"].strip() else "none", "text": r["text"],
                      "pages": r["pages"], "confidence": None,
                      "error": r["error"] or (None if r["text"].strip() else "a scan, and OCR is not available")}
        elif media == "step":
            st = att.get("step") or {}
            result = {"method": "step-header", "confidence": None, "pages": None, "error": None,
                      "text": f"3D CAD MODEL (STEP {st.get('schema') or ''})\nPART: {st.get('part_number') or ''}"
                              f" {st.get('title') or ''}\nUNITS: {st.get('units') or ''}. BOUNDING BOX: "
                              f"{st.get('bbox') or 'unknown'}"}
        else:
            result = result or {"method": "none", "text": "", "confidence": None, "pages": None,
                                "error": "OCR is not available on this server"}
    text = result.get("text") or ""
    if result.get("method") in ("ocr", "text-layer"):
        text = tidy_pdf_text(text)
    att.update(prepared=True, text=text[:PDF_TEXT_MAX_CHARS], text_method=result.get("method"),
               text_conf=result.get("confidence"), text_error=result.get("error") if not text else None,
               pages=result.get("pages"), _file_text=result)
    att["doc_type"] = classify(text, media)
    att["legend"] = detect_legend(text)
    return att


def text_from_label(att: Dict[str, Any]) -> str:
    method = att.get("text_method")
    if method == "ocr":
        conf = att.get("text_conf")
        return f"OCR {round(conf)}%" if conf is not None else "OCR"
    return {"text-layer": "text layer", "step-header": "STEP header"}.get(method or "", "no text")


_thumb_lock = threading.Lock()
_thumbs: Dict[str, Tuple[bytes, str]] = {}


def file_thumb(att: Dict[str, Any], data_dir: Any) -> Optional[Tuple[bytes, str]]:
    """A small preview of a real file: page 1 of a PDF (pdftoppm), a shrunk image (Pillow), or the
    shaded 3D view of a STEP model. None when the tools for it are missing."""
    key = att.get("sha256") or ""
    with _thumb_lock:
        if key in _thumbs:
            return _thumbs[key]
    path = resolve_data_path(data_dir, att.get("path", ""))
    if not path:
        return None
    media, out = att.get("media"), None
    try:
        if media == "step":
            out = (stepfile.iso_svg(att.get("_mesh")).encode("utf-8"), "image/svg+xml")
        elif media == "pdf":
            import shutil
            if shutil.which("pdftoppm"):
                proc = subprocess.run(["pdftoppm", "-f", "1", "-l", "1", "-scale-to", "560", "-jpeg",
                                       "-jpegopt", "quality=78", path], capture_output=True, timeout=30)
                if proc.returncode == 0 and proc.stdout.startswith(b"\xff\xd8"):
                    out = (proc.stdout, "image/jpeg")
        elif media in ("png", "jpg"):
            from PIL import Image
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((560, 560))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=82)
                out = (buf.getvalue(), "image/jpeg")
    except Exception:  # noqa: BLE001 - no preview is fine, the tile shows an icon
        out = None
    if out:
        with _thumb_lock:
            _thumbs[key] = out
    return out


# --------------------------------------------------------------------------- #
# What the browser gets about an attachment (never the full spec or text)
# --------------------------------------------------------------------------- #
def describe(att: Dict[str, Any], base: Optional[str], available: bool = True) -> Dict[str, Any]:
    """base: URL prefix for this attachment, e.g. /api/att/E01/0 or /api/upload/<id> (None = no file)."""
    att = normalize(att)
    kind = att["kind"]
    media = media_of(att)
    info: Dict[str, Any] = {"name": att["name"], "kind": kind, "media": media,
                            "label": KIND_LABELS.get(kind, "File"), "legend": att.get("legend")}
    if kind == "document":
        info["label"] = DOC_TYPE_LABELS.get(att.get("doc_type"), "Document")
    for key in ("title", "part_number", "rev", "company"):
        if att.get(key):
            info[key] = clean(att[key])
    if kind in SPEC_KINDS and base:
        spec_key = _spec_key(att)
        v = hashlib.sha1(spec_key.encode("utf-8")).hexdigest()[:10]  # cache buster: changes with the spec
        info["url"] = f"{base}/file?v={v}"
        info["thumb"] = f"{base}/thumb.svg?v={v}"
        info["text_url"] = f"{base}/text"
        cached = _memo.get(("file", spec_key))
        info["size"] = len(cached[0]) if cached else None
        pages = _memo.get(("pages", spec_key))
        info["pages"] = pages
        if kind == "model":
            info["mesh"] = f"{base}/mesh.json?v={v}"
            info["model"] = {"shape": att.get("shape"), "size": att.get("size"), "units": att.get("units"),
                             "schema": att.get("schema") or "AP214",
                             "system": clean(att.get("originating_system") or ""),
                             "bbox": drawings.bbox_phrase(att)}
    elif kind == "file" and base:
        v = (att.get("sha256") or "")[:10]
        media = att.get("media")
        info.update(url=f"{base}/file?v={v}", text_url=f"{base}/text", size=att.get("size"),
                    pages=att.get("pages"), width=att.get("width"), height=att.get("height"),
                    available=bool(available and att.get("available", True)), text_from=text_from_label(att),
                    scanned=att.get("text_method") == "ocr", text_error=att.get("text_error"))
        info["thumb"] = f"{base}/thumb.{'svg' if media == 'step' else 'jpg'}?v={v}"
        doc = att.get("doc_type") or classify("", media)
        how = {"jpg": "photo", "png": "screenshot"}.get(media or "", "scan") if info["scanned"] else ""
        info["label"] = f"{doc} ({how}, {info['text_from']})" if how and doc not in ("Photo", "Image") else \
            (f"{doc} ({info['text_from']})" if info["scanned"] else doc)
        if media == "step":
            step = att.get("step") or {}
            info["kind"] = "model"  # the viewer shows it in 3D, from the mesh in the file itself
            info["mesh"] = f"{base}/mesh.json?v={v}"
            info["model"] = {"schema": step.get("schema") or "STEP", "system": step.get("system") or "",
                             "units": step.get("units"), "bbox": step.get("bbox")}
            for key in ("part_number", "title"):
                if step.get(key):
                    info[key] = step[key]
    elif kind == "upload":
        text = (att.get("text") or "").strip()
        info.update(size=att.get("size"), pages=att.get("pages"), width=att.get("width"),
                    height=att.get("height"), text_error=att.get("text_error"), text_chars=len(text),
                    preview=text[:700], available=bool(available and base),
                    scanned=att.get("text_method") == "ocr", text_from=text_from_label(att)
                    if att.get("text_method") else None)
        if base:
            info["url"] = f"{base}/file"
            info["text_url"] = f"{base}/text"
    return info


if __name__ == "__main__":
    if "--extract-pdf" in sys.argv:
        sys.exit(_extract_main())
    print(__doc__)
