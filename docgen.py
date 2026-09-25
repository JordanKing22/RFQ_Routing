"""
A tiny page canvas that writes both PDF and SVG. Standard library only.

The sample attachments (drawings, RFQ forms, purchase orders) are laid out once as a list of
drawing operations on a Page, then written out two ways from the same operations:
  * to_pdf(pages)  -> a real PDF file (Helvetica, WinAnsi text), what the viewer opens
  * to_svg(page)   -> the matching thumbnail, what the email view shows as a tile

Coordinates are PDF points (1/72 inch) with the origin at the TOP-LEFT corner, like SVG.
Text y is the baseline.
"""

from __future__ import annotations

import math
import zlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Color = Tuple[float, float, float]
BLACK: Color = (0.0, 0.0, 0.0)
WHITE: Color = (1.0, 1.0, 1.0)
LETTER = (612.0, 792.0)
LETTER_LANDSCAPE = (792.0, 612.0)

# Glyph widths (1/1000 em) for characters 32..126 of the standard Helvetica fonts, from the Adobe AFMs.
# Arial and Liberation Sans share these metrics, so the SVG thumbnails line up with the PDFs.
_HELV = [278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
         556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
         1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
         667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
         333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
         556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584]
_HELV_BOLD = [278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278,
              556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611,
              975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778,
              667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333, 278, 333, 584, 556,
              333, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611,
              611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584]
_EXTRA = {"±": 584, "°": 400, "Ø": 778, "µ": 556, "×": 584, "·": 278}

SVG_FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"


def text_width(text: str, size: float, bold: bool = False) -> float:
    table = _HELV_BOLD if bold else _HELV
    total = 0
    for ch in text:
        code = ord(ch)
        if 32 <= code <= 126:
            total += table[code - 32]
        else:
            total += _EXTRA.get(ch, 556)
    return total * size / 1000.0


def clean(text: Any) -> str:
    """Printable text for both backends: collapse control characters, keep the WinAnsi-safe set."""
    out = []
    for ch in str(text if text is not None else ""):
        if ch in "\r\n\t":
            out.append(" ")
        elif ch in "—–":
            out.append("-")
        elif ch in "‘’":
            out.append("'")
        elif ch in "“”":
            out.append('"')
        elif ord(ch) >= 32:
            out.append(ch)
    return "".join(out)


def wrap(text: str, width: float, size: float, bold: bool = False) -> List[str]:
    """Greedy word wrap by measured width. Very long words are split."""
    words = clean(text).split()
    lines: List[str] = []
    line = ""
    for word in words:
        while text_width(word, size, bold) > width and len(word) > 1:
            # split an overlong word
            cut = len(word)
            while cut > 1 and text_width(word[:cut], size, bold) > width:
                cut -= 1
            if line:
                lines.append(line)
                line = ""
            lines.append(word[:cut])
            word = word[cut:]
        candidate = f"{line} {word}" if line else word
        if text_width(candidate, size, bold) <= width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def fit_text(text: str, width: float, size: float, bold: bool = False, min_size: float = 4.5) -> Tuple[str, float]:
    """Shrink the font until the text fits, then truncate with '..' if it still does not."""
    text = clean(text)
    while size > min_size and text_width(text, size, bold) > width:
        size -= 0.25
    if text_width(text, size, bold) > width:
        while text and text_width(text + "..", size, bold) > width:
            text = text[:-1]
        text = text.rstrip() + ".."
    return text, size


class Page:
    """An ordered list of drawing operations for one page."""

    def __init__(self, width: float, height: float):
        self.width = float(width)
        self.height = float(height)
        self.ops: List[Tuple[str, Dict[str, Any]]] = []

    # -- shapes ----------------------------------------------------------- #
    def line(self, x1: float, y1: float, x2: float, y2: float, lw: float = 0.6,
             color: Color = BLACK, dash: Optional[Sequence[float]] = None) -> None:
        self.ops.append(("line", dict(x1=x1, y1=y1, x2=x2, y2=y2, lw=lw, color=color, dash=dash)))

    def rect(self, x: float, y: float, w: float, h: float, lw: float = 0.6,
             stroke: Optional[Color] = BLACK, fill: Optional[Color] = None,
             dash: Optional[Sequence[float]] = None) -> None:
        self.ops.append(("rect", dict(x=x, y=y, w=w, h=h, lw=lw, stroke=stroke, fill=fill, dash=dash)))

    def circle(self, cx: float, cy: float, r: float, lw: float = 0.6,
               stroke: Optional[Color] = BLACK, fill: Optional[Color] = None,
               dash: Optional[Sequence[float]] = None) -> None:
        self.ops.append(("circle", dict(cx=cx, cy=cy, r=r, lw=lw, stroke=stroke, fill=fill, dash=dash)))

    def polygon(self, points: Iterable[Tuple[float, float]], lw: float = 0.6,
                stroke: Optional[Color] = BLACK, fill: Optional[Color] = None,
                closed: bool = True, dash: Optional[Sequence[float]] = None) -> None:
        pts = [(float(x), float(y)) for x, y in points]
        if len(pts) >= 2:
            self.ops.append(("poly", dict(points=pts, lw=lw, stroke=stroke, fill=fill, closed=closed, dash=dash)))

    def path(self, commands: Sequence[Tuple], lw: float = 0.6, stroke: Optional[Color] = BLACK,
             fill: Optional[Color] = None, dash: Optional[Sequence[float]] = None) -> None:
        """commands: ("M", x, y), ("L", x, y), ("C", x1, y1, x2, y2, x, y), ("Z",)."""
        self.ops.append(("path", dict(cmds=list(commands), lw=lw, stroke=stroke, fill=fill, dash=dash)))

    def arc(self, cx: float, cy: float, r: float, start_deg: float, end_deg: float, lw: float = 0.6,
            color: Color = BLACK, dash: Optional[Sequence[float]] = None) -> None:
        """Circular arc, angles in degrees counterclockwise from +x (screen y points down)."""
        self.path(arc_commands(cx, cy, r, start_deg, end_deg, move=True), lw=lw, stroke=color, dash=dash)

    # -- text ------------------------------------------------------------- #
    def text(self, x: float, y: float, text: Any, size: float = 8.0, bold: bool = False,
             anchor: str = "start", color: Color = BLACK) -> None:
        text = clean(text)
        if text:
            self.ops.append(("text", dict(x=x, y=y, text=text, size=size, bold=bold, anchor=anchor, color=color)))

    def text_fit(self, x: float, y: float, text: Any, width: float, size: float = 8.0, bold: bool = False,
                 anchor: str = "start", color: Color = BLACK) -> None:
        shown, fitted = fit_text(str(text), width, size, bold)
        self.text(x, y, shown, fitted, bold, anchor, color)

    def paragraph(self, x: float, y: float, text: Any, width: float, size: float = 8.0,
                  leading: Optional[float] = None, bold: bool = False, color: Color = BLACK,
                  max_lines: Optional[int] = None) -> float:
        """Wrapped text starting at baseline y. Returns the baseline after the last line."""
        leading = leading or size * 1.25
        lines = wrap(str(text), width, size, bold)
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1]
            while last and text_width(last + "..", size, bold) > width:
                last = last[:-1]
            lines[-1] = last.rstrip() + ".."
        for line in lines:
            self.text(x, y, line, size, bold, color=color)
            y += leading
        return y


def arc_commands(cx: float, cy: float, r: float, start_deg: float, end_deg: float,
                 move: bool = True) -> List[Tuple]:
    """Bezier approximation of a circular arc (screen coordinates, y down)."""
    cmds: List[Tuple] = []
    sweep = end_deg - start_deg
    segments = max(1, int(math.ceil(abs(sweep) / 90.0)))
    step = math.radians(sweep / segments)
    a0 = math.radians(start_deg)
    k = 4.0 / 3.0 * math.tan(step / 4.0)
    x0, y0 = cx + r * math.cos(a0), cy - r * math.sin(a0)
    if move:
        cmds.append(("M", x0, y0))
    for i in range(segments):
        a1 = a0 + step
        x1, y1 = cx + r * math.cos(a1), cy - r * math.sin(a1)
        c1 = (x0 - k * r * math.sin(a0), y0 - k * r * math.cos(a0))
        c2 = (x1 + k * r * math.sin(a1), y1 + k * r * math.cos(a1))
        cmds.append(("C", c1[0], c1[1], c2[0], c2[1], x1, y1))
        a0, x0, y0 = a1, x1, y1
    return cmds


# --------------------------------------------------------------------------- #
# PDF backend
# --------------------------------------------------------------------------- #
def _num(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def _pdf_string(text: str) -> str:
    data = text.encode("cp1252", errors="replace")
    out = []
    for b in data:
        if b in (0x28, 0x29, 0x5C):  # ( ) \
            out.append("\\" + chr(b))
        elif 32 <= b <= 126:
            out.append(chr(b))
        else:
            out.append(f"\\{b:03o}")
    return "(" + "".join(out) + ")"


def _pdf_color(c: Color, op: str) -> str:
    return f"{_num(c[0])} {_num(c[1])} {_num(c[2])} {op}"


def _pdf_style(lw: float, dash: Optional[Sequence[float]]) -> str:
    style = f"{_num(lw)} w "
    style += ("[" + " ".join(_num(d) for d in dash) + "] 0 d " if dash else "[] 0 d ")
    return style


def _pdf_paint(stroke: Optional[Color], fill: Optional[Color]) -> str:
    if stroke and fill:
        return "B"
    if fill:
        return "f"
    return "S"


def _content_stream(page: Page) -> bytes:
    h = page.height
    out: List[str] = ["1 J 1 j"]  # round caps and joins

    def Y(y: float) -> float:  # noqa: N802 - flip to PDF space
        return h - y

    for kind, o in page.ops:
        if kind == "text":
            size = o["size"]
            width = text_width(o["text"], size, o["bold"])
            x = o["x"] - (width if o["anchor"] == "end" else width / 2 if o["anchor"] == "middle" else 0)
            out.append(f"BT /{'F2' if o['bold'] else 'F1'} {_num(size)} Tf {_pdf_color(o['color'], 'rg')} "
                       f"1 0 0 1 {_num(x)} {_num(Y(o['y']))} Tm {_pdf_string(o['text'])} Tj ET")
            continue
        parts = ["q", _pdf_style(o.get("lw", 0.6), o.get("dash"))]
        stroke = o.get("stroke", o.get("color"))
        fill = o.get("fill")
        if stroke:
            parts.append(_pdf_color(stroke, "RG"))
        if fill:
            parts.append(_pdf_color(fill, "rg"))
        if kind == "line":
            parts.append(f"{_num(o['x1'])} {_num(Y(o['y1']))} m {_num(o['x2'])} {_num(Y(o['y2']))} l S")
        elif kind == "rect":
            parts.append(f"{_num(o['x'])} {_num(Y(o['y'] + o['h']))} {_num(o['w'])} {_num(o['h'])} re "
                         + _pdf_paint(stroke, fill))
        elif kind == "circle":
            cmds = arc_commands(o["cx"], o["cy"], o["r"], 0, 360)
            parts.append(_path_ops(cmds, Y) + " h " + _pdf_paint(stroke, fill))
        elif kind == "poly":
            pts = o["points"]
            seg = [f"{_num(pts[0][0])} {_num(Y(pts[0][1]))} m"]
            seg += [f"{_num(x)} {_num(Y(y))} l" for x, y in pts[1:]]
            closed = o.get("closed", True)
            paint = _pdf_paint(stroke, fill) if closed else "S"
            parts.append(" ".join(seg) + (" h " if closed else " ") + paint)
        elif kind == "path":
            parts.append(_path_ops(o["cmds"], Y) + " " + _pdf_paint(stroke, fill))
        parts.append("Q")
        out.append(" ".join(parts))
    return "\n".join(out).encode("latin-1", errors="replace")


def _path_ops(cmds: Sequence[Tuple], Y) -> str:  # noqa: N803
    seg = []
    for c in cmds:
        if c[0] == "M":
            seg.append(f"{_num(c[1])} {_num(Y(c[2]))} m")
        elif c[0] == "L":
            seg.append(f"{_num(c[1])} {_num(Y(c[2]))} l")
        elif c[0] == "C":
            seg.append(f"{_num(c[1])} {_num(Y(c[2]))} {_num(c[3])} {_num(Y(c[4]))} "
                       f"{_num(c[5])} {_num(Y(c[6]))} c")
        elif c[0] == "Z":
            seg.append("h")
    return " ".join(seg)


def _pdf_text_string(text: str) -> str:
    """For the Info dictionary: plain ASCII, escaped."""
    return _pdf_string(clean(text).encode("ascii", errors="replace").decode("ascii"))


def to_pdf(pages: List[Page], title: str = "", author: str = "", subject: str = "") -> bytes:
    """Write pages as a compact PDF 1.4 file with the two standard Helvetica fonts."""
    objects: List[bytes] = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    catalog = add(b"")  # placeholder, filled below
    pages_id = add(b"")
    font1 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    font2 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    kids = []
    for page in pages:
        data = zlib.compress(_content_stream(page), 9)
        stream = add(b"<< /Length " + str(len(data)).encode() + b" /Filter /FlateDecode >>\nstream\n"
                     + data + b"\nendstream")
        page_id = add(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {_num(page.width)} {_num(page.height)}] "
            f"/Resources << /Font << /F1 {font1} 0 R /F2 {font2} 0 R >> >> /Contents {stream} 0 R >>".encode())
        kids.append(page_id)
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode()
    objects[pages_id - 1] = (f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] "
                             f"/Count {len(kids)} >>").encode()
    info = add(("<< /Producer (RFQ Router demo) /Creator (RFQ Router demo, docgen.py)"
                + (f" /Title {_pdf_text_string(title)}" if title else "")
                + (f" /Author {_pdf_text_string(author)}" if author else "")
                + (f" /Subject {_pdf_text_string(subject)}" if subject else "")
                + " /CreationDate (D:20260925080000Z) >>").encode("latin-1", errors="replace"))

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R /Info {info} 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


# --------------------------------------------------------------------------- #
# SVG backend
# --------------------------------------------------------------------------- #
def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _svg_color(c: Optional[Color]) -> str:
    if not c:
        return "none"
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(v * 255)))) for v in c)


def _svg_style(o: Dict[str, Any], stroke: Optional[Color], fill: Optional[Color]) -> str:
    attrs = [f'fill="{_svg_color(fill)}"']
    if stroke:
        attrs.append(f'stroke="{_svg_color(stroke)}" stroke-width="{_num(o.get("lw", 0.6))}"')
        if o.get("dash"):
            attrs.append(f'stroke-dasharray="{" ".join(_num(d) for d in o["dash"])}"')
    return " ".join(attrs)


def to_svg(page: Page, width_px: Optional[float] = None, background: Color = WHITE) -> str:
    w, h = page.width, page.height
    size = f' width="{_num(width_px)}" height="{_num(width_px * h / w)}"' if width_px else ""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_num(w)} {_num(h)}"{size} '
           f'stroke-linecap="round" stroke-linejoin="round">',
           f'<rect width="{_num(w)}" height="{_num(h)}" fill="{_svg_color(background)}"/>']
    for kind, o in page.ops:
        if kind == "text":
            anchor = {"start": "start", "middle": "middle", "end": "end"}[o["anchor"]]
            weight = ' font-weight="bold"' if o["bold"] else ""
            out.append(f'<text x="{_num(o["x"])}" y="{_num(o["y"])}" font-family="{SVG_FONT}" '
                       f'font-size="{_num(o["size"])}"{weight} text-anchor="{anchor}" '
                       f'fill="{_svg_color(o["color"])}">{_esc(o["text"])}</text>')
        elif kind == "line":
            out.append(f'<line x1="{_num(o["x1"])}" y1="{_num(o["y1"])}" x2="{_num(o["x2"])}" '
                       f'y2="{_num(o["y2"])}" {_svg_style(o, o["color"], None)}/>')
        elif kind == "rect":
            out.append(f'<rect x="{_num(o["x"])}" y="{_num(o["y"])}" width="{_num(o["w"])}" '
                       f'height="{_num(o["h"])}" {_svg_style(o, o["stroke"], o["fill"])}/>')
        elif kind == "circle":
            out.append(f'<circle cx="{_num(o["cx"])}" cy="{_num(o["cy"])}" r="{_num(o["r"])}" '
                       f'{_svg_style(o, o["stroke"], o["fill"])}/>')
        elif kind == "poly":
            tag = "polygon" if o.get("closed", True) else "polyline"
            pts = " ".join(f"{_num(x)},{_num(y)}" for x, y in o["points"])
            fill = o["fill"] if o.get("closed", True) else None
            out.append(f'<{tag} points="{pts}" {_svg_style(o, o["stroke"], fill)}/>')
        elif kind == "path":
            d = []
            for c in o["cmds"]:
                if c[0] == "Z":
                    d.append("Z")
                else:
                    d.append(c[0] + " ".join(_num(v) for v in c[1:]))
            out.append(f'<path d="{" ".join(d)}" {_svg_style(o, o["stroke"], o["fill"])}/>')
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Export-control and proprietary legends, shared by drawings, RFQ forms, and POs
# --------------------------------------------------------------------------- #
def legend_text(kind: Optional[str], company: str = "", eccn: str = "", poc: str = "") -> str:
    company = clean(company).upper() or "THE OWNER"
    if kind == "itar":
        return ("WARNING: THIS DOCUMENT CONTAINS TECHNICAL DATA WHOSE EXPORT IS RESTRICTED BY THE ARMS EXPORT "
                "CONTROL ACT (TITLE 22, U.S.C., SEC 2751, ET SEQ.) AND THE INTERNATIONAL TRAFFIC IN ARMS "
                "REGULATIONS (ITAR, 22 CFR 120-130). EXPORT, OR RELEASE TO A FOREIGN PERSON, WITHOUT AN "
                "APPROVED LICENSE OR EXEMPTION IS A VIOLATION OF U.S. LAW. U.S. PERSONS ONLY.")
    if kind == "ear":
        code = clean(eccn).upper() or "9E610"
        return (f"WARNING: THIS DOCUMENT CONTAINS TECHNOLOGY SUBJECT TO THE EXPORT ADMINISTRATION REGULATIONS "
                f"(EAR, 15 CFR 730-774), ECCN {code}. EXPORT, REEXPORT, OR RELEASE TO A FOREIGN PERSON WITHOUT "
                f"AUTHORIZATION FROM THE U.S. DEPARTMENT OF COMMERCE IS PROHIBITED.")
    if kind == "cui":
        contact = clean(poc).upper()
        return (f"CUI. CONTROLLED BY: {company}. CUI CATEGORY: CTI (CONTROLLED TECHNICAL INFORMATION). "
                f"DISTRIBUTION STATEMENT D: DISTRIBUTION AUTHORIZED TO THE DEPARTMENT OF DEFENSE AND U.S. DOD "
                f"CONTRACTORS ONLY. HANDLE PER DFARS 252.204-7012 AND NIST SP 800-171."
                + (f" POC: {contact}." if contact else ""))
    if kind == "proprietary":
        return (f"PROPRIETARY AND CONFIDENTIAL. THE INFORMATION IN THIS DOCUMENT IS THE PROPERTY OF {company}. "
                f"DO NOT REPRODUCE IT OR USE IT FOR ANY PURPOSE OTHER THAN QUOTING OR MAKING PARTS FOR "
                f"{company} WITHOUT WRITTEN PERMISSION.")
    return ""


EXPORT_LEGENDS = ("itar", "ear", "cui")
