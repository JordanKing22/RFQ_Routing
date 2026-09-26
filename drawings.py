"""
Engineering drawings and 3D models for the sample attachments. Standard library only.

    drawing_pages(spec)    -> [Page]  an 11 x 8.5 in drawing sheet: orthographic views chosen for the
                                       shape (with hidden lines, center marks, dimensions, sections, and
                                       leaders from the spec's callouts to the features they name), a
                                       shaded isometric view, notes, a revision block, the title block,
                                       and any legend (ITAR, EAR, CUI, proprietary), inside a zoned border
    mesh_for(spec)         -> {"vertices", "faces", ...}  a faceted solid for the 3D viewer
    model_thumb_page(spec) -> Page    the shaded isometric view used on a STEP file's tile
    step_file(spec)        -> str     an ISO 10303-21 (STEP) file with a faceted B-rep of the mesh

Shapes (spec["shape"]) and sizes (spec["size"], in spec["units"]):
    prismatic  plate block bracket housing cover manifold heatsink fixture enclosure  [L, W, H]
    complex    structural_fitting impeller implant contoured                         [L, W, H]
    round      shaft pin disc nozzle threaded_fitting                                 [L, D]
               bushing spacer ring                                                    [L, OD, ID]
    other      weldment sheet_metal casting assembly                                  [L, W, H]

How it fits together: part_model(spec) turns the spec (shape, size, units, callouts) into one part
description: a profile, holes, pockets, lathe steps, and so on. The orthographic views, the mesh, the
isometric views, and the STEP file are all made from that description, so they agree with each other.
Callouts such as "4X Ø.201 THRU", "M6 X 1.0 - 6H THRU", or ".375 DEEP POCKET" are parsed into features,
and each callout's leader points at the feature it produced.

Meshes: every face is counterclockwise seen from outside, z up. A mesh is a set of closed components
(solids that may touch or overlap); vertices are shared inside a component. check_mesh() verifies that.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from docgen import BLACK, LETTER_LANDSCAPE, Page, arc_commands, clean, fit_text, legend_text, text_width, wrap

PRISMATIC = {"plate", "block", "bracket", "housing", "cover", "manifold", "heatsink", "fixture", "enclosure"}
COMPLEX = {"structural_fitting", "impeller", "implant", "contoured"}
ROUND2 = {"shaft", "pin", "disc", "nozzle", "threaded_fitting"}
ROUND3 = {"bushing", "spacer", "ring"}
ROUND = ROUND2 | ROUND3
OTHER = {"weldment", "sheet_metal", "casting", "assembly"}

GREY = (0.42, 0.44, 0.48)
WHITE = (1.0, 1.0, 1.0)
THIN = 0.4
MED = 0.75
THICK = 1.25
CENTER_DASH = (10, 2.2, 1.8, 2.2)
HIDDEN_DASH = (3.2, 1.8)
PHANTOM_DASH = (11, 2, 2, 2, 2, 2)
CUT_DASH = (16, 2.6, 3.2, 2.6)
DIM = 6.6        # dimension text size
LABEL = 6.3      # callout text size
Pt = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Size helpers (also used for the text Jev reads)
# --------------------------------------------------------------------------- #
def _dims(spec: Dict[str, Any]) -> List[float]:
    try:
        vals = [float(v) for v in spec.get("size") or []]
    except (TypeError, ValueError):
        vals = []
    shape = spec.get("shape")
    # NaN or infinity (Python's json accepts them) would break every layout: use the default instead
    dflt = [1.0, 1.0, 0.5] if shape in ROUND3 else [2.0, 0.5] if shape in ROUND2 else [4.0, 3.0, 1.0]
    vals = [v if math.isfinite(v) else dflt[min(i, len(dflt) - 1)] for i, v in enumerate(vals)]
    if shape in ROUND3:
        vals = (vals + [1.0, 1.0, 0.5])[:3]
        if vals[2] >= vals[1]:
            vals[2] = vals[1] * 0.5
    elif shape in ROUND2:
        vals = (vals + [2.0, 0.5])[:2]
    else:
        vals = (vals + [4.0, 3.0, 1.0])[:3]
    return [max(v, 1e-3) for v in vals]


def fmt(value: float, units: Optional[str]) -> str:
    if units == "mm":
        return f"{value:.1f}".rstrip("0").rstrip(".") if value < 100 else f"{value:.0f}"
    text = f"{value:.3f}"
    if text.startswith("0."):
        text = text[1:]
    return text


def default_tolerances(units: Optional[str]) -> str:
    if units == "mm":
        return "X.X ±0.2  X.XX ±0.05  ANGLES ±0.5°"
    return ".XX ±.01  .XXX ±.005  ANGLES ±0.5°"


def size_phrase(spec: Dict[str, Any]) -> str:
    d, units, shape = _dims(spec), spec.get("units") or "in", spec.get("shape")
    u = "MM" if units == "mm" else "IN"
    if shape in ROUND3:
        return f"Ø{fmt(d[1], units)} OD x Ø{fmt(d[2], units)} ID x {fmt(d[0], units)} LONG ({u})"
    if shape in ROUND2:
        word = "THICK" if shape == "disc" else "LONG"
        return f"Ø{fmt(d[1], units)} x {fmt(d[0], units)} {word} ({u})"
    return f"{fmt(d[0], units)} x {fmt(d[1], units)} x {fmt(d[2], units)} ({u})"


def bbox(spec: Dict[str, Any]) -> Tuple[float, float, float]:
    d, shape = _dims(spec), spec.get("shape")
    if shape in ROUND:
        return d[0], d[1], d[1]
    return d[0], d[1], d[2]


def bbox_phrase(spec: Dict[str, Any]) -> str:
    units = spec.get("units") or "in"
    x, y, z = bbox(spec)
    return f"{fmt(x, units)} x {fmt(y, units)} x {fmt(z, units)} {'MM' if units == 'mm' else 'IN'}"


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _text_list(value: Any) -> List[Any]:
    """A list of text items from a spec field. A bare string is one item, not a list of characters."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
    return []


# --------------------------------------------------------------------------- #
# Callouts: "4X Ø.201 THRU, C'BORE Ø.344 X .20 DP" -> count, diameters, thread, depth, ...
# All lengths come back in the spec's units.
# --------------------------------------------------------------------------- #
_NUM = r"(\d*\.\d+|\d+(?:\.\d+)?)"
_UN_RE = re.compile(r"(?<![\d./])(#?\d+-\d+/\d+|\d+/\d+|#?\d+)\s*-\s*(\d+)\s*(UNJ[CF]?|UNEF|UNR[CF]?|UN[CFS]?|NPTF?)\b"
                    r"(?:\s*-\s*(\d)\s*([ABab]))?", re.I)
_M_RE = re.compile(r"\bM\s?(\d+(?:\.\d+)?)\s*[xX]\s*(\d*\.?\d+)(?:\s*-\s*\d\s*([A-Za-z]))?")
_NPT_OD = {"1/16": .3125, "1/8": .405, "1/4": .540, "3/8": .675, "1/2": .840, "3/4": 1.050, "1": 1.315}
_TAGS = [
    (r"\bFINS?\b", "fin"), (r"\bKEY\s*(?:WAY|SEAT)", "keyway"), (r"GROOVE|GASKET|O-RING|\bCHANNEL\b", "groove"),
    (r"\bSLOTS?\b", "slot"), (r"POCKET|CAVITY|RECESS|\bNEST\b", "pocket"), (r"WINDOW|CUTOUT", "window"),
    (r"CHAMFER|LEAD-IN|BEVEL", "chamfer"), (r"\bBORES?\b|\bID\b", "bore"), (r"PILOT", "pilot"),
    (r"DOWEL|PIN HOLE|LOCATING|TOOLING", "dowel"), (r"\bPORTS?\b|\bORB\b|SAE\s*-\d", "port"), (r"CROSS[\s-]*DRILL|CROSS HOLE", "cross"),
    (r"LIGHTENING", "lightening"), (r"\bPADS?\b|\bBOSS", "pad"), (r"INSERT|HELI", "insert"), (r"\bWALLS?\b", "wall"),
    (r"FLATNESS|PERPENDICULAR|PARALLEL|RUNOUT|\bTIR\b|CONCENTRIC|COPLANAR|PROFILE|\bFACE\b", "gdt"),
    (r"COMPOUND|\bANGLES?\b|DRAFT|TWIST|TAPER", "angle"), (r"STIFFENER|\bRIBS?\b", "rib"), (r"\bFLANGE", "flange"),
    (r"BLADES?|SPLITTER", "blade"), (r"KNURL", "knurl"), (r"\bFLATS\b|\bHEX\b|A/F", "hex"),
    (r"JOURNAL|BEARING", "journal"), (r"\bHEAD\b", "head"), (r"COTTER|CROSS HOLE", "cotter"), (r"ORIFICE", "orifice"),
    (r"\bCONE\b|\bSEAT\b|ENTRY", "cone"), (r"RELIEF|UNDERCUT", "relief"), (r"\bTANG\b", "tang"),
    (r"SET\s*SCREW", "setscrew"), (r"SHOULDER", "shoulder"), (r"SHANK|PACKING|SEAL AREA|PISTON|\bLAND\b", "land"),
    (r"\bOD\b", "od"), (r"\bOAL\b|OVERALL", "oal"), (r"CORNERS?|FILLET|(?<![A-Z])R\s*\.?\d", "radius"),
    (r"TRUNNION", "trunnion"), (r"\bLE\b|\bTE\b|LEADING|TRAILING", "edge"), (r"FLUTES?", "flute"),
    (r"\bPEGS?\b", "peg"), (r"NOTCH", "notch"), (r"PATELLAR|TROCHLE", "trochlea"), (r"BULLET|\bNOSE\b", "nose"),
    (r"K-?WIRE", "kwire"), (r"LOCKING HOLE|SCREW HOLE|FOR \d", "screwhole"), (r"MARKER", "marker"),
    (r"\bBEND|S-CURVE|\bFORM", "bend"), (r"\bWELD", "weld"), (r"\bTUBE|TUBING", "tube"), (r"INSPECTION", "inspect"),
    (r"SOCKET|CRIMP", "socket"), (r"THREAD", "threadword"), (r"TONGUE|DIVIDER", "tongue"), (r"\bTHK\b|THICK", "thk"),
    (r"CONTACT|SPRING", "contact"), (r"\bEACH SIDE\b|BOTH SIDES", "eachside"), (r"\bFEET\b|\bFOOT\b|BOTTOM", "bottom"),
    (r"GLAND|BULKHEAD|CONNECTOR|SENSOR|CABLE|\bSMA\b", "sidewall"), (r"\bLUG\b", "lug"), (r"BOTH ENDS|EACH END", "bothends"),
    (r"PATTERN|\bBC\b|B\.C\.", "pattern"), (r"MOUNT", "mount"), (r"PLATE", "plate"), (r"BALANCE", "balance"),
]


def _nominal_inch(nom: str, tpi: int) -> float:
    nom = nom.lstrip("#")
    if "/" in nom:
        whole, frac = (nom.split("-", 1) if "-" in nom else ("0", nom))
        a, b = frac.split("/")
        return float(whole) + float(a) / max(float(b), 1.0)
    n = int(nom)
    if n <= 12 and tpi >= 24:
        return 0.060 + 0.013 * n  # numbered machine screw sizes
    return float(n)


def _conv(v: float, from_units: str, to_units: str) -> float:
    if from_units == to_units:
        return v
    return v * 25.4 if to_units == "mm" else v / 25.4


def parse_callout(text: Any, units: str) -> Dict[str, Any]:
    t = clean(text)
    T = t.upper()
    c: Dict[str, Any] = {"text": t, "T": T, "count": 1, "dias": [], "thread": None, "internal": None,
                         "depth": None, "thru": bool(re.search(r"\bTHRU\b", T)), "cb": None, "csk": None,
                         "rect": None, "width": None, "length": None, "radius": None, "bc": None, "angles": [],
                         "pattern": None, "num": None, "grid": None, "tags": set()}
    m = re.match(r"\s*(\d+)\s*X\b", T)
    if m:
        c["count"] = max(1, min(int(m.group(1)), 999))
    body = T[m.end():] if m else T
    scrub = body
    mm = _M_RE.search(t[m.end():] if m else t)
    if mm:
        c["thread"] = _conv(float(mm.group(1)), "mm", units)
        if mm.group(3):
            c["internal"] = mm.group(3).isupper()
        scrub = scrub.replace(mm.group(0).upper(), " ")
    else:
        mu = _UN_RE.search(body)
        if mu:
            nom, tpi, series = mu.group(1), int(mu.group(2)), mu.group(3).upper()
            dia = _NPT_OD.get(nom.lstrip("#"), 0.5) if series.startswith("NPT") else _nominal_inch(nom, tpi)
            c["thread"] = _conv(dia, "in", units)
            if mu.group(5):
                c["internal"] = mu.group(5).upper() == "B"
            scrub = scrub.replace(mu.group(0), " ")
        else:
            mp = re.search(r"(\d+/\d+|\d+)\s*(?:-\s*\d+\s*)?NPTF?\b", body)
            if mp:
                c["thread"] = _conv(_NPT_OD.get(mp.group(1), 0.54), "in", units)
                scrub = scrub.replace(mp.group(0), " ")
    mb = re.search(r"Ø\s*" + _NUM + r"\s*(?:B\.?\s*C\.?|BOLT)", body)
    dia_text = body
    if mb:
        c["bc"] = float(mb.group(1))
        dia_text = dia_text.replace(mb.group(0), " ")
    c["dias"] = [float(x) for x in re.findall(r"Ø\s*" + _NUM, dia_text)]
    mc = re.search(r"C'?\s*BORE\s*Ø?\s*" + _NUM + r"(?:\s*X\s*" + _NUM + r"\s*(?:DP|DEEP))?", body)
    depth_text = scrub
    if mc:
        c["cb"] = (float(mc.group(1)), float(mc.group(2)) if mc.group(2) else None)
        depth_text = depth_text.replace(mc.group(0), " ")
        if c["dias"] and abs(c["dias"][-1] - c["cb"][0]) < 1e-9 and len(c["dias"]) > 1:
            c["dias"] = c["dias"][:-1]
    i = max(body.find("CSK"), body.find("C'SINK"), body.find("COUNTERSINK"))
    if i >= 0:
        mk = re.search(r"Ø\s*" + _NUM, body[i:i + 22])
        c["csk"] = float(mk.group(1)) if mk else 0.0
        if mk and len(c["dias"]) > 1 and abs(c["dias"][-1] - c["csk"]) < 1e-9:
            c["dias"] = c["dias"][:-1]
    md = re.search(_NUM + r"\s*(?:DP|DEEP)\b", depth_text)
    if md:
        c["depth"] = float(md.group(1))
    mr = re.search(_NUM + r"\s*X\s*" + _NUM + r"(?!\s*°)(?:\s*X\s*" + _NUM + r"(?!\s*°))?", scrub.replace("Ø", " "))
    if mr:
        vals = [float(v) for v in mr.groups() if v]
        if "MM" in scrub[mr.end():mr.end() + 5] and units != "mm":
            vals = [_conv(v, "mm", units) for v in vals]
        elif re.match(r"\s*(?:IN|\")", scrub[mr.end():mr.end() + 4]) and units == "mm":
            vals = [_conv(v, "in", units) for v in vals]
        c["rect"] = tuple(vals)
        if "PATTERN" in T:
            c["pattern"] = (vals[0], vals[1])
    mg = re.search(_NUM + r"\s*(MM|IN\.?)?\s*GRID\b|\bGRID\s*(?:OF\s*)?" + _NUM, T)
    if mg:
        pitch = float(mg.group(1) or mg.group(3))
        if mg.group(2) == "MM" and units != "mm":
            pitch = _conv(pitch, "mm", units)
        elif mg.group(2) and mg.group(2).startswith("IN") and units == "mm":
            pitch = _conv(pitch, "in", units)
        c["grid"] = pitch if pitch > 0 else None
    mw = re.search(_NUM + r"\s*(?:WIDE|W)\b", scrub)
    if mw:
        c["width"] = float(mw.group(1))
    ml = re.search(_NUM + r"\s*(?:LONG|LG)\b", scrub)
    if ml:
        c["length"] = float(ml.group(1))
    mrad = re.search(r"(?<![A-Z])R\s*" + _NUM, scrub) or re.search(_NUM + r"\s*R\b", scrub)
    if mrad:
        c["radius"] = float(mrad.group(1))
    c["angles"] = [float(a) for a in re.findall(_NUM + r"\s*°", T)]
    nums = [float(n) for n in re.findall(r"(?<![\w.])" + _NUM + r"(?!\s*°)(?![\d])", scrub.replace("Ø", " Ø"))]
    c["num"] = nums[0] if nums else None
    tags = c["tags"]
    for pat, tag in _TAGS:
        if re.search(pat, T):
            tags.add(tag)
    if c["thread"]:
        tags.add("thread")
    if c["cb"]:
        tags.add("cbore")
    if c["csk"] is not None:
        tags.add("csk")
    if (c["dias"] or c["thread"]) and not tags & {"slot", "keyway", "groove", "fin", "blade", "knurl", "flute"}:
        tags.add("hole")
    return c


def _external(c: Dict[str, Any]) -> bool:
    """Is a thread callout an external (male) thread?"""
    if c["internal"] is not None:
        return not c["internal"]
    T = c["T"]
    if re.search(r"\bDP\b|DEEP|INSERT|TAP|THRU", T):
        return False
    return bool(re.search(r"\bLG\b|\bLONG\b|BOTH ENDS|EACH END|\bEND\b|THREAD\b", T)) or True


# --------------------------------------------------------------------------- #
# Meshes
# --------------------------------------------------------------------------- #
def _newell(pts: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1, z1 = pts[i]
        x2, y2, z2 = pts[(i + 1) % n]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    return nx, ny, nz


def _volume(verts: Sequence[Tuple[float, float, float]], faces: Sequence[Sequence[int]]) -> float:
    total = 0.0
    for f in faces:
        ax, ay, az = verts[f[0]]
        for k in range(1, len(f) - 1):
            bx, by, bz = verts[f[k]]
            cx, cy, cz = verts[f[k + 1]]
            total += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    return total / 6.0


class Mesh:
    """Closed components built one at a time: begin(), vid()/poly()/face(), end()."""

    def __init__(self) -> None:
        self.v: List[Tuple[float, float, float]] = []
        self.f: List[List[int]] = []
        self.parts: List[Tuple[int, int]] = []
        self._reg: Dict[Tuple[float, float, float], int] = {}
        self._start: Optional[int] = None
        self._xf: Optional[Callable[[float, float, float], Tuple[float, float, float]]] = None

    # -- component bookkeeping ---------------------------------------------- #
    def begin(self, xf: Optional[Callable[[float, float, float], Tuple[float, float, float]]] = None) -> None:
        self._reg = {}
        self._start = len(self.f)
        self._xf = xf

    def vid(self, x: float, y: float, z: float) -> int:
        if self._xf is not None:
            x, y, z = self._xf(x, y, z)
        key = (round(x, 7), round(y, 7), round(z, 7))
        i = self._reg.get(key)
        if i is None:
            i = len(self.v)
            self.v.append((float(x), float(y), float(z)))
            self._reg[key] = i
        return i

    def face(self, ids: Sequence[int]) -> None:
        out: List[int] = []
        for i in ids:
            if not out or out[-1] != i:
                out.append(i)
        while len(out) > 1 and out[0] == out[-1]:
            out.pop()
        if len(set(out)) >= 3:
            self.f.append(out)

    def poly(self, pts: Sequence[Tuple[float, float, float]], expect: Optional[Tuple[float, float, float]] = None) -> None:
        """A face from local points; with expect, the winding is chosen so the normal points that way."""
        pts = list(pts)
        if expect is not None:
            n = _newell(pts)
            if n[0] * expect[0] + n[1] * expect[1] + n[2] * expect[2] < 0:
                pts.reverse()
        self.face([self.vid(*p) for p in pts])

    def end(self) -> None:
        start = self._start if self._start is not None else len(self.f)
        faces = self.f[start:]
        if faces:
            if _volume(self.v, faces) < 0:
                for fc in faces:
                    fc.reverse()
            self.parts.append((start, len(self.f)))
        self._start, self._xf, self._reg = None, None, {}

    # -- primitives (each call is one closed component) ---------------------- #
    def add(self, pts: Sequence[Tuple[float, float, float]], faces: Sequence[Sequence[int]]) -> None:
        self.begin()
        ids = [self.vid(*p) for p in pts]
        for fc in faces:
            self.face([ids[i] for i in fc])
        self.end()

    def box(self, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, xf=None) -> None:
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        if z1 < z0:
            z0, z1 = z1, z0
        if min(x1 - x0, y1 - y0, z1 - z0) <= 1e-9:
            return
        self.prism([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], z0, z1, xf)

    def prism(self, outline: Sequence[Pt], z0: float, z1: float, xf=None) -> None:
        """Extrude a simple outline (x, y) between z0 and z1."""
        pts = list(outline)
        if _area2(pts) < 0:
            pts.reverse()
        n = len(pts)
        if n < 3 or z1 - z0 <= 1e-9:
            return
        self.begin(xf)
        self.poly([(x, y, z0) for x, y in pts], (0, 0, -1))
        self.poly([(x, y, z1) for x, y in pts], (0, 0, 1))
        for i in range(n):
            (ax, ay), (bx, by) = pts[i], pts[(i + 1) % n]
            self.poly([(ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1)], (by - ay, ax - bx, 0))
        self.end()

    def cylinder_x(self, x0: float, x1: float, r: float, cy: float = 0.0, cz: float = 0.0,
                   segments: int = 28, r_inner: float = 0.0) -> None:
        if r_inner > 0:
            prof = [(x0, r_inner), (x1, r_inner), (x1, r), (x0, r)]
        else:
            prof = [(x0, 0.0), (x1, 0.0), (x1, r), (x0, r)]
        self.revolve(prof, segments, lambda x, y, z: (x, cy + y, cz + z))

    def cylinder_z(self, z0: float, z1: float, r: float, cx: float = 0.0, cy: float = 0.0,
                   segments: int = 28) -> None:
        self.revolve([(z0, 0.0), (z1, 0.0), (z1, r), (z0, r)], segments, lambda a, b, c: (cx + b, cy + c, a))

    def prism_z(self, outline: Sequence[Pt], z0: float, z1: float) -> None:
        self.prism(outline, z0, z1)

    def revolve(self, profile: Sequence[Pt], segments: int = 28, xf=None) -> None:
        """Revolve a closed (x, r) profile about the local x axis (local y = r cos t, z = r sin t)."""
        prof = [(float(x), max(0.0, float(r))) for x, r in profile]
        clean_prof: List[Pt] = []
        for p in prof:
            if not clean_prof or abs(p[0] - clean_prof[-1][0]) > 1e-12 or abs(p[1] - clean_prof[-1][1]) > 1e-12:
                clean_prof.append(p)
        while len(clean_prof) > 2 and clean_prof[0] == clean_prof[-1]:
            clean_prof.pop()
        prof = clean_prof
        if len(prof) < 3:
            return
        if _area2(prof) < 0:
            prof.reverse()
        n = max(6, int(segments))
        cs = [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)) for k in range(n)]
        eps = 1e-12
        self.begin(xf)
        m = len(prof)
        for i in range(m):
            (xa, ra), (xb, rb) = prof[i], prof[(i + 1) % m]
            dx, dr = xb - xa, rb - ra
            nx, nr = dr, -dx  # outward normal in the (x, r) plane (profile is counterclockwise)
            if ra <= eps and rb <= eps:
                continue
            if ra <= eps or rb <= eps:
                rr, xr, xo = (rb, xb, xa) if ra <= eps else (ra, xa, xb)
                if abs(xa - xb) <= eps:
                    self.poly([(xr, rr * c, rr * s) for c, s in cs], (nx, 0, 0))
                else:
                    for k in range(n):
                        c0, s0 = cs[k]
                        c1, s1 = cs[(k + 1) % n]
                        cm, sm = math.cos(2 * math.pi * (k + .5) / n), math.sin(2 * math.pi * (k + .5) / n)
                        self.poly([(xo, 0, 0), (xr, rr * c0, rr * s0), (xr, rr * c1, rr * s1)], (nx, nr * cm, nr * sm))
                continue
            for k in range(n):
                c0, s0 = cs[k]
                c1, s1 = cs[(k + 1) % n]
                cm, sm = math.cos(2 * math.pi * (k + .5) / n), math.sin(2 * math.pi * (k + .5) / n)
                self.poly([(xa, ra * c0, ra * s0), (xa, ra * c1, ra * s1), (xb, rb * c1, rb * s1), (xb, rb * c0, rb * s0)],
                          (nx, nr * cm, nr * sm))
        self.end()

    def loft(self, sections: Sequence[Sequence[Tuple[float, float, float]]], xf=None) -> None:
        """Connect rings of equal length with quads and cap both ends (caps should be convex)."""
        secs = [list(s) for s in sections if len(s) >= 3]
        if len(secs) < 2:
            return
        n = len(secs[0])
        self.begin(xf)
        for a, b in zip(secs, secs[1:]):
            for k in range(n):
                self.face([self.vid(*a[k]), self.vid(*a[(k + 1) % n]), self.vid(*b[(k + 1) % n]), self.vid(*b[k])])
        self.face([self.vid(*p) for p in secs[-1]])
        self.face([self.vid(*p) for p in reversed(secs[0])])
        self.end()

    def holed_prism(self, outline: Sequence[Pt], hole: Tuple[float, float, float], z0: float, z1: float,
                    segments: int = 24, xf=None) -> None:
        """Extrude a convex outline with one round hole through it."""
        cx, cy, r = hole
        pts = list(outline)
        if _area2(pts) < 0:
            pts.reverse()
        angles = sorted({(2 * math.pi * k / segments) for k in range(segments)}
                        | {math.atan2(y - cy, x - cx) % (2 * math.pi) for x, y in pts})
        ring = []
        for a in angles:
            d = (math.cos(a), math.sin(a))
            hit = _ray_polygon(cx, cy, d, pts)
            if hit is None:
                continue
            ring.append((a, hit))
        if len(ring) < 3:
            return
        self.begin(xf)
        m = len(ring)
        for i in range(m):
            a0, p0 = ring[i]
            a1, p1 = ring[(i + 1) % m]
            q0 = (cx + r * math.cos(a0), cy + r * math.sin(a0))
            q1 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
            self.poly([(p0[0], p0[1], z1), (p1[0], p1[1], z1), (q1[0], q1[1], z1), (q0[0], q0[1], z1)], (0, 0, 1))
            self.poly([(p0[0], p0[1], z0), (p1[0], p1[1], z0), (q1[0], q1[1], z0), (q0[0], q0[1], z0)], (0, 0, -1))
            ex, ey = p1[0] - p0[0], p1[1] - p0[1]
            self.poly([(p0[0], p0[1], z0), (p1[0], p1[1], z0), (p1[0], p1[1], z1), (p0[0], p0[1], z1)], (ey, -ex, 0))
            am = (a0 + a1) / 2 if a1 > a0 else (a0 + a1 + 2 * math.pi) / 2
            self.poly([(q0[0], q0[1], z0), (q1[0], q1[1], z0), (q1[0], q1[1], z1), (q0[0], q0[1], z1)],
                      (-math.cos(am), -math.sin(am), 0))
        self.end()

    def rect_tube(self, length: float, w: float, h: float, t: float, xf=None) -> None:
        """Square or rectangular tube along local x from 0 to length, centered on the x axis."""
        t = min(t, w * 0.45, h * 0.45)
        o = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        i = [(-w / 2 + t, -h / 2 + t), (w / 2 - t, -h / 2 + t), (w / 2 - t, h / 2 - t), (-w / 2 + t, h / 2 - t)]
        self.begin(xf)
        for k in range(4):
            (ay, az), (by, bz) = o[k], o[(k + 1) % 4]
            (cy, cz), (dy, dz) = i[k], i[(k + 1) % 4]
            out = (0, bz - az, -(by - ay))
            self.poly([(0, ay, az), (length, ay, az), (length, by, bz), (0, by, bz)], out)
            self.poly([(0, cy, cz), (length, cy, cz), (length, dy, dz), (0, dy, dz)], (0, -out[1], -out[2]))
            self.poly([(0, ay, az), (0, by, bz), (0, dy, dz), (0, cy, cz)], (-1, 0, 0))
            self.poly([(length, ay, az), (length, by, bz), (length, dy, dz), (length, cy, cz)], (1, 0, 0))
        self.end()

    def as_dict(self, units: str) -> Dict[str, Any]:
        xs = [p[0] for p in self.v] or [0]
        ys = [p[1] for p in self.v] or [0]
        zs = [p[2] for p in self.v] or [0]
        return {"units": units, "vertices": [[round(c, 4) for c in p] for p in self.v], "faces": self.f,
                "bbox": [round(max(xs) - min(xs), 4), round(max(ys) - min(ys), 4), round(max(zs) - min(zs), 4)]}


def _area2(pts: Sequence[Pt]) -> float:
    return sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))


def _ray_polygon(cx: float, cy: float, d: Pt, pts: Sequence[Pt]) -> Optional[Pt]:
    """First hit of the ray from (cx, cy) along d with a polygon's edges (the polygon surrounds the origin)."""
    best = None
    for i in range(len(pts)):
        (ax, ay), (bx, by) = pts[i], pts[(i + 1) % len(pts)]
        ex, ey = bx - ax, by - ay
        den = d[0] * ey - d[1] * ex
        if abs(den) < 1e-15:
            continue
        t = ((ax - cx) * ey - (ay - cy) * ex) / den
        u = ((ax - cx) * d[1] - (ay - cy) * d[0]) / den
        if t > 1e-12 and -1e-9 <= u <= 1 + 1e-9 and (best is None or t < best):
            best = t
    if best is None:
        return None
    return cx + d[0] * best, cy + d[1] * best


_COARSE = [False]  # set while building the lighter mesh used for a drawing's small isometric view


def plate_solid(m: Mesh, L: float, W: float, z0: float, z1: float, holes: Sequence[Dict[str, float]] = (),
                pockets: Sequence[Tuple[float, float, float, float, float]] = (), xcuts: Sequence[float] = (),
                ycuts: Sequence[float] = (), xf=None, max_holes: int = 250, split_walls: bool = True,
                merge_cells: bool = True, coarse: bool = False) -> None:
    """A rectangular plate [0,L] x [0,W] x [z0,z1] with round holes and rectangular pockets opened from the
    top, as one watertight component with no T-junctions (every face is convex).

    holes: dicts with cx, cy, r, and optionally r2 (counterbore or countersink radius), d2 (counterbore
           depth, or 0 for a countersink), depth (None or missing = through).
    pockets: (x0, y0, x1, y1, depth); depth >= the plate thickness makes a through window.
    xcuts/ycuts: extra grid lines, so a later non-linear xf (bending, tapering) has vertices to move.
    split_walls: side walls as strips of quads, so painter's-order renderers sort them locally.
    merge_cells: merge runs of grid cells into one face; turn off under a non-linear xf (bending), so every
                 face stays nearly planar."""
    T = z1 - z0
    if T <= 0 or L <= 0 or W <= 0:
        return
    tol = max(L, W) * 1e-9
    pk = []
    for x0, y0, x1, y1, dep in pockets:
        x0, x1 = max(x0, L * 0.002), min(x1, L * 0.998)
        y0, y1 = max(y0, W * 0.002), min(y1, W * 0.998)
        if x1 - x0 > tol and y1 - y0 > tol:
            pk.append((x0, y0, x1, y1, min(dep, T) if dep < T * 0.999 else T))
    tiles = []
    if coarse or _COARSE[0]:
        # a sheet's small isometric keeps a regular subset of a big pattern (enough to read as a grid)
        max_holes = min(max_holes, 40 if (coarse or _COARSE[0] == 2) else 64)
    hl = _subsample_holes([h for h in holes if all(math.isfinite(h.get(k) or 0.0) for k in ("cx", "cy", "r"))
                           and h["r"] > 0], max_holes)
    # Each hole owns a square tile around it. Sizes are settled together, so that a big hole next to a small
    # one (or a pocket) shrinks its tile instead of dropping the small hole: neighbors split the gap between
    # them in proportion to their radii, and a tile that still reaches into a pocket trims the pocket a little.
    cand = []
    for h in sorted(hl, key=lambda h: -max(h.get("r2") or 0, h["r"])):
        cx, cy, r = h["cx"], h["cy"], h["r"]
        r2 = max(r, h.get("r2") or 0.0)
        need = r2 * 1.04
        lim = min(r2 * 1.55, cx, L - cx, cy, W - cy)
        if lim < need:
            continue  # breaks out of the plate's edge
        cand.append({"h": h, "cx": cx, "cy": cy, "r": r, "r2": r2, "need": need, "lim": lim})
    for c in cand:
        for k, (x0, y0, x1, y1, dep) in enumerate(pk):
            dch = max(x0 - c["cx"], c["cx"] - x1, y0 - c["cy"], c["cy"] - y1)
            if dch >= c["need"]:
                c["lim"] = min(c["lim"], dch)
                continue
            # the smallest tile would reach into the pocket: move the pocket's nearest side back, if that is a
            # small change, else give up the hole (it really runs into the pocket)
            n_ = c["need"]
            opts = []
            if x0 > c["cx"]:
                opts.append(((c["cx"] + n_ - x0) / (x1 - x0), (c["cx"] + n_, y0, x1, y1, dep)))
            if x1 < c["cx"]:
                opts.append(((x1 - (c["cx"] - n_)) / (x1 - x0), (x0, y0, c["cx"] - n_, y1, dep)))
            if y0 > c["cy"]:
                opts.append(((c["cy"] + n_ - y0) / (y1 - y0), (x0, c["cy"] + n_, x1, y1, dep)))
            if y1 < c["cy"]:
                opts.append(((y1 - (c["cy"] - n_)) / (y1 - y0), (x0, y0, x1, c["cy"] - n_, dep)))
            opts = [o for o in opts if o[0] <= 0.3]
            if opts:
                pk[k] = min(opts)[1]
                c["lim"] = min(c["lim"], n_)
            else:
                c["lim"] = -1.0
    for a in range(len(cand)):
        ca = cand[a]
        for b in range(a + 1, len(cand)):
            cb = cand[b]
            d = max(abs(ca["cx"] - cb["cx"]), abs(ca["cy"] - cb["cy"]))
            if d >= ca["lim"] + cb["lim"]:
                continue
            share = d * ca["r2"] / (ca["r2"] + cb["r2"])
            ca["lim"] = min(ca["lim"], share)
            cb["lim"] = min(cb["lim"], d - share)
    for c in cand:
        if c["lim"] < c["need"]:
            continue  # touches a neighboring hole
        h, cx, cy, r, r2, half = c["h"], c["cx"], c["cy"], c["r"], c["r2"], c["lim"]
        depth = h.get("depth")
        blind = depth is not None and depth < T * 0.995
        tiles.append({"cx": cx, "cy": cy, "r": r, "r2": r2, "d2": h.get("d2") or 0.0, "h": half,
                      "zb": z1 - depth if blind else z0, "thru": not blind})
    nseg_base = 14 if len(tiles) <= 8 else 10 if len(tiles) <= 20 else 8
    if coarse or _COARSE[0]:
        nseg_base = 8
    xs = {0.0, L} | {x for x in xcuts if 0 < x < L}
    ys = {0.0, W} | {y for y in ycuts if 0 < y < W}
    for t in tiles:
        xs |= {t["cx"] - t["h"], t["cx"] + t["h"]}
        ys |= {t["cy"] - t["h"], t["cy"] + t["h"]}
    for x0, y0, x1, y1, _ in pk:
        xs |= {x0, x1}
        ys |= {y0, y1}
    xs_l = _uniq(sorted(xs), tol)
    ys_l = _uniq(sorted(ys), tol)
    snap_x = {round(v, 9): v for v in xs_l}
    snap_y = {round(v, 9): v for v in ys_l}
    extra: Dict[Tuple[str, float], List[float]] = {}  # ('x'|'y', line) -> coordinates of extra points on it

    def grid_val(axis: str, v: float) -> float:
        """The grid line a coordinate lies on, exactly as the grid holds it."""
        snap = snap_x if axis == "x" else snap_y
        hit = snap.get(round(v, 9))
        if hit is None:
            grid = xs_l if axis == "x" else ys_l
            hit = min(grid, key=lambda g: abs(g - v))
            snap[round(v, 9)] = hit
        return hit

    def line_key(axis: str, v: float) -> Tuple[str, float]:
        return (axis, round(grid_val(axis, v), 9))

    def edge_pts(z: float, axis: str, line: float, a: float, b: float) -> List[float]:
        """Extra points strictly between a and b on a grid line, in order from a to b (same at every level)."""
        vals = [c for c in extra.get(line_key(axis, line), []) if min(a, b) + tol * 20 < c < max(a, b) - tol * 20]
        return sorted(vals, reverse=b < a)

    # each tile's own boundary samples go on the grid lines first, so touching tiles and cells all see them
    for t in tiles:
        cx, cy, hh = t["cx"], t["cy"], t["h"]
        t["n"] = n = max(nseg_base, min(12 if (coarse or _COARSE[0]) else 28, int(t["r2"] / max(L, W) * 120)))
        x0, x1, y0, y1 = cx - hh, cx + hh, cy - hh, cy + hh
        sq = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for k in range(n):
            a = 2 * math.pi * (k + 0.5) / n
            hit = _ray_polygon(cx, cy, (math.cos(a), math.sin(a)), sq)
            if not hit:
                continue
            px, py = hit
            dists = [(abs(px - x0), "x", x0, py), (abs(px - x1), "x", x1, py), (abs(py - y0), "y", y0, px),
                     (abs(py - y1), "y", y1, px)]
            _, axis, line, coord = min(dists)
            extra.setdefault(line_key(axis, line), []).append(coord)
    for key in extra:
        vals = sorted(extra[key])
        extra[key] = [v for i, v in enumerate(vals) if i == 0 or v - vals[i - 1] > tol * 40]
    for t in tiles:
        cx, cy, hh = t["cx"], t["cy"], t["h"]
        # the tile's sides as the grid holds them: two holes in line can put a side a rounding error apart,
        # and a vertex that far from the cells' own would not weld to it (an open seam around the tile)
        x0, x1, y0, y1 = grid_val("x", cx - hh), grid_val("x", cx + hh), grid_val("y", cy - hh), grid_val("y", cy + hh)
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        pts += [(x, y0) for x in xs_l if x0 + tol < x < x1 - tol] + [(x, y1) for x in xs_l if x0 + tol < x < x1 - tol]
        pts += [(x0, y) for y in ys_l if y0 + tol < y < y1 - tol] + [(x1, y) for y in ys_l if y0 + tol < y < y1 - tol]
        pts += [(x, y0) for x in edge_pts(z1, "y", y0, x0, x1)] + [(x, y1) for x in edge_pts(z1, "y", y1, x0, x1)]
        pts += [(x0, y) for y in edge_pts(z1, "x", x0, y0, y1)] + [(x1, y) for y in edge_pts(z1, "x", x1, y0, y1)]
        merged = []
        for q in sorted(pts, key=lambda q: math.atan2(q[1] - cy, q[0] - cx) % (2 * math.pi)):
            if not merged or abs(q[0] - merged[-1][0]) + abs(q[1] - merged[-1][1]) > tol * 10:
                merged.append(q)
        if len(merged) > 1 and abs(merged[0][0] - merged[-1][0]) + abs(merged[0][1] - merged[-1][1]) <= tol * 10:
            merged.pop()
        t["ring"] = merged

    m.begin(xf)

    def run_poly(i0: int, i1: int, j: int, z: float) -> List[Tuple[float, float, float]]:
        """One convex face over cells i0..i1-1 of row j, with every grid and extra point on its edges."""
        xa, xb, ya, yb = xs_l[i0], xs_l[i1], ys_l[j], ys_l[j + 1]
        mids = xs_l[i0 + 1:i1]
        bot = sorted(set(mids) | set(edge_pts(z, "y", ya, xa, xb)))
        top = sorted(set(mids) | set(edge_pts(z, "y", yb, xa, xb)), reverse=True)
        pts = [(xa, ya, z)] + [(x, ya, z) for x in bot] + [(xb, ya, z)]
        pts += [(xb, y, z) for y in edge_pts(z, "x", xb, ya, yb)]
        pts += [(xb, yb, z)] + [(x, yb, z) for x in top] + [(xa, yb, z)]
        pts += [(xa, y, z) for y in edge_pts(z, "x", xa, yb, ya)]
        return pts

    def owner(mx: float, my: float, row_tiles: Sequence[Dict[str, Any]], row_pk: Sequence[Any]) -> Tuple[str, Any]:
        for t in row_tiles:
            if abs(mx - t["cx"]) < t["h"] - tol:
                return "tile", t
        for p in row_pk:
            if p[0] + tol < mx < p[2] - tol:
                return "pocket", p
        return "top", None

    nx = len(xs_l) - 1
    for j in range(len(ys_l) - 1):
        my = (ys_l[j] + ys_l[j + 1]) / 2
        # only the tiles and pockets this row crosses (a big hole pattern has hundreds of tiles)
        row_tiles = [t for t in tiles if abs(my - t["cy"]) < t["h"] - tol]
        row_pk = [p for p in pk if p[1] + tol < my < p[3] - tol]
        kinds = [owner((xs_l[i] + xs_l[i + 1]) / 2, my, row_tiles, row_pk) for i in range(nx)]
        # faces facing up: the top surface and pocket floors, merged along the row
        i = 0
        while i < nx:
            kind, obj = kinds[i]
            if kind == "tile" or (kind == "pocket" and obj[4] >= T):
                i += 1
                continue
            k = i + 1
            while merge_cells and k < nx and kinds[k][0] == kind and kinds[k][1] is obj:
                k += 1
            z = z1 if kind == "top" else z1 - obj[4]
            m.poly(run_poly(i, k, j, z), (0, 0, 1))
            i = k
        # the bottom, wherever the plate is not open (through holes and through windows)
        i = 0
        while i < nx:
            kind, obj = kinds[i]
            if (kind == "tile" and obj["thru"]) or (kind == "pocket" and obj[4] >= T):
                i += 1
                continue
            k = i + 1
            while merge_cells and k < nx and not ((kinds[k][0] == "tile" and kinds[k][1]["thru"]) or
                                                      (kinds[k][0] == "pocket" and kinds[k][1][4] >= T)):
                k += 1
            m.poly(run_poly(i, k, j, z0), (0, 0, -1))
            i = k

    def wall(axis: str, line: float, a: float, b: float, zlo: float, zhi: float, normal: Tuple[float, float, float]) -> None:
        grid = ys_l if axis == "x" else xs_l
        inner = [g for g in grid if min(a, b) + tol < g < max(a, b) - tol]

        def pts_at(z: float, fwd: bool) -> List[float]:
            vals = sorted(set(inner) | set(edge_pts(z, axis, line, a, b)))
            vals = [a] + vals + [b] if a < b else [a] + vals[::-1] + [b]
            return vals if fwd else vals[::-1]

        lo = pts_at(zlo, True)
        hi = pts_at(zhi, False)

        def P(c: float, z: float) -> Tuple[float, float, float]:
            return (line, c, z) if axis == "x" else (c, line, z)

        if split_walls and len(lo) == len(hi):
            for c0, c1 in zip(lo, lo[1:]):
                m.poly([P(c0, zlo), P(c1, zlo), P(c1, zhi), P(c0, zhi)], normal)
        else:
            m.poly([P(c, zlo) for c in lo] + [P(c, zhi) for c in hi], normal)

    wall("y", 0.0, 0.0, L, z0, z1, (0, -1, 0))
    wall("y", W, 0.0, L, z0, z1, (0, 1, 0))
    wall("x", 0.0, 0.0, W, z0, z1, (-1, 0, 0))
    wall("x", L, 0.0, W, z0, z1, (1, 0, 0))
    for x0, y0, x1, y1, dep in pk:
        zf = z1 - dep if dep < T else z0
        wall("y", y0, x0, x1, zf, z1, (0, 1, 0))
        wall("y", y1, x0, x1, zf, z1, (0, -1, 0))
        wall("x", x0, y0, y1, zf, z1, (1, 0, 0))
        wall("x", x1, y0, y1, zf, z1, (-1, 0, 0))
    for t in tiles:
        cx, cy, r, r2 = t["cx"], t["cy"], t["r"], t["r2"]
        ring = t["ring"]
        ang = [math.atan2(p[1] - cy, p[0] - cx) for p in ring]
        k = len(ring)

        def circ(rad: float, z: float) -> List[Tuple[float, float, float]]:
            return [(cx + rad * math.cos(a), cy + rad * math.sin(a), z) for a in ang]

        top = circ(r2, z1)
        for i in range(k):
            j = (i + 1) % k
            m.poly([(ring[i][0], ring[i][1], z1), (ring[j][0], ring[j][1], z1), top[j], top[i]], (0, 0, 1))
        prof = [(r2, z1)]
        if r2 > r * 1.001:
            if t["d2"] > 0:
                zc = max(t["zb"] + (z1 - t["zb"]) * 0.05, z1 - t["d2"])
                prof += [(r2, zc), (r, zc)]
            else:
                prof.append((r, max(t["zb"] + (z1 - t["zb"]) * 0.05, z1 - (r2 - r))))
        prof.append((r, t["zb"]))
        for (ra, za), (rb, zb) in zip(prof, prof[1:]):
            A, B = circ(ra, za), circ(rb, zb)
            for i in range(k):
                j = (i + 1) % k
                if abs(ra - rb) > 1e-12 and abs(za - zb) <= 1e-12:
                    nrm = (0.0, 0.0, 1.0)
                else:
                    nrm = (-(math.cos(ang[i]) + math.cos(ang[j])), -(math.sin(ang[i]) + math.sin(ang[j])), 0.0)
                m.poly([A[i], A[j], B[j], B[i]], nrm)
        bottom = circ(r, t["zb"])
        if t["thru"]:
            for i in range(k):
                j = (i + 1) % k
                m.poly([(ring[i][0], ring[i][1], z0), (ring[j][0], ring[j][1], z0), bottom[j], bottom[i]], (0, 0, -1))
        else:
            m.poly(bottom, (0, 0, 1))
    m.end()


def _subsample_holes(holes: List[Dict[str, float]], limit: int) -> List[Dict[str, float]]:
    """Keep a regular subset of a big hole pattern (every k-th row and column), so the grid stays small."""
    if len(holes) <= limit:
        return holes
    keyed = [(round(h["cx"], 6), round(h["cy"], 6), h) for h in holes]
    xs = sorted({kx for kx, _, _ in keyed})
    ys = sorted({ky for _, ky, _ in keyed})
    for k in range(2, 40):
        kx = xs[::k] if len(xs) > 2 else xs
        ky = ys[::k] if len(ys) > 2 else ys
        if xs[-1] not in kx:
            kx = kx + [xs[-1]]
        if ys[-1] not in ky:
            ky = ky + [ys[-1]]
        sx, sy = set(kx), set(ky)  # built once per pass, not once per hole
        sel = [h for hx, hy, h in keyed if hx in sx and hy in sy]
        if 0 < len(sel) <= limit:
            return sel
    step = len(holes) / float(limit)
    return [holes[int(i * step)] for i in range(limit)]


def _uniq(vals: Sequence[float], tol: float) -> List[float]:
    out: List[float] = []
    for v in vals:
        if not out or v - out[-1] > tol * 20:
            out.append(v)
    return out


def check_mesh(mesh: Mesh, max_faces: int = 3000) -> List[str]:
    """Problems with a mesh: open or inconsistently wound components, inward components, too many faces."""
    problems: List[str] = []
    if len(mesh.f) > max_faces:
        problems.append(f"{len(mesh.f)} faces (more than {max_faces})")
    parts = mesh.parts or [(0, len(mesh.f))]
    for pi, (a, b) in enumerate(parts):
        faces = mesh.f[a:b]
        directed: Dict[Tuple[int, int], int] = {}
        for fc in faces:
            for k in range(len(fc)):
                e = (fc[k], fc[(k + 1) % len(fc)])
                directed[e] = directed.get(e, 0) + 1
        bad = sum(1 for (i, j), c in directed.items() if c != 1 or directed.get((j, i), 0) != 1)
        if bad:
            problems.append(f"component {pi}: {bad} unmatched or repeated edges")
        vol = _volume(mesh.v, faces)
        if vol <= 0:
            problems.append(f"component {pi}: signed volume {vol:.4g} is not positive")
    return problems


# --------------------------------------------------------------------------- #
# Shaded isometric rendering (painter's algorithm), shared by drawings and model tiles
# --------------------------------------------------------------------------- #
def _iso_project(p: Tuple[float, float, float], yaw: float, pitch: float) -> Tuple[float, float, float]:
    x, y, z = p
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, y = x * cy - y * sy, x * sy + y * cy
    cp, sp = math.cos(pitch), math.sin(pitch)
    y, z = y * cp - z * sp, y * sp + z * cp
    return x, -z, y  # screen x, screen y (down), depth (bigger = farther)


def render_mesh(page: Page, mesh: Mesh, x: float, y: float, w: float, h: float,
                base: Tuple[float, float, float] = (0.78, 0.82, 0.88), edges: Tuple[float, float, float] = (0.2, 0.22, 0.26),
                yaw_deg: float = -40.0, pitch_deg: float = 30.0, edge_lw: float = 0.5, fill_frac: float = 0.92,
                shadow: Optional[Tuple[float, float, float]] = None) -> Tuple[float, float, float, float]:
    """Draw a flat-shaded view with crease and silhouette edges. Returns the drawn bounding box."""
    V, F = mesh.v, mesh.f
    if not V or not F:
        return (x, y, x, y)
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    xs, ys, zs = [p[0] for p in V], [p[1] for p in V], [p[2] for p in V]
    cx, cy, cz = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2
    proj = [_iso_project((px - cx, py - cy, pz - cz), yaw, pitch) for px, py, pz in V]
    sx, sy = [p[0] for p in proj], [p[1] for p in proj]
    scale = min(w / max(max(sx) - min(sx), 1e-9), h / max(max(sy) - min(sy), 1e-9)) * fill_frac
    ox = x + w / 2 - (max(sx) + min(sx)) / 2 * scale
    oy = y + h / 2 - (max(sy) + min(sy)) / 2 * scale
    scr = [(ox + p[0] * scale, oy + p[1] * scale) for p in proj]
    if shadow is not None:
        zmin = min(zs) - cz
        foot = [_iso_project((a - cx, b - cy, zmin), yaw, pitch) for a, b in
                ((min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys)))]
        fx = [ox + p[0] * scale for p in foot]
        fy = [oy + p[1] * scale for p in foot]
        mx, my = sum(fx) / 4, sum(fy) / 4
        rx, ry = (max(fx) - min(fx)) / 2, (max(fy) - min(fy)) / 2
        for k, grow in enumerate((1.18, 1.08, 1.0)):
            c = tuple(min(1.0, v + (0.04 * (2 - k))) for v in shadow)
            page.path(_ellipse_cmds(mx, my + ry * 0.12, rx * grow, max(ry, rx * 0.12) * grow), 0, None, c)
    light = (-0.42, -0.62, 0.66)
    ll = math.sqrt(sum(c * c for c in light))
    light = (light[0] / ll, light[1] / ll, light[2] / ll)
    cyw, syw, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    normals = []
    for fc in F:
        nx, ny, nz = _newell([V[i] for i in fc])
        norm = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        normals.append((nx / norm, ny / norm, nz / norm))
    vis = []
    for nx, ny, nz in normals:
        y1 = nx * syw + ny * cyw
        vis.append(y1 * cp - nz * sp)  # view-space depth component: negative faces the viewer
    edge_faces: Dict[Tuple[int, int], List[int]] = {}
    for fi, fc in enumerate(F):
        for k in range(len(fc)):
            a, b = fc[k], fc[(k + 1) % len(fc)]
            edge_faces.setdefault((a, b) if a < b else (b, a), []).append(fi)
    # big faces are cut into screen-space pieces (plane-interpolated depth), then pieces are ordered by the
    # viewer's rule: B goes after A when B lies on the viewer's side of A's plane (only overlapping pairs are
    # compared, found through a screen grid); depth breaks ties and cycles. Crease flags follow the pieces.
    piece_max = max(w, h) * 0.14
    pieces: List[Tuple[float, List[Tuple[float, float, bool]], Tuple[float, float, float]]] = []
    info: List[Tuple[Tuple[float, float, float], float, List[Tuple[float, float, float]], Tuple[float, float, float, float]]] = []
    inv = 1.0 / scale
    owner: List[int] = []
    adjacent = {(fs[0], fs[1]) if fs[0] < fs[1] else (fs[1], fs[0]) for fs in edge_faces.values() if len(fs) == 2}
    for fi, fc in enumerate(F):
        if vis[fi] >= -1e-6:
            continue
        nx, ny, nz = normals[fi]
        nk = len(fc)
        poly = []
        for kk in range(nk):
            a, b = fc[kk], fc[(kk + 1) % nk]
            nb = [g for g in edge_faces[(a, b) if a < b else (b, a)] if g != fi]
            sh = not nb
            for g in nb:
                if vis[g] >= -1e-6:
                    sh = True
                    break
                gx, gy, gz = normals[g]
                if nx * gx + ny * gy + nz * gz < 0.85:
                    sh = True
                    break
            poly.append((scr[a][0], scr[a][1], sh))
        diff = max(0.0, nx * light[0] + ny * light[1] + nz * light[2])
        k = 0.4 + 0.46 * diff + 0.2 * max(0.0, -vis[fi])
        col = tuple(min(1.0, c * k + 0.05) for c in base)
        ds = [proj[i][2] for i in fc]
        # the face plane in view space (x right, y = depth away from the viewer, z up), normal toward the viewer
        y1 = nx * syw + ny * cyw
        nvx, nvy, nvz = nx * cyw - ny * syw, y1 * cp - nz * sp, y1 * sp + nz * cp
        p0 = proj[fc[0]]
        dpl = nvx * p0[0] + nvy * p0[2] + nvz * (-p0[1])
        nrm = (nvx, nvy, nvz)   # visible faces point at the viewer (-y), so the viewer's side is n.v > n.p0
        dneg = -dpl
        xs_f, ys_f = [q[0] for q in poly], [q[1] for q in poly]
        if max(xs_f) - min(xs_f) <= piece_max and max(ys_f) - min(ys_f) <= piece_max:
            parts = [(poly, ds)]
        else:
            plane = _depth_plane([scr[i] for i in fc], ds)
            cut = [poly]
            for axis, lo, hi in ((0, min(xs_f), max(xs_f)), (1, min(ys_f), max(ys_f))):
                n_cut = int((hi - lo) / piece_max)
                nxt = []
                for part in cut:
                    rest = part
                    for c in range(1, n_cut + 1):
                        left, rest = _cut_poly(rest, axis, lo + (hi - lo) * c / (n_cut + 1) + 0.013)
                        if len(left) >= 3:
                            nxt.append(left)
                    if len(rest) >= 3:
                        nxt.append(rest)
                cut = nxt
            parts = []
            for part in cut:
                if plane:
                    pd = [plane[0] * q[0] + plane[1] * q[1] + plane[2] for q in part]
                else:
                    pd = [sum(ds) / nk] * len(part)
                parts.append((part, pd))
        for part, pd in parts:
            owner.append(fi)
            vv = [((q[0] - ox) * inv, d, -(q[1] - oy) * inv) for q, d in zip(part, pd)]
            px, py = [q[0] for q in part], [q[1] for q in part]
            pieces.append((max(pd) * .5 + sum(pd) / len(pd) * .5, part, col))  # type: ignore[arg-type]
            info.append((nrm, dneg, vv, (min(px), min(py), max(px), max(py))))
    n_p = len(pieces)
    edges_of: List[Optional[List[Tuple[float, float, float, float]]]] = [None] * n_p
    succ: List[List[int]] = [[] for _ in range(n_p)]
    indeg = [0] * n_p
    eps = 1e-6 * max(w, h) * inv
    pkey = [(round(nr[0], 4), round(nr[1], 4), round(nr[2], 4), round(dn / max(eps, 1e-12) / 50)) for nr, dn, _, _ in info]

    def in_front(bi: int, ai: int) -> bool:
        (nx_, ny_, nz_), dn, _, _ = info[ai]
        for vx, vy, vz in info[bi][2]:
            if nx_ * vx + ny_ * vy + nz_ * vz + dn < -eps:
                return False
        return True

    def plane_depth(i: int, sx_: float, sy_: float) -> Optional[float]:
        (nx_, ny_, nz_), dn, _, _ = info[i]
        if abs(ny_) < 1e-9:
            return None
        X, Z = (sx_ - ox) * inv, -(sy_ - oy) * inv
        return -(dn + nx_ * X + nz_ * Z) / ny_

    import heapq
    order_x = sorted(range(n_p), key=lambda i: info[i][3][0])
    active: List[Tuple[float, int]] = []
    live: set = set()
    for i in order_x:
        bi_ = info[i][3]
        while active and active[0][0] < bi_[0] + 0.02:
            live.discard(heapq.heappop(active)[1])
        for j in live:
            bj = info[j][3]
            if bi_[3] <= bj[1] + 0.02 or bj[3] <= bi_[1] + 0.02 or pkey[i] == pkey[j]:
                continue
            jf, iff = in_front(j, i), in_front(i, j)
            fa, fb = owner[i], owner[j]
            if jf == iff and not jf and ((fa, fb) if fa < fb else (fb, fa)) not in adjacent:
                # undecided by the planes: compare both planes' depths where the two pieces overlap
                if edges_of[i] is None:
                    edges_of[i] = _edges(pieces[i][1])
                if edges_of[j] is None:
                    edges_of[j] = _edges(pieces[j][1])
                pt = _common_point(pieces[i][1], pieces[j][1], edges_of[i], edges_of[j])
                if pt is not None:
                    di, dj = plane_depth(i, pt[0], pt[1]), plane_depth(j, pt[0], pt[1])
                    if di is not None and dj is not None and abs(di - dj) > eps * 20:
                        jf, iff = dj < di, di < dj
            if jf and not iff:
                succ[i].append(j)
                indeg[j] += 1
            elif iff and not jf:
                succ[j].append(i)
                indeg[i] += 1
        heapq.heappush(active, (bi_[2], i))
        live.add(i)
    heap = [(-pieces[i][0], i) for i in range(n_p) if indeg[i] == 0]
    heapq.heapify(heap)
    done = [False] * n_p
    ordered: List[int] = []
    while len(ordered) < n_p:
        if not heap:
            rest = [i for i in range(n_p) if not done[i]]
            i = max(rest, key=lambda r: pieces[r][0])
            indeg[i] = 0
        else:
            _, i = heapq.heappop(heap)
            if done[i]:
                continue
        done[i] = True
        ordered.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0 and not done[j]:
                heapq.heappush(heap, (-pieces[j][0], j))
    pieces = [pieces[i] for i in ordered]
    for _, part, col in pieces:
        pts = [(q[0], q[1]) for q in part]
        nk = len(part)
        flags = [q[2] for q in part]
        if all(flags):
            page.polygon(pts, edge_lw, edges, col)  # type: ignore[arg-type]
            continue
        page.polygon(pts, 0.35, col, col)  # type: ignore[arg-type]
        if any(flags):
            start = next(k for k in range(nk) if flags[k] and not flags[k - 1])
            chain: List[Pt] = []
            for step in range(nk):
                k = (start + step) % nk
                if flags[k]:
                    if not chain:
                        chain.append(pts[k])
                    chain.append(pts[(k + 1) % nk])
                elif chain:
                    page.polygon(chain, edge_lw, edges, None, closed=False)
                    chain = []
            if chain:
                page.polygon(chain, edge_lw, edges, None, closed=False)
    xs2, ys2 = [p[0] for p in scr], [p[1] for p in scr]
    return min(xs2), min(ys2), max(xs2), max(ys2)


def _depth_plane(pts: Sequence[Pt], depths: Sequence[float]) -> Optional[Tuple[float, float, float]]:
    """depth = a*x + b*y + c through the face's projected vertices (orthographic, so planar faces are exact)."""
    n = len(pts)
    best = None
    for i in range(1, n - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[0], pts[i], pts[i + 1]
        det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if best is None or abs(det) > abs(best[0]):
            best = (det, i)
    if best is None or abs(best[0]) < 1e-9:
        return None
    det, i = best
    (x0, y0), (x1, y1), (x2, y2) = pts[0], pts[i], pts[i + 1]
    d0, d1, d2 = depths[0], depths[i], depths[i + 1]
    a = ((d1 - d0) * (y2 - y0) - (d2 - d0) * (y1 - y0)) / det
    b = ((x1 - x0) * (d2 - d0) - (x2 - x0) * (d1 - d0)) / det
    return a, b, d0 - a * x0 - b * y0


def _edges(pa: Sequence[Tuple[float, ...]]) -> List[Tuple[float, float, float, float]]:
    """Edge lines of a convex polygon as (ax, ay, ex, ey), oriented so the inside is on the left."""
    n = len(pa)
    area = 0.0
    for i in range(n):
        area += pa[i][0] * pa[(i + 1) % n][1] - pa[(i + 1) % n][0] * pa[i][1]
    sign = 1.0 if area > 0 else -1.0
    return [(pa[i][0], pa[i][1], (pa[(i + 1) % n][0] - pa[i][0]) * sign, (pa[(i + 1) % n][1] - pa[i][1]) * sign)
            for i in range(n)]


def _separated(ea: Sequence[Tuple[float, float, float, float]], pb: Sequence[Tuple[float, ...]]) -> bool:
    """Is one of polygon a's edges (from _edges) a separating axis against the convex polygon pb?"""
    for ax, ay, ex, ey in ea:
        for q in pb:
            if ex * (q[1] - ay) - ey * (q[0] - ax) > 1e-6:
                break
        else:
            return True
    return False


def _common_point(pa: Sequence[Tuple[float, ...]], pb: Sequence[Tuple[float, ...]], ea=None, eb=None) -> Optional[Pt]:
    """A point inside both convex polygons (the centroid of their intersection), if they really overlap."""
    ea = ea if ea is not None else _edges(pa)
    eb = eb if eb is not None else _edges(pb)
    if _separated(ea, pb) or _separated(eb, pa):
        return None
    out = [(q[0], q[1]) for q in pa]
    for ax, ay, ex, ey in eb:
        inp, out = out, []
        n = len(inp)
        for i in range(n):
            P, Q = inp[i], inp[(i + 1) % n]
            sp = ex * (P[1] - ay) - ey * (P[0] - ax)
            sq = ex * (Q[1] - ay) - ey * (Q[0] - ax)
            if sp >= 0:
                out.append(P)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append((P[0] + (Q[0] - P[0]) * t, P[1] + (Q[1] - P[1]) * t))
        if len(out) < 3:
            return None
    area = sum(out[i][0] * out[(i + 1) % len(out)][1] - out[(i + 1) % len(out)][0] * out[i][1] for i in range(len(out)))
    if abs(area) < 0.02:
        return None
    return sum(q[0] for q in out) / len(out), sum(q[1] for q in out) / len(out)


def _cut_poly(poly: List[Tuple[float, float, bool]], axis: int, c: float):
    """Split a convex polygon of (x, y, edge-is-crease) at x = c (axis 0) or y = c (axis 1)."""
    left: List[Tuple[float, float, bool]] = []
    right: List[Tuple[float, float, bool]] = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        pv, qv = p[axis] - c, q[axis] - c
        if pv <= 0:
            left.append(p if (pv < 0 or qv <= 0) else (p[0], p[1], False))
        if pv >= 0:
            right.append(p if (pv > 0 or qv >= 0) else (p[0], p[1], False))
        if (pv < 0 < qv) or (qv < 0 < pv):
            t = pv / (pv - qv)
            ix, iy = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
            if pv < 0:
                left.append((ix, iy, False))
                right.append((ix, iy, p[2]))
            else:
                right.append((ix, iy, False))
                left.append((ix, iy, p[2]))
    return left, right


def _ellipse_cmds(cx: float, cy: float, rx: float, ry: float) -> List[Tuple]:
    k = 0.5523
    return [("M", cx + rx, cy), ("C", cx + rx, cy + k * ry, cx + k * rx, cy + ry, cx, cy + ry),
            ("C", cx - k * rx, cy + ry, cx - rx, cy + k * ry, cx - rx, cy),
            ("C", cx - rx, cy - k * ry, cx - k * rx, cy - ry, cx, cy - ry),
            ("C", cx + k * rx, cy - ry, cx + rx, cy - k * ry, cx + rx, cy), ("Z",)]


def iso_view(page: Page, spec: Dict[str, Any], x: float, y: float, w: float, h: float,
             base: Tuple[float, float, float] = (0.78, 0.82, 0.88), edges: Tuple[float, float, float] = (0.2, 0.22, 0.26),
             yaw_deg: float = -40.0, pitch_deg: float = 30.0) -> None:
    render_mesh(page, build_mesh(spec), x, y, w, h, base, edges, yaw_deg, pitch_deg)


def model_thumb_page(spec: Dict[str, Any]) -> Page:
    page = Page(320, 220)
    mesh = build_mesh(spec, coarse="thumb")
    render_mesh(page, mesh, 22, 14, 276, 180, base=(0.66, 0.73, 0.83), edges=(0.18, 0.21, 0.27),
                edge_lw=0.55, fill_frac=0.9, shadow=(0.86, 0.88, 0.9))
    # a small axis triad, like a CAD viewport
    yaw, pitch = math.radians(-40.0), math.radians(30.0)
    ox, oy = 20.0, 204.0
    for vec, label, col in (((1, 0, 0), "X", (0.78, 0.2, 0.18)), ((0, 1, 0), "Y", (0.2, 0.6, 0.25)),
                            ((0, 0, 1), "Z", (0.2, 0.36, 0.78))):
        px, py, _ = _iso_project(vec, yaw, pitch)
        page.line(ox, oy, ox + px * 11, oy + py * 11, 0.9, col)
        page.text(ox + px * 15, oy + py * 15 + 2, label, 5, True, "middle", col)
    return page


# --------------------------------------------------------------------------- #
# STEP (ISO 10303-21) with a faceted B-rep per closed component
# --------------------------------------------------------------------------- #
def _step_str(text: Any) -> str:
    out = []
    for ch in clean(text):
        if ch == "'":
            out.append("''")
        elif ch == "\\":
            out.append("\\\\")
        elif 32 <= ord(ch) <= 126:
            out.append(ch)
        else:
            out.append("\\X2\\%04X\\X0\\" % ord(ch))
    return "".join(out)


def step_file(spec: Dict[str, Any]) -> str:
    mesh = build_mesh(spec)
    units = spec.get("units") or "in"
    pn = _step_str(spec.get("part_number") or "PART")
    title = _step_str(spec.get("title") or spec.get("part_number") or "PART")
    rev = _step_str(spec.get("rev") or "-")
    system = _step_str(spec.get("originating_system") or "RFQ Router demo")
    author = _step_str(spec.get("author") or "engineering")
    schema = spec.get("schema") or "AP214"
    schema_name, proto, year = {
        "AP203": ("CONFIG_CONTROL_DESIGN", "config_control_design", 1994),
        "AP242": ("AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }",
                  "ap242_managed_model_based_3d_engineering", 2014),
    }.get(schema, ("AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }", "automotive_design", 2000))
    lines: List[str] = []
    n = [0]

    def ent(text: str) -> int:
        n[0] += 1
        lines.append(f"#{n[0]}={text};")
        return n[0]

    app = ent("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    ent(f"APPLICATION_PROTOCOL_DEFINITION('international standard','{proto}',{year},#{app})")
    pctx = ent(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    prod = ent(f"PRODUCT('{pn}','{title}','Rev {rev}',(#{pctx}))")
    ent(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{prod}))")
    pdf = ent(f"PRODUCT_DEFINITION_FORMATION('{rev}','',#{prod})")
    dctx = ent(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")
    pd = ent(f"PRODUCT_DEFINITION('design','',#{pdf},#{dctx})")
    pds = ent(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
    if units == "mm":
        lu = ent("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    else:
        mm = ent("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
        lmu = ent(f"LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#{mm})")
        dim = ent("DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.)")
        lu = ent(f"(CONVERSION_BASED_UNIT('INCH',#{lmu})LENGTH_UNIT()NAMED_UNIT(#{dim}))")
    au = ent("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    su = ent("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = ent(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-05),#{lu},'distance_accuracy_value','confusion accuracy')")
    ctx = ent(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))"
              f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{lu},#{au},#{su}))REPRESENTATION_CONTEXT('Context3D','3D'))")
    pts = [ent(f"CARTESIAN_POINT('',({x:.5f},{y:.5f},{z:.5f}))") for x, y, z in mesh.v]
    breps = []
    for ci, (a, b) in enumerate(mesh.parts or [(0, len(mesh.f))]):
        faces = []
        for face in mesh.f[a:b]:
            for loop_ids in _planar_pieces(mesh, face):
                loop = ent("POLY_LOOP('',(" + ",".join(f"#{pts[i]}" for i in loop_ids) + "))")
                bound = ent(f"FACE_OUTER_BOUND('',#{loop},.T.)")
                faces.append(ent(f"FACE('',(#{bound}))"))
        if faces:
            shell = ent("CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in faces) + "))")
            breps.append(ent(f"FACETED_BREP('{pn} solid {ci + 1}',#{shell})"))
    origin = ent("CARTESIAN_POINT('',(0.,0.,0.))")
    zdir = ent("DIRECTION('',(0.,0.,1.))")
    xdir = ent("DIRECTION('',(1.,0.,0.))")
    axis = ent(f"AXIS2_PLACEMENT_3D('',#{origin},#{zdir},#{xdir})")
    rep = ent(f"FACETED_BREP_SHAPE_REPRESENTATION('{pn}',(#{axis}" + "".join(f",#{b}" for b in breps) + f"),#{ctx})")
    ent(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
    name = _step_str(spec.get("name") or (clean(spec.get("part_number") or "part") + ".step"))
    date = re.sub(r"[^0-9-]", "", clean(spec.get("date") or "")) or "2026-09-01"
    header = [
        "ISO-10303-21;",
        "HEADER;",
        f"FILE_DESCRIPTION(('{title}','Faceted model for quoting'),'2;1');",
        f"FILE_NAME('{name}','{date[:10]}T08:00:00',('{author}'),(''),'RFQ Router demo','{system}','');",
        f"FILE_SCHEMA(('{schema_name}'));",
        "ENDSEC;",
        "DATA;",
    ]
    return "\n".join(header + lines + ["ENDSEC;", "END-ISO-10303-21;", ""])


def _planar_pieces(mesh: Mesh, face: List[int]) -> List[List[int]]:
    """A faceted B-rep wants planar faces: split a warped polygon into triangles."""
    if len(face) <= 3:
        return [face]
    pts = [mesh.v[i] for i in face]
    nx, ny, nz = _newell(pts)
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm <= 1e-18:
        return [face]
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    x0, y0, z0 = pts[0]
    size = max(max(abs(p[i] - pts[0][i]) for p in pts) for i in range(3)) or 1.0
    if all(abs((p[0] - x0) * nx + (p[1] - y0) * ny + (p[2] - z0) * nz) <= size * 1e-6 for p in pts):
        return [face]
    return [[face[0], face[k], face[k + 1]] for k in range(1, len(face) - 1)]


# --------------------------------------------------------------------------- #
# Part descriptions
# --------------------------------------------------------------------------- #
Anchor = Tuple[str, float, float, float]  # (view, u, v, circle radius in part units or 0)


class Part:
    """Everything the views and the mesh need to know about one part, made once from the spec."""

    def __init__(self, spec: Dict[str, Any]):
        shape = spec.get("shape")
        if shape not in PRISMATIC | COMPLEX | ROUND | OTHER:
            shape = "block"
        self.spec = spec
        self.shape: str = shape
        self.units: str = "mm" if spec.get("units") == "mm" else "in"
        self.d = _dims(dict(spec, shape=shape))
        self.callouts = [clean(c).strip() for c in _text_list(spec.get("callouts")) if clean(c).strip()][:4]
        self.cs = [parse_callout(c, self.units) for c in self.callouts]
        self.anchors: Dict[int, Anchor] = {}
        self.family = ("prismatic" if shape in PRISMATIC else "complex" if shape in COMPLEX else
                       "round" if shape in ROUND else "other")
        self.flag = ""          # a short boxed note shown under the isometric view
        self.parts_list: List[Tuple[str, str, str, int]] = []
        self.balloons: List[Tuple[int, Anchor]] = []

    def f(self, v: float) -> str:
        return fmt(v, self.units)


_PART_CACHE: "OrderedDict[str, Part]" = OrderedDict()


def part_model(spec: Dict[str, Any]) -> Part:
    keys = ("shape", "size", "units", "callouts", "part_number", "rev")
    key = json.dumps({k: spec.get(k) for k in keys}, sort_keys=True, default=str)
    hit = _PART_CACHE.get(key)
    if hit is not None:
        _PART_CACHE.move_to_end(key)
        return hit
    p = Part(spec)
    builder = {"prismatic": _build_prismatic, "round": _build_round, "complex": _build_complex,
               "other": _build_other}[p.family]
    try:
        builder(p)
    except Exception:  # noqa: BLE001 - a strange spec must still produce a drawing
        p = Part(dict(spec, shape="block", callouts=[]))
        _build_prismatic(p)
    _PART_CACHE[key] = p
    while len(_PART_CACHE) > 96:
        _PART_CACHE.popitem(last=False)
    return p


class Face2D:
    """A rectangular face [0, w] x [0, h] where features are placed without colliding."""

    def __init__(self, w: float, h: float):
        self.w, self.h = w, h
        self.circles: List[Tuple[float, float, float]] = []
        self.rects: List[Tuple[float, float, float, float]] = []
        self.unit = min(w, h)
        self.grid_corners: Optional[List[Pt]] = None

    def fits(self, x: float, y: float, r: float, gap: float) -> bool:
        edge = r + gap
        if x < edge or y < edge or x > self.w - edge or y > self.h - edge:
            return False
        for cx, cy, cr in self.circles:
            if (x - cx) ** 2 + (y - cy) ** 2 < (r + cr + gap) ** 2:
                return False
        for x0, y0, x1, y1 in self.rects:
            dx = max(x0 - x, 0.0, x - x1)
            dy = max(y0 - y, 0.0, y - y1)
            if dx * dx + dy * dy < (r + gap) ** 2:
                return False
        return True

    def rect_fits(self, x0: float, y0: float, x1: float, y1: float, gap: float) -> bool:
        if x0 < gap or y0 < gap or x1 > self.w - gap or y1 > self.h - gap:
            return False
        for cx, cy, cr in self.circles:
            dx = max(x0 - cx, 0.0, cx - x1)
            dy = max(y0 - cy, 0.0, cy - y1)
            if dx * dx + dy * dy < (cr + gap) ** 2:
                return False
        for a0, b0, a1, b1 in self.rects:
            if x0 < a1 + gap and a0 < x1 + gap and y0 < b1 + gap and b0 < y1 + gap:
                return False
        return True


def _perimeter(n: int, w: float, h: float, ex: float, ey: float) -> List[Pt]:
    sw, sh = w - 2 * ex, h - 2 * ey
    if sw <= 0 or sh <= 0 or n < 4:
        return []
    if n % 2 == 0:
        best = None
        for nx in range(1, n // 2):
            ny = n // 2 - nx
            err = abs(sw / nx - sh / ny)
            if best is None or err < best[0]:
                best = (err, nx, ny)
        _, nx, ny = best  # type: ignore[misc]
        pts = [(ex + sw * i / nx, ey) for i in range(nx)] + [(w - ex, ey + sh * j / ny) for j in range(ny)]
        pts += [(w - ex - sw * i / nx, h - ey) for i in range(nx)] + [(ex, h - ey - sh * j / ny) for j in range(ny)]
        return pts
    per = 2 * (sw + sh)
    out = []
    for k in range(n):
        s = per * k / n
        if s < sw:
            out.append((ex + s, ey))
        elif s < sw + sh:
            out.append((w - ex, ey + s - sw))
        elif s < 2 * sw + sh:
            out.append((w - ex - (s - sw - sh), h - ey))
        else:
            out.append((ex, h - ey - (s - 2 * sw - sh)))
    return out


def _patterns(n: int, w: float, h: float, e: float) -> List[List[Pt]]:
    ex = e
    c: List[List[Pt]] = []
    if n == 1:
        c += [[(w / 2, h / 2)], [(w - ex, h / 2)], [(ex, h / 2)], [(w / 2, h - e)], [(w / 2, e)], [(w - ex, h - e)],
              [(ex, e)], [(w - ex, e)], [(ex, h - e)]]
    elif n == 2:
        if w >= h:
            c += [[(ex, h / 2), (w - ex, h / 2)], [(ex, e), (w - ex, h - e)], [(w / 2, e), (w / 2, h - e)]]
        else:
            c += [[(w / 2, e), (w / 2, h - e)], [(ex, e), (w - ex, h - e)], [(ex, h / 2), (w - ex, h / 2)]]
        c += [[(ex, h - e), (w - ex, e)], [(w - ex, e), (w - ex, h - e)], [(ex, e), (ex, h - e)]]
    elif n == 3:
        if w >= h:
            c += [[(ex, h / 2), (w / 2, h / 2), (w - ex, h / 2)], [(ex, e), (w - ex, e), (w / 2, h - e)]]
        else:
            c += [[(w / 2, e), (w / 2, h / 2), (w / 2, h - e)], [(ex, e), (ex, h - e), (w - ex, h / 2)]]
        c += [[(w * .25, h - e), (w * .5, h - e), (w * .75, h - e)], [(w * .25, e), (w * .5, e), (w * .75, e)]]
    elif n == 4:
        c += [[(ex, e), (w - ex, e), (w - ex, h - e), (ex, h - e)]]
        if w >= h:
            c += [[(ex + (w - 2 * ex) * k / 3, h / 2) for k in range(4)]]
        else:
            c += [[(w / 2, e + (h - 2 * e) * k / 3) for k in range(4)]]
    if n >= 6:
        c += _grids(n, w, h, e)
    if n >= 4:
        per = _perimeter(n, w, h, ex, e)
        if per:
            c.append(per)
    if n >= 2:
        c.append([(ex + (w - 2 * ex) * k / (n - 1), h / 2) for k in range(n)])
    return c


def _grids(n: int, w: float, h: float, e: float, limit: int = 3) -> List[List[Pt]]:
    """Rectangular grids of n holes: exact a x b grids, and a x b >= n grids with the few extra positions
    left out at the corners (212 holes = 18 x 12 minus 4 corners), ordered by how well they match the face."""
    sw, sh = max(w - 2 * e, 1e-9), max(h - 2 * e, 1e-9)
    cands = []
    for a in range(2, n + 1):
        b = -(-n // a)
        if b < 2:
            continue
        extra = a * b - n
        if extra >= min(a, b) or extra > 4:
            continue
        fit = abs(math.log(max(a - 1, 1) / max(b - 1, 1) * sh / sw))
        cands.append((fit + 0.05 * extra, a, b, extra))
    cands.sort()
    out: List[List[Pt]] = []
    for _, a, b, extra in cands[:limit]:
        pts = [(e + sw * i / (a - 1), e + sh * j / (b - 1)) for j in range(b) for i in range(a)]
        if extra:
            cx, cy = w / 2, h / 2
            drop = sorted(range(len(pts)), key=lambda k: (-(abs(pts[k][0] - cx) / sw + abs(pts[k][1] - cy) / sh),
                                                          pts[k][1], pts[k][0]))[:extra]
            pts = [q for k, q in enumerate(pts) if k not in set(drop)]
        out.append(pts)
    return out


def _hole_geom(c: Dict[str, Any], units: str, unit: float) -> Dict[str, Any]:
    """Radii and style for a hole callout (drawing sizes in part units)."""
    tags = c["tags"]
    style, r, r2, d2, rt = "plain", None, 0.0, 0.0, 0.0
    if c["thread"] and not (c["internal"] is False):
        rt = c["thread"] / 2
        r = rt * 0.82
        style = "tap"
    elif c["dias"]:
        r = c["dias"][0] / 2
    if r is None:
        r = unit * 0.035
    if c["cb"]:
        style, r2 = "cbore", max(c["cb"][0] / 2, r * 1.25)
        d2 = c["cb"][1] or r2 * 0.9
    elif c["csk"] is not None:
        style, r2 = "csk", max((c["csk"] or r * 3.8) / 2, r * 1.3)
    elif "port" in tags:
        style, r2 = "port", r * 1.7 if not rt else rt * 1.6
    elif "dowel" in tags:
        style = "dowel"
    elif "pad" in tags:
        # "3X MOUNTING PAD Ø.375": a machined pad face, not a hole (unless it says THRU or gives a depth)
        style = "pad" if (c["thru"] or c["depth"]) else "spot"
    if style == "tap" and "insert" in tags:
        rt = rt * 1.15
    return {"r": r, "r2": r2, "d2": d2, "rt": rt, "style": style, "depth": None if c["thru"] else c["depth"]}


def _place_group(face: Face2D, n: int, r: float, gap: float, insets: Sequence[float],
                 prefer: Optional[List[List[Pt]]] = None) -> Optional[List[Pt]]:
    cands: List[List[Pt]] = list(prefer or [])
    for e in insets:
        cands += _patterns(n, face.w, face.h, e)
    for pts in cands:
        if len(pts) != n:
            continue
        ok = all(face.fits(x, y, r, gap) for x, y in pts)
        if ok:
            # the group's own holes must not collide either
            for i in range(n):
                for j in range(i + 1, n):
                    if (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2 < (2 * r + gap) ** 2:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            return pts
    return None


def _build_prismatic(p: Part) -> None:
    L, W, H = p.d
    shape = p.shape
    mn = min(L, W)
    p.profile = "block"
    p.holes: List[Dict[str, Any]] = []
    p.pockets: List[Dict[str, Any]] = []   # top face: x0 y0 x1 y1 depth rad (z from the top face)
    p.slots: List[Dict[str, Any]] = []     # obround on the top face: cx cy len wid vertical
    p.grooves: List[Dict[str, Any]] = []   # rect: x0 y0 x1 y1 rad wid, or circle: cx cy R wid
    p.open_slots: List[Dict[str, Any]] = []  # across the top, along y: cx wid depth
    p.fins: List[Tuple[float, float]] = []
    p.corner_ch: Dict[int, float] = {}
    p.edge_ch: List[Tuple[str, float]] = []
    p.cross: List[Dict[str, Any]] = []
    p.top = H
    faces: Dict[str, Face2D] = {}
    top = Face2D(L, W)
    faces["top"] = top
    if shape == "bracket" and H > 0.18 * mn:
        p.profile = "bracket"
        t = max(min(0.2 * H, 0.12 * L, 0.22 * W), 0.05 * H)
        p.tb, p.t2 = t, min(t * 1.15, 0.3 * L)
        p.top = t
        top.rects.append((0, 0, p.t2 + 0.25 * t, W))
        up = Face2D(W, H)
        up.rects.append((0, 0, W, t * 1.35))
        faces["upright"] = up
    elif shape == "heatsink":
        p.profile = "heatsink"
        fc = next((c for c in p.cs if "fin" in c["tags"]), None)
        hf = fc["depth"] if fc and fc["depth"] and fc["depth"] < H * 0.92 else H * 0.72
        p.base = max(H - hf, H * 0.12)
        fx0, fx1 = L * 0.1, L * 0.9
        nf = fc["count"] if fc and fc["count"] > 2 else max(5, min(16, int((fx1 - fx0) / max(H * 0.3, mn * 0.05))))
        pitch = (fx1 - fx0) / nf
        tf = (fc["num"] if fc and fc["num"] and fc["num"] < pitch * 0.8 else pitch * 0.38)
        tf = _clamp(tf, pitch * 0.18, pitch * 0.6)
        p.fins = [(fx0 + pitch * (i + 0.5) - tf / 2, fx0 + pitch * (i + 0.5) + tf / 2) for i in range(nf)]
        top.rects.append((fx0 - tf * 0.3, 0, fx1 + tf * 0.3, W))
        faces["bottom"] = Face2D(L, W)
        faces["bottom"].circles = top.circles  # the base is thin: holes from both faces must not meet
        if fc:
            i = p.cs.index(fc)
            fa, fb = p.fins[len(p.fins) // 2 + 1] if len(p.fins) > 2 else p.fins[-1]
            p.anchors[i] = ("front", (fa + fb) / 2, H, 0.0)
    elif shape in ("housing", "enclosure"):
        p.profile = "housing"
        pc = next((c for c in p.cs if "pocket" in c["tags"]), None)
        wall = _clamp(0.085 * mn, 0.035 * mn, 0.2 * mn)
        n_cav = 2 if (pc and pc["count"] == 2) else 1
        if pc and pc["rect"] and len(pc["rect"]) >= 2:
            px, py = pc["rect"][0], pc["rect"][1]
            if n_cav == 2 and px > L * 0.5:
                px = L * 0.44
            if px > L * 0.95 or py > W * 0.95 or px < L * 0.2:
                px, py = L - 2 * wall, W - 2 * wall
        else:
            px, py = L - 2 * wall, W - 2 * wall
            if n_cav == 2:
                px = (L - 3 * wall) / 2
        depth = (pc["rect"][2] if pc and pc["rect"] and len(pc["rect"]) > 2 else pc["depth"] if pc and pc["depth"] else H * 0.84)
        depth = _clamp(depth, H * 0.2, H * 0.94)
        rad = _clamp(pc["radius"] if pc and pc["radius"] else mn * 0.03, 0, min(px, py) * 0.3)
        cavs = []
        if n_cav == 2:
            divider = max(L - 2 * px - 2 * (W - py) / 2, wall * 0.6)
            gapx = (L - 2 * px - divider) / 2
            cavs = [(gapx, (W - py) / 2, gapx + px, (W + py) / 2), (L - gapx - px, (W - py) / 2, L - gapx, (W + py) / 2)]
        else:
            cavs = [((L - px) / 2, (W - py) / 2, (L + px) / 2, (W + py) / 2)]
        for x0, y0, x1, y1 in cavs:
            p.pockets.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "depth": depth, "rad": rad, "cavity": True})
            top.rects.append((x0, y0, x1, y1))
        p.wall = min(cavs[0][0], cavs[0][1])
        p.floor = H - depth
        if pc:
            cx0, cy0, cx1, cy1 = cavs[-1]
            p.anchors[p.cs.index(pc)] = ("top", cx1, cy1 - (cy1 - cy0) * 0.3, 0.0)
        faces["bottom"] = Face2D(L, W)
        fr = Face2D(L, H)
        fr.rects.append((0, H - 0.001, L, H))
        faces["front"] = fr
        faces["right"] = Face2D(W, H)
    elif shape == "cover" or (shape == "bracket"):
        if shape == "cover":
            p.profile = "cover"
            p.lip = H * 0.32
            p.inset = _clamp(0.06 * mn, 0.02 * mn, 0.12 * mn)
    if shape == "manifold":
        faces["front"] = Face2D(L, H)
    gap_base = max(mn * 0.012, 1e-4)

    order = sorted(range(len(p.cs)), key=lambda i: _feature_rank(p.cs[i], shape))
    hole_groups = 0
    port_groups = 0
    for i in order:
        c = p.cs[i]
        tags = c["tags"]
        if i in p.anchors:
            continue
        if "fin" in tags and p.fins:
            continue
        if "pocket" in tags and "hole" in tags and len(c["dias"]) >= 1 and not c["rect"]:
            # round (nest) pocket, possibly stepped: a big blind hole
            r2 = c["dias"][0] / 2
            r1 = c["dias"][1] / 2 if len(c["dias"]) > 1 and c["dias"][1] < c["dias"][0] else r2
            r2 = min(r2, mn * 0.3)
            r1 = min(r1, r2)
            depth = _clamp(c["depth"] or H * 0.5, H * 0.1, p.top * 0.9)
            pts = _place_group(top, 1, r2, gap_base * 3, [mn * 0.3])
            x, y = pts[0] if pts else (L / 2, W / 2)
            hole = {"face": "top", "axis": "z", "a": x, "b": y, "r": r1, "r2": r2 if r2 > r1 * 1.01 else 0.0,
                    "d2": depth * 0.45 if r2 > r1 * 1.01 else 0.0, "style": "nest", "rt": 0.0,
                    "lo": p.top - depth, "hi": p.top, "entry": "+", "grp": i}
            p.holes.append(hole)
            top.circles.append((x, y, r2))
            p.anchors[i] = ("top", x, y, r2)
            continue
        if "pocket" in tags or ("window" in tags and "hole" not in tags):
            if p.profile == "housing":
                continue
            rect = c["rect"]
            pw, ph = (rect[0], rect[1]) if rect and len(rect) >= 2 else (L * 0.46, W * 0.42)
            pw, ph = min(pw, L * 0.8), min(ph, W * 0.8)
            depth = rect[2] if rect and len(rect) > 2 else (c["depth"] or p.top * 0.4)
            depth = _clamp(depth, p.top * 0.05, p.top * 0.9)
            if "CUTOUT" in c["T"] or (c["thru"] and not c["depth"]):
                depth = p.top  # a cutout goes through
            rad = _clamp(c["radius"] or min(pw, ph) * 0.08, 0, min(pw, ph) * 0.45)
            n = 2 if c["count"] == 2 else 1
            placed = False
            for scale in (1.0, 0.85, 0.7, 0.55, 0.4):
                ww, hh = pw * scale / (n if n == 2 else 1), ph * scale
                spots = ([(L / 2, W / 2)] if n == 1 else [(L * 0.3, W / 2), (L * 0.7, W / 2)])
                rects = [(x - ww / 2, y - hh / 2, x + ww / 2, y + hh / 2) for x, y in spots]
                if all(top.rect_fits(*r, gap_base * 2) for r in rects):
                    for r in rects:
                        p.pockets.append({"x0": r[0], "y0": r[1], "x1": r[2], "y1": r[3], "depth": depth, "rad": rad})
                        top.rects.append(r)
                    rr = rects[-1]
                    p.anchors[i] = ("top", rr[2], rr[3] - (rr[3] - rr[1]) * 0.25, 0.0)
                    placed = True
                    break
            if not placed:
                p.anchors[i] = ("top", L * 0.75, W * 0.5, 0.0)
            continue
        if "hole" in tags and ("bore" in tags or "pilot" in tags or "window" in tags) and c["count"] == 1 and c["dias"] \
                and c["dias"][0] > mn * 0.12:
            r = min(c["dias"][0] / 2, mn * 0.38)
            target = "top"
            if p.profile == "housing":
                target = "front"
            face = faces.get(target, top)
            pts = _place_group(face, 1, r, gap_base * 2, [face.unit * 0.3])
            x, y = pts[0] if pts else (face.w / 2, face.h / 2)
            p.holes.append(_hole_on(p, target, x, y, {"r": r, "r2": 0.0, "d2": 0.0, "rt": 0.0, "style": "bore",
                                                      "depth": None if c["thru"] or not c["depth"] else c["depth"]}, i))
            face.circles.append((x, y, r))
            p.anchors[i] = (_face_view(target), x, y, r)
            p.bore = (x, y, r)
            continue
        if "groove" in tags:
            wid = _clamp(c["width"] or mn * 0.02, mn * 0.006, mn * 0.05)
            nest = next((h for h in p.holes if h["style"] in ("nest", "bore")), None)
            if nest and "o-ring" in c["T"].lower() or (nest and "O-RING" in c["T"]):
                R = max(nest["r"], nest["r2"]) + mn * 0.09
                R = min(R, mn * 0.45)
                p.grooves.append({"kind": "circle", "cx": nest["a"], "cy": nest["b"], "R": R, "wid": wid})
                top.circles.append((nest["a"], nest["b"], R + wid))
                p.anchors[i] = ("top", nest["a"] + R * 0.7071, nest["b"] + R * 0.7071, 0.0)
            else:
                if p.profile == "housing" and p.pockets:
                    # close to the cavity, so the lid screws fit between the groove and the outside
                    cav = p.pockets
                    x0 = min(q["x0"] for q in cav) - p.wall * 0.25
                    y0 = min(q["y0"] for q in cav) - p.wall * 0.25
                    x1 = max(q["x1"] for q in cav) + p.wall * 0.25
                    y1 = max(q["y1"] for q in cav) + p.wall * 0.25
                    wid = min(wid, p.wall * 0.3)
                else:
                    e = mn * 0.16
                    x0, y0, x1, y1 = e, e, L - e, W - e
                g = {"kind": "rect", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "rad": mn * 0.05, "wid": wid}
                p.grooves.append(g)
                w2 = wid * 0.55
                top.rects += [(x0 - w2, y0 - w2, x1 + w2, y0 + w2), (x0 - w2, y1 - w2, x1 + w2, y1 + w2),
                              (x0 - w2, y0 + w2, x0 + w2, y1 - w2), (x1 - w2, y0 + w2, x1 + w2, y1 - w2)]
                p.anchors[i] = ("top", x1, (y0 + y1) / 2 + (y1 - y0) * 0.18, 0.0)
            continue
        if "slot" in tags and not c["length"] and (c["depth"] or "KNIFE" in c["T"]):
            wid = _clamp(c["width"] or c["num"] or mn * 0.06, mn * 0.02, L * 0.3)
            depth = _clamp(c["depth"] or H * 0.3, H * 0.05, p.top * 0.85)
            cx = L / 2
            for trial in (0.5, 0.35, 0.65, 0.25, 0.75):
                x = L * trial
                if top.rect_fits(x - wid / 2, 0.0001, x + wid / 2, W - 0.0001, 0) or trial == 0.75:
                    cx = x
                    break
            p.open_slots.append({"cx": cx, "wid": wid, "depth": depth})
            top.rects.append((cx - wid / 2 - gap_base, -1, cx + wid / 2 + gap_base, W + 1))
            p.anchors[i] = ("front", cx + wid / 2, p.top - depth * 0.5, 0.0)
            continue
        if "slot" in tags:
            n = c["count"]
            wid = _clamp(c["width"] or mn * 0.06, mn * 0.02, mn * 0.25)
            ln = _clamp(c["length"] or wid * 3, wid * 1.2, mn * 0.5)
            placed = None
            for scale in (1.0, 0.8, 0.6, 0.45):
                lw, ww = ln * scale, wid * max(scale, 0.7)
                for vertical in ((True, False) if n == 2 else (False, True)):
                    half_a, half_b = ((ww / 2, lw / 2) if vertical else (lw / 2, ww / 2))
                    for e in (mn * 0.14, mn * 0.2, mn * 0.28, mn * 0.36):
                        for pts in _patterns(n, L, W, e + max(half_a, half_b) * 0.5):
                            if len(pts) != n:
                                continue
                            rects = [(x - half_a, y - half_b, x + half_a, y + half_b) for x, y in pts]
                            if all(top.rect_fits(*r, gap_base * 2) for r in rects):
                                placed = (pts, lw, ww, vertical, rects)
                                break
                        if placed:
                            break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                pts, lw, ww, vertical, rects = placed
                for (x, y), r in zip(pts, rects):
                    p.slots.append({"cx": x, "cy": y, "len": lw, "wid": ww, "vertical": vertical,
                                    "depth": None if c["thru"] or not c["depth"] else c["depth"]})
                    top.rects.append(r)
                best = max(range(n), key=lambda k: (pts[k][0], pts[k][1]))
                x, y = pts[best]
                p.anchors[i] = ("top", x + (0 if vertical else lw / 2), y + (lw / 2 if vertical else 0), 0.0)
            else:
                p.anchors[i] = ("top", L * 0.8, W * 0.2, 0.0)
            continue
        if "chamfer" in tags:
            size = c["num"] if c["num"] and c["num"] < mn * 0.25 else mn * 0.04
            if "LEAD" in c["T"] or "EDGE" in c["T"] and c["count"] == 1 and "EDGES" not in c["T"]:
                p.edge_ch.append(("left", size))
                p.anchors[i] = ("front", size * 0.5, p.top - size * 0.5, 0.0)
            else:
                corners = [0, 1] if c["count"] == 2 else [0, 1, 2, 3] if c["count"] != 1 or "EDGES" in c["T"] else [0]
                for k in corners:
                    p.corner_ch[k] = size
                p.anchors[i] = ("top", L - size * 0.5, W - size * 0.5, 0.0)
            continue
        if "cross" in tags and ("hole" in tags or c["thru"]):
            r = min((c["dias"][0] / 2) if c["dias"] else H * 0.06, H * 0.2, W * 0.2)
            p.cross.append({"r": r, "y": W / 2, "z": H * 0.38})
            dep = c["depth"] if c["depth"] and c["depth"] < L * .98 else L
            p.holes.append({"face": "right", "axis": "x", "a": W / 2, "b": H * 0.38, "r": r, "r2": 0.0, "d2": 0.0,
                            "style": "plain", "rt": 0.0, "lo": L - dep, "hi": L, "entry": "+", "grp": i})
            p.anchors[i] = ("right", W / 2, H * 0.38, r)
            continue
        if "hole" in tags:
            g = _hole_geom(c, p.units, mn)
            target = _hole_target(p, c, hole_groups, port_groups, faces)
            if "port" in tags:
                port_groups += 1
            hole_groups += 1
            _place_holes(p, faces, target, c, g, i, gap_base)
            continue
        # anything else just needs a sensible place to point at
        p.anchors[i] = _generic_anchor(p, c, i)

    if not any(h["face"] in ("top", "upright", "front", "right", "bottom") and h["style"] not in ("nest",) for h in p.holes):
        _default_holes(p, faces, gap_base)
    for i in range(len(p.cs)):
        if i not in p.anchors:
            p.anchors[i] = _generic_anchor(p, p.cs[i], i)
    # a leader that ends on the side of a groove or pocket must not end on a hole placed there afterwards
    tops = [(h["a"], h["b"], max(h["r"], h["r2"], h["rt"])) for h in p.holes if h["face"] == "top" and h["axis"] == "z"]
    for i, (view, u, v, r) in list(p.anchors.items()):
        if view != "top" or r > 0 or not tops:
            continue
        near = [(a, b, rh) for a, b, rh in tops if abs(a - u) < rh + mn * 0.05]
        if not any(math.hypot(a - u, b - v) < rh + mn * 0.04 for a, b, rh in near):
            continue
        best = None
        for k in range(1, 30):
            for sgn in (1, -1):
                vv = v + sgn * k * mn * 0.012
                if not (mn * 0.05 < vv < W - mn * 0.05):
                    continue
                if all(math.hypot(a - u, b - vv) >= rh + mn * 0.04 for a, b, rh in near):
                    best = vv
                    break
            if best is not None:
                break
        if best is not None:
            p.anchors[i] = (view, u, best, r)


def _feature_rank(c: Dict[str, Any], shape: str) -> Tuple[int, float]:
    tags = c["tags"]
    if "fin" in tags:
        return (0, 0)
    if "pocket" in tags or "window" in tags and "hole" not in tags:
        return (1, 0)
    if "hole" in tags and ("bore" in tags or "pilot" in tags) and c["count"] == 1:
        return (2, 0)
    if "groove" in tags:
        return (3, 0)
    if "slot" in tags:
        return (4, 0)
    if "chamfer" in tags or "cross" in tags:
        return (5, 0)
    if "hole" in tags and c.get("grid"):
        return (5, 1)
    if "hole" in tags:
        size = (c["cb"][0] if c["cb"] else c["thread"] or (c["dias"][0] if c["dias"] else 0))
        return (6, -size * (1 + math.log(c["count"] + 1) * 0.1))
    return (9, 0)


def _face_view(face: str) -> str:
    return {"top": "top", "bottom": "top", "front": "front", "right": "right", "upright": "right"}.get(face, "top")


def _hole_on(p: Part, face: str, x: float, y: float, g: Dict[str, Any], grp: int) -> Dict[str, Any]:
    L, W, H = p.d
    depth = g.get("depth")
    h = {"face": face, "r": g["r"], "r2": g["r2"], "d2": g["d2"], "style": g["style"], "rt": g["rt"], "grp": grp}
    if face == "top":
        top = p.base if p.profile == "heatsink" else p.top  # a heat sink's holes are in the base, between fins
        dep = top if depth is None else min(depth, top * 0.95)
        if p.profile == "housing" and depth is None:
            dep = min(H * 0.5, top)
        h.update(axis="z", a=x, b=y, lo=top - dep, hi=top, entry="+")
    elif face == "bottom":
        dep = min(depth or H * 0.3, H * 0.6)
        if p.profile == "heatsink":
            dep = min(dep, p.base * 0.85)
        h.update(axis="z", a=x, b=y, lo=0.0, hi=dep, entry="-")
    elif face == "front":
        dep = getattr(p, "wall", None) if p.profile == "housing" else None
        dep = dep if dep else min(depth or W * 0.4, W)
        h.update(axis="y", a=x, b=y, lo=0.0, hi=dep, entry="-")
    elif face == "right":
        dep = getattr(p, "wall", None) if p.profile == "housing" else None
        dep = dep if dep else min(depth or L * 0.3, L)
        h.update(axis="x", a=x, b=y, lo=L - dep, hi=L, entry="+")
    else:  # upright of a bracket: through its thickness
        h.update(axis="x", a=x, b=y, lo=0.0, hi=p.t2, entry="+")
    return h


def _hole_target(p: Part, c: Dict[str, Any], groups: int, ports: int, faces: Dict[str, Face2D]) -> str:
    tags = c["tags"]
    if p.profile == "bracket":
        if tags & {"pattern", "mount"} or "TRIPOD" in c["T"]:
            return "upright"
        return "upright" if groups % 2 == 1 else "top"
    if p.profile == "housing":
        if "bottom" in tags:
            return "bottom"
        if "eachside" in tags:
            return "front+right"
        if "sidewall" in tags:
            return "front"
        return "top"
    if p.profile == "heatsink":
        if tags & {"mount"} or c["thru"]:
            return "top"
        return "bottom"
    if p.shape == "manifold" and "port" in tags and ports == 1:
        return "front"
    return "top"


def _best_effort(face: Face2D, n: int, r: float, gap: float, cands: Sequence[List[Pt]]) -> List[Pt]:
    """When no candidate fits cleanly, take the one that does the least harm: holes in open air (inside a
    cavity or pocket) are the worst, then holes off the face, then holes touching other features."""
    best, best_pts = None, None
    for pts in cands:
        if len(pts) != n:
            continue
        cost = 0.0
        for x, y in pts:
            edge = min(x, y, face.w - x, face.h - y) - r
            if edge < 0:
                cost += 100.0 * (-edge / max(r, 1e-9))
            for x0, y0, x1, y1 in face.rects:
                dx = max(x0 - x, 0.0, x - x1)
                dy = max(y0 - y, 0.0, y - y1)
                if dx * dx + dy * dy < r * r:
                    inside = x0 < x < x1 and y0 < y < y1
                    cost += 1000.0 if inside else 50.0
            for cx, cy, cr in face.circles:
                d = math.hypot(x - cx, y - cy)
                if d < r + cr + gap:
                    cost += 10.0 * (r + cr + gap - d) / max(r + cr, 1e-9)
        if best is None or cost < best:
            best, best_pts = cost, pts
    if best_pts is None:
        best_pts = [(face.w * (k + 1) / (n + 1), face.h / 2) for k in range(n)]
    return list(best_pts)


def _grid_points(face: Face2D, n: int, pitch: float, rr: float, gap: float) -> List[Pt]:
    """Holes on a stated grid pitch, centered on the face, trimmed to n (corner positions go first)."""
    if pitch <= 0:
        return []
    margin = max(rr * 1.5, rr + gap)
    nx = int((face.w - 2 * margin) / pitch + 1e-9) + 1
    ny = int((face.h - 2 * margin) / pitch + 1e-9) + 1
    if nx < 1 or ny < 1 or nx * ny > 4000:
        return []
    while nx * ny > n + 4 and (nx > 1 or ny > 1):
        # a bigger face than the pattern: shrink the grid, keeping its proportions
        if (nx - 1) * ny >= n and (nx >= ny or (nx * (ny - 1)) < n):
            nx -= 1
        elif nx * (ny - 1) >= n:
            ny -= 1
        else:
            break
    x0 = (face.w - (nx - 1) * pitch) / 2
    y0 = (face.h - (ny - 1) * pitch) / 2
    pts = [(x0 + i * pitch, y0 + j * pitch) for j in range(ny) for i in range(nx)]
    face.grid_corners = [(x0, y0), (x0 + (nx - 1) * pitch, y0), (x0 + (nx - 1) * pitch, y0 + (ny - 1) * pitch),
                         (x0, y0 + (ny - 1) * pitch)]
    pts = [q for q in pts if face.fits(q[0], q[1], rr, gap)]
    if len(pts) > n:
        cx, cy = face.w / 2, face.h / 2
        order = sorted(range(len(pts)), key=lambda k: (-(abs(pts[k][0] - cx) + abs(pts[k][1] - cy)), pts[k][1], pts[k][0]))
        drop = set(order[:len(pts) - n])
        pts = [q for k, q in enumerate(pts) if k not in drop]
    return pts


def _place_holes(p: Part, faces: Dict[str, Face2D], target: str, c: Dict[str, Any], g: Dict[str, Any], i: int,
                 gap_base: float) -> None:
    L, W, H = p.d
    targets = target.split("+")
    n_total = c["count"]
    per_face = [n_total] if len(targets) == 1 else [max(1, n_total // 2), max(1, n_total - n_total // 2)]
    anchor = None
    for tgt, n in zip(targets, per_face):
        face = faces.get(tgt)
        if face is None:
            face = faces["top"]
            tgt = "top"
        gg = dict(g)
        rr = max(gg["r"], gg["r2"], gg["rt"])
        # a hole is never bigger than the face it is on: a typo such as "Ø99" must not fill the sheet
        cap = min(face.w, face.h) * (0.42 if n == 1 else 0.3 if n <= 4 else 0.2)
        if rr > cap > 0:
            k = cap / rr
            for key in ("r", "r2", "rt"):
                gg[key] = gg[key] * k
            rr = cap
        unit = face.unit
        prefer: List[List[Pt]] = []
        cx0, cy0 = face.w / 2, face.h / 2
        if tgt == "upright":
            cy0 = (p.tb * 1.35 + H) / 2
        bore = getattr(p, "bore", None)
        if bore and tgt == "top":
            cx0, cy0 = bore[0], bore[1]
        gap = max(gap_base, rr * 0.35)
        if c.get("grid"):
            pts = _grid_points(face, n, c["grid"], rr, gap)
            if len(pts) >= max(1, int(n * 0.9)):
                prefer.append(pts)
                n = len(pts)
        if "CORNER" in c["T"] and n == 4:
            if face.grid_corners:
                prefer.append(list(face.grid_corners))  # "in place of the corner grid holes"
            for e in (rr * 1.8, unit * 0.06, unit * 0.1):
                prefer.append([(e, e), (face.w - e, e), (face.w - e, face.h - e), (e, face.h - e)])
        if c["bc"]:
            R = c["bc"] / 2
            if bore and tgt == "top":
                R = max(R, bore[2] + rr * 1.6)
            R = min(R, min(cx0, face.w - cx0, cy0, face.h - cy0) - rr * 1.2)
            if R > rr:
                # the usual clocking first, then turned half a pitch (clear of slots or pockets in the way)
                for off in ((math.pi / 4, 0.0) if n == 4 else (math.pi / 2, math.pi / 2 + math.pi / n)):
                    prefer.append([(cx0 + R * math.cos(off + 2 * math.pi * k / n), cy0 + R * math.sin(off + 2 * math.pi * k / n))
                                   for k in range(n)])
        if c["pattern"]:
            a, b = c["pattern"]
            if n == 4:
                prefer.append([(cx0 - a / 2, cy0 - b / 2), (cx0 + a / 2, cy0 - b / 2), (cx0 + a / 2, cy0 + b / 2),
                               (cx0 - a / 2, cy0 + b / 2)])
            elif n == 2:
                prefer.append([(cx0 - a / 2, cy0), (cx0 + a / 2, cy0)])
        if tgt == "upright" and n <= 2:
            prefer.append([(face.w * (k + 1) / (n + 1), cy0) for k in range(n)])
        if p.profile == "housing" and tgt == "top":
            # lid screws sit on the walls, outside the gasket groove when there is one
            fracs = (0.3, 0.36, 0.25, 0.42) if p.grooves else (0.5, 0.42, 0.58, 0.35, 0.65)
            for f in fracs:
                prefer += [q for q in _patterns(n, face.w, face.h, p.wall * f) if len(q) == n]
        if p.profile == "cover" and tgt == "top" and n >= 5:
            # many holes on a lid form a bolt pattern around the edge
            for f in (0.1, 0.07, 0.13, 0.17):
                per = _perimeter(n, face.w, face.h, max(unit * f, rr * 1.8), max(unit * f, rr * 1.8))
                if per:
                    prefer.append(per)
        if tgt in ("front", "right") and p.profile == "housing":
            zc = p.floor + (H - p.floor) * 0.5
            prefer.append([(face.w * (k + 1) / (n + 1), zc) for k in range(n)])
        if n == 1:
            # a single hole stays near the middle when something already sits there, instead of taking a spot
            # on the edge where a bolt pattern placed later belongs
            prefer.append([(cx0, cy0)])
            for f in (0.18, 0.26, 0.34):
                d = unit * f
                prefer += [[(cx0 + d, cy0)], [(cx0 - d, cy0)], [(cx0, cy0 + d)], [(cx0, cy0 - d)]]
        insets = [unit * f for f in (0.1, 0.13, 0.17, 0.22, 0.28, 0.34, 0.4)]
        insets = [max(e, rr * 1.7) for e in insets]
        pts = None
        scale = 1.0
        for attempt in range(6):
            pts = _place_group(face, n, rr * scale, gap * scale, insets, prefer)
            if pts:
                break
            scale *= 0.78
        if not pts:
            scale = 0.78 ** 5
            cands = list(prefer)
            for e in insets:
                cands += _patterns(n, face.w, face.h, e)
            pts = _best_effort(face, n, rr * scale, gap * scale, cands)
        if scale < 1.0:
            for key in ("r", "r2", "rt"):
                gg[key] = gg[key] * scale
        for x, y in pts:
            p.holes.append(_hole_on(p, tgt, x, y, gg, i))
            face.circles.append((x, y, max(gg["r"], gg["r2"], gg["rt"])))
        if anchor is None:
            best = max(pts, key=lambda q: (round(q[0], 6), q[1]))
            anchor = (_face_view(tgt), best[0], best[1], max(gg["r"], gg["r2"], gg["rt"]))
    if anchor:
        p.anchors[i] = anchor


def _default_holes(p: Part, faces: Dict[str, Face2D], gap_base: float) -> None:
    L, W, H = p.d
    mn = min(L, W)
    r = _clamp(mn * 0.035, mn * 0.01, H * 1.2 if p.profile == "block" else mn)
    base = {"r": r, "r2": 0.0, "d2": 0.0, "rt": 0.0, "style": "plain", "depth": None}
    tap = {"r": r * 0.8, "r2": 0.0, "d2": 0.0, "rt": r, "style": "tap", "depth": H * 0.4}
    plan: List[Tuple[str, int, Dict[str, Any]]] = []
    shape = p.shape
    if p.profile == "housing":
        plan = [("top", 4 if mn < 3 * r * 20 else 6, dict(tap, depth=H * 0.3))]
    elif p.profile == "bracket":
        plan = [("top", 2, base), ("upright", 2, base)]
    elif p.profile == "heatsink":
        plan = [("top", 4, base)]
    elif shape == "cover":
        plan = [("top", 8 if L > 1.6 * W else 6, dict(base, style="csk", r2=r * 1.9))]
    elif shape == "manifold":
        plan = [("top", 3, dict(tap, r=r * 1.5, rt=r * 1.8, r2=r * 3.0, style="port", depth=H * 0.5)),
                ("top", 4, dict(tap, depth=H * 0.3))]
    elif shape == "fixture":
        plan = [("top", 1, dict(base, r=mn * 0.12, r2=mn * 0.17, d2=H * 0.2, style="nest", depth=H * 0.45)),
                ("top", 4, dict(base, style="cbore", r2=r * 1.7, d2=H * 0.25)), ("top", 2, dict(base, r=r * 0.7, style="dowel"))]
    elif shape == "block":
        plan = [("top", 2, tap), ("top", 2, dict(base, style="cbore", r2=r * 1.7, d2=H * 0.25))]
    else:
        plan = [("top", 4, base), ("top", 2, dict(base, r=r * 0.7, style="dowel"))]
    for tgt, n, g in plan:
        face = faces.get(tgt) or faces["top"]
        rr = max(g["r"], g["r2"], g["rt"])
        insets = [max(face.unit * f, rr * 1.7) for f in (0.1, 0.14, 0.2, 0.28, 0.36)]
        prefer = []
        if p.profile == "housing":
            prefer = _patterns(n, face.w, face.h, p.wall * 0.62)
        pts = _place_group(face, n, rr, max(gap_base, rr * 0.4), insets, prefer)
        if not pts:
            continue
        for x, y in pts:
            h = _hole_on(p, tgt, x, y, g, -1)
            if g["style"] == "nest":
                h["lo"] = p.top - (g["depth"] or H * 0.4)
            p.holes.append(h)
            face.circles.append((x, y, rr))


def _rect_path_point(g: Dict[str, Any], t: float) -> Pt:
    x0, y0, x1, y1 = g["x0"], g["y0"], g["x1"], g["y1"]
    per = 2 * ((x1 - x0) + (y1 - y0))
    s = per * t
    if s < x1 - x0:
        return x0 + s, y0
    s -= x1 - x0
    if s < y1 - y0:
        return x1, y0 + s
    s -= y1 - y0
    if s < x1 - x0:
        return x1 - s, y1
    s -= x1 - x0
    return x0, y1 - s


def _generic_anchor(p: Part, c: Dict[str, Any], i: int) -> Anchor:
    """A plausible target for callouts that are not a placed feature (walls, radii, flatness, angles)."""
    L, W, H = p.d
    tags = c["tags"]
    if p.family == "prismatic":
        cav = p.pockets[-1] if p.pockets else None
        inside = p.profile == "housing" or re.search(r"INSIDE|INTERNAL|POCKET|CAVITY", c["T"]) is not None
        if "radius" in tags and p.profile == "bracket" and not (cav and re.search(r"POCKET|CAVITY", c["T"])) \
                and re.search(r"INSIDE|INTERNAL|BEND|FILLET", c["T"]):
            # a bracket's inside corner is the fillet between the base and the upright, seen in the front view
            rf = min(p.tb, p.t2) * .6
            return ("front", p.t2 + rf * (1 - 0.7071), p.tb + rf * (1 - 0.7071), 0.0)
        if "radius" in tags and cav and inside:
            r = cav["rad"]
            return ("top", cav["x1"] - r * 0.3, cav["y0"] + r * 0.3, 0.0)
        if "wall" in tags and cav:
            return ("top", (cav["x1"] + L) / 2, (cav["y0"] + cav["y1"]) / 2, 0.0)
        if "tongue" in tags and len(p.pockets) > 1:
            return ("top", (p.pockets[0]["x1"] + p.pockets[1]["x0"]) / 2, W * 0.6, 0.0)
        if "gdt" in tags or "angle" in tags:
            return ("front", L * 0.72, p.top, 0.0)
        if "radius" in tags:
            return ("top", L, W * 0.62, 0.0)
        spots = [("top", L, W * 0.35, 0.0), ("front", L * 0.6, p.top, 0.0), ("right", W * 0.7, H, 0.0),
                 ("top", L * 0.55, W, 0.0)]
        return spots[i % len(spots)]
    return ("top", L, W / 2, 0.0)


def _prismatic_mesh(p: Part, m: Mesh) -> None:
    L, W, H = p.d
    holes = []
    for h in p.holes:
        if h["face"] != "top" or h["style"] == "spot":
            continue
        depth = None if h["lo"] <= 1e-9 else h["hi"] - h["lo"]
        style = h["style"]
        r2 = h["r2"] if style in ("cbore", "csk", "nest", "port") else 0.0
        d2 = h["d2"] if style in ("cbore", "nest") else (min(h["r2"] * 0.3, (h["hi"] - h["lo"]) * 0.2) if style == "port" else 0.0)
        r = h["r"] if style != "tap" else (h["r"] + h["rt"]) / 2
        holes.append({"cx": h["a"], "cy": h["b"], "r": r, "r2": r2, "d2": d2, "depth": depth})
    pockets = [(q["x0"], q["y0"], q["x1"], q["y1"], q["depth"]) for q in p.pockets]
    for s in p.slots:
        hw, hl = (s["wid"] / 2, s["len"] / 2) if s["vertical"] else (s["len"] / 2, s["wid"] / 2)
        pockets.append((s["cx"] - hw, s["cy"] - hl, s["cx"] + hw, s["cy"] + hl, s["depth"] or p.top))
    for o in p.open_slots:
        pockets.append((o["cx"] - o["wid"] / 2, 0.0, o["cx"] + o["wid"] / 2, W, o["depth"]))
    if p.profile == "bracket":
        plate_solid(m, L, W, 0.0, p.tb, holes, pockets)
        up = [{"cx": h["a"], "cy": h["b"] - p.tb, "r": h["r"] if h["style"] != "tap" else (h["r"] + h["rt"]) / 2,
               "r2": h["r2"] if h["style"] in ("cbore", "csk") else 0.0, "d2": h["d2"]}
              for h in p.holes if h["face"] == "upright" and h["style"] != "spot"]
        tb, t2 = p.tb, p.t2
        # local plate: x along W (y), y along height (z - tb), z through the thickness (x)
        plate_solid(m, W, H - tb, 0.0, t2, up, (), xf=lambda a, b, c: (t2 - c, a, tb + b))
    elif p.profile == "heatsink":
        plate_solid(m, L, W, 0.0, p.base, holes)
        for x0, x1 in p.fins:
            m.box(x0, 0.0, p.base, x1, W, H)
    elif p.profile == "cover":
        e, lip = p.inset, p.lip
        thru = [q for q in pockets if q[4] >= p.top * 0.999]
        plate_solid(m, L, W, lip, H, [dict(h, depth=None if h["depth"] is None else min(h["depth"], H - lip))
                                      for h in holes],
                    [(a, b, c, d, H - lip if dep >= p.top * 0.999 else min(dep, (H - lip) * 0.9)) for a, b, c, d, dep in pockets])
        # the register lip under it: through holes and cutouts carry on through it, as the views show them
        lw, lh = L - 2 * e, W - 2 * e
        lip_holes = [{"cx": h["cx"] - e, "cy": h["cy"] - e, "r": h["r"]} for h in holes
                     if h["depth"] is None and e + h["r"] * 1.1 < h["cx"] < L - e - h["r"] * 1.1
                     and e + h["r"] * 1.1 < h["cy"] < W - e - h["r"] * 1.1]
        lip_cuts = [(max(a - e, 0.0), max(b - e, 0.0), min(c - e, lw), min(d - e, lh), lip) for a, b, c, d, _ in thru
                    if min(c - e, lw) - max(a - e, 0.0) > 0 and min(d - e, lh) - max(b - e, 0.0) > 0]
        if _COARSE[0] or len(lip_holes) > 48:
            # seen from above (thumbnails, the sheet's isometric) the holes in the lip do not show, and a huge
            # pattern would double an already heavy mesh
            lip_holes = []
        if lip_holes or lip_cuts:
            plate_solid(m, lw, lh, 0.0, lip, lip_holes, lip_cuts, xf=lambda x, y, z: (x + e, y + e, z))
        else:
            m.box(e, e, 0.0, L - e, W - e, lip)
    else:
        plate_solid(m, L, W, 0.0, H, holes, pockets)


# --------------------------------------------------------------------------- #
# Round parts: an outer profile of lathe steps, an inner (bore) profile, and features
# --------------------------------------------------------------------------- #
def _seg(x0: float, x1: float, r: float, name: str, kind: str = "plain", r1: Optional[float] = None) -> Dict[str, Any]:
    return {"x0": x0, "x1": x1, "r": r, "r1": r if r1 is None else r1, "name": name, "kind": kind, "minor": r * 0.86}


def _bore_subject(c: Dict[str, Any]) -> bool:
    """Is the callout about a bore ("Ø1.238-1.240 BORE"), not a tolerance referred to one ("FACE A
    PERPENDICULAR TO BORE", "OD RUNOUT .001 TIR TO BORE")?"""
    if "bore" not in c["tags"]:
        return False
    return any(not re.search(r"\b(?:TO|FROM|WITH)\s+(?:THE\s+)?$", c["T"][:m.start()])
               for m in re.finditer(r"\bBORES?\b|\bID\b", c["T"]))


def _cut_groove(segs: List[Dict[str, Any]], xc: float, c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Cut an external groove (callout c: Ø and width) into the plain step at xc. Returns the groove step."""
    j = next((k for k, s in enumerate(segs) if s["x0"] < xc < s["x1"] and s["kind"] == "plain"), None)
    if j is None:
        return None
    s = segs[j]
    ln = s["x1"] - s["x0"]
    w = c["width"] or (c["rect"][1] if c["rect"] and len(c["rect"]) > 1 else ln * .08)
    w = _clamp(w, ln * .02, ln * .3)
    r = _clamp(c["dias"][0] / 2 if c["dias"] else s["r"] * .85, s["r"] * .5, s["r"] * .97)
    x0 = _clamp(xc - w / 2, s["x0"] + w * .5, s["x1"] - w * 1.5)
    left = dict(s, x1=x0)
    right = dict(s, x0=x0 + w)
    g = _seg(x0, x0 + w, r, "groove", "groove")
    segs[j:j + 1] = [left, g, right]
    return g


def _find(segs: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    return next((s for s in segs if s["name"] == name), None)


def _resize(segs: List[Dict[str, Any]], idx: int, new_len: float) -> None:
    """Change one segment's length, taking the difference from its biggest neighbor (the body)."""
    s = segs[idx]
    old = s["x1"] - s["x0"]
    delta = new_len - old
    others = [j for j in range(len(segs)) if j != idx]
    if not others:
        return
    j = max(others, key=lambda k: segs[k]["x1"] - segs[k]["x0"])
    if segs[j]["x1"] - segs[j]["x0"] - delta < (segs[j]["x1"] - segs[j]["x0"]) * 0.25:
        return
    lengths = [t["x1"] - t["x0"] for t in segs]
    lengths[idx] += delta
    lengths[j] -= delta
    x = segs[0]["x0"]
    for t, ln in zip(segs, lengths):
        t["x0"], t["x1"] = x, x + ln
        x += ln


def _build_round(p: Part) -> None:
    L, D = p.d[0], p.d[1]
    R = D / 2
    shape = p.shape
    rin = p.d[2] / 2 if shape in ROUND3 else 0.0
    segs: List[Dict[str, Any]] = []
    bore: List[Dict[str, Any]] = []  # x0 x1 r r1 kind
    p.keyways: List[Dict[str, Any]] = []
    p.cross: List[Dict[str, Any]] = []
    p.tines: List[Dict[str, Any]] = []
    p.hex_socket: Optional[Dict[str, Any]] = None
    p.bore_key: Optional[Dict[str, Any]] = None
    p.face_holes: List[Dict[str, Any]] = []
    p.oil_groove = False
    cs = p.cs
    used: set = set()

    def take(pred) -> List[int]:
        out = [i for i, c in enumerate(cs) if i not in used and pred(c)]
        return out

    ext_threads = take(lambda c: c["thread"] and _external(c) and "socket" not in c["tags"])
    int_threads = take(lambda c: c["thread"] and not _external(c))
    if shape == "shaft":
        segs = [_seg(0, L * .12, R * .7, "end_l"), _seg(L * .12, L * .26, R * .84, "journal_l"),
                _seg(L * .26, L * .74, R, "body"), _seg(L * .74, L * .88, R * .84, "journal_r"),
                _seg(L * .88, L, R * .7, "end_r")]
        for k, i in enumerate(ext_threads[:2]):
            c = cs[i]
            name = "end_r" if k == 0 else "end_l"
            s = _find(segs, name)
            s["kind"] = "thread"
            s["r"] = s["r1"] = _clamp(c["thread"] / 2, R * .35, R)
            s["minor"] = s["r"] * 0.86
            if c["length"]:
                _resize(segs, segs.index(s), _clamp(c["length"], L * .05, L * .38))
            p.anchors[i] = ("side", (s["x0"] + s["x1"]) / 2, s["r"], 0.0)
            used.add(i)
            if "bothends" in c["tags"] and k == 0:
                t = _find(segs, "end_l")
                t.update(kind="thread", r=s["r"], r1=s["r"], minor=s["minor"])
                if c["length"]:
                    _resize(segs, segs.index(t), _clamp(c["length"], L * .05, L * .38))
        for i in take(lambda c: "journal" in c["tags"]):
            c = cs[i]
            for name in ("journal_l", "journal_r"):
                s = _find(segs, name)
                if c["dias"]:
                    s["r"] = s["r1"] = _clamp(c["dias"][0] / 2, R * .4, R)
            s = _find(segs, "journal_r")
            p.anchors[i] = ("side", (s["x0"] + s["x1"]) / 2, s["r"], 0.0)
            used.add(i)
        for i in take(lambda c: c["tags"] & {"land", "shoulder", "od"} or ("pilot" in c["tags"])):
            c = cs[i]
            if "pilot" in c["tags"] or "PISTON" in c["T"]:
                s = _find(segs, "end_l") if _find(segs, "end_l")["kind"] == "plain" else \
                    (_find(segs, "journal_l") or _find(segs, "body"))
                if c["length"]:
                    _resize(segs, segs.index(s), _clamp(c["length"], L * .05, L * .3))
            else:
                s = _find(segs, "body")
            if c["dias"]:
                s["r"] = s["r1"] = _clamp(c["dias"][0] / 2, R * .45, R)
                if s["name"] == "body" and re.search(r"SHANK|REDUCED", c["T"]) \
                        and not any("journal" in cc["tags"] for cc in cs):
                    # a reduced shank runs between the ends: there are no separate journals on this part, the
                    # body takes their length (so relief grooves land next to the threads, where they belong)
                    for name in ("journal_l", "journal_r"):
                        j = _find(segs, name)
                        if j and j["kind"] == "plain":
                            if name == "journal_l":
                                s["x0"] = j["x0"]
                            else:
                                s["x1"] = j["x1"]
                            segs.remove(j)
            p.anchors[i] = ("side", (s["x0"] + s["x1"]) / 2, s["r"], 0.0)
            used.add(i)
        for i in take(lambda c: "hex" in c["tags"]):
            c = cs[i]
            body = _find(segs, "body")
            ln = _clamp(c["length"] or (body["x1"] - body["x0"]) * .2, L * .03, (body["x1"] - body["x0"]) * .4)
            af = c["num"] if c["num"] and c["num"] < 2 * body["r"] * .97 else 2 * body["r"] * 0.86
            j = segs.index(body)
            flats = _seg(body["x1"] - ln, body["x1"], body["r"], "flats", "flats")
            flats["af"] = af
            body["x1"] -= ln
            segs.insert(j + 1, flats)
            p.anchors[i] = ("side", (flats["x0"] + flats["x1"]) / 2, af / 2, 0.0)
            used.add(i)
        for i in take(lambda c: "tang" in c["tags"]):
            s = _find(segs, "end_l")
            s["kind"] = "flats"
            s["af"] = s["r"] * 1.1
            p.anchors[i] = ("side", (s["x0"] + s["x1"]) / 2, s["af"] / 2, 0.0)
            used.add(i)
        for i in take(lambda c: "relief" in c["tags"]):
            c = cs[i]
            body = _find(segs, "body")
            w = _clamp((c["radius"] or c["num"] or L * .012) * 2, L * .008, L * .04)
            for side in ("l", "r"):
                j = segs.index(body)
                nb = segs[j - 1] if side == "l" else segs[j + 1] if j + 1 < len(segs) else None
                if nb is None:
                    continue
                rr = min(nb["r"], body["r"]) * 0.9
                if side == "l":
                    g = _seg(body["x0"], body["x0"] + w, rr, "relief", "groove")
                    body["x0"] += w
                    segs.insert(j, g)
                else:
                    g = _seg(body["x1"] - w, body["x1"], rr, "relief", "groove")
                    body["x1"] -= w
                    segs.insert(j + 1, g)
            g = next(s for s in reversed(segs) if s["name"] == "relief") if _find(segs, "relief") else body
            p.anchors[i] = ("side", (g["x0"] + g["x1"]) / 2, g["r"], 0.0)
            used.add(i)
        for i in take(lambda c: "groove" in c["tags"] and not re.search(r"INTERNAL|\bID\b|BORE", c["T"])):
            # retaining ring grooves: at the outboard end of each journal
            c = cs[i]
            spots = []
            for name, side in (("journal_r", 1), ("journal_l", -1)):
                j = _find(segs, name)
                if j:
                    w = _clamp(c["width"] or (c["rect"][1] if c["rect"] and len(c["rect"]) > 1 else L * .01),
                               L * .004, (j["x1"] - j["x0"]) * .3)
                    spots.append(j["x1"] - w * 2 if side > 0 else j["x0"] + w * 2)
            if not spots:
                b = _find(segs, "body")
                spots = [b["x1"] - (b["x1"] - b["x0"]) * .1, b["x0"] + (b["x1"] - b["x0"]) * .1]
            gs = [_cut_groove(segs, x, c) for x in spots[:2 if c["count"] >= 2 else 1]]
            gs = [g for g in gs if g]
            if gs:
                p.anchors[i] = ("side", (gs[0]["x0"] + gs[0]["x1"]) / 2, gs[0]["r"], 0.0)
                used.add(i)
        for i in take(lambda c: "cotter" in c["tags"] or ("cross" in c["tags"] and c["dias"])):
            # cross holes for cotter pins go through the ends
            c = cs[i]
            n = 2 if (c["count"] >= 2 or "bothends" in c["tags"]) else 1
            xs = []
            for name, side in (("end_r", 1), ("end_l", -1))[:n]:
                e_ = _find(segs, name)
                r = _clamp(c["dias"][0] / 2 if c["dias"] else e_["r"] * .3, e_["r"] * .08, e_["r"] * .4)
                ln = e_["x1"] - e_["x0"]
                x = e_["x1"] - min(ln * .3, max(r * 2.5, ln * .15)) if side > 0 else e_["x0"] + min(ln * .3, max(r * 2.5, ln * .15))
                p.cross.append({"x": x, "r": r})
                xs.append((x, r))
            p.anchors[i] = ("side", xs[0][0], 0.0, xs[0][1])
            used.add(i)
        # neighboring steps of the same diameter are one step (no shoulder line between them)
        merged: List[Dict[str, Any]] = []
        for sg in segs:
            prv = merged[-1] if merged else None
            if prv and prv["kind"] == sg["kind"] == "plain" and abs(prv["r"] - sg["r"]) < 1e-9 \
                    and abs(prv["r1"] - prv["r"]) < 1e-9 and abs(sg["r1"] - sg["r"]) < 1e-9:
                prv["x1"] = sg["x1"]
                if sg["name"] == "body":
                    prv["name"] = "body"
                continue
            merged.append(sg)
        segs = merged
    elif shape == "pin":
        segs = [_seg(0, L * .9, R, "body"), _seg(L * .9, L, R * .84, "tip")]
        # the head callout ("Ø.750 HEAD X .125 THK"), not a distance measured from the head ("1.938 UNDER HEAD
        # TO HOLE CL", handled with the cross hole below), and only the first one
        for i in take(lambda c: "head" in c["tags"] and not re.search(r"UNDER\s+HEAD|HEAD\s+TO\b|TO\s+HOLE", c["T"]))[:1]:
            c = cs[i]
            hd = c["dias"][0] / 2 if c["dias"] else R * 1.5
            if hd <= R * 1.001:
                # the stated diameter is the head's (the largest): the shank under it is smaller
                hr, br = R, R * .9
            else:
                hr, br = _clamp(hd, R * 1.05, R * 2.2), R
            mt = re.search(r"X\s*" + _NUM + r"\s*(?:THK|THICK|LG|LONG)", c["T"])
            hl = _clamp(float(mt.group(1)) if mt else L * .08, L * .03, L * .3)
            segs = [_seg(0, L - hl, br, "body"), _seg(L - hl, L, hr, "head")]
            p.anchors[i] = ("side", L - hl / 2, hr, 0.0)
            used.add(i)
        for k, i in enumerate(ext_threads[:1]):
            c = cs[i]
            ln = c["length"]
            if not ln:
                mt = re.findall(r"X\s*" + _NUM, c["T"])
                ln = float(mt[-1]) if len(mt) >= 2 else L * .3
            ln = _clamp(ln, L * .1, L * .6)
            tr = _clamp(c["thread"] / 2, R * .3, R)
            segs[0]["x0"] = ln
            segs.insert(0, _seg(0, ln, tr, "thread", "thread"))
            p.anchors[i] = ("side", ln / 2, tr, 0.0)
            used.add(i)
        for i in take(lambda c: "cone" in c["tags"] and _find(segs, "head") is not None):
            head = _find(segs, "head")
            body = _find(segs, "body")
            ang = cs[i]["angles"][0] if cs[i]["angles"] else 30.0
            cl = _clamp((head["r"] - body["r"]) / max(math.tan(math.radians(max(ang, 5))), .1), L * .01, (body["x1"] - body["x0"]) * .3)
            j = segs.index(head)
            cone = _seg(head["x0"] - cl, head["x0"], body["r"], "seat", "cone", r1=head["r"] * .98)
            body["x1"] = head["x0"] - cl
            segs.insert(j, cone)
            p.anchors[i] = ("side", head["x0"] - cl / 2, (body["r"] + head["r"]) / 2, 0.0)
            used.add(i)
        for i in take(lambda c: "hex" in c["tags"]):
            c = cs[i]
            end_r = segs[-1]["r"]
            af = _clamp(c["num"] or end_r, end_r * .3, end_r * 1.4)
            dep = _clamp(c["depth"] or L * .1, L * .02, L * .4)
            p.hex_socket = {"af": af, "depth": dep}
            bore.append({"x0": L - dep, "x1": L, "r": af / 2 / math.cos(math.pi / 6), "r1": None, "kind": "hex"})
            p.anchors[i] = ("end", 0.0, af / 2, 0.0)
            used.add(i)
        for i in take(lambda c: "socket" in c["tags"] or ("bore" in c["tags"] and c["depth"])):
            c = cs[i]
            crimp = "CRIMP" in c["T"]
            r = _clamp(c["dias"][0] / 2 if c["dias"] else R * .4, R * .15, R * .8)
            dep = _clamp(c["depth"] or L * .25, L * .05, L * .45)
            x0, x1 = (0, dep) if crimp else (L - dep, L)
            bore.append({"x0": x0, "x1": x1, "r": r, "r1": None, "kind": "plain"})
            p.anchors[i] = ("side", (x0 + x1) / 2, r, 0.0)
            used.add(i)
        for i in take(lambda c: "slot" in c["tags"]):
            c = cs[i]
            ln = _clamp(c["length"] or L * .15, L * .03, L * .4)
            wd = _clamp(c["width"] or R * .12, R * .05, R * .5)
            p.tines.append({"x0": L - ln, "x1": L, "wid": wd})
            p.anchors[i] = ("side", L - ln * .4, wd / 2, 0.0)
            used.add(i)
        for i in take(lambda c: "groove" in c["tags"] and not re.search(r"INTERNAL|\bID\b|BORE", c["T"])):
            c = cs[i]
            n = 2 if c["count"] >= 2 else 1
            gs = [_cut_groove(segs, L * f, c) for f in ((.28, .72) if n == 2 else (.28,))]
            gs = [g for g in gs if g]
            if gs:
                p.anchors[i] = ("side", (gs[-1]["x0"] + gs[-1]["x1"]) / 2, gs[-1]["r"], 0.0)
                used.add(i)
        for i in take(lambda c: c["tags"] & {"cotter", "inspect"} or ("hole" in c["tags"] and c["thru"] and c["dias"])):
            c = cs[i]
            r = _clamp(c["dias"][0] / 2 if c["dias"] else R * .25, R * .08, R * .45)
            x = L * .1 if _find(segs, "head") else L * .45
            if "inspect" in c["tags"]:
                x = L * .22
            dist = next((cc for cc in cs if "HEAD TO HOLE" in cc["T"] or "TO HOLE CL" in cc["T"]), None)
            if dist and dist["num"] and _find(segs, "head") and dist["num"] < L:
                x = _clamp(_find(segs, "head")["x0"] - dist["num"], r * 1.5, L - r * 1.5)
            p.cross.append({"x": x, "r": r})
            p.anchors[i] = ("side", x, 0.0, r)
            used.add(i)
        for i in take(lambda c: "HEAD TO HOLE" in c["T"] or "TO HOLE CL" in c["T"]):
            x = p.cross[0]["x"] if p.cross else L * .2
            p.anchors[i] = ("side", x, -R * 0.5, 0.0)
            used.add(i)
    elif shape == "disc":
        segs = [_seg(0, L, R, "body")]
        for i in take(lambda c: _bore_subject(c) or ("hole" in c["tags"] and c["count"] == 1 and c["dias"]
                                                     and D * .15 < c["dias"][0] < D * .95 and not re.search(r"\bOD\b", c["T"]))):
            c = cs[i]
            r = _clamp(c["dias"][0] / 2 if c["dias"] else R * .3, R * .08, R * .8)
            bore.append({"x0": 0, "x1": L, "r": r, "r1": None, "kind": "plain"})
            p.anchors[i] = ("end", 0.0, 0.0, r)
            used.add(i)
        for i in take(lambda c: "hole" in c["tags"] and c["count"] > 1):
            c = cs[i]
            rb = bore[0]["r"] if bore else 0.0
            r = _clamp((c["thread"] or (c["dias"][0] if c["dias"] else R * .12)) / 2, R * .03, R * .15)
            bc = _clamp(c["bc"] / 2 if c["bc"] else (R + rb) / 2, rb + r * 2, R - r * 2)
            for k in range(c["count"]):
                a = math.pi / 2 + 2 * math.pi * k / c["count"]
                p.face_holes.append({"u": bc * math.cos(a), "v": bc * math.sin(a), "r": r, "tap": bool(c["thread"])})
            h = max(p.face_holes, key=lambda h: (h["u"], h["v"]))
            p.anchors[i] = ("end", h["u"], h["v"], r)
            used.add(i)
    elif shape == "nozzle":
        knurl = any("knurl" in c["tags"] for c in cs)
        segs = [_seg(0, L * .42, R, "grip", "knurl" if knurl else "hex"), _seg(L * .42, L * .8, R * .8, "thread", "thread"),
                _seg(L * .8, L, R * .7, "tip", "cone", r1=R * .42)]
        if not knurl:
            segs[0]["af"] = 2 * R * math.cos(math.pi / 6)
        for i in ext_threads[:1]:
            s = _find(segs, "thread")
            s["r"] = s["r1"] = _clamp(cs[i]["thread"] / 2, R * .4, R * .95)
            s["minor"] = s["r"] * .86
            _find(segs, "tip")["r"] = s["r"] * .88
            p.anchors[i] = ("side", (s["x0"] + s["x1"]) / 2, s["r"], 0.0)
            used.add(i)
        ro = R * .1
        for i in take(lambda c: "orifice" in c["tags"] or ("hole" in c["tags"] and c["thru"])):
            ro = _clamp(cs[i]["dias"][0] / 2 if cs[i]["dias"] else ro, R * .03, R * .35)
            used.add(i)
            p._orifice_idx = i
        ang = 60.0
        for i in take(lambda c: "cone" in c["tags"]):
            ang = cs[i]["angles"][0] if cs[i]["angles"] else ang
            used.add(i)
            p._cone_idx = i
        re_ = R * .6
        cl = _clamp((re_ - ro) / math.tan(math.radians(_clamp(ang, 20, 150) / 2)), L * .05, L * .5)
        bore = [{"x0": 0, "x1": L * .12, "r": re_, "r1": None, "kind": "plain"},
                {"x0": L * .12, "x1": L * .12 + cl, "r": re_, "r1": ro, "kind": "cone"},
                {"x0": L * .12 + cl, "x1": L, "r": ro, "r1": None, "kind": "plain"}]
        if hasattr(p, "_orifice_idx"):
            p.anchors[p._orifice_idx] = ("section", L * .12 + cl + (L - L * .12 - cl) * .6, ro, 0.0)
        if hasattr(p, "_cone_idx"):
            p.anchors[p._cone_idx] = ("section", L * .12 + cl * .5, (re_ + ro) / 2, 0.0)
        for i in take(lambda c: "knurl" in c["tags"]):
            s = _find(segs, "grip")
            p.anchors[i] = ("side", (s["x0"] + s["x1"]) / 2, s["r"], 0.0)
            used.add(i)
    elif shape == "threaded_fitting":
        tr = R * .72
        if ext_threads:
            tr = _clamp(cs[ext_threads[0]]["thread"] / 2, R * .35, R * .92)
        segs = [_seg(0, L * .36, tr, "thread_l", "thread"), _seg(L * .36, L * .62, R, "hex", "hex"),
                _seg(L * .62, L, tr, "thread_r", "thread")]
        segs[1]["af"] = 2 * R * math.cos(math.pi / 6)
        for s in segs:
            s["minor"] = s["r"] * .86
        rb = tr * .45
        for i in take(lambda c: "bore" in c["tags"] or ("hole" in c["tags"] and not c["thread"])):
            if cs[i]["dias"]:
                rb = _clamp(cs[i]["dias"][0] / 2, tr * .2, tr * .7)
            p.anchors[i] = ("end", 0.0, 0.0, rb)
            used.add(i)
        bore = [{"x0": 0, "x1": L, "r": rb, "r1": None, "kind": "plain"}]
        for k, i in enumerate(ext_threads[:2]):
            s = segs[2] if k == 0 else segs[0]
            p.anchors[i] = ("side", (s["x0"] + s["x1"]) / 2, s["r"], 0.0)
            used.add(i)
        for i in take(lambda c: "hex" in c["tags"]):
            p.anchors[i] = ("side", (segs[1]["x0"] + segs[1]["x1"]) / 2, R, 0.0)
            used.add(i)
    else:  # bushing, spacer, ring
        segs = [_seg(0, L, R, "body")]
        bore = [{"x0": 0, "x1": L, "r": rin, "r1": None, "kind": "plain"}]
        for i in take(lambda c: "flange" in c["tags"]):
            c = cs[i]
            fr = c["rect"][0] / 2 if c["rect"] else (c["dias"][0] / 2 if c["dias"] else R * 1.3)
            fr = _clamp(fr, R * 1.05, R * 2.0)
            ft = c["rect"][1] if c["rect"] and len(c["rect"]) > 1 else L * .12
            ft = _clamp(ft, L * .04, L * .4)
            segs = [_seg(0, L - ft, segs[0]["r"], "body"), _seg(L - ft, L, fr, "flange")]
            p.anchors[i] = ("side", L - ft / 2, fr, 0.0)
            used.add(i)
        for i in take(lambda c: "pilot" in c["tags"] or re.search(r"\bHUB\b", c["T"]) is not None):
            c = cs[i]
            pr = _clamp(c["dias"][0] / 2 if c["dias"] else R * .8, rin * 1.15 if rin else R * .3, R * .97)
            # "HUB Ø2.00 X .250 PROJ": the second number of the pair is how far it stands out
            pl = c["length"] or (c["rect"][1] if c["rect"] and len(c["rect"]) > 1 else L * .15)
            pl = _clamp(pl, L * .05, L * .4)
            body = segs[0]
            segs.insert(0, _seg(0, pl, pr, "pilot"))
            body["x0"] = pl
            p.anchors[i] = ("side", pl / 2, pr, 0.0)
            used.add(i)
        for i in take(lambda c: "groove" in c["tags"] and ("INTERNAL" in c["T"] or "BORE" in c["T"] or " ID" in c["T"]
                                                          or "OIL" in c["T"])):
            c = cs[i]
            if "OIL" in c["T"]:
                p.oil_groove = True
                p.anchors[i] = ("section", L * .5, rin, 0.0)
            else:
                gr = _clamp(c["dias"][0] / 2 if c["dias"] else rin * 1.12, rin * 1.04, (rin + R) / 2)
                gw = _clamp(c["width"] or L * .06, L * .02, L * .2)
                gx = L * .32
                bore.append({"x0": gx, "x1": gx + gw, "r": gr, "r1": None, "kind": "groove"})
                p.anchors[i] = ("section", gx + gw / 2, gr, 0.0)
            used.add(i)
        for i in take(lambda c: "slot" in c["tags"]):
            c = cs[i]
            wd = _clamp(c["width"] or c["num"] or R * .3, R * .08, R * 1.2)
            dp = _clamp(c["depth"] or L * .15, L * .03, L * .45)
            p.tines.append({"x0": L - dp, "x1": L, "wid": wd})
            p.anchors[i] = ("side", L - dp * .4, wd / 2, 0.0)
            used.add(i)
        for i in take(lambda c: "keyway" in c["tags"]):
            c = cs[i]
            wd = _clamp(c["rect"][0] if c["rect"] else (c["width"] or rin * .5), rin * .15, rin * 1.2)
            dp = _clamp(c["rect"][1] if c["rect"] and len(c["rect"]) > 1 else wd * .5, rin * .05, (R - rin) * .6)
            p.bore_key = {"wid": wd, "depth": dp}
            p.anchors[i] = ("end", 0.0, rin + dp, 0.0)
            used.add(i)
        for i in take(lambda c: c["count"] > 1 and "hole" in c["tags"] and "setscrew" not in c["tags"]
                      and (c["bc"] or not c["thread"]) and not re.search(r"RADIAL|CROSS|\bAT 90", c["T"])):
            # a hole pattern through the end face, on a bolt circle in the wall
            c = cs[i]
            ro = max(s_["r"] for s_ in segs)
            wall = max(ro - rin, 1e-9)
            r = _clamp((c["thread"] or (c["dias"][0] if c["dias"] else wall * .4)) / 2, wall * .06, wall * .3)
            bc = _clamp(c["bc"] / 2 if c["bc"] else (ro + rin) / 2, rin + r * 1.5, ro - r * 1.5)
            n = min(c["count"], 24)
            for k in range(n):
                a = math.pi / 2 + 2 * math.pi * k / n
                p.face_holes.append({"u": bc * math.cos(a), "v": bc * math.sin(a), "r": r, "tap": bool(c["thread"])})
            h = max(p.face_holes, key=lambda h: (round(h["v"], 9), h["u"]))  # the top hole: labels sit above
            p.anchors[i] = ("end", h["u"], h["v"], r)
            used.add(i)
        for i in take(lambda c: "setscrew" in c["tags"] or ("thread" in c["tags"] and "hole" in c["tags"])
                      or ("hole" in c["tags"] and c["count"] == 1 and re.search(r"OIL HOLE|CROSS|RADIAL", c["T"]))):
            c = cs[i]
            plain = not c["thread"]
            rt = _clamp(((c["dias"][0] if plain and c["dias"] else c["thread"]) or R * .3) / 2, R * .05,
                        (segs[-1]["x1"] - segs[-1]["x0"]) * .3)
            x = segs[-1]["x0"] + (segs[-1]["x1"] - segs[-1]["x0"]) * .5
            if _find(segs, "flange"):
                x = _find(segs, "body")["x0"] + (_find(segs, "body")["x1"] - _find(segs, "body")["x0"]) * .5
            if "MID" in c["T"]:
                x = L / 2
            p.cross.append({"x": x, "r": rt} if plain else {"x": x, "r": rt * .82, "rt": rt})
            p.anchors[i] = ("side", x, 0.0, rt)
            used.add(i)
        for i in take(lambda c: _bore_subject(c) or " ID" in " " + c["T"]):
            p.anchors[i] = ("section", L * .72, rin, 0.0)
            used.add(i)
        for i in take(lambda c: c["tags"] & {"od", "land"} or ("dias" in c and c["dias"] and "OD" in c["T"])):
            s = _find(segs, "body")
            p.anchors[i] = ("side", s["x0"] + (s["x1"] - s["x0"]) * .35, s["r"], 0.0)
            used.add(i)
    # internal threads at the ends (tapped holes in shafts and pins)
    for k, i in enumerate(int_threads):
        if i in used:
            continue
        c = cs[i]
        rt = _clamp(c["thread"] / 2, R * .12, R * .6)
        if bore and shape not in ("shaft", "pin", "disc"):
            b = max(bore, key=lambda b: b["r"])
            b["kind"] = "thread"
            b["major"] = rt
            p.anchors[i] = ("end", 0.0, 0.0, rt)
            used.add(i)
            continue
        dep = _clamp(c["depth"] or rt * 4, L * .03, L * .35)
        bore.append({"x0": L - dep, "x1": L, "r": rt * .82, "r1": None, "kind": "thread", "major": rt})
        if "bothends" in c["tags"]:
            bore.append({"x0": 0, "x1": dep, "r": rt * .82, "r1": None, "kind": "thread", "major": rt})
        p.anchors[i] = ("end", 0.0, 0.0, rt)
        used.add(i)
    # keyways on the outside
    for i in take(lambda c: "keyway" in c["tags"]):
        c = cs[i]
        target = _find(segs, "body") or max(segs, key=lambda s: s["x1"] - s["x0"])
        el = _find(segs, "end_l")
        ln = c["length"] or (target["x1"] - target["x0"]) * .45
        if el and el["kind"] == "plain" and (el["x1"] - el["x0"]) >= ln * 1.05 and "OPERATOR" in c["T"]:
            target = el
        ln = _clamp(ln, (target["x1"] - target["x0"]) * .2, (target["x1"] - target["x0"]) * .85)
        wd = _clamp(c["rect"][0] if c["rect"] else (c["width"] or c["num"] or target["r"] * .5), target["r"] * .15, target["r"] * 1.1)
        xm = (target["x0"] + target["x1"]) / 2
        p.keyways.append({"x0": xm - ln / 2, "x1": xm + ln / 2, "wid": wd, "r": target["r"]})
        p.anchors[i] = ("side", xm + ln / 2 - wd / 2, wd / 2, 0.0)
        used.add(i)
    # generic targets for everything left
    n_od = 0
    for i, c in enumerate(cs):
        if i in used or i in p.anchors:
            continue
        tags = c["tags"]
        T = c["T"]
        # the subject of a callout is what it names before any datum reference ("OD RUNOUT .001 TIR TO BORE"
        # is about the OD, "FACE A PERPENDICULAR TO BORE" about the face)
        ib = T.find("BORE")
        mo, mf = re.search(r"\bOD\b", T), re.search(r"\bFACE\b", T)
        subj_od = mo is not None and (ib < 0 or mo.start() < ib)
        subj_face = mf is not None and (ib < 0 or mf.start() < ib) and bool(tags & {"gdt"})
        if "chamfer" in tags:
            s = segs[-1]
            p.anchors[i] = ("side", L - min(s["r"] * .06, (s["x1"] - s["x0"]) * .1), s["r"] * .96, 0.0)
        elif "oal" in tags:
            p.anchors[i] = ("side", L, segs[-1]["r"] * .5, 0.0)
        elif subj_face:
            p.anchors[i] = ("side", segs[0]["x0"], segs[0]["r"] * .6, 0.0)
        elif subj_od:
            s = max(segs, key=lambda s: (s["r"], s["x1"] - s["x0"]))
            p.anchors[i] = ("side", s["x0"] + (s["x1"] - s["x0"]) * (.35 + .3 * (n_od % 2)), s["r"], 0.0)
            n_od += 1
        elif tags & {"gdt"} and "BORE" not in c["T"]:
            s = max(segs, key=lambda s: s["r"])
            if "FACE" in c["T"] or "PERPENDICULAR" in c["T"]:
                p.anchors[i] = ("side", segs[0]["x0"], segs[0]["r"] * .6, 0.0)
            else:
                s = _find(segs, "journal_l") or _find(segs, "journal_r") or s
                p.anchors[i] = ("side", (s["x0"] + s["x1"]) / 2, s["r"], 0.0)
        elif "bore" in tags or "hole" in tags and bore:
            b = max(bore, key=lambda b: b["x1"] - b["x0"]) if bore else None
            if b and shape in ROUND3 | {"nozzle", "threaded_fitting"}:
                p.anchors[i] = ("section", (b["x0"] + b["x1"]) / 2, b["r"], 0.0)
            elif b:
                p.anchors[i] = ("end", 0.0, 0.0, b["r"])
            else:
                p.anchors[i] = ("side", L / 2, R, 0.0)
        elif "thread" in tags and segs:
            s = next((s for s in segs if s["kind"] == "thread"), segs[-1])
            p.anchors[i] = ("side", (s["x0"] + s["x1"]) / 2, s["r"], 0.0)
        else:
            s = max(segs, key=lambda s: (s["x1"] - s["x0"]) * (0.2 + s["r"]))
            frac = 0.3 + 0.13 * (i % 4)
            p.anchors[i] = ("side", s["x0"] + (s["x1"] - s["x0"]) * frac, s["r"], 0.0)
    # chamfers on the outer ends and at shoulders
    for s in segs:
        s["chL"] = s["chR"] = 0.0
    if segs:
        segs[0]["chL"] = min(segs[0]["r"] * .08, (segs[0]["x1"] - segs[0]["x0"]) * .2)
        segs[-1]["chR"] = min(segs[-1]["r"] * .08, (segs[-1]["x1"] - segs[-1]["x0"]) * .2)
        for a, b in zip(segs, segs[1:]):
            if a["kind"] in ("cone", "groove") or b["kind"] in ("cone", "groove"):
                continue
            big, side = (a, "chR") if a["r1"] > b["r"] else (b, "chL")
            other = b if big is a else a
            if big["r"] - other["r"] > big["r"] * .06 and big["kind"] != "hex":
                big[side] = min((big["r"] - other["r"]) * .35, (big["x1"] - big["x0"]) * .15)
        for s in segs:
            if s["kind"] == "thread":
                s["chL"] = max(s["chL"], min(s["r"] - s["minor"], (s["x1"] - s["x0"]) * .2)) if s is segs[0] else s["chL"]
                s["chR"] = max(s["chR"], min(s["r"] - s["minor"], (s["x1"] - s["x0"]) * .2)) if s is segs[-1] else s["chR"]
    p.segs = segs
    p.bore = sorted(bore, key=lambda b: b["x0"])
    p.rmax = max([s["r"] for s in segs] + [s["r1"] for s in segs] + [R * .2])
    p.has_section = shape in ROUND3 or shape == "nozzle"


def _outer_profile(p: Part) -> List[Pt]:
    """(x, r) points of the outer silhouette from x = 0 to x = L, with chamfers."""
    pts: List[Pt] = []
    for s in p.segs:
        x0, x1, r0, r1 = s["x0"], s["x1"], s["r"], s["r1"]
        cl, cr = s.get("chL", 0.0), s.get("chR", 0.0)
        if s["kind"] == "hex":
            r0 = r1 = s["r"]
        if cl > 0:
            pts += [(x0, r0 - cl), (x0 + cl, r0)]
        else:
            pts.append((x0, r0))
        if cr > 0:
            pts += [(x1 - cr, r1), (x1, r1 - cr)]
        else:
            pts.append((x1, r1))
    return pts


def _inner_profile(p: Part) -> List[Pt]:
    """(x, r) points of the bore from x = 0 to x = L (r = 0 where the part is solid)."""
    L = p.d[0]
    events = sorted({0.0, L} | {b["x0"] for b in p.bore} | {b["x1"] for b in p.bore})
    pts: List[Pt] = []
    for a, b in zip(events, events[1:]):
        mid = (a + b) / 2
        cand = [q for q in p.bore if q["x0"] - 1e-12 <= mid <= q["x1"] + 1e-12]
        if not cand:
            pts += [(a, 0.0), (b, 0.0)]
            continue
        q = max(cand, key=lambda q: max(q["r"], q["r1"] or 0))
        ra = q["r"] if q["r1"] is None else q["r"] + (q["r1"] - q["r"]) * (a - q["x0"]) / max(q["x1"] - q["x0"], 1e-12)
        rb = q["r"] if q["r1"] is None else q["r"] + (q["r1"] - q["r"]) * (b - q["x0"]) / max(q["x1"] - q["x0"], 1e-12)
        pts += [(a, ra), (b, rb)]
    return pts


def _round_mesh(p: Part, m: Mesh) -> None:
    L = p.d[0]
    outer = _outer_profile(p)
    inner = _inner_profile(p)
    n = 32 if p.rmax * 2 > L * 0.08 else 20
    if _COARSE[0]:
        n = min(n, 24)
    hexes = [s for s in p.segs if s["kind"] in ("hex",)]
    prof: List[Pt] = []
    for x, r in outer:
        if hexes and any(h["x0"] - 1e-9 <= x <= h["x1"] + 1e-9 for h in hexes):
            h = next(h for h in hexes if h["x0"] - 1e-9 <= x <= h["x1"] + 1e-9)
            r = min(r, h.get("af", 2 * h["r"]) / 2 * 0.98)
        prof.append((x, r))
    prof += [(x, r) for x, r in reversed(inner)]
    m.revolve(prof, n)
    for h in hexes:
        af = h.get("af", 2 * h["r"] * math.cos(math.pi / 6))
        rc = af / 2 / math.cos(math.pi / 6)
        rin = max([b["r"] for b in p.bore if b["x0"] < h["x1"] and b["x1"] > h["x0"]] + [0.0])
        hexa = [(rc * math.cos(math.radians(30 + 60 * k)), rc * math.sin(math.radians(30 + 60 * k))) for k in range(6)]
        if rin > 0:
            m.holed_prism(hexa, (0.0, 0.0, rin), h["x0"], h["x1"], 24, xf=lambda a, b, c: (c, a, b))
        else:
            m.prism(hexa, h["x0"], h["x1"], xf=lambda a, b, c: (c, a, b))


# --------------------------------------------------------------------------- #
# Complex (5-axis) parts
# --------------------------------------------------------------------------- #
def _pick(p: Part, targets: Dict[str, Anchor], rules: Sequence[Tuple[Any, str]], defaults: Sequence[str],
          alts: Optional[Dict[str, Sequence[str]]] = None) -> None:
    """Point each callout at a named target: the first rule whose tag set (or predicate) matches wins. A target
    already taken gives way to one of its alternates (alts), so two leaders do not end on the same point."""
    k = 0
    taken: set = set()
    for i, c in enumerate(p.cs):
        if i in p.anchors:
            continue
        chosen = None
        for cond, name in rules:
            ok = cond(c) if callable(cond) else bool(c["tags"] & set(cond))
            if ok and name in targets:
                chosen = name
                break
        if chosen is None:
            chosen = defaults[k % len(defaults)]
            k += 1
        if chosen in taken and alts:
            chosen = next((a for a in alts.get(chosen, ()) if a in targets and a not in taken), chosen)
        taken.add(chosen)
        p.anchors[i] = targets[chosen]


def _count(p: Part, tags: set, default: int, lo: int = 1, hi: int = 24) -> Tuple[int, Optional[Dict[str, Any]]]:
    c = next((c for c in p.cs if c["tags"] & tags), None)
    n = c["count"] if c and c["count"] > 1 else default
    return int(_clamp(n, lo, hi)), c


def _build_complex(p: Part) -> None:
    L, W, H = p.d
    p.flag = f"CONTOURED SURFACES PER 3D MODEL {clean(p.spec.get('part_number') or '')}".strip()
    if p.shape == "structural_fitting":
        _build_fitting(p)
    elif p.shape == "impeller":
        _build_impeller(p)
    elif p.shape == "implant":
        ratio = L / max(W, 1e-9)
        p.variant = "plate" if ratio >= 4 else "cage" if ratio >= 1.8 else "condyle"
        {"plate": _build_boneplate, "cage": _build_cage, "condyle": _build_condyle}[p.variant](p)
    else:
        if L / max(W, 1e-9) >= 2 and H <= W * 0.6:
            p.variant = "blade"
            _build_blade(p)
        elif L >= 3 * max(W, H):
            p.variant = "handle"
            _build_handle(p)
        else:
            p.variant = "sculpt"
            _build_sculpt(p)


def _build_fitting(p: Part) -> None:
    L, W, H = p.d
    mn = min(L, W)
    if L > 3.2 * W:
        p.variant = "spar"
        p.tf = _clamp(H * .14, H * .06, W * .12)
        p.tw = _clamp(W * .1, W * .05, W * .2)
        p.wc = W * .55
        p.tc = _clamp(H * .1, H * .05, H * .2)
        zlo, zhi = p.tf, H - p.tc
        n, c = _count(p, {"lightening"}, 5, 2, 12)
        r = (c["dias"][0] / 2) if c and c["dias"] else (zhi - zlo) * .32
        r = _clamp(r, (zhi - zlo) * .12, (zhi - zlo) * .36)
        r = min(r, L / (n * 2.6))
        p.zh = (zlo + zhi) / 2
        p.light = [L * .1 + (L * .8) * (k + .5) / n for k in range(n)]
        p.rl = r
        p.stiff = [(a + b) / 2 for a, b in zip(p.light, p.light[1:])]
        tr = _clamp(mn * .03, mn * .01, mn * .06)
        dc = next((c for c in p.cs if c["tags"] & {"dowel"} and c["dias"]), None)
        if dc:
            tr = _clamp(dc["dias"][0] / 2, mn * .005, mn * .06)
        p.tool = [(L * .035, W * .22, tr), (L * .965, W * .22, tr)]
        hx = max(p.light)
        targets = {"light": ("front", hx, p.zh, r),
                   "rib": ("front", p.stiff[-1] if p.stiff else L / 2, (zlo + zhi) / 2 + r * .3, 0.0),
                   "tool": ("top", p.tool[1][0], p.tool[1][1], tr),
                   "cap": ("right", W / 2 + p.wc / 2, H - p.tc / 2, 0.0),
                   "flange": ("right", W * .9, p.tf, 0.0),
                   "web": ("front", L * .5, zhi - (zhi - zlo) * .1, 0.0)}
        _pick(p, targets, [({"lightening"}, "light"), ({"rib"}, "rib"), ({"dowel"}, "tool"), ({"hole"}, "tool"),
                           ({"angle", "flange"}, "cap"), ({"wall", "gdt"}, "web")], ["web", "flange", "cap"])
        return
    p.variant = "lug"
    p.tf = _clamp(H * .2, H * .08, W * .25)
    p.tl = _clamp(W * .22, W * .1, W * .34)
    ll = L * .46
    p.xl0, p.xl1 = (L - ll) / 2, (L + ll) / 2
    p.yl0, p.yl1 = W / 2 - p.tl / 2, W / 2 + p.tl / 2
    p.rlug = min(ll / 2, (H - p.tf) * .62)
    p.zc = H - p.rlug
    bc = next((c for c in p.cs if ("lug" in c["tags"] or "bore" in c["tags"]) and c["dias"]), None)
    rb = bc["dias"][0] / 2 if bc else p.rlug * .42
    p.rb = _clamp(rb, p.rlug * .2, p.rlug * .62)
    p.tg = p.tl * .7
    p.hg = min((H - p.tf) * .55, (p.zc - p.tf) * .95)
    p.gus = [(p.xl0, p.xl0 + p.tg), (p.xl1 - p.tg, p.xl1)]
    gap = mn * .05
    rad = next((c["radius"] for c in p.cs if c["radius"] and "radius" in c["tags"]), None) or mn * .05
    py0, py1 = p.yl1 + gap, W - gap * 1.4
    px0, px1 = p.xl0 + p.tg + gap, p.xl1 - p.tg - gap
    p.pk_depth = p.tf * .6
    p.pk_rad = _clamp(rad, 0, min(px1 - px0, py1 - py0) * .4)
    p.pockets = []
    if px1 - px0 > mn * .1 and py1 - py0 > mn * .06:
        p.pockets = [(px0, py0, px1, py1), (px0, W - py1, px1, W - py0)]
    n, hc = _count(p, {"hole", "csk", "insert"} - {"lug"}, 4, 2, 12)
    if hc and ("lug" in hc["tags"] or hc is bc):
        hc = next((c for c in p.cs if c is not bc and c["tags"] & {"hole"}), None)
        n = hc["count"] if hc else 4
    hr = _clamp((hc["dias"][0] / 2 if hc and hc["dias"] else (hc["thread"] / 2 * .82 if hc and hc["thread"] else mn * .035)),
                mn * .012, mn * .06)
    hr2 = 0.0
    if hc and hc["csk"] is not None:
        hr2 = max((hc["csk"] or hr * 3.8) / 2, hr * 1.4)
    elif hc and hc["cb"]:
        hr2 = max(hc["cb"][0] / 2, hr * 1.3)
    p.hole_r, p.hole_r2 = hr, min(hr2, mn * .09)
    per_end = max(1, n // 2)
    xe = min(p.xl0 * .5, L * .12)
    ys = [W / 2] if per_end == 1 else [W * (.18 + .64 * k / (per_end - 1)) for k in range(per_end)]
    p.fholes = [(xe, y) for y in ys] + [(L - xe, y) for y in ys]
    if n % 2 == 1:
        p.fholes = p.fholes[:n]
    # a second hole callout gets its own column of holes, inboard of the first
    p.fholes2: List[Tuple[float, float, float, float]] = []
    hc2 = next((c for c in p.cs if hc is not None and c is not hc and c is not bc and c["tags"] & {"hole", "csk", "insert"}
                and "lug" not in c["tags"]), None)
    if hc2 is not None:
        r_b = _clamp(hc2["dias"][0] / 2 if hc2["dias"] else (hc2["thread"] / 2 * .82 if hc2["thread"] else mn * .035),
                     mn * .012, mn * .06)
        r2_b = max((hc2["csk"] or r_b * 3.8) / 2, r_b * 1.4) if hc2["csk"] is not None else \
            max(hc2["cb"][0] / 2, r_b * 1.3) if hc2["cb"] else 0.0
        r2_b = min(r2_b, mn * .09)
        big = max(p.hole_r, p.hole_r2, r_b, r2_b)
        x2 = xe + max(3.2 * big, (p.xl0 - xe) * .5)
        if x2 + big < p.xl0 - mn * .02:
            n2 = int(_clamp(hc2["count"] if hc2["count"] > 1 else 2, 2, 12))
            per2 = max(1, n2 // 2)
            ys2 = [W / 2] if per2 == 1 else [W * (.18 + .64 * k / (per2 - 1)) for k in range(per2)]
            pts2 = [(x2, y) for y in ys2] + [(L - x2, y) for y in ys2]
            p.fholes2 = [(x, y, r_b, r2_b) for x, y in (pts2[:n2] if n2 % 2 == 1 else pts2)]
    hx, hy = max(p.fholes, key=lambda q: (q[0], q[1]))
    pk = p.pockets[0] if p.pockets else None
    targets = {"bore": ("front", L / 2, p.zc, p.rb),
               "holes": ("top", hx, hy, max(p.hole_r, p.hole_r2)),
               "pocket": ("top", pk[2] - p.pk_rad * .3, pk[3] - p.pk_rad * .3, 0.0) if pk else ("top", L * .6, W * .8, 0.0),
               "flange": ("right", W * .92, p.tf, 0.0),
               "gusset": ("right", (p.yl1 + W * .9) / 2, p.tf + p.hg * .45, 0.0),
               "wall": ("top", pk[2], (pk[1] + pk[3]) / 2, 0.0) if pk else ("top", L * .7, p.yl1, 0.0),
               "lugtop": ("front", L / 2 + p.rlug * .7071, p.zc + p.rlug * .7071, 0.0)}
    if p.fholes2:
        h2 = max(p.fholes2, key=lambda q: (q[0], q[1]))
        targets["holes2"] = ("top", h2[0], h2[1], max(h2[2], h2[3]))
    _pick(p, targets, [(lambda c: "lug" in c["tags"] or (c is bc), "bore"), (lambda c: c is hc2 and bool(p.fholes2), "holes2"),
                       ({"hole", "csk", "insert", "thread"}, "holes"),
                       ({"pocket", "radius"}, "pocket"), ({"flange"}, "flange"), ({"angle"}, "gusset"),
                       ({"wall"}, "wall")], ["lugtop", "gusset", "wall"])


def _impeller_curves(p: Part) -> None:
    H = p.d[2]
    p.z_top = H * .97
    p.z_ex = p.tb + H * .1


def _shroud(p: Part, t: float) -> Pt:
    """(r, z) on the blade tip (shroud) line, t = 0 at the inducer tip, 1 at the exit."""
    a = t * math.pi / 2
    return p.R - (p.R - p.r_ind) * math.cos(a), p.z_top - (p.z_top - p.z_ex) * math.sin(a)


def _hubline(p: Part, t: float) -> Pt:
    a = t * math.pi / 2
    return p.R - (p.R - p.rh) * math.cos(a), p.d[2] - (p.d[2] - p.tb) * math.sin(a)


def _blade_theta(t: float, k: int, n: int) -> float:
    return 2 * math.pi * k / n + 0.95 * t ** 1.4


def _build_impeller(p: Part) -> None:
    L, W, H = p.d
    p.R = min(L, W) / 2
    p.tb = H * .14
    p.rh = p.R * .22
    p.r_ind = p.R * .56
    _impeller_curves(p)
    nm = 7
    ns = 0
    for c in p.cs:
        if "blade" in c["tags"]:
            counts = [int(x) for x in re.findall(r"(\d+)\s*X", c["T"])]
            if "SPLITTER" in c["T"] and len(counts) >= 2:
                nm, ns = counts[0], counts[1]
            elif counts:
                nm = counts[0]
                if "SPLITTER" in c["T"]:
                    ns = counts[0]
    p.nm, p.ns = int(_clamp(nm, 3, 16)), int(_clamp(ns, 0, 16))
    bc = next((c for c in p.cs if "bore" in c["tags"] and c["dias"]), None)
    p.rb = _clamp(bc["dias"][0] / 2 if bc else p.rh * .5, p.rh * .2, p.rh * .8)
    R = p.R
    t = 0.72
    r, z = _shroud(p, t)
    th = _blade_theta(t, 0, p.nm) - math.pi / 2
    targets = {"blade": ("top", R + r * math.cos(th), R + r * math.sin(th), 0.0),
               "bore": ("top", R, R, p.rb),
               "tip": ("front", R + _shroud(p, .85)[0], _shroud(p, .85)[1], 0.0),
               "back": ("front", R * 1.55, 0.0, 0.0),
               "hub": ("front", R + p.rh, H, 0.0)}
    _pick(p, targets, [(lambda c: "TIP" in c["T"] or "PROFILE" in c["T"], "tip"), ({"blade"}, "blade"), ({"bore"}, "bore"),
                       (lambda c: "BACK" in c["T"] or "FLAT" in c["T"] or "balance" in c["tags"], "back")],
          ["hub", "tip", "back"])


def _plate_width(p: Part, x: float) -> float:
    """Scalloped width of a bone plate: full at the holes, waisted between them, tapered at the ends."""
    L, W = p.d[0], p.d[1]
    xs = p.bp_holes
    if not xs:
        return W
    if x <= xs[0]:
        t = x / max(xs[0], 1e-9)
        return W * (.62 + .38 * math.sin(t * math.pi / 2))
    if x >= xs[-1]:
        t = (L - x) / max(L - xs[-1], 1e-9)
        return W * (.62 + .38 * math.sin(t * math.pi / 2))
    for a, b in zip(xs, xs[1:]):
        if a <= x <= b:
            t = (x - a) / max(b - a, 1e-9)
            return W * (1 - .28 * math.sin(t * math.pi) ** 1.5)
    return W


def _plate_z(p: Part, x: float) -> float:
    L = p.d[0]
    u = _clamp((x / L - .3) / .36, 0.0, 1.0)
    return p.bp_amp * (u * u * (3 - 2 * u))


def _build_boneplate(p: Part) -> None:
    L, W, H = p.d
    p.bp_t = _clamp(W * .26, H * .12, H * .6)
    p.bp_amp = max(H - p.bp_t, 0.0)
    n, hc = _count(p, {"screwhole"}, 6, 2, 12)
    hr = W * .16
    if hc:
        mnum = re.search(r"(\d+(?:\.\d+)?)\s*MM", hc["T"])
        if mnum:
            hr = _conv(float(mnum.group(1)) * 1.12, "mm", p.units) / 2
        elif hc["dias"]:
            hr = hc["dias"][0] / 2
    p.bp_r = _clamp(hr, W * .08, W * .3)
    p.bp_holes = [L * .14 + L * .72 * k / max(n - 1, 1) for k in range(n)] if n > 1 else [L / 2]
    kc = next((c for c in p.cs if "kwire" in c["tags"]), None)
    p.bp_kr = _clamp((kc["dias"][0] / 2) if kc and kc["dias"] else W * .06, W * .03, W * .12)
    p.bp_k = [L * .045, L * .955]
    hx = p.bp_holes[-2] if len(p.bp_holes) > 1 else p.bp_holes[0]
    targets = {"hole": ("top", hx, W / 2, p.bp_r), "kwire": ("top", p.bp_k[1], W / 2, p.bp_kr),
               "bend": ("front", L * .48, _plate_z(p, L * .48) + p.bp_t, 0.0),
               "end": ("front", L * .985, p.bp_amp + p.bp_t * .5, 0.0), "edge": ("top", L * .35, W / 2 + _plate_width(p, L * .35) / 2, 0.0)}
    # an end taper is an angle too: test for it before the bend
    _pick(p, targets, [({"screwhole"}, "hole"), ({"kwire"}, "kwire"),
                       (lambda c: re.search(r"\bENDS?\b|TAPER", c["T"]) is not None, "end"), ({"bend", "angle"}, "bend"),
                       ({"hole"}, "hole")],
          ["edge", "bend", "end"])


def _build_cage(p: Part) -> None:
    L, W, H = p.d
    p.win = (L * .34, W * .27, L * .74, W * .73)
    mc = next((c for c in p.cs if "marker" in c["tags"]), None)
    p.mark_r = _clamp((mc["dias"][0] / 2) if mc and mc["dias"] else W * .04, W * .015, W * .08)
    p.marks = [(L * .17, W / 2), (L * .87, W / 2)]
    tc = next((c for c in p.cs if c["thread"]), None)
    p.ins_r = _clamp((tc["thread"] / 2) if tc else H * .16, H * .06, H * .3)
    p.teeth = [L * .3 + L * .62 * k / 5 for k in range(6)]
    p.tooth_h = H * .05  # the body sits inside the teeth, so body and teeth together are the stated height
    wc = next((c for c in p.cs if "window" in c["tags"] and c["dias"]), None)
    if wc:
        r = _clamp(wc["dias"][0] / 2, W * .15, W * .3)
        cx = (p.win[0] + p.win[2]) / 2
        p.win = (cx - r * 1.6, W / 2 - r, cx + r * 1.6, W / 2 + r)
    targets = {"window": ("top", p.win[2], p.win[3] - (p.win[3] - p.win[1]) * .2, 0.0),
               "thread": ("right", W / 2, H / 2, p.ins_r), "nose": ("top", L * .03, W * .62, 0.0),
               "marker": ("top", p.marks[1][0], p.marks[1][1], p.mark_r), "teeth": ("front", p.teeth[3] - L * .01, _cage_z(p, p.teeth[3], H) + p.tooth_h * .5, 0.0)}
    _pick(p, targets, [({"window"}, "window"), ({"thread"}, "thread"), ({"nose"}, "nose"), ({"marker"}, "marker"),
                       ({"hole"}, "marker")], ["teeth", "window", "nose"])


def _cage_top(p: Part, x: float) -> float:
    L, W, H = p.d
    return H * (.84 + .16 * x / L)


def _cage_k(p: Part, x: float) -> float:
    L = p.d[0]
    u = _clamp(x / (L * .24), 0.0, 1.0)
    return .45 + .55 * math.sqrt(u * (2 - u))


def _build_condyle(p: Part) -> None:
    L, W, H = p.d
    t = min(L, H) * .12
    p.fc_t = t
    # inner box cuts (x, z): posterior cut, posterior chamfer, distal cut, anterior chamfer, anterior cut
    xp = min(L, H) * .17
    p.fc_inner = [(xp, H * .52), (xp, H * .3), (xp + L * .12, t), (L * .64, t), (L - t * 1.35, H * .3),
                  (L - t * 1.05, H * .98)]
    p.fc_in_st, p.fc_out_st = _fc_stations(p.fc_inner, t, 40)
    pc = next((c for c in p.cs if "peg" in c["tags"]), None)
    p.peg_r = _clamp((pc["dias"][0] / 2) if pc and pc["dias"] else W * .05, W * .02, W * .08)
    p.pegs = [(L * .44, W * .28), (L * .44, W * .72)]
    p.notch = (0.0, W * .4, L * .4, W * .6)
    targets = {"peg": ("top", p.pegs[1][0], p.pegs[1][1], p.peg_r),
               "chamfer": ("front", (p.fc_inner[1][0] + p.fc_inner[2][0]) / 2, (p.fc_inner[1][1] + p.fc_inner[2][1]) / 2, 0.0),
               "groove": ("right", W * .5, H * .7, 0.0), "notch": ("top", p.notch[2], W * .5, 0.0),
               "outer": ("front", L * .5, 0.0, 0.0)}
    _pick(p, targets, [({"peg"}, "peg"), ({"chamfer"}, "chamfer"), ({"trochlea", "groove"}, "groove"), ({"notch"}, "notch"),
                       ({"hole"}, "peg")], ["outer", "chamfer", "groove"])


def _fc_stations(inner: Sequence[Pt], t: float, n: int) -> Tuple[List[Pt], List[Pt]]:
    """Stations on the inner box cuts and the matching outer surface (offset by t, corners rounded)."""
    ins = _resample(inner, n)
    outs = []
    for k in range(n):
        a, b = ins[max(0, k - 3)], ins[min(n - 1, k + 3)]
        tx, tz = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(tx, tz) or 1.0
        thick = t * (1.25 if k < n * .3 else 1.0)
        outs.append((ins[k][0] + tz / ln * thick, ins[k][1] - tx / ln * thick))
    return ins, outs


def _build_blade(p: Part) -> None:
    L, W, H = p.d
    tc = next((c for c in p.cs if "trunnion" in c["tags"] and c["dias"]), None)
    rt = _clamp((tc["dias"][0] / 2) if tc else min(W, H) * .32, min(W, H) * .15, min(W, H) * .48)
    p.rt = rt
    th = next((c for c in p.cs if c["thread"]), None)
    p.stub = (L * .07, _clamp(th["thread"] / 2, rt * .4, rt * .9)) if th else None
    p.x_tr0 = p.stub[0] if p.stub else 0.0
    p.x_tr1 = L * .19
    p.x_pl1 = L * .23
    tw = next((c for c in p.cs if "angle" in c["tags"] and c["angles"]), None)
    p.twist = _clamp(tw["angles"][0] if tw else 10.0, 0, 35)
    p.chord0, p.chord1 = W * .96, W * .7
    p.tmax = H * .5
    targets = {"trunnion": ("top", (p.x_tr0 + p.x_tr1) / 2, W / 2 + rt, 0.0),
               "twist": ("right", W / 2 + p.chord1 * .45 * math.cos(math.radians(p.twist)), H / 2 + p.chord1 * .45 * math.sin(math.radians(p.twist)), 0.0),
               "edge": ("right", W / 2 - p.chord1 * .45, H / 2, 0.0),
               "thread": ("top", p.x_tr0 * .5 if p.stub else L * .05, W / 2 + (p.stub[1] if p.stub else rt), 0.0),
               "airfoil": ("top", L * .7, W / 2 + _blade_le(p, L * .7), 0.0)}
    _pick(p, targets, [({"trunnion"}, "trunnion"), ({"angle"}, "twist"), ({"edge", "radius"}, "edge"), ({"thread"}, "thread")],
          ["airfoil", "twist", "edge"])


def _blade_chord(p: Part, x: float) -> float:
    L = p.d[0]
    u = _clamp((x - p.x_pl1) / max(L - p.x_pl1, 1e-9), 0, 1)
    return p.chord0 + (p.chord1 - p.chord0) * u


def _blade_twist(p: Part, x: float) -> float:
    L = p.d[0]
    u = _clamp((x - p.x_pl1) / max(L - p.x_pl1, 1e-9), 0, 1)
    return math.radians(p.twist) * u


def _blade_le(p: Part, x: float) -> float:
    """Projected half-width of the planform at x (leading edge side)."""
    c = _blade_chord(p, x)
    return c * .45 * math.cos(_blade_twist(p, x))


def _airfoil(c: float, t: float, n: int = 12) -> List[Pt]:
    """A symmetric airfoil of chord c and thickness t, centered on 40% chord, counterclockwise."""
    xs = [(1 - math.cos(math.pi * k / n)) / 2 for k in range(n + 1)]
    up = []
    for x in xs:
        yt = 5 * (t / max(c, 1e-9)) * (0.2969 * math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4)
        up.append(((x - .4) * c, yt * c))
    pts = up[::-1] + [(x, -y) for x, y in up[1:-1]]
    return pts


def _blade_section(p: Part, x: float) -> List[Pt]:
    """Airfoil at station x in the (y, z) plane of the part."""
    W, H = p.d[1], p.d[2]
    c = _blade_chord(p, x)
    tw = _blade_twist(p, x)
    t = min(p.tmax, c * .16)
    out = []
    for a, b in _airfoil(c, t):
        ca, sa = math.cos(tw), math.sin(tw)
        out.append((W / 2 + a * ca - b * sa, H / 2 + a * sa + b * ca))
    return out


def _handle_ab(p: Part, x: float) -> Tuple[float, float]:
    L, W, H = p.d
    u = x / L
    bulge = math.exp(-((u - .3) / .2) ** 2)
    waist = math.exp(-((u - .66) / .12) ** 2)
    a = W / 2 * (.72 + .28 * bulge - .1 * waist)
    b = H / 2 * (.7 + .3 * bulge - .08 * waist)
    tail = _clamp(u / .05, 0, 1)
    k = .55 + .45 * math.sqrt(tail * (2 - tail))
    return a * k, b * k


def _superellipse(a: float, b: float, n: int = 20, e: float = 2.6) -> List[Pt]:
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n
        c, s = math.cos(t), math.sin(t)
        out.append((a * math.copysign(abs(c) ** (2 / e), c), b * math.copysign(abs(s) ** (2 / e), s)))
    return out


def _build_handle(p: Part) -> None:
    L, W, H = p.d
    bc = next((c for c in p.cs if c["dias"] and ("bore" in c["tags"] or c["thru"])), None)
    a, b = _handle_ab(p, L)
    p.hb_r = _clamp((bc["dias"][0] / 2) if bc else min(a, b) * .35, min(a, b) * .15, min(a, b) * .7)
    p.hb_depth = L * .3
    tc = next((c for c in p.cs if c["thread"]), None)
    p.ht = (_clamp(tc["thread"] / 2, min(W, H) * .08, min(W, H) * .3), _clamp(tc["depth"] or L * .06, L * .02, L * .2)) if tc else None
    pc = next((c for c in p.cs if "port" in c["tags"]), None)
    p.port_r = _clamp((pc["dias"][0] / 2) if pc and pc["dias"] else W * .05, W * .02, W * .1)
    p.port = (L * .8, W / 2)
    nf, _ = _count(p, {"flute"}, 4, 2, 8)
    p.flutes = [L * (.14 + .32 * k / max(nf - 1, 1)) for k in range(nf)]
    targets = {"bore": ("right", W / 2, H / 2, p.hb_r), "port": ("top", p.port[0], p.port[1], p.port_r),
               "thread": ("front", (p.ht[1] if p.ht else L * .05) * .5, H / 2 + (p.ht[0] if p.ht else 0), 0.0),
               "flute": ("top", p.flutes[-1], W / 2 + _handle_ab(p, p.flutes[-1])[0] * .93, 0.0),
               "body": ("front", L * .45, H / 2 + _handle_ab(p, L * .45)[1], 0.0)}
    _pick(p, targets, [(lambda c: c is bc, "bore"), ({"port"}, "port"), ({"thread"}, "thread"), ({"flute"}, "flute")],
          ["body", "flute", "port"])


def _sculpt_a(p: Part, x: float) -> float:
    """Half width of the sculpted body at x: a gentle waist, rounded ends."""
    L, W = p.d[0], p.d[1]
    u = x / L
    a = W / 2 * (.9 + .1 * math.sin(math.pi * u) - .06 * math.exp(-((u - .62) / .16) ** 2))
    e = min(u, 1 - u) / .07
    return a * (.72 + .28 * math.sqrt(max(0.0, min(1.0, e) * (2 - min(1.0, e)))))


def _sculpt_top(p: Part, x: float, y: float) -> float:
    """Top surface height: a dome that blends into the compound-angle face past xa."""
    L, W, H = p.d
    a, b = p.slope
    dome = H * (.93 + .07 * math.sin(math.pi * min(1.0, x / max(p.xa * 1.6, 1e-9))))
    t = _clamp((x - p.xa * .8) / max(p.xa * .4, 1e-9), 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    plane = H - a * max(0.0, x - p.xa) - b * (y - W / 2) * (1.0 if x > p.xa else 0.0) * _clamp((x - p.xa) / max(L - p.xa, 1e-9) * 4, 0, 1)
    return max(H * .18, dome * (1 - t) + min(dome, plane) * t)


def _sculpt_section(p: Part, x: float, n_arc: int = 4) -> List[Pt]:
    """Filleted section (y, z) at x, counterclockwise, the same number of points at every x."""
    W = p.d[1]
    a = _sculpt_a(p, x)
    y0, y1 = W / 2 - a, W / 2 + a
    zl, zr = _sculpt_top(p, x, y0), _sculpt_top(p, x, y1)
    hmin = min(zl, zr)
    rt = min(a * .45, hmin * .35)
    rb = min(a * .12, hmin * .1)
    corners = [(y1 - rb, rb, -90, rb), (y1 - rt, zr - rt, 0, rt), (y0 + rt, zl - rt, 90, rt), (y0 + rb, rb, 180, rb)]
    pts: List[Pt] = []
    for cy, cz, start, r in corners:
        for k in range(n_arc + 1):
            ang = math.radians(start + 90 * k / n_arc)
            pts.append((cy + r * math.cos(ang), cz + r * math.sin(ang)))
    return pts


def _build_sculpt(p: Part) -> None:
    L, W, H = p.d
    angs = next((c["angles"] for c in p.cs if c["angles"] and "angle" in c["tags"]), [22.5, 11.25])
    ax = math.radians(_clamp(angs[0], 5, 40))
    ay = math.radians(_clamp(angs[1] if len(angs) > 1 else angs[0] / 2, 0, 30))
    p.xa = L * .45
    a, b = math.tan(ax), math.tan(ay)
    p.z1 = max(H * .35, H - a * (L - p.xa))
    a = (H - p.z1) / (L - p.xa)
    p.z2 = max(H * .2, p.z1 - b * W)
    b = (p.z1 - p.z2) / W
    p.xb = p.xa - (b / a) * W if a > 0 else p.xa
    if p.xb < L * .08:
        p.xb = L * .08
    p.slope = (a, b)
    hs = [c for c in p.cs if "hole" in c["tags"]]
    normal = [c for c in hs if "NORMAL" in c["T"]]
    other = [c for c in hs if c not in normal]
    p.face_holes = []
    for c in normal:
        r = _clamp((c["thread"] * .41 if c["thread"] else (c["dias"][0] / 2 if c["dias"] else W * .05)), W * .015, W * .08)
        n = c["count"]
        for k in range(n):
            x = p.xa + (L - p.xa) * .55 + (L - p.xa) * .15 * math.cos(2 * math.pi * k / max(n, 1)) if n > 1 else p.xa + (L - p.xa) * .55
            y = W * (.5 + (.28 * math.sin(2 * math.pi * k / n) if n > 1 else 0))
            p.face_holes.append((x, y, r, p.cs.index(c)))
    p.top_holes = []
    for g, c in enumerate(other):
        # each hole callout gets its own column on the flat top, so two patterns never land on each other
        r = _clamp((c["dias"][0] / 2 if c["dias"] else (c["thread"] * .41 if c["thread"] else W * .04)), W * .012, W * .07)
        n = min(c["count"], 8)
        x = p.xb * (.3 + .5 * (g + .5) / len(other)) if len(other) > 1 else p.xb * .5
        for k in range(n):
            p.top_holes.append((x, W * (.2 + .6 * k / (n - 1)) if n > 1 else W * .5, r, p.cs.index(c)))
    targets = {"face": ("front", (p.xa + L) / 2, (H + p.z1) / 2, 0.0), "edge": ("right", W * .5, (p.z1 + p.z2) / 2, 0.0),
               "blend": ("top", p.xa, W * .1, 0.0)}
    for i, c in enumerate(p.cs):
        fh = [h for h in p.face_holes if h[3] == i]
        th = [h for h in p.top_holes if h[3] == i]
        if fh:
            h = max(fh, key=lambda h: h[0])
            p.anchors[i] = ("top", h[0], h[1], h[2])
        elif th:
            h = max(th, key=lambda h: h[1])
            p.anchors[i] = ("top", h[0], h[1], h[2])
    _pick(p, targets, [({"angle"}, "face")], ["face", "edge", "blend"])


def _complex_mesh(p: Part, m: Mesh) -> None:
    L, W, H = p.d
    if p.shape == "structural_fitting" and p.variant == "spar":
        tf, tw, tc, wc = p.tf, p.tw, p.tc, p.wc
        plate_solid(m, L, W, 0.0, tf, [{"cx": x, "cy": y, "r": r} for x, y, r in p.tool])
        zlo, zhi = tf, H - tc
        holes = [{"cx": x, "cy": p.zh - zlo, "r": p.rl} for x in p.light]
        plate_solid(m, L, zhi - zlo, 0.0, tw, holes, xf=lambda a, b, c: (a, W / 2 - tw / 2 + c, zlo + b))
        m.box(0, W / 2 - wc / 2, zhi, L, W / 2 + wc / 2, H)
        for x in p.stiff:
            m.box(x - tw / 2, W / 2 - wc / 2, zlo, x + tw / 2, W / 2 - tw / 2, zhi)
            m.box(x - tw / 2, W / 2 + tw / 2, zlo, x + tw / 2, W / 2 + wc / 2, zhi)
        return
    if p.shape == "structural_fitting":
        holes = [{"cx": x, "cy": y, "r": p.hole_r, "r2": p.hole_r2, "d2": 0.0} for x, y in p.fholes]
        holes += [{"cx": x, "cy": y, "r": r, "r2": r2, "d2": 0.0} for x, y, r, r2 in p.fholes2]
        pockets = [(a, b, c, d, p.pk_depth) for a, b, c, d in p.pockets]
        plate_solid(m, L, W, 0.0, p.tf, holes, pockets)
        # the lug: a trapezoid with a round top around the bore, as in the front view (convex)
        out = [(p.xl0, p.tf), (p.xl1, p.tf)]
        out += [(L / 2 + p.rlug * math.cos(math.pi * k / 12), p.zc + p.rlug * math.sin(math.pi * k / 12)) for k in range(13)]
        m.holed_prism(out, (L / 2, p.zc, p.rb), p.yl0, p.yl1, 28, xf=lambda a, b, c: (a, c, b))
        for x0, x1 in p.gus:
            for side in (-1, 1):
                y0 = p.yl1 if side > 0 else p.yl0
                y1 = W - W * .1 if side > 0 else W * .1
                tri = [(y0, p.tf), (y1, p.tf), (y0, p.tf + p.hg)]
                m.prism(tri, x0, x1, xf=lambda a, b, c: (c, a, b))
        return
    if p.shape == "impeller":
        R = p.R
        prof = [(0.0, p.rb), (0.0, R), (p.tb * .75, R)]
        for k in range(11):
            r, z = _hubline(p, 1 - k / 10)
            prof.append((z, r))
        prof.append((H, p.rb))
        m.revolve(prof, 24 if _COARSE[0] else 36, xf=lambda a, b, c: (R + b, R + c, a))
        for k in range(p.nm + p.ns):
            main = k < p.nm
            idx = k if main else k - p.nm
            n = p.nm if main else p.ns
            off = 0.0 if main else math.pi / p.nm
            t0 = 0.0 if main else 0.42
            secs = []
            steps = (5 if main else 3) if _COARSE[0] else (8 if main else 5)
            thick = R * .035
            for j in range(steps + 1):
                t = t0 + (1 - t0) * j / steps
                rs, zs = _shroud(p, t)
                rh, zh = _hubline(p, t)
                rh = min(rh, rs)
                th = _blade_theta(t, idx, n) + off
                dth_s, dth_h = thick / 2 / max(rs, 1e-9), thick / 2 / max(rh, 1e-9)
                zh2 = min(zh, zs) - H * .012
                sec = [(R + rh * math.cos(th - dth_h), R + rh * math.sin(th - dth_h), zh2),
                       (R + rh * math.cos(th + dth_h), R + rh * math.sin(th + dth_h), zh2),
                       (R + rs * math.cos(th + dth_s), R + rs * math.sin(th + dth_s), zs),
                       (R + rs * math.cos(th - dth_s), R + rs * math.sin(th - dth_s), zs)]
                secs.append(sec)
            m.loft(secs)
        return
    if p.shape == "implant" and p.variant == "plate":
        holes = [{"cx": x, "cy": W / 2, "r": p.bp_r} for x in p.bp_holes] + [{"cx": x, "cy": W / 2, "r": p.bp_kr} for x in p.bp_k]
        nx = 24 if _COARSE[0] else 40
        xc = [L * k / nx for k in range(1, nx)]
        xc += [(a + b) / 2 for a, b in zip(p.bp_holes, p.bp_holes[1:])]
        t = p.bp_t
        endt = next((c for c in p.cs if "TAPER" in c["T"] and c["num"]), None)

        def tfac(x: float) -> float:
            u = min(x, L - x) / (L * .08)
            return (.55 + .45 * _clamp(u, 0, 1)) if endt else 1.0

        plate_solid(m, L, W, 0.0, t, holes, xcuts=xc,
                    xf=lambda x, y, z: (x, W / 2 + (y - W / 2) * _plate_width(p, x) / W, z * tfac(x) + _plate_z(p, x)),
                    merge_cells=False)
        return
    if p.shape == "implant" and p.variant == "cage":
        x0, y0, x1, y1 = p.win
        holes = [{"cx": x, "cy": y, "r": p.mark_r} for x, y in p.marks]
        xc = [L * f for f in (.03, .06, .1, .14, .19, .24)]

        def xf(x: float, y: float, z: float) -> Tuple[float, float, float]:
            return x, W / 2 + (y - W / 2) * _cage_k(p, x), _cage_z(p, x, z)

        plate_solid(m, L, W, 0.0, H, holes, [(x0, y0, x1, y1, H)], xcuts=xc, xf=xf, merge_cells=False)
        for x in p.teeth:
            tw = L * .035
            k = _cage_k(p, x)
            w0 = W / 2 - W / 2 * k * .92
            spans = [(w0, W - w0)]
            if x0 - tw < x < x1 + tw * .4:  # teeth stop at the graft window
                spans = [(w0, W / 2 + (y0 - W / 2) * k - W * .01), (W / 2 + (y1 - W / 2) * k + W * .01, W - w0)]
            for zb, sgn in ((_cage_z(p, x, H) - H * .004, 1), (_cage_z(p, x, 0) + H * .004, -1)):
                tri = [(x - tw, zb), (x + tw * .4, zb), (x + tw * .4, zb + sgn * (p.tooth_h + H * .004))]
                for ya, yb in spans:
                    if yb - ya > W * .02:
                        m.prism(tri, ya, yb, xf=lambda a, b, c: (a, c, b))
        return
    if p.shape == "implant":
        ip, op = p.fc_in_st, p.fc_out_st
        N = len(ip)
        k1 = int(N * .42)
        for y0, y1, a, b in ((0.0, W * .4, 0, k1 + 1), (W * .6, W, 0, k1 + 1), (0.0, W, k1, N)):
            secs = []
            for k in range(a, b):
                (xi, zi), (xo, zo) = ip[k], op[k]
                secs.append([(xo, y0, zo), (xo, y1, zo), (xi, y1, zi), (xi, y0, zi)])
            m.loft(secs)
        for x, y in p.pegs:
            m.cylinder_z(p.fc_t * .9, p.fc_t + H * .22, p.peg_r, x, y, 16)
        return
    if p.variant == "blade":
        if p.stub:
            m.revolve([(0, 0), (p.stub[0], 0), (p.stub[0], p.stub[1]), (0, p.stub[1] * .9)], 20,
                      xf=lambda a, b, c: (a, W / 2 + b, H / 2 + c))
        m.revolve([(p.x_tr0, 0), (p.x_tr1, 0), (p.x_tr1, p.rt), (p.x_tr0, p.rt)], 24, xf=lambda a, b, c: (a, W / 2 + b, H / 2 + c))
        m.box(p.x_tr1, W * .04, H * .02, p.x_pl1, W * .96, H * .98)
        secs = []
        for k in range(9):
            x = p.x_pl1 + (L - p.x_pl1) * k / 8
            secs.append([(x, y, z) for y, z in _blade_section(p, x)])
        m.loft(secs)
        return
    if p.variant == "handle":
        secs = []
        for k in range(17):
            x = L * k / 16
            a, b = _handle_ab(p, x)
            secs.append([(x, W / 2 + u, H / 2 + v) for u, v in _superellipse(a, b)])
        m.loft(secs)
        return
    # sculpted body: filleted sections lofted along x, the top blending into the compound-angle face
    secs = []
    for k in range(19):
        x = L * k / 18
        secs.append([(x, y, z) for y, z in _sculpt_section(p, x)])
    m.loft(secs)


def _resample(pts: Sequence[Pt], n: int) -> List[Pt]:
    """n points evenly spaced by arc length along a polyline."""
    seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
    total = sum(seg) or 1.0
    out = []
    j, acc = 0, 0.0
    for k in range(n):
        s = total * k / (n - 1)
        while j < len(seg) - 1 and acc + seg[j] < s - 1e-12:
            acc += seg[j]
            j += 1
        ln = seg[j] if seg else 0.0
        t = _clamp((s - acc) / ln, 0.0, 1.0) if ln > 0 else 0.0
        a, b = pts[j], pts[j + 1] if len(pts) > 1 else pts[j]
        out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


# --------------------------------------------------------------------------- #
# Other processes: weldment, sheet metal, casting, assembly
# --------------------------------------------------------------------------- #
def _build_other(p: Part) -> None:
    L, W, H = p.d
    mn = min(L, W, H)
    u = p.units
    if p.shape == "weldment":
        a = _clamp(mn * .07, mn * .03, mn * .16)
        wall = a * .09
        # the tube size and wall come from the material when it names them ("2 X 2 X .188 SQ TUBE")
        mt = re.search(r"(\d*\.?\d+)\s*X\s*(\d*\.?\d+)\s*X\s*(\d*\.?\d+)\s*(?:SQ(?:UARE)?\.?|RECT\w*\.?)?\s*(?:TUBE|TUBING|HSS)",
                       clean(p.spec.get("material") or "").upper())
        if mt:
            ta, tt = float(mt.group(1)), float(mt.group(3))
            if mn * .02 <= ta <= mn * .25 and 0 < tt < ta * .3:
                a, wall = ta, tt
        p.tube, p.twall = a, wall
        # the weld symbol shows the fillet the callouts ask for, else a fillet the size of the tube wall
        wc = next((c for c in p.cs if ("weld" in c["tags"] or "FILLET" in c["T"]) and c["num"]), None)
        p.weld = wc["num"] if wc and 0 < wc["num"] < a else wall
        p.pad = a * .18
        p.zr = H * .24
        p.pad_w = a * 1.6
        p.pad_hole = a * .22
        p.flag = "ALL WELDS PER AWS D1.1. GRIND WELDS FLUSH WHERE NOTED."
        pw = p.pad_w
        targets = {"weld": ("front", a, H - a, 0.0), "tube": ("front", a / 2, H * .6, 0.0),
                   "pad": ("front", L - pw / 2, p.pad * .5, 0.0), "pad2": ("right", W - pw / 2, p.pad * .5, 0.0),
                   "pad3": ("right", pw / 2, p.pad * .5, 0.0), "rail": ("front", L * .5, p.zr + a, 0.0),
                   "top": ("front", L * .66, H, 0.0), "top2": ("right", W * .5, H, 0.0)}
        _pick(p, targets, [(lambda c: "weld" in c["tags"] or "FILLET" in c["T"], "weld"),
                           (lambda c: re.search(r"MACHINED|FLATNESS|\bMOUNT", c["T"]) is not None
                            and re.search(r"FOOT|FEET|LEVEL", c["T"]) is None, "top"),
                           ({"tube"}, "tube"), ({"plate", "pad", "hole"}, "pad")], ["rail", "top", "tube"],
              alts={"pad": ("pad2", "pad3"), "top": ("top2",)})
        return
    if p.shape == "sheet_metal":
        t = _clamp(mn * .04, (0.8 if u == "mm" else .03), (4.0 if u == "mm" else .19))
        t = min(t, mn * .2)  # the stock minimum on a tiny part would fold the profile through itself
        tm = _sheet_thickness(clean(p.spec.get("material") or ""), u)  # "16 GA (.060)"
        if tm and mn * .002 < tm < mn * .3:
            t = tm
        tc = next((c for c in p.cs if "thk" in c["tags"] and c["num"]), None)
        if tc and tc["num"] < mn * .3:
            t = tc["num"]
        p.t = t
        p.ri = t
        ba = math.pi / 2 * (p.ri + .44 * t)
        fl = max(H - t - p.ri, t)
        web = max(W - 2 * (t + p.ri), t)
        p.flat_w = web + 2 * fl + 2 * ba
        p.vb = (fl + ba / 2, p.flat_w - fl - ba / 2)
        p.ba = ba
        hs = [c for c in p.cs if "hole" in c["tags"]]
        p.sm_holes = []
        if hs:
            for c in hs:
                r = _clamp((c["dias"][0] / 2) if c["dias"] else web * .06, web * .02, web * .15)
                n = c["count"]
                for k in range(n):
                    p.sm_holes.append((L * (.12 + .76 * k / max(n - 1, 1)) if n > 1 else L / 2, p.flat_w / 2, r, p.cs.index(c)))
        else:
            r = web * .07
            p.sm_holes = [(L * .15, p.flat_w / 2, r, -1), (L * .85, p.flat_w / 2, r, -1)]
        hx = max(p.sm_holes, key=lambda h: h[0])
        # the bend: the inside of the far bend, which lies under the label column (a leader to the near bend
        # would drop straight down its wall)
        targets = {"hole": ("flat", hx[0], hx[1], hx[2]), "bend": ("right", W - t - p.ri * .293, t + p.ri * .293, 0.0),
                   "flange": ("right", W - t / 2, H, 0.0), "flat": ("flat", L * .5, p.flat_w, 0.0)}
        _pick(p, targets, [({"hole"}, "hole"), ({"bend", "radius"}, "bend"), ({"flange"}, "flange")], ["flat", "bend", "flange"])
        return
    if p.shape == "casting":
        mnl = min(L, W)
        p.rc = mnl * .15
        p.tb = H * .28
        p.rbo = mnl * .27
        p.rf = mnl * .05
        p.rbore = p.rbo * .45
        p.draft = math.radians(2.5)
        p.ch = [(mnl * .13, mnl * .13), (L - mnl * .13, mnl * .13), (L - mnl * .13, W - mnl * .13), (mnl * .13, W - mnl * .13)]
        p.ch_r = mnl * .035
        p.rib_t = mnl * .06
        stock = ".06" if u == "in" else "1.5"
        fil = "R.12" if u == "in" else "R3"
        p.flag = f"ADD {stock} MACHINING STOCK TO SURFACES MARKED. UNTOLERANCED FILLETS {fil}, DRAFT 2° MAX."
        targets = {"bore": ("top", L / 2, W / 2, p.rbore), "hole": ("top", p.ch[2][0], p.ch[2][1], p.ch_r * 1.8),
                   "fillet": ("front", L / 2 + p.rbo + p.rf * .3, p.tb + p.rf * .3, 0.0),
                   "draft": ("front", L / 2 + p.rbo * .96, (p.tb + H) / 2, 0.0), "rib": ("front", L / 2 + p.rbo * 1.5, p.tb + (H - p.tb) * .15, 0.0),
                   "top": ("front", L / 2, H, 0.0)}
        _pick(p, targets, [({"bore"}, "bore"), ({"hole", "thread"}, "hole"), ({"radius"}, "fillet"), ({"angle"}, "draft"),
                           ({"rib"}, "rib")], ["top", "fillet", "draft"])
        return
    # assembly: the pulley reaches the stated height and clears the base plate
    p.tb = H * .16
    p.zc = p.tb + (H - p.tb) * .52
    p.rp = min(H - p.zc, W * .44)
    p.rs = max(p.rp * .2, W * .04)
    p.sw = (W * .22, W * .78)
    p.sup = [(L * .12, L * .22), (L * .78, L * .88)]
    # the supports' round tops stay under the stated height (flattened when the part is wide and low), or
    # reach it when a narrow pulley does not; the shaft fits inside them and inside the pulley
    p.sup_top = H if p.zc + p.rp < H * .999 else min(p.zc + (p.sw[1] - p.sw[0]) / 2, H)
    p.arch = max(p.sup_top - p.zc, 1e-9)
    p.rs = min(p.rs, p.arch * .7, p.rp * .6)
    p.pul = (L * .47, L * .57)
    p.bolts = [(x, y) for x0, x1 in p.sup for x in ((x0 + x1) / 2,) for y in (W * .12, W * .88)]
    p.base_holes = [(L * .05, W * .12), (L * .95, W * .12), (L * .95, W * .88), (L * .05, W * .88)]
    pn = clean(p.spec.get("part_number") or "ASSY")
    p.parts_list = [("1", f"{pn}-1", "BASE PLATE", 1), ("2", f"{pn}-2", "BEARING SUPPORT", 2),
                    ("3", f"{pn}-3", "SHAFT", 1), ("4", f"{pn}-4", "PULLEY", 1),
                    ("5", "SHCS 1/4-20 X .75" if u == "in" else "SHCS M6 X 20", "SOCKET HEAD CAP SCREW", 8)]
    p.balloons = [(1, ("top", L * .32, W * .1, 0.0)), (2, ("top", p.sup[1][1], W * .7, 0.0)),
                  (3, ("top", L * .93, W / 2, 0.0)), (4, ("top", p.pul[1], W / 2 + p.rp * .6, 0.0)),
                  (5, ("top", p.bolts[3][0], p.bolts[3][1], 0.0))]
    targets = {"hole": ("top", p.base_holes[2][0], p.base_holes[2][1], W * .03), "shaft": ("front", L * .7, p.zc + p.rs, 0.0),
               "pulley": ("front", p.pul[1], p.zc + p.rp * .5, 0.0), "support": ("front", p.sup[1][1], p.sup_top * .8, 0.0),
               "plate": ("front", L * .3, p.tb, 0.0)}
    _pick(p, targets, [({"hole"}, "hole"), (lambda c: "SHAFT" in c["T"], "shaft"), (lambda c: "PULLEY" in c["T"], "pulley"),
                       ({"journal", "flange"}, "support"), ({"plate"}, "plate")], ["plate", "support", "shaft"])


_GAUGE_IN = {7: .1793, 8: .1644, 10: .1345, 11: .1196, 12: .1046, 13: .0897, 14: .0747, 16: .0598, 18: .0478, 20: .0359,
             22: .0299, 24: .0239, 26: .0179, 28: .0149}  # sheet steel gauges


def _sheet_thickness(material: str, units: str) -> Optional[float]:
    """Sheet thickness named by a material callout: "(.060)", ".060 THK" or "16 GA", in the part's units."""
    T = material.upper()
    m = re.search(r"\(\s*(\d*\.\d+)\s*(MM|IN\.?|\")?\s*\)", T) or \
        re.search(r"(\d*\.\d+)\s*(MM|IN\.?|\")?\s*(?:THK|THICK)", T)
    if m:
        unit = "mm" if (m.group(2) or "").startswith("MM") else ("in" if m.group(2) else units)
        return _conv(float(m.group(1)), unit, units)
    g = re.search(r"\b(\d{1,2})\s*(?:GA|GAUGE)\b", T)
    if g and int(g.group(1)) in _GAUGE_IN:
        return _conv(_GAUGE_IN[int(g.group(1))], "in", units)
    return None


def _sheet_profile(p: Part, n_arc: int = 5) -> List[Tuple[Pt, Pt]]:
    """Stations along the formed U channel cross-section: (outer, inner) points in (y, z)."""
    W, H = p.d[1], p.d[2]
    t, ri = p.t, p.ri
    ro = ri + t
    st: List[Tuple[Pt, Pt]] = [((0.0, H), (t, H))]
    cz = ro
    for k in range(n_arc + 1):
        a = math.pi + (math.pi / 2) * k / n_arc
        cy = ro
        st.append(((cy + ro * math.cos(a), cz + ro * math.sin(a)), (cy + ri * math.cos(a), cz + ri * math.sin(a))))
    for k in range(n_arc + 1):
        a = 1.5 * math.pi + (math.pi / 2) * k / n_arc
        cy = W - ro
        st.append(((cy + ro * math.cos(a), cz + ro * math.sin(a)), (cy + ri * math.cos(a), cz + ri * math.sin(a))))
    st.append(((W, H), (W - t, H)))
    return st


def _other_mesh(p: Part, m: Mesh) -> None:
    L, W, H = p.d
    if p.shape == "weldment":
        a, t = p.tube, p.twall
        pad = p.pad
        pw = p.pad_w
        for x in (a / 2, L - a / 2):
            for y in (a / 2, W - a / 2):
                m.rect_tube(H - a - pad, a, a, t, xf=lambda u, v, w, x=x, y=y: (x + v, y + w, pad + u))
                # foot plates flush with the outside of the frame, as the views draw them
                px, py = (0.0 if x < L / 2 else L - pw), (0.0 if y < W / 2 else W - pw)
                m.box(px, py, 0, px + pw, py + pw, pad)
        for y in (a / 2, W - a / 2):
            m.rect_tube(L, a, a, t, xf=lambda u, v, w, y=y: (u, y + v, H - a / 2 + w))
            m.rect_tube(L - 2 * a, a, a, t, xf=lambda u, v, w, y=y: (a + u, y + v, p.zr + a / 2 + w))
        for x in (a / 2, L - a / 2):
            m.rect_tube(W - 2 * a, a, a, t, xf=lambda u, v, w, x=x: (x + v, a + u, H - a / 2 + w))
            m.rect_tube(W - 2 * a, a, a, t, xf=lambda u, v, w, x=x: (x + v, a + u, p.zr + a / 2 + w))
        return
    if p.shape == "sheet_metal":
        st = _sheet_profile(p)
        m.begin()
        n = len(st)
        for k in range(n - 1):
            (o0, i0), (o1, i1) = st[k], st[k + 1]
            m.face([m.vid(0, *o0), m.vid(0, *o1), m.vid(L, *o1), m.vid(L, *o0)])
            m.face([m.vid(0, *i1), m.vid(0, *i0), m.vid(L, *i0), m.vid(L, *i1)])
            m.face([m.vid(0, *o1), m.vid(0, *o0), m.vid(0, *i0), m.vid(0, *i1)])
            m.face([m.vid(L, *o0), m.vid(L, *o1), m.vid(L, *i1), m.vid(L, *i0)])
        (o0, i0), (o1, i1) = st[0], st[-1]
        m.face([m.vid(0, *i0), m.vid(0, *o0), m.vid(L, *o0), m.vid(L, *i0)])
        m.face([m.vid(0, *o1), m.vid(0, *i1), m.vid(L, *i1), m.vid(L, *o1)])
        m.end()
        return
    if p.shape == "casting":
        m.prism(_rounded_outline(L, W, p.rc, 5), 0.0, p.tb)
        prof = [(p.tb, p.rbore), (p.tb, p.rbo + p.rf)]
        for k in range(1, 5):
            a = math.pi / 2 * k / 4
            prof.append((p.tb + p.rf * (1 - math.cos(a)), p.rbo + p.rf * (1 - math.sin(a))))
        prof += [(H, p.rbo - (H - p.tb) * math.tan(p.draft)), (H, p.rbore)]
        m.revolve(prof, 32, xf=lambda a, b, c: (L / 2 + b, W / 2 + c, a))
        rt = p.rib_t
        for ang in (0, 90, 180, 270):
            ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            reach = (L / 2 if ang in (0, 180) else W / 2) * .86
            hr = (H - p.tb) * .55
            tri = [(p.rbo * .8, p.tb), (reach, p.tb), (p.rbo * .8, p.tb + hr)]
            m.prism(tri, -rt / 2, rt / 2, xf=lambda u, v, w, ca=ca, sa=sa: (L / 2 + u * ca - w * sa, W / 2 + u * sa + w * ca, v))
        for x, y in p.ch:
            m.cylinder_z(p.tb, p.tb + p.tb * .12, p.ch_r * 1.8, x, y, 18)
        return
    # assembly
    plate_solid(m, L, W, 0.0, p.tb, [{"cx": x, "cy": y, "r": W * .025} for x, y in p.base_holes])
    y0, y1 = p.sw
    rr = (y1 - y0) / 2
    out = [(y0, p.tb), (y1, p.tb), (y1, p.zc)]
    for k in range(1, 12):
        a = math.pi * k / 12
        out.append((W / 2 + rr * math.cos(a), p.zc + p.arch * math.sin(a)))
    out.append((y0, p.zc))
    for x0, x1 in p.sup:
        m.holed_prism(out, (W / 2, p.zc, p.rs * 1.04), x0, x1, 24, xf=lambda a, b, c: (c, a, b))
    m.revolve([(L * .03, 0), (L * .97, 0), (L * .97, p.rs * .8), (L * .94, p.rs), (L * .06, p.rs), (L * .03, p.rs * .8)], 24,
              xf=lambda a, b, c: (a, W / 2 + b, p.zc + c))
    m.revolve([(p.pul[0], p.rs * 1.02), (p.pul[1], p.rs * 1.02), (p.pul[1], p.rp * .92), (p.pul[1] - (p.pul[1] - p.pul[0]) * .15, p.rp),
               (p.pul[0] + (p.pul[1] - p.pul[0]) * .15, p.rp), (p.pul[0], p.rp * .92)], 32,
              xf=lambda a, b, c: (a, W / 2 + b, p.zc + c))
    hr = W * .045
    hexa = [(hr * math.cos(math.radians(60 * k)), hr * math.sin(math.radians(60 * k))) for k in range(6)]
    for x, y in p.bolts:
        m.prism([(x + a, y + b) for a, b in hexa], p.tb, p.tb + hr * .8)


def _rounded_outline(length: float, width: float, r: float, steps: int = 5) -> List[Pt]:
    r = max(0.0, min(r, length / 2 * 0.95, width / 2 * 0.95))
    pts: List[Pt] = []
    corners = [(length - r, r, -90), (length - r, width - r, 0), (r, width - r, 90), (r, r, 180)]
    for cx, cy, start in corners:
        for i in range(steps + 1):
            a = math.radians(start + 90 * i / steps)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def build_mesh(spec: Dict[str, Any], coarse: Any = False) -> Mesh:
    """The part's solid. coarse=True gives a light mesh for a drawing's small isometric view (big hole patterns
    subsampled, fewer facets); coarse="thumb" a medium one for a model's tile; False the full model."""
    p = part_model(spec)
    m = Mesh()
    _COARSE[0] = 2 if coarse is True else 1 if coarse else 0
    try:
        {"prismatic": _prismatic_mesh, "round": _round_mesh, "complex": _complex_mesh, "other": _other_mesh}[p.family](p, m)
    except Exception:  # noqa: BLE001 - never fail a file for a strange spec
        m = Mesh()
    finally:
        _COARSE[0] = 0
    if not m.f:
        L, W, H = bbox(spec)
        m.box(0, 0, 0, L, W, H)
    elif p.family == "complex" and m.v:
        # sculpted surfaces overshoot or fall short of the stated envelope by a few percent: stretch the solid
        # to it exactly, so the model measures what the drawing and the RFQ say (orientation is unchanged)
        want = bbox(spec)
        lo = [min(v[k] for v in m.v) for k in range(3)]
        span = [max(v[k] for v in m.v) - lo[k] for k in range(3)]
        fac = [want[k] / span[k] if span[k] > 1e-12 and 0.9 < span[k] / want[k] < 1.1 else 1.0 for k in range(3)]
        if any(abs(f - 1.0) > 1e-4 for f in fac):
            m.v = [tuple(lo[k] + (v[k] - lo[k]) * fac[k] for k in range(3)) for v in m.v]  # type: ignore[misc]
    return m


def mesh_for(spec: Dict[str, Any]) -> Dict[str, Any]:
    return build_mesh(spec).as_dict(spec.get("units") or "in")


# --------------------------------------------------------------------------- #
# Drafting primitives (sheet coordinates: points, y down)
# --------------------------------------------------------------------------- #
HID = 0.55  # hidden line weight


def _arrow(page: Page, x: float, y: float, dx: float, dy: float, size: float = 4.4) -> None:
    """Filled arrowhead with its tip at (x, y), pointing along (dx, dy)."""
    ln = math.hypot(dx, dy) or 1.0
    ux, uy = dx / ln, dy / ln
    bx, by = x - ux * size, y - uy * size
    px, py = -uy * size * 0.3, ux * size * 0.3
    page.polygon([(x, y), (bx + px, by + py), (bx - px, by - py)], 0.2, BLACK, BLACK)


def _label(page: Page, cx: float, cy: float, text: str, size: float = DIM, bold: bool = False) -> None:
    """Text centered on (cx, cy) with a white box behind it, so it breaks the line it sits on."""
    tw = text_width(text, size, bold)
    page.rect(cx - tw / 2 - 1.6, cy - size * 0.56, tw + 3.2, size * 1.1, 0, None, WHITE)
    page.text(cx, cy + size * 0.35, text, size, bold, "middle")


def _dim_h(page: Page, x1: float, x2: float, y: float, ext_from: float, label: str) -> None:
    if x2 < x1:
        x1, x2 = x2, x1
    sgn = 1 if y > ext_from else -1
    for x in (x1, x2):
        page.line(x, ext_from + sgn * 1.6, x, y + sgn * 2.6, THIN)
    tw = text_width(label, DIM)
    if x2 - x1 >= tw + 13:
        page.line(x1, y, x2, y, THIN)
        _arrow(page, x1, y, -1, 0)
        _arrow(page, x2, y, 1, 0)
        _label(page, (x1 + x2) / 2, y, label)
    else:
        page.line(x1 - 9, y, x2 + 9, y, THIN)
        _arrow(page, x1, y, 1, 0)
        _arrow(page, x2, y, -1, 0)
        page.text(x2 + 11, y + DIM * 0.35, label, DIM)


def _dim_v(page: Page, y1: float, y2: float, x: float, ext_from: float, label: str) -> None:
    if y2 < y1:
        y1, y2 = y2, y1
    sgn = 1 if x > ext_from else -1
    for y in (y1, y2):
        page.line(ext_from + sgn * 1.6, y, x + sgn * 2.6, y, THIN)
    if y2 - y1 >= 19:
        page.line(x, y1, x, y2, THIN)
        _arrow(page, x, y1, 0, -1)
        _arrow(page, x, y2, 0, 1)
    else:
        page.line(x, y1 - 9, x, y2 + 9, THIN)
        _arrow(page, x, y1, 0, 1)
        _arrow(page, x, y2, 0, -1)
    _label(page, x, (y1 + y2) / 2, label)


def _dia_dim(page: Page, x: float, y_top: float, y_bot: float, label: str, label_y: Optional[float] = None) -> None:
    """A diameter dimensioned across a turned step, inside the view (figures on the axis unless label_y says)."""
    page.line(x, y_top, x, y_bot, THIN)
    _arrow(page, x, y_top, 0, -1, 4.0)
    _arrow(page, x, y_bot, 0, 1, 4.0)
    _label(page, x, (y_top + y_bot) / 2 if label_y is None else label_y, label)


def _cmark(page: Page, cx: float, cy: float, r: float) -> None:
    if r >= 4.5:
        page.line(cx - r - 3.2, cy, cx + r + 3.2, cy, THIN, dash=CENTER_DASH)
        page.line(cx, cy - r - 3.2, cx, cy + r + 3.2, THIN, dash=CENTER_DASH)
    else:
        arm = r + 2.2
        page.path([("M", cx - arm, cy), ("L", cx + arm, cy), ("M", cx, cy - arm), ("L", cx, cy + arm)], THIN)


def _hid(page: Page, x1: float, y1: float, x2: float, y2: float) -> None:
    page.line(x1, y1, x2, y2, HID, dash=HIDDEN_DASH)


def _thread_arc(page: Page, cx: float, cy: float, r: float) -> None:
    if r >= 2.0:  # smaller than this it is only a smudge next to the tap drill circle
        page.arc(cx, cy, r, 90, 360, THIN)


def _rrect(page: Page, x0: float, y0: float, x1: float, y1: float, r: float, lw: float = MED,
           dash: Optional[Sequence[float]] = None) -> None:
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    if r < 0.3:
        page.rect(x0, y0, x1 - x0, y1 - y0, lw, BLACK, None, dash)
        return
    cmds: List[Tuple] = [("M", x0 + r, y0), ("L", x1 - r, y0)]
    cmds += arc_commands(x1 - r, y0 + r, r, 90, 0, move=False)
    cmds.append(("L", x1, y1 - r))
    cmds += arc_commands(x1 - r, y1 - r, r, 0, -90, move=False)
    cmds.append(("L", x0 + r, y1))
    cmds += arc_commands(x0 + r, y1 - r, r, -90, -180, move=False)
    cmds.append(("L", x0, y0 + r))
    cmds += arc_commands(x0 + r, y0 + r, r, 180, 90, move=False)
    cmds.append(("Z",))
    page.path(cmds, lw, BLACK, None, dash)


def _hatch(page: Page, poly: Sequence[Pt], spacing: float = 3.0, angle: float = 45.0, lw: float = 0.35) -> None:
    """Section lining: parallel lines clipped to a polygon (even-odd), on a sheet-wide grid so regions line up."""
    if len(poly) < 3:
        return
    a = math.radians(angle)
    dx, dy = math.cos(a), -math.sin(a)          # line direction on the sheet (y down)
    nx, ny = -dy, dx                            # normal
    offs = [px * nx + py * ny for px, py in poly]
    k0, k1 = int(math.floor(min(offs) / spacing)), int(math.ceil(max(offs) / spacing))
    n = len(poly)
    for k in range(k0, k1 + 1):
        c = k * spacing
        ts = []
        for i in range(n):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
            s1, s2 = x1 * nx + y1 * ny - c, x2 * nx + y2 * ny - c
            if (s1 > 0) != (s2 > 0):
                t = s1 / (s1 - s2)
                ts.append((x1 + (x2 - x1) * t) * dx + (y1 + (y2 - y1) * t) * dy)
        ts.sort()
        for j in range(0, len(ts) - 1, 2):
            t0, t1 = ts[j], ts[j + 1]
            if t1 - t0 < 0.4:
                continue
            page.line(c * nx + t0 * dx, c * ny + t0 * dy, c * nx + t1 * dx, c * ny + t1 * dy, lw)


def _leader(page: Page, sx: float, sy: float, tx: float, ty: float, shoulder: float = 6.0, dot: bool = False) -> None:
    """Leader from a label at (sx, sy): a short horizontal shoulder, then straight to the target."""
    kx = sx + (shoulder if tx > sx else -shoulder)
    page.line(sx, sy, kx, sy, THIN)
    _ko_line(page, kx, sy, tx, ty)
    if dot:
        page.circle(tx, ty, 1.2, 0, None, BLACK)
    else:
        _arrow(page, tx, ty, tx - kx, ty - sy)


def _machine_mark(page: Page, x: float, y: float, up: bool = False) -> None:
    """Surface texture symbol with material removal required, touching the surface at (x, y)."""
    s = -1 if up else 1
    page.polygon([(x - 3.2, y - s * 4.2), (x, y), (x + 3.2 * 0.62, y - s * 2.6)], 0.5, BLACK, None, closed=False)
    page.line(x, y, x + 6.4, y - s * 8.4, 0.5)
    page.line(x - 3.2, y - s * 4.2, x + 3.2 * 0.62 * 1.0 + 0.4, y - s * 2.6 - 0.4 * s, 0.5)


def _weld(page: Page, jx: float, jy: float, ex: float, ey: float, size: str, tail: str = "", all_around: bool = True) -> None:
    """A fillet weld symbol: arrow to the joint, reference line with the fillet triangle on the arrow side."""
    page.line(ex, ey, jx, jy, THIN)
    _arrow(page, jx, jy, jx - ex, jy - ey, 4.0)
    d = 1 if ex >= jx else -1
    rl = 30.0
    page.line(ex, ey, ex + d * rl, ey, THIN)
    tx = ex + d * 12
    page.polygon([(tx, ey), (tx, ey + 5.2), (tx + d * 5.2, ey)], 0.5, BLACK, None)
    page.text(tx - d * 1.6, ey + 5.4, size, 5.6, anchor="end" if d > 0 else "start")
    if all_around:
        page.circle(ex, ey, 2.0, 0.45)
    if tail:
        endx = ex + d * rl
        page.line(endx, ey, endx + d * 4, ey - 3.5, THIN)
        page.line(endx, ey, endx + d * 4, ey + 3.5, THIN)
        page.text(endx + d * 6, ey + 2, tail, 5.6, anchor="start" if d > 0 else "end")


def _zbreak(page: Page, x: float, y0: float, y1: float) -> None:
    """A long-break line across a view at x (y0 above y1)."""
    ym = (y0 + y1) / 2
    h = min(5.0, (y1 - y0) * 0.2 + 1.5)
    page.polygon([(x, y0 - 4), (x, ym - h), (x + 3, ym - h * 0.4), (x - 3, ym + h * 0.4), (x, ym + h), (x, y1 + 4)],
                 THIN, BLACK, None, closed=False)


def _ellipse_poly(cx: float, cy: float, rx: float, ry: float, rot: float = 0.0, n: int = 28) -> List[Pt]:
    c, s = math.cos(rot), math.sin(rot)
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n
        x, y = rx * math.cos(t), ry * math.sin(t)
        out.append((cx + x * c - y * s, cy + x * s + y * c))
    return out


# --------------------------------------------------------------------------- #
# Views and their layout
# --------------------------------------------------------------------------- #
class Xf:
    """Maps part coordinates of one view (u right, v up) to sheet points, with an optional break in u."""

    def __init__(self, left: float, bottom: float, s: float, sy: float, u0: float, v0: float,
                 brk: Optional[Tuple[float, float, float]] = None):
        self.left, self.bottom, self.s, self.sy, self.u0, self.v0, self.brk = left, bottom, s, sy, u0, v0, brk

    def pos(self, u: float) -> float:
        if not self.brk:
            return (u - self.u0) * self.s
        b0, b1, gap = self.brk
        if u <= b0:
            return (u - self.u0) * self.s
        if u >= b1:
            return (b0 - self.u0) * self.s + gap + (u - b1) * self.s
        return (b0 - self.u0) * self.s + gap * (u - b0) / max(b1 - b0, 1e-12)

    def x(self, u: float) -> float:
        return self.left + self.pos(u)

    def y(self, v: float) -> float:
        return self.bottom - (v - self.v0) * self.sy

    def p(self, u: float, v: float) -> Pt:
        return self.x(u), self.y(v)

    def hidden(self, u: float) -> bool:
        return bool(self.brk) and self.brk[0] < u < self.brk[1]


class View:
    def __init__(self, key: str, title: str, u0: float, u1: float, v0: float, v1: float, col: int, row: int,
                 draw: Callable[[Page, Xf], None], thin: bool = False, title_size: float = 6.5, bold_title: bool = True):
        self.key, self.title = key, title
        self.u0, self.u1, self.v0, self.v1 = u0, u1, v0, v1
        self.col, self.row = col, row
        self.draw = draw
        self.thin = thin
        self.brk: Optional[Tuple[float, float, float]] = None
        self.dims: List[Tuple[str, str, int, float, float, str, bool]] = []
        self.xf: Optional[Xf] = None
        self.extra = {"l": 0.0, "r": 0.0, "t": 0.0, "b": 0.0}
        self.title_size = title_size
        self.bold_title = bold_title
        self.title_op: Optional[int] = None

    def dim(self, kind: str, side: str, level: int, a: float, b: float, label: str, opt: bool = False) -> None:
        """opt: a feature dimension that is left out when it does not fit at the final scale."""
        if abs(b - a) > 1e-12:
            self.dims.append((kind, side, level, a, b, label, opt))

    def _levels(self, side: str) -> Dict[int, float]:
        """Offset of each dimension level from the geometry edge on one side."""
        levels = sorted({d[2] for d in self.dims if d[1] == side})
        out: Dict[int, float] = {}
        if side in ("above", "below"):
            off = 9.0
            for lv in levels:
                out[lv] = off
                off += 11.0
        else:
            off = 3.0
            for lv in levels:
                tw = max(text_width(d[5], DIM) for d in self.dims if d[1] == side and d[2] == lv)
                off += tw / 2 + 4.5
                out[lv] = off
                off += tw / 2 + 1.5
        return out

    def margins(self) -> Tuple[float, float, float, float]:
        ml = mr = mt = mb = 0.0
        for side in ("above", "below", "left", "right"):
            lv = self._levels(side)
            if not lv:
                continue
            last = max(lv)
            if side in ("above", "below"):
                val = lv[last] + 6.0
            else:
                tw = max(text_width(d[5], DIM) for d in self.dims if d[1] == side and d[2] == last)
                val = lv[last] + tw / 2 + 3.0
            if side == "above":
                mt = val
            elif side == "below":
                mb = val
            elif side == "left":
                ml = val
            else:
                mr = val
        if self.title:
            mb += 13.0
        return ml + self.extra["l"], mr + self.extra["r"], mt + self.extra["t"], mb + self.extra["b"]

    def gw(self, s: float) -> float:
        span = (self.u1 - self.u0) * s
        if self.brk:
            span = (self.u1 - self.u0 - (self.brk[1] - self.brk[0])) * s + self.brk[2]
        return max(span, 6.0)

    def gh(self, s: float) -> float:
        h = (self.v1 - self.v0) * s
        if self.thin:
            h = max(h, 12.0)
        return max(h, 4.0)

    def sy(self, s: float) -> float:
        return self.gh(s) / max(self.v1 - self.v0, 1e-12)

    def render(self, page: Page) -> None:
        T = self.xf
        assert T is not None
        self.draw(page, T)
        if self.brk:
            b0, b1, gap = self.brk
            xa, xb = T.x(b0), T.x(b1)
            ytop, ybot = T.y(self.v1) - 2.5, T.y(self.v0) + 2.5
            page.rect(xa + 0.8, ytop - 3, xb - xa - 1.6, ybot - ytop + 6, 0, None, WHITE)
            _zbreak(page, xa, ytop, ybot)
            _zbreak(page, xb, ytop, ybot)
        gx0, gx1 = T.x(self.u0), T.x(self.u1)
        gy0, gy1 = T.y(self.v1), T.y(self.v0)
        for side in ("above", "below", "left", "right"):
            lv = self._levels(side)
            for kind, sd, level, a, b, label, opt in self.dims:
                if sd != side:
                    continue
                off = lv[level]
                if opt:
                    span = abs(T.x(b) - T.x(a)) if kind == "h" else abs(T.y(b) - T.y(a))
                    need = text_width(label, DIM) + 14 if kind == "h" else 20
                    if span < need:
                        continue
                if side == "above":
                    _dim_h(page, T.x(a), T.x(b), gy0 - off, gy0, label)
                elif side == "below":
                    _dim_h(page, T.x(a), T.x(b), gy1 + off, gy1, label)
                elif side == "left":
                    _dim_v(page, T.y(a), T.y(b), gx0 - off, gx0, label)
                else:
                    _dim_v(page, T.y(a), T.y(b), gx1 + off, gx1, label)
        if self.title:
            lvb = self._levels("below")
            below = (max(lvb.values()) + 6.0) if lvb else 0.0
            self.title_op = len(page.ops)
            page.text((gx0 + gx1) / 2, gy1 + below + 10.5 + self.extra["b"], self.title, self.title_size,
                      self.bold_title, "middle")


class Block:
    """A fixed-size area in the view grid (callout labels), placed like a view."""

    def __init__(self, key: str, col: int, row: int, w: float, h: float):
        self.key, self.col, self.row, self.w, self.h = key, col, row, w, h
        self.x = self.y = 0.0


def _layout(views: List[View], blocks: List[Block], region: Tuple[float, float, float, float],
            gapx: float = 24.0, gapy: float = 14.0) -> float:
    """Solve for the largest scale that fits the grid in the region, then place everything. Returns the scale."""
    rx, ry, rw, rh = region
    ncol = max([v.col for v in views] + [b.col for b in blocks]) + 1
    nrow = max([v.row for v in views] + [b.row for b in blocks]) + 1
    mg = {id(v): v.margins() for v in views}

    def grid(s: float):
        cl = [0.0] * ncol
        cw = [0.0] * ncol
        cr = [0.0] * ncol
        rt = [0.0] * nrow
        rg = [0.0] * nrow
        rb = [0.0] * nrow
        fixed_w = [0.0] * ncol
        fixed_h = [0.0] * nrow
        for v in views:
            ml, mr, mt, mb = mg[id(v)]
            cl[v.col] = max(cl[v.col], ml)
            cr[v.col] = max(cr[v.col], mr)
            cw[v.col] = max(cw[v.col], v.gw(s))
            rt[v.row] = max(rt[v.row], mt)
            rb[v.row] = max(rb[v.row], mb)
            rg[v.row] = max(rg[v.row], v.gh(s))
        for b in blocks:
            fixed_w[b.col] = max(fixed_w[b.col], b.w)
            fixed_h[b.row] = max(fixed_h[b.row], b.h)
        colw = [max(cl[i] + cw[i] + cr[i], fixed_w[i]) for i in range(ncol)]
        rowh = [max(rt[i] + rg[i] + rb[i], fixed_h[i]) for i in range(nrow)]
        used_c = [i for i in range(ncol) if colw[i] > 0]
        used_r = [i for i in range(nrow) if rowh[i] > 0]
        tw = sum(colw) + gapx * max(0, len(used_c) - 1)
        th = sum(rowh) + gapy * max(0, len(used_r) - 1)
        return tw, th, colw, rowh, cl, rt, rg

    lo, hi = 1e-12, 1e9
    for _ in range(72):
        mid = math.sqrt(lo * hi)
        tw, th = grid(mid)[:2]
        if tw <= rw and th <= rh:
            lo = mid
        else:
            hi = mid
    s = lo
    tw, th, colw, rowh, cl, rt, rg = grid(s)
    x0 = rx + max(0.0, (rw - tw) / 2)
    y0 = ry + max(0.0, (rh - th) / 2) * 0.8
    colx, x = [], x0
    for i in range(ncol):
        colx.append(x)
        x += colw[i] + (gapx if colw[i] > 0 else 0)
    rowy, y = [], y0
    for i in range(nrow):
        rowy.append(y)
        y += rowh[i] + (gapy if rowh[i] > 0 else 0)
    for v in views:
        left = colx[v.col] + cl[v.col]
        bottom = rowy[v.row] + rt[v.row] + rg[v.row]
        if rowh[v.row] > rt[v.row] + rg[v.row] + max(mg[id(w)][3] for w in views if w.row == v.row) + 0.5:
            bottom = rowy[v.row] + rowh[v.row] - max(mg[id(w)][3] for w in views if w.row == v.row)
        v.xf = Xf(left, bottom, s, v.sy(s), v.u0, v.v0, v.brk)
    for b in blocks:
        b.x, b.y = colx[b.col], rowy[b.row]
        b.w = max(b.w, colw[b.col])
        b.h = max(b.h, rowh[b.row])
    return s


# --------------------------------------------------------------------------- #
# Callout labels and leaders
# --------------------------------------------------------------------------- #
def _label_lines(text: str, width: float) -> Tuple[List[str], float]:
    size = LABEL
    lines = wrap(text, width, size)
    while len(lines) > 3 and size > 5.2:
        size -= 0.3
        lines = wrap(text, width, size)
    if len(lines) > 3:
        lines = lines[:3]
        lines[-1] = fit_text(lines[-1] + " ...", width, size)[0]
    return lines or ["-"], size


def _anchor_uv(views: Dict[str, View], anchor: Anchor) -> Optional[Tuple[View, float, float, float]]:
    """The anchor in its view, moved out of a broken-out stretch of the view if it falls inside one."""
    view, u, v, r = anchor
    vw = views.get(view)
    if vw is None or vw.xf is None:
        return None
    brk = vw.xf.brk
    if brk and brk[0] < u < brk[1]:
        pad = 7.0 / max(vw.xf.s, 1e-9)
        u = brk[0] - pad if u - brk[0] < brk[1] - u else brk[1] + pad
    return vw, u, v, r


def _tip(views: Dict[str, View], anchor: Anchor, toward: Pt) -> Optional[Pt]:
    hit = _anchor_uv(views, anchor)
    if hit is None:
        return None
    vw, u, v, r = hit
    T = vw.xf
    cx, cy = T.p(u, v)
    rp = r * T.s
    if rp > 0.8:
        dx, dy = toward[0] - cx, toward[1] - cy
        ang = math.atan2(dy, dx)
        # a leader straight along a center line would lie on it: meet the circle a little off the axis
        for axis in (-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi):
            off = ang - axis
            if abs(off) < math.radians(10):
                ang = axis + math.copysign(math.radians(14), off if abs(off) > 1e-9 else 1.0)
                break
        return cx + math.cos(ang) * rp, cy + math.sin(ang) * rp
    return cx, cy


def _labels_column(page: Page, p: Part, views: Dict[str, View], blk: Block, items: List[Tuple[str, Anchor, str]]) -> None:
    """items: (text, anchor, kind) with kind 'callout' or 'balloon'. Labels stack in the block; leaders fan out
    to the left in angle order so they do not cross; targets under the block get the bottom labels."""
    if not items:
        return
    width = blk.w - 12
    x = blk.x + 10
    ref_y = blk.y + blk.h / 2
    entries = []
    for text, anc, kind in items:
        hit = _anchor_uv(views, anc)
        if hit is None:
            continue
        vw, u, v, _ = hit
        cx, cy = vw.xf.p(u, v)
        if kind == "balloon":
            e = {"lines": [text], "size": 7.0, "h": 15.0, "w": 13.0}
        else:
            lines, size = _label_lines(text, width)
            e = {"lines": lines, "size": size, "h": len(lines) * (size + 1.4) + 2,
                 "w": max(text_width(ln, size) for ln in lines)}
        e.update(anc=anc, kind=kind, cx=cx, cy=cy, under=cx > x - 4)
        e["ang"] = math.atan2(cy - ref_y, max(x - cx, 1e-6)) if not e["under"] else 10 - cx
        entries.append(e)
    entries.sort(key=lambda e: (e["under"], e["ang"]))
    top, bottom = blk.y + 2, blk.y + blk.h - 2
    xr = x + width

    def place(order: List[Dict[str, Any]]) -> None:
        gap = 5.0
        total = sum(e["h"] for e in order) + gap * (len(order) - 1)
        if total > bottom - top:
            gap = max(1.5, (bottom - top - sum(e["h"] for e in order)) / max(1, len(order) - 1))
        y = top
        for e in order:
            want = min(max(e["cy"], top), bottom) - e["h"] / 2
            e["y"] = max(y, min(want, bottom - e["h"]))
            y = e["y"] + e["h"] + gap
        if y - gap > bottom:
            y = bottom
            for e in reversed(order):
                e["y"] = min(e["y"], y - e["h"])
                y = e["y"] - gap

    def drop_start(e: Dict[str, Any]) -> Pt:
        """Where a leader leaves the underside of its label: inclined up to about 15 degrees off vertical, as far
        as the text reaches to the right (a vertical leader can lie right on a wall of the view it points into;
        leaning the other way would cut through the labels stacked below)."""
        sy = e["y"] + e["h"] - 0.5
        lo, hi = x + 3, x + e["w"] - 3
        lean = max(6.0, (e["cy"] - sy) * 0.27)
        return min(max(e["cx"] + lean, lo), max(hi, lo)), sy

    def geom(e: Dict[str, Any], mode: str):
        """Text box and leader polyline of a label whose target is under the block. mode 'L': text on the
        left edge, leader dropping from under the text; 'R': text on the right edge, leader from its left end."""
        if e["kind"] == "balloon":
            bx, by = x + 6.5, e["y"] + 7.5
            tip = _tip(views, e["anc"], (bx - 6.5, by)) or (bx, by)
            return (x, e["y"], x + 13.0, e["y"] + 15.0), [(bx - 6.5, by), tip]
        if mode == "L":
            sx, sy = drop_start(e)
            tip = _tip(views, e["anc"], (sx, sy)) or (sx, sy)
            return (x, e["y"], x + e["w"], e["y"] + e["h"]), [(sx, sy), tip]
        x0 = xr - e["w"]
        sx, sy = x0 - 2, e["y"] + e["size"] * 0.62
        tip = _tip(views, e["anc"], (sx - 6.0, sy)) or (sx, sy)
        return (x0, e["y"], xr, e["y"] + e["h"]), [(sx, sy), (sx - 6.0, sy), tip]

    head = [e for e in entries if not e["under"]]
    under = [e for e in entries if e["under"]]

    def head_geom(e: Dict[str, Any]):
        if e["kind"] == "balloon":
            bx, by = x + 6.5, e["y"] + 7.5
            tip = _tip(views, e["anc"], (bx - 6.5, by)) or (bx, by)
            kx = bx - 6.5 + (4.0 if tip[0] > bx - 6.5 else -4.0)
            return (x, e["y"], x + 13.0, e["y"] + 15.0), [(bx - 6.5, by), (kx, by), tip]
        sy = e["y"] + e["size"] * 0.62
        tip = _tip(views, e["anc"], (x - 2, sy)) or (x - 2, sy)
        kx = x - 2 + (6.0 if tip[0] > x - 2 else -6.0)
        return (x, e["y"], x + e["w"], e["y"] + e["h"]), [(x - 2, sy), (kx, sy), tip]

    if 1 < len(head) <= 6:
        # the fan order (by angle from the middle of the block) crosses leaders when the labels bunch up near
        # their targets: take the order with the fewest crossings, then the shortest leaders
        def hscore(order: List[Dict[str, Any]]) -> float:
            place(order + under)
            g = [head_geom(e) for e in order]
            s = 0.0
            for k, (_, pts) in enumerate(g):
                s += math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) * 1e-5
                for j, (box, _) in enumerate(g):
                    if j != k and any(_seg_hits_box(pts[q], pts[q + 1], box) for q in range(len(pts) - 1)):
                        s += 10.0
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    pa, pb = g[a][1], g[b][1]
                    if any(_segs_cross(pa[i], pa[i + 1], pb[j], pb[j + 1])
                           for i in range(len(pa) - 1) for j in range(len(pb) - 1)):
                        s += 1.0
            return s

        best_h = None
        for o in itertools.permutations(head):
            sc = hscore(list(o))
            if best_h is None or sc < best_h[0] - 1e-9:
                best_h = (sc, list(o))
        head = best_h[1]
    mode = "L"
    if len(under) > 1:
        # leaders of these labels drop through the labels stacked under them: pick the order (and side) whose
        # leaders cross the fewest other labels, then the fewest other leaders
        def score(order: List[Dict[str, Any]], md: str) -> float:
            place(head + order)
            g = [geom(e, md) for e in order]
            boxes = [(x, e["y"], x + e["w"], e["y"] + e["h"]) for e in head] + [b for b, _ in g]
            s = 0.0
            for k, (box, pts) in enumerate(g):
                for j, other in enumerate(boxes):
                    # a leader must not run back through its own text either
                    if any(_seg_hits_box(pts[q], pts[q + 1], other) for q in range(len(pts) - 1)):
                        s += 10.0
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    pa, pb = g[a][1], g[b][1]
                    if any(_segs_cross(pa[i], pa[i + 1], pb[j], pb[j + 1])
                           for i in range(len(pa) - 1) for j in range(len(pb) - 1)):
                        s += 1.0
            return s

        orders = [list(o) for o in itertools.permutations(under)] if len(under) <= 5 else [under]
        best = None
        for md in ("L", "R"):
            for o in orders:
                sc = score(o, md)
                if best is None or sc < best[0] - 1e-9:
                    best = (sc, o, md)
                if best[0] == 0:
                    break
            if best[0] == 0:
                break
        under, mode = best[1], best[2]
    entries = head + under
    place(entries)
    for e in entries:
        if e["kind"] == "balloon":
            bx, by = x + 6.5, e["y"] + 7.5
            page.circle(bx, by, 6.5, 0.6, BLACK, WHITE)
            page.text(bx, by + 2.5, e["lines"][0], 7.0, True, "middle")
            tip = _tip(views, e["anc"], (bx - 6.5, by))
            if tip:
                _leader(page, bx - 6.5, by, tip[0], tip[1], 4.0, dot=True)
            continue
        size = e["size"]
        yy = e["y"] + size
        right = e["under"] and mode == "R"
        for line in e["lines"]:
            page.text(xr if right else x, yy, line, size, anchor="end" if right else "start")
            yy += size + 1.4
        if not e["under"]:
            sy = e["y"] + size * 0.62
            tip = _tip(views, e["anc"], (x - 2, sy))
            if tip:
                _leader(page, x - 2, sy, tip[0], tip[1])
        elif right:
            _, pts = geom(e, "R")
            (sx, sy), (kx, _), tip = pts
            page.line(sx, sy, kx, sy, THIN)
            _ko_line(page, kx, sy, tip[0], tip[1])
            _arrow(page, tip[0], tip[1], tip[0] - kx, tip[1] - sy)
        else:
            sx, sy = drop_start(e)
            tip = _tip(views, e["anc"], (sx, sy))
            if tip:
                _ko_line(page, sx, sy, tip[0], tip[1])
                _arrow(page, tip[0], tip[1], tip[0] - sx, tip[1] - sy)


def _seg_hits_box(a: Pt, b: Pt, box: Tuple[float, float, float, float], pad: float = 0.6) -> bool:
    """Does the segment a-b pass through the box (shrunk by pad on every side)?"""
    x0, y0, x1, y1 = box[0] + pad, box[1] + pad, box[2] - pad, box[3] - pad
    if x1 <= x0 or y1 <= y0:
        return False
    dx, dy = b[0] - a[0], b[1] - a[1]
    u0, u1 = 0.0, 1.0
    for pv, qv in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(pv) < 1e-12:
            if qv < 0:
                return False
            continue
        t = qv / pv
        if pv < 0:
            u0 = max(u0, t)
        else:
            u1 = min(u1, t)
        if u0 > u1:
            return False
    return True


def _segs_cross(a: Pt, b: Pt, c: Pt, d: Pt) -> bool:
    """Proper crossing of segments a-b and c-d (touching end points do not count)."""
    def orient(p1: Pt, p2: Pt, p3: Pt) -> float:
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    d1, d2 = orient(c, d, a), orient(c, d, b)
    d3, d4 = orient(a, b, c), orient(a, b, d)
    return d1 * d2 < -1e-9 and d3 * d4 < -1e-9


def _labels_band(page: Page, views: Dict[str, View], band: Tuple[float, float, float, float],
                 items: List[Tuple[str, Anchor]]) -> None:
    """Round parts: labels in a band above the views, one per row, leaders dropping to the features."""
    bx, by, bw, bh = band
    rows = []
    for text, anc in items:
        hit = _anchor_uv(views, anc)
        if hit is None:
            continue
        vw, u, v, _ = hit
        cx, cy = vw.xf.p(u, v)
        lines, size = _label_lines(text, min(190.0, bw * 0.45))
        rows.append({"lines": lines, "size": size, "anc": anc, "ax": cx, "ay": cy,
                     "w": max(text_width(ln, size) for ln in lines), "h": len(lines) * (size + 1.4)})
    def plan(mirror: bool) -> Tuple[List[Tuple[Dict[str, Any], float]], int]:
        """Rows top to bottom with the x of each leader's knee. Each row's text must stay clear of the leaders
        of the rows above it: left to right by feature (texts to the right of every earlier leader), or, when
        the features sit near the right edge, mirrored (right to left, texts to the left of earlier leaders).
        Returns the rows and how many of them could not keep clear (their leaders cross a text)."""
        out, bad = [], 0
        if not mirror:
            prev = bx - 10
            for r in sorted(rows, key=lambda r: r["ax"]):
                lo = prev + 8
                start = max(lo, min(r["ax"] - 6, bx + bw - r["w"] - 8))
                if start + 8 + r["w"] > bx + bw:
                    start = bx + bw - r["w"] - 8
                bad += start < lo - 1e-6
                out.append((r, start + 2))
                prev = max(prev, r["ax"], start + 2)
        else:
            prev = bx + bw + 10
            for r in sorted(rows, key=lambda r: -r["ax"]):
                hi = prev - 8
                end = min(hi, max(r["ax"] + 6, bx + r["w"] + 8))
                if end - 8 - r["w"] < bx:
                    end = bx + r["w"] + 8
                bad += end > hi + 1e-6
                out.append((r, end - 2))
                prev = min(prev, r["ax"], end - 2)
        return out, bad

    order, bad = plan(False)
    mirrored = False
    if bad:
        order2, bad2 = plan(True)
        if bad2 < bad:
            order, mirrored = order2, True
    total = sum(r["h"] for r in rows) + 4.0 * max(0, len(rows) - 1)
    y = by + max(0.0, bh - total)
    for r, knee in order:
        size = r["size"]
        yy = y + size
        for line in r["lines"]:
            if mirrored:
                page.text(knee - 6, yy, line, size, anchor="end")
            else:
                page.text(knee + 6, yy, line, size)
            yy += size + 1.4
        sy = y + (len(r["lines"]) - 1) * (size + 1.4) + size * 0.62
        tip = _tip(views, r["anc"], (knee, sy + 10))
        if tip:
            page.line(knee + (-4 if mirrored else 4), sy, knee, sy, THIN)
            _ko_line(page, knee, sy, tip[0], tip[1])
            _arrow(page, tip[0], tip[1], tip[0] - knee, tip[1] - sy)
        y += r["h"] + 4.0


# --------------------------------------------------------------------------- #
# Prismatic views: top, front, right side (third angle), hidden lines, center marks
# --------------------------------------------------------------------------- #
def _chamfer_outline(L: float, W: float, ch: Dict[int, float]) -> List[Pt]:
    pts: List[Pt] = []
    for k, (cx, cy), (pa, pb) in ((3, (0, 0), ((0, 1), (1, 0))), (0, (L, 0), ((-1, 0), (0, 1))),
                                  (1, (L, W), ((0, -1), (-1, 0))), (2, (0, W), ((1, 0), (0, -1)))):
        c = ch.get(k, 0.0)
        if c > 0:
            pts += [(cx + pa[0] * c, cy + pa[1] * c), (cx + pb[0] * c, cy + pb[1] * c)]
        else:
            pts.append((cx, cy))
    return pts


class _Seen:
    """Drop duplicate hidden lines (many holes share the same projection)."""

    def __init__(self) -> None:
        self.keys: set = set()

    def once(self, *vals: float) -> bool:
        key = tuple(round(v, 1) for v in vals)
        if key in self.keys:
            return False
        self.keys.add(key)
        return True


def _hole_circles(page: Page, cx: float, cy: float, h: Dict[str, Any], s: float, hidden: bool = False) -> None:
    r = max(h["r"] * s, 0.7)
    dash = HIDDEN_DASH if hidden else None
    lw = HID if hidden else MED
    style = h["style"]
    if style in ("cbore", "nest", "csk") and h["r2"] > h["r"]:
        page.circle(cx, cy, h["r2"] * s, lw, dash=dash)
    if style == "port":
        page.circle(cx, cy, max(h["r2"], h["rt"] * 1.3) * s, THIN, dash=dash)
    page.circle(cx, cy, r, lw, dash=dash)
    if style in ("tap", "port") and h["rt"] > h["r"] and not hidden:
        _thread_arc(page, cx, cy, h["rt"] * s)
    _cmark(page, cx, cy, max(h["r"], h["r2"], h["rt"]) * s)


def _vertical_hole_lines(page: Page, T: Xf, h: Dict[str, Any], a: float, seen: _Seen) -> None:
    """A hole drilled along the view's v axis, seen from the side: hidden walls, steps, and a centerline."""
    s = T.s
    lo, hi = h["lo"], h["hi"]
    r, r2, rt = h["r"], h["r2"], h["rt"]
    entry_top = h.get("entry", "+") == "+"
    if not seen.once(a, lo, hi, r, r2):
        return
    x = T.x(a)
    ytop, ybot = T.y(hi), T.y(lo)
    if h["style"] in ("cbore", "nest") and r2 > r:
        d2 = min(h["d2"] or (hi - lo) * .3, (hi - lo) * .9)
        ys = T.y(hi - d2) if entry_top else T.y(lo + d2)
        yc0, yc1 = (ytop, ys) if entry_top else (ys, ybot)
        for sg in (-1, 1):
            _hid(page, x + sg * r2 * s, yc0, x + sg * r2 * s, yc1)
        _hid(page, x - r2 * s, ys, x + r2 * s, ys)
        y0, y1 = (ys, ybot) if entry_top else (ytop, ys)
    elif h["style"] == "csk" and r2 > r:
        dz = min(r2 - r, (hi - lo) * .6)
        ys = T.y(hi - dz)
        for sg in (-1, 1):
            _hid(page, x + sg * r2 * s, ytop, x + sg * r * s, ys)
        y0, y1 = ys, ybot
    else:
        y0, y1 = ytop, ybot
    for sg in (-1, 1):
        _hid(page, x + sg * r * s, y0, x + sg * r * s, y1)
        if h["style"] == "tap" and rt > r:
            page.line(x + sg * rt * s, y0, x + sg * rt * s, y1, THIN, dash=HIDDEN_DASH)
    blind = lo > 1e-9 and entry_top or (not entry_top and hi < T.y(0) and False)
    if blind and h["style"] != "nest":
        tip = min(r * s * .58, (y1 - y0) * .3)
        page.polygon([(x - r * s, y1), (x, y1 + tip), (x + r * s, y1)], HID, BLACK, None, closed=False, dash=HIDDEN_DASH)
    elif blind:
        _hid(page, x - r * s, y1, x + r * s, y1)
    page.line(x, ytop - 3, x, ybot + 3, THIN, dash=CENTER_DASH)


def _horizontal_hole_lines(page: Page, T: Xf, zc: float, r: float, u0: float, u1: float, seen: _Seen) -> None:
    if not seen.once(zc, u0, u1, r, 7):
        return
    x0, x1 = T.x(u0), T.x(u1)
    for sg in (-1, 1):
        y = T.y(zc + sg * r)
        _hid(page, x0, y, x1, y)
    y = T.y(zc)
    page.line(x0 - 3, y, x1 + 3, y, THIN, dash=CENTER_DASH)


def _pris_top(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    s = T.s
    page.polygon([T.p(u, v) for u, v in _chamfer_outline(L, W, p.corner_ch)], THICK)
    if p.profile == "bracket":
        page.line(T.x(p.t2), T.y(0), T.x(p.t2), T.y(W), MED)
    if p.profile == "heatsink":
        for x0, x1 in p.fins:
            if T.hidden((x0 + x1) / 2):
                continue
            page.line(T.x(x0), T.y(0), T.x(x0), T.y(W), MED)
            page.line(T.x(x1), T.y(0), T.x(x1), T.y(W), MED)
    if p.profile == "cover":
        e = p.inset
        page.rect(T.x(e), T.y(W - e), T.x(L - e) - T.x(e), T.y(e) - T.y(W - e), HID, BLACK, None, HIDDEN_DASH)
    for side, c in p.edge_ch:
        page.line(T.x(c), T.y(0), T.x(c), T.y(W), MED)
    for k, c in p.corner_ch.items():
        pass
    for q in p.pockets:
        _rrect(page, T.x(q["x0"]), T.y(q["y1"]), T.x(q["x1"]), T.y(q["y0"]), q["rad"] * s, MED)
    for g in p.grooves:
        w2 = g["wid"] / 2
        if g["kind"] == "circle":
            for dr in (-w2, w2):
                page.circle(T.x(g["cx"]), T.y(g["cy"]), (g["R"] + dr) * s, MED)
        else:
            for dr in (-w2, w2):
                _rrect(page, T.x(g["x0"] - dr), T.y(g["y1"] + dr), T.x(g["x1"] + dr), T.y(g["y0"] - dr),
                       max(g["rad"] + dr, 0) * s, MED)
    for sl in p.slots:
        hw, hl = (sl["wid"] / 2, sl["len"] / 2) if sl["vertical"] else (sl["len"] / 2, sl["wid"] / 2)
        x0, x1 = T.x(sl["cx"] - hw), T.x(sl["cx"] + hw)
        y0, y1 = T.y(sl["cy"] + hl), T.y(sl["cy"] - hl)
        _rrect(page, x0, y0, x1, y1, sl["wid"] / 2 * s, MED)
        cx, cy = T.p(sl["cx"], sl["cy"])
        if sl["vertical"]:
            page.line(cx, y0 - 3, cx, y1 + 3, THIN, dash=CENTER_DASH)
        else:
            page.line(x0 - 3, cy, x1 + 3, cy, THIN, dash=CENTER_DASH)
    for o in p.open_slots:
        for x in (o["cx"] - o["wid"] / 2, o["cx"] + o["wid"] / 2):
            page.line(T.x(x), T.y(0), T.x(x), T.y(W), MED)
    seen = _Seen()
    for h in p.holes:
        if h["style"] == "spot" and h["axis"] != "z":
            continue
        if h["axis"] == "z":
            if T.hidden(h["a"]):
                continue
            _hole_circles(page, T.x(h["a"]), T.y(h["b"]), h, s, hidden=h["face"] == "bottom")
        elif h["axis"] == "y":
            if T.hidden(h["a"]) or not seen.once(h["a"], h["r"], 1):
                continue
            for sg in (-1, 1):
                _hid(page, T.x(h["a"] + sg * h["r"]), T.y(h["lo"]), T.x(h["a"] + sg * h["r"]), T.y(h["hi"]))
            page.line(T.x(h["a"]), T.y(h["lo"]) + 3, T.x(h["a"]), T.y(h["hi"]) - 3, THIN, dash=CENTER_DASH)
        else:
            _horizontal_hole_lines(page, T, h["a"], h["r"], h["lo"], h["hi"], seen)


def _pris_front_outline(p: Part) -> List[Pt]:
    L, W, H = p.d
    if p.profile == "bracket":
        tb, t2 = p.tb, p.t2
        rf = min(tb, t2) * .6
        pts = [(0, 0), (L, 0), (L, tb), (t2 + rf, tb)]
        for k in range(1, 6):
            a = math.pi / 2 * k / 6
            pts.append((t2 + rf - rf * math.sin(a), tb + rf - rf * math.cos(a)))
        pts += [(t2, tb + rf), (t2, H), (0, H)]
        return pts
    if p.profile == "heatsink":
        pts = [(0, 0), (L, 0), (L, p.base)]
        for x0, x1 in reversed(p.fins):
            pts += [(x1, p.base), (x1, H), (x0, H), (x0, p.base)]
        pts.append((0, p.base))
        return pts
    if p.profile == "cover":
        e, lip = p.inset, p.lip
        return [(e, 0), (L - e, 0), (L - e, lip), (L, lip), (L, H), (0, H), (0, lip), (e, lip)]
    top = p.top
    pts = [(0, 0), (L, 0), (L, top)]
    for o in sorted(p.open_slots, key=lambda o: -o["cx"]):
        a, b = o["cx"] + o["wid"] / 2, o["cx"] - o["wid"] / 2
        pts += [(a, top), (a, top - o["depth"]), (b, top - o["depth"]), (b, top)]
    ec = next((c for side, c in p.edge_ch if side == "left"), 0.0)
    if ec > 0:
        pts += [(ec, top), (0, top - ec)]
    else:
        pts.append((0, top))
    return pts


def _pris_side_common(page: Page, T: Xf, p: Part, view: str) -> None:
    """Hidden features for the front view (view 'front', u = x) or the right view (view 'right', u = y)."""
    L, W, H = p.d
    s = T.s
    seen = _Seen()
    for q in p.pockets:
        a0, a1 = (q["x0"], q["x1"]) if view == "front" else (q["y0"], q["y1"])
        zf = p.top - q["depth"]
        if not seen.once(a0, a1, zf, 3):
            continue
        for a in (a0, a1):
            _hid(page, T.x(a), T.y(p.top), T.x(a), T.y(zf))
        _hid(page, T.x(a0), T.y(zf), T.x(a1), T.y(zf))
    for sl in p.slots:
        hw, hl = (sl["wid"] / 2, sl["len"] / 2) if sl["vertical"] else (sl["len"] / 2, sl["wid"] / 2)
        c, half = (sl["cx"], hw) if view == "front" else (sl["cy"], hl)
        zb = p.top - (sl["depth"] or p.top)
        if not seen.once(c, half, zb, 4):
            continue
        for a in (c - half, c + half):
            _hid(page, T.x(a), T.y(p.top), T.x(a), T.y(zb))
    if view == "right":
        for o in p.open_slots:
            y = T.y(p.top - o["depth"])
            _hid(page, T.x(0), y, T.x(W), y)
    for h in p.holes:
        if h["style"] == "spot" and not ((h["axis"] == "y" and view == "front") or (h["axis"] == "x" and view == "right")):
            continue  # a pad face: nothing to see edge-on
        if h["axis"] == "z":
            a = h["a"] if view == "front" else h["b"]
            if view == "front" and T.hidden(a):
                continue
            _vertical_hole_lines(page, T, h, a, seen)
        elif h["axis"] == "y":
            if view == "front":
                if T.hidden(h["a"]):
                    continue
                _hole_circles(page, T.x(h["a"]), T.y(h["b"]), h, s)
            else:
                _horizontal_hole_lines(page, T, h["b"], h["r"], h["lo"], h["hi"], seen)
        else:
            if view == "right":
                _hole_circles(page, T.x(h["a"]), T.y(h["b"]), h, s)
            else:
                _horizontal_hole_lines(page, T, h["b"], h["r"], h["lo"], h["hi"], seen)


def _pris_front(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    page.polygon([T.p(u, v) for u, v in _pris_front_outline(p)], THICK)
    for k, c in p.corner_ch.items():
        if k in (0, 3):
            x = L - c if k == 0 else c
            page.line(T.x(x), T.y(0), T.x(x), T.y(p.top), MED)
    if p.profile == "cover":
        page.line(T.x(p.inset), T.y(p.lip), T.x(L - p.inset), T.y(p.lip), MED)
    _pris_side_common(page, T, p, "front")


def _pris_right(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    if p.profile == "cover":
        e, lip = p.inset, p.lip
        pts = [(e, 0), (W - e, 0), (W - e, lip), (W, lip), (W, H), (0, H), (0, lip), (e, lip)]
    else:
        pts = [(0, 0), (W, 0), (W, H), (0, H)]
    page.polygon([T.p(u, v) for u, v in pts], THICK)
    if p.profile == "bracket":
        page.line(T.x(0), T.y(p.tb), T.x(W), T.y(p.tb), MED)
    if p.profile == "heatsink":
        page.line(T.x(0), T.y(p.base), T.x(W), T.y(p.base), MED)
    for k, c in p.corner_ch.items():
        if k in (0, 1):
            y = c if k == 0 else W - c
            page.line(T.x(y), T.y(0), T.x(y), T.y(p.top), MED)
    _pris_side_common(page, T, p, "right")


def _views_prismatic(p: Part) -> List[View]:
    L, W, H = p.d
    f = p.f
    top = View("top", "TOP VIEW", 0, L, 0, W, 0, 0, lambda pg, T: _pris_top(pg, T, p))
    front = View("front", "FRONT VIEW", 0, L, 0, H, 0, 1, lambda pg, T: _pris_front(pg, T, p), thin=True)
    right = View("right", "RIGHT SIDE VIEW", 0, W, 0, H, 1, 1, lambda pg, T: _pris_right(pg, T, p), thin=True)
    top.dim("h", "above", 2, 0, L, f(L))
    top.dim("v", "left", 2, 0, W, f(W))
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for h in p.holes:
        if h["face"] == "top":
            groups.setdefault(h["grp"], []).append(h)
    if groups:
        grp = max(groups.values(), key=lambda g: (len({round(h["a"], 6) for h in g}) + len({round(h["b"], 6) for h in g}), len(g)))
        xs = sorted({round(h["a"], 9) for h in grp})
        ys = sorted({round(h["b"], 9) for h in grp})
        if xs:
            top.dim("h", "above", 1, 0, xs[0], f(xs[0]), True)
        if len(xs) >= 2:
            top.dim("h", "above", 1, xs[0], xs[-1], f(xs[-1] - xs[0]), True)
        if ys:
            top.dim("v", "left", 1, 0, ys[0], f(ys[0]), True)
        if len(ys) >= 2:
            top.dim("v", "left", 1, ys[0], ys[-1], f(ys[-1] - ys[0]), True)
    elif p.pockets:
        q = p.pockets[0]
        top.dim("h", "above", 1, q["x0"], q["x1"], f(q["x1"] - q["x0"]), True)
        top.dim("v", "left", 1, q["y0"], q["y1"], f(q["y1"] - q["y0"]), True)
    front.dim("v", "left", 1, 0, H, f(H))
    if p.profile == "bracket":
        right.dim("v", "right", 1, 0, p.tb, f(p.tb))
    elif p.profile == "heatsink":
        right.dim("v", "right", 1, 0, p.base, f(p.base))
    elif p.profile == "housing":
        right.dim("v", "right", 1, 0, p.floor, f(p.floor))
    elif p.profile == "cover":
        right.dim("v", "right", 1, 0, p.lip, f(p.lip))
    elif p.pockets:
        right.dim("v", "right", 1, p.top - p.pockets[0]["depth"], p.top, f(p.pockets[0]["depth"]))
    return [top, front, right]


# --------------------------------------------------------------------------- #
# Round views: side view, end view, and a hatched SECTION A-A
# --------------------------------------------------------------------------- #
def _round_side(page: Page, T: Xf, p: Part) -> None:
    L = p.d[0]
    s = T.s
    for sg in p.segs:
        x0, x1, r0, r1 = sg["x0"], sg["x1"], sg["r"], sg["r1"]
        cl, cr = sg.get("chL", 0.0), sg.get("chR", 0.0)
        pts = [(x0, r0 - cl), (x0 + cl, r0), (x1 - cr, r1), (x1, r1 - cr), (x1, -(r1 - cr)), (x1 - cr, -r1), (x0 + cl, -r0),
               (x0, -(r0 - cl))]
        page.polygon([T.p(u, v) for u, v in pts], THICK)
        if cl > 0:
            page.line(T.x(x0 + cl), T.y(r0), T.x(x0 + cl), T.y(-r0), MED)
        if cr > 0:
            page.line(T.x(x1 - cr), T.y(r1), T.x(x1 - cr), T.y(-r1), MED)
        k = sg["kind"]
        if k == "thread":
            mi = sg["minor"]
            for v in (mi, -mi):
                page.line(T.x(x0 + (cl if cl else 0)), T.y(v), T.x(x1 - (cr if cr else 0)), T.y(v), THIN)
        elif k == "hex":
            for v in (r0 * .5, -r0 * .5):
                page.line(T.x(x0), T.y(v), T.x(x1), T.y(v), MED)
        elif k == "flats":
            af = sg.get("af", r0 * 1.6) / 2
            xa, xb, ya, yb = T.x(x0), T.x(x1), T.y(af), T.y(-af)
            page.line(xa, ya, xb, ya, MED)
            page.line(xa, yb, xb, yb, MED)
            page.line(xa, ya, xb, yb, THIN)
            page.line(xa, yb, xb, ya, THIN)
        elif k == "knurl":
            poly = [T.p(x0 + cl, r0), T.p(x1 - cr, r0), T.p(x1 - cr, -r0), T.p(x0 + cl, -r0)]
            _hatch(page, poly, 2.8, 30, 0.3)
            _hatch(page, poly, 2.8, -30, 0.3)
    for kw in p.keyways:
        x0, x1 = T.x(kw["x0"]), T.x(kw["x1"])
        w2 = kw["wid"] / 2 * T.sy
        cy = T.y(0)
        _rrect(page, x0, cy - w2, x1, cy + w2, w2, MED)
    for tn in p.tines:
        x0, x1 = T.x(tn["x0"]), T.x(tn["x1"])
        for v in (tn["wid"] / 2, -tn["wid"] / 2):
            page.line(x0, T.y(v), x1, T.y(v), MED)
        page.arc(x0, T.y(0), tn["wid"] / 2 * T.sy, 90, 270, MED)
    for ch in p.cross:
        cx, cy = T.p(ch["x"], 0)
        page.circle(cx, cy, max(ch["r"] * s, .8), MED)
        if ch.get("rt"):
            _thread_arc(page, cx, cy, ch["rt"] * s)
        _cmark(page, cx, cy, ch.get("rt", ch["r"]) * s)
    for b in p.bore:
        x0, x1 = b["x0"], b["x1"]
        ra = b["r"]
        rb = b["r"] if b["r1"] is None else b["r1"]
        for sgn in (1, -1):
            _hid(page, T.x(x0), T.y(sgn * ra), T.x(x1), T.y(sgn * rb))
            if b["kind"] == "thread" and b.get("major"):
                page.line(T.x(x0), T.y(sgn * b["major"]), T.x(x1), T.y(sgn * b["major"]), THIN, dash=HIDDEN_DASH)
        inner_end = x0 if x1 >= L - 1e-9 else x1 if x0 <= 1e-9 else None
        if inner_end is not None and 1e-9 < inner_end < L - 1e-9 and not any(
                o is not b and (o["x0"] - 1e-9 <= inner_end <= o["x1"] + 1e-9) for o in p.bore):
            r = ra if inner_end == x0 else rb
            xe = T.x(inner_end)
            tip = r * s * .58 * (1 if inner_end == x1 else -1)
            page.polygon([(xe, T.y(r)), (xe + tip, T.y(0)), (xe, T.y(-r))], HID, BLACK, None, closed=False, dash=HIDDEN_DASH)
    seen_v = _Seen()
    for fh in p.face_holes:
        # holes through the end face, seen edge-on: hidden lines across the steps they pass through
        xa, xb = _face_hole_span(p, fh)
        if xb > xa and seen_v.once(round(fh["v"], 6)):
            for v in (fh["v"] - fh["r"], fh["v"] + fh["r"]):
                _hid(page, T.x(xa), T.y(v), T.x(xb), T.y(v))
    page.line(T.x(0) - 7, T.y(0), T.x(L) + 7, T.y(0), THIN, dash=CENTER_DASH)
    # diameters of the main steps, inside the view where they fit
    rmax = max(sg["r"] for sg in p.segs)
    done: List[float] = [rmax]
    # the label sits on the axis: keep it off features drawn there (keyways, slots, cross holes), and off
    # bores whose hidden lines would run under its box
    hard = [(kw["x0"], kw["x1"]) for kw in p.keyways] + [(c["x"] - c["r"] * 2, c["x"] + c["r"] * 2) for c in p.cross] \
        + [(tn["x0"], tn["x1"]) for tn in p.tines]
    soft = [(b["x0"], b["x1"]) for b in p.bore if max(b["r"], b.get("major") or 0.0) * T.sy < DIM * 0.75 + 2.5]
    for sg in sorted(p.segs, key=lambda sg: -(sg["x1"] - sg["x0"])):
        if sg["kind"] not in ("plain", "groove") or any(abs(sg["r"] - d) < rmax * .02 for d in done):
            continue
        if T.hidden((sg["x0"] + sg["x1"]) / 2):
            continue
        lab = "Ø" + p.f(2 * sg["r"])
        wpt = T.x(sg["x1"]) - T.x(sg["x0"])
        hpt = 2 * sg["r"] * T.sy
        tw = text_width(lab, DIM)
        if wpt < tw + 8 or hpt < 22:
            continue
        half = (tw / 2 + 2) / max(T.s, 1e-9)  # the label's half width, in part units
        xm = None
        for spans in (hard + soft, hard):
            for frac in (.5, .3, .7, .2, .8):
                x_ = sg["x0"] + (sg["x1"] - sg["x0"]) * frac
                if sg["x0"] <= x_ - half and x_ + half <= sg["x1"] and \
                        not any(a - half < x_ < b + half for a, b in spans):
                    xm = x_
                    break
            if xm is not None:
                break
        ly = None
        if xm is None:
            # every spot on the axis would cover a slot or keyway: put the figures in the upper half instead
            if hpt < 30:
                continue
            xm = (sg["x0"] + sg["x1"]) / 2
            ly = T.y(sg["r"]) + hpt * 0.24
        _dia_dim(page, T.x(xm), T.y(sg["r"]), T.y(-sg["r"]), lab, ly)
        done.append(sg["r"])
        if len(done) >= 4:
            break


def _face_hole_span(p: Part, fh: Dict[str, Any]) -> Tuple[float, float]:
    """x range of the material an axial face hole passes through (the steps whose radius covers it)."""
    need = math.hypot(fh["u"], fh["v"]) + fh["r"]
    xs = [(sg["x0"], sg["x1"]) for sg in p.segs if min(sg["r"], sg["r1"]) >= need]
    if not xs:
        return 0.0, 0.0
    return min(a for a, _ in xs), max(b for _, b in xs)


def _round_end(page: Page, T: Xf, p: Part) -> None:
    s = T.s
    cx, cy = T.p(0, 0)
    R = p.rmax
    seen_r = -1.0
    for sg in reversed(p.segs):
        rf = max(sg["r"], sg["r1"])
        if rf <= seen_r + 1e-9:
            continue
        lw = THICK if abs(rf - R) < 1e-9 else MED
        if sg["kind"] == "hex":
            pts = [(cx + rf * s * math.cos(math.radians(90 + 60 * k)), cy - rf * s * math.sin(math.radians(90 + 60 * k)))
                   for k in range(6)]
            page.polygon(pts, lw)
            page.circle(cx, cy, rf * s * math.cos(math.pi / 6) * .96, THIN)
        elif sg["kind"] == "flats":
            af = sg.get("af", rf * 1.6) / 2
            a = math.degrees(math.acos(_clamp(af / rf, -1, 1)))
            page.arc(cx, cy, rf * s, a, 180 - a, lw)
            page.arc(cx, cy, rf * s, 180 + a, 360 - a, lw)
            h = math.sqrt(max(rf * rf - af * af, 0)) * s
            page.line(cx + af * s, cy - h, cx + af * s, cy + h, lw)
            page.line(cx - af * s, cy - h, cx - af * s, cy + h, lw)
        else:
            page.circle(cx, cy, rf * s, lw)
            if sg is p.segs[-1] and sg.get("chR", 0) > 0:
                page.circle(cx, cy, (sg["r1"] - sg["chR"]) * s, MED)
            if sg is p.segs[-1] and sg["kind"] == "thread":
                _thread_arc(page, cx, cy, sg["minor"] * s)
        seen_r = rf
    L = p.d[0]
    for b in p.bore:
        if b["x1"] < L - 1e-9:
            continue
        r = b["r"] if b["r1"] is None else b["r1"]
        if b["kind"] == "hex" and p.hex_socket:
            af = p.hex_socket["af"] / 2
            rc = af / math.cos(math.pi / 6)
            page.polygon([(cx + rc * s * math.cos(math.radians(60 * k)), cy - rc * s * math.sin(math.radians(60 * k)))
                          for k in range(6)], MED)
        else:
            page.circle(cx, cy, max(r * s, .7), MED)
            if b["kind"] == "thread" and b.get("major"):
                _thread_arc(page, cx, cy, b["major"] * s)
    open_bores = [b for b in p.bore if b["x0"] <= 1e-9 and b["x1"] >= L - 1e-9]
    if p.bore_key and open_bores:
        rin = min(b["r"] for b in open_bores)
        w2, dp = p.bore_key["wid"] / 2 * s, p.bore_key["depth"] * s
        yb = cy - math.sqrt(max((rin * s) ** 2 - w2 * w2, 0))
        page.polygon([(cx - w2, yb), (cx - w2, cy - rin * s - dp), (cx + w2, cy - rin * s - dp), (cx + w2, yb)], MED,
                     BLACK, None, closed=False)
    for fh in p.face_holes:
        hx, hy = cx + fh["u"] * s, cy - fh["v"] * s
        page.circle(hx, hy, max(fh["r"] * s, .7), MED)
        if fh["tap"]:
            _thread_arc(page, hx, hy, fh["r"] * s * 1.22)
        _cmark(page, hx, hy, fh["r"] * s)
    if p.face_holes:
        bc = math.hypot(p.face_holes[0]["u"], p.face_holes[0]["v"]) * s
        page.circle(cx, cy, bc, THIN, dash=CENTER_DASH)
    ext = R * s + 5
    page.line(cx - ext, cy, cx + ext, cy, THIN, dash=CENTER_DASH)
    page.line(cx, cy - ext, cx, cy + ext, THIN, dash=CENTER_DASH)
    if p.has_section:
        top, bot = cy - R * s - 10, cy + R * s + 10
        page.line(cx, top, cx, bot, 0.9, dash=CUT_DASH)
        for yy, ty in ((top, top - 3.5), (bot, bot + 9.5)):
            page.line(cx, yy, cx + 8, yy, 0.9)
            _arrow(page, cx + 12, yy, 1, 0, 4.6)
            page.text(cx + 14, ty, "A", 8, True)


def _section_polys(p: Part) -> Tuple[List[Pt], List[Pt]]:
    outer = _outer_profile(p)
    inner = _inner_profile(p)
    upper = outer + list(reversed(inner))
    lower = [(x, -r) for x, r in upper]
    return upper, lower


def _round_section(page: Page, T: Xf, p: Part) -> None:
    L = p.d[0]
    upper, lower = _section_polys(p)
    for poly in (upper, lower):
        pts = [T.p(u, v) for u, v in poly]
        _hatch(page, pts, 3.0, 45)
        page.polygon(pts, THICK)
    inner = _inner_profile(p)
    for (xa, ra), (xb, rb) in zip(inner, inner[1:]):
        if abs(xa - xb) < 1e-12 and abs(ra - rb) > 1e-12 and 1e-9 < xa < L - 1e-9:
            r = max(ra, rb)
            page.line(T.x(xa), T.y(r), T.x(xa), T.y(-r), MED)
    for b in p.bore:
        if b["kind"] == "thread" and b.get("major"):
            for sg in (1, -1):
                page.line(T.x(b["x0"]), T.y(sg * b["major"]), T.x(b["x1"]), T.y(sg * b["major"]), THIN)
    for fh in p.face_holes:
        if abs(fh["u"]) >= fh["r"]:
            continue  # the cut (vertical, through the axis) misses this hole
        xa, xb = _face_hole_span(p, fh)
        if xb <= xa:
            continue
        va, vb = fh["v"] - fh["r"], fh["v"] + fh["r"]
        page.rect(T.x(xa), T.y(vb), T.x(xb) - T.x(xa), T.y(va) - T.y(vb), 0, None, WHITE)
        for v in (va, vb):
            page.line(T.x(xa), T.y(v), T.x(xb), T.y(v), MED)
    if p.oil_groove and p.bore:
        rin = min(b["r"] for b in p.bore)
        n = 4
        for k in range(n):
            xa = L * (.12 + .76 * k / n)
            page.line(T.x(xa), T.y(-rin), T.x(xa + L * .76 / n * .8), T.y(rin), MED)
    page.line(T.x(0) - 7, T.y(0), T.x(L) + 7, T.y(0), THIN, dash=CENTER_DASH)


def _views_round(p: Part, stubby: bool) -> List[View]:
    L = p.d[0]
    R = p.rmax
    f = p.f
    side = View("side", "SIDE VIEW", 0, L, -R, R, 0, 0, lambda pg, T: _round_side(pg, T, p))
    end = View("end", "END VIEW", -R, R, -R, R, 2 if (stubby and p.has_section) else 1, 0,
               lambda pg, T: _round_end(pg, T, p))
    views = [side, end]
    if p.has_section:
        end.extra["t"] = 16.0
        end.extra["b"] = 16.0
    pos = [(s["x0"], s["x1"]) for s in p.segs]
    if len(pos) > 1:
        for x0, x1 in pos:
            side.dim("h", "below", 1, x0, x1, f(x1 - x0), True)
    side.dim("h", "below", 2 if len(pos) > 1 else 1, 0, L, f(L))
    side.dim("v", "left", 1, -R, R, "Ø" + f(2 * R))
    if p.has_section:
        sec = View("section", "SECTION A-A", 0, L, -R, R, 1 if stubby else 0, 0 if stubby else 1,
                   lambda pg, T: _round_section(pg, T, p), title_size=7.5)
        bores = [b for b in p.bore if b["x1"] - b["x0"] > L * .3]
        if bores:
            b = max(bores, key=lambda b: b["x1"] - b["x0"])
            sec.dim("v", "left", 1, -b["r"], b["r"], "Ø" + f(2 * b["r"]))
        views.append(sec)
    return views


# --------------------------------------------------------------------------- #
# Complex part views
# --------------------------------------------------------------------------- #
def _poly(page: Page, T: Xf, pts: Sequence[Pt], lw: float = THICK, closed: bool = True,
          dash: Optional[Sequence[float]] = None) -> None:
    page.polygon([T.p(u, v) for u, v in pts], lw, BLACK, None, closed, dash)


def _rect(page: Page, T: Xf, u0: float, v0: float, u1: float, v1: float, lw: float = THICK,
          dash: Optional[Sequence[float]] = None) -> None:
    _poly(page, T, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)], lw, True, dash)


def _hole_uv(page: Page, T: Xf, u: float, v: float, r: float, r2: float = 0.0, tap: float = 0.0) -> None:
    cx, cy = T.p(u, v)
    if r2 > r:
        page.circle(cx, cy, r2 * T.s, MED)
    page.circle(cx, cy, max(r * T.s, .7), MED)
    if tap > r:
        _thread_arc(page, cx, cy, tap * T.s)
    _cmark(page, cx, cy, max(r, r2, tap) * T.s)


def _fit_top(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    mn = min(L, W)
    if p.variant == "spar":
        _rect(page, T, 0, 0, L, W)
        _rect(page, T, 0, W / 2 - p.wc / 2, L, W / 2 + p.wc / 2, MED)
        for y in (W / 2 - p.tw / 2, W / 2 + p.tw / 2):
            _hid(page, T.x(0), T.y(y), T.x(L), T.y(y))
        for x, y, r in p.tool:
            _hole_uv(page, T, x, y, r)
        return
    _rrect(page, T.x(0), T.y(W), T.x(L), T.y(0), mn * .06 * T.s, THICK)
    _rect(page, T, p.xl0, p.yl0, p.xl1, p.yl1, MED)
    for x0, x1 in p.gus:
        _rect(page, T, x0, p.yl1, x1, W * .9, MED)
        _rect(page, T, x0, W * .1, x1, p.yl0, MED)
    for x0, y0, x1, y1 in p.pockets:
        _rrect(page, T.x(x0), T.y(y1), T.x(x1), T.y(y0), p.pk_rad * T.s, MED)
    for x, y in p.fholes:
        _hole_uv(page, T, x, y, p.hole_r, p.hole_r2)
    for x, y, r, r2 in p.fholes2:
        _hole_uv(page, T, x, y, r, r2)
    for sg in (-1, 1):
        _hid(page, T.x(L / 2 + sg * p.rb), T.y(p.yl0), T.x(L / 2 + sg * p.rb), T.y(p.yl1))
    page.line(T.x(L / 2), T.y(p.yl0) + 4, T.x(L / 2), T.y(p.yl1) - 4, THIN, dash=CENTER_DASH)


def _fit_front(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    if p.variant == "spar":
        _rect(page, T, 0, 0, L, H)
        for z in (p.tf, H - p.tc):
            page.line(T.x(0), T.y(z), T.x(L), T.y(z), MED)
        for x in p.light:
            cx, cy = T.p(x, p.zh)
            page.circle(cx, cy, p.rl * T.s, MED)
            page.circle(cx, cy, p.rl * T.s * 1.16, THIN)
            _cmark(page, cx, cy, p.rl * T.s)
        for x in p.stiff:
            _rect(page, T, x - p.tw / 2, p.tf, x + p.tw / 2, H - p.tc, MED)
        return
    _rect(page, T, 0, 0, L, p.tf)
    out = [(p.xl0, p.tf), (p.xl1, p.tf)]
    for k in range(25):
        a = math.pi * k / 24
        out.append((L / 2 + p.rlug * math.cos(a), p.zc + p.rlug * math.sin(a)))
    _poly(page, T, out, THICK)
    cx, cy = T.p(L / 2, p.zc)
    page.circle(cx, cy, p.rb * T.s, MED)
    _cmark(page, cx, cy, p.rb * T.s)
    for x0, x1 in p.gus:
        _rect(page, T, x0, p.tf, x1, p.tf + p.hg, MED)
    for x0, y0, x1, y1 in p.pockets[:1]:
        _rect(page, T, x0, p.tf - p.pk_depth, x1, p.tf, HID, HIDDEN_DASH)
    seen = _Seen()
    for x, y, r in [(x, y, p.hole_r) for x, y in p.fholes] + [(x, y, r) for x, y, r, _ in p.fholes2]:
        if seen.once(x):
            for sg in (-1, 1):
                _hid(page, T.x(x + sg * r), T.y(0), T.x(x + sg * r), T.y(p.tf))


def _fit_right(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    if p.variant == "spar":
        pts = [(0, 0), (W, 0), (W, p.tf), (W / 2 + p.tw / 2, p.tf), (W / 2 + p.tw / 2, H - p.tc), (W / 2 + p.wc / 2, H - p.tc),
               (W / 2 + p.wc / 2, H), (W / 2 - p.wc / 2, H), (W / 2 - p.wc / 2, H - p.tc), (W / 2 - p.tw / 2, H - p.tc),
               (W / 2 - p.tw / 2, p.tf), (0, p.tf)]
        _poly(page, T, pts)
        if p.stiff:
            for y in (W / 2 - p.wc / 2, W / 2 + p.wc / 2):
                page.line(T.x(y), T.y(p.tf), T.x(y), T.y(H - p.tc), MED)
        return
    _rect(page, T, 0, 0, W, p.tf)
    _rect(page, T, p.yl0, p.tf, p.yl1, H)
    for y0, y1 in ((p.yl1, W * .9), (p.yl0, W * .1)):
        page.line(T.x(y0), T.y(p.tf + p.hg), T.x(y1), T.y(p.tf), THICK)
    for sg in (-1, 1):
        _hid(page, T.x(p.yl0), T.y(p.zc + sg * p.rb), T.x(p.yl1), T.y(p.zc + sg * p.rb))
    page.line(T.x(p.yl0) - 4, T.y(p.zc), T.x(p.yl1) + 4, T.y(p.zc), THIN, dash=CENTER_DASH)
    for x0, y0, x1, y1 in p.pockets:
        _rect(page, T, y0, p.tf - p.pk_depth, y1, p.tf, HID, HIDDEN_DASH)


def _imp_top(page: Page, T: Xf, p: Part) -> None:
    R = p.R
    cx, cy = T.p(R, R)
    s = T.s
    page.circle(cx, cy, R * s, THICK)
    page.circle(cx, cy, p.r_ind * s, THIN)
    page.circle(cx, cy, p.rh * s, MED)
    page.circle(cx, cy, p.rb * s, MED)
    thick = R * .035
    for k in range(p.nm + p.ns):
        main = k < p.nm
        idx, n = (k, p.nm) if main else (k - p.nm, p.ns)
        off = 0.0 if main else math.pi / p.nm
        t0 = 0.0 if main else .42
        for side in (-1, 1):
            pts = []
            for j in range(13):
                t = t0 + (1 - t0) * j / 12
                r, _ = _shroud(p, t)
                th = _blade_theta(t, idx, n) + off + side * thick / 2 / max(r, 1e-9)
                pts.append((cx + r * s * math.cos(th), cy - r * s * math.sin(th)))
            page.polygon(pts, MED, BLACK, None, closed=False)
    ext = R * s + 5
    page.line(cx - ext, cy, cx + ext, cy, THIN, dash=CENTER_DASH)
    page.line(cx, cy - ext, cx, cy + ext, THIN, dash=CENTER_DASH)


def _imp_front(page: Page, T: Xf, p: Part) -> None:
    R, H = p.R, p.d[2]
    right = [_shroud(p, 1 - k / 16) for k in range(17)]
    pts = [(0, 0), (2 * R, 0), (2 * R, p.tb)] + [(R + r, z) for r, z in right]
    pts += [(R + p.rh, p.z_top), (R + p.rh, H), (R - p.rh, H), (R - p.rh, p.z_top)]
    pts += [(R - r, z) for r, z in reversed(right)] + [(0, p.tb)]
    _poly(page, T, pts)
    page.line(T.x(0), T.y(p.tb), T.x(2 * R), T.y(p.tb), MED)
    for k in range(p.nm):
        th = _blade_theta(1.0, k, p.nm)
        if math.sin(th) > -0.05:
            continue
        xs = [R + _shroud(p, t / 10)[0] * math.cos(_blade_theta(t / 10, k, p.nm)) for t in range(11)]
        zs = [_shroud(p, t / 10)[1] for t in range(11)]
        page.polygon([T.p(x, z) for x, z in zip(xs, zs)], MED, BLACK, None, closed=False)
        page.line(T.x(xs[-1]), T.y(p.tb), T.x(xs[-1]), T.y(zs[-1]), MED)
    for sg in (-1, 1):
        _hid(page, T.x(R + sg * p.rb), T.y(0), T.x(R + sg * p.rb), T.y(H))
    page.line(T.x(R), T.y(H) - 5, T.x(R), T.y(0) + 5, THIN, dash=CENTER_DASH)


def _bp_outline(p: Part) -> Tuple[List[Pt], List[Pt]]:
    L, W = p.d[0], p.d[1]
    xs = [L * k / 80 for k in range(81)]
    up = [(x, W / 2 + _plate_width_r(p, x) / 2) for x in xs]
    lo = [(x, W / 2 - _plate_width_r(p, x) / 2) for x in xs]
    return up, lo


def _plate_width_r(p: Part, x: float) -> float:
    """Plate width with rounded ends (the drawing and the mesh share it)."""
    L = p.d[0]
    w = _plate_width(p, x)
    d = min(L * .035, w * .5)
    e = min(x, L - x)
    if e < d:
        w *= max(math.sqrt(max(0.0, 1 - ((d - e) / d) ** 2)), .3)
    return w


def _bp_top(page: Page, T: Xf, p: Part) -> None:
    up, lo = _bp_outline(p)
    _poly(page, T, up + list(reversed(lo)))
    W = p.d[1]
    for x in p.bp_holes:
        _hole_uv(page, T, x, W / 2, p.bp_r * .88, 0.0, p.bp_r)
    for x in p.bp_k:
        _hole_uv(page, T, x, W / 2, p.bp_kr)
    page.line(T.x(0) - 5, T.y(W / 2), T.x(p.d[0]) + 5, T.y(W / 2), THIN, dash=CENTER_DASH)


def _bp_front(page: Page, T: Xf, p: Part) -> None:
    L = p.d[0]
    t = p.bp_t
    endt = next((c for c in p.cs if "TAPER" in c["T"] and c["num"]), None)
    xs = [L * k / 80 for k in range(81)]

    def tf(x: float) -> float:
        u = min(x, L - x) / (L * .08)
        return (.55 + .45 * _clamp(u, 0, 1)) if endt else 1.0

    bot = [(x, _plate_z(p, x)) for x in xs]
    top = [(x, _plate_z(p, x) + t * tf(x)) for x in xs]
    _poly(page, T, bot + list(reversed(top)))
    for x in p.bp_holes + p.bp_k:
        r = p.bp_r if x in p.bp_holes else p.bp_kr
        z0 = _plate_z(p, x)
        for sg in (-1, 1):
            _hid(page, T.x(x + sg * r), T.y(z0), T.x(x + sg * r), T.y(z0 + t * tf(x)))


def _bp_right(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    t = p.bp_t
    wend = _plate_width(p, L)
    _rrect(page, T.x(0), T.y(H), T.x(W), T.y(0), t * .5 * T.s, THICK)
    _rrect(page, T.x(W / 2 - wend / 2), T.y(H), T.x(W / 2 + wend / 2), T.y(H - t), t * .45 * T.s, MED)
    page.line(T.x(0), T.y(t), T.x(W), T.y(t), MED)


def _cage_z(p: Part, x: float, z: float) -> float:
    """Height of the body surface at x for a nominal z in [0, H] (lordosis and bullet nose), inside the teeth."""
    L, W, H = p.d
    k = _cage_k(p, x)
    top = _cage_top(p, x)
    t = getattr(p, "tooth_h", 0.0)
    return t + (top / 2 + (z - H / 2) * (top / H) * (.7 + .3 * k)) * (H - 2 * t) / H


def _cage_top_v(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    xs = [L * k / 60 for k in range(61)]
    up = [(x, W / 2 + W / 2 * _cage_k(p, x)) for x in xs]
    lo = [(x, W / 2 - W / 2 * _cage_k(p, x)) for x in xs]
    _poly(page, T, up + list(reversed(lo)))
    x0, y0, x1, y1 = p.win
    _rrect(page, T.x(x0), T.y(y1), T.x(x1), T.y(y0), (y1 - y0) * .25 * T.s, MED)
    for x in p.teeth:
        k = _cage_k(p, x)
        page.line(T.x(x), T.y(W / 2 + W / 2 * k * .92), T.x(x), T.y(W / 2 - W / 2 * k * .92), THIN)
    for x, y in p.marks:
        _hole_uv(page, T, x, y, p.mark_r)


def _cage_front(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    xs = [L * k / 60 for k in range(61)]
    top = [(x, _cage_z(p, x, H)) for x in xs]
    bot = [(x, _cage_z(p, x, 0)) for x in xs]
    _poly(page, T, bot + list(reversed(top)))
    tw = L * .035
    for x in p.teeth:
        zt, zb = _cage_z(p, x, H), _cage_z(p, x, 0)
        _poly(page, T, [(x - tw, zt), (x + tw * .4, zt + p.tooth_h), (x + tw * .4, zt)], MED, False)
        _poly(page, T, [(x - tw, zb), (x + tw * .4, zb - p.tooth_h), (x + tw * .4, zb)], MED, False)
    x0, x1 = p.win[0], p.win[2]
    for x in (x0, x1):
        _hid(page, T.x(x), T.y(_cage_z(p, x, 0)), T.x(x), T.y(_cage_z(p, x, H)))
    dep = L * .16
    for sg in (-1, 1):
        _hid(page, T.x(L - dep), T.y(H / 2 + sg * p.ins_r * .82), T.x(L), T.y(H / 2 + sg * p.ins_r * .82))


def _cage_right(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    z0, z1 = _cage_z(p, L, 0), _cage_z(p, L, H)
    _rrect(page, T.x(0), T.y(z1), T.x(W), T.y(z0), min(W, H) * .12 * T.s, THICK)
    cx, cy = T.p(W / 2, H / 2)
    page.circle(cx, cy, p.ins_r * .82 * T.s, MED)
    _thread_arc(page, cx, cy, p.ins_r * T.s)
    _cmark(page, cx, cy, p.ins_r * T.s)
    for y in (p.win[1], p.win[3]):
        _hid(page, T.x(y), T.y(z0), T.x(y), T.y(z1))


def _cond_top(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    nx0, ny0, nx1, ny1 = p.notch
    r0 = min(W * .2, L * .25)
    c = W * .12
    hw = (ny1 - ny0) / 2
    pts: List[Pt] = [(r0, 0), (L - c, 0), (L, c), (L, W - c), (L - c, W), (r0, W)]
    pts += [(r0 - r0 * math.sin(math.pi / 2 * k / 6), W - r0 + r0 * math.cos(math.pi / 2 * k / 6)) for k in range(1, 7)]
    pts += [(0, ny1), (nx1 - hw, ny1)]
    pts += [(nx1 - hw + hw * math.sin(math.pi * k / 10), (ny0 + ny1) / 2 + hw * math.cos(math.pi * k / 10)) for k in range(1, 10)]
    pts += [(nx1 - hw, ny0), (0, ny0), (0, r0)]
    pts += [(r0 - r0 * math.cos(math.pi / 2 * k / 6), r0 - r0 * math.sin(math.pi / 2 * k / 6)) for k in range(1, 6)]
    _poly(page, T, pts)
    for x in sorted({pt[0] for pt in p.fc_inner[1:5]}):
        if x < nx1:
            page.line(T.x(x), T.y(W * .03), T.x(x), T.y(ny0), MED)
            page.line(T.x(x), T.y(ny1), T.x(x), T.y(W * .97), MED)
        else:
            page.line(T.x(x), T.y(W * .03), T.x(x), T.y(W * .97), MED)
    for x, y in p.pegs:
        _hole_uv(page, T, x, y, p.peg_r)


def _cond_front(page: Page, T: Xf, p: Part) -> None:
    ins, outs = p.fc_in_st, p.fc_out_st
    _poly(page, T, list(p.fc_inner), THICK, False)
    _poly(page, T, outs, THICK, False)
    for a, b in ((ins[0], outs[0]), (ins[-1], outs[-1])):
        page.line(*T.p(*a), *T.p(*b), THICK)
    t = p.fc_t
    for x, _ in p.pegs[:1]:
        _rect(page, T, x - p.peg_r, t, x + p.peg_r, t + p.d[2] * .22, MED)


def _cond_right(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    outline = [(W * .1, H * .02), (W * .5, H * .08), (W * .9, H * .02), (W, H * .2), (W * .82, H * .92), (W * .66, H),
               (W * .34, H), (W * .18, H * .92), (0, H * .2)]
    _poly(page, T, outline)
    for sg in (-1, 1):
        page.path([("M",) + T.p(W / 2 + sg * W * .14, H),
                   ("C",) + T.p(W / 2 + sg * W * .08, H * .6) + T.p(W / 2 + sg * W * .03, H * .35) + T.p(W / 2, H * .18)], MED)
    page.line(T.x(W / 2), T.y(H) - 4, T.x(W / 2), T.y(0) + 4, THIN, dash=CENTER_DASH)


def _blade_env(p: Part, n: int = 24) -> List[Tuple[float, float, float, float, float]]:
    L = p.d[0]
    out = []
    for k in range(n + 1):
        x = p.x_pl1 + (L - p.x_pl1) * k / n
        sec = _blade_section(p, x)
        ys, zs = [q[0] for q in sec], [q[1] for q in sec]
        out.append((x, min(ys), max(ys), min(zs), max(zs)))
    return out


def _blade_top(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    if p.stub:
        _rect(page, T, 0, W / 2 - p.stub[1], p.stub[0], W / 2 + p.stub[1])
        for sg in (-1, 1):
            page.line(T.x(0), T.y(W / 2 + sg * p.stub[1] * .86), T.x(p.stub[0]), T.y(W / 2 + sg * p.stub[1] * .86), THIN)
    _rect(page, T, p.x_tr0, W / 2 - p.rt, p.x_tr1, W / 2 + p.rt)
    _rect(page, T, p.x_tr1, W * .04, p.x_pl1, W * .96)
    env = _blade_env(p)
    _poly(page, T, [(x, hi) for x, lo, hi, _, _ in env] + [(x, lo) for x, lo, hi, _, _ in reversed(env)])
    for frac in (.45, .8):
        x = p.x_pl1 + (L - p.x_pl1) * frac
        sec = _blade_section(p, x)
        ys = [q[0] for q in sec]
        page.line(T.x(x), T.y(min(ys)) + 3, T.x(x), T.y(max(ys)) - 3, THIN, dash=PHANTOM_DASH)
    page.line(T.x(0) - 5, T.y(W / 2), T.x(p.x_pl1) + 5, T.y(W / 2), THIN, dash=CENTER_DASH)


def _blade_front(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    if p.stub:
        _rect(page, T, 0, H / 2 - p.stub[1], p.stub[0], H / 2 + p.stub[1])
    _rect(page, T, p.x_tr0, H / 2 - p.rt, p.x_tr1, H / 2 + p.rt)
    _rect(page, T, p.x_tr1, H * .02, p.x_pl1, H * .98)
    env = _blade_env(p)
    _poly(page, T, [(x, hi) for x, _, _, lo, hi in env] + [(x, lo) for x, _, _, lo, hi in reversed(env)])
    page.line(T.x(0) - 5, T.y(H / 2), T.x(p.x_pl1) + 5, T.y(H / 2), THIN, dash=CENTER_DASH)


def _blade_right(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    _rect(page, T, W * .04, H * .02, W * .96, H * .98, MED)
    root = _blade_section(p, p.x_pl1)
    tip = _blade_section(p, L)
    _poly(page, T, root, MED)
    _poly(page, T, tip, THICK)
    cx, cy = T.p(W / 2, H / 2)
    page.circle(cx, cy, p.rt * T.s, HID, dash=HIDDEN_DASH)
    page.line(cx - W * .5 * T.s, cy, cx + W * .5 * T.s, cy, THIN, dash=CENTER_DASH)


def _handle_top(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    xs = [L * k / 60 for k in range(61)]
    up = [(x, W / 2 + _handle_ab(p, x)[0]) for x in xs]
    lo = [(x, W / 2 - _handle_ab(p, x)[0]) for x in xs]
    _poly(page, T, up + list(reversed(lo)))
    for x in p.flutes:
        for sg in (-1, 1):
            a = _handle_ab(p, x)[0]
            w = L * .045
            y0 = W / 2 + sg * _handle_ab(p, x - w)[0]
            y1 = W / 2 + sg * _handle_ab(p, x + w)[0]
            ym = W / 2 + sg * a * .78
            page.path([("M",) + T.p(x - w, y0), ("C",) + T.p(x - w * .4, ym) + T.p(x + w * .4, ym) + T.p(x + w, y1)], MED)
    _hole_uv(page, T, p.port[0], p.port[1], p.port_r)
    for sg in (-1, 1):
        _hid(page, T.x(L - p.hb_depth), T.y(W / 2 + sg * p.hb_r), T.x(L), T.y(W / 2 + sg * p.hb_r))
    page.line(T.x(0) - 5, T.y(W / 2), T.x(L) + 5, T.y(W / 2), THIN, dash=CENTER_DASH)


def _handle_front(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    xs = [L * k / 60 for k in range(61)]
    up = [(x, H / 2 + _handle_ab(p, x)[1]) for x in xs]
    lo = [(x, H / 2 - _handle_ab(p, x)[1]) for x in xs]
    _poly(page, T, up + list(reversed(lo)))
    for sg in (-1, 1):
        _hid(page, T.x(L - p.hb_depth), T.y(H / 2 + sg * p.hb_r), T.x(L), T.y(H / 2 + sg * p.hb_r))
    if p.ht:
        r, d = p.ht
        for sg in (-1, 1):
            _hid(page, T.x(0), T.y(H / 2 + sg * r * .82), T.x(d), T.y(H / 2 + sg * r * .82))
            page.line(T.x(0), T.y(H / 2 + sg * r), T.x(d), T.y(H / 2 + sg * r), THIN, dash=HIDDEN_DASH)
    page.line(T.x(0) - 5, T.y(H / 2), T.x(L) + 5, T.y(H / 2), THIN, dash=CENTER_DASH)


def _handle_right(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    amax = max(_handle_ab(p, L * k / 40)[0] for k in range(41))
    bmax = max(_handle_ab(p, L * k / 40)[1] for k in range(41))
    _poly(page, T, [(W / 2 + u, H / 2 + v) for u, v in _superellipse(amax, bmax, 40)], MED)
    a, b = _handle_ab(p, L)
    _poly(page, T, [(W / 2 + u, H / 2 + v) for u, v in _superellipse(a, b, 40)], THICK)
    cx, cy = T.p(W / 2, H / 2)
    page.circle(cx, cy, p.hb_r * T.s, MED)
    _cmark(page, cx, cy, p.hb_r * T.s)


def _sculpt_top_v(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    xs = [L * k / 48 for k in range(49)]
    up = [(x, W / 2 + _sculpt_a(p, x)) for x in xs]
    lo = [(x, W / 2 - _sculpt_a(p, x)) for x in xs]
    _poly(page, T, up + list(reversed(lo)))
    # fillet tangent lines along the top edges, and where the dome meets the angled face
    up2 = [(x, W / 2 + _sculpt_a(p, x) * .62) for x in xs]
    lo2 = [(x, W / 2 - _sculpt_a(p, x) * .62) for x in xs]
    _poly(page, T, up2, THIN, False)
    _poly(page, T, lo2, THIN, False)
    xa = p.xa
    page.path([("M",) + T.p(xa * .95, W / 2 - _sculpt_a(p, xa) * .62),
               ("C",) + T.p(xa * 1.05, W * .45) + T.p(xa * 1.05, W * .55) + T.p(xa * .95, W / 2 + _sculpt_a(p, xa) * .62)],
              THIN, dash=PHANTOM_DASH)
    a, b = p.slope
    tilt = math.atan(math.hypot(a, b))
    rot = math.atan2(b, a)
    for x, y, r, _ in p.face_holes:
        cx, cy = T.p(x, y)
        page.polygon(_ellipse_poly(cx, cy, r * math.cos(tilt) * T.s, r * T.s, -rot), MED)
        _cmark(page, cx, cy, r * T.s)
    for x, y, r, _ in p.top_holes:
        _hole_uv(page, T, x, y, r)


def _sculpt_front(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    xs = [L * k / 48 for k in range(49)]
    top = [(x, max(_sculpt_top(p, x, W / 2 - _sculpt_a(p, x)), _sculpt_top(p, x, W / 2 + _sculpt_a(p, x)))) for x in xs]
    _poly(page, T, [(0, 0), (L, 0)] + list(reversed(top)))
    low = [(x, min(_sculpt_top(p, x, W / 2 - _sculpt_a(p, x)), _sculpt_top(p, x, W / 2 + _sculpt_a(p, x)))) for x in xs
           if x >= p.xa]
    if len(low) > 1:
        _poly(page, T, low, HID, False, HIDDEN_DASH)
    seen = _Seen()
    for x, y, r, _ in p.top_holes:
        if seen.once(x):
            zt = _sculpt_top(p, x, y)
            for sg in (-1, 1):
                _hid(page, T.x(x + sg * r), T.y(zt), T.x(x + sg * r), T.y(zt * .6))


def _sculpt_right(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    big = max((_sculpt_section(p, L * k / 24) for k in range(25)), key=lambda sec: max(z for _, z in sec) + max(y for y, _ in sec))
    _poly(page, T, big, MED)
    _poly(page, T, _sculpt_section(p, L), THICK)


def _views_complex(p: Part) -> List[View]:
    L, W, H = p.d
    f = p.f
    if p.shape == "impeller":
        top = View("top", "TOP VIEW", 0, 2 * p.R, 0, 2 * p.R, 0, 0, lambda pg, T: _imp_top(pg, T, p))
        front = View("front", "FRONT VIEW", 0, 2 * p.R, 0, H, 0, 1, lambda pg, T: _imp_front(pg, T, p), thin=True)
        top.dim("h", "above", 1, 0, 2 * p.R, "Ø" + f(2 * p.R))
        front.dim("v", "left", 1, 0, H, f(H))
        front.dim("v", "right", 1, 0, p.tb, f(p.tb))
        return [top, front]
    draw = {"lug": (_fit_top, _fit_front, _fit_right), "spar": (_fit_top, _fit_front, _fit_right),
            "plate": (_bp_top, _bp_front, _bp_right), "cage": (_cage_top_v, _cage_front, _cage_right),
            "condyle": (_cond_top, _cond_front, _cond_right), "blade": (_blade_top, _blade_front, _blade_right),
            "handle": (_handle_top, _handle_front, _handle_right), "sculpt": (_sculpt_top_v, _sculpt_front, _sculpt_right)}[p.variant]
    top = View("top", "TOP VIEW", 0, L, 0, W, 0, 0, lambda pg, T: draw[0](pg, T, p))
    front = View("front", "FRONT VIEW", 0, L, 0, H, 0, 1, lambda pg, T: draw[1](pg, T, p), thin=True)
    right = View("right", "RIGHT SIDE VIEW", 0, W, 0, H, 1, 1, lambda pg, T: draw[2](pg, T, p), thin=True)
    top.dim("h", "above", 2, 0, L, f(L))
    top.dim("v", "left", 1, 0, W, f(W))
    front.dim("v", "left", 1, 0, H, f(H))
    if p.variant == "lug":
        front.dim("v", "right", 1, 0, p.zc, f(p.zc))
        top.dim("h", "above", 1, p.xl0, p.xl1, f(p.xl1 - p.xl0), True)
    elif p.variant == "spar":
        if len(p.light) > 1:
            top.dim("h", "above", 1, p.light[0], p.light[1], f(p.light[1] - p.light[0]), True)
        right.dim("v", "right", 1, 0, p.tf, f(p.tf))
    elif p.variant == "plate":
        if len(p.bp_holes) > 1:
            top.dim("h", "above", 1, p.bp_holes[0], p.bp_holes[1], f(p.bp_holes[1] - p.bp_holes[0]), True)
        right.dim("v", "right", 1, 0, p.bp_t, f(p.bp_t))
    elif p.variant == "sculpt":
        top.dim("h", "above", 1, 0, p.xa, f(p.xa), True)
        right.dim("v", "right", 1, 0, _sculpt_top(p, L, W / 2), f(_sculpt_top(p, L, W / 2)))
    elif p.variant == "blade":
        top.dim("h", "above", 1, p.x_tr0, p.x_tr1, f(p.x_tr1 - p.x_tr0), True)
        right.dim("v", "right", 1, H / 2 - p.rt, H / 2 + p.rt, "Ø" + f(2 * p.rt))
    elif p.variant == "cage":
        top.dim("h", "above", 1, p.win[0], p.win[2], f(p.win[2] - p.win[0]), True)
    return [top, front, right]


# --------------------------------------------------------------------------- #
# Other process views: weldment, sheet metal, casting, assembly
# --------------------------------------------------------------------------- #
def _weld_front(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    a, pad = p.tube, p.pad
    for x0 in (0.0, L - a):
        _rect(page, T, x0, pad, x0 + a, H - a)
    _rect(page, T, 0, H - a, L, H)
    _rect(page, T, a, p.zr, L - a, p.zr + a)
    pw = a * 1.6
    for x0 in (0.0, L - pw):
        _rect(page, T, x0, 0, x0 + pw, pad)
    size = p.f(p.weld)
    # the two symbols share the opening between the top rail and the lower rail: fit them into it
    gap = T.y(p.zr + a) - T.y(H - a)
    j1 = T.p(a, H - a)
    _weld(page, j1[0], j1[1], j1[0] + 16, j1[1] + _clamp(gap - 8.5, 6.0, 16.0), size, "TYP")
    j2 = T.p(L - a, p.zr + a)
    _weld(page, j2[0], j2[1], j2[0] - 16, j2[1] - _clamp(gap - 4.0, 9.0, 18.0), size)
    j3 = T.p(a, p.zr)
    if T.y(pad) - j3[1] > 26:
        _weld(page, j3[0], j3[1], j3[0] + 16, j3[1] + 14, size)


def _weld_top(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    a = p.tube
    _rect(page, T, 0, 0, L, W)
    _rect(page, T, a, a, L - a, W - a)
    for y in (a, W - a):
        page.line(T.x(0), T.y(y), T.x(a), T.y(y), MED)
        page.line(T.x(L - a), T.y(y), T.x(L), T.y(y), MED)


def _weld_right(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    a, pad = p.tube, p.pad
    for y0 in (0.0, W - a):
        _rect(page, T, y0, pad, y0 + a, H - a)
    _rect(page, T, 0, H - a, W, H)
    _rect(page, T, a, p.zr, W - a, p.zr + a)
    pw = a * 1.6
    for y0 in (0.0, W - pw):
        _rect(page, T, y0, 0, y0 + pw, pad)


def _sm_flat(page: Page, T: Xf, p: Part) -> None:
    L = p.d[0]
    _rect(page, T, 0, 0, L, p.flat_w)
    rtxt = ("R" + p.f(p.ri)).replace("R0.", "R.")
    for vb in p.vb:
        y = T.y(vb)
        page.line(T.x(0) - 4, y, T.x(L) + 4, y, THIN, dash=PHANTOM_DASH)
        page.text(T.x(L) - 4, y - 2.4, f"UP 90° {rtxt}", 5.8, anchor="end")
    for x, y, r, _ in p.sm_holes:
        _hole_uv(page, T, x, y, r)


def _sm_front(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    _rect(page, T, 0, 0, L, H)
    page.line(T.x(0), T.y(p.t + p.ri), T.x(L), T.y(p.t + p.ri), THIN)


def _sm_right(page: Page, T: Xf, p: Part) -> None:
    st = _sheet_profile(p, 6)
    outer = [o for o, _ in st]
    inner = [i for _, i in st]
    _poly(page, T, outer + list(reversed(inner)))


def _cast_top(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    _rrect(page, T.x(0), T.y(W), T.x(L), T.y(0), p.rc * T.s, THICK)
    cx, cy = T.p(L / 2, W / 2)
    page.circle(cx, cy, p.rbo * T.s, THICK)
    page.circle(cx, cy, (p.rbo + p.rf) * T.s, THIN)
    page.circle(cx, cy, p.rbore * T.s, MED)
    _cmark(page, cx, cy, p.rbo * T.s)
    rt = p.rib_t / 2
    for ang in (0, 90, 180, 270):
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        reach = (L / 2 if ang in (0, 180) else W / 2) * .86
        for sg in (-1, 1):
            u0, u1 = p.rbo + p.rf, reach
            page.line(*T.p(L / 2 + u0 * ca - sg * rt * sa, W / 2 + u0 * sa + sg * rt * ca),
                      *T.p(L / 2 + u1 * ca - sg * rt * sa, W / 2 + u1 * sa + sg * rt * ca), MED)
    for x, y in p.ch:
        _hole_uv(page, T, x, y, p.ch_r, p.ch_r * 1.8)


def _cast_side(page: Page, T: Xf, p: Part, span: float) -> None:
    H = p.d[2]
    c = span / 2
    _rect(page, T, 0, 0, span, p.tb)
    top_r = p.rbo - (H - p.tb) * math.tan(p.draft)
    rf = p.rf
    pts = [(c - p.rbo - rf, p.tb)]
    for k in range(1, 6):
        a = math.pi / 2 * k / 6
        pts.append((c - p.rbo - rf + rf * math.sin(a), p.tb + rf - rf * math.cos(a)))
    pts += [(c - p.rbo, p.tb + rf), (c - top_r, H), (c + top_r, H), (c + p.rbo, p.tb + rf)]
    for k in range(5, 0, -1):
        a = math.pi / 2 * k / 6
        pts.append((c + p.rbo + rf - rf * math.sin(a), p.tb + rf - rf * math.cos(a)))
    pts.append((c + p.rbo + rf, p.tb))
    _poly(page, T, pts, THICK, False)
    hr = (H - p.tb) * .55
    reach = c * .86
    for sg in (-1, 1):
        page.line(*T.p(c + sg * (p.rbo + rf * .5), p.tb + hr), *T.p(c + sg * reach, p.tb), THICK)
    _rect(page, T, c - p.rib_t / 2, p.tb, c + p.rib_t / 2, p.tb + hr * .9, MED)
    for sg in (-1, 1):
        _hid(page, T.x(c + sg * p.rbore), T.y(p.tb), T.x(c + sg * p.rbore), T.y(H))
    page.line(T.x(c), T.y(H) - 5, T.x(c), T.y(0) + 5, THIN, dash=CENTER_DASH)
    y = T.y(p.tb)
    page.line(T.x(0) - 6, y, T.x(span) + 6, y, THIN, dash=PHANTOM_DASH)
    # under the line, where the ribs cannot run through it, when the base below is tall enough to hold it
    page.text(T.x(0) + 3, y + 7.5 if T.y(0) - y > 10 else y - 3, "P/L", 5.6, True)
    _machine_mark(page, T.x(c + top_r * .45), T.y(H))
    _machine_mark(page, T.x(span * .1) + 4, T.y(0), up=True)


def _asm_front(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    _rect(page, T, 0, 0, L, p.tb)
    for x0, x1 in p.sup:
        _rect(page, T, x0, p.tb, x1, p.sup_top)
    x_a, x_b = L * .03, L * .97
    for sg in (-1, 1):
        y = T.y(p.zc + sg * p.rs)
        segs = [(x_a, p.sup[0][0]), (p.sup[0][1], p.pul[0]), (p.pul[1], p.sup[1][0]), (p.sup[1][1], x_b)]
        for a, b in segs:
            page.line(T.x(a), y, T.x(b), y, THICK)
        for a, b in ((p.sup[0][0], p.sup[0][1]), (p.sup[1][0], p.sup[1][1])):
            _hid(page, T.x(a), y, T.x(b), y)
    for x in (x_a, x_b):
        page.line(T.x(x), T.y(p.zc - p.rs), T.x(x), T.y(p.zc + p.rs), THICK)
    _rect(page, T, p.pul[0], p.zc - p.rp, p.pul[1], p.zc + p.rp)
    hr = W * .045
    for x, _ in p.bolts[::2]:
        _rect(page, T, x - hr, p.tb, x + hr, p.tb + hr * .8, MED)
    page.line(T.x(0) - 5, T.y(p.zc), T.x(L) + 5, T.y(p.zc), THIN, dash=CENTER_DASH)


def _asm_top(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    _rect(page, T, 0, 0, L, W)
    for x, y in p.base_holes:
        _hole_uv(page, T, x, y, W * .025)
    for x0, x1 in p.sup:
        _rect(page, T, x0, p.sw[0], x1, p.sw[1])
    for a, b in ((L * .03, p.sup[0][0]), (p.sup[0][1], p.pul[0]), (p.pul[1], p.sup[1][0]), (p.sup[1][1], L * .97)):
        _rect(page, T, a, W / 2 - p.rs, b, W / 2 + p.rs, MED)
    _rect(page, T, p.pul[0], W / 2 - p.rp, p.pul[1], W / 2 + p.rp)
    hr = W * .045
    for x, y in p.bolts:
        page.polygon([T.p(x + hr * math.cos(math.radians(60 * k)), y + hr * math.sin(math.radians(60 * k))) for k in range(6)], MED)
    page.line(T.x(0) - 5, T.y(W / 2), T.x(L) + 5, T.y(W / 2), THIN, dash=CENTER_DASH)


def _asm_right(page: Page, T: Xf, p: Part) -> None:
    L, W, H = p.d
    _rect(page, T, 0, 0, W, p.tb)
    cx, cy = T.p(W / 2, p.zc)
    page.circle(cx, cy, p.rp * T.s, MED)
    y0, y1 = p.sw
    rr = (y1 - y0) / 2
    pts = [(y0, p.tb), (y1, p.tb), (y1, p.zc)] + [(W / 2 + rr * math.cos(math.pi * k / 16), p.zc + p.arch * math.sin(math.pi * k / 16))
                                                  for k in range(1, 16)] + [(y0, p.zc)]
    page.polygon([T.p(u, v) for u, v in pts], THICK, BLACK, WHITE)
    page.circle(cx, cy, p.rs * T.s, THICK)
    _cmark(page, cx, cy, p.rp * T.s)


def _views_other(p: Part) -> List[View]:
    L, W, H = p.d
    f = p.f
    if p.shape == "sheet_metal":
        flat = View("flat", "FLAT PATTERN", 0, L, 0, p.flat_w, 0, 0, lambda pg, T: _sm_flat(pg, T, p))
        front = View("front", "FRONT VIEW", 0, L, 0, H, 0, 1, lambda pg, T: _sm_front(pg, T, p), thin=True)
        right = View("right", "RIGHT SIDE VIEW", 0, W, 0, H, 1, 1, lambda pg, T: _sm_right(pg, T, p))
        flat.dim("h", "above", 1, 0, L, f(L))
        flat.dim("v", "left", 1, 0, p.flat_w, f(p.flat_w))
        front.dim("v", "left", 1, 0, H, f(H))
        right.dim("h", "below", 1, 0, W, f(W))
        right.dim("v", "right", 1, 0, p.t, f(p.t) + " THK")
        return [flat, front, right]
    draw = {"weldment": (_weld_top, _weld_front, _weld_right),
            "casting": (_cast_top, lambda pg, T, p: _cast_side(pg, T, p, L), lambda pg, T, p: _cast_side(pg, T, p, W)),
            "assembly": (_asm_top, _asm_front, _asm_right)}[p.shape]
    top = View("top", "TOP VIEW", 0, L, 0, W, 0, 0, lambda pg, T: draw[0](pg, T, p))
    front = View("front", "FRONT VIEW", 0, L, 0, H, 0, 1, lambda pg, T: draw[1](pg, T, p))
    right = View("right", "RIGHT SIDE VIEW", 0, W, 0, H, 1, 1, lambda pg, T: draw[2](pg, T, p))
    top.dim("h", "above", 1, 0, L, f(L))
    top.dim("v", "left", 1, 0, W, f(W))
    front.dim("v", "left", 1, 0, H, f(H))
    if p.shape == "weldment":
        front.dim("v", "right", 1, 0, p.zr, f(p.zr))
    elif p.shape == "casting":
        front.extra["b"] = 7.0
        right.extra["b"] = 7.0
        front.dim("v", "right", 1, 0, p.tb, f(p.tb))
    elif p.shape == "assembly":
        front.dim("v", "right", 1, 0, p.zc, f(p.zc))
    return [top, front, right]


# --------------------------------------------------------------------------- #
# The sheet: legend, title block, notes, revision block first (so text extraction reads them first),
# then the views, the isometric view, and finally the border and zone labels.
# --------------------------------------------------------------------------- #
def _cell(page: Page, x: float, y: float, w: float, h: float, label: str, value: str, size: float,
          bold: bool = False, center: bool = False, max_lines: int = 2) -> None:
    page.rect(x, y, w, h, THIN)
    if label:
        page.text(x + 3, y + 6.2, label, 4.7, True, color=GREY)
    value = clean(value)
    if not value:
        return
    avail_w = w - 6
    top = y + (8.5 if label else 2.0)
    if max_lines == 1:
        shown, sz = fit_text(value, avail_w, size, bold, 4.2)
        yy = top + (y + h - 2.5 - top) / 2 + sz * 0.36
        if center:
            page.text(x + w / 2, yy, shown, sz, bold, "middle")
        else:
            page.text(x + 3, yy, shown, sz, bold)
        return
    avail_h = y + h - 2.5 - top
    sz = size
    while sz >= 4.6:
        lines = [value] if text_width(value, sz, bold) <= avail_w else wrap(value, avail_w, sz, bold)
        lead = sz * 1.08
        if len(lines) <= max_lines and len(lines) * lead <= avail_h + 1.0:
            break
        sz -= 0.25
    lines = [value] if text_width(value, sz, bold) <= avail_w else wrap(value, avail_w, sz, bold)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = fit_text(lines[-1] + " ..", avail_w, sz, bold)[0]
    lead = sz * 1.08
    block = len(lines) * lead
    yy = top + (avail_h - block) / 2 + sz * 0.82
    for line in lines:
        if center:
            page.text(x + w / 2, yy, line, sz, bold, "middle")
        else:
            page.text(x + 3, yy, line, sz, bold)
        yy += lead


def _title_block(page: Page, spec: Dict[str, Any], p: Part, x: float, y: float, w: float, h: float) -> None:
    units = p.units
    tol_w = 108.0
    rx, rw = x + tol_w, w - tol_w
    rows = [
        (24, [("", clean(spec.get("company")).upper(), 10.5, True, True)]),
        (26, [("TITLE", clean(spec.get("title")).upper(), 8.4, True, False)]),
        (24, [("MATERIAL", clean(spec.get("material")).upper(), 6.6, False, False)]),
        (24, [("FINISH", clean(spec.get("finish") or "NONE").upper(), 6.6, False, False)]),
        (28, [("SIZE", "A", 9.5, True, True), ("DWG NO.", clean(spec.get("part_number")), 10.0, True, False),
              ("REV", clean(spec.get("rev") or "-"), 10.0, True, True)]),
        (24, [("DRAWN", clean(spec.get("drawn_by") or "T. WALSH").upper(), 6.2, False, False),
              ("DATE", clean(spec.get("date") or ""), 6.2, False, False),
              ("SCALE", clean(spec.get("scale") or "1:1"), 6.2, False, False),
              ("SHEET", clean(spec.get("sheet") or "1 OF 1"), 6.2, False, False)]),
    ]
    yy = y
    for height, cells in rows:
        widths = ([rw] if len(cells) == 1 else [rw * .14, rw * .66, rw * .2] if len(cells) == 3
                  else [rw * .31, rw * .27, rw * .19, rw * .23])
        cx = rx
        for (label, value, size, bold, center), cw in zip(cells, widths):
            _cell(page, cx, yy, cw, height, label, value, size, bold, center, 2 if len(cells) == 1 else 1)
            cx += cw
        yy += height
    page.rect(x, y, w, h, THICK)
    page.rect(x, y, tol_w, h, MED)
    ty = y + 9
    for line in ("UNLESS OTHERWISE SPECIFIED:", f"DIMENSIONS ARE IN {'MILLIMETERS' if units == 'mm' else 'INCHES'}",
                 "TOLERANCES:"):
        page.text(x + 4, ty, line, 5.2, True)
        ty += 7.2
    tol = clean(spec.get("tolerances")) or default_tolerances(units)
    pieces = [s.strip() for s in re.split(r"\s{2,}|;|,\s", tol) if s.strip()]
    lines: List[str] = []
    for piece in pieces:
        lines += wrap(piece, tol_w - 14, 5.8) or [piece]
    for line in lines[:5]:
        page.text_fit(x + 8, ty, line, tol_w - 12, 5.8)
        ty += 7.0
    ty += 1.5
    for line in ("INTERPRET PER ASME Y14.5-2018", "BREAK SHARP EDGES", "DO NOT SCALE DRAWING"):
        if ty < y + h - 30:
            page.text(x + 4, ty, line, 5.0)
            ty += 6.8
    # third angle projection symbol: the frustum's small end faces the circles
    sx, sy = x + 30, y + h - 19
    page.circle(sx, sy, 8.0, MED)
    page.circle(sx, sy, 3.6, MED)
    page.line(sx - 11, sy, sx + 11, sy, THIN, dash=(3, 1.2, 1, 1.2))
    page.line(sx, sy - 11, sx, sy + 11, THIN, dash=(3, 1.2, 1, 1.2))
    tx = sx + 17
    page.polygon([(tx, sy - 3.6), (tx + 18, sy - 8.0), (tx + 18, sy + 8.0), (tx, sy + 3.6)], MED)
    page.line(tx - 3, sy, tx + 21, sy, THIN, dash=(3, 1.2, 1, 1.2))
    page.text(x + tol_w / 2, y + h - 3.6, "THIRD ANGLE PROJECTION", 4.7, anchor="middle")


def _rev_block(page: Page, spec: Dict[str, Any], x: float, y: float, w: float) -> float:
    revs = [r for r in (spec.get("revisions") if isinstance(spec.get("revisions"), list) else []) if isinstance(r, dict)][-4:] or [
        {"rev": spec.get("rev") or "A", "description": "RELEASED", "date": spec.get("date") or ""}]
    drawn = clean(spec.get("drawn_by") or "")
    initials = "".join(t[0] for t in re.findall(r"[A-Z][A-Z'-]+", drawn.upper().replace(".", " ")))[:3] or "TW"
    page.rect(x, y, w, 12, MED, BLACK, (0.93, 0.93, 0.92))
    page.text(x + w / 2, y + 8.6, "REVISIONS", 6.5, True, "middle")
    cols = [(0, 24, "REV"), (24, w - 24 - 48 - 34, "DESCRIPTION"), (w - 82, 48, "DATE"), (w - 34, 34, "APPR")]
    yy = y + 12
    for cx, cw, label in cols:
        page.rect(x + cx, yy, cw, 10, THIN)
        page.text(x + cx + cw / 2, yy + 7.2, label, 5.5, True, "middle")
    yy += 10
    for r in revs:
        desc = clean(r.get("description")).upper()
        dw = cols[1][1] - 5
        lines = wrap(desc, dw, 5.8) or [""]
        if len(lines) > 2:
            lines = lines[:2]
            lines[-1] = fit_text(lines[-1] + " ..", dw, 5.8)[0]
        rh = 11.0 if len(lines) == 1 else 18.0
        vals = [clean(r.get("rev")), None, clean(r.get("date")), initials]
        for (cx, cw, _), val in zip(cols, vals):
            page.rect(x + cx, yy, cw, rh, THIN)
            if val is None:
                for i, line in enumerate(lines):
                    page.text(x + cx + 2.5, yy + 7.8 + i * 6.8, line, 5.8)
            else:
                page.text_fit(x + cx + 2.5, yy + 7.8, val, cw - 5, 5.8)
        yy += rh
    return yy


def _notes(page: Page, spec: Dict[str, Any], legend: Optional[str], company: str, x: float, width: float,
           bottom: float, max_h: float) -> float:
    """Notes anchored to the bottom left, proprietary text under them. Returns the top of the block."""
    notes = [clean(n).strip() for n in _text_list(spec.get("notes")) if clean(n).strip()][:12]
    prop = legend_text("proprietary", company) if legend == "proprietary" else ""
    size, lead = 6.8, 8.3
    for _ in range(6):
        lines: List[Tuple[str, str]] = []
        for i, note in enumerate(notes, start=1):
            for j, piece in enumerate(wrap(note, width - 16, size)):
                lines.append((f"{i}." if j == 0 else "", piece))
        plines = wrap(prop, width, 5.6, True) if prop else []
        height = 13 + len(lines) * lead + (len(plines) * 6.8 + 6 if plines else 0)
        if height <= max_h or size <= 5.4:
            break
        size, lead = size - 0.35, lead - 0.4
    top = bottom - height
    y = top
    if notes:
        page.text(x, y + 8, "NOTES:", 7.6, True)
        y += 13
        for num, piece in lines:
            y += lead
            if num:
                page.text(x, y - 1.5, num, size, True)
            page.text(x + 14, y - 1.5, piece, size)
    if plines:
        y += 6
        for line in plines:
            y += 6.8
            page.text(x, y - 1.2, line, 5.6, True, color=GREY)
    return top


def _parts_table(page: Page, p: Part, x: float, bottom: float, w: float) -> float:
    rows = p.parts_list
    rh = 10.5
    cols = [(0, 26, "ITEM", "middle"), (26, 96, "PART NUMBER", "start"), (122, w - 122 - 30, "DESCRIPTION", "start"),
            (w - 30, 30, "QTY", "middle")]
    y = bottom - rh
    page.rect(x, y, w, rh, MED, BLACK, (0.93, 0.93, 0.92))
    for cx, cw, label, al in cols:
        page.rect(x + cx, y, cw, rh, THIN)
        page.text(x + cx + (cw / 2 if al == "middle" else 3), y + 7.3, label, 5.4, True, al)
    for item in rows:
        y -= rh
        for (cx, cw, _, al), val in zip(cols, (item[0], item[1], item[2], str(item[3]))):
            page.rect(x + cx, y, cw, rh, THIN)
            if al == "middle":
                page.text(x + cx + cw / 2, y + 7.4, val, 6.0, False, "middle")
            else:
                page.text_fit(x + cx + 3, y + 7.4, val, cw - 6, 6.0)
    y -= 11
    page.text(x + w / 2, y + 8, "PARTS LIST", 6.5, True, "middle")
    return y


def _break_candidates(p: Part) -> Optional[Tuple[float, float]]:
    """The longest stretch along the length with no features, where a long view may be broken."""
    L = p.d[0]
    if p.family == "round":
        segs = p.segs
        blocked = [(kw["x0"], kw["x1"]) for kw in p.keyways] + [(c["x"] - c["r"] * 3, c["x"] + c["r"] * 3) for c in p.cross]
        blocked += [(t["x0"], t["x1"]) for t in p.tines] + [(b["x0"], b["x1"]) for b in p.bore
                                                             if 0 < b["x0"] or b["x1"] < L]
        best = None
        run = None
        for k, sg in enumerate(segs):
            ok = sg["kind"] == "plain" and 0 < k < len(segs) - 1 or (len(segs) == 1 and sg["kind"] == "plain")
            if ok:
                run = (run[0], sg["x1"]) if run else (sg["x0"], sg["x1"])
            else:
                run = None
            if run:
                for a0, b0 in _free_parts(run, blocked):
                    if best is None or b0 - a0 > best[1] - best[0]:
                        best = (a0, b0)
        if best and best[1] - best[0] > L * .2:
            m = (best[1] - best[0]) * .1
            return best[0] + m, best[1] - m
        return None
    if p.family != "prismatic":
        return None
    occ: List[Tuple[float, float]] = []
    for h in p.holes:
        if h["axis"] in ("z", "y"):
            rr = max(h["r"], h["r2"], h["rt"])
            occ.append((h["a"] - rr * 2, h["a"] + rr * 2))
        elif h["hi"] - h["lo"] < L * .5:
            occ.append((h["lo"], h["hi"]))
    for q in p.pockets:
        occ.append((q["x0"], q["x1"]))
    for sl in p.slots:
        occ.append((sl["cx"] - sl["len"], sl["cx"] + sl["len"]))
    for o in p.open_slots:
        occ.append((o["cx"] - o["wid"], o["cx"] + o["wid"]))
    for x0, x1 in p.fins:
        occ.append((x0, x1))
    parts = _free_parts((L * .04, L * .96), occ)
    if not parts:
        return None
    a, b = max(parts, key=lambda f: f[1] - f[0])
    if b - a < L * .25:
        return None
    m = (b - a) * .06
    return a + m, b - m


def _free_parts(span: Tuple[float, float], blocked: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    lo, hi = span
    out, cur = [], lo
    for a, b in sorted(blocked):
        if b <= cur or a >= hi:
            continue
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        out.append((cur, hi))
    return [f for f in out if f[1] - f[0] > 1e-9]


def _make_views(p: Part, stubby: bool) -> List[View]:
    if p.family == "round":
        return _views_round(p, stubby)
    if p.family == "prismatic":
        return _views_prismatic(p)
    if p.family == "complex":
        return _views_complex(p)
    return _views_other(p)


def _label_block_size(items: List[Tuple[str, Anchor, str]], max_w: float) -> Tuple[float, float]:
    if not items:
        return 0.0, 0.0
    width = min(max((text_width(t, LABEL) if k == "callout" else 14) for t, _, k in items) + 16, max_w)
    h = 0.0
    for t, _, k in items:
        h += (15 + 5) if k == "balloon" else len(_label_lines(t, width - 12)[0]) * (LABEL + 1.4) + 2 + 5
    return width, h


def _band_height(items: Sequence[Tuple[str, Any]], rw: float) -> float:
    return sum(len(_label_lines(t, min(190.0, rw * .45))[0]) * (LABEL + 1.4) + 4 for t, _ in items) + 4


def _try_layout(p: Part, region: Tuple[float, float, float, float], stubby: bool, mode: str, col_w: float,
                items: List[Tuple[str, Anchor, str]]):
    views = _make_views(p, stubby)
    blocks: List[Block] = []
    band = None
    rx, ry, rw, rh = region
    if p.family == "round" or mode == "band":
        sec = [it for it in items if it[1][0] == "section" and not stubby and p.family == "round" and it[2] == "callout"]
        top_items = [(t, a) for t, a, k in items if (t, a, k) not in sec]
        if top_items:
            bh = _band_height(top_items, rw)
            band = (rx, ry, rw, bh)
            ry, rh = ry + bh + 4, rh - bh - 4
        if sec:
            bw, bh2 = _label_block_size(sec, col_w)
            blocks.append(Block("labels", 1, 1, bw, bh2))
    elif items:
        bw, bh = _label_block_size(items, col_w)
        blocks.append(Block("labels", 1, 0, bw, bh))
    if rh < 60:
        return None
    s = _layout(views, blocks, (rx, ry, rw, rh))
    cand = _break_candidates(p)
    if cand:
        u_views = [v for v in views if v.key in ("side", "section", "top", "front", "flat")]
        if p.family == "round":
            small = 2 * p.rmax * s < 32 and p.d[0] > 6 * p.rmax * 2
        else:
            small = min(p.d[1], p.d[2] * 3) * s < 36 and p.d[0] > 5 * p.d[1]
        if small:
            b0, b1 = cand
            for frac in (.45, .65, .85, 1.0):
                mid, half = (b0 + b1) / 2, (b1 - b0) * frac / 2
                for v in u_views:
                    v.brk = (mid - half, mid + half, 12.0)
                s = _layout(views, blocks, (rx, ry, rw, rh))
                if (2 * p.rmax if p.family == "round" else min(p.d[1], p.d[2] * 3)) * s >= 36:
                    break
    return s, views, blocks, band, (rx, ry, rw, rh)


def _draw_views(page: Page, p: Part, regions: List[Tuple[float, float, float, float]]) -> None:
    items: List[Tuple[str, Anchor, str]] = [(p.callouts[i], p.anchors[i], "callout") for i in range(len(p.callouts))
                                            if i in p.anchors]
    if p.balloons:
        items = [(str(n), anc, "balloon") for n, anc in p.balloons] + items
    best = None

    def consider(region, stubby, mode, col_w, handicap):
        nonlocal best
        got = _try_layout(p, region, stubby, mode, col_w, items)
        if got is not None and (best is None or got[0] / handicap > best[0]):
            best = (got[0] / handicap, got, mode)
        return got

    for region in regions:
        if p.family == "round":
            for stubby in (False, True):
                if not (stubby and p.d[0] > p.rmax * 3.2):
                    consider(region, stubby, "band", 176.0, 1.0)
            continue
        got = consider(region, False, "column", 176.0, 1.0)
        if not items or got is None:
            continue
        # narrower label columns only help when the views are limited by the width
        views, rg = got[1], got[4]
        xs = [v.xf.x(v.u1) for v in views if v.xf] + [b.x + b.w for b in got[2]]
        if max(xs) > rg[0] + rg[2] - 4:
            consider(region, False, "column", 140.0, 1.012)
            consider(region, False, "column", 112.0, 1.03)
            if not p.balloons and p.d[0] > 2.5 * max(p.d[1], 1e-9):
                consider(region, False, "band", 176.0, 1.22)
    if best is None:
        got = _try_layout(p, regions[0], False, "column", 176.0, items)
        if got is None:
            return
        best = (0, got, "column")
    _, (s, views, blocks, band, region), mode = best
    byk = {v.key: v for v in views}
    first_view_op = len(page.ops)
    for v in views:
        v.render(page)
    first_label_op = len(page.ops)
    # leaders drawn from here on break around the views' figures (dimensions, section letters, notes on the
    # views); view titles are not in the list, they move out of the way afterwards
    titles = {v.title_op for v in views}
    _KNOCKOUT[:] = [_text_box(o, 1.2) for k, (kind, o) in enumerate(page.ops[first_view_op:first_label_op], first_view_op)
                    if kind == "text" and k not in titles]
    try:
        col_items = [it for it in items if blocks and (mode == "column" or it[1][0] == "section")]
        band_items = [(t, a) for t, a, k in items if (t, a, k) not in col_items]
        if band and band_items:
            _labels_band(page, byk, band, band_items)
        if blocks and col_items:
            _labels_column(page, p, byk, blocks[0], col_items)
    finally:
        _KNOCKOUT[:] = []
    _clear_titles(page, views, first_view_op, first_label_op)


_KNOCKOUT: List[Tuple[float, float, float, float]] = []  # boxes that leader lines break around (set while labeling)


def _ko_line(page: Page, x1: float, y1: float, x2: float, y2: float, lw: float = THIN) -> None:
    """A leader line, left out where it would cross a dimension's figures (the gap reads as the leader passing
    behind the dimension instead of striking through it)."""
    dx, dy = x2 - x1, y2 - y1
    cuts = []
    for bx0, by0, bx1, by1 in _KNOCKOUT:
        u0, u1 = 0.0, 1.0
        ok = True
        for pv, qv in ((-dx, x1 - bx0), (dx, bx1 - x1), (-dy, y1 - by0), (dy, by1 - y1)):
            if abs(pv) < 1e-12:
                if qv < 0:
                    ok = False
                    break
                continue
            t = qv / pv
            if pv < 0:
                u0 = max(u0, t)
            else:
                u1 = min(u1, t)
            if u0 > u1:
                ok = False
                break
        if ok and u1 - u0 > 1e-6:
            cuts.append((u0, u1))
    if not cuts:
        page.line(x1, y1, x2, y2, lw)
        return
    cuts.sort()
    t = 0.0
    ln = math.hypot(dx, dy)
    for a, b in cuts + [(1.0, 1.0)]:
        if a > t and (a - t) * ln > 0.6:
            page.line(x1 + dx * t, y1 + dy * t, x1 + dx * a, y1 + dy * a, lw)
        t = max(t, b)


def _text_box(o: Dict[str, Any], pad: float = 1.0) -> Tuple[float, float, float, float]:
    w = text_width(o["text"], o["size"], o["bold"])
    x = o["x"] - (w if o["anchor"] == "end" else w / 2 if o["anchor"] == "middle" else 0.0)
    return x - pad, o["y"] - o["size"] * 0.74, x + w + pad, o["y"] + o["size"] * 0.22


def _clear_titles(page: Page, views: List[View], v0: int, l0: int) -> None:
    """Leaders are drawn after the views, so one can run through a view title: slide that title sideways, under
    its own view, to the nearest spot no leader crosses."""
    ops = page.ops
    segs = [((o["x1"], o["y1"]), (o["x2"], o["y2"])) for k, o in ops[l0:] if k == "line"]
    if not segs:
        return

    def crossed(box: Tuple[float, float, float, float]) -> bool:
        return any(_seg_hits_box(a, b, box, 0.0) for a, b in segs)

    for v in views:
        k = v.title_op
        if k is None or not (v0 <= k < l0) or ops[k][0] != "text" or v.xf is None:
            continue
        o = ops[k][1]
        if not crossed(_text_box(o)):
            continue
        half = text_width(o["text"], o["size"], o["bold"]) / 2
        gx0, gx1 = v.xf.x(v.u0), v.xf.x(v.u1)
        for step in range(1, 60):
            hit = None
            for sgn in (-1, 1):
                nx = o["x"] + sgn * step * 3.0
                if nx - half >= gx0 - 8 and nx + half <= gx1 + 8 and not crossed(_text_box(dict(o, x=nx))):
                    hit = nx
                    break
            if hit is not None:
                o["x"] = hit
                break


_SHEET_CACHE: "OrderedDict[str, List[Page]]" = OrderedDict()


def drawing_pages(spec: Dict[str, Any]) -> List[Page]:
    """The drawing sheet for a spec (memoized: the PDF, the thumbnail, and the page count all ask for it)."""
    key = json.dumps(spec, sort_keys=True, default=str)
    hit = _SHEET_CACHE.get(key)
    if hit is not None:
        _SHEET_CACHE.move_to_end(key)
        return list(hit)
    pages = _drawing_pages(spec)
    _SHEET_CACHE[key] = pages
    while len(_SHEET_CACHE) > 48:
        _SHEET_CACHE.popitem(last=False)
    return list(pages)


def _drawing_pages(spec: Dict[str, Any]) -> List[Page]:
    p = part_model(spec)
    page = Page(*LETTER_LANDSCAPE)
    PW, PH = page.width, page.height
    m, inner = 18.0, 28.0
    legend = spec.get("legend") if spec.get("legend") in ("itar", "ear", "cui", "proprietary") else None
    company = clean(spec.get("company"))
    top = inner + 5.0
    if legend == "cui":
        page.text(PW / 2, 14.2, "CUI", 11, True, "middle")
        page.text(PW / 2, PH - 5.6, "CUI", 11, True, "middle")
    elif legend in ("itar", "ear"):
        text = legend_text(legend, company, spec.get("eccn", ""))
        lines = wrap(text, PW - 2 * inner - 24, 6.3, True)
        box_h = 9.0 + len(lines) * 7.6
        page.rect(inner + 4, top, PW - 2 * inner - 8, box_h, 1.2, BLACK, (1.0, 0.965, 0.88))
        for i, line in enumerate(lines):
            page.text(inner + 11, top + 10.8 + i * 7.6, line, 6.3, True)
        top += box_h + 6
    tb_w, tb_h = 300.0, 150.0
    tb_x, tb_y = PW - inner - tb_w, PH - inner - tb_h
    _title_block(page, spec, p, tb_x, tb_y, tb_w, tb_h)
    notes_x = inner + 10
    notes_w = tb_x - notes_x - 14
    notes_top = _notes(page, spec, legend, company, notes_x, notes_w, PH - inner - 6, (PH - top) * .42)
    rb_w = 250.0
    rb_x = PW - inner - rb_w - 4
    rev_bottom = _rev_block(page, spec, rb_x, top, rb_w)
    col_bottom = tb_y - 6
    if legend == "cui":
        text = legend_text("cui", company, poc=spec.get("drawn_by", ""))
        lines = wrap(text, tb_w - 12, 5.3, True)
        h = 16 + len(lines) * 6.5
        page.rect(tb_x, col_bottom - h, tb_w, h, MED, BLACK, (0.97, 0.97, 0.97))
        page.text(tb_x + 5, col_bottom - h + 9, "CUI DESIGNATION INDICATOR", 5.8, True)
        for i, line in enumerate(lines):
            page.text(tb_x + 5, col_bottom - h + 17 + i * 6.5, line, 5.3, True)
        col_bottom -= h + 6
    if p.parts_list:
        col_bottom = _parts_table(page, p, tb_x, col_bottom, tb_w) - 4
    flag_lines = wrap(p.flag, rb_w - 16, 5.8, True) if p.flag else []
    flag_h = (10 + len(flag_lines) * 7.0) if flag_lines else 0.0
    x0 = inner + 12
    vtop = top + 4
    regions = [(x0, vtop, tb_x - 16 - x0, notes_top - 8 - vtop),
               (x0, vtop, rb_x - 14 - x0, min(notes_top, col_bottom, tb_y) - 8 - vtop)]
    regions = [r for r in regions if r[2] > 120 and r[3] > 120] or [regions[0]]
    _draw_views(page, p, regions)
    iso_top = rev_bottom + 10
    iso_bottom = col_bottom - (flag_h + 8 if flag_h else 0) - 14
    if iso_bottom - iso_top > 50:
        drawn = render_mesh(page, build_mesh(spec, coarse=True), rb_x + 8, iso_top, rb_w - 16, iso_bottom - iso_top,
                            base=(0.86, 0.88, 0.91), edges=(0.12, 0.13, 0.15), edge_lw=0.45)
        # the title goes right under the part, not at the bottom of the area the view was given
        page.text(rb_x + rb_w / 2, min(iso_bottom + 9, drawn[3] + 14), "ISOMETRIC VIEW", 6.5, True, "middle")
    if flag_lines:
        fy = col_bottom - flag_h
        page.rect(rb_x + 4, fy, rb_w - 8, flag_h - 2, MED)
        for i, line in enumerate(flag_lines):
            page.text(rb_x + rb_w / 2, fy + 9 + i * 7.0, line, 5.8, True, "middle")
    # border and zones, last
    page.rect(m, m, PW - 2 * m, PH - 2 * m, THICK)
    page.rect(inner, inner, PW - 2 * inner, PH - 2 * inner, MED)
    for i in range(8):
        zx = inner + (PW - 2 * inner) * (i + 0.5) / 8
        page.text(zx, m + 7.5, str(8 - i), 6, anchor="middle")
        page.text(zx, PH - m - 2.5, str(8 - i), 6, anchor="middle")
        if i:
            bx = inner + (PW - 2 * inner) * i / 8
            page.line(bx, m, bx, inner, THIN)
            page.line(bx, PH - inner, bx, PH - m, THIN)
    for i, letter in enumerate("DCBA"):
        zy = inner + (PH - 2 * inner) * (i + 0.5) / 4
        page.text(m + 5, zy + 2, letter, 6, anchor="middle")
        page.text(PW - m - 5, zy + 2, letter, 6, anchor="middle")
        if i:
            by = inner + (PH - 2 * inner) * i / 4
            page.line(m, by, inner, by, THIN)
            page.line(PW - inner, by, PW - m, by, THIN)
    return [page]
