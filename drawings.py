"""
Engineering drawings and 3D models for the sample attachments. Standard library only.

    drawing_pages(spec)    -> [Page]  an 11 x 8.5 in drawing sheet: border and zones, orthographic
                                       views with dimensions, a shaded isometric view, notes, a
                                       revision block, the title block, and any legend (ITAR, EAR,
                                       CUI, proprietary)
    mesh_for(spec)         -> {"vertices", "faces", ...}  a faceted solid for the 3D viewer
    model_thumb_page(spec) -> Page    the shaded isometric view used on a STEP file's tile
    step_file(spec)        -> str     an ISO 10303-21 (STEP) file with a faceted B-rep of the mesh

Shapes (spec["shape"]) and sizes (spec["size"], in spec["units"]):
    prismatic  plate block bracket housing cover manifold heatsink fixture enclosure  [L, W, H]
    complex    structural_fitting impeller implant contoured                         [L, W, H]
    round      shaft pin disc nozzle threaded_fitting                                 [L, D]
               bushing spacer ring                                                    [L, OD, ID]
    other      weldment sheet_metal casting assembly                                  [L, W, H]
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from docgen import BLACK, LETTER_LANDSCAPE, Page, clean, legend_text, text_width, wrap

PRISMATIC = {"plate", "block", "bracket", "housing", "cover", "manifold", "heatsink", "fixture", "enclosure"}
COMPLEX = {"structural_fitting", "impeller", "implant", "contoured"}
ROUND2 = {"shaft", "pin", "disc", "nozzle", "threaded_fitting"}
ROUND3 = {"bushing", "spacer", "ring"}
ROUND = ROUND2 | ROUND3
OTHER = {"weldment", "sheet_metal", "casting", "assembly"}

GREY = (0.42, 0.44, 0.48)
THIN = 0.45
MED = 0.8
THICK = 1.3
CENTER_DASH = (9, 2.5, 1.5, 2.5)
HIDDEN_DASH = (3, 2)


# --------------------------------------------------------------------------- #
# Size helpers (also used for the text Jev reads)
# --------------------------------------------------------------------------- #
def _dims(spec: Dict[str, Any]) -> List[float]:
    try:
        vals = [float(v) for v in spec.get("size") or []]
    except (TypeError, ValueError):
        vals = []
    shape = spec.get("shape")
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


# --------------------------------------------------------------------------- #
# Meshes: faces are lists of vertex indices, counterclockwise seen from outside, z up.
# --------------------------------------------------------------------------- #
class Mesh:
    def __init__(self) -> None:
        self.v: List[Tuple[float, float, float]] = []
        self.f: List[List[int]] = []

    def add(self, pts: Sequence[Tuple[float, float, float]], faces: Sequence[Sequence[int]]) -> None:
        base = len(self.v)
        self.v.extend((float(x), float(y), float(z)) for x, y, z in pts)
        self.f.extend([base + i for i in face] for face in faces)

    def box(self, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> None:
        pts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
               (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        self.add(pts, [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])

    def cylinder_x(self, x0: float, x1: float, r: float, cy: float = 0.0, cz: float = 0.0,
                   segments: int = 28, r_inner: float = 0.0) -> None:
        """A solid (or hollow, r_inner > 0) cylinder along +x."""
        n = segments
        ring = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
        pts = [(x0, cy + r * c, cz + r * s) for c, s in ring] + [(x1, cy + r * c, cz + r * s) for c, s in ring]
        faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        if r_inner > 0:
            pts += [(x0, cy + r_inner * c, cz + r_inner * s) for c, s in ring]
            pts += [(x1, cy + r_inner * c, cz + r_inner * s) for c, s in ring]
            a, b = 2 * n, 3 * n
            faces += [[a + i, b + i, b + (i + 1) % n, a + (i + 1) % n] for i in range(n)]  # bore, facing in
            faces += [[i, a + i, a + (i + 1) % n, (i + 1) % n] for i in range(n)]          # x0 end, normal -x
            faces += [[n + i, n + (i + 1) % n, b + (i + 1) % n, b + i] for i in range(n)]  # x1 end, normal +x
        else:
            faces.append(list(range(n - 1, -1, -1)))  # x0 cap, normal -x
            faces.append([n + i for i in range(n)])     # x1 cap, normal +x
        self.add(pts, faces)

    def cylinder_z(self, z0: float, z1: float, r: float, cx: float = 0.0, cy: float = 0.0,
                   segments: int = 28) -> None:
        n = segments
        ring = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
        pts = [(cx + r * c, cy + r * s, z0) for c, s in ring] + [(cx + r * c, cy + r * s, z1) for c, s in ring]
        faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        faces.append(list(range(n - 1, -1, -1)))
        faces.append([n + i for i in range(n)])
        self.add(pts, faces)

    def prism_z(self, outline: Sequence[Tuple[float, float]], z0: float, z1: float) -> None:
        """Extrude a counterclockwise outline (seen from +z) between z0 and z1."""
        n = len(outline)
        pts = [(x, y, z0) for x, y in outline] + [(x, y, z1) for x, y in outline]
        faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        faces.append(list(range(n - 1, -1, -1)))
        faces.append([n + i for i in range(n)])
        self.add(pts, faces)

    def as_dict(self, units: str) -> Dict[str, Any]:
        xs = [p[0] for p in self.v] or [0]
        ys = [p[1] for p in self.v] or [0]
        zs = [p[2] for p in self.v] or [0]
        return {"units": units, "vertices": [[round(c, 4) for c in p] for p in self.v], "faces": self.f,
                "bbox": [round(max(xs) - min(xs), 4), round(max(ys) - min(ys), 4), round(max(zs) - min(zs), 4)]}


def _rounded_outline(length: float, width: float, r: float, steps: int = 5) -> List[Tuple[float, float]]:
    r = max(0.0, min(r, length / 2 * 0.95, width / 2 * 0.95))
    pts: List[Tuple[float, float]] = []
    corners = [(length - r, r, -90), (length - r, width - r, 0), (r, width - r, 90), (r, r, 180)]
    for cx, cy, start in corners:
        for i in range(steps + 1):
            a = math.radians(start + 90 * i / steps)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def build_mesh(spec: Dict[str, Any]) -> Mesh:
    shape, d, m = spec.get("shape") or "block", _dims(spec), Mesh()
    if shape in ROUND:
        length = d[0]
        od = d[1]
        idia = d[2] if shape in ROUND3 else 0.0
        r = od / 2
        if shape == "shaft":
            # a stepped shaft: journal, body, journal, short threaded end
            steps = [(0.0, 0.14, 0.72), (0.14, 0.78, 1.0), (0.78, 0.92, 0.72), (0.92, 1.0, 0.55)]
            for a, b, k in steps:
                m.cylinder_x(a * length, b * length, r * k)
        elif shape == "pin":
            m.cylinder_x(0, length * 0.94, r)
            m.cylinder_x(length * 0.94, length, r * 0.82)
        elif shape == "nozzle":
            m.cylinder_x(0, length * 0.35, r)
            m.cylinder_x(length * 0.35, length * 0.8, r * 0.7)
            m.cylinder_x(length * 0.8, length, r * 0.42)
        elif shape == "threaded_fitting":
            m.cylinder_x(0, length * 0.38, r * 0.72)
            hexr = r
            hexagon = [(hexr * math.cos(math.radians(30 + 60 * i)), hexr * math.sin(math.radians(30 + 60 * i)))
                       for i in range(6)]
            n = len(hexagon)
            x0, x1 = length * 0.38, length * 0.62
            pts = [(x0, y, z) for y, z in hexagon] + [(x1, y, z) for y, z in hexagon]
            faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
            faces.append(list(range(n - 1, -1, -1)))
            faces.append([n + i for i in range(n)])
            m.add(pts, faces)
            m.cylinder_x(length * 0.62, length, r * 0.72)
        elif shape in ROUND3:
            m.cylinder_x(0, length, r, r_inner=idia / 2)
        else:  # disc
            m.cylinder_x(0, length, r)
        return m
    L, W, H = d
    if shape == "bracket":
        t = min(H * 0.3, L * 0.18, W * 0.25) if H > 0 else 0.2
        m.box(0, 0, 0, L, W, t)
        m.box(0, 0, t, t * 1.2, W, H)
    elif shape == "heatsink":
        base = H * 0.28
        m.box(0, 0, 0, L, W, base)
        fins = max(4, min(14, int(L / (H * 0.35 + 1e-6))))
        pitch = L / fins
        for i in range(fins):
            x = i * pitch + pitch * 0.3
            m.box(x, 0, base, x + pitch * 0.4, W, H)
    elif shape in ("housing", "enclosure"):
        wall = min(L, W) * 0.08
        m.box(0, 0, 0, L, W, H * 0.12)
        m.box(0, 0, H * 0.12, L, wall, H)
        m.box(0, W - wall, H * 0.12, L, W, H)
        m.box(0, wall, H * 0.12, wall, W - wall, H)
        m.box(L - wall, wall, H * 0.12, L, W - wall, H)
    elif shape == "manifold":
        m.box(0, 0, 0, L, W, H)
        for i in range(3):
            m.cylinder_z(H, H + H * 0.08, min(W, L) * 0.08, L * (0.25 + 0.25 * i), W / 2, 16)
    elif shape == "fixture":
        m.box(0, 0, 0, L, W, H * 0.35)
        m.box(L * 0.08, W * 0.2, H * 0.35, L * 0.22, W * 0.8, H)
        m.box(L * 0.78, W * 0.2, H * 0.35, L * 0.92, W * 0.8, H)
    elif shape == "structural_fitting":
        m.box(0, 0, 0, L, W, H * 0.22)
        m.box(0, W * 0.4, H * 0.22, L, W * 0.6, H)
        m.box(0, 0, H * 0.22, L * 0.12, W, H * 0.75)
    elif shape == "impeller":
        r = min(L, W) / 2
        m.cylinder_z(0, H * 0.25, r, r, r, 36)
        m.cylinder_z(H * 0.25, H, r * 0.28, r, r, 20)
        blades = 7
        for i in range(blades):
            a = 2 * math.pi * i / blades
            ca, sa = math.cos(a), math.sin(a)
            inner, outer, t = r * 0.3, r * 0.95, r * 0.05
            quad = [(inner, -t), (outer, -t * 0.6), (outer, t * 0.6), (inner, t)]
            outline = [(r + x * ca - y * sa, r + x * sa + y * ca) for x, y in quad]
            m.prism_z(outline, H * 0.25, H * 0.25 + (H * 0.75) * 0.85)
    elif shape in ("implant", "contoured"):
        m.prism_z(_rounded_outline(L, W, min(L, W) * 0.45, 7), 0, H * 0.55)
        m.prism_z([(x * 0.7 + L * 0.15, y * 0.7 + W * 0.15) for x, y in _rounded_outline(L, W, min(L, W) * 0.45, 7)],
                  H * 0.55, H)
    elif shape == "weldment":
        t = min(L, W, H) * 0.08
        for x0, y0 in ((0, 0), (L - t, 0), (0, W - t), (L - t, W - t)):
            m.box(x0, y0, 0, x0 + t, y0 + t, H)
        m.box(0, 0, H - t, L, t, H)
        m.box(0, W - t, H - t, L, W, H)
        m.box(0, t, H - t, t, W - t, H)
        m.box(L - t, t, H - t, L, W - t, H)
        m.box(L * 0.3, W * 0.2, H, L * 0.7, W * 0.8, H + t)
    elif shape == "sheet_metal":
        t = max(min(L, W, H) * 0.04, 0.02)
        m.box(0, 0, 0, L, W, t)
        m.box(0, 0, t, L, t, H)
        m.box(0, W - t, t, L, W, H)
    elif shape == "casting":
        m.prism_z(_rounded_outline(L, W, min(L, W) * 0.2, 5), 0, H * 0.6)
        m.cylinder_z(H * 0.6, H, min(L, W) * 0.28, L / 2, W / 2, 24)
    elif shape == "assembly":
        m.box(0, 0, 0, L, W, H * 0.25)
        m.cylinder_x(0, L, min(W, H) * 0.12, W / 2, H * 0.55)
        m.box(L * 0.1, W * 0.3, H * 0.25, L * 0.25, W * 0.7, H)
        m.box(L * 0.75, W * 0.3, H * 0.25, L * 0.9, W * 0.7, H)
    elif shape == "cover":
        m.box(0, 0, 0, L, W, H * 0.5)
        m.box(L * 0.06, W * 0.06, H * 0.5, L * 0.94, W * 0.94, H)
    else:  # plate, block
        m.box(0, 0, 0, L, W, H)
    return m


def mesh_for(spec: Dict[str, Any]) -> Dict[str, Any]:
    return build_mesh(spec).as_dict(spec.get("units") or "in")


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


def iso_view(page: Page, spec: Dict[str, Any], x: float, y: float, w: float, h: float,
             base: Tuple[float, float, float] = (0.78, 0.82, 0.88), edges: Tuple[float, float, float] = (0.2, 0.22, 0.26),
             yaw_deg: float = -38.0, pitch_deg: float = 57.0) -> None:
    mesh = build_mesh(spec)
    if not mesh.v:
        return
    cx = sum(p[0] for p in mesh.v) / len(mesh.v)
    cy = sum(p[1] for p in mesh.v) / len(mesh.v)
    cz = sum(p[2] for p in mesh.v) / len(mesh.v)
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    proj = [_iso_project((px - cx, py - cy, pz - cz), yaw, pitch) for px, py, pz in mesh.v]
    xs, ys = [p[0] for p in proj], [p[1] for p in proj]
    scale = min(w / max(max(xs) - min(xs), 1e-6), h / max(max(ys) - min(ys), 1e-6)) * 0.92
    ox = x + w / 2 - (max(xs) + min(xs)) / 2 * scale
    oy = y + h / 2 - (max(ys) + min(ys)) / 2 * scale
    light = (-0.35, -0.55, 0.76)
    faces = []
    for face in mesh.f:
        if len(face) < 3:
            continue
        a, b, c = mesh.v[face[0]], mesh.v[face[1]], mesh.v[face[2]]
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        norm = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        nx, ny, nz = nx / norm, ny / norm, nz / norm
        # rotate the normal the same way as the points to test visibility
        sx, sy_, depth_n = _iso_project((nx, ny, nz), yaw, pitch)
        if depth_n > 1e-6:  # facing away from the viewer
            continue
        shade = 0.55 + 0.45 * max(0.0, nx * light[0] + ny * light[1] + nz * light[2])
        depth = sum(proj[i][2] for i in face) / len(face)
        pts = [(ox + proj[i][0] * scale, oy + proj[i][1] * scale) for i in face]
        faces.append((depth, pts, shade))
    faces.sort(key=lambda f: -f[0])
    for _, pts, shade in faces:
        fill = tuple(min(1.0, c * shade + 0.08) for c in base)
        page.polygon(pts, 0.35, edges, fill)  # type: ignore[arg-type]


def model_thumb_page(spec: Dict[str, Any]) -> Page:
    page = Page(320, 220)
    iso_view(page, spec, 26, 18, 268, 170, base=(0.72, 0.78, 0.86), edges=(0.24, 0.27, 0.32))
    return page


# --------------------------------------------------------------------------- #
# Drawing sheet
# --------------------------------------------------------------------------- #
def _dim_h(page: Page, x1: float, x2: float, y: float, ext_from: float, label: str) -> None:
    """Horizontal dimension at height y with extension lines from ext_from."""
    page.line(x1, ext_from, x1, y + (3 if y > ext_from else -3), THIN)
    page.line(x2, ext_from, x2, y + (3 if y > ext_from else -3), THIN)
    page.line(x1, y, x2, y, THIN)
    for xa, direction in ((x1, 1), (x2, -1)):
        page.polygon([(xa, y), (xa + 5 * direction, y - 1.6), (xa + 5 * direction, y + 1.6)], 0.3, BLACK, BLACK)
    tw = text_width(label, 7)
    page.rect((x1 + x2) / 2 - tw / 2 - 2, y - 5.5, tw + 4, 8, 0, None, (1, 1, 1))
    page.text((x1 + x2) / 2, y + 2.4, label, 7, anchor="middle")


def _dim_v(page: Page, y1: float, y2: float, x: float, ext_from: float, label: str) -> None:
    page.line(ext_from, y1, x + (3 if x > ext_from else -3), y1, THIN)
    page.line(ext_from, y2, x + (3 if x > ext_from else -3), y2, THIN)
    page.line(x, y1, x, y2, THIN)
    for ya, direction in ((y1, 1), (y2, -1)):
        page.polygon([(x, ya), (x - 1.6, ya + 5 * direction), (x + 1.6, ya + 5 * direction)], 0.3, BLACK, BLACK)
    tw = text_width(label, 7)
    page.rect(x - tw / 2 - 2, (y1 + y2) / 2 - 5.5, tw + 4, 8, 0, None, (1, 1, 1))
    page.text(x, (y1 + y2) / 2 + 2.4, label, 7, anchor="middle")


def _views(page: Page, spec: Dict[str, Any], x: float, y: float, w: float, h: float) -> None:
    """Front and side (or top and front) orthographic views with overall dimensions."""
    shape, d, units = spec.get("shape"), _dims(spec), spec.get("units") or "in"
    if shape in ROUND:
        length, od = d[0], d[1]
        idia = d[2] if shape in ROUND3 else 0.0
        avail_w, avail_h = w * 0.62, h * 0.62
        scale = min(avail_w / length, avail_h / od)
        vx, vy = x + 30, y + h * 0.18
        L, D = length * scale, od * scale
        cy = vy + D / 2
        profile = []
        if shape == "shaft":
            steps = [(0.0, 0.14, 0.72), (0.14, 0.78, 1.0), (0.78, 0.92, 0.72), (0.92, 1.0, 0.55)]
        elif shape == "pin":
            steps = [(0.0, 0.94, 1.0), (0.94, 1.0, 0.82)]
        elif shape == "nozzle":
            steps = [(0.0, 0.35, 1.0), (0.35, 0.8, 0.7), (0.8, 1.0, 0.42)]
        elif shape == "threaded_fitting":
            steps = [(0.0, 0.38, 0.72), (0.38, 0.62, 1.0), (0.62, 1.0, 0.72)]
        else:
            steps = [(0.0, 1.0, 1.0)]
        for a, b, k in steps:
            page.rect(vx + a * L, cy - D * k / 2, (b - a) * L, D * k, THICK)
            profile.append((a, b, k))
        if shape == "threaded_fitting":
            for a, b in ((0.0, 0.38), (0.62, 1.0)):
                xx = vx + a * L + 3
                while xx < vx + b * L - 2:
                    page.line(xx, cy - D * 0.36, xx + 1.5, cy - D * 0.36 + 3, THIN)
                    page.line(xx, cy + D * 0.36, xx + 1.5, cy + D * 0.36 - 3, THIN)
                    xx += 4
        if idia:
            page.line(vx, cy - idia * scale / 2, vx + L, cy - idia * scale / 2, THIN, dash=HIDDEN_DASH)
            page.line(vx, cy + idia * scale / 2, vx + L, cy + idia * scale / 2, THIN, dash=HIDDEN_DASH)
        page.line(vx - 10, cy, vx + L + 10, cy, THIN, dash=CENTER_DASH)
        _dim_h(page, vx, vx + L, cy + D / 2 + 26, cy + D / 2 + 2, fmt(length, units))
        _dim_v(page, cy - D / 2, cy + D / 2, vx - 22, vx - 2, f"Ø{fmt(od, units)}")
        # end view
        ex = vx + L + 40 + D / 2
        page.circle(ex, cy, D / 2, THICK)
        if idia:
            page.circle(ex, cy, idia * scale / 2, THICK)
            _dim_h(page, ex - idia * scale / 2, ex + idia * scale / 2, cy - D / 2 - 18, cy,
                   f"Ø{fmt(idia, units)} THRU")
        page.line(ex - D / 2 - 8, cy, ex + D / 2 + 8, cy, THIN, dash=CENTER_DASH)
        page.line(ex, cy - D / 2 - 8, ex, cy + D / 2 + 8, THIN, dash=CENTER_DASH)
        page.text(vx + L / 2, cy + D / 2 + 46, "SIDE VIEW", 7, True, "middle")
        page.text(ex, cy + D / 2 + 46, "END VIEW", 7, True, "middle")
        return
    L, W, H = d
    scale = min((w * 0.58) / L, (h * 0.5) / W, (h * 0.28) / H)
    Ls, Ws, Hs = L * scale, W * scale, H * scale
    tx, ty = x + 34, y + 18
    # top view
    if shape in ("implant", "contoured", "casting"):
        pts = [(tx + px * scale, ty + py * scale) for px, py in _rounded_outline(L, W, min(L, W) * 0.4, 7)]
        page.polygon(pts, THICK)
        inner = [(tx + (px * 0.7 + L * 0.15) * scale, ty + (py * 0.7 + W * 0.15) * scale)
                 for px, py in _rounded_outline(L, W, min(L, W) * 0.4, 7)]
        page.polygon(inner, MED)
    elif shape == "impeller":
        r = min(Ls, Ws) / 2
        c = (tx + r, ty + r)
        page.circle(c[0], c[1], r, THICK)
        page.circle(c[0], c[1], r * 0.28, THICK)
        for i in range(7):
            a = 2 * math.pi * i / 7
            page.path([("M", c[0] + r * 0.3 * math.cos(a), c[1] + r * 0.3 * math.sin(a)),
                       ("C", c[0] + r * 0.6 * math.cos(a + 0.2), c[1] + r * 0.6 * math.sin(a + 0.2),
                        c[0] + r * 0.8 * math.cos(a + 0.5), c[1] + r * 0.8 * math.sin(a + 0.5),
                        c[0] + r * 0.95 * math.cos(a + 0.75), c[1] + r * 0.95 * math.sin(a + 0.75))], MED)
        Ls = Ws = 2 * r
    else:
        page.rect(tx, ty, Ls, Ws, THICK)
        hole = max(1.6, min(Ls, Ws) * 0.035)
        inset = max(hole * 2.6, min(Ls, Ws) * 0.1)
        if shape in ("plate", "cover", "block", "manifold", "fixture", "enclosure", "housing", "bracket",
                     "structural_fitting", "assembly", "weldment", "sheet_metal"):
            for hx, hy in ((tx + inset, ty + inset), (tx + Ls - inset, ty + inset),
                           (tx + inset, ty + Ws - inset), (tx + Ls - inset, ty + Ws - inset)):
                page.circle(hx, hy, hole, MED)
                page.line(hx - hole - 3, hy, hx + hole + 3, hy, THIN, dash=CENTER_DASH)
                page.line(hx, hy - hole - 3, hx, hy + hole + 3, THIN, dash=CENTER_DASH)
        if shape in ("housing", "enclosure", "cover", "structural_fitting"):
            page.rect(tx + Ls * 0.12, ty + Ws * 0.18, Ls * 0.76, Ws * 0.64, MED)
        if shape == "heatsink":
            fins = max(4, min(14, int(L / (H * 0.35 + 1e-6))))
            pitch = Ls / fins
            for i in range(fins):
                fx = tx + i * pitch + pitch * 0.3
                page.rect(fx, ty, pitch * 0.4, Ws, THIN)
        if shape == "manifold":
            for i in range(3):
                page.circle(tx + Ls * (0.25 + 0.25 * i), ty + Ws / 2, min(Ls, Ws) * 0.08, MED)
        if shape == "sheet_metal":
            page.line(tx, ty + Ws * 0.15, tx + Ls, ty + Ws * 0.15, THIN, dash=HIDDEN_DASH)
            page.line(tx, ty + Ws * 0.85, tx + Ls, ty + Ws * 0.85, THIN, dash=HIDDEN_DASH)
            page.text(tx + Ls - 4, ty + Ws * 0.15 - 3, "BEND UP 90°", 5.5, anchor="end")
    _dim_h(page, tx, tx + Ls, ty - 10, ty, fmt(L, units))
    _dim_v(page, ty, ty + Ws, tx - 18, tx, fmt(W, units))
    page.text(tx + Ls / 2, ty + Ws + 16, "TOP VIEW", 7, True, "middle")
    # front view below
    fy = ty + Ws + 38
    if shape == "bracket":
        t = min(H * 0.3, L * 0.18, W * 0.25) * scale
        page.polygon([(tx, fy), (tx + t * 1.2, fy), (tx + t * 1.2, fy + Hs - t), (tx + Ls, fy + Hs - t),
                      (tx + Ls, fy + Hs), (tx, fy + Hs)], THICK)
    elif shape == "heatsink":
        base = Hs * 0.28
        page.rect(tx, fy + Hs - base, Ls, base, THICK)
        fins = max(4, min(14, int(L / (H * 0.35 + 1e-6))))
        pitch = Ls / fins
        for i in range(fins):
            fx = tx + i * pitch + pitch * 0.3
            page.rect(fx, fy, pitch * 0.4, Hs - base, MED)
    else:
        page.rect(tx, fy, Ls, Hs, THICK)
        if shape in ("housing", "enclosure", "cover"):
            page.line(tx + Ls * 0.12, fy, tx + Ls * 0.12, fy + Hs * 0.85, THIN, dash=HIDDEN_DASH)
            page.line(tx + Ls * 0.88, fy, tx + Ls * 0.88, fy + Hs * 0.85, THIN, dash=HIDDEN_DASH)
    _dim_v(page, fy, fy + Hs, tx + Ls + 18, tx + Ls, fmt(H, units))
    page.text(tx + Ls / 2, fy + Hs + 16, "FRONT VIEW", 7, True, "middle")


def _callouts(page: Page, spec: Dict[str, Any], x: float, y: float) -> None:
    for i, text in enumerate((spec.get("callouts") or [])[:4]):
        yy = y + i * 13
        page.line(x - 18, yy - 12, x - 2, yy - 3, THIN)
        page.polygon([(x - 18, yy - 12), (x - 13.5, yy - 11.2), (x - 15.5, yy - 8.6)], 0.3, BLACK, BLACK)
        page.text(x, yy, clean(text), 7)


def drawing_pages(spec: Dict[str, Any]) -> List[Page]:
    page = Page(*LETTER_LANDSCAPE)
    W, H = page.width, page.height
    m = 18.0
    legend = spec.get("legend")
    company = clean(spec.get("company"))
    units = spec.get("units") or "in"
    # border with zones
    page.rect(m, m, W - 2 * m, H - 2 * m, THICK)
    inner = m + 10
    page.rect(inner, inner, W - 2 * inner, H - 2 * inner, MED)
    for i in range(8):
        zx = inner + (W - 2 * inner) * (i + 0.5) / 8
        page.text(zx, m + 7.5, str(8 - i), 6, anchor="middle")
        page.text(zx, H - m - 2.5, str(8 - i), 6, anchor="middle")
        if i:
            bx = inner + (W - 2 * inner) * i / 8
            page.line(bx, m, bx, inner, THIN)
            page.line(bx, H - inner, bx, H - m, THIN)
    for i, letter in enumerate("DCBA"):
        zy = inner + (H - 2 * inner) * (i + 0.5) / 4
        page.text(m + 5, zy + 2, letter, 6, anchor="middle")
        page.text(W - m - 5, zy + 2, letter, 6, anchor="middle")
        if i:
            by = inner + (H - 2 * inner) * i / 4
            page.line(m, by, inner, by, THIN)
            page.line(W - inner, by, W - m, by, THIN)

    top = inner + 4
    if legend == "cui":
        page.text(W / 2, inner + 12, "CUI", 12, True, "middle")
        page.text(W / 2, H - inner - 5, "CUI", 12, True, "middle")
        top += 12
    elif legend in ("itar", "ear"):
        text = legend_text(legend, company, spec.get("eccn", ""))
        lines = wrap(text, W - 2 * inner - 16, 6.2, True)
        box_h = 8 + len(lines) * 7.6
        page.rect(inner + 4, top, W - 2 * inner - 8, box_h, 1.2, BLACK, (1.0, 0.97, 0.9))
        for i, line in enumerate(lines):
            page.text(inner + 10, top + 10 + i * 7.6, line, 6.2, True)
        top += box_h + 4

    # revision block, top right
    rb_w = 250.0
    rb_x = W - inner - rb_w
    revs = [r for r in (spec.get("revisions") or []) if isinstance(r, dict)][:4] or [
        {"rev": spec.get("rev") or "A", "description": "RELEASED", "date": spec.get("date") or ""}]
    page.rect(rb_x, top, rb_w, 12, MED, BLACK, (0.93, 0.93, 0.92))
    page.text(rb_x + rb_w / 2, top + 8.6, "REVISIONS", 6.5, True, "middle")
    cols = [(0, 26, "REV"), (26, 150, "DESCRIPTION"), (176, 44, "DATE"), (220, 30, "APPR")]
    y = top + 12
    for cx, cw, label in cols:
        page.rect(rb_x + cx, y, cw, 10, THIN)
        page.text(rb_x + cx + cw / 2, y + 7.2, label, 5.5, True, "middle")
    y += 10
    for r in revs:
        values = [clean(r.get("rev")), clean(r.get("description")), clean(r.get("date")), "TW"]
        for (cx, cw, _), val in zip(cols, values):
            page.rect(rb_x + cx, y, cw, 11, THIN)
            page.text_fit(rb_x + cx + 2.5, y + 7.8, val, cw - 5, 5.8)
        y += 11
    rev_bottom = y

    # title block, bottom right
    tb_w, tb_h = 300.0, 150.0
    tb_x, tb_y = W - inner - tb_w, H - inner - tb_h
    page.rect(tb_x, tb_y, tb_w, tb_h, THICK)
    # tolerance block on the left of the title block
    tol_w = 108.0
    page.rect(tb_x, tb_y, tol_w, tb_h, MED)
    ty = tb_y + 10
    for line in ["UNLESS OTHERWISE SPECIFIED:", f"DIMENSIONS ARE IN {'MILLIMETERS' if units == 'mm' else 'INCHES'}",
                 "TOLERANCES:"]:
        page.text(tb_x + 4, ty, line, 5.4, True)
        ty += 7.5
    for piece in (clean(spec.get("tolerances")) or default_tolerances(units)).split("  "):
        if piece.strip():
            page.text_fit(tb_x + 8, ty, piece.strip(), tol_w - 12, 5.8)
            ty += 7.5
    ty += 2
    for line in ["INTERPRET PER ASME Y14.5-2018", "BREAK SHARP EDGES", "DO NOT SCALE DRAWING"]:
        page.text(tb_x + 4, ty, line, 5.2)
        ty += 7
    # third angle projection symbol
    sx, sy = tb_x + 30, tb_y + tb_h - 20
    page.polygon([(sx - 12, sy - 6), (sx + 4, sy - 9), (sx + 4, sy + 9), (sx - 12, sy + 6)], MED)
    page.circle(sx + 22, sy, 9, MED)
    page.circle(sx + 22, sy, 4, MED)
    page.text(tb_x + tol_w / 2, tb_y + tb_h - 4, "THIRD ANGLE PROJECTION", 4.8, anchor="middle")

    rx = tb_x + tol_w
    rw = tb_w - tol_w
    rows = [
        (26, [("", company.upper(), 11, True)]),
        (24, [("TITLE", clean(spec.get("title")).upper(), 9, True)]),
        (22, [("MATERIAL", clean(spec.get("material")).upper(), 6.6, False)]),
        (22, [("FINISH", clean(spec.get("finish") or "NONE").upper(), 6.6, False)]),
        (28, [("SIZE", "A", 9, True), ("DWG NO.", clean(spec.get("part_number")), 10, True),
              ("REV", clean(spec.get("rev") or "-"), 10, True)]),
        (28, [("DRAWN", clean(spec.get("drawn_by") or "T. WALSH").upper(), 6.4, False),
              ("DATE", clean(spec.get("date") or ""), 6.4, False),
              ("SCALE", clean(spec.get("scale") or "1:1"), 6.4, False),
              ("SHEET", clean(spec.get("sheet") or "1 OF 1"), 6.4, False)]),
    ]
    yy = tb_y
    for height, cells in rows:
        if len(cells) == 3:
            widths = [rw * 0.14, rw * 0.66, rw * 0.2]
        elif len(cells) == 4:
            widths = [rw * 0.3, rw * 0.26, rw * 0.2, rw * 0.24]
        else:
            widths = [rw]
        cx = rx
        for (label, value, size, bold), cw in zip(cells, widths):
            page.rect(cx, yy, cw, height, THIN)
            if label:
                page.text(cx + 3, yy + 6.6, label, 4.8, True, color=GREY)
                page.text_fit(cx + 3, yy + height - 6, value, cw - 6, size, bold)
            else:
                page.text_fit(cx + cw / 2, yy + height - 8.5, value, cw - 8, size, bold, "middle")
            cx += cw
        yy += height

    # notes, bottom left
    notes = [clean(n) for n in spec.get("notes") or [] if clean(n)]
    notes_w = tb_x - inner - 26
    note_lines: List[Tuple[str, str]] = []
    for i, note in enumerate(notes, start=1):
        for j, piece in enumerate(wrap(note, notes_w - 18, 7)):
            note_lines.append((f"{i}." if j == 0 else "", piece))
    ny = H - inner - 12 - len(note_lines) * 9.2
    if legend == "proprietary":
        prop = wrap(legend_text("proprietary", company), notes_w, 5.6)
        ny -= len(prop) * 7 + 8
    page.text(inner + 12, ny - 6, "NOTES:", 8, True)
    for num, piece in note_lines:
        page.text(inner + 12, ny + 6, num, 7, True)
        page.text(inner + 26, ny + 6, piece, 7)
        ny += 9.2
    if legend == "proprietary":
        ny += 6
        for line in wrap(legend_text("proprietary", company), notes_w, 5.6):
            page.text(inner + 12, ny + 4, line, 5.6, True, color=GREY)
            ny += 7
    if legend == "cui":
        cb_w, cb_h = 190.0, 46.0
        cbx, cby = tb_x - cb_w - 8, H - inner - cb_h - 4
        page.rect(cbx, cby, cb_w, cb_h, MED)
        for i, line in enumerate(wrap(legend_text("cui", company, poc=spec.get("drawn_by", "")), cb_w - 8, 5.2, True)[:6]):
            page.text(cbx + 4, cby + 8 + i * 6.6, line, 5.2, True)

    # views and isometric
    views_top = top + 16
    notes_top = H - inner - 12 - len(note_lines) * 9.2 - 30
    area_h = max(160.0, min(notes_top, tb_y) - views_top - 10)
    _views(page, spec, inner + 20, views_top + 14, rb_x - inner - 40, area_h)
    iso_top = rev_bottom + 12
    iso_h = max(90.0, tb_y - iso_top - 30)
    iso_view(page, spec, rb_x + 10, iso_top, rb_w - 20, iso_h, base=(0.86, 0.88, 0.9), edges=(0.15, 0.15, 0.17))
    page.text(rb_x + rb_w / 2, iso_top + iso_h + 10, "ISOMETRIC VIEW", 6.5, True, "middle")
    if spec.get("callouts"):
        _callouts(page, spec, rb_x - 150, views_top + 30)
    return [page]


# --------------------------------------------------------------------------- #
# STEP (ISO 10303-21) with a faceted B-rep
# --------------------------------------------------------------------------- #
def step_file(spec: Dict[str, Any]) -> str:
    mesh = build_mesh(spec)
    units = spec.get("units") or "in"
    pn = clean(spec.get("part_number") or "PART").replace("'", "")
    title = clean(spec.get("title") or pn).replace("'", "")
    rev = clean(spec.get("rev") or "-").replace("'", "")
    system = clean(spec.get("originating_system") or "RFQ Router demo").replace("'", "")
    author = clean(spec.get("author") or "engineering").replace("'", "")
    schema = spec.get("schema") or "AP214"
    schema_name = {"AP203": "CONFIG_CONTROL_DESIGN",
                   "AP242": "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }"}.get(
        schema, "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }")
    lines: List[str] = []
    n = [0]

    def ent(text: str) -> int:
        n[0] += 1
        lines.append(f"#{n[0]}={text};")
        return n[0]

    app = ent("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    ent(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})")
    pctx = ent(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    prod = ent(f"PRODUCT('{pn}','{title}','Rev {rev}',(#{pctx}))")
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
    faces = []
    for face in mesh.f:
        loop = ent("POLY_LOOP('',(" + ",".join(f"#{pts[i]}" for i in face) + "))")
        bound = ent(f"FACE_OUTER_BOUND('',#{loop},.T.)")
        faces.append(ent(f"FACE('',(#{bound}))"))
    shell = ent("CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in faces) + "))")
    brep = ent(f"FACETED_BREP('{pn}',#{shell})")
    origin = ent("CARTESIAN_POINT('',(0.,0.,0.))")
    zdir = ent("DIRECTION('',(0.,0.,1.))")
    xdir = ent("DIRECTION('',(1.,0.,0.))")
    axis = ent(f"AXIS2_PLACEMENT_3D('',#{origin},#{zdir},#{xdir})")
    rep = ent(f"FACETED_BREP_SHAPE_REPRESENTATION('{pn}',(#{axis},#{brep}),#{ctx})")
    ent(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
    header = [
        "ISO-10303-21;",
        "HEADER;",
        f"FILE_DESCRIPTION(('{title}','Faceted model for quoting'),'2;1');",
        f"FILE_NAME('{clean(spec.get('name') or pn + '.step')}','{clean(spec.get('date') or '2026-09-01')}T08:00:00',"
        f"('{author}'),(''),'RFQ Router demo','{system}','');",
        f"FILE_SCHEMA(('{schema_name}'));",
        "ENDSEC;",
        "DATA;",
    ]
    return "\n".join(header + lines + ["ENDSEC;", "END-ISO-10303-21;", ""])
