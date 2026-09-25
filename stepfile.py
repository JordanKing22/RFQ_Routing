"""
Read a STEP (ISO 10303-21) file the way a quoting tool would: the header (file name, originating
system, schema), the PRODUCT (part number and title), the length unit, and, for faceted B-rep models,
the mesh itself so the browser can show the part in 3D. Standard library only.

    info = stepfile.parse(data)     -> {"part_number", "title", "units", "schema", "system", "mesh"}
    stepfile.iso_svg(info["mesh"])  -> a shaded isometric SVG for the file's tile

Solid models from real CAD systems (advanced B-rep with curved faces) still give the header and
product fields; only their mesh is left empty.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import docgen

_ENTITY = re.compile(r"#(\d+)\s*=\s*(.*?);\s*(?=#\d+\s*=|ENDSEC;)", re.S)
_REF = re.compile(r"#(\d+)")


def _strings(args: str) -> List[str]:
    return [s.replace("''", "'") for s in re.findall(r"'((?:[^']|'')*)'", args)]


def parse(data: bytes, max_bytes: int = 40_000_000) -> Dict[str, Any]:
    """Parse the parts of a STEP file a quote needs. Never raises on bad input."""
    info: Dict[str, Any] = {"part_number": None, "title": None, "units": None, "schema": None,
                            "system": None, "file_name": None, "mesh": None, "error": None}
    try:
        text = data[:max_bytes].decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        info["error"] = "not a text file"
        return info
    if "ISO-10303-21" not in text[:200]:
        info["error"] = "not a STEP file (no ISO-10303-21 header)"
        return info
    header = text.split("DATA;", 1)[0]
    m = re.search(r"FILE_NAME\s*\((.*?)\)\s*;", header, re.S)
    if m:
        names = _strings(m.group(1))
        if names:
            info["file_name"] = names[0]
        if len(names) >= 6:
            info["system"] = names[-2] or names[-3] or None
    m = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'", header)
    if m:
        schema = m.group(1).upper()
        info["schema"] = ("AP242" if "AP242" in schema or "442" in schema else
                          "AP214" if "AUTOMOTIVE" in schema or "214" in schema else
                          "AP203" if "CONFIG_CONTROL" in schema or "203" in schema else m.group(1))
    body = text.split("DATA;", 1)[-1]
    entities: Dict[int, Tuple[str, str]] = {}
    for num, rest in _ENTITY.findall(body + "ENDSEC;"):
        kind, _, args = rest.strip().partition("(")
        entities[int(num)] = (kind.strip().upper(), args)
    for kind, args in entities.values():
        if kind == "PRODUCT":
            vals = _strings(args)
            if vals:
                info["part_number"] = vals[0] or None
                info["title"] = (vals[1] if len(vals) > 1 else None) or None
            break
    units_blob = " ".join(args for kind, args in entities.values() if "UNIT" in kind or kind == "")
    raw = body.upper()
    if "CONVERSION_BASED_UNIT('INCH'" in raw.replace(" ", ""):
        info["units"] = "in"
    elif ".MILLI.,.METRE." in raw.replace(" ", ""):
        info["units"] = "mm"
    del units_blob
    info["mesh"] = _mesh(entities, info["units"] or "in")
    return info


def _mesh(entities: Dict[int, Tuple[str, str]], units: str, max_faces: int = 20000) -> Optional[Dict[str, Any]]:
    """Faceted B-rep: CARTESIAN_POINT -> POLY_LOOP -> FACE_OUTER_BOUND -> FACE."""
    point_index: Dict[int, int] = {}
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    def point(ref: int) -> Optional[int]:
        if ref in point_index:
            return point_index[ref]
        ent = entities.get(ref)
        if not ent or ent[0] != "CARTESIAN_POINT":
            return None
        nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", ent[1].split(",", 1)[-1])
        if len(nums) < 3:
            return None
        point_index[ref] = len(vertices)
        vertices.append([float(n) for n in nums[:3]])
        return point_index[ref]

    for kind, args in entities.values():
        if kind != "POLY_LOOP":
            continue
        idx = [point(int(r)) for r in _REF.findall(args)]
        if len(idx) >= 3 and None not in idx:
            faces.append(idx)  # type: ignore[arg-type]
        if len(faces) >= max_faces:
            break
    if not faces:
        return None
    xs, ys, zs = zip(*vertices)
    return {"units": units, "vertices": [[round(c, 4) for c in v] for v in vertices], "faces": faces,
            "bbox": [round(max(xs) - min(xs), 4), round(max(ys) - min(ys), 4), round(max(zs) - min(zs), 4)]}


def bbox_phrase(mesh: Optional[Dict[str, Any]], units: Optional[str]) -> Optional[str]:
    if not mesh:
        return None
    u = "MM" if units == "mm" else "IN"
    fmt = (lambda v: f"{v:.1f}") if units == "mm" else (lambda v: f"{v:.3f}")
    return " x ".join(fmt(v) for v in mesh["bbox"]) + f" {u}"


def iso_svg(mesh: Optional[Dict[str, Any]], width: float = 320, height: float = 220) -> str:
    """Shaded isometric view of a mesh (painter's algorithm), the same look as the sample STEP tiles."""
    page = docgen.Page(width, height)
    if mesh and mesh.get("faces"):
        verts = mesh["vertices"]
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        cz = sum(v[2] for v in verts) / len(verts)
        yaw, pitch = math.radians(-38.0), math.radians(57.0)

        def project(p: List[float]) -> Tuple[float, float, float]:
            x, y, z = p[0] - cx, p[1] - cy, p[2] - cz
            x, y = x * math.cos(yaw) - y * math.sin(yaw), x * math.sin(yaw) + y * math.cos(yaw)
            y, z = y * math.cos(pitch) - z * math.sin(pitch), y * math.sin(pitch) + z * math.cos(pitch)
            return x, -z, y

        proj = [project(v) for v in verts]
        xs, ys = [p[0] for p in proj], [p[1] for p in proj]
        scale = min((width - 52) / max(max(xs) - min(xs), 1e-6), (height - 50) / max(max(ys) - min(ys), 1e-6))
        ox = width / 2 - (max(xs) + min(xs)) / 2 * scale
        oy = height / 2 - (max(ys) + min(ys)) / 2 * scale
        light = (-0.35, -0.55, 0.76)
        drawn = []
        for face in mesh["faces"]:
            a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
            u = [b[i] - a[i] for i in range(3)]
            w = [c[i] - a[i] for i in range(3)]
            n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]]
            norm = math.sqrt(sum(v * v for v in n)) or 1.0
            n = [v / norm for v in n]
            if project([n[0] + cx, n[1] + cy, n[2] + cz])[2] > 1e-6:
                continue  # facing away
            shade = 0.55 + 0.45 * max(0.0, sum(n[i] * light[i] for i in range(3)))
            depth = sum(proj[i][2] for i in face) / len(face)
            drawn.append((depth, [(ox + proj[i][0] * scale, oy + proj[i][1] * scale) for i in face], shade))
        drawn.sort(key=lambda f: -f[0])
        for _, pts, shade in drawn:
            page.polygon(pts, 0.35, (0.24, 0.27, 0.32), tuple(min(1.0, c * shade + 0.08) for c in (0.72, 0.78, 0.86)))
    return docgen.to_svg(page, background=(0.96, 0.97, 0.98))
