"""
Write Outlook .msg files for the test fixtures, since there is no Outlook here to save them. Test
tooling only: the demo reads .msg files (mailfile.py) and never writes them. Standard library only.

    from tools.msgwriter import build_msg
    data = build_msg({"subject": ..., "from_name": ..., "from_email": ..., "to": [...], ...})

A .msg file is a Compound File (MS-CFB) of MAPI properties (MS-OXMSG). This writes version 3
(512-byte sectors): streams under 4096 bytes go in the mini stream, the FAT gets DIFAT sectors once
it outgrows the 109 slots in the header, and every storage's children form a real red-black tree.

Also here, for PR_RTF_COMPRESSED bodies: compress_rtf, an LZFu compressor (MS-OXRTFCP) that finds
back-references in the 4096-byte dictionary and writes the CRC, and html_to_rtf, which wraps HTML
in RTF the way Outlook does (\\fromhtml1 with \\htmltag groups, MS-OXRTFEX).
"""

from __future__ import annotations

import datetime as _dt
import email.utils
import html
import re
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union

# --------------------------------------------------------------------------- #
# Compound File writer (MS-CFB)
# --------------------------------------------------------------------------- #
FREESECT = 0xFFFFFFFF
ENDOFCHAIN = 0xFFFFFFFE
FATSECT = 0xFFFFFFFD
DIFSECT = 0xFFFFFFFC
NOSTREAM = 0xFFFFFFFF
MINI_CUTOFF = 4096
SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
# The CLSID Outlook gives the root storage of a saved message.
MSG_CLSID = bytes.fromhex("0B0D020000000000C000000000000046")

Tree = Dict[str, Union[bytes, "Tree"]]


def _cfb_key(name: str) -> Tuple[int, str]:
    """MS-CFB sibling order: shorter names first, then by upper-cased characters."""
    return (len(name.encode("utf-16-le")) // 2, name.upper())


class _Entry:
    __slots__ = ("name", "kind", "data", "left", "right", "child", "red", "start", "size", "clsid")

    def __init__(self, name: str, kind: int, data: bytes = b"", clsid: bytes = b"\0" * 16):
        self.name = name
        self.kind = kind          # 1 storage, 2 stream, 5 root
        self.data = data
        self.left = self.right = self.child = NOSTREAM
        self.red = False
        self.start = ENDOFCHAIN
        self.size = 0
        self.clsid = clsid


def _red_black(entries: List[_Entry], ids: List[int]) -> int:
    """Link sorted sibling ids into a balanced binary tree and color it so it is a valid
    red-black tree: every path has the same number of black nodes, and a red node only has black
    parents. A midpoint-split tree has all its empty links on the last two levels, so coloring
    the deepest level red (unless that level is full) does it. Returns the root id."""
    depth: Dict[int, int] = {}

    def build(lo: int, hi: int, d: int) -> int:
        if lo >= hi:
            return NOSTREAM
        mid = (lo + hi) // 2
        node = ids[mid]
        depth[node] = d
        entries[node].left = build(lo, mid, d + 1)
        entries[node].right = build(mid + 1, hi, d + 1)
        return node

    root = build(0, len(ids), 0)
    if ids:
        deepest = max(depth.values())
        full = (len(ids) + 1) & len(ids) == 0
        for node, d in depth.items():
            entries[node].red = d == deepest and not full and d > 0
    return root


def write_cfb(tree: Tree, *, root_clsid: bytes = MSG_CLSID, sector_shift: int = 9) -> bytes:
    """A Compound File from a tree of {name: bytes (a stream) or dict (a storage)}. sector_shift
    9 writes version 3 (512-byte sectors); 12 writes version 4 (4096), for reader tests."""
    ssize = 1 << sector_shift
    per = ssize // 4
    entries: List[_Entry] = [_Entry("Root Entry", 5, clsid=root_clsid)]

    def add(storage_id: int, node: Tree) -> None:
        ids = []
        for name in sorted(node, key=_cfb_key):
            value = node[name]
            if len(name) > 31:
                raise ValueError(f"CFB names are at most 31 characters: {name}")
            if isinstance(value, dict):
                entries.append(_Entry(name, 1))
                ids.append(len(entries) - 1)
                add(len(entries) - 1, value)
            else:
                entries.append(_Entry(name, 2, bytes(value)))
                ids.append(len(entries) - 1)
        entries[storage_id].child = _red_black(entries, ids)

    add(0, tree)

    # Small streams go in the mini stream, 64 bytes a mini sector.
    ministream = bytearray()
    minifat: List[int] = []
    big: List[_Entry] = []
    for e in entries:
        if e.kind != 2:
            continue
        e.size = len(e.data)
        if e.size == 0:
            e.start = ENDOFCHAIN
        elif e.size < MINI_CUTOFF:
            count = (e.size + 63) // 64
            e.start = len(minifat)
            minifat.extend(range(e.start + 1, e.start + count))
            minifat.append(ENDOFCHAIN)
            ministream += e.data + b"\0" * (count * 64 - e.size)
        else:
            big.append(e)

    def sectors(n_bytes: int) -> int:
        return (n_bytes + ssize - 1) // ssize

    dir_count = sectors(len(entries) * 128)
    minifat_count = sectors(len(minifat) * 4)
    mini_count = sectors(len(ministream))
    big_counts = [sectors(e.size) for e in big]
    data_count = dir_count + minifat_count + mini_count + sum(big_counts)
    # The FAT covers every sector, its own and the DIFAT's included.
    fat_count = 1
    while True:
        difat_count = max(0, (fat_count - 109 + per - 2) // (per - 1))
        if fat_count * per >= data_count + fat_count + difat_count:
            break
        fat_count += 1

    fat = [FREESECT] * (fat_count * per)
    nxt = 0

    def run(count: int, mark: Optional[int] = None) -> int:
        nonlocal nxt
        start = nxt
        for k in range(count):
            fat[start + k] = mark if mark is not None else (start + k + 1 if k < count - 1 else ENDOFCHAIN)
        nxt += count
        return start if count else ENDOFCHAIN

    fat_start = run(fat_count, FATSECT)
    difat_start = run(difat_count, DIFSECT)
    dir_start = run(dir_count)
    minifat_start = run(minifat_count)
    mini_start = run(mini_count)
    for e, count in zip(big, big_counts):
        e.start = run(count)
    root = entries[0]
    root.start = mini_start if ministream else ENDOFCHAIN
    root.size = len(ministream)

    fat_ids = list(range(fat_start, fat_start + fat_count))
    difat_array = fat_ids[:109] + [FREESECT] * (109 - min(109, fat_count))
    difat_sectors = []
    rest = fat_ids[109:]
    for k in range(difat_count):
        chunk = rest[k * (per - 1):(k + 1) * (per - 1)]
        chunk += [FREESECT] * (per - 1 - len(chunk))
        chunk.append(difat_start + k + 1 if k < difat_count - 1 else ENDOFCHAIN)
        difat_sectors.append(struct.pack(f"<{per}I", *chunk))

    header = bytearray(512)
    header[0:8] = SIGNATURE
    struct.pack_into("<HHHHH", header, 24, 0x003E, 4 if sector_shift == 12 else 3, 0xFFFE, sector_shift, 6)
    struct.pack_into("<IIIIIIIII", header, 40,
                     dir_count if sector_shift == 12 else 0, fat_count, dir_start, 0, MINI_CUTOFF,
                     minifat_start if minifat else ENDOFCHAIN, minifat_count,
                     difat_start if difat_count else ENDOFCHAIN, difat_count)
    struct.pack_into("<109I", header, 76, *difat_array)

    directory = bytearray()
    for e in entries:
        name = e.name.encode("utf-16-le") + b"\0\0"
        directory += struct.pack("<64sHBBIII16sIQQIQ", name, len(name), e.kind, 0 if e.red else 1,
                                 e.left, e.right, e.child, e.clsid, 0, 0, 0,
                                 e.start if e.kind != 1 else 0, e.size)
    empty = struct.pack("<64sHBBIII16sIQQIQ", b"", 0, 0, 0, NOSTREAM, NOSTREAM, NOSTREAM, b"\0" * 16, 0, 0, 0, 0, 0)
    while len(directory) < dir_count * ssize:
        directory += empty

    def padded(raw: bytes, count: int) -> bytes:
        return raw + b"\0" * (count * ssize - len(raw))

    out = bytearray(header)
    if sector_shift == 12:
        out += b"\0" * (ssize - 512)
    out += struct.pack(f"<{len(fat)}I", *fat)
    for sector in difat_sectors:
        out += sector
    out += directory
    out += padded(struct.pack(f"<{len(minifat)}I", *minifat), minifat_count) if minifat else b""
    out += padded(bytes(ministream), mini_count)
    for e, count in zip(big, big_counts):
        out += padded(e.data, count)
    return bytes(out)


# --------------------------------------------------------------------------- #
# Compressed RTF (MS-OXRTFCP)
# --------------------------------------------------------------------------- #
RTF_PREBUF = (b"{\\rtf1\\ansi\\mac\\deff0\\deftab720{\\fonttbl;}{\\f0\\fnil \\froman \\fswiss \\fmodern "
              b"\\fscript \\fdecor MS Sans SerifSymbolArialTimes New RomanCourier{\\colortbl\\red0\\green0"
              b"\\blue0\r\n\\par \\pard\\plain\\f0\\fs20\\b\\i\\u\\tab\\tx")


def rtf_crc(data: bytes) -> int:
    """MS-OXRTFCP's CRC: the CRC-32 table with no inversion at the start or the end."""
    return zlib.crc32(data, 0xFFFFFFFF) ^ 0xFFFFFFFF


def compress_rtf(raw: bytes, *, candidates: int = 256) -> bytes:
    """LZFu-compress RTF: an 8-token run per control byte, each token a literal byte or a 2-byte
    reference (12-bit dictionary offset, 4-bit length - 2) to 2 to 17 bytes already in the 4096-byte
    dictionary, which starts with the prebuffer. The longest match wins, and among equal matches the
    lowest dictionary offset, which is what the specification's own example output shows. A
    reference may run into the bytes it is writing (a repeat). Ends with the end marker, a
    reference to the current write position."""
    raw = bytes(raw)
    # One linear buffer: 4096 zeros, the prebuffer, then the input. Padded index i is dictionary
    # offset (i - 4096) % 4096, and a match that runs past the write position reads the input
    # itself, exactly as the decompressor will have written it.
    buf = bytes(4096) + RTF_PREBUF + raw
    base = 4096 + len(RTF_PREBUF)
    chains: Dict[bytes, List[int]] = {}
    indexed = 4096  # padded indices below this are in `chains`
    out = bytearray()
    tokens: List[bytes] = []
    flags = 0

    def emit(token: bytes, is_ref: bool) -> None:
        nonlocal flags
        if is_ref:
            flags |= 1 << len(tokens)
        tokens.append(token)
        if len(tokens) == 8:
            flush()

    def flush() -> None:
        nonlocal flags
        out.append(flags)
        for t in tokens:
            out.extend(t)
        tokens.clear()
        flags = 0

    pos = base
    end = len(buf)
    while pos < end:
        while indexed < pos:  # every offset already written, up to the byte before this one
            chains.setdefault(buf[indexed:indexed + 2], []).append(indexed)
            indexed += 1
        best_len, best_off = 0, 0
        limit = min(17, end - pos)
        if limit >= 2:
            lowest = max(4096, pos - 4095)  # before the ring wraps, only offsets written so far
            for j in reversed(chains.get(buf[pos:pos + 2], ())[-candidates:]):
                if j < lowest:
                    break
                length = 2
                while length < limit and buf[j + length] == buf[pos + length]:
                    length += 1
                off = (j - 4096) & 0xFFF
                if length > best_len or (length == best_len and off < best_off):
                    best_len, best_off = length, off
        step = best_len if best_len >= 2 else 1
        if best_len >= 2:
            emit(struct.pack(">H", (best_off << 4) | (best_len - 2)), True)
        else:
            emit(buf[pos:pos + 1], False)
        pos += step
    write = (pos - 4096) & 0xFFF
    emit(struct.pack(">H", write << 4), True)
    if tokens:
        flush()
    return struct.pack("<IIII", len(out) + 12, len(raw), 0x75465A4C, rtf_crc(bytes(out))) + bytes(out)


def _rtf_escape(text: str) -> str:
    """Text as RTF: \\ { } escaped, anything outside ASCII as \\uN? (a UTF-16 unit each)."""
    parts = []
    for ch in text:
        code = ord(ch)
        if ch in "\\{}":
            parts.append("\\" + ch)
        elif ch == "\n":
            parts.append("\\par\n")
        elif ch == "\t":
            parts.append("\\tab ")
        elif 32 <= code < 128:
            parts.append(ch)
        elif code < 32:
            continue
        else:
            for unit in struct.unpack(f"<{len(ch.encode('utf-16-le')) // 2}H", ch.encode("utf-16-le")):
                parts.append(f"\\u{unit - 65536 if unit > 32767 else unit}?")
    return "".join(parts)


_RTF_HEAD = ("{\\rtf1\\ansi\\ansicpg1252\\deff0{\\fonttbl{\\f0\\fswiss\\fcharset0 Arial;}"
             "{\\f1\\fmodern Courier New;}}{\\colortbl;\\red0\\green0\\blue0;}"
             "{\\*\\generator msgwriter;}{\\info{\\author Fixture Writer}}\\uc1\\pard\\plain\\f0\\fs20 ")


def text_to_rtf(text: str) -> bytes:
    """Plain text as a small RTF document."""
    return (_RTF_HEAD.replace("\\deff0", "\\fromtext \\deff0", 1) + _rtf_escape(text) + "\\par\n}").encode("ascii")


_HTML_TOKEN = re.compile(r"(<!--.*?-->|<[^>]*>)|(&(?:#\d+|#x[0-9a-fA-F]+|\w+);)|([^<&]+|&)", re.S)


def html_to_rtf(markup: str) -> bytes:
    """HTML wrapped in RTF the way Outlook does it (MS-OXRTFEX): every tag in an \\htmltag
    group, the text as RTF, entities as an \\htmltag with their character in \\htmlrtf (RTF-only
    text a de-encapsulating reader must drop), and a \\par in \\htmlrtf after each block so a
    plain RTF reader still sees lines."""
    parts = [_RTF_HEAD.replace("\\deff0", "\\fromhtml1 \\deff0", 1)]
    for m in _HTML_TOKEN.finditer(markup):
        tag, entity, text = m.groups()
        if tag is not None:
            parts.append("{\\*\\htmltag64 " + _rtf_escape(tag).replace("\\par\n", "\\par ") + "}")
            low = tag.lower()
            if low.startswith(("</p", "<br", "</div", "</tr", "</li", "</h")):
                parts.append("\\htmlrtf \\par\n\\htmlrtf0 ")
        elif entity is not None:
            parts.append("{\\*\\htmltag84 " + entity + "}\\htmlrtf " + _rtf_escape(html.unescape(entity))
                         + "\\htmlrtf0 ")
        elif text:
            parts.append(_rtf_escape(text))
    parts.append("}")
    return "".join(parts).encode("ascii")


# --------------------------------------------------------------------------- #
# MAPI properties (MS-OXMSG)
# --------------------------------------------------------------------------- #
PT_LONG, PT_BOOLEAN, PT_SYSTIME = 0x0003, 0x000B, 0x0040
PT_STRING8, PT_UNICODE, PT_BINARY, PT_OBJECT = 0x001E, 0x001F, 0x0102, 0x000D


def _filetime(iso: Optional[str]) -> Optional[int]:
    if not iso:
        return None
    when = _dt.datetime.fromisoformat(str(iso))
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    delta = when - _dt.datetime(1601, 1, 1, tzinfo=_dt.timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 10_000_000 + delta.microseconds * 10


class _Props:
    """The properties of one storage: fixed values for __properties_version1.0 and the
    __substg1.0_ streams for strings and binaries."""

    def __init__(self, unicode: bool):
        self.unicode = unicode
        self.fixed: Dict[int, bytes] = {}
        self.streams: Tree = {}

    def string(self, pid: int, text: Optional[str]) -> None:
        if text is None:
            return
        if self.unicode:
            data, ptype, extra = str(text).encode("utf-16-le"), PT_UNICODE, 2
        else:
            data, ptype, extra = str(text).encode("cp1252", "replace"), PT_STRING8, 1
        tag = (pid << 16) | ptype
        self.streams[f"__substg1.0_{tag:08X}"] = data
        self.fixed[tag] = struct.pack("<II", len(data) + extra, 0)

    def binary(self, pid: int, data: bytes) -> None:
        tag = (pid << 16) | PT_BINARY
        self.streams[f"__substg1.0_{tag:08X}"] = bytes(data)
        self.fixed[tag] = struct.pack("<II", len(data), 0)

    def long(self, pid: int, value: int) -> None:
        self.fixed[(pid << 16) | PT_LONG] = struct.pack("<II", value & 0xFFFFFFFF, 0)

    def boolean(self, pid: int, value: bool) -> None:
        self.fixed[(pid << 16) | PT_BOOLEAN] = struct.pack("<H6x", 1 if value else 0)

    def systime(self, pid: int, ticks: Optional[int]) -> None:
        if ticks is not None:
            self.fixed[(pid << 16) | PT_SYSTIME] = struct.pack("<Q", ticks)

    def obj(self, pid: int, storage: Tree) -> None:
        tag = (pid << 16) | PT_OBJECT
        self.streams[f"__substg1.0_{tag:08X}"] = storage
        self.fixed[tag] = struct.pack("<II", 0xFFFFFFFF, 0)

    def table(self, header: bytes) -> bytes:
        """__properties_version1.0: the header, then 16 bytes a property (tag, flags readable and
        writable, value or size)."""
        rows = b"".join(struct.pack("<II", tag, 0x6) + self.fixed[tag] for tag in sorted(self.fixed))
        return header + rows


def _split_address(value: str) -> Tuple[str, str]:
    return email.utils.parseaddr(str(value))


def _legacy_dn(addr: str) -> str:
    """An Exchange legacy DN, what Outlook stores as the address of a sender in the same tenant."""
    who = re.sub(r"[^A-Za-z0-9]", "", addr.split("@")[0]).upper() or "USER"
    return f"/O=EXCHANGELABS/OU=EXCHANGE ADMINISTRATIVE GROUP (FYDIBOHF23SPDLT)/CN=RECIPIENTS/CN={who}"


def _short_name(name: str) -> str:
    """An 8.3 name for PR_ATTACH_FILENAME, the way Outlook abbreviates."""
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = re.sub(r"[^A-Za-z0-9]", "", stem).upper() or "ATTACH"
    ext = re.sub(r"[^A-Za-z0-9]", "", ext).upper()[:3]
    short = stem[:6] + "~1" if len(stem) > 8 else stem
    return short + ("." + ext if ext else "")


def _text_to_html(text: str) -> str:
    paras = "".join(f"<p class=MsoNormal>{html.escape(line) or '&nbsp;'}</p>\r\n" for line in text.split("\n"))
    return ("<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\">"
            "<style>p.MsoNormal{margin:0}</style></head><body>\r\n" + paras + "</body></html>")


def message_tree(email_dict: Dict[str, Any], unicode: bool = True, body: str = "text",
                 exchange_sender: bool = False, embedded: bool = False) -> Tree:
    """The storage tree of one message (the file's root, or an embedded message), for
    write_cfb. build_msg is write_cfb(message_tree(...))."""
    if body not in ("text", "html", "rtf", "all"):
        raise ValueError(f"body must be text, html, rtf, or all, not {body!r}")
    e = email_dict
    p = _Props(unicode)
    subject = str(e.get("subject") or "")
    p.string(0x001A, "IPM.Note")
    p.string(0x0037, subject)
    p.string(0x0E1D, subject)
    p.string(0x003D, "")
    from_name = str(e.get("from_name") or "")
    from_email = str(e.get("from_email") or "")
    if from_name or from_email:
        p.string(0x0C1A, from_name or from_email)
        p.string(0x0042, from_name or from_email)
        if exchange_sender:
            dn = _legacy_dn(from_email)
            p.string(0x0C1F, dn)
            p.string(0x0C1E, "EX")
            p.string(0x0065, dn)
            p.string(0x0064, "EX")
            p.string(0x5D01, from_email)
        else:
            p.string(0x0C1F, from_email)
            p.string(0x0C1E, "SMTP")
            p.string(0x0065, from_email)
            p.string(0x0064, "SMTP")
            p.string(0x5D01, from_email)  # Outlook writes it for SMTP senders too
    to = [str(x) for x in e.get("to") or []]
    cc = [str(x) for x in e.get("cc") or []]
    p.string(0x0E04, "; ".join(_split_address(x)[0] or _split_address(x)[1] for x in to))
    p.string(0x0E03, "; ".join(_split_address(x)[0] or _split_address(x)[1] for x in cc))
    if e.get("message_id"):
        mid = str(e["message_id"]).strip("<>")
        p.string(0x1035, f"<{mid}>")
    ticks = _filetime(e.get("date"))
    p.systime(0x0039, ticks)
    p.systime(0x0E06, ticks)
    if e.get("transport_headers"):
        p.string(0x007D, str(e["transport_headers"]))
    p.long(0x0E07, 1)            # PR_MESSAGE_FLAGS: read
    p.long(0x3FFD, 1252)         # PR_MESSAGE_CODEPAGE, for the 8-bit strings
    if unicode:
        p.long(0x340D, 0x00040000)  # PR_STORE_SUPPORT_MASK: STORE_UNICODE_OK

    text = str(e.get("body") or "")
    markup = e.get("html")
    if body in ("text", "all"):
        p.string(0x1000, text)
    if body in ("html", "all"):
        page = str(markup) if markup else _text_to_html(text)
        if unicode:
            p.binary(0x1013, page.encode("utf-8"))
            p.long(0x3FDE, 65001)   # PR_INTERNET_CPID
        else:
            p.binary(0x1013, page.encode("cp1252", "replace"))
            p.long(0x3FDE, 1252)
    if body in ("rtf", "all"):
        rtf = html_to_rtf(str(markup)) if markup else text_to_rtf(text)
        p.binary(0x1009, compress_rtf(rtf))
        p.boolean(0x0E1F, True)  # PR_RTF_IN_SYNC

    tree: Tree = {}
    recips = [(1, x) for x in to] + [(2, x) for x in cc]
    for i, (kind, value) in enumerate(recips):
        name, addr = _split_address(value)
        r = _Props(unicode)
        r.long(0x0C15, kind)
        r.string(0x3001, name or addr)
        r.string(0x5FF6, name or addr)
        r.string(0x3002, "SMTP")
        r.string(0x3003, addr)
        r.string(0x39FE, addr)
        r.long(0x3000, i)        # PR_ROWID
        r.long(0x0FFE, 6)        # PR_OBJECT_TYPE: MAPI_MAILUSER
        r.long(0x3900, 0)        # PR_DISPLAY_TYPE: DT_MAILUSER
        node: Tree = dict(r.streams)
        node["__properties_version1.0"] = r.table(bytes(8))
        tree[f"__recip_version1.0_#{i:08X}"] = node

    atts: List[Tuple[str, Any]] = [("file", a) for a in e.get("attachments") or []]
    atts += [("msg", m) for m in e.get("embedded") or []]
    for i, (kind, att) in enumerate(atts):
        a = _Props(unicode)
        a.long(0x0E21, i)        # PR_ATTACH_NUM
        a.long(0x0FFE, 7)        # PR_OBJECT_TYPE: MAPI_ATTACH
        a.long(0x370B, 0xFFFFFFFF)  # PR_RENDERING_POSITION: not placed in an RTF body
        if kind == "msg":
            inner_subject = str(att.get("subject") or "")
            a.long(0x3705, 5)    # ATTACH_EMBEDDED_MSG
            a.string(0x3001, inner_subject)
            a.obj(0x3701, message_tree(att, unicode, body, exchange_sender, True))
        else:
            name = str(att.get("name") or "attachment")
            data = bytes(att.get("data") or b"")
            a.long(0x3705, 1)    # ATTACH_BY_VALUE
            a.string(0x3707, name)
            a.string(0x3704, _short_name(name))
            a.string(0x3001, name)
            if "." in name:
                a.string(0x3703, "." + name.rsplit(".", 1)[1])
            if att.get("content_type"):
                a.string(0x370E, str(att["content_type"]))
            if att.get("content_id"):
                a.string(0x3712, str(att["content_id"]).strip("<>"))
            if att.get("inline"):
                a.boolean(0x7FFE, True)   # PR_ATTACHMENT_HIDDEN
                a.long(0x3714, 0x4)       # PR_ATTACH_FLAGS: ATT_MHTML_REF
            a.long(0x0E20, len(data))
            a.binary(0x3701, data)
        node = dict(a.streams)
        node["__properties_version1.0"] = a.table(bytes(8))
        tree[f"__attach_version1.0_#{i:08X}"] = node

    tree.update(p.streams)
    counts = struct.pack("<IIII", len(recips), len(atts), len(recips), len(atts))
    header = bytes(8) + counts + (b"" if embedded else bytes(8))
    tree["__properties_version1.0"] = p.table(header)
    if not embedded:
        tree["__nameid_version1.0"] = {"__substg1.0_00020102": b"", "__substg1.0_00030102": b"",
                                       "__substg1.0_00040102": b""}
    return tree


def build_msg(email: Dict[str, Any], *, unicode: bool = True, body: str = "text",
              exchange_sender: bool = False) -> bytes:
    """email: {"subject", "from_name", "from_email", "to": [...], "cc": [...], "date" (ISO),
    "message_id", "body" (plain text), "html" (optional), "attachments": [{"name", "content_type",
    "data": bytes, "inline": bool, "content_id"}], "embedded": [email dicts, written as
    ATTACH_EMBEDDED_MSG attachments]}.
    body: "text" (PR_BODY), "html" (PR_HTML only, no PR_BODY), "rtf" (PR_RTF_COMPRESSED only, real
    LZFu compression with back-references and the CRC), or "all".
    unicode=False writes 001E strings in cp1252. exchange_sender=True writes an EX legacy DN in
    PR_SENDER_EMAIL_ADDRESS and the SMTP address only in PR_SENDER_SMTP_ADDRESS.
    Optional "transport_headers" (a string) is written as PR_TRANSPORT_MESSAGE_HEADERS."""
    return write_cfb(message_tree(email, unicode, body, exchange_sender, False))
