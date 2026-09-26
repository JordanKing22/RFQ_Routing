"""
Read uploaded emails into plain dicts the importer can add to the inbox: .eml files, Outlook .msg
files, and .zip files of them. Standard library only.

    result = mailfile.load("RFQ 123.msg", data)
    -> {"emails": [ParsedEmail, ...], "skipped": [{"source": ..., "reason": ...}, ...]}

Every byte here comes from outside, so nothing trusts a size, count, offset, or charset it reads:
each one is checked against the file and against Limits, and a problem becomes a warning on the
email or a skipped entry, never an exception.

.eml    email.parser with policy.default. Emails attached as message/rfc822 (how Outlook on the web
        and new Outlook export several at once: select them, Forward, save) are listed after the
        email that carried them.
.msg    Outlook's own format: a Compound File (MS-CFB) of MAPI properties (MS-OXMSG), read here with
        a small read-only CFB reader. The body is PR_BODY, else PR_HTML, else PR_RTF_COMPRESSED
        (LZFu, MS-OXRTFCP), with the HTML taken back out of the RTF when Outlook wrapped it
        (MS-OXRTFEX).
.zip    zipfile, with the uncompressed bytes counted while reading, because a zip's own header can
        say anything.
"""

from __future__ import annotations

import array
import base64
import binascii
import codecs
import datetime as _dt
import email.feedparser
import email.parser
import email.policy
import email.utils
import html.parser
import io
import mimetypes
import posixpath
import quopri
import re
import struct
import sys
import urllib.parse
import zipfile
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Limits:
    max_file_bytes: int = 40 * 1024 * 1024        # one uploaded .eml/.msg/.zip
    max_zip_entries: int = 1000
    max_zip_total_bytes: int = 200 * 1024 * 1024  # uncompressed, enforced while reading (a lying header must not help)
    max_depth: int = 3                            # emails inside emails inside zips
    max_emails: int = 500
    max_attachments: int = 25                     # per email; the rest are reported
    max_attachment_bytes: int = 10 * 1024 * 1024  # bigger ones keep their name but lose their data (data=None)
    max_body_chars: int = 200_000


LIMITS = Limits()

# An inline image under this size is a signature logo or a social icon, not a pasted screenshot.
SIGNATURE_IMAGE_BYTES = 30 * 1024
MAX_NAME_CHARS = 120

CFB_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
EMAIL_EXTENSIONS = (".eml", ".msg")

# Work caps for hostile input that the Limits do not cover by themselves.
_MAX_MIME_PARTS = 5000             # parts looked at in one .eml
_MAX_HTML_CHARS = 5_000_000        # HTML read into text (Outlook HTML is often 10x its text)
_MAX_RTF_BYTES = 10 * 1024 * 1024  # decompressed RTF (pictures inside RTF make it big)
_MAX_MSG_OBJECTS = 5000            # recipient or attachment storages looked at in one .msg

_UTC = _dt.timezone.utc


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
_SURROGATES = re.compile("[\ud800-\udfff]+")
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")
# Direction overrides can disguise a name ("invoice", U+202E, "fdp.exe" shows as "invoiceexe.pdf"),
# and zero-width marks hide in names.
_NAME_JUNK = re.compile("[\x00-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_SPACES = re.compile(r"[ \t\u00a0\u2000-\u200a\u3000]+")


def _c1_table() -> Dict[int, Optional[str]]:
    table: Dict[int, Optional[str]] = {}
    for code in range(0x80, 0xA0):
        try:
            table[code] = bytes([code]).decode("cp1252")
        except UnicodeDecodeError:
            table[code] = None
    return table


# C1 control characters in text are Windows-1252 bytes that someone decoded as Latin-1 (Outlook
# 2003 stores the trademark sign as U+0099), so give them their Windows-1252 meaning.
_C1 = _c1_table()


def _latin(raw: bytes) -> str:
    """Bytes of an unknown or broken charset: Windows-1252 (a superset of printable Latin-1 that
    also has the curly quotes Outlook writes), else Latin-1, which decodes anything."""
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _unsurrogate(text: str) -> str:
    """The email package keeps bytes it could not decode as lone surrogates. JSON and the browser
    cannot carry those, so turn each run back into text: UTF-8 when it is UTF-8, else Latin-1."""
    if not _SURROGATES.search(text):
        return text

    def fix(m: "re.Match[str]") -> str:
        run = m.group()
        if all("\udc80" <= c <= "\udcff" for c in run):
            raw = bytes(ord(c) - 0xDC00 for c in run)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return _latin(raw)
        return "\ufffd" * len(run)

    return _SURROGATES.sub(fix, text)


def _one_line(text: Any) -> str:
    """A header-like value on one line: no control characters, single spaces, stripped."""
    text = _unsurrogate(str(text if text is not None else "")).translate(_C1)
    text = _CONTROL.sub(" ", text.replace("\t", " ").replace("\r", " ").replace("\n", " "))
    return _SPACES.sub(" ", text).strip()


def _clean_body(text: str) -> str:
    """Plain body text: \\n line ends, no control characters, no trailing spaces, at most one blank
    line in a row."""
    text = _unsurrogate(text).translate(_C1).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text.replace("\x0b", "\n").replace("\x0c", "\n"))
    lines = [line.rstrip() for line in text.split("\n")]
    out: List[str] = []
    for line in lines:
        if not line and out and not out[-1]:
            continue
        out.append(line)
    return "\n".join(out).strip("\n")


def _cap_body(text: str, limits: Limits, warnings: List[str]) -> str:
    if len(text) > limits.max_body_chars:
        warnings.append(f"body cut to {limits.max_body_chars:,} characters")
        return text[:limits.max_body_chars]
    return text


def _safe_name(name: Any, fallback: str = "attachment") -> str:
    """An attachment or entry name safe to show and to store: no folders, no control or direction
    characters, at most 120 characters with the extension kept, never empty."""
    text = _unsurrogate(str(name if name is not None else ""))
    text = text.replace("\\", "/").split("/")[-1]
    text = _NAME_JUNK.sub("", text)
    text = re.sub(r"^[A-Za-z]:", "", text)          # a drive letter left after the split
    text = _SPACES.sub(" ", text).strip().rstrip(".").strip()
    if text in ("", ".", ".."):
        text = fallback
    if len(text) > MAX_NAME_CHARS:
        stem, ext = posixpath.splitext(text)
        if 0 < len(ext) <= 16:
            text = stem[:MAX_NAME_CHARS - len(ext)].rstrip() + ext
        else:
            text = text[:MAX_NAME_CHARS]
    return text


def _ext(name: str) -> str:
    return posixpath.splitext(name)[1].lower()


_TEXT_CODECS_BLOCKED = {"unicode_escape", "raw_unicode_escape", "idna", "punycode", "undefined",
                        "mbcs", "oem", "unicode_internal"}
# Charset names mail programs use that Python spells differently.
_CHARSET_ALIASES = {
    "iso-8859-8-i": "iso-8859-8", "iso-8859-6-i": "iso-8859-6", "windows-874": "cp874",
    "x-sjis": "shift_jis", "x-euc-jp": "euc_jp", "x-gbk": "gbk", "ks_c_5601-1987": "cp949",
    "unicode-1-1-utf-7": "utf-7", "x-mac-roman": "mac_roman", "macintosh": "mac_roman",
    "ansi_x3.4-1968": "ascii", "utf8": "utf-8", "cp-850": "cp850", "unicode": "utf-16",
}


def _codec(charset: Optional[str]) -> Optional[str]:
    """Python's name for a mail charset, or None when Python has no such text codec."""
    if not charset:
        return None
    name = str(charset).strip().strip("\"'").lower()
    name = _CHARSET_ALIASES.get(name, name)
    if not name or len(name) > 40:
        return None
    try:
        info = codecs.lookup(name)
    except (LookupError, TypeError, ValueError):
        return None
    if info.name in _TEXT_CODECS_BLOCKED or not getattr(info, "_is_text_encoding", True):
        return None
    return info.name


def _decode_bytes(raw: bytes, charset: Optional[str], warnings: Optional[List[str]] = None,
                  what: str = "text") -> str:
    """Bytes in a declared charset. No charset or US-ASCII (often wrong): UTF-8 if it is UTF-8,
    else Windows-1252. An unknown charset: Latin-1, with a warning."""
    if not raw:
        return ""
    name = (charset or "").strip().strip("\"'").lower()
    if name in ("", "us-ascii", "ascii", "unknown-8bit", "x-unknown", "default", "binary", "utf-8", "utf8"):
        # Byte runs that are not UTF-8 fall back to Windows-1252 one run at a time, so one stray
        # byte does not garble the rest.
        return _unsurrogate(raw.decode("utf-8", "surrogateescape"))
    if name in ("iso-8859-1", "latin-1", "latin1", "iso_8859-1", "l1", "windows-1252", "cp1252"):
        return _latin(raw)
    codec = _codec(name)
    if codec is None:
        if warnings is not None:
            shown = _one_line(name)[:40]
            warnings.append(f"the {what} used an unknown character set ({shown}); read as Latin-1")
        return raw.decode("latin-1")
    try:
        return raw.decode(codec, errors="replace")
    except Exception:  # noqa: BLE001 - a codec can fail in ways other than UnicodeError
        return raw.decode("latin-1")


_ENCODED_WORD = re.compile(r"=\?([^?\s]{1,75})\?([QqBb])\?([^?\s]*)\?=")
_BETWEEN_WORDS = re.compile(r"(\?=)[ \t\r\n]+(=\?)")


def _decode_words(raw: Optional[str], warnings: Optional[List[str]] = None, what: str = "header") -> str:
    """RFC 2047 encoded words to text, by hand, so an unknown charset gives Latin-1 and a warning
    instead of replacement characters. Raw 8-bit text (common, not allowed) is kept."""
    if raw is None:
        return ""
    text = _BETWEEN_WORDS.sub(r"\1\2", str(raw))  # space between two encoded words is not text

    def word(m: "re.Match[str]") -> str:
        charset = m.group(1).split("*", 1)[0]      # RFC 2231 language suffix
        enc, payload = m.group(2).upper(), m.group(3)
        try:
            if enc == "B":
                data = base64.b64decode(payload + "=" * (-len(payload) % 4), validate=False)
            else:
                data = quopri.decodestring(payload.replace("_", " ").encode("ascii", "surrogateescape"),
                                           header=True)
        except (binascii.Error, ValueError, UnicodeError):
            return m.group()
        return _decode_bytes(data, charset, warnings, what)

    try:
        text = _ENCODED_WORD.sub(word, text)
    except Exception:  # noqa: BLE001
        pass
    return _unsurrogate(text)


# --------------------------------------------------------------------------- #
# HTML to text
# --------------------------------------------------------------------------- #
_BLOCK_TAGS = {"div", "tbody", "thead", "tfoot", "dt", "dd", "address", "center", "form", "fieldset",
               "section", "article", "header", "footer", "nav", "aside", "main", "figure", "hr",
               "caption", "body", "html", "tr"}
_PARAGRAPH_TAGS = {"table", "ul", "ol", "dl", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre"}
_SKIP_TAGS = {"script", "style", "head", "title", "template", "noscript", "xml", "object", "svg"}


class _HTMLText(html.parser.HTMLParser):
    """Readable text from email HTML: blocks and <br> become line breaks, paragraphs, tables, and
    lists are set apart by a blank line, table cells are joined with " | ", list items start with
    "- ", and scripts, styles, and the head are dropped. Line ends in the HTML source are only
    spaces, as in a browser, except inside <pre>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self.skip = 0
        self.pre = 0
        self.cells = 0
        self.trail = 0          # line ends at the end of the output so far
        self.p_lines = 2        # how the open <p> ends: 1 for Outlook's lines, 2 for paragraphs
        self.started = False    # any visible text yet

    def _break(self, lines: int) -> None:
        """End the current line, and leave `lines - 1` blank lines, without piling them up."""
        if self.started and self.trail < lines:
            self.out.append("\n" * (lines - self.trail))
            self.trail = lines

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in _SKIP_TAGS:
            self.skip += 1
            return
        if tag == "body":
            self.skip = 0  # a head that never closed must not hide the whole email
        if tag == "br":
            self.out.append("\n")
            self.trail += 1
        elif tag == "tr":
            self._break(1)
            self.cells = 0
        elif tag in ("td", "th"):
            if self.cells:
                self.out.append(" | ")
            self.cells += 1
        elif tag == "li":
            self._break(1)
            self.out.append("- ")
        elif tag == "p":
            # Outlook writes every line as <p class=MsoNormal> with no margin, and its blank lines
            # as empty paragraphs; other mailers mean a paragraph break.
            cls = " ".join(v or "" for k, v in attrs if k == "class").lower()
            self.p_lines = 1 if "mso" in cls else 2
            self._break(self.p_lines)
        elif tag in _PARAGRAPH_TAGS:
            self._break(2)
        elif tag in _BLOCK_TAGS:
            self._break(1)
        if tag == "pre":
            self.pre += 1

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag not in _SKIP_TAGS:
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
        elif tag == "p":
            self._break(self.p_lines)
        elif tag in _PARAGRAPH_TAGS:
            self._break(2)
        elif tag in _BLOCK_TAGS:
            self._break(1)
        if tag == "pre":
            self.pre = max(0, self.pre - 1)

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        if not self.pre:
            data = re.sub(r"[ \t\r\n\f]+", " ", data)
        if data.strip(" \t\r\n\f"):  # a no-break space counts: Outlook's blank lines are made of them
            self.started = True
            self.trail = len(data) - len(data.rstrip("\n")) if self.pre else 0
        self.out.append(data)


def html_to_text(markup: str) -> str:
    """Readable plain text from an HTML body. Never raises."""
    markup = markup[:_MAX_HTML_CHARS]
    parser = _HTMLText()
    try:
        parser.feed(markup)
        parser.close()
        text = "".join(parser.out)
    except Exception:  # noqa: BLE001 - fall back to dropping the tags
        text = re.sub(r"<[^>]*>", " ", markup)
        text = html.unescape(text)
    lines = [_SPACES.sub(" ", line.replace("\r", " ")).strip() for line in text.split("\n")]
    return _clean_body("\n".join(lines))


_CID_REF = re.compile(r"cid:([^\"'\s>)]+)", re.I)


def _cids(markup: str) -> set:
    """Content-IDs an HTML body shows as pictures (src="cid:...")."""
    out = set()
    for m in _CID_REF.finditer(markup[:_MAX_HTML_CHARS]):
        out.add(urllib.parse.unquote(m.group(1)).strip("<>").lower())
    return out


# --------------------------------------------------------------------------- #
# Headers, addresses, dates
# --------------------------------------------------------------------------- #
def _raw_header(msg: Any, name: str) -> Optional[str]:
    """A header as it was in the file, without the header registry (which can raise on hostile
    input and turns unknown charsets into replacement characters)."""
    try:
        want = name.lower()
        for key, value in msg.raw_items():
            if str(key).lower() == want:
                return str(value)
    except Exception:  # noqa: BLE001
        pass
    return None


def _strip_quotes(name: str) -> str:
    name = name.strip()
    while len(name) >= 2 and name[0] == name[-1] and name[0] in "'\"":
        name = name[1:-1].strip()
    return name


def _address_list(raw: Optional[str], warnings: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """[(display name, lowercased address)] from a raw To/From/Cc header. An entry without an
    address keeps its name with an empty address."""
    if not raw or not raw.strip():
        return []
    raw = raw.replace("\r", " ").replace("\n", " ")
    try:
        pairs = email.utils.getaddresses([raw])
        if any(p == ("", "") for p in pairs):
            try:
                pairs = email.utils.getaddresses([raw], strict=False)
            except TypeError:
                pass
    except Exception:  # noqa: BLE001
        pairs = []
    out: List[Tuple[str, str]] = []
    for name, addr in pairs:
        name = _strip_quotes(_one_line(_decode_words(name, warnings, "sender or recipient names")))
        addr = _one_line(_unsurrogate(addr)).strip("<>").strip()
        if "@" not in addr:
            name, addr = (name or addr), ""
        addr = addr.lower()
        if name.lower() == addr:
            name = ""
        if name or addr:
            out.append((name, addr))
    if not out:
        found = re.findall(r"[\w.+'-]+@[\w-]+(?:\.[\w-]+)+", raw)
        out = [("", a.lower()) for a in found]
    return out


def _format_address(name: str, addr: str) -> str:
    if name and addr:
        return f"{name} <{addr}>"
    return addr or name


def _iso_date(raw: Optional[str]) -> Optional[str]:
    """An RFC 5322 date as ISO 8601 with its offset. A date without a zone counts as UTC."""
    if not raw or not str(raw).strip():
        return None
    try:
        when = email.utils.parsedate_to_datetime(_one_line(raw))
    except Exception:  # noqa: BLE001 - ValueError, TypeError, IndexError, OverflowError
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=_UTC)
    try:
        return when.replace(microsecond=0).isoformat()
    except Exception:  # noqa: BLE001
        return None


def _message_id(raw: Optional[str]) -> Optional[str]:
    text = _one_line(raw or "")
    if not text:
        return None
    m = re.search(r"<([^<>\s]+)>", text)
    value = m.group(1) if m else text.strip("<> ").split(" ")[0]
    return value[:998] or None


# --------------------------------------------------------------------------- #
# One run of load(): the result, the limits, and the email count
# --------------------------------------------------------------------------- #
class _Run:
    def __init__(self, limits: Limits):
        self.limits = limits
        self.emails: List[Dict[str, Any]] = []
        self.skipped: List[Dict[str, str]] = []
        self.full = False

    def skip(self, source: str, reason: str) -> None:
        self.skipped.append({"source": source, "reason": reason})

    def room(self, source: str) -> bool:
        """True while another email fits under max_emails. The first one that does not is
        reported once, for itself and everything after it."""
        if len(self.emails) < self.limits.max_emails:
            return True
        if not self.full:
            self.full = True
            self.skip(source, f"more than {self.limits.max_emails} emails; this one and the rest were not read")
        return False


def _blank_email(source: str, fmt: str) -> Dict[str, Any]:
    return {"source": source, "format": fmt, "message_id": None, "subject": "", "from_name": "",
            "from_email": "", "to": [], "cc": [], "date": None, "body": "", "body_format": "none",
            "attachments": [], "container_only": False, "warnings": []}


class _Candidate:
    """An attachment before the limits and the signature filter decide what happens to it. The
    data is read only when it is kept (or to size an inline picture), so a message with thousands
    of attachments costs little."""
    __slots__ = ("name", "content_type", "inline", "content_id", "loader", "size_hint")

    def __init__(self, name: str, content_type: str, inline: bool, content_id: str,
                 loader: Callable[[], Optional[bytes]], size_hint: Optional[int]):
        self.name = name
        self.content_type = content_type
        self.inline = inline
        self.content_id = content_id
        self.loader = loader
        self.size_hint = size_hint


def _no_data() -> Optional[bytes]:
    return None


def _keep_attachments(run: _Run, candidates: List[_Candidate], cids: set,
                      warnings: List[str]) -> List[Dict[str, Any]]:
    limits = run.limits
    kept: List[Dict[str, Any]] = []
    signatures = 0
    over: List[str] = []
    big: List[str] = []
    for c in candidates:
        referenced = bool(c.content_id) and c.content_id.strip("<>").lower() in cids
        inline = c.inline or referenced
        data: Any = ...
        if c.content_type.startswith("image/") and inline:
            if c.size_hint is None or c.size_hint < 4 * SIGNATURE_IMAGE_BYTES:
                data = _load(c, warnings)
                size = len(data) if data is not None else (c.size_hint or 0)
            else:
                size = c.size_hint
            if 0 < size < SIGNATURE_IMAGE_BYTES:
                signatures += 1
                continue
        if len(kept) >= limits.max_attachments:
            over.append(c.name)
            continue
        if data is ... and c.size_hint is not None and c.size_hint > limits.max_attachment_bytes:
            data = None  # known to be too big: do not decode it just to drop it
            big.append(c.name)
        elif data is ...:
            data = _load(c, warnings)
        size = len(data) if data is not None else (c.size_hint or 0)
        if data is not None and len(data) > limits.max_attachment_bytes:
            big.append(c.name)
            data = None
        kept.append({"name": c.name, "content_type": c.content_type, "data": data, "size": size,
                     "inline": inline})
    if signatures:
        warnings.append(f"{signatures} signature image{'s' if signatures != 1 else ''} left out")
    if big:
        mb = limits.max_attachment_bytes / (1024 * 1024)
        warnings.append(f"{_name_list(big)}: larger than {mb:g} MB, only the name is kept")
    if over:
        warnings.append(f"{len(over)} more attachment{'s' if len(over) != 1 else ''} left out "
                        f"(the limit is {limits.max_attachments}): {_name_list(over)}")
    return kept


def _load(c: _Candidate, warnings: List[str]) -> Optional[bytes]:
    try:
        data = c.loader()
    except Exception:  # noqa: BLE001
        warnings.append(f"{c.name} could not be decoded")
        return None
    return bytes(data) if data is not None else None


def _name_list(names: List[str], most: int = 5) -> str:
    shown = ", ".join(names[:most])
    return shown + (f", and {len(names) - most} more" if len(names) > most else "")


_MIME = mimetypes.MimeTypes()  # Python's built-in table only, the same on every system


def _guess_type(name: str) -> str:
    guess = _MIME.guess_type(name, strict=False)[0] if name else None
    return guess or "application/octet-stream"


def _guess_ext(content_type: str) -> str:
    fixed = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "application/pdf": ".pdf",
             "text/plain": ".txt", "text/html": ".html", "message/rfc822": ".eml"}
    return fixed.get(content_type) or _MIME.guess_extension(content_type, strict=False) or ""


def _is_tnef(name: str, content_type: str) -> bool:
    return content_type in ("application/ms-tnef", "application/vnd.ms-tnef") or name.lower() == "winmail.dat"


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
_HEADER_LINE = re.compile(rb"^([!-9;-~]{1,76})[ \t]*:")
_KNOWN_HEADERS = {b"from", b"to", b"cc", b"subject", b"date", b"message-id", b"received", b"return-path",
                  b"mime-version", b"content-type", b"delivered-to", b"reply-to", b"sender",
                  b"x-scrubbed-attachments", b"thread-topic", b"in-reply-to", b"references"}


def sniff(data: bytes) -> Optional[str]:
    """'msg' (CFB signature D0CF11E0A1B11AE1), 'zip' (PK\\x03\\x04 or an empty zip), 'eml' (looks like
    RFC 5322 headers), or None."""
    if not isinstance(data, (bytes, bytearray, memoryview)) or len(data) == 0:
        return None
    head = bytes(data[:8])
    if head == CFB_SIGNATURE:
        return "msg"
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return "zip"
    text = bytes(data[:65536])
    if text.startswith(b"\xef\xbb\xbf"):
        text = text[3:]
    lines = text.lstrip(b" \t\r\n").split(b"\n")
    if lines and lines[0].startswith(b"From "):  # an mbox separator line
        lines = lines[1:]
    if not lines or not _HEADER_LINE.match(lines[0]):
        return None
    for line in lines[:500]:
        line = line.rstrip(b"\r")
        if not line:
            break
        if line[:1] in (b" ", b"\t"):
            continue
        m = _HEADER_LINE.match(line)
        if not m:
            break
        if m.group(1).lower() in _KNOWN_HEADERS:
            return "eml"
    return None


def load(filename: str, data: bytes, limits: Limits = LIMITS) -> Dict[str, Any]:
    """{"emails": [ParsedEmail, ...], "skipped": [{"source": str, "reason": str}, ...]}"""
    run = _Run(limits if isinstance(limits, Limits) else LIMITS)
    source = _safe_name(filename, "upload")
    try:
        if isinstance(data, str):
            data = data.encode("utf-8", "surrogateescape")
        data = bytes(data) if data is not None else b""
        _load_bytes(run, data, source, 0, top=True)
        if not run.emails and not run.skipped:
            run.skip(source, "no emails found")
    except Exception as exc:  # noqa: BLE001 - the last guard: load() never raises
        run.skip(source, f"could not be read ({type(exc).__name__})")
    for em in run.emails:
        em["warnings"] = _dedupe(em["warnings"])
    return {"emails": run.emails, "skipped": run.skipped}


def _load_bytes(run: _Run, data: bytes, source: str, depth: int, top: bool = False) -> None:
    """One file's bytes, whatever they turn out to be."""
    limits = run.limits
    if depth > limits.max_depth:
        run.skip(source, f"nested too deeply (more than {limits.max_depth} levels)")
        return
    if len(data) > limits.max_file_bytes:
        mb = limits.max_file_bytes / (1024 * 1024)
        run.skip(source, f"larger than {mb:g} MB")
        return
    kind = sniff(data)
    if kind == "zip":
        if top:
            _load_zip(run, data, source, depth)
        else:
            run.skip(source, "a zip inside a zip or an email is not opened")
        return
    if kind == "msg":
        _load_msg(run, data, source, depth)
        return
    if kind == "eml":
        _load_eml(run, data, source, depth)
        return
    ext = _ext(source.split(" > ")[-1])
    if not data:
        run.skip(source, "empty file")
    elif ext == ".msg":
        run.skip(source, "not an Outlook .msg file (no Compound File signature)")
    elif ext == ".zip":
        run.skip(source, "not a zip file")
    elif ext == ".eml":
        run.skip(source, "does not look like an email (no mail headers at the top)")
    else:
        run.skip(source, "not an email file (.eml, .msg, or .zip)")


# --------------------------------------------------------------------------- #
# .eml
# --------------------------------------------------------------------------- #
def _load_eml(run: _Run, data: bytes, source: str, depth: int) -> None:
    msg = _parse_mime(data)  # None on RecursionError from absurd nesting, and on parser bugs
    if msg is None:
        run.skip(source, "not a readable email")
        return
    _eml_email(run, msg, source, depth)


def _part_type(part: Any) -> str:
    try:
        return str(part.get_content_type()).lower()
    except Exception:  # noqa: BLE001
        return "application/octet-stream"


def _part_disposition(part: Any) -> Optional[str]:
    try:
        value = part.get_content_disposition()
        return str(value).lower() if value else None
    except Exception:  # noqa: BLE001
        return None


def _part_filename(part: Any) -> str:
    for getter in (lambda: part.get_filename(), lambda: part.get_param("name")):
        try:
            value = getter()
        except Exception:  # noqa: BLE001
            continue
        if isinstance(value, tuple):  # an RFC 2231 value the email package left encoded
            try:
                value = email.utils.collapse_rfc2231_value(value)
            except Exception:  # noqa: BLE001
                value = value[-1]
        if value:
            return _decode_words(str(value))
    return ""


def _part_bytes(part: Any) -> bytes:
    """A leaf part's decoded bytes (base64 and quoted-printable undone, damage tolerated)."""
    try:
        data = part.get_payload(decode=True)
    except Exception:  # noqa: BLE001
        data = None
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    try:
        payload = part.get_payload()
    except Exception:  # noqa: BLE001
        return b""
    if isinstance(payload, str):
        return payload.encode("utf-8", "surrogateescape")
    return b""


def _part_size_hint(part: Any) -> Optional[int]:
    """A leaf part's decoded size without decoding it: exact for base64 and unencoded parts,
    None for quoted-printable (which has to be decoded to know)."""
    try:
        payload = part.get_payload()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, str):
        return None
    cte = _one_line(_raw_header(part, "content-transfer-encoding")).lower()
    if cte == "base64":
        chars = len(payload) - sum(payload.count(c) for c in "\r\n\t ")
        pad = len(payload.rstrip("\r\n\t ")) - len(payload.rstrip("\r\n\t =").rstrip("\r\n\t "))
        return max(0, chars * 3 // 4 - min(pad, 2))
    if cte in ("", "7bit", "8bit", "binary"):
        return len(payload)
    return None


def _part_charset(part: Any) -> Optional[str]:
    try:
        value = part.get_param("charset")
    except Exception:  # noqa: BLE001
        return None
    if isinstance(value, tuple):
        value = value[-1]
    return str(value) if value else None


def _part_text(part: Any, warnings: List[str], what: str) -> str:
    return _decode_bytes(_part_bytes(part), _part_charset(part), warnings, what)


def _attached_message(part: Any) -> Tuple[Optional[Any], Optional[bytes]]:
    """A message/rfc822 part: the parsed message, or its bytes when it arrived base64 or
    quoted-printable encoded (not allowed, but some mailers do it) and must be parsed again."""
    try:
        payload = part.get_payload()
    except Exception:  # noqa: BLE001
        return None, None
    inner = payload[0] if isinstance(payload, list) and payload else None
    cte = _one_line(_raw_header(part, "content-transfer-encoding")).lower()
    if cte in ("base64", "quoted-printable") and inner is not None:
        try:
            text = inner.get_payload()
            if not isinstance(text, str):
                text = inner.as_string()
            raw = text.encode("ascii", "ignore")
            data = base64.b64decode(raw, validate=False) if cte == "base64" else quopri.decodestring(raw)
            return None, data
        except Exception:  # noqa: BLE001
            return inner, None
    if isinstance(payload, (str, bytes)):  # a message/rfc822 the parser did not open
        data = payload.encode("utf-8", "surrogateescape") if isinstance(payload, str) else payload
        return None, data
    return inner, None


_SIGNATURE_TYPES = ("application/pkcs7-signature", "application/x-pkcs7-signature", "application/pgp-signature")
_OPAQUE_TYPES = ("application/pkcs7-mime", "application/x-pkcs7-mime", "application/pgp-encrypted")

Nested = List[Tuple[str, Callable[[str, int], None]]]


def _open_nested(run: _Run, source: str, depth: int, nested: Nested) -> None:
    """The emails an email carried, each listed after it as "outer > inner"."""
    limits = run.limits
    for label, open_fn in nested:
        child_source = f"{source} > {label}"
        if depth + 1 > limits.max_depth:
            run.skip(child_source, f"nested too deeply (more than {limits.max_depth} levels)")
        else:
            open_fn(child_source, depth + 1)


def _mime_walk(run: _Run, msg: Any, warnings: List[str]
               ) -> Tuple[List[str], List[str], List[_Candidate], Nested]:
    """(plain bodies, HTML bodies, attachment candidates, attached emails) of a MIME message,
    walked without recursion. Attached emails are not opened here: they become their own
    ParsedEmails after the one that carried them."""
    plain: List[str] = []
    htmls: List[str] = []
    leaves: List[Tuple[Any, str]] = []   # (part, parent content type)
    nested: Nested = []
    stack: List[Tuple[Any, str, int]] = [(msg, "", 0)]
    seen = 0

    def add_message(name: str, inner: Any, data: Optional[bytes]) -> None:
        if inner is not None:
            label = name or _one_line(_decode_words(_raw_header(inner, "subject")))
        else:
            label = name
        label = _safe_name(label, f"attached email {len(nested) + 1}")
        if _ext(label) not in EMAIL_EXTENSIONS:
            label += ".eml"
        if inner is not None:
            nested.append((label, lambda src, d, m=inner: _eml_email(run, m, src, d)))
        elif data:
            nested.append((label, lambda src, d, b=data: _load_bytes(run, b, src, d)))
        else:
            nested.append((label, lambda src, d: run.skip(src, "an attached email that could not be read")))

    while stack:
        part, parent, level = stack.pop()
        seen += 1
        if seen > _MAX_MIME_PARTS:
            warnings.append(f"more than {_MAX_MIME_PARTS:,} MIME parts; the rest were not read")
            break
        ctype = _part_type(part)
        if part is not msg and ctype in ("message/rfc822", "message/global"):
            add_message(_part_filename(part), *_attached_message(part))
            continue
        try:
            multipart = part.is_multipart()
        except Exception:  # noqa: BLE001
            multipart = False
        if multipart:
            if level >= 60:
                warnings.append("MIME parts nested too deeply; the deepest were not read")
                continue
            try:
                children = list(part.iter_parts())
            except Exception:  # noqa: BLE001
                children = []
            for child in reversed(children):
                stack.append((child, ctype, level + 1))
            continue
        leaves.append((part, parent))

    candidates: List[_Candidate] = []
    for part, parent in leaves:
        ctype = _part_type(part)
        disposition = _part_disposition(part)
        filename = _part_filename(part)
        if ctype in ("text/plain", "text/html") and disposition != "attachment" and not filename:
            text = _part_text(part, warnings, "body")
            (plain if ctype == "text/plain" else htmls).append(text)
            continue
        if ctype in _SIGNATURE_TYPES:
            warnings.append("a signed email; the signature was not checked")
            continue
        name = _safe_name(filename, "attachment" + _guess_ext(ctype))
        if ctype in _OPAQUE_TYPES or _ext(name) == ".p7m":
            warnings.append(f"{name}: encrypted or signed (S/MIME) content is not read")
        if _is_tnef(name, ctype):
            warnings.append("winmail.dat (Outlook rich text) is not read")
        if _ext(name) in EMAIL_EXTENSIONS or ctype in ("application/vnd.ms-outlook", "application/x-msg"):
            data = _part_bytes(part)
            if sniff(data) in ("eml", "msg"):
                add_message(name, None, data)
                continue
        cid = _raw_header(part, "content-id") or ""
        inline = disposition == "inline" or (disposition is None and parent == "multipart/related")
        candidates.append(_Candidate(name, ctype, inline, _one_line(cid).strip("<> "),
                                     (lambda p=part: _part_bytes(p)), _part_size_hint(part)))
    return plain, htmls, candidates, nested


def _set_body(em: Dict[str, Any], plain: List[str], htmls: List[str], limits: Limits,
              warnings: List[str]) -> None:
    """The text/plain parts if any say something, else the HTML parts as text."""
    body_plain = "\n\n".join(t for t in plain if t.strip())
    if body_plain.strip():
        em["body"], em["body_format"] = _clean_body(body_plain), "text"
    elif any(h.strip() for h in htmls):
        text = "\n\n".join(html_to_text(h) for h in htmls if h.strip())
        em["body"], em["body_format"] = text, "html"
    em["body"] = _cap_body(em["body"], limits, warnings)


def _parse_mime(data: bytes) -> Optional[Any]:
    """The parsed message, or None (RecursionError on absurd nesting, and parser bugs). Fed a
    megabyte at a time: parsebytes() would first copy the whole file into one string, and a big
    .eml already costs several times its size in memory."""
    try:
        parser = email.feedparser.BytesFeedParser(policy=email.policy.default)
        view = memoryview(data)
        for start in range(0, len(data), 1 << 20):
            parser.feed(bytes(view[start:start + (1 << 20)]))
        return parser.close()
    except Exception:  # noqa: BLE001
        return None


def _eml_email(run: _Run, msg: Any, source: str, depth: int) -> None:
    """One parsed message: add its ParsedEmail, then the emails attached to it."""
    if not run.room(source):
        return
    limits = run.limits
    em = _blank_email(source, "eml")
    warnings: List[str] = em["warnings"]

    em["message_id"] = _message_id(_raw_header(msg, "message-id"))
    em["subject"] = _one_line(_decode_words(_raw_header(msg, "subject"), warnings, "subject"))
    sender = _address_list(_raw_header(msg, "from"), warnings)
    if sender:
        em["from_name"], em["from_email"] = sender[0]
    else:
        warnings.append("no sender address")
    em["to"] = [_format_address(n, a) for n, a in _address_list(_raw_header(msg, "to"), warnings)]
    em["cc"] = [_format_address(n, a) for n, a in _address_list(_raw_header(msg, "cc"), warnings)]
    raw_date = _raw_header(msg, "date")
    em["date"] = _iso_date(raw_date)
    if em["date"] is None:
        warnings.append("no date" if not (raw_date or "").strip() else "the date could not be read")

    plain, htmls, candidates, nested = _mime_walk(run, msg, warnings)

    scrubbed = _decode_words(_raw_header(msg, "x-scrubbed-attachments"), warnings, "attachment list")
    for item in scrubbed.split(";"):
        if item.strip():
            candidates.append(_Candidate(_safe_name(item), "", False, "", _no_data, 0))

    _set_body(em, plain, htmls, limits, warnings)

    cids = set()
    for h in htmls:
        cids |= _cids(h)
    em["attachments"] = _keep_attachments(run, candidates, cids, warnings)
    em["container_only"] = bool(nested) and not em["attachments"]
    run.emails.append(em)
    _open_nested(run, source, depth, nested)


# --------------------------------------------------------------------------- #
# .zip
# --------------------------------------------------------------------------- #
_ZIP_METHODS = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
_ZIP_JUNK = (".DS_Store", "Thumbs.db", "desktop.ini")


def _zip_path(name: str) -> str:
    """An entry name as a clean relative path: no drive, no leading slash, no "..", no control
    characters. Nothing is written to disk, but the name ends up in `source` and on screen."""
    parts = []
    for part in _unsurrogate(str(name)).replace("\\", "/").split("/"):
        part = _SPACES.sub(" ", _NAME_JUNK.sub("", part)).strip()
        if part in ("", ".", "..") or re.fullmatch(r"[A-Za-z]:", part):
            continue
        parts.append(part[:MAX_NAME_CHARS])
    return "/".join(parts)


def _read_zip_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo, file_budget: int,
                    total_budget: int, peek: bool = False) -> Tuple[Optional[bytes], int, str]:
    """(bytes, bytes read, problem). Counts what actually comes out of the decompressor, a chunk
    at a time, and stops at a budget whatever the header claimed. peek: just the first
    file_budget bytes."""
    chunks: List[bytes] = []
    got = 0
    try:
        with zf.open(info) as fh:
            while True:
                chunk = fh.read(min(1 << 16, file_budget + 1 - got) if peek else 1 << 16)
                if not chunk:
                    break
                got += len(chunk)
                if got > total_budget:
                    return None, got, "total"
                if peek and got >= file_budget:
                    chunks.append(chunk)
                    break
                if got > file_budget:
                    return None, got, "file"
                chunks.append(chunk)
    except RuntimeError:
        return None, got, "password protected"
    except NotImplementedError:
        return None, got, "compressed with a method this reader does not open"
    except Exception as exc:  # noqa: BLE001 - BadZipFile (CRC), zlib.error, EOFError, OSError
        return None, got, f"could not be unzipped ({type(exc).__name__})"
    return b"".join(chunks), got, ""


def _load_zip(run: _Run, data: bytes, source: str, depth: int) -> None:
    limits = run.limits
    # zipfile builds an object per central directory record before it can be asked anything.
    # Count the record signatures first (an over-count only makes this stricter) so a zip of a
    # million empty entries is turned away before it costs memory.
    if data.count(b"PK\x01\x02") > 20 * max(limits.max_zip_entries, 50):
        run.skip(source, f"the zip has far more than {limits.max_zip_entries} entries")
        return
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        infos = sorted(zf.infolist(), key=lambda i: i.filename)
    except Exception as exc:  # noqa: BLE001
        run.skip(source, f"not a readable zip file ({type(exc).__name__})")
        return
    total = 0
    mb_total = limits.max_zip_total_bytes / (1024 * 1024)
    mb_file = limits.max_file_bytes / (1024 * 1024)
    with zf:
        for n, info in enumerate(infos):
            if n >= limits.max_zip_entries:
                run.skip(source, f"the zip has {len(infos)} entries; only the first "
                         f"{limits.max_zip_entries} were read")
                break
            path = _zip_path(info.filename)
            if not path or info.filename.endswith(("/", "\\")):
                continue  # a folder
            parts = path.split("/")
            if "__MACOSX" in parts or parts[-1].startswith("._") or parts[-1] in _ZIP_JUNK:
                continue  # Finder and Explorer leftovers, not the user's files
            entry = f"{source}/{path}"
            ext = _ext(parts[-1])
            if info.flag_bits & 0x1:
                run.skip(entry, "password protected")
                continue
            if ext == ".zip":
                run.skip(entry, "a zip inside a zip is not opened")
                continue
            if info.compress_type not in _ZIP_METHODS:
                # bzip2 and LZMA decompress a whole chunk at once, with no cap on what comes out.
                run.skip(entry, "compressed with a method this reader does not open")
                continue
            problem, content = "", None
            if ext not in EMAIL_EXTENSIONS:
                # Maildir and some exports name emails without .eml: look at the first bytes
                # rather than unzipping every drawing and spreadsheet in the archive.
                head, got, problem = _read_zip_entry(zf, info, 4096, limits.max_zip_total_bytes - total,
                                                     peek=True)
                total += got
                if not problem and sniff(head or b"") not in ("eml", "msg"):
                    problem = "not an email file"
            if not problem:
                if not run.room(entry):
                    break
                content, got, problem = _read_zip_entry(zf, info, limits.max_file_bytes,
                                                        limits.max_zip_total_bytes - total)
                total += got
            if problem == "total":
                run.skip(entry, f"the zip holds more than {mb_total:g} MB uncompressed; this entry "
                         "and the rest were not read")
                break
            if problem == "file":
                run.skip(entry, f"larger than {mb_file:g} MB")
                continue
            if problem:
                run.skip(entry, problem)
                continue
            _load_bytes(run, content or b"", entry, depth + 1)


# --------------------------------------------------------------------------- #
# .msg
# --------------------------------------------------------------------------- #
class _CFBError(Exception):
    pass


_FREESECT = 0xFFFFFFFF
_ENDOFCHAIN = 0xFFFFFFFE
_MAXREGSECT = 0xFFFFFFFA
_NOSTREAM = 0xFFFFFFFF
_DIR_ENTRY = struct.Struct("<64sHBBIII16sIQQIQ")


def _u32_array(raw: bytes) -> "array.array[int]":
    out = array.array("I")
    if out.itemsize != 4:  # pragma: no cover - every platform Python runs on has a 4-byte "I"
        out = array.array("L")
    out.frombytes(raw[:len(raw) - len(raw) % 4])
    if sys.byteorder == "big":
        out.byteswap()
    return out


class _CFB:
    """A read-only Compound File (MS-CFB) reader for Outlook .msg files: 512 or 4096-byte sectors,
    the FAT and its DIFAT sectors, the mini FAT and mini stream, and the directory's red-black
    trees. Every sector number, chain, and size is checked against the file; a loop or an
    out-of-range number marks the file damaged instead of reading forever or out of bounds."""

    def __init__(self, data: bytes):
        if len(data) < 512 or data[:8] != CFB_SIGNATURE:
            raise _CFBError("no Compound File signature")
        _minor, major, order, shift, mini_shift = struct.unpack_from("<HHHHH", data, 24)
        if order != 0xFFFE or shift not in (9, 12) or mini_shift != 6:
            raise _CFBError("the header is not a valid Compound File header")
        self.data = data
        self.v4 = shift == 12
        self.ssize = 1 << shift
        self.nsect = (len(data) - 1) // self.ssize  # sectors after the header, a partial last one included
        self.damaged = False
        # Streams that share sectors could make a small file read as gigabytes: count what is read.
        self.budget = 2 * len(data) + (1 << 20)
        (_nfat, first_dir, _tx, _cutoff, first_minifat, _nminifat,
         first_difat, _ndifat) = struct.unpack_from("<IIIIIIII", data, 44)

        # The FAT's own sectors: 109 listed in the header, the rest in a chain of DIFAT sectors.
        # Only as many as the file has sectors to describe: more could only point past the end.
        per = self.ssize // 4
        need = (self.nsect + per - 1) // per
        fat_sectors = [s for s in struct.unpack_from("<109I", data, 76) if s <= _MAXREGSECT]
        seen = set()
        s = first_difat
        while s <= _MAXREGSECT and len(fat_sectors) < need:
            if s in seen or s >= self.nsect:
                self.damaged = True
                break
            seen.add(s)
            vals = struct.unpack(f"<{per}I", self._sector(s))
            fat_sectors.extend(v for v in vals[:-1] if v <= _MAXREGSECT)
            s = vals[-1]
        raw = bytearray()
        for fs in fat_sectors[:need]:
            if fs >= self.nsect:
                self.damaged = True
                raw += b"\xff" * self.ssize
            else:
                raw += self._sector(fs)
        self.fat = _u32_array(bytes(raw))[:self.nsect]
        if not len(self.fat):
            raise _CFBError("no sector allocation table")

        self.dir = self._chain_bytes(first_dir, self.fat, self.nsect, self.ssize, self._sector)
        self.nentries = len(self.dir) // 128
        if self.nentries == 0:
            raise _CFBError("no directory")
        root = self.entry(0)
        if root is None or root[1] != 5:
            raise _CFBError("no root entry")
        self.children_cache: Dict[int, Dict[str, int]] = {}
        self.visited = {0}
        # The mini stream (the root entry's stream) holds every stream under 4096 bytes, in
        # 64-byte mini sectors listed by the mini FAT.
        start, size = root[5], root[6]
        mini_sectors = (size + 63) // 64
        try:
            self.ministream = self._chain_bytes(start, self.fat, (size + self.ssize - 1) // self.ssize,
                                                self.ssize, self._sector)[:size]
        except _CFBError:
            self.damaged = True
            self.ministream = b""
        try:
            self.minifat = _u32_array(self._chain_bytes(first_minifat, self.fat, self.nsect, self.ssize,
                                                        self._sector))[:mini_sectors]
        except _CFBError:
            self.damaged = True
            self.minifat = _u32_array(b"")

    def _sector(self, s: int) -> bytes:
        start = (s + 1) * self.ssize
        chunk = self.data[start:start + self.ssize]
        if len(chunk) < self.ssize:  # a truncated last sector reads as zeros past the end
            chunk += b"\0" * (self.ssize - len(chunk))
        return chunk

    def _mini_sector(self, s: int) -> bytes:
        chunk = self.ministream[s * 64:s * 64 + 64]
        return chunk + b"\0" * (64 - len(chunk))

    def _chain_bytes(self, start: int, fat: Any, most: int, size: int,
                     read: Callable[[int], bytes]) -> bytes:
        """The bytes of a sector chain, at most `most` sectors. Raises _CFBError on a loop or a
        sector number past the table; a chain that ends early just gives fewer bytes."""
        out = bytearray()
        seen = set()
        s = start
        count = 0
        while s != _ENDOFCHAIN and count < most:
            if s >= len(fat):
                if count == 0 and s in (_FREESECT, _ENDOFCHAIN):
                    break
                raise _CFBError("a sector number is out of range")
            if s in seen:
                raise _CFBError("a sector chain loops")
            seen.add(s)
            out += read(s)
            count += 1
            s = fat[s]
        return bytes(out)

    def entry(self, i: int) -> Optional[Tuple[str, int, int, int, int, int, int]]:
        """Directory entry i as (name, type, left, right, child, start sector, size), or None
        when there is no such entry. Types: 1 storage, 2 stream, 5 root."""
        if not 0 <= i < self.nentries:
            return None
        (raw_name, name_len, kind, _color, left, right, child, _clsid, _state, _ct, _mt,
         start, size) = _DIR_ENTRY.unpack_from(self.dir, i * 128)
        name_len = min(name_len, 64)
        name = raw_name[:max(0, name_len - 2)].decode("utf-16-le", "replace")
        if not self.v4:
            size &= 0xFFFFFFFF  # version 3 files may leave junk in the high half
        return (name, kind, left, right, child, start, size)  # type: ignore[return-value]

    def children(self, i: int) -> Dict[str, int]:
        """A storage's children by upper-case name: its red-black tree walked without recursion.
        The visited set is shared by the whole file, so an entry reached twice (a loop, or two
        storages sharing a subtree) is read once and the file is marked damaged."""
        if i in self.children_cache:
            return self.children_cache[i]
        out: Dict[str, int] = {}
        e = self.entry(i)
        stack = [e[4]] if e is not None and e[1] in (1, 5) else []
        while stack:
            j = stack.pop()
            if j == _NOSTREAM:
                continue
            if j in self.visited or not 0 <= j < self.nentries:
                self.damaged = True
                continue
            self.visited.add(j)
            c = self.entry(j)
            if c is None:
                continue
            stack.append(c[2])
            stack.append(c[3])
            if c[1] in (1, 2):
                out.setdefault(c[0].upper(), j)
            else:
                self.damaged = True
        self.children_cache[i] = out
        return out

    def is_storage(self, i: Optional[int]) -> bool:
        e = self.entry(i) if i is not None else None
        return e is not None and e[1] == 1

    def size(self, i: Optional[int]) -> Optional[int]:
        """A stream's declared size, or None when there is no such stream or the size is more
        than the file could hold (the file is then marked damaged)."""
        e = self.entry(i) if i is not None else None
        if e is None or e[1] != 2:
            return None
        if e[6] > len(self.data):
            self.damaged = True
            return None
        return e[6]

    def read(self, i: Optional[int]) -> Optional[bytes]:
        """A stream's bytes, or None when it is missing or damaged (the file is marked damaged)."""
        e = self.entry(i) if i is not None else None
        if e is None or e[1] != 2:
            return None
        start, size = e[5], e[6]
        if size == 0:
            return b""
        try:
            if size < 4096:
                data = self._chain_bytes(start, self.minifat, (size + 63) // 64, 64, self._mini_sector)
            else:
                if size > len(self.data):
                    raise _CFBError("a stream is larger than the file")
                data = self._chain_bytes(start, self.fat, (size + self.ssize - 1) // self.ssize,
                                         self.ssize, self._sector)
        except _CFBError:
            self.damaged = True
            return None
        if len(data) < size:
            self.damaged = True
        data = data[:size]
        self.budget -= len(data)
        if self.budget < 0:
            self.damaged = True
            return None
        return data


# --------------------------------------------------------------------------- #
# Compressed RTF (MS-OXRTFCP) and RTF to text
# --------------------------------------------------------------------------- #
_RTF_PREBUF = (b"{\\rtf1\\ansi\\mac\\deff0\\deftab720{\\fonttbl;}{\\f0\\fnil \\froman \\fswiss \\fmodern "
               b"\\fscript \\fdecor MS Sans SerifSymbolArialTimes New RomanCourier{\\colortbl\\red0\\green0"
               b"\\blue0\r\n\\par \\pard\\plain\\f0\\fs20\\b\\i\\u\\tab\\tx")
_LZFU = 0x75465A4C
_MELA = 0x414C454D


def rtf_crc(data: bytes) -> int:
    """The CRC of MS-OXRTFCP: the CRC-32 table without the usual inversions at start and end."""
    return zlib.crc32(data, 0xFFFFFFFF) ^ 0xFFFFFFFF


def decompress_rtf(data: bytes, max_bytes: int = _MAX_RTF_BYTES) -> Tuple[Optional[bytes], List[str]]:
    """PR_RTF_COMPRESSED to RTF bytes. A CRC or size mismatch is a warning, not a failure: the
    text is usually still fine. Returns (None, warnings) when nothing can be read."""
    warnings: List[str] = []
    if len(data) < 16:
        return None, ["the RTF body is too short to read"]
    comp_size, raw_size, comp_type, crc = struct.unpack_from("<IIII", data, 0)
    if comp_type == _MELA:  # stored without compression
        return bytes(data[16:16 + min(raw_size, max_bytes)]), warnings
    if comp_type != _LZFU:
        return None, ["the RTF body has an unknown compression type"]
    end = 4 + comp_size if 12 <= comp_size <= len(data) - 4 else len(data)
    if 4 + comp_size > len(data):
        warnings.append("the RTF body is cut short")
    body = data[16:end]
    if rtf_crc(body) != crc:
        warnings.append("the RTF body's checksum does not match; it was read anyway")
    # The dictionary is a 4096-byte ring that starts with the prebuffer. Kept here as one growing
    # buffer behind 4096 bytes of zeros, a reference is a slice at a fixed distance back.
    out = bytearray(4096)
    out += _RTF_PREBUF
    cap = len(out) + max_bytes
    pos, n = 0, len(body)
    ended = False
    while pos < n and not ended:
        control = body[pos]
        pos += 1
        if control == 0 and pos + 8 <= n:  # eight literals: the common case in text
            out += body[pos:pos + 8]
            pos += 8
            if len(out) > cap:
                warnings.append("the RTF body is very large; only the start was read")
                break
            continue
        for bit in range(8):
            if pos >= n:
                break
            if control & (1 << bit):
                if pos + 1 >= n:
                    pos = n
                    break
                word = (body[pos] << 8) | body[pos + 1]
                pos += 2
                offset, length = word >> 4, (word & 0xF) + 2
                write = (len(out) - 4096) & 0xFFF
                if offset == write:
                    ended = True
                    break
                j = len(out) - ((write - offset) & 0xFFF)
                if j + length <= len(out):
                    out += out[j:j + length]
                else:  # the copy runs into bytes it is writing (a repeat)
                    for k in range(length):
                        out.append(out[j + k])
            else:
                out.append(body[pos])
                pos += 1
        if len(out) > cap:
            warnings.append("the RTF body is very large; only the start was read")
            break
    result = bytes(out[4096 + len(_RTF_PREBUF):])
    if not ended and len(result) < raw_size:
        warnings.append("the RTF body ends early")
    if len(result) > raw_size:
        result = result[:raw_size]
    return result[:max_bytes], warnings


_RTF_TOKEN = re.compile(
    rb"\\([a-zA-Z]{1,32})(-?\d{1,10})? ?"   # control word, optional number, one space eaten
    rb"|\\'([0-9a-fA-F]{2})"                # a byte in the document's code page
    rb"|\\([^a-zA-Z'])"                     # control symbol
    rb"|([{}])"
    rb"|[\r\n]+"
    rb"|([^\\{}\r\n]+)", re.S)

# Destinations whose text is not part of the message.
_RTF_SKIP = {
    "colortbl", "stylesheet", "info", "pict", "object", "objdata", "header", "headerl",
    "headerr", "headerf", "footer", "footerl", "footerr", "footerf", "fldinst", "listtable",
    "listoverridetable", "revtbl", "rsidtbl", "generator", "xmlnstbl", "themedata", "datastore",
    "colorschememapping", "latentstyles", "pgdsctbl", "filetbl", "author", "operator", "title",
    "subject", "company", "bkmkstart", "bkmkend", "nonshppict", "private", "mhtmltag", "pnseclvl",
    "defchp", "defpap", "wgrffmtfilter", "mmathpr", "fchars", "lchars", "userprops", "docvar",
    "xe", "tc", "atnid", "atnauthor", "annotation", "comment", "doccomm", "keywords", "category",
}
_RTF_SYMBOLS = {
    "par": "\n", "line": "\n", "sect": "\n", "page": "\n", "row": "\n", "tab": "\t", "cell": "\t",
    "emdash": "-", "endash": "-", "emspace": " ", "enspace": " ", "qmspace": " ", "bullet": "\u2022",
    "lquote": "'", "rquote": "'", "ldblquote": '"', "rdblquote": '"',
}
_WINDOWS_CODEPAGES = {65001: "utf-8", 20127: "ascii", 1200: "utf-16-le", 1201: "utf-16-be",
                      50220: "iso2022_jp", 50221: "iso2022_jp", 50222: "iso2022_jp", 51932: "euc_jp",
                      20866: "koi8_r", 21866: "koi8_u", 54936: "gb18030", 52936: "hz", 10000: "mac_roman"}


def _codepage_codec(cp: Optional[int]) -> str:
    """A Windows code page number to a Python codec, Windows-1252 when unknown."""
    if not cp:
        return "cp1252"
    name = _WINDOWS_CODEPAGES.get(cp)
    if name is None and 28591 <= cp <= 28606:
        name = f"iso-8859-{cp - 28590}"
    return _codec(name or f"cp{cp}") or "cp1252"


def rtf_to_text(rtf: bytes) -> Tuple[str, bool]:
    """(text, was_html). RTF that Outlook made from HTML (\\fromhtml1) gives the HTML back
    (MS-OXRTFEX) turned into text; other RTF gives its visible text."""
    rtf = bytes(rtf[:_MAX_RTF_BYTES])
    from_html = b"\\fromhtml" in rtf[:4096]
    out = _rtf_walk(rtf, from_html)
    if from_html:
        return html_to_text(out), True
    return _clean_body(out), False


# \fcharsetN in the font table: the code page of the \'hh bytes written in that font.
_FONT_CHARSETS = {77: "mac_roman", 128: "cp932", 129: "cp949", 130: "cp1361", 134: "cp936", 136: "cp950",
                  161: "cp1253", 162: "cp1254", 163: "cp1258", 177: "cp1255", 178: "cp1256", 186: "cp1257",
                  204: "cp1251", 222: "cp874", 238: "cp1250", 255: "cp437"}


def _rtf_walk(rtf: bytes, from_html: bool) -> str:
    out: List[str] = []
    pending = bytearray()          # \\'hh bytes, decoded together (double-byte code pages)
    doc_codec = "cp1252"           # \\ansicpg
    fonts: Dict[int, str] = {}     # font number -> codec, from the font table
    deff = 0
    defining = None                # the font being defined inside the font table
    # Group state: skip (a destination we drop), rtfonly (\\htmlrtf: RTF that has no HTML
    # counterpart), tag (inside \\*\\htmltag: HTML to keep), uc (fallback characters after \\u),
    # codec (the current font's code page), fonttbl (inside the font table).
    skip, rtfonly, tag, uc, codec, fonttbl = False, False, False, 1, "cp1252", False
    stack: List[Tuple[bool, bool, bool, int, str, bool]] = []
    star = False
    fallback = 0                   # characters still to drop after a \\uN
    high: Optional[int] = None     # a UTF-16 high surrogate waiting for its low half
    pos, n = 0, len(rtf)

    def visible() -> bool:
        return not skip and (not from_html or tag or not rtfonly)

    def flush() -> None:
        if pending:
            try:
                out.append(pending.decode(codec, errors="replace"))
            except Exception:  # noqa: BLE001
                out.append(pending.decode("latin-1"))
            pending.clear()

    def emit(text: str) -> None:
        flush()
        out.append(text)

    while pos < n:
        m = _RTF_TOKEN.match(rtf, pos)
        if m is None:
            pos += 1
            continue
        pos = m.end()
        word, hexbyte, symbol, brace, text = m.group(1), m.group(3), m.group(4), m.group(5), m.group(6)
        if hexbyte is not None:
            if fallback:
                fallback -= 1
            elif visible():
                pending.append(int(hexbyte, 16))
            continue
        if text is not None:
            if fallback:
                drop = min(fallback, len(text))
                fallback -= drop
                text = text[drop:]
            if text and visible():
                emit(text.decode(codec, errors="replace") if codec != "cp1252" else _latin(text))
            continue
        if brace is not None:
            fallback = 0
            star = False
            if brace == b"{":
                if len(stack) < 1000:
                    stack.append((skip, rtfonly, tag, uc, codec, fonttbl))
            elif stack:
                flush()
                skip, rtfonly, tag, uc, codec, fonttbl = stack.pop()
            continue
        if symbol is not None:
            fallback = 0
            if symbol == b"*":
                star = True
            elif visible():
                ch = symbol.decode("latin-1")
                if ch in "\\{}":
                    emit(ch)
                elif ch == "~":
                    emit(" ")
                elif ch == "_":
                    emit("-")
                elif ch in "\r\n":
                    emit("\n")
            continue
        if word is None:  # a run of line ends: not text in RTF
            continue
        name = word.decode("ascii").lower()
        arg = m.group(2)
        num = int(arg) if arg is not None else None
        if name == "bin":  # raw binary data follows; skip it
            pos += max(0, min(num or 0, n - pos))
            continue
        if fonttbl:
            # Only the code page of each font matters here; the names are not text.
            if name == "f" and num is not None:
                defining = num
            elif name == "fcharset" and defining is not None and num in _FONT_CHARSETS:
                fonts[defining] = _codec(_FONT_CHARSETS[num]) or doc_codec
            elif name == "cpg" and defining is not None and num:
                fonts[defining] = _codepage_codec(num)
            star = False
            continue
        if name == "fonttbl":
            skip, fonttbl, star = True, True, False
            continue
        if name == "htmltag" and from_html:
            skip, tag, star = False, True, False
            continue
        if star or name in _RTF_SKIP:
            skip, star = True, False
            continue
        if name == "u" and num is not None:
            fallback = uc
            if skip or not visible():
                continue
            code = num + 65536 if num < 0 else num
            if not 0 <= code <= 0xFFFF:  # \u is a signed 16-bit number
                code = 0xFFFD
            if 0xD800 <= code <= 0xDBFF:
                high = code
                continue
            if 0xDC00 <= code <= 0xDFFF and high is not None:
                code = 0x10000 + ((high - 0xD800) << 10) + (code - 0xDC00)
            elif 0xD800 <= code <= 0xDFFF:
                code = 0xFFFD
            high = None
            emit(chr(code))
            continue
        fallback = 0
        if name == "uc":
            uc = max(0, min(num or 0, 10))
        elif name == "ansicpg":
            flush()
            doc_codec = codec = _codepage_codec(num)
        elif name == "deff" and num is not None:
            deff = num
        elif name == "f" and num is not None:
            flush()
            codec = fonts.get(num, doc_codec)
        elif name == "plain":
            flush()
            codec = fonts.get(deff, doc_codec)
        elif name == "htmlrtf":
            rtfonly = num != 0
        elif name in _RTF_SYMBOLS and visible():
            emit(_RTF_SYMBOLS[name])
    flush()
    return "".join(out)


# --------------------------------------------------------------------------- #
# MAPI properties in a .msg (MS-OXMSG)
# --------------------------------------------------------------------------- #
class _Props:
    """The properties of one message, recipient, or attachment storage: fixed-size values from
    __properties_version1.0, strings and binaries from the __substg1.0_ streams beside it."""

    def __init__(self, cfb: _CFB, storage: int, header: int, codepage: Optional[int] = None):
        self.cfb = cfb
        self.kids = cfb.children(storage)
        self.fixed: Dict[int, bytes] = {}
        raw = cfb.read(self.kids.get("__PROPERTIES_VERSION1.0")) or b""
        for off in range(header, len(raw) - 15, 16):
            tag = struct.unpack_from("<I", raw, off)[0]
            self.fixed.setdefault(tag, raw[off + 8:off + 16])
        self.codepage = self.long(0x3FFD) or self.long(0x3FDE) or codepage

    def long(self, pid: int) -> Optional[int]:
        value = self.fixed.get((pid << 16) | 0x0003)
        return struct.unpack("<I", value[:4])[0] if value else None

    def boolean(self, pid: int) -> bool:
        value = self.fixed.get((pid << 16) | 0x000B)
        return bool(value and value[0])

    def time(self, pid: int) -> Optional[str]:
        """A PT_SYSTIME (FILETIME, 100 ns since 1601) as ISO 8601 in UTC."""
        value = self.fixed.get((pid << 16) | 0x0040)
        if not value:
            return None
        ticks = struct.unpack("<Q", value[:8])[0]
        if ticks == 0:
            return None
        try:
            when = _dt.datetime(1601, 1, 1, tzinfo=_UTC) + _dt.timedelta(microseconds=ticks // 10)
        except OverflowError:
            return None
        if not 1970 <= when.year <= 2200:
            return None
        return when.replace(microsecond=0).isoformat()

    def stream(self, name: str) -> Optional[int]:
        return self.kids.get(name)

    def binary(self, pid: int) -> Optional[bytes]:
        return self.cfb.read(self.kids.get(f"__SUBSTG1.0_{pid:04X}0102"))

    def string(self, pid: int) -> str:
        """A PT_UNICODE (UTF-16LE) or PT_STRING8 (the message's code page) property, or ""."""
        raw = self.cfb.read(self.kids.get(f"__SUBSTG1.0_{pid:04X}001F"))
        if raw is not None:
            if len(raw) % 2:
                raw = raw[:-1]
            return _unsurrogate(raw.decode("utf-16-le", "replace").split("\0", 1)[0])
        raw = self.cfb.read(self.kids.get(f"__SUBSTG1.0_{pid:04X}001E"))
        if raw is not None:
            raw = raw.split(b"\0", 1)[0]
            try:
                return raw.decode(_codepage_codec(self.codepage), errors="replace")
            except Exception:  # noqa: BLE001
                return _latin(raw)
        return ""


def _smtp(addr: str) -> str:
    """An SMTP address, or "" for an Exchange legacy DN (/O=EXCHANGELABS/...) or junk."""
    addr = _one_line(addr).strip("<>'\" ")
    if "@" not in addr or addr.startswith("/") or " " in addr:
        return ""
    return addr.lower()


def _suffix(name: str) -> int:
    try:
        return int(name.rsplit("#", 1)[1], 16)
    except (IndexError, ValueError):
        return 1 << 40


def _load_msg(run: _Run, data: bytes, source: str, depth: int) -> None:
    try:
        cfb = _CFB(data)
    except _CFBError as exc:
        run.skip(source, f"not a readable Outlook .msg file ({exc})")
        return
    except Exception as exc:  # noqa: BLE001 - struct errors on a mangled header
        run.skip(source, f"not a readable Outlook .msg file ({type(exc).__name__})")
        return
    kids = cfb.children(0)
    if "__PROPERTIES_VERSION1.0" not in kids and not any(k.startswith("__SUBSTG1.0_") for k in kids):
        run.skip(source, "not an Outlook email (a Compound File without message properties)")
        return
    _msg_email(run, cfb, 0, source, depth, 32, None)


def _msg_email(run: _Run, cfb: _CFB, storage: int, source: str, depth: int, header: int,
               parent_codepage: Optional[int]) -> None:
    """One message storage (the file's root, or an embedded message): add its ParsedEmail, then
    the messages attached to it."""
    if not run.room(source):
        return
    limits = run.limits
    # Damage found while opening the file belongs to the file's own (root) email.
    was_damaged, cfb.damaged = cfb.damaged, (cfb.damaged and storage == 0)
    em = _blank_email(source, "msg")
    warnings: List[str] = em["warnings"]
    p = _Props(cfb, storage, header, parent_codepage)

    headers = None
    transport = p.string(0x007D)
    if transport.strip():
        try:
            headers = email.parser.Parser(policy=email.policy.default).parsestr(transport, headersonly=True)
        except Exception:  # noqa: BLE001
            headers = None

    def header(name: str) -> Optional[str]:
        return _raw_header(headers, name) if headers is not None else None

    em["subject"] = _one_line(p.string(0x0037)) or _one_line(_decode_words(header("subject")))
    em["message_id"] = _message_id(p.string(0x1035)) or _message_id(header("message-id"))

    # The sender. From Exchange the address is a legacy DN, so look for an SMTP one.
    em["from_name"] = _strip_quotes(_one_line(p.string(0x0C1A) or p.string(0x0042)))
    from_email = (_smtp(p.string(0x0C1F)) or _smtp(p.string(0x5D01)) or _smtp(p.string(0x0065))
                  or _smtp(p.string(0x5D02)))
    from_header = _address_list(header("from"))
    if not from_email and from_header:
        from_email = from_header[0][1]
    if not em["from_name"] and from_header:
        em["from_name"] = from_header[0][0]
    em["from_email"] = from_email
    if not from_email:
        warnings.append("no sender address")

    # Recipients: one storage each. An Exchange recipient may have only a legacy DN; the transport
    # headers can still give its SMTP address, found by display name. Some files list a
    # recipient twice.
    by_name: Dict[str, str] = {}
    for key in ("to", "cc"):
        for n, a in _address_list(header(key)):
            if n and a:
                by_name.setdefault(n.lower(), a)
    listed_once = set()
    recips = sorted((k for k in p.kids if k.startswith("__RECIP_VERSION1.0_#")), key=_suffix)
    for name in recips[:_MAX_MSG_OBJECTS]:
        idx = p.kids[name]
        if not cfb.is_storage(idx):
            continue
        r = _Props(cfb, idx, 8, p.codepage)
        kind = (r.long(0x0C15) or 1) & 0xF
        display = _strip_quotes(_one_line(r.string(0x3001) or r.string(0x5FF6)))
        addr = _smtp(r.string(0x39FE)) or _smtp(r.string(0x3003)) or by_name.get(display.lower(), "")
        if display.lower() == addr:
            display = ""
        entry = _format_address(display, addr)
        once = (kind, addr or display.lower())
        if entry and kind in (1, 2) and once not in listed_once:
            listed_once.add(once)
            em["to" if kind == 1 else "cc"].append(entry)
    for key, prop in (("to", 0x0E04), ("cc", 0x0E03)):
        if any("@" in entry for entry in em[key]):
            continue
        # No recipient storage gave an address (none there, or only Exchange DNs): the transport
        # headers, else the display names Outlook shows.
        listed = [(n, a) for n, a in _address_list(header(key)) if a]
        if listed:
            em[key] = [_format_address(n, a) for n, a in listed]
        elif not recips:
            em[key] = [s for s in (_one_line(x) for x in p.string(prop).split(";")) if s]

    em["date"] = _iso_date(header("date")) or p.time(0x0039) or p.time(0x0E06)
    if em["date"] is None:
        warnings.append("no date")

    # The body: PR_BODY, else PR_HTML, else the compressed RTF.
    html_text = ""
    raw_html = p.binary(0x1013)
    if raw_html is not None:
        m = re.search(rb"charset\s*=\s*[\"']?([A-Za-z0-9_.:-]+)", raw_html[:4096], re.I)
        charset = m.group(1).decode("ascii") if m else None
        if p.long(0x3FDE):
            html_text = raw_html.decode(_codepage_codec(p.long(0x3FDE)), errors="replace")
        else:
            html_text = _decode_bytes(raw_html, charset, warnings, "body")
    else:
        html_text = p.string(0x1013)
    plain = p.string(0x1000)
    if plain.strip():
        em["body"], em["body_format"] = _clean_body(plain), "text"
    elif html_text.strip():
        em["body"], em["body_format"] = html_to_text(html_text), "html"
    else:
        rtf = p.binary(0x1009)
        if rtf:
            raw_rtf, problems = decompress_rtf(rtf)
            warnings.extend(problems)
            if raw_rtf:
                if b"\\fromhtml" in raw_rtf[:4096]:
                    html_text = _rtf_walk(raw_rtf, True)
                    em["body"] = html_to_text(html_text)
                else:
                    em["body"] = rtf_to_text(raw_rtf)[0]
                em["body_format"] = "rtf"
    em["body"] = _cap_body(em["body"], limits, warnings)

    # Attachments: one storage each. Embedded messages become their own ParsedEmails.
    candidates: List[_Candidate] = []
    nested: Nested = []
    cids = _cids(html_text)
    atts = sorted((k for k in p.kids if k.startswith("__ATTACH_VERSION1.0_#")), key=_suffix)
    if len(atts) > _MAX_MSG_OBJECTS:
        warnings.append(f"{len(atts) - _MAX_MSG_OBJECTS:,} more attachments were not looked at")
    for name in atts[:_MAX_MSG_OBJECTS]:
        idx = p.kids[name]
        if not cfb.is_storage(idx):
            continue
        a = _Props(cfb, idx, 8, p.codepage)
        obj = a.stream("__SUBSTG1.0_3701000D")
        method = a.long(0x3705)
        if method is None:
            method = 5 if cfb.is_storage(obj) else 1
        filename = _one_line(a.string(0x3707) or a.string(0x3704) or a.string(0x3001))
        if method == 5 and cfb.is_storage(obj):
            label = filename
            if not label:
                label = _one_line(_Props(cfb, obj, 24, p.codepage).string(0x0037))
            label = _safe_name(label, f"attached email {len(nested) + 1}")
            if _ext(label) not in EMAIL_EXTENSIONS:
                label += ".msg"
            nested.append((label, lambda src, d, o=obj, cp=p.codepage: _msg_email(run, cfb, o, src, d, 24, cp)))
            continue
        if method == 0:
            continue
        mime = _one_line(a.string(0x370E)).lower()
        aname = _safe_name(filename, "attachment" + _guess_ext(mime))
        mime = mime or _guess_type(aname)
        data_idx = a.stream("__SUBSTG1.0_37010102")
        if method == 1 and mime == "multipart/signed":
            # A signed email: Outlook keeps the whole signed MIME message as one attachment
            # (smime.p7m), and the real attachments are inside it.
            signed = _parse_mime(cfb.read(data_idx) or b"")
            if signed is not None:
                plain2, htmls2, found, nested2 = _mime_walk(run, signed, warnings)
                candidates.extend(found)
                nested.extend(nested2)
                for h in htmls2:
                    cids |= _cids(h)
                if em["body_format"] == "none":
                    _set_body(em, plain2, htmls2, limits, warnings)
                continue
        if method == 1 and (_ext(aname) in EMAIL_EXTENSIONS or mime in ("message/rfc822", "application/vnd.ms-outlook")):
            data = cfb.read(data_idx)
            if data and sniff(data) in ("eml", "msg"):
                nested.append((aname, lambda src, d, b=data: _load_bytes(run, b, src, d)))
                continue
        if method == 6:
            warnings.append(f"{aname}: an embedded OLE object is not read")
        if _is_tnef(aname, mime):
            warnings.append("winmail.dat (Outlook rich text) is not read")
        if mime in _OPAQUE_TYPES or _ext(aname) == ".p7m":
            warnings.append(f"{aname}: encrypted or signed (S/MIME) content is not read")
        inline = a.boolean(0x7FFE) or bool((a.long(0x3714) or 0) & 0x4)
        loader: Callable[[], Optional[bytes]] = _no_data
        if method == 1 and data_idx is not None:
            loader = (lambda i=data_idx: cfb.read(i))
        candidates.append(_Candidate(aname, mime, inline, _one_line(a.string(0x3712)).strip("<> "),
                                     loader, cfb.size(data_idx) if method == 1 else 0))
    em["attachments"] = _keep_attachments(run, candidates, cids, warnings)
    em["container_only"] = bool(nested) and not em["attachments"]
    if cfb.damaged:
        warnings.append("parts of this Outlook file are damaged and were not read")
    cfb.damaged = was_damaged or cfb.damaged
    run.emails.append(em)
    _open_nested(run, source, depth, nested)
