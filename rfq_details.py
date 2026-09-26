"""
RFQ details: pull what a quoting manager needs out of each RFQ email and its files, and write it all
to one consolidated file with one row per part line.

    python rfq_details.py                  # beta inbox -> data/rfq_beta/rfq_details.csv and .json
    python rfq_details.py --check          # field accuracy against tests/rfq_beta_fields_truth.json

Standard library only. The text of each attachment comes from ocr.file_text (text layer, OCR, or
STEP header); this module never runs tesseract itself. It reads four kinds of source:
    the email       subject and body: quantities, dates, material and finish, requirements
    RFQ forms       RFQ number, respond-by date, one table row per part, quote requirements
    drawings        the title block (part number, rev, title, material, finish) and legends
    STEP models     part number, title, and revision from the header
A file's kind comes from what is printed on it, not its name or format: a phone photo or a
screenshot whose text carries title block markers is a drawing. A part number is only taken from a
place that holds one (the DWG NO. cell, a P/N label, an RFQ form row, the email, a STEP header),
so a date or a note on a drawing never becomes a part line.
When sources disagree the RFQ form wins for quantities and dates, the drawing title block wins for
part number, rev, material, finish, and description, and the email counts when it is the only
source. Every disagreement is written to "check" so a person can look. OCR text is noisy, so labels
are matched loosely, O/0 and I/1 style confusions are folded before values are compared, and when a
clean source (email, text layer, STEP) and an OCR reading agree up to those confusions, the clean
spelling is kept. Nothing is invented: a value that is not found stays null.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
BETA_EMAILS = DATA_DIR / "rfq_beta" / "emails.json"
BETA_OUT = DATA_DIR / "rfq_beta" / "rfq_details"
BETA_CACHE = DATA_DIR / "rfq_beta" / "ocr_cache.json"
FIELDS_TRUTH = HERE / "tests" / "rfq_beta_fields_truth.json"

# The sample emails carry a time but no date; the demo's story is that they arrived today, on the
# day the beta inbox was built. The CLI resolves "next Friday" and "within two weeks" against this
# date so the committed CSV does not change from one day to the next.
SAMPLE_INBOX_DATE = dt.date(2026, 9, 25)

# Below this line confidence an OCR value that no other source confirms gets a "verify" note.
LOW_OCR_CONF = 60.0

FILE_TYPES = ("RFQ form", "drawing", "3D model", "photo", "screenshot", "other")

# --------------------------------------------------------------------------- #
# Text cleanup and fuzzy helpers
# --------------------------------------------------------------------------- #
_TRANS = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "−": "-", "‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"',
    "”": '"', " ": " ", "´": "'", "′": "'", "″": '"', "­": "",
})


def clean(text: Any) -> str:
    """One line of text: dashes and quotes made plain (OCR loves em dashes), spaces collapsed."""
    return re.sub(r"[ \t\f\v]+", " ", str(text or "").translate(_TRANS)).strip()


def clean_block(text: Any) -> str:
    return "\n".join(clean(line) for line in str(text or "").translate(_TRANS).splitlines())


def norm(text: Any) -> str:
    """Uppercase words only, for comparing wordings."""
    return " ".join(re.sub(r"[^A-Z0-9./]+", " ", clean(text).upper()).split())


def similar(a: Any, b: Any) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# OCR swaps look-alike characters. Comparing keys with these folded makes "CI-10442", "Cl-10442",
# and "C1-10442" the same part number without guessing which spelling is right.
_FOLD = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1", "!": "1", "S": "5",
                       "B": "8", "Z": "2", "G": "6", "T": "7"})


def fold(text: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", clean(text).upper()).translate(_FOLD)


_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1", "!": "1", "S": "5", "B": "8",
             "Z": "2", "G": "6"}
_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}


def fix_part_number(token: str) -> str:
    """Repair a part number read by OCR: letters in the prefix, digits in the numeric groups.
    'C1-1O442' -> 'CI-10442', '8WM-3106' -> 'BWM-3106'. A lowercase l is almost always an I in an
    uppercase prefix and a 1 among digits."""
    token = clean(token).strip(".,;:|()[]{}'\"")
    parts = token.split("-")
    if len(parts) < 2:
        return token.upper()
    out = ["".join(_TO_LETTER.get(c, c) for c in parts[0].replace("l", "I").upper())]
    for part in parts[1:]:
        up = part.upper()
        digits = sum(c.isdigit() for c in up)
        if digits and digits >= len(up) / 2:
            up = "".join(_TO_DIGIT.get(c, c) for c in part.replace("l", "1").upper())
        out.append(up)
    return "-".join(out)


def ocr_fix_spec(text: str) -> str:
    """Fix the OCR slips that change a material or finish callout: an aluminum temper read as
    6061-16511 or 6061-7651, TYPE Ill for TYPE III, CLASS l for CLASS 1."""
    text = re.sub(r"\b([1-7]\d{3})-[1I7l|]([0-9OIl]{1,4})\b",
                  lambda m: f"{m.group(1)}-T{m.group(2).replace('O', '0').replace('I', '1').replace('l', '1')}", text)
    text = re.sub(r"\bTYPE\s+([IlL1|]{1,3})\b", lambda m: "TYPE " + "I" * len(m.group(1)), text)
    text = re.sub(r"\bCLASS\s+[lI|]\b", "CLASS 1", text)
    text = re.sub(r"(?<=[A-Z])!(?=[\s,.]|$)", "I", text)  # AIS! 4140 -> AISI 4140
    text = re.sub(r"\bDATUM([A-Z])\b", r"DATUM \1", text)  # the space before a datum letter is thin
    text = re.sub(r"\b(ASTM|AMS|MIL)\s*[-]?\s*", lambda m: m.group(0), text)
    return text


def _num(token: str) -> Optional[int]:
    """'1,000' -> 1000, '10k' -> 10000, '5O' -> 50 (OCR letter O). None when it is not a count."""
    t = clean(token).upper().replace(",", "").rstrip(".")
    t = "".join(_TO_DIGIT.get(c, c) if c in "OIlLSB|" else c for c in t) if re.search(r"\d", t) else t
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*K", t)
    if m:
        return int(round(float(m.group(1)) * 1000))
    if re.fullmatch(r"\d{1,7}", t):
        return int(t)
    return None


def _tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Z0-9]+", clean(text).upper()) if t]


_STOP = {"THE", "A", "AN", "AND", "OR", "OF", "TO", "FOR", "ON", "IN", "WITH", "BY", "PER", "IS", "ARE", "BE",
         "WILL", "YOUR", "OUR", "ANY", "EACH", "AS", "AT", "IT", "PLEASE", "WE", "YOU", "THIS", "THAT", "IF"}


def _content(text: str) -> set:
    return {t for t in _tokens(text) if t not in _STOP}


def _covered(a: str, b: str) -> bool:
    """Every content word of a (at least three of them) appears in b."""
    ta = _content(a)
    return len(ta) >= 3 and ta <= _content(b)


def overlap(a: str, b: str) -> float:
    """Share of the shorter phrase's content words found in the other one."""
    ta, tb = _content(a), _content(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #
# Part numbers look like QA-41127, HPV-2045, BWM-3140-08, TO-5520. The lookbehind keeps spec
# numbers out (MIL-A-8625 must not give "A-8625", BWM-QS-0412 must not give "QS-0412").
# A letter group may sit between prefix and number (BWM-T-0415, BFW-SS-5812).
PN_BODY = r"[A-Z][A-Z0-9]{0,4}(?:-[A-Z]{1,3})?-\d{2,6}(?:-[A-Z0-9]{1,4})?"
PN_RE = re.compile(r"(?<![A-Za-z0-9-])(" + PN_BODY + r")(?![A-Za-z0-9-])")
# The same with OCR slack: digits allowed in the prefix, look-alike letters in the digits.
PN_OCR_RE = re.compile(r"(?<![A-Za-z0-9-])([A-Za-z0-9|]{1,5}(?:-[A-Za-z]{1,3})?-[0-9OIlSBZ|]{2,6}(?:-[A-Za-z0-9]{1,4})?)"
                       r"(?![A-Za-z0-9-])")
# Standards that are written like part numbers. "SP", "MS", and "AN" stay out of this list: they are
# real part number prefixes too, and the standards that use them are written with a space.
SPEC_PREFIXES = {"MIL", "AMS", "ASTM", "SAE", "NAS", "ISO", "DFARS", "NIST", "UNC", "UNF", "UNS", "ANSI", "ASME",
                 "QQ", "DTL", "STD", "AWS", "RAL", "NASM", "PRF"}
RFQ_NO_RE = re.compile(r"\bRFQ(?:[-\s#:]*(?:NO\.?|NUMBER|#))?[\s#:]*((?:[A-Z]{1,5}-)?\d{2}-\d{3,5}|[A-Z]{2,5}-\d{3,6})\b",
                       re.IGNORECASE)
RFQ_ID_RE = re.compile(r"\b((?:RFQ|RF[O0Q]|[A-Z]{1,4})-?[0-9OISB]{2}-[0-9OISB]{3,5})\b")
QUOTE_REF_RE = re.compile(r"\b(Q(?:T|UOTE)?-?\d{2}-\d{3,6}|Q\d{5,8})\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

MATERIAL_STRONG = re.compile(
    r"\b(?:[1-7]\d{3}-T\d{1,4}(?:\s*OR\s*T\d{1,4})?|[1-7]\d{3}\s+(?:ALUMINUM|ALUMINIUM|AL)\b|(?:AL|ALUMINUM)\s+[1-7]\d{3}"
    r"|(?:17-4|15-5|13-8)\s?PH|30[34]L?\s+(?:STAINLESS|SS|SST|CRES)|31[06]L\b|31[06]\s+(?:STAINLESS|SS|SST)"
    r"|STAINLESS(?:\s+STEEL)?\s+\d{3}L?|(?:AISI\s+)?(?:4140|4340|8620|12L14)\b|(?:1018|1020|1045|1215)\s+STEEL"
    r"|A36\b|C3[46]\d{1,3}\b|(?:IMPLANT[- ]GRADE\s+)?PEEK\b|(?:UNFILLED\s+)?PEEK\b|DELRIN|ACETAL|ULTEM|PTFE|TITANIUM"
    r"|TI-?6AL-?4V|6AL-?4V|INCONEL\s*\d*|BRASS|BRONZE|COPPER)",
    re.IGNORECASE)
MATERIAL_WORD = re.compile(r"\b(?:ALUMINUM|ALUMINIUM|STAINLESS|STEEL|SST|CRES|BRASS|BRONZE|COPPER|TITANIUM|PEEK|"
                           r"PLASTIC|NYLON|DELRIN|ACETAL|INCONEL|AL)\b", re.IGNORECASE)
FINISH_RE = re.compile(
    r"\b(?:(?:HARD|BLACK|CLEAR|COLOR|RED|BLUE|GOLD)\s+)?ANODI[ZS](?:E|ED|ING)\b|\bPASSIVAT(?:E|ED|ION)\b"
    r"|\b(?:CLEAR\s+)?CHEM(?:ICAL)?\s+FILM\b|\bCONVERSION\s+COAT(?:ING)?\b|\bALODINE\b|\bIRIDITE\b"
    r"|\bELECTROLESS\s+NICKEL\b|\bNICKEL\s+PLAT\w*\b|\bZINC(?:\s+PLAT\w*)?\b|\bBLACK\s+OXIDE\b"
    r"|\bPOWDER\s*COAT\w*\b|\bPAINT(?:ED)?\b|\bGOLD\s+(?:FLASH|PLAT\w*)\b|\bSILVER\s+PLAT\w*\b"
    r"|\bTIN\s+PLAT\w*\b|\bCHROME\s+PLAT\w*\b|\bCADMIUM\b|\bELECTROPOLISH\w*\b|\bNITRID\w*\b",
    re.IGNORECASE)
SPEC_WORDS = re.compile(r"\b(?:PER|CLASS|TYPE|METHOD|COND|CONDITION|ASTM|AMS|MIL|GRADE|NITRIC|CITRIC|THK)\b")
PART_NOUNS = {"BLOCK", "PLATE", "FRAME", "SHAFT", "BRACKET", "HOUSING", "COVER", "PIN", "SPACER", "BOX", "BODY",
              "MANIFOLD", "CAGE", "MOUNT", "RING", "CELL", "PART", "PARTS", "BAR", "ROD", "TUBE"}

MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                                      "NOV", "DEC"], start=1)}
WEEKDAYS = {d: i for i, d in enumerate(["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"])}
NUM_WORDS = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8,
             "NINE": 9, "TEN": 10, "A": 1, "AN": 1}

EXPORT_PATTERNS = [
    ("ITAR", re.compile(r"\bITAR\b|INTERNATIONAL\s+TRAFFIC\s+IN\s+ARMS|ARMS\s+EXPORT\s+CONTROL\s+ACT|22\s*CFR\s*12\d",
                        re.IGNORECASE)),
    ("EAR", re.compile(r"EXPORT\s+ADMINISTRATION\s+REGULATIONS|\bECCN\b|15\s*CFR\s*7[3-7]\d", re.IGNORECASE)),
    ("CUI", re.compile(r"\bCUI\b|CONTROLLED\s+UNCLASSIFIED|CONTROLLED\s+TECHNICAL\s+INFORMATION|DISTRIBUTION\s+"
                       r"STATEMENT\s+[B-F]\b|DFARS\s*252\.204-7012|NIST\s*SP\s*800-171", re.IGNORECASE)),
]
# The same legends as fuzzy phrases, for OCR text where a letter or two is wrong.
EXPORT_PHRASES = [
    ("ITAR", "INTERNATIONAL TRAFFIC IN ARMS REGULATIONS"), ("ITAR", "ARMS EXPORT CONTROL ACT"),
    ("EAR", "EXPORT ADMINISTRATION REGULATIONS"), ("CUI", "CONTROLLED UNCLASSIFIED INFORMATION"),
    ("CUI", "CONTROLLED TECHNICAL INFORMATION"), ("CUI", "DISTRIBUTION AUTHORIZED TO THE DEPARTMENT OF DEFENSE"),
]
NOT_EXPORT = re.compile(r"\b(?:NOT|NON|NO)[- ](?:ITAR|EXPORT[- ]CONTROLLED|CUI|CONTROLLED)\b", re.IGNORECASE)

REQUIREMENT_RE = re.compile(
    r"\bCERT(?:S|IFICATION|IFICATIONS|IFIED)?\b|\bC\s?OF\s?C\b|\bCOC\b|CERTIFICATE OF CONFORM|\bFAI\b|FIRST ARTICLE|AS9102"
    r"|\bISO\s*\d{4,5}|\bAS\s?9100|ITAR REGISTRATION|\bNIST\b|\bDFARS\b|COMPLIANCE|\bNRE\b|\bTOOLING\b|SETUP (?:AND|CHARGE)"
    r"|BREAK OUT|SEPARATE LINE|OWN LINE|WITH AND WITHOUT|PRICE BREAKS|\bBLANKET\b|\bEXPEDITE|\bOVERTIME\b|NO-?BID"
    r"|DOUBLE BAG|PRECISION CLEAN|BARE AND OILED|FURNISHED|TRACEAB|INSPECTION",
    re.IGNORECASE)

# Words around a date that make it the date the quote is due, and words that make it something else.
RESPOND_CUES = re.compile(
    r"QUOTE (?:IS )?DUE|DUE (?:BY|IN|ON)|QUOTE BY|RESPOND(?:S|ED)? BY|RESPONSES? (?:ARE |IS )?DUE|REPLY BY|"
    r"GET BACK TO (?:ME|US)|QUOTE BACK|QUOTE DATE|QUOTE (?:IT )?TODAY|QUOTES? (?:NEEDED|REQUIRED) BY|BID DUE|"
    r"PRICING BY|NEED (?:THE |A |YOUR )?(?:QUOTE|PRICING|PRICE)", re.IGNORECASE)
NOT_RESPOND = re.compile(r"\bLINK\b|GOOD THROUGH|EXPIRES|PARTS BY|NEED THEM|FIRST PARTS|DELIVER|SHIP|PROMISE|ON DOCK",
                         re.IGNORECASE)
DELIVERY_CUES = re.compile(r"NEED (?:THEM|THE PARTS|PARTS|IT)\b|PARTS BY|FIRST PARTS|DELIVERY|DELIVER(?:ED)? BY|"
                           r"SHIP BY|ON DOCK BY|REQUIRED DELIVERY", re.IGNORECASE)


def _find_export(text: str, fuzzy: bool) -> List[Tuple[str, str]]:
    """Export-control kinds in a text, with the phrase that proved each one."""
    found: List[Tuple[str, str]] = []
    if not text:
        return found
    for kind, pat in EXPORT_PATTERNS:
        m = pat.search(text)
        if m:
            phrase = m.group(0)
            if kind == "EAR":
                code = re.search(r"\bECCN\s*[:#]?\s*([0-9][A-E][0-9]{3}[A-Z]?)", text, re.IGNORECASE)
                kind = f"EAR (ECCN {code.group(1).upper()})" if code else "EAR"
            found.append((kind, phrase))
    if fuzzy:
        words = norm(text).split()
        have = {k.split(" ")[0] for k, _ in found}
        for kind, phrase in EXPORT_PHRASES:
            if kind in have:
                continue
            target = phrase.split()
            n = len(target)
            for i in range(max(0, len(words) - n + 1)):
                window = " ".join(words[i:i + n])
                if all(difflib.SequenceMatcher(None, a, b).ratio() >= 0.7 for a, b in zip(words[i:i + n], target)) \
                        and difflib.SequenceMatcher(None, window, phrase).ratio() >= 0.8:
                    found.append((kind, window))
                    have.add(kind)
                    break
    return found


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
def _year_for(month: int, day: int, today: dt.date) -> Optional[dt.date]:
    """A month and day with no year: this year, unless that is well in the past."""
    for year in (today.year, today.year + 1):
        try:
            d = dt.date(year, month, day)
        except ValueError:
            return None
        if d >= today - dt.timedelta(days=60):
            return d
    return None


def add_business_days(start: dt.date, days: int) -> dt.date:
    current, added = start, 0
    while added < days:
        current += dt.timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def parse_date(text: str, today: dt.date) -> Optional[Tuple[dt.date, str]]:
    """The first date in a phrase, absolute or relative to today, with the words that gave it."""
    t = clean(text)
    up = t.upper()
    m = re.search(r"\b(20\d\d)-(\d\d)-(\d\d)\b", t)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.group(0)
        except ValueError:
            pass
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
    if m:
        mo, da = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= da <= 31:
            if m.group(3):
                year = int(m.group(3))
                year += 2000 if year < 100 else 0
                try:
                    return dt.date(year, mo, da), m.group(0)
                except ValueError:
                    pass
            else:
                d = _year_for(mo, da, today)
                if d:
                    return d, m.group(0)
    m = re.search(r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\.?\s+(\d{1,2})(?:ST|ND|RD|TH)?"
                  r"(?:,?\s*(20\d\d))?\b", up)
    if m:
        mo, da = MONTHS[m.group(1)[:3]], int(m.group(2))
        try:
            d = dt.date(int(m.group(3)), mo, da) if m.group(3) else _year_for(mo, da, today)
        except ValueError:
            d = None
        if d:
            return d, t[m.start():m.end()]
    m = re.search(r"\bWITHIN\s+(\w+)\s+(BUSINESS\s+DAYS?|WORKING\s+DAYS?|DAYS?|WEEKS?)\b|\bIN\s+(\w+)\s+"
                  r"(BUSINESS\s+DAYS?|WORKING\s+DAYS?|DAYS?|WEEKS?)\b", up)
    if m:
        count_word = m.group(1) or m.group(3)
        unit = m.group(2) or m.group(4)
        count = int(count_word) if count_word.isdigit() else NUM_WORDS.get(count_word)
        if count:
            if unit.startswith(("BUSINESS", "WORKING")):
                d = add_business_days(today, count)
            elif unit.startswith("WEEK"):
                d = today + dt.timedelta(weeks=count)
            else:
                d = today + dt.timedelta(days=count)
            return d, t[m.start():m.end()]
    m = re.search(r"\b(NEXT|THIS|BY|ON)?\s*(MON|TUE|WED|THU|FRI|SAT|SUN)[A-Z]*DAY\b", up)
    if m:
        wd = WEEKDAYS[m.group(2)]
        if m.group(1) == "NEXT":  # the named day in the following calendar week
            d = today + dt.timedelta(days=7 - today.weekday() + wd)
        else:
            d = today + dt.timedelta(days=(wd - today.weekday()) % 7)
        return d, t[m.start():m.end()].strip()
    m = re.search(r"\b(TODAY|TOMORROW|END OF (?:THE )?WEEK|EOW|END OF (?:THE )?DAY|EOD)\b", up)
    if m:
        word = m.group(1)
        if word == "TOMORROW":
            d = today + dt.timedelta(days=1)
        elif word in ("TODAY", "END OF DAY", "END OF THE DAY", "EOD"):
            d = today
        else:
            d = today + dt.timedelta(days=(4 - today.weekday()) % 7)
        return d, t[m.start():m.end()]
    return None


def _iso_dates(text: str) -> List[Tuple[dt.date, str]]:
    """ISO dates in OCR text, with look-alike letters folded ('2026-1O-O9')."""
    out = []
    for m in re.finditer(r"\b([2Z][0O][0-9OIlSB]{2})-([0-9OIlSB]{2})-([0-9OIlSB]{2})\b", text):
        y, mo, d = (int("".join(_TO_DIGIT.get(c, c) for c in g.upper())) for g in m.groups())
        try:
            out.append((dt.date(y, mo, d), m.group(0)))
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------------- #
# Attachment documents
# --------------------------------------------------------------------------- #
class Line:
    __slots__ = ("text", "conf", "page", "box", "i")

    def __init__(self, text: str, conf: Optional[float], page: int, box: Optional[Sequence[float]], i: int):
        self.text, self.conf, self.page, self.i = text, conf, page, i
        self.box = tuple(float(v) for v in box) if box and len(box) == 4 else None

    @property
    def h(self) -> float:
        return (self.box[3] - self.box[1]) if self.box else 0.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Line({self.text!r}, {self.box})"


class Doc:
    """One attachment's text as the extractor sees it."""

    def __init__(self, name: str, media: str, result: Optional[Dict[str, Any]]):
        r = result or {}
        self.name = name
        self.media = (media or "").lower()
        self.method = r.get("method") or "none"
        self.text = clean_block(r.get("text") or "")
        self.conf = r.get("confidence")
        self.error = r.get("error")
        self.settings = r.get("settings") or {}
        self.ocr = self.method == "ocr"
        self.lines: List[Line] = []
        for i, ln in enumerate(r.get("lines") or []):
            if isinstance(ln, dict) and clean(ln.get("text")):
                self.lines.append(Line(clean(ln.get("text")), ln.get("conf"), int(ln.get("page") or 1),
                                       ln.get("bbox"), i))
        self.boxed = sum(1 for ln in self.lines if ln.box) >= 5
        # Reading-order lines for text parsing. OCR text joins the cells of one row with wide gaps;
        # those gaps are kept as " | " so a cell boundary is still visible.
        self.rows: List[str] = [re.sub(r"\s{3,}", " | ", ln).strip() for ln in
                                (r.get("text") or "").translate(_TRANS).splitlines()]
        self.rows = [clean(x) for x in self.rows if clean(x)]
        self.kind = "other"
        self.capture = "digital"
        self.parsed: Dict[str, Any] = {}

    @property
    def label(self) -> str:
        if self.method == "text-layer":
            return f"{self.name} (text layer)"
        if self.method == "ocr":
            return f"{self.name} (OCR {self.conf:.0f}%)" if isinstance(self.conf, (int, float)) else f"{self.name} (OCR)"
        if self.method == "step-header":
            return f"{self.name} (STEP header)"
        return self.name

    @property
    def text_from(self) -> str:
        if self.method == "text-layer":
            return "text layer"
        if self.method == "ocr":
            return f"OCR {self.conf:.0f}%" if isinstance(self.conf, (int, float)) else "OCR"
        if self.method == "step-header":
            return "STEP header"
        return "none"


FORM_MARKERS = [r"REQUEST\s*F[O0]R\s*QU[O0]TAT", r"\bRF[QO0]\s*N[O0]", r"RESP[O0]ND\s*BY", r"QU[O0]TE\s+REQUIREMENTS",
                r"QUANTITIES", r"UNIT\s*PRICE", r"LEAD\s*TIME", r"SUPPLIER\s+RESP[O0]NSE", r"TO\s+SUPPLIER"]
# Words printed on nearly every engineering drawing: the border and tolerance block phrases, and
# the title block cell labels (matched as whole lines, loosely, because OCR bends them).
DRAWING_PHRASES = [r"REVISIONS", r"THIRD\s+ANGLE", r"UNLESS\s+OTHERWISE\s+SPECIFIED", r"TOLERANCES",
                   r"D[O0]\s+N[O0]T\s+SCALE", r"\b[DO0]{1,2}WG\.?\s*N[O0]",
                   r"(?:ISOMETRIC|FRONT|TOP|SIDE|BOTTOM|REAR)\s+VIEW|SECTION\s+[A-Z]-[A-Z]\b|DETAIL\s+[A-Z]\b",
                   r"INTERPRET\s+(?:DRAWING\s+)?PER\s+ASME", r"BREAK\s+(?:ALL\s+)?SHARP\s+EDGES"]
DRAWING_CELLS = ["TITLE", "MATERIAL", "FINISH", "SIZE", "SCALE", "SHEET", "DRAWN", "REV", "NOTES", "DWG NO"]


def _capture(doc: Doc) -> str:
    """How an OCR'd file was made (scan, fax, photo, screenshot), from what ocr.py measured."""
    if not doc.ocr:
        return "digital"
    sources = " ".join(str(p.get("source", "")) for p in doc.settings.get("pages") or [] if isinstance(p, dict))
    sources = (sources + " " + json.dumps(doc.settings)).lower()
    if "photo" in sources:
        return "photo"
    if "screen" in sources:
        return "screenshot"
    if "fax" in sources or "bilevel" in sources or "1-bit" in sources:
        return "fax"
    if doc.media == "jpg":
        return "photo"
    if doc.media == "png":
        return "screenshot"
    return "scan"


def classify(doc: Doc) -> Tuple[str, str]:
    """(file type, how it was captured), from what is printed on the file, not its name or format.
    A phone photo or a screenshot whose text carries title block markers is a drawing."""
    up = doc.text.upper()
    capture = _capture(doc)
    if doc.method == "step-header" or "ISO-10303" in up[:200] or re.search(r"^PART NUMBER:", up, re.M):
        return "3D model", "digital"
    form_hits = sum(bool(re.search(p, up)) for p in FORM_MARKERS)
    cells = [c.strip() for ln in (doc.lines or []) for c in ln.text.split("|")] + \
            [c.strip() for row in doc.rows for c in row.split("|")]
    cell_hits = sum(any(_label_score(c, lb, tail_ok=False) >= 0.9 for c in cells) for lb in DRAWING_CELLS)
    drawing_hits = sum(bool(re.search(p, up)) for p in DRAWING_PHRASES) + cell_hits
    if form_hits >= 3 and form_hits >= drawing_hits - 3:
        return "RFQ form", capture
    if drawing_hits >= 4:
        return "drawing", capture
    if form_hits >= 2:
        return "RFQ form", capture
    if doc.media in ("jpg", "png") or capture in ("photo", "screenshot"):
        kind = capture if capture in ("photo", "screenshot") else ("photo" if doc.media == "jpg" else "screenshot")
        return kind, capture
    return "other", capture


# ---- label and value helpers ----------------------------------------------- #
def _lab(text: str) -> str:
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", text.upper()).split())


def _label_score(text: str, label: str, tail_ok: bool = True) -> float:
    """How well a line is just this label (1.0), ends with it (0.8, a merged OCR line), or is a
    near miss ('MATERlAL', 'DWG N0')."""
    t, lb = _lab(text), _lab(label)
    if not t:
        return 0.0
    if t == lb:
        return 1.0
    if abs(len(t) - len(lb)) <= 2 and difflib.SequenceMatcher(None, t, lb).ratio() >= 0.8:
        return 0.9
    if tail_ok and t.endswith(" " + lb):
        return 0.7
    return 0.0


def _below(doc: Doc, label: Line, accept: Callable[[str], Optional[str]], max_rows: float = 4.0,
           stop: Optional[Callable[[str], bool]] = None) -> Optional[Tuple[str, Line]]:
    """The value printed under a label: the nearest line below it that starts near the label's
    left edge or overlaps it, on the same page."""
    if not label.box:
        return None
    lx0, ly0, lx1, ly1 = label.box
    h = max(label.h, 8.0)
    cands = []
    for ln in doc.lines:
        if ln is label or not ln.box or ln.page != label.page:
            continue
        x0, y0, x1, y1 = ln.box
        if y0 < ly0 + 0.35 * h or y0 > ly1 + max_rows * h:
            continue
        horiz = min(x1, lx1 + 4 * h) - max(x0, lx0 - 2 * h)
        if horiz <= 0 and abs(x0 - lx0) > 3 * h:
            continue
        cands.append((y0 - ly1 + 0.2 * abs(x0 - lx0), ln))
    for _, ln in sorted(cands, key=lambda c: c[0]):
        if stop and stop(ln.text):
            break
        val = accept(ln.text)
        if val:
            return val, ln
    return None


def _after(rows: List[str], i: int, accept: Callable[[str], Optional[str]], ahead: int = 3,
           stop: Optional[Callable[[str], bool]] = None) -> Optional[Tuple[str, int]]:
    for j in range(i + 1, min(len(rows), i + 1 + ahead)):
        if stop and stop(rows[j]):
            break
        val = accept(rows[j])
        if val:
            return val, j
    return None


def _labelled(doc: Doc, labels: Sequence[str], accept: Callable[[str], Optional[str]], ahead: int = 3,
              stop: Optional[Callable[[str], bool]] = None, tail_ok: bool = True,
              where: Optional[Dict[str, Any]] = None) -> Optional[Tuple[str, Optional[Line]]]:
    """Find a value by its label: on the same line ('MATERIAL: X'), under it (boxes), or after
    it in reading order. Whole-line labels are tried before merged-line tails. `where`, when
    given, is filled with how the value was found ("line" under a label, or reading-order "row"),
    so a caller can follow a value that wrapped onto the next line."""
    where = where if where is not None else {}
    for lb in labels:  # same line: "MATERIAL: ALUMINUM 6061-T6"
        pat = re.compile(r"(?:^|\|\s*)" + r"\s*".join(re.escape(w) for w in lb.split()) + r"\s*[:.]\s*(.+)$",
                         re.IGNORECASE)
        for ln in (doc.lines or []):
            m = pat.search(ln.text)
            if m and accept(m.group(1)):
                where["same_line"] = True
                return accept(m.group(1)), ln
        for row in doc.rows:
            m = pat.search(row)
            if m and accept(m.group(1)):
                where["same_line"] = True
                return accept(m.group(1)), None
    if doc.boxed:
        scored = []
        for ln in doc.lines:
            s = max(_label_score(ln.text, lb, tail_ok) for lb in labels)
            if s:
                scored.append((-s, ln.i, ln))
        for _, _, ln in sorted(scored, key=lambda x: (x[0], x[1])):
            got = _below(doc, ln, accept, stop=stop)
            if got:
                where["line"] = got[1]
                return got
    scored_rows = []
    for i, row in enumerate(doc.rows):
        cells = [c.strip() for c in row.split("|")]
        s = max(max(_label_score(c, lb, tail_ok) for lb in labels) for c in cells)
        if s:
            scored_rows.append((-s, i))
    for _, i in sorted(scored_rows):
        got = _after(doc.rows, i, accept, ahead, stop)
        if got:
            where["row"] = got[1]
            return got[0], None
    return None


def _tail_segment(text: str, pattern: "re.Pattern[str]") -> Optional[str]:
    """The value at the end of a line that OCR may have merged with a drawing note on its left:
    the last sentence that holds the pattern, through the end of the line."""
    text = clean(text.replace("|", "   ")).strip()
    text = re.sub(r"\s{2,}", " ", text)
    if not pattern.search(text):
        return None
    starts = [0] + [m.end() for m in re.finditer(r"(?<=[A-Za-z)])\.\s+(?=[A-Z0-9])", text)]
    for s in reversed(starts):
        seg = text[s:]
        first = re.split(r"(?<=[A-Za-z)])\.\s+(?=[A-Z0-9])", seg)[0]
        if pattern.search(first):
            return _strip_junk(seg)
    return _strip_junk(text)


def _strip_junk(text: str) -> str:
    """Drop OCR crumbs at the start of a value: stray lowercase bits, symbols, and item numbers."""
    words = text.split()
    while words and (re.fullmatch(r"[^A-Za-z0-9]+", words[0]) or re.fullmatch(r"[a-z]{1,3}", words[0])
                     or re.fullmatch(r"\d{1,2}[.)]", words[0])):
        words.pop(0)
    while words and re.fullmatch(r"[^A-Za-z0-9.)%]+|[a-z]{1,2}", words[-1]):
        words.pop()
    return " ".join(words).strip(" ,;:")


# ---- drawings ---------------------------------------------------------------- #
DRAWING_LABELS = ["TITLE", "MATERIAL", "FINISH", "SIZE", "DWG NO", "DWG NO.", "REV", "DRAWN", "DATE", "SCALE",
                  "SHEET", "REVISIONS", "NOTES", "THIRD ANGLE PROJECTION", "UNLESS OTHERWISE SPECIFIED"]


def _is_drawing_label(text: str) -> bool:
    return any(_label_score(c, lb, tail_ok=False) >= 0.9 for c in text.split("|") for lb in DRAWING_LABELS)


def _looks_like_date(token: str) -> bool:
    """'2025-06-17', 'Z0Z5-O6-17', '2026-10': a date, even with OCR look-alikes in the digits."""
    m = re.fullmatch(r"([0-9OQDIlSBZ|]{4})-([0-9OQDIlSBZ|]{2})(?:-([0-9OQDIlSBZ|]{2}))?", clean(token))
    if not m:
        return False
    y, mo = (int("".join(_TO_DIGIT.get(c, c) for c in g.upper())) for g in m.groups()[:2])
    d = int("".join(_TO_DIGIT.get(c, c) for c in m.group(3).upper())) if m.group(3) else 1
    return 1900 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= 31


def _pn_candidates(text: str, ocr: bool) -> List[str]:
    pat = PN_OCR_RE if ocr else PN_RE
    out = []
    for m in pat.finditer(text):
        tok = m.group(1)
        raw_prefix = tok.split("-")[0]
        # The look-alike repair below turns digits into letters, so a date has to be caught first:
        # "2025-06-17" in a title block would otherwise come out as the part number "ZOZS-06-17".
        # For the same reason the prefix needs one character that was printed as a letter.
        if _looks_like_date(tok) or (ocr and not re.search(r"[A-Za-z]", raw_prefix.replace("l", ""))):
            continue
        fixed = fix_part_number(tok) if ocr else tok.upper()
        prefix = fixed.split("-")[0]
        if prefix in SPEC_PREFIXES or not re.search(r"[A-Z]", prefix) or re.search(r"(?:^|-)RF[QO0P](?:-|$)", fixed):
            continue
        if re.fullmatch(r"\d{4}", fixed.split("-")[1]) and re.fullmatch(r"[1-7]\d{3}", fixed.split("-")[1]) and prefix in ("AL", "T"):
            continue
        if re.fullmatch(r"Q\d{0,2}", prefix) or re.fullmatch(r"20\d\d-\d\d(-\d\d)?", fixed):
            continue
        if not re.fullmatch(PN_BODY, fixed):
            continue
        out.append(fixed)
    return out


def _clean_rev(text: str) -> Optional[str]:
    t = clean(text).strip(" .,:;|()[]'\"*-_")
    if not t:
        return None
    t = t.split()[0] if len(t.split()) <= 2 else ""
    if re.fullmatch(r"[A-Za-z]{1,2}", t):
        up = t.upper()
        if len(up) == 2 and up[0] == up[1]:  # "Cc": one letter read twice
            up = up[0]
        return up
    if re.fullmatch(r"\d{1,2}", t):
        return t
    return None


def _material_value(text: str, cut: bool = True) -> Optional[str]:
    """A material callout. cut: the line may hold a neighbouring note too (OCR joins them), so keep
    the sentence with the material and what follows; a text layer line is the value as printed."""
    t = ocr_fix_spec(clean(text))
    if _is_drawing_label(t) and not MATERIAL_STRONG.search(t):
        return None
    if not (MATERIAL_STRONG.search(t) or MATERIAL_WORD.search(t)):
        return None
    if not cut and "|" not in t:
        return _strip_junk(t) or None
    seg = _tail_segment(t, re.compile(MATERIAL_STRONG.pattern + "|" + MATERIAL_WORD.pattern, re.IGNORECASE))
    return seg or None


def _finish_value(text: str, cut: bool = True) -> Optional[str]:
    t = ocr_fix_spec(clean(text))
    if re.fullmatch(r"(?:\|\s*)?NONE\.?(?:\s*\|)?", t.strip(), re.IGNORECASE):
        return "NONE"
    if _is_drawing_label(t) and not FINISH_RE.search(t):
        return None
    if not FINISH_RE.search(t):
        return None
    if not cut and "|" not in t:
        return _strip_junk(t) or None  # "NONE. FINAL POLISH AND PASSIVATION BY ALDERCREST" stays whole
    return _tail_segment(t, FINISH_RE) or None


def _plain_value(text: str) -> Optional[str]:
    """Any title block value from a text layer: words, not a label, a date, or a lone number."""
    t = _strip_junk(clean(text.replace("|", " ")))
    if len(re.findall(r"[A-Za-z]", t)) < 3 or _is_drawing_label(t) or _looks_like_date(t.split()[0]):
        return None
    if len(t.split()) == 1 and _pn_candidates(t, False):
        return None
    return t


def _title_value(text: str) -> Optional[str]:
    t = clean(text.replace("|", " "))
    t = _strip_junk(t)
    if not t or _is_drawing_label(t) or len(re.findall(r"[A-Z]", t)) < 4:
        return None
    if re.search(r"\b(?:INTERPRET|TOLERANCES|DIMENSIONS|ANGLES|UNLESS|THIRD ANGLE|BREAK SHARP|DO NOT SCALE)\b", t):
        # a merged line: the title is the tail after the tolerance block text
        m = re.search(r"([A-Z][A-Z0-9 ,.&/()'-]{5,})$", t)
        t = m.group(1).strip() if m else ""
    if sum(c.islower() for c in t) > len(t) * 0.3:
        return None
    return t or None


TITLE_BLOCK_LABELS = ["TITLE", "MATERIAL", "FINISH", "DWG NO", "DWG NO.", "SIZE", "DRAWN"]


def _title_block_labels(doc: Doc) -> List[Tuple[str, Line]]:
    """The title block labels OCR could read, as (label, line), on the page that has the most."""
    found = []
    for ln in doc.lines:
        if not ln.box:
            continue
        for lb in TITLE_BLOCK_LABELS:
            if _label_score(ln.text, lb, tail_ok=False) >= 0.9:
                found.append((lb.rstrip("."), ln))
                break
    if not found:
        return []
    pages: Dict[int, int] = {}
    for _, ln in found:
        pages[ln.page] = pages.get(ln.page, 0) + 1
    page = max(pages, key=lambda p: pages[p])
    return [(lb, ln) for lb, ln in found if ln.page == page]


def _title_block_region(doc: Doc) -> Optional[Tuple[int, float, float]]:
    """(page, left, top) of the title block: the box around its labels, with some slack. A number
    inside it that stands alone is the drawing number even when its DWG NO. label is unreadable."""
    labels = _title_block_labels(doc)
    if len(labels) < 2:
        return None
    h = sorted(ln.h for _, ln in labels)[len(labels) // 2] or 10.0
    left = min(ln.box[0] for _, ln in labels) - 3 * h
    top = min(ln.box[1] for _, ln in labels) - 3 * h
    return labels[0][1].page, left, top


def _title_block_column(doc: Doc) -> Optional[Dict[str, Any]]:
    """The left edge of the TITLE / MATERIAL / FINISH cells, from whichever of those labels were read."""
    labels = [(lb, ln) for lb, ln in _title_block_labels(doc) if lb in ("TITLE", "MATERIAL", "FINISH")]
    if not labels:
        return None
    xs = sorted(ln.box[0] for _, ln in labels)
    h = sorted(ln.h for _, ln in labels)[len(labels) // 2] or 10.0
    return {"page": labels[0][1].page, "x": xs[len(xs) // 2], "h": max(h, 8.0),
            "top": min(ln.box[1] for _, ln in labels), "labels": labels}


def _column_value(doc: Doc, column: Dict[str, Any], field: str, accept: Callable[[str], Optional[str]],
                  used: set) -> Optional[Tuple[str, Line]]:
    """A title block value whose label OCR lost: the first line in the TITLE / MATERIAL / FINISH
    column that reads like the field and does not sit under a label for some other field."""
    x, h = column["x"], column["h"]
    col = sorted((ln for ln in doc.lines if ln.box and ln.page == column["page"] and abs(ln.box[0] - x) <= 2.5 * h
                  and ln.box[1] >= column["top"] - h), key=lambda ln: ln.box[1])
    for k, ln in enumerate(col):
        val = accept(ln.text)
        if not val or val in used or _is_drawing_label(ln.text):
            continue
        above = next((p for p in reversed(col[:k]) if p.box[3] <= ln.box[1] + 0.3 * h), None)
        if above is not None and _is_drawing_label(above.text) and \
                _label_score(above.text, field.upper(), tail_ok=False) < 0.9:
            continue  # the value of the label above it, which is not this field
        return val, ln
    return None


def _more_value(field: str) -> Callable[[str], bool]:
    """Whether a line can be the second line of a title block value that wrapped in its cell:
    'BLACK ANODIZE PER MIL-A-8625 TYPE II CLASS 2. MASK' / 'PADS, BORES, AND DATUM A'."""
    def ok(text: str) -> bool:
        t = clean(text.replace("|", " ")).strip()
        letters = re.findall(r"[A-Za-z]", t)
        # a short spec word counts ('H1025', 'THK'); OCR crumbs and lowercase noise do not
        if not letters or len(re.findall(r"[A-Za-z0-9]", t)) < 3 or sum(c.islower() for c in letters) > len(letters) * 0.3:
            return False
        if _is_drawing_label(t) or _looks_like_date(t.split()[0]) or re.fullmatch(r"[A-Z0-9]{1,2}", t):
            return False
        if len(t.split()) <= 2 and _pn_candidates(t, True):
            return False  # the drawing number, not a wrapped value
        if field == "material" and (FINISH_RE.search(t) or re.match(r"NONE\b", t)):
            return False
        if field == "finish" and MATERIAL_STRONG.search(t) and not FINISH_RE.search(t):
            return False
        if field == "title" and (MATERIAL_STRONG.search(t) or FINISH_RE.search(t)):
            return False
        return True
    return ok


def _continue_value(doc: Doc, value: str, first: Optional[Line], row_index: Optional[int], field: str) -> str:
    """Add the lines a title block value wrapped onto. With boxes: lines right under the value,
    starting at its left edge, closer than a line apart. From reading order: the next rows,
    up to the next label."""
    more = _more_value(field)
    extra: List[str] = []
    src = first.text if first is not None else doc.rows[row_index] if row_index is not None else ""
    if len(clean(src.replace("|", " "))) > len(value) + 12:
        return value  # the value shares its line with another column's text; what is below is not its
    if first is not None and first.box:
        cur = first
        for _ in range(3):
            nxt = None
            for ln in doc.lines:
                if ln is cur or not ln.box or ln.page != cur.page:
                    continue
                # the text height, not the box: a speck read as a word makes a box taller
                h = max(min(cur.h, ln.h), 6.0)
                below = (ln.box[1] + ln.box[3]) / 2 > (cur.box[1] + cur.box[3]) / 2 + 0.5 * h
                gap = ln.box[1] - cur.box[3]
                if below and gap <= 1.0 * h and abs(ln.box[0] - first.box[0]) <= 1.5 * h \
                        and (nxt is None or ln.box[1] < nxt.box[1]):
                    nxt = ln
            if nxt is None or (nxt.conf is not None and nxt.conf < 50) or not more(nxt.text):
                break
            extra.append(_strip_junk(clean(nxt.text.replace("|", " "))))
            cur = nxt
    elif row_index is not None:
        for j in range(row_index + 1, min(len(doc.rows), row_index + 3)):
            if not more(doc.rows[j]):
                break
            extra.append(_strip_junk(clean(doc.rows[j].replace("|", " "))))
    extra = [ocr_fix_spec(e) if field in ("material", "finish") else e for e in extra if e]
    if not extra:
        return value
    if clean(src).rstrip().endswith(":") and not value.endswith(":"):
        value += ":"  # "MARKERS:" lost its colon when the single-line value was trimmed
    return " ".join([value] + extra)


def parse_drawing(doc: Doc, hint_pns: Sequence[str] = ()) -> Dict[str, Any]:
    """Title block and legend values from a drawing's text."""
    out: Dict[str, Any] = {"part_number": None, "rev": None, "title": None, "material": None, "finish": None,
                           "company": None, "revisions": [], "conf": {}}

    def keep(field: str, got: Optional[Tuple[str, Optional[Line]]]) -> None:
        if got and got[0]:
            out[field] = got[0]
            ln = got[1]
            out["conf"][field] = ln.conf if ln is not None and ln.conf is not None else doc.conf

    stop = _is_drawing_label
    cut = doc.ocr
    for field, labels, accept, ahead in (("title", ["TITLE"], _title_value, 2),
                                         ("material", ["MATERIAL", "MATL", "MATERIAL SPEC"],
                                          lambda t: _material_value(t, cut), 3),
                                         ("finish", ["FINISH", "FINISH SPEC"], lambda t: _finish_value(t, cut), 3)):
        where: Dict[str, Any] = {}
        got = _labelled(doc, labels, accept, ahead=ahead, stop=stop, where=where)
        if not got and not doc.ocr and field != "title":
            # A typed title block says what it says: 'SEE PARTS LIST', '1045 INDUCTION HARDENED ...
            # BAR', 'AS MACHINED' are the values even though they read like no known material or finish.
            # OCR text keeps the stricter test, where the line under a label can be noise.
            where = {}
            got = _labelled(doc, labels, _plain_value, ahead=1, stop=stop, tail_ok=False, where=where)
        if got and not where.get("same_line") and got[0] != "NONE":
            # a long callout wraps inside its cell; the second line belongs to the value
            got = (_continue_value(doc, got[0], where.get("line"), where.get("row"), field), got[1])
        keep(field, got)

    # A label the scan lost ('MATERIAL' unreadable) still leaves its value in the title block column.
    column = _title_block_column(doc) if doc.boxed else None
    if column:
        used = {out.get("title"), out.get("finish"), out.get("material")}
        for field, accept in (("material", _material_value), ("finish", _finish_value)):
            if not out[field]:
                got = _column_value(doc, column, field, accept, used)
                if got:
                    got = (_continue_value(doc, got[0], got[1], None, field), got[1])
                    keep(field, got)
                    used.add(got[0])

    # Part number: only from a place where a part number is printed. That is the DWG NO. cell (the
    # value under or after the label), a P/N label, the size / number / rev row a title block reads
    # as, a short line inside the title block, or a number the email, RFQ form, or model names. A
    # number anywhere else on the sheet (a date, a note, a view callout) never becomes a part line.
    scores: Dict[str, float] = {}
    spelled: Dict[str, str] = {}
    confs: Dict[str, Optional[float]] = {}
    ctx: Dict[str, set] = {}
    hint_keys = {fold(p): p for p in hint_pns}
    block = _title_block_region(doc) if doc.boxed else None

    def cand(pn: str, bonus: float, conf: Optional[float], where: Optional[str] = None) -> None:
        key = fold(pn)
        scores[key] = scores.get(key, 0.0) + bonus
        spelled.setdefault(key, pn)
        ctx.setdefault(key, set())
        if where:
            ctx[key].add(where)
        if conf is not None and key not in confs:
            confs[key] = conf

    tb_row = re.compile(r"^\W*(?:[A-E]\s*\|?\s*)?(\S+)\s*\|?\s*(?:[A-Z0-9]{1,2})?\W*$")
    for ln in (doc.lines or []):
        n_words = len(ln.text.split())
        for pn in _pn_candidates(ln.text, doc.ocr):
            where = None
            if n_words <= 3 and block and ln.box and ln.page == block[0] and ln.box[0] >= block[1] \
                    and ln.box[1] >= block[2]:
                where = "title block"
            cand(pn, 1.0 + (0.5 if n_words <= 3 else 0.0) - (1.5 if n_words > 8 else 0.0), ln.conf, where)
    for row in doc.rows:
        n_words = len(row.replace("|", " ").split())
        m = tb_row.match(row.replace("|", " | "))
        for pn in _pn_candidates(row, doc.ocr):
            where = "title block row" if m and n_words <= 3 and _pn_candidates(m.group(1), doc.ocr) else None
            cand(pn, (0.0 if doc.lines else 1.0) + (0.5 if n_words <= 3 else 0.0), None, where)
    got = _labelled(doc, ["DWG NO", "DWG NO.", "DRAWING NO", "DWG", "PART NO", "PART NUMBER", "P/N"],
                    lambda t: (_pn_candidates(t, doc.ocr) or [None])[0], ahead=3, tail_ok=True)
    if got:
        cand(got[0], 4.0, got[1].conf if got[1] is not None else None, "DWG NO")
    # the label and its number inside a longer line: "TITLE BLOCK: ... DWG NO: TIB-0725 REV B."
    for row in doc.rows:
        for m in re.finditer(r"\b(?:[DO0]WG|DRAWING)\s*(?:N[O0]|NUMBER|#)\.?\s*[:#]?\s*|\b(?:PART\s*(?:NO|NUMBER)|P/N)\.?\s*[:#]\s*",
                             row, re.IGNORECASE):
            after = row[m.end():].split()
            pns = _pn_candidates(after[0].strip(".,;|"), doc.ocr) if after else []
            if pns:
                cand(pns[0], 4.0, None, "DWG NO")
    for key in list(scores):
        for hk, spelled_hint in hint_keys.items():
            # the same number, or one is a dash number of the other (drawing BWM-3140, form BWM-3140-08)
            if key == hk or _variant_of(spelled[key], spelled_hint):
                scores[key] += 2.5
                ctx[key].add("named")
    if block and not any(ctx.get(k) for k in scores):
        # OCR broke the number's dash ('HPV. 2045'); a short title block line that folds to a number
        # another source names is that number
        for ln in doc.lines:
            if ln.box and len(ln.text.split()) <= 3 and ln.page == block[0] and ln.box[0] >= block[1] \
                    and ln.box[1] >= block[2]:
                for hk, spelled_hint in hint_keys.items():
                    if len(hk) >= 5 and fold(ln.text) == hk:
                        cand(spelled_hint, 3.0, ln.conf, "named")
    placed = [k for k in scores if ctx.get(k)]
    if placed:
        best = max(placed, key=lambda k: (scores[k], -len(k)))
        out["part_number"] = spelled[best]
        out["conf"]["part_number"] = confs.get(best, doc.conf)
        out["pn_context"] = sorted(ctx[best])

    # Rev: under the REV label next to the drawing number, or right after the part number on a
    # merged title block row ("A CI-10442 C"), else the newest row of the revision table.
    pn = out["part_number"]
    rev = None
    if pn:
        key = fold(pn)
        for row in doc.rows:
            toks = row.replace("|", " ").split()
            for k, tok in enumerate(toks[:-1]):
                if _pn_candidates(tok, doc.ocr) and fold(fix_part_number(tok) if doc.ocr else tok) == key:
                    r = _clean_rev(toks[k + 1])
                    if r and len(toks) - k <= 3:
                        rev = (r, None)
                    elif k + 2 < len(toks) and re.fullmatch(r"REV\.?", toks[k + 1].upper()):
                        r = _clean_rev(toks[k + 2])  # "DWG NO: TIB-0725 REV B."
                        rev = (r, None) if r else rev
        if doc.boxed and not rev:
            for ln in doc.lines:
                if _label_score(ln.text, "REV", tail_ok=True) >= 0.7 and len(ln.text) <= 12:
                    got = _below(doc, ln, _clean_rev, max_rows=3.5)
                    pn_line = next((p for p in doc.lines if p.box and fold(" ".join(_pn_candidates(p.text, doc.ocr))) == key), None)
                    if got and pn_line and abs(got[1].box[1] - pn_line.box[1]) < 3 * max(pn_line.h, 10):
                        rev = got
                        break
        if not rev:
            idx = [i for i, row in enumerate(doc.rows) if _label_score(row, "REV", tail_ok=False) >= 0.9]
            dwg = [i for i, row in enumerate(doc.rows) if re.match(r"^[DO0]WG\.?\s*N", row.upper())]
            for i in idx:
                if dwg and 0 < i - dwg[0] <= 3:
                    got = _after(doc.rows, i, _clean_rev, ahead=1)
                    if got:
                        rev = (got[0], None)
        if doc.boxed and not rev:
            # the REV cell ends the size / drawing number / rev row: the first letter to the right of
            # the number on the same baseline, even when both labels above it are unreadable
            pn_line = next((ln for ln in doc.lines if ln.box and len(ln.text.split()) <= 3 and any(
                fold(c) == key for c in _pn_candidates(ln.text, doc.ocr))), None)
            if pn_line is not None:
                right = sorted((ln for ln in doc.lines if ln.box and ln.page == pn_line.page
                                and ln.box[0] > pn_line.box[2]
                                and min(ln.box[3], pn_line.box[3]) - max(ln.box[1], pn_line.box[1])
                                >= 0.5 * min(ln.h, pn_line.h) and (ln.conf is None or ln.conf >= 50)),
                               key=lambda ln: ln.box[0])
                for ln in right:
                    t = ln.text.strip(" |_-.:;'\"")
                    if re.fullmatch(r"[A-Z]{1,2}|\d{1,2}", t) and _clean_rev(t):
                        rev = (_clean_rev(t), ln)
                        break
    # revision table rows: "B ADDED KEYWAY EDGE BREAK NOTE 2025-08-04 TW"
    revs = []
    for row in doc.rows:
        m = re.match(r"^\|?\s*([A-Z]{1,2})\s+(?:\|\s*)?[A-Z].*\b20\d\d-\d\d-\d\d\b", row)
        if m and not re.match(r"^(?:REV|SIZE|DWG|DRAWN)\b", row):
            revs.append(m.group(1))
    out["revisions"] = revs
    if rev:
        out["rev"] = rev[0]
        ln = rev[1] if len(rev) > 1 else None
        out["conf"]["rev"] = ln.conf if isinstance(ln, Line) and ln.conf is not None else doc.conf
    elif revs:
        out["rev"] = max(revs)
        out["conf"]["rev"] = doc.conf
    if revs and out["rev"] and out["rev"] not in revs and out["rev"].isdigit():
        out["rev"] = max(revs)  # a digit where the revision table has letters is an OCR slip (8 for B)

    # company: the line above TITLE in the title block
    for i, row in enumerate(doc.rows):
        if _label_score(row.split("|")[-1], "TITLE", tail_ok=False) >= 0.9 and i:
            prev = _strip_junk(doc.rows[i - 1].split("|")[-1])
            if re.fullmatch(r"[A-Z][A-Z&.,' -]{3,}", prev or "") and not _is_drawing_label(prev):
                out["company"] = prev.title()
            break
    out["export"] = _find_export(doc.text, doc.ocr)
    return out


# ---- STEP -------------------------------------------------------------------- #
def parse_step(doc: Doc) -> Dict[str, Any]:
    t = doc.text
    out: Dict[str, Any] = {"part_number": None, "title": None, "rev": None, "units": None, "size": None}

    def grab(pattern: str) -> Optional[str]:
        m = re.search(pattern, t, re.IGNORECASE | re.MULTILINE)
        return clean(m.group(1)) if m and clean(m.group(1)) else None

    out["part_number"] = grab(r"^\s*(?:PART NUMBER|P/N|PART)\s*[:=]\s*([A-Z0-9][A-Z0-9-]*)")
    m = re.search(r"PRODUCT\s*\(\s*'([^']*)'\s*,\s*'([^']*)'(?:\s*,\s*'([^']*)')?", t)
    if m:
        out["part_number"] = out["part_number"] or clean(m.group(1))
        out["title"] = clean(m.group(2)) or None
        r = re.match(r"REV\.?\s*(\S+)", clean(m.group(3) or ""), re.IGNORECASE)
        out["rev"] = r.group(1) if r else None
    out["title"] = grab(r"^\s*TITLE\s*[:=]\s*(.+)$") or out["title"]
    out["rev"] = grab(r"^\s*(?:REVISION|REV)\s*[:=]\s*(\S+)") or out["rev"]
    if not out["rev"]:
        out["rev"] = grab(r"^\s*PRODUCT DESCRIPTION\s*[:=]\s*REV\.?\s*(\S+)")
    if not out["title"]:
        d = grab(r"^\s*DESCRIPTION\s*[:=]\s*([^;\n]+)")
        out["title"] = d
    out["units"] = grab(r"^\s*UNITS?\s*[:=]\s*(\w+)")
    out["size"] = grab(r"^\s*(?:BOUNDING BOX|SIZE|OVERALL SIZE)\s*[:=]\s*(.+)$")
    if out["rev"] in ("-", "", None):
        out["rev"] = None
    if out["part_number"]:
        out["part_number"] = out["part_number"].upper()
    return out


# ---- RFQ forms --------------------------------------------------------------- #
FORM_COLUMNS = [
    ("item", r"^(?:ITEM|LINE|NO\.?)$"),
    ("pn", r"PART\s*(?:NUMBER|NO\.?|#)|^P/?N$|PARTNUMBER"),
    ("rev", r"^REV\.?$"),
    ("desc", r"DESCRIPTI[O0]N|^DESC\.?$"),
    ("matfin", r"MATERIAL\s*[/|Il1]\s*FINISH"),
    ("material", r"^MATERIAL$|^MAT'?L$"),
    ("finish", r"^FINISH$"),
    ("qty", r"QUANTIT|^QTY"),
    ("price", r"UNIT\s*PRICE|^PRICE$"),
    ("lead", r"LEAD\s*TIME"),
]


def _col_of(text: str) -> Optional[str]:
    t = clean(text).upper().strip(" .:")
    for key, pat in FORM_COLUMNS:
        if re.search(pat, t):
            return key
    return None


def parse_quantities(text: str) -> Tuple[Optional[List[int]], bool]:
    """A quantity cell ('25 / 75 / 150', '250 / 500 / 1,000', '25, 50, 100') -> ([ints], complete).
    complete is False when OCR left a dangling separator or an unreadable piece."""
    t = clean(text).replace("|", " ")
    t = re.sub(r"\bPCS?\b|\bPIECES\b|\bEA\b", " ", t, flags=re.IGNORECASE)
    if "/" in t:
        parts = [p.strip() for p in t.split("/")]
    elif re.search(r"\d{1,3}(?:,\d{3})+", t) and not re.search(r"\d,\s", t):
        parts = t.split()
    else:
        parts = re.split(r"[,;]|\s+AND\s+|\s+", t, flags=re.IGNORECASE)
    nums, complete = [], True
    for p in parts:
        p = p.strip(" .*'\"`~-_")
        if not p:
            if nums:
                complete = False
            continue
        n = _num(p)
        if n is None or n == 0:
            complete = False
            continue
        nums.append(n)
    if t.rstrip(" *'\"`~_-").endswith("/"):
        complete = False
    return (nums or None), complete


def _split_matfin(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = clean(text).strip(" /")
    if not t:
        return None, None
    pieces = [p.strip() for p in re.split(r"\s+/\s+|\s/|/\s", t) if p.strip()]
    if len(pieces) >= 2:
        for k in range(1, len(pieces)):
            right = " / ".join(pieces[k:])
            if FINISH_RE.search(pieces[k]) or re.match(r"NONE\b", pieces[k], re.IGNORECASE):
                return ocr_fix_spec(" / ".join(pieces[:k])), ocr_fix_spec(right)
        return ocr_fix_spec(" / ".join(pieces[:-1])), ocr_fix_spec(pieces[-1])
    if FINISH_RE.search(t) and not (MATERIAL_STRONG.search(t) or MATERIAL_WORD.search(t)):
        return None, ocr_fix_spec(t)
    return ocr_fix_spec(t), None


_MATFIN_ANCHOR = re.compile(r"\b(?:AL|SST|ALUMINUM|STAINLESS|STEEL|BRASS|PEEK|TITANIUM|DELRIN|ACETAL|NYLON|"
                            r"(?:17-4|15-5)\s?PH|[1-7]\d{3}-T\d|ASTM|AMS|MIL-|ANODIZE|PASSIVATE|NONE)\b", re.IGNORECASE)


def _split_desc_matfin(text: str) -> Tuple[str, str]:
    """A merged table row fragment: description words come first (left column), then material."""
    m = _MATFIN_ANCHOR.search(text)
    if not m:
        if SPEC_WORDS.search(text.upper()) or re.match(r"^\s*[\d.]+\b", text) or "/" in text:
            return "", text
        return text, ""
    before = text[:m.start()].strip(" ,")
    if "/" in before or re.match(r"^[A-Z]?\d", before) or SPEC_WORDS.search(before.upper()):
        return "", text.strip()  # "H1025 / PASSIVATE PER": all of it continues the material cell
    return before, text[m.start():].strip()


class _FormTable:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []


def _deskew_slope(lines: List[Line]) -> float:
    """dy/dx from label boxes that were printed on one baseline (the table header)."""
    pts = [((ln.box[0] + ln.box[2]) / 2, (ln.box[1] + ln.box[3]) / 2) for ln in lines if ln.box]
    if len(pts) < 3:
        return 0.0
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    if not den:
        return 0.0
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / den
    return slope if abs(slope) < 0.06 else 0.0


QTY_LIST_RE = re.compile(r"(?:\d[\d,OIl]*\s*/\s*)+\d[\d,OIl]*\s*/?\*?|\d[\d,]*\s*/\s*\*?$")


def _row_piece(text: str, ocr: bool) -> Dict[str, Any]:
    """Split a table row fragment that OCR read across several columns, by what the words are:
    item number, part number, rev, quantity list, then description words before material words."""
    out: Dict[str, Any] = {"pn": None, "rev": None, "qty": None, "desc": "", "matfin": "", "tail_number": None}
    t = clean(text.replace("|", " "))
    m = (PN_OCR_RE if ocr else PN_RE).search(t)
    if m and _pn_candidates(m.group(1), ocr):
        out["pn"] = _pn_candidates(m.group(1), ocr)[0]
        after = t[m.end():].strip(" .,")
        toks = after.split()
        if toks and len(toks[0].strip(".,")) <= 2 and _clean_rev(toks[0]) and not toks[0].isdigit():
            out["rev"] = _clean_rev(toks[0])
            after = after[len(toks[0]):]
        t = after
    q = QTY_LIST_RE.search(t)
    if q:
        out["qty"] = q.group(0)
        t = (t[:q.start()] + " " + t[q.end():]).strip()
    m = re.search(r"(?:^|\s)(\d{1,3}(?:,\d{3})+|\d{2,6})\s*$", t)
    if m and not re.search(r"(?:ASTM|AMS|MIL|TYPE|CLASS|METHOD|GRADE|COND|F|H)\s*$", t[:m.start()].upper()):
        out["tail_number"] = m.group(1)
        t = t[:m.start()].strip()
    t = re.sub(r"^\s*[:;.,_*\-]*\s*\d{1,2}\s+(?=[A-Z])", "", t)  # a leading item number
    d, mf = _split_desc_matfin(_strip_junk(t))
    out["desc"], out["matfin"] = d, mf
    return out


def _form_rows_boxed(doc: Doc) -> Optional[List[Dict[str, Any]]]:
    """Table rows from OCR lines with boxes: find the header labels, give every line below them a
    column by where it sits and a row by the part number cell it lines up with (after taking out
    the page skew measured on the header). A line that runs across several columns is split by
    what its words are, not by guessing where each word sits."""
    lines = [ln for ln in doc.lines if ln.box]
    header: Dict[str, Line] = {}
    for ln in lines:
        key = _col_of(ln.text)
        if key and key not in header and len(ln.text) < 40:
            header[key] = ln
    merged_header = None
    if "pn" not in header or "qty" not in header:
        for ln in lines:
            t = ln.text.upper()
            if re.search(r"PART\s*(?:NUMBER|NO)", t) and re.search(r"QUANTIT|QTY", t):
                merged_header = ln
                break
        if not merged_header:
            return None
        header = {}
        x0, _, x1, _ = merged_header.box
        text = merged_header.text
        for key, pat in FORM_COLUMNS:
            m = re.search(pat.replace("^", r"\b").replace("$", r"\b"), text.upper())
            if m:
                cx0 = x0 + (x1 - x0) * m.start() / max(1, len(text))
                cx1 = x0 + (x1 - x0) * m.end() / max(1, len(text))
                header[key] = Line(m.group(0), merged_header.conf, merged_header.page,
                                   (cx0, merged_header.box[1], cx1, merged_header.box[3]), -1)
        if "pn" not in header:
            return None
    slope = _deskew_slope(list(header.values())) if not merged_header else 0.0
    page = header["pn"].page
    x_ref = header["pn"].box[0]

    def dy(ln: Line) -> float:
        return ln.box[1] - slope * (ln.box[0] - x_ref)

    head_y = max(dy(h) for h in header.values())
    head_bottom = max(h.box[3] - slope * (h.box[0] - x_ref) for h in header.values())
    end_y = float("inf")
    for ln in lines:
        if ln.page == page and dy(ln) > head_y and re.search(
                r"REQUIREMENTS|^\W*TERMS\b|SUPPLIER\s+RESP|^NOTES\b", ln.text.upper()):
            end_y = min(end_y, dy(ln))
    body = [ln for ln in lines if ln.page == page and dy(ln) > head_bottom - 2 and dy(ln) < end_y - 2
            and ln not in header.values() and ln is not merged_header]
    if not body:
        return None
    heights = sorted(ln.h for ln in body if ln.h)
    unit = heights[len(heights) // 2] if heights else 30.0
    # a merged header line is taller than its text by the skew across its width
    skew_mag = 0.0
    if merged_header:
        skew_mag = max(0.0, (merged_header.h - unit) / max(1.0, merged_header.box[2] - merged_header.box[0]))
    cols = sorted(header.items(), key=lambda kv: kv[1].box[0])
    margin = 0.4 * unit
    spans = []
    for k, (key, ln) in enumerate(cols):
        left = ln.box[0] - margin
        right = cols[k + 1][1].box[0] - margin if k + 1 < len(cols) else float("inf")
        spans.append((key, left, right))

    def column_of(ln: Line) -> Optional[str]:
        x0, _, x1, _ = ln.box
        width = max(1.0, x1 - x0)
        hits = [(key, max(0.0, min(x1, r) - max(x0, l))) for key, l, r in spans]
        hits = [h for h in hits if h[1] > 0]
        if not hits:
            return spans[0][0] if x1 < spans[0][1] + margin else None
        best = max(hits, key=lambda h: h[1])
        return best[0] if best[1] >= 0.85 * width else None

    pieces = []  # (y, x, line, column or None for a line across columns, text)
    for ln in body:
        text = ln.text.strip(" |")
        if text.strip(" |.,'`*_:;-"):
            pieces.append((dy(ln), ln.box[0], ln, column_of(ln), text))
    anchors = []
    for y, x, ln, col, text in pieces:
        pn = None
        if col == "pn" and _pn_candidates(text, doc.ocr):
            pn = _pn_candidates(text, doc.ocr)[0]
        elif col is None:
            pn = _row_piece(text, doc.ocr)["pn"]
        if pn and not any(fold(a[1]) == fold(pn) and abs(a[0] - y) < 2 * unit for a in anchors):
            anchors.append((y, pn, ln))
    if not anchors:
        return None
    anchors.sort(key=lambda a: a[0])
    rows = [{"pn": a[1], "cells": {}, "conf": a[2].conf, "qty": [], "qty_conf": None} for a in anchors]
    for y, x, ln, col, text in sorted(pieces, key=lambda p: (p[0], p[1])):
        cx = (ln.box[0] + ln.box[2]) / 2

        def reach(a: Tuple[float, str, Line]) -> float:
            return 0.5 * unit + skew_mag * abs(cx - (a[2].box[0] + a[2].box[2]) / 2)

        # A piece that shares height with a part number cell is in that row (a cell centred in a
        # tall row starts above its part number); anything else continues the row above it.
        over = [(min(y + ln.h, a[0] + a[2].h) - max(y, a[0]), i) for i, a in enumerate(anchors)]
        best = max(over)
        if best[0] > 0.25 * min(ln.h or unit, anchors[best[1]][2].h or unit):
            k = best[1]
        else:
            k = max((i for i, a in enumerate(anchors) if a[0] - reach(a) <= y), default=None)
        if k is None:
            continue
        row = rows[k]
        cells = row["cells"]
        if col is not None:
            if col == "qty":
                if not re.search(r"\d", text):
                    continue  # a speck in the quantity cell ("E" at 0% confidence) is not a break
                row["qty"].append(text)
                row["qty_conf"] = min(row["qty_conf"] or 100.0, ln.conf if ln.conf is not None else 100.0)
            elif col != "pn" or not _pn_candidates(text, doc.ocr) or fold(_pn_candidates(text, doc.ocr)[0]) != fold(row["pn"]):
                cells.setdefault(col, []).append(text)
            elif col == "pn":
                cells.setdefault("pn", []).append(text)
            continue
        piece = _row_piece(text, doc.ocr)
        if piece["rev"] and not cells.get("rev"):
            cells["rev"] = [piece["rev"]]
        if piece["qty"]:
            row["qty"].append(piece["qty"])
            row["qty_conf"] = min(row["qty_conf"] or 100.0, ln.conf if ln.conf is not None else 100.0)
        if piece["tail_number"]:
            row["qty"].append(piece["tail_number"])
        if piece["desc"]:
            cells.setdefault("desc", []).append(piece["desc"])
        if piece["matfin"]:
            cells.setdefault("matfin" if "matfin" in header or "material" not in header else "material", []).append(
                piece["matfin"])
    out = []
    for r in rows:
        c = {k: " ".join(v) for k, v in r["cells"].items()}
        material, finish = (c.get("material"), c.get("finish"))
        if "matfin" in c:
            material, finish = _split_matfin(c["matfin"])
        rev = _clean_rev(c.get("rev", "")) if c.get("rev") else None
        if not rev and c.get("pn"):
            after = c["pn"].split()
            idx = next((i for i, tok in enumerate(after) if _pn_candidates(tok, doc.ocr)
                        and fold(_pn_candidates(tok, doc.ocr)[0]) == fold(r["pn"])), None)
            if idx is not None and idx + 1 < len(after):
                rev = _clean_rev(after[idx + 1])
        qty, complete = parse_quantities(" / ".join(x.strip(" /*'\"`~_.:;") for x in r["qty"])) if r["qty"] \
            else (None, True)
        if r["qty"] and r["qty"][0].rstrip(" *").endswith("/") and len(r["qty"]) == 1:
            complete = False
        desc = _strip_junk(c.get("desc", "")) or None
        out.append({"part_number": r["pn"], "rev": rev, "description": desc, "material": material,
                    "finish": finish, "quantities": qty, "qty_complete": complete,
                    "conf": r["qty_conf"] if r["qty_conf"] is not None else r["conf"]})
    return out


def _form_rows_text(doc: Doc) -> List[Dict[str, Any]]:
    """Table rows from reading-order text (a text layer, or OCR without boxes)."""
    rows = doc.rows
    start = next((i for i, r in enumerate(rows) if re.search(r"PART\s*(?:NUMBER|NO)|DESCRIPTI", r.upper())), None)
    if start is None:
        start = 0
    end = next((i for i in range(start + 1, len(rows)) if re.search(r"REQUIREMENTS|^TERMS\b|SUPPLIER\s+RESP",
                                                                      rows[i].upper())), len(rows))
    header_end = start
    while header_end + 1 < end and _col_of(rows[header_end + 1]) and not _pn_candidates(rows[header_end + 1], doc.ocr):
        header_end += 1
    region = rows[header_end + 1:end]
    groups: List[List[str]] = []
    for row in region:
        if _pn_candidates(row, doc.ocr) and not re.search(r"\bRFQ\b", row.upper()):
            groups.append([row])
        elif groups:
            groups[-1].append(row)
    out = []
    for g in groups:
        first = g[0]
        pn = _pn_candidates(first, doc.ocr)[0]
        m = (PN_OCR_RE if doc.ocr else PN_RE).search(first)
        rest_first = first[m.end():] if m else first
        rev = None
        toks = rest_first.replace("|", " ").split()
        if toks and _clean_rev(toks[0]) and len(toks[0]) <= 2:
            rev = _clean_rev(toks[0])
            rest_first = rest_first.split(toks[0], 1)[1]
        parts = [rest_first] + g[1:]
        qty, complete, desc, matfin = None, True, [], []
        for p in parts:
            p = p.replace("|", " ")
            q = re.search(r"(?:\d[\d,OIl]*\s*/\s*)+\d[\d,OIl]*\s*/?|\d[\d,]*\s*/\s*$", p)
            if q and qty is None:
                qty, complete = parse_quantities(q.group(0))
                p = (p[:q.start()] + " " + p[q.end():]).strip()
            elif qty is not None and not complete and re.fullmatch(r"\s*\d[\d,]*\s*", p):
                more, _ = parse_quantities(p)
                if more:
                    qty, complete = qty + more, True
                    continue
            elif re.fullmatch(r"\s*\d[\d,]*\s*", p) and qty is None and len(parts) > 1:
                qty, complete = parse_quantities(p)
                continue
            d, mf = _split_desc_matfin(p.strip())
            if d:
                desc.append(d)
            if mf:
                matfin.append(mf)
        material, finish = _split_matfin(" ".join(matfin)) if matfin else (None, None)
        out.append({"part_number": pn, "rev": rev, "description": _strip_junk(" ".join(desc)) or None,
                    "material": material, "finish": finish, "quantities": qty, "qty_complete": complete,
                    "conf": doc.conf})
    return out


HEADER_LABELS = ["RFQ NO", "RFQ NUMBER", "DATE", "RESPOND BY", "RESPONSE DUE", "QUOTE DUE", "DUE DATE", "BUYER",
                 "PAGE", "REVISION", "PO NUMBER"]


def _cell_value(rows: List[str], labels: Sequence[str], accept: Callable[[str], Optional[str]]) -> Optional[Tuple[str, None]]:
    """A label in a row of label cells ('RFQ NO.  DATE  RESPOND BY') and its value in the same
    position of the next row, or right after the label on its own line. Returns None when the
    value in that position is unreadable, rather than borrowing a neighbour's value."""
    for i, row in enumerate(rows):
        up = _lab(row)
        hits = []
        for lb in set(HEADER_LABELS) | set(labels):
            for m in re.finditer(r"(?:^| )" + re.escape(_lab(lb)) + r"(?= |$)", up):
                hits.append((m.start(), m.end(), lb))
        # keep the longest label at each position ("RESPOND BY", not "BY")
        hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
        kept = []
        for h in hits:
            if not kept or h[0] >= kept[-1][1]:
                kept.append(h)
        target = next((k for k, h in enumerate(kept) if h[2] in labels), None)
        if target is None:
            continue
        tail = up[kept[target][1]:].strip()
        if len(kept) == 1 and tail and accept(row[len(row) - len(tail):] if len(tail) < len(row) else row):
            val = accept(row.split(kept[target][2].split()[-1], 1)[-1]) if kept[target][2].split()[-1] in row.upper() \
                else None
            if val:
                return val, None
        if i + 1 >= len(rows):
            return None
        nxt = rows[i + 1]
        if len(kept) == 1:
            val = accept(nxt)
            return (val, None) if val else None
        cells = [c.strip() for c in nxt.split("|") if c.strip()]
        if len(cells) != len(kept):
            cells = nxt.split()
        if len(cells) == len(kept):
            val = accept(cells[target])
            return (val, None) if val else None
        return None
    return None


REQ_HEADING = re.compile(r"REQUIREMENTS|QUALITY\s+CLAUSES|SPECIAL\s+INSTRUCTIONS")
REQ_END = re.compile(r"^\W*TERMS\b|SUPPLIER\s+RESP|QUOTED\s+BY|^\W*PAGE\s+\d|^\W*SIGNATURE")
ITEM_NO = re.compile(r"^\W{0,2}\d{1,2}\s*(?:[.,)]+\W*|\|)\s*")


def _req_text(text: str) -> str:
    """One requirement without its item number and the OCR crumbs around it."""
    t = ITEM_NO.sub("", clean(text).strip())
    t = clean(t.replace("|", " "))
    t = re.sub(r"\bC\s?OF\s?C\b", "C OF C", t)  # OCR drops the space: "C OFC"
    words = t.split()
    while words and (re.fullmatch(r"[^A-Za-z0-9]+", words[-1]) or re.fullmatch(r"[a-z]{1,2}", words[-1])):
        words.pop()  # "LUBRICANTS . a", "ON FIRST LOT \u00b0"
    return _strip_junk(" ".join(words)).strip(" -_.")


def _is_req(text: str) -> bool:
    words = re.findall(r"[A-Za-z]{3,}", text)
    return len(words) >= 2 and sum(w.isupper() for w in words) >= len(words) / 2 or \
        (len(words) >= 3 and not re.search(r"[a-z]{2,}[A-Z]", text))


def _form_requirements(doc: Doc) -> List[str]:
    """The numbered quote requirements under their heading, until the terms or the response box.
    With OCR boxes, one item is one line of text in the item column; a line continues the item
    above only when that one ran to the right edge of the block without a period (it wrapped).
    A tall thin 'line' is the column of item numbers read as one word, and is dropped."""
    items: List[str] = []
    lines = [ln for ln in doc.lines if ln.box]
    head = next((ln for ln in lines if REQ_HEADING.search(ln.text.upper()) and len(ln.text) < 60), None)
    if head is not None:
        end_y = min((ln.box[1] for ln in lines if ln.page == head.page and ln.box[1] > head.box[3]
                     and REQ_END.search(ln.text.upper())), default=float("inf"))
        block = [ln for ln in lines if ln.page == head.page and head.box[3] - 0.3 * head.h < ln.box[1] < end_y - 2
                 and ln is not head]
        heights = sorted(ln.h for ln in block if ln.h)
        unit = heights[len(heights) // 2] if heights else 20.0
        block = [ln for ln in block if ln.h <= 2.2 * unit and (ln.conf is None or ln.conf >= 40 or len(ln.text) > 12)]
        rows: List[List[Line]] = []
        for ln in sorted(block, key=lambda l: (l.box[1], l.box[0])):
            if rows and abs(ln.box[1] - rows[-1][0].box[1]) < 0.5 * unit:
                rows[-1].append(ln)
            else:
                rows.append([ln])
        # A wrapped item ran into the right edge of the printed area; measure that edge on the whole
        # page (the header and table reach it), not on the requirement lines, which may all be short.
        left = min((ln.box[0] for ln in block), default=0.0)
        page_right = max((ln.box[2] for ln in lines if ln.page == head.page and ln.h <= 3 * unit), default=0.0)
        right = left + 0.85 * (page_right - left)
        prev_right, prev_text = 0.0, ""
        for row in rows:
            row.sort(key=lambda l: l.box[0])
            raw = " | ".join(l.text for l in row)
            text = _req_text(raw)
            if not text or not _is_req(text):
                continue
            numbered = bool(ITEM_NO.match(clean(raw)))
            joiner = re.search(r"(?:,|\bAND|\bOR|\bPER|\bWITH|\bTO|-|&)$", prev_text.upper())
            wrapped = items and not numbered and not prev_text.endswith(".") and (prev_right >= right > 0 or joiner)
            if wrapped or (items and text[:1].islower()):
                items[-1] += " " + text
            elif not any(overlap(text, x) > 0.9 for x in items):
                items.append(text)
            prev_right, prev_text = row[-1].box[2], clean(row[-1].text)
        return items
    rows_ = doc.rows
    start = next((i for i, r in enumerate(rows_) if REQ_HEADING.search(r.upper())), None)
    if start is None:
        return items
    prev = ""
    for r in rows_[start + 1:]:
        if REQ_END.search(r.upper()):
            break
        text = _req_text(r)
        if not text or not _is_req(text):
            continue
        # in reading order a wrapped item shows as a row that starts in lowercase, or one after
        # a row that stops on a joining word
        if items and not ITEM_NO.match(r) and (text[:1].islower() or
                                               re.search(r"(?:,|\bAND|\bOR|\bPER|\bWITH|-|&)$", prev.upper())):
            items[-1] += " " + text
        elif not any(overlap(text, x) > 0.9 for x in items):
            items.append(text)
        prev = text
    return items


def parse_form(doc: Doc) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rfq_number": None, "date": None, "respond_by": None, "respond_text": None,
                           "company": None, "rows": [], "requirements": [], "terms": None, "delivery": None}
    head_text = "\n".join(doc.rows[:12])

    def rfq_id(text: str) -> Optional[str]:
        for m in RFQ_ID_RE.finditer(text.upper()):
            tok = m.group(1)
            if re.fullmatch(r"[2Z][0O][0-9OISB]{2}-[0-9OISB]{2}-[0-9OISB]{2}", tok):
                continue  # an ISO date
            fixed = fix_part_number(tok)
            fixed = re.sub(r"^RF[O0]", "RFQ", fixed)
            return fixed
        return None

    got = _labelled(doc, ["RFQ NO", "RFQ NO.", "RFQ NUMBER", "RFQ #", "RFO NO"], rfq_id, ahead=3)
    out["rfq_number"] = got[0] if got else rfq_id(head_text)
    resp_labels = ["RESPOND BY", "RESPONSE DUE", "QUOTE DUE", "DUE DATE", "BID DUE", "REPLY BY", "RESPOND"]
    first_date = lambda t: _iso_dates(t)[0][1] if _iso_dates(t) else None  # noqa: E731
    header_dates = sorted({d for d, _ in _iso_dates(head_text)})
    got = None
    label_seen = False
    if doc.boxed:
        for ln in doc.lines:
            if max(_label_score(ln.text, lb, tail_ok=False) for lb in resp_labels) >= 0.9:
                label_seen = True
                hit = _below(doc, ln, first_date, max_rows=3.0)
                if hit:
                    got = hit
                    break
    if not got and not label_seen:
        got = _cell_value(doc.rows, resp_labels, first_date)
    if got:
        d = _iso_dates(got[0])[0][0]
        out["respond_by"], out["respond_text"] = d.isoformat(), f"RESPOND BY {d.isoformat()}"
    elif len(header_dates) >= 2 and not label_seen:
        # DATE and RESPOND BY sit side by side in the header; with the labels unreadable, the
        # respond-by date is the later one. With only one date readable nothing says which it is.
        d = header_dates[-1]
        out["respond_by"], out["respond_text"] = d.isoformat(), f"RESPOND BY {d.isoformat()}"
    elif label_seen and not got:
        out["respond_unreadable"] = True
    if len(header_dates) >= 2:
        out["date"] = header_dates[0].isoformat()
    for row in doc.rows[:3]:
        t = _strip_junk(row.split("|")[0])
        if re.fullmatch(r"[A-Z][A-Z&.,' -]{3,}", t or "") and "REQUEST" not in t:
            out["company"] = re.sub(r"\s*\.?\s*REQUEST FOR QUOTATION.*$", "", t).title()
            break
    out["rows"] = (_form_rows_boxed(doc) if doc.boxed else None) or _form_rows_text(doc)
    reqs = _form_requirements(doc)
    out["requirements"] = reqs
    rows = doc.rows
    for r in rows:
        m = re.match(r"^\s*TERMS\s*[:.]\s*(.+)$", r, re.IGNORECASE)
        if m:
            out["terms"] = _strip_junk(m.group(1).replace("|", " "))
            break
    for rq in reqs:
        if re.search(r"REQUIRED DELIVERY|DELIVERY\s*:", rq.upper()):
            out["delivery"] = rq
    out["export"] = _find_export(doc.text, doc.ocr)
    return out


# --------------------------------------------------------------------------- #
# The email
# --------------------------------------------------------------------------- #
SIGNOFF = re.compile(r"^(?:thanks|thank you|thanks again|many thanks|best regards|best|regards|kind regards|cheers|"
                     r"sincerely|thx|respectfully)[,!.]?$", re.IGNORECASE)
TITLE_WORDS = re.compile(r"\b(?:buyer|manager|engineer|specialist|agent|administrator|purchasing|sourcing|"
                         r"procurement|supply chain|director|president|owner|ceo|coordinator|planner|technician|"
                         r"analyst|lead|officer|representative|sales|assistant)\b", re.IGNORECASE)
COMPANY_SUFFIXES = ["hydraulics", "robotics", "aerospace", "defense", "medical", "optics", "optomechanics", "energy",
                    "instruments", "packaging", "systems", "industries", "controls", "vacuum", "solar", "precision",
                    "machining", "manufacturing", "technologies", "engineering", "automation", "motorsports",
                    "agriculture", "devices", "conveyor", "electronics", "labs", "group", "tooling", "fabrication",
                    "machinery", "products", "components", "dynamics", "pump", "valve", "fluid"]


def split_body(body: str, from_name: str) -> Tuple[List[str], List[str]]:
    """(content lines, signature lines)."""
    lines = [clean(x) for x in (body or "").splitlines()]
    first = (from_name or "").split()[0].lower() if from_name else ""
    cut = None
    for i, ln in enumerate(lines):
        low = ln.lower().strip()
        if SIGNOFF.match(low) or (low and (low == (from_name or "").lower() or (first and low == first))):
            cut = i
            if not SIGNOFF.match(low):
                break
    if cut is None:
        return lines, []
    return lines[:cut], lines[cut:]


def sentences(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        if not ln:
            continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", ln) if s.strip())
    return out


def email_company(email: Dict[str, Any], signature: List[str]) -> Optional[str]:
    domain = (email.get("from_email") or "").split("@")[-1].lower()
    label = re.sub(r"[^a-z0-9]", "", domain.split(".")[0]) if domain else ""
    name = (email.get("from_name") or "").lower()
    for ln in signature[1:]:
        for part in re.split(r"\s*[,|]\s*", ln):
            words = re.findall(r"[A-Za-z0-9&]+", part)
            if not words or part.lower() == name or re.search(r"\d{3}", part):
                continue
            if label and words[0].lower() in label:
                return part.strip()
    for ln in signature[1:]:
        for part in re.split(r"\s*[,|]\s*", ln):
            if (part and part.lower() != name and not TITLE_WORDS.search(part) and not re.search(r"\d|@", part)
                    and re.fullmatch(r"[A-Z][\w&.'-]*(?:\s+[A-Z&][\w&.'-]*)+", part)):
                return part.strip()
    return None


def domain_company(email: str) -> Optional[str]:
    domain = (email or "").split("@")[-1].lower()
    if not domain or domain.split(".")[0] in ("gmail", "yahoo", "outlook", "hotmail", "icloud", "aol"):
        return None
    label = re.sub(r"[^a-z]", "", domain.split(".")[0])
    for suffix in sorted(COMPANY_SUFFIXES, key=len, reverse=True):
        if label.endswith(suffix) and len(label) > len(suffix) + 2:
            return f"{label[:-len(suffix)].title()} {suffix.title()}"
    return label.title() or None


def _clauses(sentence: str) -> List[str]:
    """Split at commas and joining words, but keep 'Type II, Class 2' together."""
    parts = re.split(r",\s+(?!(?:CLASS|TYPE|METHOD|GRADE|NITRIC|CITRIC|COND)\b)|;\s*|:\s+|\s+with\s+|\s+then\s+|"
                     r"\s+and then\s+|\s+after\s+(?=machining|welding)", sentence, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p and p.strip()]


def _phrase_for(pattern: "re.Pattern[str]", text: str, trim_nouns: bool = False) -> Optional[str]:
    """The clause of a sentence that holds a pattern, trimmed of lead-in words."""
    for clause in _clauses(text):
        m = pattern.search(clause)
        if not m:
            continue
        # start at the match, or a little earlier for adjectives ("black anodize", "unfilled PEEK")
        before = clause[:m.start()].split()
        keep = []
        for w in reversed(before[-2:]):
            if re.fullmatch(r"(?:black|clear|hard|gold|implant[- ]grade|unfilled|glass[- ]filled|annealed|"
                            r"a36|aisi|type)", w, re.IGNORECASE):
                keep.insert(0, w)
            else:
                break
        phrase = " ".join(keep + [clause[m.start():]]).strip(" .,;:")
        phrase = re.sub(r"^(?:then|and|a|an|the|it's|it is|in)\s+", "", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"\s+(?:after machining|after plating|per the print)$", lambda mm: mm.group(0)
                        if "print" in mm.group(0) else "", phrase, flags=re.IGNORECASE)
        if trim_nouns:
            words = phrase.split()
            while len(words) > 1 and words[-1].upper().strip(".,") in PART_NOUNS:
                words.pop()
            phrase = " ".join(words)
        return phrase.strip(" .,;:") or None
    return None


REQ_SPLIT = re.compile(r";\s+|,\s+(?=(?:and|so|then|but|plus)\s)|,?\s+(?:and\s+)?then\s+", re.IGNORECASE)
REQ_POINTER = re.compile(r"\b(?:SEE|ARE (?:ALL )?(?:LISTED )?ON|IS ON|LISTED ON)\s+THE\s+(?:ATTACHED\s+)?(?:RFQ\s+)?FORM\b")
RELEASES = re.compile(r"\bRELEASED\s+(MONTHLY|QUARTERLY|WEEKLY|ANNUALLY)\b|\b(MONTHLY|QUARTERLY|WEEKLY)\s+RELEASES\b",
                      re.IGNORECASE)


def email_requirements(content: List[str], sents: List[str]) -> List[str]:
    """What the quote has to include or allow for, one item per demand, in the sender's words.
    A labelled line ('Inspection: standard, FAI on the first lot') stays whole. Other sentences are
    cut into clauses at ', and' / ', then' / ';' and only the clauses that make a demand stay:
    'Passivate per ASTM A967, then precision clean for high vacuum and double bag' gives the
    cleaning and packaging demand (the finish is its own column). An 'If ...' clause stays with
    the clause it conditions. A blanket request keeps its term and release schedule."""
    reqs: List[str] = []
    labelled = set()
    for ln in content:
        m = re.match(r"^\s*([A-Za-z][A-Za-z /&-]{2,30}):\s*(.+)$", ln)
        if m and not re.match(r"(?:material|finish|quantit|qty|part|p/n|due|quote due|lead time)", m.group(1),
                              re.IGNORECASE) and REQUIREMENT_RE.search(ln) and not URL_RE.search(ln):
            reqs.append(clean(ln).rstrip(" ."))
            labelled.add(clean(ln))
    body = " ".join(sents)
    release = RELEASES.search(body)
    for s in sents:
        if any(s.strip() in lab or lab in s for lab in labelled) or REQ_POINTER.search(s.upper()):
            continue
        if re.match(r"^\s*(?:material|finish|quantit\w*|qty)\s*[:=]", s, re.IGNORECASE):
            continue
        clauses = [c.strip(" ,") for c in REQ_SPLIT.split(s) if c and c.strip(" ,")]
        merged: List[str] = []
        for c in clauses:
            if merged and re.match(r"(?:if|when|unless|although|once)\b", merged[-1], re.IGNORECASE) \
                    and "," not in merged[-1][-1:]:
                merged[-1] = merged[-1] + ", " + c
            else:
                merged.append(c)
        for c in merged:
            if not REQUIREMENT_RE.search(c):
                continue
            c = re.sub(r"^(?:and|so|then|but|plus)\s+", "", c, flags=re.IGNORECASE).strip(" .")
            if re.search(r"\bBLANKET\b", c, re.IGNORECASE):
                m = re.search(r"(?:\d+[- ](?:month|year|yr)\s+)?blanket\s+(?:pricing|order|po|agreement|quote)", c,
                              re.IGNORECASE)
                if m:
                    c = m.group(0)
                    if release:
                        c += ", released " + (release.group(1) or release.group(2)).lower()
            if c and not any(overlap(c, x) > 0.9 for x in reqs):
                reqs.append(c)
    return reqs


def parse_email(email: Dict[str, Any], today: dt.date) -> Dict[str, Any]:
    subject = clean(email.get("subject"))
    content, signature = split_body(email.get("body") or "", email.get("from_name") or "")
    sents = sentences(content)
    body_text = " ".join(sents)
    out: Dict[str, Any] = {"subject": subject, "content": content, "signature": signature, "sentences": sents}

    # identifiers
    rfq = None
    for src, text in (("subject", subject), ("email body", body_text)):
        m = re.search(r"\bRFQ-\d{2}-\d{3,5}\b", text, re.IGNORECASE)
        if m:
            rfq = (m.group(0).upper(), src)
            break
        m = RFQ_NO_RE.search(text)
        if m:
            rfq = (m.group(1).upper(), src)
            break
    out["rfq_number"] = rfq
    qref = None
    for src, text in (("subject", subject), ("email body", body_text)):
        m = QUOTE_REF_RE.search(text)
        if m:
            qref = (m.group(1).upper(), src)
            break
    out["quote_ref"] = qref
    revision = bool(qref) and bool(re.search(r"\b(?:UPDATE|REVISE|REVISED|REVISION|REQUOTE|RE-QUOTE|AGAINST REV|"
                                              r"NEW REV|RELEASED REV)\b", (subject + " " + body_text).upper()))
    revision = revision or bool(re.search(r"\b(?:REVISED QUOTE|UPDATE(?:D)? (?:OUR |THE |YOUR )?QUOTE|REQUOTE)\b",
                                          body_text.upper()))
    out["request"] = "quote revision" if revision else "new RFQ"

    # part numbers named in the email, with a rev when one sits next to them
    excluded = {fold(rfq[0])} if rfq else set()
    if qref:
        excluded.add(fold(qref[0]))
    pns: List[Dict[str, Any]] = []
    for src, text in (("subject", subject), ("email body", "\n".join(content))):
        for m in PN_RE.finditer(text):
            pn = m.group(1).upper()
            if (fold(pn) in excluded or pn.split("-")[0] in SPEC_PREFIXES or re.search(r"(?:^|-)RF[QP](?:-|$)", pn)
                    or any(fold(pn) == fold(e) or fold(pn).endswith(fold(e)) for e in excluded)):
                continue
            if re.search(r"\bPER\s+$", text[max(0, m.start() - 5):m.start()], re.IGNORECASE):
                continue  # "label per BWM-QS-0412": a process spec, not a part to quote
            # "the same program as the TO-5520 lens cell", "our quality clauses CV-QR-004": a number
            # named for reference, which only joins a line some file already gives
            lead = re.split(r"[.!?;:\n]", text[max(0, m.start() - 40):m.start()])[-1]
            ref = bool(re.search(r"\b(?:SAME|SIMILAR|LIKE|PREVIOUS|PRIOR|REPLACES|REPLACED|SUPERSEDES?|OLD|LAST|"
                                 r"CLAUSES?|SPEC(?:IFICATION)?S?|PROCEDURES?|STANDARDS?|INSTRUCTIONS?|QUALITY)\b",
                                 lead, re.IGNORECASE))
            after = text[m.end():m.end() + 16]
            r = re.match(r"\s*,?\s*REV(?:ISION)?\.?\s*([A-Z0-9]{1,2})\b", after, re.IGNORECASE)
            rev = r.group(1).upper() if r else None
            if not rev:
                sent = next((s for s in re.split(r"(?<=[.!?])\s+", text) if pn in s.upper()), "")
                r = re.search(r"\bREV(?:ISION)?\.?\s+([A-Z0-9]{1,2})\s+OF\b", sent, re.IGNORECASE)
                if r and len(set(PN_RE.findall(sent))) == 1:
                    rev = r.group(1).upper()
            hit = next((p for p in pns if fold(p["pn"]) == fold(pn)), None)
            if hit:
                hit["rev"] = hit["rev"] or rev
                continue
            pns.append({"pn": pn, "rev": rev, "src": src, "pos": m.start(), "ref": ref})
    out["part_numbers"] = pns

    # per-part lines in the body ("BWM-3105 Rev A, pivot pin, 17-4 PH stainless, condition H1025, passivated")
    per_part: Dict[str, Dict[str, Optional[str]]] = {}
    part_rows = set()
    for ln in content:
        m = re.match(r"^\s*(?:P/N\s*)?(" + PN_BODY + r")\s*(?:,?\s*REV\.?\s*([A-Z0-9]{1,2}))?\s*[,:-]\s*(.+)$",
                     ln, re.IGNORECASE)
        if not m:
            continue
        fields = [f.strip() for f in re.split(r",\s*", m.group(3)) if f.strip()]
        info: Dict[str, Optional[str]] = {"description": None, "material": None, "finish": None}
        mat: List[str] = []
        for f in fields:
            if FINISH_RE.search(f) and not info["finish"]:
                info["finish"] = f
            elif MATERIAL_STRONG.search(f) or MATERIAL_WORD.search(f):
                mat.append(f)
            elif mat and re.match(r"(?:condition|cond\.?)\s", f, re.IGNORECASE):
                mat.append(f)
            elif not info["description"] and not mat:
                info["description"] = f
        info["material"] = ", ".join(mat) or None
        per_part[fold(m.group(1))] = info
        part_rows.add(ln)
    out["per_part"] = per_part
    # material and finish said for the whole email come from the other sentences: a per-part row's
    # "passivated" belongs to that part only
    general = sentences([ln for ln in content if ln not in part_rows])

    # labelled lines: "Material: ...", "Finish: ...", "Quantities: ..."
    labels: Dict[str, str] = {}
    for ln in content:
        m = re.match(r"^\s*(material|finish|quantit(?:y|ies)|qty|inspection|lead time|due date|quote due)\s*[:=]\s*(.+)$",
                     ln, re.IGNORECASE)
        if m:
            labels[m.group(1).lower()[:5]] = m.group(2).strip()

    # material and finish for the whole email
    material = None
    if "mater" in labels:
        material = labels["mater"]
    else:
        for s in general:
            if MATERIAL_STRONG.search(s):
                material = _phrase_for(MATERIAL_STRONG, s, trim_nouns=True)
                if material:
                    break
    out["material"] = material
    finish = labels.get("finis")
    if not finish:
        for s in general:
            if FINISH_RE.search(s) and not re.search(r"\bWITH AND WITHOUT\b|\bIF THE\b", s.upper()):
                finish = _phrase_for(FINISH_RE, s)
                if finish:
                    break
    out["finish"] = finish

    # quantities and annual usage
    qty: Optional[List[int]] = None
    annual: Optional[int] = None
    qty_text = None
    for src_text in ([labels["quant"]] if "quant" in labels else []) + ([labels["qty"]] if "qty" in labels else []):
        q, _ = parse_quantities(re.split(r"\b(?:pcs|pieces|ea)\b", src_text, flags=re.IGNORECASE)[0])
        if q:
            qty, qty_text = q, src_text
            break
    for s in [subject] + sents:
        up = s.upper()
        if re.search(r"\bANNUAL|PER YEAR|/\s*YR\b|A YEAR\b|\bEAU\b|YEARLY|USAGE\b|\bVOLUME\b", up):
            m = re.search(r"(?:ANNUAL\s+(?:USAGE|VOLUME)|USAGE|EAU|VOLUME)[A-Z ]{0,30}?(?:IS|OF|:)?\s*(?:ABOUT|AROUND|"
                          r"APPROX\.?|APPROXIMATELY|ROUGHLY|~)?\s*(\d[\d,.]*K?)\s*(?:PCS|PIECES|EA|UNITS)?", up)
            if not m:
                m = re.search(r"(\d[\d,.]*K?)\s*(?:PCS|PIECES)?\s*(?:/\s*YR|PER YEAR|A YEAR|ANNUALLY)", up)
            if m and _num(m.group(1)) and annual is None:
                annual = _num(m.group(1))
        if qty:
            continue
        m = re.search(r"\bRELEASE(?:\s+QUANTITY)?\s*(?:IS|OF|:)?\s*(\d[\d,]*)", up)
        if m:
            qty, qty_text = [_num(m.group(1))], s
            continue
        m = re.search(r"\bQUANTIT(?:Y|IES)\s*[:=]?\s*((?:\d[\d,]*\s*(?:PCS|PIECES)?\s*(?:/|,|AND|&)?\s*)+)", up)
        if m and _num(m.group(1).split()[0].strip(",/")):
            q, _ = parse_quantities(m.group(1))
            if q:
                qty, qty_text = q, s
                continue
        m = re.search(r"\bQTY\.?\s*[:=]?\s*(\d[\d,]*(?:\s*(?:/|,)\s*\d[\d,]*)*)", up)
        if m:
            q, _ = parse_quantities(m.group(1))
            if q:
                qty, qty_text = q, s
                continue
        scrub = PN_RE.sub(" ", up)
        scrub = re.sub(r"\b[A-Z]+\d[\w-]*|\b\d+[A-Z][\w-]*", " ", scrub)  # 316L, 6061-T6, H1025
        scrub = re.sub(r"\(.*?\)", " ", scrub)
        scrub = re.sub(r"\b(?:POSSIBLY|MAYBE|PERHAPS)\s+\d[\d,]*\s+MORE\b.*", " ", scrub)
        scrub = re.sub(r"\b\d[\d,]*\s+MORE\b", " ", scrub)
        if re.search(r"ANNUAL|PER YEAR|/\s*YR|USAGE|VOLUME", scrub):
            continue
        m = re.search(r"((?:\d[\d,]*\s*(?:,|/|AND|&|OR)\s*)*\d[\d,]*)\s*(?:PCS|PC|PIECES|EA|UNITS)\b", scrub)
        if m:
            q, _ = parse_quantities(m.group(1))
            if q:
                qty, qty_text = q, s
                continue
        m = re.search(r"\b(?:QUOTE|QUOTATION|PRICING|PRICE)\s+(?:FOR|ON)\s+(\d[\d,]*)\s+[A-Z]", scrub)
        # "blanket pricing for 3 enclosure covers" counts part numbers, not pieces
        if m and s is not subject and not (_num(m.group(1)) == len(pns) and len(pns) > 1):
            qty, qty_text = [_num(m.group(1))], s
    out["quantities"] = (qty, qty_text)
    out["annual_usage"] = annual

    # descriptions: the words before a part number, the subject, or "12 gimbal yokes"
    descs: Dict[str, str] = {}
    body_join = "\n".join(content)
    for p in pns:
        text = subject if p["src"] == "subject" else body_join
        idx = text.upper().find(p["pn"])
        before = text[:idx]
        before = re.split(r"[.!?:;(\n]\s*", before)[-1]
        before = re.sub(r"[,\s]*(?:P/N|PN|PART(?:\s+NUMBER)?|#)?[\s,]*$", "", before, flags=re.IGNORECASE)
        m = re.search(r"(?:\b(?:the attached|attached|the|a|an|our|your|this|of|on|for)\s+)((?:[a-z][a-z0-9-]*\s*){1,5})$",
                      before)
        if m:
            d = re.sub(r"^(?:a|an|the)\s+", "", m.group(1).strip(), flags=re.IGNORECASE)
            if not re.search(r"\b(?:drawing|print|rfq|file|scan|photo|model|step|quote|pdf|package|revision|rev)\b", d):
                descs.setdefault(fold(p["pn"]), d)
    sub_desc = None
    m = re.search(r"(?:\bRFQ\b[^:]*|QUOTE REQUEST|REQUEST FOR QUOTE|QUOTE|RFQ)\s*:\s*(.+)$", subject, re.IGNORECASE)
    if m:
        d = re.split(r",|\s+-\s+|\s+(?=[A-Z][A-Z0-9]{0,4}-\d)", m.group(1))[0].strip()
        if d and not PN_RE.fullmatch(d) and len(d.split()) <= 7:
            sub_desc = d
    if not sub_desc:
        m = re.search(r"\bQUOTE\s+ON\s+([a-z][a-z -]+)$", subject, re.IGNORECASE)
        if m:
            sub_desc = m.group(1).strip()
    qty_desc = None
    m = re.search(r"\b(?:quote|quotation|pricing|price)\s+(?:for|on)\s+\d[\d,]*\s+([a-z][a-z -]{2,40}?)(?:\s+in\b|\s+of\b|,|\.|$)",
                  body_text)
    if m:
        qty_desc = m.group(1).strip()
    out["descriptions"], out["subject_description"], out["qty_description"] = descs, sub_desc, qty_desc

    # sizes stated in the email
    size = None
    m = re.search(r"((?:about\s+|approx\.?\s+)?(?:\d*\.\d+|\d+)\"?(?:\s*(?:OD|ID|dia(?:meter)?|long|thick|wide|lg))?"
                  r"(?:\s*x\s*(?:\d*\.\d+|\d+)\"?(?:\s*(?:OD|ID|dia(?:meter)?|long|thick|wide|lg))?)+)", body_text,
                  re.IGNORECASE)
    if m and re.search(r"\sx\s", m.group(1), re.IGNORECASE):
        size = m.group(1).strip()
    out["size"] = size

    # dates: when the quote is due, and when parts are needed
    respond = None
    for s in [subject] + sents:
        if RESPOND_CUES.search(s) and not NOT_RESPOND.search(s):
            got = parse_date(s, today)
            if got:
                respond = (got[0].isoformat(), s.strip())
                break
    out["respond_by"] = respond
    delivery = None
    for s in sents:
        if DELIVERY_CUES.search(s) and not RESPOND_CUES.search(s):
            got = parse_date(s, today)
            delivery = (s.strip(), got[0].isoformat() if got else None)
            break
    out["delivery"] = delivery

    out["requirements"] = email_requirements(content, sents)

    # export control, drawing links, drawings promised later
    full = subject + "\n" + "\n".join(content)
    marks = [] if NOT_EXPORT.search(full) and not re.search(r"\bITAR CONTROLLED\b|\bMARKED CUI\b", full.upper()) \
        else _find_export(full, fuzzy=False)
    out["export"] = marks
    links = []
    for m in URL_RE.finditer("\n".join(content)):
        around = body_text[max(0, body_text.find(m.group(0)) - 300):body_text.find(m.group(0)) + 50]
        if re.search(r"\b(?:DRAWING|DRAWINGS|STEP|CAD|MODEL|TECHNICAL DATA|TDP|FILE SHARE|PORTAL|PRINTS?|FILES)\b",
                     around.upper()):
            links.append(m.group(0).rstrip(".,"))
    out["drawing_links"] = links
    out["drawing_later"] = bool(re.search(r"\b(?:SEND|EMAIL|FORWARD)\s+(?:YOU\s+)?THE\s+(?:DRAWING|PRINT|CAD|STEP)"
                                          r"[^.]*\b(?:TOMORROW|LATER|SOON|NEXT WEEK|WHEN)|DRAWING(?:S)?\s+(?:TO|WILL)\s+"
                                          r"FOLLOW", body_text.upper()))
    out["company"] = email_company(email, signature)
    return out


# --------------------------------------------------------------------------- #
# Material and finish comparison (for check notes)
# --------------------------------------------------------------------------- #
def material_keys(text: Optional[str]) -> set:
    """What identifies a material, whatever the wording: alloy, temper, condition."""
    if not text:
        return set()
    t = ocr_fix_spec(clean(text).upper())
    keys = set()
    for m in re.finditer(r"\b([1-7]\d{3})(?:-(T\d{1,4}))?(?:\s*OR\s*(T\d{1,4}))?\b", t):
        if re.search(r"\b(?:AL|ALUMINUM|ALUMINIUM)\b", t) or m.group(2):
            keys.add(m.group(1))
            for g in (m.group(2), m.group(3)):
                if g:
                    keys.add(f"{m.group(1)}-{g}")
    for m in re.finditer(r"\b(17-4|15-5|13-8)\s?PH\b", t):
        keys.add(m.group(1) + "PH")
    for m in re.finditer(r"\b(H\d{3,4})\b", t):
        keys.add(m.group(1))
    for m in re.finditer(r"\b(30[34]L?|31[06]L?|41[06]|420|440C)\b", t):
        if re.search(r"STAINLESS|SS\b|SST|CRES", t) or m.group(1).endswith("L"):
            keys.add(m.group(1))
    for m in re.finditer(r"\b(4140|4340|8620|12L14|1018|1020|1045|A36)\b", t):
        keys.add(m.group(1))
    for m in re.finditer(r"\bC(3\d{2})(?:00)?\b", t):
        keys.add("C" + m.group(1))
    for word in ("PEEK", "DELRIN", "ACETAL", "ULTEM", "TITANIUM", "BRASS", "INCONEL"):
        if word in t:
            keys.add(word)
    return keys


def materials_agree(a: Optional[str], b: Optional[str]) -> bool:
    ka, kb = material_keys(a), material_keys(b)
    if not ka or not kb:
        return True
    if ka <= kb or kb <= ka:
        return True
    # different wording of the same alloy: compare alloys, then tempers only where both give one
    alloys_a = {k for k in ka if "-" not in k}
    alloys_b = {k for k in kb if "-" not in k}
    if alloys_a and alloys_b and not (alloys_a & alloys_b):
        return False
    tempers_a = {k for k in ka if "-T" in k}
    tempers_b = {k for k in kb if "-T" in k}
    if tempers_a and tempers_b and not (tempers_a & tempers_b):
        return False
    conds_a = {k for k in ka if re.fullmatch(r"H\d+", k)}
    conds_b = {k for k in kb if re.fullmatch(r"H\d+", k)}
    return not (conds_a and conds_b and not (conds_a & conds_b))


FINISH_FAMILIES = [("anodize", r"ANODI"), ("passivate", r"PASSIVAT"), ("chem film", r"CHEM\w*\s+FILM|CONVERSION|ALODINE"),
                   ("electroless nickel", r"ELECTROLESS"), ("black oxide", r"BLACK\s+OXIDE"),
                   ("powder coat", r"POWDER"), ("paint", r"PAINT"), ("zinc", r"ZINC"), ("gold", r"GOLD"),
                   ("none", r"^NONE\b")]


def finishes_agree(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return True
    ta, tb = clean(a).upper(), clean(b).upper()
    fa = {name for name, pat in FINISH_FAMILIES if re.search(pat, ta)}
    fb = {name for name, pat in FINISH_FAMILIES if re.search(pat, tb)}
    if fa and fb and not (fa <= fb or fb <= fa):
        return False
    if "anodize" in fa & fb:
        for colors in (("BLACK", "CLEAR"),):
            ca = {c for c in colors if c in ta}
            cb = {c for c in colors if c in tb}
            if ca and cb and ca != cb:
                return False
        tya = re.search(r"TYPE\s+(III|II|I)\b", ta)
        tyb = re.search(r"TYPE\s+(III|II|I)\b", tb)
        hard_a = "HARD" in ta or (tya and tya.group(1) == "III")
        hard_b = "HARD" in tb or (tyb and tyb.group(1) == "III")
        if (tya or "HARD" in ta) and (tyb or "HARD" in tb) and hard_a != hard_b:
            return False
    return True


# --------------------------------------------------------------------------- #
# Putting one RFQ together
# --------------------------------------------------------------------------- #
def _v(value: Any = None, source: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    out = {"value": value, "source": source if value not in (None, [], "") else None}
    out.update(extra)
    return out


def lookup_customer(from_email: str, shop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    domain = (from_email or "").split("@")[-1].lower().strip()
    for customer in (shop or {}).get("customers", []):
        if customer.get("domain", "").lower() == domain:
            return customer
    return None


def is_rfq(email: Dict[str, Any], decision: Optional[Dict[str, Any]] = None) -> bool:
    """With a Jev decision, Jev's call. Without one: the sender asks for a price on parts, and the
    email is not an order (placing, changing, releasing, or chasing a PO) or a sales pitch.
    A question about what 'something like this' would cost, with no part named, is a capability
    question for a person to answer, not an RFQ (E17 in the beta inbox)."""
    if decision and "is_rfq" in decision:
        return bool(decision["is_rfq"])
    subject = clean(email.get("subject")).upper()
    content, _ = split_body(email.get("body") or "", email.get("from_name") or "")
    body = " ".join(content).upper()
    text = subject + " " + body
    pitch = re.search(r"\d+\s*% OFF|\bUNSUBSCRIBE\b|THIS MONTH ONLY|FREE SHIPPING|\bWEBINAR\b|\bNEWSLETTER\b|"
                      r"\bOUR (?:ONLINE )?STORE\b|\bRESUME\b|\bJOB OPENING\b|\bWE ARE HIRING\b|SET UP A (?:QUICK )?CALL|"
                      r"\bSTOCK (?:AND DROP )?LIST\b|\bDROP LIST\b|OPEN AN ACCOUNT|CREDIT APP", text)
    order = re.search(r"\b(?:PO|P\.O\.|PURCHASE ORDER)\s*#?\s*\d+[^.?!]*\b(?:STATUS|TRACKING|SHIP|PROMISE|ON TRACK)|"
                      r"\bSTATUS\?|\bTRACKING\b", text)
    # an order being placed, changed, or released: it may cite a quote, but it does not ask for one
    placed = re.search(r"^(?:RE:\s*|FW:\s*)?(?:PO|P\.O\.|PURCHASE ORDER|CHANGE ORDER|BLANKET PO|RELEASE)\b", subject) \
        or re.search(r"\bATTACHED PO\b|\bPO\s+\S+\s+IS ATTACHED|\bRELEASE\s+\d+\s+AGAINST\b|"
                     r"\bNOT ASKING FOR A NEW QUOTE\b|\bOF OUR PO\b", body)
    ask = re.search(r"\bRFQ\b|\bRFP\b|REQUEST FOR (?:A )?QUOT|\bQUOTE REQUEST\b|\bQUOTATION\b|"
                    r"\b(?:PLEASE|CAN YOU|COULD YOU|WOULD YOU)\s+(?:\w+\s+){0,2}QUOTE\b|\bQUOTE (?:ON|FOR)\b|"
                    r"\bREQUESTING A QUOTE\b|\bNEED A (?:QUOTE|PRICE)\b|\bSEND (?:ME |US )?(?:BUDGETARY )?PRICING\b|"
                    r"\bPRICING (?:ON|FOR)\b|\bPRICE ON\b|\bBUDGETARY (?:QUOTE|PRICING|PRICE)\b|"
                    r"\bBLANKET PRICING\b|\bLOOKING FOR PRICING\b|\bQUOTE\b.*\b(?:PCS|QTY|DRAWING)\b|\bUPDATE (?:OUR |THE )?QUOTE\b|"
                    r"\bA QUOTE\b",
                    text)
    explicit = re.search(r"\bRFQ\b|REQUEST FOR (?:A )?QUOT|\bQUOTE REQUEST\b|\bPLEASE QUOTE\b|\bNEED A QUOTE\b|"
                         r"\bUPDATE (?:OUR |THE )?QUOTE\b", text)
    if pitch and not re.search(r"\bRFQ\b|REQUEST FOR QUOT", subject):
        return False
    if placed and not explicit:
        return False
    if order and not ask:
        return False
    return bool(ask)


def _near_key(a: str, b: str) -> bool:
    """Two folded part numbers one or two characters apart, with the same start."""
    return bool(a and b) and a[:2] == b[:2] and abs(len(a) - len(b)) <= 1 and \
        difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def _variant_of(a: str, b: str) -> bool:
    """One part number is the other plus a dash number: BWM-3140-08 and BWM-3140."""
    a, b = clean(a).upper(), clean(b).upper()
    long_, short = (a, b) if len(a) > len(b) else (b, a)
    return long_.count("-") > short.count("-") and fold(long_.rsplit("-", 1)[0]) == fold(short)


def _pick(cands: List[Tuple[str, Any, str, Optional[float]]], order: Sequence[str]) -> Optional[Tuple[str, Any, str, Optional[float]]]:
    """cands: (kind, value, source label, OCR conf or None). The first kind in precedence order."""
    for kind in order:
        for c in cands:
            if c[0] == kind and c[1] not in (None, "", []):
                return c
    return None


def _email_date(email: Dict[str, Any]) -> Optional[dt.date]:
    for key in ("date", "received_date", "received"):
        m = re.match(r"\s*(20\d\d-\d\d-\d\d)", str(email.get(key) or ""))
        if m:
            try:
                return dt.date.fromisoformat(m.group(1))
            except ValueError:
                pass
    return None


def extract(email: Dict[str, Any], texts: Dict[str, Dict[str, Any]], shop: Dict[str, Any],
            decision: Optional[Dict[str, Any]] = None, *, today: Optional[dt.date] = None) -> Dict[str, Any]:
    """One RFQ record from an email and the ocr.file_text result of each attachment. Relative dates
    ("next Friday") resolve against `today`, else the day the email arrived when it says, else today."""
    today = today or _email_date(email) or dt.date.today()
    texts = texts or {}
    em = parse_email(email, today)
    check: List[str] = []

    # attachments
    docs: List[Doc] = []
    for att in email.get("attachments") or []:
        name = att.get("name") if isinstance(att, dict) else str(att)
        media = (att.get("media") if isinstance(att, dict) else "") or Path(name or "").suffix.lstrip(".").lower()
        media = {"stp": "step", "jpeg": "jpg"}.get(media, media)
        doc = Doc(name, media, texts.get(name))
        doc.kind, doc.capture = classify(doc) if doc.text else (
            ("3D model" if media == "step" else "photo" if media == "jpg" else "screenshot" if media == "png" else "other"),
            "digital")
        docs.append(doc)
    email_pns = [p["pn"] for p in em["part_numbers"] if not p.get("ref")]
    for doc in docs:
        if doc.kind == "RFQ form":
            doc.parsed = parse_form(doc)
        elif doc.kind == "3D model":
            doc.parsed = parse_step(doc)
    # A drawing number is trusted when it sits in a part number cell or another source names it,
    # so the drawings are read last, with every number the email, forms, and models give.
    named = email_pns + [row["part_number"] for d in docs if d.kind == "RFQ form"
                         for row in d.parsed.get("rows") or [] if row.get("part_number")]
    named += [d.parsed["part_number"] for d in docs if d.kind == "3D model" and d.parsed.get("part_number")]
    for doc in docs:
        if doc.kind == "drawing":
            doc.parsed = parse_drawing(doc, named)
    forms = [d for d in docs if d.kind == "RFQ form"]
    drawings = [d for d in docs if d.kind == "drawing" and d.parsed.get("part_number")]
    models = [d for d in docs if d.kind == "3D model" and d.parsed.get("part_number")]

    # customer
    cust = lookup_customer(email.get("from_email", ""), shop)
    if cust:
        customer, customer_src = cust.get("name"), "shop customer list"
    elif em["company"]:
        customer, customer_src = em["company"], "email signature"
    else:
        customer, customer_src = None, None
        domain = (email.get("from_email") or "").split("@")[-1].lower()
        label = re.sub(r"[^a-z0-9]", "", domain.split(".")[0]) if domain else ""
        for d in forms + drawings:
            comp = d.parsed.get("company")
            if comp and label and comp.split()[0].lower() in label:
                customer, customer_src = comp, d.name
                break
        if not customer:
            customer, customer_src = domain_company(email.get("from_email", "")), "email domain"

    # RFQ number, respond-by, requirements from forms
    rfq_cands = []
    if em["rfq_number"]:
        rfq_cands.append(("email", em["rfq_number"][0], em["rfq_number"][1]))
    for f in forms:
        if f.parsed.get("rfq_number"):
            rfq_cands.append(("form", f.parsed["rfq_number"], f.label))
    rfq_number = _v()
    if rfq_cands:
        # the typed subject is exact; a form number that matches it up to OCR slips adds nothing
        best = rfq_cands[0]
        rfq_number = _v(best[1], best[2])
        base = lambda s: fold(re.sub(r"^RF[QO0]-?", "", s.upper()))  # noqa: E731
        for kind, val, src in rfq_cands[1:]:
            if base(val) != base(best[1]) and difflib.SequenceMatcher(None, base(val), base(best[1])).ratio() < 0.8:
                check.append(f"RFQ number differs: {best[2]} says {best[1]}, {src} says {val}")
    quote_ref = _v(*em["quote_ref"]) if em["quote_ref"] else _v()

    respond = _v(text=None)
    form_resp = next(((f.parsed["respond_by"], f.parsed.get("respond_text"), f.label) for f in forms
                      if f.parsed.get("respond_by")), None)
    if form_resp:
        respond = _v(form_resp[0], form_resp[2], text=form_resp[1])
        if em["respond_by"] and em["respond_by"][0] != form_resp[0]:
            check.append(f"Respond-by differs: email says {em['respond_by'][0]} (\"{em['respond_by'][1]}\"), "
                         f"{form_resp[2]} says {form_resp[0]}; used the RFQ form")
    elif em["respond_by"]:
        respond = _v(em["respond_by"][0], "email body", text=em["respond_by"][1])
    for f in forms:
        if f.parsed.get("respond_unreadable") and not f.parsed.get("respond_by"):
            check.append(f"The respond-by date on {f.name} could not be read" +
                         ("; used the email's" if respond["value"] else ""))

    requirements: List[Dict[str, Any]] = []
    for f in forms:
        for r in f.parsed.get("requirements") or []:
            requirements.append({"value": r, "source": f.label})
    # an email line that only repeats what the RFQ form already lists adds nothing
    form_words = " ".join(x["value"] for x in requirements)
    for r in em["requirements"]:
        if not any(overlap(r, x["value"]) >= 0.8 for x in requirements) and not (form_words and
                                                                                  _covered(r, form_words)):
            requirements.append({"value": r, "source": "email body"})

    delivery = _v()
    if em["delivery"]:
        delivery = _v(em["delivery"][0], "email body", date=em["delivery"][1])
    else:
        for f in forms:
            if f.parsed.get("delivery"):
                delivery = _v(f.parsed["delivery"], f.label, date=None)
                break
    terms = next((_v(f.parsed["terms"], f.label) for f in forms if f.parsed.get("terms")), _v())

    # export control: every place a marking shows up
    marks: List[Tuple[str, str]] = [(k, "email body") for k, _ in em["export"]]
    for d in docs:
        for k, _ in (d.parsed.get("export") or []):
            marks.append((k, d.label))
    export = _v()
    if marks:
        kinds = []
        for k, _ in marks:
            if k not in kinds and not (k == "EAR" and any(x.startswith("EAR (") for x, _ in marks)):
                kinds.append(k)
        order = {"ITAR": 0, "EAR": 1, "CUI": 2}
        kinds.sort(key=lambda k: order.get(k.split(" ")[0], 3))
        sources = []
        for _, s in marks:
            if s not in sources:
                sources.append(s)
        export = _v(", ".join(kinds), "; ".join(sources))
        if all(s != "email body" for _, s in marks):
            check.append(f"{', '.join(kinds)} marking found only in {', '.join(sources)}; the email does not mention it")

    # ---- part lines -------------------------------------------------------- #
    lines: List[Dict[str, Any]] = []

    def find_line(pn: Optional[str], allow_prefix: bool = True) -> List[Dict[str, Any]]:
        if not pn:
            return []
        key = fold(pn)
        exact = [ln for ln in lines if ln["_key"] == key]
        if exact or not allow_prefix:
            return exact
        return [ln for ln in lines if ln["_key"].startswith(key) and len(ln["_key"]) > len(key)]

    def near_line(pn: Optional[str]) -> List[Dict[str, Any]]:
        """The one line whose part number only OCR gave and which differs from pn by a slip
        ('QA-41172' read for 'QA-41127'): the same part, not a new line."""
        if not pn:
            return []
        key = fold(pn)
        near = [ln for ln in lines if ln["_key"] and ln["_key"] != key and _near_key(ln["_key"], key)
                and ln["_c"]["part_number"] and all("(OCR" in c[2] for c in ln["_c"]["part_number"])]
        if len(near) != 1:
            return []
        near[0]["_key"] = key  # from here on the line goes by the typed number
        return near

    def new_line(pn: Optional[str]) -> Dict[str, Any]:
        ln = {"_key": fold(pn) if pn else "", "_c": {f: [] for f in ("part_number", "rev", "description", "material",
                                                                       "finish", "quantities", "annual_usage", "size")},
              "_qty_complete": True}
        lines.append(ln)
        return ln

    def add(ln: Dict[str, Any], field: str, kind: str, value: Any, source: str, conf: Optional[float] = None) -> None:
        if value in (None, "", []):
            return
        ln["_c"][field].append((kind, value, source, conf))

    for f in forms:
        for row in f.parsed.get("rows") or []:
            hit = find_line(row["part_number"], allow_prefix=False)
            ln = hit[0] if hit else new_line(row["part_number"])
            conf = row.get("conf") if f.ocr else None
            add(ln, "part_number", "form", row["part_number"], f.label, conf)
            add(ln, "rev", "form", row.get("rev"), f.label, conf)
            add(ln, "description", "form", row.get("description"), f.label, conf)
            add(ln, "material", "form", row.get("material"), f.label, conf)
            add(ln, "finish", "form", row.get("finish"), f.label, conf)
            add(ln, "quantities", "form", row.get("quantities"), f.label, conf)
            if not row.get("qty_complete", True):
                ln["_qty_complete"] = False
    # drawings: one line each, or every dash-number line of the form that starts with the drawing number
    ordered = sorted(drawings, key=lambda d: next((i for i, p in enumerate(email_pns)
                                                   if fold(p) == fold(d.parsed["part_number"])), 99))
    for d in ordered:
        p = d.parsed
        targets = find_line(p["part_number"]) or [new_line(p["part_number"])]
        conf = p.get("conf", {})
        for ln in targets:
            variant = ln["_key"] != fold(p["part_number"])
            add(ln, "part_number", "drawing_variant" if variant else "drawing", p["part_number"], d.label,
                conf.get("part_number") if d.ocr else None)
            add(ln, "rev", "drawing", p.get("rev"), d.label, conf.get("rev") if d.ocr else None)
            add(ln, "description", "drawing_variant" if variant else "drawing", p.get("title"), d.label,
                conf.get("title") if d.ocr else None)
            add(ln, "material", "drawing", p.get("material"), d.label, conf.get("material") if d.ocr else None)
            add(ln, "finish", "drawing", p.get("finish"), d.label, conf.get("finish") if d.ocr else None)
    unnumbered = [d for d in docs if d.kind == "drawing" and not d.parsed.get("part_number")]
    for d in models:
        p = d.parsed
        targets = find_line(p["part_number"]) or near_line(p["part_number"]) or [new_line(p["part_number"])]
        for ln in targets:
            add(ln, "part_number", "model", p["part_number"], d.label)
            add(ln, "rev", "model", p.get("rev"), d.label)
            add(ln, "description", "model", p.get("title"), d.label)
            add(ln, "size", "model", p.get("size"), d.label)
    for p in em["part_numbers"]:
        targets = find_line(p["pn"]) or near_line(p["pn"])
        if not targets and p.get("ref"):
            continue
        if not targets:
            if lines and (forms or drawings) and p["src"] == "subject" and len(em["part_numbers"]) > len(lines):
                continue
            targets = [new_line(p["pn"])]
        src = "subject" if p["src"] == "subject" else "email body"
        for ln in targets:
            add(ln, "part_number", "email", p["pn"], src)
            add(ln, "rev", "email", p.get("rev"), src)
    if not lines:
        new_line(None)
    # A drawing whose number OCR could not read still describes a part: when the RFQ has one
    # drawing like that and one line no other drawing covers, its title block belongs to that line.
    if len(unnumbered) == 1:
        free = [ln for ln in lines if not any(k == "drawing" for k, *_ in ln["_c"]["material"] + ln["_c"]["finish"]
                                              + ln["_c"]["description"])]
        if len(free) == 1 and len(lines) == 1:
            d, ln = unnumbered[0], free[0]
            p, conf = d.parsed, d.parsed.get("conf", {})
            add(ln, "rev", "drawing", p.get("rev"), d.label, conf.get("rev"))
            add(ln, "description", "drawing", p.get("title"), d.label, conf.get("title"))
            add(ln, "material", "drawing", p.get("material"), d.label, conf.get("material"))
            add(ln, "finish", "drawing", p.get("finish"), d.label, conf.get("finish"))
            check.append(f"The drawing number on {d.name} could not be read; its title block was used for line 1")
    # email-wide values reach every line; per-part listings reach their own line
    qty, _ = em["quantities"]
    for ln in lines:
        info = next((v for k, v in em["per_part"].items() if ln["_key"] and (ln["_key"] == k or ln["_key"].startswith(k))),
                    None)
        if info:
            add(ln, "description", "email", info.get("description"), "email body")
            add(ln, "material", "email", info.get("material"), "email body")
            add(ln, "finish", "email", info.get("finish"), "email body")
        desc = em["descriptions"].get(ln["_key"]) or next(
            (v for k, v in em["descriptions"].items() if ln["_key"].startswith(k)), None)
        if not desc and len(lines) == 1:
            desc = em["subject_description"] or em["qty_description"]
        add(ln, "description", "email", desc, "email body" if desc in em["descriptions"].values() else "subject")
        if not info:
            add(ln, "material", "email", em["material"], "email body")
            add(ln, "finish", "email", em["finish"], "email body")
        add(ln, "quantities", "email", qty, "email body")
        add(ln, "annual_usage", "email", em["annual_usage"], "email body")
        add(ln, "size", "email", em["size"], "email body")

    # ---- resolve each field with the precedence rules ---------------------- #
    out_lines = []
    for n, ln in enumerate(lines, start=1):
        c = ln["_c"]
        rec: Dict[str, Any] = {"line": n}
        for field, order in (("part_number", ("drawing", "form", "model", "email", "drawing_variant")),
                             ("rev", ("drawing", "form", "model", "email")),
                             ("description", ("drawing", "form", "model", "email", "drawing_variant")),
                             ("material", ("drawing", "form", "email")),
                             ("finish", ("drawing", "form", "email")),
                             ("quantities", ("form", "email")),
                             ("annual_usage", ("form", "email")),
                             ("size", ("email", "model"))):
            if field == "description" and any(k == "drawing_variant" for k, *_ in c[field]):
                # a dash-number line: the form's description names the variant, the drawing the family
                order = ("form", "drawing_variant", "model", "email")
            if field == "part_number" and any(k == "drawing_variant" for k, *_ in c[field]):
                order = ("form", "email", "drawing_variant")
            win = _pick(c[field], order)
            if not win:
                rec[field] = _v()
                continue
            kind, value, source, conf = win
            ocr_win = conf is not None or "(OCR" in source
            # identity fields: when a clean source agrees up to OCR slips, keep its spelling
            if field in ("part_number", "rev") and ocr_win:
                for k2, v2, s2, c2 in c[field]:
                    if "(OCR" not in s2 and fold(v2) == fold(value) and v2 != value:
                        value = v2
                        break
            # a part number OCR read one character off from what a typed source says: the typed one
            if field == "part_number" and ocr_win:
                typed = [(v2, s2) for k2, v2, s2, c2 in c[field] if "(OCR" not in s2 and fold(v2) != fold(value)
                         and _near_key(fold(v2), fold(value)) and not _variant_of(value, v2)]
                if typed:
                    check.append(f"Line {n} part number: OCR read {value} in {source}, {typed[0][1]} says "
                                 f"{typed[0][0]}; used {typed[0][0]}")
                    value, source, conf = typed[0][0], typed[0][1], None
                    ocr_win = False
            if field in ("part_number", "rev"):
                for k2, v2, s2, c2 in c[field]:
                    if k2 == "drawing_variant" or v2 in (None, ""):
                        continue
                    if field == "part_number" and (_variant_of(value, v2) or (
                            "(OCR" in s2 and _near_key(fold(v2), fold(value)))):
                        continue  # a dash number of the drawing (BWM-3140-08), or the OCR slip noted above
                    if fold(v2) != fold(value):
                        if field == "rev" and ("(OCR" in s2 or ocr_win) and len(fold(v2)) == len(fold(value)) == 1 \
                                and not (("(OCR" in s2) and ocr_win):
                            # one OCR letter against a clean one: trust the clean source, say so
                            clean_v = v2 if "(OCR" not in s2 else value
                            if clean_v != value:
                                check.append(f"Line {n} rev: OCR read {value} in {source}, {s2} says {v2}; "
                                             f"used {clean_v}")
                                value, source = clean_v, s2
                            continue
                        label = "Part number" if field == "part_number" else f"Line {n} rev"
                        check.append(f"{label} differs: {source} says {value}, {s2} says {v2}; used {value}")
            if field == "material":
                for k2, v2, s2, c2 in c[field]:
                    if v2 is not value and not materials_agree(value, v2):
                        check.append(f"Line {n} material differs: {source} says {value}, {s2} says {v2}; "
                                     f"used {'the drawing' if kind == 'drawing' else source}")
            if field == "finish":
                for k2, v2, s2, c2 in c[field]:
                    if v2 is not value and not finishes_agree(value, v2):
                        check.append(f"Line {n} finish differs: {source} says {value}, {s2} says {v2}; "
                                     f"used {'the drawing' if kind == 'drawing' else source}")
            if field == "quantities":
                for k2, v2, s2, c2 in c[field]:
                    if v2 is not value and list(v2) != list(value):
                        check.append(f"Line {n} quantities differ: {source} says {' / '.join(map(str, value))}, "
                                     f"{s2} says {' / '.join(map(str, v2))}; used {source}")
                if not ln["_qty_complete"] and kind == "form":
                    check.append(f"Line {n} quantities may be incomplete: OCR could not read every break in {source}")
            if ocr_win and conf is not None and conf < LOW_OCR_CONF and field in ("part_number", "rev", "quantities"):
                confirmed = any("(OCR" not in s2 and fold(str(v2)) == fold(str(value)) for _, v2, s2, _ in c[field])
                if not confirmed:
                    check.append(f"Line {n} {field.replace('_', ' ')} read by OCR at {conf:.0f}% confidence "
                                 f"from {source}: verify")
            rec[field] = _v(value, source)
        out_lines.append(rec)

    # ---- files, missing info, routing --------------------------------------- #
    files = []
    for d in docs:
        entry = {"name": d.name, "type": d.kind, "capture": d.capture, "text_from": d.text_from, "chars": len(d.text)}
        if d.error:
            entry["error"] = d.error
        files.append(entry)
    missing = []
    if not any(ln["quantities"]["value"] or ln["annual_usage"]["value"] for ln in out_lines):
        missing.append("quantity")
    elif any(not (ln["quantities"]["value"] or ln["annual_usage"]["value"]) for ln in out_lines):
        missing.append("quantity for some lines")
    has_drawing = any(d.kind in ("drawing", "3D model") for d in docs)
    if not has_drawing and not em["drawing_links"]:
        missing.append("drawing")
    if em["drawing_links"] and not has_drawing:
        check.append("Drawings are behind a link, not attached: " + ", ".join(em["drawing_links"]))
    if em["drawing_later"] and not has_drawing:
        check.append("The sender says the drawing will follow")
    if not any(ln["material"]["value"] for ln in out_lines) and not em["drawing_links"]:
        missing.append("material")
    for d in docs:
        if d.method == "none" or not d.text:
            check.append(f"No text could be read from {d.name}" + (f" ({d.error})" if d.error else ""))

    routing = None
    if decision:
        due = decision.get("due") or {}
        routing = {"lane": decision.get("lane_name") or decision.get("lane"), "lane_id": decision.get("lane"),
                   "estimator": decision.get("owner"), "priority": decision.get("priority"),
                   "quote_by": due.get("date") if isinstance(due, dict) else None}

    seen = set()
    check = [c for c in check if not (c in seen or seen.add(c))]
    return {
        "email_id": email.get("id"), "received": email.get("received"), "subject": clean(email.get("subject")),
        "is_rfq": is_rfq(email, decision),
        "customer": customer, "customer_source": customer_src, "customer_tier": cust.get("tier") if cust else None,
        "contact": clean(email.get("from_name")) or None, "contact_email": clean(email.get("from_email")) or None,
        "rfq_number": rfq_number, "quote_ref": quote_ref, "request": em["request"],
        "respond_by": respond, "delivery": delivery, "terms": terms,
        "requirements": requirements, "export_control": export,
        "lines": out_lines, "files": files, "missing": missing, "check": check, "routing": routing,
    }


def extract_all(emails: List[Dict[str, Any]], texts_by_email: Dict[str, Dict[str, Dict[str, Any]]],
                shop: Dict[str, Any], decisions: Optional[Dict[str, Dict[str, Any]]] = None, *,
                today: Optional[dt.date] = None) -> List[Dict[str, Any]]:
    """Records for the emails that are RFQs, in inbox order."""
    out = []
    for email in emails:
        dec = (decisions or {}).get(email.get("id"))
        if not is_rfq(email, dec):
            continue
        out.append(extract(email, (texts_by_email or {}).get(email.get("id"), {}), shop, dec, today=today))
    return out


# --------------------------------------------------------------------------- #
# The consolidated file
# --------------------------------------------------------------------------- #
CSV_COLUMNS = ["Email", "Received", "Customer", "Tier", "Contact", "Contact email", "RFQ number", "Quote ref",
               "Request", "Respond by", "Delivery", "Line", "Part number", "Rev", "Description", "Material", "Finish",
               "Size", "Quantities", "Annual usage", "Export control", "Export control found in", "Requirements",
               "Files", "Missing info", "Check", "Lane", "Estimator", "Priority", "Quote by"]


def _cell(value: Any) -> str:
    """A spreadsheet-safe cell: text that starts like a formula gets a leading apostrophe."""
    if value is None:
        return ""
    s = re.sub(r"\s*[\r\n]+\s*", " ", str(value)).replace("\x00", "")  # one spreadsheet row per part line
    if s[:1] in ("=", "+", "@", "\t", "\r") or (s[:1] == "-" and len(s) > 1 and not re.match(r"-\d", s)):
        return "'" + s
    return s


def to_csv(records: List[Dict[str, Any]], bom: bool = False) -> str:
    """One row per part line, RFC 4180 (CRLF line ends, quoted where needed). bom=True puts a UTF-8
    byte order mark first so Excel opens the file as UTF-8."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    w.writerow(CSV_COLUMNS)
    for r in records:
        files = "; ".join(f"{f['name']} ({f['type']}, " + (f"{f['capture']}, " if f.get("capture") not in
                                                                (None, "digital") else "") + f"{f['text_from']})"
                          for f in r.get("files") or [])
        routing = r.get("routing") or {}
        for ln in r.get("lines") or [{}]:
            q = (ln.get("quantities") or {}).get("value")
            au = (ln.get("annual_usage") or {}).get("value")
            w.writerow([_cell(x) for x in [
                r.get("email_id"), r.get("received"), r.get("customer"), r.get("customer_tier"), r.get("contact"),
                r.get("contact_email"), (r.get("rfq_number") or {}).get("value"),
                (r.get("quote_ref") or {}).get("value"), r.get("request"), (r.get("respond_by") or {}).get("value"),
                (r.get("delivery") or {}).get("value"), ln.get("line"), (ln.get("part_number") or {}).get("value"),
                (ln.get("rev") or {}).get("value"), (ln.get("description") or {}).get("value"),
                (ln.get("material") or {}).get("value"), (ln.get("finish") or {}).get("value"),
                (ln.get("size") or {}).get("value"), " / ".join(str(x) for x in q) if q else "",
                au if au else "", (r.get("export_control") or {}).get("value"),
                (r.get("export_control") or {}).get("source"),
                "; ".join(x["value"] for x in r.get("requirements") or []), files,
                "; ".join(r.get("missing") or []), "; ".join(r.get("check") or []),
                routing.get("lane"), routing.get("estimator"), routing.get("priority"), routing.get("quote_by")]])
    text = buf.getvalue()
    return ("﻿" + text) if bom else text


def to_json(records: List[Dict[str, Any]]) -> str:
    return json.dumps({"about": "RFQ details extracted by rfq_details.py: one record per RFQ email, every value "
                                "with the place it came from.", "records": records}, indent=1, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------- #
# CLI: read the files, write the consolidated file, or grade against the answer key
# --------------------------------------------------------------------------- #
def _fallback_text(data: bytes, media: str) -> Dict[str, Any]:
    """Without ocr.py: PDF text layers through pypdf (attachments.extract_pdf_text) and STEP
    headers. Scans come back empty, which --check will show."""
    base = {"method": "none", "text": "", "confidence": None, "pages": None, "lines": [], "settings": {},
            "seconds": 0.0, "error": None}
    if media == "pdf":
        try:
            import attachments
            got = attachments.extract_pdf_text(data)
        except Exception as exc:  # noqa: BLE001
            return dict(base, error=f"could not read the PDF ({exc})")
        if len((got.get("text") or "").strip()) > 40:
            return dict(base, method="text-layer", text=got["text"], pages=got.get("pages"))
        return dict(base, error="no text layer and ocr.py is not available")
    if media == "step":
        head = data[:2_000_000].decode("latin-1", "replace")
        m = re.search(r"PRODUCT\s*\(\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'", head)
        if m:
            text = f"Part number: {m.group(1)}\nTitle: {m.group(2)}\nProduct description: {m.group(3)}"
            return dict(base, method="step-header", text=text)
    return dict(base, error="ocr.py is not available")


def load_texts(emails: List[Dict[str, Any]], cache_path: Optional[Path] = BETA_CACHE, allow_ocr: bool = True,
               texts_file: Optional[Path] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """email id -> attachment name -> file_text result, for attachments that are real files.
    The OCR cache is only read: ocr.py --build-cache owns it. A file the cache does not hold
    (or every file, when there is no cache yet) is read live by ocr.file_text."""
    pre = json.loads(Path(texts_file).read_text(encoding="utf-8")) if texts_file else None
    ocr_mod, cache = None, None
    if pre is None:
        try:
            import ocr as ocr_mod  # noqa: F811
            if cache_path and Path(cache_path).exists():
                cache = ocr_mod.OcrCache(cache_path)
        except Exception:  # noqa: BLE001 - ocr.py missing or broken: fall back to text layers
            ocr_mod = None
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for email in emails:
        for att in email.get("attachments") or []:
            if not isinstance(att, dict) or not att.get("path"):
                continue
            rel = att["path"]
            if pre is not None:
                if rel in pre:
                    out.setdefault(email["id"], {})[att["name"]] = pre[rel]
                continue
            path = DATA_DIR / rel
            try:
                data = path.read_bytes()
            except OSError as exc:
                out.setdefault(email["id"], {})[att["name"]] = {"method": "none", "text": "", "error": str(exc)}
                continue
            media = att.get("media") or path.suffix.lstrip(".").lower()
            if ocr_mod is not None:
                res = ocr_mod.file_text(data, media, att["name"], cache=cache, allow_ocr=allow_ocr)
            else:
                res = _fallback_text(data, media)
            out.setdefault(email["id"], {})[att["name"]] = res
    return out


def load_shop() -> Dict[str, Any]:
    try:
        return json.loads((HERE / "shop_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ---- grading ----------------------------------------------------------------- #
def _text_ok(got: Any, truth: Dict[str, Any]) -> bool:
    if truth.get("value") is None:
        return got in (None, "", [])
    if got in (None, ""):
        return False
    options = [truth["value"]] + list(truth.get("accept") or [])
    return any(norm(got) == norm(o) or similar(got, o) >= 0.85 for o in options)


def _id_ok(got: Any, truth: Dict[str, Any], strip_rfq: bool = False) -> bool:
    tv = truth.get("value")
    if tv is None:
        return got in (None, "")
    if not got:
        return False
    a, b = str(got).upper(), str(tv).upper()
    if strip_rfq:
        a, b = re.sub(r"^RFQ-?", "", a), re.sub(r"^RFQ-?", "", b)
    return re.sub(r"[^A-Z0-9]", "", a) == re.sub(r"[^A-Z0-9]", "", b)


def _source_kind(field: str, where: List[str], files: Dict[str, Dict[str, Any]], value: Any) -> str:
    """Which kind of source a correct answer has to come from: the one that wins by precedence."""
    if value in (None, [], ""):
        return "empty"
    typed = [(w, files.get(w, {})) for w in where if w in files]
    by_type = {"drawing": [w for w, f in typed if f.get("type") == "drawing"],
               "RFQ form": [w for w, f in typed if f.get("type") == "RFQ form"],
               "3D model": [w for w, f in typed if f.get("type") == "3D model"]}
    emailish = any(w in ("email", "subject") for w in where)
    if field in ("rfq_number", "quote_ref", "customer", "export_control", "requirements", "annual_usage"):
        if emailish:
            return "email"
        for w, f in typed:
            return f.get("text", "text layer")
        return "email"
    order = ("RFQ form", "drawing") if field in ("quantities", "respond_by") else ("drawing", "RFQ form", "3D model")
    for t in order:
        if by_type.get(t):
            return files[by_type[t][0]].get("text", "text layer")
    return "email"


def grade(records: List[Dict[str, Any]], truth: Dict[str, Any]) -> Dict[str, Any]:
    """Field-by-field comparison with the answer key."""
    files = truth.get("files") or {}
    by_id = {r["email_id"]: r for r in records}
    rows: List[Dict[str, Any]] = []

    def add(eid: str, field: str, ok: bool, kind: str, got: Any, want: Any, attach_only: bool) -> None:
        rows.append({"email": eid, "field": field, "ok": bool(ok), "kind": kind, "got": got, "want": want,
                     "attachment_only": attach_only})

    for eid, t in (truth.get("rfqs") or {}).items():
        r = by_id.get(eid)
        if r is None:
            add(eid, "is_rfq", False, "email", "skipped", "RFQ", False)
            continue
        for field in ("customer",):
            add(eid, field, _text_ok(r.get(field), {"value": t[field]}), "email", r.get(field), t[field], False)
        add(eid, "customer_tier", r.get("customer_tier") == t.get("customer_tier"), "email", r.get("customer_tier"),
            t.get("customer_tier"), False)
        for field in ("rfq_number", "quote_ref"):
            tv = t[field]
            got = (r.get(field) or {}).get("value")
            where = tv.get("where") or []
            add(eid, field, _id_ok(got, tv, strip_rfq=field == "rfq_number"),
                _source_kind(field, where, files, tv.get("value")), got, tv.get("value"),
                bool(where) and not any(w in ("email", "subject") for w in where))
        add(eid, "request", r.get("request") == t.get("request"), "email", r.get("request"), t.get("request"), False)
        tv = t["respond_by"]
        got = (r.get("respond_by") or {}).get("value")
        where = tv.get("where") or []
        add(eid, "respond_by", got == tv.get("value"), _source_kind("respond_by", where, files, tv.get("value")),
            got, tv.get("value"), bool(where) and not any(w in ("email", "subject") for w in where))
        tv = t["export_control"]
        got = (r.get("export_control") or {}).get("value")
        where = tv.get("where") or []
        add(eid, "export_control", (got or None) == tv.get("value"),
            _source_kind("export_control", where, files, tv.get("value")), got, tv.get("value"),
            bool(where) and not any(w in ("email", "subject") for w in where))
        # requirements: recall of the answer key's items, one extracted item per key item (so two
        # items run together count as one hit and one miss), and the extracted items that match none
        got_reqs = [x["value"] for x in r.get("requirements") or []]
        taken: set = set()
        for item in t.get("requirements") or []:
            scored = [(overlap(item["value"], g), j) for j, g in enumerate(got_reqs) if j not in taken]
            best = max(scored, default=(0.0, None))
            ok = best[0] >= 0.7
            if ok:
                taken.add(best[1])
            add(eid, "requirements", ok, _source_kind("requirements", item.get("where") or [], files, item["value"]),
                "" if ok else "; ".join(got_reqs)[:120], item["value"],
                not any(w in ("email", "subject") for w in item.get("where") or []))
        extra = [g for j, g in enumerate(got_reqs) if j not in taken]
        for g in extra:
            rows.append({"email": eid, "field": "requirements (extra)", "ok": False, "kind": "extra", "got": g,
                         "want": None, "attachment_only": False})
        add(eid, "missing", sorted(r.get("missing") or []) == sorted(t.get("missing") or []), "email",
            r.get("missing"), t.get("missing"), False)
        # lines: pair by part number, then by position
        got_lines = list(r.get("lines") or [])
        used = set()
        for k, tl in enumerate(t.get("lines") or []):
            tpn = (tl["part_number"] or {}).get("value")
            match = None
            for j, gl in enumerate(got_lines):
                if j in used:
                    continue
                if tpn and fold((gl.get("part_number") or {}).get("value") or "") == fold(tpn):
                    match = j
                    break
            if match is None and k < len(got_lines) and k not in used and not tpn:
                match = k
            if match is None:
                match = next((j for j in range(len(got_lines)) if j not in used and
                              not (got_lines[j].get("part_number") or {}).get("value")), None)
            gl = got_lines[match] if match is not None else {}
            if match is not None:
                used.add(match)
            for field in ("part_number", "rev", "description", "material", "finish", "quantities", "annual_usage"):
                tv = tl.get(field) or {"value": None}
                got = (gl.get(field) or {}).get("value")
                if field in ("part_number", "rev"):
                    ok = _id_ok(got, tv)
                elif field in ("quantities", "annual_usage"):
                    ok = (got or None) == tv.get("value")
                else:
                    ok = _text_ok(got, tv)
                where = tv.get("where") or []
                add(eid, field, ok, _source_kind(field, where, files, tv.get("value")), got, tv.get("value"),
                    bool(where) and not any(w in ("email", "subject") for w in where))
                rows[-1]["accept"] = list(tv.get("accept") or [])
        for j, gl in enumerate(got_lines):
            if j not in used:
                rows.append({"email": eid, "field": "lines (extra)", "ok": False, "kind": "extra",
                             "got": (gl.get("part_number") or {}).get("value"), "want": None,
                             "attachment_only": False})
        # file types
        for f in r.get("files") or []:
            tf = files.get(f["name"])
            if tf:
                add(eid, "file type", f["type"] == tf["type"], tf.get("text", "text layer"), f["type"], tf["type"],
                    False)
    for eid in truth.get("not_rfq") or []:
        if eid in by_id:
            add(eid, "is_rfq", False, "email", "RFQ", "not an RFQ", False)
    return {"rows": rows}


def print_grade(result: Dict[str, Any], verbose: bool = False) -> Tuple[int, int]:
    rows = result["rows"]
    scored = [r for r in rows if r["kind"] != "extra"]
    fields = []
    for r in scored:
        if r["field"] not in fields:
            fields.append(r["field"])
    kinds = ["email", "text layer", "STEP header", "OCR", "empty"]
    print(f"{'field':<16}{'all':>12}" + "".join(f"{k:>14}" for k in kinds))
    for f in fields:
        sub = [r for r in scored if r["field"] == f]
        line = f"{f:<16}{_frac(sub):>12}"
        for k in kinds:
            ks = [r for r in sub if r["kind"] == k]
            line += f"{_frac(ks) if ks else '-':>14}"
        print(line)
    line = f"{'TOTAL':<16}{_frac(scored):>12}"
    for k in kinds:
        ks = [r for r in scored if r["kind"] == k]
        line += f"{_frac(ks) if ks else '-':>14}"
    print(line)
    att = [r for r in scored if r["attachment_only"]]
    print(f"\nvalues printed only on attachments: {_frac(att)}"
          f"   (OCR-only: {_frac([r for r in att if r['kind'] == 'OCR'])})")
    # the text fields pass at 85% similarity; this says how many are word for word
    words = [dict(r, ok=any(norm(r["got"]) == norm(w) for w in [r["want"]] + r.get("accept", []))) for r in scored
             if r["field"] in ("description", "material", "finish") and r["want"] is not None]
    print("word for word (description, material, finish): " + "   ".join(
        f"{k} {_frac([r for r in words if r['kind'] == k])}" for k in kinds if any(r["kind"] == k for r in words)))
    extras = [r for r in rows if r["kind"] == "extra"]
    print(f"extra items not in the answer key: {len(extras)}"
          + (" (" + ", ".join(f"{r['email']} {r['field']}" for r in extras[:12]) + ")" if extras else ""))
    wrong = [r for r in scored if not r["ok"]]
    if wrong:
        print(f"\n{len(wrong)} wrong:")
        for r in wrong if verbose else wrong[:40]:
            print(f"  {r['email']:<4} {r['field']:<15} [{r['kind']}] got {str(r['got'])[:70]!r} want {str(r['want'])[:70]!r}")
    return sum(r["ok"] for r in scored), len(scored)


def _frac(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "-"
    ok = sum(r["ok"] for r in rows)
    return f"{ok}/{len(rows)} {100 * ok / len(rows):.0f}%"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract RFQ details into one consolidated file.")
    ap.add_argument("--emails", default=str(BETA_EMAILS), help="inbox JSON (default: the beta inbox)")
    ap.add_argument("--out", default=str(BETA_OUT), help="output path without extension (.csv and .json)")
    ap.add_argument("--cache", default=str(BETA_CACHE), help="OCR cache (default: data/rfq_beta/ocr_cache.json)")
    ap.add_argument("--no-ocr", action="store_true", help="never run tesseract; use the cache and text layers only")
    ap.add_argument("--today", default=SAMPLE_INBOX_DATE.isoformat(), help="date the emails arrived (YYYY-MM-DD)")
    ap.add_argument("--texts", help="precomputed file_text results keyed by path under data/ (for evaluation)")
    ap.add_argument("--check", action="store_true", help="grade against tests/rfq_beta_fields_truth.json")
    ap.add_argument("--verbose", action="store_true", help="with --check, list every wrong field")
    args = ap.parse_args(argv)
    today = dt.date.fromisoformat(args.today)
    inbox = json.loads(Path(args.emails).read_text(encoding="utf-8"))
    emails = inbox.get("emails", inbox) if isinstance(inbox, dict) else inbox
    texts = load_texts(emails, Path(args.cache) if args.cache else None, allow_ocr=not args.no_ocr,
                       texts_file=Path(args.texts) if args.texts else None)
    records = extract_all(emails, texts, load_shop(), today=today)
    if args.check:
        truth = json.loads(FIELDS_TRUTH.read_text(encoding="utf-8"))
        ok, total = print_grade(grade(records, truth), verbose=args.verbose)
        print(f"\n{ok}/{total} fields correct ({100 * ok / max(1, total):.1f}%)")
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # plain UTF-8: the server adds the byte order mark Excel wants when it sends the download
    out.with_suffix(".csv").write_text(to_csv(records), encoding="utf-8", newline="")
    out.with_suffix(".json").write_text(to_json(records), encoding="utf-8")
    n_lines = sum(len(r["lines"]) for r in records)
    print(f"{len(records)} RFQs, {n_lines} part lines -> {out.with_suffix('.csv')} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
