"""
RFQ routing for a CNC job shop.

Two halves, on purpose:
  1. JEV decides the judgment calls. Every email gets one Jev request with eight
     questions (what kind of email, which estimator, export-controlled or not,
     urgency, and so on). Jev reads the email and the text of its attachments
     (drawing notes, title blocks, legends, RFQ forms) and answers all of the
     questions in parallel with probabilities.
  2. CODE decides what happens. Thresholds, customer lookups, attachment checks,
     priorities, SLAs, and reply templates are plain Python you can read and change.

Jev never writes text, so every draft reply below is a template filled in by code.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Dict, List, Optional

import attachments as att_mod

QUESTION_SET_VERSION = "rfq-cnc-v2"

# --------------------------------------------------------------------------- #
# What we ask Jev about every email
# --------------------------------------------------------------------------- #
EMAIL_TYPES = {
    "new_rfq": "The sender asks for a price quote on parts for the shop to machine. "
               "This includes first-time requests and requests to quote a part again.",
    "quote_revision": "The sender sends changes to a quote the shop is already working on or "
                      "has already sent, such as a new drawing revision or new quantities, "
                      "and asks for an updated price.",
    "purchase_order": "The sender places an order or awards a quote: a purchase order, "
                      "a PO number to start work, or an acceptance of a quote.",
    "order_followup": "The sender asks about an existing order: status, ship date, tracking, "
                      "invoices, material certifications, or inspection reports.",
    "vendor_or_solicitation": "The sender is selling something to the shop (tools, materials, "
                              "software, services, marketing), or is a recruiter or job seeker.",
    "other": "Anything else: newsletters, internal notes, personal messages, or spam.",
}

EMAIL_TYPE_LABELS = {
    "new_rfq": "New RFQ",
    "quote_revision": "Quote revision",
    "purchase_order": "Purchase order",
    "order_followup": "Order follow-up",
    "vendor_or_solicitation": "Vendor / solicitation",
    "other": "Other",
}

MIXED_OPTION = {
    "mixed_or_unclear": "The main work is a different process, such as welding, fabrication, "
                        "sheet metal, casting, or assembly, or the email does not describe the "
                        "parts well enough to tell which machine would make them.",
}

VOLUME_LABELS = {
    "prototype": "Prototype",
    "low_volume": "Low volume",
    "production": "Production",
    "not_stated": "Qty not stated",
}

QUESTION_LABELS = {
    "email_type": "What kind of email is this?",
    "process": "Which estimator should quote it?",
    "export_controlled": "Export-controlled (ITAR / CUI)?",
    "urgency": "How urgent?",
    "quantity_given": "Quantity stated?",
    "drawings_provided": "Drawings or CAD provided?",
    "outside_processing": "Needs finishing or outside processing?",
    "volume": "Production volume",
}

DEFAULT_THRESHOLDS = {
    # Confidence below these sends the email to a human instead of auto-routing.
    "email_type_confidence": 0.5,
    "process_confidence": 0.5,
    # Export control: route to the restricted queue at a LOW bar, because a miss is costly.
    "export_control_restrict": 0.5,
    # A vendor pitch that merely mentions ITAR skips the restricted queue only when Jev is this
    # sure it is a vendor pitch. Deliberately separate from the adjustable auto-route bar.
    "itar_exempt_confidence": 0.8,
    "export_control_warn": 0.2,
    # Flags
    "rush_score": 2.3,        # urgency is 0..3; 3 = line down / AOG / today
    "soon_score": 1.5,
    "quantity_given": 0.5,
    "drawings_provided": 0.5,
    "outside_processing": 0.6,
    "volume_confidence": 0.35,
}


def build_questions(shop: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The eight Jev questions, as raw TypeSafe System One question objects."""
    process_options = {lane["id"]: lane["jev_description"] for lane in shop["lanes"]["estimating"]}
    process_options.update(MIXED_OPTION)
    return {
        "email_type": {
            "type": "choice",
            "instructions": "This email arrived in the shared quoting inbox of a CNC machine shop. "
                            "What is the sender asking the shop to do?",
            "criteria": dict(EMAIL_TYPES),
        },
        "process": {
            "type": "choice",
            "instructions": "Which estimator at the machine shop should quote the parts in this email? "
                            "Decide from the part names, descriptions, materials, drawing file names, "
                            "and the attached drawings.",
            "criteria": process_options,
        },
        "export_controlled": {
            "type": "noul",
            "instructions": "Do the email or its attachments say the parts, drawings, or technical data "
                            "are export-controlled, ITAR, or controlled unclassified information (CUI)? "
                            "Check the drawing legends, title blocks, and RFQ forms as well as the email text.",
            "criteria": {
                "true": "The email or an attachment mentions ITAR, EAR or export control, an ECCN, CUI, "
                        "DFARS 252.204-7012, controlled technical data, U.S. persons only, or defense "
                        "drawings with access restrictions.",
                "false": "Neither the email nor its attachments mention export control, ITAR, EAR, CUI, "
                         "or limits on who may see the technical data. An ordinary proprietary or "
                         "confidentiality notice alone does not count.",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How fast does the sender need a reply or a quote?",
            "criteria": [
                "No deadline mentioned, or a budgetary or planning request.",
                "Normal turnaround: a due date a week or more away, or a standard request.",
                "Soon: the sender wants a quote or answer within a few days or by the end of the week.",
                "Emergency: rush, expedite, ASAP, line down, AOG, or needed today or tomorrow.",
            ],
        },
        "quantity_given": {
            "type": "noul",
            "instructions": "Do the email or its attachments state how many parts to quote? "
                            "Check an attached RFQ form or purchase order as well as the email text.",
            "criteria": {
                "true": "A quantity, quantity breaks (for example 10 / 25 / 50), or an annual usage is "
                        "given in the email or in an attachment.",
                "false": "Neither the email nor its attachments give a quantity for the parts.",
            },
        },
        "drawings_provided": {
            "type": "noul",
            "instructions": "Do the email or its attachments provide part drawings or CAD models?",
            "criteria": {
                "true": "Drawings, prints, or CAD models (PDF, STEP, IGES, SolidWorks) are attached, "
                        "linked, or shared through a portal.",
                "false": "No drawings are provided yet, for example the drawing will follow later, "
                         "drawings are not mentioned, or the only attachments are forms, orders, or "
                         "other documents.",
            },
        },
        "outside_processing": {
            "type": "noul",
            "instructions": "Do the email or its attachments call for a finish or outside processing "
                            "after machining? Check the finish block and notes on attached drawings.",
            "criteria": {
                "true": "Anodizing, plating, passivation, heat treating, painting, powder coating, "
                        "chem film, or another finish or treatment is required by the email or an "
                        "attachment.",
                "false": "Neither the email nor its attachments call for a finish or post-machining "
                         "treatment (a drawing finish of NONE counts as no finish).",
            },
        },
        "volume": {
            "type": "choice",
            "instructions": "What production volume is the sender asking about?",
            "criteria": {
                "prototype": "Prototypes or a handful of parts, about 1 to 10 pieces.",
                "low_volume": "A small batch, about 11 to 250 pieces.",
                "production": "Production quantities: more than 250 pieces, blanket orders, or annual usage.",
                "not_stated": "No quantity is given.",
            },
        },
    }


def jev_state(email: Dict[str, Any]) -> Dict[str, Any]:
    """Only what the questions need: sender, subject, body, and each attachment's name and text.

    Attachment text comes from attachments.py: sample files from their specs (title block, notes,
    legends, RFQ-form quantities), uploaded PDFs from pypdf. It has its own size budget.
    """
    entries = att_mod.jev_entries(email.get("attachments") or [])
    body = (email.get("body") or "").strip()
    if len(body) > 8000:  # keep the state small; Jev accuracy drops with filler
        body = body[:8000] + " [truncated]"
    return {
        "from": f"{email.get('from_name', '').strip()} <{email.get('from_email', '').strip()}>".strip(),
        "subject": (email.get("subject") or "").strip(),
        "body": body,
        "attachments": entries if entries else "none",
    }


# --------------------------------------------------------------------------- #
# Deterministic helpers (code, not Jev)
# --------------------------------------------------------------------------- #
CAD_EXTENSIONS = {".step", ".stp", ".igs", ".iges", ".sldprt", ".sldasm", ".x_t", ".x_b",
                  ".dxf", ".dwg", ".prt", ".ipt", ".iam", ".stl", ".3mf", ".par", ".catpart"}
NOT_A_DRAWING = re.compile(r"(^|[^a-z])(po|purchase|invoice|cert|certs|coc|resume|cv|brochure|catalog|quote_form|"
                           r"rfq|form|linecard|newsletter|packing|slip|statement|remit|ncr|rma|terms)([^a-z]|$)",
                           re.IGNORECASE)
EXPORT_MARKINGS = re.compile(r"\b(itar|arms export control|export administration regulations|eccn|cui|"
                             r"controlled unclassified|dfars 252\.204-7012|u\.s\. persons only)\b", re.IGNORECASE)


def cad_attachments(attachments: List[Any]) -> List[str]:
    """Attachments that code recognizes as drawings or CAD. Sample files know their kind;
    plain file names and uploads are judged by extension and name."""
    found = []
    for raw in attachments or []:
        att = att_mod.normalize(raw)
        name, kind = att["name"], att.get("kind")
        if kind in ("drawing", "model"):
            found.append(name)
            continue
        if kind in ("rfq_form", "po", "document"):
            continue
        if kind == "upload" and att.get("media") != "pdf":
            continue
        lower = name.lower().strip()
        ext = "." + lower.rsplit(".", 1)[-1] if "." in lower else ""
        if ext in CAD_EXTENSIONS or (ext == ".pdf" and not NOT_A_DRAWING.search(lower)):
            found.append(name)
    return found


def export_marked_attachments(attachments: List[Any]) -> List[str]:
    """Attachments whose text carries an export-control marking. Code uses this only to explain
    a decision ("the marking is on the drawing"); Jev makes the call."""
    found = []
    for raw in attachments or []:
        att = att_mod.normalize(raw)
        if EXPORT_MARKINGS.search(att_mod.jev_text(att)):
            found.append(att["name"])
    return found


def lookup_customer(from_email: str, shop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    domain = (from_email or "").split("@")[-1].lower().strip()
    for customer in shop.get("customers", []):
        if customer["domain"].lower() == domain:
            return customer
    return None


def add_business_days(start: dt.date, days: int) -> dt.date:
    current = start
    added = 0
    while added < days:
        current += dt.timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    while current.weekday() >= 5:  # never land on a weekend
        current += dt.timedelta(days=1)
    return current


def friendly_date(day: dt.date, today: dt.date) -> str:
    if day == today:
        return "today"
    if day == today + dt.timedelta(days=1):
        return "tomorrow"
    return f"{day.strftime('%a')}, {day.strftime('%b')} {day.day}"


GENERIC_NAME_WORDS = {
    "quality", "supplier", "purchasing", "procurement", "accounts", "accounting", "sales", "team",
    "the", "unknown", "customer", "info", "admin", "support", "buyer", "receiving", "shipping",
    "engineering", "department", "dept", "office", "orders", "service", "noreply", "no-reply",
}


def first_name(name: str) -> str:
    cleaned = re.sub(r"^(dr|mr|mrs|ms)\.?\s+", "", (name or "").strip(), flags=re.IGNORECASE)
    words = [w.lower().strip(".,") for w in cleaned.split()]
    if not words or any(w in GENERIC_NAME_WORDS for w in words) or "@" in cleaned:
        return "there"
    return cleaned.split()[0].strip(".,")


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{round(x * 100)}%"


def top_two(answer: Dict[str, Any], labels: Dict[str, str]) -> str:
    probs = sorted(answer.get("probabilities", {}).items(), key=lambda kv: kv[1], reverse=True)[:2]
    return " vs ".join(f"{labels.get(k, k)} {pct(v)}" for k, v in probs)


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #
def decide(email: Dict[str, Any], answers: Dict[str, Dict[str, Any]], shop: Dict[str, Any],
           thresholds: Optional[Dict[str, float]] = None, today: Optional[dt.date] = None) -> Dict[str, Any]:
    th = dict(DEFAULT_THRESHOLDS)
    th.update(thresholds or {})
    today = today or dt.date.today()
    lanes = lane_index(shop)
    process_labels = {lane["id"]: lane["name"] for lane in shop["lanes"]["estimating"]}
    process_labels["mixed_or_unclear"] = "Mixed / unclear"

    et = answers["email_type"]
    pr = answers["process"]
    ex = answers["export_controlled"]
    ur = answers["urgency"]
    qty = answers["quantity_given"]
    drw = answers["drawings_provided"]
    osp = answers["outside_processing"]
    vol = answers["volume"]

    trace: List[Dict[str, str]] = []
    flags: List[Dict[str, str]] = []

    def step(label: str, detail: str, outcome: str = "info") -> None:
        trace.append({"label": label, "detail": detail, "outcome": outcome})

    customer = lookup_customer(email.get("from_email", ""), shop)
    cad = cad_attachments(email.get("attachments") or [])
    et_label = EMAIL_TYPE_LABELS.get(et["choice"], et["choice"])
    et_conf = et.get("confidence") or 0.0
    type_confident = et_conf >= th["email_type_confidence"]
    is_rfq = et["choice"] in ("new_rfq", "quote_revision")
    meter = {"label": "Confidence", "value": et_conf, "threshold": th["email_type_confidence"]}
    lane_id = "review"
    reason = ""

    # 1. Export control first, at a low bar: a miss is far costlier than a false alarm.
    #    A confident vendor pitch that merely mentions ITAR does not count.
    restricted = ex["p"] >= th["export_control_restrict"] and not (
        et_conf >= max(th["itar_exempt_confidence"], th["email_type_confidence"])
        and et["choice"] in ("vendor_or_solicitation", "other"))
    marked = export_marked_attachments(email.get("attachments") or [])
    where = ""
    if marked and ex["p"] >= th["export_control_warn"]:
        where = " Marking found in the attachment" + ("s " if len(marked) > 1 else " ") + ", ".join(marked) + "."
    step("Export-controlled?",
         f"Jev: {pct(ex['p'])} likely ITAR / CUI. Restricted queue at "
         f"{pct(th['export_control_restrict'])} or above.{where}",
         "stop" if restricted else "pass")

    # 2. What kind of email is it?
    if restricted:
        step("What kind of email?",
             f"Jev: {et_label} ({pct(et.get('p'))}). Restricted handling applies either way.", "info")
    else:
        step("What kind of email?",
             f"Jev: {et_label} ({pct(et.get('p'))}), confidence {pct(et_conf)}. "
             f"Auto-route bar is {pct(th['email_type_confidence'])}.",
             "pass" if type_confident else "stop")

    if restricted:
        lane_id = "itar"
        reason = ("Export-controlled technical data. Only U.S. persons may handle it, "
                  "so it skips the shared queues.")
        meter = {"label": "ITAR / CUI likelihood", "value": ex["p"],
                 "threshold": th["export_control_restrict"]}
    elif not type_confident:
        lane_id = "review"
        reason = f"Jev is unsure what kind of email this is: {top_two(et, EMAIL_TYPE_LABELS)}."
    elif et["choice"] in ("vendor_or_solicitation", "other"):
        lane_id = "filtered"
        reason = f"{et_label}, not customer work."
    elif et["choice"] in ("purchase_order", "order_followup"):
        lane_id = "orders"
        reason = f"{et_label} for an existing job."
    else:
        # 3. RFQ: which estimator?
        process_label = process_labels.get(pr["choice"], pr["choice"])
        pr_conf = pr.get("confidence") or 0.0
        process_ok = pr["choice"] != "mixed_or_unclear" and pr_conf >= th["process_confidence"]
        step("Which estimator?",
             f"Jev: {process_label} ({pct(pr.get('p'))}), confidence {pct(pr_conf)}. "
             f"Auto-route bar is {pct(th['process_confidence'])}.",
             "pass" if process_ok else "stop")
        weakest = min(et_conf, pr_conf)
        meter = {"label": "Confidence", "value": weakest,
                 "threshold": th["process_confidence"] if pr_conf <= et_conf else th["email_type_confidence"]}
        if pr["choice"] == "mixed_or_unclear":
            lane_id = "review"
            reason = "Mixed processes or not enough detail to pick an estimator."
        elif pr_conf < th["process_confidence"]:
            lane_id = "review"
            reason = f"Jev is torn between estimators: {top_two(pr, process_labels)}."
        else:
            lane_id = pr["choice"]
            reason = f"{process_label} work, {pct(pr.get('p'))} probability."

    # ---- flags ------------------------------------------------------------ #
    missing: List[str] = []
    if is_rfq:
        if not cad and drw["p"] < th["drawings_provided"]:
            missing.append("drawings (PDF plus STEP if you have it)")
            flags.append({"id": "missing_drawings", "label": "Missing drawings", "tone": "warning"})
        if qty["p"] < th["quantity_given"]:
            missing.append("the quantities you want quoted")
            flags.append({"id": "missing_qty", "label": "Missing quantity", "tone": "warning"})
        if osp["p"] >= th["outside_processing"]:
            flags.append({"id": "outside_processing", "label": "Outside processing", "tone": "neutral"})
        if (vol.get("confidence") or 0) >= th["volume_confidence"] and vol["choice"] != "not_stated":
            flags.append({"id": "volume", "label": VOLUME_LABELS.get(vol["choice"], vol["choice"]),
                          "tone": "neutral"})
        step("Anything missing?",
             ("CAD / drawing files attached: " + (", ".join(cad) if cad else "none") +
              f". Jev: drawings provided {pct(drw['p'])}, quantity stated {pct(qty['p'])}."),
             "warn" if missing else "pass")
    if lane_id == "itar":
        flags.insert(0, {"id": "itar", "label": "ITAR / CUI", "tone": "restricted"})
    elif ex["p"] >= th["export_control_warn"]:
        flags.insert(0, {"id": "itar_check", "label": "Check export control", "tone": "warning"})

    rush = ur["score"] >= th["rush_score"]
    soon = ur["score"] >= th["soon_score"]
    if rush and lane_id != "filtered":
        flags.insert(0, {"id": "rush", "label": "Rush", "tone": "serious"})

    # ---- priority + SLA (code) ------------------------------------------- #
    priority = None
    priority_reason = ""
    tier = customer.get("tier") if customer else None
    production = vol["choice"] == "production" and (vol.get("confidence") or 0) >= th["volume_confidence"]
    if lane_id == "filtered":
        priority = None
    elif rush:
        priority, priority_reason = "P1", "rush language"
    elif is_rfq and tier == "A" and production:
        priority, priority_reason = "P1", "A-tier customer and production volume"
    elif lane_id == "orders" or (lane_id == "itar" and et["choice"] in ("purchase_order", "order_followup")):
        priority, priority_reason = "P2", "customer is waiting on an existing order"
    elif soon or tier == "A" or production:
        why = []
        if soon:
            why.append("short turnaround")
        if tier == "A":
            why.append("A-tier customer")
        if production:
            why.append("production volume")
        priority, priority_reason = "P2", ", ".join(why)
    else:
        priority, priority_reason = "P3", "standard request"

    sla_days = shop.get("sla_business_days", {"P1": 0, "P2": 2, "P3": 5})
    due = None
    if priority:
        days = int(sla_days.get(priority, 5))
        reply_only = lane_id == "orders" or (lane_id == "itar" and not is_rfq)
        if reply_only:
            days = 0
        due_date = add_business_days(today, days) if days > 0 else today
        due = {"date": due_date.isoformat(), "label": friendly_date(due_date, today),
               "kind": "reply" if reply_only else "quote"}
    if priority:
        step("Priority", f"{priority}: {priority_reason}." +
             (f" {due['kind'].capitalize()} due {due['label']}." if due else ""), "info")

    lane = lanes[lane_id]
    action = suggest_action(email, lane_id, lane, et, pr, missing, due, shop, process_labels, customer)
    step("Route", f"{lane['name']} ({lane['owner']}). {reason}",
         "review" if lane_id == "review" else "done")

    expected = (email.get("expected") or {}).get("lane")
    return {
        "lane": lane_id,
        "lane_name": lane["name"],
        "owner": lane["owner"],
        "auto_routed": lane_id != "review",
        "reason": reason,
        "meter": meter,
        "is_rfq": is_rfq,
        "flags": flags,
        "missing": missing,
        "priority": priority,
        "priority_reason": priority_reason,
        "due": due,
        "customer": customer,
        "cad_attachments": cad,
        "export_marked": marked,
        "process_guess": {"choice": pr["choice"], "label": process_labels.get(pr["choice"], pr["choice"]),
                          "p": pr.get("p")},
        "trace": trace,
        "action": action,
        "expected_lane": expected,
        "matches_expected": (expected == lane_id) if expected else None,
    }


def lane_index(shop: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lanes = {lane["id"]: dict(lane, group="estimating") for lane in shop["lanes"]["estimating"]}
    for key in ("itar", "orders", "review", "filtered"):
        lane = shop["lanes"][key]
        lanes[lane["id"]] = dict(lane, group="other" if key != "itar" else "estimating")
    return lanes


def suggest_action(email, lane_id, lane, et, pr, missing, due, shop, process_labels, customer):
    """Code-generated next step and draft reply. Jev decides, templates write."""
    shop_name = shop["shop"]["name"]
    hi = f"Hi {first_name(email.get('from_name', ''))},"
    raw_subject = (email.get("subject") or "").strip()
    received = f"your email \"{raw_subject}\"" if raw_subject and raw_subject != "(no subject)" else "your email"
    owner_first = lane["owner"].split(" ")[0]
    sign = f"Best,\n{shop_name} Quoting Team"

    if lane_id == "filtered":
        return {"title": "Archive. No reply needed.", "to": None, "draft": None}

    if lane_id == "orders":
        if et["choice"] == "purchase_order":
            title = f"Enter the PO in the ERP and send an order acknowledgment ({lane['owner']})"
            body = (f"{hi}\n\nThanks for the order. We received {received} and {owner_first} is entering it now. "
                    f"We'll confirm the ship date by end of day.\n\n{sign.replace('Quoting', 'Customer Service')}")
        else:
            title = f"Look up the order in the ERP and reply ({lane['owner']})"
            body = (f"{hi}\n\nThanks for checking in. {owner_first} is pulling up the order details now and will "
                    f"get back to you by end of day with an update.\n\n{sign.replace('Quoting', 'Customer Service')}")
        return {"title": title, "to": email.get("from_email"), "draft": body}

    if lane_id == "itar":
        title = (f"Move to the restricted ITAR folder and notify {lane['owner']}. "
                 f"Do not forward attachments outside the controlled environment.")
        if et["choice"] in ("new_rfq", "quote_revision"):
            body = (f"{hi}\n\nThanks for the RFQ. We received {received} and it is being handled by our "
                    f"export-controlled quoting team (U.S. persons only)." +
                    (f" You'll have our quote by {due['label']}." if due else "") + f"\n\n{sign}")
        elif et["choice"] in ("purchase_order", "order_followup"):
            body = (f"{hi}\n\nThanks for your note. We received {received} and our export-controlled "
                    f"team (U.S. persons only) will get back to you by end of day.\n\n"
                    f"{sign.replace('Quoting', 'Customer Service')}")
        else:
            title = (f"{lane['owner']} reviews this in the restricted ITAR folder before anyone replies. "
                     f"Do not forward it outside the controlled environment.")
            body = None
        return {"title": title, "to": email.get("from_email") if body else None, "draft": body}

    if lane_id == "review":
        guess = top_two(pr, process_labels) if et["choice"] in ("new_rfq", "quote_revision") \
            else top_two(et, EMAIL_TYPE_LABELS)
        title = f"{lane['owner']} ({lane['role']}) confirms where this goes. Jev's best guesses: {guess}."
        return {"title": title, "to": None, "draft": None}

    # Estimating lane
    ask = ""
    if missing:
        ask = (" To give you an accurate price, could you send " + " and ".join(missing) + "?")
    promise = f" You'll have our quote by {due['label']}." if (due and not missing) else ""
    if missing:
        title = f"Assign to {lane['owner']} and request the missing information"
    else:
        title = f"Assign to {lane['owner']} and send the acknowledgment"
    body = (f"{hi}\n\nThanks for the RFQ. We received {received} and {owner_first} on our "
            f"{lane['name']} team is reviewing it now.{ask}{promise}\n\n{sign}")
    return {"title": title, "to": email.get("from_email"), "draft": body}
