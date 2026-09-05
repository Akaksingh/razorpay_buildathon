"""Print-ready rendering of a CE3.0 evidence packet.

The compiler's JSON is the contract: the console and any acquirer
integration consume it, and the packet hash is computed over it. But the
people who decide a dispute -- the merchant's case handler, the acquirer's
chargeback desk, the issuer's reviewer -- read documents, not JSON. This
module turns one packet dict into one self-contained HTML document that can
be printed, e-mailed, or dropped into a case-management system that strips
scripts and external assets.

Two properties matter more than the styling:

  * The document is a VIEW of the packet, never a second source of truth.
    Every value on the page is copied from the packet dict; nothing is
    recomputed, so the SHA-256 printed at the bottom is the hash of the JSON
    the page was rendered from and a reviewer can re-derive it from the raw
    packet alone.
  * Rendering is a pure function of (packet, generated_at). Same packet,
    same timestamp, byte-identical document -- which is what makes the
    "generated deterministically" footer a testable claim rather than a
    slogan.

No templating engine: the packet has a fixed shape, and an f-string per
section keeps the escaping visible at every interpolation point.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

ELEMENT_LABELS = {
    "ip_hash": "IP address (hashed)",
    "device_hash": "Device fingerprint (hashed)",
    "addr_hash": "Shipping address (normalised, hashed)",
    "account_id": "Account / login ID",
}
CRITERION_LABELS = {
    "reason_code_10_4": "Dispute is Visa reason code 10.4 (card-absent fraud)",
    "prior_txns_in_120_365d_window": "At least two prior transactions on the credential 120 to 365 days before the dispute",
    "prior_txns_undisputed": "At least two of those prior transactions were never disputed",
    "prior_txns_with_2_matching_elements_incl_ip_or_device":
        "At least two prior transactions share two or more data elements, one being IP address or device ID",
}
DEFAULT_MERCHANT = "Merchant of record (acquirer MID on file)"

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #ffffff; color: #0b0b0b; font: 13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.page { max-width: 820px; margin: 0 auto; padding: 32px 36px 40px; }
h1 { font-size: 21px; margin: 0 0 2px; letter-spacing: -0.01em; }
h2 { font-size: 14px; margin: 26px 0 8px; text-transform: uppercase; letter-spacing: 0.06em; color: #52514e; }
h3 { font-size: 13.5px; margin: 14px 0 6px; }
.sub { color: #52514e; margin: 0 0 14px; }
.status { display: inline-block; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 12px; letter-spacing: 0.04em; }
.status.ok { background: #e2f5e2; color: #006300; border: 1px solid #0ca30c; }
.status.no { background: #fbe6e6; color: #8f1f1f; border: 1px solid #d03b3b; }
.kv { display: grid; grid-template-columns: 170px 1fr; gap: 4px 14px; margin: 12px 0; padding: 12px 14px; border: 1px solid #e1e0d9; border-radius: 8px; }
.kv dt { color: #52514e; } .kv dd { margin: 0; }
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #e1e0d9; vertical-align: top; }
th { font-weight: 600; color: #52514e; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.04em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pass, .fail { font-weight: 700; white-space: nowrap; }
.pass { color: #006300; } .fail { color: #b22a2a; }
.mono { font: 11.5px ui-monospace, Menlo, Consolas, monospace; word-break: break-all; }
.txn { border: 1px solid #e1e0d9; border-radius: 8px; padding: 10px 14px 12px; margin: 10px 0; break-inside: avoid; }
.txn .head { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 6px; }
.txn .head b { font-size: 13.5px; }
tr.match td { background: #fff4d6; }
tr.match td:first-child { border-left: 3px solid #eda100; }
.tag { display: inline-block; font-size: 10.5px; font-weight: 700; padding: 1px 6px; border-radius: 4px; background: #eda100; color: #0b0b0b; margin-left: 6px; vertical-align: middle; }
.narrative { border-left: 3px solid #c3c2b7; padding: 6px 12px; margin: 8px 0; color: #222; }
.hashbox { border: 1px solid #e1e0d9; border-radius: 8px; padding: 10px 14px; margin-top: 8px; background: #f9f9f7; }
footer { margin-top: 28px; padding-top: 12px; border-top: 1px solid #e1e0d9; color: #52514e; font-size: 11.5px; }
.print { float: right; font: inherit; padding: 6px 12px; border: 1px solid #c3c2b7; border-radius: 6px; background: #fff; cursor: pointer; }
@page { size: A4; margin: 16mm; }
@media print { .page { padding: 0; max-width: none; } .print { display: none; } a { color: inherit; text-decoration: none; } }
"""


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _inr(value: object) -> str:
    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return _e(value)


def _criterion_row(key: str, crit: dict) -> str:
    detail = []
    if "value" in crit:
        detail.append(f"reason code {_e(crit['value'])}")
    if "count" in crit:
        detail.append(f"count {_e(crit['count'])}")
    if "window" in crit and isinstance(crit["window"], (list, tuple)) and len(crit["window"]) == 2:
        detail.append(f"window {_e(crit['window'][0])} to {_e(crit['window'][1])}")
    verdict = '<span class="pass">PASS</span>' if crit.get("pass") else '<span class="fail">FAIL</span>'
    label = CRITERION_LABELS.get(key, key.replace("_", " "))
    return f"<tr><td>{verdict}</td><td>{_e(label)}<br><span class=\"mono\">{_e(key)}</span></td><td>{' · '.join(detail)}</td></tr>"


def _elements_table(prior: dict, disputed_elements: dict) -> str:
    matched = set(prior.get("matched_elements", []))
    hashes = prior.get("element_hashes", {})
    rows = []
    for key, label in ELEMENT_LABELS.items():
        is_match = key in matched
        own = hashes.get(key)
        own_cell = f'<span class="mono">{_e(own)}</span>' if own else '<span class="mono">not matched</span>'
        disputed_cell = f'<span class="mono">{_e(disputed_elements.get(key, ""))}</span>'
        tag = '<span class="tag">MATCH</span>' if is_match else ""
        rows.append(f'<tr class="{"match" if is_match else ""}"><td>{_e(label)}{tag}</td><td>{own_cell}</td><td>{disputed_cell}</td></tr>')
    return ("<table><thead><tr><th>Data element</th><th>This prior transaction</th><th>Disputed transaction</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def _prior_block(prior: dict, disputed_elements: dict) -> str:
    delivered = "delivered" if prior.get("delivered") else "not delivered / returned"
    return (
        '<div class="txn"><div class="head">'
        f'<span><b>{_e(prior.get("transaction_id", ""))}</b> · {_e(prior.get("date", ""))} '
        f'({_e(prior.get("days_before_dispute", ""))} days before the dispute)</span>'
        f'<span>{_inr(prior.get("amount_inr"))} · {_e(prior.get("items", ""))} · {delivered}</span>'
        f'</div>{_elements_table(prior, disputed_elements)}</div>'
    )


def render_packet_html(packet: dict, *, merchant: str = DEFAULT_MERCHANT, generated_at: datetime | None = None) -> str:
    """Render one compiled CE3.0 packet as a self-contained, printable HTML document.

    ``packet`` is the dict returned by :func:`compile_ce3` (also the shape of
    the JSON route). ``generated_at`` is injectable so tests can assert
    byte-for-byte determinism; it defaults to the current UTC time.
    """
    ts = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    eligible = bool(packet.get("eligible"))
    criteria: dict = packet.get("criteria") or {}
    evidence: dict = packet.get("evidence") or {}
    disputed: dict = evidence.get("disputed_transaction") or {}
    disputed_elements: dict = disputed.get("elements") or {}
    priors: list = evidence.get("prior_transactions") or []
    extra = int(evidence.get("additional_prior_transactions") or 0)
    reason_code = (criteria.get("reason_code_10_4") or {}).get("value", "")
    txn_id = packet.get("transaction_id", "")
    packet_hash = packet.get("packet_hash") or ""
    standard = packet.get("standard", "Visa CE3.0")

    status = ('<span class="status ok">ELIGIBLE — LIABILITY SHIFT REQUESTED</span>' if eligible
              else '<span class="status no">NOT ELIGIBLE</span>')

    header = f"""
<header>
  <button class="print" onclick="window.print()">Print</button>
  <h1>{_e(standard)} evidence packet</h1>
  <p class="sub">Dispute response compiled from ledger records · {status}</p>
  <dl class="kv">
    <dt>Merchant</dt><dd>{_e(merchant)}</dd>
    <dt>Disputed transaction</dt><dd><span class="mono">{_e(txn_id)}</span></dd>
    <dt>Dispute reason</dt><dd>{_e(reason_code) if reason_code else "not recorded"}</dd>
    <dt>Dispute date</dt><dd>{_e(evidence.get("dispute_date", "not recorded"))}</dd>
    <dt>Outcome</dt><dd>{_e(packet.get("reason", ""))}</dd>
    <dt>Packet hash</dt><dd><span class="mono">{_e(packet_hash) if packet_hash else "none (packet could not be compiled)"}</span></dd>
    <dt>Generated</dt><dd>{_e(ts)}</dd>
  </dl>
</header>"""

    if criteria:
        crit_rows = "".join(_criterion_row(k, v) for k, v in criteria.items())
        criteria_html = (f"<h2>CE3.0 criteria</h2><table><thead><tr><th>Result</th><th>Criterion</th><th>Detail</th></tr></thead>"
                         f"<tbody>{crit_rows}</tbody></table>")
    else:
        criteria_html = "<h2>CE3.0 criteria</h2><p>No criteria were evaluated: the transaction was not found in the ledger.</p>"

    if disputed:
        element_rows = "".join(
            f"<tr><td>{_e(label)}</td><td><span class=\"mono\">{_e(disputed_elements.get(key, ''))}</span></td></tr>"
            for key, label in ELEMENT_LABELS.items())
        disputed_html = f"""
<h2>Disputed transaction</h2>
<dl class="kv">
  <dt>Transaction</dt><dd><span class="mono">{_e(disputed.get("transaction_id", ""))}</span></dd>
  <dt>Date</dt><dd>{_e(disputed.get("date", ""))}</dd>
  <dt>Amount</dt><dd>{_inr(disputed.get("amount_inr"))}</dd>
  <dt>Items</dt><dd>{_e(disputed.get("items", ""))}</dd>
  <dt>Payment credential</dt><dd><span class="mono">{_e(disputed.get("credential", ""))}</span></dd>
</dl>
<table><thead><tr><th>Data element</th><th>Value (hashed)</th></tr></thead><tbody>{element_rows}</tbody></table>"""
    else:
        disputed_html = ""

    if priors:
        title = "Qualifying prior transactions" if eligible else "Prior transactions found (insufficient for CE3.0)"
        note = (f"<p>{extra} further prior transaction(s) also qualify and are held in the ledger; the two most recent are presented.</p>"
                if extra else "")
        priors_html = f"<h2>{title}</h2>{note}" + "".join(_prior_block(p, disputed_elements) for p in priors)
    elif disputed:
        priors_html = "<h2>Qualifying prior transactions</h2><p>None: no undisputed prior transaction in the 120 to 365 day window shares the required data elements.</p>"
    else:
        priors_html = ""

    narrative = evidence.get("merchant_narrative")
    narrative_html = f'<h2>Merchant statement</h2><div class="narrative">{_e(narrative)}</div>' if narrative else ""

    hash_html = f"""
<h2>Integrity</h2>
<div class="hashbox">
  <div>SHA-256 of the canonical packet (JSON with sorted keys, compact separators, UTF-8; fields <span class="mono">transaction_id, standard, criteria, evidence</span>):</div>
  <div class="mono">{_e(packet_hash) if packet_hash else "not computed"}</div>
</div>"""

    footer = """
<footer>
  This document was generated deterministically from ledger records by the ChakraShield CE3.0 compiler.
  Every value is copied from the compiled packet; nothing on this page is free text or model output.
  Re-running the compiler against the same ledger, transaction and dispute date reproduces the packet hash above.
</footer>"""

    return (f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{_e(standard)} packet {_e(txn_id)}</title><style>{_CSS}</style></head>"
            f"<body><div class=\"page\">{header}{criteria_html}{disputed_html}{priors_html}{narrative_html}{hash_html}{footer}"
            f"</div></body></html>\n")
