from datetime import datetime, timezone

from chakrashield.dispute.ce3 import TransactionLedger, TxnRecord, compile_ce3
from chakrashield.dispute.render import render_packet_html

DAY = 86400.0
T = 1_750_000_000.0
GEN = datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc)


def rec(tid, ts, *, ip="ipA", dev="devA", addr="addrA", acct="accA", disputed=False):
    return TxnRecord(transaction_id=tid, ts=ts, card_token="tok1", ip_hash=ip, device_hash=dev, addr_hash=addr,
                     account_id=acct, amount=1799.0, items="2 item(s), 640 g", disputed=disputed, delivered=True)


def ledger(extra):
    led = TransactionLedger()
    led.add(rec("disputed", T))
    for r in extra:
        led.add(r)
    return led


def eligible_packet():
    return compile_ce3(ledger([rec("p1", T + 45 * DAY - 150 * DAY), rec("p2", T + 45 * DAY - 200 * DAY)]), "disputed")


def test_eligible_packet_renders_hash_priors_and_matches():
    p = eligible_packet()
    html = render_packet_html(p, generated_at=GEN)
    assert html.startswith("<!doctype html>") and "<script src" not in html and "<link" not in html
    assert p["packet_hash"] in html and p["packet_hash"][:16] in html
    assert "p1" in html and "p2" in html and "ELIGIBLE" in html and "NOT ELIGIBLE" not in html
    assert html.count('<span class="pass">PASS</span>') == 4 and '<span class="fail">' not in html
    # each prior transaction shares all four elements, so eight highlighted rows across the two blocks
    assert html.count('<tr class="match">') == 8 and html.count("MATCH</span>") == 8
    assert "2026-09-05 12:00:00 UTC" in html and "generated deterministically" in html
    assert "₹1,799.00" in html and "10.4" in html


def test_ineligible_packet_marks_failed_criterion():
    led = ledger([rec("p1", T + 45 * DAY - 150 * DAY), rec("p2", T + 45 * DAY - 30 * DAY)])
    p = compile_ce3(led, "disputed")
    html = render_packet_html(p, generated_at=GEN)
    assert not p["eligible"] and "NOT ELIGIBLE" in html and p["packet_hash"] in html
    assert '<span class="fail">FAIL</span>' in html and "insufficient for CE3.0" in html
    assert "p1" in html and "p2" not in html.split("<h2>Prior transactions")[1]   # p2 is out of window, never matched


def test_unknown_transaction_renders_without_hash():
    html = render_packet_html(compile_ce3(TransactionLedger(), "nope"), generated_at=GEN)
    assert "not found in ledger" in html and "packet could not be compiled" in html and "NOT ELIGIBLE" in html


def test_render_is_deterministic_and_escapes():
    p = eligible_packet()
    assert render_packet_html(p, generated_at=GEN) == render_packet_html(p, generated_at=GEN)
    hostile = {**p, "transaction_id": "<script>alert(1)</script>"}
    html = render_packet_html(hostile, generated_at=GEN)
    assert "<script>alert" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
