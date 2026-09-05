from chakrashield.dispute.ce3 import TransactionLedger, TxnRecord, compile_ce3

DAY = 86400.0
T = 1_750_000_000.0


def rec(tid, ts, *, ip="ipA", dev="devA", addr="addrA", acct="accA", disputed=False):
    return TxnRecord(transaction_id=tid, ts=ts, card_token="tok1", ip_hash=ip, device_hash=dev, addr_hash=addr,
                     account_id=acct, amount=999.0, items="1 item", disputed=disputed, delivered=True)


def build(extra=()):
    led = TransactionLedger()
    led.add(rec("disputed", T))
    for r in extra:
        led.add(r)
    return led


def test_eligible_with_two_qualifying_priors():
    led = build([rec("p1", T + 45 * DAY - 150 * DAY), rec("p2", T + 45 * DAY - 200 * DAY)])
    out = compile_ce3(led, "disputed")
    assert out["eligible"] and len(out["evidence"]["prior_transactions"]) == 2
    assert all(c["pass"] for c in out["criteria"].values())
    assert len(out["packet_hash"]) == 64


def test_recent_prior_does_not_count():
    led = build([rec("p1", T + 45 * DAY - 150 * DAY), rec("p2", T + 45 * DAY - 30 * DAY)])
    out = compile_ce3(led, "disputed")
    assert not out["eligible"] and out["criteria"]["prior_txns_in_120_365d_window"]["count"] == 1


def test_disputed_prior_does_not_count():
    led = build([rec("p1", T + 45 * DAY - 150 * DAY), rec("p2", T + 45 * DAY - 200 * DAY, disputed=True)])
    assert not compile_ce3(led, "disputed")["eligible"]


def test_two_elements_must_include_ip_or_device():
    # shares address + account only -> fails the primary-element requirement
    led = build([rec("p1", T + 45 * DAY - 150 * DAY, ip="other", dev="other"), rec("p2", T + 45 * DAY - 200 * DAY)])
    out = compile_ce3(led, "disputed")
    assert not out["eligible"]
    assert out["criteria"]["prior_txns_with_2_matching_elements_incl_ip_or_device"]["count"] == 1


def test_packet_is_deterministic():
    led = build([rec("p1", T + 45 * DAY - 150 * DAY), rec("p2", T + 45 * DAY - 200 * DAY)])
    a, b = compile_ce3(led, "disputed"), compile_ce3(led, "disputed")
    assert a["packet_hash"] == b["packet_hash"]


def test_unknown_transaction():
    assert not compile_ce3(TransactionLedger(), "nope")["eligible"]
