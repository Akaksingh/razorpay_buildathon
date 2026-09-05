"""End-to-end gateway tests. Skipped until artifacts exist (run scripts 01-02)."""
import pytest

from chakrashield.config import MODEL_DIR

pytestmark = pytest.mark.skipif(not (MODEL_DIR / "chakra_rto.txt.json").exists(), reason="train the model first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from chakrashield.serving.app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    h = client.get("/healthz").json()
    assert h["ok"] and h["scorer_backend"] in ("onnxruntime", "lightgbm")


def test_scenarios_resolve_to_valid_actions(client):
    for sc in client.get("/v1/scenarios").json():
        r = client.post("/v1/risk/evaluate?commit=false", json=sc["req"])
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["decision"] in ("ALLOW_COD", "STEP_UP_DEPOSIT", "FORCE_PREPAID")
        assert 0 <= body["p_loss"] <= 1 and 0 <= body["tau_star"] <= 1
        assert body["decision"] in body["admissible_actions"]
        assert set(body["conformal"]["prediction_set"]) <= {0, 1}
        assert "X-Chakra-Latency-Ms" in r.headers
        assert body["latency_ms"]["total"] < 10 * body["latency_ms"]["budget_ms"]


def test_junk_address_is_never_frictionless(client):
    sc = [s for s in client.get("/v1/scenarios").json() if s["tag"] == "prepaid"][0]
    body = client.post("/v1/risk/evaluate?commit=false", json=sc["req"]).json()
    assert body["decision"] != "ALLOW_COD"
    assert any(c["code"].startswith("RSK_ADDR") for c in body["reason_codes"])


def test_validation_errors(client):
    r = client.post("/v1/risk/evaluate", json={"customer_phone": "1", "delivery_pin": "12", "shipping_address": "x", "cart_gmv": 0, "device_fingerprint_hash": "abcd"})
    assert r.status_code == 422


def test_ce3_endpoint(client):
    cands = client.get("/v1/dispute/candidates?n=3").json()
    if not cands:
        pytest.skip("no card history in ledger")
    out = client.post("/v1/dispute/ce3-compile", json={"transaction_id": cands[0]["transaction_id"]}).json()
    assert out["standard"] == "Visa CE3.0" and "criteria" in out and len(out["packet_hash"]) == 64


def test_ce3_printable_packet(client):
    cands = client.get("/v1/dispute/candidates?n=3").json()
    if not cands:
        pytest.skip("no card history in ledger")
    tid = cands[0]["transaction_id"]
    body = {"transaction_id": tid, "dispute_date": "2026-01-15"}
    js = client.post("/v1/dispute/ce3-compile", json=body)
    assert js.headers["content-type"].startswith("application/json")
    packet = js.json()
    r = client.post("/v1/dispute/ce3-compile?format=html", json=body)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert packet["packet_hash"] in r.text and tid in r.text
    for prior in packet["evidence"]["prior_transactions"]:
        assert prior["transaction_id"] in r.text
    g = client.get(f"/v1/dispute/packet/{tid}.html?dispute_date=2026-01-15")
    assert g.status_code == 200 and g.headers["content-type"].startswith("text/html") and packet["packet_hash"] in g.text
    assert client.post("/v1/dispute/ce3-compile?format=pdf", json=body).status_code == 422
    assert client.get(f"/v1/dispute/packet/{tid}.html?dispute_date=15-01-2026").status_code == 422


def test_graph_endpoints(client):
    rings = client.get("/v1/graph/rings?top=3").json()
    assert "stats" in rings
    if rings["rings"]:
        sg = client.get(f"/v1/graph/subgraph?seed={rings['rings'][0]['ring_id']}").json()
        assert sg["nodes"] and sg["edges"]


def test_explain_auto_skips_shap_on_allow(client):
    scs = client.get("/v1/scenarios").json()
    allow = [s for s in scs if s["tag"] == "frictionless"][0]
    a = client.post("/v1/risk/evaluate?commit=false&explain=auto", json=allow["req"]).json()
    assert a["decision"] == "ALLOW_COD" and a["explained"] is False and a["reason_codes"] == []
    assert a["latency_ms"]["score.treeshap"] == 0.0
    b = client.post("/v1/risk/evaluate?commit=false&explain=always", json=allow["req"]).json()
    assert b["explained"] is True and len(b["reason_codes"]) > 0
    hi = [s for s in scs if s["tag"] == "prepaid"][0]
    c = client.post("/v1/risk/evaluate?commit=false&explain=auto", json=hi["req"]).json()
    assert c["decision"] != "ALLOW_COD" and c["explained"] is True and len(c["reason_codes"]) > 0
    assert client.post("/v1/risk/evaluate?commit=false&explain=sometimes", json=hi["req"]).status_code == 422


def test_friction_shadow_price_and_budget(client):
    sc = [s for s in client.get("/v1/scenarios").json() if s["tag"] == "step-up"][0]
    base = client.post("/v1/risk/evaluate?commit=false", json=sc["req"]).json()
    assert base["friction"]["shadow_price"] == 0.0 and base["friction"]["source"] == "config"
    priced = client.post("/v1/risk/evaluate?commit=false", json={**sc["req"], "friction_shadow_price": 250.0}).json()
    assert priced["friction"]["shadow_price"] == 250.0 and priced["friction"]["source"] == "request"
    assert priced["tau_star"] > base["tau_star"] and priced["tau_soft"] > base["tau_soft"]
    budget = client.post("/v1/risk/evaluate?commit=false", json={**sc["req"], "friction_budget": 0.3}).json()
    assert budget["friction"]["budget"] == 0.3 and budget["friction"]["source"] in ("frontier", "no_frontier")


def test_control_cohort_pass_through_is_logged(client):
    import chakrashield.serving.app as appmod
    sc = [s for s in client.get("/v1/scenarios").json() if s["tag"] == "prepaid"][0]
    old = appmod.EXPLORATION_EPSILON
    appmod.EXPLORATION_EPSILON = 1.0          # every flagged order becomes a control pass-through
    try:
        r = client.post("/v1/risk/evaluate?commit=true", json={**sc["req"], "order_id": "test_ctrl_1"}).json()
    finally:
        appmod.EXPLORATION_EPSILON = old
    assert r["policy_action"] != "ALLOW_COD" and r["decision"] == "ALLOW_COD"
    assert r["exploration"]["is_control_cohort"] and r["exploration"]["propensity"] == 1.0
    assert r["rationale"].startswith("CONTROL COHORT") and r["friction_level"] == 0
    stats = client.get("/v1/ledger/stats").json()
    assert stats["decisions"] >= 1 and stats["control_cohort"] >= 1
    off = client.post("/v1/risk/evaluate?commit=false", json=sc["req"]).json()
    assert off["decision"] == off["policy_action"] and not off["exploration"]["is_control_cohort"]
    assert off["exploration"]["propensity"] == 1.0 and off["exploration"]["epsilon"] == 0.0


def test_outcome_callback_teaches_the_learner(client):
    sc = [s for s in client.get("/v1/scenarios").json() if s["tag"] == "step-up"][0]
    n0 = client.get("/v1/behaviour").json()["observations"]
    r = client.post("/v1/risk/evaluate?commit=true", json={**sc["req"], "order_id": "test_learn_1"}).json()
    assert r["behaviour"]["segment"].startswith("SOCIAL|T4") and "delta_s" in r["behaviour"]
    client.get("/v1/ledger/stats")                     # drains the async worker so the order is registered
    o = client.post("/v1/risk/outcome/test_learn_1?rto=false&stepup_result=paid").json()
    stepped = r["decision"] == "STEP_UP_DEPOSIT"
    assert o["learned"] == ("stepup" if stepped else None) and o["shipped"] is True
    assert client.get("/v1/behaviour").json()["observations"] == n0 + (1 if stepped else 0)
    o2 = client.post("/v1/risk/outcome/test_learn_1?rto=false&stepup_result=abandoned").json()
    assert o2["shipped"] is False
    assert client.post("/v1/risk/outcome/never_seen?rto=true").status_code == 404
