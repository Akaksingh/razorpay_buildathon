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


def test_graph_endpoints(client):
    rings = client.get("/v1/graph/rings?top=3").json()
    assert "stats" in rings
    if rings["rings"]:
        sg = client.get(f"/v1/graph/subgraph?seed={rings['rings'][0]['ring_id']}").json()
        assert sg["nodes"] and sg["edges"]
