"""Per-merchant configuration, API-key enforcement and shadow mode.

The registry tests run anywhere; the gateway tests are skipped until artifacts exist (scripts 01-02).
"""
import json

import pytest

from chakrashield.config import ECONOMICS, MODEL_DIR, ROOT
from chakrashield.serving.merchants import MerchantConfig, MerchantRegistry, api_key_required

CONFIG_PATH = ROOT / "config" / "merchants.json"
ORDER = {"customer_phone": "7012349876", "delivery_pin": "845401", "shipping_address": "H.No 7, Ward 4, near Hanuman Temple",
         "cart_gmv": 1799, "items_count": 2, "device_fingerprint_hash": "fp_demo_new_02", "payment_method": "COD",
         "acquisition_channel": "META_ADS", "checkout_seconds": 40, "hour_of_day": 20, "is_new_customer": True}


# ------------------------------------------------------------------ registry (no artifacts needed)
def test_repo_config_loads_three_merchants_with_a_shadow_one():
    reg = MerchantRegistry.load(CONFIG_PATH)
    assert len(reg) == 3 and "demo_merchant" in reg and reg.default_id == "demo_merchant"
    demo, known = reg.resolve("demo_merchant")
    assert known and demo.economics == ECONOMICS and demo.overrides == {} and not demo.shadow
    nyk, _ = reg.resolve("nykaa_style_d2c")
    assert nyk.economics.default_margin == 0.32 and nyk.economics.default_cac == 650.0
    assert nyk.economics.forward_shipping == ECONOMICS.forward_shipping      # untouched fields keep the global value
    assert nyk.friction_budget == 0.25 and nyk.epsilon == 0.02 and not nyk.shadow
    trial, _ = reg.resolve("trial_merchant")
    assert trial.shadow and trial.epsilon is None


def test_unknown_merchant_resolves_to_the_default_and_says_so():
    reg = MerchantRegistry.load(CONFIG_PATH)
    m, known = reg.resolve("nobody_registered_this")
    assert not known and m.merchant_id == "demo_merchant"
    block = m.response_block("nobody_registered_this", known)
    assert block["known"] is False and block["requested_id"] == "nobody_registered_this" and "unknown merchant_id" in block["note"]
    assert reg.resolve("demo_merchant")[0].response_block("demo_merchant", True).get("note") is None


def test_registry_without_a_file_still_has_the_default(tmp_path):
    reg = MerchantRegistry.load(tmp_path / "missing.json")
    m, known = reg.resolve("demo_merchant")
    assert known and m.economics == ECONOMICS and len(reg) == 1
    # a file that omits the default gets one added, on the global economics
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"only_one": {"economics": {"default_margin": 0.25}}}), encoding="utf-8")
    reg2 = MerchantRegistry.load(p)
    assert len(reg2) == 2 and reg2.resolve("demo_merchant")[0].economics == ECONOMICS
    assert reg2.resolve("only_one")[0].economics.default_margin == 0.25


def test_bad_entries_fail_at_load_not_at_request_time():
    with pytest.raises(ValueError, match="unknown Economics fields"):
        MerchantConfig.from_entry("x", {"economics": {"margin": 0.3}}, ECONOMICS)
    with pytest.raises(ValueError, match="friction_budget"):
        MerchantConfig.from_entry("x", {"friction_budget": 1.5}, ECONOMICS)
    with pytest.raises(ValueError, match="epsilon"):
        MerchantConfig.from_entry("x", {"epsilon": -0.1}, ECONOMICS)


def test_public_view_never_carries_keys_and_authorize_is_exact():
    reg = MerchantRegistry.load(CONFIG_PATH)
    for row in reg.public():
        assert "api_key" not in row and row["has_api_key"] is True
    nyk, known = reg.resolve("nykaa_style_d2c")
    assert reg.authorize(nyk, known, nyk.api_key)
    assert not reg.authorize(nyk, known, nyk.api_key + "x") and not reg.authorize(nyk, known, None)
    demo, unknown = reg.resolve("nobody")
    assert not reg.authorize(demo, unknown, demo.api_key)       # an unknown id cannot borrow the default's key
    keyless = MerchantConfig(merchant_id="k")
    assert not reg.authorize(keyless, True, "anything")


def test_api_key_required_reads_the_environment_at_call_time(monkeypatch):
    monkeypatch.delenv("CHAKRA_REQUIRE_API_KEY", raising=False)
    assert api_key_required() is False
    monkeypatch.setenv("CHAKRA_REQUIRE_API_KEY", "1")
    assert api_key_required() is True
    monkeypatch.setenv("CHAKRA_REQUIRE_API_KEY", "0")
    assert api_key_required() is False


# ------------------------------------------------------------------ gateway
pytestmark_api = pytest.mark.skipif(not (MODEL_DIR / "chakra_rto.txt.json").exists(), reason="train the model first")


@pytest.fixture(scope="module")
def client():
    if not (MODEL_DIR / "chakra_rto.txt.json").exists():
        pytest.skip("train the model first")
    from fastapi.testclient import TestClient
    from chakrashield.serving.app import app

    with TestClient(app) as c:
        yield c


def _eval(client, body, **q):
    query = "&".join(f"{k}={v}" for k, v in {"commit": "false", **q}.items())
    r = client.post(f"/v1/risk/evaluate?{query}", json=body)
    assert r.status_code == 200, r.text
    return r.json()


@pytestmark_api
def test_merchants_endpoint_lists_config_without_keys(client):
    out = client.get("/v1/merchants").json()
    ids = {m["id"] for m in out["merchants"]}
    assert ids == {"demo_merchant", "nykaa_style_d2c", "trial_merchant"} and out["default_id"] == "demo_merchant"
    assert all("api_key" not in m for m in out["merchants"])
    assert out["api_key_required"] is False and 0 < out["global_epsilon"] <= 1


@pytestmark_api
def test_unknown_merchant_falls_back_to_demo_defaults(client):
    demo = _eval(client, {**ORDER, "merchant_id": "demo_merchant"})
    other = _eval(client, {**ORDER, "merchant_id": "never_registered"})
    assert demo["merchant"] == {"id": "demo_merchant", "known": True, "shadow": False, "economics": {}, "economics_source": "default"}
    assert other["merchant"]["known"] is False and other["merchant"]["id"] == "demo_merchant"
    assert other["merchant"]["requested_id"] == "never_registered" and "unknown merchant_id" in other["merchant"]["note"]
    assert other["tau_star"] == demo["tau_star"] and other["decision"] == demo["decision"] and other["shadow"] is False


@pytestmark_api
def test_per_merchant_economics_change_tau_star(client):
    demo = _eval(client, {**ORDER, "merchant_id": "demo_merchant"})
    nyk = _eval(client, {**ORDER, "merchant_id": "nykaa_style_d2c"})
    assert nyk["economics"]["merchant_margin"] == 0.32 and nyk["economics"]["cac"] == 650.0
    assert demo["economics"]["merchant_margin"] == ECONOMICS.default_margin and demo["economics"]["cac"] == ECONOMICS.default_cac
    assert nyk["p_loss"] == demo["p_loss"]                       # same order, same score: only the pricing differs
    assert nyk["tau_star"] > demo["tau_star"]                    # a dearer good buyer raises the indifference point
    assert nyk["merchant"]["economics"]["stepup_abandon_rate"] == 0.18 and nyk["merchant"]["economics_source"] == "merchant"
    # with no learned data for the segment, the prior the resolver applied is the merchant's, and the response says so
    if not nyk["behaviour"]["applied"]:
        assert nyk["behaviour"]["delta_s"] == 0.18 and nyk["behaviour"]["delta_p"] == 0.45
        assert demo["behaviour"]["delta_s"] == ECONOMICS.stepup_abandon_rate
    # a request-level override still beats the merchant default
    explicit = _eval(client, {**ORDER, "merchant_id": "nykaa_style_d2c", "merchant_margin": 0.10, "cac": 100})
    assert explicit["economics"]["merchant_margin"] == 0.10 and explicit["tau_star"] < demo["tau_star"]


@pytestmark_api
def test_per_merchant_friction_budget_and_epsilon(client):
    nyk = _eval(client, {**ORDER, "merchant_id": "nykaa_style_d2c"}, commit="true", explain="never")
    assert nyk["friction"]["budget"] == 0.25 and nyk["friction"]["budget_source"] == "merchant"
    assert nyk["friction"]["source"] in ("frontier", "no_frontier")
    assert nyk["exploration"]["epsilon"] == 0.02
    demo = _eval(client, {**ORDER, "merchant_id": "demo_merchant"}, commit="true", explain="never")
    assert demo["friction"]["budget"] is None and demo["friction"]["source"] == "config"
    assert demo["exploration"]["epsilon"] == client.get("/v1/merchants").json()["global_epsilon"]
    req_budget = _eval(client, {**ORDER, "merchant_id": "nykaa_style_d2c", "friction_budget": 0.5})
    assert req_budget["friction"]["budget"] == 0.5 and req_budget["friction"]["budget_source"] == "request"


@pytestmark_api
def test_shadow_merchant_is_always_served_allow_but_the_policy_is_reported(client):
    scs = client.get("/v1/scenarios").json()
    policies = set()
    for sc in scs:
        body = {**sc["req"], "merchant_id": "trial_merchant", "order_id": f"shadow_{sc['tag']}"}
        body.pop("merchant_margin"), body.pop("cac")           # let the merchant's economics apply
        r = _eval(client, body, commit="true")
        assert r["decision"] == "ALLOW_COD" and r["shadow"] is True and r["friction_level"] == 0
        assert r["merchant"]["shadow"] is True and r["merchant"]["id"] == "trial_merchant"
        assert r["exploration"]["propensity"] == 1.0 and r["exploration"]["is_control_cohort"] is False
        assert r["economics"]["merchant_margin"] == 0.12
        policies.add(r["policy_action"])
        if r["policy_action"] != "ALLOW_COD":
            assert r["rationale"].startswith("SHADOW MODE (trial_merchant)")
            assert r["explained"] is True and r["reason_codes"]     # the merchant sees why it *would* have frictioned
    assert policies - {"ALLOW_COD"}, "the scenario set must contain orders the policy would have frictioned"
    stats = client.get("/v1/ledger/stats").json()                # drains the async writer
    assert stats["shadow"] >= len(scs)
    # a shipped shadow order carries an untreated delivery label for the learner and the graph
    o = client.post("/v1/risk/outcome/shadow_prepaid?rto=true").json()
    assert o["shipped"] is True and o["learned"] is None


@pytestmark_api
def test_api_key_enforced_only_when_required(client, monkeypatch):
    body = {**ORDER, "merchant_id": "nykaa_style_d2c"}
    monkeypatch.delenv("CHAKRA_REQUIRE_API_KEY", raising=False)
    assert client.post("/v1/risk/evaluate?commit=false", json=body).status_code == 200
    monkeypatch.setenv("CHAKRA_REQUIRE_API_KEY", "1")
    assert client.get("/v1/merchants").json()["api_key_required"] is True
    assert client.post("/v1/risk/evaluate?commit=false", json=body).status_code == 401
    assert client.post("/v1/risk/evaluate?commit=false", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    key = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["nykaa_style_d2c"]["api_key"]
    ok = client.post("/v1/risk/evaluate?commit=false", json=body, headers={"X-API-Key": key})
    assert ok.status_code == 200 and ok.json()["merchant"]["id"] == "nykaa_style_d2c"
    demo_key = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["demo_merchant"]["api_key"]
    assert client.post("/v1/risk/evaluate?commit=false", json=body, headers={"X-API-Key": demo_key}).status_code == 401
    unknown = client.post("/v1/risk/evaluate?commit=false", json={**ORDER, "merchant_id": "nobody"}, headers={"X-API-Key": demo_key})
    assert unknown.status_code == 401 and "unknown merchant_id" in unknown.json()["detail"]
    monkeypatch.delenv("CHAKRA_REQUIRE_API_KEY")
    assert client.post("/v1/risk/evaluate?commit=false", json=body).status_code == 200
