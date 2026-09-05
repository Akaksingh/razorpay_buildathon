import math

from chakrashield.data import pincodes
from chakrashield.features.address import score_address
from chakrashield.features.vectorizer import FEATURE_NAMES, build_features, to_vector
from chakrashield.features.velocity import VelocityFeatures, hash_entity, read_velocity, record_order, record_outcome
from chakrashield.store.feature_store import MemoryStore


def test_address_defects_rank_sensibly():
    good = score_address("Flat 402, Prestige Lakeside, Whitefield, Bengaluru 560066", "560066")
    landmark = score_address("Near Hanuman Temple, Ward 4", "845401")
    junk = score_address("asdf", "110001")
    mismatch = score_address("12, MG Road, Mumbai, Maharashtra 400001", "560001")
    assert good.defect_score < landmark.defect_score < junk.defect_score
    assert good.has_house_no and not landmark.has_house_no
    assert landmark.landmark_only and junk.has_junk
    assert mismatch.state_mismatch and mismatch.pin_in_text_mismatch


def test_pincode_hierarchy():
    assert pincodes.lookup("560034").tier == 1 and pincodes.lookup("560034").state == "Karnataka"
    assert pincodes.lookup("845401").tier == 4 and pincodes.lookup("845401").state == "Bihar"
    assert not pincodes.lookup("12").valid
    unseen = pincodes.lookup("562123")   # unlisted district in a metro state
    assert unseen.valid and unseen.tier in (3, 4)


def test_hashing_is_stable_and_pii_free():
    a, b = hash_entity("phone", " 9876543210 "), hash_entity("phone", "9876543210")
    assert a == b and len(a) == 16 and "9876" not in a
    assert hash_entity("vpa", None) == hash_entity("vpa", float("nan")) == hash_entity("vpa", "")


def test_velocity_is_point_in_time():
    s = MemoryStore()
    ph, dv, ad, pin = "p1", "d1", "a1", "560034"
    v0 = read_velocity(s, phone_h=ph, device_h=dv, addr_h=ad, pin=pin, now_ts=1000.0)
    assert v0.phone_is_new and v0.phone_orders_30d == 0 and v0.device_distinct_phones == 0
    record_order(s, phone_h=ph, device_h=dv, addr_h=ad, pin=pin, ts=1000.0, gmv=900)
    record_order(s, phone_h="p2", device_h=dv, addr_h=ad, pin=pin, ts=2000.0, gmv=900)
    record_outcome(s, phone_h=ph, device_h=dv, addr_h=ad, pin=pin, rto=True)
    v1 = read_velocity(s, phone_h=ph, device_h=dv, addr_h=ad, pin=pin, now_ts=3000.0)
    assert v1.phone_orders_30d == 1 and not v1.phone_is_new
    assert v1.device_distinct_phones == 2 and v1.addr_distinct_phones == 2
    assert v1.phone_rto_rate > v0.phone_rto_rate
    v_old = read_velocity(s, phone_h=ph, device_h=dv, addr_h=ad, pin=pin, now_ts=1000.0 + 40 * 86400)
    assert v_old.phone_orders_30d == 0   # window slid past the order


def test_feature_vector_shape_and_order():
    f = build_features(gmv=1500, items_count=1, weight_grams=450, pin="560034",
                       address=score_address("12 MG Road Bengaluru", "560034"), velocity=VelocityFeatures(), graph={},
                       payment_method="COD", payment_switch_from=float("nan"), channel="ORGANIC", coupon_applied=False,
                       checkout_seconds=60, hour_of_day=14)
    assert tuple(f) == FEATURE_NAMES
    v = to_vector(f)
    assert v.shape == (1, len(FEATURE_NAMES)) and not any(math.isnan(x) for x in v[0])
    assert f["pay_switch_from_failed"] == 0.0
