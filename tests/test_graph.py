from chakrashield.features.vectorizer import graph_features_from_store
from chakrashield.graph.syndicate import SyndicateGraph
from chakrashield.store.feature_store import MemoryStore


def test_household_is_not_a_ring_but_syndicate_is():
    s = MemoryStore()
    g = SyndicateGraph(store=s)
    # household: two phones on one device, one address
    for i, ph in enumerate(("h1", "h2")):
        g.ingest(f"o{i}", {"phone": ph, "device": "homedev", "addr": "homeaddr"})
    assert not g.lookup("device", "homedev").is_ring
    # syndicate: ten burner phones through two handsets and one drop address
    for i in range(10):
        g.ingest(f"r{i}", {"phone": f"burner{i}", "device": f"dev{i % 2}", "addr": "drop"})
        g.outcome(f"r{i}", rto=True)
    st = g.lookup("addr", "drop")
    assert st.is_ring and st.phones == 10 and st.devices == 2 and st.rto_rate == 1.0
    assert g.rings()[0]["ring_id"] == st.ring_id


def test_ring_features_are_point_in_time_and_read_from_store():
    s = MemoryStore()
    g = SyndicateGraph(store=s)
    unseen = graph_features_from_store(s, {"phone": "x", "device": "y", "addr": "z"})
    assert unseen["ring_size"] == 0 and not unseen["is_ring"]
    for i in range(6):
        g.ingest(f"r{i}", {"phone": f"b{i}", "device": "dev", "addr": "drop"})
    # a brand-new phone arriving on the ring's device sees the ring BEFORE being linked
    feats = graph_features_from_store(s, {"phone": "newphone", "device": "dev", "addr": "fresh"})
    assert feats["is_ring"] and feats["ring_phones"] == 6 and feats["entity_max_degree"] >= 6


def test_subgraph_and_snapshot(tmp_path):
    s = MemoryStore()
    g = SyndicateGraph(store=s)
    for i in range(5):
        g.ingest(f"r{i}", {"phone": f"b{i}", "device": "dev", "addr": "drop"})
    sg = g.subgraph(g.lookup("device", "dev").ring_id)
    assert len(sg["nodes"]) == 7 and len(sg["edges"]) >= 11
    g.dump(tmp_path / "g.pkl")
    g2 = SyndicateGraph.load(tmp_path / "g.pkl", store=MemoryStore())
    assert g2.stats() == g.stats()
    g2.ingest("later", {"phone": "b9", "device": "dev", "addr": "drop"})   # lock re-created, still mutable
