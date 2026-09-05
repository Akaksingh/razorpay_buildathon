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


def test_soft_identifiers_never_merge():
    """Two strangers behind one NAT share an IP; that must not put them in one component."""
    s = MemoryStore()
    g = SyndicateGraph(store=s)
    g.ingest("a", {"phone": "pa", "device": "da", "addr": "aa", "ip": "nat"})
    g.ingest("b", {"phone": "pb", "device": "db", "addr": "ab", "ip": "nat"})
    assert g.lookup("phone", "pa").ring_id != g.lookup("phone", "pb").ring_id
    assert g.lookup("ip", "nat").size == 1                      # the IP is a singleton property node
    assert g.degree("ip", "nat") == 6                            # ... but keeps bipartite adjacency
    feats = graph_features_from_store(s, {"phone": "pc", "device": "dc", "addr": "ac", "ip": "nat"})
    assert feats["ring_size"] == 0 and not feats["is_ring"]


def test_public_address_stops_bridging_and_is_never_a_ring():
    """A hostel: 30 residents, own phones and devices, normal deliveries."""
    from chakrashield.graph.syndicate import ADDR_MERGE_CEILING

    def build(guard: bool):
        s = MemoryStore()
        g = SyndicateGraph(store=s, guard=guard)
        for i in range(30):
            g.ingest(f"h{i}", {"phone": f"res{i}", "device": f"own{i}", "addr": "hostel"})
            g.outcome(f"h{i}", rto=False)
        # a syndicate (two handsets, one payout VPA) whose members once had a parcel sent to the hostel
        for i in range(8):
            g.ingest(f"r{i}", {"phone": f"burner{i}", "device": f"ringdev{i % 2}", "addr": "hostel", "vpa": "payout"})
            g.outcome(f"r{i}", rto=True)
        return s, g

    s, g = build(guard=True)
    assert g.is_shared("addr", "hostel") and g.stats()["shared_entities"] == 1
    hostel_component = g.lookup("phone", "res0")
    assert not hostel_component.is_ring and hostel_component.phones <= ADDR_MERGE_CEILING
    assert g.lookup("phone", "res29").ring_id != hostel_component.ring_id   # arrived after the ceiling: not bridged
    ring = g.lookup("phone", "burner0")
    assert ring.is_ring and ring.phones == 8 and ring.ring_id != hostel_component.ring_id
    resident = graph_features_from_store(s, {"phone": "res5", "device": "own5", "addr": "hostel"})
    assert not resident["is_ring"] and resident["entity_shared"] and resident["ring_rto_rate"] == 0.0

    s2, g2 = build(guard=False)                                  # the naive graph: everyone is one component
    naive = g2.lookup("phone", "res0")
    assert naive.phones == 38 and naive.ring_id == g2.lookup("phone", "burner0").ring_id
    assert g2.stats()["largest_component"] > g.stats()["largest_component"]
