import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from chakrashield.graph.embeddings import (
    KINDS, N_FEATURES, FEATURE_NAMES, GraphTensors, TrainConfig, build_entity_graph, fit_sage, hash_orders,
    out_of_fold_scores, standardiser, training_step,
)
from chakrashield.features.velocity import hash_entity


def orders(rows: list[tuple[str, str, str, str, bool]], t0: float = 0.0) -> pd.DataFrame:
    """(phone, device, address, vpa, rto) rows -> the columns entity_hashes needs, one order per row."""
    recs = []
    for i, (ph, dv, ad, vpa, rto) in enumerate(rows):
        ts = t0 + 100.0 * i
        recs.append({"order_id": f"o{i}", "ts": ts, "outcome_ts": ts + 5.0, "customer_phone": ph, "device_fingerprint": dv,
                     "shipping_address": ad, "delivery_pin": "560034", "vpa": vpa, "ip": f"ip{i}", "rto": rto})
    return pd.DataFrame(recs)


def world(n_ring: int = 12, n_legit: int = 40, seed: int = 0) -> list[tuple]:
    """A syndicate (n_ring phones through two handsets, one payout VPA, always RTO) among households."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_legit):
        for _ in range(2):
            rows.append((f"legit{i}", f"own{i}", f"home {i}", "", bool(rng.random() < 0.15)))
    for i in range(n_ring):
        rows.append((f"burner{i}", f"ringdev{i % 2}", f"drop {i % 3}", "payout@upi", True))
    rng.shuffle(rows)
    return rows


def test_graph_construction_counts_edges_orders_and_outcomes():
    df = orders([("p1", "d1", "a1", "", False), ("p2", "d1", "a2", "v", True), ("p2", "d1", "a2", "v", True)])
    g = build_entity_graph(hash_orders(df))
    # nodes: p1 p2 d1 a1 a2 v + 3 ips
    assert g.n_nodes == 9 and g.features.shape == (9, N_FEATURES) and len(FEATURE_NAMES) == N_FEATURES
    d1 = g.node("device", hash_entity("device", "d1"))
    p2 = g.node("phone", hash_entity("phone", "p2"))
    assert g.orders[d1] == 3 and g.orders[p2] == 2 and g.rto[p2] == 2 and g.delivered[d1] == 1
    assert g.neighbours_by_kind[d1, KINDS.index("phone")] == 2          # distinct phones on the handset
    assert g.neighbours_by_kind[d1, KINDS.index("ip")] == 3
    assert g.degree[d1] == 2 + 2 + 1 + 3                                 # phones, addrs, vpa, ips
    # every edge is stored once per direction and never twice
    assert g.n_edges * 2 == len(g.src) and len(np.unique(g.src * g.n_nodes + g.dst)) == len(g.src)
    assert g.node("phone", hash_entity("phone", "nobody")) == -1
    # the mean aggregation is a mean over distinct neighbours
    t = GraphTensors.from_graph(g, *standardiser(g))
    ones = torch.ones(g.n_nodes, 1)
    agg = torch.zeros_like(ones).index_add_(0, t.dst, ones[t.src]) * t.inv_deg
    assert torch.allclose(agg, ones)


def test_point_in_time_cutoffs_hide_later_orders_and_outcomes():
    df = orders([("p1", "d1", "a1", "", True), ("p1", "d1", "a1", "", True), ("p9", "d9", "a9", "", False)])
    early = build_entity_graph(hash_orders(df), order_cutoff=150.0, outcome_cutoff=50.0)
    p1 = early.node("phone", hash_entity("phone", "p1"))
    assert early.node("phone", hash_entity("phone", "p9")) == -1           # placed after the order cutoff
    assert early.orders[p1] == 2 and early.rto[p1] == 1                     # second outcome not yet resolved
    late = build_entity_graph(hash_orders(df))
    assert late.rto[late.node("phone", hash_entity("phone", "p1"))] == 2


def test_one_training_step_reduces_loss():
    g = build_entity_graph(hash_orders(orders(world())))
    torch.manual_seed(0)
    from chakrashield.graph.embeddings import GraphSAGE
    model = GraphSAGE(N_FEATURES, hidden=16, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    t = GraphTensors.from_graph(g, *standardiser(g))
    y = torch.zeros(g.n_nodes)
    mask = torch.from_numpy(g.kind == 0)
    for i, nid in enumerate(g.node_ids):
        y[i] = 1.0 if nid.startswith("phone:") and g.rto[i] == g.orders[i] and g.orders[i] > 0 else 0.0
    first = training_step(model, opt, t, y, mask)
    for _ in range(5):
        last = training_step(model, opt, t, y, mask)
    assert last < first


def test_inductive_scoring_of_unseen_phones():
    """Train on one world, score a later snapshot: a new burner on a ring handset outranks a new household."""
    rows = world()
    g = build_entity_graph(hash_orders(orders(rows)))
    burners = {f"phone:{hash_entity('phone', f'burner{i}')}" for i in range(12)}
    labels = np.full(g.n_nodes, np.nan)
    for i, nid in enumerate(g.node_ids):
        if nid.startswith("phone:"):
            labels[i] = 1.0 if nid in burners else 0.0
    scorer = fit_sage(g, labels, TrainConfig(hidden=16, epochs=150, lr=0.03, valid_frac=0.0, dropout=0.0, seed=1))
    assert scorer.history[-1]["train_loss"] < scorer.history[0]["train_loss"]

    later = rows + [("newburner", "ringdev0", "drop 0", "payout@upi", False), ("newlegit", "newdev", "new home", "mine@upi", False)]
    g2 = build_entity_graph(hash_orders(orders(later)))
    s = scorer.score(g2)
    burner, legit = g2.node("phone", hash_entity("phone", "newburner")), g2.node("phone", hash_entity("phone", "newlegit"))
    assert burner >= 0 and legit >= 0 and g2.n_nodes > g.n_nodes         # both phones are new to the scorer
    assert np.isfinite(s).all() and 0.0 <= s.min() and s.max() <= 1.0
    assert s[burner] > s[legit] + 0.2
    # the new phones have identical own-node features (one order, no outcome): only the neighbourhood separates them
    assert np.allclose(g2.features[burner, len(KINDS):], g2.features[legit, len(KINDS):])
    assert scorer.embed(g2).shape == (g2.n_nodes, 16)


def test_out_of_fold_scores_cover_exactly_the_labelled_nodes():
    g = build_entity_graph(hash_orders(orders(world(n_ring=8, n_legit=20))))
    labels = np.full(g.n_nodes, np.nan)
    phones = np.flatnonzero(g.kind == 0)
    labels[phones] = (g.rto[phones] == g.orders[phones]).astype(float)
    oof = out_of_fold_scores(g, labels, folds=3, config=TrainConfig(hidden=8, epochs=20, valid_frac=0.0))
    assert np.isnan(oof[g.kind != 0]).all() and np.isfinite(oof[phones]).all()
