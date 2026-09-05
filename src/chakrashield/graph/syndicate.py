"""Subgraph Abuse Sentinel: incremental entity graph + syndicate ring detection.

Nodes are hashed entities (phone / device / addr / vpa / ip). Every order
adds a clique among the entities it touches. A *ring* is a connected
component that violates the one-human-one-device prior: many phones on few
devices or few addresses. Legit families share a device between two or
three phones; a syndicate puts 40 burner phones through three handsets.

Two data structures, two jobs:

* union-find with per-root aggregates -> O(alpha(n)) ring membership and
  stats on the *hot* path. This is what the serving vectorizer reads via
  the feature store, so scoring never waits on a graph traversal.
* a NetworkX view built lazily for the ops console (subgraph extraction,
  centrality, layout). Cold path only.

Ring stats are point-in-time: we read them for the incoming order *before*
we union that order's entities, so training features never see the future.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field

KINDS = ("phone", "device", "addr", "vpa", "ip")
RING_MIN_PHONES = 3          # fewer than this is a household, not a ring
RING_PHONE_DEVICE_RATIO = 2.5  # phones per device above which we call it a ring
_GRAPH_TTL = None


@dataclass
class RingStats:
    ring_id: str
    size: int = 0
    phones: int = 0
    devices: int = 0
    addrs: int = 0
    orders: int = 0
    rto: int = 0
    delivered: int = 0
    gmv: float = 0.0

    @property
    def rto_rate(self) -> float:
        n = self.rto + self.delivered
        return self.rto / n if n else 0.0

    @property
    def is_ring(self) -> bool:
        if self.phones < RING_MIN_PHONES:
            return False
        dev_ratio = self.phones / max(1, self.devices)
        addr_ratio = self.phones / max(1, self.addrs)
        return dev_ratio >= RING_PHONE_DEVICE_RATIO or addr_ratio >= RING_PHONE_DEVICE_RATIO

    def as_dict(self) -> dict:
        return {
            "ring_id": self.ring_id, "size": self.size, "phones": self.phones, "devices": self.devices,
            "addrs": self.addrs, "orders": self.orders, "rto": self.rto, "delivered": self.delivered,
            "rto_rate": round(self.rto_rate, 4), "gmv": round(self.gmv, 2), "is_ring": self.is_ring,
        }


class SyndicateGraph:
    def __init__(self, store=None) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}
        self._stats: dict[str, RingStats] = {}
        self._adj: dict[str, set[str]] = defaultdict(set)
        self._kind: dict[str, str] = {}
        self._lock = threading.RLock()
        self._store = store
        self._order_nodes: dict[str, tuple[str, ...]] = {}   # order_id -> nodes, for outcome updates

    # ------------------------------------------------------------ snapshot
    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        d.pop("_lock", None)
        d.pop("_store", None)
        d["_adj"] = dict(self._adj)
        return d

    def __setstate__(self, d: dict) -> None:
        self.__dict__.update(d)
        self._adj = defaultdict(set, d.get("_adj", {}))
        self._lock = threading.RLock()
        self._store = None

    def dump(self, path) -> None:
        import pickle

        with self._lock, open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path, store=None) -> "SyndicateGraph":
        import pickle

        with open(path, "rb") as fh:
            g = pickle.load(fh)
        g._store = store
        return g

    # ------------------------------------------------------------ union-find
    def _find(self, x: str) -> str:
        p = self._parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def _add_node(self, node: str, kind: str) -> None:
        if node not in self._parent:
            self._parent[node] = node
            self._rank[node] = 0
            self._kind[node] = kind
            st = RingStats(ring_id=node, size=1)
            setattr(st, {"phone": "phones", "device": "devices", "addr": "addrs"}.get(kind, "size"), 1)
            if kind not in ("phone", "device", "addr"):
                st.size = 1
            self._stats[node] = st

    def _union(self, a: str, b: str) -> str:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return ra
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        sa, sb = self._stats[ra], self._stats.pop(rb)
        sa.size += sb.size
        sa.phones += sb.phones
        sa.devices += sb.devices
        sa.addrs += sb.addrs
        sa.orders += sb.orders
        sa.rto += sb.rto
        sa.delivered += sb.delivered
        sa.gmv += sb.gmv
        return ra

    # ---------------------------------------------------------------- reads
    def node_id(self, kind: str, h: str) -> str:
        return f"{kind}:{h}"

    def lookup(self, kind: str, h: str) -> RingStats | None:
        with self._lock:
            n = self.node_id(kind, h)
            if n not in self._parent:
                return None
            return self._stats[self._find(n)]

    def degree(self, kind: str, h: str) -> int:
        return len(self._adj.get(self.node_id(kind, h), ()))

    def features_for(self, entities: dict[str, str]) -> dict:
        """Point-in-time ring features for an order that has NOT been ingested yet."""
        best = RingStats(ring_id="none")
        max_deg = 0
        for kind, h in entities.items():
            if not h:
                continue
            st = self.lookup(kind, h)
            if st and (st.size, st.rto_rate) > (best.size, best.rto_rate):
                best = st
            max_deg = max(max_deg, self.degree(kind, h))
        return {
            "ring_id": best.ring_id if best.size > 1 else None,
            "ring_size": best.size if best.size > 1 else 0,
            "ring_phones": best.phones if best.size > 1 else 0,
            "ring_devices": best.devices if best.size > 1 else 0,
            "ring_rto_rate": best.rto_rate if best.size > 1 else 0.0,
            "ring_orders": best.orders if best.size > 1 else 0,
            "is_ring": best.is_ring if best.size > 1 else False,
            "entity_max_degree": max_deg,
        }

    # --------------------------------------------------------------- writes
    def ingest(self, order_id: str, entities: dict[str, str], gmv: float = 0.0) -> RingStats:
        """Link all entities of an order; returns the (post-union) ring stats."""
        with self._lock:
            nodes = [self.node_id(k, h) for k, h in entities.items() if h]
            for n, (k, h) in zip(nodes, [(k, h) for k, h in entities.items() if h]):
                self._add_node(n, k)
            root = None
            for n in nodes:
                root = n if root is None else self._union(root, n)
            for i, a in enumerate(nodes):
                for b in nodes[i + 1:]:
                    self._adj[a].add(b)
                    self._adj[b].add(a)
            st = self._stats[self._find(root)]
            st.orders += 1
            st.gmv += gmv
            self._order_nodes[order_id] = tuple(nodes)
            self._publish(nodes, st)
            return st

    def outcome(self, order_id: str, rto: bool) -> None:
        with self._lock:
            nodes = self._order_nodes.get(order_id)
            if not nodes:
                return
            st = self._stats[self._find(nodes[0])]
            if rto:
                st.rto += 1
            else:
                st.delivered += 1
            self._publish(nodes, st)

    def _publish(self, nodes, st: RingStats) -> None:
        """Push ring stats to the feature store so the serving path is graph-free."""
        if self._store is None:
            return
        payload = {"ring_id": st.ring_id, "ring_size": st.size, "ring_phones": st.phones,
                   "ring_devices": st.devices, "ring_rto": st.rto, "ring_delivered": st.delivered,
                   "ring_orders": st.orders, "is_ring": int(st.is_ring)}
        for n in nodes:
            self._store.hset(f"graph:{n}", {**payload, "degree": len(self._adj.get(n, ()))})

    # ------------------------------------------------------------ analytics
    def rings(self, min_phones: int = RING_MIN_PHONES, top: int = 50) -> list[dict]:
        with self._lock:
            out = [s.as_dict() for s in self._stats.values() if s.phones >= min_phones and s.is_ring]
        out.sort(key=lambda d: (d["rto_rate"] * d["orders"], d["phones"]), reverse=True)
        return out[:top]

    def subgraph(self, seed: str, max_nodes: int = 120) -> dict:
        """BFS neighbourhood as vis.js-ready nodes/edges. seed = 'kind:hash' or ring id."""
        import networkx as nx  # cold path only

        with self._lock:
            if seed not in self._parent:
                return {"nodes": [], "edges": [], "ring": None}
            root = self._find(seed)
            st = self._stats[root]
            seen, frontier, order = {seed}, [seed], [seed]
            while frontier and len(order) < max_nodes:
                nxt = []
                for n in frontier:
                    for m in self._adj.get(n, ()):
                        if m not in seen:
                            seen.add(m)
                            order.append(m)
                            nxt.append(m)
                            if len(order) >= max_nodes:
                                break
                    if len(order) >= max_nodes:
                        break
                frontier = nxt
            g = nx.Graph()
            for n in order:
                g.add_node(n)
            for n in order:
                for m in self._adj.get(n, ()):
                    if m in seen:
                        g.add_edge(n, m)
            cent = nx.degree_centrality(g) if g.number_of_nodes() > 1 else {n: 0.0 for n in g}
            nodes = [{"id": n, "kind": self._kind.get(n, "?"), "label": n.split(":")[1][:6],
                      "degree": g.degree(n), "centrality": round(cent.get(n, 0.0), 3)} for n in g.nodes]
            edges = [{"from": a, "to": b} for a, b in g.edges]
            return {"nodes": nodes, "edges": edges, "ring": st.as_dict()}

    def stats(self) -> dict:
        with self._lock:
            comps = [s for s in self._stats.values()]
            return {
                "nodes": len(self._parent),
                "components": len(comps),
                "rings": sum(1 for s in comps if s.is_ring),
                "ring_phones": sum(s.phones for s in comps if s.is_ring),
                "orders_ingested": len(self._order_nodes),
            }
