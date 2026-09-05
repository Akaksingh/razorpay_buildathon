"""Subgraph Abuse Sentinel: incremental entity graph + syndicate ring detection.

Nodes are hashed entities (phone / device / addr / vpa / card / ip). Every
order adds a clique among the entities it touches. A *ring* is a connected
component that violates the one-human-one-device prior: many phones on few
devices, or a drop address whose parcels keep coming back.

Edge typing -- the component-collapse guard
-------------------------------------------
Plain union-find treats every shared entity as proof of the same actor. One
corporate NAT, one college hostel, one dynamic IP then folds thousands of
strangers into a single giant component and every one of them inherits the
ring's RTO rate. So edges are typed:

* HARD identifiers (phone, device, vpa, card) are transitive merge edges.
* SOFT identifiers (ip) never merge. They are kept as bipartite properties:
  adjacency for the console and degree for features, no transitivity.
* addr is semi-hard: it merges until it looks *public* -- ADDR_MERGE_CEILING
  distinct phones -- after which it is marked SHARED and stops bridging.

Any non-hard node whose degree exceeds SHARED_DEGREE_CEILING is likewise
flagged SHARED. The flag is published to the feature store as
``entity_shared`` so the model can learn that a public address is not a ring.

Ring status needs corroboration: a phone/device ratio (hard evidence), or an
address ratio *with* a high observed RTO rate. A hostel with a normal return
rate is never a ring.

Two data structures, two jobs: union-find with per-root aggregates on the
hot path (published to the feature store), a NetworkX view built lazily for
the console. Ring stats are point-in-time: read before the incoming order's
entities are unioned, so training features never see the future.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass

KINDS = ("phone", "device", "addr", "vpa", "card", "ip")
HARD_KINDS = frozenset({"phone", "device", "vpa", "card"})
SOFT_KINDS = frozenset({"ip"})
ADDR_MERGE_CEILING = 25        # distinct phones at one address before it is public (hostel, office, PG)
SHARED_DEGREE_CEILING = 50     # any non-hard node above this degree is flagged shared
RING_MIN_PHONES = 3            # fewer than this is a household, not a ring
RING_PHONE_DEVICE_RATIO = 2.5  # phones per device (or per address) above which we call it a ring
RING_ADDR_RTO_MIN = 0.45       # address-ratio rings need behavioural corroboration ...
RING_ADDR_MIN_OUTCOMES = 5     # ... on at least this many resolved orders


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
        if self.phones / max(1, self.devices) >= RING_PHONE_DEVICE_RATIO:
            return True                                   # many phones through few handsets: hard evidence
        addr_ratio = self.phones / max(1, self.addrs)
        n = self.rto + self.delivered
        return addr_ratio >= RING_PHONE_DEVICE_RATIO and n >= RING_ADDR_MIN_OUTCOMES and self.rto_rate >= RING_ADDR_RTO_MIN

    def as_dict(self) -> dict:
        return {
            "ring_id": self.ring_id, "size": self.size, "phones": self.phones, "devices": self.devices,
            "addrs": self.addrs, "orders": self.orders, "rto": self.rto, "delivered": self.delivered,
            "rto_rate": round(self.rto_rate, 4), "gmv": round(self.gmv, 2), "is_ring": self.is_ring,
        }


class SyndicateGraph:
    def __init__(self, store=None, guard: bool = True) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}
        self._stats: dict[str, RingStats] = {}
        self._adj: dict[str, set[str]] = defaultdict(set)
        self._kind: dict[str, str] = {}
        self._phone_deg: dict[str, int] = defaultdict(int)   # addr node -> distinct phone neighbours
        self._shared: set[str] = set()
        self.guard = guard
        self._lock = threading.RLock()
        self._store = store
        self._order_nodes: dict[str, tuple[str, ...]] = {}   # order_id -> nodes, for outcome updates

    # ------------------------------------------------------------ snapshot
    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        d.pop("_lock", None)
        d.pop("_store", None)
        d["_adj"] = dict(self._adj)
        d["_phone_deg"] = dict(self._phone_deg)
        return d

    def __setstate__(self, d: dict) -> None:
        self.__dict__.update(d)
        self._adj = defaultdict(set, d.get("_adj", {}))
        self._phone_deg = defaultdict(int, d.get("_phone_deg", {}))
        self._shared = set(d.get("_shared", ()))
        self.guard = d.get("guard", True)
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
            if kind == "phone":
                st.phones = 1
            elif kind == "device":
                st.devices = 1
            elif kind == "addr":
                st.addrs = 1
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

    def _merges(self, node: str) -> bool:
        """May this node act as a transitive merge edge right now?"""
        if not self.guard:
            return True
        k = self._kind[node]
        if k in HARD_KINDS:
            return True
        if k == "addr":
            return node not in self._shared
        return False                      # soft identifiers never merge

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

    def is_shared(self, kind: str, h: str) -> bool:
        return self.node_id(kind, h) in self._shared

    def features_for(self, entities: dict[str, str]) -> dict:
        """Point-in-time ring features for an order that has NOT been ingested yet."""
        best = RingStats(ring_id="none")
        max_deg, shared = 0, False
        for kind, h in entities.items():
            if not h:
                continue
            st = self.lookup(kind, h)
            if st and (st.size, st.rto_rate) > (best.size, best.rto_rate):
                best = st
            max_deg = max(max_deg, self.degree(kind, h))
            shared = shared or self.is_shared(kind, h)
        return {
            "ring_id": best.ring_id if best.size > 1 else None,
            "ring_size": best.size if best.size > 1 else 0,
            "ring_phones": best.phones if best.size > 1 else 0,
            "ring_devices": best.devices if best.size > 1 else 0,
            "ring_rto_rate": best.rto_rate if best.size > 1 else 0.0,
            "ring_orders": best.orders if best.size > 1 else 0,
            "is_ring": best.is_ring if best.size > 1 else False,
            "entity_max_degree": max_deg,
            "entity_shared": shared,
        }

    # --------------------------------------------------------------- writes
    def ingest(self, order_id: str, entities: dict[str, str], gmv: float = 0.0) -> RingStats:
        """Link all entities of an order; returns the (post-union) ring stats of its phone."""
        with self._lock:
            pairs = [(k, h) for k, h in entities.items() if h]
            nodes = [self.node_id(k, h) for k, h in pairs]
            for n, (k, _) in zip(nodes, pairs):
                self._add_node(n, k)
            # bipartite adjacency for every kind (console + degree features)
            for i, a in enumerate(nodes):
                for b in nodes[i + 1:]:
                    if b in self._adj[a]:
                        continue
                    self._adj[a].add(b)
                    self._adj[b].add(a)
                    ka, kb = self._kind[a], self._kind[b]
                    if ka == "addr" and kb == "phone":
                        self._phone_deg[a] += 1
                    elif kb == "addr" and ka == "phone":
                        self._phone_deg[b] += 1
            # shared-entity guard: a public address / high-degree soft node stops bridging
            for n in nodes:
                k = self._kind[n]
                if k in HARD_KINDS:
                    continue
                if (k == "addr" and self._phone_deg[n] >= ADDR_MERGE_CEILING) or len(self._adj[n]) >= SHARED_DEGREE_CEILING:
                    self._shared.add(n)
            # transitive merge only through merge-eligible nodes
            merge = [n for n in nodes if self._merges(n)]
            root = None
            for n in merge:
                root = n if root is None else self._union(root, n)
            if root is None:
                root = nodes[0]
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
        """Push ring stats to the feature store so the serving path is graph-free.

        Nodes that do not merge (soft / shared) publish their *own* singleton
        stats, so a stranger on the same IP or in the same hostel never reads
        the ring's numbers.
        """
        if self._store is None:
            return
        payload = {"ring_id": st.ring_id, "ring_size": st.size, "ring_phones": st.phones,
                   "ring_devices": st.devices, "ring_rto": st.rto, "ring_delivered": st.delivered,
                   "ring_orders": st.orders, "is_ring": int(st.is_ring)}
        for n in nodes:
            own = payload if (self._find(n) == st.ring_id or not self.guard) else self._own_payload(n)
            self._store.hset(f"graph:{n}", {**own, "degree": len(self._adj.get(n, ())), "shared": int(n in self._shared)})

    def _own_payload(self, n: str) -> dict:
        s = self._stats[self._find(n)]
        return {"ring_id": s.ring_id, "ring_size": s.size, "ring_phones": s.phones, "ring_devices": s.devices,
                "ring_rto": s.rto, "ring_delivered": s.delivered, "ring_orders": s.orders, "is_ring": int(s.is_ring)}

    # ------------------------------------------------------------ analytics
    def rings(self, min_phones: int = RING_MIN_PHONES, top: int = 50) -> list[dict]:
        with self._lock:
            out = [s.as_dict() for s in self._stats.values() if s.phones >= min_phones and s.is_ring]
        out.sort(key=lambda d: (d["rto_rate"] * d["orders"], d["phones"]), reverse=True)
        return out[:top]

    def shared_entities(self, top: int = 50) -> list[dict]:
        with self._lock:
            out = [{"id": n, "kind": self._kind[n], "degree": len(self._adj.get(n, ())),
                    "phones": self._phone_deg.get(n, 0)} for n in self._shared]
        out.sort(key=lambda d: d["degree"], reverse=True)
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
                      "degree": g.degree(n), "centrality": round(cent.get(n, 0.0), 3),
                      "shared": n in self._shared, "merges": self._merges(n)} for n in g.nodes]
            edges = [{"from": a, "to": b, "hard": self._merges(a) and self._merges(b)} for a, b in g.edges]
            return {"nodes": nodes, "edges": edges, "ring": st.as_dict()}

    def stats(self) -> dict:
        with self._lock:
            comps = list(self._stats.values())
            return {
                "nodes": len(self._parent),
                "components": len(comps),
                "rings": sum(1 for s in comps if s.is_ring),
                "ring_phones": sum(s.phones for s in comps if s.is_ring),
                "largest_component": max((s.size for s in comps), default=0),
                "shared_entities": len(self._shared),
                "guard": self.guard,
                "orders_ingested": len(self._order_nodes),
            }
