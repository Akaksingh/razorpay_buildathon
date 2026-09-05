"""Inductive GraphSAGE over the entity graph: does message passing beat union-find?

The syndicate guard (syndicate.py) decides ring membership with hand-written
rules on union-find aggregates: phones per device, phones per address, an RTO
rate as corroboration. Those rules are a fixed, two-hop, order-agnostic
summary of the neighbourhood. A graph neural network learns the summary
instead, from labelled rings, and can weigh things the rules cannot (how
risky the *other* phones on this handset have been, whether the drop address
is a hostel or a mule) without collapsing a component. This module is the
experiment that measures whether that is worth anything here.

Why inductive, and why plain torch
----------------------------------
A transductive embedding (node2vec, a matrix factorisation) has to be refit
whenever a new phone arrives, which is exactly when a burner needs scoring.
GraphSAGE learns weights over *features of a neighbourhood*, so a phone that
did not exist at training time is scored by the same forward pass. The graph
is small (fewer than 10^5 nodes, under 10^6 directed edges) so full-batch
mean aggregation with ``index_add_`` is a few milliseconds per layer; there is
no reason to add a geometric-learning dependency for that.

Point-in-time honesty
---------------------
Each snapshot has two clocks: ``order_cutoff`` bounds which orders contribute
structure, ``outcome_cutoff`` bounds which deliveries / returns contribute to
the RTO-rate features. Keeping the second strictly before the window whose
phones are being labelled means the model never sees an outcome it is asked
to predict. Structure is *not* point-in-time within a snapshot (a phone's
later orders are in the graph when it is scored), which is a stated caveat of
the experiment, not of a serving path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data.replay import entity_hashes

KINDS: tuple[str, ...] = ("phone", "device", "addr", "vpa", "ip")
RTO_SHRINK = 5.0          # pseudo-count pulling a node's observed RTO rate toward the graph prior
FEATURE_NAMES: tuple[str, ...] = tuple(f"kind_{k}" for k in KINDS) + (
    "degree_log", "orders_log", "resolved_log", "rto_rate_shrunk",
) + tuple(f"nbr_{k}_log" for k in KINDS)
N_FEATURES = len(FEATURE_NAMES)


# ------------------------------------------------------------------ graph
def hash_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Hash every order's entities once; graph snapshots are cut from this frame."""
    rows = [entity_hashes(r) for r in df.to_dict("records")]
    out = pd.DataFrame(rows, index=df.index)
    out["ts"] = df["ts"].astype(float)
    out["outcome_ts"] = df["outcome_ts"].astype(float)
    out["rto"] = df["rto"].astype(bool)
    out["customer_phone"] = df["customer_phone"].astype(str)
    return out


@dataclass
class EntityGraph:
    """Undirected entity co-occurrence graph with per-node features.

    ``src``/``dst`` hold every edge in both directions, deduplicated, so a mean
    aggregation over ``dst`` is a mean over distinct neighbours.
    """
    node_ids: list[str]
    kind: np.ndarray                       # int index into KINDS
    src: np.ndarray
    dst: np.ndarray
    orders: np.ndarray                     # orders touching each node
    rto: np.ndarray                        # resolved RTOs by outcome_cutoff
    delivered: np.ndarray
    neighbours_by_kind: np.ndarray         # (n, len(KINDS)) distinct neighbours of each kind
    features: np.ndarray                   # (n, N_FEATURES) float32
    order_cutoff: float | None
    outcome_cutoff: float | None
    index: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return int(len(self.src) // 2)

    @property
    def degree(self) -> np.ndarray:
        return self.neighbours_by_kind.sum(axis=1)

    def node(self, kind: str, h: str) -> int:
        """Node index, or -1 for an entity the snapshot has never seen."""
        return self.index.get(f"{kind}:{h}", -1)

    def phone_nodes(self, phone_hashes) -> np.ndarray:
        return np.array([self.node("phone", h) for h in phone_hashes], dtype=np.int64)


def build_entity_graph(hashed: pd.DataFrame, order_cutoff: float | None = None,
                       outcome_cutoff: float | None = None) -> EntityGraph:
    """Snapshot of the entity graph from orders with ts <= order_cutoff.

    Outcomes count only if resolved by ``outcome_cutoff`` (defaults to the
    order cutoff). Node features are the per-node aggregates the syndicate
    rules are built from, so a logistic regression on them is a fair
    "no message passing" control for the GraphSAGE.
    """
    sub = hashed if order_cutoff is None else hashed[hashed["ts"] <= order_cutoff]
    outcome_cutoff = order_cutoff if outcome_cutoff is None else outcome_cutoff
    n_orders = len(sub)
    cols = {}
    for k in KINDS:
        h = sub[k].fillna("").astype(str).to_numpy()
        cols[k] = np.where(h != "", k + ":" + h, "")
    all_ids = np.concatenate([cols[k] for k in KINDS])
    node_ids = [str(x) for x in pd.unique(all_ids[all_ids != ""])]
    index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    lookup = pd.Index(node_ids)
    slots = np.full((n_orders, len(KINDS)), -1, dtype=np.int64)     # order -> node per kind
    for j, k in enumerate(KINDS):
        present = cols[k] != ""
        slots[present, j] = lookup.get_indexer(cols[k][present])
    kind = np.zeros(n, dtype=np.int64)
    for j in range(len(KINDS)):
        kind[slots[slots[:, j] >= 0, j]] = j

    # edges: every pair of entities on an order, both directions, deduplicated
    pairs = []
    for a in range(len(KINDS)):
        for b in range(a + 1, len(KINDS)):
            ok = (slots[:, a] >= 0) & (slots[:, b] >= 0)
            pairs.append(np.stack([slots[ok, a], slots[ok, b]], axis=1))
    if pairs and n_orders:
        e = np.concatenate(pairs, axis=0)
        e = np.concatenate([e, e[:, ::-1]], axis=0)
        e = np.unique(e[:, 0] * n + e[:, 1])
        src, dst = e // n, e % n
    else:
        src = dst = np.zeros(0, dtype=np.int64)

    valid = slots[slots >= 0]
    orders = np.bincount(valid, minlength=n).astype(np.float64)
    resolved_mask = (sub["outcome_ts"].to_numpy() <= outcome_cutoff) if outcome_cutoff is not None else np.ones(n_orders, bool)
    rto_row = sub["rto"].to_numpy().astype(bool) & resolved_mask
    del_row = ~sub["rto"].to_numpy().astype(bool) & resolved_mask
    rto = np.zeros(n)
    delivered = np.zeros(n)
    for j in range(len(KINDS)):
        ok = slots[:, j] >= 0
        rto += np.bincount(slots[ok, j], weights=rto_row[ok], minlength=n)
        delivered += np.bincount(slots[ok, j], weights=del_row[ok], minlength=n)
    nbk = np.zeros((n, len(KINDS)), dtype=np.float64)
    np.add.at(nbk, (dst, kind[src]), 1.0)

    resolved = rto + delivered
    prior = float(rto_row.sum() / resolved_mask.sum()) if resolved_mask.sum() > 0 else 0.2   # order-level rate of the snapshot
    feats = np.zeros((n, N_FEATURES), dtype=np.float32)
    feats[np.arange(n), kind] = 1.0
    feats[:, len(KINDS)] = np.log1p(nbk.sum(axis=1))
    feats[:, len(KINDS) + 1] = np.log1p(orders)
    feats[:, len(KINDS) + 2] = np.log1p(resolved)
    feats[:, len(KINDS) + 3] = (rto + RTO_SHRINK * prior) / (resolved + RTO_SHRINK)
    feats[:, len(KINDS) + 4:] = np.log1p(nbk)
    return EntityGraph(node_ids=node_ids, kind=kind, src=src, dst=dst, orders=orders, rto=rto, delivered=delivered,
                       neighbours_by_kind=nbk, features=feats, order_cutoff=order_cutoff, outcome_cutoff=outcome_cutoff,
                       index=index)


# ------------------------------------------------------------------ model
class SAGELayer(nn.Module):
    """h_u' = W_self h_u + W_nbr mean_{v in N(u)} h_v  (Hamilton et al. 2017, mean aggregator)."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_nbr = nn.Linear(d_in, d_out, bias=False)

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, inv_deg: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros_like(h).index_add_(0, dst, h[src]) * inv_deg
        return self.lin_self(h) + self.lin_nbr(agg)


class GraphSAGE(nn.Module):
    def __init__(self, d_in: int, hidden: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.l1 = SAGELayer(d_in, hidden)
        self.l2 = SAGELayer(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def embed(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, inv_deg: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.l1(x, src, dst, inv_deg))
        h = nn.functional.dropout(h, self.dropout, self.training)
        h = torch.relu(self.l2(h, src, dst, inv_deg))
        return nn.functional.dropout(h, self.dropout, self.training)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, inv_deg: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x, src, dst, inv_deg)).squeeze(-1)


@dataclass
class GraphTensors:
    x: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    inv_deg: torch.Tensor

    @classmethod
    def from_graph(cls, g: EntityGraph, mean: np.ndarray, std: np.ndarray) -> "GraphTensors":
        x = torch.from_numpy(((g.features - mean) / std).astype(np.float32))
        deg = torch.from_numpy(g.degree.astype(np.float32)).clamp(min=1.0)
        return cls(x=x, src=torch.from_numpy(g.src.astype(np.int64)), dst=torch.from_numpy(g.dst.astype(np.int64)),
                   inv_deg=(1.0 / deg).unsqueeze(1))


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 32
    dropout: float = 0.1
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 1e-4
    valid_frac: float = 0.15       # labelled nodes held out for early stopping (never the test era)
    patience: int = 30
    seed: int = 7


def training_step(model: GraphSAGE, opt: torch.optim.Optimizer, t: GraphTensors,
                  y: torch.Tensor, mask: torch.Tensor) -> float:
    """One full-batch gradient step of BCE on the masked nodes; returns the loss before the step."""
    model.train()
    opt.zero_grad()
    logits = model(t.x, t.src, t.dst, t.inv_deg)
    loss = nn.functional.binary_cross_entropy_with_logits(logits[mask], y[mask])
    loss.backward()
    opt.step()
    return float(loss.item())


@torch.no_grad()
def masked_loss(model: GraphSAGE, t: GraphTensors, y: torch.Tensor, mask: torch.Tensor) -> float:
    model.eval()
    logits = model(t.x, t.src, t.dst, t.inv_deg)
    return float(nn.functional.binary_cross_entropy_with_logits(logits[mask], y[mask]).item())


@dataclass
class SageScorer:
    """A trained GraphSAGE plus the feature standardisation it was fitted with.

    Scoring any snapshot, including nodes that did not exist at training time,
    is one forward pass: the weights depend on neighbourhood features only.
    """
    model: GraphSAGE
    mean: np.ndarray
    std: np.ndarray
    config: TrainConfig
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0

    def tensors(self, g: EntityGraph) -> GraphTensors:
        return GraphTensors.from_graph(g, self.mean, self.std)

    @torch.no_grad()
    def score(self, g: EntityGraph) -> np.ndarray:
        """P(label | neighbourhood) for every node of the snapshot."""
        self.model.eval()
        t = self.tensors(g)
        return torch.sigmoid(self.model(t.x, t.src, t.dst, t.inv_deg)).numpy().astype(np.float64)

    @torch.no_grad()
    def embed(self, g: EntityGraph) -> np.ndarray:
        self.model.eval()
        t = self.tensors(g)
        return self.model.embed(t.x, t.src, t.dst, t.inv_deg).numpy()


def standardiser(g: EntityGraph) -> tuple[np.ndarray, np.ndarray]:
    mean = g.features.mean(axis=0)
    std = np.maximum(g.features.std(axis=0), 1e-3)
    mean[: len(KINDS)] = 0.0                # keep the kind one-hot as a plain indicator
    std[: len(KINDS)] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def fit_sage(g: EntityGraph, labels: np.ndarray, config: TrainConfig = TrainConfig()) -> SageScorer:
    """Train on the nodes whose label is not NaN; early-stop on a seeded slice of them.

    Full-batch Adam; the best validation-loss weights are restored, so the
    returned scorer is the epoch the history reports as ``best_epoch``.
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    mean, std = standardiser(g)
    t = GraphTensors.from_graph(g, mean, std)
    labelled = np.flatnonzero(~np.isnan(labels))
    if len(labelled) == 0:
        raise ValueError("no labelled nodes")
    rng.shuffle(labelled)
    n_val = int(round(config.valid_frac * len(labelled))) if len(labelled) >= 20 else 0
    val_idx, tr_idx = labelled[:n_val], labelled[n_val:]
    y = torch.from_numpy(np.nan_to_num(labels).astype(np.float32))
    tr_mask = torch.zeros(g.n_nodes, dtype=torch.bool)
    tr_mask[tr_idx] = True
    val_mask = torch.zeros(g.n_nodes, dtype=torch.bool)
    val_mask[val_idx] = True

    model = GraphSAGE(N_FEATURES, config.hidden, config.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict] = []
    best, best_state, best_epoch, since = math.inf, None, 0, 0
    for epoch in range(config.epochs):
        tr_loss = training_step(model, opt, t, y, tr_mask)
        val_loss = masked_loss(model, t, y, val_mask) if n_val else tr_loss
        history.append({"epoch": epoch, "train_loss": tr_loss, "valid_loss": val_loss})
        if val_loss < best - 1e-5:
            best, best_epoch, since = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return SageScorer(model=model, mean=mean, std=std, config=config, history=history, best_epoch=best_epoch)


def out_of_fold_scores(g: EntityGraph, labels: np.ndarray, folds: int = 5,
                       config: TrainConfig = TrainConfig()) -> np.ndarray:
    """Cross-fitted scores for the labelled nodes (NaN elsewhere).

    Used when a score becomes a *feature* of a downstream model trained on the
    same era: an in-sample GNN score would let the booster copy the labels.
    """
    labelled = np.flatnonzero(~np.isnan(labels))
    rng = np.random.default_rng(config.seed)
    fold_of = rng.integers(0, folds, size=len(labelled))
    out = np.full(g.n_nodes, np.nan)
    for f in range(folds):
        held = labelled[fold_of == f]
        train_labels = labels.copy()
        train_labels[held] = np.nan
        scorer = fit_sage(g, train_labels, config)
        out[held] = scorer.score(g)[held]
    return out
