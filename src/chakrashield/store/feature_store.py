"""Feature store: Redis in production, an in-process store everywhere else.

One interface, two backends. The serving path only needs four primitives
(get / set-with-ttl / incr-with-ttl / hash & set ops), so keeping the
surface tiny lets the same code run on a laptop with no Redis and on a
cluster with one. If REDIS_URL is set and reachable we use it; otherwise we
fall back silently and announce which backend is active on /healthz.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Iterable


class MemoryStore:
    """Thread-safe dict with per-key expiry. Semantics mirror the Redis subset used."""

    backend = "memory"

    def __init__(self) -> None:
        self._kv: dict[str, Any] = {}
        self._exp: dict[str, float] = {}
        self._lock = threading.RLock()

    # -- internal ----------------------------------------------------------
    def _alive(self, key: str) -> bool:
        exp = self._exp.get(key)
        if exp is not None and exp < time.time():
            self._kv.pop(key, None)
            self._exp.pop(key, None)
            return False
        return key in self._kv

    def _touch(self, key: str, ttl: int | None) -> None:
        if ttl:
            self._exp[key] = time.time() + ttl

    # -- string ------------------------------------------------------------
    def get(self, key: str) -> str | None:
        with self._lock:
            return self._kv.get(key) if self._alive(key) else None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        with self._lock:
            self._kv[key] = value if isinstance(value, str) else json.dumps(value)
            self._touch(key, ttl)

    def incr(self, key: str, ttl: int | None = None, by: int = 1) -> int:
        with self._lock:
            cur = int(self._kv.get(key, 0)) if self._alive(key) else 0
            cur += by
            self._kv[key] = str(cur)
            if ttl and key not in self._exp:
                self._touch(key, ttl)
            return cur

    # -- hash --------------------------------------------------------------
    def hgetall(self, key: str) -> dict[str, str]:
        with self._lock:
            return dict(self._kv.get(key, {})) if self._alive(key) else {}

    def hset(self, key: str, mapping: dict[str, Any], ttl: int | None = None) -> None:
        with self._lock:
            h = self._kv.get(key) if self._alive(key) else None
            if not isinstance(h, dict):
                h = {}
            h.update({k: str(v) for k, v in mapping.items()})
            self._kv[key] = h
            self._touch(key, ttl)

    def hincrby(self, key: str, field: str, by: int = 1, ttl: int | None = None) -> int:
        with self._lock:
            h = self._kv.get(key) if self._alive(key) else None
            if not isinstance(h, dict):
                h = {}
            h[field] = str(int(h.get(field, 0)) + by)
            self._kv[key] = h
            self._touch(key, ttl)
            return int(h[field])

    # -- set ---------------------------------------------------------------
    def sadd(self, key: str, *members: str, ttl: int | None = None) -> int:
        with self._lock:
            s = self._kv.get(key) if self._alive(key) else None
            if not isinstance(s, set):
                s = set()
            before = len(s)
            s.update(members)
            self._kv[key] = s
            self._touch(key, ttl)
            return len(s) - before

    def scard(self, key: str) -> int:
        with self._lock:
            s = self._kv.get(key) if self._alive(key) else None
            return len(s) if isinstance(s, set) else 0

    def smembers(self, key: str) -> set[str]:
        with self._lock:
            s = self._kv.get(key) if self._alive(key) else None
            return set(s) if isinstance(s, set) else set()

    # -- admin -------------------------------------------------------------
    def keys(self, pattern_prefix: str) -> list[str]:
        with self._lock:
            return [k for k in list(self._kv) if k.startswith(pattern_prefix) and self._alive(k)]

    def flush(self) -> None:
        with self._lock:
            self._kv.clear()
            self._exp.clear()

    def ping(self) -> bool:
        return True

    # -- snapshot (demo warm-start; Redis persists itself) -----------------
    def dump(self, path, clock_ts: float | None = None) -> None:
        import pickle

        with self._lock, open(path, "wb") as fh:
            pickle.dump({"kv": self._kv, "exp": self._exp, "clock_ts": clock_ts}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path) -> float | None:
        import pickle

        with open(path, "rb") as fh:
            d = pickle.load(fh)
        with self._lock:
            self._kv, self._exp = d["kv"], d["exp"]
        return d.get("clock_ts")


class RedisStore:
    backend = "redis"

    def __init__(self, url: str) -> None:
        import redis  # local import: optional dependency

        self._r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=0.05, socket_connect_timeout=0.2)

    def get(self, key):                      return self._r.get(key)
    def set(self, key, value, ttl=None):
        v = value if isinstance(value, str) else json.dumps(value)
        self._r.set(key, v, ex=ttl)
    def incr(self, key, ttl=None, by=1):
        p = self._r.pipeline()
        p.incrby(key, by)
        if ttl:
            p.expire(key, ttl, nx=True)
        return int(p.execute()[0])
    def hgetall(self, key):                  return self._r.hgetall(key)
    def hset(self, key, mapping, ttl=None):
        p = self._r.pipeline()
        p.hset(key, mapping={k: str(v) for k, v in mapping.items()})
        if ttl:
            p.expire(key, ttl)
        p.execute()
    def hincrby(self, key, field, by=1, ttl=None):
        p = self._r.pipeline()
        p.hincrby(key, field, by)
        if ttl:
            p.expire(key, ttl)
        return int(p.execute()[0])
    def sadd(self, key, *members, ttl=None):
        p = self._r.pipeline()
        p.sadd(key, *members)
        if ttl:
            p.expire(key, ttl)
        return int(p.execute()[0])
    def scard(self, key):                    return int(self._r.scard(key))
    def smembers(self, key):                 return set(self._r.smembers(key))
    def keys(self, pattern_prefix):          return list(self._r.scan_iter(match=pattern_prefix + "*"))
    def flush(self):                         self._r.flushdb()
    def ping(self):
        try:
            return bool(self._r.ping())
        except Exception:
            return False


_STORE: MemoryStore | RedisStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> MemoryStore | RedisStore:
    """Process-wide singleton. Redis if REDIS_URL is reachable, else memory."""
    global _STORE
    if _STORE is not None:
        return _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            return _STORE
        url = os.environ.get("REDIS_URL")
        if url:
            try:
                rs = RedisStore(url)
                if rs.ping():
                    _STORE = rs
                    return _STORE
            except Exception:
                pass
        _STORE = MemoryStore()
        return _STORE


def reset_store() -> None:
    global _STORE
    with _STORE_LOCK:
        _STORE = None
