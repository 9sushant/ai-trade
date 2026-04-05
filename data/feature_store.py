"""Real-time feature store — sub-second feature computation for live trading."""
from __future__ import annotations
import json
import threading
import time
import hashlib
from datetime import datetime
from utils.logger import logger

_DEFAULT_TTL = 60   # 1 minute for real-time features


class FeatureStore:
    """
    In-memory feature store with optional Redis backend.

    Caches computed features (technical indicators, ML scores, etc.)
    so the live trading loop doesn't recompute for every signal check.

    Features expire after TTL seconds — stale features are recomputed.
    """

    def __init__(self, use_redis: bool = False, redis_host: str = "localhost",
                 redis_port: int = 6379, default_ttl: int = _DEFAULT_TTL):
        self._use_redis  = use_redis and self._check_redis()
        self._ttl        = default_ttl
        self._store: dict[str, tuple[object, float]] = {}   # key → (value, expiry_ts)
        self._lock       = threading.Lock()
        self._redis      = None
        self._hits       = 0
        self._misses     = 0

        if self._use_redis:
            try:
                import redis
                self._redis = redis.Redis(host=redis_host, port=redis_port,
                                          decode_responses=True)
                self._redis.ping()
                logger.info("FeatureStore: Redis backend connected")
            except Exception as exc:
                logger.warning(f"FeatureStore: Redis failed ({exc}), using in-memory")
                self._use_redis = False

    # ------------------------------------------------------------------
    def get(self, key: str) -> object | None:
        """Retrieve a feature value. Returns None if expired or missing."""
        if self._use_redis:
            return self._redis_get(key)

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, expiry = entry
            if time.time() > expiry:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return value

    def set(self, key: str, value: object, ttl: int = None):
        """Store a feature value with TTL."""
        ttl = ttl or self._ttl
        if self._use_redis:
            self._redis_set(key, value, ttl)
            return

        with self._lock:
            self._store[key] = (value, time.time() + ttl)

    def get_or_compute(self, key: str, compute_fn, ttl: int = None) -> object:
        """Get cached value or compute and cache it."""
        value = self.get(key)
        if value is not None:
            return value
        value = compute_fn()
        self.set(key, value, ttl)
        return value

    def set_features(self, symbol: str, features: dict, ttl: int = None):
        """Store all features for a symbol at once."""
        for feat_name, feat_val in features.items():
            self.set(f"{symbol}:{feat_name}", feat_val, ttl)
        self.set(f"{symbol}:__ts__", datetime.now().isoformat(), ttl)

    def get_features(self, symbol: str, feature_names: list[str]) -> dict:
        """Retrieve multiple features for a symbol."""
        return {
            feat: self.get(f"{symbol}:{feat}")
            for feat in feature_names
        }

    def get_all_symbol_features(self, symbol: str) -> dict:
        """Get all cached features for a symbol."""
        prefix = f"{symbol}:"
        with self._lock:
            result = {}
            for key, (val, expiry) in list(self._store.items()):
                if key.startswith(prefix) and time.time() <= expiry:
                    feat_name = key[len(prefix):]
                    result[feat_name] = val
            return result

    def invalidate(self, symbol: str = None):
        """Invalidate cache for a symbol or all symbols."""
        if symbol is None:
            with self._lock:
                self._store.clear()
        else:
            prefix = f"{symbol}:"
            with self._lock:
                keys = [k for k in self._store if k.startswith(prefix)]
                for k in keys:
                    del self._store[k]

    # ------------------------------------------------------------------
    def compute_and_store_features(self, symbol: str, df, row=None):
        """Compute all standard features and cache them."""
        if df is None or len(df) < 2:
            return {}

        import numpy as np
        last = df.iloc[-1] if row is None else row

        features = {}
        for col in ["rsi", "macd", "adx", "atr", "volume_ratio",
                    "composite_score", "momentum_score", "mean_rev_z",
                    "bb_upper", "bb_lower", "beta_60"]:
            val = last.get(col, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                features[col] = float(val)

        features["close"]       = float(last.get("close",  0) or 0)
        features["ret_1d"]      = float(df["close"].pct_change().iloc[-1] or 0)
        features["ret_5d"]      = float(df["close"].iloc[-1] / df["close"].iloc[-6] - 1) \
                                   if len(df) >= 6 else 0
        features["vol_20d"]     = float(df["close"].pct_change().rolling(20).std().iloc[-1] or 0)
        features["updated_at"]  = datetime.now().isoformat()

        self.set_features(symbol, features, ttl=30)   # 30s TTL for live features
        return features

    def stats(self) -> dict:
        with self._lock:
            n       = len(self._store)
            expired = sum(1 for _, (_, exp) in self._store.items() if time.time() > exp)
        total = self._hits + self._misses
        return {
            "keys":      n,
            "expired":   expired,
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  round(self._hits / max(total, 1), 3),
        }

    # ------------------------------------------------------------------
    def _redis_get(self, key: str) -> object | None:
        try:
            val = self._redis.get(key)
            if val is None:
                self._misses += 1
                return None
            self._hits += 1
            return json.loads(val)
        except Exception:
            return None

    def _redis_set(self, key: str, value: object, ttl: int):
        try:
            self._redis.setex(key, ttl, json.dumps(value, default=str))
        except Exception:
            pass

    @staticmethod
    def _check_redis() -> bool:
        try:
            import redis  # noqa
            return True
        except ImportError:
            return False
