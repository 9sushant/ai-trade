"""
Hidden Markov Model for market regime detection.
Automatically discovers BULL/BEAR/SIDEWAYS states from price data.
Falls back to rule-based detection if hmmlearn not installed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger

try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMM = True
except ImportError:
    _HAS_HMM = False


class HMMRegimeDetector:
    def __init__(self, n_states: int = 3):
        self.n_states  = n_states
        self._model    = None
        self._state_map: dict[int, str] = {}   # state_idx -> "BULL"/"BEAR"/"SIDEWAYS"
        self.is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        if not _HAS_HMM:
            logger.debug("hmmlearn not installed — HMM using rule-based fallback")
            return self
        if df is None or len(df) < 60:
            return self
        try:
            features = self._build_features(df)
            self._model = GaussianHMM(
                n_components=self.n_states, covariance_type="diag",
                n_iter=200, random_state=42,
            )
            self._model.fit(features)
            self._label_states(features)
            self.is_fitted = True
        except Exception as exc:
            logger.debug(f"HMM fit error: {exc}")
        return self

    # ------------------------------------------------------------------
    def get_current_regime(self, df: pd.DataFrame) -> str:
        if not self.is_fitted or self._model is None:
            return self._rule_based_regime(df)
        try:
            features  = self._build_features(df)
            states    = self._model.predict(features)
            last_state = int(states[-1])
            return self._state_map.get(last_state, "SIDEWAYS")
        except Exception:
            return self._rule_based_regime(df)

    def get_regime_series(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not self.is_fitted or self._model is None:
            out["hmm_regime"] = self._rule_based_regime(df)
            return out
        try:
            features   = self._build_features(df)
            states     = self._model.predict(features)
            regimes    = [self._state_map.get(int(s), "SIDEWAYS") for s in states]
            # Pad to match df length (features start from row 1)
            padded     = ["SIDEWAYS"] + regimes
            out["hmm_regime"] = padded[:len(df)]
        except Exception:
            out["hmm_regime"] = "SIDEWAYS"
        return out

    def regime_to_int(self, regime: str) -> int:
        return {"BULL": 1, "SIDEWAYS": 0, "BEAR": -1}.get(regime, 0)

    # ------------------------------------------------------------------
    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        close   = df["close"]
        ret     = close.pct_change().fillna(0)
        vol5    = ret.rolling(5).std().fillna(0)
        abs_ret = ret.abs()
        vol_r   = df.get("volume_ratio", pd.Series(1, index=df.index)).fillna(1)
        feats   = np.column_stack([
            ret.values,
            vol5.values,
            abs_ret.values,
            vol_r.values,
        ])[1:]   # drop first NaN row
        return feats

    def _label_states(self, features: np.ndarray):
        """Assign BULL/BEAR/SIDEWAYS labels by each state's mean return."""
        states   = self._model.predict(features)
        means    = {}
        for s in range(self.n_states):
            mask   = states == s
            if mask.sum() > 0:
                means[s] = float(features[mask, 0].mean())
            else:
                means[s] = 0.0
        sorted_states = sorted(means, key=means.get)
        if self.n_states == 3:
            self._state_map = {
                sorted_states[0]: "BEAR",
                sorted_states[1]: "SIDEWAYS",
                sorted_states[2]: "BULL",
            }
        else:
            mid  = self.n_states // 2
            for i, s in enumerate(sorted_states):
                if i < mid:         self._state_map[s] = "BEAR"
                elif i == mid:      self._state_map[s] = "SIDEWAYS"
                else:               self._state_map[s] = "BULL"

    def _rule_based_regime(self, df: pd.DataFrame) -> str:
        """Fallback: EMA200 + ADX rule."""
        if df is None or len(df) < 20:
            return "SIDEWAYS"
        try:
            close  = df["close"]
            ema200 = close.ewm(span=200, adjust=False).mean()
            adx    = df.get("adx", pd.Series(25, index=df.index))
            last_close = float(close.iloc[-1])
            last_ema   = float(ema200.iloc[-1])
            last_adx   = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 25
            if last_adx < 20:
                return "SIDEWAYS"
            return "BULL" if last_close > last_ema else "BEAR"
        except Exception:
            return "SIDEWAYS"
