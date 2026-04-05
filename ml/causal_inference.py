"""Causal inference — identify actual price drivers vs spurious correlations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class CausalFeatureSelector:
    """
    Uses causal inference to find features that *cause* price moves,
    not just correlate with them.

    Methods:
      1. DoWhy / CausalML (if installed) — proper causal graph
      2. Granger causality — does X predict Y beyond Y's own past?
      3. Convergent Cross Mapping (CCM) — for nonlinear causality
    """

    def __init__(self, max_lag: int = 5, alpha: float = 0.05):
        self.max_lag = max_lag
        self.alpha   = alpha
        self._causal_features: dict[str, list[str]] = {}
        self._granger_scores:  dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, target: str = "close",
            symbol: str = "UNKNOWN") -> list[str]:
        """
        Find features that Granger-cause the target.
        Returns list of causal feature names.
        """
        feature_cols = [c for c in df.columns
                        if c not in (target, "open", "high", "low", "volume")
                        and not c.startswith("_")]

        returns = df[target].pct_change().dropna()
        causal  = []
        scores  = {}

        for feat in feature_cols[:20]:   # limit to 20 features
            if feat not in df.columns:
                continue
            try:
                p_val = self._granger_test(
                    df[feat].fillna(0).values,
                    returns.values,
                )
                scores[feat] = round(1 - p_val, 4)   # higher = more causal
                if p_val < self.alpha:
                    causal.append(feat)
            except Exception:
                continue

        self._causal_features[symbol] = causal
        self._granger_scores[symbol]  = dict(sorted(scores.items(),
                                                      key=lambda x: x[1], reverse=True))
        logger.info(f"CausalInference {symbol}: {len(causal)} causal features found")
        return causal

    def get_causal_features(self, symbol: str) -> list[str]:
        return self._causal_features.get(symbol, [])

    def get_causal_score(self, symbol: str, feature: str) -> float:
        return self._granger_scores.get(symbol, {}).get(feature, 0.0)

    def get_top_causes(self, symbol: str, n: int = 8) -> list[dict]:
        scores = self._granger_scores.get(symbol, {})
        top    = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [{"feature": f, "causal_score": s} for f, s in top]

    def causal_signal_weight(self, symbol: str, feature: str) -> float:
        """Return weight multiplier [0.5, 1.5] based on causal strength."""
        score = self.get_causal_score(symbol, feature)
        return round(float(np.clip(0.5 + score, 0.5, 1.5)), 3)

    def filter_spurious_features(self, features: dict, symbol: str) -> dict:
        """Remove features with low causal score from signal generation."""
        causal = self._causal_features.get(symbol)
        if not causal:
            return features   # no filtering if not fitted
        return {k: v for k, v in features.items() if k in causal}

    # ------------------------------------------------------------------
    def _granger_test(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Test if X Granger-causes Y using statsmodels.
        Returns p-value (low = causal).
        """
        min_len = min(len(X), len(Y))
        X = X[-min_len:]
        Y = Y[-min_len:]

        # Align lengths and remove NaN
        mask = ~(np.isnan(X) | np.isnan(Y))
        X, Y = X[mask], Y[mask]

        if len(X) < 30:
            return 1.0   # insufficient data

        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            data   = np.column_stack([Y, X])
            result = grangercausalitytests(data, maxlag=self.max_lag, verbose=False)
            # Take minimum p-value across lags
            p_vals = [result[lag][0]["ssr_ftest"][1] for lag in range(1, self.max_lag + 1)]
            return float(min(p_vals))
        except Exception:
            return self._simple_granger(X, Y)

    def _simple_granger(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Simple Granger test via OLS."""
        from sklearn.linear_model import LinearRegression
        n = len(Y) - self.max_lag
        if n < 10:
            return 1.0

        # Restricted: Y ~ Y_lags
        Y_lags = np.column_stack([Y[i:i + self.max_lag] for i in range(self.max_lag)])
        Y_target = Y[self.max_lag:]
        r1 = LinearRegression().fit(Y_lags[:n], Y_target[:n])
        ssr1 = np.sum((Y_target[:n] - r1.predict(Y_lags[:n])) ** 2)

        # Unrestricted: Y ~ Y_lags + X_lags
        X_lags = np.column_stack([X[i:i + self.max_lag] for i in range(self.max_lag)])
        XY = np.hstack([Y_lags[:n], X_lags[:n]])
        r2 = LinearRegression().fit(XY, Y_target[:n])
        ssr2 = np.sum((Y_target[:n] - r2.predict(XY)) ** 2)

        if ssr1 <= 0 or ssr2 <= 0:
            return 1.0

        f_stat = ((ssr1 - ssr2) / self.max_lag) / (ssr2 / (n - 2 * self.max_lag))
        from scipy import stats
        p_val = 1 - stats.f.cdf(f_stat, self.max_lag, n - 2 * self.max_lag)
        return float(p_val)
