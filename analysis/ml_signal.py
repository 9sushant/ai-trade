"""
ML-based signal filter using LightGBM binary classifier.

Learns which technical/quant feature combinations lead to profitable 5-bar
trades — going beyond hand-coded composite score thresholds.

Label (per signal bar i, in direction of composite_score):
    label = 1  if price moves > +0.3% in the correct direction within 5 bars
    label = 0  otherwise (failed or ambiguous)

Features (19):
    rsi, macd_histogram, adx, adx_pos, adx_neg, stoch_k, stoch_d,
    bb_width, bb_position, volume_ratio, supertrend_dir,
    ema9_vs_ema21, ema21_vs_ema50, close_vs_ema50, atr_pct,
    momentum_score, mean_rev_z, composite_norm, price_change_5d

Usage:
    model = MLSignalModel()
    model.fit(df_list)           # list of indicator-enriched DataFrames
    prob = model.predict_proba(row)  # row = pd.Series from df.iloc[i]
"""
import numpy as np
import pandas as pd
from utils.logger import logger

try:
    import lightgbm as lgb
    _BACKEND = "lightgbm"
except ImportError:
    try:
        from sklearn.ensemble import GradientBoostingClassifier as _SKLearnGBC
        _BACKEND = "sklearn"
    except ImportError:
        _BACKEND = None

_FORWARD_BARS   = 5        # label horizon
_MIN_MOVE_PCT   = 0.003    # +0.3% directional move to count as a win
_MIN_SCORE      = 25       # composite_score threshold to include a bar as a sample
_FEATURE_NAMES  = [
    "rsi", "macd_histogram", "adx", "adx_pos", "adx_neg",
    "stoch_k", "stoch_d", "bb_width", "bb_position", "volume_ratio",
    "supertrend_dir", "ema9_vs_ema21", "ema21_vs_ema50", "close_vs_ema50",
    "atr_pct", "momentum_score", "mean_rev_z", "composite_norm", "price_change_5d",
]


def _extract_features(row: pd.Series, df: pd.DataFrame, i: int) -> np.ndarray | None:
    """Extract feature vector for bar i.  Returns None if data is insufficient."""
    close = float(row.get("close", 0) or 0)
    if close <= 0:
        return None

    bb_upper = float(row.get("bb_upper", 0) or 0)
    bb_lower = float(row.get("bb_lower", 0) or 0)
    bb_range  = bb_upper - bb_lower
    bb_pos    = (close - bb_lower) / bb_range if bb_range > 0 else 0.5

    ema9  = float(row.get("ema_9",  close) or close)
    ema21 = float(row.get("ema_21", close) or close)
    ema50 = float(row.get("ema_50", close) or close)

    atr = float(row.get("atr", 0) or 0)

    price_change_5d = 0.0
    if i >= 5:
        prev_close = float(df["close"].iloc[i - 5])
        if prev_close > 0:
            price_change_5d = (close - prev_close) / prev_close

    feats = [
        float(row.get("rsi", 50)           or 50),
        float(row.get("macd_histogram", 0) or 0),
        float(row.get("adx", 0)            or 0),
        float(row.get("adx_pos", 0)        or 0),
        float(row.get("adx_neg", 0)        or 0),
        float(row.get("stoch_k", 50)       or 50),
        float(row.get("stoch_d", 50)       or 50),
        float(row.get("bb_width", 0)       or 0),
        bb_pos,
        float(row.get("volume_ratio", 1)   or 1),
        float(row.get("supertrend_dir", 0) or 0),
        (ema9 - ema21) / close,
        (ema21 - ema50) / close,
        (close - ema50) / close,
        atr / close if close > 0 else 0,
        float(row.get("momentum_score", 0) or 0),
        float(row.get("mean_rev_z", 0)     or 0),
        float(row.get("composite_score", 0) or 0) / 100.0,
        price_change_5d,
    ]

    arr = np.array(feats, dtype=np.float32)
    if np.any(~np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _build_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) arrays from a single enriched DataFrame."""
    X_rows, y_rows = [], []
    n = len(df)

    for i in range(50, n - _FORWARD_BARS):
        row   = df.iloc[i]
        score = float(row.get("composite_score", 0) or 0)

        if abs(score) < _MIN_SCORE:
            continue

        features = _extract_features(row, df, i)
        if features is None:
            continue

        # Determine signal direction and forward outcome
        close_now = float(row.get("close", 0) or 0)
        if close_now <= 0:
            continue

        future_closes = df["close"].iloc[i + 1: i + _FORWARD_BARS + 1].values

        if score > 0:  # BUY candidate
            # Profitable if any future close is >0.3% above entry
            label = int(np.max(future_closes) >= close_now * (1 + _MIN_MOVE_PCT))
        else:          # SELL candidate
            label = int(np.min(future_closes) <= close_now * (1 - _MIN_MOVE_PCT))

        X_rows.append(features)
        y_rows.append(label)

    if not X_rows:
        return np.empty((0, len(_FEATURE_NAMES))), np.empty(0)

    return np.vstack(X_rows), np.array(y_rows)


class MLSignalModel:
    """
    Wrapper around LightGBM (or sklearn GBM fallback) for signal filtering.

    Call fit() once with a list of enriched DataFrames, then use
    predict_proba() per bar to get the probability that the trade is profitable.
    """

    def __init__(self):
        self._model   = None
        self.is_fitted = False
        self._backend  = _BACKEND
        self._pos_rate = 0.5   # base rate (fallback)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df_list: list[pd.DataFrame], verbose: bool = False) -> "MLSignalModel":
        """Train on a list of enriched DataFrames (train split only — no future data)."""
        if self._backend is None:
            logger.warning("MLSignalModel: neither lightgbm nor sklearn found — ML filter disabled")
            return self

        all_X, all_y = [], []
        for df in df_list:
            if df is None or len(df) < 80:
                continue
            # Use only the first 70% of bars (walk-forward training split)
            cutoff = int(len(df) * 0.70)
            X, y = _build_dataset(df.iloc[:cutoff])
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)

        if not all_X:
            logger.warning("MLSignalModel: no training samples generated")
            return self

        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        if len(y) < 50:
            logger.warning(f"MLSignalModel: only {len(y)} samples — skipping fit")
            return self

        self._pos_rate = float(y.mean())
        pos = int(y.sum())
        neg = len(y) - pos

        if verbose:
            logger.info(f"MLSignalModel: {len(y)} samples | {pos} pos / {neg} neg | "
                        f"base rate {self._pos_rate:.2%}")

        if self._backend == "lightgbm":
            scale = neg / max(pos, 1)
            self._model = lgb.LGBMClassifier(
                n_estimators    = 300,
                learning_rate   = 0.05,
                max_depth       = 4,
                num_leaves      = 15,
                min_child_samples=20,
                scale_pos_weight= scale,
                random_state    = 42,
                n_jobs          = -1,
                verbose         = -1,
            )
        else:  # sklearn fallback
            self._model = _SKLearnGBC(
                n_estimators = 200,
                learning_rate= 0.05,
                max_depth    = 3,
                random_state = 42,
            )

        # Fit with pure numpy arrays so predict_proba won't raise feature-name warnings
        self._model.fit(X, y, feature_name=_FEATURE_NAMES if self._backend == "lightgbm" else "auto")
        self.is_fitted = True

        if verbose:
            logger.info("MLSignalModel: training complete")

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, row: pd.Series, df: pd.DataFrame, i: int) -> float:
        """
        Return probability [0, 1] that the current bar's signal leads to profit.
        Falls back to base rate if model not fitted or features unavailable.
        """
        if not self.is_fitted or self._model is None:
            return self._pos_rate

        features = _extract_features(row, df, i)
        if features is None:
            return self._pos_rate

        try:
            proba = self._model.predict_proba(features.reshape(1, -1))[0][1]
            return float(proba)
        except Exception as exc:
            logger.debug(f"MLSignalModel.predict_proba error: {exc}")
            return self._pos_rate

    # ------------------------------------------------------------------
    # Feature importance (for reporting)
    # ------------------------------------------------------------------

    def feature_importance(self) -> dict:
        if not self.is_fitted:
            return {}
        try:
            if self._backend == "lightgbm":
                imp = self._model.feature_importances_
            else:
                imp = self._model.feature_importances_
            return dict(zip(_FEATURE_NAMES, imp.tolist()))
        except Exception:
            return {}
