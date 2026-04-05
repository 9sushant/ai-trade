"""
Ensemble ML signal model: LightGBM + XGBoost + GradientBoosting.
Trade only when majority (2/3) of models agree.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger
from analysis.ml_signal import _extract_features, _build_dataset, _FEATURE_NAMES

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from sklearn.ensemble import GradientBoostingClassifier


class EnsembleSignalModel:
    """
    Trains up to 3 classifiers and uses majority vote for signal filtering.
    Falls back gracefully when libraries are missing.
    """

    def __init__(self):
        self._models:   list  = []
        self._names:    list  = []
        self.is_fitted        = False
        self._pos_rate        = 0.5

    # ------------------------------------------------------------------
    def fit(self, df_list: list[pd.DataFrame], verbose: bool = False) -> "EnsembleSignalModel":
        all_X, all_y = [], []
        for df in df_list:
            if df is None or len(df) < 80:
                continue
            cutoff = int(len(df) * 0.70)
            X, y   = _build_dataset(df.iloc[:cutoff])
            if len(X) > 0:
                all_X.append(X); all_y.append(y)

        if not all_X:
            return self
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        if len(y) < 50:
            return self

        self._pos_rate = float(y.mean())
        pos = int(y.sum()); neg = len(y) - pos
        scale = neg / max(pos, 1)

        candidates = []
        if _HAS_LGB:
            candidates.append(("LightGBM", lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                num_leaves=15, min_child_samples=20,
                scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1,
            )))
        if _HAS_XGB:
            candidates.append(("XGBoost", xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                scale_pos_weight=scale, random_state=42,
                eval_metric="logloss", verbosity=0,
            )))
        candidates.append(("GradientBoosting", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42,
        )))

        for name, model in candidates:
            try:
                model.fit(X, y)
                self._models.append(model)
                self._names.append(name)
                if verbose:
                    logger.info(f"Ensemble: trained {name}")
            except Exception as exc:
                logger.debug(f"Ensemble: {name} failed — {exc}")

        self.is_fitted = len(self._models) > 0
        if verbose:
            logger.info(f"Ensemble: {len(self._models)} models fitted | "
                        f"{len(y)} samples | base rate {self._pos_rate:.2%}")
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, row: pd.Series, df: pd.DataFrame, i: int) -> float:
        """Average probability across all fitted models."""
        if not self.is_fitted:
            return self._pos_rate
        feats = _extract_features(row, df, i)
        if feats is None:
            return self._pos_rate
        probs = []
        for model in self._models:
            try:
                p = float(model.predict_proba(feats.reshape(1, -1))[0][1])
                probs.append(p)
            except Exception:
                pass
        return float(np.mean(probs)) if probs else self._pos_rate

    def predict_vote(self, row: pd.Series, df: pd.DataFrame, i: int,
                     threshold: float = 0.52) -> bool:
        """True if majority of models predict probability > threshold."""
        if not self.is_fitted:
            return True
        feats = _extract_features(row, df, i)
        if feats is None:
            return True
        votes = 0
        for model in self._models:
            try:
                p = float(model.predict_proba(feats.reshape(1, -1))[0][1])
                if p >= threshold:
                    votes += 1
            except Exception:
                pass
        return votes > len(self._models) / 2

    def model_agreement(self, row: pd.Series, df: pd.DataFrame, i: int,
                        threshold: float = 0.52) -> float:
        """Fraction of models that agree (0.0–1.0)."""
        if not self.is_fitted or not self._models:
            return 1.0
        feats = _extract_features(row, df, i)
        if feats is None:
            return 1.0
        agree = 0
        for model in self._models:
            try:
                p = float(model.predict_proba(feats.reshape(1, -1))[0][1])
                if p >= threshold:
                    agree += 1
            except Exception:
                pass
        return agree / len(self._models)

    def feature_importance(self) -> dict:
        if not self.is_fitted:
            return {}
        all_imp = []
        for model in self._models:
            try:
                all_imp.append(model.feature_importances_)
            except Exception:
                pass
        if not all_imp:
            return {}
        avg = np.mean(all_imp, axis=0)
        return dict(zip(_FEATURE_NAMES, avg.tolist()))
