"""Regime-specific ensemble models — separate model per BULL/BEAR/SIDEWAYS."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class RegimeSpecificModels:
    """
    Trains a separate EnsembleSignalModel per market regime.
    At prediction time, uses the model corresponding to the current regime.
    """

    REGIMES = ("BULL", "BEAR", "SIDEWAYS")

    def __init__(self):
        self._models: dict[str, object] = {}
        self.is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, df_list: list[pd.DataFrame], regime_labels: list[str],
            verbose: bool = False):
        """
        df_list      : list of enriched DataFrames
        regime_labels: regime per DataFrame (same length as df_list)
        """
        from analysis.ensemble import EnsembleSignalModel

        regime_dfs: dict[str, list] = {r: [] for r in self.REGIMES}
        for df, regime in zip(df_list, regime_labels):
            if regime in regime_dfs:
                regime_dfs[regime].append(df)

        for regime, dfs in regime_dfs.items():
            if not dfs:
                continue
            model = EnsembleSignalModel()
            model.fit(dfs, verbose=verbose)
            self._models[regime] = model
            if verbose:
                logger.info(f"RegimeModels: fitted {regime} on {len(dfs)} dfs")

        self.is_fitted = bool(self._models)

    def predict_proba(self, regime: str, row, df, i: int) -> float:
        """Use regime-specific model; fall back to any available model."""
        model = self._models.get(regime) or next(iter(self._models.values()), None)
        if model is None or not model.is_fitted:
            return 0.5
        return model.predict_proba(row, df, i)

    def predict_vote(self, regime: str, row, df, i: int,
                     threshold: float = 0.52) -> bool:
        model = self._models.get(regime) or next(iter(self._models.values()), None)
        if model is None or not model.is_fitted:
            return True
        return model.predict_vote(threshold=threshold)

    def get_model(self, regime: str):
        return self._models.get(regime)

    def available_regimes(self) -> list[str]:
        return list(self._models.keys())
