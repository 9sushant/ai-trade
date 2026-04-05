"""SHAP-based explainability — why did the model generate this signal?"""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger

_FEATURE_NAMES = [
    "rsi", "macd", "macd_signal", "bb_position", "adx",
    "volume_ratio", "atr_pct", "ema_cross", "momentum_5d",
    "momentum_20d", "mean_rev_z", "composite_score",
    "quant_score", "trend_strength", "volatility_regime",
    "sector_score", "fii_score", "sentiment_score",
    "beta_60",
]


class SignalExplainer:
    """
    Explains model predictions using SHAP values.

    Falls back to permutation importance if SHAP not installed.
    """

    def __init__(self):
        self._explainer = None
        self._model     = None
        self._has_shap  = self._check_shap()

    @staticmethod
    def _check_shap() -> bool:
        try:
            import shap  # noqa
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    def attach_model(self, model, X_background: np.ndarray):
        """Attach a fitted sklearn/LightGBM/XGBoost model."""
        self._model = model
        if self._has_shap:
            try:
                import shap
                # TreeExplainer for tree models, KernelExplainer otherwise
                try:
                    self._explainer = shap.TreeExplainer(model)
                except Exception:
                    bg = shap.sample(X_background, 100)
                    self._explainer = shap.KernelExplainer(
                        model.predict_proba, bg
                    )
                logger.info("SignalExplainer: SHAP explainer ready")
            except Exception as exc:
                logger.debug(f"SHAP init error: {exc}")

    def explain_prediction(
        self,
        features: np.ndarray,
        feature_names: list[str] = None,
        top_n: int = 8,
    ) -> dict:
        """
        Explain a single prediction.

        Returns:
          - top contributing features (positive = pushed toward BUY)
          - base value (average model output)
          - predicted probability
        """
        feature_names = feature_names or _FEATURE_NAMES[:len(features)]

        if self._model is None:
            return {"error": "No model attached"}

        # Get prediction
        x = features.reshape(1, -1)
        try:
            pred_prob = float(self._model.predict_proba(x)[0][1])
        except Exception:
            pred_prob = 0.5

        if self._has_shap and self._explainer is not None:
            return self._shap_explanation(x, feature_names, pred_prob, top_n)
        else:
            return self._permutation_explanation(x, feature_names, pred_prob, top_n)

    def explain_signal(self, row: pd.Series, df: pd.DataFrame, i: int,
                       model, top_n: int = 8) -> dict:
        """Explain a signal generated at bar i."""
        from analysis.ml_signal import _extract_features, _FEATURE_NAMES as FN
        try:
            features = _extract_features(row, df, i)
            if self._model is None:
                self.attach_model(model, np.array([features]))
            return self.explain_prediction(np.array(features), FN, top_n)
        except Exception as exc:
            return {"error": str(exc)}

    def explain_ensemble(self, ensemble_model, row, df, i, top_n: int = 8) -> dict:
        """Explain the ensemble decision by explaining each sub-model."""
        results = {}
        for name, model in zip(ensemble_model._names, ensemble_model._models):
            try:
                from analysis.ml_signal import _extract_features, _FEATURE_NAMES as FN
                feat = _extract_features(row, df, i)
                exp  = SignalExplainer()
                exp.attach_model(model, np.array([feat]))
                results[name] = exp.explain_prediction(np.array(feat), FN, top_n)
            except Exception:
                results[name] = {}
        return {
            "ensemble_explanations": results,
            "agreement": ensemble_model.model_agreement(row, df, i)
                         if hasattr(ensemble_model, "model_agreement") else None,
        }

    def feature_importance_report(self, model, X: np.ndarray,
                                   feature_names: list[str] = None) -> dict:
        """Return global feature importance from the model."""
        feature_names = feature_names or _FEATURE_NAMES[:X.shape[1]]

        # Try SHAP global importance
        if self._has_shap:
            try:
                import shap
                exp = shap.TreeExplainer(model)
                sv  = exp.shap_values(X[:min(500, len(X))])
                if isinstance(sv, list):
                    sv = sv[1]   # positive class
                importance = np.abs(sv).mean(axis=0)
                sorted_idx = np.argsort(importance)[::-1]
                return {
                    "method": "SHAP",
                    "features": [
                        {"feature": feature_names[j],
                         "importance": round(float(importance[j]), 4)}
                        for j in sorted_idx[:15]
                    ],
                }
            except Exception:
                pass

        # Fallback: model's built-in importance
        try:
            imp = model.feature_importances_
            sorted_idx = np.argsort(imp)[::-1]
            return {
                "method": "built_in",
                "features": [
                    {"feature": feature_names[j] if j < len(feature_names) else f"f{j}",
                     "importance": round(float(imp[j]), 4)}
                    for j in sorted_idx[:15]
                ],
            }
        except Exception:
            return {"error": "Feature importance not available"}

    # ------------------------------------------------------------------
    def _shap_explanation(self, x, feature_names, pred_prob, top_n) -> dict:
        import shap
        try:
            sv   = self._explainer.shap_values(x)
            if isinstance(sv, list):
                sv = sv[1]   # positive class
            sv   = sv.flatten()
            base = float(self._explainer.expected_value
                         if not isinstance(self._explainer.expected_value, list)
                         else self._explainer.expected_value[1])

            sorted_idx = np.argsort(np.abs(sv))[::-1][:top_n]
            contribs   = [
                {
                    "feature":      feature_names[i] if i < len(feature_names) else f"f{i}",
                    "shap_value":   round(float(sv[i]), 4),
                    "direction":    "BUY" if sv[i] > 0 else "SELL",
                }
                for i in sorted_idx
            ]
            return {
                "method":       "SHAP",
                "pred_prob":    round(pred_prob, 4),
                "base_value":   round(base, 4),
                "signal":       "BUY" if pred_prob >= 0.52 else "SELL",
                "contributions": contribs,
                "explanation":  self._natural_language(contribs[:3]),
            }
        except Exception as exc:
            return {"error": f"SHAP failed: {exc}", "pred_prob": pred_prob}

    def _permutation_explanation(self, x, feature_names, pred_prob, top_n) -> dict:
        """Fallback: permutation-based importance for single prediction."""
        if self._model is None:
            return {"pred_prob": pred_prob, "method": "none"}

        importances = []
        base_pred   = pred_prob
        for j in range(x.shape[1]):
            x_perm    = x.copy()
            x_perm[0, j] *= np.random.normal(1, 0.2)   # small perturbation
            try:
                new_pred  = float(self._model.predict_proba(x_perm)[0][1])
                importances.append((j, base_pred - new_pred))
            except Exception:
                importances.append((j, 0.0))

        importances.sort(key=lambda t: abs(t[1]), reverse=True)
        contribs = [
            {
                "feature":    feature_names[j] if j < len(feature_names) else f"f{j}",
                "shap_value": round(v, 4),
                "direction":  "BUY" if v > 0 else "SELL",
            }
            for j, v in importances[:top_n]
        ]
        return {
            "method":        "permutation",
            "pred_prob":     round(pred_prob, 4),
            "signal":        "BUY" if pred_prob >= 0.52 else "SELL",
            "contributions": contribs,
            "explanation":   self._natural_language(contribs[:3]),
        }

    @staticmethod
    def _natural_language(top_contribs: list[dict]) -> str:
        """Generate a one-line natural language explanation."""
        if not top_contribs:
            return "Insufficient data for explanation."
        parts = []
        for c in top_contribs:
            f  = c["feature"].replace("_", " ").upper()
            d  = "supporting" if c["shap_value"] > 0 else "opposing"
            parts.append(f"{f} is {d} the signal")
        return "; ".join(parts) + "."
