"""Online incremental learning — updates signal model after each completed trade."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from utils.logger import logger
from analysis.ml_signal import _extract_features

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

from sklearn.linear_model import SGDClassifier


class OnlineLearner:
    """
    SGDClassifier with partial_fit for incremental updates.
    Combines with ensemble model: 30% online weight + 70% ensemble weight.
    """

    def __init__(self):
        self._model = SGDClassifier(
            loss="log_loss", learning_rate="adaptive",
            eta0=0.01, random_state=42, max_iter=1000,
        )
        self._n_seen   = 0
        self._classes  = np.array([0, 1])
        self._is_init  = False

    def fit_batch(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)
        self._n_seen  = len(y)
        self._is_init = True

    def partial_fit(self, features: np.ndarray, label: int):
        self._model.partial_fit(
            features.reshape(1, -1), [label], classes=self._classes
        )
        self._n_seen  += 1
        self._is_init  = True

    def predict_proba(self, features: np.ndarray) -> float:
        if not self._is_init:
            return 0.5
        try:
            return float(self._model.predict_proba(features.reshape(1, -1))[0][1])
        except Exception:
            return 0.5

    def update_from_trade(self, row, df, i: int, was_profitable: bool):
        features = _extract_features(row, df, i)
        if features is None:
            return
        self.partial_fit(features, int(was_profitable))

    def combined_proba(self, ensemble_prob: float, features: np.ndarray) -> float:
        """Blend: 70% ensemble + 30% online."""
        if not self._is_init:
            return ensemble_prob
        online_prob = self.predict_proba(features)
        return 0.70 * ensemble_prob + 0.30 * online_prob

    @property
    def n_samples_seen(self) -> int:
        return self._n_seen

    def should_retrain_full(self) -> bool:
        return self._n_seen >= 500

    def save(self, path: str = "models/online_learner.pkl"):
        Path(path).parent.mkdir(exist_ok=True)
        if _HAS_JOBLIB:
            joblib.dump(self._model, path)
        else:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self._model, f)
        logger.info(f"OnlineLearner saved to {path}")

    def load(self, path: str = "models/online_learner.pkl"):
        try:
            if _HAS_JOBLIB:
                self._model = joblib.load(path)
            else:
                import pickle
                with open(path, "rb") as f:
                    self._model = pickle.load(f)
            self._is_init = True
            logger.info(f"OnlineLearner loaded from {path}")
        except Exception as exc:
            logger.debug(f"OnlineLearner load failed: {exc}")
