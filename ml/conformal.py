"""Conformal prediction — calibrated uncertainty intervals for ML signals."""
from __future__ import annotations
import numpy as np
from utils.logger import logger


class ConformalPredictor:
    """
    Inductive Conformal Prediction for classification.

    Provides statistically valid coverage guarantees:
      "With probability ≥ 1-alpha, the true label is in the prediction set."

    Usage:
      1. calibrate(cal_probs, cal_labels) — fit on held-out calibration set
      2. predict(test_prob) — returns (is_confident_buy, confidence_level)
    """

    def __init__(self, alpha: float = 0.10):
        """
        alpha: desired error rate (0.10 = 90% coverage guarantee).
        """
        self.alpha      = alpha
        self._threshold = None
        self._cal_scores: np.ndarray | None = None
        self.is_calibrated = False

    # ------------------------------------------------------------------
    def calibrate(self, cal_probs: list[float], cal_labels: list[int]):
        """
        Compute nonconformity scores on calibration set.

        cal_probs:  model output probabilities [0, 1]
        cal_labels: true labels (0 or 1)
        """
        probs  = np.array(cal_probs)
        labels = np.array(cal_labels)

        # Nonconformity score: 1 - p(true_class)
        scores = np.where(labels == 1, 1 - probs, probs)
        self._cal_scores = scores

        # Threshold at (1-alpha) quantile
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = np.clip(level, 0, 1)
        self._threshold = float(np.quantile(scores, level))
        self.is_calibrated = True

        coverage = float(np.mean(scores <= self._threshold))
        logger.info(f"ConformalPredictor: calibrated on {n} samples, "
                    f"threshold={self._threshold:.4f}, coverage={coverage:.3f}")

    def predict(self, prob: float) -> dict:
        """
        Returns whether the signal is confident given calibration.

        prob: model output probability for the positive (BUY) class.
        """
        if not self.is_calibrated:
            return {
                "is_confident": True,   # uncalibrated — don't block
                "confidence":   prob,
                "in_set_buy":   prob >= 0.52,
                "in_set_sell":  prob < 0.48,
            }

        # Nonconformity score for BUY: 1 - prob
        # Nonconformity score for SELL: prob
        nc_buy  = 1 - prob
        nc_sell = prob

        in_set_buy  = nc_buy  <= self._threshold
        in_set_sell = nc_sell <= self._threshold

        # p-value: fraction of calibration scores >= test score
        p_buy  = float(np.mean(self._cal_scores >= nc_buy))
        p_sell = float(np.mean(self._cal_scores >= nc_sell))

        # Confidence = 1 - min_p_value (higher = more certain)
        confidence = 1 - min(p_buy, p_sell)

        return {
            "is_confident": in_set_buy or in_set_sell,
            "confidence":   round(confidence, 4),
            "in_set_buy":   in_set_buy,
            "in_set_sell":  in_set_sell,
            "p_value_buy":  round(p_buy,  4),
            "p_value_sell": round(p_sell, 4),
            "threshold":    round(self._threshold, 4),
        }

    def should_trade(self, prob: float, direction: str = "BUY") -> bool:
        """Return True only if prediction is within conformal set."""
        result = self.predict(prob)
        if direction == "BUY":
            return result["in_set_buy"]
        return result["in_set_sell"]

    def prediction_set(self, prob: float) -> list[str]:
        """Return which labels are in the conformal prediction set."""
        result = self.predict(prob)
        labels = []
        if result["in_set_buy"]:  labels.append("BUY")
        if result["in_set_sell"]: labels.append("SELL")
        if not labels:            labels.append("ABSTAIN")
        return labels

    def update_calibration(self, new_prob: float, true_label: int):
        """Online update: add one calibration point."""
        if self._cal_scores is None:
            self._cal_scores = np.array([])

        score = 1 - new_prob if true_label == 1 else new_prob
        self._cal_scores = np.append(self._cal_scores, score)

        # Recompute threshold with updated calibration set
        n     = len(self._cal_scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = np.clip(level, 0, 1)
        self._threshold    = float(np.quantile(self._cal_scores, level))
        self.is_calibrated = True

    def coverage_report(self) -> dict:
        """Report empirical coverage on calibration set."""
        if self._cal_scores is None:
            return {}
        scores = self._cal_scores
        return {
            "n_calibration":      len(scores),
            "threshold":          round(self._threshold, 4),
            "empirical_coverage": round(float(np.mean(scores <= self._threshold)), 4),
            "target_coverage":    round(1 - self.alpha, 4),
            "mean_score":         round(float(scores.mean()), 4),
            "std_score":          round(float(scores.std()),  4),
        }
