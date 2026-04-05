"""Model drift detection — alerts when live accuracy drops vs backtest baseline."""
from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from utils.logger import logger


class ModelDriftDetector:
    """
    Detects when the live model performance has drifted from backtest baseline.

    Methods:
      - PSI (Population Stability Index): detects feature distribution shift
      - Rolling accuracy: tracks prediction accuracy over recent N trades
      - KL divergence: measures prediction confidence distribution shift
    """

    PSI_THRESHOLDS = {"stable": 0.1, "monitor": 0.2, "alert": 0.25}

    def __init__(
        self,
        baseline_win_rate: float = 0.65,
        baseline_accuracy: float = 0.60,
        alert_threshold: float = 0.10,   # drop > 10% triggers alert
        window: int = 50,                # rolling window for live stats
    ):
        self.baseline_wr    = baseline_win_rate
        self.baseline_acc   = baseline_accuracy
        self.alert_threshold = alert_threshold
        self.window         = window

        self._predictions: list[dict] = []   # {pred_prob, actual, timestamp}
        self._baseline_probs: np.ndarray | None = None

    # ------------------------------------------------------------------
    def set_baseline(self, pred_probabilities: list[float], actuals: list[int]):
        """Store baseline prediction distribution from backtest."""
        self._baseline_probs = np.array(pred_probabilities)
        logger.info(f"DriftDetector: baseline set ({len(pred_probabilities)} samples, "
                    f"mean_prob={np.mean(pred_probabilities):.3f})")

    def record_prediction(self, pred_prob: float, actual: int | None = None):
        """Record a live prediction (actual can be added later when trade closes)."""
        self._predictions.append({
            "pred_prob": pred_prob,
            "actual":    actual,
            "timestamp": datetime.now().isoformat(),
        })
        # Keep rolling window
        if len(self._predictions) > self.window * 3:
            self._predictions = self._predictions[-self.window * 2:]

    def update_actual(self, idx: int, actual: int):
        """Update actual outcome for a prediction."""
        if 0 <= idx < len(self._predictions):
            self._predictions[idx]["actual"] = actual

    # ------------------------------------------------------------------
    def check_drift(self) -> dict:
        """Run all drift checks and return status report."""
        report = {
            "timestamp":        datetime.now().isoformat(),
            "n_live_samples":   len(self._predictions),
            "drift_detected":   False,
            "alerts":           [],
        }

        if len(self._predictions) < 20:
            report["status"] = "INSUFFICIENT_DATA"
            return report

        # 1. Rolling accuracy check
        acc_check = self._check_accuracy()
        report.update(acc_check)
        if acc_check.get("accuracy_drift"):
            report["drift_detected"] = True
            report["alerts"].append(acc_check["accuracy_alert"])

        # 2. PSI on prediction probabilities
        if self._baseline_probs is not None:
            psi_check = self._compute_psi()
            report.update(psi_check)
            if psi_check.get("psi_alert"):
                report["drift_detected"] = True
                report["alerts"].append(psi_check["psi_alert"])

        # 3. Win rate drift
        wr_check = self._check_win_rate()
        report.update(wr_check)
        if wr_check.get("wr_drift"):
            report["drift_detected"] = True
            report["alerts"].append(wr_check["wr_alert"])

        # 4. Confidence calibration
        cal_check = self._check_calibration()
        report.update(cal_check)

        report["status"] = "DRIFT_DETECTED" if report["drift_detected"] else "STABLE"

        if report["drift_detected"]:
            logger.warning(f"DriftDetector: DRIFT DETECTED | {report['alerts']}")
            self._notify(report)

        return report

    def should_retrain(self) -> bool:
        """Return True if model should be retrained."""
        report = self.check_drift()
        return report["drift_detected"] or len(self._predictions) >= self.window * 2

    # ------------------------------------------------------------------
    def _check_accuracy(self) -> dict:
        completed = [p for p in self._predictions[-self.window:]
                     if p.get("actual") is not None]
        if len(completed) < 10:
            return {"accuracy_drift": False}

        preds   = [1 if p["pred_prob"] >= 0.52 else 0 for p in completed]
        actuals = [p["actual"] for p in completed]
        acc     = sum(p == a for p, a in zip(preds, actuals)) / len(completed)
        drop    = self.baseline_acc - acc

        return {
            "live_accuracy":   round(acc, 3),
            "baseline_accuracy": self.baseline_acc,
            "accuracy_drop":   round(drop, 3),
            "accuracy_drift":  drop > self.alert_threshold,
            "accuracy_alert":  f"Accuracy dropped {drop:.1%} (live={acc:.1%} vs baseline={self.baseline_acc:.1%})"
                               if drop > self.alert_threshold else None,
        }

    def _check_win_rate(self) -> dict:
        completed = [p for p in self._predictions[-self.window:]
                     if p.get("actual") is not None]
        if len(completed) < 10:
            return {"wr_drift": False}

        wr   = sum(p["actual"] == 1 for p in completed) / len(completed)
        drop = self.baseline_wr - wr
        return {
            "live_win_rate":   round(wr, 3),
            "baseline_win_rate": self.baseline_wr,
            "wr_drop":         round(drop, 3),
            "wr_drift":        drop > self.alert_threshold,
            "wr_alert":        f"Win rate dropped {drop:.1%} (live={wr:.1%} vs baseline={self.baseline_wr:.1%})"
                               if drop > self.alert_threshold else None,
        }

    def _compute_psi(self) -> dict:
        """Population Stability Index between baseline and live distributions."""
        live_probs = np.array([p["pred_prob"] for p in self._predictions[-self.window:]])
        bins       = np.linspace(0, 1, 11)

        base_counts = np.histogram(self._baseline_probs, bins=bins)[0]
        live_counts = np.histogram(live_probs,           bins=bins)[0]

        base_freq = (base_counts + 0.001) / (base_counts.sum() + 0.01)
        live_freq = (live_counts + 0.001) / (live_counts.sum() + 0.01)

        psi = float(np.sum((live_freq - base_freq) * np.log(live_freq / base_freq)))

        if psi < self.PSI_THRESHOLDS["stable"]:
            psi_status = "STABLE"
        elif psi < self.PSI_THRESHOLDS["monitor"]:
            psi_status = "MONITOR"
        elif psi < self.PSI_THRESHOLDS["alert"]:
            psi_status = "WARNING"
        else:
            psi_status = "ALERT"

        return {
            "psi":       round(psi, 4),
            "psi_status": psi_status,
            "psi_alert": f"PSI={psi:.4f} — distribution shift detected ({psi_status})"
                         if psi >= self.PSI_THRESHOLDS["monitor"] else None,
        }

    def _check_calibration(self) -> dict:
        """Check if predicted probabilities match actual win rates (calibration)."""
        completed = [p for p in self._predictions if p.get("actual") is not None]
        if len(completed) < 20:
            return {}

        # Split into 5 probability buckets
        buckets = [(0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
        cal_data = []
        for lo, hi in buckets:
            bucket = [p for p in completed if lo <= p["pred_prob"] < hi]
            if len(bucket) >= 3:
                mean_pred = np.mean([p["pred_prob"] for p in bucket])
                actual_wr = np.mean([p["actual"] for p in bucket])
                cal_data.append({"range": f"{lo:.1f}-{hi:.1f}",
                                  "mean_pred": round(mean_pred, 3),
                                  "actual_wr": round(actual_wr, 3),
                                  "count": len(bucket)})
        return {"calibration": cal_data}

    def _notify(self, report: dict):
        try:
            from notifications.telegram_bot import TelegramNotifier
            msg = f"⚠️ Model Drift Detected\n" \
                  f"Status: {report['status']}\n" \
                  f"Alerts: {'; '.join(a for a in report['alerts'] if a)}"
            TelegramNotifier().send_message(msg)
        except Exception:
            pass

    def save_state(self, path: str = "models/drift_state.json"):
        Path(path).parent.mkdir(exist_ok=True)
        state = {
            "predictions":    self._predictions[-200:],
            "baseline_probs": self._baseline_probs.tolist() if self._baseline_probs is not None else [],
        }
        with open(path, "w") as f:
            json.dump(state, f)

    def load_state(self, path: str = "models/drift_state.json"):
        if not Path(path).exists():
            return
        with open(path) as f:
            state = json.load(f)
        self._predictions    = state.get("predictions", [])
        bp = state.get("baseline_probs", [])
        if bp:
            self._baseline_probs = np.array(bp)
