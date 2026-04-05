"""Automated weekly ML model retraining pipeline."""
from __future__ import annotations
import time
from datetime import datetime
from pathlib import Path
from utils.logger import logger

try:
    import schedule
    _HAS_SCHEDULE = True
except ImportError:
    _HAS_SCHEDULE = False

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    import pickle
    _HAS_JOBLIB = False

MODELS_DIR = Path("models")


class RetrainPipeline:
    def __init__(self, symbols: list[str] = None,
                 retrain_day: str = "sunday", retrain_hour: int = 8):
        from config.settings import MarketConfig
        self.symbols      = symbols or MarketConfig.NIFTY_50_SYMBOLS[:20]
        self.retrain_day  = retrain_day
        self.retrain_hour = retrain_hour
        MODELS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    def run_retrain(self) -> dict:
        """Full retrain: fetch data → train ensemble + online learner → save → notify."""
        logger.info("RetrainPipeline: starting retraining...")
        start = datetime.now()

        # 1. Fetch fresh data
        dfs = self._fetch_data()
        if not dfs:
            logger.error("RetrainPipeline: no data fetched")
            return {"error": "No data"}

        # 2. Train ensemble model
        from analysis.ensemble import EnsembleSignalModel
        ensemble = EnsembleSignalModel()
        ensemble.fit(list(dfs.values()), verbose=True)

        # 3. Train online learner base
        from ml.online_learner import OnlineLearner
        from analysis.ml_signal import _build_dataset
        import numpy as np
        online = OnlineLearner()
        all_X, all_y = [], []
        for df in dfs.values():
            if df is not None and len(df) >= 80:
                X, y = _build_dataset(df)
                if len(X) > 0:
                    all_X.append(X); all_y.append(y)
        if all_X:
            online.fit_batch(np.vstack(all_X), np.concatenate(all_y))

        # 4. Save models with timestamp
        ts_str = datetime.now().strftime("%Y%m%d")
        self._save_model(ensemble, MODELS_DIR / f"ensemble_{ts_str}.pkl")
        self._save_model(online,   MODELS_DIR / f"online_{ts_str}.pkl")

        # 5. Log metrics
        stats = {
            "retrain_date":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            "duration_secs": (datetime.now() - start).seconds,
            "n_samples":     sum(len(_build_dataset(df)[0])
                                 for df in dfs.values() if df is not None and len(df) >= 80),
            "pos_rate":      ensemble._pos_rate,
            "models":        ensemble._names,
            "symbols":       len(dfs),
        }
        logger.info(f"RetrainPipeline: complete | {stats}")

        # 6. Telegram notification
        try:
            from notifications.telegram_bot import TelegramNotifier
            TelegramNotifier().send_retrain_complete(stats)
        except Exception:
            pass

        return stats

    def schedule_retrain(self):
        """Schedule weekly retraining using schedule library."""
        if not _HAS_SCHEDULE:
            logger.warning("schedule library not installed — run: pip install schedule")
            return
        getattr(schedule.every(), self.retrain_day).at(
            f"{self.retrain_hour:02d}:00"
        ).do(self.run_retrain)
        logger.info(f"Retraining scheduled: every {self.retrain_day} at {self.retrain_hour:02d}:00")

    def run_forever(self):
        """Blocking loop — runs scheduled retraining indefinitely."""
        if not _HAS_SCHEDULE:
            return
        self.schedule_retrain()
        logger.info("RetrainPipeline: running scheduler (Ctrl+C to stop)")
        while True:
            schedule.run_pending()
            time.sleep(60)

    def load_latest_models(self) -> dict:
        """Load most recently saved ensemble and online models."""
        result = {}
        for prefix in ("ensemble", "online"):
            files = sorted(MODELS_DIR.glob(f"{prefix}_*.pkl"), reverse=True)
            if files:
                result[prefix] = self._load_model(files[0])
                logger.info(f"Loaded {prefix} model from {files[0]}")
        return result

    def get_retrain_status(self) -> dict:
        from datetime import date
        files = sorted(MODELS_DIR.glob("ensemble_*.pkl"), reverse=True)
        last  = files[0].stem.replace("ensemble_", "") if files else None
        return {
            "last_retrain_date": last,
            "models_loaded":     len(files) > 0,
        }

    def run_retrain_now(self):
        """CLI entrypoint."""
        result = self.run_retrain()
        print(result)

    # ------------------------------------------------------------------
    def _fetch_data(self) -> dict:
        from data.fetcher import DataFetcher
        from analysis.technical import TechnicalAnalyzer
        from analysis.quant import QuantAnalyzer
        fetcher  = DataFetcher()
        analyzer = TechnicalAnalyzer()
        quant    = QuantAnalyzer()
        result   = {}
        for symbol in self.symbols:
            try:
                df = fetcher.get_historical_data(symbol, period="2y", interval="1d")
                if df is not None and len(df) >= 60:
                    df = analyzer.compute_all_indicators(df)
                    df = quant.compute_all(df)
                    result[symbol] = df
            except Exception as exc:
                logger.debug(f"Retrain fetch {symbol}: {exc}")
        return result

    def _save_model(self, model, path: Path):
        try:
            if _HAS_JOBLIB:
                joblib.dump(model, path)
            else:
                import pickle
                with open(path, "wb") as f:
                    pickle.dump(model, f)
        except Exception as exc:
            logger.debug(f"Save model failed: {exc}")

    def _load_model(self, path: Path):
        try:
            if _HAS_JOBLIB:
                return joblib.load(path)
            else:
                import pickle
                with open(path, "rb") as f:
                    return pickle.load(f)
        except Exception:
            return None


if __name__ == "__main__":
    RetrainPipeline().run_retrain_now()
