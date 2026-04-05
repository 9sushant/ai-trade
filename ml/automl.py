"""AutoML hyperparameter tuning via Optuna for ensemble models."""
from __future__ import annotations
import numpy as np
from utils.logger import logger


class AutoMLTuner:
    """
    Uses Optuna to tune LightGBM / XGBoost hyperparameters.
    Falls back to sensible defaults if Optuna not installed.
    """

    def __init__(self, n_trials: int = 30, timeout_secs: int = 120):
        self.n_trials    = n_trials
        self.timeout     = timeout_secs
        self.best_params: dict = {}

    # ------------------------------------------------------------------
    def tune_lightgbm(self, X_train, y_train, X_val, y_val) -> dict:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                import lightgbm as lgb
                params = {
                    "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
                    "num_leaves":        trial.suggest_int("num_leaves", 16, 128),
                    "max_depth":         trial.suggest_int("max_depth", 3, 10),
                    "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                    "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
                    "random_state":      42,
                    "verbose":           -1,
                }
                m = lgb.LGBMClassifier(**params)
                m.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(20, verbose=False)])
                from sklearn.metrics import roc_auc_score
                return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials,
                           timeout=self.timeout, show_progress_bar=False)
            self.best_params["lightgbm"] = study.best_params
            logger.info(f"AutoML LGB best AUC={study.best_value:.4f} | {study.best_params}")
            return study.best_params

        except ImportError:
            logger.info("AutoML: Optuna not installed — using defaults")
            return self._default_lgb_params()
        except Exception as exc:
            logger.debug(f"AutoML tune error: {exc}")
            return self._default_lgb_params()

    def tune_xgboost(self, X_train, y_train, X_val, y_val) -> dict:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                import xgboost as xgb
                params = {
                    "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
                    "max_depth":        trial.suggest_int("max_depth", 3, 10),
                    "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "gamma":            trial.suggest_float("gamma", 0, 5),
                    "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                    "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
                    "eval_metric":      "logloss",
                    "random_state":     42,
                }
                m = xgb.XGBClassifier(**params, verbosity=0)
                m.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
                from sklearn.metrics import roc_auc_score
                return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials,
                           timeout=self.timeout, show_progress_bar=False)
            self.best_params["xgboost"] = study.best_params
            return study.best_params

        except ImportError:
            return self._default_xgb_params()
        except Exception as exc:
            logger.debug(f"AutoML XGB error: {exc}")
            return self._default_xgb_params()

    # ------------------------------------------------------------------
    @staticmethod
    def _default_lgb_params() -> dict:
        return {
            "n_estimators": 200, "num_leaves": 31, "max_depth": 6,
            "learning_rate": 0.05, "min_child_samples": 20,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "verbose": -1,
        }

    @staticmethod
    def _default_xgb_params() -> dict:
        return {
            "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "gamma": 0.1,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
        }
