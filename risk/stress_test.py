"""Stress testing against historical Indian market crashes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger

SCENARIOS = {
    "COVID_2020":          ("2020-02-17", "2020-03-24",  "COVID crash — Nifty -38%"),
    "GFC_2008":            ("2008-01-01", "2008-10-27",  "Global Financial Crisis — Nifty -60%"),
    "DEMONETIZATION_2016": ("2016-11-08", "2016-12-26",  "Demonetisation shock"),
    "RUSSIA_UKRAINE_2022": ("2022-02-24", "2022-03-08",  "Russia-Ukraine war"),
    "ADANI_2023":          ("2023-01-24", "2023-02-10",  "Hindenburg/Adani selloff"),
}


class StressTest:
    def run_scenario(self, df: pd.DataFrame, scenario_name: str,
                     capital: float, quantity: int) -> dict:
        if scenario_name not in SCENARIOS:
            return {"error": f"Unknown scenario: {scenario_name}"}
        start, end, desc = SCENARIOS[scenario_name]
        try:
            idx = df.index
            if hasattr(idx, "tz") and idx.tz is not None:
                idx_naive = idx.tz_localize(None)
            else:
                idx_naive = idx
            mask = (idx_naive >= pd.Timestamp(start)) & (idx_naive <= pd.Timestamp(end))
            period = df[mask]
            if len(period) < 2:
                return {"scenario": scenario_name, "available": False,
                        "note": "Historical data predates this crash"}
            entry_price = float(period["close"].iloc[0])
            exit_price  = float(period["close"].min())
            max_loss    = (exit_price - entry_price) * quantity
            max_loss_pct = (exit_price - entry_price) / entry_price * 100
            # Days to recover (if data available)
            post = df[idx_naive > pd.Timestamp(end)]
            recover_days = None
            if not post.empty:
                recovered = post[post["close"] >= entry_price]
                if not recovered.empty:
                    recover_days = (recovered.index[0] - period.index[-1]).days
            return {
                "scenario":       scenario_name,
                "description":    desc,
                "available":      True,
                "period_start":   start,
                "period_end":     end,
                "entry_price":    round(entry_price, 2),
                "min_price":      round(exit_price, 2),
                "max_loss":       round(max_loss, 2),
                "max_loss_pct":   round(max_loss_pct, 2),
                "recover_days":   recover_days,
            }
        except Exception as exc:
            logger.debug(f"StressTest scenario {scenario_name}: {exc}")
            return {"scenario": scenario_name, "error": str(exc)}

    def run_all_scenarios(self, df: pd.DataFrame, capital: float,
                          quantity: int) -> list[dict]:
        return [self.run_scenario(df, s, capital, quantity) for s in SCENARIOS]

    def is_position_safe(self, df: pd.DataFrame, capital: float,
                         quantity: int, max_acceptable_loss_pct: float = 0.05) -> bool:
        results = self.run_all_scenarios(df, capital, quantity)
        for r in results:
            if not r.get("available"):
                continue
            loss_pct = abs(r.get("max_loss", 0)) / capital
            if loss_pct > max_acceptable_loss_pct:
                return False
        return True

    def generate_report(self, df: pd.DataFrame, capital: float,
                        quantity: int) -> dict:
        results = self.run_all_scenarios(df, capital, quantity)
        available = [r for r in results if r.get("available")]
        worst = min(available, key=lambda x: x.get("max_loss", 0), default={})
        return {
            "scenarios":      results,
            "worst_scenario": worst.get("scenario"),
            "worst_loss":     worst.get("max_loss", 0),
            "worst_loss_pct": worst.get("max_loss_pct", 0),
            "is_safe":        self.is_position_safe(df, capital, quantity),
        }
