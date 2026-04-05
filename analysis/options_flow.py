"""
Options flow analysis for signal confirmation.

Uses Put/Call ratio, Open Interest buildup, and unusual options activity
to generate a directional bias score for each stock.

Score range: -10 (very bearish) to +10 (very bullish)

Signals:
  PCR < 0.7   → bullish (more calls than puts)
  PCR > 1.3   → bearish (more puts than calls)
  OI buildup at strikes → support/resistance levels
  IV spike    → event risk (reduce position size)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, date
from utils.logger import logger


class OptionsFlowAnalyzer:
    """
    Analyses options chain data to extract directional flow signals.

    Usage:
        analyzer = OptionsFlowAnalyzer()
        score    = analyzer.score("RELIANCE")
        signal   = analyzer.get_signal("RELIANCE")
    """

    _CACHE_TTL = 1_800   # 30 minutes (options data changes fast)

    def __init__(self):
        self._cache: dict[str, tuple[dict, datetime]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, symbol: str) -> float:
        """Return options flow score [-10, +10]. Positive = bullish."""
        data = self._get_options_data(symbol)
        if not data:
            return 0.0

        s = 0.0

        # ── Put/Call Ratio ────────────────────────────────────── ±4
        pcr = data.get("pcr", 1.0)
        if pcr is not None:
            if pcr < 0.5:
                s += 4.0    # extremely bullish
            elif pcr < 0.7:
                s += 2.5
            elif pcr < 0.9:
                s += 1.0
            elif pcr < 1.1:
                s += 0.0    # neutral
            elif pcr < 1.3:
                s -= 1.0
            elif pcr < 1.6:
                s -= 2.5
            else:
                s -= 4.0    # extremely bearish

        # ── OI Change direction ───────────────────────────────── ±3
        call_oi_change = data.get("call_oi_change", 0)
        put_oi_change  = data.get("put_oi_change",  0)
        net_oi = call_oi_change - put_oi_change
        if net_oi != 0:
            s += float(np.clip(net_oi / (abs(net_oi) + 1e-9) * 3, -3, 3))

        # ── IV rank (high IV = caution) ───────────────────────── ±2
        iv_rank = data.get("iv_rank", 50)
        if iv_rank is not None:
            if iv_rank > 80:
                s -= 1.5    # IV spike = uncertainty, reduce all signals
            elif iv_rank < 20:
                s += 1.0    # Low IV = calm market, signals more reliable

        # ── Max Pain vs Current Price ─────────────────────────── ±1
        max_pain     = data.get("max_pain")
        current_price = data.get("current_price")
        if max_pain and current_price:
            deviation = (current_price - max_pain) / current_price
            if deviation > 0.02:   # price well above max pain → gravitates down
                s -= 1.0
            elif deviation < -0.02:  # price below max pain → gravitates up
                s += 1.0

        return float(np.clip(s, -10, 10))

    def label(self, symbol: str) -> str:
        s = self.score(symbol)
        if s >= 5:   return "VERY BULLISH"
        if s >= 2:   return "BULLISH"
        if s >= -2:  return "NEUTRAL"
        if s >= -5:  return "BEARISH"
        return "VERY BEARISH"

    def get_signal(self, symbol: str) -> dict:
        """Return full options signal dict."""
        data  = self._get_options_data(symbol)
        score = self.score(symbol)
        return {
            "symbol":         symbol,
            "options_score":  round(score, 2),
            "options_label":  self.label(symbol),
            "pcr":            data.get("pcr")            if data else None,
            "iv_rank":        data.get("iv_rank")        if data else None,
            "max_pain":       data.get("max_pain")       if data else None,
            "call_oi":        data.get("total_call_oi")  if data else None,
            "put_oi":         data.get("total_put_oi")   if data else None,
            "iv_spike":       (data.get("iv_rank", 0) or 0) > 80 if data else False,
        }

    def should_avoid_trade(self, symbol: str) -> bool:
        """True if IV spike detected (don't size up into unknown event risk)."""
        data = self._get_options_data(symbol)
        if not data:
            return False
        iv_rank = data.get("iv_rank", 50) or 50
        return float(iv_rank) > 85

    def get_support_resistance(self, symbol: str) -> dict:
        """
        Derive S/R levels from max OI strikes.
        High call OI = resistance; high put OI = support.
        """
        data = self._get_options_data(symbol)
        if not data:
            return {"support": None, "resistance": None}
        return {
            "support":    data.get("max_put_oi_strike"),
            "resistance": data.get("max_call_oi_strike"),
        }

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _get_options_data(self, symbol: str) -> dict | None:
        """Fetch and parse options chain data."""
        cached = self._cache.get(symbol)
        if cached:
            data, ts = cached
            if (datetime.now() - ts).seconds < self._CACHE_TTL:
                return data

        result = self._fetch_yfinance_options(symbol)
        if result:
            self._cache[symbol] = (result, datetime.now())
        return result

    def _fetch_yfinance_options(self, symbol: str) -> dict | None:
        """Fetch options chain via yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")

            # Get nearest expiry
            expiries = ticker.options
            if not expiries:
                return None

            nearest_expiry = expiries[0]
            chain = ticker.option_chain(nearest_expiry)

            calls = chain.calls
            puts  = chain.puts

            if calls is None or puts is None or calls.empty or puts.empty:
                return None

            # Current price
            fast_info = ticker.fast_info
            current_price = float(getattr(fast_info, "last_price", 0) or 0)

            # Total OI
            total_call_oi = int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
            total_put_oi  = int(puts["openInterest"].sum())  if "openInterest" in puts.columns  else 0

            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

            # OI change (use volume as proxy if OI change not available)
            call_oi_change = int(calls["volume"].sum()) if "volume" in calls.columns else 0
            put_oi_change  = int(puts["volume"].sum())  if "volume" in puts.columns  else 0

            # Max pain = strike where total option loss is minimized
            max_pain = self._compute_max_pain(calls, puts, current_price)

            # Max OI strikes
            max_call_oi_strike = None
            max_put_oi_strike  = None
            if "openInterest" in calls.columns and not calls.empty:
                max_call_oi_strike = float(calls.loc[calls["openInterest"].idxmax(), "strike"])
            if "openInterest" in puts.columns and not puts.empty:
                max_put_oi_strike  = float(puts.loc[puts["openInterest"].idxmax(),  "strike"])

            # IV rank (use ATM IV as proxy)
            iv_rank = self._estimate_iv_rank(calls, puts, current_price)

            return {
                "pcr":                 round(pcr, 3),
                "total_call_oi":       total_call_oi,
                "total_put_oi":        total_put_oi,
                "call_oi_change":      call_oi_change,
                "put_oi_change":       put_oi_change,
                "max_pain":            max_pain,
                "max_call_oi_strike":  max_call_oi_strike,
                "max_put_oi_strike":   max_put_oi_strike,
                "iv_rank":             iv_rank,
                "current_price":       current_price,
                "expiry":              nearest_expiry,
            }

        except Exception as exc:
            logger.debug(f"OptionsFlowAnalyzer fetch failed for {symbol}: {exc}")
            return None

    def _compute_max_pain(self, calls: pd.DataFrame, puts: pd.DataFrame,
                          current_price: float) -> float | None:
        """
        Max pain = strike where total value of expiring options is minimized.
        """
        try:
            if "strike" not in calls.columns or "openInterest" not in calls.columns:
                return None

            strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
            if not strikes:
                return None

            min_pain = float("inf")
            max_pain_strike = strikes[0]

            for s in strikes:
                # Loss for call holders if expires at s
                call_loss = calls[calls["strike"] >= s].apply(
                    lambda r: (s - r["strike"]) * r.get("openInterest", 0), axis=1
                ).sum()
                # Loss for put holders if expires at s
                put_loss = puts[puts["strike"] <= s].apply(
                    lambda r: (r["strike"] - s) * r.get("openInterest", 0), axis=1
                ).sum()
                total = float(call_loss) + float(put_loss)
                if total < min_pain:
                    min_pain = total
                    max_pain_strike = s

            return float(max_pain_strike)
        except Exception:
            return None

    def _estimate_iv_rank(self, calls: pd.DataFrame, puts: pd.DataFrame,
                          current_price: float) -> float | None:
        """Estimate IV rank using ATM implied volatility (0-100 scale)."""
        try:
            if "impliedVolatility" not in calls.columns:
                return None

            # Find ATM options
            atm_calls = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:3]]
            atm_iv    = float(atm_calls["impliedVolatility"].mean())

            # Map to 0-100 scale: IV 0.1 = 0, IV 0.8+ = 100
            iv_rank = min(100, max(0, (atm_iv - 0.10) / 0.70 * 100))
            return round(iv_rank, 1)
        except Exception:
            return None
