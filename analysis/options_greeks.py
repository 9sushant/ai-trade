"""Options Greeks signals — Delta, Gamma, Vega for intraday pinning and flow."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date, datetime
from utils.logger import logger

_RISK_FREE = 0.065   # India 10Y ~6.5%


def _bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return intrinsic
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _greeks(S, K, T, r, sigma, option_type="call"):
    """Compute Delta, Gamma, Vega, Theta."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}
    try:
        from scipy.stats import norm
        d1  = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2  = d1 - sigma * np.sqrt(T)
        pdf = norm.pdf(d1)
        cdf = norm.cdf(d1)

        gamma = pdf / (S * sigma * np.sqrt(T))
        vega  = S * pdf * np.sqrt(T) / 100   # per 1% vol change

        if option_type == "call":
            delta = cdf
            theta = (-(S * pdf * sigma) / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            delta = cdf - 1
            theta = (-(S * pdf * sigma) / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}
    except Exception:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}


class OptionsGreeksAnalyzer:
    """
    Computes aggregated Greeks from live option chains.

    Key signals:
      - Net Delta Exposure (DE): overall market directional bias
      - Gamma Exposure (GEX): high GEX → price pinned near strike
      - Vega: IV sensitivity, risk of vol crush/spike
      - Charm: delta change over time → pin magnet near expiry
    """

    _CACHE_TTL = 1800   # 30 min

    def __init__(self):
        self._cache: dict[str, tuple[dict, datetime]] = {}

    # ------------------------------------------------------------------
    def get_greeks_signal(self, symbol: str) -> dict:
        cached = self._cache.get(symbol)
        if cached:
            data, ts = cached
            if (datetime.now() - ts).seconds < self._CACHE_TTL:
                return data

        result = self._compute_greeks(symbol)
        self._cache[symbol] = (result, datetime.now())
        return result

    def _compute_greeks(self, symbol: str) -> dict:
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            S      = getattr(ticker.fast_info, "last_price", 0)
            exps   = ticker.options

            if not exps or S == 0:
                return self._empty()

            # Use nearest expiry
            exp_str = exps[0]
            chain   = ticker.option_chain(exp_str)
            T       = self._time_to_expiry(exp_str)

            calls = chain.calls.copy() if hasattr(chain, "calls") else pd.DataFrame()
            puts  = chain.puts.copy()  if hasattr(chain, "puts")  else pd.DataFrame()

            if calls.empty and puts.empty:
                return self._empty()

            net_delta  = 0.0
            net_gamma  = 0.0
            net_vega   = 0.0
            total_oi   = 0
            pin_strike = S

            for _, row in calls.iterrows():
                K     = float(row.get("strike", S))
                iv    = float(row.get("impliedVolatility", 0.3) or 0.3)
                oi    = int(row.get("openInterest",  0) or 0)
                g     = _greeks(S, K, T, _RISK_FREE, iv, "call")
                net_delta += g["delta"] * oi
                net_gamma += g["gamma"] * oi
                net_vega  += g["vega"]  * oi
                total_oi  += oi

            for _, row in puts.iterrows():
                K     = float(row.get("strike", S))
                iv    = float(row.get("impliedVolatility", 0.3) or 0.3)
                oi    = int(row.get("openInterest", 0) or 0)
                g     = _greeks(S, K, T, _RISK_FREE, iv, "put")
                net_delta += g["delta"] * oi
                net_gamma += g["gamma"] * oi
                net_vega  += g["vega"]  * oi
                total_oi  += oi

            # Pin strike: highest total OI
            all_oi = pd.concat([
                calls[["strike", "openInterest"]].rename(columns={"openInterest": "oi"}),
                puts[["strike", "openInterest"]].rename(columns={"openInterest": "oi"}),
            ]).groupby("strike")["oi"].sum()
            if not all_oi.empty:
                pin_strike = float(all_oi.idxmax())

            # Normalize by total OI
            scale = max(total_oi, 1)
            gex_signal = "PINNED"  if abs(net_gamma / scale) > 0.001 else "FREE"
            dir_signal = "BULLISH" if net_delta > 0 else "BEARISH"

            result = {
                "symbol":          symbol,
                "spot":            round(S, 2),
                "pin_strike":      round(pin_strike, 2),
                "distance_to_pin": round(abs(S - pin_strike) / S * 100, 2),
                "net_delta":       round(net_delta / scale, 4),
                "net_gamma":       round(net_gamma / scale, 6),
                "net_vega":        round(net_vega  / scale, 4),
                "gex_signal":      gex_signal,
                "direction_bias":  dir_signal,
                "total_oi":        total_oi,
                "should_avoid":    gex_signal == "PINNED" and abs(S - pin_strike) / S < 0.005,
            }
            return result

        except Exception as exc:
            logger.debug(f"OptionsGreeks {symbol}: {exc}")
            return self._empty()

    def composite_score_boost(self, symbol: str) -> float:
        """Return score boost [-5, +5] based on options positioning."""
        sig    = self.get_greeks_signal(symbol)
        delta  = sig.get("net_delta", 0)
        return round(float(np.clip(delta * 10, -5, 5)), 2)

    @staticmethod
    def _time_to_expiry(exp_str: str) -> float:
        """Days to expiry as fraction of year."""
        try:
            exp   = datetime.strptime(exp_str, "%Y-%m-%d").date()
            delta = (exp - date.today()).days
            return max(delta / 365, 1 / 365)
        except Exception:
            return 7 / 365

    @staticmethod
    def _empty() -> dict:
        return {"symbol": "", "spot": 0, "pin_strike": 0, "net_delta": 0,
                "net_gamma": 0, "net_vega": 0, "gex_signal": "UNKNOWN",
                "direction_bias": "NEUTRAL", "should_avoid": False}
