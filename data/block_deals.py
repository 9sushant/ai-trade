"""Block deal / bulk deal tracker — NSE institutional conviction signals."""
from __future__ import annotations
import requests
import pandas as pd
from datetime import datetime, date
from utils.logger import logger

_NSE_BULK_URL  = "https://www.nseindia.com/api/snapshot-capital-market-largeDeals"
_NSE_BLOCK_URL = "https://www.nseindia.com/api/block-deal"
_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer":         "https://www.nseindia.com/",
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}
_CACHE_TTL = 1800   # 30 min


class BlockDealTracker:
    """
    Tracks NSE block deals and bulk deals.
    Large institutional buy = bullish signal for the stock.
    Large institutional sell = bearish signal.
    """

    def __init__(self):
        self._bulk_cache:  tuple[list, datetime] | None = None
        self._block_cache: tuple[list, datetime] | None = None

    # ------------------------------------------------------------------
    def get_deals_today(self) -> list[dict]:
        """Return all block + bulk deals for today."""
        return self._fetch_bulk() + self._fetch_block()

    def get_symbol_deals(self, symbol: str) -> list[dict]:
        """Return deals for a specific symbol today."""
        all_deals = self.get_deals_today()
        sym_upper = symbol.upper()
        return [d for d in all_deals
                if d.get("symbol", "").upper() == sym_upper]

    def get_signal(self, symbol: str) -> dict:
        """
        Aggregate buy/sell deal values for a symbol.
        Returns signal: BULLISH / BEARISH / NEUTRAL + deal_value_cr.
        """
        deals = self.get_symbol_deals(symbol)
        if not deals:
            return {"signal": "NEUTRAL", "deal_value_cr": 0,
                    "buyer_deals": 0, "seller_deals": 0}

        buy_val  = sum(d.get("deal_value", 0) for d in deals if d.get("side") == "BUY")
        sell_val = sum(d.get("deal_value", 0) for d in deals if d.get("side") == "SELL")
        net      = buy_val - sell_val

        if net > 50:      signal = "BULLISH"
        elif net < -50:   signal = "BEARISH"
        else:             signal = "NEUTRAL"

        return {
            "signal":        signal,
            "deal_value_cr": round((buy_val + sell_val) / 1e7, 2),  # in Crores
            "net_value_cr":  round(net / 1e7, 2),
            "buyer_deals":   sum(1 for d in deals if d.get("side") == "BUY"),
            "seller_deals":  sum(1 for d in deals if d.get("side") == "SELL"),
        }

    def get_top_bought(self, n: int = 5) -> list[dict]:
        """Return top N symbols by institutional buying today."""
        deals = self.get_deals_today()
        sym_buy: dict[str, float] = {}
        for d in deals:
            if d.get("side") == "BUY":
                sym = d.get("symbol", "")
                sym_buy[sym] = sym_buy.get(sym, 0) + d.get("deal_value", 0)
        sorted_syms = sorted(sym_buy.items(), key=lambda x: x[1], reverse=True)[:n]
        return [{"symbol": s, "buy_value_cr": round(v / 1e7, 2)} for s, v in sorted_syms]

    def composite_score_boost(self, symbol: str) -> float:
        """Return score boost [-5, +5] based on deal activity."""
        signal = self.get_signal(symbol)
        net    = signal.get("net_value_cr", 0)
        # Scale: ±100 Cr → ±5 boost
        return round(float(max(-5, min(5, net / 20))), 2)

    # ------------------------------------------------------------------
    def _fetch_bulk(self) -> list:
        if self._bulk_cache:
            data, ts = self._bulk_cache
            if (datetime.now() - ts).seconds < _CACHE_TTL:
                return data
        try:
            sess = requests.Session()
            sess.get("https://www.nseindia.com/", headers=_HEADERS, timeout=8)
            resp = sess.get(_NSE_BULK_URL, headers=_HEADERS, timeout=8)
            if resp.ok:
                raw  = resp.json()
                data = self._parse_deals(raw.get("data", []), deal_type="BULK")
                self._bulk_cache = (data, datetime.now())
                return data
        except Exception as exc:
            logger.debug(f"BlockDeal bulk fetch: {exc}")
        return []

    def _fetch_block(self) -> list:
        if self._block_cache:
            data, ts = self._block_cache
            if (datetime.now() - ts).seconds < _CACHE_TTL:
                return data
        try:
            sess = requests.Session()
            sess.get("https://www.nseindia.com/", headers=_HEADERS, timeout=8)
            resp = sess.get(_NSE_BLOCK_URL, headers=_HEADERS, timeout=8)
            if resp.ok:
                raw  = resp.json()
                data = self._parse_deals(raw.get("data", []), deal_type="BLOCK")
                self._block_cache = (data, datetime.now())
                return data
        except Exception as exc:
            logger.debug(f"BlockDeal block fetch: {exc}")
        return []

    @staticmethod
    def _parse_deals(raw: list, deal_type: str) -> list:
        out = []
        for row in raw:
            try:
                qty   = float(str(row.get("quantity", "0")).replace(",", ""))
                price = float(str(row.get("tradePrice", "0")).replace(",", ""))
                side  = "BUY" if str(row.get("buyerName", "")).strip() else "SELL"
                out.append({
                    "symbol":     row.get("symbol", ""),
                    "side":       side,
                    "quantity":   qty,
                    "price":      price,
                    "deal_value": qty * price,
                    "deal_type":  deal_type,
                    "client":     row.get("buyerName", "") or row.get("sellerName", ""),
                })
            except Exception:
                continue
        return out
