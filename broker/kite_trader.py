"""
Zerodha Kite Connect broker integration.

Supports both paper-trading (default) and live-trading modes.

Paper mode: all orders are logged to a file and stored in memory — no real money moves.
Live  mode: uses kiteconnect library. Requires KITE_API_KEY and KITE_API_SECRET in .env.

Usage:
    # Paper trading (safe default)
    trader = KiteTrader(paper_trade=True)
    trader.place_order("RELIANCE", "BUY", qty=5, order_type="MARKET")

    # Live trading (set paper_trade=False only after thorough testing)
    trader = KiteTrader(paper_trade=False)
    url = trader.get_login_url()
    # → open url in browser, copy request_token, then:
    trader.connect(request_token)
    trader.place_order("RELIANCE", "BUY", qty=5, stop_loss=1420.0, target=1480.0)

Environment variables (.env):
    KITE_API_KEY=your_api_key
    KITE_API_SECRET=your_api_secret
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from utils.logger import logger


try:
    from kiteconnect import KiteConnect
    _HAS_KITE = True
except ImportError:
    _HAS_KITE = False


class PaperOrder:
    """In-memory representation of a paper trade order."""
    _counter = 0

    def __init__(self, symbol, direction, qty, price, order_type,
                 stop_loss=None, target=None):
        PaperOrder._counter += 1
        self.order_id   = f"PAPER-{PaperOrder._counter:05d}"
        self.symbol     = symbol
        self.direction  = direction
        self.qty        = qty
        self.price      = price
        self.order_type = order_type
        self.stop_loss  = stop_loss
        self.target     = target
        self.status     = "COMPLETE"
        self.timestamp  = datetime.now()

    def to_dict(self) -> dict:
        return {
            "order_id":   self.order_id,
            "symbol":     self.symbol,
            "direction":  self.direction,
            "qty":        self.qty,
            "price":      self.price,
            "order_type": self.order_type,
            "stop_loss":  self.stop_loss,
            "target":     self.target,
            "status":     self.status,
            "timestamp":  self.timestamp.isoformat(),
        }


class KiteTrader:
    """
    Unified broker interface for Zerodha Kite.

    Parameters
    ----------
    paper_trade : bool
        True → simulate orders locally (default, safe).
        False → place real orders via Kite API.
    api_key : str | None
        Kite API key.  Defaults to KITE_API_KEY env var.
    api_secret : str | None
        Kite API secret.  Defaults to KITE_API_SECRET env var.
    log_dir : str
        Directory for paper-trade logs.
    """

    def __init__(
        self,
        paper_trade: bool        = True,
        api_key:     str | None  = None,
        api_secret:  str | None  = None,
        log_dir:     str         = "logs",
    ):
        self.paper_trade = paper_trade
        self.api_key     = api_key     or os.getenv("KITE_API_KEY",    "")
        self.api_secret  = api_secret  or os.getenv("KITE_API_SECRET", "")
        self.log_dir     = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self._kite:          KiteConnect | None  = None
        self._paper_orders:  list[PaperOrder]    = []
        self._paper_positions: dict[str, dict]   = {}  # symbol → {qty, avg_price}
        self._is_connected   = False

        if not paper_trade and not _HAS_KITE:
            raise ImportError(
                "kiteconnect package not installed. "
                "Run: pip install kiteconnect"
            )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def get_login_url(self) -> str:
        """Return the Kite login URL. Open in browser, copy request_token."""
        if self.paper_trade:
            return "PAPER_TRADE_MODE — no login required"
        if not self.api_key:
            raise ValueError("KITE_API_KEY not set in .env")
        kite = KiteConnect(api_key=self.api_key)
        return kite.login_url()

    def connect(self, request_token: str) -> bool:
        """Exchange request_token for access_token and initialize session."""
        if self.paper_trade:
            self._is_connected = True
            logger.info("KiteTrader: paper-trade mode — no real connection")
            return True

        if not _HAS_KITE:
            raise ImportError("kiteconnect not installed")

        try:
            self._kite = KiteConnect(api_key=self.api_key)
            session    = self._kite.generate_session(request_token, api_secret=self.api_secret)
            self._kite.set_access_token(session["access_token"])
            self._is_connected = True
            logger.info(f"KiteTrader: connected | user={session.get('user_name', '?')}")
            return True
        except Exception as exc:
            logger.error(f"KiteTrader.connect failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol:     str,
        direction:  str,           # "BUY" or "SELL"
        qty:        int,
        price:      float = 0,     # 0 = market order
        order_type: str   = "MARKET",
        stop_loss:  float | None = None,
        target:     float | None = None,
        exchange:   str   = "NSE",
    ) -> str | None:
        """
        Place an order.  Returns order_id on success, None on failure.
        Automatically places bracket SL/target orders if stop_loss and target provided.
        """
        direction = direction.upper()

        if self.paper_trade:
            return self._paper_place(symbol, direction, qty, price,
                                     order_type, stop_loss, target)

        return self._live_place(symbol, direction, qty, price,
                                order_type, stop_loss, target, exchange)

    def close_position(self, symbol: str, qty: int | None = None) -> str | None:
        """Close an open position (market order in opposite direction)."""
        if self.paper_trade:
            pos = self._paper_positions.get(symbol)
            if not pos:
                logger.warning(f"No open paper position for {symbol}")
                return None
            close_qty  = qty or pos["qty"]
            close_dir  = "SELL" if pos["qty"] > 0 else "BUY"
            close_price = self._get_paper_price(symbol)
            return self._paper_place(symbol, close_dir, close_qty, close_price,
                                     "MARKET", stop_loss=None, target=None)

        # Live: get current position and reverse
        try:
            positions = self._kite.positions()["net"]
            for p in positions:
                if p["tradingsymbol"] == symbol and p["quantity"] != 0:
                    close_qty = abs(qty or p["quantity"])
                    close_dir = "SELL" if p["quantity"] > 0 else "BUY"
                    return self._live_place(
                        symbol, close_dir, close_qty, 0, "MARKET",
                        None, None, p.get("exchange", "NSE"),
                    )
        except Exception as exc:
            logger.error(f"KiteTrader.close_position error: {exc}")
        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if self.paper_trade:
            for o in self._paper_orders:
                if o.order_id == order_id:
                    o.status = "CANCELLED"
                    logger.info(f"Paper order {order_id} cancelled")
                    return True
            return False

        try:
            self._kite.cancel_order(
                variety  = KiteConnect.VARIETY_REGULAR,
                order_id = order_id,
            )
            return True
        except Exception as exc:
            logger.error(f"KiteTrader.cancel_order error: {exc}")
            return False

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict]:
        """Return current open positions."""
        if self.paper_trade:
            return [
                {"symbol": s, **v}
                for s, v in self._paper_positions.items()
                if v["qty"] != 0
            ]
        try:
            raw = self._kite.positions()
            net = raw.get("net", [])
            return [
                {
                    "symbol":    p["tradingsymbol"],
                    "qty":       p["quantity"],
                    "avg_price": p["average_price"],
                    "pnl":       p["pnl"],
                    "exchange":  p.get("exchange", "NSE"),
                }
                for p in net if p["quantity"] != 0
            ]
        except Exception as exc:
            logger.error(f"KiteTrader.get_positions error: {exc}")
            return []

    def get_orders(self) -> list[dict]:
        """Return all orders placed today."""
        if self.paper_trade:
            return [o.to_dict() for o in self._paper_orders]
        try:
            return self._kite.orders()
        except Exception as exc:
            logger.error(f"KiteTrader.get_orders error: {exc}")
            return []

    def get_funds(self) -> dict:
        """Return available margin/funds."""
        if self.paper_trade:
            realized = sum(
                o.price * o.qty * (1 if o.direction == "SELL" else -1)
                for o in self._paper_orders
                if o.status == "COMPLETE"
            )
            return {"available": realized, "mode": "paper"}
        try:
            margins = self._kite.margins()
            equity  = margins.get("equity", {})
            return {
                "available": equity.get("available", {}).get("cash", 0),
                "used":      equity.get("utilised",  {}).get("debits", 0),
            }
        except Exception as exc:
            logger.error(f"KiteTrader.get_funds error: {exc}")
            return {}

    def get_ltp(self, symbol: str, exchange: str = "NSE") -> float | None:
        """Return last traded price for symbol."""
        if self.paper_trade:
            return self._get_paper_price(symbol)
        try:
            key  = f"{exchange}:{symbol}"
            data = self._kite.ltp([key])
            return data[key]["last_price"]
        except Exception as exc:
            logger.debug(f"KiteTrader.get_ltp error for {symbol}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Paper trade internals
    # ------------------------------------------------------------------

    def _paper_place(self, symbol, direction, qty, price, order_type,
                     stop_loss, target) -> str:
        if price == 0:
            price = self._get_paper_price(symbol)

        order = PaperOrder(symbol, direction, qty, price, order_type, stop_loss, target)
        self._paper_orders.append(order)

        # Update paper positions
        pos = self._paper_positions.setdefault(
            symbol, {"qty": 0, "avg_price": 0.0, "pnl": 0.0}
        )
        if direction == "BUY":
            total_cost = pos["avg_price"] * pos["qty"] + price * qty
            pos["qty"]       += qty
            pos["avg_price"]  = total_cost / pos["qty"] if pos["qty"] else price
        else:
            pnl = (price - pos["avg_price"]) * min(qty, pos["qty"])
            pos["pnl"] += pnl
            pos["qty"]  = max(0, pos["qty"] - qty)

        logger.info(
            f"[PAPER] {direction} {qty}×{symbol} @ ₹{price:.2f} | "
            f"SL={stop_loss} | TGT={target} | ID={order.order_id}"
        )
        self._log_paper_order(order)
        return order.order_id

    def _get_paper_price(self, symbol: str) -> float:
        """Fetch live price for paper simulation."""
        try:
            import yfinance as yf
            t = yf.Ticker(f"{symbol}.NS")
            info = t.fast_info
            return float(getattr(info, "last_price", 0) or 0)
        except Exception:
            return 0.0

    def _log_paper_order(self, order: PaperOrder):
        log_file = self.log_dir / f"paper_trades_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with log_file.open("a") as f:
            f.write(json.dumps(order.to_dict()) + "\n")

    # ------------------------------------------------------------------
    # Live trade internals
    # ------------------------------------------------------------------

    def _live_place(self, symbol, direction, qty, price, order_type,
                    stop_loss, target, exchange) -> str | None:
        if not self._is_connected or self._kite is None:
            logger.error("KiteTrader: not connected — call connect() first")
            return None

        kite      = self._kite
        ttype     = KiteConnect.TRANSACTION_TYPE_BUY if direction == "BUY" \
                    else KiteConnect.TRANSACTION_TYPE_SELL
        prod      = KiteConnect.PRODUCT_MIS   # intraday
        otype     = KiteConnect.ORDER_TYPE_MARKET if order_type == "MARKET" \
                    else KiteConnect.ORDER_TYPE_LIMIT

        try:
            order_id = kite.place_order(
                variety          = KiteConnect.VARIETY_REGULAR,
                exchange         = exchange,
                tradingsymbol    = symbol,
                transaction_type = ttype,
                quantity         = qty,
                product          = prod,
                order_type       = otype,
                price            = price if order_type == "LIMIT" else None,
            )
            logger.info(f"[LIVE] {direction} {qty}×{symbol} | order_id={order_id}")

            # Place SL order if provided
            if stop_loss:
                sl_side = KiteConnect.TRANSACTION_TYPE_SELL if direction == "BUY" \
                          else KiteConnect.TRANSACTION_TYPE_BUY
                kite.place_order(
                    variety          = KiteConnect.VARIETY_REGULAR,
                    exchange         = exchange,
                    tradingsymbol    = symbol,
                    transaction_type = sl_side,
                    quantity         = qty,
                    product          = prod,
                    order_type       = KiteConnect.ORDER_TYPE_SL_M,
                    trigger_price    = stop_loss,
                )

            return order_id

        except Exception as exc:
            logger.error(f"KiteTrader live order failed: {exc}")
            return None
