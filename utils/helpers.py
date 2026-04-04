from datetime import datetime, time


def is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    market_open = time(9, 15)
    market_close = time(15, 30)
    return market_open <= now.time() <= market_close


def is_pre_market() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    return time(9, 0) <= now.time() < time(9, 15)


def round_to_tick(price: float, tick_size: float = 0.05) -> float:
    return round(round(price / tick_size) * tick_size, 2)


def calculate_quantity(capital: float, price: float, max_risk: float, sl_pct: float) -> int:
    risk_per_share = price * (sl_pct / 100)
    qty_by_risk = int(max_risk / risk_per_share) if risk_per_share > 0 else 0
    qty_by_capital = int(capital / price) if price > 0 else 0
    return min(qty_by_risk, qty_by_capital)


def format_currency(amount: float) -> str:
    if amount >= 0:
        return f"₹{amount:,.2f}"
    return f"-₹{abs(amount):,.2f}"
