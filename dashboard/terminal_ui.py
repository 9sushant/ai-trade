from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box
from datetime import datetime
from utils.helpers import format_currency


console = Console()


def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║          AI TRADE - Smart Stock Trading System           ║
    ║       NSE/BSE | Technical Analysis | Auto Trading        ║
    ╚══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")
    console.print(f"  Date: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}", style="dim")
    console.print()


def print_top_10_picks(picks: list[dict]):
    table = Table(
        title="TOP 10 STOCK PICKS FOR TODAY",
        box=box.DOUBLE_EDGE,
        show_lines=True,
        title_style="bold yellow",
    )

    table.add_column("#", style="dim", width=3)
    table.add_column("Symbol", style="cyan bold", width=12)
    table.add_column("Type", style="magenta", width=10)
    table.add_column("Price", justify="right", width=10)
    table.add_column("Entry", justify="right", width=10)
    table.add_column("Stop Loss", justify="right", style="red", width=10)
    table.add_column("Target", justify="right", style="green", width=10)
    table.add_column("R:R", justify="center", width=6)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Signal", style="bold", width=14)

    for i, pick in enumerate(picks, 1):
        score = pick.get("composite_score", pick.get("intraday_score", pick.get("swing_score", 0)))
        signal = pick.get("recommendation", "")
        signal_style = "green" if "BUY" in signal else "red" if "SELL" in signal else "yellow"

        table.add_row(
            str(i),
            pick.get("symbol", ""),
            pick.get("trade_type", pick.get("recommendation", "")[:8]),
            f"₹{pick.get('close', 0):,.2f}",
            f"₹{pick.get('entry', 0):,.2f}",
            f"₹{pick.get('stop_loss', 0):,.2f}",
            f"₹{pick.get('target', pick.get('target_1', 0)):,.2f}",
            f"{pick.get('risk_reward', 0):.1f}",
            f"[{'green' if score > 0 else 'red'}]{score:.0f}[/]",
            f"[{signal_style}]{signal}[/]",
        )

    console.print(table)
    console.print()


def print_intraday_picks(picks: list[dict]):
    table = Table(
        title="INTRADAY OPPORTUNITIES",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold green",
    )

    table.add_column("Symbol", style="cyan bold", width=12)
    table.add_column("Direction", width=8)
    table.add_column("Entry", justify="right", width=10)
    table.add_column("SL", justify="right", style="red", width=10)
    table.add_column("Target", justify="right", style="green", width=10)
    table.add_column("RSI", justify="center", width=6)
    table.add_column("Vol Ratio", justify="center", width=9)
    table.add_column("Score", justify="center", width=8)

    for pick in picks:
        direction = pick.get("direction", "BUY")
        dir_style = "green" if direction == "BUY" else "red"

        table.add_row(
            pick.get("symbol", ""),
            f"[{dir_style}]{direction}[/]",
            f"₹{pick.get('entry', 0):,.2f}",
            f"₹{pick.get('stop_loss', 0):,.2f}",
            f"₹{pick.get('target', 0):,.2f}",
            f"{pick.get('rsi', 0):.0f}",
            f"{pick.get('volume_ratio', 0):.1f}x",
            f"{pick.get('intraday_score', 0):.0f}",
        )

    console.print(table)
    console.print()


def print_swing_picks(picks: list[dict]):
    table = Table(
        title="SWING TRADE OPPORTUNITIES (3-7 Days)",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold blue",
    )

    table.add_column("Symbol", style="cyan bold", width=12)
    table.add_column("Entry", justify="right", width=10)
    table.add_column("SL", justify="right", style="red", width=10)
    table.add_column("Target 1", justify="right", style="green", width=10)
    table.add_column("Target 2", justify="right", style="green", width=10)
    table.add_column("Trend", width=14)
    table.add_column("RSI", justify="center", width=6)
    table.add_column("ADX", justify="center", width=6)
    table.add_column("Patterns", width=20)

    for pick in picks:
        trend = pick.get("trend", "")
        trend_style = "green" if "UP" in trend else "red" if "DOWN" in trend else "yellow"
        patterns = ", ".join(pick.get("patterns", [])[:2]) or "-"

        table.add_row(
            pick.get("symbol", ""),
            f"₹{pick.get('entry', 0):,.2f}",
            f"₹{pick.get('stop_loss', 0):,.2f}",
            f"₹{pick.get('target_1', 0):,.2f}",
            f"₹{pick.get('target_2', 0):,.2f}",
            f"[{trend_style}]{trend}[/]",
            f"{pick.get('rsi', 0):.0f}",
            f"{pick.get('adx', 0):.0f}",
            patterns,
        )

    console.print(table)
    console.print()


def print_option_trades(trades: list[dict]):
    if not trades:
        console.print("[dim]No option trade opportunities found.[/dim]")
        return

    table = Table(
        title="OPTIONS TRADING OPPORTUNITIES",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold magenta",
    )

    table.add_column("Symbol", style="cyan bold", width=12)
    table.add_column("Action", width=12)
    table.add_column("Strike", justify="right", width=10)
    table.add_column("Premium", justify="right", width=10)
    table.add_column("IV%", justify="center", width=8)
    table.add_column("OI", justify="right", width=10)
    table.add_column("Direction", width=10)
    table.add_column("Reason", width=30)

    for trade in trades:
        action = trade.get("action", "")
        action_style = "green" if "BUY" in action else "red"

        table.add_row(
            trade.get("symbol", ""),
            f"[{action_style}]{action}[/]",
            f"₹{trade.get('strike', trade.get('call_strike', 0)):,.0f}",
            f"₹{trade.get('premium', trade.get('total_premium', 0)):,.2f}",
            f"{trade.get('iv', 0):.1f}",
            f"{trade.get('oi', 0):,}",
            trade.get("direction", ""),
            trade.get("reason", "")[:30],
        )

    console.print(table)
    console.print()


def print_risk_summary(risk: dict):
    pnl = risk.get("daily_pnl", 0)
    pnl_style = "green" if pnl >= 0 else "red"

    panel_content = (
        f"[{pnl_style}]Daily P&L: {format_currency(pnl)}[/]\n"
        f"  Realized: {format_currency(risk.get('realized_pnl', 0))}\n"
        f"  Unrealized: {format_currency(risk.get('unrealized_pnl', 0))}\n"
        f"Positions: {risk.get('open_positions', 0)}/{risk.get('max_positions', 5)}\n"
        f"Target: {format_currency(risk.get('daily_target', 500))}\n"
        f"Loss Limit: {format_currency(risk.get('daily_loss_limit', 1000))}"
    )

    console.print(Panel(panel_content, title="RISK DASHBOARD", border_style="bold"))
    console.print()


def print_trade_execution(result: dict):
    status = result.get("status", "unknown")
    trade = result.get("trade", {})

    if status == "executed":
        console.print(Panel(
            f"[green bold]ORDER EXECUTED[/]\n"
            f"Symbol: {trade.get('symbol', '')}\n"
            f"Direction: {trade.get('direction', 'BUY')}\n"
            f"Qty: {trade.get('quantity', 0)}\n"
            f"Entry: ₹{trade.get('entry', 0):,.2f}\n"
            f"SL: ₹{trade.get('stop_loss', 0):,.2f}\n"
            f"Target: ₹{trade.get('target', 0):,.2f}\n"
            f"Risk: ₹{trade.get('risk_amount', 0):,.2f} | "
            f"Reward: ₹{trade.get('reward_amount', 0):,.2f}",
            border_style="green",
        ))
    elif status == "rejected":
        console.print(Panel(
            f"[red]TRADE REJECTED[/]\n"
            f"Symbol: {trade.get('symbol', '')}\n"
            f"Reason: {result.get('reason', 'Unknown')}",
            border_style="red",
        ))
    elif status == "ready":
        console.print(Panel(
            f"[yellow]TRADE READY (Manual Confirmation Required)[/]\n"
            f"Symbol: {trade.get('symbol', '')}\n"
            f"Entry: ₹{trade.get('entry', 0):,.2f} | "
            f"SL: ₹{trade.get('stop_loss', 0):,.2f} | "
            f"Target: ₹{trade.get('target', 0):,.2f}\n"
            f"Qty: {trade.get('quantity', 0)} | "
            f"RR: {trade.get('rr_ratio', 0)}",
            border_style="yellow",
        ))


def print_menu():
    menu = """
[bold cyan]COMMANDS:[/]
  [bold]1[/] - Show Top 10 Picks
  [bold]2[/] - Intraday Opportunities
  [bold]3[/] - Swing Trade Picks
  [bold]4[/] - Options Analysis
  [bold]5[/] - Risk Dashboard
  [bold]6[/] - Execute Recommendations (Manual)
  [bold]7[/] - Start Auto Trading Loop
  [bold]8[/] - Refresh Scan
  [bold]9[/] - Portfolio & Positions
  [bold]0[/] - Exit
"""
    console.print(Panel(menu, title="MENU", border_style="cyan"))
