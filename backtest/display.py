from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from utils.helpers import format_currency

console = Console()


def print_backtest_results(results: dict):
    """Print full backtest results to terminal."""
    if "error" in results:
        console.print(f"[red]{results['error']}[/]")
        return

    s = results["summary"]
    exits = results["exit_reasons"]
    daily = results["daily_stats"]
    streaks = results["streaks"]

    # === OVERVIEW PANEL ===
    pnl_color = "green" if s["net_pnl"] >= 0 else "red"
    overview = (
        f"[bold]Initial Capital:[/] {format_currency(s['initial_capital'])}\n"
        f"[bold]Final Capital:[/]   [{pnl_color}]{format_currency(s['final_capital'])}[/]\n"
        f"[bold {pnl_color}]Net P&L:          {format_currency(s['net_pnl'])} ({s['return_pct']}%)[/]\n"
        f"[bold]Max Drawdown:[/]    [red]{s['max_drawdown_pct']}%[/]\n"
        f"[bold]Profit Factor:[/]   {s['profit_factor']}"
    )
    console.print(Panel(overview, title="BACKTEST RESULTS", border_style="bold cyan"))

    # === TRADE STATS TABLE ===
    stats_table = Table(title="Trade Statistics", box=box.SIMPLE, show_lines=False)
    stats_table.add_column("Metric", style="cyan", width=22)
    stats_table.add_column("Value", justify="right", width=15)

    stats_table.add_row("Total Trades", str(s["total_trades"]))
    stats_table.add_row("Rejected (limits)", str(s["rejected_trades"]))
    stats_table.add_row("Wins", f"[green]{s['wins']}[/]")
    stats_table.add_row("Losses", f"[red]{s['losses']}[/]")
    stats_table.add_row("Win Rate", f"{s['win_rate']}%")
    stats_table.add_row("Avg Win", f"[green]{format_currency(s['avg_win'])}[/]")
    stats_table.add_row("Avg Loss", f"[red]{format_currency(s['avg_loss'])}[/]")
    stats_table.add_row("Largest Win", f"[green]{format_currency(s['largest_win'])}[/]")
    stats_table.add_row("Largest Loss", f"[red]{format_currency(s['largest_loss'])}[/]")
    stats_table.add_row("Avg Hold Days", f"{s['avg_hold_days']}")
    stats_table.add_row("Max Win Streak", f"[green]{streaks['max_win_streak']}[/]")
    stats_table.add_row("Max Lose Streak", f"[red]{streaks['max_lose_streak']}[/]")
    console.print(stats_table)

    # === EXIT REASONS ===
    exit_table = Table(title="Exit Reasons", box=box.SIMPLE)
    exit_table.add_column("Reason", style="cyan", width=15)
    exit_table.add_column("Count", justify="right", width=8)
    exit_table.add_column("% of Trades", justify="right", width=12)
    total = s["total_trades"]
    exit_table.add_row("Target Hit", str(exits["target_hit"]),
                        f"[green]{exits['target_hit']/total*100:.1f}%[/]")
    exit_table.add_row("Stop Loss", str(exits["stop_loss"]),
                        f"[red]{exits['stop_loss']/total*100:.1f}%[/]")
    exit_table.add_row("Time Exit", str(exits["time_exit"]),
                        f"[yellow]{exits['time_exit']/total*100:.1f}%[/]")
    console.print(exit_table)

    # === DAILY STATS ===
    daily_panel = (
        f"Trading Days:     {daily['trading_days']}\n"
        f"Profitable Days:  [green]{daily['profitable_days']}[/]\n"
        f"Losing Days:      [red]{daily['losing_days']}[/]\n"
        f"Avg Daily P&L:    {format_currency(daily['avg_daily_pnl'])}\n"
        f"Best Day:         [green]{format_currency(daily['best_day'])}[/]\n"
        f"Worst Day:        [red]{format_currency(daily['worst_day'])}[/]"
    )
    console.print(Panel(daily_panel, title="Daily Performance", border_style="blue"))

    # === TOP SYMBOLS ===
    sym_table = Table(title="Top Performing Symbols", box=box.ROUNDED, show_lines=True)
    sym_table.add_column("Symbol", style="cyan bold", width=12)
    sym_table.add_column("Trades", justify="right", width=7)
    sym_table.add_column("Win Rate", justify="right", width=9)
    sym_table.add_column("P&L", justify="right", width=12)
    sym_table.add_column("Avg Win", justify="right", width=10)
    sym_table.add_column("Avg Loss", justify="right", width=10)

    for sym in results.get("top_symbols", []):
        pnl = sym.get("total_pnl", 0)
        pnl_c = "green" if pnl >= 0 else "red"
        sym_table.add_row(
            sym["symbol"],
            str(sym.get("total_trades", 0)),
            f"{sym.get('win_rate', 0)}%",
            f"[{pnl_c}]{format_currency(pnl)}[/]",
            f"[green]{format_currency(sym.get('avg_win', 0))}[/]",
            f"[red]{format_currency(sym.get('avg_loss', 0))}[/]",
        )
    console.print(sym_table)

    # === WORST SYMBOLS ===
    if results.get("bottom_symbols"):
        bot_table = Table(title="Worst Performing Symbols", box=box.ROUNDED, show_lines=True)
        bot_table.add_column("Symbol", style="cyan bold", width=12)
        bot_table.add_column("Trades", justify="right", width=7)
        bot_table.add_column("Win Rate", justify="right", width=9)
        bot_table.add_column("P&L", justify="right", width=12)

        for sym in results["bottom_symbols"]:
            pnl = sym.get("total_pnl", 0)
            pnl_c = "green" if pnl >= 0 else "red"
            bot_table.add_row(
                sym["symbol"],
                str(sym.get("total_trades", 0)),
                f"{sym.get('win_rate', 0)}%",
                f"[{pnl_c}]{format_currency(pnl)}[/]",
            )
        console.print(bot_table)

    # === RECENT TRADES (last 20) ===
    trades = results.get("trades", [])
    if trades:
        trade_table = Table(
            title=f"Trade Log (showing last 20 of {len(trades)})",
            box=box.SIMPLE, show_lines=False,
        )
        trade_table.add_column("Date", width=12)
        trade_table.add_column("Symbol", style="cyan", width=12)
        trade_table.add_column("Dir", width=5)
        trade_table.add_column("Entry", justify="right", width=9)
        trade_table.add_column("Exit", justify="right", width=9)
        trade_table.add_column("Qty", justify="right", width=5)
        trade_table.add_column("P&L", justify="right", width=10)
        trade_table.add_column("Exit Reason", width=12)
        trade_table.add_column("Days", justify="right", width=5)

        for t in trades[-20:]:
            pnl_c = "green" if t["pnl"] >= 0 else "red"
            dir_c = "green" if t["direction"] == "BUY" else "red"
            trade_table.add_row(
                t["entry_date"].strftime("%Y-%m-%d"),
                t["symbol"],
                f"[{dir_c}]{t['direction']}[/]",
                f"{t['entry_price']:,.1f}",
                f"{t['exit_price']:,.1f}",
                str(t["quantity"]),
                f"[{pnl_c}]{format_currency(t['pnl'])}[/]",
                t["exit_reason"],
                str(t["hold_days"]),
            )
        console.print(trade_table)

    # === EQUITY CURVE (text sparkline) ===
    equity = results.get("equity_curve", [])
    if len(equity) > 2:
        values = [e["equity"] for e in equity]
        _print_equity_sparkline(values, s["initial_capital"])


def _print_equity_sparkline(values: list[float], initial: float):
    """Print a simple text-based equity curve."""
    width = 60
    if len(values) <= 1:
        return

    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1

    lines = []
    height = 12
    for row in range(height, -1, -1):
        threshold = mn + (rng * row / height)
        line = ""
        step = max(1, len(values) // width)
        for i in range(0, min(len(values), width * step), step):
            if values[i] >= threshold:
                if values[i] >= initial:
                    line += "[green]\u2588[/]"
                else:
                    line += "[red]\u2588[/]"
            else:
                line += " "
        lines.append(line)

    chart = "\n".join(lines)
    label = (
        f"  High: {format_currency(mx)}  |  Low: {format_currency(mn)}  |  "
        f"Final: {format_currency(values[-1])}"
    )
    console.print(Panel(
        chart + "\n" + label,
        title="Equity Curve",
        border_style="cyan",
    ))
