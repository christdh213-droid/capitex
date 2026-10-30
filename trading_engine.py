"""Trading Execution Engine, Risk Management, and Portfolio Analytics.

Implements:
1. Atomic Buy and Sell execution with cash/share validation inside single SQLite transactions.
2. Recomputation of weighted average buy price and tracking of peak holding prices.
3. Dynamic Trailing Stop-Loss (relative to peak price) and Take-Profit automated triggers.
4. Scaled Tranche Position Sizing and execution for autonomous AI agent.
5. Comprehensive portfolio performance analytics (Net Worth, Unrealized P&L,
   Win/Loss Ratio, Maximum Drawdown %, and Buy & Hold ROI %).
"""

from datetime import datetime, timezone
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import config
import database
import data_loader
from formatting import signed_money

logger = logging.getLogger(__name__)

# Backend price cache for 60-second quote durability
_PRICE_CACHE: Dict[str, Tuple[float, float]] = {}  # ticker -> (price, timestamp_epoch)


def get_cached_live_price(ticker: str) -> float:
    """Retrieve live price with a 60-second in-memory cache to avoid yfinance rate-limiting."""
    clean_ticker = ticker.strip().upper()
    now_epoch = datetime.now(timezone.utc).timestamp()

    if clean_ticker in _PRICE_CACHE:
        cached_price, cached_time = _PRICE_CACHE[clean_ticker]
        if now_epoch - cached_time < config.BACKEND_PRICE_CACHE_TTL and cached_price > 0:
            return cached_price

    quote = data_loader.get_live_price(clean_ticker)
    price = float(quote.get("price", 0.0))
    if price > 0:
        _PRICE_CACHE[clean_ticker] = (price, now_epoch)
        return price

    return 0.0


def get_execution_price(ticker: str) -> Dict[str, Any]:
    """One authoritative fill resolver for human, AI and risk orders.

    Always bypass both display caches. Never accept a client's displayed price or
    a predictor's historical close as a fill. Persist the fallback label/source.
    """
    ticker = data_loader._symbol(ticker)
    try:
        quote = data_loader.get_live_quote(ticker, force_refresh=True)
        price = float(quote['price'])
        if not np.isfinite(price) or price <= 0:
            raise ValueError('Invalid execution quote')
        fallback = bool(quote.get('is_fallback') or quote.get('is_stale'))
        if not fallback:
            return dict(price=price, execution_source='live_quote',
                        execution_label='Live quote (provider may be delayed)',
                        quote_timestamp=quote.get('timestamp'), is_fallback=False)
        # A stale intraday quote is not a fresh fill: resolve the last daily close.
        if quote.get('is_fallback') and quote.get('interval') == '1d':
            return dict(price=price, execution_source='last_close',
                        execution_label='FALLBACK: last historical close',
                        quote_timestamp=quote.get('timestamp'), is_fallback=True)
    except Exception as exc:
        logger.warning('Fresh execution quote unavailable for %s: %s', ticker, exc)
    history = data_loader._history(ticker, period='1mo', interval='1d')
    history = history.loc[history.index <= data_loader._utc_now()]
    if history.empty:
        raise ValueError(f'No execution quote or historical close for {ticker}.')
    price = float(history.Close.iloc[-1])
    if not np.isfinite(price) or price <= 0:
        raise ValueError(f'No valid execution price for {ticker}.')
    return dict(price=price, execution_source='last_close',
                execution_label='FALLBACK: last historical close',
                quote_timestamp=history.index[-1].isoformat(), is_fallback=True)


def compute_net_worth_in_tx(
    cursor: Any,
    wallet_type: str,
    override_prices: Optional[Dict[str, float]] = None,
) -> float:
    """Calculate total net worth (cash + mark-to-market holdings) within an active transaction."""
    cursor.execute("SELECT cash FROM wallets WHERE wallet_type = ?", (wallet_type,))
    row = cursor.fetchone()
    cash = float(row["cash"]) if row else 0.0

    cursor.execute(
        "SELECT ticker, quantity, avg_buy_price FROM holdings WHERE wallet_type = ? AND quantity > 0",
        (wallet_type,)
    )
    holdings = cursor.fetchall()

    holdings_val = 0.0
    override_prices = override_prices or {}
    for h in holdings:
        t = h["ticker"]
        qty = float(h["quantity"])
        p = override_prices.get(t) or get_cached_live_price(t) or float(h["avg_buy_price"])
        holdings_val += qty * p

    return cash + holdings_val


def execute_buy(
    wallet_type: str,
    ticker: str,
    quantity: float,
    current_price: Optional[float] = None,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    *, cash_budget: Optional[float] = None, max_position_value: Optional[float] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Execute a paper BUY trade inside a single atomic database transaction.

    Validates:
    - quantity > 0 and price > 0
    - available cash >= total cost (no negative cash)

    Updates:
    - wallet cash
    - holdings (updates quantity, recomputes weighted avg_buy_price, sets/updates peak_price)
    - trade_ledger (records action='BUY')
    - portfolio_snapshots (records net worth after execution)
    """
    clean_ticker = ticker.strip().upper()
    qty = float(quantity)
    # current_price is retained for caller compatibility only; never trusted.
    try:
        execution = get_execution_price(clean_ticker)
    except Exception as exc:
        return False, f'Cannot execute {clean_ticker}: {exc}', {}
    price = execution['price']

    if not np.isfinite(qty) or qty <= 0:
        return False, "Quantity must be strictly greater than zero.", {}
    if price <= 0:
        return False, f"Invalid purchase price (${price:.2f}).", {}

    total_cost = float(cash_budget) if cash_budget is not None else qty * price
    now_iso = datetime.now(timezone.utc).isoformat()

    with database.get_legacy_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()

        # 1. Validate Cash
        cursor.execute("SELECT cash FROM wallets WHERE wallet_type = ?", (wallet_type,))
        wallet_row = cursor.fetchone()
        if not wallet_row:
            return False, f"Wallet '{wallet_type}' not found.", {}

        available_cash = float(wallet_row["cash"])
        if available_cash < total_cost:
            return (
                False,
                f"Insufficient funds: Required ${total_cost:,.2f}, Available: ${available_cash:,.2f}.",
                {},
            )

        # AI budgets are converted to quantity at this same authoritative fill.
        if cash_budget is not None:
            spend = min(float(cash_budget), available_cash)
            if max_position_value is not None:
                held = cursor.execute('SELECT quantity FROM holdings WHERE wallet_type=? AND ticker=?',
                                      (wallet_type, clean_ticker)).fetchone()
                existing_value = (float(held['quantity']) if held else 0.0) * price
                spend = min(spend, max(0.0, max_position_value-existing_value))
            if not np.isfinite(spend) or spend < 10:
                return False, 'AI allocation is below the minimum $10.00.', {}
            qty = spend / price
            total_cost = spend

        # 2. Deduct Cash
        new_cash = available_cash - total_cost
        cursor.execute("UPDATE wallets SET cash = ? WHERE wallet_type = ?", (new_cash, wallet_type))

        # 3. Update Holdings
        cursor.execute(
            """
            SELECT quantity, avg_buy_price, peak_price, stop_loss, take_profit
            FROM holdings
            WHERE wallet_type = ? AND ticker = ?
            """,
            (wallet_type, clean_ticker),
        )
        existing = cursor.fetchone()

        sl = float(stop_loss_pct) if stop_loss_pct is not None else (existing["stop_loss"] if existing else None)
        tp = float(take_profit_pct) if take_profit_pct is not None else (existing["take_profit"] if existing else None)

        if existing and float(existing["quantity"]) > 0:
            old_qty = float(existing["quantity"])
            old_avg = float(existing["avg_buy_price"])
            old_peak = float(existing["peak_price"]) if existing["peak_price"] is not None else old_avg

            new_qty = old_qty + qty
            new_avg = ((old_qty * old_avg) + total_cost) / new_qty
            new_peak = max(old_peak, price)

            cursor.execute(
                """
                UPDATE holdings
                SET quantity = ?, avg_buy_price = ?, peak_price = ?, stop_loss = ?, take_profit = ?
                WHERE wallet_type = ? AND ticker = ?
                """,
                (new_qty, new_avg, new_peak, sl, tp, wallet_type, clean_ticker),
            )
        else:
            cursor.execute(
                """
                INSERT INTO holdings (wallet_type, ticker, quantity, avg_buy_price, peak_price, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(wallet_type, ticker) DO UPDATE SET
                    quantity = excluded.quantity,
                    avg_buy_price = excluded.avg_buy_price,
                    peak_price = excluded.peak_price,
                    stop_loss = excluded.stop_loss,
                    take_profit = excluded.take_profit
                """,
                (wallet_type, clean_ticker, qty, price, price, sl, tp),
            )

        # 4. Insert into Trade Ledger
        cursor.execute(
            """
            INSERT INTO trade_ledger (timestamp, wallet_type, ticker, action, quantity, price, realized_pnl, execution_source, execution_label, quote_timestamp)
            VALUES (?, ?, ?, 'BUY', ?, ?, 0.0, ?, ?, ?)
            """,
            (now_iso, wallet_type, clean_ticker, qty, price, execution["execution_source"], execution["execution_label"], execution["quote_timestamp"]),
        )

        # 5. Record Net Worth Snapshot
        net_worth = compute_net_worth_in_tx(cursor, wallet_type, override_prices={clean_ticker: price})
        cursor.execute(
            "INSERT INTO portfolio_snapshots (timestamp, wallet_type, net_worth, reason) VALUES (?, ?, ?, 'trade')",
            (now_iso, wallet_type, net_worth),
        )

        logger.info(
            f"[{wallet_type.upper()}] BUY {qty:.4f} {clean_ticker} @ ${price:.2f} "
            f"(Cost: ${total_cost:,.2f}, Remaining Cash: ${new_cash:,.2f})"
        )

    return (
        True,
        f"Bought {qty:.4f} shares of {clean_ticker} at ${price:.2f} (Total: ${total_cost:,.2f}) — {execution['execution_label']}",
        {
            **execution,
            "action": "BUY",
            "ticker": clean_ticker,
            "quantity": qty,
            "price": price,
            "cost": total_cost,
            "remaining_cash": new_cash,
            "net_worth": net_worth,
        },
    )


def execute_sell(
    wallet_type: str,
    ticker: str,
    quantity: float,
    current_price: Optional[float] = None,
    reason: str = "MANUAL_SELL",
) -> Tuple[bool, str, Dict[str, Any]]:
    """Execute a paper SELL trade inside a single atomic database transaction.

    Validates:
    - quantity > 0 and price > 0
    - position exists and available shares >= quantity (no short selling)

    Updates:
    - wallet cash (+ proceeds)
    - holdings (deducts quantity, deletes or zeroes row if closed)
    - trade_ledger (records action='SELL', 'STOP_LOSS', or 'TAKE_PROFIT' with realized_pnl)
    - portfolio_snapshots (records net worth after execution)
    """
    clean_ticker = ticker.strip().upper()
    qty = float(quantity)
    # current_price is retained for caller compatibility only; never trusted.
    try:
        execution = get_execution_price(clean_ticker)
    except Exception as exc:
        return False, f'Cannot execute {clean_ticker}: {exc}', {}
    price = execution['price']

    if not np.isfinite(qty) or qty <= 0:
        return False, "Sell quantity must be strictly greater than zero.", {}
    if price <= 0:
        return False, f"Invalid execution price (${price:.2f}).", {}

    now_iso = datetime.now(timezone.utc).isoformat()

    with database.get_legacy_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()

        # 1. Validate Holding
        cursor.execute(
            """
            SELECT quantity, avg_buy_price
            FROM holdings
            WHERE wallet_type = ? AND ticker = ? AND quantity > 0
            """,
            (wallet_type, clean_ticker),
        )
        existing = cursor.fetchone()
        if not existing:
            return False, f"No active position in {clean_ticker} to sell.", {}

        current_qty = float(existing["quantity"])
        avg_buy_price = float(existing["avg_buy_price"])

        if current_qty < qty - 1e-6:
            return (
                False,
                f"Insufficient shares: Attempting to sell {qty:.4f}, but only {current_qty:.4f} held.",
                {},
            )

        # Cap qty to current_qty to eliminate floating-point epsilon
        qty_to_sell = min(qty, current_qty)
        proceeds = qty_to_sell * price
        cost_basis = qty_to_sell * avg_buy_price
        realized_pnl = proceeds - cost_basis

        # 2. Credit Cash
        cursor.execute("SELECT cash FROM wallets WHERE wallet_type = ?", (wallet_type,))
        cash_row = cursor.fetchone()
        old_cash = float(cash_row["cash"]) if cash_row else 0.0
        new_cash = old_cash + proceeds
        cursor.execute("UPDATE wallets SET cash = ? WHERE wallet_type = ?", (new_cash, wallet_type))

        # 3. Update or Remove Holding
        remaining_qty = current_qty - qty_to_sell
        if remaining_qty <= 1e-6:
            cursor.execute(
                "DELETE FROM holdings WHERE wallet_type = ? AND ticker = ?",
                (wallet_type, clean_ticker),
            )
        else:
            cursor.execute(
                "UPDATE holdings SET quantity = ? WHERE wallet_type = ? AND ticker = ?",
                (remaining_qty, wallet_type, clean_ticker),
            )

        # 4. Determine Action Type for Ledger
        ledger_action = "SELL"
        if "STOP_LOSS" in reason.upper():
            ledger_action = "STOP_LOSS"
        elif "TAKE_PROFIT" in reason.upper():
            ledger_action = "TAKE_PROFIT"

        cursor.execute(
            """
            INSERT INTO trade_ledger (timestamp, wallet_type, ticker, action, quantity, price, realized_pnl, execution_source, execution_label, quote_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (now_iso, wallet_type, clean_ticker, ledger_action, qty_to_sell, price, realized_pnl, execution["execution_source"], execution["execution_label"], execution["quote_timestamp"]),
        )

        # 5. Record Net Worth Snapshot
        net_worth = compute_net_worth_in_tx(cursor, wallet_type, override_prices={clean_ticker: price})
        cursor.execute(
            "INSERT INTO portfolio_snapshots (timestamp, wallet_type, net_worth, reason) VALUES (?, ?, ?, 'trade')",
            (now_iso, wallet_type, net_worth),
        )

        logger.info(
            f"[{wallet_type.upper()}] {ledger_action} {qty_to_sell:.4f} {clean_ticker} @ ${price:.2f} "
            f"(Proceeds: ${proceeds:,.2f}, Realized P&L: {signed_money(realized_pnl)})"
        )

    return (
        True,
        f"Sold {qty_to_sell:.4f} shares of {clean_ticker} at ${price:.2f} "
        f"(Realized P&L: {signed_money(realized_pnl)}) — {execution['execution_label']}",
        {
            **execution,
            "action": ledger_action,
            "ticker": clean_ticker,
            "quantity": qty_to_sell,
            "price": price,
            "proceeds": proceeds,
            "realized_pnl": realized_pnl,
            "new_cash": new_cash,
            "net_worth": net_worth,
            "reason": reason,
        },
    )


def check_risk_triggers(wallet_type: str) -> List[Dict[str, Any]]:
    """Evaluate Dynamic Trailing Stop-Loss and Take-Profit rules for all active positions.

    Logic:
    - Updates peak_price = max(peak_price, live_price)
    - Trailing Stop: If live_price <= peak_price * (1 - stop_loss_pct / 100), auto-sells.
    - Take Profit: If live_price >= avg_buy_price * (1 + take_profit_pct / 100), auto-sells.

    Returns:
        List of triggered trade events.
    """
    holdings = database.get_holdings(wallet_type)
    triggered_events = []

    for h in holdings:
        ticker = h["ticker"]
        qty = float(h["quantity"])
        avg_price = float(h["avg_buy_price"])
        peak_price = float(h["peak_price"]) if h["peak_price"] is not None else avg_price
        sl_pct = float(h["stop_loss"]) if h["stop_loss"] is not None else None
        tp_pct = float(h["take_profit"]) if h["take_profit"] is not None else None

        live_price = get_cached_live_price(ticker)
        if live_price <= 0:
            continue

        # Update peak price if new high reached
        if live_price > peak_price:
            peak_price = live_price
            database.update_holding_peak_price(wallet_type, ticker, peak_price)

        # 1. Trailing Stop-Loss Check
        if sl_pct is not None and sl_pct > 0:
            stop_threshold = peak_price * (1.0 - (sl_pct / 100.0))
            if live_price <= stop_threshold:
                reason = f"STOP_LOSS (Price ${live_price:.2f} <= Trailing Stop ${stop_threshold:.2f} from Peak ${peak_price:.2f})"
                success, msg, data = execute_sell(wallet_type, ticker, qty, live_price, reason=reason)
                if success:
                    triggered_events.append(data)
                continue  # Position closed, skip take profit

        # 2. Take-Profit Check
        if tp_pct is not None and tp_pct > 0:
            target_threshold = avg_price * (1.0 + (tp_pct / 100.0))
            if live_price >= target_threshold:
                reason = f"TAKE_PROFIT (Price ${live_price:.2f} >= Target ${target_threshold:.2f} from Avg ${avg_price:.2f})"
                success, msg, data = execute_sell(wallet_type, ticker, qty, live_price, reason=reason)
                if success:
                    triggered_events.append(data)

    get_portfolio_summary(wallet_type, snapshot_reason="risk")
    return triggered_events


def run_ai_trading_step(
    ticker: str,
    signal: str,
    predicted_return_pct: float,
    current_price: float,
    tranche_pct: float = config.DEFAULT_AI_POSITION_PCT,
    max_position_pct: float = config.DEFAULT_AI_MAX_POSITION_PCT,
    stop_loss_pct: float = config.DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = config.DEFAULT_TAKE_PROFIT_PCT,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Execute rule-based trading step for autonomous AI portfolio agent.

    Scaled Tranche Buying:
    - On BUY signal: spends tranche_pct (default 10%) of available cash,
      up to max_position_pct of total net worth.
    - On SELL signal: liquidates entire holding in ticker.
    - On HOLD signal: maintains position.
    """
    clean_ticker = ticker.strip().upper()
    signal_upper = signal.strip().upper()

    # Display mark for preliminary allocation only; execute_buy resolves the fill.
    current_price = get_cached_live_price(clean_ticker) or 1.0

    # 1. BUY Signal
    if signal_upper == "BUY":
        wallet = database.get_wallet(config.WALLET_TYPE_AI)
        if not wallet:
            return False, "AI wallet not found.", {}

        available_cash = float(wallet["cash"])
        if available_cash < 10.0:
            return False, "AI has insufficient cash (< $10.00) to execute buy.", {}

        # Check existing position cap
        summary = get_portfolio_summary(config.WALLET_TYPE_AI)
        net_worth = summary["net_worth"]
        existing_holding = database.get_holding_for_ticker(config.WALLET_TYPE_AI, clean_ticker)
        existing_val = (
            float(existing_holding["quantity"]) * current_price if existing_holding else 0.0
        )

        max_allowed_val = net_worth * (max_position_pct / 100.0)
        if existing_val >= max_allowed_val:
            return (
                False,
                f"AI position cap reached for {clean_ticker} (${existing_val:,.2f} / ${max_allowed_val:,.2f} limit).",
                {},
            )

        # Allocate tranche
        tranche_cash = available_cash * (tranche_pct / 100.0)
        # Cap tranche so position does not exceed max_allowed_val
        tranche_cash = min(tranche_cash, max_allowed_val - existing_val)
        if tranche_cash < 10.0:
            return False, "Allocated tranche is less than minimum $10.00.", {}

        qty_to_buy = tranche_cash / current_price
        success, msg, data = execute_buy(
            wallet_type=config.WALLET_TYPE_AI,
            ticker=clean_ticker,
            quantity=1.0,
            cash_budget=tranche_cash,
            max_position_value=max_allowed_val,
            current_price=current_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        return success, f"AI Tranche BUY: {msg}", data

    # 2. SELL Signal
    elif signal_upper == "SELL":
        existing_holding = database.get_holding_for_ticker(config.WALLET_TYPE_AI, clean_ticker)
        if not existing_holding or float(existing_holding["quantity"]) <= 0:
            return False, f"AI has no open position in {clean_ticker} to liquidate.", {}

        qty_to_sell = float(existing_holding["quantity"])
        success, msg, data = execute_sell(
            wallet_type=config.WALLET_TYPE_AI,
            ticker=clean_ticker,
            quantity=qty_to_sell,
            current_price=current_price,
            reason="AI_SIGNAL_SELL",
        )
        return success, f"AI Liquidation SELL: {msg}", data

    # 3. HOLD Signal
    else:
        return True, f"AI holding steady on {clean_ticker} (Signal: HOLD)", {"action": "HOLD"}


def get_portfolio_summary(
    wallet_type: str,
    live_prices: Optional[Dict[str, float]] = None,
    *, snapshot_reason: Optional[str] = "refresh",
) -> Dict[str, Any]:
    """Calculate complete mark-to-market performance metrics and analytics for a wallet.

    Returns:
        Dictionary containing:
        - cash: float
        - holdings: List of holdings with live mark-to-market valuations
        - total_holdings_value: float
        - net_worth: float
        - total_unrealized_pnl: float
        - total_realized_pnl: float
        - win_rate_pct: float
        - win_loss_ratio: float
        - max_drawdown_pct: float
        - benchmark_roi_pct: float
        - snapshots: List of historical net-worth records
    """
    wallet = database.get_wallet(wallet_type)
    cash = float(wallet["cash"]) if wallet else 0.0

    raw_holdings = database.get_holdings(wallet_type)
    holdings_enriched = []
    total_holdings_val = 0.0
    total_unrealized = 0.0

    live_prices = live_prices or {}

    for h in raw_holdings:
        ticker = h["ticker"]
        qty = float(h["quantity"])
        avg_price = float(h["avg_buy_price"])
        peak_p = float(h["peak_price"]) if h["peak_price"] is not None else avg_price
        sl = float(h["stop_loss"]) if h["stop_loss"] is not None else None
        tp = float(h["take_profit"]) if h["take_profit"] is not None else None

        current_p = live_prices.get(ticker) or get_cached_live_price(ticker)
        if current_p <= 0:
            current_p = avg_price  # fallback to purchase cost

        peak_p = max(peak_p, current_p)
        database.update_holding_peak_price(wallet_type, ticker, peak_p)

        mkt_val = qty * current_p
        cost_basis = qty * avg_price
        unrealized = mkt_val - cost_basis
        unrealized_pct = (unrealized / cost_basis) * 100.0 if cost_basis > 0 else 0.0

        # Trailing stop trigger price
        trailing_stop_price = peak_p * (1.0 - (sl / 100.0)) if sl else None
        take_profit_price = avg_price * (1.0 + (tp / 100.0)) if tp else None

        total_holdings_val += mkt_val
        total_unrealized += unrealized

        holdings_enriched.append(
            {
                "ticker": ticker,
                "quantity": qty,
                "avg_buy_price": avg_price,
                "current_price": current_p,
                "peak_price": peak_p,
                "market_value": round(mkt_val, 2),
                "unrealized_pnl": round(unrealized, 2),
                "unrealized_pnl_pct": round(unrealized_pct, 2),
                "stop_loss_pct": sl,
                "take_profit_pct": tp,
                "trailing_stop_price": round(trailing_stop_price, 2) if trailing_stop_price else None,
                "take_profit_price": round(take_profit_price, 2) if take_profit_price else None,
            }
        )

    net_worth = cash + total_holdings_val

    # Win / Loss Ratio from trade ledger
    with database.get_legacy_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_sells,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
                SUM(realized_pnl) as total_pnl
            FROM trade_ledger
            WHERE wallet_type = ? AND action IN ('SELL', 'STOP_LOSS', 'TAKE_PROFIT')
            """,
            (wallet_type,),
        )
        pnl_stats = cursor.fetchone()

        total_sells = int(pnl_stats["total_sells"] or 0)
        wins = int(pnl_stats["wins"] or 0)
        losses = int(pnl_stats["losses"] or 0)
        total_realized_pnl = float(pnl_stats["total_pnl"] or 0.0)

    win_rate_pct = (wins / total_sells * 100.0) if total_sells > 0 else None
    win_loss_ratio = ((wins / losses) if losses > 0 else float(wins)) if total_sells else None

    with database.get_legacy_connection() as conn:
        total_trades = conn.execute(
            "SELECT COUNT(*) FROM trade_ledger WHERE wallet_type=? AND action IN ('BUY','SELL','STOP_LOSS','TAKE_PROFIT')",
            (wallet_type,)).fetchone()[0]
    if snapshot_reason is not None and wallet:
        database.record_snapshot(wallet_type, net_worth, reason=snapshot_reason,
                                 min_interval_seconds=30 if snapshot_reason == 'refresh' else 0)

    # Maximum Drawdown % from snapshots
    snapshots = database.get_portfolio_snapshots(wallet_type)
    max_drawdown_pct = 0.0 if snapshots else None
    if len(snapshots) >= 2:
        nw_series = np.array([float(s["net_worth"]) for s in snapshots])
        peak = np.maximum.accumulate(nw_series)
        drawdowns = np.divide(nw_series-peak, peak, out=np.zeros_like(nw_series), where=peak>0) * 100.0
        max_drawdown_pct = float(np.min(drawdowns))  # negative percentage

    # ROI % vs starting cash ($100k)
    benchmark_roi_pct = ((net_worth - config.STARTING_CASH) / config.STARTING_CASH) * 100.0

    return {
        "wallet_type": wallet_type,
        "cash": round(cash, 2),
        "total_holdings_value": round(total_holdings_val, 2),
        "net_worth": round(net_worth, 2),
        "unrealized_pnl": round(total_unrealized, 2),
        "realized_pnl": round(total_realized_pnl, 2),
        "holdings": holdings_enriched,
        "win_rate_pct": round(win_rate_pct, 1) if win_rate_pct is not None else None,
        "win_loss_ratio": round(win_loss_ratio, 2) if win_loss_ratio is not None else None,
        "total_closed_trades": total_sells,
        "total_trades": total_trades,
        "max_drawdown_pct": round(max_drawdown_pct, 2) if max_drawdown_pct is not None else None,
        "benchmark_roi_pct": round(benchmark_roi_pct, 2),
        "snapshots": snapshots,
    }

# CAPITEX single-wallet engine. Shares the exact fill resolver with legacy orders.
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
import math
import branding


def asset_metadata(ticker):
    ticker=data_loader._symbol(ticker)
    asset=config.ASSET_UNIVERSE.get(ticker,{})
    return {'ticker':ticker,'name':asset.get('name',ticker),
            'kind':asset.get('kind','stock'),'logo_url':asset.get('logo_url'),
            'logo':branding.asset_logo_source(ticker),
            'logo_fallback':branding.asset_logo_source(ticker,False)}


def _market_asset(ticker):
    asset=asset_metadata(ticker)
    try:
        quote=data_loader.get_live_quote(ticker)
        if not np.isfinite(quote['price']) or quote['price']<=0:
            raise ValueError('Invalid market quote')
        return dict(asset,quote=quote,price=quote['price'],available=True,
                    change_24h=quote.get('change_24h'),change_24h_pct=quote.get('change_24h_pct'))
    except Exception as exc:
        logger.warning('Market quote unavailable for %s: %s',ticker,exc)
        return dict(asset,quote=None,price=None,available=False,change_24h=None,change_24h_pct=None)


def _market_assets(tickers):
    names=list(dict.fromkeys(tickers))
    if not names: return {}
    with ThreadPoolExecutor(max_workers=min(4,len(names))) as pool:
        return {row['ticker']:row for row in pool.map(_market_asset,names)}


def _capitex_snapshot(conn, prices, reason):
    """Called within a write transaction; refresh throttling is cross-process."""
    now=datetime.now(timezone.utc)
    if reason=='refresh':
        previous=conn.execute("SELECT timestamp FROM portfolio_snapshots WHERE reason='refresh' ORDER BY id DESC LIMIT 1").fetchone()
        if previous and (now-datetime.fromisoformat(previous['timestamp'])).total_seconds()<30:
            return
    cash=float(conn.execute('SELECT cash FROM wallet WHERE id=1').fetchone()[0])
    invested=sum(float(row['quantity'])*prices.get(row['ticker'],float(row['avg_buy_price']))
                 for row in conn.execute('SELECT * FROM holdings WHERE quantity>0'))
    conn.execute('INSERT INTO portfolio_snapshots(timestamp,net_worth,cash,invested_value,reason) VALUES(?,?,?,?,?)',
                 (now.isoformat(),cash+invested,cash,invested,reason))


def capitex_wallet_summary(*, include_watchlist=True, snapshot_reason='refresh'):
    with database.get_connection() as conn:
        raw=[dict(r) for r in conn.execute('SELECT * FROM holdings WHERE quantity>0 ORDER BY ticker')]
    symbols=[row['ticker'] for row in raw]
    markets=_market_assets(symbols+(config.DEFAULT_TICKERS if include_watchlist else []))
    prices={t:m['price'] for t,m in markets.items() if m['available']}
    with database.get_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        # Re-read inside the transaction: a concurrent order may have changed cash/quantity.
        wallet=dict(conn.execute('SELECT * FROM wallet WHERE id=1').fetchone())
        raw=[dict(r) for r in conn.execute('SELECT * FROM holdings WHERE quantity>0 ORDER BY ticker')]
        holdings=[]
        estimated=False
        for row in raw:
            ticker=row['ticker']
            asset=markets.get(ticker)
            if asset is None:
                # No network while holding a database write lock.
                asset={'ticker':ticker,'name':row['name'],'price':None,'available':False,
                       'logo':branding.asset_logo_source(ticker,False),'change_24h_pct':None}
            available=asset['available']
            mark=float(asset['price'] if available else row['avg_buy_price'])
            estimated |= not available
            peak=max(float(row['peak_price'] or row['avg_buy_price']),mark)
            if available:
                conn.execute('UPDATE holdings SET peak_price=MAX(COALESCE(peak_price,avg_buy_price),?) WHERE ticker=?',(peak,ticker))
            value=mark*row['quantity']
            cost=row['quantity']*row['avg_buy_price']
            holdings.append({**asset, **row, 'current_price':mark, 'peak_price':peak,
                             'market_value':value, 'unrealized_pnl':value-cost,
                             'unrealized_pnl_pct':(value/cost-1)*100 if cost else 0,
                             'valuation_source':'market_quote' if available else 'cost_estimate'})
        invested=sum(h['market_value'] for h in holdings)
        net_worth=wallet['cash']+invested
        if snapshot_reason and not estimated:
            _capitex_snapshot(conn,prices,snapshot_reason)
        snapshots=[dict(r) for r in conn.execute('SELECT * FROM portfolio_snapshots ORDER BY id')]
        stats=conn.execute("""SELECT COUNT(*) n,
            SUM(CASE WHEN realized_pnl>0 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN realized_pnl<0 THEN 1 ELSE 0 END) losses,
            COALESCE(SUM(realized_pnl),0) pnl FROM trade_ledger
            WHERE action IN ('SELL','STOP_LOSS','TAKE_PROFIT')""").fetchone()
        trades=conn.execute("SELECT COUNT(*) FROM trade_ledger WHERE action IN ('BUY','SELL','STOP_LOSS','TAKE_PROFIT')").fetchone()[0]
        ledger=[dict(r) for r in conn.execute('SELECT * FROM trade_ledger ORDER BY id DESC LIMIT 200')]
    cutoff=datetime.now(timezone.utc).timestamp()-86400
    prior=[s for s in snapshots if datetime.fromisoformat(s['timestamp']).timestamp()<=cutoff]
    baseline=prior[-1] if prior else snapshots[0] if snapshots else {'net_worth':wallet['starting_cash']}
    change=net_worth-baseline['net_worth']
    values=np.array([s['net_worth'] for s in snapshots],dtype=float)
    drawdown=None
    if len(values):
        peaks=np.maximum.accumulate(values)
        drawdown=float(np.min(np.divide(values-peaks,peaks,out=np.zeros_like(values),where=peaks>0))*100)
    return dict(cash=wallet['cash'],starting_cash=wallet['starting_cash'],invested_value=invested,
                net_worth=net_worth,change_24h=change,
                change_24h_pct=change/baseline['net_worth']*100 if baseline['net_worth'] else 0,
                change_period='24h' if prior else 'since_start',holdings=holdings,
                watchlist=[markets[t] for t in config.DEFAULT_TICKERS if t in markets] if include_watchlist else [],
                valuation_estimated=estimated,trade_ledger=ledger,snapshots=snapshots,
                total_trades=trades,total_closed_trades=stats['n'],realized_pnl=stats['pnl'],
                win_rate_pct=stats['wins']/stats['n']*100 if stats['n'] else None,
                win_loss_ratio=(stats['wins']/stats['losses'] if stats['losses'] else float(stats['wins'] or 0)) if stats['n'] else None,
                max_drawdown_pct=drawdown,as_of=datetime.now(timezone.utc).isoformat())


def _order_receipt(row, cash, replay=False):
    return dict(row,cash=cash,is_fallback=row['execution_source']=='last_close',replayed=replay)


def execute_capitex_trade(ticker, action, *, amount=None, quantity=None, client_order_id=None,
                          stop_loss_pct=None, take_profit_pct=None, reason=None):
    """Atomic cash/quantity validation with fresh fills and idempotent client retries."""
    ticker=data_loader._symbol(ticker)
    action=action.upper()
    if action not in ('BUY','SELL'): raise ValueError('Choose BUY or SELL.')
    if (amount is None)==(quantity is None): raise ValueError('Provide exactly one dollar amount or quantity.')
    value=amount if amount is not None else quantity
    if isinstance(value,bool) or not math.isfinite(float(value)) or float(value)<=0:
        raise ValueError('Order size must be finite and positive.')
    for pct in (stop_loss_pct,take_profit_pct):
        if pct is not None and (not math.isfinite(float(pct)) or not 0<float(pct)<=100):
            raise ValueError('Protection percentages must be in (0,100].')
    fingerprint=hashlib.sha256(json.dumps([ticker,action,amount,quantity,stop_loss_pct,take_profit_pct],sort_keys=True).encode()).hexdigest()
    def replay(conn):
        old=conn.execute('SELECT * FROM trade_ledger WHERE client_order_id=?',(client_order_id,)).fetchone() if client_order_id else None
        if old:
            if old['order_fingerprint']!=fingerprint: raise ValueError('This order ID was already used for different order details.')
            return _order_receipt(dict(old),conn.execute('SELECT cash FROM wallet WHERE id=1').fetchone()[0],True)
    with database.get_connection() as conn:
        previous=replay(conn)
        if previous: return previous
        held_symbols=[r[0] for r in conn.execute('SELECT ticker FROM holdings WHERE quantity>0')]
    markets=_market_assets([t for t in held_symbols if t!=ticker])
    prices={t:m['price'] for t,m in markets.items() if m['available']}
    execution=get_execution_price(ticker)
    price=Decimal(str(execution['price']))
    if not price.is_finite() or price<=0: raise ValueError('No valid execution price.')
    qty=(Decimal(str(amount))/price if amount is not None else Decimal(str(quantity)))
    qty=qty.quantize(Decimal('0.000000000001'),rounding=ROUND_DOWN)
    total=(qty*price).quantize(Decimal('0.00000001'))
    if qty<=0 or total<=0: raise ValueError('Order is too small to execute.')
    with database.get_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        previous=replay(conn)
        if previous: return previous
        wallet=dict(conn.execute('SELECT * FROM wallet WHERE id=1').fetchone())
        cash=Decimal(str(wallet['cash']))
        existing=conn.execute('SELECT * FROM holdings WHERE ticker=?',(ticker,)).fetchone()
        old_qty=Decimal(str(existing['quantity'])) if existing else Decimal(0)
        avg=Decimal(str(existing['avg_buy_price'])) if existing else Decimal(0)
        pnl=Decimal(0)
        if action=='BUY':
            if total>cash: raise ValueError('Insufficient virtual cash for this order.')
            new_qty=old_qty+qty
            average=(old_qty*avg+total)/new_qty
            peak=max(float(existing['peak_price'] or avg) if existing else 0,float(price))
            name=config.ASSET_UNIVERSE.get(ticker,{}).get('name',ticker)
            sl=stop_loss_pct if stop_loss_pct is not None else existing['stop_loss_pct'] if existing else None
            tp=take_profit_pct if take_profit_pct is not None else existing['take_profit_pct'] if existing else None
            conn.execute('''INSERT INTO holdings(ticker,name,quantity,avg_buy_price,peak_price,stop_loss_pct,take_profit_pct)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(ticker) DO UPDATE SET quantity=excluded.quantity,
                avg_buy_price=excluded.avg_buy_price,peak_price=MAX(COALESCE(holdings.peak_price,0),excluded.peak_price),
                stop_loss_pct=excluded.stop_loss_pct,take_profit_pct=excluded.take_profit_pct''',
                (ticker,name,float(new_qty),float(average),peak,sl,tp))
            cash-=total
        else:
            if qty>old_qty: raise ValueError('Insufficient shares or tokens for this sale.')
            remainder=old_qty-qty
            pnl=total-qty*avg
            if remainder<=Decimal('0.000000000001'):
                conn.execute('DELETE FROM holdings WHERE ticker=?',(ticker,))
            else:
                conn.execute('UPDATE holdings SET quantity=? WHERE ticker=?',(float(remainder),ticker))
            cash+=total
        conn.execute('UPDATE wallet SET cash=? WHERE id=1',(float(cash),))
        ledger_action=reason if action=='SELL' and reason in ('STOP_LOSS','TAKE_PROFIT') else action
        now=datetime.now(timezone.utc).isoformat()
        cursor=conn.execute('''INSERT INTO trade_ledger(timestamp,ticker,action,quantity,price,total_value,realized_pnl,
                            execution_source,execution_label,quote_timestamp,client_order_id,order_fingerprint)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
                            (now,ticker,ledger_action,float(qty),float(price),float(total),float(pnl),
                             execution['execution_source'],execution['execution_label'],execution['quote_timestamp'],client_order_id,fingerprint))
        prices[ticker]=float(price)
        _capitex_snapshot(conn,prices,'trade')
        receipt=dict(conn.execute('SELECT * FROM trade_ledger WHERE id=?',(cursor.lastrowid,)).fetchone())
    # Prime display price after a fill; never leave the old cached UI quote authoritative.
    data_loader.get_live_quote.clear()
    _PRICE_CACHE.pop(ticker,None)
    return _order_receipt(receipt,float(cash))


def reset_capitex_wallet():
    with database.get_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('DELETE FROM holdings')
        conn.execute('DELETE FROM trade_ledger')
        conn.execute('DELETE FROM portfolio_snapshots')
        conn.execute('UPDATE wallet SET cash=?,starting_cash=? WHERE id=1',(config.STARTING_CASH,config.STARTING_CASH))
        conn.execute("INSERT INTO trade_ledger(timestamp,ticker,action,quantity,price,total_value,realized_pnl,execution_source,execution_label) VALUES(?,'SYSTEM','RESET',0,0,0,0,'reset','Wallet reset')",(datetime.now(timezone.utc).isoformat(),))
        _capitex_snapshot(conn,{},'reset')
    return {'cash':config.STARTING_CASH,'message':'Your virtual wallet is ready for a fresh start.'}


def check_capitex_risk():
    summary=capitex_wallet_summary(include_watchlist=False,snapshot_reason=None)
    triggered=[]
    for holding in summary['holdings']:
        if not holding['available']: continue
        mark=holding['current_price']
        sl=holding['stop_loss_pct']; tp=holding['take_profit_pct']
        reason=None
        if sl and mark<=holding['peak_price']*(1-sl/100): reason='STOP_LOSS'
        elif tp and mark>=holding['avg_buy_price']*(1+tp/100): reason='TAKE_PROFIT'
        if reason:
            triggered.append(execute_capitex_trade(holding['ticker'],'SELL',quantity=holding['quantity'],reason=reason))
    capitex_wallet_summary(include_watchlist=False,snapshot_reason='risk')
    return triggered
