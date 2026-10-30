"""Walk-Forward Strategy Backtesting Engine & Multi-Ticker Benchmark.

Pure-Python algorithmic backtesting module (no Streamlit dependencies).
Simulates Phase 4 autonomous AI trading agent across out-of-sample test splits:
- Scaled tranche position sizing
- Dynamic trailing stop-loss (relative to peak price)
- Take-profit profit targets
- Full transaction accounting (realized P&L, win/loss stats, max drawdown)
- Direct comparison with passive Buy & Hold benchmark and alpha computation
"""

from datetime import date, datetime, timedelta
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import data_loader
import indicators
import lstm_model

logger = logging.getLogger(__name__)


def run_single_ticker_backtest(
    ticker: str,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    lookback_years: int = 5,
    window_size: int = config.DEFAULT_WINDOW_SIZE,
    epochs: int = 10,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    tranche_pct: float = config.DEFAULT_AI_POSITION_PCT,
    max_position_pct: float = config.DEFAULT_AI_MAX_POSITION_PCT,
    stop_loss_pct: float = config.DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = config.DEFAULT_TAKE_PROFIT_PCT,
    threshold_pct: float = config.DEFAULT_SIGNAL_THRESHOLD_PCT,
    starting_cash: float = config.STARTING_CASH,
) -> Dict[str, Any]:
    """Execute complete out-of-sample walk-forward backtest for a single ticker.

    Steps:
    1. Ingest historical data from yfinance REST API and clean time series.
    2. Add technical indicators (SMA, EMA, Wilder's RSI, MACD).
    3. Construct sliding windows and split into Train (80%) and Test (20%).
    4. Train or load pre-cached LSTM neural network from disk (.keras).
    5. Generate out-of-sample log-return predictions and reconstruct price levels.
    6. Evaluate forecast precision: RMSE, MAE, and Directional Accuracy %.
    7. Simulate Phase 4 AI execution logic strictly on out-of-sample test split.
    8. Calculate Strategy ROI %, Buy & Hold ROI %, Alpha %, Max Drawdown %, and Win Rate %.

    Returns:
        Structured dictionary with metrics, equity curves, and performance stats.
    """
    clean_ticker = ticker.strip().upper()
    end_dt = end_date if end_date else date.today()
    if isinstance(end_dt, str):
        end_dt = datetime.strptime(end_dt, "%Y-%m-%d").date()

    if start_date:
        start_dt = start_date if isinstance(start_date, date) else datetime.strptime(start_date, "%Y-%m-%d").date()
    else:
        start_dt = end_dt - timedelta(days=lookback_years * 365)

    logger.info(f"Backtesting {clean_ticker} from {start_dt} to {end_dt} (Epochs: {epochs})...")

    # 1. Ingest and Clean Data
    df_raw, report = data_loader.load_from_yfinance(clean_ticker, start=start_dt, end=end_dt)
    if df_raw is None or len(df_raw) < window_size + 30:
        raise ValueError(
            f"Insufficient data for {clean_ticker}: retrieved {len(df_raw) if df_raw is not None else 0} bars, "
            f"minimum required is {window_size + 30}."
        )

    # 2. Compute Technical Indicators
    df_ind = indicators.add_all_indicators(df_raw, dropna=False)

    # 3. Prepare Train/Test Split
    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        train_dates,
        test_dates,
        test_prev_actual,
        test_actual_close,
    ) = lstm_model.prepare_lstm_dataset(
        df_ind,
        window_size=window_size,
        train_split=config.DEFAULT_TRAIN_SPLIT_RATIO,
    )

    if len(X_test) < 5:
        raise ValueError(f"Insufficient test bars for {clean_ticker} ({len(X_test)} bars).")

    # 4. Train or Load Model
    model, was_cached, info = lstm_model.train_or_load_lstm(
        ticker=clean_ticker,
        start_date=start_dt,
        end_date=end_dt,
        X_train=X_train,
        y_train=y_train,
        window_size=window_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=config.DEFAULT_LEARNING_RATE,
        force_retrain=False,
    )

    # 5. Generate Out-of-Sample Predictions
    pred_scaled = model.predict(X_test, verbose=0)
    pred_log_ret = scaler.inverse_transform(pred_scaled).flatten()
    y_test_pred = test_prev_actual * np.exp(pred_log_ret)

    # 6. Forecast Precision Metrics
    eval_metrics = lstm_model.evaluate_forecast_with_baseline(
        y_actual=test_actual_close,
        y_pred=y_test_pred,
        prev_actual=test_prev_actual,
        pred_log_returns=pred_log_ret,
    )
    rmse = eval_metrics.get("rmse", 0.0)
    mae = eval_metrics.get("mae", 0.0)
    dir_acc = eval_metrics.get("directional_accuracy", 0.0)

    # 7. Generate Algorithmic Signals on Test Bars
    signals_df = lstm_model.generate_trading_signals(
        df=df_ind,
        test_dates=test_dates,
        actual_prices=test_actual_close,
        predicted_prices=y_test_pred,
        prev_actual_prices=test_prev_actual,
        pred_log_returns=pred_log_ret,
        threshold_pct=threshold_pct,
        rsi_oversold=config.DEFAULT_RSI_OVERSOLD,
        rsi_overbought=config.DEFAULT_RSI_OVERBOUGHT,
        rsi_period=config.DEFAULT_RSI_PERIOD,
    )

    # 8. Simulate Walk-Forward Strategy Execution
    cash = float(starting_cash)
    position_qty = 0.0
    avg_buy_price = 0.0
    peak_price = 0.0

    closed_trades: List[Dict[str, Any]] = []
    equity_curve: List[float] = []
    equity_dates: List[str] = []

    for i in range(len(test_dates)):
        dt_val = test_dates[i]
        dt_str = dt_val.strftime("%Y-%m-%d") if hasattr(dt_val, "strftime") else str(dt_val)
        close_p = float(test_actual_close[i])
        sig = str(signals_df.loc[dt_val, "Signal"]) if dt_val in signals_df.index else "HOLD"

        # A. Evaluate Risk Triggers if holding open position
        if position_qty > 0:
            if close_p > peak_price:
                peak_price = close_p

            # 1. Trailing Stop-Loss
            if stop_loss_pct > 0:
                stop_threshold = peak_price * (1.0 - (stop_loss_pct / 100.0))
                if close_p <= stop_threshold:
                    proceeds = position_qty * close_p
                    cost = position_qty * avg_buy_price
                    pnl = proceeds - cost
                    cash += proceeds
                    closed_trades.append({
                        "date": dt_str,
                        "action": "STOP_LOSS",
                        "qty": position_qty,
                        "exit_price": close_p,
                        "realized_pnl": pnl,
                    })
                    position_qty = 0.0
                    avg_buy_price = 0.0
                    peak_price = 0.0

            # 2. Take-Profit
            if position_qty > 0 and take_profit_pct > 0:
                target_threshold = avg_buy_price * (1.0 + (take_profit_pct / 100.0))
                if close_p >= target_threshold:
                    proceeds = position_qty * close_p
                    cost = position_qty * avg_buy_price
                    pnl = proceeds - cost
                    cash += proceeds
                    closed_trades.append({
                        "date": dt_str,
                        "action": "TAKE_PROFIT",
                        "qty": position_qty,
                        "exit_price": close_p,
                        "realized_pnl": pnl,
                    })
                    position_qty = 0.0
                    avg_buy_price = 0.0
                    peak_price = 0.0

        # B. Process Algorithmic Signal
        if sig == "BUY":
            if cash >= 10.0:
                net_worth_temp = cash + (position_qty * close_p)
                cur_pos_val = position_qty * close_p
                max_pos_val = net_worth_temp * (max_position_pct / 100.0)

                if cur_pos_val < max_pos_val:
                    # Allocate tranche
                    tranche_budget = min(cash * (tranche_pct / 100.0), max_pos_val - cur_pos_val)
                    if tranche_budget >= 10.0:
                        shares_to_buy = tranche_budget / close_p
                        new_qty = position_qty + shares_to_buy
                        new_avg = ((position_qty * avg_buy_price) + tranche_budget) / new_qty
                        position_qty = new_qty
                        avg_buy_price = new_avg
                        peak_price = max(peak_price if peak_price > 0 else close_p, close_p)
                        cash -= tranche_budget

        elif sig == "SELL":
            if position_qty > 0:
                proceeds = position_qty * close_p
                cost = position_qty * avg_buy_price
                pnl = proceeds - cost
                cash += proceeds
                closed_trades.append({
                    "date": dt_str,
                    "action": "SIGNAL_SELL",
                    "qty": position_qty,
                    "exit_price": close_p,
                    "realized_pnl": pnl,
                })
                position_qty = 0.0
                avg_buy_price = 0.0
                peak_price = 0.0

        # C. Record Daily Net Worth
        cur_nw = cash + (position_qty * close_p)
        equity_curve.append(cur_nw)
        equity_dates.append(dt_str)

    # 9. Mark-to-Market Final Valuation
    final_close = float(test_actual_close[-1])
    final_net_worth = cash + (position_qty * final_close)
    strategy_roi_pct = ((final_net_worth - starting_cash) / starting_cash) * 100.0

    # 10. Passive Buy & Hold Benchmark ROI %
    day1_close = float(test_actual_close[0])
    buy_and_hold_roi_pct = ((final_close - day1_close) / day1_close) * 100.0
    alpha_pct = strategy_roi_pct - buy_and_hold_roi_pct

    # 11. Win Rate & Trade Stats
    total_trades = len(closed_trades)
    winning_trades = sum(1 for t in closed_trades if t["realized_pnl"] > 0)
    win_rate_pct = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    # 12. Maximum Drawdown Calculation
    eq_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - peaks) / peaks * 100.0
    max_drawdown_pct = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "ticker": clean_ticker,
        "test_bars": len(test_dates),
        "test_start_date": str(test_dates[0].strftime("%Y-%m-%d") if hasattr(test_dates[0], "strftime") else test_dates[0]),
        "test_end_date": str(test_dates[-1].strftime("%Y-%m-%d") if hasattr(test_dates[-1], "strftime") else test_dates[-1]),
        "starting_cash": starting_cash,
        "final_net_worth": round(final_net_worth, 2),
        "strategy_roi_pct": round(strategy_roi_pct, 2),
        "buy_and_hold_roi_pct": round(buy_and_hold_roi_pct, 2),
        "alpha_pct": round(alpha_pct, 2),
        "win_rate_pct": round(win_rate_pct, 1),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "total_trades": total_trades,
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "directional_accuracy_pct": round(dir_acc, 1),
        "was_cached": was_cached,
        "equity_curve": equity_curve,
        "equity_dates": equity_dates,
    }


def run_bulk_backtest(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback_years: int = 5,
    window_size: int = config.DEFAULT_WINDOW_SIZE,
    epochs: int = 10,
    tranche_pct: float = config.DEFAULT_AI_POSITION_PCT,
    stop_loss_pct: float = config.DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = config.DEFAULT_TAKE_PROFIT_PCT,
    progress_callback: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Execute walk-forward backtests across a list of tickers.

    Catches per-ticker errors gracefully so one failed ticker does not abort the batch.
    Returns:
        (results_list_sorted_by_roi_descending, skipped_tickers_list)
    """
    results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    total = len(tickers)
    for idx, raw_ticker in enumerate(tickers):
        t = raw_ticker.strip().upper()
        if not t:
            continue

        try:
            res = run_single_ticker_backtest(
                ticker=t,
                start_date=start_date,
                end_date=end_date,
                lookback_years=lookback_years,
                window_size=window_size,
                epochs=epochs,
                tranche_pct=tranche_pct,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )
            results.append(res)
        except Exception as exc:
            logger.warning(f"Backtest skipped for {t}: {exc}")
            skipped.append({"ticker": t, "reason": str(exc)})

        if progress_callback:
            progress_callback(idx + 1, total, t)

    # Sort results by Strategy ROI % descending
    results.sort(key=lambda x: x["strategy_roi_pct"], reverse=True)

    # Add Rank
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    return results, skipped
