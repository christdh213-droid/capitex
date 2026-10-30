"""Multi-row interactive and static technical charting engines.

Provides:
1. Plotly multi-row interactive candlestick chart (Candlesticks + SMA/EMA, RSI, MACD).
2. Matplotlib high-performance static fallback chart with shared x-axis.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config


def plot_technical_chart_plotly(
    df: pd.DataFrame,
    ticker: str,
    sma_period: int = config.DEFAULT_SMA_PERIOD,
    ema_period: int = config.DEFAULT_EMA_PERIOD,
    rsi_period: int = config.DEFAULT_RSI_PERIOD,
    rsi_oversold: float = config.DEFAULT_RSI_OVERSOLD,
    rsi_overbought: float = config.DEFAULT_RSI_OVERBOUGHT,
    macd_fast: int = config.DEFAULT_MACD_FAST,
    macd_slow: int = config.DEFAULT_MACD_SLOW,
    macd_signal: int = config.DEFAULT_MACD_SIGNAL,
) -> go.Figure:
    """Generate a 3-row synchronized interactive Plotly candlestick and indicator chart.

    Panels:
    - Row 1 (Main): Candlesticks + SMA overlay + EMA overlay.
    - Row 2: RSI oscillator with overbought and oversold threshold lines.
    - Row 3: MACD line, Signal line, and colored momentum histogram.

    Args:
        df: DataFrame containing Open, High, Low, Close, and precomputed indicator columns.
        ticker: Ticker symbol string.
        sma_period: Lookback window for SMA.
        ema_period: Lookback window for EMA.
        rsi_period: Lookback window for RSI.
        rsi_oversold: Lower RSI threshold level (e.g. 30).
        rsi_overbought: Upper RSI threshold level (e.g. 70).
        macd_fast: Fast period for MACD.
        macd_slow: Slow period for MACD.
        macd_signal: Signal period for MACD.

    Returns:
        Plotly Figure object configured with shared x-axes and dark theme.
    """
    sma_col = f"SMA_{sma_period}"
    ema_col = f"EMA_{ema_period}"
    rsi_col = f"RSI_{rsi_period}"

    # 3-row subplot with shared x-axis
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.56, 0.22, 0.22],
        subplot_titles=(
            f"<b>{ticker}</b> Price Action (Candlesticks, SMA {sma_period}, EMA {ema_period})",
            f"Relative Strength Index (RSI {rsi_period})",
            f"MACD ({macd_fast}, {macd_slow}, {macd_signal})",
        ),
    )

    # -------------------------------------------------------------
    # Row 1: Candlestick & Moving Averages
    # -------------------------------------------------------------
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlesticks",
            increasing_line_color=config.COLOR_BULLISH,
            decreasing_line_color=config.COLOR_BEARISH,
            increasing_fillcolor=config.COLOR_BULLISH,
            decreasing_fillcolor=config.COLOR_BEARISH,
        ),
        row=1,
        col=1,
    )

    if sma_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[sma_col],
                mode="lines",
                name=f"SMA ({sma_period})",
                line=dict(color=config.COLOR_SMA, width=1.8),
                hovertemplate="SMA: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if ema_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[ema_col],
                mode="lines",
                name=f"EMA ({ema_period})",
                line=dict(color=config.COLOR_EMA, width=1.8),
                hovertemplate="EMA: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # -------------------------------------------------------------
    # Row 2: Relative Strength Index (RSI)
    # -------------------------------------------------------------
    if rsi_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[rsi_col],
                mode="lines",
                name=f"RSI ({rsi_period})",
                line=dict(color=config.COLOR_RSI, width=2.0),
                hovertemplate="RSI: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Overbought reference line (70)
    fig.add_hline(
        y=rsi_overbought,
        line_dash="dash",
        line_color="rgba(239, 83, 80, 0.75)",
        line_width=1.2,
        annotation_text=f"Overbought ({int(rsi_overbought)})",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color="#ef5350",
        row=2,
        col=1,
    )

    # Oversold reference line (30)
    fig.add_hline(
        y=rsi_oversold,
        line_dash="dash",
        line_color="rgba(38, 166, 154, 0.75)",
        line_width=1.2,
        annotation_text=f"Oversold ({int(rsi_oversold)})",
        annotation_position="bottom right",
        annotation_font_size=10,
        annotation_font_color="#26a69a",
        row=2,
        col=1,
    )

    # Centerline (50)
    fig.add_hline(
        y=50.0,
        line_dash="dot",
        line_color="rgba(255, 255, 255, 0.25)",
        line_width=1.0,
        row=2,
        col=1,
    )

    # -------------------------------------------------------------
    # Row 3: MACD (Line, Signal, Histogram)
    # -------------------------------------------------------------
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                mode="lines",
                name="MACD Line",
                line=dict(color=config.COLOR_MACD_LINE, width=1.6),
                hovertemplate="MACD: %{y:.3f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                mode="lines",
                name="Signal Line",
                line=dict(color=config.COLOR_MACD_SIGNAL, width=1.6, dash="dot"),
                hovertemplate="Signal: %{y:.3f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    if "MACD_Hist" in df.columns:
        hist_colors = [
            config.COLOR_BULLISH if val >= 0 else config.COLOR_BEARISH
            for val in df["MACD_Hist"]
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_Hist"],
                name="MACD Histogram",
                marker_color=hist_colors,
                opacity=0.85,
                hovertemplate="Hist: %{y:.3f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    # MACD zero line
    fig.add_hline(
        y=0.0,
        line_color="rgba(255, 255, 255, 0.3)",
        line_width=1.0,
        row=3,
        col=1,
    )

    # -------------------------------------------------------------
    # Layout & Aesthetics
    # -------------------------------------------------------------
    fig.update_layout(
        template=config.CHART_THEME,
        height=740,
        margin=dict(l=50, r=50, t=50, b=40),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0, 0, 0, 0.4)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
        ),
    )

    # Format Y axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], tickvals=[0, 30, 50, 70, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig


def plot_technical_chart_matplotlib(
    df: pd.DataFrame,
    ticker: str,
    sma_period: int = config.DEFAULT_SMA_PERIOD,
    ema_period: int = config.DEFAULT_EMA_PERIOD,
    rsi_period: int = config.DEFAULT_RSI_PERIOD,
    rsi_oversold: float = config.DEFAULT_RSI_OVERSOLD,
    rsi_overbought: float = config.DEFAULT_RSI_OVERBOUGHT,
    macd_fast: int = config.DEFAULT_MACD_FAST,
    macd_slow: int = config.DEFAULT_MACD_SLOW,
    macd_signal: int = config.DEFAULT_MACD_SIGNAL,
) -> plt.Figure:
    """Generate a static 3-row candlestick and indicator chart using Matplotlib as a robust fallback.

    Args:
        df: DataFrame containing OHLCV and indicator values.
        ticker: Symbol string.

    Returns:
        Matplotlib Figure object.
    """
    sma_col = f"SMA_{sma_period}"
    ema_col = f"EMA_{ema_period}"
    rsi_col = f"RSI_{rsi_period}"

    # Use dark styling to match modern UI
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 1.2], "hspace": 0.08},
    )

    fig.patch.set_facecolor("#0e1117")
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#161b22")
        ax.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
        ax.tick_params(colors="#c9d1d9", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    dates = df.index

    # 1. Candlestick drawing on ax1
    up_mask = df["Close"] >= df["Open"]
    down_mask = ~up_mask

    # High-Low wicks
    ax1.vlines(
        dates[up_mask],
        df.loc[up_mask, "Low"],
        df.loc[up_mask, "High"],
        color=config.COLOR_BULLISH,
        linewidth=1.0,
        alpha=0.9,
    )
    ax1.vlines(
        dates[down_mask],
        df.loc[down_mask, "Low"],
        df.loc[down_mask, "High"],
        color=config.COLOR_BEARISH,
        linewidth=1.0,
        alpha=0.9,
    )

    # Open-Close bodies (using bar chart approximation or thick vlines)
    body_width = timedelta_to_float = pd.Timedelta(days=0.7)
    ax1.vlines(
        dates[up_mask],
        df.loc[up_mask, "Open"],
        df.loc[up_mask, "Close"],
        color=config.COLOR_BULLISH,
        linewidth=3.5,
        label="Bullish",
    )
    ax1.vlines(
        dates[down_mask],
        df.loc[down_mask, "Close"],
        df.loc[down_mask, "Open"],
        color=config.COLOR_BEARISH,
        linewidth=3.5,
        label="Bearish",
    )

    # Overlays
    if sma_col in df.columns:
        ax1.plot(dates, df[sma_col], color=config.COLOR_SMA, linewidth=1.5, label=f"SMA ({sma_period})")
    if ema_col in df.columns:
        ax1.plot(dates, df[ema_col], color=config.COLOR_EMA, linewidth=1.5, label=f"EMA ({ema_period})")

    ax1.set_ylabel("Price ($)", color="#c9d1d9", fontsize=10)
    ax1.set_title(f"{ticker} Technical Candlestick Analysis (Matplotlib Fallback)", color="#f0f6fc", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", fontsize=8)

    # 2. RSI on ax2
    if rsi_col in df.columns:
        ax2.plot(dates, df[rsi_col], color=config.COLOR_RSI, linewidth=1.8, label=f"RSI ({rsi_period})")
    ax2.axhline(rsi_overbought, color=config.COLOR_BEARISH, linestyle="--", linewidth=1.0, label=f"Overbought ({int(rsi_overbought)})")
    ax2.axhline(rsi_oversold, color=config.COLOR_BULLISH, linestyle="--", linewidth=1.0, label=f"Oversold ({int(rsi_oversold)})")
    ax2.axhline(50.0, color="#8b949e", linestyle=":", linewidth=0.8)
    ax2.axhspan(rsi_oversold, rsi_overbought, color=config.COLOR_RSI, alpha=0.08)
    ax2.set_ylabel("RSI", color="#c9d1d9", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([20, int(rsi_oversold), 50, int(rsi_overbought), 80])
    ax2.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", fontsize=8)

    # 3. MACD on ax3
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        ax3.plot(dates, df["MACD"], color=config.COLOR_MACD_LINE, linewidth=1.4, label="MACD")
        ax3.plot(dates, df["MACD_Signal"], color=config.COLOR_MACD_SIGNAL, linewidth=1.4, linestyle="--", label="Signal")
    if "MACD_Hist" in df.columns:
        bar_colors = [config.COLOR_BULLISH if h >= 0 else config.COLOR_BEARISH for h in df["MACD_Hist"]]
        ax3.bar(dates, df["MACD_Hist"], color=bar_colors, width=0.8, alpha=0.8, label="Hist")
    ax3.axhline(0, color="#8b949e", linewidth=0.8)
    ax3.set_ylabel("MACD", color="#c9d1d9", fontsize=10)
    ax3.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", fontsize=8)

    # Format dates on shared X axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    return fig


def plot_forecast_and_signals_plotly(
    dates: pd.DatetimeIndex,
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    signals_df: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """Generate interactive Plotly chart comparing Actual vs. LSTM Predicted prices with Buy/Sell markers.

    Features:
    - Actual price line (Blue)
    - LSTM predicted price line (Amber/Gold dashed)
    - Distinct Green Up-Triangles on test dates triggering a BUY signal
    - Distinct Red Down-Triangles on test dates triggering a SELL signal
    - Unified hover tooltip with price comparisons and trigger rationale
    """
    fig = go.Figure()

    # 1. Actual Close Price Trace
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actual_prices,
            mode="lines",
            name="Actual Close",
            line=dict(color=config.COLOR_ACTUAL, width=2.2),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Actual</b>: $%{y:.2f}<extra></extra>",
        )
    )

    # 2. Predicted Close Price Trace
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=predicted_prices,
            mode="lines",
            name="LSTM Predicted Close",
            line=dict(color=config.COLOR_PREDICTED, width=2.0, dash="dot"),
            hovertemplate="<b>Predicted</b>: $%{y:.2f}<extra></extra>",
        )
    )

    # 3. Buy Signal Markers
    buy_mask = signals_df["Signal"] == "BUY"
    if buy_mask.any():
        buy_dates = signals_df.index[buy_mask]
        buy_prices = signals_df.loc[buy_mask, "Actual_Close"]
        buy_reasons = signals_df.loc[buy_mask, "Reason"]

        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers",
                name="BUY Signal",
                marker=dict(
                    symbol="triangle-up",
                    size=13,
                    color=config.COLOR_BUY_MARKER,
                    line=dict(width=1.5, color="#FFFFFF"),
                ),
                customdata=buy_reasons,
                hovertemplate="<b>BUY SIGNAL</b><br>Price: $%{y:.2f}<br>Reason: %{customdata}<extra></extra>",
            )
        )

    # 4. Sell Signal Markers
    sell_mask = signals_df["Signal"] == "SELL"
    if sell_mask.any():
        sell_dates = signals_df.index[sell_mask]
        sell_prices = signals_df.loc[sell_mask, "Actual_Close"]
        sell_reasons = signals_df.loc[sell_mask, "Reason"]

        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode="markers",
                name="SELL Signal",
                marker=dict(
                    symbol="triangle-down",
                    size=13,
                    color=config.COLOR_SELL_MARKER,
                    line=dict(width=1.5, color="#FFFFFF"),
                ),
                customdata=sell_reasons,
                hovertemplate="<b>SELL SIGNAL</b><br>Price: $%{y:.2f}<br>Reason: %{customdata}<extra></extra>",
            )
        )

    fig.update_layout(
        template=config.CHART_THEME,
        height=540,
        title=dict(
            text=f"<b>{ticker}</b> Test Split: Actual vs. LSTM Predicted Close with Algorithmic Signals",
            font=dict(size=14, color="#f0f6fc"),
        ),
        xaxis_title="Trading Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        margin=dict(l=50, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
    )

    return fig


def plot_forecast_and_signals_matplotlib(
    dates: pd.DatetimeIndex,
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    signals_df: pd.DataFrame,
    ticker: str,
) -> plt.Figure:
    """Generate static Matplotlib chart for Actual vs. LSTM Predicted prices with Buy/Sell markers."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")
    ax.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    ax.tick_params(colors="#c9d1d9", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    # Lines
    ax.plot(dates, actual_prices, color=config.COLOR_ACTUAL, linewidth=2.0, label="Actual Close")
    ax.plot(dates, predicted_prices, color=config.COLOR_PREDICTED, linewidth=1.8, linestyle="--", label="LSTM Predicted")

    # Markers
    buy_mask = signals_df["Signal"] == "BUY"
    if buy_mask.any():
        ax.scatter(
            signals_df.index[buy_mask],
            signals_df.loc[buy_mask, "Actual_Close"],
            marker="^",
            color=config.COLOR_BUY_MARKER,
            s=120,
            edgecolors="#FFFFFF",
            linewidth=1.2,
            label="BUY Signal",
            zorder=5,
        )

    sell_mask = signals_df["Signal"] == "SELL"
    if sell_mask.any():
        ax.scatter(
            signals_df.index[sell_mask],
            signals_df.loc[sell_mask, "Actual_Close"],
            marker="v",
            color=config.COLOR_SELL_MARKER,
            s=120,
            edgecolors="#FFFFFF",
            linewidth=1.2,
            label="SELL Signal",
            zorder=5,
        )

    ax.set_title(f"{ticker} Actual vs. LSTM Forecast with Signals (Matplotlib)", color="#f0f6fc", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price ($)", color="#c9d1d9", fontsize=10)
    ax.set_xlabel("Date", color="#c9d1d9", fontsize=10)
    ax.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", fontsize=9)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    return fig


def plot_arena_net_worth_history(
    human_snapshots: list,
    ai_snapshots: list,
    starting_cash: float = config.STARTING_CASH,
) -> go.Figure:
    """Generate an interactive Plotly chart comparing Human vs. AI Net Worth progression.

    Features:
    - Cyan line for Human Net Worth
    - Purple line for AI Bot Net Worth
    - Dashed horizontal reference line for Starting Cash ($100,000)
    - Dark theme matching application design system
    """
    fig = go.Figure()

    # A single observation cannot show a trajectory. Plot each wallet only once
    # it has three snapshots, even if the other wallet already has history.
    human_snapshots = human_snapshots if len(human_snapshots) >= 3 else []
    ai_snapshots = ai_snapshots if len(ai_snapshots) >= 3 else []
    if not human_snapshots or not ai_snapshots:
        missing = 'Not enough history yet' if not human_snapshots and not ai_snapshots else (
            'Human: Not enough history yet' if not human_snapshots else 'AI: Not enough history yet')
        fig.add_annotation(text=missing, x=.5, y=.5, xref='paper', yref='paper',
                           showarrow=False, font=dict(color='#AB9FF2', size=15))
    values = [float(row['net_worth']) for row in human_snapshots + ai_snapshots]
    pad = max(abs(starting_cash) * .005, 1.0)
    low = min(values + [starting_cash])
    high = max(values + [starting_cash])
    extra = max((high-low) * .05, pad)
    axis_range = [min(low-extra, starting_cash-pad), max(high+extra, starting_cash+pad)]

    # 1. Starting Cash Reference Line
    fig.add_hline(
        y=starting_cash,
        line_dash="dash",
        line_color="#8b949e",
        line_width=1.5,
        annotation_text=f"Starting Cash (${starting_cash:,.0f})",
        annotation_position="bottom right",
        annotation_font=dict(color="#8b949e", size=10),
    )

    # 2. Human Snapshots
    if human_snapshots:
        h_df = pd.DataFrame(human_snapshots)
        h_df["dt"] = pd.to_datetime(h_df["timestamp"], utc=True, format="mixed").dt.tz_convert("Asia/Kolkata")
        h_df = h_df.sort_values("dt")
        fig.add_trace(
            go.Scatter(
                x=h_df["dt"],
                y=h_df["net_worth"],
                mode="lines+markers",
                name="👤 Human Portfolio",
                line=dict(color="#38bdf8", width=2.5),
                marker=dict(size=6, color="#38bdf8"),
                hovertemplate="<b>Human Net Worth</b>: $%{y:,.2f}<br>Time: %{x}<extra></extra>",
            )
        )

    # 3. AI Bot Snapshots
    if ai_snapshots:
        ai_df = pd.DataFrame(ai_snapshots)
        ai_df["dt"] = pd.to_datetime(ai_df["timestamp"], utc=True, format="mixed").dt.tz_convert("Asia/Kolkata")
        ai_df = ai_df.sort_values("dt")
        fig.add_trace(
            go.Scatter(
                x=ai_df["dt"],
                y=ai_df["net_worth"],
                mode="lines+markers",
                name="🤖 AI Bot Portfolio",
                line=dict(color="#a855f7", width=2.5),
                marker=dict(size=6, color="#a855f7"),
                hovertemplate="<b>AI Net Worth</b>: $%{y:,.2f}<br>Time: %{x}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="🏆 Net Worth Trajectory: Human vs. AI Bot",
            font=dict(size=14, color="#f0f6fc", family="sans-serif"),
            x=0.01,
            y=0.97,
        ),
        paper_bgcolor=getattr(config, "PLOTLY_PAPER_BG", "rgba(0, 0, 0, 0)"),
        plot_bgcolor=getattr(config, "PLOTLY_PLOT_BG", "rgba(0, 0, 0, 0)"),
        font=dict(color=getattr(config, "PLOTLY_FONT_COLOR", "#c9d1d9"), size=11),
        margin=dict(l=55, r=30, t=45, b=45),
        height=380,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=11, color="#c9d1d9"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=getattr(config, "PLOTLY_GRID_COLOR", "rgba(255, 255, 255, 0.08)"),
            linecolor=getattr(config, "PLOTLY_LINE_COLOR", "#30363d"),
            title="Timestamp",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=getattr(config, "PLOTLY_GRID_COLOR", "rgba(255, 255, 255, 0.08)"),
            linecolor=getattr(config, "PLOTLY_LINE_COLOR", "#30363d"),
            tickformat="$,.2f",
            range=axis_range,
            title="Portfolio Net Worth ($)",
        ),
    )

    return fig


def plot_backtest_leaderboard_chart(results: List[Dict[str, Any]]) -> go.Figure:
    """Grouped bar chart comparing AI Strategy ROI % vs Buy & Hold Benchmark ROI % per ticker."""
    if not results:
        return go.Figure()

    df = pd.DataFrame(results)
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["ticker"],
            y=df["strategy_roi_pct"],
            name="🤖 AI Bot Strategy ROI %",
            marker_color="#00E676",
            hovertemplate="<b>%{x} Strategy ROI</b>: %{y:+.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["ticker"],
            y=df["buy_and_hold_roi_pct"],
            name="📈 Buy & Hold Benchmark ROI %",
            marker_color="#29B6F6",
            hovertemplate="<b>%{x} B&H ROI</b>: %{y:+.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="🏆 AI Strategy vs. Passive Buy & Hold Benchmark ROI (%)",
            font=dict(size=14, color="#f0f6fc"),
            x=0.01,
            y=0.97,
        ),
        barmode="group",
        paper_bgcolor=getattr(config, "PLOTLY_PAPER_BG", "rgba(0, 0, 0, 0)"),
        plot_bgcolor=getattr(config, "PLOTLY_PLOT_BG", "rgba(0, 0, 0, 0)"),
        font=dict(color=getattr(config, "PLOTLY_FONT_COLOR", "#c9d1d9"), size=11),
        margin=dict(l=55, r=30, t=45, b=45),
        height=380,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=11, color="#c9d1d9"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=getattr(config, "PLOTLY_GRID_COLOR", "rgba(255, 255, 255, 0.08)"),
            linecolor=getattr(config, "PLOTLY_LINE_COLOR", "#30363d"),
            title="Ticker Symbol",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=getattr(config, "PLOTLY_GRID_COLOR", "rgba(255, 255, 255, 0.08)"),
            linecolor=getattr(config, "PLOTLY_LINE_COLOR", "#30363d"),
            ticksuffix="%",
            title="Total Return (ROI %)",
        ),
    )
    return fig




def plot_asset_history(payload: dict, timeframe: str = '90D') -> go.Figure:
    """Clean wallet price chart with exact requested lookback and latest quote."""
    frame=pd.DataFrame(payload['history'])
    frame['time']=pd.to_datetime(frame.timestamp,utc=True)
    span={'24H':1,'7D':7,'30D':30,'90D':90}[timeframe]
    now=pd.Timestamp.now(tz='UTC')
    selected=pd.Timestamp(payload['historical_timestamp'])
    shown=frame.loc[(frame.time>=now-pd.Timedelta(days=span)) | (frame.time==selected)].copy()
    quote_time=pd.Timestamp(payload['quote']['timestamp'])
    latest=pd.DataFrame({'time':[quote_time],'close':[payload['current_price']]})
    shown=pd.concat([shown[['time','close']],latest]).sort_values('time').drop_duplicates('time',keep='last')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=shown.time,y=shown.close,mode='lines',name='Market price',
        line=dict(color='#AB9FF2',width=2.6),fill='tozeroy',fillcolor='rgba(171,159,242,.045)',
        hovertemplate='%{x|%d %b %Y, %H:%M UTC}<br>$%{y:,.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=[selected],y=[payload['historical_price']],mode='markers',
        name=f"{payload['days_ago']} days ago",marker=dict(color='#14F195',size=11,line=dict(color='#0E0E14',width=3)),
        hovertemplate='Selected observation<br>%{x|%d %b %Y, %H:%M UTC}<br>$%{y:,.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=[quote_time],y=[payload['current_price']],mode='markers',name='Latest quote',
        marker=dict(color='#AB9FF2',size=8),hovertemplate='Latest quote<br>$%{y:,.2f}<extra></extra>'))
    minimum=float(shown.close.min()); maximum=float(shown.close.max())
    pad=max((maximum-minimum)*.15,maximum*.003)
    fig.update_layout(height=350,margin=dict(l=8,r=15,t=12,b=12),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI, Arial',color='#8F8DAB',size=11),
        showlegend=False,hovermode='x unified',dragmode='pan',
        xaxis=dict(showgrid=False,zeroline=False,tickformat='%d %b',fixedrange=False),
        yaxis=dict(side='right',range=[minimum-pad,maximum+pad],showgrid=True,
                   gridcolor='rgba(255,255,255,.045)',zeroline=False,tickformat='$,.2f'))
    return fig


def plot_wallet_history(snapshots: list, starting_cash: float) -> go.Figure:
    """Single-wallet trajectory with meaningful scale and short-history handling."""
    fig=go.Figure()
    vals=[float(s['net_worth']) for s in snapshots]
    pad=max(abs(starting_cash)*.005,1)
    low=min(vals+[starting_cash]); high=max(vals+[starting_cash])
    if len(snapshots)>=3:
        fig.add_trace(go.Scatter(x=[s['timestamp'] for s in snapshots],y=vals,
                      mode='lines+markers',line=dict(color='#AB9FF2',width=2),
                      marker=dict(size=5),hovertemplate='$%{y:,.2f}<extra></extra>'))
    else:
        fig.add_annotation(text='Not enough history yet',x=.5,y=.5,xref='paper',yref='paper',
                           showarrow=False,font=dict(color='#AB9FF2',size=15))
    fig.add_hline(y=starting_cash,line_dash='dot',line_color='#4B485E')
    fig.update_layout(height=280,margin=dict(l=10,r=20,t=15,b=15),showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#8F8DAB'),
        xaxis=dict(showgrid=False),yaxis=dict(tickformat='$,.2f',range=[low-pad,high+pad],gridcolor='rgba(255,255,255,.05)'))
    return fig


def plot_prediction_corridor(payload: dict) -> go.Figure:
    """Recent price, an anchored 80% funnel, and one labeled median target."""
    import numpy as np
    corridor=payload['corridor']
    dates=pd.to_datetime(corridor['timestamps'],utc=True)
    median=np.asarray(corridor['median'],dtype=float)
    lower=np.asarray(corridor['lower'],dtype=float)
    upper=np.asarray(corridor['upper'],dtype=float)
    if not (len(dates)==len(median)==len(lower)==len(upper)) or len(dates)<2:
        raise ValueError('The forecast path is incomplete. Please refresh the outlook.')
    if not np.isfinite(np.r_[median,lower,upper]).all() or (lower<=0).any() or (lower>median).any() or (median>upper).any():
        raise ValueError('The forecast corridor is invalid. Please refresh the outlook.')
    anchor=float(payload['current_price'])
    if not np.allclose([median[0],lower[0],upper[0]],anchor):
        raise ValueError('The forecast must start at the current quote.')
    history=pd.DataFrame(payload.get('history',[]),columns=['timestamp','close'])
    history['timestamp']=pd.to_datetime(history.timestamp,utc=True)
    context={1:7,7:20,30:45}.get(len(dates)-1,45)
    history=history.loc[history.timestamp<dates[0]].tail(context)
    history=pd.concat([history,pd.DataFrame({'timestamp':[dates[0]],'close':[anchor]})])
    fig=go.Figure()
    # Nested fills create a subtle gradient; only the outer boundary is the 80% interval.
    for fraction,opacity in [(1.,.09),(.68,.065),(.34,.06)]:
        lo=median+(lower-median)*fraction
        hi=median+(upper-median)*fraction
        fig.add_trace(go.Scatter(x=dates,y=lo,mode='lines',line=dict(width=0),
            hoverinfo='skip',showlegend=False))
        fig.add_trace(go.Scatter(x=dates,y=hi,mode='lines',line=dict(width=0),
            fill='tonexty',fillcolor=f'rgba(171,159,242,{opacity})',hoverinfo='skip',showlegend=False))
    fig.add_trace(go.Scatter(x=history.timestamp,y=history.close,mode='lines',name='Market price',
        line=dict(color='#D6D0ED',width=2.5),hovertemplate='%{x|%d %b, %H:%M UTC}<br>$%{y:,.2f}<extra>Market price</extra>'))
    fig.add_trace(go.Scatter(x=dates,y=median,mode='lines',name='Median outlook',
        line=dict(color='#AB9FF2',width=2.4,dash='dot'),customdata=np.column_stack([lower,upper]),
        hovertemplate='%{x|%d %b, %H:%M UTC}<br>Median $%{y:,.2f}<br>80% range $%{customdata[0]:,.2f} – $%{customdata[1]:,.2f}<extra></extra>'))
    target=float(median[-1]); move=(target/anchor-1)*100
    fig.add_trace(go.Scatter(x=[dates[-1]],y=[target],mode='markers',name='Target glow',
        marker=dict(color='rgba(171,159,242,.16)',size=28),hoverinfo='skip',showlegend=False))
    fig.add_trace(go.Scatter(x=[dates[-1]],y=[target],mode='markers',name='Median target',
        marker=dict(color='#C4B6FF',size=11,line=dict(color='#F3EDFF',width=2)),
        hovertemplate='Target $%{y:,.2f}<extra></extra>'))
    fig.add_annotation(x=dates[-1],y=target,text=f'Target: ${target:,.2f} ({move:+.1f}%)',
        showarrow=False,xanchor='right',yshift=29,bgcolor='#292235',borderpad=8,
        font=dict(color='#EAE3FF',size=12))
    fig.add_vline(x=dates[0].timestamp()*1000,line_width=1,line_dash='dot',line_color='#464052')
    all_values=np.r_[history.close.to_numpy(dtype=float),lower,upper]
    pad=max(float(np.ptp(all_values))*.18,anchor*.005)
    fig.update_layout(height=390,margin=dict(l=8,r=12,t=35,b=15),showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI, Arial',color='#8F8DAB',size=11),hovermode='x unified',
        xaxis=dict(showgrid=False,zeroline=False,tickformat='%d %b',fixedrange=False),
        yaxis=dict(side='right',range=[float(all_values.min())-pad,float(all_values.max())+pad],
            showgrid=True,gridcolor='rgba(255,255,255,.045)',zeroline=False,tickformat='$,.2f'))
    return fig


def plot_academic_chart(records: list, ticker: str) -> go.Figure:
    import academic
    frame=academic.records_frame(records)
    fig=plot_technical_chart_plotly(frame,ticker,sma_period=20,ema_period=12)
    for name,color in [('SMA_50','#E6B96B'),('EMA_26','#14F195')]:
        fig.add_trace(go.Scatter(x=frame.index,y=frame[name],name=name.replace('_',' '),
            mode='lines',line=dict(color=color,width=1.4),
            hovertemplate=name+' $%{y:,.2f}<extra></extra>'),row=1,col=1)
    fig.update_layout(paper_bgcolor='#0E0E14',plot_bgcolor='#0E0E14',height=650,
        margin=dict(l=40,r=25,t=85,b=30),legend=dict(orientation='h',y=1.1,x=0,xanchor='left'))
    return fig
