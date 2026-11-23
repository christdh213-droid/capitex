"""FastAPI Backend Server for Quantitative Market Analyzer and Paper Trading Arena.

Provides RESTful endpoints for:
- System health checks
- Cached live quotes (60s TTL to prevent yfinance rate limits)
- LSTM neural forecast and signal generation
- Atomic trade execution (BUY / SELL)
- Autonomous AI agent trading steps (scaled tranche sizing)
- Dynamic trailing stop-loss and take-profit risk trigger evaluation
- Mark-to-market portfolio summaries, win/loss ratios, drawdowns, and ledger exports
- Global wallet reset
"""

from contextlib import asynccontextmanager
import asyncio
import os
from datetime import date, datetime, timedelta, timezone
import logging
import pandas as pd
import academic
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import backtester
import config
import database
import data_loader
import indicators
import lstm_model
import trading_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# Lifespan Context Manager
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure database schema is initialized and wallets are seeded on server startup."""
    try:
        database.init_db()
        logger.info("FastAPI: SQLite database and wallets verified.")
    except Exception as exc:
        logger.error(f"FastAPI: Error during database initialization: {exc}")
        raise
    task=None
    if os.getenv('CAPITEX_DISABLE_BACKGROUND_GRADER')!='1':
        from forecast_grader import grader_loop
        task=asyncio.create_task(grader_loop())
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:await task
            except asyncio.CancelledError:pass


# ==============================================================================
# FastAPI Application Declaration
# ==============================================================================
app = FastAPI(
    title="CAPITEX API",
    description="High-performance REST API for market predictions, risk triggers, and virtual trading.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Pydantic Request & Response Schemas
# ==============================================================================
class TradeRequest(BaseModel):
    wallet_type: str = Field(..., description="Target wallet: 'human' or 'ai'")
    ticker: str = Field(..., description="Asset symbol (e.g. AAPL, NVDA)")
    action: str = Field(..., description="Action: 'BUY' or 'SELL'")
    quantity: float = Field(..., gt=0, description="Number of shares")
    price: Optional[float] = Field(None, gt=0, description="Execution price (defaults to live price if None)")
    stop_loss_pct: Optional[float] = Field(None, ge=0, description="Stop loss percentage")
    take_profit_pct: Optional[float] = Field(None, ge=0, description="Take profit percentage")


class AiStepRequest(BaseModel):
    ticker: str = Field("AAPL", description="Asset symbol for AI agent")
    tranche_pct: Optional[float] = Field(10.0, gt=0, le=100, description="Percentage of cash per BUY tranche")
    threshold_pct: Optional[float] = Field(0.05, gt=0, description="Signal trigger threshold %")
    stop_loss_pct: Optional[float] = Field(config.DEFAULT_STOP_LOSS_PCT, ge=0)
    take_profit_pct: Optional[float] = Field(config.DEFAULT_TAKE_PROFIT_PCT, ge=0)


class PredictRequest(BaseModel):
    ticker: str = Field("AAPL", description="Asset symbol to forecast")
    lookback_years: Optional[int] = Field(5, ge=1, le=10, description="Years of historical data")
    window_size: Optional[int] = Field(config.DEFAULT_WINDOW_SIZE, ge=5, le=120)
    epochs: Optional[int] = Field(config.DEFAULT_EPOCHS, ge=5, le=100)
    threshold_pct: Optional[float] = Field(config.DEFAULT_SIGNAL_THRESHOLD_PCT, ge=0.01)


class SingleBacktestRequest(BaseModel):
    ticker: str = Field(..., description="Asset symbol (e.g. AAPL)")
    start_date: Optional[str] = Field(None, description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="End date YYYY-MM-DD")
    lookback_years: Optional[int] = Field(5, ge=1, le=10)
    window_size: Optional[int] = Field(config.DEFAULT_WINDOW_SIZE, ge=5, le=120)
    epochs: Optional[int] = Field(10, ge=1, le=100)
    tranche_pct: Optional[float] = Field(config.DEFAULT_AI_POSITION_PCT, ge=1.0, le=100.0)
    stop_loss_pct: Optional[float] = Field(config.DEFAULT_STOP_LOSS_PCT, ge=0.0)
    take_profit_pct: Optional[float] = Field(config.DEFAULT_TAKE_PROFIT_PCT, ge=0.0)


class BulkBacktestRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of ticker symbols")
    start_date: Optional[str] = Field(None, description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="End date YYYY-MM-DD")
    lookback_years: Optional[int] = Field(5, ge=1, le=10)
    window_size: Optional[int] = Field(config.DEFAULT_WINDOW_SIZE, ge=5, le=120)
    epochs: Optional[int] = Field(10, ge=1, le=100)
    tranche_pct: Optional[float] = Field(config.DEFAULT_AI_POSITION_PCT, ge=1.0, le=100.0)
    stop_loss_pct: Optional[float] = Field(config.DEFAULT_STOP_LOSS_PCT, ge=0.0)
    take_profit_pct: Optional[float] = Field(config.DEFAULT_TAKE_PROFIT_PCT, ge=0.0)



# ==============================================================================
# REST Endpoints
# ==============================================================================
@app.get("/api/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Check backend operational health and database connection status."""
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        wallets = database.get_all_wallets()
        core_wallet = database.get_capitex_wallet()
        if core_wallet is None:
            raise RuntimeError("CAPITEX wallet has not been initialized")
        return {
            "status": "ok",
            "timestamp": now_iso,
            "database": "connected",
            "wallets_count": len(wallets),
            "capitex_database": "connected",
            "version": "1.0.0",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database check failed: {exc}")


@app.get("/api/price/{ticker}", tags=["Market Data"])
def get_price(ticker: str) -> Dict[str, Any]:
    """Retrieve real-time market price with 60-second in-memory caching to avoid rate-limiting."""
    clean_ticker = ticker.strip().upper()
    price = trading_engine.get_cached_live_price(clean_ticker)
    if price <= 0:
        quote = data_loader.get_live_price(clean_ticker)
        price = float(quote.get("price", 0.0))

    return {
        "ticker": clean_ticker,
        "price": price,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cached": True,
    }


@app.post("/api/predict", tags=["Forecasting"])
def predict_asset(req: PredictRequest) -> Dict[str, Any]:
    """Run data ingestion, technical indicators, and cached/trained LSTM forecast."""
    clean_ticker = req.ticker.strip().upper()
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=req.lookback_years * 365)

    try:
        df, report = data_loader.load_from_yfinance(clean_ticker, start=start_dt, end=end_dt)
        df_ind = indicators.add_all_indicators(df)

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
        ) = lstm_model.prepare_lstm_dataset(df_ind, window_size=req.window_size)

        model, was_cached, info = lstm_model.train_or_load_lstm(
            ticker=clean_ticker,
            start_date=start_dt,
            end_date=end_dt,
            X_train=X_train,
            y_train=y_train,
            window_size=req.window_size,
            epochs=req.epochs,
            batch_size=config.DEFAULT_BATCH_SIZE,
            learning_rate=config.DEFAULT_LEARNING_RATE,
            force_retrain=False,
        )

        next_day = lstm_model.predict_next_day(
            model=model,
            df=df_ind,
            scaler=scaler,
            window_size=req.window_size,
            threshold_pct=req.threshold_pct,
            rsi_oversold=config.DEFAULT_RSI_OVERSOLD,
            rsi_overbought=config.DEFAULT_RSI_OVERBOUGHT,
            rsi_period=config.DEFAULT_RSI_PERIOD,
        )

        return {
            "ticker": clean_ticker,
            "was_cached": was_cached,
            "forecast": next_day,
            "current_close": next_day["current_close"],
            "predicted_next_close": next_day["predicted_next_close"],
            "predicted_return_pct": next_day["predicted_return_pct"],
            "signal": next_day["signal"],
            "reason": next_day["reason"],
        }
    except Exception as exc:
        logger.exception(f"Prediction failed for {clean_ticker}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/trade/execute", tags=["Trading"])
def execute_trade(req: TradeRequest) -> Dict[str, Any]:
    """Execute paper BUY or SELL order inside an atomic database transaction."""
    w_type = req.wallet_type.strip().lower()
    if w_type not in config.VALID_WALLET_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid wallet_type '{w_type}'. Must be one of {config.VALID_WALLET_TYPES}.",
        )

    clean_ticker = req.ticker.strip().upper()
    # req.price is a legacy display estimate, never an execution authority.
    act_upper = req.action.strip().upper()
    if act_upper == "BUY":
        success, msg, data = trading_engine.execute_buy(
            wallet_type=w_type,
            ticker=clean_ticker,
            quantity=req.quantity,
            stop_loss_pct=req.stop_loss_pct,
            take_profit_pct=req.take_profit_pct,
        )
    elif act_upper == "SELL":
        success, msg, data = trading_engine.execute_sell(
            wallet_type=w_type,
            ticker=clean_ticker,
            quantity=req.quantity,
            reason="MANUAL_SELL",
        )
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action '{act_upper}'. Must be 'BUY' or 'SELL'.")

    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {"success": True, "message": msg, "trade": data}


@app.post("/trade/ai-step", tags=["Trading"])
def ai_trade_step(req: AiStepRequest) -> Dict[str, Any]:
    """Evaluate Phase 3 forecast signal and execute scaled tranche trade for the AI bot."""
    clean_ticker = req.ticker.strip().upper()

    # 1. Fetch signal via prediction helper
    pred_res = predict_asset(
        PredictRequest(
            ticker=clean_ticker,
            threshold_pct=req.threshold_pct,
        )
    )
    signal = pred_res["signal"]
    pred_ret = pred_res["predicted_return_pct"]
    signal_reference_price = float(pred_res["current_close"])

    # 2. Run AI trade execution
    success, msg, trade_data = trading_engine.run_ai_trading_step(
        ticker=clean_ticker,
        signal=signal,
        predicted_return_pct=pred_ret,
        current_price=signal_reference_price,
        tranche_pct=req.tranche_pct or config.DEFAULT_AI_POSITION_PCT,
        stop_loss_pct=req.stop_loss_pct or config.DEFAULT_STOP_LOSS_PCT,
        take_profit_pct=req.take_profit_pct or config.DEFAULT_TAKE_PROFIT_PCT,
    )

    return {
        "success": success,
        "ticker": clean_ticker,
        "signal": signal,
        "predicted_return_pct": pred_ret,
        "price": trade_data.get("price"),
        "signal_reference_price": signal_reference_price,
        "message": msg,
        "trade": trade_data,
    }


@app.post("/trade/check-risk/{wallet_type}", tags=["Risk Management"])
def check_risk(wallet_type: str) -> Dict[str, Any]:
    """Evaluate dynamic trailing stop-loss and take-profit triggers for a wallet."""
    w_type = wallet_type.strip().lower()
    if w_type not in config.VALID_WALLET_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid wallet_type '{w_type}'.")

    triggered = trading_engine.check_risk_triggers(w_type)
    return {
        "wallet_type": w_type,
        "triggered_count": len(triggered),
        "triggered_trades": triggered,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/portfolio/{wallet_type}", tags=["Portfolio"])
def get_portfolio(wallet_type: str) -> Dict[str, Any]:
    """Retrieve complete mark-to-market portfolio analytics, drawdowns, and ledger."""
    w_type = wallet_type.strip().lower()
    if w_type not in config.VALID_WALLET_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid wallet_type '{w_type}'.")

    summary = trading_engine.get_portfolio_summary(w_type)
    ledger = database.get_trade_ledger(w_type, limit=50)
    summary["trade_ledger"] = ledger

    return summary


@app.post("/portfolio/reset", tags=["Portfolio"])
def reset_portfolio() -> Dict[str, Any]:
    """Reset both Human and AI wallets back to $100,000 and clear holdings/trades."""
    try:
        database.reset_wallets(starting_cash=config.STARTING_CASH)
        return {
            "success": True,
            "message": f"Successfully reset both Human and AI wallets to ${config.STARTING_CASH:,.2f}.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Wallet reset failed: {exc}")


@app.post("/api/backtest/ticker", tags=["Backtesting"])
def backtest_ticker(req: SingleBacktestRequest) -> Dict[str, Any]:
    """Execute walk-forward out-of-sample backtest for a single asset."""
    try:
        res = backtester.run_single_ticker_backtest(
            ticker=req.ticker,
            start_date=req.start_date,
            end_date=req.end_date,
            lookback_years=req.lookback_years or 5,
            window_size=req.window_size or config.DEFAULT_WINDOW_SIZE,
            epochs=req.epochs or 10,
            tranche_pct=req.tranche_pct or config.DEFAULT_AI_POSITION_PCT,
            stop_loss_pct=req.stop_loss_pct or config.DEFAULT_STOP_LOSS_PCT,
            take_profit_pct=req.take_profit_pct or config.DEFAULT_TAKE_PROFIT_PCT,
        )
        return {"success": True, "result": res}
    except Exception as exc:
        logger.exception(f"Backtest failed for {req.ticker}")
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/backtest/run", tags=["Backtesting"])
def backtest_bulk(req: BulkBacktestRequest) -> Dict[str, Any]:
    """Execute walk-forward out-of-sample backtest across a batch of tickers."""
    try:
        results, skipped = backtester.run_bulk_backtest(
            tickers=req.tickers,
            start_date=req.start_date,
            end_date=req.end_date,
            lookback_years=req.lookback_years or 5,
            window_size=req.window_size or config.DEFAULT_WINDOW_SIZE,
            epochs=req.epochs or 10,
            tranche_pct=req.tranche_pct or config.DEFAULT_AI_POSITION_PCT,
            stop_loss_pct=req.stop_loss_pct or config.DEFAULT_STOP_LOSS_PCT,
            take_profit_pct=req.take_profit_pct or config.DEFAULT_TAKE_PROFIT_PCT,
        )
        return {
            "success": True,
            "total_requested": len(req.tickers),
            "successful_count": len(results),
            "skipped_count": len(skipped),
            "results": results,
            "skipped_tickers": skipped,
        }
    except Exception as exc:
        logger.exception("Bulk backtest error")
        raise HTTPException(status_code=500, detail=str(exc))


# CAPITEX v3 routes. Legacy prototype endpoints above remain backward compatible.
from typing import Literal
from pydantic import ConfigDict, StrictInt, field_validator, model_validator
from fastapi import Query
from threading import RLock
from time import monotonic
import reasoning_engine
import forecast_log
import market_context
from time_utils import utc_z
from market_calendar import completed_history
from frozen_forecast import load_frozen


class ThesisFields(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    forecast_log_ids:Optional[List[StrictInt]|Dict[str,StrictInt]]=None
    selected_horizon:Optional[StrictInt]=None
    context_summary_id:Optional[StrictInt]=None
    prepared_at:Optional[str]=None


class CapitexOrder(ThesisFields):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    ticker:str=Field(min_length=1,max_length=25)
    action:Literal['BUY','SELL','SHORT_OPEN','SHORT_COVER']
    amount:Optional[float]=Field(None,gt=0,le=1e12)
    quantity:Optional[float]=Field(None,gt=0,le=1e12)
    client_order_id:Optional[str]=Field(None,min_length=8,max_length=100)
    stop_loss_pct:Optional[float]=Field(None,gt=0,le=100)
    take_profit_pct:Optional[float]=Field(None,gt=0,le=100)

    @field_validator('ticker')
    @classmethod
    def clean_ticker(cls,value): return data_loader._symbol(value)

    @field_validator('action',mode='before')
    @classmethod
    def clean_action(cls,value): return str(value).strip().upper()

    @model_validator(mode='after')
    def exactly_one_size(self):
        if (self.amount is None)==(self.quantity is None):
            raise ValueError('Provide a dollar amount or a quantity, not both.')
        return self


@app.get('/api/wallet',tags=['CAPITEX Wallet'])
def capitex_wallet():
    return trading_engine.capitex_wallet_summary()


@app.get('/api/asset/{ticker}',tags=['CAPITEX Markets'])
def capitex_asset(ticker:str,days_ago:int=Query(44,ge=1,le=90)):
    try:
        ticker=data_loader._symbol(ticker)
        result=data_loader.get_90d_history_and_lookback(ticker,days_ago)
        summary=trading_engine.capitex_wallet_summary(include_watchlist=False)
        result.update(trading_engine.asset_metadata(ticker))
        result['holding']=next((h for h in summary['holdings'] if h['ticker']==ticker),None)
        result['short_position']=next((h for h in summary['short_positions'] if h['symbol']==ticker),None)
        result['cash']=summary['cash']
        return result
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc
    except data_loader.MarketDataError as exc:
        raise HTTPException(503,'Market history is temporarily unavailable. Please try again.') from exc


@app.post('/api/trade',tags=['CAPITEX Wallet'])
def capitex_trade(order:CapitexOrder):
    try:
        trade=trading_engine.execute_capitex_trade(**order.model_dump())
        return {'success':True,'trade':trade,
                'message':f"{trade['action']} filled at ${trade['price']:,.2f}. {trade['execution_label']}"}
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc
    except data_loader.MarketDataError as exc:
        raise HTTPException(503,'A valid execution quote is unavailable. No order was placed.') from exc


@app.post('/api/wallet/reset',tags=['CAPITEX Wallet'])
def capitex_reset():
    return trading_engine.reset_capitex_wallet()


@app.post('/api/wallet/risk-check',tags=['CAPITEX Wallet'])
def capitex_risk():
    events=trading_engine.check_capitex_risk()
    return {'triggered_count':len(events),'trades':events}


_BUNDLE_CACHE={}
_BUNDLE_LOCK=RLock()


def get_capitex_forecast_bundle(ticker):
    # Lock prevents duplicate expensive training within this backend process.
    with _BUNDLE_LOCK:
        cached=_BUNDLE_CACHE.get(ticker)
        if cached and monotonic()-cached[0]<config.HISTORICAL_CACHE_TTL:
            return cached[1],cached[2]
        history=data_loader.load_market_data(ticker,'5y')
        history=completed_history(ticker,history)
        bundle=load_frozen(history,ticker)
        if len(_BUNDLE_CACHE)>=8: _BUNDLE_CACHE.pop(next(iter(_BUNDLE_CACHE)))
        _BUNDLE_CACHE[ticker]=(monotonic(),bundle,history)
        return bundle,history


@app.get('/api/predict/{ticker}',tags=['CAPITEX Forecasting'])
def capitex_predict(ticker:str,horizon:str='7d'):
    try:
        ticker=data_loader._symbol(ticker)
        steps=lstm_model.parse_horizon(horizon)
        bundle,history=get_capitex_forecast_bundle(ticker)
        rows,history=forecast_log.log_forecasts(ticker,bundle,history,lstm_model.forecast_multihorizon)
        row=rows[str(steps)]
        forecast=__import__('json').loads(row['payload_json'])
        # Display the persisted forecast, never a newly sampled unlogged path.
        quote=dict(ticker=ticker,price=row['starting_price'],timestamp=row['data_timestamp'],
                   source='Completed unadjusted daily close',is_fallback=False,is_stale=False)
        recent=[{'timestamp':utc_z(stamp),'close':float(r.Close)}
                for stamp,r in history.tail(60).iterrows() if pd.notna(r.Close)]
        articles=market_context.get_articles(ticker)
        news=[{'content':{'title':a['title'],'pubDate':a['published_at']}} for a in articles]
        why=reasoning_engine.build_reasoning(ticker,history,news=news,calendar={},fetch_context=False)
        context=market_context.get_context(ticker,signals=why.get('technical_summary',{}).get('tags',[]),technical=why.get('technical_summary',{}))
        why['market_context']=context
        why.setdefault('sentiment',{})['headlines_used']=context['headline_count']
        return dict(forecast,hero={k:forecast[k] for k in
                    ('outlook','expected_change_pct','confidence','median_target_price','horizon_label')},
                    why=why,quote=quote,history=recent,forecast_as_of=row['data_timestamp'],context_summary_id=context['context_summary_id'],
                    forecast_log_ids={h:r['id'] for h,r in rows.items()},
                    logged_forecasts={h:{k:v for k,v in r.items() if k!='payload_json'} for h,r in rows.items()})
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc
    except data_loader.MarketDataError as exc:
        raise HTTPException(503,'Forecast data is temporarily unavailable.') from exc


@app.get('/api/inspect/{ticker}',tags=['CAPITEX Academic'])
def inspect_market(ticker:str):
    try:
        ticker=data_loader._symbol(ticker)
        history=data_loader.load_market_data(ticker,'5y')
        return {'ticker':ticker,'history':academic.technical_records(history)}
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc
    except data_loader.MarketDataError as exc:
        raise HTTPException(503,'Technical history is temporarily unavailable.') from exc


class CsvInspection(BaseModel):
    model_config=ConfigDict(extra='forbid')
    csv_text:str=Field(min_length=1,max_length=2_000_000)
    horizon:Literal['24h','7d','30d']='7d'


@app.post('/api/inspect/csv',tags=['CAPITEX Academic'])
def inspect_csv(request:CsvInspection):
    import hashlib
    import io
    try:
        if len(request.csv_text.encode('utf-8'))>2_000_000:
            raise ValueError('CSV file must be no larger than 2 MB.')
        history,report=data_loader.load_from_csv(io.StringIO(request.csv_text))
        history=indicators.validate_daily_prices(history)
        if len(history)<400:
            raise ValueError('Use at least 400 cleaned daily rows for reliable train, calibration and test splits.')
        if len(history)>10000:
            raise ValueError('Use at most 10,000 daily rows for a CSV inspection.')
        # A content-based identity isolates uploaded datasets from ticker model caches.
        identity='CSV_'+hashlib.sha256(request.csv_text.encode()).hexdigest()[:16]
        with _BUNDLE_LOCK:
            bundle=lstm_model.train_multihorizon_model(history,identity)
            forecast=lstm_model.forecast_multihorizon(bundle,request.horizon,ticker=identity)
        forecast['history']=[{'timestamp':pd.Timestamp(stamp).isoformat(),'close':float(row.Close)}
            for stamp,row in history.tail(60).iterrows()]
        return {'forecast':forecast,'technical_history':academic.technical_records(history),
                'cleaning_report':report,'rows':len(history),'source':'uploaded_csv_last_close'}
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc


@app.get('/api/wallet/ledger',tags=['CAPITEX Academic'])
def capitex_ledger_export():
    # Export every record, independent of the wallet screen's 200-row preview.
    with database.get_connection() as conn:
        rows=[dict(row) for row in conn.execute('SELECT * FROM trade_ledger ORDER BY id DESC')]
    return {'rows':rows}


@app.get('/api/report-card',tags=['CAPITEX Forecasting'])
def capitex_report_card(symbol:Optional[str]=None,asset_type:Optional[Literal['stock','crypto']]=None,
                        horizon:Optional[Literal[1,7,30]]=None):
    import report_card
    try:
        return report_card.get_report(data_loader._symbol(symbol) if symbol else None,asset_type,horizon)
    except ValueError as exc:raise HTTPException(400,str(exc)) from exc


class ShortOrder(ThesisFields):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    symbol:str=Field(min_length=1,max_length=25)
    quantity:float=Field(gt=0,le=1e12)
    client_order_id:Optional[str]=Field(None,min_length=8,max_length=100)


@app.post('/api/short/open',tags=['CAPITEX Wallet'])
def capitex_short_open(order:ShortOrder):
    values=order.model_dump();values['ticker']=values.pop('symbol')
    return capitex_trade(CapitexOrder(action='SHORT_OPEN',**values))


@app.post('/api/short/cover',tags=['CAPITEX Wallet'])
def capitex_short_cover(order:ShortOrder):
    values=order.model_dump();values['ticker']=values.pop('symbol')
    return capitex_trade(CapitexOrder(action='SHORT_COVER',**values))


class ShortPreview(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    symbol:str=Field(min_length=1,max_length=25)
    quantity:float=Field(gt=0,le=1e12)
    entry_price:float=Field(gt=0,le=1e12)
    pct_move:float=Field(ge=-30,le=120)


@app.post('/api/short-preview',tags=['CAPITEX Wallet'])
def capitex_short_preview(order:ShortPreview):
    from short_preview import preview
    try:return preview(**order.model_dump())
    except ValueError as exc:raise HTTPException(400,str(exc)) from exc
