"""Configuration settings for Stock Market Trend Analyzer and Paper-Trading Arena.

All global constants, database paths, cache durations, and default parameters
are defined here to eliminate hard-coded magic numbers.
"""

from pathlib import Path
from typing import List, Tuple

# ==========================================
# Paths & Directories
# ==========================================
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
DB_PATH: Path = BASE_DIR / "capitex.db"
LEGACY_DB_PATH: Path = BASE_DIR / "market_sim.db"
ASSETS_DIR: Path = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Database & Wallet Defaults
# ==========================================
STARTING_CASH: float = 100_000.0
WALLET_TYPE_HUMAN: str = "human"
WALLET_TYPE_AI: str = "ai"
VALID_WALLET_TYPES: Tuple[str, ...] = (WALLET_TYPE_HUMAN, WALLET_TYPE_AI)

# ==========================================
# Data Fetcher & Cache Defaults
# ==========================================
DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
DEFAULT_TICKER: str = "AAPL"
DEFAULT_LOOKBACK_YEARS: int = 5

# Cache TTLs (in seconds)
HISTORICAL_CACHE_TTL: int = 3600  # 1 hour
LIVE_PRICE_CACHE_TTL: int = 30

# Required columns for price series
REQUIRED_PRICE_COLUMNS: Tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")

# ==========================================
# Technical Indicators Defaults (Module 2)
# ==========================================
DEFAULT_SMA_PERIOD: int = 20
DEFAULT_EMA_PERIOD: int = 50
DEFAULT_RSI_PERIOD: int = 14
DEFAULT_RSI_OVERSOLD: float = 30.0
DEFAULT_RSI_OVERBOUGHT: float = 70.0
DEFAULT_MACD_FAST: int = 12
DEFAULT_MACD_SLOW: int = 26
DEFAULT_MACD_SIGNAL: int = 9

# Indicator Chart Aesthetics
CHART_THEME: str = "plotly_dark"
COLOR_BULLISH: str = "#26A69A"
COLOR_BEARISH: str = "#EF5350"
COLOR_SMA: str = "#29B6F6"
COLOR_EMA: str = "#FFA726"
COLOR_RSI: str = "#AB47BC"
COLOR_MACD_LINE: str = "#29B6F6"
COLOR_MACD_SIGNAL: str = "#FF7043"
PLOTLY_PAPER_BG: str = "rgba(0, 0, 0, 0)"
PLOTLY_PLOT_BG: str = "rgba(0, 0, 0, 0)"
PLOTLY_FONT_COLOR: str = "#c9d1d9"
PLOTLY_GRID_COLOR: str = "rgba(255, 255, 255, 0.08)"
PLOTLY_LINE_COLOR: str = "#30363d"

# ==========================================
# LSTM Forecasting Defaults (Module 3)
# ==========================================
DEFAULT_WINDOW_SIZE: int = 60
DEFAULT_EPOCHS: int = 25
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LEARNING_RATE: float = 0.001
DEFAULT_TRAIN_SPLIT_RATIO: float = 0.80
RANDOM_SEED: int = 42

# Network Architecture
LSTM_UNITS_L1: int = 50
LSTM_UNITS_L2: int = 50
LSTM_DROPOUT: float = 0.2
LSTM_DENSE_UNITS: int = 25

# Visual Aesthetics for Forecast Charts
COLOR_ACTUAL: str = "#2962FF"
COLOR_PREDICTED: str = "#FFB300"
COLOR_BUY_MARKER: str = "#00E676"
COLOR_SELL_MARKER: str = "#FF1744"

# Signal Generation
DEFAULT_SIGNAL_THRESHOLD_PCT: float = 0.05  # 0.05% (5 bps) threshold for log return forecasting

# ==========================================
# Risk Management & Trading Defaults (Module 4)
# ==========================================
DEFAULT_STOP_LOSS_PCT: float = 3.0       # 3% stop loss
DEFAULT_TAKE_PROFIT_PCT: float = 6.0     # 6% take profit
DEFAULT_AI_POSITION_PCT: float = 10.0    # AI spends 10% cash per trade
DEFAULT_AI_MAX_POSITION_PCT: float = 40.0 # Max 40% position cap
AUTO_REFRESH_INTERVAL_SECONDS: int = 30  # For st.fragment rerun

# FastAPI Server Defaults (Module 4)
FASTAPI_HOST: str = "127.0.0.1"
FASTAPI_PORT: int = 8000
FASTAPI_BASE_URL: str = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"
BACKEND_PRICE_CACHE_TTL: int = 30        # Marking cache; execution bypasses it

# ==========================================
# UI & Compliance
# ==========================================
APP_TITLE: str = "CAPITEX"
APP_ICON: str = "📈"
FOOTER_TEXT: str = "CAPITEX — Simulated trading for educational purposes only. Not financial advice."

BRAND_BACKGROUND = "#0E0E14"
BRAND_CARD = "#181820"
BRAND_PURPLE = "#AB9FF2"
BRAND_EMERALD = "#14F195"
LOGO_PATH = ASSETS_DIR / "capitex_logo.svg"
BANNER_PATH = ASSETS_DIR / "capitex_banner.svg"

# Issuer/project-hosted sources are optional; local circular badges always work.
ASSET_UNIVERSE = {
    "AAPL": {"name": "Apple Inc.", "logo_url": "https://www.apple.com/ac/structured-data/images/knowledge_graph_logo.png", "kind": "stock"},
    "NVDA": {"name": "NVIDIA", "logo_url": "https://images.nvidia.com/sdm/nvidia-logo.png", "kind": "stock"},
    "TSLA": {"name": "Tesla", "logo_url": "https://www.tesla.com/themes/custom/tesla_frontend/assets/favicons/favicon-196x196.png", "kind": "stock"},
    "MSFT": {"name": "Microsoft", "logo_url": "https://www.microsoft.com/favicon.ico", "kind": "stock"},
    "AMZN": {"name": "Amazon", "logo_url": "https://www.amazon.com/favicon.ico", "kind": "stock"},
    "GOOGL": {"name": "Alphabet", "logo_url": "https://www.google.com/images/branding/googleg/1x/googleg_standard_color_128dp.png", "kind": "stock"},
    "META": {"name": "Meta Platforms", "logo_url": "https://about.meta.com/favicon.ico", "kind": "stock"},
    "BTC-USD": {"name": "Bitcoin", "logo_url": "https://bitcoin.org/img/icons/opengraph.png", "kind": "crypto"},
    "ETH-USD": {"name": "Ethereum", "logo_url": "https://ethereum.org/favicon.ico", "kind": "crypto"},
    "SOL-USD": {"name": "Solana", "logo_url": "https://solana.com/favicon.ico", "kind": "crypto"},
}
for _ticker, _asset in ASSET_UNIVERSE.items():
    _asset["logo_fallback"] = str(ASSETS_DIR / "badges" / f"{_ticker}.svg")
DEFAULT_TICKERS = list(ASSET_UNIVERSE)

# CAPITEX Phase 2 defaults. Separate from prototype single-feature model caches.
FORECAST_HORIZONS = {"24h": 1, "1d": 1, "7d": 7, "30d": 30}
FORECAST_MAX_HORIZON = 30
FORECAST_MC_PASSES = 30
FORECAST_INTERVAL_LEVEL = 0.80
FORECAST_CALIBRATION_FRACTION = 0.20
FORECAST_CACHE_VERSION = "capitex-path-v1"
REASONING_CACHE_TTL = 900
