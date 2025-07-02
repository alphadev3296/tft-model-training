import time
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import ta  # For technical indicators
from dateutil.relativedelta import relativedelta
from loguru import logger
from tqdm import tqdm

from shared.config.common import config as cfg_common
from shared.config.trainer import config as cfg_trainer


def main() -> None:
    # --- Init ---
    symbol = "BTC/USDT"
    exchange = ccxt.binance()
    exchange.load_markets()
    time_delta = relativedelta(minutes=cfg_trainer.DATASET_SIZE_K * 1000)
    resolution_ms = cfg_trainer.DATASET_RESOLUTION_SECS * 1000
    limit = 1000  # Max per request

    # --- Step 1: Download OHLCV data in chunks ---
    logger.info("Downloading data from Binance...")

    all_data = []

    utc_now = datetime.now(tz=timezone.utc)  # noqa: UP017
    start_date = (utc_now - time_delta).isoformat()

    since = exchange.parse8601(start_date)
    now = exchange.milliseconds()

    progress_bar = tqdm(total=(now - since) // resolution_ms)

    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1m", since=since, limit=limit)
            if not ohlcv:
                break
            since = ohlcv[-1][0] + resolution_ms
            all_data.extend(ohlcv)
            progress_bar.update(limit)
            time.sleep(0.1)  # avoid rate limits
        except Exception as e:
            logger.error(e)
            time.sleep(5)

    progress_bar.close()

    # --- Step 2: Format into DataFrame ---
    logger.info("Processing data...")

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df = df.astype(float)

    # --- Step 3: Add time-based and cyclical features ---
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["day_of_week"] = df.index.dayofweek

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # --- Step 4: Technical Indicators ---
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["bollinger_h"] = ta.volatility.BollingerBands(df["close"]).bollinger_hband()
    df["bollinger_l"] = ta.volatility.BollingerBands(df["close"]).bollinger_lband()
    df["sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()

    # --- Step 5: Target column (next-minute return) ---
    df["target"] = df["close"].pct_change().shift(-1)

    # --- Step 6: Clean ---
    df["asset"] = "BTC"
    df = df.dropna()

    # --- Save ---
    df.to_csv(cfg_common.DATASET_FILEPATH)
    logger.info(f"✅ Dataset saved to: {cfg_common.DATASET_FILEPATH}")
    logger.info(f"Total samples: {len(df):,}")


if __name__ == "__main__":
    main()
