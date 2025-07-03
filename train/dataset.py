from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ta  # For technical indicators
from dateutil.relativedelta import relativedelta
from loguru import logger

from shared.config.common import config as cfg_common
from shared.config.train import config as cfg_train
from shared.services.binance import Binance


class Dataset:
    @classmethod
    def generate_binance_dataset(cls, dataset_size: int, filepath: str | None = None) -> pd.DataFrame:
        # --- Init ---
        binance = Binance()
        time_delta = relativedelta(minutes=dataset_size + 100)

        # --- Step 1: Download OHLCV data in chunks ---
        logger.info("Downloading data from Binance...")

        utc_now = datetime.now(tz=timezone.utc)  # noqa: UP017
        start_date = (utc_now - time_delta).isoformat()
        logger.info(f"Start date: {start_date}")

        all_data = binance.fetch_historical_ohlcvs(
            from_tstamp_ms=binance.parse8601(start_date),
            to_tstamp_ms=binance.milliseconds(),
        )

        # --- Step 2: Format into DataFrame ---
        logger.info("Processing data...")
        df = cls.convert_ohlcvs_to_dataframe(all_data)
        df = df.tail(dataset_size)

        # --- Save ---
        if filepath:
            df.to_csv(filepath)
            logger.info(f"✅ Dataset saved to: {filepath}")

        logger.info(f"Total samples: {len(df):,}")

        return df

    @classmethod
    def convert_ohlcvs_to_dataframe(cls, ohlcvs: list[list[int | float]]) -> pd.DataFrame:
        """
        Convert list of ohlcvs to DataFrame.
        The first 26+ rows will be dropped as they are incomplete in some columns.
        """
        df = pd.DataFrame(ohlcvs, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        df = df.astype(float)

        # Add time-based and cyclical features
        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["day_of_week"] = df.index.dayofweek

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Technical Indicators
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["macd"] = ta.trend.MACD(df["close"]).macd()
        df["bollinger_h"] = ta.volatility.BollingerBands(df["close"]).bollinger_hband()
        df["bollinger_l"] = ta.volatility.BollingerBands(df["close"]).bollinger_lband()
        df["sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
        df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()

        # Target column (next-minute return)
        df["target"] = df["close"].pct_change().shift(-1)

        # Clean
        df["asset"] = "BTC"
        return df.dropna()


if __name__ == "__main__":
    Dataset.generate_binance_dataset(
        dataset_size=cfg_train.DATASET_SIZE,
        filepath=cfg_common.DATASET_FILEPATH,
    )
