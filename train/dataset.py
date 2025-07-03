from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ta  # For technical indicators
from dateutil.relativedelta import relativedelta
from loguru import logger

from shared.config.common import config as cfg_common
from shared.config.train import DSCols
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
        df = pd.DataFrame(
            ohlcvs,
            columns=[
                DSCols.TIMESTAMP.value,
                DSCols.OPEN.value,
                DSCols.HIGH.value,
                DSCols.LOW.value,
                DSCols.CLOSE.value,
                DSCols.VOLUME.value,
            ],
        )
        df[DSCols.TIMESTAMP.value] = pd.to_datetime(df[DSCols.TIMESTAMP.value], unit="ms")
        df = df.set_index(DSCols.TIMESTAMP.value)
        df = df.astype(float)

        # Add time-based and cyclical features
        df[DSCols.HOUR.value] = df.index.hour
        df["minute"] = df.index.minute
        df[DSCols.DAY_OF_WEEK.value] = df.index.dayofweek

        # Cyclical encoding
        df[DSCols.HOUR_SIN.value] = np.sin(2 * np.pi * df[DSCols.HOUR.value] / 24)
        df[DSCols.HOUR_COS.value] = np.cos(2 * np.pi * df[DSCols.HOUR.value] / 24)
        df[DSCols.DOW_SIN.value] = np.sin(2 * np.pi * df[DSCols.DAY_OF_WEEK.value] / 7)
        df[DSCols.DOW_COS.value] = np.cos(2 * np.pi * df[DSCols.DAY_OF_WEEK.value] / 7)

        # Technical Indicators
        df[DSCols.RSI.value] = ta.momentum.RSIIndicator(df[DSCols.CLOSE.value], window=14).rsi()
        df[DSCols.MACD.value] = ta.trend.MACD(df[DSCols.CLOSE.value]).macd()
        df[DSCols.BOLLINGER_H.value] = ta.volatility.BollingerBands(df[DSCols.CLOSE.value]).bollinger_hband()
        df[DSCols.BOLLINGER_L.value] = ta.volatility.BollingerBands(df[DSCols.CLOSE.value]).bollinger_lband()
        df[DSCols.SMA_20.value] = ta.trend.SMAIndicator(df[DSCols.CLOSE.value], window=20).sma_indicator()
        df[DSCols.EMA_20.value] = ta.trend.EMAIndicator(df[DSCols.CLOSE.value], window=20).ema_indicator()

        # Target column (next-minute return)
        df[DSCols.TARGET.value] = df[DSCols.CLOSE.value].pct_change().shift(-1) * cfg_train.TARGET_COEFF

        # Clean
        df[DSCols.ASSET.value] = "BTC"
        return df.dropna()


if __name__ == "__main__":
    Dataset.generate_binance_dataset(
        dataset_size=cfg_train.DATASET_SIZE,
        filepath=cfg_common.DATASET_FILEPATH,
    )
