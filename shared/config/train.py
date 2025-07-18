from enum import Enum

from pydantic_settings import BaseSettings


class DSCols(Enum):
    TIME_IDX = "time_idx"
    TIMESTAMP = "timestamp"

    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"

    HOUR = "hour"
    MINUTE = "minute"
    DAY_OF_WEEK = "day_of_week"

    HOUR_SIN = "hour_sin"
    HOUR_COS = "hour_cos"
    DOW_SIN = "dow_sin"
    DOW_COS = "dow_cos"

    RSI = "rsi"
    MACD = "macd"

    BL_UPPER = "bl_upper"
    BL_LOWER = "bl_lower"

    SMA_20 = "sma_20"
    SMA_50 = "sma_50"
    SMA_200 = "sma_200"

    EMA_20 = "ema_20"
    EMA_50 = "ema_50"
    EMA_200 = "ema_200"

    TARGET = "target"

    ASSET = "asset"


class Config(BaseSettings):
    DATASET_SIZE: int = 2_000_000  # means 2M dataset which covers 3.8 years
    TIME_IDX_STEP_SECS: int = 60  # means 1min

    MAX_ENCODER_LENGTH: int = 60  # past 60 time steps
    MAX_PREDICTION_LENGTH: int = 12  # predict next 12 steps

    BATCH_SIZE: int = 64
    MAX_EPOCHS: int = 30
    SEED: int = 42
    TARGET_COL: str = DSCols.TARGET.value
    TARGET_COEFF: float = 1_000  # ensure most target value to be in range (-1, 1)

    MODEL_LEARNING_RATE: float = 1e-3
    MODEL_HIDDEN_SIZE: int = 64
    MODEL_ATTENTION_HEAD_SIZE: int = 4
    MODEL_DROP_OUT: float = 0.1
    MODEL_REDUCE_PLATEAU: int = 4
    MODEL_EARLY_STOP: int = 5
    MODEL_GRADIENT_CLIPPING: float = 0.1


config = Config()
