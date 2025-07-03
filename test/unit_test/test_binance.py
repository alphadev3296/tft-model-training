from datetime import datetime, timezone

import ccxt
from dateutil.relativedelta import relativedelta
from loguru import logger

from shared.services.binance import Binance


def test_api() -> None:
    symbol = "BTC/USDT"
    exchange = ccxt.binance()
    exchange.load_markets()
    limit = 10

    since_dt_str = "2022-01-01T00:00:00Z"
    since = exchange.parse8601(since_dt_str)

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1m", since=since, limit=limit)
    assert ohlcv

    logger.debug(f"ohlcv: {ohlcv}")


def test_service() -> None:
    limit = 10
    binance = Binance(
        symbol="BTC/USDT",
        timeframe="1m",
    )
    ohlcvs = binance.fetch_ohlcvs(limit=limit)
    assert ohlcvs
    assert len(ohlcvs) == limit

    history_limit = 1500
    time_delta = relativedelta(minutes=history_limit)
    utc_now = datetime.now(tz=timezone.utc)  # noqa: UP017
    start_date = (utc_now - time_delta).isoformat()
    logger.info(f"Start date: {start_date}")

    all_data = binance.fetch_historical_ohlcvs(
        from_tstamp_ms=binance.parse8601(start_date),
        to_tstamp_ms=binance.milliseconds(),
    )
    assert all_data
    assert len(all_data) == history_limit
