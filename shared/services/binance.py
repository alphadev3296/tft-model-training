import time

import ccxt
from loguru import logger
from tqdm import tqdm


class Binance:
    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
    ) -> None:
        self.exchange = ccxt.binance()
        self.exchange.load_markets()

        self.symbol = symbol
        self.timeframe = timeframe

    def parse8601(self, date_str: str) -> int:
        return self.exchange.parse8601(date_str)  # type: ignore  # noqa: PGH003

    def milliseconds(self) -> int:
        return self.exchange.milliseconds()  # type: ignore  # noqa: PGH003

    def fetch_ohlcvs(
        self,
        since: int | None = None,
        limit: int = 1000,
    ) -> list[list[int | float]]:
        return self.exchange.fetch_ohlcv(  # type: ignore  # noqa: PGH003
            self.symbol,
            timeframe=self.timeframe,
            since=since,
            limit=limit,
        )

    def fetch_historical_ohlcvs(
        self,
        from_tstamp_ms: int,
        to_tstamp_ms: int,
    ) -> list[list[int | float]]:
        since = from_tstamp_ms
        resolution_ms = 60 * 1000  # 1 minute

        progress_bar = tqdm(total=(to_tstamp_ms - from_tstamp_ms) // resolution_ms)

        all_data = []
        while since < to_tstamp_ms:
            try:
                ohlcvs = self.fetch_ohlcvs(since=since)
                if not ohlcvs:
                    break
                since = ohlcvs[-1][0] + resolution_ms  # type: ignore  # noqa: PGH003
                all_data.extend(ohlcvs)

                progress_bar.update(len(ohlcvs))

                time.sleep(0.1)  # avoid rate limits
            except Exception as e:
                logger.error(e)
                time.sleep(5)

        progress_bar.close()

        return all_data
