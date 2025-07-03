from loguru import logger

from train.dataset import Dataset


def test_dataset() -> None:
    df = Dataset.generate_binance_dataset(
        dataset_size=30,
        filepath=None,
    )
    logger.debug(f"shape: {df.shape}")
    logger.debug(f"head: {df.head()}")
    logger.debug(f"tail: {df.tail()}")
