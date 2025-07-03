from loguru import logger

from train.dataset import Dataset


def test_dataset() -> None:
    dataset_size = 30
    df = Dataset.generate_binance_dataset(
        dataset_size=dataset_size,
        filepath=None,
    )

    assert len(df) == dataset_size

    logger.debug(f"shape: {df.shape}")
    logger.debug(f"head: {df.head()}")
    logger.debug(f"tail: {df.tail()}")
