from pathlib import Path

from loguru import logger
from pydantic_settings import BaseSettings

cur_dir = Path(__file__).parent

root_dir = cur_dir.parent.parent

dataset_dir = root_dir / "dataset"
dataset_filename = "btc_usdt_tft_dataset_1k.csv"
dataset_filepath = dataset_dir / dataset_filename

model_dir = root_dir / "model"
model_filename = "tft"
model_filepath = model_dir / model_filename

dataset_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)


class Config(BaseSettings):
    ROOT_DIR: str = str(root_dir)

    DATASET_DIR: str = str(dataset_dir)
    DATASET_FILEPATH: str = str(dataset_filepath)

    MODEL_DIR: str = str(model_dir)
    MODEL_FILENAME: str = model_filename
    MODEL_FILEPATH: str = str(model_filepath)

    CHECKPOINT_FILENAME: str = "checkpoint"


config = Config()

if __name__ == "__main__":
    logger.info(config.model_dump_json(indent=2))
