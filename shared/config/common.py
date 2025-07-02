from pathlib import Path

from pydantic_settings import BaseSettings

cur_dir = Path(__file__).parent
root_dir = cur_dir.parent.parent
dataset_dir = root_dir / "dataset"
dataset_filename = "btc_usdt_tft_dataset.csv"
dataset_filepath = dataset_dir / dataset_filename

dataset_dir.mkdir(parents=True, exist_ok=True)


class Config(BaseSettings):
    ROOT_DIR: str = str(root_dir)
    DATASET_DIR: str = str(dataset_dir)
    DATASET_FILEPATH: str = str(dataset_filepath)


config = Config()
