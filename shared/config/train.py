from pydantic_settings import BaseSettings


class Config(BaseSettings):
    DATASET_SIZE: int = 2_000_000  # means 2M dataset
    DATASET_RESOLUTION_SECS: int = 60  # means 1min


config = Config()
