import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from shared.config.common import config as cfg_common
from shared.config.train import config as cfg_train


class Train:
    @classmethod
    def train(cls, dataset_filepath: str, model_filepath: str) -> None:
        # Set seed for reproducibility
        seed_everything(cfg_train.SEED)

        # Load data
        logger.info("Loading data...")
        df = pd.read_csv(dataset_filepath, parse_dates=["timestamp"])
        df = df.sort_values(["asset", "timestamp"])

        # Create time index (relative to earliest timestamp)
        logger.info("Creating time index...")
        df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // cfg_train.TIME_IDX_STEP_SECONDS
        df["time_idx"] = df["time_idx"].astype(int)

        # Normalize continuous features (except sin/cos or target)
        logger.info("Normalizing continuous features...")
        features_to_normalize = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "macd",
            "bollinger_h",
            "bollinger_l",
            "sma_20",
            "ema_20",
        ]
        for col in features_to_normalize:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

        # Convert categoricals to string type for proper encoding
        logger.info("Converting categoricals to string type...")
        categorical_cols = ["hour", "minute", "day_of_week", "asset"]
        for col in categorical_cols:
            df[col] = df[col].astype(str)

        # Fill missing values
        df = df.fillna(method="ffill").fillna(method="bfill")

        # Train/val split by time
        logger.info("Creating train/val split...")
        last_train_time = df["time_idx"].max() - cfg_train.MAX_PREDICTION_LENGTH
        train_df = df[df["time_idx"] <= last_train_time]
        val_df = df[df["time_idx"] > last_train_time - cfg_train.MAX_ENCODER_LENGTH]

        # Define TimeSeriesDataSet
        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=cfg_train.TARGET_COL,
            group_ids=["asset"],
            max_encoder_length=cfg_train.MAX_ENCODER_LENGTH,
            max_prediction_length=cfg_train.MAX_PREDICTION_LENGTH,
            static_categoricals=["asset"],
            time_varying_known_categoricals=["hour", "minute", "day_of_week"],
            time_varying_known_reals=["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
            time_varying_unknown_reals=[*features_to_normalize, cfg_train.TARGET_COL],
            target_normalizer=GroupNormalizer(groups=["asset"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)

        # Create dataloaders
        train_loader = training.to_dataloader(train=True, batch_size=cfg_train.BATCH_SIZE, num_workers=4)
        val_loader = validation.to_dataloader(train=False, batch_size=cfg_train.BATCH_SIZE, num_workers=4)

        # Define model
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=1e-3,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        # Callbacks
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        checkpoint_callback = ModelCheckpoint(
            dirpath=model_filepath,
            filename="best_model",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )

        # Train
        logger.info("Training...")
        trainer = Trainer(
            max_epochs=cfg_train.EPOCHS,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, checkpoint_callback],
            accelerator="auto",
            devices=1,
        )

        trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Load best model
        best_model_path = checkpoint_callback.best_model_path
        tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        # ---------------- EVALUATION ---------------- #
        actuals = torch.cat([y[0] for x, y in iter(val_loader)])
        predictions = tft.predict(val_loader)

        # Plot predictions vs actuals (first asset batch)
        plt.figure(figsize=(10, 5))
        plt.plot(actuals[: cfg_train.MAX_PREDICTION_LENGTH].detach().cpu().numpy(), label="Actual")
        plt.plot(predictions[: cfg_train.MAX_PREDICTION_LENGTH].detach().cpu().numpy(), label="Prediction")
        plt.legend()
        plt.title("Prediction vs Actual")
        plt.xlabel("Time step")
        plt.ylabel("Target")
        plt.grid()
        plt.show()

        logger.success("✅ Training complete. Best model saved to:", best_model_path)


if __name__ == "__main__":
    Train.train(
        dataset_filepath=cfg_common.DATASET_FILEPATH,
        model_filepath=cfg_common.MODEL_FILEPATH,
    )
