import multiprocessing

import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from pytorch_forecasting import MAE, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from shared.config.common import config as cfg_common
from shared.config.train import DSCols
from shared.config.train import config as cfg_train


class Train:
    @classmethod
    def train(cls, dataset_filepath: str, model_filepath: str) -> None:
        """
        Train model

        Args:
            dataset_filepath (str): Path to dataset
            model_filepath (str): Path to model to be saved
        """

        # Set seed for reproducibility
        seed_everything(cfg_train.SEED)

        # Load data
        logger.info("Loading data...")
        df = pd.read_csv(dataset_filepath, parse_dates=[DSCols.TIMESTAMP.value])
        df = df.sort_values([DSCols.ASSET.value, DSCols.TIMESTAMP.value])

        # Create time index (relative to earliest timestamp)
        logger.info("Creating time index...")

        df[DSCols.TIME_IDX.value] = (
            df[DSCols.TIMESTAMP.value] - df[DSCols.TIMESTAMP.value].min()
        ).dt.total_seconds() // cfg_train.TIME_IDX_STEP_SECS
        df[DSCols.TIME_IDX.value] = df[DSCols.TIME_IDX.value].astype(int)

        # Convert categoricals to string type for proper encoding
        logger.info("Converting categoricals to string type...")
        categorical_cols = [
            DSCols.HOUR.value,
            DSCols.MINUTE.value,
            DSCols.DAY_OF_WEEK.value,
            DSCols.ASSET.value,
        ]
        for col in categorical_cols:
            df[col] = df[col].astype(str)

        # Fill missing values
        df = df.fillna(method="ffill").fillna(method="bfill")

        # Train/val split by time
        logger.info("Creating train/val split...")
        training_cutoff = df[DSCols.TIME_IDX.value].max() - cfg_train.MAX_PREDICTION_LENGTH
        train_df = df[df[DSCols.TIME_IDX.value] <= training_cutoff]

        # Define TimeSeriesDataSet
        training = TimeSeriesDataSet(
            train_df,
            time_idx=DSCols.TIME_IDX.value,
            target=cfg_train.TARGET_COL,
            group_ids=[
                DSCols.ASSET.value,
            ],
            max_encoder_length=cfg_train.MAX_ENCODER_LENGTH,
            max_prediction_length=cfg_train.MAX_PREDICTION_LENGTH,
            static_categoricals=[
                DSCols.ASSET.value,
            ],
            time_varying_known_categoricals=[
                DSCols.HOUR.value,
                DSCols.MINUTE.value,
                DSCols.DAY_OF_WEEK.value,
            ],
            time_varying_known_reals=[
                DSCols.HOUR_SIN.value,
                DSCols.HOUR_COS.value,
                DSCols.DOW_SIN.value,
                DSCols.DOW_COS.value,
            ],
            time_varying_unknown_reals=[
                DSCols.OPEN.value,
                DSCols.HIGH.value,
                DSCols.LOW.value,
                DSCols.CLOSE.value,
                DSCols.VOLUME.value,
                DSCols.RSI.value,
                DSCols.MACD.value,
                DSCols.BOLLINGER_H.value,
                DSCols.BOLLINGER_L.value,
                DSCols.SMA_20.value,
                DSCols.EMA_20.value,
                DSCols.TARGET.value,
            ],
            target_normalizer=GroupNormalizer(
                groups=[
                    DSCols.ASSET.value,
                ],
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

        # Create dataloaders
        train_loader = training.to_dataloader(
            train=True,
            batch_size=cfg_train.BATCH_SIZE,
            num_workers=multiprocessing.cpu_count() // 2,
        )
        val_loader = validation.to_dataloader(
            train=False,
            batch_size=cfg_train.BATCH_SIZE,
            num_workers=multiprocessing.cpu_count() // 2,
        )

        # Define model
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=cfg_train.MODEL_LEARNING_RATE,
            hidden_size=cfg_train.MODEL_HIDDEN_SIZE,
            attention_head_size=cfg_train.MODEL_ATTENTION_HEAD_SIZE,
            dropout=cfg_train.MODEL_DROP_OUT,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        logger.debug(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

        # Callbacks
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg_common.MODEL_DIR,
            filename=cfg_common.CHECKPOINT_FILENAME,
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        tb_logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        # Train
        logger.info("Training...")
        trainer = Trainer(
            max_epochs=cfg_train.MAX_EPOCHS,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=tb_logger,
        )
        trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        logger.success(f"Training complete. Best checkpoint saved to: {checkpoint_callback.best_model_path}")

        # Save model
        logger.info("Saving model...")
        trainer.save_checkpoint(model_filepath)
        logger.success(f"Model saved to: {model_filepath}")

        # Load best model
        trained_tft = TemporalFusionTransformer.load_from_checkpoint(model_filepath)

        # ---------------- EVALUATION ---------------- #
        actuals = torch.cat([y[0] for x, y in iter(val_loader)])
        predictions = trained_tft.predict(val_loader, return_y=True)
        logger.debug(f"MAE: {MAE()(predictions.output, predictions.y)}")

        raw_predictions = trained_tft.predict(val_loader, mode="raw", return_x=True)
        for idx in range(1):
            a = trained_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
            a.savefig(f"prediction_{idx}.png")

        # Plot predictions vs actuals (first asset batch)
        plt.figure(figsize=(10, 5))
        plt.plot(actuals[: cfg_train.MAX_PREDICTION_LENGTH].detach().cpu().numpy(), label="Actual")
        plt.plot(predictions.y[: cfg_train.MAX_PREDICTION_LENGTH].detach().cpu().numpy(), label="Prediction")
        plt.legend()
        plt.title("Prediction vs Actual")
        plt.xlabel("Time step")
        plt.ylabel("Target")
        plt.grid()
        plt.savefig("prediction_vs_actual.png")


if __name__ == "__main__":
    Train.train(
        dataset_filepath=cfg_common.DATASET_FILEPATH,
        model_filepath=cfg_common.MODEL_FILEPATH,
    )
