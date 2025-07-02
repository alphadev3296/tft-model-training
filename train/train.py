import pandas as pd
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE
from sklearn.model_selection import train_test_split

# Load your CSV
df = pd.read_csv("your_data.csv", parse_dates=["timestamp"])

# Ensure timestamp is sorted
df = df.sort_values(["asset", "timestamp"])

# Create time_idx as integer index
df["time_idx"] = df.groupby("asset").cumcount()

# Normalize volume
df["volume"] = np.log1p(df["volume"])

# Parameters
max_encoder_length = 48
max_prediction_length = 12

# Define training dataset
training = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="target",
    group_ids=["asset"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["target"],
    time_varying_known_reals=[
        "hour", "minute", "day_of_week", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "open", "high", "low", "close", "volume", "rsi", "macd", 
        "bollinger_h", "bollinger_l", "sma_20", "ema_20"
    ],
    target_normalizer=NaNLabelEncoder(),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Split dataset
train_dataset, val_dataset = training.split_before(0.8)

# Dataloaders
from torch.utils.data import DataLoader

train_dataloader = train_dataset.to_dataloader(train=True, batch_size=64, num_workers=4)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=4)

# Define model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=SMAPE(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="min")
lr_logger = LearningRateMonitor()

# Trainer
trainer = Trainer(
    max_epochs=30,
    accelerator="auto",
    callbacks=[early_stop, lr_logger],
    gradient_clip_val=0.1,
)

# Train
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# Save model
tft.save("tft_model")

# Evaluate
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = tft.predict(val_dataloader)
print(f"SMAPE: {SMAPE()(predictions, actuals)}")
