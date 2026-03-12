"""
train_model.py
--------------
Builds, trains, and saves the LSTM and XGBoost models.
"""

import os
import numpy as np
import joblib

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from xgboost import XGBRegressor

os.makedirs("models", exist_ok=True)


# ── LSTM ──────────────────────────────────────────────────────

def build_lstm(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(64, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(32),
        Dropout(0.2),

        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    model.summary()
    return model


def train_lstm(
    X_tr, y_tr, X_val, y_val,
    epochs: int = 50,
    batch_size: int = 32,
) -> Sequential:
    model = build_lstm((X_tr.shape[1], X_tr.shape[2]))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint("models/lstm_best.keras", save_best_only=True),
    ]
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    model.save("models/bitcoin_lstm.keras")
    print("[train_model] LSTM saved → models/bitcoin_lstm.keras")
    return model


# ── XGBoost ───────────────────────────────────────────────────

def train_xgb(X_tr, y_tr, X_val, y_val) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_tr.reshape(X_tr.shape[0], -1), y_tr,
        eval_set=[(X_val.reshape(X_val.shape[0], -1), y_val)],
        verbose=100,
    )
    joblib.dump(model, "models/bitcoin_xgb.pkl")
    print("[train_model] XGBoost saved → models/bitcoin_xgb.pkl")
    return model


# ── Loaders ───────────────────────────────────────────────────

def load_lstm() -> Sequential:
    return load_model("models/bitcoin_lstm.keras")

def load_xgb() -> XGBRegressor:
    return joblib.load("models/bitcoin_xgb.pkl")
