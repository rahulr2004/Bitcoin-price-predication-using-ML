"""
predict.py
----------
Ensemble prediction engine: 70% LSTM + 30% XGBoost.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor

from src.preprocessing import inverse_close

LSTM_WEIGHT = 0.70
XGB_WEIGHT  = 0.30


def predict_ensemble(lstm, xgb, X, scaler, n_features):
    lstm_pred = lstm.predict(X, verbose=0).flatten()
    xgb_pred  = xgb.predict(X.reshape(X.shape[0], -1)).flatten()
    ensemble  = LSTM_WEIGHT * lstm_pred + XGB_WEIGHT * xgb_pred
    return inverse_close(scaler, ensemble, n_features)


def predict_next_day(lstm, xgb, last_window, scaler, n_features) -> float:
    return float(predict_ensemble(lstm, xgb, last_window, scaler, n_features)[0])
