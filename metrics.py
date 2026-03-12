"""
metrics.py
----------
Comprehensive ML metrics calculation for regression models.
Includes error metrics, statistical measures, and directional accuracy.
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    median_absolute_error,
)
from scipy.stats import pearsonr, spearmanr


class MetricsCalculator:
    """Calculate comprehensive ML metrics for regression predictions."""
    
    @staticmethod
    def calculate_all(y_true, y_pred, currency="USD"):
        """
        Calculate all metrics.
        
        Args:
            y_true: Ground truth values (inverse scaled)
            y_pred: Predicted values (inverse scaled)
            currency: "USD" or "INR" for symbol display
            
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # ── Error Metrics ──────────────────────────────────────
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['rmsle'] = np.sqrt(np.sum((np.log1p(y_pred) - np.log1p(y_true))**2) / len(y_true))
        
        # ── Percentage Errors ──────────────────────────────────
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        metrics['mdape'] = median_absolute_error(y_true, y_pred)
        
        # ── R² and Variance ────────────────────────────────────
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (len(y_true) - 1) / (len(y_true) - 2)
        
        # ── Correlation ────────────────────────────────────────
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        metrics['pearson_corr'] = pearson_corr
        metrics['spearman_corr'] = spearman_corr
        
        # ── Directional Metrics ────────────────────────────────
        y_true_direction = np.diff(y_true)
        y_pred_direction = np.diff(y_pred)
        direction_match = np.sign(y_true_direction) == np.sign(y_pred_direction)
        metrics['directional_accuracy'] = np.mean(direction_match) * 100
        
        # ── Residual Analysis ──────────────────────────────────
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['min_residual'] = np.min(residuals)
        metrics['max_residual'] = np.max(residuals)
        
        # ── Min/Max Errors ────────────────────────────────────
        absolute_errors = np.abs(residuals)
        metrics['min_abs_error'] = np.min(absolute_errors)
        metrics['max_abs_error'] = np.max(absolute_errors)
        
        # ── Theil's U Statistic (for time series) ──────────────
        metrics['theils_u'] = MetricsCalculator._theils_u(y_true, y_pred)
        
        # ── Mean Absolute Percentage Error per range ──────────
        metrics['mape_range'] = MetricsCalculator._mape_by_range(y_true, y_pred)
        
        return metrics
    
    @staticmethod
    def _theils_u(y_true, y_pred):
        """Calculate Theil's U statistic (better for time series)."""
        numerator = np.sum((y_true[1:] - y_pred[1:])**2)
        denominator = np.sum((y_true[1:])**2)
        return np.sqrt(numerator / denominator) if denominator > 0 else 0
    
    @staticmethod
    def _mape_by_range(y_true, y_pred, n_bins=5):
        """Calculate MAPE for different price ranges."""
        bins = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
        mape_ranges = {}
        
        for i in range(len(bins) - 1):
            mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
            if np.sum(mask) > 0:
                mape_val = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
                range_str = f"${bins[i]:.0f}-${bins[i+1]:.0f}"
                mape_ranges[range_str] = mape_val * 100
        
        return mape_ranges
    
    @staticmethod
    def print_metrics(metrics, label="", currency="USD"):
        """Pretty print all metrics."""
        sym = "₹" if currency.upper() == "INR" else "$"
        
        print(f"\n{'═'*80}")
        print(f"  METRICS FOR: {label.upper()}")
        print(f"{'═'*80}")
        
        # Error Metrics
        print(f"\n  ▢ ERROR METRICS")
        print(f"    MAE (Mean Absolute Error)        : {sym}{metrics['mae']:>12,.2f}")
        print(f"    RMSE (Root Mean Squared Error)   : {sym}{metrics['rmse']:>12,.2f}")
        print(f"    MSE (Mean Squared Error)         : {sym}{metrics['mse']:>12,.2f}")
        print(f"    RMSLE (Root Mean Squared Log Err): {metrics['rmsle']:>18.6f}")
        
        # Percentage Errors
        print(f"\n  ▢ PERCENTAGE ERRORS")
        print(f"    MAPE (Mean Absolute %)           : {metrics['mape']*100:>18.2f}%")
        print(f"    MdAPE (Median Absolute %)        : {metrics['mdape']:>18,.2f}")
        
        # Model Fit
        print(f"\n  ▢ MODEL FIT")
        print(f"    R² Score                         : {metrics['r2']:>18.6f}")
        print(f"    Adjusted R² Score                : {metrics['adjusted_r2']:>18.6f}")
        
        # Correlation
        print(f"\n  ▢ CORRELATION")
        print(f"    Pearson Correlation              : {metrics['pearson_corr']:>18.6f}")
        print(f"    Spearman Correlation             : {metrics['spearman_corr']:>18.6f}")
        
        # Directional Accuracy
        print(f"\n  ▢ DIRECTIONAL ACCURACY")
        print(f"    Correct Direction Prediction     : {metrics['directional_accuracy']:>18.2f}%")
        
        # Residual Stats
        print(f"\n  ▢ RESIDUAL ANALYSIS")
        print(f"    Mean Residual                    : {sym}{metrics['mean_residual']:>12,.2f}")
        print(f"    Std Dev of Residuals             : {sym}{metrics['std_residual']:>12,.2f}")
        print(f"    Min Residual                     : {sym}{metrics['min_residual']:>12,.2f}")
        print(f"    Max Residual                     : {sym}{metrics['max_residual']:>12,.2f}")
        
        # Error Range
        print(f"\n  ▢ ERROR RANGE")
        print(f"    Min Absolute Error               : {sym}{metrics['min_abs_error']:>12,.2f}")
        print(f"    Max Absolute Error               : {sym}{metrics['max_abs_error']:>12,.2f}")
        
        # Advanced Metrics
        print(f"\n  ▢ ADVANCED METRICS")
        print(f"    Theil's U Statistic              : {metrics['theils_u']:>18.6f}")
        
        # MAPE by Range
        if metrics['mape_range']:
            print(f"\n  ▢ MAPE BY PRICE RANGE")
            for price_range, mape_val in metrics['mape_range'].items():
                print(f"    {price_range:20s}          : {mape_val:>18.2f}%")
        
        print(f"\n{'═'*80}\n")
        
        return metrics
