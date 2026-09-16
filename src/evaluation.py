"""
Chronological splitting and forecast accuracy metrics.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def chronological_split(table, train_frac=0.70, val_frac=0.15):
    """
    Split BY TIME, never at random.

    WHY RANDOM SPLITTING IS INVALID HERE (temporal leakage)
    -------------------------------------------------------
    A random split scatters test weeks in among training weeks. Two things
    then go wrong:

      1. The model gets trained on weeks that come AFTER the weeks it is
         tested on -- it effectively sees the future.
      2. Worse, this data is heavily autocorrelated: consecutive weeks are
         nearly identical. Week 300's target IS week 301's lag_1. So a
         random split puts near-duplicate rows on both sides of the split,
         and the model can score well by near-memorisation.

    The result is an optimistic score that would collapse in real
    deployment, where you genuinely only have the past. Splitting by time
    reproduces the real forecasting situation: train on the earliest weeks,
    test on the latest, never the reverse.
    """
    n = len(table)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return table.iloc[:i_train], table.iloc[i_train:i_val], table.iloc[i_val:]


def evaluate(y_true, y_pred) -> dict:
    """
    MAE  - mean absolute error, in CASES. Average size of a miss.
           Easy to explain, treats all errors proportionally.
    RMSE - root mean squared error, also in CASES, but squares errors first,
           so it punishes large misses much harder. For epidemic data this
           mostly measures how badly you miss the outbreak PEAKS.
           RMSE >> MAE therefore signals a few big misses.
    R2   - fraction of variance explained, vs. a baseline that always
           predicts the training mean. 1.0 is perfect, 0.0 means no better
           than the mean, and NEGATIVE means worse than the mean.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def naive_baseline(table) -> dict:
    """
    The PERSISTENCE baseline: predict next week = this week (lag_1).

    Any forecasting model must beat this to have earned its complexity.
    It is the single most important sanity check in the whole study.
    """
    return evaluate(table["target"], table["lag_1"])
