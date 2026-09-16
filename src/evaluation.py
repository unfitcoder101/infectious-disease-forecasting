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


def rolling_origin_splits(table, min_train: int, test_block: int, n_folds: int | None = None):
    """
    ROLLING-ORIGIN (walk-forward) cross-validation splits.

    WHY THIS EXISTS, ON TOP OF chronological_split()
    -------------------------------------------------
    chronological_split() gives ONE train/val/test cut, so "does history
    length matter" gets answered by ONE fixed 139-week slice of the future.
    robustness_check() (in experiment.py) already showed that the val-period
    ranking and the test-period ranking of windows DISAGREE -- i.e. that one
    slice is not representative of the next one. Rolling-origin CV is the
    direct fix: instead of one test slice, we generate a SEQUENCE of them and
    look at the whole distribution of scores per window, not a single point.

    HOW A FOLD IS BUILT (expanding window, walk-forward)
    ------------------------------------------------------
    Fold i:
        train = rows[0            : min_train + i*test_block]
        test  = rows[min_train + i*test_block : min_train + (i+1)*test_block]

    Two properties make this leakage-safe, same standard as chronological_split:
      1. EXPANDING window: training data only ever grows forward in time and
         never includes a row later than the fold's test block. This mirrors
         how a real surveillance system accumulates history -- it never gets
         to see next quarter's data early.
      2. NON-OVERLAPPING test blocks: fold i's test rows and fold j's test
         rows (i != j) never share a week. This keeps folds independent, so
         averaging MAE across folds is not double-counting any observation.

    Returns
    -------
    list of (fold_index, train_df, test_df), fold_index starting at 0.
    Any leftover rows at the series end that don't fill one more whole
    test_block are simply not used by any fold (never leaked into training
    partway through, which padding a short final block would risk).
    """
    n = len(table)
    max_folds = (n - min_train) // test_block
    if max_folds < 1:
        raise ValueError(
            f"Not enough rows for even one fold: n={n}, min_train={min_train}, "
            f"test_block={test_block}"
        )
    k = max_folds if n_folds is None else min(n_folds, max_folds)

    folds = []
    for i in range(k):
        train_end = min_train + i * test_block
        test_end = train_end + test_block
        folds.append((i, table.iloc[:train_end], table.iloc[train_end:test_end]))
    return folds


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
