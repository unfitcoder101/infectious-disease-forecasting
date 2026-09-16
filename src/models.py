"""
The two gradient-boosting models compared in this study.

WHAT GRADIENT BOOSTING IS, BRIEFLY
----------------------------------
Both XGBoost and LightGBM build an ENSEMBLE of small decision trees, added
one at a time. Each new tree is fitted to the errors left over by the trees
built so far, so the ensemble gradually corrects itself. They differ mainly
in how they grow each tree: XGBoost grows level-by-level, LightGBM grows
leaf-by-leaf (splitting whichever leaf reduces error most), which usually
makes LightGBM faster but slightly more prone to overfitting on small data.

WHY IDENTICAL SETTINGS ACROSS WINDOWS
-------------------------------------
The research question is "does more history help?", NOT "what is the best
possible model?". If each window got its own tuned hyperparameters, a
difference between windows could just be a difference in tuning luck. So
every window gets exactly the same configuration and the same random seed,
and the ONLY thing that varies is the number of lag features.
"""

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

RANDOM_SEED = 42

# Deliberately modest, fixed settings. n_estimators/depth kept small because
# the dataset has ~650 training rows -- large ensembles would overfit.
_COMMON = dict(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=RANDOM_SEED)


def get_models() -> dict:
    """Return a fresh, identically-configured model per algorithm."""
    return {
        "XGBoost": XGBRegressor(
            **_COMMON,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=1,          # single-threaded so timing comparisons are fair
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            **_COMMON,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=1,
            verbose=-1,
            min_child_samples=10,
        ),
    }
