"""Cross-validation, OOF predictions, and train/test diagnostics."""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from src.features import load_train_test, prepare_features, split_features_target


def best_threshold(y_true, y_prob, thresholds=None):
    from src.advanced import best_threshold as _bt
    return _bt(y_true, y_prob, thresholds)


def temporal_diagnostics(
    train_raw: pd.DataFrame | None = None,
    test_raw: pd.DataFrame | None = None,
) -> dict:
    """Compare date ranges and categorical coverage between train and test."""
    if train_raw is None or test_raw is None:
        train_raw, test_raw = load_train_test()

    X_tr = prepare_features(train_raw.drop(columns=["Accept"], errors="ignore"))
    X_te = prepare_features(test_raw)

    report: dict = {}

    for col in ["ApprovalDate_year", "DisbursementDate_year"]:
        if col in X_tr.columns and col in X_te.columns:
            report[col] = {
                "train_min": int(X_tr[col].min()),
                "train_max": int(X_tr[col].max()),
                "test_min": int(X_te[col].min()),
                "test_max": int(X_te[col].max()),
            }

    for col in ["Bank", "City"]:
        if col in train_raw.columns:
            tr_levels = set(train_raw[col].astype(str))
            te_levels = set(test_raw[col].astype(str))
            unseen = te_levels - tr_levels
            report[f"{col}_unseen_in_test"] = len(unseen)
            report[f"{col}_unseen_pct"] = len(unseen) / max(len(te_levels), 1)

    if "State" in train_raw.columns:
        report["State_unique_train"] = train_raw["State"].nunique()
        report["State_unique_test"] = test_raw["State"].nunique()

    return report


def adversarial_validation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[float, pd.Series]:
    """
    Train a classifier to distinguish train (0) vs test (1) rows.
    Returns (ROC-AUC, feature importances as Series).
    AUC ~0.5 => no drift; >0.7 => strong drift.
    """
    X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y = np.array([0] * len(X_train) + [1] * len(X_test))

    from src.advanced import make_preprocessor

    prep = make_preprocessor(X_train)
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=random_state,
    )
    pipe = Pipeline([("prep", prep), ("clf", clf)])

    oof = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, va_idx in cv.split(X, y):
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = pipe.predict_proba(X.iloc[va_idx])[:, 1]

    auc = roc_auc_score(y, oof)
    pipe.fit(X, y)
    prep_fitted = pipe.named_steps["prep"]
    clf_fitted = pipe.named_steps["clf"]
    try:
        names = prep_fitted.get_feature_names_out()
    except Exception:
        names = [f"f{i}" for i in range(len(clf_fitted.feature_importances_))]
    importances = pd.Series(clf_fitted.feature_importances_, index=names)
    return auc, importances.sort_values(ascending=False)


def time_based_splits(
    X: pd.DataFrame,
    y: pd.Series,
    year_col: str = "ApprovalDate_year",
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV: train on earlier years, validate on later."""
    if year_col not in X.columns:
        raise KeyError(f"{year_col} not in features")

    years = X[year_col].fillna(X[year_col].median()).astype(int)
    unique_years = sorted(years.unique())
    if len(unique_years) < n_splits + 1:
        n_splits = max(2, len(unique_years) - 1)

    # Split year range into n_splits validation chunks
    year_arr = np.array(unique_years)
    chunks = np.array_split(year_arr, n_splits)

    for chunk in chunks:
        val_years = set(chunk.tolist())
        val_mask = years.isin(val_years).values
        train_mask = ~val_mask
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(np.unique(y.iloc[val_idx])) < 2:
            continue
        yield train_idx, val_idx


def oof_predict_proba(
    pipeline_factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | RepeatedStratifiedKFold | None = None,
    n_splits: int = 5,
    n_repeats: int = 1,
) -> np.ndarray:
    """Out-of-fold positive-class probabilities."""
    if cv is None:
        if n_repeats > 1:
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=42
            )
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        pipe = pipeline_factory()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = pipe.predict_proba(X.iloc[val_idx])[:, 1]
    return oof


def oof_macro_f1(
    pipeline_factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | RepeatedStratifiedKFold | None = None,
    n_splits: int = 5,
    n_repeats: int = 1,
) -> tuple[float, float]:
    """Return (best_oof_macro_f1, best_threshold)."""
    oof = oof_predict_proba(
        pipeline_factory, X, y, cv=cv, n_splits=n_splits, n_repeats=n_repeats
    )
    return best_threshold(y.values, oof)


def time_cv_macro_f1(
    pipeline_factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> tuple[float, float]:
    """Time-based CV with threshold tuning per fold, averaged."""
    fold_scores: list[float] = []
    for train_idx, val_idx in time_based_splits(X, y, n_splits=n_splits):
        pipe = pipeline_factory()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        _, f = best_threshold(y.iloc[val_idx].values, proba)
        fold_scores.append(f)
    if not fold_scores:
        return 0.0, 0.5
    return float(np.mean(fold_scores)), 0.5


def base_rate_threshold(y_train: pd.Series | np.ndarray) -> float:
    """Threshold so predicted approval rate matches training prevalence."""
    rate = float(np.asarray(y_train).mean())
    return rate  # predict proba >= rate for class 1


def threshold_for_target_rate(proba: np.ndarray, target_rate: float) -> float:
    """Pick threshold so fraction of preds==1 equals target_rate."""
    return float(np.quantile(proba, 1.0 - target_rate))


def print_diagnostics() -> dict:
    """Run Phase 0 diagnostics and print summary."""
    train_raw, test_raw = load_train_test()
    X_raw, y = split_features_target(train_raw)
    X_train = prepare_features(X_raw)
    X_test = prepare_features(test_raw)

    print("=== Phase 0: Temporal / coverage ===")
    temp = temporal_diagnostics(train_raw, test_raw)
    for k, v in temp.items():
        print(f"  {k}: {v}")

    print("\n=== Phase 0: Adversarial validation ===")
    auc, imp = adversarial_validation(X_train, X_test)
    print(f"  Train vs test ROC-AUC: {auc:.4f}")
    if auc > 0.7:
        print("  -> Strong drift detected; prefer time-based CV.")
    elif auc > 0.6:
        print("  -> Moderate drift; use both random and time CV.")
    else:
        print("  -> Low drift; random stratified CV is reasonable.")
    print("  Top drift features:")
    for name, val in imp.head(10).items():
        print(f"    {name}: {val:.4f}")

    return {"temporal": temp, "adversarial_auc": auc, "top_importances": imp.head(15)}


if __name__ == "__main__":
    print_diagnostics()
