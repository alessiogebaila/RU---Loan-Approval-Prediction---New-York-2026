"""
Advanced training: TargetEncoder + Optuna tuning + soft-vote ensemble.

Usage
-----
python -m src.advanced --no-optuna
python -m src.advanced --trials 20
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from src.cv_utils import (
    base_rate_threshold,
    oof_macro_f1,
    oof_predict_proba,
    threshold_for_target_rate,
    time_cv_macro_f1,
)
from src.features import (
    ID_COL,
    TARGET_COL,
    get_rare_cities,
    load_train_test,
    prepare_features,
    split_features_target,
)

warnings.filterwarnings("ignore", category=UserWarning)

TARGET_ENCODE_COLS = ["Bank", "City"]
# Higher smoothing for high-cardinality City reduces overfitting on rare levels
TARGET_ENCODER_SMOOTH = 50.0


def _get_col_groups(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric = X.select_dtypes(include=["number", "Int8", "Int16", "Float64"]).columns.tolist()
    te = [c for c in TARGET_ENCODE_COLS if c in X.columns]
    ohe = [
        c
        for c in X.select_dtypes(include=["object", "string"]).columns
        if c not in TARGET_ENCODE_COLS
    ]
    return numeric, te, ohe


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric, te_cols, ohe_cols = _get_col_groups(X)

    transformers: list = []

    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))

    if te_cols:
        transformers.append(
            (
                "te",
                TargetEncoder(cv=5, smooth=TARGET_ENCODER_SMOOTH),
                te_cols,
            )
        )

    if ohe_cols:
        ohe_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=30,
                ),
            ),
        ])
        transformers.append(("ohe", ohe_pipe, ohe_cols))

    if not transformers:
        raise ValueError("No feature columns found.")

    return ColumnTransformer(transformers)


def make_lgbm_pipeline(X: pd.DataFrame, **lgbm_params) -> Pipeline:
    from lightgbm import LGBMClassifier

    defaults = dict(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        is_unbalance=True,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    defaults.update(lgbm_params)
    return Pipeline([("prep", make_preprocessor(X)), ("clf", LGBMClassifier(**defaults))])


def make_rf_pipeline(X: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(X)),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ])


def make_xgb_pipeline(X: pd.DataFrame) -> Pipeline:
    from xgboost import XGBClassifier

    return Pipeline([
        ("prep", make_preprocessor(X)),
        (
            "clf",
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ])


def best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_f = 0.5, 0.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f = f1_score(y_true, preds, average="macro", zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


def run_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 20,
    cv_splits: int = 3,
) -> dict:
    """
    Optuna objective = OOF Macro F1 with tuned threshold (not default 0.5).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
        )

        def factory():
            return make_lgbm_pipeline(X_train, **params)

        f1, _ = oof_macro_f1(factory, X_train, y_train, n_splits=cv_splits, n_repeats=1)
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nOptuna best OOF f1_macro (tuned threshold): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def evaluate_cv(
    pipe_factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
    n_splits: int = 5,
) -> float:
    f1, t = oof_macro_f1(pipe_factory, X, y, n_splits=n_splits)
    print(f"[{label}] OOF f1_macro: {f1:.4f}  (threshold={t:.2f})")
    return f1


def main(n_optuna_trials: int = 20, run_optuna_flag: bool = True) -> None:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_raw, test_raw = load_train_test()
    X_raw, y = split_features_target(train_raw)
    test_ids = test_raw[ID_COL]
    rare = get_rare_cities(X_raw)
    X_train = prepare_features(X_raw, rare_cities=rare)
    X_test = prepare_features(test_raw, rare_cities=rare)
    print(f"Train: {X_train.shape}   Test: {X_test.shape}")

    numeric, te_cols, ohe_cols = _get_col_groups(X_train)
    print(f"Numeric: {len(numeric)}  TargetEncode: {te_cols}  OHE: {ohe_cols}")

    print("\n--- Model OOF scores (5-fold) ---")
    lgbm_cv = evaluate_cv(lambda: make_lgbm_pipeline(X_train), X_train, y, "lgbm")
    rf_cv = evaluate_cv(lambda: make_rf_pipeline(X_train), X_train, y, "rf")
    xgb_cv = evaluate_cv(lambda: make_xgb_pipeline(X_train), X_train, y, "xgb")

    time_f1, _ = time_cv_macro_f1(
        lambda: make_lgbm_pipeline(X_train), X_train, y, n_splits=5
    )
    print(f"[lgbm] Time-based CV f1_macro: {time_f1:.4f}")

    best_lgbm_params: dict = {}
    if run_optuna_flag and n_optuna_trials > 0:
        print(f"\n--- Optuna ({n_optuna_trials} trials, OOF threshold objective) ---")
        best_lgbm_params = run_optuna(X_train, y, n_trials=n_optuna_trials)
        lgbm_tuned_cv = evaluate_cv(
            lambda: make_lgbm_pipeline(X_train, **best_lgbm_params),
            X_train,
            y,
            "lgbm_tuned",
        )
    else:
        lgbm_tuned_cv = lgbm_cv

    print("\n--- RF + LGBM ensemble (no XGB) ---")
    scores = np.array([lgbm_tuned_cv, rf_cv])
    weights = list(scores / scores.sum())

    def ensemble_proba(X_in: pd.DataFrame) -> np.ndarray:
        p_lgbm = make_lgbm_pipeline(X_train, **best_lgbm_params).fit(X_train, y).predict_proba(
            X_in
        )[:, 1]
        p_rf = make_rf_pipeline(X_train).fit(X_train, y).predict_proba(X_in)[:, 1]
        return weights[0] * p_lgbm + weights[1] * p_rf

    oof_ens = (
        oof_predict_proba(lambda: make_lgbm_pipeline(X_train, **best_lgbm_params), X_train, y)
        + oof_predict_proba(lambda: make_rf_pipeline(X_train), X_train, y)
    ) / 2
    t_oof, f_oof = best_threshold(y.values, oof_ens)
    print(f"Ensemble OOF f1_macro: {f_oof:.4f}  threshold={t_oof:.2f}")

    train_rate = base_rate_threshold(y)
    t_base = threshold_for_target_rate(
        ensemble_proba(X_train), train_rate
    )
    print(f"Base-rate threshold (~{train_rate:.1%} approvals): {t_base:.2f}")

    test_proba = ensemble_proba(X_test)

    # Primary: OOF-tuned threshold
    preds_oof = (test_proba >= t_oof).astype(int)
    path_oof = out_dir / "ensemble_improved.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds_oof}).to_csv(path_oof, index=False)
    print(f"\nWrote {path_oof}  Accept=1: {preds_oof.mean():.1%}")

    # Alternate: match training approval prevalence
    preds_base = (test_proba >= t_base).astype(int)
    path_base = out_dir / "ensemble_baserate.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds_base}).to_csv(path_base, index=False)
    print(f"Wrote {path_base}  Accept=1: {preds_base.mean():.1%}")

    print("\n=== SUMMARY ===")
    print(f"  LGBM OOF:     {lgbm_tuned_cv:.4f}")
    print(f"  RF OOF:       {rf_cv:.4f}")
    print(f"  XGB OOF:      {xgb_cv:.4f}")
    print(f"  Time-CV LGBM: {time_f1:.4f}")
    print(f"  Ensemble OOF: {f_oof:.4f} @ t={t_oof:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--no-optuna", action="store_true")
    args = parser.parse_args()
    main(n_optuna_trials=args.trials, run_optuna_flag=not args.no_optuna)
