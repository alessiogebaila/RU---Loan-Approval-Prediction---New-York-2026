"""
Advanced training: TargetEncoder + Optuna tuning + soft-vote ensemble.

Usage
-----
# Full pipeline: Optuna tune → ensemble → write best submission
python -m src.advanced

# Skip Optuna (use default LightGBM params) — faster
python -m src.advanced --no-optuna

# Control number of Optuna trials (default 50)
python -m src.advanced --trials 100
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from src.features import (
    ID_COL,
    TARGET_COL,
    load_train_test,
    prepare_features,
    split_features_target,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Column groups (determined after prepare_features) ────────────────────────
# High-cardinality categoricals → TargetEncoder (learns approval rate per level)
TARGET_ENCODE_COLS = ["Bank", "City"]
# Low-cardinality categoricals → OneHotEncoder
OHE_COLS = ["BankState"]   # State is constant (NY only) and already excluded implicitly


def _get_col_groups(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (numeric_cols, target_encode_cols, ohe_cols)."""
    numeric = X.select_dtypes(include=["number", "Int8", "Int16"]).columns.tolist()
    te = [c for c in TARGET_ENCODE_COLS if c in X.columns]
    ohe = [c for c in X.select_dtypes(include=["object", "string"]).columns
           if c not in TARGET_ENCODE_COLS]
    return numeric, te, ohe


# ── Preprocessor factory ─────────────────────────────────────────────────────

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer with:
    - Median imputation + no scaling for numeric (trees don't need it)
    - TargetEncoder for Bank and City (avoids OHE dimension explosion)
    - OneHotEncoder (max 30 categories) for remaining low-card categoricals
    """
    numeric, te_cols, ohe_cols = _get_col_groups(X)

    transformers: list = []

    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))

    if te_cols:
        # TargetEncoder replaces each category with its smoothed mean of the target.
        # cv=5 internally cross-fits to prevent target leakage on train folds.
        transformers.append(("te", TargetEncoder(cv=5, smooth="auto"), te_cols))

    if ohe_cols:
        ohe_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)),
        ])
        transformers.append(("ohe", ohe_pipe, ohe_cols))

    if not transformers:
        raise ValueError("No feature columns found.")

    return ColumnTransformer(transformers)


# ── Model builders ────────────────────────────────────────────────────────────

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
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])


def make_xgb_pipeline(X: pd.DataFrame) -> Pipeline:
    from xgboost import XGBClassifier
    return Pipeline([
        ("prep", make_preprocessor(X)),
        ("clf", XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            # No scale_pos_weight — is_unbalance equivalent handled via threshold tuning
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )),
    ])


# ── Threshold tuning ─────────────────────────────────────────────────────────

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


# ── Optuna tuning for LightGBM ───────────────────────────────────────────────

def run_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
) -> dict:
    """
    Search LightGBM hyperparameters with Optuna.
    Returns the best param dict (to pass directly to make_lgbm_pipeline).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        )
        pipe = make_lgbm_pipeline(X_train, **params)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="f1_macro", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nOptuna best CV f1_macro: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


# ── CV helper ────────────────────────────────────────────────────────────────

def evaluate_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, label: str) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"[{label}] CV f1_macro: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main(n_optuna_trials: int = 50, run_optuna_flag: bool = True) -> None:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & engineer features ──────────────────────────────────────────────
    print("Loading data...")
    train_raw, test_raw = load_train_test()
    X_raw, y = split_features_target(train_raw)
    test_ids = test_raw[ID_COL]
    X_train = prepare_features(X_raw)
    X_test = prepare_features(test_raw)
    print(f"Train: {X_train.shape}   Test: {X_test.shape}")

    numeric, te_cols, ohe_cols = _get_col_groups(X_train)
    print(f"Numeric: {len(numeric)}  TargetEncode: {te_cols}  OHE: {ohe_cols}")

    # ── Step 1: Cross-validate base models ───────────────────────────────────
    print("\n--- Step 1: Cross-validate base models ---")
    lgbm_pipe = make_lgbm_pipeline(X_train)
    rf_pipe   = make_rf_pipeline(X_train)
    xgb_pipe  = make_xgb_pipeline(X_train)

    lgbm_cv = evaluate_cv(lgbm_pipe,  X_train, y, "lgbm_base")
    rf_cv   = evaluate_cv(rf_pipe,    X_train, y, "rf")
    xgb_cv  = evaluate_cv(xgb_pipe,   X_train, y, "xgb")

    # ── Step 2: Optuna tune LightGBM ─────────────────────────────────────────
    best_lgbm_params: dict = {}
    if run_optuna_flag:
        print(f"\n--- Step 2: Optuna ({n_optuna_trials} trials) ---")
        best_lgbm_params = run_optuna(X_train, y, n_trials=n_optuna_trials)
        lgbm_tuned_pipe = make_lgbm_pipeline(X_train, **best_lgbm_params)
        lgbm_tuned_cv = evaluate_cv(lgbm_tuned_pipe, X_train, y, "lgbm_tuned")
    else:
        print("\n--- Step 2: Skipping Optuna ---")
        lgbm_tuned_pipe = lgbm_pipe
        lgbm_tuned_cv = lgbm_cv

    # ── Step 3: Soft-vote ensemble ────────────────────────────────────────────
    print("\n--- Step 3: Soft-vote ensemble (RF + LightGBM + XGBoost) ---")
    # Weights proportional to CV score so best models dominate
    scores = np.array([lgbm_tuned_cv, rf_cv, xgb_cv])
    weights = list(scores / scores.sum())
    print(f"Ensemble weights — lgbm: {weights[0]:.3f}  rf: {weights[1]:.3f}  xgb: {weights[2]:.3f}")

    # VotingClassifier needs the full pipeline as the estimator
    # We build each pipeline fresh so they share the same preprocessor design
    lgbm_e = make_lgbm_pipeline(X_train, **best_lgbm_params)
    rf_e   = make_rf_pipeline(X_train)
    xgb_e  = make_xgb_pipeline(X_train)

    ensemble = Pipeline([
        ("clf", VotingClassifier(
            estimators=[("lgbm", lgbm_e), ("rf", rf_e), ("xgb", xgb_e)],
            voting="soft",
            weights=weights,
        ))
    ])
    # VotingClassifier wraps whole pipelines — no separate prep step needed
    ensemble_cv = evaluate_cv(
        VotingClassifier(
            estimators=[("lgbm", make_lgbm_pipeline(X_train, **best_lgbm_params)),
                        ("rf", make_rf_pipeline(X_train)),
                        ("xgb", make_xgb_pipeline(X_train))],
            voting="soft",
            weights=weights,
        ),
        X_train, y, "ensemble"
    )

    # ── Step 4: Threshold tuning on held-out val ──────────────────────────────
    print("\n--- Step 4: Threshold tuning ---")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y, test_size=0.2, stratify=y, random_state=42
    )

    ens_voter = VotingClassifier(
        estimators=[("lgbm", make_lgbm_pipeline(X_train, **best_lgbm_params)),
                    ("rf", make_rf_pipeline(X_train)),
                    ("xgb", make_xgb_pipeline(X_train))],
        voting="soft",
        weights=weights,
    )
    ens_voter.fit(X_tr, y_tr)
    val_proba = ens_voter.predict_proba(X_val)[:, 1]
    t_best, f_best = best_threshold(y_val.values, val_proba)
    print(f"Ensemble best threshold: {t_best:.2f}  val f1_macro: {f_best:.4f}")

    # ── Step 5: Final fit on all training data ────────────────────────────────
    print("\n--- Step 5: Final fit on all training data ---")
    final_voter = VotingClassifier(
        estimators=[("lgbm", make_lgbm_pipeline(X_train, **best_lgbm_params)),
                    ("rf", make_rf_pipeline(X_train)),
                    ("xgb", make_xgb_pipeline(X_train))],
        voting="soft",
        weights=weights,
    )
    final_voter.fit(X_train, y)
    test_proba = final_voter.predict_proba(X_test)[:, 1]
    preds = (test_proba >= t_best).astype(int)

    # ── Step 6: Write submission ──────────────────────────────────────────────
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds})
    out_path = out_dir / "ensemble_advanced.csv"
    submission.to_csv(out_path, index=False)

    print(f"\nWrote {out_path}  ({len(submission)} rows)")
    print(f"Accept=1: {preds.sum()}  Accept=0: {(preds==0).sum()}"
          f"  ratio: {preds.mean():.2%}")
    print("\n=== SUMMARY ===")
    print(f"  RF CV:              {rf_cv:.4f}")
    print(f"  XGB CV:             {xgb_cv:.4f}")
    print(f"  LightGBM base CV:   {lgbm_cv:.4f}")
    print(f"  LightGBM tuned CV:  {lgbm_tuned_cv:.4f}")
    print(f"  Ensemble CV:        {ensemble_cv:.4f}")
    print(f"  Ensemble val F1:    {f_best:.4f}  @ threshold={t_best:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials (default 50)")
    parser.add_argument("--no-optuna", action="store_true",
                        help="Skip Optuna and use default LightGBM params")
    args = parser.parse_args()
    main(n_optuna_trials=args.trials, run_optuna_flag=not args.no_optuna)
