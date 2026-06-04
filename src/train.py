"""
Train and evaluate models, write a Kaggle submission CSV.

Usage
-----
# Logistic Regression (fast sanity check)
python -m src.train --model lr --output submissions/lr.csv

# Random Forest
python -m src.train --model rf --output submissions/rf.csv

# XGBoost  (pip install xgboost)
python -m src.train --model xgb --output submissions/xgb.csv

# LightGBM  (pip install lightgbm)
python -m src.train --model lgbm --output submissions/lgbm.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.features import (
    ID_COL,
    TARGET_COL,
    build_preprocessor,
    load_train_test,
    prepare_features,
    split_features_target,
)

# ── Model registry ────────────────────────────────────────────────────────────

def _make_clf(name: str):
    """Return an unfitted classifier by short name."""
    if name == "lr":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("Run: pip install xgboost")
        # scale_pos_weight compensates for 80/20 imbalance
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=4,   # approx neg/pos ratio
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
    if name == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("Run: pip install lightgbm")
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {name!r}. Choose from: lr, rf, xgb, lgbm")


# ── Threshold tuning ─────────────────────────────────────────────────────────

def _best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Grid-search the decision threshold that maximises Macro F1.

    Returns (best_threshold, best_f1_macro).
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f = 0.5, 0.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f = f1_score(y_true, preds, average="macro", zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


# ── Core training function ────────────────────────────────────────────────────

def train_and_predict(
    data_dir: str | Path | None = None,
    model_name: str = "lgbm",
    tune_threshold: bool = True,
) -> tuple[Pipeline, float, pd.DataFrame]:
    """
    Load data, engineer features, cross-validate, fit on full train, predict test.

    Returns
    -------
    model       : fitted sklearn Pipeline
    threshold   : decision threshold used for predictions
    submission  : DataFrame with columns [id, Accept]
    """
    train, test = load_train_test(data_dir)
    X_raw, y_train = split_features_target(train)

    test_ids = test[ID_COL].copy()
    X_train = prepare_features(X_raw)
    X_test = prepare_features(test)

    clf = _make_clf(model_name)
    preprocessor = build_preprocessor(X_train)

    # Tree-based models don't need StandardScaler — skip it for them
    if model_name in ("rf", "xgb", "lgbm"):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder

        numeric_cols, categorical_cols = _get_col_groups(X_train)

        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
        ])
        transformers = []
        if numeric_cols:
            transformers.append(("num", num_pipe, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", cat_pipe, categorical_cols))
        preprocessor = ColumnTransformer(transformers)

    model = Pipeline([("prep", preprocessor), ("clf", clf)])

    # ── Cross-validation ─────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    print(
        f"[{model_name}] CV f1_macro: "
        f"{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
    )

    # ── Threshold tuning on a single held-out fold ────────────────────────────
    threshold = 0.5
    if tune_threshold and hasattr(clf, "predict_proba"):
        from sklearn.model_selection import train_test_split

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        model.fit(X_tr, y_tr)
        val_proba = model.predict_proba(X_val)[:, 1]
        threshold, val_f1 = _best_threshold(y_val.values, val_proba)
        print(f"[{model_name}] Best threshold: {threshold:.2f}  val f1_macro: {val_f1:.4f}")

    # ── Final fit on all training data ────────────────────────────────────────
    model.fit(X_train, y_train)

    if hasattr(model.named_steps["clf"], "predict_proba"):
        test_proba = model.predict_proba(X_test)[:, 1]
        preds = (test_proba >= threshold).astype(int)
    else:
        preds = model.predict(X_test).astype(int)

    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds})
    return model, threshold, submission


def _get_col_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include=["number", "Int8", "Int16", "Float64"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Loan approval model trainer")
    parser.add_argument(
        "--model",
        choices=["lr", "rf", "xgb", "lgbm"],
        default="lgbm",
        help="Classifier to use (default: lgbm)",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Submission CSV path (default: submissions/<model>.csv)",
    )
    parser.add_argument(
        "--no-tune-threshold",
        action="store_true",
        help="Skip threshold tuning and use 0.5",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = args.output or root / "submissions" / f"{args.model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    _, _, submission = train_and_predict(
        data_dir=args.data_dir,
        model_name=args.model,
        tune_threshold=not args.no_tune_threshold,
    )
    submission.to_csv(out, index=False)
    print(f"Wrote {out} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
