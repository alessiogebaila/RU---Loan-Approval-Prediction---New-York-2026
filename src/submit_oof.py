"""
Generate submissions with out-of-fold (OOF) threshold tuning.

Usage
-----
python -m src.submit_oof --model all
python -m src.submit_oof --model lgbm --repeats 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.advanced import (
    make_lgbm_pipeline,
    make_rf_pipeline,
    make_xgb_pipeline,
)
from src.cv_utils import (
    base_rate_threshold,
    oof_macro_f1,
    oof_predict_proba,
    threshold_for_target_rate,
)
from src.features import (
    ID_COL,
    TARGET_COL,
    get_rare_cities,
    load_train_test,
    prepare_features,
    split_features_target,
)


def fit_predict_proba(pipeline_factory, X, y, X_test) -> np.ndarray:
    pipe = pipeline_factory()
    pipe.fit(X, y)
    return pipe.predict_proba(X_test)[:, 1]


def write_submission(ids, proba, threshold, path: Path, label: str = "") -> None:
    preds = (proba >= threshold).astype(int)
    out = pd.DataFrame({ID_COL: ids, TARGET_COL: preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(
        f"Wrote {path}  {label}  threshold={threshold:.2f}  "
        f"Accept=1: {preds.mean():.1%} ({preds.sum()}/{len(preds)})"
    )


def try_catboost_pipeline(X):
    try:
        from catboost import CatBoostClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "string"]
        num_cols = [c for c in X.columns if c not in cat_cols]

        class _CatBoostPrepPipeline:
            def __init__(self):
                self.num_cols = num_cols
                self.cat_cols = cat_cols
                self.clf = None

            def fit(self, X_fit, y_fit):
                Xn = X_fit[self.num_cols].copy()
                Xc = X_fit[self.cat_cols].copy()
                for c in self.cat_cols:
                    Xc[c] = Xc[c].astype(str).fillna("__missing__")
                X_mat = pd.concat([Xn, Xc], axis=1)
                self.clf = CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.05,
                    depth=6,
                    auto_class_weights="Balanced",
                    cat_features=list(range(len(self.num_cols), X_mat.shape[1])),
                    verbose=0,
                    random_state=42,
                )
                self.clf.fit(X_mat, y_fit)
                return self

            def predict_proba(self, X_pred):
                Xn = X_pred[self.num_cols].copy()
                Xc = X_pred[self.cat_cols].copy()
                for c in self.cat_cols:
                    Xc[c] = Xc[c].astype(str).fillna("__missing__")
                X_mat = pd.concat([Xn, Xc], axis=1)
                return self.clf.predict_proba(X_mat)

        return lambda: _CatBoostPrepPipeline()
    except ImportError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["rf", "lgbm", "xgb", "rf_lgbm", "catboost", "all"],
        default="all",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=2, help="RepeatedStratifiedKFold repeats")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = args.out_dir or root / "submissions"

    train, test = load_train_test()
    X_raw, y = split_features_target(train)
    rare = get_rare_cities(X_raw)
    X = prepare_features(X_raw, rare_cities=rare)
    X_test = prepare_features(test, rare_cities=rare)
    ids = test[ID_COL]
    train_rate = base_rate_threshold(y)

    def run_one(name: str, factory) -> None:
        print(f"\n--- {name} ---")
        f1, t = oof_macro_f1(factory, X, y, n_splits=5, n_repeats=args.repeats)
        print(f"OOF macro-F1: {f1:.4f}  threshold={t:.2f}")
        proba_test = fit_predict_proba(factory, X, y, X_test)
        write_submission(ids, proba_test, t, out_dir / f"{name}.csv", "OOF-tuned")
        t_br = threshold_for_target_rate(proba_test, train_rate)
        write_submission(
            ids,
            proba_test,
            t_br,
            out_dir / f"{name}_baserate.csv",
            "base-rate",
        )

    models = []
    if args.model in ("rf", "all"):
        models.append(("rf_oof", lambda: make_rf_pipeline(X)))
    if args.model in ("lgbm", "all"):
        models.append(("lgbm_oof", lambda: make_lgbm_pipeline(X)))
    if args.model in ("xgb", "all"):
        models.append(("xgb_oof", lambda: make_xgb_pipeline(X)))

    for name, factory in models:
        run_one(name, factory)

    if args.model in ("rf_lgbm", "all"):
        print("\n--- rf_lgbm ---")
        oof = (
            oof_predict_proba(lambda: make_rf_pipeline(X), X, y, n_splits=5, n_repeats=args.repeats)
            + oof_predict_proba(lambda: make_lgbm_pipeline(X), X, y, n_splits=5, n_repeats=args.repeats)
        ) / 2
        from src.cv_utils import best_threshold

        t, f = best_threshold(y.values, oof)
        print(f"OOF macro-F1: {f:.4f}  threshold={t:.2f}")
        proba_test = (
            fit_predict_proba(lambda: make_rf_pipeline(X), X, y, X_test)
            + fit_predict_proba(lambda: make_lgbm_pipeline(X), X, y, X_test)
        ) / 2
        write_submission(ids, proba_test, t, out_dir / "rf_lgbm_oof.csv", "OOF-tuned")
        t_br = threshold_for_target_rate(proba_test, train_rate)
        write_submission(ids, proba_test, t_br, out_dir / "rf_lgbm_baserate.csv", "base-rate")

    if args.model in ("catboost", "all"):
        cb_factory = try_catboost_pipeline(X)
        if cb_factory:
            run_one("catboost_oof", cb_factory)
        else:
            print("\n--- catboost: skipped (pip install catboost) ---")


if __name__ == "__main__":
    main()
