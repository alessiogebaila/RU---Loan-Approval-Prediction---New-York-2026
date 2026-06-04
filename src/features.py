"""Shared feature definitions and sklearn preprocessing pipeline."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Accept"
ID_COL = "id"

# Columns dropped unconditionally (identifiers / zero-variance / constant)
_DROP_COLS = (ID_COL, "LoanNr_ChkDgt", "Name", "BalanceGross", "State")

# Raw currency columns → parsed to float
_CURRENCY_COLS = ("DisbursementGross",)

# Raw date strings → year + month integers
_DATE_COLS = ("ApprovalDate", "DisbursementDate")

# LowDoc: Y→1, else 0 (RevLineCr kept as multi-category string)
_BINARY_FLAG_MAP: dict[str, dict] = {
    "LowDoc": {"Y": 1},
}

# RevLineCr codes preserved as categories (Y, N, 0, T)
_REVLINE_MAP = {"Y": "Y", "N": "N", "0": "0", "T": "T"}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_paths(data_dir: str | Path | None = None) -> tuple[Path, Path]:
    root = Path(data_dir) if data_dir else project_root() / "data"
    return root / "train.csv", root / "test_nolabel.csv"


def load_train_test(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path, test_path = data_paths(data_dir)
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    return train, test


def split_features_target(
    train: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in train.columns:
        raise KeyError(
            f"Target column {target_col!r} not in train columns: {list(train.columns)}"
        )
    y = train[target_col]
    X = train.drop(columns=[target_col])
    return X, y


def _parse_currency(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )


def _parse_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = pd.to_datetime(df[col], errors="coerce", format="mixed")
    df[f"{col}_year"] = dt.dt.year.astype("Int16")
    df[f"{col}_month"] = dt.dt.month.astype("Int8")
    return df.drop(columns=[col])


def _binary_flag(series: pd.Series, yes_values: dict) -> pd.Series:
    return series.map(lambda v: yes_values.get(v, 0) if pd.notna(v) else np.nan)


def _encode_revline(series: pd.Series) -> pd.Series:
    """Preserve RevLineCr codes as uniform string categories."""
    return (
        series.astype(str)
        .map(lambda v: _REVLINE_MAP.get(v, "__other__") if v not in ("nan", "<NA>") else "__missing__")
    )


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering applied identically to train and test.
    """
    out = X.drop(columns=[c for c in _DROP_COLS if c in X.columns]).copy()

    for col in _CURRENCY_COLS:
        if col in out.columns:
            out[col] = _parse_currency(out[col])

    if "DisbursementGross" in out.columns and "NoEmp" in out.columns:
        out["loan_per_emp"] = (
            out["DisbursementGross"] / out["NoEmp"].replace(0, np.nan)
        ).clip(upper=1e7)

    if "DisbursementGross" in out.columns:
        out["log_DisbursementGross"] = np.log1p(out["DisbursementGross"])
        try:
            out["loan_size_bin"] = (
                pd.qcut(
                    out["log_DisbursementGross"],
                    q=5,
                    labels=False,
                    duplicates="drop",
                )
                .astype("float")
            )
        except ValueError:
            out["loan_size_bin"] = 0.0
        out = out.drop(columns=["DisbursementGross"])

    if {"NoEmp", "RetainedJob", "CreateJob"}.issubset(out.columns):
        total_jobs = out["RetainedJob"] + out["CreateJob"]
        out["jobs_per_emp"] = (total_jobs / out["NoEmp"].replace(0, np.nan)).clip(
            upper=10
        )

    for col in _DATE_COLS:
        if col in out.columns:
            out = _parse_date_col(out, col)

    if {"ApprovalDate_year", "DisbursementDate_year"}.issubset(out.columns):
        out["disburse_lag_years"] = (
            out["DisbursementDate_year"] - out["ApprovalDate_year"]
        ).clip(lower=0, upper=5)

    if "ApprovalDate_year" in out.columns:
        out["is_crisis_period"] = (
            out["ApprovalDate_year"].between(2007, 2010)
        ).astype("Int8")

    if "ApprovalFY" in out.columns:
        out = out.drop(columns=["ApprovalFY"])

    if "RevLineCr" in out.columns:
        out["RevLineCr"] = _encode_revline(out["RevLineCr"])

    for col, yes_vals in _BINARY_FLAG_MAP.items():
        if col in out.columns:
            out[col] = _binary_flag(out[col], yes_vals)

    if "FranchiseCode" in out.columns:
        out["is_franchised"] = (
            out["FranchiseCode"].isin([0, 1])
            .map({True: 0, False: 1})
            .astype("Int8")
        )
        out = out.drop(columns=["FranchiseCode"])

    if "NewExist" in out.columns:
        ne = pd.to_numeric(out["NewExist"], errors="coerce")
        ne = ne.where(ne != 0, np.nan)
        out["NewExist"] = ne.fillna(ne.median()).astype("Int8")

    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str).replace(
            {"nan": "__missing__", "<NA>": "__missing__"}
        )

    return out


def infer_column_groups(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include=["number", "Int8", "Int16", "Float64"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = infer_column_groups(X)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                max_categories=50,
            ),
        ),
    ])

    transformers: list = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No feature columns found after excluding id.")

    return ColumnTransformer(transformers=transformers)
