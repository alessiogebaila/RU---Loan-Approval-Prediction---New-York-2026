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

# Columns dropped unconditionally (identifiers / zero-variance)
_DROP_COLS = (ID_COL, "LoanNr_ChkDgt", "Name", "BalanceGross")

# Raw currency columns → parsed to float
_CURRENCY_COLS = ("DisbursementGross",)

# Raw date strings → year + month integers
_DATE_COLS = ("ApprovalDate", "DisbursementDate")

# Binary flag columns with heterogeneous codes → 0/1 int
_BINARY_FLAG_MAP: dict[str, dict] = {
    # RevLineCr: revolving line of credit. Y→1, everything else→0
    "RevLineCr": {"Y": 1},
    # LowDoc: low-documentation loan programme. Y→1, everything else→0
    "LowDoc": {"Y": 1},
}


# ── I/O ───────────────────────────────────────────────────────────────────────

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_paths(data_dir: str | Path | None = None) -> tuple[Path, Path]:
    root = Path(data_dir) if data_dir else project_root() / "data"
    return root / "train.csv", root / "test_nolabel.csv"


def load_train_test(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path, test_path = data_paths(data_dir)
    # low_memory=False silences the DtypeWarning on ApprovalFY (mixed int/str)
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


# ── Feature engineering ───────────────────────────────────────────────────────

def _parse_currency(series: pd.Series) -> pd.Series:
    """'$350,000.00 ' → 350000.0 (NaN on failure)."""
    return (
        series.astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )


def _parse_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Replace a raw date string column with _year and _month integer columns."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = pd.to_datetime(df[col], errors="coerce", format="mixed")
    df[f"{col}_year"] = dt.dt.year.astype("Int16")
    df[f"{col}_month"] = dt.dt.month.astype("Int8")
    return df.drop(columns=[col])


def _binary_flag(series: pd.Series, yes_values: dict) -> pd.Series:
    """Map a messy flag column to 0/1 int (NaN for truly missing rows)."""
    return series.map(lambda v: yes_values.get(v, 0) if pd.notna(v) else np.nan)


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering applied identically to train and test.

    Steps
    -----
    1. Drop identifier / zero-variance columns.
    2. Parse currency strings → float.
    3. Engineer log-loan-amount and job-ratio features.
    4. Parse date strings → year + month integers; add disbursement lag.
    5. Standardise ApprovalFY to int.
    6. Encode binary flag columns (RevLineCr, LowDoc).
    7. Encode FranchiseCode as a boolean is_franchised flag.
    8. Coerce NewExist to int (1 = existing business, 2 = new).
    9. Cast all remaining object/string columns to uniform str dtype.
    """
    out = X.drop(columns=[c for c in _DROP_COLS if c in X.columns]).copy()

    # ── 2. Currency parsing ───────────────────────────────────────────────────
    for col in _CURRENCY_COLS:
        if col in out.columns:
            out[col] = _parse_currency(out[col])

    # ── 3. Engineered numeric features ───────────────────────────────────────
    if "DisbursementGross" in out.columns:
        # Log-transform reduces right skew; +1 avoids log(0)
        out["log_DisbursementGross"] = np.log1p(out["DisbursementGross"])

    if {"NoEmp", "RetainedJob", "CreateJob"}.issubset(out.columns):
        total_jobs = out["RetainedJob"] + out["CreateJob"]
        # Ratio of jobs affected relative to company size (bounded)
        out["jobs_per_emp"] = (total_jobs / out["NoEmp"].replace(0, np.nan)).clip(
            upper=10
        )

    # ── 4. Date parsing + disbursement lag ───────────────────────────────────
    for col in _DATE_COLS:
        if col in out.columns:
            out = _parse_date_col(out, col)

    # Lag in years between approval and actual disbursement (risk signal)
    if {"ApprovalDate_year", "DisbursementDate_year"}.issubset(out.columns):
        out["disburse_lag_years"] = (
            out["DisbursementDate_year"] - out["ApprovalDate_year"]
        ).clip(lower=0, upper=5)

    # Indicator: approved during / just after the 2008 financial crisis
    if "ApprovalDate_year" in out.columns:
        out["is_crisis_period"] = (
            out["ApprovalDate_year"].between(2007, 2010)
        ).astype("Int8")

    # ── 5. ApprovalFY → int ───────────────────────────────────────────────────
    if "ApprovalFY" in out.columns:
        out["ApprovalFY"] = pd.to_numeric(out["ApprovalFY"], errors="coerce").astype(
            "Int16"
        )

    # ── 6. Binary flag columns ────────────────────────────────────────────────
    for col, yes_vals in _BINARY_FLAG_MAP.items():
        if col in out.columns:
            out[col] = _binary_flag(out[col], yes_vals)

    # ── 7. FranchiseCode → boolean ───────────────────────────────────────────
    # Codes 0 and 1 both mean "not a franchise" in the SBA data dictionary
    if "FranchiseCode" in out.columns:
        out["is_franchised"] = (
            out["FranchiseCode"].isin([0, 1])
            .map({True: 0, False: 1})
            .astype("Int8")
        )
        out = out.drop(columns=["FranchiseCode"])

    # ── 8. NewExist → clean int ───────────────────────────────────────────────
    if "NewExist" in out.columns:
        out["NewExist"] = pd.to_numeric(out["NewExist"], errors="coerce").astype(
            "Int8"
        )

    # ── 9. Uniform string dtype for remaining categoricals ───────────────────
    # sklearn encoders require uniform column types; mixed int/str breaks OHE
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str).replace({"nan": "__missing__", "<NA>": "__missing__"})

    return out


# ── Preprocessor (sklearn) ────────────────────────────────────────────────────

def infer_column_groups(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include=["number", "Int8", "Int16", "Float64"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build an unfitted ColumnTransformer inferred from X column dtypes.
    Fit only on training data; apply identically to test.
    """
    numeric_cols, categorical_cols = infer_column_groups(X)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # max_categories limits explosion from City (2 552 levels)
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
