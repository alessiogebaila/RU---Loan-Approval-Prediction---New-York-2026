# Models and methods — reference for report / slides

Related docs: [PRESENTATION_BRIEF.md](PRESENTATION_BRIEF.md) · [BUSINESS_USAGE.md](BUSINESS_USAGE.md) · [DIAGNOSTICS.md](DIAGNOSTICS.md) · [KAGGLE_SUBMISSIONS.md](KAGGLE_SUBMISSIONS.md) · [NEXT_STEPS.md](NEXT_STEPS.md)

## Problem framing

Binary classification: predict `Accept` (1 = loan granted, 0 = denied) for small-business loan applications. Metric: **Macro F1** (equal weight on grant and deny classes). Training data is imbalanced (~80% granted).

---

## Feature engineering (`src/features.py`)

| Step | Method | Description |
|------|--------|-------------|
| Drop IDs / noise | Column filter | Remove `id`, `Name`, `LoanNr_ChkDgt`, constant `BalanceGross`, constant `State` (NY only). |
| Currency parsing | Regex + `to_numeric` | `DisbursementGross` from `"$350,000.00"` to float. |
| Log loan amount | `log1p` | `log_DisbursementGross` reduces right skew of loan sizes. |
| Loan per employee | Ratio | `loan_per_emp = DisbursementGross / NoEmp` (business size context). |
| Loan size bins | `pd.qcut` (5 bins) | Ordinal bucket on log amount. |
| Dates | `pd.to_datetime` | `ApprovalDate` / `DisbursementDate` → year + month. |
| Disbursement lag | Difference | Years between approval and disbursement (processing risk). |
| Crisis flag | Rule | `is_crisis_period` if approved 2007–2010. |
| RevLineCr | Category preserve | Keep codes Y/N/0/T instead of collapsing to binary. |
| LowDoc | Binary flag | Y → 1, else 0. |
| Franchise | Rule | `is_franchised` from SBA franchise codes. |
| NewExist | Cleaning | Values 1/2 kept; 0 treated as missing. |

---

## Encoding (`src/advanced.py`)

**TargetEncoder** (sklearn, `cv=5`, `smooth=25`): replaces `Bank` and `City` with smoothed historical approval rates. Cross-fitting inside each fold prevents target leakage. Higher smoothing for `City` (2552 levels) reduces noise on rare cities.

**OneHotEncoder** (`max_categories=30`): remaining low-cardinality strings (e.g. `BankState`, `RevLineCr`).

---

## Models

### Logistic Regression
Linear baseline on scaled numeric + one-hot features. `class_weight='balanced'` handles imbalance. Fast and interpretable; weaker on non-linear patterns.

### Random Forest
Ensemble of bagged decision trees (400 trees, `min_samples_leaf=5`). `class_weight='balanced'`. Robust, low variance; strong OOF on this dataset (~0.74).

### LightGBM
Gradient-boosted decision trees (leaf-wise growth). `is_unbalance=True` up-weights the minority class. Fast training; top single-model performer.

### XGBoost
Gradient boosting (level-wise). Imbalance handled via **threshold tuning** (not `scale_pos_weight` here). Often needs a higher decision threshold (~0.66 OOF).

### CatBoost (optional)
Gradient boosting with native categorical handling. Useful when `Bank`/`City` are left as raw categories; requires `pip install catboost`.

---

## Validation methods

| Method | Purpose |
|--------|---------|
| **Stratified K-Fold OOF** | Out-of-fold probabilities; honest Macro F1 estimate. |
| **Repeated Stratified K-Fold** | 5 folds × 2 repeats; stabler threshold selection. |
| **Time-based CV** | Train on earlier `ApprovalDate_year`, validate on later years; used when train/test drift detected. |
| **Adversarial validation** | Classifier to separate train vs test rows; ROC-AUC > 0.6 suggests distribution shift. |

---

## Threshold tuning

Default 0.5 is **not** optimal for Macro F1 on imbalanced data. We grid-search thresholds on OOF probabilities (0.05–0.95) to maximize Macro F1.

**Base-rate threshold**: set cutoff so predicted approval rate ≈ training rate (~79.6%). Robust fallback when OOF threshold overfits.

---

## Ensembling

**Soft-vote ensemble**: average predicted probabilities from RF + LightGBM (optionally + XGBoost), weights proportional to OOF Macro F1. Apply tuned threshold to the blended probability.

---

## Scripts

| Command | Output |
|---------|--------|
| `python -m src.cv_utils` | Phase 0 diagnostics (drift, temporal, coverage) |
| `python -m src.submit_oof --model all` | OOF submissions + base-rate variants |
| `python -m src.advanced --no-optuna` | `ensemble_improved.csv`, `ensemble_baserate.csv` |
| `python -m src.advanced --trials 20` | Optuna with OOF-threshold objective |

---

## Kaggle submission

Upload CSV with columns `id`, `Accept` (integer). Recommended candidates after improvements:

- `submissions/ensemble_improved.csv` — RF+LGBM, OOF threshold
- `submissions/ensemble_baserate.csv` — RF+LGBM, training approval rate
- `submissions/lgbm_oof.csv` / `submissions/rf_lgbm_oof.csv` — from `submit_oof`

Mark your **2 best** public scores as final before the deadline.
