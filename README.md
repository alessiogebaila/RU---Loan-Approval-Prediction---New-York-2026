# RU — Loan Approval Prediction (New York)

Competition: [Kaggle — ru-loan-approval-prediction-new-york](https://www.kaggle.com/competitions/ru-loan-approval-prediction-new-york)  
Metric: **Macro F1-Score** | Up to 100 Kaggle submissions/day  
Deadline: **Friday 5 June 2026, 22:00** (notebooks + slides + authorship PDF)

---

## Project layout

```
ru-loan-approval-ny/
├── data/
│   ├── train.csv            # 40 385 rows × 21 cols  (not in git)
│   ├── test_nolabel.csv     # 7 050 rows × 20 cols   (not in git)
│   └── sample_submission.csv
├── notebooks/
│   ├── 01_eda.ipynb                 # Target balance, nulls, cardinality, leakage
│   ├── 02_baseline.ipynb            # Logistic Regression → first submission
│   ├── 03_feature_engineering.ipynb # All transformations documented
│   └── 04_model_comparison.ipynb    # RF / XGB / LGBM + threshold tuning
├── src/
│   ├── features.py   # prepare_features(), build_preprocessor()
│   └── train.py      # CLI trainer — supports lr / rf / xgb / lgbm
├── submissions/       # Generated CSVs for Kaggle (not in git)
├── models/            # Saved .joblib artefacts (not in git)
├── requirements.txt
└── scripts/
    └── setup-kaggle.ps1
```

---

## Quick start

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Download competition data

```powershell
kaggle competitions download -c ru-loan-approval-prediction-new-york -p data
Expand-Archive data\ru-loan-approval-prediction-new-york.zip -DestinationPath data -Force
```

### 3. Run baseline (Logistic Regression)

```powershell
python -m src.train --model lr --output submissions/lr.csv
```

### 4. Run best model (LightGBM + threshold tuning)

```powershell
python -m src.train --model lgbm --output submissions/lgbm_tuned.csv
```

Available `--model` options: `lr`, `rf`, `xgb`, `lgbm`

---

## Improvement roadmap

### Phase 1 — Feature Engineering (`src/features.py`)

| Task | Detail | Status |
|------|--------|--------|
| Drop identifiers | `id`, `LoanNr_ChkDgt`, `Name` | ✅ Done |
| Drop zero-variance | `BalanceGross` is `$0.00` for all rows | ✅ Done |
| Parse currency | `DisbursementGross` `"$350,000.00"` → `350000.0` | ✅ Done |
| Log-transform loan amount | `log1p(DisbursementGross)` reduces right skew | ✅ Done |
| Parse dates | `ApprovalDate`, `DisbursementDate` → `_year`, `_month` | ✅ Done |
| Disbursement lag | `DisbursementDate_year - ApprovalDate_year` (risk signal) | ✅ Done |
| Crisis indicator | `is_crisis_period` = 1 if approved 2007–2010 | ✅ Done |
| Binary flag: RevLineCr | `Y → 1`, all other codes → 0 | ✅ Done |
| Binary flag: LowDoc | `Y → 1`, all other codes → 0 | ✅ Done |
| FranchiseCode → bool | `is_franchised = 1` if code not in {0, 1} | ✅ Done |
| Job ratio | `(RetainedJob + CreateJob) / NoEmp` clipped at 10 | ✅ Done |
| OHE max_categories | Limit City (2 552 levels) to top 50 to avoid dimension explosion | ✅ Done |

**Next feature ideas (not yet implemented):**

| Task | Detail |
|------|--------|
| Target-encode Bank | Bank approval rate computed on train fold only (use `sklearn.TargetEncoder`) |
| Target-encode City | Same — city-level approval rate |
| State stays as-is | Only 1 unique value (NY) → drop it |
| Loan-size bins | `pd.qcut(DisbursementGross, 5)` → ordinal bucket feature |
| Sector proxy | `FranchiseCode` detail maps to SIC sector (requires external lookup) |

---

### Phase 2 — Model Selection (`src/train.py`)

| Model | Scikit-learn compatible | Expected CV F1 | Notes |
|-------|------------------------|---------------|-------|
| Logistic Regression | Yes | ~0.65 | Baseline, fast |
| Random Forest | Yes | ~0.70–0.73 | Good generalisation |
| **XGBoost** | Yes (wrapper) | ~0.72–0.76 | `scale_pos_weight=4` for imbalance |
| **LightGBM** | Yes (wrapper) | ~0.72–0.76 | `is_unbalance=True`, fastest |
| CatBoost | Yes (wrapper) | ~0.73–0.77 | Best for high-cardinality cats |

Run all four and compare: `notebooks/04_model_comparison.ipynb`

**Class imbalance** (80 % Accept=1, 20 % Accept=0):
- Tree models: use `is_unbalance=True` / `scale_pos_weight`
- Alternative: SMOTE (`imblearn.over_sampling.SMOTE`) applied inside CV folds only

---

### Phase 3 — Threshold Tuning

Default threshold (0.5) is not optimal for Macro F1. The minority class (denied loans) requires a lower threshold to improve recall.

```python
# Already implemented in src/train.py::_best_threshold()
thresholds = np.linspace(0.1, 0.9, 81)
best_t = max(thresholds, key=lambda t: f1_macro(y_val, proba_val >= t))
```

Typical optimal range: **0.30–0.45** for this dataset.

---

### Phase 4 — Hyperparameter Tuning (Optuna)

```python
# Already scaffolded in notebooks/04_model_comparison.ipynb, cell 7
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

Key LightGBM parameters to search:

| Parameter | Search range |
|-----------|-------------|
| `n_estimators` | 200–1000 |
| `learning_rate` | 0.01–0.2 (log) |
| `num_leaves` | 15–127 |
| `subsample` | 0.5–1.0 |
| `colsample_bytree` | 0.5–1.0 |
| `reg_alpha` | 1e-4–10 (log) |
| `reg_lambda` | 1e-4–10 (log) |
| `min_child_samples` | 5–100 |

Expected gain over default LightGBM: **+1–3 pp Macro F1**.

---

### Phase 5 — Ensembling

```python
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(
    estimators=[('rf', rf_pipe), ('xgb', xgb_pipe), ('lgbm', lgbm_pipe)],
    voting='soft',
)
```

Soft voting on predicted probabilities before applying the tuned threshold.

---

## Submission workflow

```powershell
# 1. Generate submission
python -m src.train --model lgbm --output submissions/lgbm_tuned.csv

# 2. Upload on Kaggle website → Submit Predictions
# 3. Record public Macro F1 score
# 4. Before deadline: mark your 2 best submissions as final
```

---

## Course deliverables (due Friday 5 June 22:00)

- [ ] **Presentation slides** — frame as "bank risk analysis"; explain features, model choice, and which loans are risky and why
- [ ] **Notebooks** — 01 through 04, all cells run with output
- [ ] **Authorship PDF** — table: rows = notebooks, columns = group members
- [ ] **2 Kaggle submissions marked** as your final entries

---

## Git workflow

```powershell
# From ru-loan-approval-ny/
git add .
git commit -m "feat: add LightGBM model and feature engineering"
git remote add origin <your-github-url>
git push -u origin main
```

`data/` and `submissions/` are in `.gitignore` — only code and notebooks are tracked.
