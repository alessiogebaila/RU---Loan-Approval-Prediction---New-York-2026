# Next steps — model improvement after 0.727 public

Prioritized backlog after **rf_lgbm_oof.csv** reached **0.72694** public Macro F1 (OOF **0.7452**). Gap OOF ↔ public ≈ **0.018** (improved from ~0.04).

---

## Done (improvement plan)

- [x] Adversarial validation + drift report → [DIAGNOSTICS.md](DIAGNOSTICS.md)
- [x] OOF CV, time-CV, base-rate threshold variant
- [x] Optuna objective fixed (OOF + tuned threshold)
- [x] Features: drop `State`, loan/emp, size bins, `NewExist`, RevLineCr categories, City smoothing
- [x] Model docs → [MODELS.md](MODELS.md)
- [x] Submissions: `rf_lgbm_oof`, `rf_lgbm_baserate`, ensemble variants

---

## Before deadline (high priority)

| # | Action | Command / location | Expected benefit |
|---|--------|-------------------|------------------|
| 1 | Mark **2 Kaggle finals** | Kaggle UI | `rf_lgbm_oof` + `ensemble_advanced` (see [KAGGLE_SUBMISSIONS.md](KAGGLE_SUBMISSIONS.md)) |
| 2 | Optuna-tuned LGBM + ensemble | `python -m src.advanced --trials 20` | May beat 0.727 on public |
| 3 | Repeated OOF for stabler threshold | `python -m src.submit_oof --model all --repeats 2` | Less threshold overfit |
| 4 | Update README score table after each upload | README.md | Team traceability |
| 5 | Course deliverables | [PRESENTATION_BRIEF.md](PRESENTATION_BRIEF.md), notebooks Run All, authorship PDF | Grading |

---

## Model improvements (if time remains)

### Close OOF ↔ public gap (~0.018)

| Idea | Effort | Notes |
|------|--------|-------|
| **Unseen Bank/City fallback** | Medium | Global mean + `BankState` when level absent in train |
| **Rare City → `OTHER` bucket** | Low | Reduces noisy target encoding; 158 unseen cities in test |
| **Stronger City smoothing** | Low | Increase `smooth` in `TargetEncoder` |
| **3-way RF+LGBM+XGB** with improved features | Low | Re-run `submit_oof` / `advanced` |
| **CatBoost** on raw categoricals | Medium | `pip install catboost`; optional path in plan |
| **CalibratedClassifierCV** before blend | Medium | Optional; may help under drift |
| **Time-based sample weights** | Medium | Up-weight recent `ApprovalDate_year` if time-CV lags |

### Do not repeat

- **Base-rate threshold (~79.6% approvals)** for Kaggle finals — public **0.63594** on `rf_lgbm_baserate.csv`.
- Trusting OOF alone without a leaderboard check when drift is moderate.

---

## After the competition (optional)

- SHAP / permutation importance for slide examples
- Error analysis: false grants vs false denies on validation slice
- Fairness-style breakdown: predicted approval rate by `Bank` / `City` bucket

---

## Commands quick reference

```powershell
cd "c:\Users\florin.ostafe\RU - Loan Approval Prediction - New York 2026\ru-loan-approval-ny"

# Diagnostics
python -m src.cv_utils

# Best current pipeline
python -m src.submit_oof --model rf_lgbm --repeats 1

# All OOF variants + base-rate
python -m src.submit_oof --model all --repeats 2

# Ensemble + Optuna
python -m src.advanced --no-optuna
python -m src.advanced --trials 20

# Kaggle upload
kaggle competitions submit -c ru-loan-approval-prediction-new-york `
  -f "submissions\rf_lgbm_oof.csv" -m "RF+LGBM OOF improved features"
```

---

## Success criteria

| Goal | Target |
|------|--------|
| Public Macro F1 | Beat **0.727** or lock best as final |
| OOF ↔ public gap | Stay under ~0.02 |
| Course | 2 finals marked, slides + notebooks + PDF emailed |
