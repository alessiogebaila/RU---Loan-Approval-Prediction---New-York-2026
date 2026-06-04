# Presentation brief — slide deck guide

Hand this file to whoever builds the course slides. Technical detail lives in [MODELS.md](MODELS.md) and [DIAGNOSTICS.md](DIAGNOSTICS.md); business framing in [BUSINESS_USAGE.md](BUSINESS_USAGE.md).

**Target length:** 12–15 slides, ~10–12 minutes  
**Audience:** Course assessors + bank-risk narrative (non-technical stakeholders on slides 2, 12–13)

---

## Key numbers to put on slides

| Metric | Value | Source |
|--------|-------|--------|
| Training approval rate | ~79.6% | EDA / train target |
| LR baseline OOF Macro F1 | 0.6452 | `python -m src.train --model lr` |
| First public Kaggle score | **0.70211** | `ensemble_advanced.csv` |
| Best public Kaggle score | **0.72694** | `rf_lgbm_oof.csv` |
| Best OOF Macro F1 (RF+LGBM) | **0.7452** | `python -m src.submit_oof --model rf_lgbm` |
| Base-rate public (failed) | 0.63594 | `rf_lgbm_baserate.csv` — do not present as final |
| Adversarial train vs test AUC | 0.6355 | `python -m src.cv_utils` |
| Unseen banks in test | 14 (7.7%) | DIAGNOSTICS.md |
| Unseen cities in test | 158 (13.7%) | DIAGNOSTICS.md |
| Test predictions (rows) | 7 050 | All submissions |

**Story arc for slide 10:** 0.702 → 0.727 public Macro F1 after drift diagnosis, feature engineering, OOF threshold tuning, and RF+LGBM ensemble.

---

## Recommended slide outline

| # | Title | What to show | Repo / asset |
|---|--------|--------------|--------------|
| 1 | Title & team | Project name, members, RU + Kaggle link | README |
| 2 | Business problem | Grant vs deny; SBA-style small-business loans in NY; why banks care | [BUSINESS_USAGE.md](BUSINESS_USAGE.md) § Problem |
| 3 | Data overview | ~28k train, 7k test; target `Accept`; key columns | `notebooks/01_eda.ipynb` |
| 4 | Class imbalance | ~80% granted in train; Macro F1 treats grant and deny equally | EDA notebook |
| 5 | Risk factors (EDA) | Loan size, LowDoc, franchise, crisis years, jobs | `01_eda` plots |
| 6 | Feature engineering | Log amount, loan/employee, size bins, dates, flags; drop constant `State` | [MODELS.md](MODELS.md) table |
| 7 | Encoding | TargetEncoder on Bank/City (CV, smooth=25); unseen levels risk | `03_feature_engineering` + DIAGNOSTICS |
| 8 | Models compared | LR → RF → LGBM → ensemble; OOF scores | `04_model_comparison.ipynb` |
| 9 | Validation & drift | OOF + threshold tuning; adversarial AUC 0.64; time-CV as check | [DIAGNOSTICS.md](DIAGNOSTICS.md) |
| 10 | Kaggle progress | Table or screenshot: 0.702 → **0.727** | [KAGGLE_SUBMISSIONS.md](KAGGLE_SUBMISSIONS.md) |
| 11 | Final system | RF + LGBM soft vote → threshold ~0.41 → CSV; 2 finals marked | `src/submit_oof.py` |
| 12 | Example companies | 2–3 rows: features → probability → grant/deny + plain-language “why” | Pick from EDA / manual examples |
| 13 | Limitations | Drift, unseen bank/city, threshold ≠ business profit | BUSINESS_USAGE § Limitations |
| 14 | Conclusion & deliverables | Score, repo, notebooks Run All, authorship PDF | TASKS.md |

---

## Speaker notes (copy to presenter notes)

**Slide 2 — Metric**  
“We optimize **Macro F1**: average of F1 for grant and F1 for deny. Missing rare denials hurts as much as missing grants, unlike accuracy on an 80% approval dataset.”

**Slide 9 — Why we trust OOF**  
“Out-of-fold predictions mimic scoring unseen data. We tune the decision threshold on OOF probabilities, not on 0.5, because the classes are imbalanced.”

**Slide 9 — Drift**  
“A model can tell train rows from test rows (ROC-AUC **0.64**). Loan size, approval year, and bank differ — so public leaderboard can be below cross-validation.”

**Slide 10 — Improvement**  
“First upload: three-model ensemble with target encoding, **0.702**. After planned improvements (features, OOF RF+LGBM, tuned threshold), **0.727** on the public leaderboard.”

**Slide 11 — Base-rate lesson**  
“Forcing ~80% approvals to match training scored **0.636** on Kaggle. The test set needs a lower cutoff (~68% predicted approvals) for Macro F1.”

**Slide 12 — Examples**  
Use concrete fields: e.g. large `DisbursementGross`, `LowDoc=Y`, rare `City` → lower approval probability → deny or manual review.

---

## Visuals checklist

- [ ] Bar chart: approval rate train vs predicted test (ensemble vs base-rate)
- [ ] Line or bar: model progression (LR 0.65 → RF/LGBM ~0.74 OOF → public 0.73)
- [ ] Screenshot: Kaggle Submissions page (three recent uploads)
- [ ] One confusion-matrix-style slide (grant/deny) from validation — optional
- [ ] Feature importance or simple rules for “example companies” slide

---

## Authorship PDF (course requirement)

| Notebook | Suggested content | Who did it (fill in) |
|----------|-------------------|----------------------|
| `01_eda.ipynb` | Distributions, target, missingness | |
| `02_baseline.ipynb` | Logistic regression baseline | |
| `03_feature_engineering.ipynb` | Features + encoding narrative | |
| `04_model_comparison.ipynb` | RF / LGBM / XGB / ensemble | |
| `src/` (optional row) | `features.py`, `advanced.py`, `submit_oof.py` | |

Columns = group member names. One checkmark or initials per cell.

---

## Files to run before exporting slides

```powershell
cd ru-loan-approval-ny
jupyter nbconvert --execute notebooks/01_eda.ipynb --inplace
jupyter nbconvert --execute notebooks/02_baseline.ipynb --inplace
jupyter nbconvert --execute notebooks/03_feature_engineering.ipynb --inplace
jupyter nbconvert --execute notebooks/04_model_comparison.ipynb --inplace
```

Or **Run All** in Jupyter and save. Slides should reference plots that exist in the notebooks.

---

## Suggested 2 Kaggle finals (mark on competition site)

| Slot | File | Public Macro F1 | Rationale |
|------|------|-----------------|-----------|
| 1 | `rf_lgbm_oof.csv` | **0.72694** | Best public; OOF-aligned pipeline |
| 2 | `ensemble_advanced.csv` | 0.70211 | Backup until a newer upload beats 0.727 |

Do **not** mark `rf_lgbm_baserate.csv` as final.
