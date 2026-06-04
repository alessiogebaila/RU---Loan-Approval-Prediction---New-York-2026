# RU — Loan Approval Prediction (New York)

Competition: [Kaggle — ru-loan-approval-prediction-new-york](https://www.kaggle.com/competitions/ru-loan-approval-prediction-new-york)  
Metric: **Macro F1-Score** | Up to 100 Kaggle submissions/day  
Deadline: **Friday 5 June 2026, 22:00** (notebooks + slides + authorship PDF by email)  
Repo: [github.com/floringk/RU---Loan-Approval-Prediction---New-York-2026](https://github.com/floringk/RU---Loan-Approval-Prediction---New-York-2026)

---

## Team tasks (deadline checklist)

See **[TASKS.md](TASKS.md)** for the full list with role split. Summary:

| Priority | Task | Owner | Done |
|----------|------|-------|------|
| P0 | Upload `submissions/ensemble_advanced.csv` to Kaggle | | [x] |
| P0 | Mark **2 best** Kaggle submissions as final | | [ ] |
| P0 | Run All on notebooks `01`–`04`, save outputs | | [ ] |
| P0 | Presentation slides (bank risk narrative) | | [ ] |
| P0 | Authorship PDF (members × notebooks) | | [ ] |
| P0 | Email slides + notebooks + PDF before 22:00 Fri | | [ ] |
| P1 | Record public Kaggle Macro F1 in score table below | | [x] |
| P2 | Optional: `python -m src.advanced --trials 15` | | [ ] |

---

## Project layout

```
ru-loan-approval-ny/
├── data/                    # train/test CSVs (not in git — download locally)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_comparison.ipynb
├── src/
│   ├── features.py          # prepare_features()
│   ├── train.py             # CLI: lr / rf / xgb / lgbm
│   ├── advanced.py          # TargetEncoder + ensemble + Optuna (OOF objective)
│   ├── cv_utils.py          # OOF CV, adversarial validation, time-CV
│   └── submit_oof.py        # OOF submissions + base-rate variants
├── docs/
│   ├── MODELS.md            # Model/method descriptions for report
│   ├── DIAGNOSTICS.md       # Phase 0 drift analysis
│   ├── PRESENTATION_BRIEF.md  # Slide outline + speaker notes
│   ├── BUSINESS_USAGE.md    # Bank use cases + limitations
│   ├── KAGGLE_SUBMISSIONS.md  # Public scores + final picks
│   └── NEXT_STEPS.md        # Post-0.727 improvement backlog
├── submissions/             # Kaggle CSVs (not in git)
├── scripts/setup-kaggle.ps1
├── requirements.txt
├── TASKS.md                 # Team checklist
└── README.md
```

---

## Documentation (slides & report)

| Doc | Use for |
|-----|---------|
| [docs/PRESENTATION_BRIEF.md](docs/PRESENTATION_BRIEF.md) | Slide outline, speaker notes, authorship table template |
| [docs/BUSINESS_USAGE.md](docs/BUSINESS_USAGE.md) | Bank risk narrative, operational use cases |
| [docs/KAGGLE_SUBMISSIONS.md](docs/KAGGLE_SUBMISSIONS.md) | Public scores, which 2 files to mark final |
| [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) | Further model improvements |
| [docs/MODELS.md](docs/MODELS.md) | Technical model/method reference |
| [docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md) | Train/test drift numbers |

---

## Quick start

### 1. Clone and install

```powershell
git clone https://github.com/floringk/RU---Loan-Approval-Prediction---New-York-2026.git
cd RU---Loan-Approval-Prediction---New-York-2026
pip install -r requirements.txt
```

### 2. Download competition data

```powershell
kaggle competitions download -c ru-loan-approval-prediction-new-york -p data
Expand-Archive data\ru-loan-approval-prediction-new-york.zip -DestinationPath data -Force
```

### 3. Diagnose train/test drift (Phase 0)

```powershell
python -m src.cv_utils
```

### 4. Generate improved submissions

```powershell
python -m src.advanced --no-optuna
# → ensemble_improved.csv, ensemble_baserate.csv

python -m src.submit_oof --model rf_lgbm --repeats 1
# → rf_lgbm_oof.csv, rf_lgbm_baserate.csv
```

### 5. Upload to Kaggle

```powershell
kaggle competitions submit -c ru-loan-approval-prediction-new-york `
  -f "submissions\rf_lgbm_oof.csv" -m "RF+LGBM OOF improved"

kaggle competitions submit -c ru-loan-approval-prediction-new-york `
  -f "submissions\ensemble_baserate.csv" -m "RF+LGBM base-rate 79.6pct"
```

---

## Score tracker

| Run | OOF Macro F1 | Public Macro F1 | Submission file |
|-----|-------------|-----------------|-----------------|
| LR baseline | 0.6452 | — | `baseline.csv` |
| Ensemble RF+LGBM+XGB (first upload) | ~0.745 | **0.70211** | `ensemble_advanced.csv` |
| **RF+LGBM OOF (improved features)** | **0.7452** | **0.72694** | **`rf_lgbm_oof.csv`** ← best |
| RF+LGBM base-rate (~79.6% approve) | — | 0.63594 | `rf_lgbm_baserate.csv` (do not final) |
| Ensemble OOF | 0.7440 | — | `ensemble_improved.csv` |
| Ensemble base-rate | — | — | `ensemble_baserate.csv` |
| Optuna (OOF threshold objective) | — | — | `python -m src.advanced --trials 20` |

Details: [docs/KAGGLE_SUBMISSIONS.md](docs/KAGGLE_SUBMISSIONS.md)

---

## Assignment alignment

| Course requirement | Status |
|--------------------|--------|
| Classifier: grant vs deny (`Accept`) | Done |
| Same preprocessing on train and test | `prepare_features()` |
| No test rows dropped | 7 050 predictions |
| CSV format: `id`, `Accept` (int) | Done |
| Compete on Macro F1 | Public **0.72694** (`rf_lgbm_oof.csv`) |
| Explain bank risk (why grant/deny) | [docs/BUSINESS_USAGE.md](docs/BUSINESS_USAGE.md) + slides — **todo** |
| 2 final Kaggle picks | **todo** |
| Slides + notebooks + authorship PDF | **todo** |

Submission matches the course example:

```python
submission = test[["id"]].copy()
submission["Accept"] = preds.astype(int)
submission.to_csv("submissions/ensemble_advanced.csv", index=False)
```

---

## Model pipeline (implemented)

1. **Features** (`src/features.py`) — currency/dates, flags, `log_DisbursementGross`, job ratio, crisis period  
2. **Encoding** (`src/advanced.py`) — `TargetEncoder` on `Bank` and `City`  
3. **Models** — RF + LightGBM + XGBoost soft-vote ensemble  
4. **Threshold** — tuned on validation (~0.43) for Macro F1  

---

## Publish to GitHub

From `ru-loan-approval-ny/` (only code and notebooks are tracked; not `data/` or `submissions/`):

```powershell
cd "c:\Users\florin.ostafe\RU - Loan Approval Prediction - New York 2026\ru-loan-approval-ny"

git status
git add README.md TASKS.md docs/ src/ notebooks/ requirements.txt scripts/ .gitignore data/.gitkeep submissions/.gitkeep
git commit -m "docs: presentation brief, business usage, Kaggle scores, next steps"

git remote -v
# should show: origin  https://github.com/floringk/RU---Loan-Approval-Prediction---New-York-2026.git

git push origin master
```

First-time setup (if remote missing):

```powershell
git remote add origin https://github.com/floringk/RU---Loan-Approval-Prediction---New-York-2026.git
git push -u origin master
```

**Do not commit:** `data/*.csv`, `submissions/*.csv`, `.kaggle/` tokens (see `.gitignore`).

---

## Course deliverables (due Friday 5 June 22:00)

- [ ] **Presentation slides** — use [docs/PRESENTATION_BRIEF.md](docs/PRESENTATION_BRIEF.md)  
- [ ] **Notebooks** — `01`–`04` with all cells executed  
- [ ] **Authorship PDF** — table: rows = notebooks, columns = group members  
- [ ] **2 Kaggle submissions marked** as final  

---

## Optional improvements

| Task | Command / location |
|------|------------------|
| Faster Optuna | `python -m src.advanced --trials 15` |
| Drop `State` (constant NY) | `src/features.py` |
| SMOTE in CV | `notebooks/04_model_comparison.ipynb` |
| Loan-size bins | `src/features.py` |
