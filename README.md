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
| P0 | Upload `submissions/ensemble_advanced.csv` to Kaggle | | [ ] |
| P0 | Mark **2 best** Kaggle submissions as final | | [ ] |
| P0 | Run All on notebooks `01`–`04`, save outputs | | [ ] |
| P0 | Presentation slides (bank risk narrative) | | [ ] |
| P0 | Authorship PDF (members × notebooks) | | [ ] |
| P0 | Email slides + notebooks + PDF before 22:00 Fri | | [ ] |
| P1 | Record public Kaggle Macro F1 in score table below | | [ ] |
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
│   └── advanced.py          # TargetEncoder + ensemble + Optuna
├── submissions/             # Kaggle CSVs (not in git)
├── scripts/setup-kaggle.ps1
├── requirements.txt
├── TASKS.md                 # Team checklist
└── README.md
```

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

### 3. Generate best submission (ensemble)

```powershell
python -m src.advanced --no-optuna
# → submissions/ensemble_advanced.csv
```

### 4. Single-model alternatives

```powershell
python -m src.train --model lgbm --output submissions/lgbm.csv
python -m src.train --model rf   --output submissions/rf.csv
```

---

## Score tracker

| Run | CV Macro F1 | Val Macro F1 | Command | Submission |
|-----|-------------|-------------|---------|------------|
| LR baseline | 0.6452 | — | `python -m src.train --model lr` | `baseline.csv` |
| RF | 0.7342 | 0.7445 | `python -m src.train --model rf` | `rf.csv` |
| LightGBM | 0.7291 | 0.7505 | `python -m src.train --model lgbm` | `lgbm.csv` |
| **Ensemble (recommended)** | **0.7441** | **0.7542** | `python -m src.advanced --no-optuna` | **`ensemble_advanced.csv`** |
| Optuna 50 trials | 0.7371 @ trial 12 | — | `python -m src.advanced --trials 50` | aborted (20/50) |
| **Kaggle public score** | — | — | upload above | _fill in after submit_ |

---

## Assignment alignment

| Course requirement | Status |
|--------------------|--------|
| Classifier: grant vs deny (`Accept`) | Done |
| Same preprocessing on train and test | `prepare_features()` |
| No test rows dropped | 7 050 predictions |
| CSV format: `id`, `Accept` (int) | Done |
| Compete on Macro F1 | Local val 0.7542; upload for public LB |
| Explain bank risk (why grant/deny) | Slides + notebook narrative — **todo** |
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
git add README.md TASKS.md src/ notebooks/ requirements.txt scripts/ .gitignore data/.gitkeep submissions/.gitkeep
git commit -m "docs: add team task list, update README and publish instructions"

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

- [ ] **Presentation slides** — bank risk analysis; features; model; example companies  
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
