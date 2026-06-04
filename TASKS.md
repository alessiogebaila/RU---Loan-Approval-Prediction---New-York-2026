# Team task list

**Deadline:** Friday 5 June 2026, 22:00 (email to instructor)  
**Repo:** https://github.com/floringk/RU---Loan-Approval-Prediction---New-York-2026

Update checkboxes in this file and in `README.md` when a task is done.

---

## Kaggle (competition)

- [ ] Upload `submissions/ensemble_advanced.csv` to [Kaggle competition](https://www.kaggle.com/competitions/ru-loan-approval-prediction-new-york)
- [ ] Record public Macro F1 score in `README.md` score tracker
- [ ] Try a second submission (e.g. `lgbm.csv`) if time allows
- [ ] **Mark 2 best submissions** as final on Kaggle before deadline

---

## Notebooks (course submission)

- [ ] `01_eda.ipynb` — Run All, save outputs, short findings in markdown
- [ ] `02_baseline.ipynb` — Run All, save outputs
- [ ] `03_feature_engineering.ipynb` — Run All, save outputs
- [ ] `04_model_comparison.ipynb` — Run All, save outputs (Optuna cell optional / long)

```powershell
cd ru-loan-approval-ny
pip install -r requirements.txt
jupyter lab
# Kernel → Restart & Run All on each notebook
```

---

## Presentation & report

- [ ] Slides: bank role, data, imbalance, features, models, metrics, 1–2 approve/deny examples
- [ ] Authorship PDF: rows = notebooks, columns = group members
- [ ] Email instructor: slides + notebooks (+ PDF) before **22:00 Friday**

---

## Code / repo (optional improvements)

- [x] Feature engineering (`src/features.py`)
- [x] Multi-model CLI (`src/train.py`)
- [x] Ensemble + target encoding (`src/advanced.py`)
- [ ] Drop constant `State` column in `prepare_features` (NY only)
- [ ] Optuna tuning: `python -m src.advanced --trials 15` (optional; aborted at 20/50)
- [ ] Update README public Kaggle score after upload

---

## Role split (fill in names)

| Task | Owner | Status |
|------|-------|--------|
| Kaggle upload + final 2 picks | | |
| Notebooks Run All | | |
| Slides | | |
| Authorship PDF | | |
| Email package to instructor | | |
