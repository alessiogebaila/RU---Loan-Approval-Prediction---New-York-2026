# Team task list

**Deadline:** Friday 5 June 2026, 22:00 (email to instructor)  
**Repo:** https://github.com/floringk/RU---Loan-Approval-Prediction---New-York-2026

Update checkboxes in this file and in `README.md` when a task is done.

---

## Kaggle (competition)

- [x] Upload `ensemble_advanced.csv` — public score **0.70211**
- [ ] Upload `rf_lgbm_oof.csv` and/or `rf_lgbm_baserate.csv` (improved pipeline)
- [ ] Record new public Macro F1 scores in `README.md`
- [ ] Try `ensemble_baserate.csv` if OOF-threshold underperforms on LB
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

- [x] Feature engineering (`src/features.py`) — State dropped, loan bins, RevLineCr categories
- [x] Multi-model CLI (`src/train.py`)
- [x] Ensemble + target encoding (`src/advanced.py`) — Optuna uses OOF threshold objective
- [x] OOF diagnostics (`src/cv_utils.py`) — adversarial AUC 0.635, moderate drift
- [x] OOF submissions (`src/submit_oof.py`) — base-rate + tuned variants
- [x] Model docs (`docs/MODELS.md`, `docs/DIAGNOSTICS.md`)
- [ ] Optional: `python -m src.advanced --trials 20`

---

## Role split (fill in names)

| Task | Owner | Status |
|------|-------|--------|
| Kaggle upload + final 2 picks | | |
| Notebooks Run All | | |
| Slides | | |
| Authorship PDF | | |
| Email package to instructor | | |
