# Kaggle submissions — interpreted results

Public leaderboard scores for the team. CSV files stay in `submissions/` (gitignored); only this summary is in the repo.

Competition: [ru-loan-approval-prediction-new-york](https://www.kaggle.com/competitions/ru-loan-approval-prediction-new-york)  
Metric: **Macro F1** (higher is better)

---

## Leaderboard history

| Date (approx) | File | Message | Public Macro F1 | OOF Macro F1 | Pred. approval % | Notes |
|---------------|------|---------|-----------------|--------------|------------------|-------|
| First upload | `ensemble_advanced.csv` | Ensemble RF+LGBM+XGB + target encoding | **0.70211** | ~0.745 (3-way) | ~71.7% | Previous best |
| Improved | `rf_lgbm_oof.csv` | RF+LGBM OOF improved features | **0.72694** | **0.7452** | ~67.8% | **Current best — mark as final #1** |
| Experiment | `rf_lgbm_baserate.csv` | RF+LGBM base-rate 79.6% | **0.63594** | — | ~79.6% | Failed; do not mark final |

**Lift:** 0.70211 → 0.72694 = **+0.02483** Macro F1 on the public leaderboard.

---

## Recommended finals (2/2 on Kaggle site)

| Slot | File | Public score | Why |
|------|------|--------------|-----|
| **Final 1** | `rf_lgbm_oof.csv` | 0.72694 | Best public; matches improved feature + OOF pipeline |
| **Final 2** | `ensemble_advanced.csv` | 0.70211 | Solid backup until a newer run beats 0.727 |

Replace Final 2 if `python -m src.advanced --trials 20` or a new `submit_oof` upload scores higher.

---

## Local vs public (CV/LB gap)

| Model | OOF Macro F1 | Public Macro F1 | Gap |
|-------|--------------|-----------------|-----|
| RF+LGBM OOF (`rf_lgbm_oof`) | 0.7452 | 0.72694 | ~0.018 |
| First ensemble (`ensemble_advanced`) | ~0.745 | 0.70211 | ~0.043 |

Drift explains part of the gap — see [DIAGNOSTICS.md](DIAGNOSTICS.md) (adversarial AUC **0.6355**).

---

## How each file was produced

| File | Command |
|------|---------|
| `ensemble_advanced.csv` | Earlier `python -m src.advanced` (3-model ensemble) |
| `rf_lgbm_oof.csv` | `python -m src.submit_oof --model rf_lgbm --repeats 1` |
| `rf_lgbm_baserate.csv` | Same script (base-rate threshold branch) |
| `ensemble_improved.csv` | `python -m src.advanced --no-optuna` |
| `ensemble_baserate.csv` | `python -m src.advanced --no-optuna` (base-rate variant) |

---

## Upload commands

```powershell
cd ru-loan-approval-ny

kaggle competitions submit -c ru-loan-approval-prediction-new-york `
  -f "submissions\rf_lgbm_oof.csv" -m "RF+LGBM OOF improved features"

kaggle competitions submit -c ru-loan-approval-prediction-new-york `
  -f "submissions\ensemble_advanced.csv" -m "Ensemble RF+LGBM+XGB trial"
```

After each upload, add the public score to the table above and to [README.md](../README.md) score tracker.
