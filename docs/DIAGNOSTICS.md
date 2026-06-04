# Phase 0 diagnostics (auto-generated summary)

Run: `python -m src.cv_utils`

## Temporal / coverage

| Check | Result |
|-------|--------|
| ApprovalDate_year train | 1976 – 2075 |
| ApprovalDate_year test | 1976 – 2074 |
| Bank levels unseen in test | 14 (7.7%) |
| City levels unseen in test | 158 (13.7%) |
| State | constant NY (train and test) |

## Adversarial validation

- **Train vs test ROC-AUC: 0.6355** → moderate drift
- Recommendation: use **random stratified OOF** and **time-based CV** side by side

### Top drift features

1. `log_DisbursementGross`
2. `ApprovalDate_year`
3. `Bank` (target-encoded)
4. `DisbursementDate_year`
5. `loan_size_bin`

## Validation strategy chosen

- Primary: **5-fold OOF** with threshold tuned on pooled OOF probabilities
- Secondary: **time-based CV** on `ApprovalDate_year` for LightGBM monitoring
- Fallback submission: **base-rate threshold** (~79.6% predicted approvals)
