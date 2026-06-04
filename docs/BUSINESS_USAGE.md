# Business usage — how a bank would use this model

Interpreted narrative for slides, reports, and stakeholder Q&A. Not legal or compliance advice.

---

## Context

The data describes **small-business loan applications** in **New York** (SBA-style fields): loan amount (`DisbursementGross`), jobs (`NoEmp`), lender (`Bank`), location (`City`), documentation (`LowDoc`), revolving credit line (`RevLineCr`), franchise status, and approval/disbursement dates. The historical label **`Accept`** is **1 = granted**, **0 = denied**.

The competition asks for a model that scores **new applications** the same way the training data was labeled, evaluated with **Macro F1** (equal importance on grant and deny classes).

---

## What the system does

1. **Ingest** an application record (same fields as training, after cleaning).
2. **Engineer features** — amounts, ratios, time features, flags (see [MODELS.md](MODELS.md)).
3. **Encode** high-cardinality lender and city via cross-fitted target encoding (historical approval rates with smoothing).
4. **Score** with an ensemble (Random Forest + LightGBM): **probability of approval**.
5. **Decide** grant vs deny using a **tuned threshold** (optimized for Macro F1 on out-of-fold validation, not fixed 0.5).

Output: binary decision plus probability for routing and reporting.

---

## Operational use cases

### 1. Pre-screening and triage

- **High probability of approval** → fast-track standard processing.
- **Low probability** → senior underwriter or additional documentation.
- **Borderline scores** → queue for human review.

Reduces inconsistent branch-level judgment while keeping humans on difficult cases.

### 2. Consistency across lenders and regions

`Bank` and `City` are strong signals in the data (and in drift analysis). The model encodes **historical approval patterns by lender and geography**, which can standardize decisions when policy allows data-driven support.

### 3. Capacity and portfolio planning

Predicted approval volume (~68% on the test submission vs ~80% in training) signals that **the scoring population may differ from history** — useful for staffing, capital allocation, and monitoring shift (economic period, loan size mix).

### 4. Risk communication

Features aligned with diagnostics — **loan size**, **approval year**, **bank**, **city** — support explainable narratives: “larger loans in certain periods and geographies were associated with different outcomes in training.” Slides should tie 1–2 **example companies** to these drivers (see [PRESENTATION_BRIEF.md](PRESENTATION_BRIEF.md)).

---

## What Macro F1 means for the business story

- **Not** the same as maximizing profit or minimizing default.
- **Yes** a balanced view of **catching denials** and **catching approvals** — relevant when both false grants and false denials have cost.
- Drives **threshold choice**: a cutoff that matches historical approval rate (~80%) can hurt this metric on shifted test data (our base-rate submission scored **0.636** public vs **0.727** with OOF-tuned threshold).

---

## Limitations (state clearly in slides)

| Limitation | Implication |
|------------|-------------|
| **Train/test drift** (adversarial AUC ~0.64) | Scores on live data may differ from cross-validation; monitor performance. |
| **Unseen banks/cities** (7.7% / 13.7% of test levels) | Weak encoding fallback; higher uncertainty → manual review. |
| **Target encoding** | Uses historical approval rates; can perpetuate past bias if not governed. |
| **No default/outcome modeling** | `Accept` is approval, not post-disbursement repayment risk. |
| **Threshold is policy** | Macro F1 optimal ≠ regulator or board optimal approval rate. |

---

## Elevator pitch (one sentence)

*We built an ensemble model that estimates small-business loan approval likelihood from application data, tuned for balanced grant/deny performance, to support faster triage and more consistent underwriting while flagging uncertain cases for human review.*

---

## Example talking points for “why grant / why deny”

Use these patterns on the example-companies slide (fill with real or anonymized rows from EDA):

| Signal | Plain language |
|--------|----------------|
| High `DisbursementGross` / top loan-size bin | Larger exposure; training data links size to different approval odds. |
| `LowDoc = Y` | Reduced documentation; often associated with different risk review. |
| Approval in **2007–2010** (`is_crisis_period`) | Economic stress period in training history. |
| Rare or unseen **City** / **Bank** | Less reliable encoded signal → recommend manual review. |
| Low `loan_per_emp` | Very large loan relative to headcount; scale mismatch flag. |

---

## Relation to course requirements

| Requirement | How this doc helps |
|-------------|-------------------|
| Explain bank risk | § Operational use cases + example talking points |
| Why grant/deny | Feature-driven narratives + limitations |
| Compete on Macro F1 | § What Macro F1 means |

Technical implementation: [MODELS.md](MODELS.md). Next modeling work: [NEXT_STEPS.md](NEXT_STEPS.md).
