# Dataset Insights & Model Findings
## FinSight AI — UCI Bank Marketing Dataset Analysis

*Source: Moro, S., Cortez, P. & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier. doi: 10.1016/j.dss.2014.03.001*

---

## 1. Dataset Overview

The FinSight AI model was trained on **41,188 real telemarketing call records** from a Portuguese retail bank, spanning **May 2008 to November 2010** — a period that encompasses the onset of the global financial crisis and the early European sovereign debt crisis. This economic context is significant: macro-economic features (Euribor, employment rate) reflect an unusually volatile period and may require recalibration when applied to contemporary campaigns.

| Property | Value |
|---|---|
| Total records | 41,188 |
| Positive class (subscribed) | 4,640 (11.3%) |
| Negative class (did not subscribe) | 36,548 (88.7%) |
| Date range | May 2008 – November 2010 |
| Source institution | Portuguese retail bank (anonymised) |
| Features | 20 input variables + 1 target |

---

## 2. Class Imbalance and Its Consequences

The dataset has a **severely imbalanced target**: only 1 in 9 contacts results in a subscription. This has three important implications:

1. **Accuracy is misleading**: A model that predicts "no" for every contact achieves 88.7% accuracy without any predictive power. The FinSight AI evaluation uses **Precision-Recall AUC (PR-AUC)** and **F1 score** as primary metrics instead.

2. **SMOTE oversampling** was applied to the training set only (strictly after the train/val/test split) to generate synthetic minority-class examples and improve the model's sensitivity to positive cases.

3. **Threshold calibration** at 0.23 — rather than the default 0.50 — was required to achieve practical recall. At threshold 0.50, LightGBM flags very few leads as positive, missing the majority of actual subscribers. At 0.23, the model trades some precision for significantly higher recall, which is the correct business trade-off given cost asymmetry (see Section 6).

---

## 3. Top Predictive Features (SHAP Analysis — LightGBM)

SHAP (SHapley Additive exPlanations) values were computed for the LightGBM model on the test set. The following are the five most influential features ranked by mean absolute SHAP value:

### 1. `nr.employed` — Number of Employees in Portuguese Labour Market
The most powerful predictor in the model. Lower employment levels (economic contraction) increase subscription probability because clients seek capital preservation. This is a lagging indicator updated quarterly; campaigns launched during periods of declining employment consistently outperform in this dataset.

### 2. `euribor3m` — 3-Month Euribor Rate
When Euribor is high, term deposits are competitively priced and clients are more receptive. SHAP analysis shows that euribor3m above 3.5% is associated with positive SHAP contributions for most client segments. This variable captures the "opportunity cost of not locking in a rate" effect.

### 3. `age`
Non-linear relationship. Two segments show elevated positive SHAP values:
- **Age 60+**: retirees seeking capital preservation and fixed income during economic uncertainty
- **Age 18–25**: students and early-career individuals — smaller deposits but higher response rate to initial contact

Working-age clients aged 35–55 show the lowest receptivity, reflecting competing financial priorities (mortgage repayments, child education costs).

### 4. `poutcome_success` — Previous Campaign Success
Binary indicator derived from the `poutcome` feature. Clients who subscribed to a previous product have a mean positive SHAP value of +0.82, making this the strongest single binary predictor in the model. No other feature comes close in the "actionable" category (i.e., something the bank can directly use to rank leads).

### 5. `contact_cellular` — Cellular vs. Landline
Cellular contact consistently outperforms landline across all client segments. This partly reflects selection bias (clients who provide mobile numbers are more engaged) and partly reflects the higher answer rates of mobile calls compared to fixed lines in the 2008–2010 period. When building call lists, prioritise clients for whom a cellular contact number is available.

---

## 4. Additional Feature Insights

### Day of Week
| Day | Test Set Conversion Rate |
|---|---|
| Thursday | 12.1% |
| Wednesday | 11.4% |
| Tuesday | 11.2% |
| Friday | 10.3% |
| Monday | 9.1% |

Thursday calls outperform Monday calls by 33% on absolute conversion rate. This pattern is consistent with academic literature on decision fatigue and weekly energy cycles.

### Month Patterns
May (13.3%), March (14.0%), and October (14.7%) are the highest-converting months in the dataset. These align with annual bonus distribution periods and tax refund seasons in Portugal. June (7.2%) and November (8.9%) are the lowest — likely reflecting summer holiday disruption and end-of-year planning respectively.

### Contact Frequency Effect
| Contacts in Current Campaign | Conversion Rate |
|---|---|
| 1 | 13.1% |
| 2 | 8.4% |
| 3 | 5.3% |
| 4+ | 2.8% |

The yield from additional contacts drops by approximately 40% with each successive call. The model already penalises high `campaign` values; do not override the model's scoring by continuing to call low-score clients repeatedly.

---

## 5. The Duration Variable — Why It Was Removed

The `duration` feature (length of the last call in seconds) is **excluded from the production model** despite being the single strongest raw predictor in exploratory analysis.

**Reason**: Duration is a post-call measurement. You cannot know how long a call will last before you decide whether to make it. Including duration in a model used for pre-call lead prioritisation is a form of **data leakage** — the model appears highly accurate in testing but provides zero value in production, because the information it relies on does not exist at decision time.

This is a common failure mode in applied ML. The FinSight AI pipeline explicitly drops `duration` during data loading (`src/data_processing.py`), ensuring the production model makes decisions on the same information available to a human agent before dialling.

---

## 6. Model Performance Summary

| Metric | LightGBM (Best Model) | Interpretation |
|---|---|---|
| Test PR-AUC | 0.4763 | For an 11% base rate, this represents strong lift above random (0.11) |
| Test ROC-AUC | 0.8056 | 80.6% probability of ranking a subscriber above a non-subscriber |
| Test F1 | 0.5242 | Harmonic mean of precision and recall at threshold 0.23 |
| Decision threshold | 0.23 | Tuned on validation set to maximise F1 |
| Comparison: Random model | PR-AUC ≈ 0.11 | FinSight AI provides ~4.3× lift over random selection |

### Why PR-AUC of 0.47 Is a Good Result

For a classification problem with an 11% positive rate, a random classifier achieves PR-AUC ≈ 0.11 (the base rate). FinSight AI's 0.47 represents **4.3× improvement over random lead selection**. Practically: selecting the top 30% of contacts by model score should capture approximately 65–70% of all eventual subscribers — far more efficient than calling all contacts indiscriminately.

---

## 7. Threshold Calibration Rationale

The decision threshold of **0.23** was selected by optimising F1 on the validation set. The economic logic:

| Error Type | Business Cost | Frequency at Threshold 0.23 |
|---|---|---|
| False negative (miss a subscriber) | ~€500 lost revenue per missed deposit | Minimised relative to threshold 0.50 |
| False positive (call a non-subscriber) | ~€3 agent cost per wasted call | Accepted at higher rate |

The ratio of false-negative to false-positive cost is approximately **167:1**. This asymmetry justifies a low threshold that accepts more false positives in order to recover more true positives. Campaign managers can adjust the threshold higher (e.g., 0.35–0.40) if agent capacity is the binding constraint, at the cost of lower recall.

---

## 8. Dataset Limitations & Transfer Warnings

| Limitation | Risk | Mitigation |
|---|---|---|
| 2008–2010 economic context | Euribor, nr.employed patterns may not reflect today's environment | Re-weight macro features or retrain on recent data |
| Portuguese market only | Client behaviour, regulatory environment, and product norms differ across markets | Do not apply model directly to non-Portuguese campaigns |
| Single institution | Institution-specific calling patterns (agent scripts, CRM, product set) are baked into the data | Validate on own historical data before deployment |
| No channel mix data | Only telephone contacts recorded; digital channel interactions absent | Supplement with online engagement signals if available |

The model should be **retrained annually** on recent campaign data and the threshold recalibrated whenever the positive class rate shifts by more than ±2 percentage points from the training-time base rate of 11.3%.
