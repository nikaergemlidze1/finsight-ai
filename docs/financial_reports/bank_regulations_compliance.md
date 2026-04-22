# Bank Marketing Compliance & Regulatory Framework
## Telemarketing for Term Deposits — Portuguese & EU Context

*Prepared for internal compliance reference. Last reviewed: Q4 2024.*

---

## 1. GDPR Requirements (Regulation (EU) 2016/679)

### Consent & Lawful Basis
Outbound telemarketing for financial products requires a **legitimate interest or explicit prior consent** as the lawful basis for processing. Banks must document which basis applies to each contact record before initiating a call.

- **Explicit consent** must be granular: consent to "banking communications" does not cover telemarketing for new products.
- **Opt-out must be immediate and unconditional**: customers who request removal from calling lists must be suppressed within 10 business days across all downstream systems, including CRM, dialler, and the ML scoring pipeline.
- **Do-Not-Call (DNC) list integration**: the production model's batch prediction output must be filtered against the current DNC registry before call lists are generated. Calling a registered DNC contact constitutes a GDPR violation regardless of the model's predicted probability.

### Data Retention
| Data Category | Retention Limit | Action at Expiry |
|---|---|---|
| Call outcome records | 24 months | Anonymise or delete |
| Consent logs | 36 months from withdrawal | Cryptographic deletion |
| Call recordings | 12 months (MiFID II extends to 60 months for advisory calls) | Secure deletion |
| Model training data | No re-identification after 24 months | Aggregate-only retention |

### Call Recording Disclosure
Agents must disclose call recording at the **start of every interaction** before any product information is shared. Failure to disclose invalidates the recording for regulatory archival purposes and exposes the bank to individual fines up to **€20 million or 4% of global annual turnover**, whichever is higher.

### Anonymisation for ML Models
Training data used in the FinSight AI model has been processed to remove direct identifiers (name, address, account number). The `age` field is retained as a continuous variable. Under GDPR Article 89, anonymised data used for statistical modelling is exempt from most data-subject rights, but re-identification risk must be assessed annually by the Data Protection Officer.

---

## 2. MiFID II Requirements (Directive 2014/65/EU)

### Pre-Sale Disclosures
Before a term deposit offer may be accepted, agents must verbally disclose — and follow up in writing within 24 hours:

- **Minimum deposit period** and early withdrawal penalty (typically loss of accrued interest for the period prior to withdrawal)
- **Annual Percentage Rate (APR)** including all fees, presented in a standardised Key Information Document (KID)
- **Deposit Guarantee Scheme coverage**: up to **€100,000 per depositor** under the Portuguese deposit guarantee fund (Fundo de Garantia de Depósitos)
- **Tax treatment**: interest income is subject to Portuguese withholding tax at 28%; non-residents may be eligible for reduced rates under applicable tax treaties

### Suitability Assessment
For term deposits exceeding **€10,000**, a suitability assessment questionnaire is mandatory. This assesses:
- Client's investment objectives and time horizon
- Financial situation and liquidity needs
- Knowledge and experience with similar products

If the product is deemed unsuitable, the bank must issue a written warning. Proceeding over a negative suitability determination requires documented client acknowledgement.

### Cooling-Off Period
Retail clients have a **14-day right of withdrawal** from the date of signature with no penalty. Campaign agents must not frame urgency in a way that discourages clients from exercising this right — doing so constitutes aggressive selling under MiFID II Article 16.

### Archival
All call recordings, KIDs, suitability assessments, and client communications related to a term deposit sale must be archived for **five years** from the date of the transaction, extendable to seven years at the request of the regulator.

---

## 3. Banco de Portugal Directives

The Banco de Portugal (BdP) has issued supplementary guidelines on outbound telemarketing for retail banking products under Instruction 14/2021:

- **Maximum contact frequency**: **3 contacts per client per 30-day period** for the same product. The FinSight AI `campaign` feature records contacts within the current campaign; the scoring pipeline must suppress clients where `campaign >= 3`.
- **Permitted call hours**: **Monday to Saturday, 09:00–21:00 local time (WET/WEST)**. Calls may not be initiated on Sundays or Portuguese national public holidays.
- **APR presentation**: The APR must be stated verbally during the call in a standardised format — *"The Annual Percentage Rate for this product is X%, inclusive of all fees"* — before moving to close.
- **Agent identification**: Agents must identify themselves by full name and employer institution at the start of each call.

---

## 4. Prohibited Practices

- Misrepresenting the APR, deposit guarantee coverage, or withdrawal terms
- Using high-pressure closing techniques or artificial urgency ("this rate expires today")
- Calling clients who have registered a DNC preference
- Targeting clients based on protected characteristics (disability status, religion, ethnicity) — the ML model must not use proxies for these characteristics
- Failing to disclose call recording before substantive conversation begins

---

## 5. Campaign Compliance Checklist

Before any call list is approved for dialling:

- [ ] DNC list suppression applied and dated
- [ ] Contact frequency check: `campaign` < 3 for all records
- [ ] Call scheduling system enforces 09:00–21:00 window and public holiday calendar
- [ ] All agents have completed annual MiFID II product training (certificate on file)
- [ ] KID document version approved by Compliance for the current rate offer
- [ ] Call recording system active and disclosure script included in agent brief
- [ ] GDPR lawful basis documented in CRM for all records in the call list

---

## 6. ML Model Compliance

### Protected Characteristics
The FinSight AI model uses `age`, `job`, and `marital` as features. These are legitimate predictors under fair-lending guidelines provided:
- The model is not used as a mechanism to systematically exclude protected groups
- Disparate impact analysis is conducted quarterly across gender (inferred from marital status patterns), age bands, and employment categories

### Bias Audits
An annual **algorithmic bias audit** is required, comparing conversion rates between demographic segments at the model's recommended threshold. If the false negative rate for any protected group exceeds 1.5× the overall rate, the threshold must be recalibrated or the feature set reviewed.

### Explainability
SHAP (SHapley Additive exPlanations) values are generated for every prediction above the decision threshold. Agents who challenge a model-generated "do not call" recommendation can request a SHAP explanation report through the compliance portal.

### Fair-Lending Review
Annual submission of model documentation to the BdP's internal fintech supervision team is recommended under the EU AI Act (Regulation (EU) 2024/1689), which classifies credit and deposit marketing models as **limited-risk AI systems** requiring transparency obligations.

---

## 7. Penalty Summary

| Regulation | Maximum Penalty |
|---|---|
| GDPR (data breach or consent violation) | €20,000,000 or 4% of global annual turnover |
| MiFID II (disclosure failure) | €5,000,000 or 10% of annual turnover (firm); €700,000 (individual) |
| BdP Instruction 14/2021 (contact frequency) | Administrative fine up to €5,000,000 |
| Portuguese Consumer Credit Code (DL 133/2009) | Up to €44,900 per infraction |
