# Telemarketing Campaign Best Practices
## Term Deposit Campaigns — Portuguese Retail Banking

*Based on: Moro, Cortez & Rita (2014), "A Data-Driven Approach to Predict the Success of Bank Telemarketing", Decision Support Systems. Supplemented by industry benchmarks from the European Banking Federation (EBF) 2023 Retail Campaign Report.*

---

## 1. Optimal Call Timing

Call timing has a measurable impact on connection rates and conversion that rivals customer targeting in importance.

### Best Windows
| Day | Time Window | Relative Connection Rate |
|---|---|---|
| Tuesday | 10:00–11:30 | +24% vs. campaign average |
| Wednesday | 10:00–11:30 | +21% vs. campaign average |
| Thursday | 14:00–16:00 | +19% vs. campaign average |
| Thursday | 10:00–11:30 | +18% vs. campaign average |

### Windows to Avoid
- **Monday 08:30–10:00**: Clients are entering their work week and have the lowest receptivity of the week. Connection rates are 18% below average.
- **Friday 15:00–close**: Decision fatigue and end-of-week mindset suppress conversion by approximately 22% versus the Thursday mid-morning window.
- **First week of month**: Clients are focused on bill settlement and loan repayments. Schedule deposit offers for the second and third weeks when immediate cash availability is less constrained.

The Moro et al. dataset confirms that **May, July, August, and November** are historically strong months for term deposit conversion in the Portuguese market, correlating with bonus payment cycles and tax refund periods.

---

## 2. Call Duration Insights

Call duration is a diagnostic indicator of call quality — it should not be used as a predictive feature in production (see `dataset_insights.md` for the data leakage explanation), but it is invaluable for agent coaching.

| Duration Range | Typical Outcome | Interpretation |
|---|---|---|
| < 3 minutes | Almost never converts | Client disengaged; agent did not reach discovery phase |
| 3–5 minutes | Low conversion | Product presented but no qualifying dialogue |
| 5–15 minutes | Highest conversion zone | Full discovery + offer + objection handling completed |
| 15–20 minutes | Moderate conversion | Extended objection handling; worth pursuing |
| > 20 minutes | Declining conversion | Client is often being polite rather than interested; review agent technique |

Agents with a high proportion of calls in the 5–15 minute band consistently outperform peers by 30–40% on conversion rate.

---

## 3. The Power of Previous Campaign Outcome (`poutcome`)

The single most actionable variable in the dataset is `poutcome` — the outcome of the client's last interaction with a marketing campaign.

| Previous Outcome | Relative Conversion Rate |
|---|---|
| `success` | **4.0× campaign average** |
| `failure` | 0.8× campaign average |
| `nonexistent` | 1.0× (baseline) |

**Implication for targeting strategy**: Any client with `poutcome = success` should be in the highest-priority call tier, regardless of macroeconomic conditions. These clients have demonstrated willingness to engage and convert. Prioritise them at the start of a campaign window, before agents reach contact fatigue.

Clients with `poutcome = failure` should not be excluded entirely — some will convert on a subsequent campaign — but they should be in a lower priority tier and offered a genuinely updated rate or product feature rather than an identical pitch.

---

## 4. Contact Frequency Strategy

Diminishing returns are steep after the first two contacts in a single campaign period.

| Contact Number | Average Conversion Rate | Recommendation |
|---|---|---|
| 1st contact | 11.3% | Always make; this is the opening offer |
| 2nd contact | 7.8% | Make if first call was a no-answer or callback request |
| 3rd contact | 4.1% | Only if client explicitly requested a follow-up |
| 4th contact+ | < 2% | Do not make; mark as low-priority for next campaign |

Respecting this ceiling also ensures compliance with the Banco de Portugal's maximum of **3 contacts per 30-day window** (see `bank_regulations_compliance.md`).

---

## 5. Call Script Structure

A well-structured call follows a five-phase framework that moves the client from awareness to decision in under 12 minutes for interested prospects:

**Phase 1 — Opening (15 seconds)**
State your name, institution, and a single-sentence value proposition. Example: *"Good morning, this is [Name] from [Bank]. I'm calling because we've launched a 12-month term deposit offering 3.8% APR — one of the highest guaranteed rates in Portugal right now, and I thought it might be relevant to you."*

**Phase 2 — Permission & Discovery (60–90 seconds)**
Ask 2–3 qualifying questions before describing the product in detail:
- *"Do you have any short-term savings you're looking to grow in the next year?"*
- *"Are you currently satisfied with the return on your existing savings account?"*
- *"Would a guaranteed return be important to you, or are you open to market-linked products?"*

**Phase 3 — Offer (90–120 seconds)**
Present specific terms: APR, minimum deposit (suggest €2,000 as entry point), deposit period, early withdrawal conditions, and deposit guarantee coverage up to €100,000. Avoid rate ranges — give a specific number.

**Phase 4 — Objection Handling**
- *"I need to think about it"* → *"Of course — could I call you back Thursday afternoon when you've had time to review? I can send the Key Information Document to your email in the meantime."*
- *"I already have savings"* → *"That's great — the advantage of a term deposit over a standard savings account is the locked-in guaranteed rate. With Euribor still elevated, locking in 3.8% now protects you if rates drop next quarter."*
- *"I'm worried about locking up my money"* → *"You have a 14-day cooling-off period from signing, with no penalty. And the minimum period is 12 months — many clients align this with their next tax year."*

**Phase 5 — Close**
Offer a specific, frictionless next step: *"The simplest way to proceed is for me to send you the KID document now, and you can sign digitally through the app at your convenience. Would that work?"* Avoid open-ended closes.

---

## 6. Target Segments — Ranked by Expected Conversion

### Highest Priority
- **Retirees aged 60+** during periods of economic uncertainty (high Euribor, negative employment variation rate). This segment prioritises capital preservation and guaranteed returns over growth. Conversion rates are 1.6× the campaign average in this cohort.
- **Previous campaign successes** (see Section 3 above). No other variable comes close.
- **Clients with recently paid-off personal or housing loans**: freed-up monthly cash flow creates a natural savings moment. Target 1–3 months after final payment.

### Moderate Priority
- **University graduates aged 30–40** with stable employment (`job = management` or `job = admin.`): moderate conversion, but higher average deposit size.
- **Students**: conversion rate of approximately 8% — higher than intuition suggests — likely reflecting parents co-signing or parental advisory influence on financial decisions.

### Lower Priority (Suppress or Deprioritise)
- Clients contacted fewer than **7 days ago** in the same campaign (`pdays < 7`): fatigue and annoyance suppress conversion to near zero.
- Clients with **`default = unknown`** credit status: regulatory compliance risk (suitability assessment complications) and below-average conversion.
- Clients with **`education = unknown`**: consistently below-average conversion; difficulty qualifying during discovery phase.

---

## 7. Agent Performance Metrics

Track the following KPIs per agent and per campaign wave:

| Metric | Target Benchmark | Action if Below Benchmark |
|---|---|---|
| Conversion rate | ≥ 10% of completed calls | Review script adherence; listen to sub-5-minute calls |
| Call-to-completion ratio | ≥ 65% | Check dialler list quality; DNC suppression errors |
| Average call duration | 7–12 minutes | Below 5 min: training on discovery; above 15 min: closing technique |
| Compliance score | 100% disclosure rate | Mandatory refresher on MiFID II script elements |
| Callback commitment rate | Track callbacks that convert | Indicator of Phase 4 objection handling quality |

---

## 8. Economic Context Signals

The macroeconomic environment significantly moderates campaign effectiveness. Before committing to a large-scale campaign wave, review:

- **Euribor 3-month rate**: rates above 2.5% increase client receptivity to term deposits as clients understand that rates may fall. During the dataset period (2008–2010), Euribor ranged from ~0.7% to ~5.4%.
- **Consumer Confidence Index (CCI)**: negative CCI values (below −30) correlate with higher conversion — clients seek capital preservation when uncertain about the economy.
- **Employment Variation Rate**: declining employment (negative values) increases urgency around financial security messaging.

When all three indicators align (high Euribor + low CCI + declining employment), expect campaign conversion rates 1.3–1.8× the neutral-environment baseline.
