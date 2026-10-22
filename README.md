# Card Fraud Analytics at Scale (PySpark)

An end-to-end fraud pipeline on **10 million card transactions** (200,000 cardholders, 90 days) in PySpark: Parquet storage partitioned by day, behavioural features written as **Spark SQL window functions**, a **Spark MLlib gradient-boosted model**, and a head-to-head comparison against fraud rules **at the same alert budget**, the way a fraud-strategy team would judge it.

![Recall vs review rate](results/recall_vs_review_rate.png)

## Headline results (test window: days 61–90, 3.3M transactions, 0.08% fraud)

Both approaches flag the same **0.34% of transactions** for review:

| | Rules | GBT model |
|---|---|---|
| Fraud transactions caught | 47.2% | **82.2%** |
| Fraud value caught | 53.8% | **86.9%** |
| Precision (alerts that are fraud) | 10.2% | **17.7%** |
| Compromised cards caught | 73.1% | **93.5%** |
| Median fraud txns before first alert | 2 | **1** |

**At tighter review budgets:** reviewing just 0.1% of transactions catches 74.8% of fraud transactions and 81.3% of fraud value at 54.5% precision. Overall ROC-AUC is 0.976 and PR-AUC is 0.746, against a 0.08% base rate.

**By fraud style** (at the rules' alert volume): burst 91%, card-testing 96%, stealth 38%. The rules catch **0%** of stealth fraud, which only shows up as a shift in the card's own behaviour.

**Runtime** on a 2-core, 7 GB machine (Spark local mode): features in 65s, training in 85s, scoring 3.3M rows in 30s.

## Pipeline
1. **`generate_data.py`:** simulates 10M transactions in Spark and writes Parquet partitioned by `txn_date`. Legitimate spend varies by card (typical amount, online habit, merchant mix, daytime-heavy hours). Look-alike noise is included on purpose: travel spend abroad, occasional large purchases, online sprees. About 1,000 cards (0.5%) are compromised, with three fraud styles:
   - **burst:** rapid high-value online spending, often foreign
   - **card test:** several tiny charges within minutes
   - **stealth:** amounts and channels that match the cardholder's normal behaviour
2. **`features.sql`:** per-card features that use only past activity, via `RANGE BETWEEN … PRECEDING` windows:
   - velocity: transactions and spend in the last 1h and 24h, small charges in the last hour, seconds since the previous transaction
   - the card's 30-day baseline: average amount and z-score, online and foreign rates, and how surprising this transaction is against them
3. **`train_model.py`:**
   - **Time-based split:** train on days 1–60, test on days 61–90.
   - **Merchant-category risk:** smoothed target encoding, learned on training days only.
   - **Model:** Spark MLlib `GBTClassifier`, trained with legitimate rows down-sampled to 5% and all fraud kept.
   - **Evaluation:** the model is compared with a typical rule set at the **same alert volume**, then at fixed review budgets of 0.1%, 0.2% and 0.5%, with transaction-level, value-level and card-level metrics.

**Top features (GBT importance):** 1-hour velocity (0.32), hour of day (0.12), 24-hour velocity (0.09), merchant-category risk (0.08), time since previous transaction (0.07), 24-hour spend vs the card's average (0.06), foreign surprise (0.06).

## Limits (say these before you're asked)
- **Synthetic data.** I wrote both the fraud patterns and the features, so the model has an easier job than on real data. The numbers show the method and the scale, not real-world fraud performance.
- **Hour of day ranks high** because legitimate spend was simulated as daytime-heavy while fraud is spread evenly. Real fraud timing is messier.
- **Single machine.** This is Spark in local mode. The same code runs on a cluster, but this project doesn't demonstrate cluster tuning.
- **Labels arrive instantly here.** Real chargebacks arrive weeks later, which delays training data. There is also no cost model per alert or per missed fraud yet; [credit-default-scorecard](https://github.com/Drakeion2532/credit-default-scorecard) shows that approach.

## Run it
Needs Python 3.10+ and Java 17 or 21.
```bash
pip install -r requirements.txt
python generate_data.py data/tx 10000000      # ~10M rows, about 190 MB of Parquet
python train_model.py data/tx data/work        # writes results/results.json and the chart
```
Use a smaller first argument (e.g. `200000`) for a quick test run.
