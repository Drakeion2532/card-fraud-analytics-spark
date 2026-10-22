-- @fraud_by_merchant_category
-- Where is fraud concentrated? Share of fraud value vs share of all spend, by merchant category.
SELECT mcc,
       ROUND(100 * SUM(is_fraud * amount) / SUM(SUM(is_fraud * amount)) OVER (), 1) AS pct_of_fraud_value,
       ROUND(100 * SUM(amount) / SUM(SUM(amount)) OVER (), 1)                      AS pct_of_all_spend,
       ROUND(10000 * AVG(is_fraud), 2)                                            AS fraud_bps
FROM fraud.transactions
GROUP BY mcc
ORDER BY pct_of_fraud_value DESC;

-- @fraud_by_channel
-- Fraud rate (basis points) by channel and geography.
SELECT CASE WHEN online THEN 'online' ELSE 'in-store' END AS channel,
       CASE WHEN foreign THEN 'foreign' ELSE 'domestic' END AS geo,
       COUNT(*) AS txns,
       ROUND(10000 * AVG(is_fraud), 2) AS fraud_bps
FROM fraud.transactions
GROUP BY 1, 2
ORDER BY fraud_bps DESC;

-- @rules_vs_model_by_fraud_type
-- Test window: fraud value caught by the rules vs. the model at a 0.2% review budget.
SELECT t.fraud_type,
       COUNT(*)                                                         AS fraud_txns,
       ROUND(SUM(t.amount))                                             AS fraud_value,
       ROUND(100 * COALESCE(SUM(CASE WHEN s.rule_flag = 1 THEN t.amount END), 0) / SUM(t.amount), 1)         AS pct_value_caught_rules,
       ROUND(100 * COALESCE(SUM(CASE WHEN s.score >= ${THRESH} THEN t.amount END), 0) / SUM(t.amount), 1)  AS pct_value_caught_model
FROM fraud.test_scores s JOIN fraud.transactions t ON s.txn_id = t.txn_id
WHERE t.is_fraud = 1
GROUP BY t.fraud_type
ORDER BY fraud_value DESC;

-- @daily_review_workload
-- What would a review team face each day at the 0.2% budget? (test window, 30 days)
SELECT ROUND(AVG(alerts))       AS avg_alerts_per_day,
       MAX(alerts)              AS peak_alerts_per_day,
       ROUND(AVG(fraud_caught)) AS avg_fraud_caught_per_day,
       ROUND(100 * SUM(fraud_caught) / SUM(alerts), 1) AS pct_alerts_that_are_fraud
FROM (
  SELECT t.txn_date,
         SUM(CASE WHEN s.score >= ${THRESH} THEN 1 ELSE 0 END)                AS alerts,
         SUM(CASE WHEN s.score >= ${THRESH} AND t.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_caught
  FROM fraud.test_scores s JOIN fraud.transactions t ON s.txn_id = t.txn_id
  GROUP BY t.txn_date
) d;

-- @loss_view_test_window
-- Fraud value prevented vs. let through, rules vs. model (0.2% budget), test window in rupees.
SELECT ROUND(SUM(t.amount))                                                      AS total_fraud_value,
       ROUND(SUM(CASE WHEN s.rule_flag = 1 THEN t.amount ELSE 0 END))            AS caught_by_rules,
       ROUND(SUM(CASE WHEN s.score >= ${THRESH} THEN t.amount ELSE 0 END))       AS caught_by_model,
       SUM(CASE WHEN s.rule_flag = 1 THEN 1 ELSE 0 END)                          AS rule_alerts_on_fraud,
       SUM(CASE WHEN s.score >= ${THRESH} THEN 1 ELSE 0 END)                     AS model_alerts_on_fraud
FROM fraud.test_scores s JOIN fraud.transactions t ON s.txn_id = t.txn_id
WHERE t.is_fraud = 1;
