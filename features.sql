-- Per-transaction behavioural features, computed only from each card's PAST activity.
-- Input view: tx(card_id, ts, amount, mcc, online, foreign, is_fraud, fraud_type, txn_id, txn_date)
WITH w AS (
  SELECT
    t.*,
    -- velocity: transactions on this card in the last 1 hour / 24 hours (including this one)
    COUNT(*)    OVER (PARTITION BY card_id ORDER BY ts RANGE BETWEEN 3600  PRECEDING AND CURRENT ROW) AS cnt_1h,
    COUNT(*)    OVER (PARTITION BY card_id ORDER BY ts RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW) AS cnt_24h,
    SUM(amount) OVER (PARTITION BY card_id ORDER BY ts RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW) AS spend_24h,
    -- card's normal behaviour over the previous 30 days (excluding the current transaction)
    AVG(amount)             OVER (PARTITION BY card_id ORDER BY ts RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING) AS avg_amt_30d,
    STDDEV_POP(amount)      OVER (PARTITION BY card_id ORDER BY ts RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING) AS std_amt_30d,
    COUNT(*)                OVER (PARTITION BY card_id ORDER BY ts RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING) AS cnt_30d,
    AVG(CAST(online AS INT))  OVER (PARTITION BY card_id ORDER BY ts RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING) AS online_rate_30d,
    AVG(CAST(foreign AS INT)) OVER (PARTITION BY card_id ORDER BY ts RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING) AS foreign_rate_30d,
    ts - LAG(ts) OVER (PARTITION BY card_id ORDER BY ts) AS secs_since_prev,
    -- small-amount probing: charges under 200 in the last hour
    SUM(CASE WHEN amount < 200 THEN 1 ELSE 0 END)
                OVER (PARTITION BY card_id ORDER BY ts RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW) AS small_cnt_1h
  FROM tx t
)
SELECT
  txn_id, card_id, ts, txn_date, amount, mcc, is_fraud, fraud_type,
  CAST(online AS INT)  AS online,
  CAST(foreign AS INT) AS foreign,
  HOUR(FROM_UNIXTIME(ts)) AS hour,
  cnt_1h, cnt_24h, spend_24h, small_cnt_1h,
  COALESCE(cnt_30d, 0) AS cnt_30d,
  COALESCE(amount / NULLIF(avg_amt_30d, 0), 1.0)                     AS amt_vs_avg,
  COALESCE((amount - avg_amt_30d) / NULLIF(std_amt_30d, 0), 0.0)     AS amt_zscore,
  COALESCE(spend_24h / NULLIF(avg_amt_30d, 0), 1.0)                  AS spend24h_vs_avg,
  COALESCE(online_rate_30d, 0.5)  AS online_rate_30d,
  COALESCE(foreign_rate_30d, 0.0) AS foreign_rate_30d,
  CAST(foreign AS INT) * (1 - COALESCE(foreign_rate_30d, 0.0)) AS foreign_surprise,
  CAST(online AS INT)  * (1 - COALESCE(online_rate_30d, 0.5))  AS online_surprise,
  COALESCE(LEAST(secs_since_prev, 2592000), 2592000) AS secs_since_prev
FROM w
