"""Build the day's fraud-review queue: what an analyst team actually works from.

Model scores alone don't help an analyst. They need a short list, in priority order, sized to
what the team can review, and for each alert the reasons it was raised. This script takes one
day of scored transactions, keeps the top `--budget` share (default 0.2% of the day's volume),
attaches plain-language reasons computed from the same features the model used, and writes a CSV.

Usage: python alert_queue.py WORK_DIR 2025-03-15 [--budget 0.002]
       (WORK_DIR is the directory train_model.py wrote features/ and test_scores/ to;
        the date must be in the test window, days 61-90.)
"""
import argparse
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

HIGH_RISK_MCC = ("electronics", "gift_cards", "online_marketplace", "travel")


def reasons_for(r) -> list[str]:
    """Plain-language reasons for one alert, strongest signals first."""
    out = []
    if r["small_cnt_1h"] >= 3:
        out.append(f"{r['small_cnt_1h']} small charges on this card in the last hour (card-testing pattern)")
    if r["cnt_1h"] >= 3:
        out.append(f"{r['cnt_1h']} transactions on this card in the last hour")
    if r["cnt_30d"] >= 5 and r["amt_vs_avg"] >= 3:
        out.append(f"amount is {r['amt_vs_avg']:.1f}x this card's 30-day average")
    if r["cnt_30d"] >= 5 and r["spend24h_vs_avg"] >= 5:
        out.append(f"spent {r['spend24h_vs_avg']:.0f}x the card's usual ticket in the last 24 hours")
    if r["foreign"] and r["foreign_rate_30d"] < 0.05:
        out.append("foreign transaction on a card that is rarely used abroad")
    if r["online"] and r["online_rate_30d"] < 0.3:
        out.append("online purchase on a card mostly used in store")
    if r["mcc"] in HIGH_RISK_MCC:
        out.append(f"high-risk merchant category ({r['mcc'].replace('_', ' ')})")
    if r["secs_since_prev"] < 120:
        out.append(f"only {r['secs_since_prev']}s after the previous transaction")
    if r["hour"] < 5:
        out.append(f"made at {r['hour']:02d}:00, between midnight and 5 am")
    return out or ["unusual combination of signals for this card (see model score)"]


def build_queue(spark, work, day, budget):
    feats = spark.read.parquet(f"{work}/features").where(F.col("txn_date") == day)
    scores = spark.read.parquet(f"{work}/test_scores").select("txn_id", "score")
    scored = feats.join(scores, "txn_id")
    volume = scored.count()
    if volume == 0:
        raise SystemExit(f"No scored transactions for {day}; pick a date in the test window (days 61-90).")
    k = max(1, round(budget * volume))
    top = scored.orderBy(F.desc("score")).limit(k).toPandas()
    top["reasons"] = ["; ".join(reasons_for(r)) for _, r in top.iterrows()]
    top["priority"] = range(1, len(top) + 1)
    queue = top[["priority", "txn_id", "card_id", "amount", "mcc", "online", "foreign", "score", "reasons",
                 "is_fraud", "fraud_type"]].rename(columns={"is_fraud": "label_hindsight", "fraud_type": "type_hindsight"})
    return queue, volume


def main():
    ap = argparse.ArgumentParser(description="Build one day's fraud-review queue.")
    ap.add_argument("work_dir")
    ap.add_argument("date", help="YYYY-MM-DD, in the test window")
    ap.add_argument("--budget", type=float, default=0.002, help="share of the day's transactions to review")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    spark = (SparkSession.builder.master("local[*]").appName("alert-queue")
             .config("spark.ui.enabled", "false").config("spark.driver.memory", "3g").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    queue, volume = build_queue(spark, a.work_dir, a.date, a.budget)
    out = Path(a.out or f"results/alert_queue_{a.date}.csv")
    queue.to_csv(out, index=False)
    caught = int(queue["label_hindsight"].sum())
    print(f"{a.date}: {volume:,} transactions -> {len(queue)} alerts ({a.budget:.1%} budget); "
          f"{caught} of them turned out to be fraud ({caught / len(queue):.0%}). Written to {out}")
    print(queue.head(5)[["priority", "amount", "mcc", "score", "reasons"]].to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
