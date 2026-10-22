"""Build features with Spark SQL, train a Spark MLlib gradient-boosted model, and compare it
with hand-written fraud rules at the same alert budget.

Time-based split: train on days 1-60, test on days 61-90 (no future data leaks into training).
Usage: python train_model.py TX_PARQUET_DIR WORK_DIR
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F

SEED = 7
START = 1_735_689_600
SPLIT_TS = START + 60 * 86400
LEGIT_SAMPLE = 0.05          # keep 5% of legit rows in training for speed; fraud kept in full
BUDGETS = [0.001, 0.002, 0.005]   # share of transactions an analyst team can review
FEATURES = ["amount", "online", "foreign", "hour", "cnt_1h", "cnt_24h", "spend_24h", "small_cnt_1h",
            "cnt_30d", "amt_vs_avg", "amt_zscore", "spend24h_vs_avg", "online_rate_30d",
            "foreign_rate_30d", "foreign_surprise", "online_surprise", "secs_since_prev", "mcc_risk"]
OUT = Path("results")


def rules(df):
    """A typical first-line rule set, written the way a fraud-strategy team might."""
    return ((F.col("cnt_1h") >= 4)
            | (F.col("small_cnt_1h") >= 3)
            | ((F.col("foreign_surprise") > 0.9) & (F.col("online") == 1) & (F.col("amt_vs_avg") > 3))
            | (F.col("amt_vs_avg") > 8)).cast("int")


def summarise(pdf, flag_col):
    flagged = pdf[flag_col] == 1
    fraud = pdf["is_fraud"] == 1
    tp = int((flagged & fraud).sum())
    return {"alert_rate": round(float(flagged.mean()), 5), "alerts": int(flagged.sum()),
            "recall": round(tp / max(int(fraud.sum()), 1), 3),
            "precision": round(tp / max(int(flagged.sum()), 1), 3),
            "fraud_value_caught": round(float(pdf.loc[flagged & fraud, "amount"].sum()
                                              / pdf.loc[fraud, "amount"].sum()), 3),
            "recall_by_type": {t: round(float(g[flag_col].mean()), 3)
                               for t, g in pdf[fraud].groupby("fraud_type")}}


def card_level(pdf, flag_col):
    """For each compromised card in the test window: was it caught, and after how many fraud txns?"""
    f = pdf[pdf.is_fraud == 1].sort_values(["card_id", "ts"])
    caught, before = 0, []
    for _, g in f.groupby("card_id"):
        hits = np.flatnonzero(g[flag_col].to_numpy() == 1)
        if len(hits):
            caught += 1
            before.append(int(hits[0]))  # fraud txns that went through before the first alert
    n = f.card_id.nunique()
    return {"compromised_cards": int(n), "cards_caught": round(caught / max(n, 1), 3),
            "median_fraud_txns_before_first_alert": float(np.median(before)) if before else None}


def main(tx_dir, work):
    OUT.mkdir(exist_ok=True)
    timings = {}
    spark = (SparkSession.builder.master("local[*]").appName("card-fraud-model")
             .config("spark.ui.enabled", "false").config("spark.driver.memory", "5g")
             .config("spark.sql.shuffle.partitions", "32").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    t = time.time()
    spark.read.parquet(tx_dir).createOrReplaceTempView("tx")
    feats = spark.sql(Path("features.sql").read_text())
    feats.write.mode("overwrite").partitionBy("txn_date").parquet(f"{work}/features")
    feats = spark.read.parquet(f"{work}/features")
    timings["feature_engineering_s"] = round(time.time() - t)

    train, test = feats.where(F.col("ts") < SPLIT_TS), feats.where(F.col("ts") >= SPLIT_TS)
    # merchant-category risk, learned on the training window only (smoothed target encoding)
    prior = train.agg(F.avg("is_fraud")).first()[0]
    mcc = (train.groupBy("mcc").agg(F.sum("is_fraud").alias("f"), F.count("*").alias("n"))
           .select("mcc", ((F.col("f") + 1000 * prior) / (F.col("n") + 1000)).alias("mcc_risk")))
    train = train.join(F.broadcast(mcc), "mcc", "left").fillna({"mcc_risk": prior})
    test = test.join(F.broadcast(mcc), "mcc", "left").fillna({"mcc_risk": prior})

    counts = feats.agg(F.count("*").alias("n"), F.sum("is_fraud").alias("f")).first()
    n_train = train.count(); n_test = test.count()

    t = time.time()
    sample = train.sampleBy("is_fraud", {0: LEGIT_SAMPLE, 1: 1.0}, seed=SEED)
    gbt = GBTClassifier(featuresCol="features", labelCol="is_fraud", maxDepth=5, maxIter=60,
                        stepSize=0.1, subsamplingRate=0.8, seed=SEED)
    model = Pipeline(stages=[VectorAssembler(inputCols=FEATURES, outputCol="features"), gbt]).fit(sample)
    timings["model_training_s"] = round(time.time() - t)

    t = time.time()
    scored = (model.transform(test)
              .withColumn("score", vector_to_array("probability")[1])
              .withColumn("rule_flag", rules(test)))
    pdf = scored.select("card_id", "ts", "amount", "is_fraud", "fraud_type", "score", "rule_flag").toPandas()
    timings["scoring_s"] = round(time.time() - t)

    report = {"transactions": int(counts["n"]), "fraud_transactions": int(counts["f"]),
              "fraud_rate": round(counts["f"] / counts["n"], 5), "train_rows": n_train, "test_rows": n_test,
              "split": "train days 1-60, test days 61-90", "timings": timings}

    rule_s = summarise(pdf, "rule_flag")
    report["rules"] = {**rule_s, **card_level(pdf, "rule_flag")}
    # model at the SAME alert volume as the rules, then at fixed review budgets
    def at_rate(rate, name):
        k = max(int(round(rate * len(pdf))), 1)
        thr = np.sort(pdf.score.to_numpy())[::-1][k - 1]
        pdf[name] = (pdf.score >= thr).astype(int)
        return {**summarise(pdf, name), **card_level(pdf, name), "threshold": round(float(thr), 4)}
    report["model_same_alerts_as_rules"] = at_rate(rule_s["alert_rate"], "m_rules")
    report["model_at_budget"] = {f"{b:.1%}": at_rate(b, f"m_{b}") for b in BUDGETS}

    from sklearn.metrics import average_precision_score, roc_auc_score
    report["model_pr_auc"] = round(float(average_precision_score(pdf.is_fraud, pdf.score)), 3)
    report["model_roc_auc"] = round(float(roc_auc_score(pdf.is_fraud, pdf.score)), 4)
    imp = model.stages[-1].featureImportances.toArray()
    report["feature_importance"] = {f: round(float(v), 3) for f, v in
                                    sorted(zip(FEATURES, imp), key=lambda x: -x[1])[:8]}

    # plot: recall vs share of transactions reviewed
    order = np.argsort(-pdf.score.to_numpy())
    y = pdf.is_fraud.to_numpy()[order]
    amt = pdf.amount.to_numpy()[order] * y
    rates = np.linspace(0.0002, 0.01, 200)
    ks = (rates * len(y)).astype(int)
    cum, cum_v = np.cumsum(y), np.cumsum(amt)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(rates * 100, cum[ks - 1] / y.sum(), label="GBT model: fraud transactions caught")
    ax.plot(rates * 100, cum_v[ks - 1] / amt.sum(), "--", label="GBT model: fraud value caught")
    ax.scatter([rule_s["alert_rate"] * 100], [rule_s["recall"]], color="crimson", zorder=3,
               label=f"Rules ({rule_s['alert_rate']:.2%} of txns flagged)")
    ax.set(xlabel="Share of transactions sent for review (%)", ylabel="Share caught",
           title=f"Test window: {n_test/1e6:.1f}M transactions, days 61-90")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "recall_vs_review_rate.png", dpi=150)

    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
