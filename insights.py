"""Load the pipeline outputs into Hive tables and answer business questions in HiveQL.

Creates database `fraud` with:
  fraud.transactions   raw transactions, partitioned by txn_date
  fraud.txn_features   engineered features, partitioned by txn_date
  fraud.test_scores    model score and rule flag for every test-window transaction
Then runs the queries in insights.sql and writes results/insights.json.

Usage: python insights.py TX_PARQUET_DIR WORK_DIR
"""
import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession

THRESH = json.loads(Path("results/results.json").read_text())["model_at_budget"]["0.2%"]["threshold"]


def main(tx_dir, work):
    spark = (SparkSession.builder.master("local[*]").appName("card-fraud-insights")
             .config("spark.ui.enabled", "false").config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "32")
             .config("spark.sql.warehouse.dir", f"{work}/warehouse")
             .config("javax.jdo.option.ConnectionURL", f"jdbc:derby:;databaseName={work}/metastore_db;create=true")
             .enableHiveSupport().getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    spark.sql("CREATE DATABASE IF NOT EXISTS fraud")
    spark.read.parquet(tx_dir).write.mode("overwrite").partitionBy("txn_date").saveAsTable("fraud.transactions")
    spark.read.parquet(f"{work}/features").write.mode("overwrite").partitionBy("txn_date").saveAsTable("fraud.txn_features")
    spark.read.parquet(f"{work}/test_scores").select("txn_id", "score", "rule_flag").write.mode("overwrite").saveAsTable("fraud.test_scores")

    out = {"hive_tables": {t.name: spark.table(f"fraud.{t.name}").count()
                           for t in spark.catalog.listTables("fraud")},
           "partitions_transactions": spark.sql("SHOW PARTITIONS fraud.transactions").count(),
           "alert_threshold_for_0.2pct_budget": THRESH}
    queries = Path("insights.sql").read_text().replace("${THRESH}", str(THRESH))
    for block in queries.split("-- @")[1:]:
        name, sql = block.split("\n", 1)
        out[name.strip()] = [r.asDict() for r in spark.sql(sql.strip().rstrip(";")).collect()]
    Path("results/insights.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
