"""Checks the Spark SQL features on a hand-made card history, and the generator on a small run."""
import datetime as dt
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parent.parent
T0 = 1_735_689_600


@pytest.fixture(scope="module")
def spark():
    s = (SparkSession.builder.master("local[1]").appName("tests")
         .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "2").getOrCreate())
    yield s
    s.stop()


def features_for(spark, rows):
    cols = ["card_id", "ts", "amount", "mcc", "online", "foreign", "is_fraud", "fraud_type", "txn_id", "txn_date"]
    spark.createDataFrame(rows, cols).createOrReplaceTempView("tx")
    out = spark.sql((ROOT / "features.sql").read_text()).orderBy("ts").collect()
    return {r["txn_id"]: r for r in out}


def test_velocity_and_baseline_use_only_the_past(spark):
    day = 86400
    rows = [  # one card: three normal purchases over two days, then a 3-transaction burst
        (1, T0, 100.0, "grocery", False, False, 0, "none", 1, dt.date(2025, 1, 1)),
        (1, T0 + day, 300.0, "grocery", False, False, 0, "none", 2, dt.date(2025, 1, 1)),
        (1, T0 + 2 * day, 200.0, "fuel", False, False, 0, "none", 3, dt.date(2025, 1, 1)),
        (1, T0 + 2 * day + 600, 50.0, "electronics", True, True, 1, "burst", 4, dt.date(2025, 1, 1)),
        (1, T0 + 2 * day + 1200, 2000.0, "electronics", True, True, 1, "burst", 5, dt.date(2025, 1, 1)),
        (2, T0 + 2 * day + 1200, 10.0, "grocery", False, False, 0, "none", 6, dt.date(2025, 1, 1)),  # other card
    ]
    f = features_for(spark, rows)
    assert f[1]["cnt_1h"] == 1 and f[1]["cnt_30d"] == 0          # first ever txn: no history
    assert f[5]["cnt_1h"] == 3                                    # txns 3, 4, 5 within the hour
    assert f[5]["cnt_24h"] == 3                                   # txn 2 is exactly 24h+20min earlier
    assert f[5]["secs_since_prev"] == 600
    assert f[6]["cnt_1h"] == 1                                    # other card's activity is separate
    # baseline for txn 5 = average of the 4 earlier txns (current one excluded)
    assert f[5]["amt_vs_avg"] == pytest.approx(2000 / ((100 + 300 + 200 + 50) / 4))
    assert f[5]["foreign_surprise"] == pytest.approx(1 - 1 / 4)  # 1 of 4 earlier txns was foreign


def test_generator_small_run(spark, tmp_path):
    import generate_data as g
    spark.stop()  # the generator creates its own session
    g.main(str(tmp_path / "tx"), 20_000)
    s = SparkSession.builder.master("local[1]").getOrCreate()
    d = s.read.parquet(str(tmp_path / "tx"))
    stats = d.groupBy("fraud_type").count().collect()
    types = {r["fraud_type"] for r in stats}
    assert {"none", "burst", "card_test", "stealth"} <= types
    assert d.where("is_fraud = 1").count() > 0
    assert d.select("txn_date").distinct().count() >= 60     # partitioned by day over ~90 days


def test_alert_reasons_are_specific_and_readable():
    from alert_queue import reasons_for
    base = dict(small_cnt_1h=0, cnt_1h=1, cnt_30d=40, amt_vs_avg=1.0, spend24h_vs_avg=1.0, foreign=0,
                foreign_rate_30d=0.0, online=0, online_rate_30d=0.5, mcc="grocery", secs_since_prev=86400, hour=14)
    assert reasons_for(base) == ["unusual combination of signals for this card (see model score)"]
    burst = reasons_for({**base, "cnt_1h": 5, "amt_vs_avg": 6.2, "foreign": 1, "mcc": "electronics", "secs_since_prev": 40})
    assert "5 transactions on this card in the last hour" in burst
    assert "amount is 6.2x this card's 30-day average" in burst
    assert "foreign transaction on a card that is rarely used abroad" in burst
    assert any("electronics" in r for r in burst) and any("40s" in r for r in burst)
    testing = reasons_for({**base, "small_cnt_1h": 4, "cnt_1h": 4})
    assert testing[0].startswith("4 small charges") and "card-testing" in testing[0]
    # a new card (little history) shouldn't get "x times its average" reasons
    assert not any("average" in r for r in reasons_for({**base, "cnt_30d": 1, "amt_vs_avg": 9.0}))
