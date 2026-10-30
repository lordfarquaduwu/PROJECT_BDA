import time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.fpm import FPGrowth

DB = "ci3237_project"
CONF = 0.3

spark = (SparkSession.builder.appName("fp_sensitivity")
    .config("spark.jars.packages",
            "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017")
    .config("spark.sql.session.timeZone", "Asia/Brunei")
    .config("spark.driver.extraJavaOptions", "-Xss64m")
    .config("spark.executor.extraJavaOptions", "-Xss64m")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

raw = (spark.read.format("mongodb").option("database", DB)
       .option("collection", "transactions").load().drop("_id"))

n_inv_raw = raw.select("Invoice").distinct().count()
n_inv_cancel = (raw.filter(F.col("Invoice").startswith("C"))
                .select("Invoice").distinct().count())
print("distinct invoices in raw:", n_inv_raw)
print("cancelled invoices:", n_inv_cancel)
print("non-cancelled invoices:", n_inv_raw - n_inv_cancel)

clean = (raw.filter(~F.col("Invoice").startswith("C"))
            .filter((F.col("Quantity") > 0) & (F.col("Price") > 0))
            .filter(F.col("Invoice") != "581483")
            .filter(F.col("StockCode").rlike(r"^\d{5}")))
baskets = (clean.groupBy("Invoice")
           .agg(F.collect_set("StockCode").alias("items")).cache())
n = baskets.count()
print("baskets after cleaning:", n)

print("min_support | min_baskets | itemsets | rules | max_itemset_size | seconds")
for s in [0.02, 0.01, 0.005]:
    t = time.time()
    m = FPGrowth(itemsCol="items", minSupport=s, minConfidence=CONF).fit(baskets)
    fi = m.freqItemsets.cache()
    n_fi = fi.count()
    n_rules = m.associationRules.count()
    max_size = fi.select(F.max(F.size("items"))).first()[0]
    print(s, "|", round(s * n), "|", n_fi, "|", n_rules, "|", max_size,
          "|", round(time.time() - t))

spark.stop()