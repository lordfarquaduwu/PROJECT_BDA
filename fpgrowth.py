from pyspark.sql import SparkSession, functions as F
from pyspark.ml.fpm import FPGrowth

DB = "ci3237_project"
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.3

spark = (SparkSession.builder.appName("fpgrowth")
    .config("spark.jars.packages",
            "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017")
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017")
    .config("spark.sql.session.timeZone", "Asia/Brunei")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

raw = (spark.read.format("mongodb").option("database", DB)
       .option("collection", "transactions").load().drop("_id"))

clean = (raw.filter(~F.col("Invoice").startswith("C"))
            .filter((F.col("Quantity") > 0) & (F.col("Price") > 0))
            .filter(F.col("Invoice") != "581483")
            .filter(F.col("StockCode").rlike(r"^\d{5}")))

baskets = (clean.groupBy("Invoice")
           .agg(F.collect_set("StockCode").alias("items")).cache())
n_baskets = baskets.count()
print("baskets (invoices):", n_baskets)
print("avg distinct items per basket:",
      round(baskets.select(F.avg(F.size("items"))).first()[0], 2))
print("min support:", MIN_SUPPORT, "= at least", round(MIN_SUPPORT * n_baskets), "baskets")
print("min confidence:", MIN_CONFIDENCE)

model = FPGrowth(itemsCol="items", minSupport=MIN_SUPPORT,
                 minConfidence=MIN_CONFIDENCE).fit(baskets)
itemsets = model.freqItemsets
rules = model.associationRules
print("frequent itemsets:", itemsets.count())
print("association rules:", rules.count())
itemsets.groupBy(F.size("items").alias("size")).count().orderBy("size").show()

desc = {r["StockCode"]: r["Description"] for r in
        spark.read.format("mongodb").option("database", DB)
        .option("collection", "product_summary").load()
        .select("StockCode", "Description").collect()}
print("Top 10 rules by lift:")
for r in rules.orderBy(F.desc("lift")).limit(10).collect():
    a = [desc.get(x, x) for x in r["antecedent"]]
    c = [desc.get(x, x) for x in r["consequent"]]
    print(a, "->", c, "| support", round(r["support"], 4),
          "| confidence", round(r["confidence"], 3), "| lift", round(r["lift"], 1))

for name, df in [("freq_itemsets", itemsets), ("association_rules", rules)]:
    (df.write.format("mongodb").option("database", DB)
       .option("collection", name).mode("overwrite").save())

spark.stop()