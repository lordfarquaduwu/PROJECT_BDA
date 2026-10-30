from pyspark.sql import SparkSession, functions as F

DB = "ci3237_project"

spark = (SparkSession.builder.appName("aggregate")
    .config("spark.jars.packages",
            "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017")
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017")
    .config("spark.sql.session.timeZone", "Asia/Brunei")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

raw = (spark.read.format("mongodb")
       .option("database", DB).option("collection", "transactions")
       .load().drop("_id"))

# ---- Cleaning, with a count at every step (cite these in the report) ----
n_raw = raw.count()
is_cancel = F.col("Invoice").startswith("C")
n_cancel_lines = raw.filter(is_cancel).count()
n_cancel_invoices = raw.filter(is_cancel).select("Invoice").distinct().count()

clean = (raw.filter(~is_cancel)
            .filter((F.col("Quantity") > 0) & (F.col("Price") > 0))
            .filter(F.col("Invoice") != "581483")
            .withColumn("Revenue", F.col("Quantity") * F.col("Price"))
            .cache())
n_clean = clean.count()
n_other_excluded = n_raw - n_cancel_lines - n_clean

with_cust = clean.filter(F.col("CustomerID").isNotNull())
n_with_cust = with_cust.count()

print("raw rows:", n_raw)
print("cancelled lines (Invoice starts with C):", n_cancel_lines)
print("cancelled invoices:", n_cancel_invoices)
print("other excluded (Quantity<=0 or Price<=0, not cancellations):", n_other_excluded)
print("clean sales lines:", n_clean)
print("clean lines with no CustomerID:", n_clean - n_with_cust)
print("clean lines with CustomerID:", n_with_cust)

# ---- Per-customer view ----
customer_summary = (with_cust.groupBy("CustomerID").agg(
    F.first("Country").alias("Country"),
    F.countDistinct("Invoice").alias("n_invoices"),
    F.count("*").alias("n_lines"),
    F.countDistinct("StockCode").alias("n_distinct_products"),
    F.sum("Quantity").alias("total_items"),
    F.round(F.sum("Revenue"), 2).alias("total_spend"),
    F.min("InvoiceDate").alias("first_purchase"),
    F.max("InvoiceDate").alias("last_purchase"))
    .withColumn("avg_order_value",
                F.round(F.col("total_spend") / F.col("n_invoices"), 2)))

# ---- Per-product view (real products only: StockCode starts with 5 digits) ----
products = clean.filter(F.col("StockCode").rlike(r"^\d{5}"))
print("clean lines on real product codes:", products.count())

product_summary = (products.groupBy("StockCode").agg(
    F.first("Description").alias("Description"),
    F.sum("Quantity").alias("units_sold"),
    F.round(F.sum("Revenue"), 2).alias("revenue"),
    F.countDistinct("Invoice").alias("n_invoices"),
    F.countDistinct("CustomerID").alias("n_customers"),
    F.round(F.avg("Price"), 2).alias("avg_price")))

for name, df in [("customer_summary", customer_summary),
                 ("product_summary", product_summary)]:
    (df.write.format("mongodb").option("database", DB)
       .option("collection", name).mode("overwrite").save())

print("customer_summary rows:", customer_summary.count())
print("product_summary rows:", product_summary.count())
customer_summary.orderBy(F.desc("total_spend")).show(5, truncate=False)
product_summary.orderBy(F.desc("revenue")).show(5, truncate=False)

spark.stop()