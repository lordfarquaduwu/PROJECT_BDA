import math
import time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array

DB = "ci3237_project"
SEED = 42
K_RANGE = range(2, 9)

# Step 2: after reading the comparison table, set these and run again
FINAL_ALGO = "kmeans"     # KMeans with k-means|| initialisation
FINAL_K = 4

spark = (SparkSession.builder.appName("rfm_kmeans")
    .config("spark.jars.packages",
            "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017")
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017")
    .config("spark.sql.session.timeZone", "Asia/Brunei")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")


# ---------------- Davies-Bouldin (from lab 5) ----------------
def davies_bouldin(pred, d):
    arr = pred.withColumn("f", vector_to_array("features")).cache()
    rows = (arr.groupBy("prediction")
            .agg(*[F.avg(F.col("f")[i]).alias(f"c{i}") for i in range(d)])
            .collect())
    centres = {int(r["prediction"]): [float(r[f"c{i}"]) for i in range(d)]
               for r in rows}
    centres_df = pred.sparkSession.createDataFrame(
        [(cid, vals) for cid, vals in centres.items()],
        schema="prediction int, centre array<double>")
    spread_rows = (arr.join(F.broadcast(centres_df), "prediction")
                   .withColumn("dist", F.sqrt(F.aggregate(
                       F.zip_with("f", "centre", lambda a, b: (a - b) * (a - b)),
                       F.lit(0.0), lambda acc, x: acc + x)))
                   .groupBy("prediction")
                   .agg(F.avg("dist").alias("spread"))
                   .collect())
    spread = {int(r["prediction"]): float(r["spread"]) for r in spread_rows}
    arr.unpersist()
    ids = sorted(centres)
    if len(ids) < 2:
        return float("nan")
    total = 0.0
    for i in ids:
        worst = 0.0
        for j in ids:
            if i == j:
                continue
            sep = math.dist(centres[i], centres[j])
            if sep > 0:
                worst = max(worst, (spread[i] + spread[j]) / sep)
        total += worst
    return total / len(ids)


# ---------------- RFM features ----------------
cs = (spark.read.format("mongodb").option("database", DB)
      .option("collection", "customer_summary").load().drop("_id"))

ref = cs.agg(F.max("last_purchase")).first()[0]          # snapshot date
print("reference date (latest purchase in data):", ref)

rfm = (cs.select("CustomerID",
                 (F.datediff(F.lit(ref).cast("date"),
                             F.col("last_purchase").cast("date")) + 1).alias("recency"),
                 F.col("n_invoices").alias("frequency"),
                 F.col("total_spend").alias("monetary"))
         .filter(F.col("monetary") > 0))
print("customers:", rfm.count())
rfm.describe("recency", "frequency", "monetary").show()

# RFM is heavily right-skewed, so log-transform before standardising
for c in ["recency", "frequency", "monetary"]:
    rfm = rfm.withColumn("log_" + c, F.log1p(F.col(c)))

assembler = VectorAssembler(inputCols=["log_recency", "log_frequency", "log_monetary"],
                            outputCol="rawFeatures")
assembled = assembler.transform(rfm)
scaled = (StandardScaler(inputCol="rawFeatures", outputCol="features",
                         withMean=True, withStd=True)
          .fit(assembled).transform(assembled)).cache()
scaled.count()
D = 3

evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction",
                                metricName="silhouette",
                                distanceMeasure="squaredEuclidean")


def make(algo, k):
    if algo == "kmeans":
        return KMeans(featuresCol="features", k=k, seed=SEED,
                      initMode="k-means||", maxIter=50)
    raise ValueError("Only KMeans with k-means|| is supported")


# ---------------- Step 1: evaluate KMeans with k-means|| over a range of k ----------------
print("\nalgorithm | k | silhouette | davies_bouldin | within-cluster cost | fit s")
for algo in ["kmeans"]:
    for k in K_RANGE:
        t = time.time()
        model = make(algo, k).fit(scaled)
        secs = time.time() - t
        pred = model.transform(scaled).select("features", "prediction").cache()
        sil = evaluator.evaluate(pred)
        db = davies_bouldin(pred, D)
        cost = model.summary.trainingCost
        pred.unpersist()
        print(algo, "|", k, "|", round(sil, 4), "|", round(db, 4), "|",
              round(cost, 1), "|", round(secs, 1))

# ---------------- Step 2: final model, profile and store ----------------
print(f"\nFinal model: {FINAL_ALGO}, k={FINAL_K}")
final = make(FINAL_ALGO, FINAL_K).fit(scaled)
result = final.transform(scaled)

profile = (result.groupBy("prediction").agg(
    F.count("*").alias("customers"),
    F.round(F.avg("recency"), 1).alias("avg_recency_days"),
    F.round(F.avg("frequency"), 1).alias("avg_orders"),
    F.round(F.avg("monetary"), 0).alias("avg_spend"))
    .orderBy("prediction"))
profile.show()

(result.select("CustomerID", "recency", "frequency", "monetary",
               F.col("prediction").alias("cluster"))
       .write.format("mongodb").option("database", DB)
       .option("collection", "customer_clusters").mode("overwrite").save())
print("saved customer_clusters")

spark.stop()
