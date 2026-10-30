from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp

spark = (SparkSession.builder.appName("ingest")
    .config("spark.jars.packages",
            "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017")
    .getOrCreate())

df = (spark.read.option("header", True).option("inferSchema", True)
      .csv(r"C:\BDA_Project\online_retail_II.csv")
      .withColumnRenamed("Customer ID", "CustomerID")
      .withColumn("InvoiceDate", to_timestamp("InvoiceDate")))

n = df.count()
print("rows read from CSV:", n)
df.printSchema()

(df.write.format("mongodb")
   .option("database", "ci3237_project")
   .option("collection", "transactions")
   .mode("overwrite").save())

print("write finished")
spark.stop()