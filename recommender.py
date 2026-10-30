import time
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS

DB = "ci3237_project"
SEED = 42
K_LIST = [5, 10, 20]          # precision@K / recall@K reported for each
GRID = [(10, 0.01), (10, 0.1), (30, 0.01), (30, 0.1)]   # (rank, regParam)
ALPHA = 40.0
TOP_N = 10                    # recommendations stored per customer / product

spark = (SparkSession.builder.appName("recommender")
    .config("spark.jars.packages",
            "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017")
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017")
    .config("spark.sql.session.timeZone", "Asia/Brunei")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# ---------------- 1. Customer x product interactions ----------------
raw = (spark.read.format("mongodb").option("database", DB)
       .option("collection", "transactions").load().drop("_id"))

clean = (raw.filter(~F.col("Invoice").startswith("C"))
            .filter((F.col("Quantity") > 0) & (F.col("Price") > 0))
            .filter(F.col("Invoice") != "581483")
            .filter(F.col("StockCode").rlike(r"^\d{5}"))
            .filter(F.col("CustomerID").isNotNull()))

# strength = number of distinct invoices in which the customer bought the product
pairs = (clean.groupBy("CustomerID", "StockCode")
              .agg(F.countDistinct("Invoice").alias("strength")))

indexer = StringIndexer(inputCol="StockCode", outputCol="item_f").fit(pairs)
item_codes = indexer.labels                      # index -> StockCode
inter = (indexer.transform(pairs)
         .select(F.col("CustomerID").alias("user"),
                 F.col("item_f").cast("int").alias("item"),
                 F.col("strength").cast("float").alias("strength"))
         .cache())

n_inter = inter.count()
n_users = inter.select("user").distinct().count()
n_items = inter.select("item").distinct().count()
print("customers:", n_users, "| products:", n_items, "| interactions:", n_inter)
print("matrix density:", round(100 * n_inter / (n_users * n_items), 3), "%")

desc = {r["StockCode"]: r["Description"] for r in
        spark.read.format("mongodb").option("database", DB)
        .option("collection", "product_summary").load()
        .select("StockCode", "Description").collect()}

# ---------------- 2. Splits: train / validation (tuning) / test (final) ----------------
train_full, test = inter.randomSplit([0.8, 0.2], seed=SEED)
train, val = train_full.randomSplit([0.875, 0.125], seed=SEED)
for c in (train_full, test, train, val):
    c.cache()
print("train_full:", train_full.count(), "| test:", test.count(),
      "| train:", train.count(), "| val:", val.count())


def to_map(df):
    rows = df.groupBy("user").agg(F.collect_set("item").alias("items")).collect()
    return {int(r["user"]): set(int(x) for x in r["items"]) for r in rows}


def fit_als(df, rank, reg):
    return ALS(userCol="user", itemCol="item", ratingCol="strength",
               implicitPrefs=True, rank=rank, regParam=reg, alpha=ALPHA,
               maxIter=10, seed=SEED).fit(df)


def factors(model):
    u = model.userFactors.collect()
    i = model.itemFactors.collect()
    uid = np.array([r["id"] for r in u])
    U = np.array([r["features"] for r in u], dtype=np.float32)
    iid = np.array([r["id"] for r in i])
    I = np.array([r["features"] for r in i], dtype=np.float32)
    return uid, U, iid, I


def score_recs(rec_fn, users, held, ks):
    """rec_fn(user, k) -> set of k recommended item ids."""
    out = {}
    for k in ks:
        p, r, h = [], [], []
        for u in users:
            rec = rec_fn(u, k)
            hits = len(rec & held[u])
            p.append(hits / k)
            r.append(hits / len(held[u]))
            h.append(1.0 if hits > 0 else 0.0)
        out[k] = (float(np.mean(p)), float(np.mean(r)), float(np.mean(h)))
    return out


def evaluate(model, seen, heldout, ks):
    """Evaluate ALS and a popularity baseline on the same users and items."""
    seen_map, held_map = to_map(seen), to_map(heldout)
    uid, U, iid, I = factors(model)
    col_of = {int(x): j for j, x in enumerate(iid)}      # only items seen in training
    row_of = {int(x): j for j, x in enumerate(uid)}
    held = {}
    for u, items in held_map.items():
        keep = {x for x in items if x in col_of}
        if keep and u in row_of:
            held[u] = keep
    users = list(held)

    scores = U @ I.T                                      # users x items

    def als_rec(u, k):
        s = scores[row_of[u]].copy()
        for it in seen_map.get(u, ()):
            j = col_of.get(it)
            if j is not None:
                s[j] = -np.inf
        top = np.argpartition(-s, k)[:k]
        return {int(iid[j]) for j in top}

    pop_count = {}
    for items in seen_map.values():
        for it in items:
            pop_count[it] = pop_count.get(it, 0) + 1
    pop_rank = [it for it, _ in sorted(pop_count.items(), key=lambda x: -x[1])
                if it in col_of]

    def pop_rec(u, k):
        s = seen_map.get(u, set())
        out = []
        for it in pop_rank:
            if it not in s:
                out.append(it)
                if len(out) == k:
                    break
        return set(out)

    return (len(users), score_recs(als_rec, users, held, ks),
            score_recs(pop_rec, users, held, ks))


# ---------------- 3. Tune on the validation split ----------------
print("\nTuning (precision@10 / recall@10 on validation split)")
print("rank | regParam | users | precision@10 | recall@10 | fit s")
best, best_recall = None, -1
train_seen = train
for rank, reg in GRID:
    t = time.time()
    m = fit_als(train, rank, reg)
    secs = time.time() - t
    n_u, als_m, _ = evaluate(m, train_seen, val, [10])
    p, r, h = als_m[10]
    print(rank, "|", reg, "|", n_u, "|", round(p, 4), "|", round(r, 4), "|", round(secs, 1))
    if r > best_recall:
        best, best_recall = (rank, reg), r
print("best (rank, regParam):", best)

# ---------------- 4. Final evaluation on the held-out test split ----------------
model = fit_als(train_full, *best)
n_u, als_m, pop_m = evaluate(model, train_full, test, K_LIST)
print("\nTEST SET: evaluated on", n_u, "customers with held-out purchases")
print("K | ALS precision | ALS recall | ALS hit-rate | Pop precision | Pop recall | Pop hit-rate")
for k in K_LIST:
    a, b = als_m[k], pop_m[k]
    print(k, "|", round(a[0], 4), "|", round(a[1], 4), "|", round(a[2], 4),
          "|", round(b[0], 4), "|", round(b[1], 4), "|", round(b[2], 4))

# ---------------- 5. Final model on all data: recommend + store ----------------
final = fit_als(inter, *best)
uid, U, iid, I = factors(final)
col_of = {int(x): j for j, x in enumerate(iid)}
all_seen = to_map(inter)

# customer -> top-N products (already-bought products excluded)
scores = U @ I.T
cust_rows = []
for r, u in enumerate(uid):
    u = int(u)
    s = scores[r].copy()
    for it in all_seen.get(u, ()):
        j = col_of.get(it)
        if j is not None:
            s[j] = -np.inf
    top = np.argsort(-s)[:TOP_N]
    cust_rows.append((u, [item_codes[int(iid[j])] for j in top]))

# product -> top-N similar products (cosine similarity of item factors)
In = I / (np.linalg.norm(I, axis=1, keepdims=True) + 1e-9)
sim = In @ In.T
np.fill_diagonal(sim, -np.inf)
prod_rows = []
for j in range(len(iid)):
    top = np.argsort(-sim[j])[:TOP_N]
    prod_rows.append((item_codes[int(iid[j])], [item_codes[int(iid[x])] for x in top]))

(spark.createDataFrame(cust_rows, "CustomerID int, recommendations array<string>")
      .write.format("mongodb").option("database", DB)
      .option("collection", "customer_recommendations").mode("overwrite").save())
(spark.createDataFrame(prod_rows, "StockCode string, similar_products array<string>")
      .write.format("mongodb").option("database", DB)
      .option("collection", "product_similar").mode("overwrite").save())
print("\nsaved customer_recommendations:", len(cust_rows),
      "| product_similar:", len(prod_rows))

# ---------------- 6. Sanity demo ----------------
lookup = dict(prod_rows)
print("\nSimilar products (sanity check against FP-Growth sets):")
for code in ["22745", "85123A", "22423"]:
    if code in lookup:
        print(code, desc.get(code, "?"))
        for c in lookup[code][:5]:
            print("    ->", c, desc.get(c, "?"))

cust = dict(cust_rows)
for cid in [18102, 14646]:
    if cid in cust:
        print("\nTop recommendations for customer", cid)
        for c in cust[cid][:5]:
            print("    ->", c, desc.get(c, "?"))

spark.stop()
