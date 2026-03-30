"""
StreamLens — PySpark Feature Engineering Pipeline
===================================================
Computes user features, item features, co-watch matrix, and genre
co-occurrence from MovieLens ratings data.

At Netflix scale this same logic runs on EMR/Databricks over:
  - 238M user watch histories
  - 15K+ titles with metadata
  - Daily incremental refresh via Spark Structured Streaming

Locally: runs on ml-latest-small (100K ratings, 600 users, 9K movies)
Same code — different execution engine and data path.

Usage:
  pip install pyspark
  python spark/feature_engineering.py

Output artifacts:
  artifacts/spark/user_features.parquet     — per-user behavioral features
  artifacts/spark/item_features.parquet     — per-item popularity/quality
  artifacts/spark/cowatch_pairs.parquet     — collaborative filtering signal
  artifacts/spark/genre_stats.parquet       — genre co-occurrence matrix
  artifacts/spark/user_features.json        — exported for Redis feature store
  artifacts/spark/item_features.json        — exported for Redis feature store
"""

from __future__ import annotations

import json
import pathlib
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ── Constants ────────────────────────────────────────────────────────────────

RATINGS_PATH = "data/raw/movielens/ml-latest-small/ratings.csv"
MOVIES_PATH  = "data/raw/movielens/ml-latest-small/movies.csv"
OUT_DIR      = pathlib.Path("artifacts/spark")

MIN_COWATCH_SUPPORT = 5   # minimum co-viewers for a pair to be included


# ── Spark session ─────────────────────────────────────────────────────────────

def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("StreamLens-FeatureEngineering")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .master("local[*]")
        .getOrCreate()
    )


# ── Stage 1: User features ────────────────────────────────────────────────────

def compute_user_features(spark: SparkSession, ratings_path: str):
    """
    Per-user behavioral features for personalization and LTR scoring.

    At Netflix scale: derived from play events, watch completion,
    thumbs up/down signals — billions of rows in S3 Parquet.
    Here: derived from explicit ratings as a proxy.

    Features:
      watch_count         — total titles rated (proxy for engagement depth)
      avg_rating          — mean rating (taste calibration)
      rating_stddev       — rating variance (opinionated vs casual viewer)
      high_rating_ratio   — fraction of ratings >= 4.0 (positivity bias)
      low_rating_ratio    — fraction of ratings <= 2.0 (critical tendency)
      unique_titles       — distinct titles (breadth of consumption)
      taste_breadth       — log1p(unique_titles) (bounded diversity signal)
      last_active_ts      — recency of last interaction
    """
    ratings = spark.read.csv(ratings_path, header=True, inferSchema=True)

    user_features = ratings.groupBy("userId").agg(
        F.count("movieId").alias("watch_count"),
        F.avg("rating").alias("avg_rating"),
        F.stddev("rating").alias("rating_stddev"),
        F.max("rating").alias("max_rating"),
        F.min("rating").alias("min_rating"),
        F.countDistinct("movieId").alias("unique_titles"),
        F.sum(F.when(F.col("rating") >= 4.0, 1).otherwise(0)).alias("high_rating_count"),
        F.sum(F.when(F.col("rating") <= 2.0, 1).otherwise(0)).alias("low_rating_count"),
        F.max("timestamp").alias("last_active_ts"),
    ).withColumn(
        "high_rating_ratio",
        F.col("high_rating_count") / (F.col("watch_count") + 1e-9)
    ).withColumn(
        "low_rating_ratio",
        F.col("low_rating_count") / (F.col("watch_count") + 1e-9)
    ).withColumn(
        "taste_breadth",
        F.log1p(F.col("unique_titles"))
    )

    out = OUT_DIR / "user_features.parquet"
    user_features.write.parquet(str(out), mode="overwrite")
    n = user_features.count()
    print(f"  users: {n} → {out}")
    return user_features


# ── Stage 2: Item features ────────────────────────────────────────────────────

def compute_item_features(spark: SparkSession, ratings_path: str, movies_path: str):
    """
    Per-item quality and popularity features for retrieval scoring and LTR.

    At Netflix scale: also includes play_rate, completion_rate,
    rewatch_rate, social_share_count — signals unavailable in explicit ratings.

    Features:
      rating_count        — total ratings received (raw popularity)
      item_avg_rating     — mean rating (quality signal)
      item_rating_stddev  — rating variance (controversy / polarization)
      positive_ratings    — count of ratings >= 4.0
      negative_ratings    — count of ratings <= 2.0
      unique_viewers      — distinct users who rated
      popularity_score    — log1p(rating_count) (Zipf-smoothed popularity)
      sentiment_score     — positive_ratings / rating_count (net sentiment)
      controversy_score   — stddev (high = polarizing, e.g. Marmite films)
      release_year        — extracted from title string "(YYYY)"
      recency_score       — bucketed recency weight [0.2, 0.4, 0.7, 1.0]
    """
    ratings = spark.read.csv(ratings_path, header=True, inferSchema=True)
    movies  = spark.read.csv(movies_path,  header=True, inferSchema=True)

    item_features = ratings.groupBy("movieId").agg(
        F.count("userId").alias("rating_count"),
        F.avg("rating").alias("item_avg_rating"),
        F.stddev("rating").alias("item_rating_stddev"),
        F.sum(F.when(F.col("rating") >= 4.0, 1).otherwise(0)).alias("positive_ratings"),
        F.sum(F.when(F.col("rating") <= 2.0, 1).otherwise(0)).alias("negative_ratings"),
        F.countDistinct("userId").alias("unique_viewers"),
    ).join(movies, on="movieId", how="left") \
     .withColumn("popularity_score", F.log1p(F.col("rating_count"))) \
     .withColumn("sentiment_score",
                 F.col("positive_ratings") / (F.col("rating_count") + 1e-9)) \
     .withColumn("controversy_score", F.col("item_rating_stddev")) \
     .withColumn("release_year",
                 F.when(F.regexp_extract(F.col("title"), r"\((\d{4})\)", 1) != "", F.regexp_extract(F.col("title"), r"\((\d{4})\)", 1).cast("int")).otherwise(None)) \
     .withColumn("recency_score",
                 F.when(F.col("release_year") >= 2010, 1.0)
                  .when(F.col("release_year") >= 2000, 0.7)
                  .when(F.col("release_year") >= 1990, 0.4)
                  .otherwise(0.2))

    out = OUT_DIR / "item_features.parquet"
    item_features.write.parquet(str(out), mode="overwrite")
    n = item_features.count()
    print(f"  items: {n} → {out}")
    return item_features


# ── Stage 3: Co-watch pairs ───────────────────────────────────────────────────

def compute_cowatch_pairs(spark: SparkSession, ratings_path: str):
    """
    Collaborative filtering signal: pairs of movies co-watched by the same user.

    At Netflix scale: this is the input to ALS matrix factorization,
    graph neural network training, and 'Because you watched X' generation.
    Here: produces a co-watch count matrix usable as an LTR feature.

    Output: (movieId_a, movieId_b, cowatch_count, avg_combined_rating)
    Only pairs with cowatch_count >= MIN_COWATCH_SUPPORT are kept.
    """
    ratings = spark.read.csv(ratings_path, header=True, inferSchema=True)

    r1 = ratings.select(
        F.col("userId"),
        F.col("movieId").alias("movieId_a"),
        F.col("rating").alias("rating_a"),
    )
    r2 = ratings.select(
        F.col("userId"),
        F.col("movieId").alias("movieId_b"),
        F.col("rating").alias("rating_b"),
    )

    cowatch = (
        r1.join(r2, on="userId")
          .filter(F.col("movieId_a") < F.col("movieId_b"))
          .groupBy("movieId_a", "movieId_b")
          .agg(
              F.count("userId").alias("cowatch_count"),
              F.avg(F.col("rating_a") + F.col("rating_b")).alias("avg_combined_rating"),
          )
          .filter(F.col("cowatch_count") >= MIN_COWATCH_SUPPORT)
    )

    out = OUT_DIR / "cowatch_pairs.parquet"
    cowatch.write.parquet(str(out), mode="overwrite")
    n = cowatch.count()
    print(f"  co-watch pairs (support>={MIN_COWATCH_SUPPORT}): {n} → {out}")
    return cowatch


# ── Stage 4: Genre co-occurrence ──────────────────────────────────────────────

def compute_genre_stats(spark: SparkSession, movies_path: str, ratings_path: str):
    """
    Genre-level statistics and user-genre affinity signals.

    At Netflix scale: powers row-level genre personalization,
    cold-start genre matching, and diversity constraints in slate optimization.

    Output: per-genre user count, avg rating, total watches, popularity rank.
    """
    movies  = spark.read.csv(movies_path,  header=True, inferSchema=True)
    ratings = spark.read.csv(ratings_path, header=True, inferSchema=True)

    movies_genres = movies.withColumn(
        "genre", F.explode(F.split(F.col("genres"), r"\|"))
    ).filter(F.col("genre") != "(no genres listed)")

    genre_stats = (
        ratings.join(movies_genres, on="movieId")
               .groupBy("genre")
               .agg(
                   F.countDistinct("userId").alias("user_count"),
                   F.avg("rating").alias("avg_rating"),
                   F.count("movieId").alias("total_ratings"),
                   F.countDistinct("movieId").alias("title_count"),
               )
               .withColumn("popularity_rank",
                           F.rank().over(Window.orderBy(F.desc("total_ratings"))))
               .orderBy("popularity_rank")
    )

    out = OUT_DIR / "genre_stats.parquet"
    genre_stats.write.parquet(str(out), mode="overwrite")
    n = genre_stats.count()
    print(f"  genres: {n} → {out}")
    return genre_stats


# ── Stage 5: Export to Redis feature store format ─────────────────────────────

def export_for_feature_store(spark: SparkSession) -> None:
    """
    Export Spark Parquet features to JSON for the Redis online feature store.

    At Netflix scale: Flink or Kafka Streams would read the Parquet files
    and write to the online feature store (DynamoDB / Redis Cluster) in
    real-time. Here: batch export to JSON files consumed by the FastAPI app.

    This bridges the offline Spark batch pipeline to the online serving path.
    """
    # User features
    user_f = spark.read.parquet(str(OUT_DIR / "user_features.parquet"))
    user_dict = {}
    for row in user_f.select(
        "userId", "watch_count", "avg_rating",
        "high_rating_ratio", "taste_breadth", "rating_stddev"
    ).collect():
        user_dict[str(row["userId"])] = {
            "watch_count":        int(row["watch_count"] or 0),
            "avg_rating":         round(float(row["avg_rating"] or 0), 4),
            "high_rating_ratio":  round(float(row["high_rating_ratio"] or 0), 4),
            "taste_breadth":      round(float(row["taste_breadth"] or 0), 4),
            "rating_stddev":      round(float(row["rating_stddev"] or 0), 4),
        }
    out_u = OUT_DIR / "user_features.json"
    json.dump(user_dict, open(out_u, "w"), indent=2)
    print(f"  user features JSON: {len(user_dict)} users → {out_u}")

    # Item features
    item_f = spark.read.parquet(str(OUT_DIR / "item_features.parquet"))
    item_dict = {}
    for row in item_f.select(
        "movieId", "popularity_score", "item_avg_rating",
        "sentiment_score", "recency_score", "controversy_score"
    ).collect():
        item_dict[str(row["movieId"])] = {
            "popularity":    round(float(row["popularity_score"] or 0), 4),
            "avg_rating":    round(float(row["item_avg_rating"] or 0), 4),
            "sentiment":     round(float(row["sentiment_score"] or 0), 4),
            "recency":       round(float(row["recency_score"] or 0), 4),
            "controversy":   round(float(row["controversy_score"] or 0), 4),
        }
    out_i = OUT_DIR / "item_features.json"
    json.dump(item_dict, open(out_i, "w"), indent=2)
    print(f"  item features JSON: {len(item_dict)} items → {out_i}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("StreamLens — PySpark Feature Engineering Pipeline")
    print("=" * 60)

    # Check data exists
    if not pathlib.Path(RATINGS_PATH).exists():
        print(f"\nERROR: {RATINGS_PATH} not found.")
        print("Download: https://grouplens.org/datasets/movielens/latest/")
        print("Extract to: data/raw/movielens/ml-latest-small/")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\nSpark version: {spark.version}")
    print(f"Input:  {RATINGS_PATH}")
    print(f"Output: {OUT_DIR}/\n")

    print("[1/5] Computing user features...")
    compute_user_features(spark, RATINGS_PATH)

    print("[2/5] Computing item features...")
    compute_item_features(spark, RATINGS_PATH, MOVIES_PATH)

    print("[3/5] Computing co-watch pairs...")
    compute_cowatch_pairs(spark, RATINGS_PATH)

    print("[4/5] Computing genre co-occurrence stats...")
    compute_genre_stats(spark, MOVIES_PATH, RATINGS_PATH)

    print("[5/5] Exporting to Redis feature store format...")
    export_for_feature_store(spark)

    print("\n" + "=" * 60)
    print("DONE — all feature artifacts written to artifacts/spark/")
    print()
    print("Production notes:")
    print("  • Replace local[*]  → EMR/Databricks cluster")
    print("  • Replace CSV paths → s3://your-bucket/movielens/")
    print("  • Replace JSON export → Flink → DynamoDB/Redis Cluster")
    print("  • Add incremental refresh via Spark Structured Streaming")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
