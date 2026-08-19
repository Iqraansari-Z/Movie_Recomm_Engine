"""
Phase 1 - Data Acquisition (cont.)
Loads the MovieLens 1M .dat files into Spark DataFrames and runs basic
sanity checks: row counts, nulls, duplicates.

MovieLens 1M schema:
    ratings.dat -> UserID::MovieID::Rating::Timestamp
    movies.dat  -> MovieID::Title::Genres
    users.dat   -> UserID::Gender::Age::Occupation::Zip-code

Usage:
    python src/load_data.py
"""
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, LongType
)
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def get_spark(app_name: str = "MovieRecEngine") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def load_ratings(spark: SparkSession):
    schema = StructType([
        StructField("user_id", IntegerType(), False),
        StructField("movie_id", IntegerType(), False),
        StructField("rating", IntegerType(), False),
        StructField("timestamp", LongType(), False),
    ])
    path = str(RAW_DIR / "ratings.dat")
    return (
        spark.read
        .option("encoding", "ISO-8859-1")
        .csv(path, sep="::", schema=schema)
    )


def load_movies(spark: SparkSession):
    schema = StructType([
        StructField("movie_id", IntegerType(), False),
        StructField("title", StringType(), False),
        StructField("genres", StringType(), False),
    ])
    path = str(RAW_DIR / "movies.dat")
    return (
        spark.read
        .option("encoding", "ISO-8859-1")
        .csv(path, sep="::", schema=schema)
    )


def load_users(spark: SparkSession):
    schema = StructType([
        StructField("user_id", IntegerType(), False),
        StructField("gender", StringType(), False),
        StructField("age", IntegerType(), False),
        StructField("occupation", IntegerType(), False),
        StructField("zip_code", StringType(), False),
    ])
    path = str(RAW_DIR / "users.dat")
    return (
        spark.read
        .option("encoding", "ISO-8859-1")
        .csv(path, sep="::", schema=schema)
    )


def sanity_check(name: str, df) -> None:
    print(f"\n--- {name} ---")
    n_rows = df.count()
    n_cols = len(df.columns)
    print(f"Rows: {n_rows:,} | Columns: {n_cols}")

    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns
    ]).collect()[0].asDict()
    total_nulls = sum(null_counts.values())
    print(f"Null values: {total_nulls}")
    if total_nulls:
        print(f"  Breakdown: {null_counts}")

    n_dupes = n_rows - df.dropDuplicates().count()
    print(f"Duplicate rows: {n_dupes}")

    df.show(5, truncate=False)


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    ratings = load_ratings(spark)
    movies = load_movies(spark)
    users = load_users(spark)

    sanity_check("ratings", ratings)
    sanity_check("movies", movies)
    sanity_check("users", users)

    print("\n--- Rating value distribution ---")
    ratings.groupBy("rating").count().orderBy("rating").show()

    print("\n--- Sparsity of user-item matrix ---")
    n_users = ratings.select("user_id").distinct().count()
    n_movies = ratings.select("movie_id").distinct().count()
    n_ratings = ratings.count()
    sparsity = 1 - (n_ratings / (n_users * n_movies))
    print(f"Users: {n_users:,} | Movies: {n_movies:,} | Ratings: {n_ratings:,}")
    print(f"Sparsity: {sparsity:.4%}")

    spark.stop()


if __name__ == "__main__":
    main()
