# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Download Video Data
# MAGIC
# MAGIC Downloads the public Kaggle CCTV anomaly dataset, stages it to a Unity Catalog Volume, and builds a Delta reference table with per-file duration metadata.
# MAGIC
# MAGIC **Cluster:** classic CPU compute (e.g., `Standard_E96ads_v5`). **Do not use serverless** — ffprobe is invoked via `subprocess` and serverless has restricted shell access.
# MAGIC
# MAGIC **Prerequisite — Kaggle API token.** This notebook downloads the dataset from Kaggle, which requires authentication. Two supported paths:
# MAGIC
# MAGIC 1. **Databricks secrets (recommended).** Create a secret scope `kaggle` with keys `username` and `key`. The notebook will read them automatically.
# MAGIC 2. **`~/.kaggle/kaggle.json` on the driver.** Upload your `kaggle.json` to the cluster's `/root/.kaggle/` directory before running.
# MAGIC
# MAGIC See https://www.kaggle.com/docs/api for token setup.

# COMMAND ----------

# DBTITLE 1,Install Required Packages
# MAGIC %pip install ffmpeg-python kaggle
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Define catalog, schema, volume
dbutils.widgets.text("CATALOG","main",label="CATALOG")
dbutils.widgets.text("SCHEMA", "default",label="SCHEMA")
dbutils.widgets.text("VOLUME", "transcribe-video",label="VOLUME")

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
VOLUME = dbutils.widgets.get("VOLUME")

# COMMAND ----------

# DBTITLE 1,Create catalog, schema, volume if missing
# Create catalog in Unity Catalog if it doesn't already exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS `{CATALOG}`")
# Create schema in Unity Catalog if it doesn't already exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
# Create volume in Unity Catalog if it doesn't already exist
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`.`{VOLUME}`")

# COMMAND ----------

# DBTITLE 1,Create directory for raw video files
video_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/raw_video/CCTV"
dbutils.fs.mkdirs(video_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Kaggle credentials
# MAGIC Reads from a Databricks secret scope `kaggle` if available, otherwise expects `~/.kaggle/kaggle.json` to exist on the driver. Fails fast with a clear error if neither is set.

# COMMAND ----------

# DBTITLE 1,Configure Kaggle credentials
import os
from pathlib import Path

try:
    os.environ["KAGGLE_USERNAME"] = dbutils.secrets.get(scope="kaggle", key="username")
    os.environ["KAGGLE_KEY"] = dbutils.secrets.get(scope="kaggle", key="key")
    print("Kaggle credentials loaded from secret scope 'kaggle'.")
except Exception:
    if not Path("/root/.kaggle/kaggle.json").exists() and not (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        raise RuntimeError(
            "No Kaggle credentials found. Either create a secret scope 'kaggle' with "
            "keys 'username' and 'key', or place kaggle.json at /root/.kaggle/. "
            "See https://www.kaggle.com/docs/api"
        )
    print("Kaggle credentials loaded from /root/.kaggle/kaggle.json.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download the dataset
# MAGIC Public Kaggle dataset: [Real-Time Anomaly Detection in CCTV Surveillance](https://www.kaggle.com/datasets/webadvisor/real-time-anomaly-detection-in-cctv-surveillance/data). Download + unzip typically ~20 minutes on standard cluster networking.

# COMMAND ----------

# DBTITLE 1,Download & extract via kaggle CLI
import subprocess

result = subprocess.run(
    [
        "kaggle", "datasets", "download",
        "-d", "webadvisor/real-time-anomaly-detection-in-cctv-surveillance",
        "-p", video_path,
        "--unzip",
    ],
    capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    raise RuntimeError(f"Kaggle download failed: {result.stderr}")

# COMMAND ----------

# DBTITLE 1,Build file reference DataFrame
# Build a Spark DataFrame referencing all video files
import pyspark.sql.functions as F
from functools import reduce

# List subdirectories (abuse categories) from the unzipped data
abuse_categories = [i.path for i in dbutils.fs.ls(f"{video_path}/data/")]

# Helper to convert file listings into Spark DataFrames
def create_abuse_df(folder_path):
    return spark.createDataFrame(dbutils.fs.ls(folder_path))\
        .withColumn("file_path", F.expr("substring(path, 6, length(path))"))

# Merge all folder listings into a single DataFrame
abuse_df_list = list(map(create_abuse_df, abuse_categories))
file_reference_df = reduce(lambda df1, df2: df1.union(df2), abuse_df_list)

# COMMAND ----------

# DBTITLE 1,Add video duration metadata
# UDF to extract video duration using ffprobe
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType

@udf(FloatType())
def get_length_udf(filename):
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    try:
        return float(result.stdout)
    except Exception:
        return None

# COMMAND ----------

# Apply duration extraction only on .mp4 files
file_reference_df = file_reference_df.filter(col("file_path").endswith(".mp4"))
file_reference_df = file_reference_df.withColumn("duration_seconds", get_length_udf("file_path"))

# COMMAND ----------

# DBTITLE 1,Save reference table to Unity Catalog
# Save the enriched file reference table
file_reference_df.write.mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.videos_file_reference")

# COMMAND ----------

# Display for validation
display(file_reference_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis

# COMMAND ----------

summary_df = file_reference_df.selectExpr(
    "percentile(duration_seconds, 0.0) as min",
    "percentile(duration_seconds, 0.25) as Q1",
    "percentile(duration_seconds, 0.5) as median",
    "percentile(duration_seconds, 0.75) as Q3",
    "percentile(duration_seconds, 1.0) as max"
)

display(summary_df)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

file_reference_pd_df = file_reference_df.select("duration_seconds").toPandas()

plt.figure(figsize=(10, 6))
sns.kdeplot(file_reference_pd_df["duration_seconds"], fill=True)
plt.title('Density Plot of Video Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Density')
plt.show()