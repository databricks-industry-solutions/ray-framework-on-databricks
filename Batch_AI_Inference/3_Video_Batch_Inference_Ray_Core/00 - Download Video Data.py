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

# DBTITLE 1,Install ffmpeg system binary
# MAGIC %sh
# MAGIC apt-get update -qq && apt-get install -y -qq ffmpeg
# MAGIC ffprobe -version | head -1

# COMMAND ----------

# DBTITLE 1,Install required Python packages
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
# MAGIC Public Kaggle dataset: [Real-Time Anomaly Detection in CCTV Surveillance](https://www.kaggle.com/datasets/webadvisor/real-time-anomaly-detection-in-cctv-surveillance/data).
# MAGIC
# MAGIC Set `CATEGORY_FILTER` to a specific anomaly category (e.g. `shoplifting`, `burglary`, `arson`)
# MAGIC to download only that subfolder via per-file requests. Set to `ALL` to download the full
# MAGIC ~95 GB dataset. Single-category downloads take ~10 min and ~2 GB; full takes ~30 min and ~95 GB.

# COMMAND ----------

dbutils.widgets.text("CATEGORY_FILTER", "shoplifting", label="CATEGORY_FILTER")
CATEGORY_FILTER = dbutils.widgets.get("CATEGORY_FILTER")
print(f"Downloading category: {CATEGORY_FILTER}")

# COMMAND ----------

# DBTITLE 1,Download & extract via kaggle CLI
import subprocess

DATASET = "webadvisor/real-time-anomaly-detection-in-cctv-surveillance"

if CATEGORY_FILTER.upper() == "ALL":
    # Full dataset download (~95 GB).
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", video_path, "--unzip"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")
else:
    # Single-category download. The Kaggle list_files API has pagination quirks for
    # large datasets (only returns the first page), so we enumerate by filename
    # pattern instead. Each file is downloaded directly via `kaggle datasets download
    # -f`. 404s for non-existent file numbers are silently skipped.
    #
    # The dataset's known categories: shoplifting (001..055 with gaps), burglary,
    # abuse, arson, assault, robbery, stealing, vandalism, etc. Each category folder
    # uses the same `<Category>NNN_x264.mp4` naming pattern.
    capitalised = CATEGORY_FILTER[0].upper() + CATEGORY_FILTER[1:]
    candidate_paths = [
        f"data/{CATEGORY_FILTER}/{capitalised}{i:03d}_x264.mp4"
        for i in range(1, 200)  # generous upper bound; non-existent files 404 quietly
    ]
    print(f"Attempting download of up to {len(candidate_paths)} files from "
          f"data/{CATEGORY_FILTER}/...")

    success_count = 0
    fail_count = 0
    for i, fname in enumerate(candidate_paths, 1):
        proc = subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET,
             "-f", fname, "-p", video_path, "--unzip", "--force"],
            capture_output=True, text=True,
        )
        # Kaggle CLI returns non-zero for 404, but we don't know in advance which
        # numbers exist. Treat "Not Found" / "404" stderr as a silent skip.
        stderr_lower = (proc.stderr or "").lower()
        if proc.returncode == 0:
            success_count += 1
            if success_count == 1 or success_count % 10 == 0:
                print(f"  [{success_count} downloaded] last: {fname}")
        elif "404" in stderr_lower or "not found" in stderr_lower:
            # File doesn't exist at that number — keep going.
            fail_count += 1
        else:
            fail_count += 1
            print(f"  WARN: {fname}: {proc.stderr.strip()[:200]}")

        # Early termination: if we've had 30 consecutive failures after at least
        # one success, assume we've passed the end of the numbering.
        if success_count > 0 and i - success_count > 30:
            print(f"  Stopping early after {i} attempts ({success_count} downloaded)")
            break

    print(f"Downloaded {success_count} files to {video_path}")
    if success_count == 0:
        raise RuntimeError(
            f"No files downloaded for category '{CATEGORY_FILTER}'. Check the "
            f"category spelling. Common categories: shoplifting, burglary, abuse, "
            f"arson, assault, robbery, stealing, vandalism. "
            f"Set CATEGORY_FILTER=ALL for the full dataset."
        )

    # Single-file downloads from Kaggle CLI come back as .mp4.zip even with --unzip
    # AND are placed flat at the top level instead of in data/<category>/. Reorganize
    # so the layout matches what notebook 02 expects: data/<category>/*.mp4
    import os, zipfile, shutil
    target_dir = f"{video_path}/data/{CATEGORY_FILTER}"
    os.makedirs(target_dir, exist_ok=True)

    # Step 1: unzip every .zip in video_path
    for fname in os.listdir(video_path):
        if fname.endswith(".zip"):
            zpath = os.path.join(video_path, fname)
            try:
                with zipfile.ZipFile(zpath, "r") as z:
                    z.extractall(video_path)
                os.remove(zpath)
            except zipfile.BadZipFile:
                print(f"  WARN: bad zip {fname}")

    # Step 2: move all .mp4 files at top level into data/<category>/
    relocated = 0
    for fname in os.listdir(video_path):
        if fname.endswith(".mp4"):
            shutil.move(
                os.path.join(video_path, fname),
                os.path.join(target_dir, fname),
            )
            relocated += 1
    print(f"Reorganized {relocated} .mp4 files into {target_dir}")

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