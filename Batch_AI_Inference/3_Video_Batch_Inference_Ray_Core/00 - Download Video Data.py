# Databricks notebook source
# MAGIC %md
# MAGIC This notebook downloads CCTV video data, uploads it to a Unity Catalog Volume, extracts the video files, and creates a reference table with file metadata and durations. (DO not run on serverless)

# COMMAND ----------

# DBTITLE 1,⚙️ Install Required Packages
# MAGIC %pip install ffmpeg-python
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,🔧 Setup: Define Catalog, Schema, Volume
dbutils.widgets.text("CATALOG","main",label="CATALOG")
dbutils.widgets.text("SCHEMA", "default",label="SCHEMA")
dbutils.widgets.text("VOLUME", "transcribe-video",label="VOLUME")

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
VOLUME = dbutils.widgets.get("VOLUME")

# COMMAND ----------

# DBTITLE 1,📁 Create Catalog, Schema, Volume If It Doesn’t Exist
# Create catalog in Unity Catalog if it doesn't already exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS `{CATALOG}`")
# Create schema in Unity Catalog if it doesn't already exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
# Create volume in Unity Catalog if it doesn't already exist
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`.`{VOLUME}`")

# COMMAND ----------

# DBTITLE 1,📂 Create Directory Path for Raw Video Files
# Define and create a raw video folder path
video_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/raw_video/CCTV"
dbutils.fs.mkdirs(video_path)

# Register the path as an environment variable
import os
os.environ["video_path"] = video_path

# COMMAND ----------

# MAGIC %md
# MAGIC [Real Time Anomaly Detection in CCTV Surveillance](https://www.kaggle.com/datasets/webadvisor/real-time-anomaly-detection-in-cctv-surveillance/data) and copy zip folder to your volume. This may take about 20 minutes.

# COMMAND ----------

# DBTITLE 1,⬇️ Download & Copy ZIP File to UC Volume
# MAGIC %sh
# MAGIC curl -L -o /tmp/data.zip \
# MAGIC   https://www.kaggle.com/api/v1/datasets/download/webadvisor/real-time-anomaly-detection-in-cctv-surveillance
# MAGIC cp /tmp/data.zip $video_path
# MAGIC rm /tmp/data.zip

# COMMAND ----------

# DBTITLE 1,🗜️ Extract Video Files From ZIP
# Unzip the downloaded file inside the volume path
import zipfile

zip_file_path = f"{video_path}/data.zip"
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(video_path)

# COMMAND ----------

# DBTITLE 1,📋 Create File Reference DataFrame
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

# DBTITLE 1,⏱️ Add Video Duration Metadata
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

# DBTITLE 1,💾 Save to Table in Unity Catalog
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

# Convert to Pandas DataFrame for plotting
file_reference_pd_df = file_reference_df.select("duration_seconds").toPandas()

# Generate the density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(file_reference_pd_df["duration_seconds"], shade=True)
plt.title('Density Plot of Video Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Density')
plt.show()