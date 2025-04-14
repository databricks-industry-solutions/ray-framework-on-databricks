# Databricks notebook source
# MAGIC %md
# MAGIC # Download sample `.mp4` files and upload to UC Volume

# COMMAND ----------

# DBTITLE 1,Set Catalog, Schema, and Volume Widgets
dbutils.widgets.text("CATALOG","main",label="CATALOG")
dbutils.widgets.text("SCHEMA", "default",label="SCHEMA")
dbutils.widgets.text("VOLUME", "transcribe-video",label="VOLUME")

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
VOLUME = dbutils.widgets.get("VOLUME")

# COMMAND ----------

# DBTITLE 1,Creating Spark SQL Volume if Not Exists
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`.`{VOLUME}`")

# COMMAND ----------

# DBTITLE 1,Create Raw Video Directory Path
video_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/raw_video/CCTV"
dbutils.fs.mkdirs(video_path)

# COMMAND ----------

# MAGIC %md
# MAGIC [Real Time Anomaly Detection in CCTV Surveillance](https://www.kaggle.com/datasets/webadvisor/real-time-anomaly-detection-in-cctv-surveillance/data) and copy zip folder to your volume

# COMMAND ----------

# DBTITLE 1,Download and Copy CCTV Anomaly Detection Dataset
# MAGIC %sh
# MAGIC curl -L -o /tmp/data.zip \
# MAGIC   https://www.kaggle.com/api/v1/datasets/download/webadvisor/real-time-anomaly-detection-in-cctv-surveillance
# MAGIC cp /tmp/data.zip /Volumes/samantha_wise/coop_video/transcribe-video/raw_video/CCTV
# MAGIC rm /tmp/data.zip

# COMMAND ----------

# DBTITLE 1,Unzip (may take a while so feel free to cancel manually after 10mins)
import zipfile


zip_file_path = f"{video_path}/data.zip"
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(video_path)

# COMMAND ----------

# DBTITLE 1,Create DataFrame for CCTV Video File Paths
import pyspark.sql.functions as F
from functools import reduce

abuse_categories = [i.path for i in dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/raw_video/CCTV/data/")]

def create_abuse_df(i):
    return spark.createDataFrame(dbutils.fs.ls(i))\
        .withColumn("file_path", F.expr("substring(path, 6, length(path))"))

abuse_df_list = list(map(create_abuse_df, abuse_categories))

file_reference_df = reduce(lambda df1, df2: df1.union(df2), abuse_df_list)

# COMMAND ----------

from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType

@udf(FloatType())
def get_length_udf(filename):
    import subprocess
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

file_reference_df = spark.table(f"{CATALOG}.{SCHEMA}.videos_file_reference")
file_reference_df = file_reference_df.filter(col("file_path").endswith(".mp4"))
file_reference_df = file_reference_df.withColumn("duration_seconds", get_length_udf("file_path"))

display(file_reference_df)

file_reference_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.videos_file_reference")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis

# COMMAND ----------

# DBTITLE 1,Calculate Percentiles for Duration in Seconds
summary_df = file_reference_df.selectExpr(
    "percentile(duration_seconds, 0.0) as min",
    "percentile(duration_seconds, 0.25) as Q1",
    "percentile(duration_seconds, 0.5) as median",
    "percentile(duration_seconds, 0.75) as Q3",
    "percentile(duration_seconds, 1.0) as max"
)

display(summary_df)

# COMMAND ----------

# DBTITLE 1,Generate Density Plot for Video Durations
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