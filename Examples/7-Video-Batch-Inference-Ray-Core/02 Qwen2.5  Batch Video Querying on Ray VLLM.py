# Databricks notebook source
# MAGIC %md
# MAGIC # Video querying batch inference reference solution
# MAGIC
# MAGIC Tested on:
# MAGIC ```
# MAGIC * MLR 16.1LTS GPU Runtime
# MAGIC * Collection of `.mp4` files from the Real Time Anomaly Detection in CCTV Surveillance dataset from Kaggle
# MAGIC * GPU Cluster `Standard_NC48ads_A100_v3 [A100]`
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC **IMPORTANT:**
# MAGIC
# MAGIC Set these `spark configs` on the cluster before starting it:
# MAGIC
# MAGIC * `spark.databricks.pyspark.dataFrameChunk.enabled true`
# MAGIC * `spark.task.resource.gpu.amount 0`

# COMMAND ----------

# DBTITLE 1,Install and Update Required Python Libraries
# MAGIC %pip install -qU databricks-sdk numba==0.60.0 pydub ray vllm==0.8.2 ffmpeg-python git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Set Up Database Catalog, Schema, and Volume Variables
dbutils.widgets.text("CATALOG","main",label="CATALOG")
dbutils.widgets.text("SCHEMA", "default",label="SCHEMA")
dbutils.widgets.text("VOLUME", "transcribe-video",label="VOLUME")

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
VOLUME = dbutils.widgets.get("VOLUME")

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import os
import time
import ssl
import numpy as np
import pandas as pd
import torch
import ray
import pydub
import librosa
import ffmpeg
import pyspark.sql.functions as F
import pyspark.sql.types as T

from mlflow.utils.databricks_utils import get_databricks_env_vars
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
from transformers import pipeline
from util import stage_registered_model, flatten_folder, run_on_every_node
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.video import VideoAsset

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and start Ray cluster
# MAGIC Some best practices for scaling up Ray clusters [here](https://docs.databricks.com/en/machine-learning/ray/scale-ray.html#scale-ray-clusters-on-databricks) :
# MAGIC * `num_cpus_*` always leave 1 CPU core for spark so value should be <= max cores per worker - 1

# COMMAND ----------

# num_cpu_cores_per_worker = 48-1 # total cpu's present in each worker node (g5.12xlarge)
# num_cpus_head_node = 	48-1 # total cpu's present in the driver node (g5.12xlarge)
# num_gpu_per_worker = 4
# num_gpus_head_node = 4

# Set databricks credentials as env vars
mlflow_dbrx_creds = get_databricks_env_vars("databricks")
os.environ["DATABRICKS_HOST"] = mlflow_dbrx_creds['DATABRICKS_HOST']
os.environ["DATABRICKS_TOKEN"] = mlflow_dbrx_creds['DATABRICKS_TOKEN']

# ray_conf = setup_ray_cluster(
#   min_worker_nodes=1,
#   max_worker_nodes=1,
#   num_cpus_head_node= num_cpus_head_node,
#   num_gpus_head_node= num_gpus_head_node,
#   num_cpus_per_node=num_cpu_cores_per_worker,
#   num_gpus_per_node=num_gpu_per_worker
#   )

# COMMAND ----------

# MODEL_NAME = "qwen2_5_vl-32b"
MODEL_NAME = "qwen2_5_vl-32b-awq"
MODEL_ALIAS = "Production"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-requisite: download the models from MLflow registry into every node once to avoid multiple download conflicts

# COMMAND ----------

# ray.init()
context = ray.init(
  include_dashboard=True,
  dashboard_host="0.0.0.0",
  dashboard_port=9999
  )

# COMMAND ----------

import os

def get_dashboard_url(spark,dbutils):  
  base_url='https://' + spark.conf.get("spark.databricks.workspaceUrl")
  workspace_id=spark.conf.get("spark.databricks.clusterUsageTags.orgId")
  cluster_id=spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
  dashboard_port= '9999'

  pathname_prefix='/driver-proxy/o/' + workspace_id + '/' + cluster_id + '/' + dashboard_port+"/" 

  apitoken = dbutils.notebook().entry_point.getDbutils().notebook().getContext().apiToken().get()
  dashboard_url=base_url + pathname_prefix  # ?token=' + apitoken[0:10] + " " + apitoken[10:]

  return dashboard_url

# COMMAND ----------

# MAGIC %md
# MAGIC Link to Ray Dashboard to check on the status of the clusters:

# COMMAND ----------

get_dashboard_url(spark,dbutils) 

# COMMAND ----------

# DBTITLE 1,Create remote function which will get model from mlflow directly into local worker disk
# Function to download model on each Ray node
@ray.remote(num_cpus=1)
def download_model(catalog, schema, model_name, alias="Production", local_base_path="/local_disk0/models/", overwrite=False):
    model_weights_path = stage_registered_model(
        catalog=catalog,
        schema=schema,
        model_name=model_name,
        alias=alias,
        local_base_path=local_base_path,
        overwrite=overwrite
    )
    flatten_folder(model_weights_path)
    return model_weights_path

# COMMAND ----------

# import mlflow

# # Set MLflow registry URI
# mlflow.set_registry_uri("databricks-uc")

# # Download model across all Ray nodes
# model_weights_path = run_on_every_node(download_model, **{
#     "catalog": CATALOG,
#     "schema": SCHEMA,
#     "model_name": MODEL_NAME,
#     "alias": MODEL_ALIAS
# })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write ray-friendly `__call__`-able classes for batch processing
# MAGIC
# MAGIC For VLLM, one parameter to configure would be:
# MAGIC
# MAGIC * `gpu_memory_utilization`: will define how many model instances will be created in a single GPU. This would depend on model's size and GPU VRAM in theory.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ingest and transcribe pipelines

# COMMAND ----------

class ConvertToPrompt:
    """
    Class to process video files and prepare them for the Qwen model.
    """
    
    def __init__(self, query, num_frames=16):
        self.num_frames = num_frames
        self.query = query

    def transform(self, video_filename):
        """
        Extracts frames from a video file and returns as NumPy array.
        """

        probe = ffmpeg.probe(video_filename)
        streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
        if streams:
            num_frames = int(float(streams[0]['r_frame_rate'].split('/')[0]) / float(streams[0]['r_frame_rate'].split('/')[1]))
        else:
            num_frames = self.num_frames
        
        video_data = VideoAsset(name=video_filename, num_frames=num_frames) 
        return video_data.np_ndarrays

    def __call__(self, row):
        """
        Converts video files into model-ready prompts.
        """
        row["frames"] = self.transform(row["file_path"])
        # row["frames"] = f"file:///{row['file_path']}"
        row["prompt"] = self.create_prompt(self.query) 
        return row

    def create_prompt(self, question):
        """
        Constructs a model prompt including video data.
        """
        return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n")

class QwenVideoProcessing:
    """
    Class which will handle transcription of audio files (in batch fashion using VLLM)
    """
    def __init__(self, catalog:str, schema:str, model_name:str, model_kwargs:dict, model_alias:str = "Production"):
        self.unverified_context = ssl._create_unverified_context()
        print("Loading model from UC registry...")
        model_weights_path = stage_registered_model(
                            catalog = catalog, 
                            schema = schema, 
                            model_name = model_name, 
                            alias = model_alias,
                            local_base_path = "/local_disk0/models/",
                            overwrite = False)
        
        flatten_folder(model_weights_path)
        model_weights_path = str(model_weights_path)  #convert from Posit to string required by TF
        self.QWEN_MODEL_PATH = model_weights_path

        # Create VLLM pipeline object
        self.video_inference_pipeline = LLM(
                            model=model_weights_path,
                            max_model_len=model_kwargs["max_model_len"],
                            quantization=model_kwargs["quantization"],
                            max_num_seqs=model_kwargs["max_num_seqs"],
                            mm_processor_kwargs={
                                "min_pixels": model_kwargs["min_pixels"],
                                "max_pixels": model_kwargs["max_pixels"],
                                "fps": model_kwargs["fps"],
                                                    },
                            disable_mm_preprocessor_cache=True,
                            tensor_parallel_size=model_kwargs["tensor_parallel_size"],
                            kv_cache_dtype="fp8",
                            enforce_eager=True,
                            limit_mm_per_prompt={"image": 1, "video": 1},
                            gpu_memory_utilization = .95)
        print("Model loaded...")

    def transform(self, row):
        """
        Converts frames and questions into Qwen-compatible input format.
        """
        prompts = [{
            "prompt": row["prompt"],
            "multi_modal_data": {"video": row["frames"]}
        }]
        return prompts


    def __call__(self, row) -> str:
        """
        Call method applying all pipeline steps (in batch)
        """

        # Create a sampling params inference object
        sampling_params = SamplingParams(
        temperature=0.1,
        top_p=1,
        repetition_penalty=1.05,
        max_tokens=2500,
        stop_token_ids=[],
        )

        prompts = self.transform(row)

        outputs = self.video_inference_pipeline.generate(prompts, sampling_params)

        row["generated_text"] = [output.outputs[0].text for output in outputs][0]

        return row

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare batch job
# MAGIC
# MAGIC 1. Point to input Delta table containing file paths
# MAGIC 2. Select UC model names and `@alias` _(or version)_
# MAGIC 3. Write ray inference code
# MAGIC 4. Apply batch job and write/materialize outputs to Delta table

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Read input Delta table containing video files' path

# COMMAND ----------

from pyspark.sql.functions import split, element_at
import pyspark.sql.functions as F

TABLENAME = f"{CATALOG}.{SCHEMA}.videos_file_reference" 
video_files_reference_df = spark.table(TABLENAME)

display(video_files_reference_df)

# COMMAND ----------

total_size = video_files_reference_df.agg(F.sum("size").alias("total_size")).collect()[0]["total_size"]
print(f"{total_size/1000000000} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Write ray batch pipeline
# MAGIC
# MAGIC using [`map`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map.html#ray.data.Dataset.map) and [`map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html)
# MAGIC
# MAGIC Some relevant parameters to define are:
# MAGIC * `num_cpus`: for CPU intensive workloads (i.e. read the audio files) - defines how many ray-cores to use for individual tasks (default is `1 Ray-Core/CPU == 1 CPU Physical Core`). It can be defined as a fraction to oversubscribe a single physical core with multiple tasks
# MAGIC
# MAGIC * `num_gpus`: for GPU intensive workloads - defines how many (fractionnal) GPU(s) a single task/batch will use
# MAGIC
# MAGIC * `concurrency`: how many parallel tasks to run in parallel `Tuple(min,max)`

# COMMAND ----------

num_frames = 16

# query = """
# Analyze the surveillance video and determine if any shoplifting activity occurs. Focus on suspicious behaviors such as: hiding merchandise in clothing or bags, intentionally bypassing the checkout, switching price tags, or attempting to distract staff.  If shoplifting is detected, describe what happened, when it occurred (with timestamps), and which person was involved. Be specific and detailed in your explanation.
# """

query = """
You are analyzing CCTV surveillance footage. Your task is to observe the video and describe what is happening in detail. Based on your observations, identify whether the event shown is normal or an anomaly. If it is an anomaly, classify it into one of the following categories: fighting, arson, burglary, assault, robbery, stealing, roadaccidents, shoplifting, shooting, vandalism, arrest, explosion, or abuse. For each significant event, provide the timestamp in the format [mm:ss] when it occurs. Be as specific as possible in your description, and mention the most relevant actions, objects, and people. Start your response with: 'Description:' followed by 'Classification:' and 'Timestamps:'. 
"""

model_kwargs = {
"max_model_len":32000,
"max_num_seqs": 5,
"min_pixels" : 28 * 28,
"max_pixels" : 1280 * 28 * 28,
"fps" : 1,
# "tensor_parallel_size": 2,
"tensor_parallel_size": 1, 
"quantization": "awq"
# "quantization": None
}

# ## adjust these parameters depending on your cluster/model size

# num_gpus=2 #####1
num_gpus=1 #####1
# concurrency = (2,12)
concurrency = (4,12)

# Convert to Ray dataset
ds = ray.data.from_spark(video_files_reference_df)

ds = ds.repartition(200)\
        .map(
            ConvertToPrompt, 
            fn_constructor_kwargs={
                  "num_frames": num_frames,
                  "query": query
                  },
            concurrency=(10,24), # Can go up to total sum of cores
            num_cpus=1,
        )\
        .map(
              QwenVideoProcessing,
              fn_constructor_kwargs={
                  "catalog": CATALOG,
                  "schema": SCHEMA,
                  "model_name": MODEL_NAME,
                  "model_kwargs": model_kwargs,
                  "model_alias": MODEL_ALIAS
                  },
              concurrency=concurrency, 
              num_gpus=num_gpus, # Individual batches will utilize  up to 60% of GPU's memory <==> 2 batches in parallel per GPU
          )\
        .drop_columns(["frames"])

# COMMAND ----------

# Temporary directory for ray-uc-volumes-fuse (to write to Delta natively)
VOLUME = "temp"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

tmp_dir_fs = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/tempDoc"
dbutils.fs.mkdirs(tmp_dir_fs)
os.environ["RAY_UC_VOLUMES_FUSE_TEMP_DIR"] = tmp_dir_fs

# COMMAND ----------

table_name = "responses"

ds.write_databricks_table(
  f"{CATALOG}.{SCHEMA}.{table_name}",
  mode = "overwrite", #append/merge
  mergeSchema = True
)

# COMMAND ----------

# spark.table(f"{CATALOG}.{SCHEMA}.{table_name}").display()