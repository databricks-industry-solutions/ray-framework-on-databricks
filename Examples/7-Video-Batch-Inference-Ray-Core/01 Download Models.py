# Databricks notebook source
# MAGIC %md
# MAGIC # Fetch models from Hugging face hub, tag and store to mlflow UC registry

# COMMAND ----------

# DBTITLE 1,Install Required Python Packages and Restart Python
PIP_REQUIREMENTS = (
    "openai "
    "intel_extension_for_pytorch "
    "autoawq>=0.1.8 "
    "vllm==0.8.2 "
    # "vllm==0.7.3 "
    "httpx==0.27.2 "
    "torch>=2.6.0 torchvision>=0.21.0 torchaudio>=2.6.0 xformers==0.0.29.post2 accelerate " 
    "git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775 "
    "mlflow==2.19.0 "
    "git+https://github.com/stikkireddy/mlflow-extensions.git@v0.17.0 "
    "qwen-vl-utils"
)
%pip install {PIP_REQUIREMENTS}

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Set Catalog, Schema, and Volume Widgets
dbutils.widgets.text("CATALOG","main",label="CATALOG")
dbutils.widgets.text("SCHEMA", "default",label="SCHEMA")
dbutils.widgets.text("VOLUME", "transcribe-video",label="VOLUME")

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
VOLUME = dbutils.widgets.get("VOLUME")

# COMMAND ----------

# DBTITLE 1,Connect to Databricks UC with Mlflow Client
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client  = MlflowClient()

# COMMAND ----------

# DBTITLE 1,Configure Multiprocessing Method for VLLM
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Qwen2.5 VL Model

# COMMAND ----------

# DBTITLE 1,Build pipeline
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, pipeline
from qwen_vl_utils import process_vision_info


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32 #.float16 for model size reduction

# model_id = "Qwen/Qwen2.5-VL-32B-Instruct" 
model_id = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ" 

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
model_kwargs = {
"max_model_len":10000,
"max_num_seqs":5,
"min_pixels" : 28 * 28,
"max_pixels" : 1280 * 28 * 28,
"fps" : 1
}

qwen_pipe = pipeline(
    "object-detection",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.image_processor,
    torch_dtype=torch_dtype,
    # device=device,
    model_kwargs=model_kwargs
)

# COMMAND ----------

# DBTITLE 1,Snap model to UC registry [Recommended]
import mlflow

input_video_example_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/download.mp4"

# QWEN_MODEL_NAME = "qwen2_5_vl-32b"
QWEN_MODEL_NAME = "qwen2_5_vl-32b-awq"

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec
import numpy as np

# Define the input schema
input_schema = Schema([
    ColSpec("string", "text_input"),  # User query
    ColSpec("string", "video_path"),  # Path to the input video file (URI)
    ColSpec("integer", "max_pixels"),  # Max pixel resolution for processing
    ColSpec("float", "fps")  # Frames per second for processing
])

# Define the output schema (generated text response)
output_schema = Schema([
    ColSpec("string", "generated_text")
])

# Create the model signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log the pipeline
# with mlflow.start_run(run_name="qwen32b-video-log-pipeline"):
with mlflow.start_run(run_name="qwen32b-awq-video-log-pipeline"):
    model_info = mlflow.transformers.log_model(
        transformers_model=qwen_pipe,
        # artifact_path="qwen32b_pipeline",
        artifact_path="qwen32b_awq_pipeline",
        # input_example=input_video_example_path,
        signature=signature,
        registered_model_name=f"{CATALOG}.{SCHEMA}.{QWEN_MODEL_NAME}",
    )

# COMMAND ----------

# DBTITLE 1,Tag model as @Production
client.set_registered_model_alias(
  name=f"{CATALOG}.{SCHEMA}.{QWEN_MODEL_NAME}",
  version=model_info.registered_model_version,
  alias="production",
)