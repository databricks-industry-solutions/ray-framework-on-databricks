# Databricks notebook source
# MAGIC %md
# MAGIC This notebook downloads a VLM model from Hugging Face, builds a processing pipeline, and registers it to Unity Catalog via MLflow for use in video-based inference tasks.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster Configurations: Summary
# MAGIC - **Driver**: 672 GB Memory, 96 Cores
# MAGIC - **Runtime**: 16.1.x-scala2.12
# MAGIC - **Unity Catalog**
# MAGIC - **Instance Type**: Standard_E96ads_v5
# MAGIC - **Cost**: 33 DBU/h

# COMMAND ----------

# DBTITLE 1,⚙️ Install Required Packages
# Install necessary packages including specific model version and MLflow extensions
PIP_REQUIREMENTS = (
    "openai vllm==0.7.3 "
    "httpx==0.27.2 "
    "torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.28.post3 accelerate "
    "git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775 "
    "mlflow==2.19.0 "
    "git+https://github.com/stikkireddy/mlflow-extensions.git@v0.17.0 "
    "qwen-vl-utils"
)
%pip install {PIP_REQUIREMENTS}
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,📚 Set Up MLflow and Registry URI
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client  = MlflowClient()

# COMMAND ----------

# DBTITLE 1,🧱 Define Catalog, Schema, and Volume
dbutils.widgets.text("CATALOG","main",label="CATALOG")
dbutils.widgets.text("SCHEMA", "default",label="SCHEMA")
dbutils.widgets.text("VOLUME", "transcribe-video",label="VOLUME")

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
VOLUME = dbutils.widgets.get("VOLUME")

# COMMAND ----------

# DBTITLE 1,🛠️ Set Torch Multiprocessing Mode (Optional)
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Qwen2.5 VL 32B [Model](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)

# COMMAND ----------

# DBTITLE 1,🧠 Load Qwen Model and Processor
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, pipeline
from qwen_vl_utils import process_vision_info

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32  # Use .float16 if memory becomes a concern

model_id = "Qwen/Qwen2.5-VL-32B-Instruct"

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Model-specific configuration
model_kwargs = {
    "max_model_len": 10000,
    "max_num_seqs": 5,
    "min_pixels": 28 * 28,
    "max_pixels": 1280 * 28 * 28,
    "fps": 1
}

# Build Hugging Face pipeline
qwen_pipe = pipeline(
    "object-detection",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.image_processor,
    torch_dtype=torch_dtype,
    model_kwargs=model_kwargs
)

# COMMAND ----------

# DBTITLE 1,📝 Register Model to Unity Catalog via MLflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Define model input schema
input_schema = Schema([
    ColSpec("string", "text_input"),
    ColSpec("string", "video_path"),
    ColSpec("integer", "max_pixels"),
    ColSpec("float", "fps")
])

# Define model output schema
output_schema = Schema([
    ColSpec("string", "generated_text")
])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# COMMAND ----------

QWEN_MODEL_NAME = "qwen2_5_vl-32b"

# Log the model pipeline to MLflow UC
with mlflow.start_run(run_name="qwen32b-video-log-pipeline"):
    model_info = mlflow.transformers.log_model(
        transformers_model=qwen_pipe,
        artifact_path="qwen32b_pipeline",
        signature=signature,
        registered_model_name=f"{CATALOG}.{SCHEMA}.{QWEN_MODEL_NAME}",
    )

# COMMAND ----------

# DBTITLE 1,🔁 Set Model Alias in Unity Catalog Registry
# Set alias for easier reference in downstream workloads
client.set_registered_model_alias(
    name=f"{CATALOG}.{SCHEMA}.{QWEN_MODEL_NAME}",
    version=model_info.registered_model_version,
    alias="production",
)