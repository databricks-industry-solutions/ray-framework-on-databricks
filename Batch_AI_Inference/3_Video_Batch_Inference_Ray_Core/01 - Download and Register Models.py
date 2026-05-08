# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Download and Register Qwen2.5-VL-32B
# MAGIC
# MAGIC Downloads `Qwen/Qwen2.5-VL-32B-Instruct` from Hugging Face and registers it to Unity Catalog via MLflow with a `production` alias. The downstream `02` notebook stages weights from this registered model into vLLM at inference time.
# MAGIC
# MAGIC **Cluster:** classic CPU compute. The model weights are ~60 GB and are downloaded to `/local_disk0/`. No GPU is required at registration time — vLLM is invoked only in `02`.
# MAGIC - **Driver:** 672 GB Memory, 96 Cores (e.g., `Standard_E96ads_v5`)
# MAGIC - **Runtime:** 16.1.x-scala2.12
# MAGIC - **Unity Catalog enabled**

# COMMAND ----------

# DBTITLE 1,Install required packages
# Minimal deps for VLM registration: transformers + torch + accelerate + mlflow.
# vLLM is NOT installed here — it's a GPU-only dependency invoked from notebook 02.
PIP_REQUIREMENTS = (
    "torch==2.5.1 torchvision==0.20.1 accelerate "
    "git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775 "
    "mlflow==2.19.0"
)
%pip install {PIP_REQUIREMENTS}
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Set up MLflow and registry URI
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# DBTITLE 1,Define catalog, schema, model name
dbutils.widgets.text("CATALOG", "main", label="CATALOG")
dbutils.widgets.text("SCHEMA", "default", label="SCHEMA")
dbutils.widgets.text("MODEL_NAME", "qwen2_5_vl-32b", label="MODEL_NAME")

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
MODEL_NAME = dbutils.widgets.get("MODEL_NAME")

# COMMAND ----------

# DBTITLE 1,Set torch multiprocessing mode
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Qwen2.5 VL 32B [Model](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)

# COMMAND ----------

# DBTITLE 1,Load Qwen model and processor
# device_map="auto" places the model on whatever GPUs are available (or CPU on this
# CPU-only registration cluster). Do NOT call .to(device) afterwards — it conflicts
# with device_map and triggers PyTorch warnings / accelerate errors.
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_id = "Qwen/Qwen2.5-VL-32B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

processor = AutoProcessor.from_pretrained(model_id)

# COMMAND ----------

# DBTITLE 1,Register model to Unity Catalog via MLflow
# We register the model + processor as a transformers artefact so the weights are
# governed and aliased. Notebook 02 stages these weights and instantiates vLLM
# directly — the registration is a governance/versioning step, not the inference path.
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_schema = Schema([
    ColSpec("string", "text_input"),
    ColSpec("string", "video_path"),
    ColSpec("integer", "max_pixels"),
    ColSpec("float", "fps"),
])
output_schema = Schema([
    ColSpec("string", "generated_text"),
])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# COMMAND ----------

# Save model + processor to a local path so we can register via the path-form of
# transformers.log_model. The dict-form triggers MLflow to internally call
# pipeline(task="image-text-to-text", **dict), which fails for multi-modal Qwen-VL
# because the unified processor isn't accepted in the dict. The path-form skips
# in-memory pipeline validation entirely (see MLflow source: validate_serving_input
# = not isinstance(transformers_model, str)).
import shutil
local_save_path = "/local_disk0/qwen_vlm_save"
if os.path.exists(local_save_path):
    shutil.rmtree(local_save_path)

model.save_pretrained(local_save_path)
processor.save_pretrained(local_save_path)
print(f"Saved model + processor to {local_save_path}")

# COMMAND ----------

with mlflow.start_run(run_name=f"{MODEL_NAME}-register"):
    model_info = mlflow.transformers.log_model(
        transformers_model=local_save_path,
        artifact_path="qwen_vlm",
        task="image-text-to-text",
        signature=signature,
        registered_model_name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
    )

# COMMAND ----------

# DBTITLE 1,Set @production alias on the registered model
# Lowercase 'production' — MLflow alias lookups are case-sensitive; notebook 02 reads
# the same casing.
client.set_registered_model_alias(
    name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
    version=model_info.registered_model_version,
    alias="production",
)
print(f"Registered {CATALOG}.{SCHEMA}.{MODEL_NAME} version {model_info.registered_model_version} with alias 'production'.")