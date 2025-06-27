# Databricks notebook source
# MAGIC %md
# MAGIC # Ray On Spark for multimodal image batch inference
# MAGIC In this example we'll cover how to perform distributed batch inference on databricks clusters for Small/Medium Multimodal Language Models (i.e. [OpenGVL/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B)), while also leveraging MLflow for model versionning and logging.
# MAGIC
# MAGIC Framework used: `@pandas_udf` + `map` & `map_batches`
# MAGIC
# MAGIC
# MAGIC **PS: Framework only supported for batch inference (for streaming please referr to another stream example _wip_)**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites:
# MAGIC
# MAGIC ### Cluster Configurations: Summary
# MAGIC - **Driver**: 192 GB Memory, 48 Cores
# MAGIC - **Workers**: 1
# MAGIC - **Machine Learniner Runtime**: 16.4LTS
# MAGIC - **Unity Catalog**
# MAGIC - **Instance Type**: g5.12xlarge (A10G - 24GB GPU RAM) _Feel Free to use a different or smaller instance_
# MAGIC
# MAGIC Set these `spark configs` on the cluster before starting it:
# MAGIC
# MAGIC * `spark.databricks.pyspark.dataFrameChunk.enabled true`
# MAGIC * `spark.task.resource.gpu.amount 0`
# MAGIC
# MAGIC Environment tested:
# MAGIC ```
# MAGIC mlflow=3.1.0
# MAGIC ray==2.47.1
# MAGIC timm==1.0.15
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧪 Step 1: Install Required Packages

# COMMAND ----------

# MAGIC %pip install -qU mlflow-skinny ray[default] ray[data] timm
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧱 Step 2: Read image dataset & Configure Environment
# MAGIC Point to CATALOG/SCHEMA/VOLUMES where images are.

# COMMAND ----------

# DBTITLE 1,Define catalog/schema/volumes
CATALOG = "amine_elhelou" # Change This/Point to an existing catalog
SCHEMA = "ray_gtm_examples" # Point to an existing schema
VOLUME = "fashion-images"

# COMMAND ----------

# DBTITLE 1,Set images folder/volumes paths
volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/data/"
image_paths = ["Apparel/Boys/Images/images_with_product_ids/", "Apparel/Girls/Images/images_with_product_ids/",
               "Footwear/Men/Images/images_with_product_ids/", "Footwear/Women/Images/images_with_product_ids/"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read images as bytestream into a Delta Table
# MAGIC (Can also keep file URLs only and drop/not use bytestreams)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, BinaryType, LongType, TimestampType


# Generic schema when reading binary files using spark
raw_binary_images_schema = StructType([ 
  StructField('path', StringType()),  
  StructField('modificationTime', TimestampType()),  
  StructField('length', LongType()),  
  StructField('content', BinaryType()),  
  ])

# Initialize empty dataframe
data_bin_df = spark.createDataFrame([], schema=raw_binary_images_schema)

# Loop and append/unionAll
for path in image_paths:
  data_bin_df = data_bin_df.unionAll(spark.read.format("binaryFile").load(f"{volume_path}/{path}")) #.drop("content")

# Materialze
data_bin_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_binary_images")

# NOTE: If images are streamed/dumped to Cloud Storage, this can be turned into a streaming ingestion job using autoloader

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.bronze_binary_images LIMIT 10"))

# COMMAND ----------

display(spark.sql(f"SELECT COUNT(*) FROM {CATALOG}.{SCHEMA}.bronze_binary_images"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔐 Step 3: Authenticate for MLflow + Define Constants

# COMMAND ----------

import os
import mlflow
from mlflow.utils.databricks_utils import get_databricks_env_vars


# Set Databricks auth for MLflow
mlflow_dbrx_creds = get_databricks_env_vars("databricks")
os.environ["DATABRICKS_HOST"] = mlflow_dbrx_creds['DATABRICKS_HOST']
os.environ["DATABRICKS_TOKEN"] = mlflow_dbrx_creds['DATABRICKS_TOKEN']

MODEL_NAME = "InternVL3-8B" # name of model registered in UC
MODEL_ALIAS = "production"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧠 Step 4: Define Inference Classes
# MAGIC
# MAGIC These custom Ray-callable classes handle image preprocessing and inference.
# MAGIC
# MAGIC 1. Task1: Read image, resize/adjust and Extract pixel values **[CPU intensive workload]** _Specific to [OpenGVL/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B)_
# MAGIC 2. Task2: Run inference **[GPU intensive workload]**

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 - Image PreProcessing class

# COMMAND ----------

import torch
import torchvision.transforms as T
import PIL
from io import BytesIO


class InternVLimagePreprocessor():
  """
  Wrapper class to perform image preparation for multimodal model inference
  """

  def __init__(self,
               imagenet_mean=(0.485, 0.456, 0.406),
               imagenet_std=(0.229, 0.224, 0.225),
               input_size=448,
               min_num=1,
               max_num=12,
               use_thumbnail=True):
    
    self.IMAGENET_MEAN = imagenet_mean
    self.IMAGENET_STD = imagenet_std
    self.input_size = input_size
    self.min_num = min_num 
    self.max_num = max_num
    self.use_thumbnail = use_thumbnail

    # Build torch transform once
    self._fn_transform = self._build_transform()

  
  def _build_transform(self):
    """Transform image into torch tensor"""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((self.input_size, self.input_size), interpolation=PIL.Image.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
    ])

    return transform
  
  def _find_closest_aspect_ratio(self, width, height, target_ratios):
    aspect_ratio = width/height
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * self.input_size * self.input_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    
    return best_ratio
  
  def dynamic_preprocess(self, image: PIL.Image.Image, use_thumbnail: bool=False):
    """Dynamic adjusting of aspect ratio"""
    orig_width, orig_height = image.size

    target_ratios = set(
        (i, j) for n in range(self.min_num, self.max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= self.max_num and i * j >= self.min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = self._find_closest_aspect_ratio(orig_width, orig_height, target_ratios)

    target_width = self.input_size * target_aspect_ratio[0]
    target_height = self.input_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // self.input_size)) * self.input_size,
            (i // (target_width // self.input_size)) * self.input_size,
            ((i % (target_width // self.input_size)) + 1) * self.input_size,
            ((i // (target_width // self.input_size)) + 1) * self.input_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((self.input_size, self.input_size))
        processed_images.append(thumbnail_img)
    
    return processed_images
  
  def __call__(self, row: dict, read_binary: bool=False) -> dict:
    """
    Main ingest & resize function, outputs pixel values as torch Tensor
    """
    # Read image from path or binary stream (NOTE: FOR STREAMING JOBS READ DIRECTLY FROM IMAGE PATH)
    if read_binary:
      image = PIL.Image.open(BytesIO(row["content"])).convert('RGB')
    
    else:
      image = PIL.Image.open(row["path"]).convert('RGB')

    # Adjust aspect ratio
    images = self.dynamic_preprocess(image)

    # Transform into torch tensor, stack and 'detach' to numpy array type
    pixel_values_unstacked = [self._fn_transform(image) for image in images]
    row["pixel_values"] = torch.stack(pixel_values_unstacked).detach().cpu().numpy()

    return row

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Debug
# MAGIC **IF YOU RUN IT, MAKE SURE TO FLUSH THE MODEL FROM THE DRIVER NODE TO AVOID OOM WHEN USING RAY OR ONCE DEBUG IS DONE. REMOVE/DON'T EXECUTE THE DEBUG BLOCKS WHEN RUNNING THE FULL JOB**

# COMMAND ----------

# DBTITLE 1,Create input dataframe containing raw image stream
image_binary_df = spark.read.table(f"{CATALOG}.{SCHEMA}.bronze_images_path")
test_pdf = image_binary_df.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Load model weights from UC registry
import mlflow
from util import stage_registered_model, flatten_folder


mlflow.set_registry_uri("databricks-uc")

model_weights_path = stage_registered_model(
                    catalog = CATALOG, 
                    schema = SCHEMA, 
                    model_name = MODEL_NAME, 
                    alias = MODEL_ALIAS,
                    local_base_path = "/local_disk0/models/",
                    overwrite = False)

flatten_folder(model_weights_path)
hf_model_path = str(model_weights_path)  #convert from Posit to string required by TF

# COMMAND ----------

import transformers
from transformers import AutoTokenizer, AutoModel


# hf_model_path = "OpenGVLab/InternVL3-8B"
hf_model = AutoModel.from_pretrained(
  hf_model_path,
  torch_dtype=torch.bfloat16,
  low_cpu_mem_usage=True,
  use_flash_attn=True,
  trust_remote_code=True).eval().cuda()
  
tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Single Inference

# COMMAND ----------

import pandas as pd


TestImageProcessing = InternVLimagePreprocessor()

# COMMAND ----------

# Single
out_dict = TestImageProcessing(test_pdf.iloc[0].to_dict())
print(out_dict["pixel_values"])

# COMMAND ----------

pixel_values_cuda = torch.from_numpy(out_dict["pixel_values"]).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=True)

# COMMAND ----------

question = '<image>\nPlease describe the image shortly.'
response = hf_model.chat(tokenizer, pixel_values_cuda, question, generation_config)
print(f'User: {question}\nAssistant: {response}')

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Batch

# COMMAND ----------

test_pdf['pixel_values'] = test_pdf.apply(TestImageProcessing, axis=1)['pixel_values']

# COMMAND ----------

pixel_values = [torch.from_numpy(pix).to(torch.bfloat16).cuda() for pix in test_pdf['pixel_values']]
num_patches_list = [pix.size(0) for pix in pixel_values ]
pixel_values_cat = torch.cat(tuple(pixel_values), dim=0)

# COMMAND ----------

# batch inference, single image per sample
questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
responses = hf_model.batch_chat(tokenizer, pixel_values_cat,
                             num_patches_list=num_patches_list,
                             questions=questions,
                             generation_config=generation_config)
for question, response in zip(questions, responses):
    print(f'User: {question}\nAssistant: {response}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 - Batch Inference class
# MAGIC Wrap LLM multimodal inference in ray-friendly class which will sequentially do:
# MAGIC 1. Image bytsream ingestion
# MAGIC 2. Image processing into TF.Tensor
# MAGIC 3. Run multimodal batch inference

# COMMAND ----------

import transformers
import ssl
from typing import Dict
from transformers import AutoTokenizer, AutoModel
from util import stage_registered_model, flatten_folder


class MMAnalysis_batch:
    """
    Wrapper class to perform batch multimodal model inference on prepared pixel values (formatted as numpy array)
    """

    def __init__(self, catalog:str, schema:str, model_name:str, model_alias:str = "Production", generation_config:dict =dict(max_new_tokens=1024, do_sample=True), prompt: str="Describe the image:"):
        self.unverified_context = ssl._create_unverified_context()
        print("Loading model from UC registry if weights are not already staged locally...")

        import mlflow
        mlflow.set_registry_uri("databricks-uc")
        model_weights_path = stage_registered_model(
                            catalog = catalog, 
                            schema = schema, 
                            model_name = model_name, 
                            alias = model_alias,
                            local_base_path = "/local_disk0/models/",
                            overwrite = False)

        flatten_folder(model_weights_path)
        self.path = str(model_weights_path)

        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(self.path,
                                               torch_dtype=torch.bfloat16,
                                               low_cpu_mem_usage=True,
                                               use_flash_attn=True,
                                               trust_remote_code=True).eval().cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.generation_config = generation_config
        self.prompt = prompt
        
    
    def __call__(self, batch: Dict[str, list]) -> Dict[str, list]:

        pixel_values = [torch.from_numpy(pix).to(torch.bfloat16).cuda() for pix in batch['pixel_values']]
        num_patches_list = [pix.size(0) for pix in pixel_values ]
        pixel_values_cat = torch.cat(tuple(pixel_values), dim=0)

        batch['response'] = self.model.batch_chat(self.tokenizer, pixel_values_cat,
                                    num_patches_list=num_patches_list,
                                    questions=self.prompt* len(num_patches_list),
                                    generation_config=self.generation_config)
        del batch['pixel_values']
        return batch

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Debug
# MAGIC NB: You may need to flush/release the GPU memory first

# COMMAND ----------

import torch
import gc


# Empty GPU memory
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Run garbage collection
gc.collect()

# COMMAND ----------

TestInference = MMAnalysis_batch(catalog=CATALOG, schema=SCHEMA, model_name=MODEL_NAME)
TestInference(test_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Step 5: Start Ray Cluster
# MAGIC
# MAGIC Some best practices for scaling up Ray clusters [here](https://docs.databricks.com/en/machine-learning/ray/scale-ray.html#scale-ray-clusters-on-databricks) :
# MAGIC * `num_cpus_*` always leave 1 CPU core for spark so value should be <= max cores per worker - 1

# COMMAND ----------

# DBTITLE 1,Setup Ray cluster configs
import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES


# Cluster cleanup
restart = True
if restart is True:
  try:
    shutdown_ray_cluster()

  except:
    pass

  try:
    ray.shutdown()
    
  except:
    pass

# Configs for g5.12xlarge[1xA10G] (1 worker node)
num_cpu_cores_per_worker = 47 # total cpu to use in each worker node (total_cores - 1)
num_cpus_head_node = 45 # Cores to use in driver node (Leave 3 cores for driver node)
num_gpu_per_worker = 4 # GPUs per worker node (to use)
num_gpus_head_node = 4 # GPUs in driver node (to use)
max_worker_nodes = 1

ray_conf = setup_ray_cluster(
  min_worker_nodes=1,
  max_worker_nodes=max_worker_nodes,
  num_cpus_head_node= num_cpus_head_node,
  num_gpus_head_node= num_gpus_head_node,
  num_cpus_per_node=num_cpu_cores_per_worker,
  num_gpus_per_node=num_gpu_per_worker
)

if not ray.is_initialized():
    ray.init(address=ray_conf[0], dashboard_host="0.0.0.0", dashboard_port=9999, include_dashboard=True, ignore_reinit_error=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre-requisite: download the models from MLflow registry into every node once to avoid multiple download conflicts
# MAGIC
# MAGIC **Not necessary but better to do here once ahead of time.**

# COMMAND ----------

from util import run_on_every_node


@ray.remote(num_cpus=1)
def download_model(catalog,
                  schema ,
                  model_name, 
                  alias = "Production",
                  local_base_path = "/local_disk0/models/",
                  overwrite = False):
    model_weights_path = stage_registered_model(
                  catalog = catalog,
                  schema =  schema,
                  model_name = model_name,
                  alias = alias,
                  local_base_path = local_base_path,
                  overwrite = overwrite)
    flatten_folder(model_weights_path)

# COMMAND ----------

# Point to UC registry (in case not default)
mlflow.set_registry_uri("databricks-uc")

# Execute
resulsts = run_on_every_node(download_model , **{
                  "catalog": CATALOG,
                  "schema": SCHEMA,
                  "model_name": MODEL_NAME,
                  "alias": MODEL_ALIAS
                  })

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 Step 6: Build Ray Batch Inference Pipeline
# MAGIC
# MAGIC Using [`map`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map.html#ray.data.Dataset.map) and [`map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html)
# MAGIC
# MAGIC Some relevant parameters to define are:
# MAGIC * `num_cpus`: for CPU intensive workloads (i.e. read the audio files) - defines how many ray-cores to use for individual tasks (default is `1 Ray-Core/CPU == 1 CPU Physical Core`). It can be defined as a fraction to oversubscribe a single physical core with multiple tasks
# MAGIC
# MAGIC * `num_gpus`: for GPU intensive workloads - defines how many (fractionnal) GPU(s) a single task/batch will use
# MAGIC
# MAGIC * `concurrency`: how many parallel tasks to run in parallel `Tuple(min,max)`
# MAGIC
# MAGIC * `batch_size`: if model generation loop supports batch, increase this (based on model size, single image/record size and how much GPU RAM remains available)

# COMMAND ----------

image_binary_df = spark.read.table(f"{CATALOG}.{SCHEMA}.bronze_binary_images")

ds_r = ray.data.from_spark(image_binary_df)

preds = (
ds_r.repartition(1) #num_cpu_cores_per_worker*max_worker_nodes+num_cpus_head_node
.map(
    InternVLimagePreprocessor,
    fn_constructor_kwargs={},
    concurrency=(1,92), # Can go up to total sum of cores
    num_cpus=1,
)
.map_batches(
    MMAnalysis_batch,
    fn_constructor_kwargs={
          "catalog": CATALOG,
          "schema": SCHEMA,
          "model_name": MODEL_NAME,
          "model_alias": MODEL_ALIAS,
          "prompt": "What is the image about?"
          },
    concurrency=(1,8), # [multiple of] Total number of GPUs ??
    num_gpus=1, # 1 model loaded on each GPU
    batch_size=8, # Increase/Maximize this but depends on image sizes and model size
).drop_columns(["content", "length"])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 Step 7: Point to input, run pipeline and materialize results to Delta Table

# COMMAND ----------

# DBTITLE 1,Set temporary directory for ray-uc-volumes-fuse (to write to Delta natively)
tmp_dir_fs = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/tempDoc"
dbutils.fs.mkdirs(tmp_dir_fs)
os.environ["RAY_UC_VOLUMES_FUSE_TEMP_DIR"] = tmp_dir_fs

# COMMAND ----------

mode = "overwrite" # "append"
preds.write_databricks_table(
  f"{CATALOG}.{SCHEMA}.processed_image_data_batch",
  mode = mode,
  mergeSchema = True
)

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.processed_image_data_batch"))

# COMMAND ----------

# MAGIC %md
# MAGIC Time on 1 driver (N = 4 GPUS) + 1 worker (N = 4 GPUs) ~10mins on full dataset (N = 2906 images)
