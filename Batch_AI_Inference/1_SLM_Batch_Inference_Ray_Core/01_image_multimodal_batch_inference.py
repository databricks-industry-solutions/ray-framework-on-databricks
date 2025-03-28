# Databricks notebook source
# MAGIC %md
# MAGIC # Ray On Spark for multimodal image batch inference
# MAGIC In this example we'll cover how to perform distributed batch inference on databricks clusters for Small/Medium Multimodal Language Models (i.e. SLM), while also leveraging MLflow for model versionning and logging.
# MAGIC
# MAGIC
# MAGIC **Pre-Requisites:**
# MAGIC - Set this spark config at the cluster level: `spark.task.resource.gpu.amount 0` before starting the cluster
# MAGIC
# MAGIC Tested on:
# MAGIC ```
# MAGIC Databricks Machine Learning Runtime 15.4LTS
# MAGIC mlflow=2.19.0
# MAGIC ray==2.40.2
# MAGIC ```
# MAGIC **WORK-IN-PROGRESS/TO-DO:**
# MAGIC - Switch from `Mini-Intern-VL` to `Qwen` and log/snap model and load back from mlflow/UC Registry

# COMMAND ----------

# MAGIC %pip install -qU ray[default] ray[data] timm
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Define catalog/schema/volumes
catalog = "amine_elhelou" # Change This/Point to an existing catalog
schema = "ray_gtm_examples" # Point to an existing schema
volume = "fashion-images"

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE WIDGET TEXT catalog DEFAULT 'amine_elhelou';
# MAGIC CREATE WIDGET TEXT schema DEFAULT 'ray_gtm_examples';
# MAGIC CREATE WIDGET TEXT volume DEFAULT 'fashion-images';

# COMMAND ----------

# DBTITLE 1,Create/Set catalog, schema and volume
# MAGIC %sql
# MAGIC USE CATALOG ${catalog};
# MAGIC USE SCHEMA ${schema};
# MAGIC CREATE VOLUME IF NOT EXISTS `${volume}`

# COMMAND ----------

# DBTITLE 1,Set images folder/volumes paths
image_paths = ["Apparel/Boys/Images/images_with_product_ids/", "Apparel/Girls/Images/images_with_product_ids/",
               "Footwear/Men/Images/images_with_product_ids/", "Footwear/Women/Images/images_with_product_ids/"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read images as bytestream into a Delta Table

# COMMAND ----------

# DBTITLE 1,Read directly as binary stream and persist as DELTA table
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
  data_bin_df = data_bin_df.unionAll(spark.read.format("binaryFile").load(f"{volume_path}/{path}"))

# Materialze
data_bin_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.bronze_binary_images")

# NOTE: If images are streamed/dumped to Cloud Storage, this can be turned into a streaming ingestion job using autoloader

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ${catalog}.${schema}.bronze_binary_images LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1- Batch inference using Ray on Spark
# MAGIC 1. Task1: Read image, resize/adjust and Extract pixel values **[CPU intensive workload]** _Specific to [OpenGVL/Mini-Intern-VL](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5)_
# MAGIC 2. Task2: Run inference **[GPU intensive workload]**

# COMMAND ----------

# DBTITLE 1,Task1: Wrap image preprocessing in ray-friendly class
import torch
import torchvision.transforms as T
import PIL
from io import BytesIO


class MiniInternVLimagePreprocessor():
  """
  Wrapper class to perform image preparation for multimodal model inference
  """

  def __init__(self,
               imagenet_mean=(0.485, 0.456, 0.406),
               imagenet_std=(0.229, 0.224, 0.225),
               input_size=448,
               min_num=1,
               max_num=6,
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
  
  def __call__(self, row: dict) -> dict:
    """
    Main ingest & resize function, outputs pixel values as torch Tensor
    """
    # Read byte stream
    image = PIL.Image.open(BytesIO(row["content"])).convert('RGB')

    # Adjust aspect ratio
    images = self.dynamic_preprocess(image)

    # Transform into torch tensor, stack and 'detach' to numpy array type
    pixel_values_unstacked = [self._fn_transform(image) for image in images]
    row["pixel_values"] = torch.stack(pixel_values_unstacked).detach().cpu().numpy()

    return row

# COMMAND ----------

# DBTITLE 1,Task2: Download model from hugging face and snap weights to mlflow
import transformers
from transformers import AutoTokenizer, AutoModel


hf_model_path = "OpenGVLab/InternVL2_5-4B" # OpenGVLab/Mini-InternVL-Chat-2B-V1-5"
hf_model = AutoModel.from_pretrained(hf_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)

# COMMAND ----------

import mlflow


with mlflow.start_run(run_name="Model Weights Snapping") as run:

  mlflow.transformers.log_model(
    transformers_model = hf_model,
    artifact_path = "hf_model",
  )

# COMMAND ----------

# DBTITLE 1,Wrap LLM multimodal inference in ray-friendly class
from typing import Dict


test_prompt = "What do you see in the image?"


class MMAnalysis_batch:
    """
    Wrapper class to perform batch multimodal model inference on prepared pixel values (formatted as numpy array)
    """

    def __init__(self, model_path:str = hf_model_path, prompt: str=test_prompt):
        self.path = model_path
        self.model = AutoModel.from_pretrained(self.path,
                                               torch_dtype=torch.bfloat16,
                                               low_cpu_mem_usage=True,
                                               trust_remote_code=True).eval().cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.generation_config = dict(num_beams=1,
                                      max_new_tokens=512,
                                      do_sample=False)
        self.prompt = prompt
        
    
    def __call__(self, batch: Dict[str, list]) -> Dict[str, list]:

        pixel_values = [torch.from_numpy(pix).to(torch.bfloat16).cuda() for pix in batch['pixel_values'] ]
        num_patches_list = [pix.size(0) for pix in pixel_values ]
        pixel_values_cat = torch.cat(tuple(pixel_values), dim=0)

        batch['final_answer'] = self.model.batch_chat(self.tokenizer, pixel_values_cat,
                                    num_patches_list=num_patches_list,
                                    questions=self.prompt* len(num_patches_list),
                                    generation_config=self.generation_config)
        del batch['pixel_values']
        return batch

# COMMAND ----------

# MAGIC %md
# MAGIC ### Debug
# MAGIC **IF YOU RUN IT, MAKE SURE TO FLUSH THE MODEL FROM THE DRIVER NODE TO AVOID OOM WHEN USING RAY**

# COMMAND ----------

# DBTITLE 1,Create input dataframe containing raw image stream
image_binary_df = spark.read.table(f"{catalog}.{schema}.raw_binary_data")

# COMMAND ----------

# DBTITLE 1,Debug Image preparation
import pandas as pd


TestImageProcessing = MiniInternVLimagePreprocessor()

# Single
test_pdf = image_binary_df.limit(5).toPandas()
out_dict = TestImageProcessing(test_pdf.iloc[0].to_dict())
print(out_dict["pixel_values"])

# COMMAND ----------

# DBTITLE 1,Debug Image inference
# NOT WORKING FOR NOW - PLEASE IGNORE
model_path = model_hf_path # Pointing to HF hub right now
TestInference = MMAnalysis_batch(model_path=model_hf_path)

TestInference(out_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run batch inference

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

# Configs for g5.4xlarge[1xA10G] (2 worker nodes)
num_cpu_cores_per_worker = 15 # total cpu to use in each worker node (total_cores - 1)
num_cpus_head_node = 12 # Cores to use in driver node (BP is to leave 3 cores for driver)
num_gpu_per_worker = 1 # GPUs per worker node (to use)
num_gpus_head_node = 1 # GPUs in driver node (to use)
max_worker_nodes = 4


ray_conf = setup_ray_cluster(
  min_worker_nodes=1,
  max_worker_nodes=max_worker_nodes,
  num_cpus_head_node= num_cpus_head_node,
  num_gpus_head_node= num_gpus_head_node,
  num_cpus_per_node=num_cpu_cores_per_worker,
  num_gpus_per_node=num_gpu_per_worker
)

if not ray.is_initialized():
    ray.init(address=ray_conf[0], ignore_reinit_error=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run image preparation using `map` and inference using `map_batches` on raw image stream stored in Delta Table

# COMMAND ----------

# DBTITLE 1,Wrap processing and inference classes into pandas_udf + ray tasks
import pandas as pd
import time
from pyspark.sql.types import StringType
from pyspark.sql.functions import pandas_udf


@pandas_udf(StringType())
def process_images_batch(contents: pd.Series) -> pd.Series:
    # start = time.time()

    ray.init(ray_conf[1],ignore_reinit_error=True)

    # Persist to Ray Object Store (in case column content/bytestream is too big)
    ref_contents = ray.put(contents) 

    @ray.remote
    def ray_data_task(ds = None):
        # Cast pandas Series to Dataframe to MaterializedDataset
        ds_r = ray.data.from_pandas(pd.DataFrame(ds.to_list(),columns = ['content']))

        print("shape:",ds.shape[0])
        preds = (
        ds_r.repartition(ds.shape[0])
        .map(
            MiniInternVLimagePreprocessor,
            compute=ray.data.ActorPoolStrategy(min_size=1,max_size=100), # Modify these based on total cluster size
            num_cpus=1,
        )
        .map_batches(
            MMAnalysis_batch,
            concurrency=(1,20), # [multiple of] Total number of GPUs ??
            num_gpus=1, # 1 model loaded on each GPU
            batch_size=12, # Goal is to maximize this but depends on image sizes and model size
        )
        )
        # end = time.time()
        # print("Loaded model dependencies" ,end - start)
        final_df = preds.to_pandas() 
        # Output as pandas series
        return final_df['final_answer'].astype(str) 
    
    return ray.get(ray_data_task.remote(ref_contents))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Execute and materialize text completions into a new Delta Table

# COMMAND ----------

# DBTITLE 1,Read image streams in spark dataframe and repartition to 1
from pyspark.sql.functions import col


image_binary_df = spark.read.table(f"{catalog}.{schema}.raw_binary_data")

# Repartition the data to 1 worker node
df = image_binary_df.repartition(1)

# Add the column
df = df.withColumn("image_description", process_images_batch(col("content")))

# COMMAND ----------

# DBTITLE 1,Execute & Materialize outputs
# Persist output to new delta table
df.write.mode('overwrite').option("mergeSchema", "true").saveAsTable(f"{catalog}.{schema}.processed_image_data_batch")

# COMMAND ----------

# MAGIC %md
# MAGIC Time on 4 worker nodes using `map_batches` ~10mins on full dataset
